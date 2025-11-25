from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for the freezing pipeline.") from exc

from fusion_transformer_model.global_local_fusion import (
    FusionConfig,
    GlobalLocalFusionClassifier,
)


@dataclass
class FreezingConfig:

    train_dir: str = "C:/Users/haric/Downloads/aptos2019/train_images/"
    test_dir: str = "C:/Users/haric/Downloads/aptos2019/test_images/"
    num_classes: int = 5
    model_name: str = "vit_base_patch16_224"
    epochs: int = 20
    freeze_after_epochs: int = 15
    train_split: float = 0.8
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 3e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    image_size: int = 224
    expected_accuracy_gain: float = 0.25
    use_amp: bool = True
    checkpoint_dir: str = "results/"


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def _create_dataloaders(cfg: FreezingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_path = Path(cfg.train_dir)
    if not train_path.exists():  # pragma: no cover - defensive check
        raise FileNotFoundError(f"Training directory not found: {train_path}")

    train_tf, eval_tf = _build_transforms(cfg.image_size)
    full_dataset = datasets.ImageFolder(train_path, transform=train_tf)
    train_len = int(len(full_dataset) * cfg.train_split)
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    # Validation and test transforms should not include augmentation.
    val_dataset.dataset.transform = eval_tf

    test_path = Path(cfg.test_dir)
    if test_path.exists():
        test_dataset = datasets.ImageFolder(test_path, transform=eval_tf)
    else:
        # When no hold-out set is provided reuse validation split for smoke tests.
        test_dataset = val_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _freeze_transformers(model: GlobalLocalFusionClassifier) -> None:
    for module in [model.local_encoder, model.global_encoder, model.fusion_vit]:
        for param in module.parameters():
            param.requires_grad = False

    if hasattr(model.global_encoder, "vit"):
        for param in model.global_encoder.vit.parameters():
            param.requires_grad = False


def _unfreeze_classifier_head(model: GlobalLocalFusionClassifier) -> None:
    for param in model.linear_head.parameters():
        param.requires_grad = True


def _train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def _evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_with_freezing(cfg: FreezingConfig) -> dict:
    """Train then freeze transformer encoders, fine-tuning the classifier head."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = _create_dataloaders(cfg)

    fusion_cfg = FusionConfig(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
    )
    model = GlobalLocalFusionClassifier(fusion_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    head_only = False

    for epoch in range(cfg.epochs):
        if epoch == cfg.freeze_after_epochs and not head_only:
            _freeze_transformers(model)
            _unfreeze_classifier_head(model)
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=cfg.head_learning_rate,
                weight_decay=0.0,
            )
            head_only = True

        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_acc = _evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

    test_acc = _evaluate(model, test_loader, device)
    val_gain = history["val_acc"][-1] - history["val_acc"][cfg.freeze_after_epochs - 1]
    print(
        f"Validation accuracy improved by {val_gain:.2f}, "
        f"target gain: {cfg.expected_accuracy_gain:.2f}"
    )
    print(f"Test accuracy (transformer-backed inference): {test_acc:.4f}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.model_name}_fusion_freeze.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved classifier checkpoint to {ckpt_path}")

    return {"history": history, "test_accuracy": test_acc, "checkpoint": str(ckpt_path)}


if __name__ == "__main__":  
    config = FreezingConfig()
    metrics = train_with_freezing(config)
    print("Training complete", metrics)


