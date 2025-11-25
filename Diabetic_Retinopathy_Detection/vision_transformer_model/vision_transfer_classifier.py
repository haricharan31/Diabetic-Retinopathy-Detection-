
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  
    timm = None


@dataclass
class VisionTransferConfig:
    """Configuration holder mirroring the legacy `config_robust.py` layout."""

    train_split: float = 0.8
    learning_rate: float = 5e-5
    train_batch_size: int = 8
    valid_batch_size: int = 8
    num_workers: int = 0
    epochs: int = 20
    img_width: int = 224
    img_height: int = 224
    num_classes: int = 5
    model_name: str = "vit_base_patch16_224"
    model_path: str = "./checkpoints/vit_base_patch16_224.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class VisionTransformerEncoder(nn.Module):

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required to instantiate VisionTransformerEncoder. "
                "Install it with `pip install timm`."
            )
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.embed_dim = self.vit.embed_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.vit.pos_embed
        tokens = self.vit.pos_drop(tokens)
        for blk in self.vit.blocks:
            tokens = blk(tokens)
        tokens = self.vit.norm(tokens)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features(x)
        cls_token = tokens[:, 0]
        return self.vit.head(cls_token)


class LinearClassifier(nn.Module):

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class VisionTransferModel(nn.Module):

    def __init__(self, cfg: VisionTransferConfig):
        super().__init__()
        self.encoder = VisionTransformerEncoder(cfg.model_name, num_classes=cfg.num_classes)
        self.classifier = LinearClassifier(self.encoder.embed_dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(x)
        cls_token = tokens[:, 0]
        return self.classifier(cls_token)


def _step(
    model: VisionTransferModel,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    images, labels = batch
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu())


def train_one_epoch(
    model: VisionTransferModel,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running = 0.0
    for batch in dataloader:
        running += _step(model, batch, optimizer)
    return running / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: VisionTransferModel,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    model.eval()
    total, correct, loss = 0, 0, 0.0
    for images, labels in dataloader:
        logits = model(images)
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    avg_loss = loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def save_metrics(report: str, output_path: Path) -> None:
    output_path.write_text(report.strip() + "\n", encoding="utf-8")


def run_pipeline(
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: Optional[VisionTransferConfig] = None,
) -> Dict[str, float]:
    cfg = cfg or VisionTransferConfig()
    torch.manual_seed(42)
    model = VisionTransferModel(cfg)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    for _ in range(cfg.epochs):
        train_one_epoch(model, train_loader, optimizer)
    metrics = evaluate(model, val_loader)
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.model_path)
    return metrics


if __name__ == "__main__":  # pragma: no cover
    print(
        "This module defines the Vision Transfer training pipeline.\n"
        "Hook it up to your dataloaders and call `run_pipeline` to launch training."
    )

