
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


class MedicalTransformerBackbone(nn.Module):
    """Toy Medical Transformer inspired encoder that outputs feature vectors."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(self.stem(x))
        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # B, HW, C
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return pooled


@dataclass
class HybridConfig:
    model_name: str = "vit_base_patch16_224"
    num_classes: int = 5
    learning_rate: float = 3e-5
    epochs: int = 10


class MedTWithVisionTransformer(nn.Module):

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for MedTWithVisionTransformer.")
        self.cfg = cfg
        self.medt = MedicalTransformerBackbone()
        self.vit = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
        self.vit_head = nn.Identity()
        self.vit_classifier = self.vit.get_classifier()
        vit_in_dim = self.vit_classifier.in_features
        self.vit.reset_classifier(0, "")
        fusion_dim = vit_in_dim + 256
        self.linear_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_tokens = self.vit.forward_features(x)
        cls_token = global_tokens[:, 0]
        local_vector = self.medt(x)
        fused = torch.cat([cls_token, local_vector], dim=-1)
        return self.linear_head(fused)


def train_loop(
    model: nn.Module,
    loaders: Tuple[
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ],
    cfg: HybridConfig,
) -> float:
    train_loader, val_loader = loaders
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    for _ in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


if __name__ == "__main__":  # pragma: no cover
    print(
        "Instantiate `MedTWithVisionTransformer(HybridConfig())` and call "
        "`train_loop` with your dataloaders to reproduce the Medical Transformer "
        "to Vision Transformer classification experiment."
    )

