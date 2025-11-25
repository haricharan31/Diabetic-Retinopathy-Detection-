

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


class LocalFeatureEncoder(nn.Module):
    """Medical Transformer inspired encoder that produces local feature vectors."""

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=6,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        return tokens.mean(dim=1)


class GlobalFeatureEncoder(nn.Module):
    """Vision Transformer encoder that exposes CLS token representations."""

    def __init__(self, model_name: str):
        super().__init__()
        if timm is None:
            raise ImportError("timm must be installed to use GlobalFeatureEncoder.")
        self.vit = timm.create_model(model_name, pretrained=True)
        head_in = self.vit.get_classifier().in_features
        self.vit.reset_classifier(0, "")
        self.embed_dim = head_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)
        return tokens[:, 0]


@dataclass
class FusionConfig:
    model_name: str = "vit_base_patch16_224"
    num_classes: int = 5
    fusion_layers: int = 2
    learning_rate: float = 3e-5
    epochs: int = 12


class GlobalLocalFusionClassifier(nn.Module):

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.local_encoder = LocalFeatureEncoder()
        self.global_encoder = GlobalFeatureEncoder(cfg.model_name)
        head_dim = self.local_encoder.transformer.layers[0].linear1.in_features
        fusion_dim = head_dim + self.global_encoder.embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=8,
            batch_first=True,
        )
        self.fusion_vit = nn.TransformerEncoder(encoder_layer, num_layers=cfg.fusion_layers)
        self.linear_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_vec = self.local_encoder(x)
        global_vec = self.global_encoder(x)
        fused_tokens = torch.stack([local_vec, global_vec], dim=1)
        attended = self.fusion_vit(fused_tokens)
        combined = attended.mean(dim=1)
        return self.linear_head(combined)


def train_fusion_model(
    model: nn.Module,
    loaders: Tuple[
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ],
    cfg: FusionConfig,
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
    cfg = FusionConfig()
    model = GlobalLocalFusionClassifier(cfg)
    print(
        "Fusion classifier instantiated. "
        "Provide dataloaders to `train_fusion_model` to reproduce experiments."
    )

