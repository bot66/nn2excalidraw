"""ViT-B/16, matching ``torchvision.models.vit_b_16`` module names.

Configuration (Dosovitskiy et al., 2020; torchvision defaults):

- image 224, patch 16, sequence length 196 + class token = 197
- hidden 768, MLP 3072, 12 heads, 12 encoder blocks, 1000 classes
- ``conv_proj``: Conv2d 16×16 stride 16
- ``class_token`` prepended on the sequence axis
- ``encoder.pos_embedding`` added before the block stack
- each ``EncoderBlock`` is pre-norm: LayerNorm → MultiheadAttention → residual,
  LayerNorm → MLP → residual
- class-token readout ``x[:, 0]`` then ``heads``

Twelve identical blocks collapse to one expanded block tagged × 12.
No pretrained weights are loaded. Construct with ``VisionTransformer()`` or ``vit_b_16()``.
"""

from __future__ import annotations

import torch
from torch import nn

DIAGRAM_TITLE = "ViT-B/16"


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear_1(x))
        x = self.linear_2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        h, _ = self.self_attention(h, h, h, need_weights=False)
        x = x + h
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, seq_length: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))
        layers = []
        for _ in range(num_layers):
            layers.append(EncoderBlock(num_heads, hidden_dim, mlp_dim))
        self.layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding
        x = self.layers(x)
        x = self.ln(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.conv_proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length = (image_size // patch_size) ** 2 + 1
        self.encoder = Encoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim)
        self.heads = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        return self.heads(x)


def vit_b_16(num_classes: int = 1000) -> VisionTransformer:
    """Torchvision-equivalent ViT-B/16 constructor. Weights are randomly initialized."""

    return VisionTransformer(num_classes=num_classes)
