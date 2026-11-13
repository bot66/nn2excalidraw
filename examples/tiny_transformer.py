"""Two pre-norm transformer blocks, below the repeat-collapse threshold."""

from __future__ import annotations

import torch
from torch import nn

DIAGRAM_TITLE = "Tiny Transformer"


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        h, _ = self.self_attention(h, h, h, need_weights=False)
        x = x + h
        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        return x


class TinyTransformer(nn.Module):
    def __init__(self, dim: int = 32, heads: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Linear(16, dim)
        self.block1 = EncoderBlock(dim, heads, hidden)
        self.block2 = EncoderBlock(dim, heads, hidden)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.norm(x)
        return self.head(x)
