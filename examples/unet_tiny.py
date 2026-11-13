"""Tiny encoder-decoder. Skip edges are the saved encoder activations."""

from __future__ import annotations

import torch
from torch import nn

DIAGRAM_TITLE = "Tiny U-Net"


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(8, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.bot = ConvBlock(16, 32)
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec2 = ConvBlock(32 + 16, 16)
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec1 = ConvBlock(16 + 8, 8)
        self.head = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        bottom = self.bot(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(bottom), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
