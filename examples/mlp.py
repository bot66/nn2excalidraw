"""Small MLP used by the static-parser tests.

Three linear stages. ReLU is fused into the preceding linear in the block view.
"""

from __future__ import annotations

import torch
from torch import nn

DIAGRAM_TITLE = "MLP"


class MLP(nn.Module):
    def __init__(self, in_features: int = 32, hidden: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.head(x)
