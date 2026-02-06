from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([x, y], dim=-1)))
        return gate * x + (1 - gate) * y
