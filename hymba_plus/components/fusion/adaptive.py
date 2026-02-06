from __future__ import annotations

import torch
from torch import nn


class AdaptiveFusion(nn.Module):
    def __init__(self, d_model: int, init_beta: float = 0.5) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(x))
        beta = torch.clamp(self.beta, 0.0, 1.0)
        return beta * gate * x + (1 - beta) * (1 - gate) * y
