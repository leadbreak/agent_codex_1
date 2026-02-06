from __future__ import annotations

import torch
from torch import nn


class LearnableFusion(nn.Module):
    def __init__(self, init_beta: float = 0.5, learnable: bool = True) -> None:
        super().__init__()
        beta = torch.tensor(init_beta)
        self.beta = nn.Parameter(beta) if learnable else beta

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(self.beta, 0.0, 1.0)
        return beta * x + (1 - beta) * y
