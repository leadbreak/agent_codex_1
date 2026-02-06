from __future__ import annotations

import torch
from torch import nn


class AverageFusion(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x + y)
