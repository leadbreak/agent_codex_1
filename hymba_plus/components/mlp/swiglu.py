from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = d_model * expand
        self.in_proj = nn.Linear(d_model, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(torch.nn.functional.silu(gate) * x))
