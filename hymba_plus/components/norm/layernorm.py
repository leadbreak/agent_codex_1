from __future__ import annotations

from torch import nn


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__(dim, eps=eps)
