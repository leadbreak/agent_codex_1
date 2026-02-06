from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, max_position_embeddings: int = 2048) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("i,j->ij", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb

    @staticmethod
    def apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        return x * cos + rot * sin
