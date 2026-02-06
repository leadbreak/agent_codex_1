from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """RoPE(회전 위치 임베딩) 구현.

    rope_scaling:
    - None: 기본 RoPE
    - linear: 길이 스케일을 선형 확장
    - ntk: NTK 스케일링 근사
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        rope_scaling: str | None = None,
        rope_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_scale_factor = rope_scale_factor

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_scaling(self, position_ids: torch.Tensor) -> torch.Tensor:
        if self.rope_scaling is None:
            return position_ids
        if self.rope_scaling == "linear":
            return position_ids / self.rope_scale_factor
        if self.rope_scaling == "ntk":
            return position_ids / (self.rope_scale_factor ** (position_ids / self.max_position_embeddings))
        raise ValueError(f"알 수 없는 rope_scaling: {self.rope_scaling}")

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        position_ids = self._apply_scaling(position_ids)
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
