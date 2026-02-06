from __future__ import annotations

import torch
from torch import nn

from hymba_plus.optim.kernels import TRITON_AVAILABLE, gate_mul


class SimpleSSM(nn.Module):
    """깊이별 합성곱 + 게이팅 기반의 경량 SSM 블록.

    실제 Mamba 커널을 대체하지 않으며, 구성 검증과 실동작을 위한
    벡터화된 대체 구현입니다.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        dropout: float = 0.0,
        use_triton: bool = False,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv - 1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_state = d_state
        self.use_triton = use_triton and TRITON_AVAILABLE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        value = value.transpose(1, 2)
        value = self.conv(value)[..., : x.size(1)].transpose(1, 2)
        if self.use_triton and x.is_cuda:
            out = gate_mul(value.contiguous(), gate.contiguous())
        else:
            out = torch.sigmoid(gate) * value
        out = self.out_proj(out)
        return self.dropout(out)
