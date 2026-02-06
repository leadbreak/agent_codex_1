from __future__ import annotations

import torch
from torch import nn

from hymba_plus.components.mlp.swiglu import SwiGLU
from hymba_plus.components.norm.rmsnorm import RMSNorm
from hymba_plus.components.ssm.simple_ssm import SimpleSSM


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, dropout: float, use_triton: bool) -> None:
        super().__init__()
        self.ssm = SimpleSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout,
            use_triton=use_triton,
        )
        self.mlp = SwiGLU(d_model, expand=4, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
