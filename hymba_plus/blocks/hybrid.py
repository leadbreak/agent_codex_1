from __future__ import annotations

import torch
from torch import nn

from hymba_plus.components.attention.standard import StandardAttention
from hymba_plus.components.fusion.average import AverageFusion
from hymba_plus.components.mlp.swiglu import SwiGLU
from hymba_plus.components.norm.rmsnorm import RMSNorm
from hymba_plus.components.ssm.simple_ssm import SimpleSSM
from hymba_plus.components.embedding.rope import RotaryEmbedding


class HybridBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_state: int,
        d_conv: int,
        dropout: float,
        rope: RotaryEmbedding,
        use_triton: bool,
        use_flash: bool,
        flash_version: int,
    ) -> None:
        super().__init__()
        self.attn = StandardAttention(
            d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            dropout=dropout,
            rope=rope,
            use_flash=use_flash,
            flash_version=flash_version,
        )
        self.ssm = SimpleSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout,
            use_triton=use_triton,
        )
        self.fusion = AverageFusion()
        self.mlp = SwiGLU(d_model, expand=4, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), attention_mask=attention_mask, position_ids=position_ids, is_causal=True)
        ssm_out = self.ssm(self.norm1(x))
        x = x + self.fusion(attn_out, ssm_out)
        x = x + self.mlp(self.norm2(x))
        return x
