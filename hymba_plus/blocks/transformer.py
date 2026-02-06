from __future__ import annotations

import torch
from torch import nn

from hymba_plus.components.attention.standard import StandardAttention
from hymba_plus.components.mlp.swiglu import SwiGLU
from hymba_plus.components.norm.rmsnorm import RMSNorm
from hymba_plus.components.embedding.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        rope: RotaryEmbedding,
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
        self.mlp = SwiGLU(d_model, expand=4, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask, position_ids=position_ids, is_causal=True)
        x = x + self.mlp(self.norm2(x))
        return x
