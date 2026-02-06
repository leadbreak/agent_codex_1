from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from hymba_plus.components.embedding.rope import RotaryEmbedding


class StandardAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        rope: Optional[RotaryEmbedding] = None,
        use_flash: bool = False,
        flash_version: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope
        self.use_flash = use_flash
        self.flash_version = flash_version

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        if not q.is_cuda:
            raise RuntimeError("Flash attention requires CUDA tensors")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope is not None and position_ids is not None:
            freqs = self.rope(position_ids)
            q = RotaryEmbedding.apply_rotary(q, freqs)
            k = RotaryEmbedding.apply_rotary(k, freqs)

        if self.use_flash:
            try:
                out = self._flash_attention(q, k, v, is_causal=is_causal)
            except RuntimeError:
                out = None
        else:
            out = None

        if out is None:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                scores = scores + attention_mask
            if is_causal:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.proj(out)
