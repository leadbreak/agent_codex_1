from __future__ import annotations

import torch
from torch import nn


class MoE(nn.Module):
    """벡터화된 Top-k MoE 구현.

    - 토큰 단위 Python 루프를 제거하고, 배치/시퀀스 차원에서
      전문가 출력을 일괄 계산합니다.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if top_k > n_experts:
            raise ValueError("top_k는 n_experts 이하여야 합니다.")
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.weight = nn.Parameter(torch.empty(n_experts, d_model, d_model))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

        weights_full = torch.zeros(batch, seq_len, self.n_experts, device=x.device, dtype=x.dtype)
        weights_full.scatter_(-1, topk_idx, topk_weights)

        expert_out = torch.einsum("bsd,edh->bseh", x, self.weight)
        out = torch.einsum("bse,bseh->bsd", weights_full, expert_out)
        return self.dropout(out)
