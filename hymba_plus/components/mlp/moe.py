from __future__ import annotations

import torch
from torch import nn


class MoE(nn.Module):
    """Vectorized top-k MoE.

    This implementation is simple and correct, not optimized.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_experts)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        expert_out = torch.zeros_like(x)
        x_flat = x.view(-1, dim)
        weights_flat = weights.view(-1, self.top_k)
        indices_flat = indices.view(-1, self.top_k)

        for expert_id in range(self.n_experts):
            mask = indices_flat.eq(expert_id)
            if not mask.any():
                continue
            token_ids = mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
            expert_tokens = x_flat[token_ids]
            expert_output = self.experts[expert_id](expert_tokens)
            expert_weights = weights_flat[token_ids] * mask[token_ids].float()
            expert_output = expert_output * expert_weights.sum(dim=-1, keepdim=True)
            expert_out.view(-1, dim)[token_ids] += expert_output

        return self.dropout(expert_out)
