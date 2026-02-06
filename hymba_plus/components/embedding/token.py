from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)
