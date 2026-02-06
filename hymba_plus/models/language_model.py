from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from hymba_plus.blocks.hybrid import HybridBlock
from hymba_plus.blocks.mamba import MambaBlock
from hymba_plus.blocks.transformer import TransformerBlock
from hymba_plus.components.embedding.rope import RotaryEmbedding
from hymba_plus.components.embedding.token import TokenEmbedding
from hymba_plus.components.norm.rmsnorm import RMSNorm
from hymba_plus.core.config import HymbaPlusConfig
from hymba_plus.models.base import BaseModel


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class HymbaPlus(BaseModel, nn.Module):
    """Minimal but runnable Hymba+ language model."""

    def __init__(self, config: HymbaPlusConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            dim=config.d_model // config.n_heads,
            theta=config.architecture.embedding.rope_theta,
            max_position_embeddings=config.architecture.embedding.max_position_embeddings,
        )
        self.layers = nn.ModuleList()
        use_triton = config.optimization.kernels.use_triton
        use_flash = config.architecture.attention.use_flash
        flash_version = config.architecture.attention.flash_version
        for _ in range(config.n_layers):
            if config.architecture.transformer_ratio >= 0.99:
                self.layers.append(
                    TransformerBlock(
                        d_model=config.d_model,
                        num_heads=config.n_heads,
                        dropout=config.architecture.attention.dropout,
                        rope=self.rope,
                        use_flash=use_flash,
                        flash_version=flash_version,
                    )
                )
            elif config.architecture.transformer_ratio <= 0.01:
                self.layers.append(
                    MambaBlock(
                        d_model=config.d_model,
                        d_state=config.architecture.ssm.d_state,
                        d_conv=config.architecture.ssm.d_conv,
                        dropout=config.architecture.ssm.dropout,
                        use_triton=use_triton,
                    )
                )
            else:
                self.layers.append(
                    HybridBlock(
                        d_model=config.d_model,
                        num_heads=config.n_heads,
                        d_state=config.architecture.ssm.d_state,
                        d_conv=config.architecture.ssm.d_conv,
                        dropout=config.architecture.attention.dropout,
                        rope=self.rope,
                        use_triton=use_triton,
                        use_flash=use_flash,
                        flash_version=flash_version,
                    )
                )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @classmethod
    def from_recipe(cls, recipe: str | Dict[str, Any], **overrides: Any) -> "HymbaPlus":
        config = HymbaPlusConfig.from_dict({"model": overrides})
        return cls(config)

    @classmethod
    def from_ratio(cls, transformer_ratio: float, **overrides: Any) -> "HymbaPlus":
        config = HymbaPlusConfig.from_dict(
            {
                "model": overrides,
                "architecture": {"transformer_ratio": transformer_ratio},
            }
        )
        return cls(config)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> ModelOutput:
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, seq_len)
        attention_mask = torch.zeros(batch, 1, seq_len, seq_len, device=input_ids.device)

        x = self.embed(input_ids)
        for layer in self.layers:
            if isinstance(layer, TransformerBlock) or isinstance(layer, HybridBlock):
                x = layer(x, attention_mask=attention_mask, position_ids=position_ids)
            else:
                x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return ModelOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 1) -> torch.Tensor:
        for _ in range(max_new_tokens):
            out = self.forward(input_ids)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids
