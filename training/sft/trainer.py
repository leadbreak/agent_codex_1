from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from training.utils.scheduler import CosineSchedule


@dataclass
class SFTConfig:
    """Nanochat-style SFT config."""

    max_steps: int = 100
    warmup_steps: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    bf16: bool = True


class SFTTrainer:
    """Minimal SFT trainer with cosine LR, warmup, grad accumulation, and bf16."""

    def __init__(self, model: nn.Module, config: SFTConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(self._build_param_groups(), lr=config.learning_rate)
        self.schedule = CosineSchedule(
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            base_lr=config.learning_rate,
        )

    def _build_param_groups(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def train(self, dataloader) -> None:
        self.model.train()
        step = 0
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.bf16 and torch.cuda.is_available())
        for batch in dataloader:
            lr = self.schedule.get_lr(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.bf16 and torch.cuda.is_available()):
                output = self.model(batch["input_ids"], labels=batch["labels"])
                loss = output.loss / self.config.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % self.config.grad_accum_steps == 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if step >= self.config.max_steps:
                break
            step += 1
