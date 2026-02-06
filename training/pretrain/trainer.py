from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from training.utils.scheduler import CosineSchedule


@dataclass
class PretrainConfig:
    max_steps: int = 100
    warmup_steps: int = 10
    learning_rate: float = 3e-4
    grad_accum_steps: int = 1
    bf16: bool = True


class PretrainTrainer:
    def __init__(self, model: nn.Module, config: PretrainConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.schedule = CosineSchedule(
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            base_lr=config.learning_rate,
        )

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
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if step >= self.config.max_steps:
                break
            step += 1
