from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CosineSchedule:
    warmup_steps: int
    max_steps: int
    base_lr: float
    min_lr: float = 0.0

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(max(1, self.warmup_steps))
        progress = min(1.0, float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine
