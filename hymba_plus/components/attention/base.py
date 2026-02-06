from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hymba_plus.core.base import Component


@dataclass
class AttentionOutput:
    hidden_states: Any


class BaseAttention(Component):
    def forward(self, hidden_states: Any, **kwargs: Any) -> AttentionOutput:  # pragma: no cover - stub
        raise NotImplementedError
