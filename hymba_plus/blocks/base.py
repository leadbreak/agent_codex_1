from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBlock(ABC):
    @abstractmethod
    def forward(self, hidden_states: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
