from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    """Base class for all components."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - interface only
        raise NotImplementedError
