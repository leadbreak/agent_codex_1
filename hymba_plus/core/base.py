from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    """모든 구성요소의 추상 기본 클래스."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
