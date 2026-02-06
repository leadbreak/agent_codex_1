from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
