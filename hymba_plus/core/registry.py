from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type


@dataclass(frozen=True)
class RegistryItem:
    name: str
    cls: Type[Any]


class Registry:
    """Simple registry for pluggable components."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, RegistryItem] = {}

    def register(self, key: str) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(cls: Type[Any]) -> Type[Any]:
            if key in self._items:
                raise KeyError(f"{self.name} registry already has key '{key}'")
            self._items[key] = RegistryItem(name=key, cls=cls)
            return cls

        return decorator

    def get(self, key: str) -> Type[Any]:
        if key not in self._items:
            raise KeyError(f"Unknown {self.name} key '{key}'")
        return self._items[key].cls

    def build(self, key: str, **kwargs: Any) -> Any:
        cls = self.get(key)
        return cls(**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items.keys())
