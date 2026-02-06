from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type


@dataclass(frozen=True)
class RegistryItem:
    name: str
    cls: Type[Any]


class Registry:
    """플러그인 컴포넌트를 관리하는 간단한 레지스트리."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, RegistryItem] = {}

    def register(self, key: str) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(cls: Type[Any]) -> Type[Any]:
            if key in self._items:
                raise KeyError(f"{self.name} 레지스트리에 이미 존재하는 키: '{key}'")
            self._items[key] = RegistryItem(name=key, cls=cls)
            return cls

        return decorator

    def get(self, key: str) -> Type[Any]:
        if key not in self._items:
            raise KeyError(f"알 수 없는 {self.name} 키: '{key}'")
        return self._items[key].cls

    def build(self, key: str, **kwargs: Any) -> Any:
        cls = self.get(key)
        return cls(**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items.keys())
