"""A tiny registry for backbone adapter factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prismatic_adapter.backbones.base import BackboneAdapter

Factory = Callable[..., BackboneAdapter]


class BackboneRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Factory] = {}

    def register(self, name: str, factory: Factory) -> None:
        if not name:
            raise ValueError("name cannot be empty")
        if name in self._factories:
            raise ValueError(f"backbone adapter already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> BackboneAdapter:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"unknown backbone adapter {name!r}; available: {available}") from exc
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)


BACKBONES = BackboneRegistry()
