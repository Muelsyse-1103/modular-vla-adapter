"""Policy interface for remote rollouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class RolloutPolicy(ABC):
    """Runtime policy that returns an action chunk for an observation."""

    def reset(self) -> None:
        pass

    @abstractmethod
    def act(self, observation: dict[str, Any], instruction: str) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class ConstantActionPolicy(RolloutPolicy):
    action_dim: int = 7
    horizon: int = 1
    value: float = 0.0

    def act(self, observation: dict[str, Any], instruction: str) -> list[list[float]]:
        return [[self.value for _ in range(self.action_dim)] for _ in range(self.horizon)]
