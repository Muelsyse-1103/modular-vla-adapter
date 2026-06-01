"""Abstract environment backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from env_process.protocols import TaskSpec


@dataclass
class EnvObs:
    observation: dict[str, Any]
    instruction: str
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "instruction": self.instruction,
            "info": self.info,
        }


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    success: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "done": self.done,
            "success": self.success,
            "info": self.info,
        }


class EnvBackend(ABC):
    """Backend contract implemented inside the simulator environment."""

    @abstractmethod
    def list_tasks(self) -> list[TaskSpec]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task_id: int, instruction: str | None = None, seed: int | None = None, **kwargs: Any) -> EnvObs:
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> StepResult:
        raise NotImplementedError

    def render(self) -> dict[str, Any]:
        raise NotImplementedError("render is optional for EnvBackend")

    def close(self) -> None:
        pass
