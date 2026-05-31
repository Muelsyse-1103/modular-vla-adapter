"""Episode recording containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Transition:
    step: int
    action: list[float]
    reward: float
    done: bool
    success: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    episode_id: str
    task_id: int
    instruction: str
    success: bool
    steps: int
    total_reward: float
    transitions: list[Transition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "success": self.success,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "transitions": [transition.__dict__ for transition in self.transitions],
        }
