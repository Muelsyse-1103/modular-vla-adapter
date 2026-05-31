"""Dependency-light fake environment for protocol smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env_process.backends.base import EnvBackend, EnvObs, StepResult
from env_process.protocols import TaskSpec


@dataclass
class FakeEnvBackend(EnvBackend):
    max_steps: int = 8
    image_size: int = 64
    action_dim: int = 7

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._step = 0
        self._instruction = "move to the target"

    def list_tasks(self) -> list[TaskSpec]:
        return [
            TaskSpec(task_id=0, instruction="move to the red target"),
            TaskSpec(task_id=1, instruction="move to the blue target"),
        ]

    def reset(self, task_id: int, instruction: str | None = None, seed: int | None = None) -> EnvObs:
        self._rng = np.random.default_rng(seed if seed is not None else task_id)
        self._step = 0
        tasks = {task.task_id: task.instruction for task in self.list_tasks()}
        self._instruction = instruction or tasks.get(task_id, "move to the target")
        return EnvObs(
            observation=self._make_obs(),
            instruction=self._instruction,
            info={"task_id": task_id, "step": self._step},
        )

    def step(self, action) -> StepResult:
        action = np.asarray(action, dtype=np.float32)
        self._step += 1
        done = self._step >= self.max_steps
        success = bool(done and np.isfinite(action).all())
        return StepResult(
            observation=self._make_obs(),
            reward=1.0 if success else 0.0,
            done=done,
            success=success,
            info={"step": self._step, "action_norm": float(np.linalg.norm(action))},
        )

    def render(self) -> dict[str, np.ndarray]:
        return {"image_primary": self._image()}

    def _image(self) -> np.ndarray:
        return self._rng.integers(
            0,
            255,
            size=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )

    def _make_obs(self) -> dict[str, np.ndarray]:
        return {
            "image_primary": self._image(),
            "image_wrist": self._image(),
            "proprio": self._rng.normal(size=(8,)).astype(np.float32),
        }
