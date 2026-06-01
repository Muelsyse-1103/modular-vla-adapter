"""LIBERO backend for the remote environment process.

This module intentionally keeps LIBERO imports lazy. The training/model Python
environment can import `env_process` without having MuJoCo, robosuite, or
LIBERO installed; only the separate environment process needs those packages.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from env_process.backends.base import EnvBackend, EnvObs, StepResult
from env_process.protocols import TaskSpec

TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass(frozen=True)
class LiberoBackendConfig:
    task_suite_name: str = "libero_object"
    resolution: int = 256
    num_steps_wait: int = 10
    initial_states_path: str = "DEFAULT"
    model_family: str = "openvla"
    normalize_gripper: bool = True
    binarize_gripper: bool = True
    invert_gripper: bool | None = None
    seed: int = 0
    max_steps: int | None = None


class LiberoBackend(EnvBackend):
    """A single-episode LIBERO environment backend.

    `EnvZmqServer` creates one backend instance per episode id, so the backend
    can hold the simulator instance and latest observation directly.
    """

    def __init__(self, config: LiberoBackendConfig | None = None) -> None:
        self.config = config or LiberoBackendConfig()
        if self.config.task_suite_name not in TASK_SUITES:
            raise ValueError(f"unknown LIBERO task suite: {self.config.task_suite_name}")

        benchmark, get_libero_path, env_cls = _import_libero()
        self._benchmark = benchmark
        self._get_libero_path = get_libero_path
        self._env_cls = env_cls
        self._task_suite = self._benchmark.get_benchmark_dict()[self.config.task_suite_name]()

        self._env = None
        self._task_id: int | None = None
        self._instruction = ""
        self._step = 0
        self._max_steps = self.config.max_steps or TASK_MAX_STEPS[self.config.task_suite_name]
        self._last_raw_obs: dict[str, Any] | None = None

    def list_tasks(self) -> list[TaskSpec]:
        specs: list[TaskSpec] = []
        for task_id in range(self._task_suite.n_tasks):
            task = self._task_suite.get_task(task_id)
            specs.append(
                TaskSpec(
                    task_id=task_id,
                    instruction=task.language,
                    metadata={
                        "task_suite": self.config.task_suite_name,
                        "problem_folder": task.problem_folder,
                        "bddl_file": task.bddl_file,
                        "max_steps": self.config.max_steps or TASK_MAX_STEPS[self.config.task_suite_name],
                    },
                )
            )
        return specs

    def reset(
        self,
        task_id: int,
        instruction: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> EnvObs:
        self.close()
        self._task_id = int(task_id)
        self._step = 0
        self._max_steps = int(kwargs.get("max_steps") or self.config.max_steps or TASK_MAX_STEPS[self.config.task_suite_name])

        task = self._task_suite.get_task(self._task_id)
        self._env, task_description = self._make_env(task, seed=seed)
        self._instruction = instruction or task_description

        raw_obs = self._env.reset()
        initial_state, initial_state_index = self._select_initial_state(
            task_id=self._task_id,
            seed=seed,
            initial_state_index=kwargs.get("initial_state_index"),
            initial_states_path=str(kwargs.get("initial_states_path") or self.config.initial_states_path),
        )
        if initial_state is not None:
            raw_obs = self._env.set_init_state(initial_state)
        if raw_obs is None:
            raw_obs = self._env.get_observation()

        num_steps_wait = int(kwargs.get("num_steps_wait") if kwargs.get("num_steps_wait") is not None else self.config.num_steps_wait)
        for _ in range(num_steps_wait):
            raw_obs, _, _, _ = self._env.step(_dummy_action())

        self._last_raw_obs = raw_obs
        return EnvObs(
            observation=self._make_observation(raw_obs),
            instruction=self._instruction,
            info={
                "task_id": self._task_id,
                "task_suite": self.config.task_suite_name,
                "step": self._step,
                "max_steps": self._max_steps,
                "initial_state_index": initial_state_index,
                "num_steps_wait": num_steps_wait,
            },
        )

    def step(self, action) -> StepResult:
        if self._env is None:
            raise RuntimeError("LIBERO backend has not been reset")

        env_action = self._prepare_action(action)
        raw_obs, reward, raw_done, info = self._env.step(env_action.tolist())
        self._step += 1

        success = bool(raw_done)
        done = success or self._step >= self._max_steps
        self._last_raw_obs = raw_obs

        step_info = dict(info or {})
        step_info.update(
            {
                "task_id": self._task_id,
                "step": self._step,
                "max_steps": self._max_steps,
                "success": success,
                "timeout": bool(not success and self._step >= self._max_steps),
            }
        )
        return StepResult(
            observation=self._make_observation(raw_obs),
            reward=float(reward if reward is not None else success),
            done=done,
            success=success,
            info=step_info,
        )

    def render(self) -> dict[str, np.ndarray]:
        if self._last_raw_obs is None:
            if self._env is None:
                raise RuntimeError("LIBERO backend has not been reset")
            self._last_raw_obs = self._env.get_observation()

        frames = {"image_primary": _get_primary_image(self._last_raw_obs)}
        if "robot0_eye_in_hand_image" in self._last_raw_obs:
            frames["image_wrist"] = _get_wrist_image(self._last_raw_obs)
        return frames

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._last_raw_obs = None

    def _make_env(self, task, seed: int | None):
        task_bddl_file = os.path.join(
            self._get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        env = self._env_cls(
            bddl_file_name=task_bddl_file,
            camera_heights=self.config.resolution,
            camera_widths=self.config.resolution,
        )
        env.seed(self.config.seed if seed is None else int(seed))
        return env, task.language

    def _select_initial_state(
        self,
        task_id: int,
        seed: int | None,
        initial_state_index: Any,
        initial_states_path: str,
    ) -> tuple[Any | None, int | None]:
        if initial_states_path and initial_states_path != "DEFAULT":
            initial_states = _load_initial_states(initial_states_path, task_id)
        else:
            initial_states = self._load_default_initial_states(task_id)

        if initial_states is None or len(initial_states) == 0:
            return None, None

        if initial_state_index is None:
            index = (int(seed) if seed is not None else 0) % len(initial_states)
        else:
            index = int(initial_state_index) % len(initial_states)
        return initial_states[index], index

    def _load_default_initial_states(self, task_id: int):
        try:
            return self._task_suite.get_task_init_states(task_id)
        except Exception as exc:
            if "Weights only load failed" not in str(exc):
                raise
            task = self._task_suite.get_task(task_id)
            path = os.path.join(
                self._get_libero_path("init_states"),
                task.problem_folder,
                task.init_states_file,
            )
            return _torch_load(path)

    def _make_observation(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        proprio = np.concatenate(
            [
                np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
                _quat2axisangle(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)),
                np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
            ]
        ).astype(np.float32)

        observation = {
            "image_primary": _get_primary_image(raw_obs),
            "proprio": proprio,
            "state": proprio,
        }
        if "robot0_eye_in_hand_image" in raw_obs:
            observation["image_wrist"] = _get_wrist_image(raw_obs)
        return observation

    def _prepare_action(self, action) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if action_array.shape[0] != 7:
            raise ValueError(f"LIBERO expects 7D actions, got shape {action_array.shape}")

        if self.config.normalize_gripper:
            action_array = _normalize_gripper_action(action_array, binarize=self.config.binarize_gripper)

        invert = self.config.invert_gripper
        if invert is None:
            invert = self.config.model_family == "openvla"
        if invert:
            action_array = _invert_gripper_action(action_array)
        return action_array


def _import_libero():
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "LiberoBackend requires LIBERO and simulator dependencies in the env process. "
            "Install LIBERO in the separate environment process, then run "
            "`python scripts/serve_libero_env.py` from that environment."
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv


def _load_initial_states(path: str, task_id: int):
    suffix = os.path.splitext(path)[1]
    if suffix in {".init", ".pruned_init", ".pt", ".pth"}:
        return _torch_load(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if str(task_id) in data:
            return data[str(task_id)]
        if task_id in data:
            return data[task_id]
        raise KeyError(f"initial state file {path} has no task id {task_id}")

    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[task_id]
    return data


def _torch_load(path: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading LIBERO initial states requires torch in the env process.") from exc

    try:
        return torch.load(path)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(path, weights_only=False)


def _dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _get_primary_image(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["agentview_image"])[::-1, ::-1].copy()


def _get_wrist_image(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1].copy()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized = action.copy()
    normalized[..., -1] = 2.0 * normalized[..., -1] - 1.0
    if binarize:
        normalized[..., -1] = np.sign(normalized[..., -1])
    return normalized


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted = action.copy()
    inverted[..., -1] *= -1.0
    return inverted
