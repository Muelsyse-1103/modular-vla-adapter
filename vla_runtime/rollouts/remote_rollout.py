"""Remote rollout loop over a ZMQ env process."""

from __future__ import annotations

from collections import deque

from vla_runtime.buffers.episode import EpisodeResult, Transition
from vla_runtime.env_client import RemoteEnvClient
from vla_runtime.policies.base import RolloutPolicy


class RemoteRolloutRunner:
    def __init__(
        self,
        client: RemoteEnvClient,
        policy: RolloutPolicy,
        max_steps: int = 500,
    ) -> None:
        self.client = client
        self.policy = policy
        self.max_steps = max_steps

    def run_episode(
        self,
        task_id: int,
        instruction: str | None = None,
        seed: int | None = None,
    ) -> EpisodeResult:
        reset = self.client.reset(task_id=task_id, instruction=instruction, seed=seed)
        episode_id = reset["episode_id"]
        obs = reset["observation"]
        instruction = reset["instruction"]
        self.policy.reset()

        action_queue: deque[list[float]] = deque()
        transitions: list[Transition] = []
        total_reward = 0.0
        success = False

        for step_idx in range(self.max_steps):
            if not action_queue:
                chunk = self.policy.act(obs, instruction)
                action_queue.extend(chunk)
            action = action_queue.popleft()
            result = self.client.step(episode_id, action)
            obs = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            success = bool(result["success"])
            total_reward += reward
            transitions.append(
                Transition(
                    step=step_idx,
                    action=list(action),
                    reward=reward,
                    done=done,
                    success=success,
                    info=dict(result.get("info", {})),
                )
            )
            if done or success:
                break

        return EpisodeResult(
            episode_id=episode_id,
            task_id=task_id,
            instruction=instruction,
            success=success,
            steps=len(transitions),
            total_reward=total_reward,
            transitions=transitions,
        )
