"""Evaluation runner over a remote env process."""

from __future__ import annotations

from dataclasses import dataclass

from env_process.protocols import TaskSpec
from vla_runtime.recorder import EpisodeRecorder
from vla_runtime.rollouts.remote_rollout import RemoteRolloutRunner


@dataclass
class EvalSummary:
    episodes: int
    successes: int
    success_rate: float
    average_steps: float

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "average_steps": self.average_steps,
        }


class RemoteEvalRunner:
    def __init__(
        self,
        rollout_runner: RemoteRolloutRunner,
        recorder: EpisodeRecorder | None = None,
    ) -> None:
        self.rollout_runner = rollout_runner
        self.recorder = recorder

    def run(
        self,
        tasks: list[TaskSpec],
        trials_per_task: int = 1,
        seed: int = 0,
    ) -> EvalSummary:
        successes = 0
        total_steps = 0
        episodes = 0
        for task in tasks:
            for trial in range(trials_per_task):
                episode = self.rollout_runner.run_episode(
                    task_id=task.task_id,
                    instruction=task.instruction,
                    seed=seed + trial,
                )
                episodes += 1
                successes += int(episode.success)
                total_steps += episode.steps
                if self.recorder is not None:
                    self.recorder.write_episode(episode)

        summary = EvalSummary(
            episodes=episodes,
            successes=successes,
            success_rate=successes / max(episodes, 1),
            average_steps=total_steps / max(episodes, 1),
        )
        if self.recorder is not None:
            self.recorder.write_metrics(summary.to_dict())
        return summary
