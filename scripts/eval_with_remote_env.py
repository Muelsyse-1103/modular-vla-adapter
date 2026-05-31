"""Run a smoke evaluation against a remote ZMQ environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_runtime.env_client import RemoteEnvClient
from vla_runtime.policies import ConstantActionPolicy
from vla_runtime.recorder import EpisodeRecorder
from vla_runtime.rollouts import RemoteRolloutRunner
from vla_runtime.runners import RemoteEvalRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", default="outputs/remote_eval_smoke")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = RemoteEnvClient(args.endpoint)
    print(client.hello())
    tasks = client.list_tasks()
    policy = ConstantActionPolicy(action_dim=args.action_dim, horizon=args.action_horizon)
    rollout = RemoteRolloutRunner(client=client, policy=policy, max_steps=args.max_steps)
    runner = RemoteEvalRunner(rollout_runner=rollout, recorder=EpisodeRecorder(args.output_dir))
    summary = runner.run(tasks, trials_per_task=args.trials_per_task)
    print(summary.to_dict())
    client.close()
    client.close_socket()


if __name__ == "__main__":
    main()
