"""Start a LIBERO environment process over ZMQ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env_process.backends.libero import LiberoBackend, LiberoBackendConfig, TASK_SUITES
from env_process.clients.zmq_server import EnvZmqServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--task-suite", choices=TASK_SUITES, default="libero_object")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--initial-states-path", default="DEFAULT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--model-family", default="openvla")
    parser.add_argument("--no-normalize-gripper", action="store_true")
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--invert-gripper", choices=("auto", "true", "false"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invert_gripper = None
    if args.invert_gripper == "true":
        invert_gripper = True
    elif args.invert_gripper == "false":
        invert_gripper = False

    config = LiberoBackendConfig(
        task_suite_name=args.task_suite,
        resolution=args.resolution,
        num_steps_wait=args.num_steps_wait,
        initial_states_path=args.initial_states_path,
        model_family=args.model_family,
        normalize_gripper=not args.no_normalize_gripper,
        binarize_gripper=not args.no_binarize_gripper,
        invert_gripper=invert_gripper,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    server = EnvZmqServer(
        backend_factory=lambda: LiberoBackend(config),
        endpoint=args.endpoint,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
