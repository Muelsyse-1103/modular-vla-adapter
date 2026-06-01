"""Start a LIBERO environment process over ZMQ."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env_process.backends.libero import LiberoBackend, LiberoBackendConfig, TASK_SUITES
from env_process.clients.zmq_server import EnvZmqServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--libero-root",
        default=None,
        help="Path to the copied LIBERO repo or benchmark root. Example: ./LIBERO",
    )
    parser.add_argument(
        "--libero-config-dir",
        default="outputs/libero_config",
        help="Writable directory for LIBERO_CONFIG_PATH when --libero-root is provided.",
    )
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


def configure_local_libero(libero_root: str | None, config_dir: str) -> None:
    if libero_root is None:
        return

    root = Path(libero_root).expanduser().resolve()
    benchmark_root = _resolve_benchmark_root(root)
    python_root = _resolve_python_root(root, benchmark_root)
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    config_path = Path(config_dir).expanduser().resolve()
    config_path.mkdir(parents=True, exist_ok=True)
    dataset_root = benchmark_root.parent / "datasets"
    payload = "\n".join(
        [
            f"assets: {benchmark_root / 'assets'}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"benchmark_root: {benchmark_root}",
            f"datasets: {dataset_root}",
            f"init_states: {benchmark_root / 'init_files'}",
            "",
        ]
    )
    (config_path / "config.yaml").write_text(payload, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path)


def _resolve_benchmark_root(root: Path) -> Path:
    candidates = [
        root,
        root / "libero" / "libero",
        root / "LIBERO" / "libero" / "libero",
    ]
    for candidate in candidates:
        if (candidate / "bddl_files").exists() and (candidate / "init_files").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find LIBERO bddl_files/init_files under {root}. "
        "Pass either the LIBERO repo root or the inner libero/libero benchmark root."
    )


def _resolve_python_root(root: Path, benchmark_root: Path) -> Path:
    if (root / "libero" / "libero" / "__init__.py").exists():
        return root
    if benchmark_root.name == "libero" and benchmark_root.parent.name == "libero":
        return benchmark_root.parents[1]
    return root


def main() -> None:
    args = parse_args()
    configure_local_libero(args.libero_root, args.libero_config_dir)
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
