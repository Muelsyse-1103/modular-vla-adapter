"""Start a fake environment process over ZMQ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env_process.backends.fake import FakeEnvBackend
from env_process.clients.zmq_server import EnvZmqServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = EnvZmqServer(
        backend_factory=lambda: FakeEnvBackend(max_steps=args.max_steps, image_size=args.image_size),
        endpoint=args.endpoint,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
