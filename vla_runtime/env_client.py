"""Remote environment client using ZMQ REQ/REP."""

from __future__ import annotations

from typing import Any

from env_process.codecs import decode_observation
from env_process.protocols import TaskSpec, request


class RemoteEnvClient:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", timeout_ms: int = 30_000) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("RemoteEnvClient requires pyzmq; install `pyzmq`.") from exc

        self.zmq = zmq
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(endpoint)
        self.endpoint = endpoint

    def close_socket(self) -> None:
        self.socket.close(linger=0)

    def _send(self, message: dict[str, Any]) -> dict[str, Any]:
        self.socket.send_json(message)
        response = self.socket.recv_json()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "remote env error"))
        return response

    def hello(self) -> dict[str, Any]:
        return self._send(request("HELLO"))

    def list_tasks(self) -> list[TaskSpec]:
        response = self._send(request("LIST_TASKS"))
        return [TaskSpec.from_dict(item) for item in response["tasks"]]

    def reset(
        self,
        task_id: int,
        instruction: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        response = self._send(
            request(
                "RESET",
                task_id=task_id,
                instruction=instruction,
                seed=seed,
                episode_id=episode_id,
            )
        )
        response["observation"] = decode_observation(response["observation"])
        return response

    def step(self, episode_id: str, action) -> dict[str, Any]:
        response = self._send(request("STEP", episode_id=episode_id, action=list(action)))
        response["observation"] = decode_observation(response["observation"])
        return response

    def close(self, episode_id: str | None = None) -> dict[str, Any]:
        return self._send(request("CLOSE", episode_id=episode_id))
