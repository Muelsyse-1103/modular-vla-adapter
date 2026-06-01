"""ZMQ REP server wrapping an `EnvBackend`."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import Callable
from typing import Any

from env_process.backends.base import EnvBackend
from env_process.codecs import encode_array, encode_observation
from env_process.protocols import error, ok


class EnvZmqServer:
    """Single-process environment server.

    The server supports multiple episode ids, but handles requests
    synchronously. That keeps the first version predictable and easy to debug.
    """

    def __init__(
        self,
        backend_factory: Callable[[], EnvBackend],
        endpoint: str = "tcp://127.0.0.1:5555",
    ) -> None:
        self.backend_factory = backend_factory
        self.endpoint = endpoint
        self.backends: dict[str, EnvBackend] = {}
        self._running = False

    def serve_forever(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("EnvZmqServer requires pyzmq; install `pyzmq` in env process.") from exc

        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(self.endpoint)
        self._running = True
        print(f"[env_process] listening on {self.endpoint}")

        try:
            while self._running:
                message = socket.recv_json()
                response = self.handle(message)
                socket.send_json(response)
        finally:
            for backend in self.backends.values():
                backend.close()
            socket.close(linger=0)

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            message_type = message.get("type")
            if message_type == "HELLO":
                return ok("HELLO", name="env_process", protocol_version=1)
            if message_type == "LIST_TASKS":
                backend = self.backend_factory()
                try:
                    return ok("LIST_TASKS", tasks=[task.to_dict() for task in backend.list_tasks()])
                finally:
                    backend.close()
            if message_type == "RESET":
                return self._reset(message)
            if message_type == "STEP":
                return self._step(message)
            if message_type == "RENDER":
                return self._render(message)
            if message_type == "CLOSE":
                return self._close(message)
            return error(f"unknown message type: {message_type}")
        except Exception as exc:  # pragma: no cover - defensive server boundary
            return error(str(exc), traceback=traceback.format_exc())

    def _backend(self, episode_id: str) -> EnvBackend:
        if episode_id not in self.backends:
            raise KeyError(f"unknown episode_id: {episode_id}")
        return self.backends[episode_id]

    def _reset(self, message: dict[str, Any]) -> dict[str, Any]:
        episode_id = message.get("episode_id") or str(uuid.uuid4())
        if episode_id in self.backends:
            self.backends[episode_id].close()
        backend = self.backend_factory()
        self.backends[episode_id] = backend
        reset_kwargs = {
            key: message[key]
            for key in (
                "initial_state_index",
                "initial_states_path",
                "max_steps",
                "num_steps_wait",
            )
            if key in message
        }
        obs = backend.reset(
            task_id=int(message.get("task_id", 0)),
            instruction=message.get("instruction"),
            seed=message.get("seed"),
            **reset_kwargs,
        )
        return ok(
            "RESET",
            episode_id=episode_id,
            instruction=obs.instruction,
            observation=encode_observation(obs.observation),
            info=obs.info,
        )

    def _step(self, message: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(message["episode_id"])
        action = message["action"]
        result = self._backend(episode_id).step(action)
        return ok(
            "STEP",
            episode_id=episode_id,
            observation=encode_observation(result.observation),
            reward=result.reward,
            done=result.done,
            success=result.success,
            info=result.info,
        )

    def _render(self, message: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(message["episode_id"])
        frames = self._backend(episode_id).render()
        return ok(
            "RENDER",
            episode_id=episode_id,
            frames={key: encode_array(value) for key, value in frames.items()},
        )

    def _close(self, message: dict[str, Any]) -> dict[str, Any]:
        episode_id = message.get("episode_id")
        if episode_id is None:
            for backend in self.backends.values():
                backend.close()
            self.backends.clear()
            self._running = False
            return ok("CLOSE", closed="all")
        episode_id = str(episode_id)
        backend = self.backends.pop(episode_id, None)
        if backend is not None:
            backend.close()
        return ok("CLOSE", episode_id=episode_id)
