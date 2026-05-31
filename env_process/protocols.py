"""ZMQ message protocol shared by env process and runtime client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MessageType = Literal["HELLO", "LIST_TASKS", "RESET", "STEP", "RENDER", "CLOSE", "ERROR"]


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    instruction: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=int(data["task_id"]),
            instruction=str(data["instruction"]),
            metadata=dict(data.get("metadata", {})),
        )


def request(message_type: MessageType, **payload: Any) -> dict[str, Any]:
    return {"type": message_type, **payload}


def ok(message_type: MessageType, **payload: Any) -> dict[str, Any]:
    return {"ok": True, "type": message_type, **payload}


def error(message: str, **payload: Any) -> dict[str, Any]:
    return {"ok": False, "type": "ERROR", "error": message, **payload}


def require_type(message: dict[str, Any], expected: MessageType) -> None:
    actual = message.get("type")
    if actual != expected:
        raise ValueError(f"expected message type {expected}, got {actual}")
