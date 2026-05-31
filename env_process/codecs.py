"""Serialization helpers for observations, actions, and array payloads."""

from __future__ import annotations

import base64
import io
from typing import Any


def encode_array(array: Any) -> dict[str, Any]:
    """Encode a numpy-like array as base64 `.npy` bytes."""

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Array encoding requires numpy in the env process.") from exc

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return {
        "encoding": "npy.base64",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]):
    """Decode an array encoded by `encode_array`."""

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Array decoding requires numpy in the runtime process.") from exc

    if payload.get("encoding") != "npy.base64":
        raise ValueError(f"unsupported array encoding: {payload.get('encoding')}")
    raw = base64.b64decode(payload["data"].encode("ascii"))
    return np.load(io.BytesIO(raw), allow_pickle=False)


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    encoded = {}
    for key, value in observation.items():
        if key.startswith("image") or key in {"proprio", "state"}:
            encoded[key] = encode_array(value)
        else:
            encoded[key] = value
    return encoded


def decode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    decoded = {}
    for key, value in observation.items():
        if isinstance(value, dict) and value.get("encoding") == "npy.base64":
            decoded[key] = decode_array(value)
        else:
            decoded[key] = value
    return decoded
