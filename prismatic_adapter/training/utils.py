"""Training utilities."""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from typing import Any

import torch

from prismatic_adapter.types import AdapterBatch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    return value


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def move_batch_to_device(batch: AdapterBatch, device: torch.device) -> AdapterBatch:
    return AdapterBatch(
        input_ids=move_to_device(batch.input_ids, device),
        attention_mask=move_to_device(batch.attention_mask, device),
        pixel_values=move_to_device(batch.pixel_values, device),
        action_mask=move_to_device(batch.action_mask, device),
        actions=move_to_device(batch.actions, device) if batch.actions is not None else None,
        labels=move_to_device(batch.labels, device) if batch.labels is not None else None,
        proprio=move_to_device(batch.proprio, device) if batch.proprio is not None else None,
        metadata=batch.metadata,
    )


def autocast_dtype(name: str) -> torch.dtype | None:
    if name == "none":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported amp dtype: {name}")
