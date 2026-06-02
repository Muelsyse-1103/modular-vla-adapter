"""Dataset-facing interfaces and collation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch

from prismatic_adapter.types import AdapterBatch


class SampleAdapter(ABC):
    """Convert one dataset-native sample into an `AdapterBatch`.

    RLDS, LIBERO, CALVIN, ALOHA, and custom PyTorch datasets can each keep their
    own storage format. Only this small adapter has to know those details.
    """

    @abstractmethod
    def __call__(self, sample: Any) -> AdapterBatch:
        raise NotImplementedError


class PaddedBatchCollator:
    """Pad variable-length language fields and stack fixed-size tensors."""

    def __init__(
        self,
        pad_token_id: int,
        padding_side: str = "right",
    ) -> None:
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def _pad_1d(self, values: Sequence[torch.Tensor], pad_value: int | bool) -> torch.Tensor:
        values = [value.squeeze(0) if value.ndim == 2 and value.shape[0] == 1 else value for value in values]
        max_len = max(value.shape[0] for value in values)
        padded = []
        for value in values:
            pad_len = max_len - value.shape[0]
            pad = torch.full((pad_len,), pad_value, dtype=value.dtype, device=value.device)
            padded.append(torch.cat([value, pad], dim=0) if self.padding_side == "right" else torch.cat([pad, value], dim=0))
        return torch.stack(padded, dim=0)

    def __call__(self, items: Sequence[AdapterBatch]) -> AdapterBatch:
        input_ids = self._pad_1d([item.input_ids for item in items], self.pad_token_id)
        attention_mask = self._pad_1d([item.attention_mask for item in items], 0)
        action_mask = self._pad_1d([item.action_mask for item in items], False)

        pixel_values = _collate_nested([item.pixel_values for item in items])

        actions = None
        if all(item.actions is not None for item in items):
            actions = torch.stack([item.actions for item in items if item.actions is not None], dim=0)

        proprio = None
        if all(item.proprio is not None for item in items):
            proprio = torch.stack([item.proprio for item in items if item.proprio is not None], dim=0)

        labels = None
        if all(item.labels is not None for item in items):
            labels = self._pad_1d([item.labels for item in items if item.labels is not None], -100)

        return AdapterBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            action_mask=action_mask,
            actions=actions,
            labels=labels,
            proprio=proprio,
            metadata={"items": [item.metadata for item in items]},
        )


def _collate_nested(values: Sequence[Any]) -> Any:
    if all(isinstance(value, torch.Tensor) for value in values):
        return torch.stack(values, dim=0)
    if all(isinstance(value, dict) for value in values):
        keys = tuple(values[0].keys())
        if not all(tuple(value.keys()) == keys for value in values):
            return list(values)
        return {key: _collate_nested([value[key] for value in values]) for key in keys}
    return list(values)
