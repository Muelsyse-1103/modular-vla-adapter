"""LIBERO-style sample adapters for adapter training.

The adapter expects already materialized Python samples, so this module does not
own RLDS/HDF5 loading. Dataset-specific loaders can stay outside the framework
and hand records to `LiberoSampleAdapter`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from prismatic_adapter.components.actions import ActionNormalizer, ActionStats
from prismatic_adapter.data import SampleAdapter
from prismatic_adapter.types import AdapterBatch


@dataclass(frozen=True)
class LiberoSampleKeys:
    instruction: str = "instruction"
    actions: str = "actions"
    proprio: str = "proprio"
    image_primary: str = "image_primary"
    image_wrist: str = "image_wrist"


@dataclass(frozen=True)
class LiberoAdapterConfig:
    image_keys: tuple[str, ...] = ("image_primary", "image_wrist")
    image_size: int = 224
    prompt_template: str = "What action should the robot take to {instruction}?"
    action_query_tokens: int = 64
    placeholder_token_id: int | None = None
    add_special_tokens: bool = True
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class LiberoSampleAdapter(SampleAdapter):
    """Convert a LIBERO-style record into an `AdapterBatch` sample."""

    def __init__(
        self,
        tokenizer,
        config: LiberoAdapterConfig | None = None,
        keys: LiberoSampleKeys | None = None,
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or LiberoAdapterConfig()
        self.keys = keys or LiberoSampleKeys()
        self.action_normalizer = action_normalizer

    def __call__(self, sample: Mapping[str, Any]) -> AdapterBatch:
        instruction = str(sample[self.keys.instruction])
        input_ids, attention_mask, action_mask = self._encode_instruction(instruction)
        actions = _as_tensor(sample[self.keys.actions]).float()
        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)

        return AdapterBatch(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            pixel_values=self._prepare_images(sample).squeeze(0),
            action_mask=action_mask.squeeze(0),
            actions=actions,
            proprio=_as_tensor(sample[self.keys.proprio]).float().reshape(-1),
            metadata={
                "instruction": instruction,
                "image_keys": self.config.image_keys,
            },
        )

    def _encode_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self.config.prompt_template.format(instruction=instruction)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.config.add_special_tokens,
        )
        input_ids = _encoded_value(encoded, "input_ids")
        attention_mask = _encoded_value(encoded, "attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        placeholders = torch.full(
            (1, self.config.action_query_tokens),
            self._placeholder_token_id(),
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, placeholders], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(placeholders)], dim=1)
        action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_mask[:, -self.config.action_query_tokens :] = True
        return input_ids, attention_mask, action_mask

    def _placeholder_token_id(self) -> int:
        if self.config.placeholder_token_id is not None:
            return int(self.config.placeholder_token_id)
        for name in ("pad_token_id", "eos_token_id", "unk_token_id"):
            value = getattr(self.tokenizer, name, None)
            if value is not None:
                return int(value)
        return 0

    def _prepare_images(self, sample: Mapping[str, Any]) -> torch.Tensor:
        views = []
        for key_name in self.config.image_keys:
            sample_key = getattr(self.keys, key_name, key_name)
            if sample_key not in sample:
                raise KeyError(f"sample is missing image key: {sample_key}")
            views.append(_prepare_image(sample[sample_key], self.config))
        return torch.stack(views, dim=1) if len(views) > 1 else views[0]


def compute_action_stats(
    samples: Iterable[Mapping[str, Any]],
    action_key: str = "actions",
    mask: Sequence[bool] | None = None,
) -> ActionStats:
    """Compute bounds stats for action normalization from an iterable of samples."""

    chunks = []
    for sample in samples:
        chunks.append(_as_tensor(sample[action_key]).float().reshape(-1, _as_tensor(sample[action_key]).shape[-1]))
    if not chunks:
        raise ValueError("cannot compute action stats from an empty sample iterable")
    actions = torch.cat(chunks, dim=0)
    stats_mask = torch.tensor(mask, dtype=torch.bool) if mask is not None else None
    return ActionStats(low=actions.amin(dim=0), high=actions.amax(dim=0), mask=stats_mask)


def save_action_stats(stats: ActionStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "low": stats.low.detach().cpu().tolist(),
        "high": stats.high.detach().cpu().tolist(),
    }
    if stats.mask is not None:
        payload["mask"] = stats.mask.detach().cpu().tolist()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _prepare_image(image: Any, config: LiberoAdapterConfig) -> torch.Tensor:
    tensor = _as_tensor(image).float()
    if tensor.ndim != 3:
        raise ValueError(f"image must have shape HWC or CHW, got {tuple(tensor.shape)}")
    if tensor.shape[0] != 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(config.image_size, config.image_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(config.image_mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(config.image_std, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _encoded_value(encoded: Any, key: str):
    if isinstance(encoded, dict):
        return encoded.get(key)
    return getattr(encoded, key, None)


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Sequence):
        return torch.tensor(value)
    raise TypeError(f"cannot convert {type(value)!r} to tensor")
