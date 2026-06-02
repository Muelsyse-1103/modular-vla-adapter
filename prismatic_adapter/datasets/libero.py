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

import torch

from prismatic_adapter.components.actions import ActionNormalizer, ActionStats
from prismatic_adapter.data import SampleAdapter
from prismatic_adapter.processors.standard import StandardBatchProcessor, StandardProcessorConfig, as_tensor
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
        self.processor = StandardBatchProcessor(
            tokenizer=tokenizer,
            config=StandardProcessorConfig(
                image_keys=self.config.image_keys,
                image_size=self.config.image_size,
                prompt_template=self.config.prompt_template,
                action_query_tokens=self.config.action_query_tokens,
                placeholder_token_id=self.config.placeholder_token_id,
                add_special_tokens=self.config.add_special_tokens,
                image_mean=self.config.image_mean,
                image_std=self.config.image_std,
            ),
        )

    def __call__(self, sample: Mapping[str, Any]) -> AdapterBatch:
        instruction = str(sample[self.keys.instruction])
        actions = as_tensor(sample[self.keys.actions]).float()
        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)
        normalized_sample = {"proprio": sample[self.keys.proprio]}
        for image_key in self.config.image_keys:
            sample_key = getattr(self.keys, image_key, image_key)
            normalized_sample[image_key] = sample[sample_key]
        return self.processor(
            normalized_sample,
            instruction=instruction,
            actions=actions,
            metadata={
                "instruction": instruction,
                "image_keys": self.config.image_keys,
            },
        )


def compute_action_stats(
    samples: Iterable[Mapping[str, Any]],
    action_key: str = "actions",
    mask: Sequence[bool] | None = None,
) -> ActionStats:
    """Compute bounds stats for action normalization from an iterable of samples."""

    chunks = []
    for sample in samples:
        chunks.append(as_tensor(sample[action_key]).float().reshape(-1, as_tensor(sample[action_key]).shape[-1]))
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
