"""Standard tensor processors for non-chat-template VLA backbones."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from prismatic_adapter.processors.base import PromptProcessor
from prismatic_adapter.types import AdapterBatch


@dataclass(frozen=True)
class StandardProcessorConfig:
    image_keys: tuple[str, ...] = ("image_primary", "image_wrist")
    image_size: int = 224
    prompt_template: str = "What action should the robot take to {instruction}?"
    action_query_tokens: int = 64
    placeholder_token_id: int | None = None
    add_special_tokens: bool = True
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    proprio_key: str = "proprio"


class StandardBatchProcessor:
    """Build `AdapterBatch` objects from simple image/proprio mappings."""

    def __init__(
        self,
        tokenizer: Any,
        config: StandardProcessorConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or StandardProcessorConfig()
        self.device = torch.device(device) if device is not None else None
        self.prompt = PromptProcessor(
            tokenizer=tokenizer,
            prompt_template=self.config.prompt_template,
            action_query_tokens=self.config.action_query_tokens,
            placeholder_token_id=self.config.placeholder_token_id,
            add_special_tokens=self.config.add_special_tokens,
        )

    def __call__(
        self,
        sample: Mapping[str, Any],
        instruction: str,
        actions: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AdapterBatch:
        input_ids, attention_mask, action_mask = self.prompt.encode(
            instruction,
            device=self.device,
        )
        proprio = self.prepare_proprio(sample)
        return AdapterBatch(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            pixel_values=self.prepare_images(sample).squeeze(0),
            action_mask=action_mask.squeeze(0),
            actions=as_tensor(actions).float() if actions is not None else None,
            proprio=proprio.squeeze(0) if proprio is not None else None,
            metadata=dict(metadata or {}),
        )

    def prepare_images(self, sample: Mapping[str, Any]) -> torch.Tensor:
        views = []
        for key in self.config.image_keys:
            if key not in sample:
                raise KeyError(f"sample is missing image key: {key}")
            views.append(prepare_image(sample[key], self.config))
        pixel_values = torch.stack(views, dim=1) if len(views) > 1 else views[0]
        return pixel_values.to(self.device) if self.device is not None else pixel_values

    def prepare_proprio(self, sample: Mapping[str, Any]) -> torch.Tensor | None:
        if self.config.proprio_key not in sample:
            return None
        proprio = as_tensor(sample[self.config.proprio_key]).float().reshape(1, -1)
        return proprio.to(self.device) if self.device is not None else proprio


def prepare_image(image: Any, config: StandardProcessorConfig) -> torch.Tensor:
    tensor = as_tensor(image).float()
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


def as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return torch.tensor(value)
    raise TypeError(f"cannot convert {type(value)!r} to tensor")
