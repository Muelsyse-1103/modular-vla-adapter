"""MiniCPM-V processor adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from prismatic_adapter.types import AdapterBatch


@dataclass(frozen=True)
class MiniCPMProcessorConfig:
    image_keys: tuple[str, ...] = ("image_primary", "image_wrist")
    action_query_tokens: int = 64
    placeholder_token_id: int | None = None
    prompt_template: str = "What action should the robot take to {instruction}?"
    downsample_mode: str = "16x"
    max_slice_nums: int = 1
    add_generation_prompt: bool = True
    proprio_key: str = "proprio"


class MiniCPMVBatchProcessor:
    """Build MiniCPM-V `AdapterBatch` samples from raw observations."""

    def __init__(
        self,
        processor: Any,
        config: MiniCPMProcessorConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.processor = processor
        self.config = config or MiniCPMProcessorConfig()
        self.device = torch.device(device) if device is not None else None

    def __call__(
        self,
        sample: Mapping[str, Any],
        instruction: str | None = None,
        actions: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AdapterBatch:
        instruction = str(sample["instruction"] if instruction is None else instruction)
        actions = sample.get("actions") if actions is None else actions
        encoded = self._encode(sample, instruction)
        input_ids = encoded.pop("input_ids")
        attention_mask = encoded.pop("attention_mask", torch.ones_like(input_ids))
        input_ids, attention_mask, action_mask = self._append_action_queries(input_ids, attention_mask)
        proprio = self._prepare_proprio(sample)
        return AdapterBatch(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            pixel_values={key: self._squeeze_batch(value) for key, value in encoded.items()},
            action_mask=action_mask.squeeze(0),
            actions=as_tensor(actions).float() if actions is not None else None,
            proprio=proprio.squeeze(0) if proprio is not None else None,
            metadata={
                "instruction": instruction,
                "processor": "minicpm_v",
                "downsample_mode": self.config.downsample_mode,
                **dict(metadata or {}),
            },
        )

    def _encode(self, sample: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        content = []
        for key in self.config.image_keys:
            if key not in sample:
                raise KeyError(f"sample is missing image key: {key}")
            content.append({"type": "image", "image": to_pil_image(sample[key])})
        content.append({"type": "text", "text": self.config.prompt_template.format(instruction=instruction)})
        messages = [{"role": "user", "content": content}]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.config.add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=self.config.downsample_mode,
            max_slice_nums=self.config.max_slice_nums,
        )
        result = dict(encoded)
        if self.device is not None:
            result = {key: _to_device(value, self.device) for key, value in result.items()}
        return result

    def _append_action_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        placeholders = torch.full(
            (input_ids.shape[0], self.config.action_query_tokens),
            self.placeholder_id(),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, placeholders], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(placeholders)], dim=1)
        action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_mask[:, -self.config.action_query_tokens :] = True
        return input_ids, attention_mask, action_mask

    def placeholder_id(self) -> int:
        if self.config.placeholder_token_id is not None:
            return int(self.config.placeholder_token_id)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        for name in ("pad_token_id", "eos_token_id", "unk_token_id"):
            value = getattr(tokenizer, name, None)
            if value is not None:
                return int(value)
        return 0

    def _prepare_proprio(self, sample: Mapping[str, Any]) -> torch.Tensor | None:
        if self.config.proprio_key not in sample:
            return None
        proprio = as_tensor(sample[self.config.proprio_key]).float().reshape(1, -1)
        return proprio.to(self.device) if self.device is not None else proprio

    @staticmethod
    def _squeeze_batch(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.shape[:1] == (1,):
            return value.squeeze(0)
        return value


def to_pil_image(image: Any) -> Image.Image:
    array = as_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"image must have shape HWC or CHW, got {array.shape}")
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, Sequence) and not isinstance(value, str):
        return np.asarray(value)
    raise TypeError(f"cannot convert {type(value)!r} to numpy array")


def as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return torch.tensor(value)
    raise TypeError(f"cannot convert {type(value)!r} to tensor")


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value
