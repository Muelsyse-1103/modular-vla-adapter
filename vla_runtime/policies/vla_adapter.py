"""Rollout policy wrapper for `prismatic_adapter` models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from prismatic_adapter.inference import ActionPredictor
from prismatic_adapter.types import AdapterBatch
from vla_runtime.policies.base import RolloutPolicy


@dataclass(frozen=True)
class ObservationBatchConfig:
    image_keys: tuple[str, ...] = ("image_primary", "image_wrist")
    image_size: int = 224
    prompt_template: str = "What action should the robot take to {instruction}?"
    action_query_tokens: int = 64
    placeholder_token_id: int | None = None
    add_special_tokens: bool = True
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    proprio_key: str = "proprio"


class ObservationBatchBuilder:
    """Convert remote-env observations into single-sample `AdapterBatch` objects."""

    def __init__(
        self,
        tokenizer,
        config: ObservationBatchConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or ObservationBatchConfig()
        self.device = torch.device(device)

    def __call__(self, observation: dict[str, Any], instruction: str) -> AdapterBatch:
        input_ids, attention_mask, action_mask = self._encode_instruction(instruction)
        pixel_values = self._prepare_images(observation)
        proprio = self._prepare_proprio(observation)
        return AdapterBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            action_mask=action_mask,
            proprio=proprio,
            metadata={"instruction": instruction, "image_keys": self.config.image_keys},
        )

    def _encode_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self.config.prompt_template.format(instruction=instruction)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.config.add_special_tokens,
        )
        input_ids = _encoded_value(encoded, "input_ids").to(self.device)
        attention_mask = _encoded_value(encoded, "attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)

        placeholder_id = self._placeholder_token_id()
        placeholders = torch.full(
            (input_ids.shape[0], self.config.action_query_tokens),
            placeholder_id,
            dtype=input_ids.dtype,
            device=self.device,
        )
        placeholder_attention = torch.ones_like(placeholders)
        input_ids = torch.cat([input_ids, placeholders], dim=1)
        attention_mask = torch.cat([attention_mask, placeholder_attention], dim=1)

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

    def _prepare_images(self, observation: dict[str, Any]) -> torch.Tensor:
        views = []
        for key in self.config.image_keys:
            if key not in observation:
                raise KeyError(f"observation is missing image key: {key}")
            views.append(self._prepare_image(observation[key]))
        return torch.stack(views, dim=1) if len(views) > 1 else views[0]

    def _prepare_image(self, image: Any) -> torch.Tensor:
        tensor = _as_tensor(image).float()
        if tensor.ndim != 3:
            raise ValueError(f"image must have shape HWC or CHW, got {tuple(tensor.shape)}")
        if tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(self.config.image_mean, dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.config.image_std, dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.to(self.device)

    def _prepare_proprio(self, observation: dict[str, Any]) -> torch.Tensor | None:
        if self.config.proprio_key not in observation:
            return None
        proprio = _as_tensor(observation[self.config.proprio_key]).float().reshape(1, -1)
        return proprio.to(self.device)


class VLAAdapterRolloutPolicy(RolloutPolicy):
    """Run a `prismatic_adapter` policy against observations from `RemoteEnvClient`."""

    def __init__(
        self,
        predictor: ActionPredictor,
        batch_builder: ObservationBatchBuilder,
        device: str | torch.device = "cpu",
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.predictor = predictor
        self.batch_builder = batch_builder
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype
        self.predictor.model.to(self.device)
        self.predictor.model.eval()

    def reset(self) -> None:
        self.predictor.model.eval()

    def act(self, observation: dict[str, Any], instruction: str) -> list[list[float]]:
        batch = self.batch_builder(observation, instruction)
        with _maybe_autocast(self.device, self.amp_dtype):
            output = self.predictor.predict(batch)
        actions = output.actions.detach().float().cpu()
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"expected action chunk [H, A], got {tuple(actions.shape)}")
        return actions.tolist()


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


class _maybe_autocast:
    def __init__(self, device: torch.device, dtype: torch.dtype | None) -> None:
        self.device = device
        self.dtype = dtype
        self._context = None

    def __enter__(self):
        if self.dtype is None or self.device.type == "cpu":
            return None
        self._context = torch.autocast(device_type=self.device.type, dtype=self.dtype)
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, traceback):
        if self._context is None:
            return None
        return self._context.__exit__(exc_type, exc, traceback)
