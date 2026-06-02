"""Rollout policy wrapper for `prismatic_adapter` models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from prismatic_adapter.inference import ActionPredictor
from prismatic_adapter.processors.standard import StandardBatchProcessor, StandardProcessorConfig
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
                proprio_key=self.config.proprio_key,
            ),
            device=self.device,
        )

    def __call__(self, observation: dict[str, Any], instruction: str) -> AdapterBatch:
        sample = self.processor(
            observation,
            instruction=instruction,
            metadata={"instruction": instruction, "image_keys": self.config.image_keys},
        )
        return AdapterBatch(
            input_ids=sample.input_ids.unsqueeze(0),
            attention_mask=sample.attention_mask.unsqueeze(0),
            pixel_values=(
                sample.pixel_values.unsqueeze(0)
                if isinstance(sample.pixel_values, torch.Tensor)
                else sample.pixel_values
            ),
            action_mask=sample.action_mask.unsqueeze(0),
            proprio=sample.proprio.unsqueeze(0) if sample.proprio is not None else None,
            metadata=sample.metadata,
        )


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
