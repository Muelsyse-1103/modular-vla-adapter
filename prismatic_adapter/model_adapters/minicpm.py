"""MiniCPM-V model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from prismatic_adapter.config import SequenceConfig
from prismatic_adapter.model_adapters.base import BackboneAdapter
from prismatic_adapter.sequence import replace_masked_embeddings
from prismatic_adapter.types import AdapterBatch, BackboneOutput, SegmentSlices


class MiniCPMVLAAdapter(BackboneAdapter):
    """Adapter for MiniCPM-V style `AutoModelForImageTextToText` backbones."""

    def __init__(
        self,
        model: nn.Module,
        sequence_config: SequenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.sequence_config = sequence_config or SequenceConfig()
        self.hidden_size = int(_config_attr(model.config, "hidden_size"))
        self.num_hidden_layers = int(_config_attr(model.config, "num_hidden_layers"))
        self.image_token_id = int(getattr(model.config, "image_token_id"))

    @property
    def language_model(self) -> nn.Module:
        return self.model.model.language_model

    @language_model.setter
    def language_model(self, module: nn.Module) -> None:
        self.model.model.language_model = module

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = "pretrained_models/MiniCPM-V-4.6",
        sequence_config: SequenceConfig | None = None,
        torch_dtype: torch.dtype | str | None = "auto",
        device_map: str | dict[str, Any] | None = None,
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
    ) -> "MiniCPMVLAAdapter":
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError(
                "MiniCPMVLAAdapter requires transformers with AutoModelForImageTextToText."
            ) from exc

        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
        return cls(model=model, sequence_config=sequence_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward_with_action_queries(
        self,
        batch: AdapterBatch,
        action_queries: torch.Tensor,
    ) -> BackboneOutput:
        if not isinstance(batch.pixel_values, dict):
            raise TypeError("MiniCPMVLAAdapter expects batch.pixel_values to be a dict")
        input_embeddings = self.embed_input_ids(batch.input_ids)
        input_embeddings = replace_masked_embeddings(
            input_embeddings=input_embeddings,
            action_mask=batch.action_mask,
            action_queries=action_queries.to(device=input_embeddings.device, dtype=input_embeddings.dtype),
        )
        image_inputs = _normalize_image_inputs(batch.pixel_values)
        output = self.model(
            input_ids=batch.input_ids,
            inputs_embeds=input_embeddings,
            attention_mask=batch.attention_mask,
            pixel_values=image_inputs.get("pixel_values"),
            target_sizes=image_inputs.get("target_sizes"),
            pixel_values_videos=image_inputs.get("pixel_values_videos"),
            target_sizes_videos=image_inputs.get("target_sizes_videos"),
            downsample_mode=image_inputs.get("downsample_mode"),
            labels=batch.labels,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        segments = _segments_from_minicpm_ids(
            input_ids=batch.input_ids,
            action_mask=batch.action_mask,
            image_token_id=self.image_token_id,
            bos_tokens=self.sequence_config.bos_tokens,
        )
        return BackboneOutput(
            hidden_states=output.hidden_states,
            segments=segments,
            fused_attention_mask=batch.attention_mask,
        )


def _config_attr(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, name):
        return getattr(text_config, name)
    return default


def _normalize_image_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    for key in ("downsample_mode",):
        value = normalized.get(key)
        if isinstance(value, list) and value:
            normalized[key] = value[0]
    return normalized


def _segments_from_minicpm_ids(
    input_ids: torch.Tensor,
    action_mask: torch.Tensor,
    image_token_id: int,
    bos_tokens: int,
) -> SegmentSlices:
    image_positions = torch.where(input_ids[0] == image_token_id)[0]
    if image_positions.numel() == 0:
        vision = slice(bos_tokens, bos_tokens)
    else:
        vision = slice(int(image_positions.min()), int(image_positions.max()) + 1)
    return SegmentSlices(
        bos=slice(0, bos_tokens),
        vision=vision,
        text=slice(vision.stop, input_ids.shape[1]),
        action_mask=action_mask,
    )
