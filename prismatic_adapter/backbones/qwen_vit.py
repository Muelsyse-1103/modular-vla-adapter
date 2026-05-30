"""Qwen language backbone plus a standard fused ViT vision stack.

This is the concrete example requested for local `pretrained_models/Qwen3.5-2B`.
It intentionally keeps image preprocessing outside the adapter so LIBERO,
CALVIN, real-robot cameras, and future processors can each provide tensors in
their own preferred way.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.config import SequenceConfig
from prismatic_adapter.sequence import build_multimodal_embeddings, replace_masked_embeddings
from prismatic_adapter.types import AdapterBatch, BackboneOutput

DEFAULT_VISION_MODEL_IDS = (
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_so400m_patch14_siglip_224",
)


def _config_attr(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, name):
        return getattr(text_config, name)
    return default


class TimmFusedVisionBackbone(nn.Module):
    """Fuse one or more TIMM ViT models by concatenating patch features."""

    def __init__(
        self,
        model_ids: Sequence[str] = DEFAULT_VISION_MODEL_IDS,
        pretrained: bool = True,
        num_views: int = 1,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("TimmFusedVisionBackbone requires `timm`; install with `pip install timm`.") from exc

        if not model_ids:
            raise ValueError("at least one TIMM vision model id is required")
        if num_views <= 0:
            raise ValueError("num_views must be positive")

        self.model_ids = tuple(model_ids)
        self.num_views = num_views
        self.models = nn.ModuleList(
            [timm.create_model(model_id, pretrained=pretrained, num_classes=0) for model_id in self.model_ids]
        )
        self.embed_dim = sum(int(getattr(model, "embed_dim")) for model in self.models)

    def _features(self, model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "forward_features"):
            features = model.forward_features(pixel_values)
        else:
            features = model(pixel_values)
        if isinstance(features, dict):
            features = features.get("x_norm_patchtokens", features.get("features", next(iter(features.values()))))
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        if features.ndim != 3:
            raise ValueError(f"vision features must be [B, P, D], got {tuple(features.shape)}")
        return features

    def _split_views(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        if pixel_values.ndim == 5:
            if pixel_values.shape[1] != self.num_views:
                raise ValueError("pixel_values view count does not match num_views")
            return [pixel_values[:, view] for view in range(self.num_views)]
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be [B, C, H, W] or [B, V, C, H, W]")
        if self.num_views == 1:
            return [pixel_values]
        expected_channels = 3 * self.num_views
        if pixel_values.shape[1] != expected_channels:
            raise ValueError(f"expected {expected_channels} channels for {self.num_views} RGB views")
        return list(torch.chunk(pixel_values, self.num_views, dim=1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        view_tokens = []
        for view in self._split_views(pixel_values):
            fused = [self._features(model, view) for model in self.models]
            view_tokens.append(torch.cat(fused, dim=-1))
        return torch.cat(view_tokens, dim=1)


class QwenTimmVLAAdapter(BackboneAdapter):
    """Backbone adapter for local Qwen3.5 + fused TIMM ViT features."""

    def __init__(
        self,
        language_model: nn.Module,
        vision_backbone: TimmFusedVisionBackbone,
        sequence_config: SequenceConfig | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.vision_backbone = vision_backbone
        self.sequence_config = sequence_config or SequenceConfig()
        self.torch_dtype = torch_dtype

        self.hidden_size = int(_config_attr(language_model.config, "hidden_size"))
        self.num_hidden_layers = int(_config_attr(language_model.config, "num_hidden_layers"))
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_backbone.embed_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    @classmethod
    def from_pretrained(
        cls,
        qwen_path: str | Path = "pretrained_models/Qwen3.5-2B",
        vision_model_ids: Sequence[str] = DEFAULT_VISION_MODEL_IDS,
        vision_pretrained: bool = True,
        num_views: int = 2,
        sequence_config: SequenceConfig | None = None,
        torch_dtype: torch.dtype | None = torch.bfloat16,
        device_map: str | dict[str, Any] | None = None,
        trust_remote_code: bool = True,
    ) -> "QwenTimmVLAAdapter":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "QwenTimmVLAAdapter.from_pretrained requires `transformers`."
            ) from exc

        language_model = AutoModelForCausalLM.from_pretrained(
            str(qwen_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        vision_backbone = TimmFusedVisionBackbone(
            model_ids=vision_model_ids,
            pretrained=vision_pretrained,
            num_views=num_views,
        )
        return cls(
            language_model=language_model,
            vision_backbone=vision_backbone,
            sequence_config=sequence_config,
            torch_dtype=torch_dtype,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(input_ids)

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_features = self.vision_backbone(pixel_values)
        return self.vision_projector(patch_features.to(dtype=self.vision_projector[0].weight.dtype))

    def run_language_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> Any:
        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward_with_action_queries(
        self,
        batch: AdapterBatch,
        action_queries: torch.Tensor,
    ) -> BackboneOutput:
        input_embeddings = self.embed_input_ids(batch.input_ids)
        input_embeddings = replace_masked_embeddings(
            input_embeddings=input_embeddings,
            action_mask=batch.action_mask,
            action_queries=action_queries.to(device=input_embeddings.device, dtype=input_embeddings.dtype),
        )
        vision_tokens = self.encode_vision(batch.pixel_values).to(
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )
        fused_embeddings, fused_attention, fused_labels, segments = build_multimodal_embeddings(
            input_embeddings=input_embeddings,
            vision_tokens=vision_tokens,
            attention_mask=batch.attention_mask,
            action_mask=batch.action_mask,
            labels=batch.labels,
            cfg=self.sequence_config,
        )
        output = self.run_language_model(fused_embeddings, fused_attention, fused_labels)
        return BackboneOutput(
            hidden_states=output.hidden_states,
            segments=segments,
            fused_attention_mask=fused_attention,
            projected_vision_tokens=vision_tokens,
        )
