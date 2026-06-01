"""Qwen language backbone plus a standard fused ViT vision stack.

This is the concrete example requested for local `pretrained_models/Qwen3.5-2B`.
It intentionally keeps image preprocessing outside the adapter so LIBERO,
CALVIN, real-robot cameras, and future processors can each provide tensors in
their own preferred way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.config import SequenceConfig
from prismatic_adapter.sequence import build_multimodal_embeddings, replace_masked_embeddings
from prismatic_adapter.types import AdapterBatch, BackboneOutput

DEFAULT_VISION_MODEL_IDS = (
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_so400m_patch14_siglip_224",
)


@dataclass(frozen=True)
class VisionBackboneSpec:
    """One TIMM vision tower in the fused DINOv2/SigLIP stack."""

    model_id: str
    image_size: int = 224


DEFAULT_VISION_BACKBONE_SPECS = (
    VisionBackboneSpec("vit_large_patch14_reg4_dinov2.lvd142m", image_size=224),
    VisionBackboneSpec("vit_so400m_patch14_siglip_224", image_size=224),
)


def _config_attr(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, name):
        return getattr(text_config, name)
    return default


class TimmFusedVisionBackbone(nn.Module):
    """Fuse TIMM ViT towers such as DINOv2 and SigLIP.

    Each tower can own its expected input size. Patch token counts are aligned
    before feature concatenation, so a DINOv2 tower and a SigLIP tower can be
    swapped or resized independently without changing the VLA adapter contract.
    """

    def __init__(
        self,
        model_ids: Sequence[str] | None = None,
        specs: Sequence[VisionBackboneSpec] | None = None,
        pretrained: bool = True,
        num_views: int = 1,
        token_align: str = "interpolate",
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("TimmFusedVisionBackbone requires `timm`; install with `pip install timm`.") from exc

        self.specs = _resolve_vision_specs(model_ids=model_ids, specs=specs)
        if not self.specs:
            raise ValueError("at least one TIMM vision backbone is required")
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        if token_align not in {"interpolate", "truncate", "error"}:
            raise ValueError("token_align must be one of: interpolate, truncate, error")

        self.model_ids = tuple(spec.model_id for spec in self.specs)
        self.num_views = num_views
        self.token_align = token_align
        self.models = nn.ModuleList(
            [
                timm.create_model(spec.model_id, pretrained=pretrained, num_classes=0)
                for spec in self.specs
            ]
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
            tower_tokens = [
                self._features(model, self._resize_for_tower(view, spec))
                for model, spec in zip(self.models, self.specs)
            ]
            view_tokens.append(self._align_and_concat(tower_tokens))
        return torch.cat(view_tokens, dim=1)

    def _resize_for_tower(self, pixel_values: torch.Tensor, spec: VisionBackboneSpec) -> torch.Tensor:
        if pixel_values.shape[-2:] == (spec.image_size, spec.image_size):
            return pixel_values
        return F.interpolate(
            pixel_values,
            size=(spec.image_size, spec.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _align_and_concat(self, tower_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        token_counts = [tokens.shape[1] for tokens in tower_tokens]
        if len(set(token_counts)) == 1:
            return torch.cat(list(tower_tokens), dim=-1)
        if self.token_align == "error":
            raise ValueError(f"vision tower token counts do not match: {token_counts}")

        target = min(token_counts) if self.token_align == "truncate" else max(token_counts)
        aligned = [_align_token_count(tokens, target) for tokens in tower_tokens]
        return torch.cat(aligned, dim=-1)


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
        vision_model_ids: Sequence[str] | None = None,
        vision_specs: Sequence[VisionBackboneSpec] | None = None,
        vision_image_sizes: Sequence[int] | None = None,
        vision_token_align: str = "interpolate",
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
            specs=_resolve_vision_specs(
                model_ids=vision_model_ids,
                specs=vision_specs,
                image_sizes=vision_image_sizes,
            ),
            pretrained=vision_pretrained,
            num_views=num_views,
            token_align=vision_token_align,
        )
        return cls(
            language_model=language_model,
            vision_backbone=vision_backbone,
            sequence_config=sequence_config,
            torch_dtype=torch_dtype,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(input_ids)

    def adapter_modules(self):
        return (self.vision_projector,)

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


def _resolve_vision_specs(
    model_ids: Sequence[str] | None = None,
    specs: Sequence[VisionBackboneSpec] | None = None,
    image_sizes: Sequence[int] | None = None,
) -> tuple[VisionBackboneSpec, ...]:
    if specs is not None:
        if image_sizes is not None:
            raise ValueError("provide either specs or image_sizes, not both")
        return tuple(specs)

    ids = tuple(model_ids or DEFAULT_VISION_MODEL_IDS)
    if image_sizes is None:
        defaults = {spec.model_id: spec.image_size for spec in DEFAULT_VISION_BACKBONE_SPECS}
        return tuple(
            VisionBackboneSpec(model_id=model_id, image_size=defaults.get(model_id, 224))
            for model_id in ids
        )
    if len(image_sizes) != len(ids):
        raise ValueError("vision_image_sizes must have the same length as vision_model_ids")
    return tuple(
        VisionBackboneSpec(model_id=model_id, image_size=int(image_size))
        for model_id, image_size in zip(ids, image_sizes)
    )


def _align_token_count(tokens: torch.Tensor, target: int) -> torch.Tensor:
    if tokens.shape[1] == target:
        return tokens
    if tokens.shape[1] > target:
        return tokens[:, :target]
    return F.interpolate(
        tokens.transpose(1, 2),
        size=target,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
