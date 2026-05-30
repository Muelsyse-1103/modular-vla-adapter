"""A practical adapter for Hugging Face Prismatic/OpenVLA-like models."""

from __future__ import annotations

from typing import Any

import torch

from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.config import SequenceConfig
from prismatic_adapter.sequence import build_multimodal_embeddings, replace_masked_embeddings
from prismatic_adapter.types import AdapterBatch, BackboneOutput


class HuggingFacePrismaticAdapter(BackboneAdapter):
    """Wrap models that expose `vision_backbone`, `projector`, and `language_model`.

    This class intentionally covers the common Prismatic/OpenVLA shape without
    pretending that every VLM is identical. For a model with a different image
    tower or language forward signature, subclass it and override
    `encode_vision` or `run_language_model`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sequence_config: SequenceConfig | None = None,
        force_output_hidden_states: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.sequence_config = sequence_config or SequenceConfig()
        self.force_output_hidden_states = force_output_hidden_states
        self.hidden_size = int(getattr(model, "llm_dim", model.config.text_config.hidden_size))
        self.num_hidden_layers = int(getattr(model.config.text_config, "num_hidden_layers"))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def encode_vision(self, pixel_values: Any) -> torch.Tensor:
        patch_features = self.model.vision_backbone(pixel_values)
        return self.model.projector(patch_features)

    def run_language_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> Any:
        return self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=self.force_output_hidden_states,
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
            action_queries=action_queries,
        )
        vision_tokens = self.encode_vision(batch.pixel_values)
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
