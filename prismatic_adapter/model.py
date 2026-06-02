"""Top-level VLA adapter policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.config import AdapterConfig
from prismatic_adapter.components.conditioning import (
    ConditionProjector,
    LayerSelector,
    MeanPoolTokenCompressor,
)
from prismatic_adapter.policy.bridge import BridgeActionHead, ProprioProjector
from prismatic_adapter.sequence import HiddenStateExtractor
from prismatic_adapter.types import AdapterBatch


class PrismaticAdapterPolicy(nn.Module):
    """Compose a VLM backbone adapter with ActionQuery and Bridge policy."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        config: AdapterConfig,
        proprio_dim: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()

        self.backbone = backbone
        self.config = config
        self.action_queries = nn.Parameter(
            torch.zeros(config.sequence.action_query_tokens, backbone.hidden_size)
        )
        self.layer_selector = LayerSelector(config.conditioning)
        self.hidden_extractor = HiddenStateExtractor(include_embedding_state=True)
        self.condition_projector = ConditionProjector(
            input_dim=backbone.hidden_size,
            output_dim=config.policy.hidden_size,
            mode=config.conditioning.projection,
        )
        self.raw_token_compressor = MeanPoolTokenCompressor(
            token_budget=(
                config.conditioning.raw_token_budget
                if config.conditioning.raw_compression == "mean_pool"
                else None
            )
        )
        self.action_head = BridgeActionHead(
            hidden_size=config.policy.hidden_size,
            action_dim=config.policy.action_dim,
            action_horizon=config.policy.action_horizon,
            num_layers=config.policy.num_layers,
            num_heads=config.policy.num_heads,
            dropout=config.policy.dropout,
            use_rope=config.policy.use_rope,
            gate_raw_branch=config.policy.gate_raw_branch,
            ffn_multiplier=config.policy.ffn_multiplier,
        )
        self.proprio_projector = (
            ProprioProjector(proprio_dim, config.policy.hidden_size)
            if proprio_dim is not None
            else None
        )
        self.configure_trainable_parameters()

    def configure_trainable_parameters(self) -> None:
        trainable = self.config.resolved_trainable()
        self.requires_grad_(False)
        found_language = self._set_backbone_module_trainable(
            "language_model",
            trainable.language_model,
        )
        found_vision = self._set_backbone_module_trainable(
            "vision_backbone",
            trainable.vision_backbone,
        )
        if not found_language and not found_vision and (
            trainable.language_model or trainable.vision_backbone
        ):
            self.backbone.requires_grad_(True)
        for module in self.backbone.adapter_modules():
            module.requires_grad_(trainable.vision_projector)
        self.action_queries.requires_grad_(trainable.action_queries)
        self.condition_projector.requires_grad_(trainable.conditioning)
        self.action_head.requires_grad_(trainable.action_head)
        if self.proprio_projector is not None:
            self.proprio_projector.requires_grad_(trainable.proprio_projector)

    def _set_backbone_module_trainable(self, name: str, trainable: bool) -> bool:
        module = getattr(self.backbone, name, None)
        if module is not None:
            module.requires_grad_(trainable)
            return True
        return False

    def forward(self, batch: AdapterBatch) -> torch.Tensor:
        backbone_output = self.backbone.forward_with_action_queries(batch, self.action_queries)
        selected_states = self.layer_selector.select(backbone_output.hidden_states)
        condition = self.hidden_extractor(selected_states, backbone_output.segments)
        condition = self.condition_projector(condition)
        condition = self.raw_token_compressor(condition)

        proprio_token = None
        if self.proprio_projector is not None:
            if batch.proprio is None:
                raise ValueError("batch.proprio is required when proprio_projector is configured")
            proprio_token = self.proprio_projector(batch.proprio).unsqueeze(1)

        return self.action_head(
            raw_tokens=condition.raw_tokens,
            action_query_tokens=condition.action_query_tokens,
            proprio_token=proprio_token,
        )
