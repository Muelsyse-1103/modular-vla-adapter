"""Top-level VLA adapter policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.config import AdapterConfig
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
        if backbone.hidden_size != config.policy.hidden_size:
            raise ValueError("backbone.hidden_size must match config.policy.hidden_size")

        self.backbone = backbone
        self.config = config
        self.action_queries = nn.Parameter(
            torch.zeros(config.sequence.action_query_tokens, config.policy.hidden_size)
        )
        self.hidden_extractor = HiddenStateExtractor(
            include_embedding_state=config.hidden_state_source == "all_states"
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
        self.backbone.requires_grad_(self.config.train_backbone)
        self.action_queries.requires_grad_(self.config.train_action_queries)
        self.action_head.requires_grad_(self.config.train_policy)
        if self.proprio_projector is not None:
            self.proprio_projector.requires_grad_(self.config.train_policy)

    def forward(self, batch: AdapterBatch) -> torch.Tensor:
        backbone_output = self.backbone.forward_with_action_queries(batch, self.action_queries)
        condition = self.hidden_extractor(backbone_output.hidden_states, backbone_output.segments)

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
