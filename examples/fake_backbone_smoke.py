"""Smoke example showing how little a new backbone adapter must provide."""

from __future__ import annotations

import torch
import torch.nn as nn

from prismatic_adapter import (
    AdapterConfig,
    ConditioningConfig,
    PolicyConfig,
    PrismaticAdapterPolicy,
    SequenceConfig,
)
from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.sequence import build_multimodal_embeddings, replace_masked_embeddings
from prismatic_adapter.types import AdapterBatch, BackboneOutput


class TinyBackboneAdapter(BackboneAdapter):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.vision = nn.Linear(16, hidden_size)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True) for _ in range(layers)]
        )

    def forward_with_action_queries(
        self,
        batch: AdapterBatch,
        action_queries: torch.Tensor,
    ) -> BackboneOutput:
        embeddings = self.embed(batch.input_ids)
        embeddings = replace_masked_embeddings(embeddings, batch.action_mask, action_queries)
        vision_tokens = self.vision(batch.pixel_values)
        fused, fused_attention, _, segments = build_multimodal_embeddings(
            embeddings,
            vision_tokens,
            batch.attention_mask,
            batch.action_mask,
        )
        hidden_states = [fused]
        x = fused
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        return BackboneOutput(hidden_states, segments, fused_attention, vision_tokens)


def main() -> None:
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 12
    query_tokens = 4
    action_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    action_mask[:, -query_tokens:] = True
    batch = AdapterBatch(
        input_ids=torch.randint(0, 128, (batch_size, seq_len)),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        pixel_values=torch.randn(batch_size, 5, 16),
        action_mask=action_mask,
        actions=torch.randn(batch_size, 3, 2),
        proprio=torch.randn(batch_size, 6),
    )
    cfg = AdapterConfig(
        sequence=SequenceConfig(action_query_tokens=query_tokens),
        conditioning=ConditioningConfig(
            num_condition_layers=3,
            raw_token_budget=3,
            projection="linear",
        ),
        policy=PolicyConfig(hidden_size=16, action_dim=2, action_horizon=3, num_layers=4, num_heads=4),
    )
    model = PrismaticAdapterPolicy(TinyBackboneAdapter(), cfg, proprio_dim=6)
    actions = model(batch)
    print(tuple(actions.shape))


if __name__ == "__main__":
    main()
