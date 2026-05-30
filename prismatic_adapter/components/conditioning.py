"""Condition adapters for model-size and token-count compatibility."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from prismatic_adapter.config import ConditioningConfig
from prismatic_adapter.types import LayerCondition


@dataclass(frozen=True)
class LayerSelector:
    """Select a stable number of hidden-state layers from any backbone."""

    cfg: ConditioningConfig

    def select(self, hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        start = 0 if self.cfg.include_embedding_state else 1
        available = list(hidden_states[start:])
        if not available:
            raise ValueError("backbone did not return selectable hidden states")

        if self.cfg.layer_strategy == "all":
            return available

        if self.cfg.layer_strategy == "indices":
            assert self.cfg.layer_indices is not None
            selected = []
            for idx in self.cfg.layer_indices:
                resolved = idx if idx >= 0 else len(hidden_states) + idx
                if resolved < 0 or resolved >= len(hidden_states):
                    raise IndexError(f"layer index out of range: {idx}")
                if not self.cfg.include_embedding_state and resolved == 0:
                    raise ValueError("embedding state index 0 is disabled by include_embedding_state=False")
                selected.append(hidden_states[resolved])
            return selected

        count = self.cfg.num_condition_layers or len(available)
        count = min(count, len(available))
        if self.cfg.layer_strategy == "last":
            return available[-count:]

        # Uniform sampling keeps small and large models on the same policy depth.
        if count == 1:
            return [available[-1]]
        positions = torch.linspace(0, len(available) - 1, steps=count).round().to(torch.long).tolist()
        return [available[pos] for pos in positions]


class ConditionProjector(nn.Module):
    """Project backbone hidden size to policy hidden size."""

    def __init__(self, input_dim: int, output_dim: int, mode: str = "linear") -> None:
        super().__init__()
        if input_dim == output_dim and mode == "identity":
            self.proj = nn.Identity()
        elif input_dim == output_dim and mode == "linear":
            self.proj = nn.Identity()
        elif mode == "linear":
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"unsupported projection mode: {mode}")

    def forward(self, condition: LayerCondition) -> LayerCondition:
        return LayerCondition(
            raw_tokens=self.proj(condition.raw_tokens),
            action_query_tokens=self.proj(condition.action_query_tokens),
        )


class IdentityProjector(ConditionProjector):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.proj = nn.Identity()


class TokenCompressor(nn.Module):
    """Base class for Raw token compressors."""

    def forward(self, condition: LayerCondition) -> LayerCondition:
        raise NotImplementedError


class MeanPoolTokenCompressor(TokenCompressor):
    """Compress Raw visual tokens to a fixed budget by contiguous mean pooling."""

    def __init__(self, token_budget: int | None) -> None:
        super().__init__()
        self.token_budget = token_budget

    def forward(self, condition: LayerCondition) -> LayerCondition:
        if self.token_budget is None:
            return condition
        raw = condition.raw_tokens
        if raw.shape[2] <= self.token_budget:
            return condition

        batch_size, layers, tokens, hidden = raw.shape
        groups = self.token_budget
        padded_tokens = math.ceil(tokens / groups) * groups
        if padded_tokens != tokens:
            pad = raw[:, :, -1:].expand(batch_size, layers, padded_tokens - tokens, hidden)
            raw = torch.cat([raw, pad], dim=2)
        raw = raw.reshape(batch_size, layers, groups, padded_tokens // groups, hidden).mean(dim=3)
        return LayerCondition(raw_tokens=raw, action_query_tokens=condition.action_query_tokens)
