"""Configuration objects for the adapter framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SequenceConfig:
    """How visual and action-query tokens are arranged around the language prompt."""

    bos_tokens: int = 1
    action_query_tokens: int = 64
    insert_vision_after_bos: bool = True

    def validate(self) -> None:
        if self.bos_tokens < 0:
            raise ValueError("bos_tokens must be non-negative")
        if self.action_query_tokens <= 0:
            raise ValueError("action_query_tokens must be positive")


@dataclass(frozen=True)
class PolicyConfig:
    """Bridge policy dimensions and behavior."""

    hidden_size: int
    action_dim: int = 7
    action_horizon: int = 8
    num_layers: int = 24
    num_heads: int = 8
    dropout: float = 0.0
    use_rope: bool = True
    gate_raw_branch: bool = True
    ffn_multiplier: int = 4

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class ConditioningConfig:
    """How backbone hidden states are normalized before the Bridge policy."""

    layer_strategy: Literal["all", "uniform", "last", "indices"] = "uniform"
    num_condition_layers: int | None = None
    layer_indices: tuple[int, ...] | None = None
    include_embedding_state: bool = False
    raw_token_budget: int | None = 512
    raw_compression: Literal["none", "mean_pool"] = "mean_pool"
    projection: Literal["identity", "linear"] = "linear"

    def validate(self) -> None:
        if self.layer_strategy not in {"all", "uniform", "last", "indices"}:
            raise ValueError("unsupported layer_strategy")
        if self.num_condition_layers is not None and self.num_condition_layers <= 0:
            raise ValueError("num_condition_layers must be positive")
        if self.layer_strategy == "indices" and not self.layer_indices:
            raise ValueError("layer_indices must be provided when layer_strategy='indices'")
        if self.raw_token_budget is not None and self.raw_token_budget <= 0:
            raise ValueError("raw_token_budget must be positive")
        if self.raw_compression not in {"none", "mean_pool"}:
            raise ValueError("unsupported raw_compression")
        if self.projection not in {"identity", "linear"}:
            raise ValueError("unsupported projection")


@dataclass(frozen=True)
class AdapterConfig:
    """Top-level VLA adapter configuration."""

    sequence: SequenceConfig
    policy: PolicyConfig
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    train_backbone: bool = False
    train_action_queries: bool = True
    train_policy: bool = True

    def validate(self) -> None:
        self.sequence.validate()
        self.policy.validate()
        self.conditioning.validate()
