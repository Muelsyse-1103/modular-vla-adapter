"""Small data containers shared by the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class SegmentSlices:
    """Token ranges in the fused sequence consumed by the language model."""

    bos: slice
    vision: slice
    text: slice
    action_mask: torch.Tensor


@dataclass
class AdapterBatch:
    """Minimal batch expected by the adapter policy.

    `action_mask` marks action-placeholder positions in the original text sequence
    before visual tokens are inserted. Those positions are replaced by learnable
    ActionQuery embeddings.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Any
    action_mask: torch.Tensor
    actions: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    proprio: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackboneOutput:
    """Output of a backbone adapter before the continuous action policy."""

    hidden_states: Sequence[torch.Tensor]
    segments: SegmentSlices
    fused_attention_mask: torch.Tensor
    projected_vision_tokens: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class LayerCondition:
    """Per-layer condition tensors for the Bridge policy."""

    raw_tokens: torch.Tensor
    action_query_tokens: torch.Tensor
