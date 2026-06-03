"""Sequence assembly and hidden-state extraction.

The official VLA-Adapter implementation interleaves several concerns in one
Hugging Face model class. This module keeps the contracts explicit:

1. replace action placeholders with learnable ActionQuery embeddings;
2. insert visual tokens after the BOS token;
3. shift the action mask into the fused sequence;
4. extract Raw visual tokens and ActionQuery-aligned states per layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from prismatic_adapter.config import SequenceConfig
from prismatic_adapter.types import LayerCondition, SegmentSlices


def expand_action_queries(action_queries: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return action queries as `[B, Q, D]`."""

    if action_queries.ndim == 2:
        return action_queries.unsqueeze(0).expand(batch_size, -1, -1)
    if action_queries.ndim == 3 and action_queries.shape[0] == batch_size:
        return action_queries
    raise ValueError("action_queries must have shape [Q, D] or [B, Q, D]")


def replace_masked_embeddings(
    input_embeddings: torch.Tensor,
    action_mask: torch.Tensor,
    action_queries: torch.Tensor,
) -> torch.Tensor:
    """Replace masked positions with ActionQuery embeddings.

    Every sample must expose the same number of action-placeholder positions.
    This catches silent prompt/action-horizon mismatches early.
    """

    if input_embeddings.ndim != 3:
        raise ValueError("input_embeddings must have shape [B, S, D]")
    if action_mask.shape != input_embeddings.shape[:2]:
        raise ValueError("action_mask must have shape [B, S]")

    batch_size = input_embeddings.shape[0]
    queries = expand_action_queries(action_queries, batch_size)
    counts = action_mask.to(torch.long).sum(dim=1)
    expected = queries.shape[1]
    if not torch.all(counts == expected):
        raise ValueError(
            f"each sample must have exactly {expected} action positions; got {counts.tolist()}"
        )

    output = input_embeddings.clone()
    batch_ids = torch.arange(batch_size, device=input_embeddings.device).unsqueeze(1)
    action_positions = torch.stack([torch.where(mask)[0] for mask in action_mask], dim=0)
    output[batch_ids, action_positions] = queries
    return output


def shift_mask_after_vision_insert(
    action_mask: torch.Tensor,
    num_vision_tokens: int,
    bos_tokens: int = 1,
) -> torch.Tensor:
    """Shift an original text-sequence mask into `BOS + vision + text` layout."""

    if bos_tokens == 0:
        false_vision = torch.zeros(
            action_mask.shape[0],
            num_vision_tokens,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return torch.cat([false_vision, action_mask], dim=1)

    false_vision = torch.zeros(
        action_mask.shape[0],
        num_vision_tokens,
        dtype=torch.bool,
        device=action_mask.device,
    )
    return torch.cat(
        [
            action_mask[:, :bos_tokens],
            false_vision,
            action_mask[:, bos_tokens:],
        ],
        dim=1,
    )


def build_multimodal_embeddings(
    input_embeddings: torch.Tensor,
    vision_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    action_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
    cfg: SequenceConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SegmentSlices]:
    """Insert visual tokens into the language sequence and return fused masks."""

    cfg = cfg or SequenceConfig()
    cfg.validate()
    if not cfg.insert_vision_after_bos:
        raise NotImplementedError("only BOS -> vision -> text layout is implemented")

    batch_size, _, hidden_size = input_embeddings.shape
    if vision_tokens.ndim != 3 or vision_tokens.shape[0] != batch_size:
        raise ValueError("vision_tokens must have shape [B, P, D]")
    if vision_tokens.shape[2] != hidden_size:
        raise ValueError("vision hidden size must match language hidden size")

    num_vision_tokens = vision_tokens.shape[1]
    bos = cfg.bos_tokens
    fused_embeddings = torch.cat(
        [input_embeddings[:, :bos], vision_tokens, input_embeddings[:, bos:]],
        dim=1,
    )

    vision_attention = torch.ones(
        batch_size,
        num_vision_tokens,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    fused_attention = torch.cat(
        [attention_mask[:, :bos], vision_attention, attention_mask[:, bos:]],
        dim=1,
    )

    fused_labels = None
    if labels is not None:
        vision_labels = torch.full(
            (batch_size, num_vision_tokens),
            ignore_index,
            dtype=labels.dtype,
            device=labels.device,
        )
        fused_labels = torch.cat([labels[:, :bos], vision_labels, labels[:, bos:]], dim=1)

    fused_action_mask = shift_mask_after_vision_insert(action_mask, num_vision_tokens, bos)
    segments = SegmentSlices(
        bos=slice(0, bos),
        vision=slice(bos, bos + num_vision_tokens),
        text=slice(bos + num_vision_tokens, fused_embeddings.shape[1]),
        action_mask=fused_action_mask,
    )
    return fused_embeddings, fused_attention, fused_labels, segments


@dataclass(frozen=True)
class HiddenStateExtractor:
    """Extract per-layer Raw and ActionQuery conditions for the Bridge policy."""

    include_embedding_state: bool = False
    raw_token_budget: int | None = None

    def layer_indices(self, hidden_states: Sequence[torch.Tensor]) -> Iterable[int]:
        start = 0 if self.include_embedding_state else 1
        return range(start, len(hidden_states))

    def __call__(
        self,
        hidden_states: Sequence[torch.Tensor],
        segments: SegmentSlices,
    ) -> LayerCondition:
        raw_layers = []
        aq_layers = []
        for idx in self.layer_indices(hidden_states):
            state = hidden_states[idx]
            batch_size = state.shape[0]
            raw_layers.append(_mean_pool_tokens(state[:, segments.vision], self.raw_token_budget).unsqueeze(1))
            aq = state[segments.action_mask].reshape(batch_size, 1, -1, state.shape[-1])
            aq_layers.append(aq)

        if not raw_layers:
            raise ValueError("no hidden-state layers selected")

        return LayerCondition(
            raw_tokens=torch.cat(raw_layers, dim=1),
            action_query_tokens=torch.cat(aq_layers, dim=1),
        )


def _mean_pool_tokens(tokens: torch.Tensor, token_budget: int | None) -> torch.Tensor:
    """Reduce token count before stacking layers to keep peak memory bounded."""

    if token_budget is None or tokens.shape[1] <= token_budget:
        return tokens
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    batch_size, token_count, hidden_size = tokens.shape
    padded_tokens = math.ceil(token_count / token_budget) * token_budget
    if padded_tokens != token_count:
        pad = tokens[:, -1:].expand(batch_size, padded_tokens - token_count, hidden_size)
        tokens = torch.cat([tokens, pad], dim=1)
    return tokens.reshape(batch_size, token_budget, padded_tokens // token_budget, hidden_size).mean(dim=2)
