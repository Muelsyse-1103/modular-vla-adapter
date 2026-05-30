"""Loss functions for continuous action training."""

from __future__ import annotations

import torch


def normalized_action_l1_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 loss over normalized action chunks."""

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            f"predicted and target actions must share shape; got "
            f"{tuple(predicted_actions.shape)} and {tuple(target_actions.shape)}"
        )
    per_dim = (predicted_actions - target_actions).abs()
    if mask is None:
        return per_dim.mean()

    if mask.shape != predicted_actions.shape[:2]:
        raise ValueError("mask must have shape [B, H]")
    weighted = per_dim * mask.unsqueeze(-1).to(per_dim.dtype)
    denom = mask.sum().clamp_min(1).to(per_dim.dtype) * predicted_actions.shape[-1]
    return weighted.sum() / denom
