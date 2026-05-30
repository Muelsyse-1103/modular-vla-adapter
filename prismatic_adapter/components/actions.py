"""Action normalization utilities for dataset compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class ActionStats:
    """Per-dimension action statistics."""

    low: torch.Tensor
    high: torch.Tensor
    mask: torch.Tensor | None = None


class ActionNormalizer:
    """Normalize and unnormalize actions with dataset statistics."""

    def __init__(
        self,
        stats: ActionStats,
        mode: Literal["bounds"] = "bounds",
        eps: float = 1e-8,
    ) -> None:
        if mode != "bounds":
            raise ValueError("only bounds normalization is implemented")
        self.stats = stats
        self.mode = mode
        self.eps = eps

    def _mask(self, ref: torch.Tensor) -> torch.Tensor:
        if self.stats.mask is None:
            return torch.ones_like(ref, dtype=torch.bool)
        return self.stats.mask.to(device=ref.device, dtype=torch.bool)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        low = self.stats.low.to(actions.device, actions.dtype)
        high = self.stats.high.to(actions.device, actions.dtype)
        normalized = 2.0 * (actions - low) / (high - low + self.eps) - 1.0
        return torch.where(self._mask(actions), normalized, actions)

    def unnormalize(self, actions: torch.Tensor) -> torch.Tensor:
        low = self.stats.low.to(actions.device, actions.dtype)
        high = self.stats.high.to(actions.device, actions.dtype)
        unnormalized = 0.5 * (actions + 1.0) * (high - low + self.eps) + low
        return torch.where(self._mask(actions), unnormalized, actions)
