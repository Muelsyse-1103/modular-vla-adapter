"""Inference helpers for normalized and environment-scale actions."""

from __future__ import annotations

import torch

from prismatic_adapter.components.actions import ActionNormalizer
from prismatic_adapter.model import PrismaticAdapterPolicy
from prismatic_adapter.types import AdapterBatch, PredictionOutput


class ActionPredictor:
    """Thin inference wrapper around `PrismaticAdapterPolicy`."""

    def __init__(
        self,
        model: PrismaticAdapterPolicy,
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        self.model = model
        self.action_normalizer = action_normalizer

    @torch.inference_mode()
    def predict(self, batch: AdapterBatch) -> PredictionOutput:
        normalized = self.model(batch)
        actions = (
            self.action_normalizer.unnormalize(normalized)
            if self.action_normalizer is not None
            else normalized
        )
        return PredictionOutput(normalized_actions=normalized, actions=actions)
