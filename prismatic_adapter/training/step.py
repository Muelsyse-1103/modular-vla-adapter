"""A small train-step wrapper suitable for scripts or Lightning/Accelerate glue."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prismatic_adapter.components.actions import ActionNormalizer
from prismatic_adapter.model import PrismaticAdapterPolicy
from prismatic_adapter.training.losses import normalized_action_l1_loss
from prismatic_adapter.types import AdapterBatch


@dataclass
class AdapterTrainStep:
    """Compute the VLA-Adapter forward pass and L1 objective."""

    model: PrismaticAdapterPolicy
    action_normalizer: ActionNormalizer | None = None

    def __call__(self, batch: AdapterBatch) -> tuple[torch.Tensor, dict[str, float]]:
        if batch.actions is None:
            raise ValueError("batch.actions is required for training")
        target_actions = batch.actions
        if self.action_normalizer is not None:
            target_actions = self.action_normalizer.normalize(target_actions)
        predicted = self.model(batch)
        loss = normalized_action_l1_loss(predicted, target_actions)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "action_l1": float(loss.detach().cpu()),
        }
        if predicted.shape[1] > 1:
            current = normalized_action_l1_loss(predicted[:, :1], target_actions[:, :1])
            future = normalized_action_l1_loss(predicted[:, 1:], target_actions[:, 1:])
            metrics["current_action_l1"] = float(current.detach().cpu())
            metrics["future_action_l1"] = float(future.detach().cpu())
        return loss, metrics
