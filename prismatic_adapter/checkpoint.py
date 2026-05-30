"""Checkpoint helpers with explicit component boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from prismatic_adapter.model import PrismaticAdapterPolicy


def adapter_state_dict(model: PrismaticAdapterPolicy) -> dict[str, Any]:
    """Return a checkpoint dictionary grouped by framework component."""

    state = {
        "config": model.config,
        "action_queries": model.action_queries.detach().cpu(),
        "condition_projector": model.condition_projector.state_dict(),
        "action_head": model.action_head.state_dict(),
        "proprio_projector": (
            model.proprio_projector.state_dict() if model.proprio_projector is not None else None
        ),
    }
    if model.config.train_backbone:
        state["backbone"] = model.backbone.state_dict()
    return state


def save_adapter_checkpoint(model: PrismaticAdapterPolicy, path: str | Path) -> None:
    """Save only adapter-owned components by default."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state_dict(model), path)


def load_adapter_checkpoint(
    model: PrismaticAdapterPolicy,
    path: str | Path,
    strict: bool = True,
    load_backbone: bool = False,
) -> dict[str, Any]:
    """Load adapter components into an already constructed model."""

    checkpoint = torch.load(path, map_location="cpu")
    with torch.no_grad():
        model.action_queries.copy_(checkpoint["action_queries"].to(model.action_queries.device))
    model.condition_projector.load_state_dict(checkpoint["condition_projector"], strict=strict)
    model.action_head.load_state_dict(checkpoint["action_head"], strict=strict)
    if model.proprio_projector is not None and checkpoint.get("proprio_projector") is not None:
        model.proprio_projector.load_state_dict(checkpoint["proprio_projector"], strict=strict)
    if load_backbone and "backbone" in checkpoint:
        model.backbone.load_state_dict(checkpoint["backbone"], strict=strict)
    return checkpoint
