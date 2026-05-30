"""Small construction helpers for scripts."""

from __future__ import annotations

from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.config import AdapterConfig
from prismatic_adapter.model import PrismaticAdapterPolicy


def build_policy(
    backbone: BackboneAdapter,
    config: AdapterConfig,
    proprio_dim: int | None = None,
) -> PrismaticAdapterPolicy:
    """Construct the top-level policy after validating config/backbone metadata."""

    return PrismaticAdapterPolicy(backbone=backbone, config=config, proprio_dim=proprio_dim)
