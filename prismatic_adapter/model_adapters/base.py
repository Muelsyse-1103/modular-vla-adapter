"""Model adapter protocol.

This is the preferred public import path. `prismatic_adapter.backbones.base`
remains as a compatibility alias for existing code.
"""

from prismatic_adapter.backbones.base import BackboneAdapter, BackboneAdapter as ModelAdapter

__all__ = ["BackboneAdapter", "ModelAdapter"]
