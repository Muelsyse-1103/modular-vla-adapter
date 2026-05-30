"""Backbone adapter interfaces."""

from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.backbones.hf_prismatic import HuggingFacePrismaticAdapter

__all__ = ["BackboneAdapter", "HuggingFacePrismaticAdapter"]
