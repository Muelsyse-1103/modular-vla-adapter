"""Model-specific adapters.

Add new large-model integrations here. A model adapter is responsible for
turning model-native inputs and hidden states into the framework's stable
`BackboneOutput` contract.
"""

from prismatic_adapter.backbones.base import BackboneAdapter as ModelAdapter
from prismatic_adapter.backbones.hf_prismatic import HuggingFacePrismaticAdapter
from prismatic_adapter.backbones.qwen_vit import (
    DEFAULT_VISION_BACKBONE_SPECS,
    DEFAULT_VISION_MODEL_IDS,
    QwenTimmVLAAdapter,
    TimmFusedVisionBackbone,
    VisionBackboneSpec,
)

__all__ = [
    "DEFAULT_VISION_BACKBONE_SPECS",
    "DEFAULT_VISION_MODEL_IDS",
    "HuggingFacePrismaticAdapter",
    "ModelAdapter",
    "QwenTimmVLAAdapter",
    "TimmFusedVisionBackbone",
    "VisionBackboneSpec",
]
