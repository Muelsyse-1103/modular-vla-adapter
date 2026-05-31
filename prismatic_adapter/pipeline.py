"""Public VLA adapter pipeline.

This module gives the framework its primary mental model:

    ModelAdapter -> Conditioning -> ActionHead

The implementation reuses `PrismaticAdapterPolicy` for backward compatibility,
but new code should import `VLAAdapter` when it wants the complete adapter
pipeline.
"""

from __future__ import annotations

from prismatic_adapter.model import PrismaticAdapterPolicy


class VLAAdapter(PrismaticAdapterPolicy):
    """End-to-end VLA adapter pipeline.

    The class composes:
    - a model-specific adapter that exposes hidden states and token segments;
    - conditioning modules that select layers, project hidden sizes, and compress
      visual tokens;
    - a Bridge action head that predicts continuous action chunks.
    """


__all__ = ["VLAAdapter"]
