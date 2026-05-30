"""Backbone adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from prismatic_adapter.types import AdapterBatch, BackboneOutput


class BackboneAdapter(ABC, torch.nn.Module):
    """Expose any VLM as a Prismatic-style condition provider.

    To adapt a new large model, implement this interface. The continuous policy
    does not need to know whether the language model is Qwen, Llama, Phi,
    InternVL, MiniCPM-V, or a local research model.
    """

    hidden_size: int
    num_hidden_layers: int

    @abstractmethod
    def forward_with_action_queries(
        self,
        batch: AdapterBatch,
        action_queries: torch.Tensor,
    ) -> BackboneOutput:
        """Run the VLM with ActionQuery embeddings inserted into the prompt."""
