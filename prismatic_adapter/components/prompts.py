"""Prompt and action-placeholder helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ActionPlaceholderSpec:
    """How action placeholder tokens are represented in tokenized prompts."""

    token_id: int
    count: int


class TokenizerLike(Protocol):
    def __call__(self, text: str, **kwargs): ...


class PromptAdapter:
    """Build prompt ids and action masks while leaving templates model-specific."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        placeholder: ActionPlaceholderSpec,
        template: str = "What action should the robot take to {instruction}?",
    ) -> None:
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.template = template

    def format_instruction(self, instruction: str) -> str:
        return self.template.format(instruction=instruction)

    def encode(self, instruction: str, **tokenizer_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.format_instruction(instruction)
        encoded = self.tokenizer(text, return_tensors="pt", **tokenizer_kwargs)
        input_ids = encoded.input_ids
        placeholders = torch.full(
            (input_ids.shape[0], self.placeholder.count),
            self.placeholder.token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, placeholders], dim=1)
        action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_mask[:, -self.placeholder.count :] = True
        return input_ids, action_mask
