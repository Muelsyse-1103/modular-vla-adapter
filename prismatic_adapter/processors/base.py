"""Prompt processing primitives shared by datasets and rollout policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PromptProcessor:
    """Tokenize text prompts and append ActionQuery placeholder positions."""

    tokenizer: Any
    prompt_template: str = "What action should the robot take to {instruction}?"
    action_query_tokens: int = 64
    placeholder_token_id: int | None = None
    add_special_tokens: bool = True

    def encode(
        self,
        instruction: str,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self.prompt_template.format(instruction=instruction)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )
        input_ids = _encoded_value(encoded, "input_ids")
        attention_mask = _encoded_value(encoded, "attention_mask")
        if input_ids is None:
            raise ValueError("tokenizer output is missing input_ids")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        target_device = torch.device(device) if device is not None else input_ids.device
        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)
        placeholders = torch.full(
            (input_ids.shape[0], self.action_query_tokens),
            self.placeholder_id(),
            dtype=input_ids.dtype,
            device=target_device,
        )
        input_ids = torch.cat([input_ids, placeholders], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(placeholders)], dim=1)
        action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_mask[:, -self.action_query_tokens :] = True
        return input_ids, attention_mask, action_mask

    def placeholder_id(self) -> int:
        if self.placeholder_token_id is not None:
            return int(self.placeholder_token_id)
        for name in ("pad_token_id", "eos_token_id", "unk_token_id"):
            value = getattr(self.tokenizer, name, None)
            if value is not None:
                return int(value)
        return 0


def _encoded_value(encoded: Any, key: str):
    if isinstance(encoded, dict):
        return encoded.get(key)
    return getattr(encoded, key, None)
