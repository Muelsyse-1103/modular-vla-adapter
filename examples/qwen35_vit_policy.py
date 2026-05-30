"""Instantiate the Qwen3.5 + DINOv2/SigLIP example policy.

This example is construction-only by default. A real training script should
provide tokenized prompts, action masks, camera tensors, action chunks, and
optional proprio through `AdapterBatch`.
"""

from __future__ import annotations

import torch

from prismatic_adapter import AdapterConfig, ConditioningConfig, PolicyConfig, SequenceConfig
from prismatic_adapter.backbones import QwenTimmVLAAdapter
from prismatic_adapter.factory import build_policy


def build_qwen35_vit_policy():
    backbone = QwenTimmVLAAdapter.from_pretrained(
        qwen_path="pretrained_models/Qwen3.5-2B",
        vision_pretrained=True,
        num_views=2,
        torch_dtype=torch.bfloat16,
    )
    cfg = AdapterConfig(
        sequence=SequenceConfig(action_query_tokens=64),
        conditioning=ConditioningConfig(
            layer_strategy="uniform",
            num_condition_layers=24,
            raw_token_budget=512,
            raw_compression="mean_pool",
            projection="linear",
        ),
        policy=PolicyConfig(
            hidden_size=1024,
            action_dim=7,
            action_horizon=8,
            num_layers=24,
            num_heads=8,
        ),
        train_backbone=False,
        train_action_queries=True,
        train_policy=True,
    )
    return build_policy(backbone=backbone, config=cfg, proprio_dim=8)


if __name__ == "__main__":
    model = build_qwen35_vit_policy()
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(model)
    print(f"trainable parameters: {trainable:,}")
