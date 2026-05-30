"""Optimizer, scheduler, and LoRA helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from prismatic_adapter.model import PrismaticAdapterPolicy
from prismatic_adapter.training.config import LoraConfig, OptimizerConfig, SchedulerConfig


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    cfg: OptimizerConfig,
) -> AdamW:
    return AdamW(
        list(parameters),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: SchedulerConfig,
    max_steps: int,
) -> LambdaLR:
    cfg.validate()

    def lr_lambda(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            warmup = 0.1 + 0.9 * float(step + 1) / float(cfg.warmup_steps)
        else:
            warmup = 1.0

        if cfg.name == "constant":
            scale = 1.0
        elif cfg.name == "multistep":
            passed = sum(step >= milestone for milestone in cfg.milestones)
            scale = cfg.gamma**passed
        elif cfg.name == "cosine":
            start = cfg.warmup_steps
            progress = min(max(step - start, 0) / max(max_steps - start, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine
        else:
            raise ValueError(f"unsupported scheduler: {cfg.name}")

        return warmup * scale

    return LambdaLR(optimizer, lr_lambda)


def apply_lora(policy: PrismaticAdapterPolicy, cfg: LoraConfig) -> PrismaticAdapterPolicy:
    """Attach PEFT LoRA to the selected pretrained component."""

    cfg.validate()
    if not cfg.enabled:
        return policy
    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires `peft`; install optional dependencies.") from exc

    if cfg.target == "language_model":
        if not hasattr(policy.backbone, "language_model"):
            raise ValueError("LoRA target language_model requires backbone.language_model")
        target = policy.backbone.language_model
    else:
        target = policy.backbone

    peft_cfg = PeftLoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha or 2 * cfg.rank,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        init_lora_weights=cfg.init_lora_weights,
    )
    wrapped = get_peft_model(target, peft_cfg)
    if cfg.target == "language_model":
        policy.backbone.language_model = wrapped
    else:
        policy.backbone = wrapped
    policy.configure_trainable_parameters()
    for name, param in wrapped.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    return policy
