"""Train the Qwen3.5 + fused ViT adapter policy.

The script expects a dataset factory so the framework stays independent of
RLDS/LIBERO/CALVIN storage details.

Dataset factory contract:
    factory(**kwargs) -> Dataset | (train_dataset, val_dataset) | {"train": ..., "val": ...}

Each dataset item should be an `AdapterBatch`; use `SampleAdapter` wrappers for
native RLDS/LIBERO records.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from prismatic_adapter import AdapterConfig, ConditioningConfig, PolicyConfig, SequenceConfig
from prismatic_adapter.adapters import QwenTimmVLAAdapter
from prismatic_adapter.components.actions import ActionNormalizer, ActionStats
from prismatic_adapter.data import PaddedBatchCollator
from prismatic_adapter.factory import build_policy
from prismatic_adapter.training.config import (
    CheckpointConfig,
    LoggingConfig,
    LoraConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
)
from prismatic_adapter.training.optim import apply_lora, count_trainable_parameters
from prismatic_adapter.training.trainer import AdapterTrainer


def load_object(path: str):
    module_name, object_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def load_action_normalizer(path: str | None) -> ActionNormalizer | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    stats = ActionStats(
        low=torch.tensor(data["low"], dtype=torch.float32),
        high=torch.tensor(data["high"], dtype=torch.float32),
        mask=torch.tensor(data["mask"], dtype=torch.bool) if "mask" in data else None,
    )
    return ActionNormalizer(stats)


def parse_dataset_factory(factory_path: str, kwargs_json: str | None):
    factory = load_object(factory_path)
    kwargs: dict[str, Any] = {}
    if kwargs_json is not None:
        kwargs = json.loads(kwargs_json)
    result = factory(**kwargs)
    if isinstance(result, dict):
        return result["train"], result.get("val")
    if isinstance(result, tuple):
        return result[0], result[1] if len(result) > 1 else None
    return result, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-factory", required=True, help="Import path such as my_data:build_dataset")
    parser.add_argument("--dataset-kwargs-json", default=None, help="JSON object passed to dataset factory")
    parser.add_argument("--qwen-path", default="pretrained_models/Qwen3.5-2B")
    parser.add_argument("--vision-pretrained", action="store_true", help="Load pretrained TIMM ViT weights")
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/qwen35_vit")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=10_000)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--validate-every-steps", type=int, default=0)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--action-stats-json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["none", "float16", "bfloat16"])
    parser.add_argument("--policy-hidden-size", type=int, default=1024)
    parser.add_argument("--raw-token-budget", type=int, default=512)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="vla_adapter_qwen35_vit")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    sequence_cfg = SequenceConfig(action_query_tokens=64)
    backbone = QwenTimmVLAAdapter.from_pretrained(
        qwen_path=args.qwen_path,
        vision_pretrained=args.vision_pretrained,
        num_views=args.num_views,
        sequence_config=sequence_cfg,
        torch_dtype=torch.bfloat16 if args.amp_dtype == "bfloat16" else None,
    )
    adapter_cfg = AdapterConfig(
        sequence=sequence_cfg,
        conditioning=ConditioningConfig(
            layer_strategy="uniform",
            num_condition_layers=24,
            raw_token_budget=args.raw_token_budget,
            raw_compression="mean_pool",
            projection="linear",
        ),
        policy=PolicyConfig(
            hidden_size=args.policy_hidden_size,
            action_dim=7,
            action_horizon=8,
            num_layers=24,
            num_heads=8,
        ),
        train_backbone=False,
        train_action_queries=True,
        train_policy=True,
    )
    model = build_policy(backbone=backbone, config=adapter_cfg, proprio_dim=8)
    if args.use_lora:
        model = apply_lora(
            model,
            LoraConfig(enabled=True, rank=args.lora_rank, target="language_model"),
        )
    return model


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        trainer=TrainerConfig(
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accumulation_steps=args.grad_accumulation_steps,
            device=args.device,
            amp_dtype=args.amp_dtype,
            validate_every_steps=args.validate_every_steps or None,
        ),
        optimizer=OptimizerConfig(learning_rate=args.learning_rate),
        scheduler=SchedulerConfig(name="multistep", warmup_steps=args.warmup_steps),
        checkpoint=CheckpointConfig(
            output_dir=Path(args.output_dir),
            save_every_steps=args.save_every_steps,
            resume_path=Path(args.resume_path) if args.resume_path else None,
        ),
        logging=LoggingConfig(
            log_every_steps=args.log_every_steps,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_mode=args.wandb_mode,
        ),
    )


def main() -> None:
    args = parse_args()
    train_dataset, val_dataset = parse_dataset_factory(args.dataset_factory, args.dataset_kwargs_json)
    model = build_model(args)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    trainer = AdapterTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=build_training_config(args),
        collator=PaddedBatchCollator(pad_token_id=args.pad_token_id),
        action_normalizer=load_action_normalizer(args.action_stats_json),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
