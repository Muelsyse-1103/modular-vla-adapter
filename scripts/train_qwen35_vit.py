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
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic_adapter import AdapterConfig, ConditioningConfig, PolicyConfig, SequenceConfig
from prismatic_adapter.config import TrainableConfig
from prismatic_adapter.config_loader import load_yaml_defaults
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


def parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_int_csv(value: str | None) -> tuple[int, ...] | None:
    parsed = parse_csv(value)
    if parsed is None:
        return None
    return tuple(int(item) for item in parsed)


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


def parse_lora_target_modules(value: str):
    if value == "all-linear":
        return value
    return list(parse_csv(value) or ())


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    initial, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file. CLI values override it.")
    parser.add_argument(
        "--dataset-format",
        default="auto",
        choices=["auto", "factory", "libero_hdf5", "rlds"],
        help="Select the data input backend. `auto` preserves legacy argument-based selection.",
    )
    parser.add_argument("--dataset-factory", default=None, help="Import path such as my_data:build_dataset")
    parser.add_argument("--dataset-kwargs-json", default=None, help="JSON object passed to dataset factory")
    parser.add_argument("--libero-hdf5-root", default=None, help="Use the built-in LIBERO HDF5 dataset factory.")
    parser.add_argument("--libero-val-ratio", type=float, default=0.0)
    parser.add_argument("--libero-action-key", default="actions")
    parser.add_argument("--libero-image-keys", default="image_primary,image_wrist")
    parser.add_argument("--libero-primary-image-keys", default=None)
    parser.add_argument("--libero-wrist-image-keys", default=None)
    parser.add_argument("--libero-proprio-keys", default=None)
    parser.add_argument("--libero-fallback-instruction", default=None)
    parser.add_argument("--libero-sample-stride", type=int, default=1)
    parser.add_argument("--libero-frame-stride", type=int, default=1)
    parser.add_argument("--libero-max-episodes", type=int, default=None)
    parser.add_argument("--write-action-stats-json", default=None)
    parser.add_argument("--rlds-tfds-name", default=None, help="TFDS builder name for an RLDS dataset.")
    parser.add_argument("--rlds-data-dir", default=None)
    parser.add_argument("--rlds-split", default="train")
    parser.add_argument("--rlds-val-split", default=None)
    parser.add_argument("--rlds-shuffle-files", action="store_true")
    parser.add_argument("--rlds-action-key", default="action")
    parser.add_argument("--rlds-steps-key", default="steps")
    parser.add_argument("--rlds-image-keys", default="image_primary,image_wrist")
    parser.add_argument("--rlds-primary-image-keys", default=None)
    parser.add_argument("--rlds-wrist-image-keys", default=None)
    parser.add_argument("--rlds-proprio-keys", default=None)
    parser.add_argument("--rlds-language-keys", default=None)
    parser.add_argument("--rlds-fallback-instruction", default=None)
    parser.add_argument("--rlds-sample-stride", type=int, default=1)
    parser.add_argument("--rlds-frame-stride", type=int, default=1)
    parser.add_argument("--rlds-max-episodes", type=int, default=None)
    parser.add_argument("--rlds-max-steps", type=int, default=None)
    parser.add_argument("--qwen-path", default="pretrained_models/Qwen3.5-2B")
    parser.add_argument("--vision-pretrained", action="store_true", help="Load pretrained TIMM ViT weights")
    parser.add_argument(
        "--vision-cache-dir",
        default="pretrained_models/vision_cache/hf",
        help="Local HF/TIMM cache directory for DINOv2/SigLIP weights.",
    )
    parser.add_argument(
        "--vision-model-ids",
        default=None,
        help="Comma-separated TIMM ids. Defaults to DINOv2 large + SigLIP SO400M.",
    )
    parser.add_argument(
        "--vision-image-sizes",
        default=None,
        help="Comma-separated input sizes matching --vision-model-ids, e.g. 224,224.",
    )
    parser.add_argument(
        "--vision-token-align",
        default="interpolate",
        choices=["interpolate", "truncate", "error"],
        help="How to align DINOv2/SigLIP patch-token counts before fusion.",
    )
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
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
    parser.add_argument("--action-query-tokens", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--raw-token-budget", type=int, default=512)
    parser.add_argument("--train-language-model", action="store_true")
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument(
        "--train-vision-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--train-action-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-action-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-proprio-projector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-target", default="language_model", choices=["language_model", "backbone"])
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="Modular-vla-adapter")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    defaults = load_yaml_defaults(initial.config)
    defaults["config"] = initial.config
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


def parse_dataset(args: argparse.Namespace):
    if args.dataset_format in ("auto", "libero_hdf5") and args.libero_hdf5_root is not None:
        from prismatic_adapter.datasets.libero_hdf5 import build_libero_hdf5_dataset

        result = build_libero_hdf5_dataset(
            root=args.libero_hdf5_root,
            tokenizer_path=args.qwen_path,
            val_ratio=args.libero_val_ratio,
            write_action_stats_json=args.write_action_stats_json,
            action_key=args.libero_action_key,
            primary_image_keys=parse_csv(args.libero_primary_image_keys),
            wrist_image_keys=parse_csv(args.libero_wrist_image_keys),
            proprio_keys=parse_csv(args.libero_proprio_keys),
            image_keys=parse_csv(args.libero_image_keys) or ("image_primary", "image_wrist"),
            image_size=args.image_size,
            action_horizon=args.action_horizon,
            action_query_tokens=args.action_query_tokens,
            fallback_instruction=args.libero_fallback_instruction,
            sample_stride=args.libero_sample_stride,
            frame_stride=args.libero_frame_stride,
            max_episodes=args.libero_max_episodes,
        )
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None
    if args.dataset_format in ("auto", "rlds") and args.rlds_tfds_name is not None:
        return build_qwen_rlds_dataset(args)
    if args.dataset_format in ("auto", "factory") and args.dataset_factory is not None:
        return parse_dataset_factory(args.dataset_factory, args.dataset_kwargs_json)
    raise ValueError(
        "provide a matching data source: --libero-hdf5-root, --rlds-tfds-name, "
        "or --dataset-factory"
    )


def build_qwen_rlds_dataset(args: argparse.Namespace):
    from transformers import AutoTokenizer

    from prismatic_adapter.datasets.libero import LiberoAdapterConfig, LiberoSampleAdapter
    from prismatic_adapter.datasets.rlds import RldsTfdsDataset

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    adapter = LiberoSampleAdapter(
        tokenizer=tokenizer,
        config=LiberoAdapterConfig(
            image_keys=parse_csv(args.rlds_image_keys) or ("image_primary", "image_wrist"),
            image_size=args.image_size,
            action_query_tokens=args.action_query_tokens,
        ),
    )
    train_dataset = RldsTfdsDataset(config=_rlds_config(args, args.rlds_split), adapter=adapter)
    val_dataset = (
        RldsTfdsDataset(config=_rlds_config(args, args.rlds_val_split), adapter=adapter)
        if args.rlds_val_split
        else None
    )
    return train_dataset, val_dataset


def _rlds_config(args: argparse.Namespace, split: str):
    from prismatic_adapter.datasets.rlds import (
        DEFAULT_RLDS_LANGUAGE_KEYS,
        DEFAULT_RLDS_PRIMARY_IMAGE_KEYS,
        DEFAULT_RLDS_PROPRIO_KEYS,
        DEFAULT_RLDS_WRIST_IMAGE_KEYS,
        RldsConfig,
    )

    return RldsConfig(
        tfds_name=args.rlds_tfds_name,
        data_dir=args.rlds_data_dir,
        split=split,
        shuffle_files=args.rlds_shuffle_files,
        action_key=args.rlds_action_key,
        steps_key=args.rlds_steps_key,
        primary_image_keys=parse_csv(args.rlds_primary_image_keys) or DEFAULT_RLDS_PRIMARY_IMAGE_KEYS,
        wrist_image_keys=parse_csv(args.rlds_wrist_image_keys) or DEFAULT_RLDS_WRIST_IMAGE_KEYS,
        proprio_keys=parse_csv(args.rlds_proprio_keys) or DEFAULT_RLDS_PROPRIO_KEYS,
        language_keys=parse_csv(args.rlds_language_keys) or DEFAULT_RLDS_LANGUAGE_KEYS,
        fallback_instruction=args.rlds_fallback_instruction,
        action_horizon=args.action_horizon,
        frame_stride=args.rlds_frame_stride,
        sample_stride=args.rlds_sample_stride,
        max_episodes=args.rlds_max_episodes,
        max_steps=args.rlds_max_steps,
    )


def build_model(args: argparse.Namespace):
    sequence_cfg = SequenceConfig(action_query_tokens=args.action_query_tokens)
    backbone = QwenTimmVLAAdapter.from_pretrained(
        qwen_path=args.qwen_path,
        vision_model_ids=parse_csv(args.vision_model_ids),
        vision_image_sizes=parse_int_csv(args.vision_image_sizes),
        vision_token_align=args.vision_token_align,
        vision_pretrained=args.vision_pretrained,
        vision_cache_dir=args.vision_cache_dir,
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
            action_dim=args.action_dim,
            action_horizon=args.action_horizon,
            num_layers=24,
            num_heads=8,
        ),
        trainable=TrainableConfig(
            language_model=args.train_language_model,
            vision_backbone=args.train_vision_backbone,
            vision_projector=args.train_vision_projector,
            action_queries=args.train_action_queries,
            conditioning=args.train_conditioning,
            action_head=args.train_action_head,
            proprio_projector=args.train_proprio_projector,
        ),
    )
    model = build_policy(backbone=backbone, config=adapter_cfg, proprio_dim=args.proprio_dim)
    if args.use_lora:
        model = apply_lora(
            model,
            LoraConfig(
                enabled=True,
                target=args.lora_target,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=parse_lora_target_modules(args.lora_target_modules),
            ),
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
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
        ),
    )


def main() -> None:
    args = parse_args()
    train_dataset, val_dataset = parse_dataset(args)
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
