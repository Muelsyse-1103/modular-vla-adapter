"""Train a MiniCPM-V adapter policy on LIBERO HDF5 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic_adapter import AdapterConfig, ConditioningConfig, PolicyConfig, SequenceConfig
from prismatic_adapter.components.actions import ActionNormalizer, ActionStats
from prismatic_adapter.config import TrainableConfig
from prismatic_adapter.config_loader import load_yaml_defaults
from prismatic_adapter.data import PaddedBatchCollator
from prismatic_adapter.datasets.libero_hdf5 import (
    DEFAULT_PRIMARY_IMAGE_KEYS,
    DEFAULT_PROPRIO_KEYS,
    DEFAULT_WRIST_IMAGE_KEYS,
    LiberoHdf5Config,
    LiberoHdf5Dataset,
)
from prismatic_adapter.factory import build_policy
from prismatic_adapter.model_adapters import MiniCPMVLAAdapter
from prismatic_adapter.processors import MiniCPMProcessorConfig, MiniCPMVBatchProcessor
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


def parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_lora_target_modules(value: str):
    if value == "all-linear":
        return value
    return list(parse_csv(value) or ())


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


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    initial, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dataset-format",
        default="libero_hdf5",
        choices=["libero_hdf5", "rlds"],
        help="Select the data input backend.",
    )
    parser.add_argument("--libero-hdf5-root", required=False)
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
    parser.add_argument("--model-path", default="pretrained_models/MiniCPM-V-4.6")
    parser.add_argument("--downsample-mode", default="16x", choices=["4x", "16x"])
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--action-stats-json", default=None)
    parser.add_argument("--output-dir", default="outputs/minicpm_v_libero")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=10_000)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--validate-every-steps", type=int, default=0)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["none", "float16", "bfloat16"])
    parser.add_argument("--policy-hidden-size", type=int, default=512)
    parser.add_argument("--action-query-tokens", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--raw-token-budget", type=int, default=256)
    parser.add_argument("--train-language-model", action="store_true")
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument("--train-vision-projector", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--wandb-project", default="vla_adapter_minicpm_v")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    defaults = load_yaml_defaults(initial.config)
    defaults["config"] = initial.config
    parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


def build_dataset(args: argparse.Namespace):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    adapter = MiniCPMVBatchProcessor(
        processor=processor,
        config=MiniCPMProcessorConfig(
            image_keys=parse_csv(
                args.rlds_image_keys if args.dataset_format == "rlds" else args.libero_image_keys
            )
            or ("image_primary", "image_wrist"),
            action_query_tokens=args.action_query_tokens,
            downsample_mode=args.downsample_mode,
            max_slice_nums=args.max_slice_nums,
        ),
    )
    if args.dataset_format == "libero_hdf5":
        dataset = LiberoHdf5Dataset(
            LiberoHdf5Config(
                root=args.libero_hdf5_root,
                action_key=args.libero_action_key,
                primary_image_keys=parse_csv(args.libero_primary_image_keys) or DEFAULT_PRIMARY_IMAGE_KEYS,
                wrist_image_keys=parse_csv(args.libero_wrist_image_keys) or DEFAULT_WRIST_IMAGE_KEYS,
                proprio_keys=parse_csv(args.libero_proprio_keys) or DEFAULT_PROPRIO_KEYS,
                fallback_instruction=args.libero_fallback_instruction,
                action_horizon=args.action_horizon,
                frame_stride=args.libero_frame_stride,
                sample_stride=args.libero_sample_stride,
                max_episodes=args.libero_max_episodes,
            ),
            adapter=adapter,
        )
        return _split_dataset(dataset, args.libero_val_ratio)
    if args.dataset_format == "rlds":
        if args.rlds_tfds_name is None:
            raise ValueError("--rlds-tfds-name is required when --dataset-format rlds")
        from prismatic_adapter.datasets.rlds import RldsTfdsDataset

        train_dataset = RldsTfdsDataset(config=_rlds_config(args, args.rlds_split), adapter=adapter)
        val_dataset = (
            RldsTfdsDataset(config=_rlds_config(args, args.rlds_val_split), adapter=adapter)
            if args.rlds_val_split
            else None
        )
        return train_dataset, val_dataset
    raise ValueError(f"unsupported dataset format: {args.dataset_format}")


def _split_dataset(dataset, val_ratio: float):
    if val_ratio <= 0:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    indices = list(range(len(dataset)))
    return Subset(dataset, indices[:train_size]), Subset(dataset, indices[train_size:])


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
    torch_dtype = {
        "none": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]
    backbone = MiniCPMVLAAdapter.from_pretrained(
        model_path=args.model_path,
        sequence_config=sequence_cfg,
        torch_dtype=torch_dtype,
    )
    config = AdapterConfig(
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
    model = build_policy(backbone=backbone, config=config, proprio_dim=args.proprio_dim)
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
            wandb_mode=args.wandb_mode,
        ),
    )


def main() -> None:
    args = parse_args()
    if args.dataset_format == "libero_hdf5" and args.libero_hdf5_root is None:
        raise ValueError("--libero-hdf5-root is required")
    train_dataset, val_dataset = build_dataset(args)
    model = build_model(args)
    processor = train_dataset.dataset.adapter.processor if isinstance(train_dataset, Subset) else train_dataset.adapter.processor
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    trainer = AdapterTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=build_training_config(args),
        collator=PaddedBatchCollator(pad_token_id=int(pad_token_id)),
        action_normalizer=load_action_normalizer(args.action_stats_json),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
