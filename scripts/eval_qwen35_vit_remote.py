"""Evaluate a Qwen3.5 + fused ViT adapter against a remote ZMQ environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic_adapter import AdapterConfig, ConditioningConfig, PolicyConfig, SequenceConfig
from prismatic_adapter.adapters import QwenTimmVLAAdapter
from prismatic_adapter.checkpoint import load_adapter_checkpoint
from prismatic_adapter.components.actions import ActionNormalizer, ActionStats
from prismatic_adapter.factory import build_policy
from prismatic_adapter.inference import ActionPredictor
from vla_runtime.env_client import RemoteEnvClient
from vla_runtime.policies import ObservationBatchBuilder, ObservationBatchConfig, VLAAdapterRolloutPolicy
from vla_runtime.recorder import EpisodeRecorder
from vla_runtime.rollouts import RemoteRolloutRunner
from vla_runtime.runners import RemoteEvalRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--qwen-path", default="pretrained_models/Qwen3.5-2B")
    parser.add_argument("--checkpoint", default=None, help="Adapter checkpoint produced by training.")
    parser.add_argument("--action-stats-json", default=None)
    parser.add_argument("--output-dir", default="outputs/qwen35_vit_remote_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["none", "float16", "bfloat16"])
    parser.add_argument("--vision-pretrained", action="store_true")
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
    parser.add_argument("--image-keys", default="image_primary,image_wrist")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-hidden-size", type=int, default=1024)
    parser.add_argument("--raw-token-budget", type=int, default=512)
    parser.add_argument("--strict-checkpoint", action="store_true")
    return parser.parse_args()


def amp_dtype(name: str) -> torch.dtype | None:
    if name == "none":
        return None
    return torch.float16 if name == "float16" else torch.bfloat16


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


def build_model(args: argparse.Namespace):
    sequence_cfg = SequenceConfig(action_query_tokens=64)
    backbone = QwenTimmVLAAdapter.from_pretrained(
        qwen_path=args.qwen_path,
        vision_model_ids=parse_csv(args.vision_model_ids),
        vision_image_sizes=parse_int_csv(args.vision_image_sizes),
        vision_token_align=args.vision_token_align,
        vision_pretrained=args.vision_pretrained,
        num_views=args.num_views,
        sequence_config=sequence_cfg,
        torch_dtype=amp_dtype(args.amp_dtype),
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
            action_dim=7,
            action_horizon=8,
            num_layers=24,
            num_heads=8,
        ),
        train_backbone=False,
        train_action_queries=True,
        train_policy=True,
    )
    return build_policy(backbone=backbone, config=config, proprio_dim=8)


def load_tokenizer(qwen_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Evaluation requires `transformers` to load the Qwen tokenizer.") from exc
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    model = build_model(args)
    if args.checkpoint is not None:
        load_adapter_checkpoint(model, args.checkpoint, strict=args.strict_checkpoint)

    tokenizer = load_tokenizer(args.qwen_path)
    batch_builder = ObservationBatchBuilder(
        tokenizer=tokenizer,
        config=ObservationBatchConfig(
            image_keys=tuple(key.strip() for key in args.image_keys.split(",") if key.strip()),
            image_size=args.image_size,
            action_query_tokens=model.config.sequence.action_query_tokens,
        ),
        device=args.device,
    )
    policy = VLAAdapterRolloutPolicy(
        predictor=ActionPredictor(model, action_normalizer=load_action_normalizer(args.action_stats_json)),
        batch_builder=batch_builder,
        device=args.device,
        amp_dtype=amp_dtype(args.amp_dtype),
    )

    client = RemoteEnvClient(args.endpoint)
    print(client.hello())
    tasks = client.list_tasks()
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]

    rollout = RemoteRolloutRunner(client=client, policy=policy, max_steps=args.max_steps)
    runner = RemoteEvalRunner(rollout_runner=rollout, recorder=EpisodeRecorder(args.output_dir))
    summary = runner.run(tasks, trials_per_task=args.trials_per_task, seed=args.seed)
    print(summary.to_dict())
    client.close()
    client.close_socket()


if __name__ == "__main__":
    main()
