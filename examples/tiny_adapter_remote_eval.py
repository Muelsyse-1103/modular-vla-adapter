"""Run a tiny VLAAdapter through the real remote-env rollout path.

This smoke script validates the framework wiring without loading Qwen, DINOv2,
SigLIP, or LIBERO:

RemoteEnvClient -> ObservationBatchBuilder -> VLAAdapter -> action chunk -> env.step
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic_adapter import AdapterConfig, ConditioningConfig, PolicyConfig, SequenceConfig
from prismatic_adapter.backbones.base import BackboneAdapter
from prismatic_adapter.factory import build_policy
from prismatic_adapter.inference import ActionPredictor
from prismatic_adapter.sequence import build_multimodal_embeddings, replace_masked_embeddings
from prismatic_adapter.types import AdapterBatch, BackboneOutput
from vla_runtime.env_client import RemoteEnvClient
from vla_runtime.policies import ObservationBatchBuilder, ObservationBatchConfig, VLAAdapterRolloutPolicy
from vla_runtime.recorder import EpisodeRecorder
from vla_runtime.rollouts import RemoteRolloutRunner
from vla_runtime.runners import RemoteEvalRunner


class WhitespaceTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True):
        del return_tensors, add_special_tokens
        ids = [2 + (abs(hash(token)) % 100) for token in text.split()]
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class TinyVisionBackbone(nn.Module):
    def __init__(self, hidden_size: int, vision_tokens: int = 8) -> None:
        super().__init__()
        self.vision_tokens = vision_tokens
        self.project = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 5:
            batch, views = pixel_values.shape[:2]
            pooled = pixel_values.mean(dim=(-1, -2)).reshape(batch, views, 3)
        else:
            pooled = pixel_values.mean(dim=(-1, -2)).unsqueeze(1)
        tokens = self.project(pooled)
        tokens = tokens.repeat_interleave(max(self.vision_tokens // tokens.shape[1], 1), dim=1)
        return tokens[:, : self.vision_tokens]


class TinyRemoteBackbone(BackboneAdapter):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.vision = TinyVisionBackbone(hidden_size)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True) for _ in range(layers)]
        )

    def forward_with_action_queries(
        self,
        batch: AdapterBatch,
        action_queries: torch.Tensor,
    ) -> BackboneOutput:
        embeddings = self.embed(batch.input_ids)
        embeddings = replace_masked_embeddings(embeddings, batch.action_mask, action_queries)
        vision_tokens = self.vision(batch.pixel_values)
        fused, fused_attention, _, segments = build_multimodal_embeddings(
            embeddings,
            vision_tokens,
            batch.attention_mask,
            batch.action_mask,
        )
        hidden_states: list[torch.Tensor] = [fused]
        x = fused
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        return BackboneOutput(hidden_states, segments, fused_attention, vision_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", default="outputs/tiny_adapter_remote_eval")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_tiny_policy(device: str) -> VLAAdapterRolloutPolicy:
    sequence = SequenceConfig(action_query_tokens=4)
    config = AdapterConfig(
        sequence=sequence,
        conditioning=ConditioningConfig(num_condition_layers=3, raw_token_budget=4),
        policy=PolicyConfig(
            hidden_size=16,
            action_dim=7,
            action_horizon=2,
            num_layers=4,
            num_heads=4,
        ),
    )
    model = build_policy(TinyRemoteBackbone(), config, proprio_dim=8)
    batch_builder = ObservationBatchBuilder(
        tokenizer=WhitespaceTokenizer(),
        config=ObservationBatchConfig(image_size=32, action_query_tokens=sequence.action_query_tokens),
        device=device,
    )
    return VLAAdapterRolloutPolicy(
        predictor=ActionPredictor(model),
        batch_builder=batch_builder,
        device=device,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    client = RemoteEnvClient(args.endpoint)
    print(client.hello())
    tasks = client.list_tasks()
    rollout = RemoteRolloutRunner(
        client=client,
        policy=build_tiny_policy(args.device),
        max_steps=args.max_steps,
    )
    runner = RemoteEvalRunner(rollout_runner=rollout, recorder=EpisodeRecorder(args.output_dir))
    summary = runner.run(tasks, trials_per_task=args.trials_per_task)
    print(summary.to_dict())
    client.close()
    client.close_socket()


if __name__ == "__main__":
    main()
