import numpy as np
import torch

from prismatic_adapter.types import AdapterBatch, PredictionOutput
from vla_runtime.policies.vla_adapter import (
    ObservationBatchBuilder,
    ObservationBatchConfig,
    VLAAdapterRolloutPolicy,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        del return_tensors, add_special_tokens
        ids = torch.arange(2, 2 + len(text.split()), dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel(torch.nn.Module):
    def forward(self, batch: AdapterBatch):
        assert batch.input_ids.shape[0] == 1
        return torch.zeros(1, 2, 7)


class FakePredictor:
    def __init__(self):
        self.model = FakeModel()

    def predict(self, batch: AdapterBatch):
        return PredictionOutput(normalized_actions=torch.zeros(1, 2, 7), actions=self.model(batch))


def test_observation_batch_builder_shapes():
    builder = ObservationBatchBuilder(
        FakeTokenizer(),
        ObservationBatchConfig(image_size=16, action_query_tokens=4),
    )
    obs = {
        "image_primary": np.zeros((32, 32, 3), dtype=np.uint8),
        "image_wrist": np.ones((32, 32, 3), dtype=np.uint8),
        "proprio": np.zeros(8, dtype=np.float32),
    }

    batch = builder(obs, "pick up the block")

    assert batch.input_ids.shape[0] == 1
    assert batch.input_ids.shape == batch.attention_mask.shape == batch.action_mask.shape
    assert batch.action_mask.sum().item() == 4
    assert batch.pixel_values.shape == (1, 2, 3, 16, 16)
    assert batch.proprio.shape == (1, 8)


def test_vla_adapter_rollout_policy_returns_action_chunk():
    builder = ObservationBatchBuilder(
        FakeTokenizer(),
        ObservationBatchConfig(image_size=8, action_query_tokens=2),
    )
    policy = VLAAdapterRolloutPolicy(FakePredictor(), builder)
    obs = {
        "image_primary": np.zeros((8, 8, 3), dtype=np.uint8),
        "image_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "proprio": np.zeros(8, dtype=np.float32),
    }

    actions = policy.act(obs, "move")

    assert actions == [[0.0] * 7, [0.0] * 7]
