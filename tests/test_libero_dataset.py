import numpy as np
import torch

from prismatic_adapter.datasets import (
    LiberoAdapterConfig,
    LiberoSampleAdapter,
    compute_action_stats,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        del return_tensors, add_special_tokens
        ids = torch.arange(2, 2 + len(text.split()), dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def sample(actions):
    return {
        "instruction": "pick the cube",
        "image_primary": np.zeros((16, 16, 3), dtype=np.uint8),
        "image_wrist": np.ones((16, 16, 3), dtype=np.uint8),
        "proprio": np.zeros(8, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def test_libero_sample_adapter_builds_adapter_batch():
    adapter = LiberoSampleAdapter(
        FakeTokenizer(),
        LiberoAdapterConfig(image_size=8, action_query_tokens=3),
    )

    batch = adapter(sample([[0, 1, 2, 3, 4, 5, 6]]))

    assert batch.pixel_values.shape == (2, 3, 8, 8)
    assert batch.actions.shape == (1, 7)
    assert batch.action_mask.sum().item() == 3
    assert batch.proprio.shape == (8,)


def test_compute_action_stats_from_samples():
    stats = compute_action_stats(
        [
            sample([[0, 1], [2, 3]]),
            sample([[-1, 4]]),
        ],
        mask=[True, False],
    )

    assert stats.low.tolist() == [-1.0, 1.0]
    assert stats.high.tolist() == [2.0, 4.0]
    assert stats.mask.tolist() == [True, False]
