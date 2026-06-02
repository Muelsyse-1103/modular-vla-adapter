import numpy as np
import torch

from prismatic_adapter.data import PaddedBatchCollator
from prismatic_adapter.processors import (
    MiniCPMProcessorConfig,
    MiniCPMVBatchProcessor,
    StandardBatchProcessor,
    StandardProcessorConfig,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        del return_tensors, add_special_tokens
        ids = torch.arange(2, 2 + len(text.split()), dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeMiniCPMProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return {
            "input_ids": torch.tensor([[101, 999, 102]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 14, 16),
            "target_sizes": torch.tensor([[16, 16]], dtype=torch.long),
            "downsample_mode": ["16x"],
        }


def test_standard_processor_builds_adapter_batch():
    processor = StandardBatchProcessor(
        FakeTokenizer(),
        StandardProcessorConfig(image_keys=("image_primary",), image_size=16, action_query_tokens=4),
    )
    batch = processor(
        {
            "image_primary": np.zeros((32, 32, 3), dtype=np.uint8),
            "proprio": np.zeros(8, dtype=np.float32),
        },
        instruction="pick up the block",
        actions=np.zeros((2, 7), dtype=np.float32),
    )

    assert batch.input_ids.shape == batch.attention_mask.shape == batch.action_mask.shape
    assert batch.action_mask.sum().item() == 4
    assert batch.pixel_values.shape == (3, 16, 16)
    assert batch.actions.shape == (2, 7)
    assert batch.proprio.shape == (8,)


def test_minicpm_processor_keeps_processor_image_dict():
    processor = MiniCPMVBatchProcessor(
        FakeMiniCPMProcessor(),
        MiniCPMProcessorConfig(image_keys=("image_primary",), action_query_tokens=2),
    )
    batch = processor(
        {
            "instruction": "move",
            "image_primary": np.zeros((8, 8, 3), dtype=np.uint8),
            "proprio": np.zeros(8, dtype=np.float32),
            "actions": np.zeros((1, 7), dtype=np.float32),
        }
    )

    assert batch.action_mask.sum().item() == 2
    assert isinstance(batch.pixel_values, dict)
    assert batch.pixel_values["pixel_values"].shape == (3, 14, 16)
    assert batch.pixel_values["target_sizes"].shape == (2,)
    assert batch.pixel_values["downsample_mode"] == ["16x"]


def test_collator_stacks_nested_pixel_values_dict():
    processor = MiniCPMVBatchProcessor(
        FakeMiniCPMProcessor(),
        MiniCPMProcessorConfig(image_keys=("image_primary",), action_query_tokens=2),
    )
    items = [
        processor({"instruction": "move", "image_primary": np.zeros((8, 8, 3), dtype=np.uint8)}),
        processor({"instruction": "move", "image_primary": np.zeros((8, 8, 3), dtype=np.uint8)}),
    ]

    batch = PaddedBatchCollator(pad_token_id=0)(items)

    assert batch.pixel_values["pixel_values"].shape == (2, 3, 14, 16)
    assert batch.pixel_values["target_sizes"].shape == (2, 2)
