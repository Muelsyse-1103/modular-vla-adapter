import numpy as np
import pytest
import torch

from prismatic_adapter.datasets.libero import LiberoAdapterConfig, LiberoSampleAdapter
from prismatic_adapter.datasets.libero_hdf5 import (
    LiberoHdf5Config,
    LiberoHdf5Dataset,
    compute_libero_hdf5_action_stats,
    discover_hdf5_files,
)

h5py = pytest.importorskip("h5py")


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        del return_tensors, add_special_tokens
        ids = torch.arange(2, 2 + len(text.split()), dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def write_demo(path):
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["language_instruction"] = "pick up the cube"
        demo.create_dataset("actions", data=np.arange(20, dtype=np.float32).reshape(5, 4))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((5, 16, 16, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.ones((5, 16, 16, 3), dtype=np.uint8))
        obs.create_dataset("ee_states", data=np.zeros((5, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((5, 2), dtype=np.float32))


def test_libero_hdf5_dataset_reads_adapter_batch(tmp_path):
    path = tmp_path / "demo.hdf5"
    write_demo(path)

    adapter = LiberoSampleAdapter(
        FakeTokenizer(),
        LiberoAdapterConfig(image_size=8, action_query_tokens=3),
    )
    dataset = LiberoHdf5Dataset(
        LiberoHdf5Config(root=tmp_path, action_horizon=3, sample_stride=2),
        adapter,
    )

    batch = dataset[0]

    assert discover_hdf5_files(tmp_path) == [path]
    assert len(dataset) == 3
    assert batch.pixel_values.shape == (2, 3, 8, 8)
    assert batch.actions.shape == (3, 4)
    assert batch.proprio.shape == (8,)
    assert batch.metadata["instruction"] == "pick up the cube"


def test_compute_libero_hdf5_action_stats(tmp_path):
    write_demo(tmp_path / "demo.hdf5")

    stats = compute_libero_hdf5_action_stats(tmp_path)

    assert stats.low.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert stats.high.tolist() == [16.0, 17.0, 18.0, 19.0]
