import numpy as np

from prismatic_adapter.datasets.rlds import RldsConfig, RldsEpisodeDataset, iter_rlds_samples


def fake_episode(length=4):
    return {
        "language_instruction": b"pick up the cube",
        "steps": [
            {
                "observation": {
                    "image": np.full((8, 8, 3), idx, dtype=np.uint8),
                    "wrist_image": np.full((8, 8, 3), idx + 1, dtype=np.uint8),
                    "proprio": np.full(8, idx, dtype=np.float32),
                },
                "action": np.full(7, idx, dtype=np.float32),
            }
            for idx in range(length)
        ],
    }


def test_iter_rlds_samples_normalizes_episode_steps():
    config = RldsConfig(action_horizon=3, sample_stride=2)

    samples = list(iter_rlds_samples([fake_episode()], config))

    assert len(samples) == 2
    assert samples[0]["instruction"] == "pick up the cube"
    assert samples[0]["image_primary"].shape == (8, 8, 3)
    assert samples[0]["image_wrist"].shape == (8, 8, 3)
    assert samples[0]["proprio"].shape == (8,)
    assert samples[0]["actions"].shape == (3, 7)
    assert samples[0]["actions"][:, 0].tolist() == [0.0, 1.0, 2.0]
    assert samples[1]["actions"][:, 0].tolist() == [2.0, 3.0, 3.0]


def test_rlds_episode_dataset_applies_adapter():
    dataset = RldsEpisodeDataset(
        episodes=[fake_episode(length=2)],
        adapter=lambda sample: {"instruction": sample["instruction"], "actions": sample["actions"]},
        config=RldsConfig(action_horizon=2),
    )

    items = list(dataset)

    assert len(items) == 2
    assert items[0]["instruction"] == "pick up the cube"
    assert items[0]["actions"].shape == (2, 7)


def test_iter_rlds_samples_accepts_dict_of_step_arrays():
    episode = {
        "steps": {
            "observation": {
                "image": np.zeros((3, 8, 8, 3), dtype=np.uint8),
                "proprio": np.zeros((3, 8), dtype=np.float32),
            },
            "action": np.arange(21, dtype=np.float32).reshape(3, 7),
            "language_instruction": np.asarray([b"open drawer", b"open drawer", b"open drawer"]),
        }
    }

    samples = list(iter_rlds_samples([episode], RldsConfig(action_horizon=2)))

    assert len(samples) == 3
    assert samples[0]["instruction"] == "open drawer"
    assert samples[0]["actions"].shape == (2, 7)
    assert samples[0]["actions"][1, 0] == 7.0
