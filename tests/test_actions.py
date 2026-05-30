import torch

from prismatic_adapter.components.actions import ActionNormalizer, ActionStats


def test_action_normalizer_round_trip():
    stats = ActionStats(low=torch.tensor([-1.0, 0.0]), high=torch.tensor([1.0, 10.0]))
    normalizer = ActionNormalizer(stats)
    actions = torch.tensor([[[0.0, 5.0], [1.0, 10.0]]])

    normalized = normalizer.normalize(actions)
    restored = normalizer.unnormalize(normalized)

    assert torch.allclose(restored, actions)
