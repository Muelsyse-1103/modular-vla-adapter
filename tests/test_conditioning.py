import torch

from prismatic_adapter.components.conditioning import (
    ConditionProjector,
    LayerSelector,
    MeanPoolTokenCompressor,
)
from prismatic_adapter.config import ConditioningConfig
from prismatic_adapter.types import LayerCondition


def test_uniform_layer_selector_skips_embedding_state_by_default():
    states = [torch.full((1, 2, 3), float(i)) for i in range(5)]
    selector = LayerSelector(ConditioningConfig(num_condition_layers=2, layer_strategy="uniform"))

    selected = selector.select(states)

    assert len(selected) == 2
    assert torch.equal(selected[0], states[1])
    assert torch.equal(selected[1], states[4])


def test_condition_projector_changes_hidden_size():
    condition = LayerCondition(
        raw_tokens=torch.randn(2, 3, 4, 8),
        action_query_tokens=torch.randn(2, 3, 5, 8),
    )
    projector = ConditionProjector(8, 6)

    projected = projector(condition)

    assert projected.raw_tokens.shape == (2, 3, 4, 6)
    assert projected.action_query_tokens.shape == (2, 3, 5, 6)


def test_mean_pool_token_compressor_reduces_raw_budget():
    condition = LayerCondition(
        raw_tokens=torch.arange(1 * 1 * 6 * 1, dtype=torch.float32).reshape(1, 1, 6, 1),
        action_query_tokens=torch.randn(1, 1, 2, 1),
    )
    compressor = MeanPoolTokenCompressor(token_budget=3)

    compressed = compressor(condition)

    assert compressed.raw_tokens.shape == (1, 1, 3, 1)
    assert compressed.raw_tokens.flatten().tolist() == [0.5, 2.5, 4.5]
