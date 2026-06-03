import torch

from prismatic_adapter.sequence import (
    HiddenStateExtractor,
    build_multimodal_embeddings,
    replace_masked_embeddings,
    shift_mask_after_vision_insert,
)
from prismatic_adapter.types import SegmentSlices


def test_replace_masked_embeddings_puts_queries_in_order():
    embeddings = torch.zeros(2, 5, 3)
    mask = torch.tensor([[False, False, True, True, False], [False, True, False, False, True]])
    queries = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    out = replace_masked_embeddings(embeddings, mask, queries)

    assert torch.equal(out[0, 2], queries[0])
    assert torch.equal(out[0, 3], queries[1])
    assert torch.equal(out[1, 1], queries[0])
    assert torch.equal(out[1, 4], queries[1])


def test_shift_mask_after_vision_insert_preserves_action_positions():
    mask = torch.tensor([[False, False, True, True]])

    shifted = shift_mask_after_vision_insert(mask, num_vision_tokens=3, bos_tokens=1)

    assert shifted.tolist() == [[False, False, False, False, False, True, True]]


def test_build_multimodal_embeddings_segments_are_explicit():
    input_embeddings = torch.randn(1, 4, 8)
    vision_tokens = torch.randn(1, 3, 8)
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    action_mask = torch.tensor([[False, False, True, True]])

    fused, fused_attention, labels, segments = build_multimodal_embeddings(
        input_embeddings,
        vision_tokens,
        attention_mask,
        action_mask,
    )

    assert fused.shape == (1, 7, 8)
    assert fused_attention.shape == (1, 7)
    assert labels is None
    assert segments.vision == slice(1, 4)
    assert segments.action_mask.tolist() == [[False, False, False, False, False, True, True]]


def test_hidden_state_extractor_compresses_raw_tokens_before_stacking_layers():
    states = [
        torch.arange(1 * 8 * 1, dtype=torch.float32).reshape(1, 8, 1),
        torch.arange(1 * 8 * 1, dtype=torch.float32).reshape(1, 8, 1) + 10.0,
    ]
    segments = SegmentSlices(
        bos=slice(0, 1),
        vision=slice(1, 7),
        text=slice(7, 8),
        action_mask=torch.tensor([[False, False, False, False, False, False, False, True]]),
    )
    extractor = HiddenStateExtractor(include_embedding_state=True, raw_token_budget=3)

    condition = extractor(states, segments)

    assert condition.raw_tokens.shape == (1, 2, 3, 1)
    assert condition.raw_tokens[0, 0, :, 0].tolist() == [1.5, 3.5, 5.5]
    assert condition.raw_tokens[0, 1, :, 0].tolist() == [11.5, 13.5, 15.5]
