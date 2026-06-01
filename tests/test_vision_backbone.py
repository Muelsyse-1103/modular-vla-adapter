import sys
import types

import torch

from prismatic_adapter.backbones.qwen_vit import (
    TimmFusedVisionBackbone,
    VisionBackboneSpec,
)


class FakeVisionModel(torch.nn.Module):
    def __init__(self, embed_dim, tokens):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokens = tokens
        self.seen_size = None

    def forward_features(self, pixel_values):
        self.seen_size = tuple(pixel_values.shape[-2:])
        batch = pixel_values.shape[0]
        return torch.ones(batch, self.tokens, self.embed_dim)


def test_dinov2_siglip_towers_resize_and_align_tokens(monkeypatch):
    created = []

    def create_model(model_id, pretrained, num_classes):
        del pretrained, num_classes
        if "dinov2" in model_id:
            model = FakeVisionModel(embed_dim=3, tokens=4)
        else:
            model = FakeVisionModel(embed_dim=5, tokens=7)
        created.append(model)
        return model

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=create_model))
    backbone = TimmFusedVisionBackbone(
        specs=(
            VisionBackboneSpec("fake_dinov2", image_size=16),
            VisionBackboneSpec("fake_siglip", image_size=24),
        ),
        pretrained=False,
        num_views=2,
        token_align="interpolate",
    )

    out = backbone(torch.zeros(1, 2, 3, 12, 12))

    assert out.shape == (1, 14, 8)
    assert created[0].seen_size == (16, 16)
    assert created[1].seen_size == (24, 24)


def test_token_align_error_reports_mismatch(monkeypatch):
    def create_model(model_id, pretrained, num_classes):
        del pretrained, num_classes
        return FakeVisionModel(embed_dim=2, tokens=3 if "a" in model_id else 5)

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=create_model))
    backbone = TimmFusedVisionBackbone(
        model_ids=("a", "b"),
        pretrained=False,
        token_align="error",
    )

    try:
        backbone(torch.zeros(1, 3, 8, 8))
    except ValueError as exc:
        assert "token counts" in str(exc)
    else:
        raise AssertionError("expected mismatched token counts to raise")
