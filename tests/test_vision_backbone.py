import sys
import types

import torch

from prismatic_adapter.config import SequenceConfig
from prismatic_adapter.backbones.qwen_vit import (
    QwenTimmVLAAdapter,
    TimmFusedVisionBackbone,
    VisionBackboneSpec,
)
from prismatic_adapter.types import AdapterBatch


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


class FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8, num_hidden_layers=2)
        self.embedding = torch.nn.Embedding(16, 8)
        self.seen_logits_to_keep = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        self.seen_logits_to_keep = kwargs.get("logits_to_keep")
        hidden = kwargs["inputs_embeds"]
        return types.SimpleNamespace(hidden_states=[hidden, hidden + 1.0])


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


def test_qwen_adapter_skips_language_logits_when_labels_absent(monkeypatch):
    def create_model(model_id, pretrained, num_classes):
        del model_id, pretrained, num_classes
        return FakeVisionModel(embed_dim=8, tokens=2)

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=create_model))
    language_model = FakeLanguageModel()
    vision = TimmFusedVisionBackbone(model_ids=("fake",), pretrained=False, num_views=1)
    adapter = QwenTimmVLAAdapter(
        language_model=language_model,
        vision_backbone=vision,
        sequence_config=SequenceConfig(action_query_tokens=2),
    )
    batch = AdapterBatch(
        input_ids=torch.tensor([[1, 2, 0, 0]], dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        pixel_values=torch.zeros(1, 3, 8, 8),
        action_mask=torch.tensor([[False, False, True, True]]),
    )

    adapter.forward_with_action_queries(batch, torch.zeros(2, 8))

    assert isinstance(language_model.seen_logits_to_keep, torch.Tensor)
    assert language_model.seen_logits_to_keep.numel() == 0
