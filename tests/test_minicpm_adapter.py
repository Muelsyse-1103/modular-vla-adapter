from types import SimpleNamespace

import torch

from prismatic_adapter.config import SequenceConfig
from prismatic_adapter.model_adapters import MiniCPMVLAAdapter
from prismatic_adapter.types import AdapterBatch


class FakeMiniCPMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=2, image_token_id=999)
        self.embedding = torch.nn.Embedding(1200, 4)
        self.model = SimpleNamespace(language_model=torch.nn.Linear(4, 4))
        with torch.no_grad():
            self.embedding.weight.zero_()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        hidden = kwargs["inputs_embeds"]
        return SimpleNamespace(hidden_states=[hidden, hidden + 1.0])


def test_minicpm_adapter_replaces_action_queries_and_infers_segments():
    adapter = MiniCPMVLAAdapter(FakeMiniCPMModel(), SequenceConfig(action_query_tokens=2))
    batch = AdapterBatch(
        input_ids=torch.tensor([[101, 999, 999, 102, 0, 0]], dtype=torch.long),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        pixel_values={"pixel_values": torch.zeros(1, 3, 14, 16)},
        action_mask=torch.tensor([[False, False, False, False, True, True]]),
    )
    action_queries = torch.tensor([[[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]])

    output = adapter.forward_with_action_queries(batch, action_queries)

    assert output.segments.vision == slice(1, 3)
    assert output.segments.action_mask.equal(batch.action_mask)
    assert torch.allclose(output.hidden_states[-1][0, 4:], action_queries[0] + 1.0)
