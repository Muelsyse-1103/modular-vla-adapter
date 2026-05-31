# VLA Adapter Framework Map

The framework is organized around adapter replacement, not RL rollout. The
main path is:

```text
datasets.AdapterBatch
        |
        v
adapters.ModelAdapter
  - owns tokenizer/model/image-tower quirks
  - returns hidden states + explicit token segments
        |
        v
conditioning
  - layer selection
  - hidden-size projection
  - raw visual token compression
        |
        v
action_heads.BridgeActionHead
  - reads Raw visual tokens
  - reads ActionQuery states
  - reads optional proprio
        |
        v
pipeline.VLAAdapter
  - normalized action chunk [B, H, A]
```

## Public Package Layout

```text
prismatic_adapter/
├── pipeline.py              # VLAAdapter: complete adapter pipeline
├── adapters/                # model-specific adapters
│   ├── base.py              # ModelAdapter protocol
│   ├── qwen_vit.py          # Qwen3.5 + DINOv2/SigLIP ViT example
│   └── hf_prismatic.py      # Prismatic/OpenVLA-like HF adapter
├── conditioning/            # hidden-state compatibility layer
├── action_heads/            # continuous action heads
├── datasets/                # AdapterBatch, SampleAdapter, collator
├── training/                # trainer, optimizer, scheduler, LoRA, logging
├── runtime/                 # inference and checkpoint helpers
├── config.py                # adapter/policy/conditioning configs
├── sequence.py              # token insertion and segment extraction
└── types.py                 # shared dataclasses
```

The older internal paths remain valid:

```text
backbones/   -> adapters implementation internals
components/  -> conditioning/action/prompt internals
policy/      -> action_heads implementation internals
data.py      -> datasets compatibility module
```

## Where To Add Things

| Goal | Add or edit |
| --- | --- |
| Support a new MLLM/VLM | `prismatic_adapter/adapters/` |
| Change Raw/AQ conditioning | `prismatic_adapter/conditioning/` |
| Add Perceiver/Q-Former visual compression | `prismatic_adapter/conditioning/` |
| Add a new action decoder | `prismatic_adapter/action_heads/` |
| Support a new dataset format | `prismatic_adapter/datasets/` or a project-local `SampleAdapter` |
| Change training mechanics | `prismatic_adapter/training/` |
| Change inference/checkpoint behavior | `prismatic_adapter/runtime/` |

## Replacement Contract

To replace the backbone, implement `ModelAdapter.forward_with_action_queries`.
Everything after that point is model-agnostic.

To replace the action head, keep the condition tensor contract:

```python
raw_tokens:          [B, L, R, D]
action_query_tokens: [B, L, Q, D]
proprio_token:       [B, 1, D] or None
```

To replace a dataset, emit `AdapterBatch` with a valid `action_mask`. The core
pipeline does not know RLDS, LIBERO, CALVIN, or robot-specific field names.
