# Architecture

## Data Flow

```text
raw sample / remote observation
        |
        v
Processor
  - tokenize prompt or apply chat template
  - prepare image/proprio tensors
  - append ActionQuery placeholder tokens
  - emit AdapterBatch
        |
        v
ModelAdapter
  - embed input ids
  - replace action placeholders with learnable ActionQuery embeddings
  - run model-specific multimodal forward
  - return hidden states and segment masks
        |
        v
Conditioning
  - select layers from arbitrary-depth backbones
  - project hidden size into policy hidden size
  - compress arbitrary visual token counts into a fixed Raw budget
        |
        v
BridgeActionHead
  - attends over Raw visual tokens
  - attends over ActionQuery states and optional proprio token
  - predicts normalized action chunks [B, H, A]
```

## Why It Is Split This Way

The reference VLA-Adapter style implementation ties ActionQuery insertion,
visual-token positions, Hugging Face forward calls, and policy inputs closely to
one model class. That is efficient for reproducing one configuration, but it
makes model replacement hard.

This framework keeps the variable parts explicit:

- `processors/` owns tokenizer, chat-template, and image-preprocessing quirks.
- `model_adapters/` owns VLM-specific forward behavior and segment extraction.
- `components/conditioning.py` absorbs differences in layer count, hidden size,
  and visual token count.
- `training/config.py` and `TrainableConfig` decide what is frozen, unfrozen, or
  adapted with LoRA.
- `env_process/` isolates simulator dependencies from the model environment.

## ModelAdapter Contract

Every new model adapter must return:

- `hidden_states`: a list or tuple of `[B, S, D]` tensors;
- `segments.vision`: the visual-token slice used as the Raw branch;
- `segments.action_mask`: a bool mask for ActionQuery states;
- `fused_attention_mask`: the attention mask used by the VLM.

Whenever possible, new adapters should support `inputs_embeds` so ActionQuery
embeddings can be inserted without modifying the tokenizer vocabulary.

## Compatibility Strategy

Different backbones can disagree on almost every shape:

```text
language depth      -> LayerSelector maps to policy layers
hidden size         -> ConditionProjector maps to policy hidden size
visual token count  -> MeanPoolTokenCompressor maps to raw_token_budget
image processors    -> processor-specific AdapterBatch.pixel_values
prompt format       -> processor-specific action_mask
action scale        -> ActionNormalizer
```

The shared policy only depends on `BackboneOutput`, so a new VLM does not need a
new action head or trainer.
