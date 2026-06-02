# Training

The framework training path mirrors the important parts of the reference
`full-vla-adapter/vla-scripts/finetune.py` without coupling the loop to one
dataset or one Hugging Face model class.

## Components

```text
Dataset / SampleAdapter
  -> AdapterBatch
  -> PaddedBatchCollator
  -> AdapterTrainer
      - AdapterTrainStep
      - AdamW
      - warmup + multistep/cosine scheduler
      - gradient accumulation
      - AMP
      - checkpoint/resume
      - JSONL and optional W&B logging
```

## Dataset Contract

A training dataset item should be an `AdapterBatch`:

```python
AdapterBatch(
    input_ids=...,
    attention_mask=...,
    pixel_values=...,
    action_mask=...,
    actions=...,      # [H, action_dim] for one sample
    proprio=...,      # optional
)
```

Use `SampleAdapter` when your native sample is RLDS/LIBERO/CALVIN-specific.
The framework deliberately does not decide how to chunk actions or preprocess
camera images; those choices belong to the dataset adapter.

For LIBERO-style records, the built-in adapter expects one Python mapping with:

```python
{
    "instruction": "...",
    "image_primary": ...,  # HWC or CHW RGB
    "image_wrist": ...,    # optional if image_keys excludes it
    "proprio": ...,        # 8D for LIBERO
    "actions": ...,        # [H, 7]
}
```

Example dataset wrapper:

```python
from prismatic_adapter.datasets import LiberoSampleAdapter, compute_action_stats

adapter = LiberoSampleAdapter(tokenizer)
item = adapter(raw_libero_sample)
stats = compute_action_stats(raw_samples)
```

## Qwen3.5 + ViT Training Entry

`scripts/train_qwen35_vit.py` builds:

- `pretrained_models/Qwen3.5-2B` as the language model;
- `vit_large_patch14_reg4_dinov2.lvd142m + vit_so400m_patch14_siglip_224` as the fused vision stack;
- a 1024-dim Bridge policy;
- optional LoRA on the Qwen language model.

Example:

```bash
python scripts/download_vision_backbones.py \
  --cache-dir pretrained_models/vision_cache \
  --hf-endpoint https://hf-mirror.com

# The downloader falls back to:
# pretrained_models/vision_cache/files/<timm-model-id>/model.safetensors
# when the mirror does not satisfy the standard Hugging Face cache metadata API.

python scripts/train_qwen35_vit.py \
  --dataset-factory my_project.datasets:build_libero_dataset \
  --dataset-kwargs-json "{\"root\":\"data/libero\",\"name\":\"libero_object_no_noops\"}" \
  --vision-pretrained \
  --vision-cache-dir pretrained_models/vision_cache/hf \
  --use-lora \
  --batch-size 8 \
  --grad-accumulation-steps 8 \
  --max-steps 100000 \
  --output-dir outputs/qwen35_vit_libero_object
```

The dataset factory can return:

```python
train_dataset
(train_dataset, val_dataset)
{"train": train_dataset, "val": val_dataset}
```

## Action Normalization

Pass an action statistics JSON if dataset actions are environment-scale:

```json
{
  "low": [-1, -1, -1, -1, -1, -1, -1],
  "high": [1, 1, 1, 1, 1, 1, 1],
  "mask": [true, true, true, true, true, true, true]
}
```

Then run with:

```bash
--action-stats-json path/to/action_stats.json
```

## Checkpoints

Each training checkpoint contains:

- adapter config;
- ActionQuery weights;
- condition projector;
- Bridge action head;
- proprio projector if present;
- adapter-owned backbone modules such as the Qwen+ViT vision projector;
- optimizer and scheduler state for exact resume.

Use `--resume-path outputs/.../latest.pt` to continue training.

## Remote Evaluation

After training, run the environment process separately and evaluate with:

```bash
python scripts/eval_qwen35_vit_remote.py \
  --endpoint tcp://127.0.0.1:5555 \
  --qwen-path pretrained_models/Qwen3.5-2B \
  --checkpoint outputs/qwen35_vit_libero_object/latest.pt \
  --action-stats-json path/to/action_stats.json \
  --vision-pretrained \
  --vision-cache-dir pretrained_models/vision_cache/hf \
  --task-limit 1
```

Use `--vision-pretrained` only after the TIMM vision weights are available in
the project-local vision cache.
