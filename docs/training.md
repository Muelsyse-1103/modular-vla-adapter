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
Use `processors/` when a model needs a different tokenizer, chat template, image
processor, or multimodal input dictionary. The dataset owns what a sample means;
the processor owns how that sample becomes an `AdapterBatch` for one model.

## Selecting Data Input Format

Training entries can select the storage backend without changing model code:

```bash
--dataset-format libero_hdf5
--dataset-format rlds
```

Both paths normalize native data into the same raw sample shape:

```python
{
    "instruction": "...",
    "image_primary": ...,
    "image_wrist": ...,
    "proprio": ...,
    "actions": ...,
}
```

Then the selected processor builds `AdapterBatch`. This keeps RLDS/HDF5 storage
details out of `model_adapters/`, `components/`, and `training/`.

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
  --libero-hdf5-root data/libero \
  --action-stats-json outputs/libero_action_stats.json \
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

## Built-in LIBERO HDF5 Dataset

If your LIBERO demonstrations are `.hdf5` or `.h5` files, first scan them and
write action normalization stats:

```bash
python scripts/prepare_libero_hdf5.py \
  --root data/libero \
  --output-json outputs/libero_action_stats.json \
  --sample-check
```

Then train with the built-in factory:

```bash
python scripts/train_qwen35_vit.py \
  --libero-hdf5-root data/libero \
  --libero-val-ratio 0.02 \
  --action-stats-json outputs/libero_action_stats.json \
  --qwen-path pretrained_models/Qwen3.5-2B \
  --vision-pretrained \
  --vision-cache-dir pretrained_models/vision_cache/hf \
  --use-lora \
  --lora-rank 64 \
  --batch-size 8 \
  --grad-accumulation-steps 8 \
  --max-steps 100000 \
  --output-dir outputs/qwen35_vit_libero_object
```

The same flow is captured in YAML:

```bash
python scripts/train_qwen35_vit.py \
  --config configs/train_libero_qwen35_vit.example.yaml
```

and in a PowerShell example:

```text
configs/train_libero_hdf5_qwen35_vit.example.ps1
```

## MiniCPM-V Training Entry

`scripts/train_minicpm_v.py` demonstrates a second model family. It reuses the
same LIBERO HDF5 dataset, trainer, LoRA path, freezing switches, ActionQuery
module, conditioning stack, and Bridge action head.

```bash
python scripts/train_minicpm_v.py \
  --config configs/train_libero_minicpm_v.example.yaml \
  --libero-hdf5-root data/libero \
  --max-steps 1000 \
  --output-dir outputs/minicpm_v_libero
```

This entry differs only at the replaceable edge:

```text
MiniCPMVBatchProcessor -> MiniCPMVLAAdapter -> shared policy/trainer
```

Common field overrides:

```bash
--libero-action-key actions
--libero-image-keys image_primary,image_wrist
--libero-primary-image-keys obs/agentview_rgb,obs/agentview_image
--libero-wrist-image-keys obs/eye_in_hand_rgb,obs/robot0_eye_in_hand_rgb
--libero-proprio-keys obs/ee_states,obs/gripper_states
--libero-fallback-instruction "pick up the object"
```

## RLDS / TFDS Dataset

RLDS support is intentionally optional. Install the extra dependencies only
when you need official-style TFDS/RLDS datasets:

```bash
pip install -e ".[rlds]"
```

Qwen example:

```bash
python scripts/train_qwen35_vit.py \
  --config configs/train_rlds_qwen35_vit.example.yaml \
  --rlds-tfds-name bridge \
  --rlds-data-dir path/to/tfds \
  --rlds-split train
```

Important mapping knobs:

```bash
--rlds-action-key action
--rlds-steps-key steps
--rlds-primary-image-keys observation/image,observation/image_primary
--rlds-wrist-image-keys observation/wrist_image,observation/image_wrist
--rlds-proprio-keys observation/proprio,observation/state
--rlds-language-keys language_instruction,natural_language_instruction,instruction
```

MiniCPM-V supports the same backend:

```bash
python scripts/train_minicpm_v.py \
  --dataset-format rlds \
  --rlds-tfds-name bridge \
  --rlds-data-dir path/to/tfds
```

Train/freeze switches:

```bash
--train-language-model
--train-vision-backbone
--train-vision-projector / --no-train-vision-projector
--train-action-queries / --no-train-action-queries
--train-conditioning / --no-train-conditioning
--train-action-head / --no-train-action-head
--train-proprio-projector / --no-train-proprio-projector
```

LoRA switches:

```bash
--use-lora
--lora-target language_model
--lora-rank 64
--lora-alpha 128
--lora-dropout 0.05
--lora-target-modules q_proj,k_proj,v_proj,o_proj
```

## W&B Logging

W&B is optional and disabled/offline in public examples. If you enable it on a
server, provide credentials through the environment, not through committed files:

```bash
export WANDB_API_KEY="<your-key>"
python scripts/train_qwen35_vit.py \
  --wandb \
  --wandb-entity your-team-or-user \
  --wandb-mode online
```

See `docs/security.md` before publishing training scripts or configs.

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
