# Pretrained Model Directory

Place local model assets here. Large weights are intentionally ignored by git.

Expected local layout for the Qwen example:

```text
pretrained_models/
├── Qwen3.5-2B/
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors...
└── vision_cache/
    ├── hf/
    └── torch/
```

Download/cache the standard ViT pair with:

```bash
python scripts/download_vision_backbones.py --cache-dir pretrained_models/vision_cache
```
