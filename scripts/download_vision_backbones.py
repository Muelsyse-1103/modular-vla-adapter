"""Download/cache the standard fused ViT backbone used by the Qwen example.

The script intentionally relies on TIMM's normal cache mechanism. Set
`--cache-dir pretrained_models/vision_cache` to keep downloaded assets under
the project workspace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_VISION_MODEL_IDS = (
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_so400m_patch14_siglip_224",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        default=None,
        help="TIMM model id to cache. Can be passed multiple times.",
    )
    parser.add_argument(
        "--cache-dir",
        default="pretrained_models/vision_cache",
        help="Cache directory for torch/timm/huggingface assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))

    try:
        import timm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `timm`. Install optional dependencies with "
            "`pip install -e .[hf]` in an environment that already has PyTorch."
        ) from exc

    model_ids = tuple(args.model_ids or DEFAULT_VISION_MODEL_IDS)
    for model_id in model_ids:
        print(f"[download] caching TIMM model: {model_id}")
        model = timm.create_model(model_id, pretrained=True, num_classes=0)
        embed_dim = getattr(model, "embed_dim", "<unknown>")
        print(f"[download] ready: {model_id} embed_dim={embed_dim}")

    print(f"[download] cache root: {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
