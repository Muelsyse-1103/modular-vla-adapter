"""Download/cache the standard fused ViT backbone used by the Qwen example.

The script uses TIMM plus Hugging Face Hub and keeps downloaded assets under
`pretrained_models/vision_cache` by default. It defaults to the hf-mirror.com
endpoint, which is useful in China; pass `--hf-endpoint ""` to use the official
Hugging Face endpoint.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic_adapter.backbones.qwen_vit import _safe_weight_dir_name

DEFAULT_VISION_MODEL_IDS = (
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_so400m_patch14_siglip_224",
)
WEIGHTS_FILENAME = "model.safetensors"


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
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint mirror. Pass an empty string to leave HF_ENDPOINT unset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    hf_cache = cache_dir / "hf"
    torch_cache = cache_dir / "torch"
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["TORCH_HOME"] = str(torch_cache)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

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
        cfg = timm.models.get_pretrained_cfg(model_id)
        local_file = cache_dir / "files" / _safe_weight_dir_name(model_id) / WEIGHTS_FILENAME
        if local_file.exists():
            print(f"[download] ready: {model_id} file={local_file}")
            continue

        overlay = {"source": "hf-hub", "url": ""} if getattr(cfg, "hf_hub_id", None) else None
        try:
            model = timm.create_model(
                model_id,
                pretrained=True,
                pretrained_cfg_overlay=overlay,
                cache_dir=str(hf_cache),
                num_classes=0,
            )
            embed_dim = getattr(model, "embed_dim", "<unknown>")
            print(f"[download] ready: {model_id} embed_dim={embed_dim}")
            del model
            gc.collect()
        except Exception as exc:
            hf_hub_id = getattr(cfg, "hf_hub_id", None)
            if not hf_hub_id:
                raise
            print(f"[download] TIMM/HF cache failed for {model_id}: {exc}")
            _download_from_resolve_url(
                endpoint=args.hf_endpoint or "https://huggingface.co",
                repo_id=hf_hub_id,
                output_path=local_file,
            )
            print(f"[download] ready: {model_id} file={local_file}")

    print(f"[download] cache root: {cache_dir.resolve()}")


def _download_from_resolve_url(endpoint: str, repo_id: str, output_path: Path) -> None:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("Missing dependency `requests`; install the runtime/hf extras first.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".incomplete")
    url = f"{endpoint.rstrip('/')}/{repo_id}/resolve/main/{WEIGHTS_FILENAME}"
    print(f"[download] direct mirror download: {url}")
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_report = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                downloaded_mb = downloaded // (1024 * 1024)
                if downloaded_mb >= next_report:
                    if total:
                        total_mb = total // (1024 * 1024)
                        print(f"[download] {downloaded_mb} / {total_mb} MB")
                    else:
                        print(f"[download] {downloaded_mb} MB")
                    next_report = downloaded_mb + 256
    tmp_path.replace(output_path)


if __name__ == "__main__":
    main()
