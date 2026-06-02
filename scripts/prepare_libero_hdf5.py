"""Prepare LIBERO HDF5 demonstrations for adapter training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic_adapter.datasets.libero import LiberoAdapterConfig, save_action_stats
from prismatic_adapter.datasets.libero_hdf5 import (
    LiberoHdf5Config,
    LiberoHdf5Dataset,
    LiberoSampleAdapter,
    compute_libero_hdf5_action_stats,
    discover_hdf5_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory or file containing LIBERO .hdf5/.h5 demos.")
    parser.add_argument("--output-json", default="outputs/libero_action_stats.json")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--tokenizer-path", default="pretrained_models/Qwen3.5-2B")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--sample-check", action="store_true", help="Read one AdapterBatch after writing stats.")
    parser.add_argument("--fallback-instruction", default=None)
    return parser.parse_args()


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    files = discover_hdf5_files(args.root)
    print(f"[prepare] found {len(files)} HDF5 files")
    for path in files[:5]:
        print(f"[prepare] file: {path}")
    if len(files) > 5:
        print(f"[prepare] ... {len(files) - 5} more")

    stats = compute_libero_hdf5_action_stats(root=args.root, action_key=args.action_key)
    save_action_stats(stats, args.output_json)
    print(f"[prepare] wrote action stats: {Path(args.output_json).resolve()}")

    if args.sample_check:
        tokenizer = load_tokenizer(args.tokenizer_path)
        dataset = LiberoHdf5Dataset(
            config=LiberoHdf5Config(
                root=args.root,
                action_key=args.action_key,
                action_horizon=args.action_horizon,
                fallback_instruction=args.fallback_instruction,
                max_episodes=1,
            ),
            adapter=LiberoSampleAdapter(
                tokenizer,
                LiberoAdapterConfig(image_size=args.image_size),
            ),
        )
        sample = dataset[0]
        print(f"[prepare] sample input_ids: {tuple(sample.input_ids.shape)}")
        print(f"[prepare] sample pixel_values: {tuple(sample.pixel_values.shape)}")
        print(f"[prepare] sample actions: {tuple(sample.actions.shape)}")
        print(f"[prepare] sample proprio: {tuple(sample.proprio.shape)}")


if __name__ == "__main__":
    main()
