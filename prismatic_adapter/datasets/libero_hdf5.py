"""HDF5-backed LIBERO demonstration datasets.

The loader is intentionally mapping-driven. Different LIBERO releases and
mirrors use slightly different dataset names, so the defaults cover the common
`data/demo_*/obs/...` layout while script arguments can override every key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from prismatic_adapter.components.actions import ActionStats
from prismatic_adapter.components.actions import ActionNormalizer
from prismatic_adapter.datasets.libero import (
    LiberoAdapterConfig,
    LiberoSampleAdapter,
    compute_action_stats,
    save_action_stats,
)
from prismatic_adapter.types import AdapterBatch


DEFAULT_PRIMARY_IMAGE_KEYS = (
    "obs/agentview_rgb",
    "obs/agentview_image",
    "obs/image_primary",
    "agentview_rgb",
    "image_primary",
)
DEFAULT_WRIST_IMAGE_KEYS = (
    "obs/eye_in_hand_rgb",
    "obs/robot0_eye_in_hand_rgb",
    "obs/wrist_image",
    "obs/image_wrist",
    "eye_in_hand_rgb",
    "image_wrist",
)
DEFAULT_PROPRIO_KEYS = (
    "obs/ee_states",
    "obs/gripper_states",
)
DEFAULT_PROPRIO_FALLBACK_KEYS = (
    "obs/proprio",
    "proprio",
    "obs/states",
    "states",
    "obs/joint_states",
)


@dataclass(frozen=True)
class LiberoHdf5Config:
    """Storage mapping for LIBERO HDF5 demonstrations."""

    root: str | Path
    action_key: str = "actions"
    primary_image_keys: tuple[str, ...] = DEFAULT_PRIMARY_IMAGE_KEYS
    wrist_image_keys: tuple[str, ...] = DEFAULT_WRIST_IMAGE_KEYS
    proprio_keys: tuple[str, ...] = DEFAULT_PROPRIO_KEYS
    proprio_fallback_keys: tuple[str, ...] = DEFAULT_PROPRIO_FALLBACK_KEYS
    language_keys: tuple[str, ...] = (
        "language_instruction",
        "instruction",
        "language",
        "task_description",
    )
    fallback_instruction: str | None = None
    action_horizon: int = 8
    frame_stride: int = 1
    sample_stride: int = 1
    max_episodes: int | None = None

    def validate(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")


@dataclass(frozen=True)
class LiberoHdf5IndexEntry:
    file_path: Path
    episode_path: str
    step: int


class LiberoHdf5Dataset(Dataset):
    """Index HDF5 demonstrations and return `AdapterBatch` training samples."""

    def __init__(
        self,
        config: LiberoHdf5Config,
        adapter: LiberoSampleAdapter,
    ) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise ImportError("LiberoHdf5Dataset requires `h5py`; install the data extras.") from exc

        config.validate()
        self.config = config
        self.adapter = adapter
        self.files = discover_hdf5_files(config.root)
        self.index = build_libero_hdf5_index(self.files, config)
        if not self.index:
            raise ValueError(f"no LIBERO HDF5 samples found under {config.root}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> AdapterBatch:
        entry = self.index[idx]
        raw_sample = read_libero_hdf5_sample(entry, self.config)
        return self.adapter(raw_sample)


def discover_hdf5_files(root: str | Path) -> list[Path]:
    path = Path(root)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"LIBERO HDF5 root does not exist: {path}")
    files = sorted([*path.rglob("*.hdf5"), *path.rglob("*.h5")])
    if not files:
        raise FileNotFoundError(f"no .hdf5/.h5 files found under {path}")
    return files


def build_libero_hdf5_index(
    files: Sequence[Path],
    config: LiberoHdf5Config,
) -> list[LiberoHdf5IndexEntry]:
    import h5py

    entries: list[LiberoHdf5IndexEntry] = []
    episode_count = 0
    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            for episode_path in _episode_paths(handle):
                actions = _read_dataset(handle[episode_path], config.action_key)
                if actions is None:
                    continue
                length = int(actions.shape[0])
                for step in range(0, length, config.sample_stride):
                    entries.append(
                        LiberoHdf5IndexEntry(
                            file_path=file_path,
                            episode_path=episode_path,
                            step=step,
                        )
                    )
                episode_count += 1
                if config.max_episodes is not None and episode_count >= config.max_episodes:
                    return entries
    return entries


def read_libero_hdf5_sample(
    entry: LiberoHdf5IndexEntry,
    config: LiberoHdf5Config,
) -> dict[str, Any]:
    import h5py

    with h5py.File(entry.file_path, "r") as handle:
        episode = handle[entry.episode_path]
        actions = _require_dataset(episode, config.action_key)
        action_chunk = _action_chunk(actions, entry.step, config.action_horizon, config.frame_stride)
        sample = {
            "instruction": _instruction(handle, episode, entry.file_path, config),
            "image_primary": _read_timestep_required(episode, config.primary_image_keys, entry.step),
            "actions": action_chunk,
            "proprio": _read_proprio(episode, config, entry.step),
        }
        wrist = _read_timestep_optional(episode, config.wrist_image_keys, entry.step)
        if wrist is not None:
            sample["image_wrist"] = wrist
        return sample


def iter_libero_hdf5_action_samples(
    root: str | Path,
    action_key: str = "actions",
) -> Iterable[Mapping[str, Any]]:
    import h5py

    for file_path in discover_hdf5_files(root):
        with h5py.File(file_path, "r") as handle:
            for episode_path in _episode_paths(handle):
                actions = _read_dataset(handle[episode_path], action_key)
                if actions is not None:
                    yield {"actions": np.asarray(actions)}


def compute_libero_hdf5_action_stats(
    root: str | Path,
    action_key: str = "actions",
    mask: Sequence[bool] | None = None,
):
    return compute_action_stats(
        iter_libero_hdf5_action_samples(root=root, action_key=action_key),
        action_key="actions",
        mask=mask,
    )


def build_libero_hdf5_dataset(
    root: str,
    tokenizer_path: str = "pretrained_models/Qwen3.5-2B",
    val_ratio: float = 0.0,
    action_stats_json: str | None = None,
    write_action_stats_json: str | None = None,
    action_key: str = "actions",
    primary_image_keys: str | Sequence[str] | None = None,
    wrist_image_keys: str | Sequence[str] | None = None,
    proprio_keys: str | Sequence[str] | None = None,
    image_size: int = 224,
    image_keys: str | Sequence[str] = "image_primary,image_wrist",
    action_horizon: int = 8,
    action_query_tokens: int = 64,
    prompt_template: str = "What action should the robot take to {instruction}?",
    fallback_instruction: str | None = None,
    sample_stride: int = 1,
    frame_stride: int = 1,
    max_episodes: int | None = None,
):
    """Factory consumed directly by `scripts/train_qwen35_vit.py`."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    hdf5_cfg = LiberoHdf5Config(
        root=root,
        action_key=action_key,
        primary_image_keys=_as_tuple(primary_image_keys) or DEFAULT_PRIMARY_IMAGE_KEYS,
        wrist_image_keys=_as_tuple(wrist_image_keys) or DEFAULT_WRIST_IMAGE_KEYS,
        proprio_keys=_as_tuple(proprio_keys) or DEFAULT_PROPRIO_KEYS,
        fallback_instruction=fallback_instruction,
        action_horizon=action_horizon,
        frame_stride=frame_stride,
        sample_stride=sample_stride,
        max_episodes=max_episodes,
    )
    adapter_cfg = LiberoAdapterConfig(
        image_keys=_as_tuple(image_keys) or ("image_primary", "image_wrist"),
        image_size=image_size,
        prompt_template=prompt_template,
        action_query_tokens=action_query_tokens,
    )
    action_normalizer = _load_action_normalizer(action_stats_json)
    adapter = LiberoSampleAdapter(
        tokenizer=tokenizer,
        config=adapter_cfg,
        action_normalizer=action_normalizer,
    )
    dataset = LiberoHdf5Dataset(config=hdf5_cfg, adapter=adapter)
    if write_action_stats_json is not None:
        stats = compute_libero_hdf5_action_stats(root=root, action_key=action_key)
        save_action_stats(stats, write_action_stats_json)
    if val_ratio <= 0:
        return dataset
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1) when validation is enabled")
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("validation split would leave no training samples")
    indices = list(range(len(dataset)))
    return Subset(dataset, indices[:train_size]), Subset(dataset, indices[train_size:])


def _episode_paths(handle: Any) -> list[str]:
    paths: list[str] = []
    root = handle["data"] if "data" in handle and hasattr(handle["data"], "keys") else handle
    for key in root.keys():
        node = root[key]
        if hasattr(node, "keys"):
            path = f"data/{key}" if root is not handle else key
            paths.append(path)
    return sorted(paths)


def _read_dataset(group: Any, key: str) -> Any | None:
    node: Any = group
    for part in key.split("/"):
        if part not in node:
            return None
        node = node[part]
    return node


def _require_dataset(group: Any, key: str) -> Any:
    value = _read_dataset(group, key)
    if value is None:
        raise KeyError(f"HDF5 episode is missing required key: {key}")
    return value


def _read_timestep_required(group: Any, keys: Sequence[str], step: int) -> np.ndarray:
    value = _read_timestep_optional(group, keys, step)
    if value is None:
        raise KeyError(f"HDF5 episode is missing one of keys: {keys}")
    return value


def _read_timestep_optional(group: Any, keys: Sequence[str], step: int) -> np.ndarray | None:
    for key in keys:
        dataset = _read_dataset(group, key)
        if dataset is not None:
            return np.asarray(dataset[min(step, dataset.shape[0] - 1)])
    return None


def _read_proprio(group: Any, config: LiberoHdf5Config, step: int) -> np.ndarray:
    values = []
    missing = False
    for key in config.proprio_keys:
        dataset = _read_dataset(group, key)
        if dataset is None:
            missing = True
            break
        values.append(np.asarray(dataset[min(step, dataset.shape[0] - 1)]).reshape(-1))
    if values and not missing:
        return np.concatenate(values, axis=0).astype(np.float32)

    for key in config.proprio_fallback_keys:
        dataset = _read_dataset(group, key)
        if dataset is not None:
            return np.asarray(dataset[min(step, dataset.shape[0] - 1)]).reshape(-1).astype(np.float32)
    raise KeyError("HDF5 episode is missing proprio keys")


def _action_chunk(actions: Any, step: int, horizon: int, stride: int) -> np.ndarray:
    indices = [min(step + offset * stride, actions.shape[0] - 1) for offset in range(horizon)]
    return np.asarray(actions)[indices].astype(np.float32)


def _instruction(handle: Any, episode: Any, file_path: Path, config: LiberoHdf5Config) -> str:
    for key in config.language_keys:
        for attrs in (episode.attrs, handle.attrs):
            if key in attrs:
                return _decode(attrs[key])
        dataset = _read_dataset(episode, key)
        if dataset is not None:
            return _decode(np.asarray(dataset)[0] if getattr(dataset, "shape", ()) else dataset[()])
    if config.fallback_instruction is not None:
        return config.fallback_instruction
    return file_path.stem.replace("_", " ")


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value.item())
        return _decode(value.reshape(-1)[0])
    return str(value)


def _as_tuple(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value)


def _load_action_normalizer(path: str | None) -> ActionNormalizer | None:
    if path is None:
        return None
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    stats = ActionStats(
        low=torch.tensor(data["low"], dtype=torch.float32),
        high=torch.tensor(data["high"], dtype=torch.float32),
        mask=torch.tensor(data["mask"], dtype=torch.bool) if "mask" in data else None,
    )
    return ActionNormalizer(stats)
