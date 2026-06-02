"""RLDS-compatible dataset adapters.

RLDS is a storage convention rather than a model input contract. This module
normalizes RLDS episodes into the same LIBERO-style raw samples consumed by the
framework processors:

```python
{
    "instruction": str,
    "image_primary": np.ndarray,
    "image_wrist": np.ndarray,  # optional
    "proprio": np.ndarray,
    "actions": np.ndarray,      # [H, action_dim]
}
```

The model-specific processor then turns that raw sample into an `AdapterBatch`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import IterableDataset

DEFAULT_RLDS_PRIMARY_IMAGE_KEYS = (
    "observation/image",
    "observation/image_primary",
    "observation/exterior_image_1_left",
    "image",
    "image_primary",
)
DEFAULT_RLDS_WRIST_IMAGE_KEYS = (
    "observation/wrist_image",
    "observation/image_wrist",
    "observation/hand_image",
    "observation/robot0_eye_in_hand_image",
    "wrist_image",
    "image_wrist",
)
DEFAULT_RLDS_PROPRIO_KEYS = (
    "observation/proprio",
    "observation/state",
    "observation/robot_state",
    "proprio",
    "state",
)
DEFAULT_RLDS_LANGUAGE_KEYS = (
    "language_instruction",
    "natural_language_instruction",
    "instruction",
    "task_description",
    "observation/natural_language_instruction",
)


@dataclass(frozen=True)
class RldsConfig:
    """Mapping and iteration options for RLDS episodes."""

    tfds_name: str | None = None
    data_dir: str | None = None
    split: str = "train"
    shuffle_files: bool = False
    action_key: str = "action"
    steps_key: str = "steps"
    primary_image_keys: tuple[str, ...] = DEFAULT_RLDS_PRIMARY_IMAGE_KEYS
    wrist_image_keys: tuple[str, ...] = DEFAULT_RLDS_WRIST_IMAGE_KEYS
    proprio_keys: tuple[str, ...] = DEFAULT_RLDS_PROPRIO_KEYS
    language_keys: tuple[str, ...] = DEFAULT_RLDS_LANGUAGE_KEYS
    fallback_instruction: str | None = None
    action_horizon: int = 8
    frame_stride: int = 1
    sample_stride: int = 1
    max_episodes: int | None = None
    max_steps: int | None = None

    def validate(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")


class RldsEpisodeDataset(IterableDataset):
    """Iterable dataset over in-memory or TFDS-style RLDS episodes."""

    def __init__(
        self,
        episodes: Iterable[Mapping[str, Any]],
        adapter: Any,
        config: RldsConfig | None = None,
    ) -> None:
        self.episodes = episodes
        self.adapter = adapter
        self.config = config or RldsConfig()
        self.config.validate()

    def __iter__(self):
        for raw_sample in iter_rlds_samples(self.episodes, self.config):
            yield self.adapter(raw_sample)


class RldsTfdsDataset(IterableDataset):
    """Load RLDS episodes from TensorFlow Datasets and emit `AdapterBatch` items."""

    def __init__(self, config: RldsConfig, adapter: Any) -> None:
        if config.tfds_name is None:
            raise ValueError("RldsTfdsDataset requires config.tfds_name")
        config.validate()
        self.config = config
        self.adapter = adapter

    def __iter__(self):
        for raw_sample in iter_rlds_samples(_iter_tfds_episodes(self.config), self.config):
            yield self.adapter(raw_sample)


def iter_rlds_samples(
    episodes: Iterable[Mapping[str, Any]],
    config: RldsConfig,
) -> Iterable[dict[str, Any]]:
    """Yield normalized raw samples from RLDS episodes."""

    config.validate()
    episode_count = 0
    emitted = 0
    for episode in episodes:
        steps = list(_steps(episode, config.steps_key))
        if not steps:
            continue
        instruction = _instruction(episode, steps, config)
        for step_idx in range(0, len(steps), config.sample_stride):
            sample = {
                "instruction": instruction,
                "image_primary": _read_required(steps[step_idx], config.primary_image_keys),
                "actions": _action_chunk(steps, step_idx, config),
                "proprio": _read_proprio(steps[step_idx], config),
            }
            wrist = _read_optional(steps[step_idx], config.wrist_image_keys)
            if wrist is not None:
                sample["image_wrist"] = wrist
            yield sample
            emitted += 1
            if config.max_steps is not None and emitted >= config.max_steps:
                return
        episode_count += 1
        if config.max_episodes is not None and episode_count >= config.max_episodes:
            return


def iter_rlds_action_samples(
    episodes: Iterable[Mapping[str, Any]],
    action_key: str = "action",
    steps_key: str = "steps",
) -> Iterable[Mapping[str, Any]]:
    """Yield action arrays for action-stat computation."""

    config = RldsConfig(action_key=action_key, steps_key=steps_key)
    for episode in episodes:
        actions = []
        for step in _steps(episode, config.steps_key):
            value = _get_path(step, config.action_key)
            if value is not None:
                actions.append(np.asarray(value))
        if actions:
            yield {"actions": np.asarray(actions)}


def build_rlds_tfds_dataset(config: RldsConfig, adapter: Any):
    """Factory wrapper used by training entry points."""

    return RldsTfdsDataset(config=config, adapter=adapter)


def _iter_tfds_episodes(config: RldsConfig) -> Iterable[Mapping[str, Any]]:
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "RLDS TFDS loading requires tensorflow-datasets. Install the rlds extras."
        ) from exc

    # Keep TensorFlow off the GPU so PyTorch training can use it exclusively.
    tf.config.set_visible_devices([], "GPU")

    dataset = tfds.builder(config.tfds_name, data_dir=config.data_dir).as_dataset(
        split=config.split,
        shuffle_files=config.shuffle_files,
    )
    for episode in dataset:
        yield _materialize_rlds_tree(episode)


def _materialize_rlds_tree(value: Any) -> Any:
    """Convert TF tensors and nested RLDS step datasets into Python/numpy values."""

    if _is_tf_dataset(value):
        return [_materialize_rlds_tree(step) for step in value]

    if isinstance(value, Mapping):
        return {key: _materialize_rlds_tree(child) for key, child in value.items()}

    if hasattr(value, "numpy"):
        converted = value.numpy()
        if isinstance(converted, bytes):
            return converted
        return np.asarray(converted)

    if isinstance(value, (list, tuple)):
        return [_materialize_rlds_tree(item) for item in value]

    return value


def _is_tf_dataset(value: Any) -> bool:
    try:
        import tensorflow as tf
    except ImportError:
        return hasattr(value, "element_spec") and type(value).__name__ == "DatasetV2"
    return isinstance(value, tf.data.Dataset)


def _steps(episode: Mapping[str, Any], steps_key: str) -> Iterable[Mapping[str, Any]]:
    value = _get_path(episode, steps_key)
    if value is None:
        raise KeyError(f"RLDS episode is missing steps key: {steps_key}")
    if isinstance(value, list):
        return value
    if _is_tf_dataset(value):
        return [_materialize_rlds_tree(step) for step in value]
    if hasattr(value, "as_numpy_iterator"):
        return (_materialize_rlds_tree(step) for step in value.as_numpy_iterator())
    if isinstance(value, Mapping):
        return _mapping_steps(value)
    return value


def _mapping_steps(value: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    length = _nested_length(value)
    for idx in range(length):
        yield _index_nested(value, idx)


def _nested_length(value: Any) -> int:
    if isinstance(value, Mapping):
        for child in value.values():
            try:
                return _nested_length(child)
            except TypeError:
                continue
        raise TypeError("could not infer RLDS steps length from mapping")
    array = np.asarray(value)
    if array.ndim == 0:
        raise TypeError("scalar value cannot define RLDS steps length")
    return int(array.shape[0])


def _index_nested(value: Any, idx: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _index_nested(child, idx) for key, child in value.items()}
    array = np.asarray(value)
    if array.ndim == 0:
        return value
    return array[idx]


def _instruction(
    episode: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    config: RldsConfig,
) -> str:
    for key in config.language_keys:
        value = _get_path(episode, key)
        if value is not None:
            return _decode(value)
    for step in steps:
        for key in config.language_keys:
            value = _get_path(step, key)
            if value is not None:
                return _decode(value)
    if config.fallback_instruction is not None:
        return config.fallback_instruction
    raise KeyError("RLDS episode is missing language instruction")


def _read_required(step: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    value = _read_optional(step, keys)
    if value is None:
        raise KeyError(f"RLDS step is missing one of keys: {keys}")
    return value


def _read_optional(step: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        value = _get_path(step, key)
        if value is not None:
            return np.asarray(value)
    return None


def _read_proprio(step: Mapping[str, Any], config: RldsConfig) -> np.ndarray:
    value = _read_optional(step, config.proprio_keys)
    if value is None:
        raise KeyError(f"RLDS step is missing one of proprio keys: {config.proprio_keys}")
    return np.asarray(value).reshape(-1).astype(np.float32)


def _action_chunk(
    steps: Sequence[Mapping[str, Any]],
    step_idx: int,
    config: RldsConfig,
) -> np.ndarray:
    actions = []
    for offset in range(config.action_horizon):
        idx = min(step_idx + offset * config.frame_stride, len(steps) - 1)
        value = _get_path(steps[idx], config.action_key)
        if value is None:
            raise KeyError(f"RLDS step is missing action key: {config.action_key}")
        actions.append(np.asarray(value).reshape(-1))
    return np.asarray(actions, dtype=np.float32)


def _get_path(container: Any, path: str) -> Any | None:
    current = container
    for part in path.split("/"):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value.item())
        return _decode(value.reshape(-1)[0])
    return str(value)


def as_tuple(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value)
