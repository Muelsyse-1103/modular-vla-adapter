"""Small YAML configuration helpers for script entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_defaults(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML configs require PyYAML. Install the data/runtime extras.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping")
    return flatten_script_config(data)


def flatten_script_config(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested script configs into argparse destination names."""

    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and key in _SECTION_PREFIXES:
            prefix = _SECTION_PREFIXES[key]
            for child_key, child_value in flatten_script_config(value).items():
                flat_key = f"{prefix}{child_key}" if prefix else child_key
                flat[_ALIASES.get(flat_key, flat_key)] = child_value
        else:
            normalized_key = key.replace("-", "_")
            flat[_ALIASES.get(normalized_key, normalized_key)] = value
    return flat


_SECTION_PREFIXES = {
    "dataset": "",
    "libero": "libero_",
    "model": "",
    "vision": "vision_",
    "policy": "",
    "training": "",
    "trainable": "train_",
    "lora": "lora_",
    "logging": "",
}

_ALIASES = {
    "lora_use_lora": "use_lora",
}
