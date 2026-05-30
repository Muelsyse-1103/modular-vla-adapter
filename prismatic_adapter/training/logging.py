"""Minimal local and optional W&B logging."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


class MetricAverager:
    def __init__(self, window: int = 50) -> None:
        self.values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.values[key].append(float(value))

    def compute(self) -> dict[str, float]:
        return {
            key: sum(values) / max(len(values), 1)
            for key, values in self.values.items()
            if values
        }


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: dict[str, Any], prefix: str = "train") -> None:
        record = {"step": step, "prefix": prefix, **metrics}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class WandbLogger:
    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: str | None,
        mode: str,
        name: str | None,
        config: dict[str, Any],
        directory: Path,
    ) -> None:
        self.enabled = enabled
        self.wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("W&B logging requested but `wandb` is not installed.") from exc
        self.wandb = wandb
        self.wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            name=name,
            dir=str(directory),
            config=config,
        )

    def log(self, step: int, metrics: dict[str, float], prefix: str = "train") -> None:
        if self.wandb is None:
            return
        self.wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
