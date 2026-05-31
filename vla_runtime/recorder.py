"""Evaluation result recording."""

from __future__ import annotations

import json
from pathlib import Path

from vla_runtime.buffers.episode import EpisodeResult


class EpisodeRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.output_dir / "episodes.jsonl"

    def write_episode(self, episode: EpisodeResult) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    def write_metrics(self, metrics: dict) -> None:
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
