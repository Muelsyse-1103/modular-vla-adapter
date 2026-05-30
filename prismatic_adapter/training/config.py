"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass(frozen=True)
class SchedulerConfig:
    name: Literal["constant", "multistep", "cosine"] = "multistep"
    warmup_steps: int = 0
    milestones: tuple[int, ...] = (100_000,)
    gamma: float = 0.1
    min_lr_ratio: float = 0.0

    def validate(self) -> None:
        if self.name not in {"constant", "multistep", "cosine"}:
            raise ValueError("scheduler name must be constant, multistep, or cosine")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")


@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: Path = Path("outputs")
    save_every_steps: int = 10_000
    save_latest_only: bool = False
    resume_path: Path | None = None
    load_backbone_on_resume: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    log_every_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "vla_adapter"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    run_name: str | None = None
    jsonl_name: str = "metrics.jsonl"


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool = False
    target: Literal["language_model", "backbone"] = "language_model"
    rank: int = 64
    alpha: int | None = None
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"
    init_lora_weights: str | bool = "gaussian"

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.target not in {"language_model", "backbone"}:
            raise ValueError("LoRA target must be language_model or backbone")


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int = 100_000
    batch_size: int = 8
    grad_accumulation_steps: int = 1
    num_workers: int = 0
    device: str = "cuda"
    amp_dtype: Literal["none", "float16", "bfloat16"] = "bfloat16"
    clip_grad_norm: float | None = 1.0
    seed: int = 7
    validate_every_steps: int | None = None
    max_validation_batches: int | None = None

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive")
        if self.amp_dtype not in {"none", "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be none, float16, or bfloat16")


@dataclass(frozen=True)
class TrainingConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)

    def validate(self) -> None:
        self.trainer.validate()
        self.scheduler.validate()
        self.lora.validate()
