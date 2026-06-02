"""Generic training loop for Prismatic VLA adapters."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from prismatic_adapter.checkpoint import load_training_checkpoint, save_training_checkpoint
from prismatic_adapter.components.actions import ActionNormalizer
from prismatic_adapter.data import PaddedBatchCollator
from prismatic_adapter.model import PrismaticAdapterPolicy
from prismatic_adapter.training.config import TrainingConfig
from prismatic_adapter.training.logging import JsonlLogger, MetricAverager, WandbLogger
from prismatic_adapter.training.optim import (
    build_optimizer,
    build_scheduler,
    count_trainable_parameters,
    trainable_parameters,
)
from prismatic_adapter.training.step import AdapterTrainStep
from prismatic_adapter.training.utils import autocast_dtype, dataclass_to_dict, move_batch_to_device, set_seed
from prismatic_adapter.types import AdapterBatch


class AdapterTrainer:
    """Small but complete trainer: grad accumulation, AMP, validation, resume."""

    def __init__(
        self,
        model: PrismaticAdapterPolicy,
        train_dataset: Dataset | Iterable[AdapterBatch],
        config: TrainingConfig,
        collator: PaddedBatchCollator | None = None,
        val_dataset: Dataset | Iterable[AdapterBatch] | None = None,
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        config.validate()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.collator = collator
        self.action_normalizer = action_normalizer
        self.device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")

        set_seed(config.trainer.seed)
        self.model.to(self.device)
        params = trainable_parameters(self.model)
        if not params:
            raise ValueError("no trainable parameters found")
        self.optimizer = build_optimizer(params, config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, config.scheduler, config.trainer.max_steps)
        self.stepper = AdapterTrainStep(self.model, action_normalizer=action_normalizer)
        self.global_step = 0

        self.run_dir = Path(config.checkpoint.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_logger = JsonlLogger(self.run_dir / config.logging.jsonl_name)
        self.metric_window = MetricAverager(window=max(config.logging.log_every_steps, 1))
        self.wandb_logger = WandbLogger(
            enabled=config.logging.use_wandb and config.logging.wandb_mode != "disabled",
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            mode=config.logging.wandb_mode,
            name=config.logging.run_name,
            config=dataclass_to_dict(config),
            directory=self.run_dir,
        )

        if config.checkpoint.resume_path is not None:
            checkpoint = load_training_checkpoint(
                model=self.model,
                path=config.checkpoint.resume_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                load_backbone=config.checkpoint.load_backbone_on_resume,
            )
            self.global_step = int(checkpoint.get("step", 0))

    def _loader(self, dataset: Dataset | Iterable[AdapterBatch], shuffle: bool) -> DataLoader:
        # IterableDataset is a Dataset subclass but must not use DataLoader.shuffle.
        use_shuffle = shuffle and isinstance(dataset, Dataset) and not isinstance(
            dataset, IterableDataset
        )
        return DataLoader(
            dataset,
            batch_size=self.config.trainer.batch_size,
            shuffle=use_shuffle,
            num_workers=self.config.trainer.num_workers,
            collate_fn=self.collator,
            pin_memory=self.device.type == "cuda",
        )

    def _amp_context(self):
        dtype = autocast_dtype(self.config.trainer.amp_dtype)
        if dtype is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _save(self, step: int, latest: bool = False) -> None:
        if latest:
            path = self.run_dir / "latest.pt"
        else:
            path = self.run_dir / f"step-{step:06d}.pt"
        save_training_checkpoint(
            model=self.model,
            path=path,
            step=step,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            extra={"trainable_parameters": count_trainable_parameters(self.model)},
        )

    def _log(self, step: int, metrics: dict[str, float], prefix: str) -> None:
        self.jsonl_logger.log(step=step, metrics=metrics, prefix=prefix)
        self.wandb_logger.log(step=step, metrics=metrics, prefix=prefix)

    def fit(self) -> None:
        self.model.train()
        loader = self._loader(self.train_dataset, shuffle=True)
        accumulation = self.config.trainer.grad_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)

        while self.global_step < self.config.trainer.max_steps:
            for batch_idx, batch in enumerate(loader):
                batch = move_batch_to_device(batch, self.device)
                with self._amp_context():
                    loss, metrics = self.stepper(batch)
                    scaled_loss = loss / accumulation
                scaled_loss.backward()
                self.metric_window.update(metrics)

                if (batch_idx + 1) % accumulation != 0:
                    continue

                if self.config.trainer.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(self.model),
                        self.config.trainer.clip_grad_norm,
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.global_step % self.config.logging.log_every_steps == 0:
                    smoothed = self.metric_window.compute()
                    smoothed["learning_rate"] = float(self.scheduler.get_last_lr()[0])
                    self._log(self.global_step, smoothed, "train")

                if (
                    self.config.trainer.validate_every_steps is not None
                    and self.val_dataset is not None
                    and self.global_step % self.config.trainer.validate_every_steps == 0
                ):
                    self.evaluate()
                    self.model.train()

                if (
                    self.config.checkpoint.save_every_steps > 0
                    and self.global_step % self.config.checkpoint.save_every_steps == 0
                ):
                    self._save(self.global_step, latest=self.config.checkpoint.save_latest_only)

                if self.global_step >= self.config.trainer.max_steps:
                    break

        self._save(self.global_step, latest=True)
        self.wandb_logger.finish()

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        if self.val_dataset is None:
            return {}
        self.model.eval()
        loader = self._loader(self.val_dataset, shuffle=False)
        averager = MetricAverager(window=10_000)
        for idx, batch in enumerate(loader):
            if (
                self.config.trainer.max_validation_batches is not None
                and idx >= self.config.trainer.max_validation_batches
            ):
                break
            batch = move_batch_to_device(batch, self.device)
            with self._amp_context():
                _, metrics = self.stepper(batch)
            averager.update(metrics)
        metrics = averager.compute()
        self._log(self.global_step, metrics, "val")
        return metrics
