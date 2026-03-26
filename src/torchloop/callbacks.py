"""Callback utilities for training lifecycle hooks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import torch


class StopTraining(Exception):
    """Raised by callbacks to request early stop of training."""


class Callback:
    """Base callback class.

    Subclass this and override hook methods to customize training behavior.
    """

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Called before the first epoch starts.

        Args:
            logs: Optional training state dictionary.
        """

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Called after training ends.

        Args:
            logs: Optional training state dictionary.
        """

    def on_epoch_begin(
        self,
        epoch: int,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number.
            logs: Optional training state dictionary.
        """

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number.
            logs: Optional training state dictionary.
        """

    def on_batch_end(
        self,
        batch: int,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Called after each training batch.

        Args:
            batch: Zero-based batch index.
            logs: Optional batch-level metrics.
        """


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for an improvement.
        monitor: Metric name expected in logs.
        min_delta: Minimum improvement required to reset patience.
    """

    def __init__(
        self,
        patience: int = 5,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Check metric value and raise StopTraining when patience is exceeded.

        Args:
            epoch: Current epoch number.
            logs: Metrics dictionary from trainer.
        """
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if current <= self.best - self.min_delta:
            self.best = float(current)
            self.counter = 0
            return

        self.counter += 1
        if self.counter >= self.patience:
            raise StopTraining(f"Early stopping triggered at epoch {epoch}")


class ModelCheckpoint(Callback):
    """Save model checkpoints based on monitored metric.

    Args:
        monitor: Metric name expected in logs.
        save_best_only: Save only when monitored metric improves.
        filepath: Path to write checkpoint file.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        filepath: str = "checkpoint.pt",
    ) -> None:
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.filepath = Path(filepath)
        self.best = float("inf")

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist model checkpoint if criteria are met.

        Args:
            epoch: Current epoch number.
            logs: Metrics dictionary from trainer.
        """
        logs = logs or {}
        model = logs.get("model")
        if model is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        if not self.save_best_only:
            torch.save(model.state_dict(), self.filepath)
            return

        if float(current) < self.best:
            self.best = float(current)
            torch.save(model.state_dict(), self.filepath)


class CSVLogger(Callback):
    """Log epoch metrics to a CSV file.

    Args:
        log_dir: Directory where CSV logs are written.
        filename: Name of CSV file.
    """

    def __init__(self, log_dir: str = "./logs", filename: str = "metrics.csv") -> None:
        self.log_dir = Path(log_dir)
        self.filename = filename
        self._initialized = False
        self._fieldnames: list[str] = []

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append epoch metrics to CSV.

        Args:
            epoch: Current epoch number.
            logs: Metrics dictionary from trainer.
        """
        logs = dict(logs or {})
        logs["epoch"] = epoch

        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.log_dir / self.filename

        if not self._initialized:
            self._fieldnames = sorted(logs.keys())
            with file_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow({k: logs.get(k) for k in self._fieldnames})
            self._initialized = True
            return

        with file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({k: logs.get(k) for k in self._fieldnames})
