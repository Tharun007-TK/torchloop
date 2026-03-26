"""Tests for callback utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchloop import Trainer
from torchloop.callbacks import CSVLogger, Callback, EarlyStopping, ModelCheckpoint, StopTraining


class TinyNet(nn.Module):
    """Simple network for callback tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def loaders() -> tuple[DataLoader, DataLoader]:
    """Create tiny train and validation loaders."""
    x_train = torch.randn(12, 10)
    y_train = torch.randint(0, 2, (12,))
    x_val = torch.randn(8, 10)
    y_val = torch.randint(0, 2, (8,))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)
    return train_loader, val_loader


@pytest.fixture
def trainer_parts() -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """Create model, optimizer, and criterion for Trainer setup."""
    model = TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def test_early_stopping_init_values() -> None:
    """Validate early stopping initialization fields."""
    callback = EarlyStopping(patience=3, monitor="val_loss", min_delta=0.1)
    assert callback.patience == 3
    assert callback.monitor == "val_loss"
    assert callback.min_delta == 0.1
    assert callback.best == float("inf")
    assert callback.counter == 0


def test_early_stopping_validation_errors() -> None:
    """Validate constructor error handling for invalid config."""
    with pytest.raises(ValueError, match="patience must be >= 1"):
        EarlyStopping(patience=0)

    with pytest.raises(ValueError, match="min_delta must be >= 0"):
        EarlyStopping(min_delta=-0.1)


def test_early_stopping_triggers_after_patience() -> None:
    """Raise StopTraining when monitored metric does not improve."""
    callback = EarlyStopping(patience=2, monitor="val_loss")
    callback.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
    callback.on_epoch_end(epoch=1, logs={"val_loss": 1.1})

    with pytest.raises(StopTraining):
        callback.on_epoch_end(epoch=2, logs={"val_loss": 1.2})


def test_model_checkpoint_save_best_only(tmp_path: Path) -> None:
    """Save checkpoint only when monitored metric improves."""
    model = TinyNet()
    path = tmp_path / "best.pt"
    callback = ModelCheckpoint(filepath=str(path), save_best_only=True)

    callback.on_epoch_end(epoch=0, logs={"model": model, "val_loss": 1.0})
    first_mtime = path.stat().st_mtime

    callback.on_epoch_end(epoch=1, logs={"model": model, "val_loss": 1.2})
    assert path.stat().st_mtime == first_mtime

    callback.on_epoch_end(epoch=2, logs={"model": model, "val_loss": 0.8})
    assert path.stat().st_mtime >= first_mtime


def test_model_checkpoint_save_all(tmp_path: Path) -> None:
    """Always save checkpoint when save_best_only is disabled."""
    model = TinyNet()
    path = tmp_path / "all.pt"
    callback = ModelCheckpoint(filepath=str(path), save_best_only=False)

    callback.on_epoch_end(epoch=0, logs={"model": model, "val_loss": 2.0})
    first_mtime = path.stat().st_mtime
    callback.on_epoch_end(epoch=1, logs={"model": model, "val_loss": 3.0})
    assert path.stat().st_mtime >= first_mtime


def test_model_checkpoint_noop_when_logs_missing(tmp_path: Path) -> None:
    """Do not crash when required keys are missing in logs."""
    path = tmp_path / "noop.pt"
    callback = ModelCheckpoint(filepath=str(path))

    callback.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
    callback.on_epoch_end(epoch=1, logs={"model": TinyNet()})
    assert not path.exists()


def test_csv_logger_writes_header_and_rows(tmp_path: Path) -> None:
    """Write CSV header once and append per-epoch values."""
    callback = CSVLogger(log_dir=str(tmp_path), filename="metrics.csv")
    callback.on_epoch_end(epoch=0, logs={"loss": 1.0, "val_loss": 1.1})
    callback.on_epoch_end(epoch=1, logs={"loss": 0.9, "val_loss": 1.0})

    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 3
    assert "epoch" in rows[0]


def test_trainer_calls_custom_callback_hooks(
    trainer_parts: tuple[nn.Module, torch.optim.Optimizer, nn.Module],
    loaders: tuple[DataLoader, DataLoader],
) -> None:
    """Ensure custom callback receives train, epoch, and batch hooks."""

    class Recorder(Callback):
        def __init__(self) -> None:
            self.events: list[str] = []

        def on_train_begin(self, logs=None) -> None:
            self.events.append("train_begin")

        def on_epoch_begin(self, epoch: int, logs=None) -> None:
            self.events.append(f"epoch_begin_{epoch}")

        def on_batch_end(self, batch: int, logs=None) -> None:
            self.events.append("batch_end")

        def on_epoch_end(self, epoch: int, logs=None) -> None:
            self.events.append(f"epoch_end_{epoch}")

        def on_train_end(self, logs=None) -> None:
            self.events.append("train_end")

    model, optimizer, criterion = trainer_parts
    train_loader, val_loader = loaders
    callback = Recorder()
    trainer = Trainer(model, optimizer, criterion, callbacks=[callback], device="cpu")
    trainer.fit(train_loader, val_loader, epochs=2)

    assert "train_begin" in callback.events
    assert "train_end" in callback.events
    assert "epoch_begin_1" in callback.events
    assert "epoch_end_2" in callback.events
    assert "batch_end" in callback.events


def test_trainer_handles_stop_training_callback(
    trainer_parts: tuple[nn.Module, torch.optim.Optimizer, nn.Module],
    loaders: tuple[DataLoader, DataLoader],
) -> None:
    """StopTraining raised by callback should halt training gracefully."""

    class StopAfterFirstEpoch(Callback):
        def on_epoch_end(self, epoch: int, logs=None) -> None:
            if epoch >= 1:
                raise StopTraining("stop")

    model, optimizer, criterion = trainer_parts
    train_loader, val_loader = loaders
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        callbacks=[StopAfterFirstEpoch()],
        device="cpu",
    )

    history = trainer.fit(train_loader, val_loader, epochs=5)
    assert len(history["train_loss"]) == 1
