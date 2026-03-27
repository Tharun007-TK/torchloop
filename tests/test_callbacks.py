"""Tests for torchloop callback integrations."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchloop import Trainer
from torchloop.callbacks import Callback, MLflowLogger, WandBLogger


def _make_loader(n: int = 32, features: int = 8, classes: int = 3) -> DataLoader:
    x_data = torch.randn(n, features)
    y_data = torch.randint(0, classes, (n,))
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=8)


def _make_model(features: int = 8, classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.Linear(features, 16), nn.ReLU(), nn.Linear(16, classes))


def test_callback_base_methods_are_noop() -> None:
    """Ensure base callback methods are safe no-op defaults."""
    callback = Callback()
    callback.on_train_begin({"start": True})
    callback.on_epoch_end(1, {"loss": 1.0})
    callback.on_train_end({"done": True})


def test_wandb_logger_raises_if_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ImportError when wandb is unavailable at runtime."""
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = WandBLogger(project="proj")

    with pytest.raises(ImportError, match="wandb is required"):
        logger.on_train_begin({"train": True})



def test_mlflow_logger_raises_if_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ImportError when mlflow is unavailable at runtime."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    logger = MLflowLogger(experiment_name="exp")

    with pytest.raises(ImportError, match="mlflow is required"):
        logger.on_train_begin({"train": True})



def test_trainer_accepts_empty_callbacks_list() -> None:
    """Allow explicit empty callback lists during fit execution."""
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        callbacks=[],
    )

    history = trainer.fit(_make_loader(), _make_loader(), epochs=2)
    assert len(history["train_loss"]) == 2



def test_trainer_calls_on_epoch_end() -> None:
    """Invoke callback epoch-end hook once per epoch with epoch indices."""

    class TestCallback(Callback):
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def on_epoch_end(self, epoch: int, logs: dict) -> None:
            self.epochs.append(epoch)

    callback = TestCallback()
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        callbacks=[callback],
    )
    trainer.fit(_make_loader(), _make_loader(), epochs=3)

    assert callback.epochs == [1, 2, 3]
