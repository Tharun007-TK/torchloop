"""
torchlite.trainer
-----------------
Wraps the PyTorch training loop so you stop rewriting it.

Usage:
    from torchlite import Trainer

    trainer = Trainer(model, optimizer, criterion, device="cuda")
    trainer.fit(train_loader, val_loader, epochs=20)
    trainer.save("best.pt")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Minimal, opinionated PyTorch training loop.

    Args:
        model       : nn.Module to train.
        optimizer   : Any torch.optim optimizer.
        criterion   : Loss function (nn.Module or callable).
        device      : 'cuda', 'cpu', or 'mps'. Auto-detects if None.
        metric_fn   : Optional callable(preds, targets) → float for val metric.
        patience    : Early stopping patience (epochs). None = disabled.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: Optional[str] = None,
        metric_fn: Optional[Callable] = None,
        patience: Optional[int] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fn = metric_fn
        self.patience = patience

        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": [],
        }
        self._best_val_loss = float("inf")
        self._best_state: Optional[dict] = None
        self._no_improve_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> dict:
        """
        Train the model.

        Returns:
            history dict with train_loss, val_loss, val_metric per epoch.
        """
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            val_loss, val_metric = None, None
            if val_loader is not None:
                val_loss, val_metric = self._val_epoch(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_metric"].append(val_metric)
                self._checkpoint(val_loss)

            self._log(epoch, epochs, train_loss, val_loss, val_metric, time.time() - t0)

            if self._should_stop():
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            print("  Restored best model weights.")

        return self.history

    def save(self, path: str | Path) -> None:
        """Save model state dict to path."""
        torch.save(self.model.state_dict(), path)
        print(f"  Saved → {path}")

    def load(self, path: str | Path) -> None:
        """Load model state dict from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"  Loaded ← {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for inputs, targets in tqdm(loader, desc="  train", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(loader.dataset)

    def _val_epoch(self, loader: DataLoader) -> tuple[float, Optional[float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="  val  ", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                if self.metric_fn is not None:
                    all_preds.append(outputs.cpu())
                    all_targets.append(targets.cpu())
        avg_loss = total_loss / len(loader.dataset)
        metric = None
        if self.metric_fn is not None and all_preds:
            metric = self.metric_fn(
                torch.cat(all_preds), torch.cat(all_targets)
            )
        return avg_loss, metric

    def _checkpoint(self, val_loss: float) -> None:
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_state = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

    def _should_stop(self) -> bool:
        return (
            self.patience is not None
            and self._no_improve_count >= self.patience
        )

    @staticmethod
    def _log(epoch, epochs, train_loss, val_loss, val_metric, elapsed) -> None:
        parts = [f"Epoch [{epoch:>3}/{epochs}]", f"train_loss={train_loss:.4f}"]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_metric is not None:
            parts.append(f"val_metric={val_metric:.4f}")
        parts.append(f"({elapsed:.1f}s)")
        print("  " + "  ".join(parts))
