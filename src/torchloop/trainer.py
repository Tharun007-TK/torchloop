"""
torchlite.trainer
-----------------
Wraps the PyTorch training loop so you stop rewriting it.

Usage:
    from torchloop import Trainer
    import torch.optim.lr_scheduler as sched

    scheduler = sched.StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = Trainer(
        model, optimizer, criterion,
        device="cuda",
        scheduler=scheduler,
        amp=True,
        patience=5,
    )
    trainer.fit(train_loader, val_loader, epochs=30)
    trainer.save("best.pt")
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchloop.callbacks import Callback


class Trainer:
    """
    Minimal, opinionated PyTorch training loop.

    Args:
        model: nn.Module to train.
        optimizer: Any torch.optim optimizer.
        criterion: Loss function (nn.Module or callable).
        device: 'cuda', 'cpu', or 'mps'. Auto-detects if None.
        metric_fn: Optional callable(preds, targets) -> float for val metric.
        patience: Early stopping patience (epochs). None = disabled.
        scheduler: Any torch.optim.lr_scheduler. Steps once per epoch.
            ReduceLROnPlateau is handled automatically.
        amp: Deprecated alias for use_amp, kept for backward compatibility.
        use_amp: If True, enables automatic mixed precision on CUDA.
        accumulate_steps: Number of batches to accumulate gradients before an
            optimizer step.
        callbacks: Optional list of callback instances.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: Optional[str] = None,
        metric_fn: Optional[Callable] = None,
        patience: Optional[int] = None,
        scheduler: Optional[object] = None,
        amp: Optional[bool] = None,
        use_amp: bool = False,
        accumulate_steps: int = 1,
        callbacks: Optional[list[Callback]] = None,
    ):
        if accumulate_steps < 1:
            raise ValueError("accumulate_steps must be >= 1")

        if amp is not None:
            warnings.warn(
                "`amp` is deprecated and will be removed in a future release; "
                "use `use_amp` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_amp = amp

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fn = metric_fn
        self.patience = patience
        self.scheduler = scheduler
        self.accumulate_steps = accumulate_steps
        self.callbacks = list(callbacks) if callbacks is not None else []
        self._stop_early = False

        if use_amp and self.device != "cuda":
            warnings.warn(
                "AMP requested on non-CUDA device. AMP will be disabled.",
                UserWarning,
                stacklevel=2,
            )

        self.use_amp = bool(use_amp and self.device == "cuda")
        self.amp = self.use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": [],
            "lr": [],
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
    ) -> dict[str, list[Any]]:
        """
        Train the model.

        Returns:
            history dict with train_loss, val_loss, val_metric, lr per epoch.
        """
        self._run_callbacks("on_train_begin", dict(self.history))
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

            self._step_scheduler(val_loss)
            current_lr = self._current_lr()
            self.history["lr"].append(current_lr)

            self._log(
                epoch, epochs, train_loss, val_loss,
                val_metric, current_lr, time.time() - t0,
            )

            epoch_logs = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "lr": current_lr,
            }
            self._run_callbacks("on_epoch_end", epoch_logs)

            if self._should_stop():
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            print("  Restored best model weights.")

        self._run_callbacks("on_train_end", dict(self.history))
        return self.history

    def add_callback(self, callback: Callback) -> None:
        """Add a callback instance to the trainer."""
        self.callbacks.append(callback)

    def save(self, path: str | Path) -> None:
        """Save model state dict to path."""
        torch.save(self.model.state_dict(), path)
        print(f"  Saved → {path}")

    def load(self, path: str | Path) -> None:
        """Load model state dict from path."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        print(f"  Loaded ← {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        pending_steps = 0
        for batch_idx, (inputs, targets) in enumerate(
            tqdm(loader, desc="  train", leave=False)
        ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    raw_loss = self.criterion(outputs, targets)
                    loss = raw_loss / self.accumulate_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                raw_loss = self.criterion(outputs, targets)
                loss = raw_loss / self.accumulate_steps
                loss.backward()

            pending_steps += 1
            should_step = (batch_idx + 1) % self.accumulate_steps == 0
            if should_step:
                self._optimizer_step()
                pending_steps = 0

            total_loss += raw_loss.item() * inputs.size(0)

        if pending_steps > 0:
            self._optimizer_step()

        return total_loss / len(loader.dataset)

    def _optimizer_step(self) -> None:
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _val_epoch(self, loader: DataLoader) -> tuple[float, Optional[float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="  val  ", leave=False):
                inputs, targets = (
                    inputs.to(self.device), targets.to(self.device)
                )
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

    def _step_scheduler(self, val_loss: Optional[float]) -> None:
        if self.scheduler is None:
            return
        plateau = "ReduceLROnPlateau"
        if type(self.scheduler).__name__ == plateau:
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _should_stop(self) -> bool:
        return (
            self._stop_early
            or (
                self.patience is not None
                and self._no_improve_count >= self.patience
            )
        )

    def _run_callbacks(self, event: str, logs: dict[str, Any]) -> None:
        for callback in (self.callbacks or []):
            hook = getattr(callback, event, None)
            if hook is None:
                continue
            if event == "on_epoch_end":
                hook(int(logs.get("epoch", 0)), logs)
            else:
                hook(logs)

    @staticmethod
    def _log(
        epoch, epochs, train_loss, val_loss,
        val_metric, lr, elapsed,
    ) -> None:
        parts = [
            f"Epoch [{epoch:>3}/{epochs}]",
            f"train_loss={train_loss:.4f}",
        ]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_metric is not None:
            parts.append(f"val_metric={val_metric:.4f}")
        parts.append(f"lr={lr:.2e}")
        parts.append(f"({elapsed:.1f}s)")
        print("  " + "  ".join(parts))