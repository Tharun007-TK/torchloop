"""Weights & Biases callback integration for torchloop."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

from torchloop.callbacks.base import Callback


class WandBLogger(Callback):
    """Log training metrics to Weights & Biases.

    Args:
        project: Weights & Biases project name.
        name: Optional run name.
        config: Optional run configuration dictionary.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config or {}

    def on_train_begin(self, logs: dict) -> None:
        """Initialize a W&B run."""
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandBLogger. "
                "Install with: pip install torchloop[logging]"
            ) from exc

        wandb.init(project=self.project, name=self.name, config=self.config)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Log epoch metrics to W&B."""
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandBLogger. "
                "Install with: pip install torchloop[logging]"
            ) from exc

        wandb.log(logs, step=epoch)

    def on_train_end(self, logs: dict) -> None:
        """Finish the active W&B run."""
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandBLogger. "
                "Install with: pip install torchloop[logging]"
            ) from exc

        wandb.finish()
