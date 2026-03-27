"""Base callback abstractions for torchloop training events."""

from __future__ import annotations


class Callback:
    """Base callback with optional training lifecycle hooks.

    Subclasses can override any hook they need.
    """

    def on_train_begin(self, logs: dict) -> None:
        """Run before training starts."""

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Run at the end of each epoch."""

    def on_train_end(self, logs: dict) -> None:
        """Run after training finishes."""
