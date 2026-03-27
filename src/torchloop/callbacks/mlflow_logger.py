"""MLflow callback integration for torchloop."""

from __future__ import annotations

from importlib import import_module
from typing import Optional

from torchloop.callbacks.base import Callback


class MLflowLogger(Callback):
    """Log training metrics to MLflow.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: Optional tracking server URI.
        run_name: Optional MLflow run name.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name

    def on_train_begin(self, logs: dict) -> None:
        """Initialize MLflow experiment and run."""
        try:
            mlflow = import_module("mlflow")
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MLflowLogger. "
                "Install with: pip install torchloop[logging]"
            ) from exc

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Log epoch metrics to MLflow."""
        try:
            mlflow = import_module("mlflow")
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MLflowLogger. "
                "Install with: pip install torchloop[logging]"
            ) from exc

        numeric_logs = {
            key: value
            for key, value in logs.items()
            if isinstance(value, (int, float))
        }
        mlflow.log_metrics(numeric_logs, step=epoch)

    def on_train_end(self, logs: dict) -> None:
        """End the active MLflow run."""
        try:
            mlflow = import_module("mlflow")
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MLflowLogger. "
                "Install with: pip install torchloop[logging]"
            ) from exc

        mlflow.end_run()
