"""
torchloop — Lightweight PyTorch utility library.

Modules:
    trainer   : Training loop, metric logging, checkpoint management
    evaluator : Classification report, confusion matrix, per-class F1
    exporter  : PyTorch → ONNX → TFLite with optional quantization
"""

__version__ = "0.2.3"
__author__ = "Tharun Kumar"

from torchloop.callbacks import (
    CSVLogger,
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    StopTraining,
)
from torchloop.edge import deploy_to_edge, estimate_model
from torchloop.evaluator import Evaluator
from torchloop.exporter import Exporter
from torchloop.trainer import Trainer

__all__ = [
    "Trainer",
    "Evaluator",
    "Exporter",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "CSVLogger",
    "StopTraining",
    "deploy_to_edge",
    "estimate_model",
    "__version__",
]