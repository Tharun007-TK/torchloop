"""
torchlite — Lightweight PyTorch utility library.

Modules:
    trainer   : Training loop, metric logging, checkpoint management
    evaluator : Classification report, confusion matrix, per-class F1
    exporter  : PyTorch → ONNX → TFLite with optional quantization
"""

__version__ = "0.1.0"
__author__ = "Tharun Kumar"

from torchlite.evaluator import Evaluator
from torchlite.trainer import Trainer

__all__ = ["Trainer", "Evaluator", "__version__"]