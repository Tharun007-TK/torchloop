"""
torchlite.evaluator
-------------------
One-call classification diagnostics. No more assembling sklearn +
matplotlib calls manually across every project.

Usage:
    from torchlite import Evaluator

    ev = Evaluator(model, device="cuda")
    ev.report(val_loader, class_names=["cat", "dog"])
    ev.confusion_matrix(val_loader)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader


class Evaluator:
    """
    Classification model evaluator.

    Args:
        model       : Trained nn.Module.
        device      : 'cuda', 'cpu', or 'mps'. Auto-detects if None.
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(
        self,
        loader: DataLoader,
        class_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Print full sklearn classification report.

        Returns:
            dict with keys: accuracy, macro_f1, weighted_f1, per_class_f1
        """
        preds, targets = self._infer(loader)
        report = classification_report(
            targets, preds, target_names=class_names, zero_division=0
        )
        print(report)
        per_class = f1_score(targets, preds, average=None, zero_division=0).tolist()
        return {
            "accuracy": float((np.array(preds) == np.array(targets)).mean()),
            "macro_f1": float(
                f1_score(targets, preds, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(targets, preds, average="weighted", zero_division=0)
            ),
            "per_class_f1": {
                (class_names[i] if class_names else str(i)): round(v, 4)
                for i, v in enumerate(per_class)
            },
        }

    def confusion_matrix(
        self,
        loader: DataLoader,
        class_names: Optional[list[str]] = None,
        normalize: Optional[str] = "true",   # 'true' | 'pred' | 'all' | None
        figsize: tuple = (8, 6),
    ) -> plt.Figure:
        """
        Plot and return confusion matrix figure.
        """
        preds, targets = self._infer(loader)
        cm = confusion_matrix(targets, preds, normalize=normalize)
        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        return fig

    def f1_per_class(
        self,
        loader: DataLoader,
        class_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Returns per-class F1 as a dict. Clean for logging to W&B or MLflow.
        """
        preds, targets = self._infer(loader)
        scores = f1_score(targets, preds, average=None, zero_division=0)
        return {
            (class_names[i] if class_names else str(i)): round(float(s), 4)
            for i, s in enumerate(scores)
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer(self, loader: DataLoader) -> tuple[list, list]:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets.tolist())
        return all_preds, all_targets