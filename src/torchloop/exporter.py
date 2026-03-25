"""
torchloop.exporter
------------------
PyTorch → ONNX → TFLite in one place.
Requires: pip install torchloop[export]

Usage:
    from torchloop.exporter import Exporter

    exp = Exporter(model, input_shape=(1, 3, 224, 224))
    exp.to_onnx("model.onnx")
    exp.to_tflite("model.tflite", quantize=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class Exporter:
    """
    Handles model export from PyTorch to ONNX and TFLite.

    Args:
        model       : Trained nn.Module (will be set to eval mode).
        input_shape : Tuple describing one sample input e.g. (1, 3, 224, 224).
        device      : Device to run dummy forward pass on.
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.input_shape = input_shape
        self._dummy = torch.randn(*input_shape).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_onnx(self, path: str | Path, opset: int = 17) -> Path:
        """
        Export model to ONNX format.

        Args:
            path  : Output .onnx file path.
            opset : ONNX opset version. Default 17 covers most torch ops.

        Returns:
            Resolved path to exported file.
        """
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "onnx is not installed. Run: pip install torchloop[export]"
            )

        path = Path(path)
        torch.onnx.export(
            self.model,
            self._dummy,
            str(path),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        model_onnx = onnx.load(str(path))
        onnx.checker.check_model(model_onnx)
        print(f"  ONNX export verified → {path}")
        return path

    def to_tflite(
        self,
        path: str | Path,
        quantize: bool = False,
        onnx_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Export model to TFLite via ONNX → TF → TFLite pipeline.

        Args:
            path      : Output .tflite file path.
            quantize  : If True, applies dynamic range quantization.
            onnx_path : Intermediate .onnx file path. Auto-generated if None.

        Returns:
            Resolved path to exported .tflite file.

        Note:
            Requires tensorflow and onnx2tf installed.
            pip install torchloop[export] onnx2tf
        """
        try:
            import onnx2tf
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "tensorflow or onnx2tf not installed.\n"
                "Run: pip install torchloop[export] onnx2tf"
            )

        path = Path(path)

        # Step 1: Export to ONNX first
        _onnx_path = Path(onnx_path) if onnx_path else path.with_suffix(".onnx")
        self.to_onnx(_onnx_path)

        # Step 2: ONNX → SavedModel via onnx2tf
        saved_model_dir = path.parent / "_tflite_savedmodel_tmp"
        onnx2tf.convert(
            input_onnx_file_path=str(_onnx_path),
            output_folder_path=str(saved_model_dir),
            not_use_onnxsim=False,
            verbosity="error",
        )

        # Step 3: SavedModel → TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("  Quantization: dynamic range enabled.")

        tflite_model = converter.convert()
        path.write_bytes(tflite_model)
        size_kb = path.stat().st_size / 1024
        print(f"  TFLite export → {path}  ({size_kb:.1f} KB)")
        return path
