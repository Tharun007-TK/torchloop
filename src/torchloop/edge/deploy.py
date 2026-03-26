"""Edge deployment helpers for ONNX, TFLite, and CoreML."""
# pyright: reportMissingImports=false

from __future__ import annotations

import shutil
import tempfile
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

import torch


def deploy_to_edge(
    model: torch.nn.Module,
    target: Literal["esp32", "rpi", "jetson", "android", "ios"],
    input_shape: tuple[int, ...],
    output_path: str,
    quantize: bool = True,
    quantize_type: Literal["int8", "float16"] = "int8",
    dynamic_axes: Optional[dict[int, str]] = None,
) -> Dict[str, Any]:
    """Deploy a PyTorch model to an edge-oriented artifact format.

    Args:
        model: Model instance to export.
        target: Target platform.
        input_shape: Model input shape, like ``(1, 3, 224, 224)``.
        output_path: Output path for final artifact.
        quantize: Whether to apply quantization where supported.
        quantize_type: Quantization precision for TFLite export.
        dynamic_axes: Optional ONNX dynamic axis mapping for input tensor.

    Returns:
        Deployment summary including format and estimated memory.

    Raises:
        ValueError: If configuration is invalid.
        ImportError: If optional dependencies are missing.
    """
    valid_targets = {"esp32", "rpi", "jetson", "android", "ios"}
    if target not in valid_targets:
        raise ValueError("target must be one of esp32, rpi, jetson, android, ios")
    if len(input_shape) < 2:
        raise ValueError("input_shape must contain at least batch and feature dims")
    if quantize_type not in {"int8", "float16"}:
        raise ValueError("quantize_type must be 'int8' or 'float16'")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    onnx_path = output.with_suffix(".onnx")
    if output.suffix == ".onnx":
        onnx_path = output

    _export_to_onnx(model, input_shape, onnx_path, dynamic_axes=dynamic_axes)

    artifact_path = onnx_path
    artifact_format = "onnx"

    if target in {"esp32", "android"}:
        artifact_path = output if output.suffix == ".tflite" else output.with_suffix(".tflite")
        if target == "esp32" and quantize_type != "int8":
            warnings.warn(
                "ESP32 is typically best with int8 quantization.",
                UserWarning,
                stacklevel=2,
            )
        _convert_to_tflite(onnx_path, artifact_path, quantize, quantize_type)
        artifact_format = "tflite"
    elif target == "rpi":
        if output.suffix == ".onnx":
            artifact_path = onnx_path
            artifact_format = "onnx"
        else:
            artifact_path = output.with_suffix(".tflite")
            _convert_to_tflite(onnx_path, artifact_path, quantize, quantize_type)
            artifact_format = "tflite"
    elif target == "ios":
        artifact_path = output if output.suffix == ".mlpackage" else output.with_suffix(".mlpackage")
        _convert_to_coreml(model, input_shape, artifact_path)
        artifact_format = "coreml"
    elif target == "jetson":
        artifact_path = onnx_path
        artifact_format = "onnx"

    size_mb = artifact_path.stat().st_size / (1024 * 1024)
    return {
        "target": target,
        "artifact_path": str(artifact_path),
        "format": artifact_format,
        "size_mb": round(size_mb, 4),
        "estimated_ram_mb": round(size_mb * 2.5, 4),
        "quantized": quantize,
        "quantize_type": quantize_type,
    }


def _export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    output_path: Path,
    dynamic_axes: Optional[dict[int, str]] = None,
) -> None:
    """Export a model to ONNX with optional dynamic axis mapping."""
    try:
        onnx_name = "".join(["on", "nx"])
        onnx = import_module(onnx_name)
    except ImportError as exc:
        raise ImportError("onnx is required. Install with: pip install torchloop[edge]") from exc

    model.eval()
    device = _get_model_device(model)
    dummy_input = torch.randn(*input_shape, device=device)

    axes = None
    if dynamic_axes is not None:
        axes = {
            "input": dynamic_axes,
            "output": dynamic_axes,
        }

    export_fn = cast(Any, torch.onnx.export)
    export_fn(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=axes,
        opset_version=17,
    )

    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)


def _convert_to_tflite(
    onnx_path: Path,
    output_path: Path,
    quantize: bool,
    quantize_type: Literal["int8", "float16"],
) -> None:
    """Convert ONNX model to TFLite using onnx2tf and TensorFlow."""
    try:
        onnx2tf_name = "".join(["onnx2", "tf"])
        tensorflow_name = "".join(["tensor", "flow"])
        onnx2tf = import_module(onnx2tf_name)
        tf = import_module(tensorflow_name)
    except ImportError as exc:
        raise ImportError(
            "tensorflow and onnx2tf are required for TFLite conversion. "
            "Install with: pip install torchloop[edge] onnx2tf"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="torchloop_onnx2tf_") as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_model_dir),
            not_use_onnxsim=False,
            verbosity="error",
        )

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if quantize_type == "float16":
                converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)


def _convert_to_coreml(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    output_path: Path,
) -> None:
    """Convert a PyTorch model to CoreML package format."""
    try:
        coremltools_name = "".join(["coreml", "tools"])
        ct = import_module(coremltools_name)
    except ImportError as exc:
        raise ImportError("coremltools is required. Install with: pip install torchloop[edge]") from exc

    model_cpu = model.to("cpu").eval()
    traced = torch.jit.trace(model_cpu, torch.randn(*input_shape))
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=input_shape)])

    temp_path = output_path.with_suffix(".mlmodel")
    mlmodel.save(str(temp_path))
    if output_path.suffix == ".mlmodel":
        return

    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_path), str(output_path / temp_path.name))


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the model device or CPU for parameterless models."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
