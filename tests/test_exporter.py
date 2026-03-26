"""Tests for exporter module with lightweight dependency mocking."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchloop import Exporter


class SimpleCNN(nn.Module):
    """Small model used for export tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def model() -> nn.Module:
    """Create a simple CNN model."""
    return SimpleCNN()


@pytest.fixture
def input_shape() -> tuple[int, int, int, int]:
    """Default input shape for tests."""
    return (1, 3, 16, 16)


def test_exporter_init_sets_fields(model: nn.Module, input_shape: tuple[int, ...]) -> None:
    """Initialize exporter and validate key fields."""
    exporter = Exporter(model, input_shape=input_shape, device="cpu")
    assert exporter.model is model
    assert exporter.input_shape == input_shape
    assert tuple(exporter._dummy.shape) == input_shape


def test_to_onnx_raises_without_onnx(monkeypatch: pytest.MonkeyPatch, model: nn.Module, input_shape: tuple[int, ...]) -> None:
    """Raise ImportError when onnx dependency is missing."""
    monkeypatch.setitem(sys.modules, "onnx", None)
    exporter = Exporter(model, input_shape=input_shape, device="cpu")

    with pytest.raises(ImportError, match="onnx is not installed"):
        exporter.to_onnx("x.onnx")


def test_to_onnx_writes_file_with_mocked_backend(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    input_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    """Export ONNX with mocked onnx and torch exporter functions."""
    onnx_stub = types.SimpleNamespace(
        load=lambda p: object(),
        checker=types.SimpleNamespace(check_model=lambda m: None),
    )
    monkeypatch.setitem(sys.modules, "onnx", onnx_stub)

    def fake_export(_model, _dummy, path, **_kwargs):
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    exporter = Exporter(model, input_shape=input_shape, device="cpu")
    out = exporter.to_onnx(tmp_path / "model.onnx")

    assert out.exists()
    assert out.suffix == ".onnx"


def test_to_tflite_raises_without_tf(monkeypatch: pytest.MonkeyPatch, model: nn.Module, input_shape: tuple[int, ...]) -> None:
    """Raise ImportError when TensorFlow pipeline dependencies are missing."""
    monkeypatch.setitem(sys.modules, "onnx2tf", None)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    exporter = Exporter(model, input_shape=input_shape, device="cpu")

    with pytest.raises(ImportError, match="tensorflow or onnx2tf not installed"):
        exporter.to_tflite("x.tflite")


def test_to_tflite_writes_file_and_applies_quantization(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    input_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    """Export TFLite with mocked onnx2tf and tensorflow modules."""
    convert_calls: list[dict] = []

    class FakeConverter:
        def __init__(self) -> None:
            self.optimizations = []

        def convert(self) -> bytes:
            return b"tflite-bytes"

    converter = FakeConverter()

    fake_tf = types.SimpleNamespace(
        lite=types.SimpleNamespace(
            Optimize=types.SimpleNamespace(DEFAULT="DEFAULT"),
            TFLiteConverter=types.SimpleNamespace(
                from_saved_model=lambda _p: converter
            ),
        )
    )
    fake_onnx2tf = types.SimpleNamespace(
        convert=lambda **kwargs: convert_calls.append(kwargs)
    )

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "onnx2tf", fake_onnx2tf)

    exporter = Exporter(model, input_shape=input_shape, device="cpu")
    monkeypatch.setattr(
        Exporter,
        "to_onnx",
        lambda self, p, opset=17: Path(p).write_bytes(b"onnx") or Path(p),
    )

    out = exporter.to_tflite(tmp_path / "model.tflite", quantize=True)
    assert out.exists()
    assert out.read_bytes() == b"tflite-bytes"
    assert converter.optimizations == ["DEFAULT"]
    assert len(convert_calls) == 1


def test_to_tflite_uses_explicit_onnx_path(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    input_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    """Use user-provided ONNX path in TFLite conversion flow."""
    fake_tf = types.SimpleNamespace(
        lite=types.SimpleNamespace(
            Optimize=types.SimpleNamespace(DEFAULT="DEFAULT"),
            TFLiteConverter=types.SimpleNamespace(
                from_saved_model=lambda _p: types.SimpleNamespace(convert=lambda: b"x")
            ),
        )
    )
    fake_onnx2tf = types.SimpleNamespace(convert=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "onnx2tf", fake_onnx2tf)

    exporter = Exporter(model, input_shape=input_shape, device="cpu")
    called_paths: list[Path] = []

    def fake_to_onnx(self, path, opset=17):
        p = Path(path)
        called_paths.append(p)
        p.write_bytes(b"onnx")
        return p

    monkeypatch.setattr(Exporter, "to_onnx", fake_to_onnx)

    explicit = tmp_path / "explicit.onnx"
    out = exporter.to_tflite(tmp_path / "m.tflite", onnx_path=explicit)
    assert called_paths[0] == explicit
    assert out.exists()


def test_to_onnx_propagates_export_errors(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    input_shape: tuple[int, ...],
) -> None:
    """Propagate underlying export failures from torch.onnx.export."""
    onnx_stub = types.SimpleNamespace(
        load=lambda p: object(),
        checker=types.SimpleNamespace(check_model=lambda m: None),
    )
    monkeypatch.setitem(sys.modules, "onnx", onnx_stub)
    monkeypatch.setattr(torch.onnx, "export", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad path")))

    exporter = Exporter(model, input_shape=input_shape, device="cpu")
    with pytest.raises(OSError, match="bad path"):
        exporter.to_onnx("broken.onnx")
