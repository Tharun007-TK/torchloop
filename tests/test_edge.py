from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from torchloop.edge import deploy_to_edge, estimate_model
from torchloop.edge import deploy as deploy_mod


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def test_estimate_model_returns_all_keys():
    """Return all expected estimate keys for a valid model."""
    model = SimpleNet()
    stats = estimate_model(model, (1, 3, 32, 32), target_device="desktop")
    expected = {
        "params",
        "param_size_mb",
        "estimated_ram_mb",
        "estimated_latency_ms",
        "flops",
    }
    assert expected.issubset(stats.keys())


def test_estimate_model_different_devices_change_latency():
    """Use target device profile to produce different latency outputs."""
    model = SimpleNet()
    esp32 = estimate_model(model, (1, 3, 32, 32), target_device="esp32")
    desktop = estimate_model(model, (1, 3, 32, 32), target_device="desktop")
    assert esp32["estimated_latency_ms"] > desktop["estimated_latency_ms"]


def test_estimate_model_larger_input_increases_flops_and_latency():
    """Increase input shape and verify flops and latency rise."""
    model = SimpleNet()
    small = estimate_model(model, (1, 3, 32, 32), target_device="desktop")
    large = estimate_model(model, (1, 3, 224, 224), target_device="desktop")
    assert large["flops"] > small["flops"]
    assert large["estimated_latency_ms"] > small["estimated_latency_ms"]


def test_estimate_model_rejects_invalid_target():
    """Raise ValueError for unsupported estimate target."""
    with pytest.raises(ValueError, match="target_device"):
        estimate_model(
            SimpleNet(),
            (1, 3, 32, 32),
            target_device=cast(Any, "mobile"),
        )


def test_estimate_model_rejects_invalid_input_shape():
    """Raise ValueError when input shape is too small."""
    with pytest.raises(ValueError, match="input_shape"):
        estimate_model(SimpleNet(), (1,), target_device="desktop")


def test_deploy_to_edge_onnx_branch_with_mocked_export(monkeypatch, tmp_path: Path):
    """Run jetson deployment branch and verify ONNX artifact output."""
    model = SimpleNet()

    def fake_export(_model, _shape, output_path, dynamic_axes=None):
        Path(output_path).write_bytes(b"onnx")

    monkeypatch.setattr("torchloop.edge.deploy._export_to_onnx", fake_export)
    out = tmp_path / "jetson.onnx"
    result = deploy_to_edge(model, "jetson", (1, 3, 32, 32), str(out))

    assert out.exists()
    assert result["format"] == "onnx"
    assert result["target"] == "jetson"


def test_deploy_to_edge_rpi_tflite_branch(monkeypatch, tmp_path: Path):
    """Run rpi branch that converts ONNX to TFLite."""
    model = SimpleNet()

    def fake_export(_model, _shape, output_path, dynamic_axes=None):
        Path(output_path).write_bytes(b"onnx")

    def fake_tflite(_onnx, output_path, quantize, quantize_type):
        Path(output_path).write_bytes(b"tflite")

    monkeypatch.setattr("torchloop.edge.deploy._export_to_onnx", fake_export)
    monkeypatch.setattr("torchloop.edge.deploy._convert_to_tflite", fake_tflite)

    out = tmp_path / "rpi.tflite"
    result = deploy_to_edge(model, "rpi", (1, 3, 32, 32), str(out), quantize=False)
    assert out.exists()
    assert result["format"] == "tflite"
    assert result["quantized"] is False


def test_deploy_to_edge_ios_branch(monkeypatch, tmp_path: Path):
    """Run ios branch and verify CoreML artifact selection."""
    model = SimpleNet()

    def fake_export(_model, _shape, output_path, dynamic_axes=None):
        Path(output_path).write_bytes(b"onnx")

    def fake_coreml(_model, _shape, output_path):
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "model.mlmodel").write_bytes(b"ml")

    monkeypatch.setattr("torchloop.edge.deploy._export_to_onnx", fake_export)
    monkeypatch.setattr("torchloop.edge.deploy._convert_to_coreml", fake_coreml)

    out = tmp_path / "ios.mlpackage"
    result = deploy_to_edge(model, "ios", (1, 3, 32, 32), str(out))
    assert out.exists()
    assert result["format"] == "coreml"


def test_deploy_to_edge_rejects_invalid_target(tmp_path: Path):
    """Raise ValueError for unsupported deployment target."""
    with pytest.raises(ValueError, match="target must be one of"):
        deploy_to_edge(
            SimpleNet(),
            cast(Any, "invalid"),
            (1, 3, 32, 32),
            str(tmp_path / "x.onnx"),
        )


def test_deploy_to_edge_warns_for_esp32_float16(monkeypatch, tmp_path: Path):
    """Warn when ESP32 is used with non-int8 quantization type."""
    model = SimpleNet()

    def fake_export(_model, _shape, output_path, dynamic_axes=None):
        Path(output_path).write_bytes(b"onnx")

    def fake_tflite(_onnx, output_path, quantize, quantize_type):
        Path(output_path).write_bytes(b"tflite")

    monkeypatch.setattr("torchloop.edge.deploy._export_to_onnx", fake_export)
    monkeypatch.setattr("torchloop.edge.deploy._convert_to_tflite", fake_tflite)

    with pytest.warns(UserWarning, match="ESP32"):
        deploy_to_edge(
            model,
            "esp32",
            (1, 3, 32, 32),
            str(tmp_path / "esp32.tflite"),
            quantize=True,
            quantize_type="float16",
        )


def test_deploy_to_edge_rejects_invalid_quantize_type(tmp_path: Path):
    """Raise ValueError for unsupported quantize type values."""
    with pytest.raises(ValueError, match="quantize_type"):
        deploy_to_edge(
            SimpleNet(),
            "jetson",
            (1, 3, 32, 32),
            str(tmp_path / "q.onnx"),
            quantize_type=cast(Any, "int4"),
        )

def test_deploy_to_edge_rejects_invalid_input_shape(tmp_path: Path):
    """Raise ValueError for invalid deploy input shape."""
    with pytest.raises(ValueError, match="input_shape"):
        deploy_to_edge(SimpleNet(), "jetson", (1,), str(tmp_path / "bad.onnx"))


def test_export_to_onnx_raises_when_onnx_missing(monkeypatch, tmp_path: Path):
    """Raise ImportError when onnx dependency is unavailable."""
    monkeypatch.setattr(deploy_mod, "import_module", lambda _name: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(ImportError, match="onnx is required"):
        deploy_mod._export_to_onnx(SimpleNet(), (1, 3, 32, 32), tmp_path / "m.onnx")


def test_export_to_onnx_accepts_dynamic_axes(monkeypatch, tmp_path: Path):
    """Pass dynamic axes through ONNX export and verify checker call."""
    captured: dict[str, object] = {}

    onnx_stub = type(
        "OnnxStub",
        (),
        {
            "load": staticmethod(lambda _p: object()),
            "checker": type("Checker", (), {"check_model": staticmethod(lambda _m: None)}),
        },
    )

    def fake_import(name: str):
        if name == "onnx":
            return onnx_stub
        raise ImportError("missing")

    def fake_export(model, args, path, **kwargs):
        captured["dynamic_axes"] = kwargs.get("dynamic_axes")
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(deploy_mod, "import_module", fake_import)
    monkeypatch.setattr(torch.onnx, "export", fake_export)

    deploy_mod._export_to_onnx(
        SimpleNet(),
        (1, 3, 32, 32),
        tmp_path / "dynamic.onnx",
        dynamic_axes={0: "batch"},
    )
    assert captured["dynamic_axes"] == {"input": {0: "batch"}, "output": {0: "batch"}}


def test_convert_to_tflite_raises_when_dependency_missing(monkeypatch, tmp_path: Path):
    """Raise ImportError when tensorflow or onnx2tf is unavailable."""
    monkeypatch.setattr(deploy_mod, "import_module", lambda _name: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(ImportError, match="tensorflow and onnx2tf"):
        deploy_mod._convert_to_tflite(tmp_path / "in.onnx", tmp_path / "out.tflite", True, "int8")


def test_convert_to_tflite_float16_sets_supported_types(monkeypatch, tmp_path: Path):
    """Set float16 target type when quantize_type is float16."""
    onnx_in = tmp_path / "in.onnx"
    onnx_in.write_bytes(b"onnx")

    class FakeConverter:
        def __init__(self) -> None:
            self.optimizations = []
            self.target_spec = type("Spec", (), {"supported_types": []})()

        def convert(self) -> bytes:
            return b"tflite"

    converter = FakeConverter()
    fake_tf = type(
        "TF",
        (),
        {
            "float16": "float16",
            "lite": type(
                "Lite",
                (),
                {
                    "Optimize": type("Opt", (), {"DEFAULT": "DEFAULT"}),
                    "TFLiteConverter": type(
                        "Conv",
                        (),
                        {"from_saved_model": staticmethod(lambda _p: converter)},
                    ),
                },
            ),
        },
    )
    fake_onnx2tf = type("Onnx2TF", (), {"convert": staticmethod(lambda **_k: None)})

    def fake_import(name: str):
        if name == "onnx2tf":
            return fake_onnx2tf
        if name == "tensorflow":
            return fake_tf
        raise ImportError("missing")

    monkeypatch.setattr(deploy_mod, "import_module", fake_import)
    out = tmp_path / "out.tflite"
    deploy_mod._convert_to_tflite(onnx_in, out, True, "float16")

    assert out.exists()
    assert converter.optimizations == ["DEFAULT"]
    assert cast(Any, converter.target_spec).supported_types == ["float16"]


def test_convert_to_coreml_raises_when_dependency_missing(monkeypatch, tmp_path: Path):
    """Raise ImportError when coremltools is unavailable."""
    monkeypatch.setattr(deploy_mod, "import_module", lambda _name: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(ImportError, match="coremltools is required"):
        deploy_mod._convert_to_coreml(SimpleNet(), (1, 3, 32, 32), tmp_path / "x.mlmodel")


def test_get_model_device_for_parameterless_module() -> None:
    """Default helper device to CPU when model has no parameters."""

    class NoParam(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    assert str(deploy_mod._get_model_device(NoParam())) == "cpu"
