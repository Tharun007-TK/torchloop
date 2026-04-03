# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+

## Basic Install

```bash
pip install torchloop
```

## Optional Dependencies

### Logging integrations (W&B + MLflow)
```bash
pip install torchloop[logging]
```

### Export support (ONNX + TFLite)
```bash
pip install torchloop[export]
```

### Edge deployment support
```bash
pip install torchloop[edge]
```

### Development tools
```bash
pip install torchloop[dev]
```

### Everything
```bash
pip install torchloop[all]
```

## Installation Options Explained

| Package | What it includes |
|---------|------------------|
| `torchloop` | Core training, evaluation, and checkpoint management |
| `torchloop[logging]` | W&B and MLflow logging callbacks |
| `torchloop[export]` | ONNX and TFLite export utilities |
| `torchloop[edge]` | Edge deployment tools and resource estimation |
| `torchloop[dev]` | Development dependencies (testing, linting, docs) |
| `torchloop[all]` | All optional features combined |

## Verify Install

```python
import torchloop
print(torchloop.__version__)
```
