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

### Everything
```bash
pip install torchloop[all]
```

## Verify Install

```python
import torchloop
print(torchloop.__version__)
```
