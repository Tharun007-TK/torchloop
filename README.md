# torchloop

> Lightweight PyTorch utility library for training, evaluation, and TFLite export — without the framework lock-in.

[![CI](https://github.com/Tharun007-TK/torchloop/actions/workflows/ci.yml/badge.svg)](https://github.com/Tharun007-TK/torchloop/actions)
[![PyPI](https://img.shields.io/pypi/v/torchloop)](https://pypi.org/project/torchloop/)
[![Python](https://img.shields.io/pypi/pyversions/torchloop)](https://pypi.org/project/torchloop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

You write the same PyTorch training loop in every project. Same checkpoint logic. Same metric assembly. Same TFLite export steps. It's tedious and inconsistent.

`torchloop` abstracts exactly that — nothing more.

---

## Install

```bash
# Base installation
pip install torchloop

# With TFLite export support
pip install torchloop[export]

# With edge deployment support
pip install torchloop[edge]

# Development setup
pip install torchloop[dev]
```

---

## Usage

### Training

```python
from torchloop import EarlyStopping, ModelCheckpoint, Trainer

trainer = Trainer(
    model,
    optimizer=torch.optim.Adam(model.parameters()),
    criterion=torch.nn.CrossEntropyLoss(),
    device="cuda",
    use_amp=True,
    accumulate_steps=4,
    patience=5,
)

trainer.add_callback(EarlyStopping(patience=5))
trainer.add_callback(ModelCheckpoint(filepath="best.pt"))

history = trainer.fit(train_loader, val_loader, epochs=30)
trainer.save("best.pt")
```

### Evaluation

```python
from torchloop import Evaluator

ev = Evaluator(model, device="cuda")
results = ev.report(val_loader, class_names=["No Damage", "Minor", "Major", "Destroyed"])
# prints sklearn classification report

fig = ev.confusion_matrix(val_loader)
fig.savefig("cm.png")

per_class = ev.f1_per_class(val_loader)
# {'No Damage': 0.91, 'Minor': 0.78, ...}
```

### Export

```python
from torchloop.exporter import Exporter

exp = Exporter(model, input_shape=(1, 3, 224, 224))
exp.to_onnx("model.onnx")
exp.to_tflite("model.tflite", quantize=True)
```

### Edge Deployment

```python
from torchloop.edge import deploy_to_edge, estimate_model

stats = estimate_model(model, (1, 3, 224, 224), target_device="esp32")
print(f"RAM: {stats['estimated_ram_mb']} MB")
print(f"Latency: {stats['estimated_latency_ms']} ms")

deploy_to_edge(
    model,
    target="esp32",
    input_shape=(1, 3, 224, 224),
    output_path="model.tflite",
    quantize=True,
    quantize_type="int8",
)
```

---

## Design Principles

- **No lock-in**: Works with any nn.Module. No subclassing required.
- **Minimal surface area**: Three modules. That's it.
- **You own the model**: torchloop wraps your loop, doesn't replace your architecture.

---

## Roadmap

- [x] `v0.1.0` — Trainer, Evaluator, Exporter
- [x] `v0.2.0` — LR scheduler support, mixed precision (AMP)
- [x] `v0.2.1` — Gradient accumulation + callbacks
- [x] `v0.2.2` — Edge submodule
- [ ] `v0.3.0` — W&B / MLflow hooks + CoreML export
- [ ] `v0.3.1` — Model pruning utilities

---

## License

MIT
