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
pip install torchloop

# With TFLite export support
pip install torchloop[export]
```

---

## Usage

### Training

```python
from torchloop import Trainer

trainer = Trainer(
    model,
    optimizer=torch.optim.Adam(model.parameters()),
    criterion=torch.nn.CrossEntropyLoss(),
    device="cuda",
    patience=5,           # early stopping
)

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

---

## Design Principles

- **No lock-in**: Works with any nn.Module. No subclassing required.
- **Minimal surface area**: Three modules. That's it.
- **You own the model**: torchloop wraps your loop, doesn't replace your architecture.

---

## Roadmap

- [ ] `v0.1.0` — Trainer, Evaluator, Exporter
- [ ] `v0.2.0` — LR scheduler support, mixed precision (AMP)
- [ ] `v0.3.0` — W&B / MLflow logging hooks

---

## License

MIT
