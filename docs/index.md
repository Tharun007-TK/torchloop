# torchloop

> Lightweight PyTorch utility library for training, evaluation, and edge deployment — without the framework lock-in.

[![CI](https://github.com/Tharun007-TK/torchloop/actions/workflows/ci.yml/badge.svg)](https://github.com/Tharun007-TK/torchloop/actions)
[![PyPI](https://img.shields.io/pypi/v/torchloop)](https://pypi.org/project/torchloop/)
[![Python](https://img.shields.io/pypi/pyversions/torchloop)](https://pypi.org/project/torchloop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Tharun007-TK/torchloop/blob/main/LICENSE)

---

## The Problem

You write the same PyTorch training loop in every project. Same checkpoint logic. Same metric assembly. Same export steps. It's tedious and inconsistent.

`torchloop` abstracts exactly that — nothing more.

---

## Quick Start

```python
from torchloop import Trainer, Evaluator
import torch.optim.lr_scheduler as sched

scheduler = sched.ReduceLROnPlateau(optimizer, patience=5)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device="cuda",
    scheduler=scheduler,
    amp=True,
    patience=10,
)

history = trainer.fit(train_loader, val_loader, epochs=50)
trainer.save("best.pt")

ev = Evaluator(model, device="cuda")
ev.report(val_loader, class_names=["No Damage", "Minor", "Major", "Destroyed"])
fig = ev.confusion_matrix(val_loader)
fig.savefig("confusion_matrix.png")
```

---

## Features

- **Trainer** — training loop, early stopping, LR scheduler, AMP, checkpointing
- **Evaluator** — classification report, confusion matrix, per-class F1
- **Callbacks** — W&B and MLflow logging hooks
- **Exporter** — PyTorch → ONNX → TFLite pipeline
- **Edge** — deployment utilities and FLOPs estimation

---

## Install

```bash
pip install torchloop

# With logging integrations
pip install torchloop[logging]

# With export support
pip install torchloop[export]
```