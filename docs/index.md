<div class="hero-section" markdown>

# Stop rewriting PyTorch boilerplate.

torchloop handles the training loop, evaluation, logging, and export — so you can focus on the model.

[Get Started](getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Tharun007-TK/torchloop){ .md-button }

</div>

[![CI](https://github.com/Tharun007-TK/torchloop/actions/workflows/ci.yml/badge.svg)](https://github.com/Tharun007-TK/torchloop/actions)
[![PyPI](https://img.shields.io/pypi/v/torchloop)](https://pypi.org/project/torchloop/)
[![Python](https://img.shields.io/pypi/pyversions/torchloop)](https://pypi.org/project/torchloop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Tharun007-TK/torchloop/blob/main/LICENSE)

---

## The Problem

You write the same PyTorch training loop in every project. Same checkpoint logic. Same metric assembly. Same export steps. It's tedious and inconsistent.

`torchloop` abstracts exactly that — nothing more.

---

## What's Inside

| Module | What it does |
|--------|--------------|
| **Trainer** | Training loop, early stopping, AMP, LR scheduler |
| **Evaluator** | Classification report, confusion matrix, per-class F1 |
| **Callbacks** | W&B and MLflow logging hooks |
| **Exporter** | PyTorch → ONNX → TFLite |
| **Edge** | Deployment utilities and FLOPs estimation |

[Learn more about Export & Edge Deployment →](export-and-edge.md){ .md-button }

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

## Install

=== "Basic"

    ```bash
    pip install torchloop
    ```

=== "With Logging"

    ```bash
    pip install torchloop[logging]
    ```

=== "With Export"

    ```bash
    pip install torchloop[export]
    ```

=== "With Edge"

    ```bash
    pip install torchloop[edge]
    ```

=== "Everything"

    ```bash
    pip install torchloop[all]
    ```

---

## Design Principles

!!! tip "Philosophy"
    **No lock-in**: Works with any `nn.Module`. No subclassing required.
    
    **Minimal surface area**: Three core modules. That's it.
    
    **You own the model**: torchloop wraps your loop, doesn't replace your architecture.

---

## Roadmap

- [x] `v0.1.0` — Trainer, Evaluator, Exporter
- [x] `v0.2.0` — LR scheduler support, mixed precision (AMP)
- [x] `v0.2.1` — Gradient accumulation + callbacks
- [x] `v0.2.2` — Edge submodule
- [ ] `v0.3.0` — W&B / MLflow hooks + CoreML export
- [ ] `v0.3.1` — Model pruning utilities