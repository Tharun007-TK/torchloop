# Changelog

All notable changes to torchloop are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.0] - 2026-03-31

### Added
- Callback system with abstract `Callback` base class
- `WandBLogger` callback for Weights & Biases integration
- `MLflowLogger` callback for MLflow experiment tracking
- `callbacks` optional dependency group: `pip install torchloop[logging]`
- Edge deployment utilities (`torchloop.edge`)
- FLOPs and parameter estimation (`torchloop.edge.estimate`)
- MkDocs documentation site

### Changed
- `Trainer` now accepts `callbacks: list[Callback]` parameter
- `Trainer.fit()` triggers `on_train_begin`, `on_epoch_end`, `on_train_end` hooks

---

## [0.2.0] - 2026-03-28

### Added
- LR scheduler support in `Trainer` — any `torch.optim.lr_scheduler`
- `ReduceLROnPlateau` handled automatically (passes `val_loss`)
- Automatic Mixed Precision (AMP) via `amp=True` flag — CUDA only
- LR logged per epoch in `history["lr"]`

### Changed
- `Trainer.__init__` now accepts `scheduler` and `amp` parameters
- Training log now includes current LR per epoch

---

## [0.1.0] - 2026-03-27

### Added
- `Trainer` — PyTorch training loop with early stopping and checkpointing
- `Evaluator` — classification report, confusion matrix, per-class F1
- `Exporter` — PyTorch → ONNX → TFLite export pipeline
- CI via GitHub Actions across Python 3.9, 3.10, 3.11
- PyPI trusted publishing via OIDC
