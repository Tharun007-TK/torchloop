# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.2.3] - 2026-03-27

### Added

- Trainer gradient accumulation via `accumulate_steps`.
- Callback system with `Callback`, `EarlyStopping`, `ModelCheckpoint`, `CSVLogger`, and `StopTraining`.
- Edge deployment submodule with `deploy_to_edge` and `estimate_model`.
- Example script for edge deployment in `examples/edge_deployment.py`.
- Expanded test suite for callbacks, exporter, and edge deployment logic.

### Changed

- AMP configuration now prefers `use_amp`; legacy `amp` is kept as a deprecated alias.
- Public exports updated to include `Exporter`, callbacks, and edge APIs.
- Project metadata updated for v0.2.3 and edge/dev optional dependencies.
- README updated with callback usage and edge deployment documentation.

### Fixed

- Pytest import resolution for src-layout test runs.
- Deterministic early-stopping test behavior in trainer test suite.

## [0.2.0] - 2026-03-xx

### Added

- Automatic mixed precision (AMP) support in Trainer.
- Learning-rate scheduler integration in Trainer.

## [0.1.0] - 2026-03-xx

### Added

- Core `Trainer`, `Evaluator`, and `Exporter` APIs.
