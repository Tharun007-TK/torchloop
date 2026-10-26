# Contributing to torchloop

Thank you for your interest in contributing to **torchloop**! 🎉  
We welcome contributions of all kinds — bug fixes, new features, documentation improvements, and more.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Submitting a Pull Request](#submitting-a-pull-request)
- [Good First Issues](#good-first-issues)
- [Code Style](#code-style)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Commit Message Guidelines](#commit-message-guidelines)

---

## Code of Conduct

By participating in this project, you agree to be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/torchloop.git
   cd torchloop
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Tharun007-TK/torchloop.git
   ```

---

## Development Setup

We recommend using a virtual environment. The project uses [hatch](https://hatch.pypa.io/) for packaging.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

To install all optional extras:

```bash
pip install -e ".[dev,export,logging]"
```

---

## How to Contribute

### Reporting Bugs

- Search [existing issues](https://github.com/Tharun007-TK/torchloop/issues) first to avoid duplicates.
- Open a new issue using the **Bug Report** template.
- Include: Python version, torch version, OS, a minimal reproducible example, and the full traceback.

### Suggesting Features

- Check the [Roadmap](README.md#roadmap) and open issues to see if it's already planned.
- Open an issue using the **Feature Request** template with a clear description and motivation.

### Submitting a Pull Request

1. Create a new branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes, add tests, and update docs if needed.
3. Run linting and tests locally (see below).
4. Commit your changes using [conventional commits](#commit-message-guidelines).
5. Push and open a pull request against `main`.
6. Fill in the PR template — link the related issue.

> **Important:** All PRs must pass CI (lint + tests) before they can be merged.

---

## Good First Issues

New to open source or torchloop? Look for issues tagged [`good first issue`](https://github.com/Tharun007-TK/torchloop/labels/good%20first%20issue).

These are intentionally scoped, well-documented tasks that are a great entry point. Here are examples of the kind of tasks that carry this label:

- Adding docstrings to undocumented functions
- Writing a missing unit test
- Fixing a typo in docs or README
- Adding a simple usage example to the `examples/` directory
- Improving an error message to be more descriptive

If you'd like to work on one, **comment on the issue** to let others know and get guidance from maintainers.

---

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting (replaces black + isort + flake8).

```bash
# Check linting
ruff check .

# Auto-fix lint issues
ruff check --fix .

# Format code
ruff format .
```

Type hints are encouraged. We use [mypy](https://mypy.readthedocs.io/) for static type checking:

```bash
mypy src/torchloop
```

---

## Running Tests

We use [pytest](https://pytest.org/) with coverage reporting:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_trainer.py
```

Coverage is reported automatically via `--cov` (configured in `pyproject.toml`). Aim to keep coverage high and always add tests for new features.

---

## Project Structure

```
torchloop/
├── src/
│   └── torchloop/       # Main library source
├── tests/               # Pytest test suite
├── examples/            # Usage examples (Jupyter/Python scripts)
├── docs/                # MkDocs documentation source
├── .github/
│   └── workflows/       # CI/CD GitHub Actions
├── pyproject.toml       # Build & tool configuration
└── CONTRIBUTING.md      # This file
```

---

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) spec:

```
<type>(<scope>): <short summary>
```

| Type       | When to use                                  |
|------------|----------------------------------------------|
| `feat`     | A new feature                                |
| `fix`      | A bug fix                                    |
| `docs`     | Documentation changes only                   |
| `test`     | Adding or updating tests                     |
| `refactor` | Code change that is neither fix nor feature  |
| `chore`    | Tooling, CI, dependency updates              |
| `style`    | Formatting, no logic change                  |

**Examples:**

```
feat(trainer): add support for cosine annealing scheduler
fix(evaluator): handle empty batch edge case in confusion matrix
docs: add contributing guide
test(exporter): add onnx export roundtrip test
```

---

## Need Help?

Open a [Discussion](https://github.com/Tharun007-TK/torchloop/discussions) or comment on any issue — we're happy to help! 🚀
