# dl_techniques

A deep learning research library providing 150+ model architectures, 290+ custom layers, and extensive tooling for Keras 3 / TensorFlow 2.18.

**Author**: Nikolas Markou | **License**: GPL-3.0 | **Python**: >= 3.11

## Environment

Always use the `.venv` virtual environment for running code, tests, and training scripts.

## Quick Reference

```
make test       # python -m pytest tests/ -vvv
make clean      # remove build artifacts and __pycache__
make structure  # display src tree
make docs       # generate documentation
```

> **Full test suite runtime ≈ 1.5 hours** (this is also the pre-push hook). Do NOT run `make test` or `pytest tests/` as a routine regression check. Instead, scope pytest to the module(s) you changed (e.g. `pytest tests/test_models/test_video_jepa/`) plus any modules that import what you touched. Reserve the full suite for when the user explicitly asks, or immediately before a push they've requested.

## Repository Structure

```
├── src/
│   ├── dl_techniques/   # Core library — all layers, models, losses, metrics, etc.
│   ├── applications/    # Deployable ready-made applications built on dl_techniques
│   ├── train/           # Production-grade training scripts for models in dl_techniques/models/
│   ├── experiments/     # Exploratory / experimental code (not production)
│   └── results/         # Training results and outputs
├── tests/               # Mirrors src/dl_techniques/ structure
├── research/            # Research notes and references
├── docs/                # Generated documentation
└── imgs/, ww-img/       # Images and assets
```

### src/dl_techniques/ (core library)

The main codebase. Contains all custom layers, models, losses, metrics, optimizers, analyzers, visualization, datasets, and utilities. Has its own `CLAUDE.md` with detailed documentation, and each subpackage has one as well.

### src/applications/

Ready-made applications that package models from `dl_techniques` for deployment. These are end-to-end solutions, not research code.

### src/train/

Production-grade training pipelines. Each subdirectory corresponds to a model architecture in `dl_techniques/models/` (e.g., `train/cliffordnet/`, `train/vit/`, `train/resnet/`).

### src/experiments/

Experimental and exploratory code. Not production quality — used for prototyping ideas, testing hypotheses, and quick iterations.

### tests/

Pytest test suite mirroring the `src/dl_techniques/` structure. See `src/dl_techniques/CLAUDE.md` for testing conventions.

## Dependencies

- **tensorflow** 2.18.0, **keras** >=3.8.0 <4.0
- **numpy** >=1.22, **scipy** >=1.15.1, **scikit-learn** >=1.6.1, **pandas** >=2.2.3
- **matplotlib** >=3.10, **seaborn** >=0.13.2, **tqdm**
- Dev: pytest, pytest-cov, pylint, pre-commit

## Running Training Scripts

Always set matplotlib to non-interactive mode to avoid X11 crashes on headless/remote systems:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.<model>.train_<script> [args]
```

## Core Conventions

- Keras 3 patterns: `@keras.saving.register_keras_serializable()`, `keras.ops` for backend-agnostic ops
- Config-driven construction via factory functions
- Full round-trip serialization via `get_config()`
- Python 3.11+ with type hints, Google-style docstrings
- Centralized logging via `dl_techniques.utils.logger` — no print statements

When instructed to create a new model or layer, follow the guide in `research/2026_keras_custom_models_instructions.md`.

See `src/dl_techniques/CLAUDE.md` for detailed conventions, patterns, and how to add new components.
