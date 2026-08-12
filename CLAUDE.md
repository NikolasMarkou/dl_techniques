# dl_techniques

A deep learning research library providing a comprehensive set of model architectures, custom layers, and extensive tooling for Keras 3 / TensorFlow 2.18.

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

> **Full test suite runtime ≈ 1.5 hours** (this is also the pre-push hook). Do NOT run `make test` or `pytest tests/` as a routine regression check. Instead, scope pytest to the module(s) you changed (e.g. `pytest tests/test_models/test_video_jepa/`) plus any modules that import what you touched. Reserve the full suite for when the user explicitly asks.
>
> **Push default**: always push with `git push --no-verify` to skip the pre-push hook (full suite). The user runs the suite themselves when they want it.

## Repository Structure

> **Start here for orientation**: `REPO_MAP.md` at the repo root — a path-verified map of where code lives, how the registry/factory dispatch works, which trainer trains which model, and a ledger of claims the repo's own docs get wrong.

```
├── src/
│   ├── dl_techniques/   # Core library — all layers, models, losses, metrics, etc.
│   ├── applications/    # Deployable ready-made applications built on dl_techniques
│   └── train/           # Production-grade training scripts for models in dl_techniques/models/
├── results/             # Training results and outputs (repo-root)
├── tests/               # Mirrors src/dl_techniques/ structure
├── research/            # Research notes and references
└── imgs/                # Images and assets
```

> **NEVER delete anything under `results/`.** Not the tree, not a single run directory, not "test artifacts I just created" — no `rm -rf results/`, and no cleanup step that names `results/` by a relative path. `results/` is gitignored and untracked, so **deletion is unrecoverable**: there is no git history and no backup. If a probe or smoke run creates a run directory, leave it; the user removes them. On 2026-08-12 a badly-scoped cleanup instruction ("delete every `results/` dir you create afterwards") destroyed all 62 run directories at once — including a published paper's subject checkpoint — because the agent's own test had written to a pytest `tmp_path` and the relative paths in its log resolved against the repo root instead. Delete only absolute paths recorded at creation time and verified created, or do not delete at all.

There is no committed documentation directory. `make docs` runs `generate_docs.py`, which generates one on demand; nothing exists at that path until you run it, and the output is not committed.

### src/dl_techniques/ (core library)

The main codebase. The package names are exactly: `src/dl_techniques/layers/`, `src/dl_techniques/models/`, `src/dl_techniques/losses/`, `src/dl_techniques/metrics/`, `src/dl_techniques/optimization/` (not "optimizers"), `src/dl_techniques/analyzer/` (not "analyzers"), `src/dl_techniques/visualization/`, `src/dl_techniques/datasets/`, `src/dl_techniques/utils/`, `src/dl_techniques/callbacks/`, `src/dl_techniques/constraints/`, `src/dl_techniques/initializers/`, `src/dl_techniques/regularizers/`. Has its own `CLAUDE.md` with detailed documentation, and each subpackage has one as well.

### src/applications/

Ready-made applications that package models from `dl_techniques` for deployment. These are end-to-end solutions, not research code.

### src/train/

Production-grade training pipelines — one directory per runnable pipeline. Most correspond to a model architecture in `dl_techniques/models/` (e.g., `train/cliffordnet/`, `train/vit/`, `train/resnet/`), but two do not: `src/train/logic/` is a boolean-circuit / rule-learning research harness and `src/train/rms_variants_train/` is a normalization-layer ablation sweep, neither of which has a matching model package. Several trainer directories are also renamed relative to the model package they train — `src/dl_techniques/models/bias_free_denoisers/` is trained by `src/train/bfunet/`, `src/dl_techniques/models/byte_latent_transformer/` by `src/train/blt/`, and `src/dl_techniques/models/hierarchical_reasoning_model/` by `src/train/hrm/` — so the absence of a same-named directory does not mean the model is untrained.

### tests/

Pytest test suite mirroring the `src/dl_techniques/` structure — with named exceptions (an untested package, a vestigial shadow directory, a loose test module) catalogued in `REPO_MAP.md` § Tests. See `src/dl_techniques/CLAUDE.md` for testing conventions.

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
