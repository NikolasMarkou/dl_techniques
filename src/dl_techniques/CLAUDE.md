# dl_techniques

A comprehensive deep learning library providing a broad set of model architectures, custom layers, and extensive tooling for Keras 3 / TensorFlow 2.18.

**Author**: Nikolas Markou | **License**: GPL-3.0 | **Python**: >= 3.11

## Quick Reference

```
make test       # python -m pytest tests/ -vvv
make clean      # remove build artifacts and __pycache__
make structure  # display src tree
make docs       # generate documentation
```

## Dependencies

- **tensorflow** 2.18.0, **keras** >=3.8.0 <4.0
- **numpy** >=1.22, **scipy** >=1.15.1, **scikit-learn** >=1.6.1, **pandas** >=2.2.3
- **matplotlib** >=3.10, **seaborn** >=0.13.2, **tqdm**
- Dev: pytest, pytest-cov, pylint, pre-commit

## Project Layout

```
src/dl_techniques/
├── models/          # Architectures (vision, NLP, VLM, time series, graphs, etc.)
├── layers/          # Custom layers (attention, FFN, norms, embeddings, MoE, transformers, geometric, etc.)
├── losses/          # Loss functions (contrastive, focal, calibration, segmentation, GAN, etc.)
├── metrics/         # Custom metrics (capsule, CLIP, perplexity, PSNR, time series)
├── optimization/    # Config-driven optimizer/LR schedule builders, Muon optimizer, deep supervision
├── analyzer/        # ModelAnalyzer framework (weight, spectral, calibration, training dynamics)
├── visualization/   # Plugin-based visualization (training, classification, regression, data/NN inspection)
├── callbacks/       # Keras callbacks (analyzer integration during training)
├── regularizers/    # Advanced regularizers (binary/ternary preference, entropy, orthogonal, SRIP)
├── initializers/    # Weight initializers (orthonormal, He-orthonormal, hypersphere, Haar wavelet, polar)
├── constraints/     # Weight constraints (value range clipping)
├── datasets/        # Data loading (time series, vision, ARC, tabular, VQA, HuggingFace)
└── utils/           # Shared utilities (tensors, geometry, masking, alignment, export, inference)
```

Each package has its own `CLAUDE.md` with detailed documentation.

## Core Conventions

### Keras 3 Patterns
- All custom layers/models use `@keras.saving.register_keras_serializable()`
- Layers implement `__init__`, `build`, `call`, `get_config` (and optionally `from_config`)
- Use `keras.ops` for backend-agnostic tensor operations — not raw TensorFlow ops
- Models use `Dict[str, Any]` config dicts for construction parameters

### Code Style
- Python 3.11+ with comprehensive type hints
- Docstrings carry mathematical formulations where relevant. **Two styles are in use — match the package you are editing, do not convert files wholesale:**
  - **Sphinx/reST (`:param:` / `:type:` / `:raises:`)** is the convention in `layers/` (255 of 294 modules, re-derived 2026-08-14 with `grep -rl ":param " src/dl_techniques/layers --include=*.py | wc -l` over `find src/dl_techniques/layers -name '*.py' | wc -l`; the previous "248 of 285" was correct when written on 2026-08-11 and went stale when the `layers/fastvit/` package landed), and is *mandatory* in `layers/attention/`, where 34 of the 35 modules use it — the sole exception is the package `__init__.py` — and `channel_attention.py` is the reference exemplar.
  - **`models/` has NO package-wide style — it is measurably MIXED, and a blanket rule about it is false in either direction.** Measured 2026-08-14 over its 267 `.py` files: **102 Google-only, 67 Sphinx-only, 5 both, 93 neither** (the last bucket is mostly `__init__.py` and modules with no parameter docs at all). Re-derive with `find src/dl_techniques/models -name '*.py' | wc -l` → 267; `grep -rlE "^[[:space:]]*Args:[[:space:]]*$" src/dl_techniques/models --include=*.py | wc -l` → 107 (Google-only + both); `grep -rl ":param " src/dl_techniques/models --include=*.py | wc -l` → 72 (Sphinx-only + both); `comm -12` of those two sorted file lists → 5 (both); neither = 267 − 102 − 67 − 5 = 93. **`--include=*.py` is load-bearing** — without it the greps match this very file's prose and the numbers invalidate themselves.
    - **For a NEW `models/` package, follow `models/bert/bert.py` — the normative exemplar, and it is entirely Sphinx/reST.** `grep -cE "^[[:space:]]*Args:[[:space:]]*$" src/dl_techniques/models/bert/bert.py` → 0; `grep -c ":param " src/dl_techniques/models/bert/bert.py` → 81. That is why the recently added `models/gpt2/gpt2.py` (0 `Args:` / 26 `:param `) and `models/wave_field/model.py` (31 `:param `, plus 1 stray `Args:` at :232) are Sphinx.
    - Do **not** convert existing Google-style `models/` files toward Sphinx, or Sphinx files toward Google — match the file you are editing. `plans/LESSONS.md:294` records an earlier agent nearly "fixing" `bert.py` toward Google on the strength of the blanket rule this bullet replaces.
  - **Google-style (`Args:`) is the majority style** in `losses/` (32 of 42 files carry `Args:`, 7 carry `:param `), `metrics/` (12 of 15 / 2), `utils/` (22 of 39 / 9, 3 both), `optimization/` (9 of 14 / 2), `analyzer/` (10 of 24 / 0), and `visualization/` (6 of 7 / 0) — measured 2026-08-14 by running the two greps above with `models` replaced by each package name. Majority, not rule: every one of these packages except `analyzer/` and `visualization/` also contains Sphinx files.
- Centralized logging via `dl_techniques.utils.logger` — no print statements
- `__init__.py` files either export a curated public API (with `__all__`) or are empty (in which case import from submodules directly) — see `layers/CLAUDE.md` for which is which in `layers/`

### Factory Pattern
- Subpackages with `factory.py` support config-driven construction (e.g., `create_attention_layer()`, `create_ffn_layer()`, `create_normalization_layer()`)
- Factory functions accept a `type` string + config dict and return configured layer instances

### Serialization
- All custom components (layers, losses, metrics, regularizers, initializers, constraints) must support full round-trip serialization via `get_config()`
- Models can be saved/loaded with `model.save("model.keras")` / `keras.models.load_model()`

## Testing

- **Framework**: pytest (with `conftest.py` adding `src/` to path and silencing TF logging)
- **Run**: `make test` or `python -m pytest tests/ -vvv`
- **Marker**: `@pytest.mark.integration` for integration tests
- **Hooks**: `.pre-commit-config.yaml` *declares* a local hook running `python -m pytest` with `always_run: true`, but hook installation is per-clone and untracked — run `ls .git/hooks/` to see yours. On the author's machine only `pre-push` is installed, so the suite fires on push, not on commit. **The full suite takes about 1.5 hours**, which is why the standing default is `git push --no-verify` and why you should scope pytest to the modules you changed rather than running `make test` as a routine check
- **CI**: there is none. No `.github/` directory exists; nothing runs these tests except you
- **Test structure**: mirrors `src/` — e.g., `tests/test_models/test_mobilenet/`, `tests/test_layers/test_attention/`. Under `tests/test_models/` the directory is the norm; under `tests/test_layers/` a loose `test_<name>.py` is (78 loose modules against 21 subdirectories), so a missing directory there means nothing. `REPO_MAP.md` § Tests lists the named exceptions
- **Test conventions**:
  - Class-based: `class TestModelName`
  - Pytest fixtures for configs and sample data
  - Tests cover: initialization, forward pass, gradient flow, serialization round-trip, training mode, edge cases
  - Numerical tolerance: `atol=1e-6` to `1e-7` for float comparisons

## Adding New Components

> **Canonical guide for new models and layers**: `research/2026_keras_custom_models_instructions_v2.md`. Read it first — it is the authoritative reference for Keras 3 custom layer/model authoring in this repo. The checklists below are a quick summary, not a substitute.

### New Layer
1. Create file in the appropriate `layers/` subdomain
2. Inherit from `keras.layers.Layer`, decorate with `@keras.saving.register_keras_serializable()`
3. Implement `__init__`, `build`, `call`, `get_config`
4. If the subdomain has a `factory.py`, register the new layer type there
5. Add tests in `tests/test_layers/`

### New Model
1. Create subdirectory under `models/` with `__init__.py` and model module(s)
2. Inherit from `keras.Model`, decorate with `@keras.saving.register_keras_serializable()`
3. Support variant configs (tiny/small/base/large) via factory methods where appropriate
4. Add tests in `tests/test_models/`

### New Loss / Metric / Regularizer / Initializer / Constraint
1. Create file in the appropriate package
2. Inherit from the corresponding Keras base class
3. Implement required methods + `get_config()`
4. Export from `__init__.py` if the package has a public API
5. Add tests in the corresponding `tests/` subdirectory
