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
├── initializers/    # Weight initializers (orthonormal, He-orthonormal, hypersphere, Haar wavelet)
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
- Google-style docstrings with mathematical formulations where relevant
- Centralized logging via `dl_techniques.utils.logger` — no print statements
- `__init__.py` files either export the public API or are empty (import from submodules directly)

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
- **Pre-commit hook**: runs pytest on every commit
- **Test structure**: mirrors `src/` — e.g., `tests/test_models/test_mobilenet_v1.py`, `tests/test_layers/test_attention/`
- **Test conventions**:
  - Class-based: `class TestModelName`
  - Pytest fixtures for configs and sample data
  - Tests cover: initialization, forward pass, gradient flow, serialization round-trip, training mode, edge cases
  - Numerical tolerance: `atol=1e-6` to `1e-7` for float comparisons

## Adding New Components

> **Canonical guide for new models and layers**: `research/2026_keras_custom_models_instructions.md`. Read it first — it is the authoritative reference for Keras 3 custom layer/model authoring in this repo. The checklists below are a quick summary, not a substitute.

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
