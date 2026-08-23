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

- Python 3.11+ with comprehensive type hints.
- Centralized logging via `dl_techniques.utils.logger` — no print statements.
- `__init__.py` files either export a curated public API (with `__all__`) or are empty (import from
  submodules directly) — see `layers/CLAUDE.md` for which is which in `layers/`.

**Docstring style: two are in use. Match the package you are editing; never convert a file
wholesale.** Docstrings carry mathematical formulations where relevant.

| Package | Convention | Measured 2026-08-19 |
|---|---|---|
| `layers/` | **Sphinx/reST** (`:param:` / `:type:` / `:raises:`) | 256 of 296 modules |
| `layers/attention/` | Sphinx/reST, **mandatory** | 34 of 35 (the exception is the package `__init__.py`); `channel_attention.py` is the exemplar |
| `models/` | **NO package-wide style — measurably MIXED** | 98 Google-only, 68 Sphinx-only, 9 both, 92 neither, over 267 `.py` files |
| `losses/` | Google (`Args:`) majority | 32 of 42 carry `Args:`, 7 carry `:param ` |
| `metrics/` | Google majority | 12 of 15 / 2 |
| `utils/` | Google majority | 22 of 39 / 9, 3 both |
| `optimization/` | Google majority | 9 of 14 / 2 |
| `analyzer/` | Google majority | 10 of 24 / 0 |
| `visualization/` | Google majority | 6 of 7 / 0 |

Majority, not rule: every Google-majority package except `analyzer/` and `visualization/` also
contains Sphinx files. The `models/` "neither" bucket is mostly `__init__.py` and modules with no
parameter docs at all.

**For a NEW `models/` package follow `models/bert/bert.py`** — the normative exemplar, and entirely
Sphinx/reST (0 `Args:` / 78 `:param `). That is why the recently added `models/gpt2/gpt2.py`
(0 / 26) and `models/wave_field/model.py` (31 `:param `, plus 1 stray `Args:` at :232) are Sphinx.

> **Do not convert existing Google-style `models/` files toward Sphinx, or Sphinx toward Google.**
> `plans/LESSONS.md:294` records an earlier agent nearly "fixing" `bert.py` toward Google on the
> strength of a blanket rule that measurement refuted. A blanket claim about `models/` is false in
> either direction.

Re-derive (`--include=*.py` is **load-bearing** — without it the greps match this file's own prose
and the numbers invalidate themselves):

```bash
find src/dl_techniques/models -name '*.py' | wc -l                                   # 267
grep -rlE "^[[:space:]]*Args:[[:space:]]*$" src/dl_techniques/models --include=*.py | wc -l   # 107 = Google-only + both
grep -rl ":param " src/dl_techniques/models --include=*.py | wc -l                   #  77 = Sphinx-only + both
# comm -12 of those two sorted file lists -> 9 (both);  neither = 267 - 98 - 68 - 9 = 92
```

**This table rots.** It said "248 of 285" for `layers/` until 2026-08-14 (correct on 2026-08-11,
falsified by the `layers/fastvit/` package landing), and on 2026-08-19 six more figures had drifted:
`layers/` 255/294 -> 256/296, `models/` Google-only 102 -> 99, Sphinx-only 67 -> 68, both 5 -> 8,
neither 93 -> 92, and `bert.py` 81 -> 78 `:param `. Re-derived again on 2026-08-21 (step
19.1): Google-only 99 -> 98, both 8 -> 9, the `:param ` total 76 -> 77 — a drift that predates that
step's own two new module docstrings, both of which moved neither count. **Re-run the block above before quoting any of
it** — a derived number is a perishable good.

### Factory Pattern

Subpackages with `factory.py` support config-driven construction. A factory takes a `type` string
plus a config dict and returns a configured layer.

**The per-domain registry sizes and entry points live in exactly one place:**
`layers/CLAUDE.md` § Layer Reuse Policy. They are not restated here — two homes for one number is a
hand-maintained lockstep invariant, i.e. a latent defect. Re-derive with `len(<X>_REGISTRY)`, or
`typing.get_args` for the norms `Literal`. `heads/`, `memory/` and `utils/masking/` expose
constructor sets rather than a dict registry.

**Three rules, all load-bearing:**

1. **Every factory RAISES `ValueError` on a keyword the target type does not declare.** Verified by
   execution 2026-08-19 across `create_attention_layer`, `create_ffn_layer`,
   `create_normalization_layer`, `create_activation_layer` and `create_embedding_layer` — each
   rejects a `bogus_key=1`. **A new factory, or any new registry-backed dispatch, raises.** Never
   reintroduce filter-and-drop; that design previously caused:

   | Silent defect | Effect |
   |---|---|
   | `dropout=` where the registry declares `dropout_rate` | positional dropout dead repo-wide |
   | `qkv_bias=` where the spelling is `use_bias` | attention built with zero bias weights |
   | RoPE args passed to an attention type declaring none | keys evaporated |

2. **A registry's key set, `Literal` aliases and each entry's `required_params` /
   `optional_params` are public API** — consumed by config-driven callers and asserted by
   `tests/test_layers/test_factory_registry_drift.py`. Adding, renaming or removing one is a
   breaking change, not a cleanup.

3. **`create_normalization_layer` sets `epsilon=1e-6`; Keras defaults to `1e-3`** for both
   `LayerNormalization` and `BatchNormalization` (both confirmed by execution 2026-08-19) — 1000x in
   every denominator, with no shape symptom and no warning. Route normalization through the factory;
   constructing directly makes `epsilon=` mandatory with a cited reference. Do **not** blanket-fix:
   some references genuinely use `1e-3` (MobileNet's BN among them).

### Serialization
- All custom components (layers, losses, metrics, regularizers, initializers, constraints) must support full round-trip serialization via `get_config()`
- Models can be saved/loaded with `model.save("model.keras")` / `keras.models.load_model()`

## Testing

### Running

| | |
|---|---|
| Framework | pytest (`conftest.py` adds `src/` to path and silences TF logging) |
| Run | `make test` or `python -m pytest tests/ -vvv` |
| Marker | `@pytest.mark.integration` for integration tests |
| CI | **there is none.** No `.github/` directory exists; nothing runs these tests except you |
| Hooks | `.pre-commit-config.yaml` *declares* a local pytest hook with `always_run: true`, but hook installation is per-clone and untracked — run `ls .git/hooks/` to see yours. On the author's machine only `pre-push` is installed, so the suite fires on push, not commit |

**The full suite takes about 1.5 hours.** Hence the standing `git push --no-verify` default, and
hence scoping pytest to the modules you changed rather than running `make test` as a routine check.

### Layout

Mirrors `src/` — `tests/test_models/test_mobilenet/`, `tests/test_layers/test_attention/`. Under
`tests/test_models/` a directory is the norm; under `tests/test_layers/` a loose `test_<name>.py` is
(84 loose modules against 20 subdirectories, 2026-08-23), so a missing directory there means nothing.
`REPO_MAP.md` § Tests lists the named exceptions.

| File kind | Convention |
|---|---|
| Comprehensive suite | `class TestModelName`, pytest fixtures for configs and sample data |
| **Single-claim guard** | sentence-named after the claim: `test_the_attention_mask_is_honoured.py`, `test_tables_survive_stateless_build.py`, `test_the_gates_actually_gate.py` (15 such files, 2026-08-19) |
| **Meta-test** | prefixed `test_the_guard_…` / `test_the_probe_…` / `test_the_contract_…` |
| **Shared instrument** | **no `test_` prefix**, so pytest does not collect it; each has a mirrored `test_<name>.py` RED proof |

Tests cover initialization, forward pass, gradient flow, serialization round-trip, training mode and
edge cases. Numerical tolerance `atol=1e-6` to `1e-7`, **with `rtol=0`** for a pure absolute bound —
`assert_allclose`'s default `rtol=1e-7` otherwise contributes silently to a nominally-`atol` failure.

### Shared instruments — reuse these, do not reinvent them

| Module | Provides |
|---|---|
| `tests/test_models/smoke_contract_oracle.py` | `assert_contract_rejects_a_broken_forward()` — mutation-injects the model's own forward output. Requires `AssertionError` **specifically**, never `Exception`: a `TypeError` from a contract that indexed a scalar is the contract *crashing*, not *judging* |
| `tests/test_models/knob_sensitivity_oracle.py` | `assert_structural_knob_changes_weights()` / `assert_value_knob_changes_output()` / `assert_scoped_value_knob_changes_weights()`. Pick by knob class — a **structural** knob (depth/heads/filters) is pinned on the weight-SHAPE signature, because different shapes consume different RNG draws and an output-difference assertion is satisfied by random-init luck alone |
| `tests/test_models/test_sam/dead_component_oracle.py` | `fit_one_step_moved_variables()` (returns a NAME SET, never a count), `outputs_stop_gradient()`, `component_response()`, and the killers `zeroed_variables` / `destroy_negatives` / `destroy_positives` / `layer_returns_its_input` |
| `tests/numerics.py` | `reassociation_atol()`, for bounds DERIVED from a noise source rather than pasted. A tolerance below the dtype's resolution measures nothing — three assertions once shipped at `atol=1e-6` on a float32 path and were RED for their entire lifetime |

### Repo-wide guards

| Guard | Watches |
|---|---|
| `tests/test_layers/test_factory_registry_drift.py` | registry surface |
| `tests/test_serialization_registry.py` | `register_keras_serializable` key collisions, watched ACROSS imports |
| `tests/test_models/test_package_api_contract.py` | `__all__` / factory exports |
| `tests/test_repo_map_numbers.py` | re-derives every number in `REPO_MAP.md` from the live tree. **Editing a tracked `.md` pulls this into your gate set** |

### Fixtures owning process-global state

In `tests/conftest.py` and `tests/test_layers/conftest.py`:

| Fixture | Does |
|---|---|
| `golden_reference_device` | pins golden probes to CPU |
| `dtype_policy` | parametrizes float32 / mixed_float16 / float64, restores in `finally` |
| `tf32_disabled` | opt-in per module via `pytestmark = pytest.mark.usefixtures("tf32_disabled")`; asserts its own restoration |
| `_tf32_leak_canary` | fails the NEXT test after a leak |
| autouse `results/` guard | forbids any test writing into the repo-root `results/` — it **asserts**, never cleans up |

> **Never call `enable_tensor_float_32_execution` at module import.** It is process-global for the
> session and swings precision measurements by ~1000x depending on collection order.

> **`results/` is gitignored and untracked, so deletion there is unrecoverable.** Route every config
> through `tmp_path`.

### Gate discipline

- **A known-open defect is pinned with `@pytest.mark.xfail(strict=True, reason="<measured>: ...")`**
  (20 sites, 2026-08-19), so it XPASSes loudly when someone fixes it. A plain `skip` is inert; a deleted test
  leaves the gap unguarded.
- **Never gate on `--collect-only`.** An all-skip module reads as a pass, and a suite whose
  collection errored can "pass" by running almost nothing — this once hid 12 real failures across 8
  steps. Quote the passed count WITH the collected count.
- **Do** run the tree-wide collection gate after any change to a package's public surface:
  `pytest tests/test_models/ -q --collect-only`. A curated `__init__.py` shadowing its own subpackage
  breaks only at collection, invisible to per-package runs.
- **Never run pytest jobs in parallel.** Contention causes false FAILURES — same suite: 21 failed /
  77 passed contended, 89 passed alone. The tell is `cudaSetDevice() ... out of memory` at import.

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
