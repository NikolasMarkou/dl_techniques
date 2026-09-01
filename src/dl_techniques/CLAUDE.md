# dl_techniques

The core library: model architectures, custom layers, losses, metrics and tooling for Keras 3 /
TensorFlow 2.18.

Environment, the `make` targets, dependencies, the full-suite runtime and the `git push
--no-verify` default live in the repo-root `CLAUDE.md`. The package layout, the model/trainer/test
triangle and the test-tree exceptions live in `REPO_MAP.md`. The model family taxonomy and the
package catalogue live in `models/README.md`; the authoring rules in `models/CLAUDE.md`. Each
subpackage has its own `CLAUDE.md`.

## Core Conventions

### Keras 3 Patterns

- All custom layers/models are registered with
  `@register_dl_technique(package="dl_techniques.<module.path>")` from
  `dl_techniques.utils.keras_registration` — **not** with a bare
  `@keras.saving.register_keras_serializable()`. See Registration below.
- Layers implement `__init__`, `build`, `call`, `get_config` (and optionally `from_config`).
- Use `keras.ops` for backend-agnostic tensor operations, not raw TensorFlow ops.
- Import style: `import keras`, then qualify at the call site. Never `from keras import ops`.
- Models take `Dict[str, Any]` config dicts for construction parameters.

### Registration

```python
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.attention.multi_head_attention")
class MultiHeadAttention(keras.layers.Layer):
    ...
```

`keras.saving.get_registered_name(MultiHeadAttention)` then resolves to
`dl_techniques.layers.attention.multi_head_attention>MultiHeadAttention`.

**The `package` string is the defining module's dotted path**, with the `models/` family
directories and subfamily containers stripped. Those are a filing decision, not a namespace, and
they have already been reshuffled once — so `models/vision/resnet/model.py` registers under
`dl_techniques.models.resnet.model`, not its full import path.

**Why not the bare decorator.** `@keras.saving.register_keras_serializable()` mints the key
`Custom>ClassName`, which is **independent of the defining module**: two same-named classes
anywhere in the tree claim the identical slot, and whichever imports last silently wins every
deserialization of both.

**Legacy archives still load.** The helper also binds `Custom>ClassName` as an alias to the same
object, which is what an older `.keras` file reads. No name under `src/` opts out, and no bare
`__name__` is claimed by two registered objects — `tests/test_the_legacy_alias_namespace_has_no_collisions.py`
pins both. The remedy for a genuine bare-name collision is a package-prefix rename of the narrower
consumer (`PWFNetDownsample`, not a second `Downsample`); `legacy_alias=False` remains supported as
a last resort and has no users. The `dl_techniques.utils.keras_registration` module docstring
records the mechanism and the measured control.

### Serialization

Every custom component (layer, loss, metric, regularizer, initializer, constraint) supports full
round-trip serialization via `get_config()`; models save/load with `model.save("model.keras")` /
`keras.models.load_model()`.

### Code Style

Python 3.11+ with comprehensive type hints; centralized logging via `dl_techniques.utils.logger`,
no print statements; `__init__.py` files either export a curated public API (with `__all__`) or are
empty — see `layers/CLAUDE.md` for which is which in `layers/`.

**Docstring style: two are in use. Match the package you are editing; never convert a file
wholesale.** Docstrings carry mathematical formulations where relevant.

| Package | Convention |
|---|---|
| `layers/` | **Sphinx/reST** (`:param:` / `:type:` / `:raises:`) in the large majority of modules |
| `layers/attention/` | Sphinx/reST, **mandatory**; `channel_attention.py` is the exemplar |
| `models/` | **No package-wide style — measurably MIXED.** A blanket claim about `models/` is false in either direction |
| `losses/`, `metrics/`, `utils/`, `optimization/`, `analyzer/`, `visualization/` | Google (`Args:`) majority — a majority, not a rule; most of them also contain Sphinx files |

**For a NEW `models/` package follow `models/language/bert/model.py`** — the normative exemplar,
entirely Sphinx/reST, as are `models/language/gpt2/gpt2.py` and `models/language/wave_field/model.py`.

> **Do not convert existing Google-style `models/` files toward Sphinx, or Sphinx toward Google.**
> An earlier agent nearly "fixed" `bert.py` toward Google on the strength of a blanket rule that
> measurement refuted.

### Factory Pattern

Subpackages with `factory.py` support config-driven construction: a `type` string plus a config
dict returns a configured layer. **`layers/CLAUDE.md` § Layer Reuse Policy owns the factory
contract, the per-domain entry points and the registry sizes**, and is not restated here — two
homes for one rule is a hand-maintained lockstep invariant, i.e. a latent defect.

## Testing

### Running

| | |
|---|---|
| Framework | pytest (`conftest.py` adds `src/` to path and silences TF logging) |
| Marker | `@pytest.mark.integration` for integration tests |
| CI | **there is none.** No `.github/` directory exists; nothing runs these tests except you |
| Hooks | `.pre-commit-config.yaml` *declares* a local pytest hook with `always_run: true`, but hook installation is per-clone and untracked — run `ls .git/hooks/` to see yours. On the author's machine only `pre-push` is installed, so the suite fires on push, not commit |

Scope pytest to the modules you changed; the root `CLAUDE.md` explains why, and why the standing push default is `--no-verify`.

### Layout

Mirrors `src/` — `tests/test_models/test_mobilenet/`, `tests/test_layers/test_attention/`. Under
`tests/test_models/` a directory is the norm; under `tests/test_layers/` a loose `test_<name>.py` is,
so a missing directory there means nothing. `REPO_MAP.md` § Tests lists the named exceptions,
including why `tests/test_models/` is deliberately FLAT.

| File kind | Convention |
|---|---|
| Comprehensive suite | `class TestModelName`, pytest fixtures for configs and sample data |
| **Single-claim guard** | sentence-named after the claim: `test_the_attention_mask_is_honoured.py`, `test_tables_survive_stateless_build.py`, `test_the_gates_actually_gate.py` |
| **Meta-test** | prefixed `test_the_guard_…` / `test_the_probe_…` / `test_the_contract_…`. A **shared instrument** carries **no `test_` prefix**, so pytest does not collect it; each has a mirrored `test_<name>.py` RED proof |

Tests cover initialization, forward pass, gradient flow, serialization round-trip, training mode
and edge cases. Numerical tolerance `atol=1e-6` to `1e-7`, **with `rtol=0`** for a pure absolute
bound — `assert_allclose`'s default `rtol=1e-7` otherwise contributes silently to a nominally-`atol` failure.

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
> session and swings precision measurements by ~1000x depending on collection order. And since
> `results/` is gitignored and untracked, deletion there is unrecoverable — route every config
> through `tmp_path`.

### Gate discipline

- **A known-open defect is pinned with `@pytest.mark.xfail(strict=True, reason="<measured>: ...")`**
  so it XPASSes loudly when someone fixes it. A plain `skip` is inert; a deleted test leaves the gap
  unguarded.
- **Never gate on `--collect-only`.** An all-skip module reads as a pass, and a suite whose
  collection errored can "pass" by running almost nothing — this once hid 12 real failures across 8
  steps. Quote the passed count WITH the collected count.
- **Do** run the tree-wide collection gate after any change to a package's public surface:
  `pytest tests/test_models/ -q --collect-only`. A curated `__init__.py` shadowing its own
  subpackage breaks only at collection, invisible to per-package runs.
- **Never run pytest jobs in parallel.** Contention causes false FAILURES — same suite: 21 failed /
  77 passed contended, 89 passed alone. The tell is `cudaSetDevice() ... out of memory` at import.

## Adding New Components

> **Canonical guide**: `research/2026_keras_custom_models_instructions_v2.md`. Read it first — the
> checklists below are a summary, not a substitute.

### New Layer

1. Create the file in the appropriate `layers/` subdomain, inheriting `keras.layers.Layer` and
   decorated with `@register_dl_technique("dl_techniques.layers.<subdomain>.<module>")`.
2. Implement `__init__`, `build`, `call`, `get_config`.
3. If the subdomain has a `factory.py`, register the new layer type there.
4. Add tests in `tests/test_layers/`.

### New Model

1. Create a subdirectory under the appropriate **family** in `models/` (e.g.
   `models/vision/<name>/`) with `__init__.py` and model module(s) — never directly under
   `models/`, which is a family layer only. Add the package to `models/README.md` and to the
   family's `__init__.py` docstring.
2. Inherit from `keras.Model`, decorated with
   `@register_dl_technique("dl_techniques.models.<name>.<module>")` — the family directory is
   **stripped** from the package string.
3. Support variant configs (tiny/small/base/large) via factory methods where appropriate, and add
   tests in `tests/test_models/`.

### New Loss / Metric / Regularizer / Initializer / Constraint

Create the file in the appropriate package, inherit from the corresponding Keras base class,
implement the required methods plus `get_config()`, export from `__init__.py` if the package has a
public API, and add tests in the corresponding `tests/` subdirectory.
