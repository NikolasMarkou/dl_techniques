# dl_techniques

A comprehensive deep learning library providing a broad set of model architectures, custom layers, and extensive tooling for Keras 3 / TensorFlow 2.18.

**Author**: Nikolas Markou | **License**: GPL-3.0 | **Python**: >= 3.11

## Quick Reference

```
make test       # python -m pytest tests/ -vvv
make clean      # remove build artifacts and __pycache__
make structure  # display src tree
```

## Dependencies

- **tensorflow** 2.18.0, **keras** >=3.8.0 <4.0
- **numpy** >=1.22, **scipy** >=1.15.1, **scikit-learn** >=1.6.1, **pandas** >=2.2.3
- **matplotlib** >=3.10, **seaborn** >=0.13.2, **tqdm**
- Dev: pytest, pytest-cov, pylint, pre-commit

## Project Layout

```
src/dl_techniques/
├── models/          # 80 leaf model packages in 11 family dirs — catalogue in models/README.md
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
└── utils/           # Shared utilities (tensors, geometry, masking, alignment, inference)
```

Each package has its own `CLAUDE.md` with detailed documentation.

`models/` is the one package that is **two or three levels deep**: since 2026-08-24 its 85 leaf
packages sit under 12 family directories (`common`, `embeddings_experimental`,
`general_purpose`, `graph`, `language`, `memory`, `neural_computer`, `point_cloud`,
`tabular`, `time_series`, `vision`, `vision_language`), four of which nest one level further
(`vision/image_restoration/`, `vision/keypoints/`, `vision/super_resolution/`,
`vision_language/sam/`). **A family is a grouping, not a namespace** — every family
`__init__.py` holds a docstring and nothing else, so import from the leaf package
(`dl_techniques.models.vision.resnet.model`), never from the family. `time_series/` is the sole
exception and does re-export its 7 children. `models/README.md` is the catalogue and
`models/CLAUDE.md` the authoring rules; both were re-derived against the live tree on
2026-08-25.

## Core Conventions

### Keras 3 Patterns
- All custom layers/models are registered with
  `@register_dl_technique(package="dl_techniques.<module.path>")` from
  `dl_techniques.utils.keras_registration` — **not** with a bare
  `@keras.saving.register_keras_serializable()`. See "Registration" below
- Layers implement `__init__`, `build`, `call`, `get_config` (and optionally `from_config`)
- Use `keras.ops` for backend-agnostic tensor operations — not raw TensorFlow ops
- Models use `Dict[str, Any]` config dicts for construction parameters

### Registration

Every registered class or function in `src/` uses the shared helper:

```python
import keras
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.attention.multi_head_attention")
class MultiHeadAttention(keras.layers.Layer):
    ...
```

`keras.saving.get_registered_name(MultiHeadAttention)` then resolves to
`dl_techniques.layers.attention.multi_head_attention>MultiHeadAttention` (verified 2026-08-29).

**The `package` string is the defining module's dotted path**, with two things stripped under
`dl_techniques.models`: its 12 family directories (`common`, `embeddings_experimental`,
`general_purpose`, `graph`, `language`, `memory`, `neural_computer`, `point_cloud`, `tabular`,
`time_series`, `vision`, `vision_language`) and its 4 subfamily containers (`image_restoration`,
`keypoints`, `super_resolution`, `sam`). Those are a filing decision, not a namespace, and they
have already been reshuffled once — so `models/vision/resnet/model.py` registers under
`dl_techniques.models.resnet.model`, not its full import path.

**Why not the bare decorator.** A bare `@keras.saving.register_keras_serializable()` mints the key
`Custom>ClassName`, which is **independent of the defining module**: two same-named classes
anywhere in the tree claim the identical slot and whichever imports last silently wins every
deserialization of both.

**Legacy archives still load.** The helper also binds `Custom>ClassName` as an alias to the same
object, which is what a pre-2026-08-29 `.keras` file reads. Four names carry `legacy_alias=False`
and therefore have no alias. The `dl_techniques.utils.keras_registration` module docstring
records the mechanism, the exceptions, and the measured control.

### Code Style

- Python 3.11+ with comprehensive type hints.
- Centralized logging via `dl_techniques.utils.logger` — no print statements.
- `__init__.py` files either export a curated public API (with `__all__`) or are empty (import from
  submodules directly) — see `layers/CLAUDE.md` for which is which in `layers/`.

**Docstring style: two are in use. Match the package you are editing; never convert a file
wholesale.** Docstrings carry mathematical formulations where relevant.

| Package | Convention | Measured 2026-08-28 (re-run of the block below) |
|---|---|---|
| `layers/` | **Sphinx/reST** (`:param:` / `:type:` / `:raises:`) | 262 of 297 modules carry `:param ` (re-measured 2026-08-31; commit `95ec63218` deleted three top-level modules that all carried `:param `, hence -3 of the numerator and -2 of the denominator; then `layers/grid_sample.py` was merged into `layers/spatial_layer.py` and deleted, and because it carried **0** `:param ` lines only the DENOMINATOR moved, 298 -> 297. Check both terms, never the ratio.) |
| `layers/attention/` | Sphinx/reST, **mandatory** | 34 of 35 (the exception is the package `__init__.py`); `channel_attention.py` is the exemplar |
| `models/` | **NO package-wide style — measurably MIXED** | 80 Google-only, 87 Sphinx-only, 8 both, 112 neither, over 287 `.py` files |
| `losses/` | Google (`Args:`) majority | 32 of 44 carry `Args:`, 10 carry `:param `, 1 both |
| `metrics/` | Google majority | 12 of 15 / 2 |
| `utils/` | Google majority | 22 of 41 / 11, 3 both |
| `optimization/` | Google majority | 9 of 14 / 2 |
| `analyzer/` | Google majority | 10 of 24 / 0 |
| `visualization/` | Google majority | 6 of 7 / 0 |

Majority, not rule: every Google-majority package except `analyzer/` and `visualization/` also
contains Sphinx files. The `models/` "neither" bucket is mostly `__init__.py` and modules with no
parameter docs at all.

**For a NEW `models/` package follow `models/language/bert/model.py`** — the normative exemplar, and entirely
Sphinx/reST (0 `Args:` / **81** `:param `). That is why `models/language/gpt2/gpt2.py`
(0 / **27**) and `models/language/wave_field/model.py` (31 `:param `, plus 1 stray `Args:` at **:273**) are Sphinx.
All three paths gained a family segment in the 2026-08-24 `models/` restructure — they were
`models/bert/`, `models/gpt2/`, `models/wave_field/` before it.

> **Do not convert existing Google-style `models/` files toward Sphinx, or Sphinx toward Google.**
> `plans/LESSONS.md:294` records an earlier agent nearly "fixing" `bert.py` toward Google on the
> strength of a blanket rule that measurement refuted. A blanket claim about `models/` is false in
> either direction.

Re-derive (`--include=*.py` is **load-bearing** — without it the greps match this file's own prose
and the numbers invalidate themselves):

```bash
find src/dl_techniques/models -name '*.py' | wc -l                                   # 287
grep -rlE "^[[:space:]]*Args:[[:space:]]*$" src/dl_techniques/models --include=*.py | wc -l   #  88 = Google-only + both
grep -rl ":param " src/dl_techniques/models --include=*.py | wc -l                   #  95 = Sphinx-only + both
# comm -12 of those two sorted file lists -> 8 (both)
# Google-only = 88 - 8 = 80;  Sphinx-only = 95 - 8 = 87;  neither = 287 - 80 - 87 - 8 = 112
```

> **The `Args:` anchor is load-bearing, and the numbers are not comparable without it.** An
> unanchored `grep -rl "Args:"` over the same tree returns **89**, not 88 — it also matches `Args:`
> mid-line, inside prose and inside code. Re-running the commands exactly as printed above is
> what produced 80/81/8/106; a hand-rolled regex over the same tree on the same day produced
> 81/73/14/102. Do not mix a number from one instrument into a table derived by another.

**This table rots.** It said "248 of 285" for `layers/` until 2026-08-14 (correct on 2026-08-11,
falsified by the `layers/fastvit/` package landing), and on 2026-08-19 six more figures had drifted:
`layers/` 255/294 -> 256/296, `models/` Google-only 102 -> 99, Sphinx-only 67 -> 68, both 5 -> 8,
neither 93 -> 92. Re-derived again on 2026-08-21 (step 19.1): Google-only 99 -> 98, both 8 -> 9, the
`:param ` total 76 -> 77 — a drift that predates that step's own two new module docstrings, both of
which moved neither count. Re-derived on **2026-08-25** by re-running the printed commands after the
`models/` family restructure: `layers/` 256/296 -> 258/298, `models/` 267 -> 270 files, Google-only
98 -> **82**, Sphinx-only 68 -> **75**, both 9 -> **8**, neither 92 -> **105**, `losses/` `:param `
7 -> 8, `utils/` 22 of 39/9 -> 22 of 41/11, and `bert.py` back to **81** `:param ` (the "78" recorded
on 2026-08-19 does not reproduce under the printed command). The restructure was a pure `git mv` and
moved no docstring, so the `models/` swing is the instrument, not the tree — see the anchor note
above. Re-derived again on 2026-08-25 (later the same day) by
`plan-2026-08-25-c71fc3ad/iter-1/step-9` after the `models/language/colbert/` package landed:
`models/` 270 -> **275** files and `:param ` 85 -> **89**. **Two of the figures recorded in the
sentence before this one never reproduced**: re-running the printed commands against the commit
*preceding* that plan (`0222f3044`) returns anchored-`Args:` **88**, not 90, and `:param ` **85**,
not 83 — so Google-only was 80 and Sphinx-only 77 on 2026-08-25 morning, not 82 and 75. The
arithmetic in the block was carried forward from mis-transcribed inputs; the `neither` figure
matching (105) was a coincidence of two compensating errors. That drift is **not** this plan's and
predates it; it is repaired here only because leaving half a measurement block updated would make
the file contradict itself. **Re-run the block above before quoting any of it** — a derived number
is a perishable good, and this table has now been wrong twice in the same week in two different
ways.

Re-derived a third time on **2026-08-25** by `plan-2026-08-25-704a9bcb/iter-1/step-7`, which
re-ran the WHOLE of `REPO_MAP.md`'s Numbers table (all 76 enforceable rows) rather than only the
red ones: `layers/` **258 of 298 -> 259 of 299**. That drift is **not** that plan's either — it was
already red at its base commit and comes from `4ab68e323` adding
`src/dl_techniques/layers/activations/common.py` (3 `:param ` lines); it is repaired here because
`REPO_MAP.md` § Numbers requires a prose digit to move in the SAME edit as the table row that
sources it. **Every other figure in the table above was re-run in the same edit and reproduces
exactly** — `layers/attention/` 34 of 35, `models/` 88 anchored-`Args:` / 89 `:param ` / 8 both over
275 files (so 80 Google-only, 81 Sphinx-only, 106 neither), `losses/` 32 of 43 / 9 / 1 both,
`metrics/` 12 of 15 / 2, `utils/` 22 of 41 / 11 / 3 both, `optimization/` 9 of 14 / 2, `analyzer/`
10 of 24 / 0, `visualization/` 6 of 7 / 0, `bert.py` 0 / 81, `gpt2.py` 0 / 27, `wave_field/model.py`
31 `:param ` with the stray `Args:` still at `:273`. Note that the drift log in
the paragraph before this one records what was measured *on those dates* and is deliberately left
as-is: it is a history of readings, not a set of live claims.

Re-derived a **fourth** time on **2026-08-26** by `plan-2026-08-26-fb07cf4e/iter-1/step-5.1`, again
by re-running the WHOLE of `REPO_MAP.md`'s Numbers table (all 76 enforceable rows, now executed by
`tests/test_repo_map_numbers.py` rather than by hand) plus every command printed in this section:
`layers/` **260 of 300 -> 259 of 299**. The mover is `plan-2026-08-26-f3744602/iter-1/step-6`
(`0e6b2ea5b`), which deleted `src/dl_techniques/layers/moe/integration.py`; that file carried 14
`:param ` lines, so it belonged to BOTH populations and its deletion decremented the numerator and
the denominator together. **A ratio whose terms move in lockstep looks unchanged; check both of
them, never the ratio.** Every other figure in the table
and the code block above was re-run in the same edit and reproduces EXACTLY: `layers/attention/` 34
of 35, `models/` 275 files with 88 anchored-`Args:` / 89 `:param ` / 8 both (so 80 Google-only, 81
Sphinx-only, 106 neither), `losses/` 32 of 43 / 9 / 1 both, `metrics/` 12 of 15 / 2, `utils/` 22 of
41 / 11 / 3 both, `optimization/` 9 of 14 / 2, `analyzer/` 10 of 24 / 0, `visualization/` 6 of 7 / 0,
`bert.py` 0 / 81, `gpt2.py` 0 / 27, `wave_field/model.py` 31 `:param ` with the stray `Args:` still
at `:273`. The MoE deletion touched `layers/` and nothing else, and the measurement says so.

Re-derived a **fifth** time on **2026-08-28** by `plan-2026-08-28-6de2095b/iter-1/step-6.1`, again
by re-running the WHOLE of `REPO_MAP.md`'s Numbers table (all 76 enforceable rows, executed by
`tests/test_repo_map_numbers.py`) plus every command printed in this section: `layers/`
**259 of 299 -> 264 of 299**. The denominator did not move, and that is the whole story of this
one — five files changed style without any file being added or deleted. **Only three of the five
are that plan's**: `class_token.py`, `mask_token.py` and `register_tokens.py` under
`layers/embedding/`, converted Google -> Sphinx by its first step. The other two were already red at
its base commit `f43881697`, which measured 261, not the tabulated 259:
`layers/norms/polar_weight_norm.py` (`4b8dd2559`) and `layers/thera_heat_field.py` (`7648dbc7c`).
**A red row is the sum of every mover since the last sweep, so a plan cannot read its own
contribution off the size of the failure** — here the failure was +5 and the plan owned +3. Every
other figure in the table and the code block above was re-run in the same edit and reproduces
EXACTLY: `layers/attention/` 34 of 35, `models/` 275 files with 88 anchored-`Args:` / 89 `:param ` /
8 both (so 80 Google-only, 81 Sphinx-only, 106 neither), `losses/` 32 of 43 / 9 / 1 both, `metrics/`
12 of 15 / 2, `utils/` 22 of 41 / 11 / 3 both, `optimization/` 9 of 14 / 2, `analyzer/` 10 of 24 / 0,
`visualization/` 6 of 7 / 0, `bert.py` 0 / 81, `gpt2.py` 0 / 27, `wave_field/model.py` 31 `:param `
with the stray `Args:` still at `:273`, and the unanchored-grep trap still returning 89 against 88.

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
(83 loose modules against 20 subdirectories, re-measured 2026-08-31), so a missing directory there means nothing.
`REPO_MAP.md` § Tests lists the named exceptions.

> **`tests/test_models/` is FLAT and deliberately does not mirror the `models/` family nesting**
> (plan `plan-2026-08-24T205033-8fd4f20d` D-001). There is no `test_vision/` or `test_language/`
> level. The rule that holds is the one that always mattered: **leaf package `x` is tested by
> `tests/test_models/test_<x>/`** — measured 2026-08-25, 80 test directories against 80 leaf
> packages, agreeing on 78 names; the two that differ are `sam1` (covered by `test_sam/`, which
> also owns the shared `dead_component_oracle.py`) and `lewm` (untested), against the grouping
> directories `test_sam/` and `test_time_series/`. Nesting them was evaluated and rejected: 215
> relative imports (`from ..gradient_flow_oracle` and friends) reach shared oracles at
> `tests/test_models/*.py`, and a family level changes what `..` resolves to in every one of
> them, in a suite that cannot be verified in a single process. Do not "finish the job".

| File kind | Convention |
|---|---|
| Comprehensive suite | `class TestModelName`, pytest fixtures for configs and sample data |
| **Single-claim guard** | sentence-named after the claim: `test_the_attention_mask_is_honoured.py`, `test_tables_survive_stateless_build.py`, `test_the_gates_actually_gate.py`. **65** carry the `test_the_` form (`find tests -name 'test_the_*.py' -not -path '*__pycache__*' | wc -l`, 2026-08-25: 47 under `test_models/`, 6 under `test_layers/`, 6 under `test_train/`, 6 loose at `tests/`); the earlier "15 such files, 2026-08-19" quoted no command and does not reproduce |
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
  (**27** sites across 17 files — `grep -rn "xfail(strict=True" tests --include=*.py | wc -l`, 2026-08-25), so it XPASSes loudly when someone fixes it. A plain `skip` is inert; a deleted test
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
2. Inherit from `keras.layers.Layer`, decorate with
   `@register_dl_technique("dl_techniques.layers.<subdomain>.<module>")` (see "Registration")
3. Implement `__init__`, `build`, `call`, `get_config`
4. If the subdomain has a `factory.py`, register the new layer type there
5. Add tests in `tests/test_layers/`

### New Model
1. Create a subdirectory under the appropriate **family** in `models/` (e.g.
   `models/vision/<name>/`, `models/language/<name>/`) with `__init__.py` and model module(s) —
   never directly under `models/`, which is a family layer only. Add the package to
   `models/README.md` and to the family's `__init__.py` docstring
2. Inherit from `keras.Model`, decorate with
   `@register_dl_technique("dl_techniques.models.<name>.<module>")` — the family directory is
   **stripped** from the package string (see "Registration")
3. Support variant configs (tiny/small/base/large) via factory methods where appropriate
4. Add tests in `tests/test_models/`

### New Loss / Metric / Regularizer / Initializer / Constraint
1. Create file in the appropriate package
2. Inherit from the corresponding Keras base class
3. Implement required methods + `get_config()`
4. Export from `__init__.py` if the package has a public API
5. Add tests in the corresponding `tests/` subdirectory
