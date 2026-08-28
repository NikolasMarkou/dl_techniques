# Models Package

Model architectures, grouped into 11 families under `src/dl_techniques/models/`.
`src/dl_techniques/models/README.md` is the catalogue: every family, every leaf package,
one line each. This file is the authoring contract.

> **ALL model work MUST follow `research/2026_keras_custom_models_instructions_v2.md`.**
> Read it before creating a model directory *or authoring any layer inside one* — it is the
> canonical guide for Keras 3 authoring here (serialization, `build`, `get_config`,
> factories, tests). Non-negotiable, new packages and existing ones alike.

## Layout

Families: `common` `embeddings_experimental` `general_purpose` `graph` `language` `memory`
`neural_computer` `point_cloud` `tabular` `time_series` `vision` `vision_language`.

A **leaf package** is a directory with an `__init__.py` and no `__init__.py`-bearing child.
Four families nest one level further (`vision/image_restoration`, `vision/keypoints`,
`vision/super_resolution`, `vision_language/sam`), so "direct child of `models/`" is the
wrong test. Re-derive any count with the command beside it; never quote one from memory.

```bash
find src/dl_techniques/models -name '__init__.py' -not -path '*__pycache__*' | wc -l   # 100 packages
find src/dl_techniques/models -name '*.py' -not -path '*__pycache__*' | wc -l          # 283 .py
```

## Conventions

- `models/__init__.py` (the root) is **0 bytes** and stays that way. Never import from
  `dl_techniques.models` itself — import from the leaf package.
- **Family and subfamily `__init__.py` carry a docstring and nothing else** — no imports, no
  `__all__`. Re-exporting `vision/` would pull 35 packages behind one statement: a real import
  cost and a real circular-import surface. `plan-2026-08-24T205033-8fd4f20d/D-002`. Do not
  "finish the job" by adding re-exports.
- **`time_series/__init__.py` is the one exception** — 7 children, curated re-exports, and
  consumers that rely on them. Leave it. It is also why `time_series/{deepar,prism,tirex}`
  can carry an empty `__init__.py` without being broken.
- This diverges from `layers/`, where most subpackages are curated. The discriminator is child
  count and import weight. Not an inconsistency to fix.
- Leaf packages are curated: nearly all bind a `create_*` factory and declare `__all__`, and
  every one has a `README.md`. Import the public name straight from the package.
- Each leaf typically holds its model module(s), any architecture-specific blocks, an optional
  `train.py`, and a `README.md`. All models use Keras 3 with full `get_config()`
  serialization, config dicts for construction, and factories for variants.

### Docstring style — measurably mixed

```bash
grep -rlE "^[[:space:]]*Args:[[:space:]]*$" src/dl_techniques/models --include=*.py | wc -l  # 88
grep -rl ":param " src/dl_techniques/models --include=*.py | wc -l                           # 93
```

Over 283 files: Google-only 80, Sphinx-only 85, both 8, neither 110. No package-wide rule.
**Match the file you are editing; never convert one wholesale.** Perishable — re-run the
greps. A different instrument gives a different answer (unanchored `Args:` returns 89).

## Tests are FLAT and do not mirror this layout

`tests/test_models/` is one directory per leaf package, one level deep — `test_beit/`, not
`test_vision/test_beit/`. This is deliberate: 218 relative imports (`from ..oracle`) reach
shared oracle modules at `tests/test_models/*.py`, and nesting would rewrite all of them for
no behavioural gain. `plan-2026-08-24T205033-8fd4f20d/D-001`. Do not "fix" it.

## Layer Reuse Policy (factory-first)

> **Before implementing ANY new layer, check for an existing one to reuse.** A bespoke layer
> is the last resort. Check in order; only move on when nothing fits.

1. **The relevant factory** — pass a `type` string + config; do not hand-roll what a factory
   already builds.

   | Domain | Entry point |
   |---|---|
   | Normalization | `create_normalization_layer()` — `layers/norms/factory.py` |
   | Attention | `create_attention_layer()` — `layers/attention/factory.py` |
   | FFN / MLP | `create_ffn_layer()` — `layers/ffn/factory.py` |
   | Embeddings | `create_embedding_layer()` — `layers/embedding/factory.py` |
   | Activations | `create_activation_layer()` — `layers/activations/factory.py` |
   | Transformer blocks | `TransformerLayer` — `layers/transformers/transformer.py` (direct import) |

   `transformers/` has no `create_*_layer`; `TransformerLayer` is highly configurable and
   composes the factories above, so it covers most cases. Higher-level `create_*_encoder`
   builders live in `vision_encoder.py` / `text_encoder.py`.

2. **The broader `layers/` package** — search it before writing your own.
3. **Only then a new layer** — follow the authoring guide, and put it in the right `layers/`
   subpackage (and its factory registry) rather than burying it in the model directory.

## House Model Module Shape

Reference implementation: `src/dl_techniques/models/vision/resnet/model.py`. Read the spec
rather than copying the file — this shape spread by copy-paste, and that is how its defects
(placeholder weight URLs, mutable default args, a `logger.info` inside `call()`) reached
`convnext/` and `mobilenet/`.

Applies to a package implementing **one architecture with named variants**. A target, not a
universal law — see "When it does not apply".

- **Module skeleton** — substantive module docstring (what it is, the composition rule, a
  `References:` section with real citations), then imports, then constants, then the class,
  then factories. No function definitions between classes.
- **Class API** — `@keras.saving.register_keras_serializable()`; explicit `__init__` args, no
  mutable defaults; validate config early; `build()` creates sublayers; `get_config()` round
  trips every constructor arg; `from_config()` only when it does measured real work.
- **Variant tables** — `MODEL_VARIANTS` is the public-name registry a caller passes;
  `SCALE_CONFIGS` (where present) holds the hyperparameters. Some packages carry both.
- **Pretrained weights** — if none are distributed, `pretrained=True` must raise
  `NotImplementedError`, never silently return random weights.
- **Factory and exports** — a `create_<name>()` factory, bound in `__init__.py` with `__all__`.
- **Hygiene** — `keras.ops` over raw `tf`; central `dl_techniques.utils.logger`, never `print`;
  no logging inside `call()`.

### Things you must NOT do

- **Never delete or reword a `# DECISION <plan-id>/D-NNN` comment.** They resolve through the
  append-only manifest `plans/ANCHORS.md`; removing one destroys the record. If a comment must
  go, record the removal in that manifest first — see its "Retired anchors" section.
- **A module rename must carry every referencing site in the same commit** — documents and
  path-keyed test data included, not just `import` lines.
- **Never convert docstring style.** See above.
- **Never re-export the deep-supervision helpers from a `models/` package.**
  `get_model_output_info` and `create_inference_model_from_training_model` live in
  `dl_techniques/utils/deep_supervision.py`. A **registrar** import (cross-package,
  `# noqa: F401`, bound for its `@keras.saving.register_keras_serializable` side effect) is
  NOT a re-export, is load-bearing, and must not be swept.
- **Never re-export a family.** See Conventions.

### When the shape does not apply

- **No genuine named variants** — do not invent a `MODEL_VARIANTS` table to satisfy the
  template. Apply the skeleton, factory and hygiene rules only, and say why in the README.
- **Functional builders** (`vision/bias_free_denoisers/`, `vision/convunext/`,
  `vision/image_restoration/darkir/`) return `keras.Model(inputs, outputs)` and have no
  subclass. Keep them functional — converting breaks existing checkpoints, and the
  `bias_free_denoisers` ones are used by `src/train/bfunet/` and `src/applications/`.
  Verify with `grep -n "^class .*(.*Model)" <pkg>/*.py` before calling a package functional:
  `vision/detr/` looks functional and is not.
- **Multi-model families** (`vision_language/sam/`, `time_series/`, `vision/dino/`,
  `vision_language/ideogram4/`, `vision_language/sd3_mmdit/`) apply the shape per inner
  architecture, not per directory.

## Verification conventions

- A guard must be proven RED: inject the defect it targets and watch it fail, then restore.
  A guard that has never failed is not known to work.
- Re-derive counts; never transcribe them. Print the command beside the number.
- `--collect-only` is not a test run. Modules with deferred imports collect clean and fail
  when executed.
- Re-run a failure ALONE before calling it real — the suite has ordering-dependent failures.
- Never loosen a pinned population count to silence it. Re-derive it and justify the change.

## Testing

Tests live in `tests/test_models/test_<leaf>/`. Run scoped to what you changed, in separate
processes — the full suite takes ~1.5h and OOMs in a single process. Use
`CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg`.
