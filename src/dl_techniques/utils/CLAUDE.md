# Utils Package

Shared utilities used across the library — tensor operations, geometry, masking, alignment, and more.

## Structure

### Top-level Modules
- `tensors.py` — Core tensor ops: `gram_matrix()`, `power_iteration()`, `window_partition()`/`window_reverse()`, Gaussian kernels, orthonormality validation. Also the shared home for two pure dimension-contract predicates that used to be private copies in `layers/`:
  - `is_power_of_two(n)` — `True` only for `n >= 1` with one set bit, so `0` and negatives are rejected. Used by `layers/orthogonal_butterfly.py` and `layers/norms/polar_weight_norm.py` to refuse a last dimension they cannot pair. `polar_weight_norm._next_power_of_two` is a different function and stays where it is
  - `canonical_binary_input_shape(input_shape)` — collapses one-shape-or-list-of-shapes into the single shape a binary elementwise layer builds from, raising on more than two shapes or on a mismatched pair. Used by `layers/logic/{logic,arithmetic}_operators.py` from three sites each (`build`, `compute_output_shape`, `_assert_call_shape_contract`). Import it by its bare name: `tests/test_layers/test_logic/test_the_nine_source_fixes_stay_fixed.py` asserts the call sites resolve as `ast.Name`, so a qualified `tensors.canonical_binary_input_shape(...)` call would go unseen. It is NOT the `is_list_of_shapes` idiom inlined three times under `layers/attention/`, which is deliberately kept separate
- `constants.py` — Shared constants for config keys
- `random.py` — `rayleigh()` distribution generator with statistical validation
- `logger.py` — Centralized library logger
- `bounding_box.py` — IoU (GIoU/DIoU/CIoU), format conversion, NMS
- `corruption.py` — 10 image corruption types (noise, blur, distortion, color) with severity levels, backend-agnostic
- `inference.py` — `FullImageInference`: sliding window patch extraction + aggregation for YOLOv12, with profiling
- `tokenizer.py` — `TiktokenPreprocessor`: BERT-compatible tokenization with special tokens, attention masks
- `deep_supervision.py` — Deep-supervision output helpers (auxiliary heads, weight scheduling glue)
- `drop_path.py` — `linear_drop_path_rates(num_blocks, max_rate)`: computes per-block stochastic-depth (drop-path) rates for transformer/ConvNeXt block stacks. The actual drop-path layer is `StochasticDepth` in `dl_techniques/layers/stochastic_depth.py`
- `weight_transfer.py` — two loaders, deliberately not interchangeable:
  - `load_weights_from_checkpoint(target, ckpt_path, skip_prefixes, strict)`: layer-by-layer *partial* transfer between different architectures (a pretrain trunk into a fine-tune model). Use this instead of `model.load_weights(by_name=True)` (broken in Keras 3.8 for `.keras` files). A zero-layer result **reports rather than raises** — `src/train/beit/` and `src/train/energy_transformer/` build their own warm-start guards on that report
  - `load_weights_or_raise(model, weights_path, skip_mismatch)`: whole-file `.keras` restore into the **same** architecture, returning the number of variables whose value changed and raising when that is zero. `model.load_weights(path, skip_mismatch=True)` restores nothing and returns normally when names or shapes do not match, so never call it bare
- `yolo_decode.py` — YOLOv12 output decoder (anchors-free decoding to boxes + scores)
- `activation_serialization.py` — the canonical `activation`-argument (de)serialization pair, used directly by ~60 pre-existing files plus `layers/activations/common.py`'s wrapper trio (see `layers/CLAUDE.md`):
  - `serialize_activation(activation)` — makes a stored `activation` value JSON-safe for `get_config()`. A `keras.layers.Layer` goes through `keras.saving.serialize_keras_object`; any other callable goes through `keras.activations.serialize`; **everything else — including a bare `str`, `None`, and a dl_techniques factory key like `'mish'`/`'sparsemax'` — passes through unchanged.** That passthrough is mandatory, not incidental: `keras.activations.serialize` raises on a bare string, and a factory key is not a Keras activation at all
  - `deserialize_activation(activation, custom_objects=None, allow_layer=True)` — inverts `serialize_activation`. A `dict` goes through `keras.saving.deserialize_keras_object` (dispatches both the function form and the Layer form); everything else — `str`, `None`, an already-live callable — is returned **unchanged** (this function does not resolve a bare string to a callable; that is `resolve_activation`'s job, see below)
    - `allow_layer` (default `True`): preserves the original, Layer-permissive contract all ~60 pre-existing callers rely on. Pass `allow_layer=False` only when the caller's contract must reject a `keras.layers.Layer` activation (weights created in `call()` rather than `build()` break `.keras` weight loading) — `layers/activations/common.py::resolve_activation` is the one caller that does this
  - **Call-site placement rule**: `serialize_activation` belongs in `get_config()`; `deserialize_activation` belongs in `__init__`, on the value's way into the stored attribute — **never in `from_config`**. MEASURED: `keras.models.load_model(..., custom_objects=...)` builds sub-objects inside a `custom_object_scope`, so the `custom_objects=None` default already resolves an unregistered function correctly from `__init__`; that call site additionally covers hand-rolled `Cls(**cfg)` reconstruction and nested layers whose parent never calls their `from_config`
  - Consumed directly by ~60 pre-existing files and, as of this plan (`plan-2026-09-15T034909-a7edc8da`), migrated onto by 37 further files via `layers/activations/common.py`'s wrapper — see `layers/CLAUDE.md` for why those 37 go through the wrapper rather than this pair directly
- `matplotlib_backend.py` — `import_pyplot(with_cm=False)`: the one place library code acquires `matplotlib.pyplot`. **Setdefault semantics, not an override**: a non-empty `MPLBACKEND` is respected as-is; only when it is unset does the helper select `Agg` (via `matplotlib.use` *and* the env var, so subprocesses inherit it). Returns `plt`, or `(plt, cm)` when `with_cm=True`. All **five** matplotlib-using callbacks in `dl_techniques/callbacks/` go through it.
  - What it fixes is **divergence, not a crash**: three callbacks imported `pyplot` bare and two forced `Agg`, so the process-global backend depended on which plotted first. MEASURED on matplotlib 3.10.0: with `MPLBACKEND` unset a bare `import matplotlib.pyplot` resolves to `agg` on its own, with `DISPLAY` unset *and* with a bogus `DISPLAY=:99` (savefig OK) — matplotlib's own headless fallback. An earlier revision of this bullet claimed the bare importers "crashed on a headless host"; that claim was false and is retracted.
  - Scoped rule: **inside `dl_techniques/callbacks/`, never import `matplotlib.pyplot` directly** — use the helper. This is *not* yet a tree-wide invariant: `grep -rln "import matplotlib" src/dl_techniques --include=*.py` returns **29** files, of which **25** are outside `callbacks/` and outside the helper (all 10 of `analyzer/`, all 6 of `visualization/`, 2 in `datasets/`, `losses/clustering_loss.py`, `models/memory/som/model.py`, and 5 under `utils/`). None of the 25 is a Keras callback, so all are out of scope for the guard — but note that `callbacks/analyzer_callback.py` reaches matplotlib **transitively** through `dl_techniques.analyzer`, so an epoch hook does hit a bare import. Widening the guard to `analyzer/` is a separate, untaken decision.
  - The guard is `tests/test_callbacks/test_the_matplotlib_backend_is_headless.py`. Its subject list is a **derived AST census** of `callbacks/*.py` (not a hardcoded literal) with an anti-vacuity floor, and its behavioural arms run in fresh subprocesses whose `MPLBACKEND` the harness sets explicitly — the suite itself runs under `MPLBACKEND=Agg`, so an in-process assertion would read the environment rather than the code

### Subpackages
- `alignment/` — Feature alignment framework:
  - `alignment.py` — Core alignment logic
  - `metrics.py` — Alignment quality metrics
  - `utils.py` — Alignment helpers
- `masking/` — Masking strategy framework:
  - `strategies.py` — Masking strategies (random, block, etc.)
  - `factory.py` — Config-driven masking construction
- `geometry/` — Geometric math:
  - `poincare_math.py` — Poincare ball model operations for hyperbolic geometry

## Conventions

- `__init__.py` is empty — import from submodules directly
- Subpackages with `factory.py` support config-driven construction
- Geometry utilities support hyperbolic space operations used by graph models

## Testing

Tests in `tests/test_utils/`.
