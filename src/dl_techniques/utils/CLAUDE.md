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
