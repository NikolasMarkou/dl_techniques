# Utils Package

Shared utilities used across the library — tensor operations, geometry, masking, alignment, export, and more.

## Structure

### Top-level Modules
- `tensors.py` — Core tensor ops: `gram_matrix()`, `power_iteration()`, `window_partition()`/`window_reverse()`, Gaussian kernels, orthonormality validation. Also the shared home for two pure dimension-contract predicates that used to be private copies in `layers/`:
  - `is_power_of_two(n)` — `True` only for `n >= 1` with one set bit, so `0` and negatives are rejected. Used by `layers/orthogonal_butterfly.py` and `layers/norms/polar_weight_norm.py` to refuse a last dimension they cannot pair. `polar_weight_norm._next_power_of_two` is a different function and stays where it is
  - `canonical_binary_input_shape(input_shape)` — collapses one-shape-or-list-of-shapes into the single shape a binary elementwise layer builds from, raising on more than two shapes or on a mismatched pair. Used by `layers/logic/{logic,arithmetic}_operators.py` from three sites each (`build`, `compute_output_shape`, `_assert_call_shape_contract`). Import it by its bare name: `tests/test_layers/test_logic/test_the_nine_source_fixes_stay_fixed.py` asserts the call sites resolve as `ast.Name`, so a qualified `tensors.canonical_binary_input_shape(...)` call would go unseen. It is NOT the `is_list_of_shapes` idiom inlined three times under `layers/attention/`, which is deliberately kept separate
- `constants.py` — Shared constants for config keys
- `convert.py` — `convert_numpy_to_python()` for JSON serialization
- `random.py` — `rayleigh()` distribution generator with statistical validation
- `scaling.py` — Quantization utilities: `range_from_bits()`, `round_clamp()`, `scale()`
- `logger.py` — Centralized library logger
- `filesystem.py` — File discovery and streaming (`image_file_generator()`)
- `bounding_box.py` — IoU (GIoU/DIoU/CIoU), format conversion, NMS
- `corruption.py` — 10 image corruption types (noise, blur, distortion, color) with severity levels, backend-agnostic
- `inference.py` — `FullImageInference`: sliding window patch extraction + aggregation for YOLOv12, with profiling
- `train.py` — `TrainingConfig` + `train_model()` with early stopping, checkpointing, CSV logging
- `tokenizer.py` — `TiktokenPreprocessor`: BERT-compatible tokenization with special tokens, attention masks
- `graphs.py` — Adjacency normalization (symmetric/row), sparse ops, random graph generation, negative sampling
- `visualization.py` — `collage()`, `draw_figure_to_buffer()`, `plot_confusion_matrices()`
- `visualization_manager.py` — `VisualizationManager` with consistent styling and timestamped output
- `conformal_forecaster.py` — `ConformalForecaster`: model-agnostic uncertainty quantification with finite-sample coverage guarantees. Supports multiple nonconformity measures (absolute, normalized, CQR, locally_weighted) and multi-horizon strategies
- `forecastability_analyzer.py` — `ForecastabilityAssessor`: permutation entropy, AMI-based delay estimation, Cao's embedding dimension, baseline benchmarking, and forecastability scoring [0-100]
- `deep_supervision.py` — Deep-supervision output helpers (auxiliary heads, weight scheduling glue)
- `drop_path.py` — `linear_drop_path_rates(num_blocks, max_rate)`: computes per-block stochastic-depth (drop-path) rates for transformer/ConvNeXt block stacks. The actual drop-path layer is `StochasticDepth` in `dl_techniques/layers/stochastic_depth.py`
- `weight_transfer.py` — two loaders, deliberately not interchangeable:
  - `load_weights_from_checkpoint(target, ckpt_path, skip_prefixes, strict)`: layer-by-layer *partial* transfer between different architectures (a pretrain trunk into a fine-tune model). Use this instead of `model.load_weights(by_name=True)` (broken in Keras 3.8 for `.keras` files). A zero-layer result **reports rather than raises** — `src/train/beit/` and `src/train/energy_transformer/` build their own warm-start guards on that report
  - `load_weights_or_raise(model, weights_path, skip_mismatch)`: whole-file `.keras` restore into the **same** architecture, returning the number of variables whose value changed and raising when that is zero. `model.load_weights(path, skip_mismatch=True)` restores nothing and returns normally when names or shapes do not match, so never call it bare
- `yolo_decode.py` — YOLOv12 output decoder (anchors-free decoding to boxes + scores)
- `matplotlib_backend.py` — `import_pyplot(with_cm=False)`: the one headless-safe matplotlib importer. Calls `matplotlib.use("Agg")` BEFORE importing `pyplot` and returns `plt`, or `(plt, cm)` when `with_cm=True`. All four plotting callbacks in `dl_techniques/callbacks/` go through it; two of them previously imported `pyplot` with no backend forced, which crashed on a headless host unless the trainer happened to set `MPLBACKEND`. Never import `matplotlib.pyplot` directly in library code. Backend selection is process-global and effectively once-only, so the guard (`tests/test_callbacks/test_the_matplotlib_backend_is_headless.py`) runs each arm in a fresh subprocess with `MPLBACKEND` deleted — an in-process assertion reads the environment, not the code

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
- `export/` — Model export utilities:
  - `onnx.py` — ONNX export
  - `tflite.py` — TFLite export

## Conventions

- `__init__.py` is empty — import from submodules directly
- Subpackages with `factory.py` support config-driven construction
- Geometry utilities support hyperbolic space operations used by graph models

## Testing

Tests in `tests/test_utils/`.
