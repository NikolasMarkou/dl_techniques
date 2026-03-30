# Utils Package

Shared utilities used across the library — tensor operations, geometry, masking, alignment, export, and more.

## Structure

### Top-level Modules
- `tensors.py` — Core tensor ops: `gram_matrix()`, `power_iteration()`, `create_causal_mask()`, `window_partition()`/`window_reverse()`, Gaussian kernels, orthonormality validation
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
