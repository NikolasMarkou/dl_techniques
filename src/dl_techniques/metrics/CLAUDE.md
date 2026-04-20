# Metrics Package

Custom Keras metrics for specialized evaluation tasks.

## Modules

- `capsule_accuracy.py` — Accuracy metric for capsule network outputs
- `clip_accuracy.py` — CLIP model retrieval accuracy
- `hrm_metrics.py` — Hierarchical reasoning model metrics
- `multi_label_metrics.py` — Multi-label classification metrics (F1, precision, recall per label)
- `perplexity_metric.py` — Language model perplexity
- `psnr_metric.py` — Peak Signal-to-Noise Ratio for image quality
- `time_series_metrics.py` — Time series forecasting metrics (MASE, SMAPE, quantile loss, etc.)
- `depth_metrics.py` — Monocular depth estimation metrics (AbsRel, SqRel, RMSE, RMSE log, delta threshold)

## Conventions

- `__init__.py` is empty — import from submodules directly
- All metrics inherit from `keras.metrics.Metric`
- Must implement `update_state()`, `result()`, `reset_state()`, and `get_config()`

## Testing

Tests in `tests/test_metrics/`.
