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
- `embedding_quality.py` — Pool-level text-embedding metrics: ranking (`rank_of_ground_truth`, `recall_at_k`, `mrr_at_k`, `ndcg_at_k`) and geometry (`anisotropy`, `effective_rank`, `alignment`, `uniformity`, `embedding_norm_stats`). **Plain functions, not `keras.metrics.Metric`** — see Conventions below.
- `brier_score.py` — Brier Score (proper scoring rule for probabilistic predictions): `BrierScore` for binary / multi-label classification, `CategoricalBrierScore` for multi-class (with sparse-label fast path for segmentation). See `research/brier_score.md` for background.

## Conventions

- `__init__.py` is empty — import from submodules directly
- **Streaming** metrics inherit from `keras.metrics.Metric` and must implement
  `update_state()`, `result()`, `reset_state()`, and `get_config()`
- **Pool-level** metrics are plain functions on numpy arrays. Some quantities
  cannot be expressed as a streaming `update_state` at all: an SVD, a mean over
  every pair, a ranking against a whole candidate pool all need the complete
  matrix at once. `embedding_quality.py` is the worked example; the package has
  always also carried plain functions for this reason (`llm_metrics.self_bleu`,
  `llm_metrics.distinct_n`, `perplexity_metric.perplexity`,
  `time_series_metrics.calculate_comprehensive_metrics`)

## Testing

Tests in `tests/test_metrics/`.
