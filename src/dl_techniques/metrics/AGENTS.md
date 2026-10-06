# Metrics Package

Custom Keras metrics for specialized evaluation tasks.

## Modules

- `capsule_accuracy.py` — Accuracy metric for capsule network outputs
- `clip_accuracy.py` — CLIP model retrieval accuracy
- `hrm_metrics.py` — Hierarchical reasoning model metrics
- `pass_at_k.py` — Sampling **coverage**: `pass_at_k` (unbiased and plug-in estimators), `pass_at_k_curve`, `per_problem_pass_at_k`, `solved_counts`. Plain functions on an `(n_problems, n_samples)` outcome matrix — the shape a verifier-scored sampler produces, not a `keras.metrics.Metric`, because every quantity is a functional of the whole matrix (see Conventions). `> 0` marks a sample solved; threshold partial credit upstream. Consumed by `dl_techniques.optimization.ssp` — do not add a second estimator anywhere else
- `multi_label_metrics.py` — Multi-label classification metrics (F1, precision, recall per label)
- `perplexity_metric.py` — Language model perplexity
- `psnr_metric.py` — Peak Signal-to-Noise Ratio for image quality
- `keypoint_matching.py` — `KeypointMatchMetric` (`precision` / `recall` / `f1` of LightGlue-style matches vs homography labels; packed labels and `log_assignments` as in `LightGlueLoss`; wire through `compile(metrics={"log_assignments": [...]})`)
- `time_series_metrics.py` — Time series forecasting metrics (MASE, SMAPE, quantile loss, etc.)
- `topographic.py` — Topographic-VAE equivariance metrics (Keller & Welling 2022). `equivariance_error` (Eq. 13) and `capcorr_correlation` / `capcorr_per_capsule` (Eq. 15/16), plus `roll_capsules` and `observed_roll`. **Plain NumPy functions, not `keras.metrics.Metric`** — `equivariance_error` compares every timestep pair and `capcorr` cross-correlates a canonical timestep against timestep 0, so both need the whole sequence at once and neither reduces to a streaming `update_state`. Read the module docstring before using `equivariance_error` alone: it is a SMOOTHNESS measure and is low for an INVARIANT representation too, which is exactly why `capcorr` is here beside it
- `spatial_autocorrelation.py` — Moran's I of a value map: `morans_i` (general sparse form), `morans_i_grid` (offset-walk, never materialises an `N x N` matrix), `grid_weights` / `mesh_weights` (the mesh path exists because a cortical surface gives each vertex six neighbours, which no rectangular grid can express), `label_islands`, `islands_morans_i` (each island scored on its own mean and its own internal adjacency) and `morans_i_permutation_test`. **Plain NumPy functions, not `keras.metrics.Metric`** — every one needs the whole map. Two rules are load-bearing: take the map **UNTHRESHOLDED** (a patch of zeros is a patch of agreement, so thresholding first manufactures the clustering the statistic measures), and use **queen** contiguity, which is the same l-infinity neighbourhood the spatial loss trains with. Chance is `-1/(N-1)`, not zero. `islands_morans_i` is reported *alongside* the standard value and is **not** a strict improvement: on a small island it is high-variance and sometimes negative (Anselin's queen-contiguity instability)
- `topographic_selectivity.py` — the post-hoc measurement chain: `contrast_tmap` (two-sample t per unit; a unit constant in both conditions is reported `t = 0, p = 1` rather than SciPy's `nan`), `fdr_bh` (delegates to SciPy — do not add a second transcription of the step-up rule), `fdr_across_layers` (**one** correction over every tap, then split back; correcting each tap separately admits ~470 false positives per 784-unit layer), `grow_clusters` (most-selective-cell-first; provably the connected components of the same-sign significant cells when `max_size=None`, pinned against `label_islands`), `cluster_response` and `profile_consistency`. **Plain NumPy functions** — see Conventions below
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
