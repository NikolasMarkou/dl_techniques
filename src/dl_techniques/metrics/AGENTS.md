# Metrics Package

This directory contains custom Keras metrics and pool-level evaluation functions for specialized deep learning tasks. It covers a wide range of domains including causal language modeling, vision-language contrastive learning, time-series forecasting, topographic capsule networks, and spatial autocorrelation.

## Core Architectural Rules & Conventions

Before modifying or adding to this package, agents must adhere to the following strict architectural rules:

1.  **Streaming vs. Pool-Level Metrics:**
    *   **Streaming Metrics:** Inherit from `keras.metrics.Metric`. They must implement `update_state()`, `result()`, `reset_state()`, and `get_config()`. They are used for quantities that can be accumulated batch-by-batch.
    *   **Pool-Level Metrics:** Plain Python/NumPy functions. If a metric requires an SVD, ranking against a whole candidate pool, or cross-correlating entire sequences (e.g., `embedding_quality.py`, `pass_at_k.py`, `generation_metrics.py`, `topographic.py`), it must be a plain function. Do not attempt to force these into a `keras.metrics.Metric` subclass using unbounded state accumulation.

2.  **Keras 3 Graph-Mode Compliance:**
    *   Always use `keras.ops` for tensor operations in `keras.metrics.Metric` subclasses to ensure backend agnosticism (JAX, TensorFlow, PyTorch).
    *   **Shape Ranks:** Prefer `len(tensor.shape)` over `len(keras.ops.shape(tensor))` as a style choice. `keras.ops.shape` is not the failing call: on Keras 3.8 it returns a tuple on TF, JAX and torch (ints for known dims, scalar tensors for unknown dims on TF), so `len()` of it works in graph mode. What does fail is using an unknown dimension as a Python bool, e.g. `ops.shape(y)[-1] > 1` when the last dim is `None` (a symbolic tensor, `OperatorNotAllowedInGraphError` under `tf.function(autograph=False)`). Branch on the static `.shape` instead and decide the `None` case explicitly (see `primary_output_metrics._true_class_indices`). A tensor of unknown rank fails with a `ValueError` either way.
    *   Ensure all denominators use `keras.config.epsilon()` or explicit `ops.maximum` clipping to avoid `NaN` results. Use `ops.divide_no_nan` where appropriate.

3.  **Serialization:**
    *   All Keras metrics must be decorated with `@register_dl_technique("dl_techniques.metrics.[module_name]")`.
    *   All constructor arguments must be saved as instance attributes before calling `super().__init__()` and exposed in `get_config()`.

## Module Manifest

### Natural Language & Generation
*   `llm_metrics.py`: Compile-time CLM metrics (`BitsPerToken`, `BitsPerCharacter`). Accumulators use `float32` for AMP safety. Also provides pure-Python, NLTK-free probe helpers (`self_bleu`, `distinct_n`). Note: `self_bleu` measures diversity, not quality.
*   `perplexity_metric.py`: Language model perplexity (`Perplexity`), defined as `exp(cross_entropy)`. Includes both a stateful Keras metric and a functional interface. Supports padding omission via `ignore_class`.
*   `generation_metrics.py`: Corpus-level NLP metrics (`bleu`, `rouge`, `chrf`, `meteor`, `cider`). Plain functions, not Keras metrics. 
    *   *Hard Rules:* Do not introduce NLTK or SacreBLEU dependencies. ROUGE-Lsum uses a global per-token hit budget to prevent F1 > 1.0. A BLEU candidate shorter than `max_order` scores 0.0 (except with `add_k` smoothing). METEOR is exact-match only unless `match_fn` is provided.

### Vision, Retrieval & Multimodal
*   `clip_accuracy.py`: Contrastive learning metrics (`CLIPAccuracy`, `CLIPRecallAtK`). `update_state` expects `y_pred` to be a dictionary containing `logits_per_image` and `logits_per_text`. Supports directional tracking (`i2t` vs `t2i`).
*   `depth_metrics.py`: Monocular depth evaluation (`AbsRelMetric`, `DeltaThresholdMetric`, `SqRelMetric`, `RMSEMetric`, `RMSELogMetric`). Expects `y_true` to be a concatenated `[depth, mask]` tensor. Includes a closed-form `align_scale_and_shift` helper.
*   `keypoint_matching.py`: `KeypointMatchMetric` for LightGlue-style matchers. Computes precision/recall/F1 from log assignments. Excludes ignored/padded keypoints (label `-2`).
*   `psnr_metric.py`: Peak Signal-to-Noise Ratio. Designed for deep supervision; automatically slices `y_pred[0]` if the model returns a list of outputs.
*   `ssim_metric.py`: Structural Similarity Index. Currently relies on `tf.image.ssim` and strictly requires the TensorFlow backend.

### Topographic & Spatial
*   `topographic.py`: Equivariance metrics for Topographic-VAEs. Plain NumPy functions. `equivariance_error` measures smoothness (low for invariant models too). `capcorr_correlation` measures actual correspondence via periodic cross-correlation.
*   `spatial_autocorrelation.py`: Moran's I computations. Supports 2-D `(H, W)` and 3-D `(D, H, W)` maps. 
    *   *Hard Rules:* Takes unthresholded maps only. Uses Queen contiguity (l-infinity neighbourhood) exclusively to match spatial loss objectives. Does not construct `N x N` dense matrices.
*   `topographic_selectivity.py`: Post-hoc measurement chain for spatial models. Includes 2-sample t-maps and Benjamini-Hochberg FDR correction. Correction is applied jointly across all layers, never per-layer, to prevent massive false positive inflation.

### Probabilistic, Classification & Sampling
*   `brier_score.py`: Proper scoring rules for probability calibration. `BrierScore` (binary/multi-label) and `CategoricalBrierScore` (multi-class). Includes a highly optimized sparse path for segmentation masks using the identity `||p||^2 - 2 p_c + 1`.
*   `probabilistic_forecast_metrics.py`: Metrics for quantile forecasts (`CoverageMetric`, `SharpnessMetric`). Extracts lower and upper bounds via `low_index` and `high_index` slices on the final tensor axis.
*   `pass_at_k.py`: Coverage metrics for verifier-scored sampling (`pass_at_k`, `pass_at_k_curve`). 
    *   *Hard Rule:* The unbiased estimator strictly uses `scipy.special.gammaln`. Do not replace this with integer binomials (`math.comb`), which will overflow `float64` for standard evaluation budgets.
*   `multi_label_metrics.py`: `MultiLabelMetrics`. Computes precision, recall, and F1 across independent binary classes. Supports background exclusion.
*   `void_aware_accuracy.py`: Pixel accuracy (`VoidAwareAccuracy`). Explicitly excludes `ignore_index` from the denominator. Standard Keras `SparseCategoricalAccuracy` fails here by scoring void pixels as incorrect.

### Embeddings & Time Series
*   `embedding_quality.py`: Pool-level text-embedding metrics. Contains Ranking metrics (`ndcg_at_k`, `mrr_at_k`) and Geometry diagnostics (`anisotropy`, `uniformity`, `effective_rank`). 
    *   *Hard Rule:* Tie-breaking in `rank_of_ground_truth` must be pessimistic to correctly penalize collapsed encoders.
*   `time_series_metrics.py`: Forecasting metrics. Includes `SMAPE` (symmetric bounding 0-200) and `PerHorizonError` for tracking scalar metrics at specific forecast steps `h`.

### Specialized Utility Metrics
*   `capsule_accuracy.py`: Evaluates accuracy based on capsule vector L2 lengths rather than softmax probabilities.
*   `half_precision_top_k.py`: `HalfPrecisionTopKAccuracy`. Casts logits to `float32` before computing top-k. Bypasses a Keras bug where `InTopKV2` operations crash on `float16`/`bfloat16` inputs.
*   `primary_output_metrics.py`: `PrimaryOutputAccuracy`, `PrimaryOutputTopKAccuracy`. Silently routes only the primary (index 0) output of a multi-output model to the accuracy calculation.
*   `hrm_metrics.py`: Custom metrics aggregator (`HRMMetrics`) for Hierarchical Reasoning Models. Not a `keras.metrics.Metric` subclass; meant for explicit `update_state()` calls inside custom `train_step` blocks.

## Testing Context
Unit tests for this module reside in `tests/test_metrics/`. When modifying pool-level metrics, ensure deterministic edge cases (e.g., all-zero matrices, fully collapsed vectors) are explicitly tested against numerical `NaN` / `Inf` outputs.