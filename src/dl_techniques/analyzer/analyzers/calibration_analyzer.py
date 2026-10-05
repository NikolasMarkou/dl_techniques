"""
Assess the reliability and uncertainty of model predictions.

This analyzer evaluates how well a model's predicted probabilities align with
the true likelihood of outcomes. Modern neural networks, particularly when
trained with techniques that encourage low-entropy outputs (e.g., cross-
entropy loss), often become overconfident. This means the model might assign
a high probability (e.g., 99%) to a prediction that is incorrect. For
high-stakes applications, understanding and quantifying this discrepancy
between confidence and accuracy is critical. This module provides the tools
to measure this gap.

Architecture
------------
The analyzer operates as a post-processing component. It does not run the
model but consumes pre-computed predictions (probabilities) and true labels.
Its primary role is to delegate these data to a suite of specialized metric
functions, each designed to probe a different aspect of probabilistic
prediction quality. The results are then structured into two distinct
categories:
-   **Calibration Metrics**: Focus on the reliability of the probabilities. It
    answers the question: "When the model says it's P% confident, is it
    correct P% of the time?" Key metrics include ECE and Brier score.
-   **Confidence Metrics**: Characterize the model's internal sense of
    certainty, irrespective of correctness. It answers: "How certain is the
    model in its predictions?" Key metrics include prediction entropy and
    the distribution of maximum probabilities.

Foundational Mathematics
------------------------
The core of the analysis rests on established metrics from statistics and
information theory to quantify the quality of probabilistic forecasts.

-   **Expected Calibration Error (ECE)**: This is the primary metric for
    miscalibration. It measures the average gap between a model's prediction
    confidence and its actual accuracy. The calculation involves partitioning
    predictions into `M` bins based on their confidence scores. For each bin
    `B_m`, the average confidence `conf(B_m)` and accuracy `acc(B_m)` are
    computed. The ECE is the weighted average of their absolute difference:
    ECE = Σ_{m=1 to M} (|B_m|/n) * |acc(B_m) - conf(B_m)|
    A perfectly calibrated model has an ECE of 0.

-   **Brier Score**: This is a "proper scoring rule" that measures both
    calibration and resolution (the model's ability to distinguish outcomes).
    It is the mean squared error between the predicted probability vector `p`
    and the one-hot encoded true label vector `o`:
    BS = (1/N) * Σ_{i=1 to N} Σ_{j=1 to K} (p_ij - o_ij)²
    A lower Brier score is better, indicating predictions that are both
    accurate and well-calibrated.

-   **Shannon Entropy**: This metric is used to quantify the uncertainty of an
    individual prediction. For a probability distribution `p` over `K`
    classes, the entropy is:
    H(p) = -Σ_{i=1 to K} p_i * log(p_i)
    A low entropy value corresponds to a "peaked," high-confidence
    prediction, while a high entropy value indicates an uncertain prediction
    with probabilities spread across multiple classes.

References
----------
1.  Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On calibration
    of modern neural networks." ICML.
2.  Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good
    probabilities with supervised learning." ICML.
3.  Brier, G. W. (1950). "Verification of forecasts expressed in terms of
    probability." Monthly Weather Review.

"""

import numpy as np
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput
from ..calibration_metrics import (
    compute_ece,
    compute_ece_binary,
    compute_brier_score,
    compute_reliability_data,
    compute_prediction_entropy_stats,
    # DECISION plan-2026-10-05-analyzer-audit/F-081: the head-type router and the
    # convention-aware confidence statistics. `is_multilabel_output` re-derives what
    # `ModelAnalyzer._infer_output_activation` decides and then discards, so the two
    # cannot disagree about which convention applies.
    is_multilabel_output,
    compute_confidence_statistics,
)

# ---------------------------------------------------------------------


class CalibrationAnalyzer(BaseAnalyzer):
    """Analyzes model confidence and calibration."""

    def requires_data(self) -> bool:
        """Calibration analysis requires input data."""
        return True

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Analyze model confidence and calibration with consolidated metric storage.
        """
        logger.info("Analyzing confidence and calibration...")

        if cache is None:
            raise ValueError("Prediction cache is required for calibration analysis")

        skipped_models = {}

        for model_name in self.models:
            if model_name not in cache:
                skipped_models[model_name] = "no prediction cache entry"
                continue

            model_cache = cache[model_name]
            # This variable now correctly holds probabilities, not logits.
            y_pred_proba = model_cache.get('predictions', None)

            if y_pred_proba is None:
                # DECISION plan-2026-10-05-analyzer-audit/F-045
                # This branch is UNREACHABLE from `ModelAnalyzer`: `_cache_predictions`
                # sets `'logits'` only in the branch where it ALSO sets `'predictions'`
                # to the softmax, so logits non-None implies predictions non-None. It is
                # retained for hand-built caches, but the `keras.ops.softmax` call it used
                # to make would have softmaxed ALREADY-softmaxed probabilities — exactly
                # the D-032 defect, reintroduced inside the analyzer. Convert to a numpy
                # array and log loudly instead: a cache that needs this repair is a
                # caller bug, not something to paper over silently.
                logger.warning(
                    f"Model '{model_name}' has no 'predictions' in the cache but does "
                    f"have 'logits'. Refusing to re-softmax them: they are already "
                    f"probabilities if they came from ModelAnalyzer, and softmaxing "
                    f"them again would corrupt every calibration metric (D-032). Supply "
                    f"'predictions' directly.")
                skipped_models[model_name] = "cache has logits but no predictions"
                continue

            if y_pred_proba is None:
                logger.warning(f"Skipping calibration analysis for {model_name}: No predictions / logits available.")
                continue

            # DECISION plan-2026-10-05-analyzer-audit/F-046
            # `y_data` is read INSIDE the guarded region. It used to sit above the
            # `try`, so a cache entry missing the key raised `KeyError`, which the
            # `except (ValueError, TypeError)` could not catch and which propagated out
            # of `analyze()` before `save_results()` — losing every other model's
            # metrics to one incomplete cache entry.
            #
            # The whole per-model body is guarded from here. `InformationFlowAnalyzer`
            # already isolates models this way; this analyzer had no per-model guard at
            # all, so a 1-D `predict` output raised `IndexError` at `y_pred_proba.shape[1]`
            # and aborted the whole run.
            try:
                y_true = np.asarray(model_cache['y_data'])

                # DECISION plan-2026-10-05-analyzer-audit/F-081
                # Route the metrics to the convention the HEAD actually implies.
                #
                # `ModelAnalyzer._infer_output_activation` already distinguishes a sigmoid
                # multi-label head from a softmax one and throws the answer away — the
                # activation name never reaches this analyzer. `is_multilabel_output`
                # re-derives it from the same row-sum test on the same array, so the two
                # cannot disagree.
                #
                # What changes under it:
                #   * `brier_score` uses the per-label (mean over classes) convention, not
                #     the multiclass sum. Applying the sum to independent sigmoids charged
                #     one sample K times over — F-080.
                #   * the pooled top-1 `ece`, the reliability diagram and
                #     `per_class_conditional_top1_ece` are NOT computed and are reported
                #     as None, because "was the single top class right" is not a
                #     calibration question when there is no single correct class.
                #   * `per_class_ece` IS computed, and is the right metric for a sigmoid
                #     head: each column is an independent probability calibrated against
                #     its own 0/1 indicator (D-015's classwise form).
                #   * the confidence statistics report the multi-label quantities and NaN
                #     the single-label ones (F-081).
                multilabel = is_multilabel_output(y_pred_proba)

                # Convert to class indices if needed
                y_true_idx = self._labels_to_indices(
                    y_true, y_pred_proba, multilabel=multilabel)

                num_classes = y_pred_proba.shape[1]
                if multilabel:
                    # Top-1 quantities are undefined; per-class and the Brier score are not.
                    ece = None
                    reliability_data = None
                else:
                    ece = compute_ece(y_true_idx, y_pred_proba,
                                      self.config.calibration_bins)
                    reliability_data = compute_reliability_data(
                        y_true_idx, y_pred_proba, self.config.calibration_bins)

                # Brier score requires one-hot encoded true labels. Convert if necessary.
                if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                    y_true_one_hot = np.zeros((y_true_idx.size, num_classes))
                    y_true_one_hot[np.arange(y_true_idx.size), y_true_idx] = 1
                else:
                    y_true_one_hot = y_true
                brier_score = compute_brier_score(
                    y_true_one_hot, y_pred_proba, multilabel=multilabel)

                # Compute per-class ECE
                per_class_ece = []
                per_class_conditional_top1_ece = []
                # DECISION plan-2026-10-05-analyzer-audit/F-044
                # `per_class_bins` halves `calibration_bins`, so the published
                # `per_class_ece` is a 5-bin ECE while `ece` is a 10-bin ECE at the
                # defaults. That is a deliberate choice (a per-class column holds far
                # less mass than the pooled top-1 score), but it was undocumented at the
                # public surface, so a reader comparing `per_class_ece[c]` against `ece`
                # was comparing two different estimators. It is now a NAMED config field
                # rather than an inline expression, so it is visible and overridable.
                per_class_bins = self._per_class_bins()

                for c in range(num_classes):
                    # DECISION plan-2026-09-01T225724-e79ad4bd/D-015
                    # `per_class_ece` is CLASSWISE ECE (Kull et al. 2019): the class-c
                    # PROBABILITY COLUMN over ALL samples, against the indicator
                    # `y_true == c`. Do NOT go back to masking samples by true label and
                    # re-running the top-1 ECE on them — that quantity is blind to the
                    # entire off-diagonal and reports 0.0 for a class the model never
                    # predicts, however wrong its column is. See decisions.md D-015.
                    #
                    # F-081: `y_true_idx` is the INDICATOR matrix for a multi-label head,
                    # so `y_true_idx[:, c]` is that label's 0/1 column. The same line
                    # therefore serves both conventions.
                    if multilabel:
                        outcomes_c = y_true_idx[:, c].astype(float)
                    else:
                        outcomes_c = (y_true_idx == c).astype(float)
                    per_class_ece.append(
                        compute_ece_binary(
                            outcomes_c,
                            y_pred_proba[:, c],
                            per_class_bins,
                        )
                    )

                    # The legacy quantity, kept under an honest name. It is a TOP-1
                    # quantity, so it does not exist for a multi-label head (F-081):
                    # reported as None there rather than as a fabricated 0.0.
                    if multilabel:
                        per_class_conditional_top1_ece.append(None)
                    else:
                        class_mask = y_true_idx == c
                        if np.any(class_mask):
                            per_class_conditional_top1_ece.append(
                                compute_ece(y_true_idx[class_mask],
                                            y_pred_proba[class_mask],
                                            per_class_bins)
                            )
                        else:
                            per_class_conditional_top1_ece.append(0.0)

                # Store only calibration-specific metrics (no entropy here)
                results.calibration_metrics[model_name] = {
                    'ece': ece,
                    'brier_score': brier_score,
                    'ece_bins': int(self.config.calibration_bins),
                    'per_class_ece': per_class_ece,
                    'per_class_ece_bins': per_class_bins,
                    'per_class_conditional_top1_ece': per_class_conditional_top1_ece,
                    # F-081: which convention produced the numbers above, so a reader
                    # never has to infer it from the magnitudes.
                    'multilabel': bool(multilabel),
                }

                # Store reliability data separately (for plotting). A multi-label head
                # has no top-1 correctness forecast, so there is no reliability diagram.
                if reliability_data is not None:
                    results.reliability_data[model_name] = reliability_data

                # Consolidate ALL confidence-related metrics including entropy. F-081:
                # entropy is the Shannon entropy of a DISTRIBUTION and is undefined over
                # independent sigmoids, so it is not computed for a multi-label head.
                confidence_metrics = compute_confidence_statistics(
                    y_pred_proba, multilabel=multilabel)
                if multilabel:
                    entropy_stats = {}
                else:
                    entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

                # Combine all confidence-related metrics into one place
                all_confidence_metrics = {**confidence_metrics, **entropy_stats}
                results.confidence_metrics[model_name] = all_confidence_metrics

            except Exception as e:
                # F-046: one model's failure skips ONLY that model. The `try` covers the
                # analyzer's own arithmetic, so a genuine bug still surfaces in the log
                # with its traceback rather than as a plausible-looking metric.
                logger.error(
                    f"Calibration analysis failed for '{model_name}': {e}. "
                    f"Skipping this model only.", exc_info=True)
                skipped_models[model_name] = f"{type(e).__name__}: {e}"
                continue

        if skipped_models:
            logger.warning(
                f"Calibration analysis skipped for {len(skipped_models)} of "
                f"{len(self.models)} model(s): "
                + "; ".join(f"{name} ({reason})" for name, reason in skipped_models.items())
            )

    def _per_class_bins(self) -> int:
        """Bin count for the per-class calibration metrics.

        Returns:
            ``config.per_class_calibration_bins`` when set, otherwise half of
            ``config.calibration_bins`` (never fewer than 2). See the F-044 anchor on
            the config field for why the two are not the same number by default.
        """
        override = self.config.per_class_calibration_bins
        if override is not None:
            return max(2, int(override))
        return max(2, int(self.config.calibration_bins) // 2)

    def _labels_to_indices(
            self, y_true: np.ndarray, y_pred_proba: np.ndarray,
            *, multilabel: bool = False,
    ) -> np.ndarray:
        """Reduce a label array to one integer class index per sample.

        Args:
            y_true: Labels as stored: 1-D class indices, a one-hot ``(N, K)`` block,
                or a multi-hot ``(N, K)`` block.
            y_pred_proba: The model's probabilities, used only for its class count.

        Returns:
            Integer array of shape ``(N,)``.

        Raises:
            ValueError: If ``y_true`` is not a recognised encoding, or if a one-hot /
                multi-hot block's width disagrees with the predicted class count. The
                caller isolates the model on any exception (F-046).
        """
        num_classes = int(y_pred_proba.shape[1]) if y_pred_proba.ndim > 1 else 0

        # DECISION plan-2026-10-05-analyzer-audit/F-081
        # For a multi-label head the 0/1 INDICATOR is the right reduction, not an index:
        # `per_class_ece` needs `(indicator[:, c] == 1)` per class, and the Brier score
        # needs the indicator itself. Return the indicator's row indices as booleans
        # cast to float is NOT wanted, so the indicator is returned unchanged and the
        # caller treats it as a matrix. This is the one place a "class index" is not an
        # index, which is why the return type is documented as "an index array OR the
        # indicator matrix".
        if multilabel:
            if y_true.ndim < 2 or y_true.shape[1] != num_classes:
                raise ValueError(
                    f"a multi-label head needs a (n_samples, {num_classes}) indicator "
                    f"matrix of labels, got shape {y_true.shape}")
            indicator = np.asarray(y_true, dtype=float)
            if not np.all(np.isin(indicator, (0.0, 1.0))):
                raise ValueError(
                    f"a multi-label indicator must be 0/1; found values outside "
                    f"[0, 1] (min {indicator.min():.3g}, max {indicator.max():.3g})")
            return indicator

        if y_true.ndim > 1 and y_true.shape[1] > 1:
            # DECISION plan-2026-10-05-analyzer-audit/F-047
            # Validate the block encoding instead of `argmax`-ing it blind.
            #
            # `np.argmax` on a MULTI-HOT row keeps only the first maximal class, so a
            # genuine multi-label target was silently scored in a single-label outcome
            # space: top-1 ECE, the reliability diagram and
            # `per_class_conditional_top1_ece` all became meaningless, and an all-zero
            # multi-hot row became class 0 with no diagnostic. Nothing distinguished a
            # valid multi-hot block from a mis-shaped one.
            #
            # The rule now: a row whose maximum is 1.0 and which sums to ~1 is treated as
            # ONE-HOT (argmax is correct and lossless); anything else is MULTI-HOT and is
            # reported as unsupported rather than silently mis-scored. The analyzer
            # cannot produce honest top-1 calibration numbers for a multi-label target,
            # so it says so instead of inventing them.
            row_max = y_true.max(axis=-1)
            row_sum = y_true.sum(axis=-1)
            is_one_hot = np.allclose(row_max, 1.0, atol=1e-6) and np.allclose(
                row_sum, 1.0, atol=1e-6)

            if not is_one_hot:
                raise ValueError(
                    f"y_true looks like a MULTI-HOT block (rows do not each sum to 1; "
                    f"max per row in [{row_max.min():.3g}, {row_max.max():.3g}], "
                    f"sum per row in [{row_sum.min():.3g}, {row_sum.max():.3g}]). "
                    f"Top-1 calibration metrics (ECE, reliability diagram, per-class "
                    f"ECE) are defined only for a single-label target: a multi-hot row "
                    f"has no single correct class, so the honest answer is to skip this "
                    f"model rather than argmax it. This analyzer's sigmoid support means "
                    f"the PREDICTIONS survive intact — see "
                    f"tests/test_analyzer/test_model_analyzer.py's sigmoid probe — but "
                    f"none of the metrics below are meaningful for it.")

            if num_classes and y_true.shape[1] != num_classes:
                raise ValueError(
                    f"y_true one-hot width {y_true.shape[1]} disagrees with the "
                    f"model's {num_classes} output columns.")

            return np.argmax(y_true, axis=-1)

        flat = y_true.flatten()
        # DECISION plan-2026-10-05-analyzer-audit/F-048
        # Round, do not TRUNCATE. `.astype(int)` mapped 1.7 -> 1 and -0.2 -> 0, silently
        # corrupting a float-valued label array. `np.rint` is the correct conversion for
        # labels that are integral-valued floats, and the range check below turns a
        # genuinely non-integral array into an error rather than a plausible number.
        as_float = np.asarray(flat, dtype=np.float64)
        if not np.all(np.isfinite(as_float)):
            raise ValueError("y_true contains NaN or inf")
        if not np.allclose(as_float, np.rint(as_float), atol=1e-9):
            raise ValueError(
                f"y_true holds non-integral class labels "
                f"(e.g. {as_float[0]!r}); refusing to truncate them to integers.")
        indices = np.rint(as_float).astype(np.int64)

        if num_classes:
            if indices.min() < 0 or indices.max() >= num_classes:
                raise ValueError(
                    f"y_true holds class indices outside [0, {num_classes}): "
                    f"min={indices.min()}, max={indices.max()}")

        return indices

# ---------------------------------------------------------------------
# DECISION plan-2026-10-05-analyzer-audit/F-081
# `_compute_confidence_metrics` is GONE, superseded by
# `calibration_metrics.compute_confidence_statistics(probabilities, multilabel=...)`.
# It could not express the multi-label case: it unconditionally took the top-1/top-2
# margin and `1 - Σp²` of what it assumed was a distribution. The replacement is a
# public function, is unit-testable on its own, and is shared with any other caller.
#
# It also carried two defects the replacement fixes: `gini` was computed from the SORTED
# probabilities (harmless, since the sum of squares is order-free, but it sorted a whole
# row to no purpose), and the single-class case returned `gini = 0.0` — the value a
# perfectly UNIFORM distribution gives, published for a model that had no distribution to
# be uniform over. That is the same "a plausible zero standing in for an undefined
# quantity" failure this audit removed elsewhere; it is now NaN.
#
# Do NOT reintroduce a private copy of this logic.
