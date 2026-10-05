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
    compute_prediction_entropy_stats
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

                # Convert to class indices if needed
                y_true_idx = self._labels_to_indices(y_true, y_pred_proba)

                # Compute calibration-specific metrics
                ece = compute_ece(y_true_idx, y_pred_proba, self.config.calibration_bins)
                reliability_data = compute_reliability_data(
                    y_true_idx, y_pred_proba, self.config.calibration_bins)

                # Brier score requires one-hot encoded true labels. Convert if necessary.
                num_classes = y_pred_proba.shape[1]
                if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                    y_true_one_hot = np.zeros((y_true_idx.size, num_classes))
                    y_true_one_hot[np.arange(y_true_idx.size), y_true_idx] = 1
                else:
                    y_true_one_hot = y_true
                brier_score = compute_brier_score(y_true_one_hot, y_pred_proba)

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
                    per_class_ece.append(
                        compute_ece_binary(
                            (y_true_idx == c).astype(float),
                            y_pred_proba[:, c],
                            per_class_bins,
                        )
                    )

                    # The legacy quantity, kept under an honest name.
                    class_mask = y_true_idx == c
                    if np.any(class_mask):
                        per_class_conditional_top1_ece.append(
                            compute_ece(y_true_idx[class_mask], y_pred_proba[class_mask],
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
                }

                # Store reliability data separately (for plotting)
                results.reliability_data[model_name] = reliability_data

                # Consolidate ALL confidence-related metrics including entropy
                confidence_metrics = self._compute_confidence_metrics(y_pred_proba)
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
            self, y_true: np.ndarray, y_pred_proba: np.ndarray
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

    def _compute_confidence_metrics(self, probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various confidence metrics (excluding entropy which comes from entropy_stats)."""
        max_prob = np.max(probabilities, axis=1)

        # Handle single-class case for margin and gini
        if probabilities.shape[1] > 1:
            # Sort probabilities to find the top two for margin calculation
            sorted_probs = np.sort(probabilities, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            gini = 1 - np.sum(sorted_probs**2, axis=1)
        else:
            # Margin and Gini are not well-defined for a single class output
            margin = np.full(probabilities.shape[0], np.nan)
            gini = np.zeros(probabilities.shape[0])

        return {
            'max_probability': max_prob,
            'margin': margin,
            'gini_coefficient': gini
        }

# ---------------------------------------------------------------------
