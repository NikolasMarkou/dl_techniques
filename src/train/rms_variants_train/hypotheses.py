"""Falsifiable per-variant hypothesis registry for the RMS-variants study.

For each of the 8 norm variants in :data:`config.NORM_VARIANTS`, this module
declares a single **falsifiable claim** with a concrete numerical STOP-IF
threshold on a metric column that the harness already collects (either the
headline metric, a probe statistic, or a calibration/distribution-shift
signal). :func:`evaluate_hypothesis` returns one of four verdicts:

- ``CONFIRMED``   — the claim's STOP-IF threshold is **not** crossed (i.e. the
  layer behaves consistently with its documented design).
- ``REJECTED``    — the STOP-IF threshold is crossed (the layer's claim does
  **not** hold on observed data).
- ``INCONCLUSIVE`` — insufficient data (n < min_samples) or the metric column
  is present but all NaN.
- ``N/A``         — this variant has no applicable hypothesis for the
  (experiment, mode) being scored (e.g. ``band_logit_norm``'s ECE-reduction
  hypothesis is N/A on regression E4).

Design constraints (cf. plan_2026-05-18_e1f12eab plan.md / D-001):
- Thresholds are stated from the **layer's own documented claim** — they are
  NOT post-hoc tuned to match observed Phase 1 data (avoid Scenario A of the
  pre-mortem).
- Every metric column referenced is one the trainers / probes already emit
  (cf. ``callbacks.py`` ActivationStatsProbe / GradNormProbe; ``report.py``
  headline + probes aggregation).
- The registry is purely **additive** to ``report.py:VARIANT_CRITERIA``
  (invariant I4): it does not modify nor replace the PASS/FAIL verdict path.

The verdict column ``hypothesis_verdict`` is emitted by ``report.py`` next to
the existing ``verdict`` column.

Plan: ``plans/plan_2026-05-18_e1f12eab/plan.md``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Tuple

import pandas as pd

from dl_techniques.utils.logger import logger
from train.rms_variants_train.config import NORM_VARIANTS


# DECISION plan_2026-05-18_e1f12eab/D-001: this is the campaign's per-variant
# falsifiable-hypothesis registry. Every entry has a STOP-IF threshold stated
# in terms of a metric column the harness already collects. Append-only on
# extension of NORM_VARIANTS — never delete a variant's entry; mark N/A via
# ``applicable_experiments=()`` if a claim no longer holds.
Verdict = Literal["CONFIRMED", "REJECTED", "INCONCLUSIVE", "N/A"]


@dataclass(frozen=True)
class HypothesisSpec:
    """A single falsifiable claim about one norm variant.

    :param claim: One-sentence English statement of the layer's design claim.
    :param metric_column: Name of the column on the aggregated ``all_runs.csv``
        (or the probe-snapshot frame merged into it) that the threshold tests.
    :param comparator: How to interpret ``threshold`` against the observed
        ``metric_column`` value:
          - ``"<="`` REJECTED if observed > threshold (claim says: stay below).
          - ``">="`` REJECTED if observed < threshold (claim says: stay above).
          - ``"<"`` / ``">"`` strict variants.
    :param threshold: Numerical threshold for STOP-IF.
    :param applicable_experiments: Tuple of experiment ids (e.g. ``("e1",)``)
        on which this hypothesis is testable. Empty tuple => N/A everywhere.
    :param applicable_modes: Tuple of mode strings (``"oob"``,
        ``"param_matched"``). Empty => all modes.
    :param min_samples: Minimum non-NaN seed count to yield CONFIRMED /
        REJECTED. Below this, verdict is INCONCLUSIVE.
    :param reduction: Pre-comparator aggregator over the per-seed values within
        a single (experiment, norm_type, mode) cell. One of ``"mean"``,
        ``"median"``, ``"max"``, ``"min"``.
    :param notes: Free-text provenance (which docstring / paper section the
        claim comes from).
    """
    claim: str
    metric_column: str
    comparator: Literal["<=", "<", ">=", ">"]
    threshold: float
    applicable_experiments: Tuple[str, ...] = ()
    applicable_modes: Tuple[str, ...] = ()
    min_samples: int = 2
    reduction: Literal["mean", "median", "max", "min"] = "mean"
    notes: str = ""


# ----------------------------------------------------------------------
# The registry — one HypothesisSpec per variant in NORM_VARIANTS.
# ----------------------------------------------------------------------
#
# Notes on metric columns referenced:
#   - ``best_val_acc`` / ``final_val_loss`` — headline metrics on
#     ``all_runs.csv`` (cf. ``report.py:HEADLINE_METRIC``).
#   - ``act_per_sample_rms_max_max`` — the max over layers of the per-sample
#     RMS upper envelope at the final epoch (cf. ``report.py``
#     ``_final_probe_snapshot``).
#   - ``act_mean_abs`` — absolute value of activation mean averaged over
#     layers, final epoch (cf. same).
#   - ``grad_norm_global`` — global grad-norm at the final epoch (cf. same).
#   - ``generalization_gap`` — added in plan_e1f12eab Step 3.
#
# All thresholds are derived from the layer's design claim, not from observed
# Phase 1 data (cf. plan_e1f12eab pre-mortem Scenario A).
VARIANT_HYPOTHESES: Dict[str, HypothesisSpec] = {
    "rms_norm": HypothesisSpec(
        claim=(
            "RMSNorm is the baseline; its grad-norm at the final epoch stays "
            "within a 2-decade window typical of well-conditioned training."
        ),
        metric_column="grad_norm_global",
        comparator="<=",
        # Order of magnitude check: grad-norm > 1e2 at final epoch is a
        # well-documented divergence signal across CV / NLP architectures.
        threshold=1.0e2,
        applicable_experiments=("e1", "e2", "e3", "e4", "e5"),
        applicable_modes=(),  # both
        notes="Baseline sanity hypothesis — grad-norm stays bounded.",
    ),
    "band_rms": HypothesisSpec(
        claim=(
            "BandRMS constrains per-sample RMS to the band [1-alpha, 1] "
            "(alpha=0.1 default), so the max per-sample RMS at any layer at "
            "the final epoch should not exceed 1 by more than a small "
            "numerical-stability margin."
        ),
        metric_column="act_per_sample_rms_max_max",
        comparator="<=",
        # The band's upper bound is 1.0 exactly. Allow 5% slack for epsilon /
        # boundary smoothing in the soft-projection.
        threshold=1.05,
        applicable_experiments=("e1", "e2", "e3", "e5"),
        applicable_modes=(),
        notes=(
            "From band_rms.py docstring — band is [1-max_band_width, 1]; "
            "max_band_width=0.1 by default."
        ),
    ),
    "zero_centered_rms_norm": HypothesisSpec(
        claim=(
            "ZeroCenteredRMSNorm subtracts the mean before RMS normalization, "
            "so the absolute activation mean at the final epoch should be "
            "near zero (<= 0.05) at every layer."
        ),
        metric_column="act_mean_abs",
        comparator="<=",
        threshold=0.05,
        applicable_experiments=("e1", "e2", "e3", "e5"),
        applicable_modes=(),
        notes=(
            "Zero-centering claim — direct consequence of subtracting the "
            "mean. Threshold 0.05 = generous tolerance for floating-point "
            "noise + small remaining bias from the affine scale."
        ),
    ),
    "zero_centered_band_rms_norm": HypothesisSpec(
        claim=(
            "ZeroCenteredBandRMSNorm combines zero-centering with a band "
            "constraint; the per-sample RMS upper bound (1+slack) holds."
        ),
        metric_column="act_per_sample_rms_max_max",
        comparator="<=",
        threshold=1.05,
        applicable_experiments=("e1", "e2", "e3", "e5"),
        applicable_modes=(),
        notes=(
            "Same band-bound claim as band_rms (the band-projection layer "
            "is mechanically the same; only the pre-normalization differs)."
        ),
    ),
    "adaptive_band_rms": HypothesisSpec(
        claim=(
            "AdaptiveBandRMS adapts the band width per-sample but still keeps "
            "per-sample RMS bounded by 1 plus epsilon at the final epoch."
        ),
        metric_column="act_per_sample_rms_max_max",
        comparator="<=",
        # Adaptive band allows a wider band per-sample but the upper-envelope
        # claim still holds (band's [1-alpha_eff, 1]).
        threshold=1.05,
        applicable_experiments=("e1", "e2", "e3", "e5"),
        applicable_modes=(),
        notes="Same upper-bound claim as band_rms; band width is per-sample.",
    ),
    "band_logit_norm": HypothesisSpec(
        claim=(
            "BandLogitNorm is designed to reduce overfitting on classification "
            "logits — its generalization gap (train_acc - val_acc) should be "
            "no worse than rms_norm on classification tasks (E1, E3)."
        ),
        # generalization_gap added in Step 3 of plan_e1f12eab.
        metric_column="generalization_gap",
        comparator="<=",
        # Threshold: generalization gap should be no more than 0.20 in absolute
        # value at final epoch (well-trained CIFAR-10 ViT typically reaches
        # train-val gap < 0.10 with proper regularization).
        threshold=0.20,
        applicable_experiments=("e1", "e3"),
        applicable_modes=(),
        notes=(
            "Off-label on E2/E4 per VARIANT_CRITERIA off_label_contexts "
            "(D-002 of plan_74a935a2). Calibration-style claim — checking the "
            "generalization gap as a proxy until ECE is plumbed."
        ),
    ),
    "dynamic_tanh": HypothesisSpec(
        claim=(
            "DynamicTanh replaces normalization with a learnable tanh; like "
            "the baseline, its grad-norm at the final epoch should stay "
            "bounded (no divergence)."
        ),
        metric_column="grad_norm_global",
        comparator="<=",
        threshold=1.0e2,
        applicable_experiments=("e1", "e2", "e3", "e4", "e5"),
        applicable_modes=(),
        notes="DyT does not normalize by RMS — only the grad-norm sanity claim applies.",
    ),
    "zero_centered_adaptive_band_rms_norm": HypothesisSpec(
        claim=(
            "ZeroCenteredAdaptiveBandRMSNorm combines zero-mean output with "
            "an adaptive band; both the act_mean_abs <= 0.05 and the "
            "per-sample RMS <= 1.05 should hold. This registry checks the "
            "band-bound claim; the act_mean_abs claim is checked structurally "
            "via the variant's design (zero_centered prefix)."
        ),
        metric_column="act_per_sample_rms_max_max",
        comparator="<=",
        threshold=1.05,
        applicable_experiments=("e1", "e2", "e3", "e5"),
        applicable_modes=(),
        notes="8th variant — composite claim, primary band-bound check here.",
    ),
}


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------

def _apply_reduction(values: pd.Series, reduction: str) -> Optional[float]:
    """Aggregate a Series of per-seed values into a scalar; return None if no
    finite values remain.
    """
    finite = values.dropna()
    if finite.empty:
        return None
    if reduction == "mean":
        return float(finite.mean())
    if reduction == "median":
        return float(finite.median())
    if reduction == "max":
        return float(finite.max())
    if reduction == "min":
        return float(finite.min())
    raise ValueError(f"Unknown reduction: {reduction}")


def _compare(observed: float, threshold: float, comparator: str) -> bool:
    """Return True iff the observation **satisfies** the claim (i.e. the
    STOP-IF threshold is NOT crossed)."""
    if comparator == "<=":
        return observed <= threshold
    if comparator == "<":
        return observed < threshold
    if comparator == ">=":
        return observed >= threshold
    if comparator == ">":
        return observed > threshold
    raise ValueError(f"Unknown comparator: {comparator}")


def evaluate_hypothesis(
    variant: str,
    df: pd.DataFrame,
    *,
    experiment: Optional[str] = None,
    mode: Optional[str] = None,
) -> Verdict:
    """Evaluate the registered hypothesis for ``variant`` against ``df``.

    :param variant: One of the keys in :data:`VARIANT_HYPOTHESES`.
    :param df: A pandas DataFrame containing per-seed rows for a single cell
        — typically the subset of ``all_runs.csv`` filtered to one
        ``(experiment, norm_type, mode)`` cell. Must contain the
        ``metric_column`` referenced by the variant's :class:`HypothesisSpec`
        (else returns INCONCLUSIVE).
    :param experiment: Optional experiment id used to gate via
        :attr:`HypothesisSpec.applicable_experiments`. If the variant's
        hypothesis is not applicable to this experiment, returns ``N/A``.
    :param mode: Optional mode string used to gate via
        :attr:`HypothesisSpec.applicable_modes`. Empty tuple in the spec
        means "all modes apply".
    :returns: One of the four :data:`Verdict` literals.

    Examples:
        >>> df = pd.DataFrame({"grad_norm_global": [0.5, 0.6, 0.7]})
        >>> evaluate_hypothesis("rms_norm", df, experiment="e1")
        'CONFIRMED'
    """
    spec = VARIANT_HYPOTHESES.get(variant)
    if spec is None:
        logger.warning(
            "evaluate_hypothesis: no hypothesis registered for variant=%s",
            variant,
        )
        return "N/A"

    # Applicability gates.
    if spec.applicable_experiments and experiment is not None:
        if experiment not in spec.applicable_experiments:
            return "N/A"
    if spec.applicable_modes and mode is not None:
        if mode not in spec.applicable_modes:
            return "N/A"

    # Empty applicable_experiments => N/A everywhere (claim retired).
    if not spec.applicable_experiments:
        return "N/A"

    # Metric availability.
    if spec.metric_column not in df.columns:
        logger.debug(
            "evaluate_hypothesis: variant=%s missing column=%s in df "
            "(columns=%s) -> INCONCLUSIVE",
            variant, spec.metric_column, list(df.columns),
        )
        return "INCONCLUSIVE"

    values = df[spec.metric_column]
    finite = values.dropna()
    if len(finite) < spec.min_samples:
        return "INCONCLUSIVE"

    observed = _apply_reduction(values, spec.reduction)
    if observed is None:
        return "INCONCLUSIVE"

    return "CONFIRMED" if _compare(observed, spec.threshold, spec.comparator) else "REJECTED"


def evaluate_all(
    df: pd.DataFrame,
    *,
    variant_col: str = "norm_type",
    experiment_col: str = "experiment",
    mode_col: str = "mode",
) -> pd.DataFrame:
    """Evaluate hypotheses for every ``(experiment, variant, mode)`` cell in
    ``df``.

    Returns a frame with columns ``[experiment, norm_type, mode,
    hypothesis_verdict, hypothesis_metric, hypothesis_threshold,
    hypothesis_observed]``. Rows where the variant has no registered
    hypothesis are emitted with verdict ``N/A`` and NaN observed value.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            experiment_col, variant_col, mode_col,
            "hypothesis_verdict",
            "hypothesis_metric",
            "hypothesis_threshold",
            "hypothesis_observed",
        ])
    out: List[Dict[str, Any]] = []
    keys = [experiment_col, variant_col, mode_col]
    for key_tuple, sub in df.groupby(keys, sort=False):
        exp, variant, mode = key_tuple
        spec = VARIANT_HYPOTHESES.get(variant)
        verdict = evaluate_hypothesis(
            variant, sub, experiment=exp, mode=mode,
        )
        observed: Optional[float]
        if spec is not None and spec.metric_column in sub.columns:
            observed = _apply_reduction(sub[spec.metric_column], spec.reduction)
        else:
            observed = None
        out.append({
            experiment_col: exp,
            variant_col: variant,
            mode_col: mode,
            "hypothesis_verdict": verdict,
            "hypothesis_metric": spec.metric_column if spec else "",
            "hypothesis_threshold": spec.threshold if spec else float("nan"),
            "hypothesis_observed": observed if observed is not None else float("nan"),
        })
    return pd.DataFrame(out)


# Self-check at import time: registry covers all NORM_VARIANTS.
def _self_check() -> None:
    missing = [v for v in NORM_VARIANTS if v not in VARIANT_HYPOTHESES]
    if missing:
        raise RuntimeError(
            f"VARIANT_HYPOTHESES missing entries for NORM_VARIANTS: {missing}"
        )
    extra = [v for v in VARIANT_HYPOTHESES if v not in NORM_VARIANTS]
    if extra:
        logger.warning(
            "VARIANT_HYPOTHESES has entries not in NORM_VARIANTS: %s "
            "(stale claims — consider removing)", extra,
        )


_self_check()


__all__ = [
    "VARIANT_HYPOTHESES",
    "HypothesisSpec",
    "Verdict",
    "evaluate_hypothesis",
    "evaluate_all",
]
