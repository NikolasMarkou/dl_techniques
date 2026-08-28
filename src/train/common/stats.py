"""Pure-function statistics utilities for multi-seed / multi-run sweeps.

These helpers are deliberately small, NaN-tolerant, and degenerate-case-safe.
Every public function is deterministic given a fixed `numpy.random.Generator`.

This module was previously forked in two places -- ``train.logic
.multiseed_stats`` and ``train.rms_variants_train.stats`` -- whose function
bodies were character-identical (the latter's own docstring described itself as
"byte-equivalent" to the former). Both forks now import from here.

Public surface
--------------
- ``mean_std(values, ddof=1)``                  → ``(mean, std)``
- ``bootstrap_ci(values, ...)``                 → ``(ci_low, ci_high)``
- ``paired_permutation_test(a, b, ...)``        → ``(observed_mean_diff, p_value)``
- ``format_mean_std(mean, std, decimals=4)``    → ``"0.7006 ± 0.0123"``
- ``holm_bonferroni(p_values, ...)``            → ``(rejected, p_adjusted)``
- ``benjamini_hochberg(p_values, ...)``         → ``(rejected, p_adjusted)``
- ``min_reachable_p_signflip(n_pairs)``         → smallest two-sided p attainable
- ``min_pairs_for_significance(family_size)``   → seeds needed to reject at all

Design notes
------------
- All inputs are 1-D ``np.ndarray`` (or array-like coerced via ``np.asarray``).
- NaN handling: ``mean_std`` uses ``np.nanmean`` / ``np.nanstd`` (so a missing
  attribution row in E3 doesn't poison the aggregate). Bootstrap and permutation
  resample only finite values.
- Zero-variance degenerate cases are explicitly handled:
    * ``bootstrap_ci`` on all-identical input → ``(value, value)``.
    * ``paired_permutation_test`` on all-zero paired diffs → ``(0.0, 1.0)``.
- All callers should pass an explicit ``rng = np.random.default_rng(SEED)`` to
  guarantee reproducibility of bootstrap CIs and permutation p-values.

Plan: ``plans/plan_2026-05-14_9c6387a3``  (multi-seed sweep, D-002, D-004).
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np

__all__ = [
    "mean_std",
    "bootstrap_ci",
    "paired_permutation_test",
    "format_mean_std",
    "holm_bonferroni",
    "benjamini_hochberg",
    "min_reachable_p_signflip",
    "min_pairs_for_significance",
]

ArrayLike = Union[np.ndarray, list, tuple]


# ---------------------------------------------------------------------------
# Mean / std
# ---------------------------------------------------------------------------

def mean_std(values: ArrayLike, ddof: int = 1) -> Tuple[float, float]:
    """Return (mean, std) of ``values``, NaN-tolerant, with sample-std default.

    Parameters
    ----------
    values : array_like
        1-D collection of floats. NaNs are skipped.
    ddof : int, default 1
        Delta degrees of freedom for the standard deviation. ``ddof=1`` is the
        unbiased sample estimator (Bessel correction); use ``ddof=0`` for the
        population estimator.

    Returns
    -------
    (mean, std) : tuple of float
        If all values are NaN or the array is empty, returns ``(nan, nan)``.
        If only one finite value remains, ``std`` is ``0.0`` (not NaN) so
        downstream formatting does not break.
    """
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1:
        return float(finite[0]), 0.0
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=ddof))
    return mean, std


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: ArrayLike,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean.

    Parameters
    ----------
    values : array_like
        1-D sample. NaNs are dropped before resampling.
    confidence : float, default 0.95
        Two-sided coverage probability.
    n_boot : int, default 2000
        Number of bootstrap resamples.
    rng : numpy.random.Generator
        Required — caller owns reproducibility.

    Returns
    -------
    (ci_low, ci_high) : tuple of float
        Percentile-bootstrap CI of the sample mean.

        Degenerate cases:
          * Empty / all-NaN input → ``(nan, nan)``.
          * Single finite value → ``(value, value)``.
          * All-identical finite values → ``(value, value)`` (CI width 0).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")

    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1 or np.all(finite == finite[0]):
        v = float(finite[0])
        return v, v

    # Vectorized bootstrap: draw (n_boot, n) integer indices in one call.
    n = finite.size
    idx = rng.integers(low=0, high=n, size=(n_boot, n))
    means = finite[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------

def paired_permutation_test(
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_perm: int = 10000,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Two-sided paired sign-flip permutation test on the mean difference.

    Tests H0: mean(a - b) == 0 against H1: mean(a - b) != 0.

    Parameters
    ----------
    a, b : array_like
        Paired 1-D arrays of equal length. NaN pairs (NaN in either) dropped.
    n_perm : int, default 10000
        Number of sign-flip permutations to sample (Monte Carlo).
    rng : numpy.random.Generator
        Required — caller owns reproducibility.

    Returns
    -------
    (observed_mean_diff, p_value) : tuple of float
        ``observed_mean_diff = mean(a - b)`` over the retained pairs.
        ``p_value`` is the two-sided proportion of permutations whose absolute
        mean-difference equals or exceeds the observed absolute mean-difference
        (with +1 numerator and denominator add-one correction per
        Phipson & Smyth 2010 to prevent p=0).

        Degenerate cases:
          * No retained pairs → ``(nan, nan)``.
          * All paired diffs zero → ``(0.0, 1.0)``.
    """
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1; got {n_perm}")

    arr_a = np.asarray(a, dtype=float).ravel()
    arr_b = np.asarray(b, dtype=float).ravel()
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"a and b must have the same shape; got {arr_a.shape} vs {arr_b.shape}"
        )

    finite_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    diffs = (arr_a - arr_b)[finite_mask]
    if diffs.size == 0:
        return float("nan"), float("nan")

    observed = float(diffs.mean())
    if np.all(diffs == 0.0):
        return 0.0, 1.0

    # Sign-flip: each permutation is a sign vector in {-1, +1}^n.
    n = diffs.size
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    abs_obs = abs(observed)
    n_extreme = int(np.sum(np.abs(perm_means) >= abs_obs))
    # Phipson & Smyth (2010) add-one correction: avoids p=0 in finite Monte
    # Carlo enumeration.
    p = (n_extreme + 1) / (n_perm + 1)
    return observed, float(p)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Render ``mean ± std`` to a fixed-decimals string.

    NaN inputs are rendered as the literal string ``"nan"`` to keep tables
    aligned.
    """
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0; got {decimals}")
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan ± nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------
#
# A sweep that reports M metrics across A arms performs M*A tests. At
# alpha = 0.05 uncorrected, ~1 in 20 of those is expected to look "significant"
# on noise alone, which for a study whose entire output is significance claims
# is not acceptable. Neither correction below existed here before; the only
# prior treatment in the repository is prose in
# ``research/2026_correlations.md``.
#
# Both functions share one contract:
#   * input and output are in the SAME order (never sorted in place);
#   * NaN entries are EXCLUDED from the family size m and returned as NaN /
#     False -- a comparison that could not be computed must not inflate the
#     penalty applied to the ones that could;
#   * adjusted p-values are made monotone, so a smaller raw p never yields a
#     larger adjusted p;
#   * at m == 1 both are the identity.


def _prepare_p_values(p_values: ArrayLike) -> Tuple[np.ndarray, np.ndarray, int]:
    """Coerce, validate and locate the finite entries of a p-value vector.

    :param p_values: Raw p-values; NaN marks a comparison that could not be run.
    :return: ``(flat_array, finite_mask, m)`` where ``m`` is the family size,
        counting only finite entries.
    :raises ValueError: If any finite entry falls outside ``[0, 1]``.
    """
    arr = np.asarray(p_values, dtype=float).ravel()
    finite = np.isfinite(arr)
    if np.any((arr[finite] < 0.0) | (arr[finite] > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    return arr, finite, int(finite.sum())


def holm_bonferroni(
    p_values: ArrayLike,
    *,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Holm step-down correction, controlling the family-wise error rate.

    Sorts ascending and compares ``p_(i)`` against ``alpha / (m - i)``, stopping
    at the first failure. Adjusted values are ``(m - i) * p_(i)`` made monotone
    by a running maximum and clipped to 1.

    Prefer this over :func:`benjamini_hochberg` for a small family backing a
    single categorical claim ("arm X beats the baseline"), where the family-wise
    rate is the error you care about, and because Holm is valid under arbitrary
    dependence between the tests while BH needs independence or PRDS.

    :param p_values: Raw p-values; NaN entries are excluded from the family.
    :param alpha: Family-wise error rate.
    :return: ``(rejected, p_adjusted)``, both in input order.
    :raises ValueError: If ``alpha`` is outside ``(0, 1)`` or a p-value is
        outside ``[0, 1]``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    arr, finite, m = _prepare_p_values(p_values)
    rejected = np.zeros(arr.shape, dtype=bool)
    adjusted = np.full(arr.shape, np.nan, dtype=float)
    if m == 0:
        return rejected, adjusted

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(arr[idx], kind="stable")]
    scaled = (m - np.arange(m)) * arr[order]
    # Running maximum: a later (larger raw p) entry can never be reported as
    # more significant than an earlier one.
    monotone = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted[order] = monotone
    rejected[order] = monotone <= alpha
    return rejected, adjusted


def benjamini_hochberg(
    p_values: ArrayLike,
    *,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up correction, controlling the false discovery rate.

    Adjusted values are ``m / (i + 1) * p_(i)`` made monotone by a running
    minimum taken from the largest p-value downward, then clipped to 1.

    Prefer this over :func:`holm_bonferroni` for an exploratory family where the
    tolerable error is a *proportion* of false discoveries rather than any
    false discovery at all, and where the tests are positively dependent --
    which is BH's PRDS validity case.

    :param p_values: Raw p-values; NaN entries are excluded from the family.
    :param alpha: False-discovery rate.
    :return: ``(rejected, p_adjusted)``, both in input order.
    :raises ValueError: If ``alpha`` is outside ``(0, 1)`` or a p-value is
        outside ``[0, 1]``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    arr, finite, m = _prepare_p_values(p_values)
    rejected = np.zeros(arr.shape, dtype=bool)
    adjusted = np.full(arr.shape, np.nan, dtype=float)
    if m == 0:
        return rejected, adjusted

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(arr[idx], kind="stable")]
    ranks = np.arange(1, m + 1)
    scaled = arr[order] * m / ranks
    # Running minimum from the largest p downward keeps the sequence monotone.
    monotone = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted[order] = monotone
    rejected[order] = monotone <= alpha
    return rejected, adjusted


def min_reachable_p_signflip(n_pairs: int) -> float:
    """Smallest two-sided p-value a sign-flip permutation test can produce.

    A paired sign-flip test over ``n`` pairs enumerates ``2**n`` sign vectors.
    The observed assignment and its mirror are always at least as extreme as
    the observation, so the smallest attainable two-sided p is ``2 / 2**n``.

    This is why :func:`paired_permutation_test` cannot report significance below
    a certain seed count *for any effect size* -- a property of the test, not of
    the data.

    :param n_pairs: Number of paired observations (seeds).
    :return: ``2 ** (1 - n_pairs)``; 1.0 for ``n_pairs <= 1``.
    :raises ValueError: If ``n_pairs`` is negative.
    """
    if n_pairs < 0:
        raise ValueError(f"n_pairs must be non-negative; got {n_pairs}")
    if n_pairs <= 1:
        return 1.0
    return float(2.0 ** (1 - n_pairs))


def min_pairs_for_significance(family_size: int = 1, *, alpha: float = 0.05) -> int:
    """Fewest paired observations at which a corrected sign-flip test can reject.

    Both Holm and BH compare the smallest p-value in a family of size ``m``
    against ``alpha / m``, so rejection requires
    ``2 ** (1 - n) <= alpha / m``, i.e. ``n >= 1 + log2(m / alpha)``.

    MEASURED against this module's own :func:`paired_permutation_test` on
    maximally separated inputs: family size 1 needs 6 pairs, 3 needs 7, 18
    needs 10, 21 needs 10, 63 needs 12. Correcting a larger family is therefore
    not free -- it is paid for in seeds, and the choice of family is a choice
    about how much compute the study costs.

    :param family_size: Number of tests corrected together.
    :param alpha: Significance level before correction.
    :return: Minimum number of pairs.
    :raises ValueError: If ``family_size`` is below 1 or ``alpha`` is outside
        ``(0, 1)``.
    """
    if family_size < 1:
        raise ValueError(f"family_size must be >= 1; got {family_size}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    return int(math.ceil(1.0 + math.log2(family_size / alpha)))
