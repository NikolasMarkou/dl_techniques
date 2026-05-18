"""Pure-function statistics utilities for the rms_variants_train sweep.

Per plan_2026-05-18_6776f8ba (Refinement E, step 1): replace the prior thin
re-export of ``train.logic.multiseed_stats`` with a self-contained pure-function
module owned by ``rms_variants_train``. This decouples the rms_variants_train
report pipeline from the unrelated multi-seed sweep logic and gives this
package an explicit, fully-tested statistics surface.

Behavioural contract is byte-equivalent to ``train.logic.multiseed_stats`` so
existing wiring tests (``test_stats_wiring.py``) continue to pass unchanged.

Public surface
--------------
- ``mean_std(values, ddof=1)``                  -> ``(mean, std)``
- ``bootstrap_ci(values, *, confidence, n_boot, rng)`` -> ``(ci_low, ci_high)``
- ``paired_permutation_test(a, b, *, n_perm, rng)`` -> ``(observed_mean_diff, p_value)``
- ``format_mean_std(mean, std, decimals=4)``    -> ``"mean ± std"``

Design notes
------------
- All inputs are 1-D ``np.ndarray`` (or array-like coerced via ``np.asarray``).
- NaN handling: ``mean_std`` uses ``np.nanmean``/``np.nanstd`` (so a missing
  attribution row does not poison the aggregate). Bootstrap and permutation
  resample only finite values.
- Zero-variance degenerate cases are explicitly handled:
    * ``bootstrap_ci`` on all-identical input -> ``(value, value)``.
    * ``paired_permutation_test`` on all-zero paired diffs -> ``(0.0, 1.0)``.
- All callers pass an explicit ``rng = np.random.default_rng(SEED)`` to
  guarantee reproducibility of bootstrap CIs and permutation p-values.

Plan: ``plans/plan_2026-05-18_6776f8ba`` (RMS-Variants Phase 3 refinement E).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]

__all__ = [
    "mean_std",
    "bootstrap_ci",
    "paired_permutation_test",
    "format_mean_std",
]


# ---------------------------------------------------------------------------
# Mean / std
# ---------------------------------------------------------------------------

def mean_std(values: ArrayLike, ddof: int = 1) -> Tuple[float, float]:
    """Return ``(mean, std)`` of ``values``, NaN-tolerant, sample-std default.

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
        - All-NaN or empty -> ``(nan, nan)``.
        - Single finite value -> ``(value, 0.0)``.
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
        Two-sided coverage probability. Must be in ``(0, 1)``.
    n_boot : int, default 2000
        Number of bootstrap resamples. Must be ``>= 1``.
    rng : numpy.random.Generator
        Required. Caller owns reproducibility.

    Returns
    -------
    (ci_low, ci_high) : tuple of float
        Degenerate cases:
          - Empty/all-NaN -> ``(nan, nan)``.
          - Single finite value -> ``(value, value)``.
          - All-identical finite values -> ``(value, value)`` (CI width 0).
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
        Paired 1-D arrays of equal length. Pairs containing NaN in either
        side are dropped.
    n_perm : int, default 10000
        Monte Carlo sign-flip sample count. Must be ``>= 1``.
    rng : numpy.random.Generator
        Required. Caller owns reproducibility.

    Returns
    -------
    (observed_mean_diff, p_value) : tuple of float
        ``observed_mean_diff = mean(a - b)`` over the retained pairs. ``p_value``
        is the two-sided proportion of permutations whose absolute mean
        difference equals or exceeds the observed absolute mean difference,
        using the Phipson & Smyth (2010) add-one correction:
        ``p = (n_extreme + 1) / (n_perm + 1)``.

        Degenerate cases:
          - No retained pairs -> ``(nan, nan)``.
          - All paired diffs zero -> ``(0.0, 1.0)``.

    Raises
    ------
    ValueError
        If ``n_perm < 1`` or ``a.shape != b.shape``.
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

    n = diffs.size
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    abs_obs = abs(observed)
    n_extreme = int(np.sum(np.abs(perm_means) >= abs_obs))
    p = (n_extreme + 1) / (n_perm + 1)
    return observed, float(p)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Render ``mean ± std`` to a fixed-decimals string.

    NaN inputs render as ``"nan ± nan"`` to keep aligned table output.
    """
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0; got {decimals}")
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan ± nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
