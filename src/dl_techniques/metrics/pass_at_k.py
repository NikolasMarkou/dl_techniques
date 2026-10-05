"""Sampling coverage metrics: ``pass@k`` over a pool of candidate solutions.

**Why this module exists.** Every method that samples several candidate solutions per
input and checks them against a verifier has one obvious summary statistic, and it
is not the mean success rate. Mean success rate answers "how good is my first
guess"; ``pass@k`` answers "does a correct solution exist in my candidate pool at
all". The second question is the one that matters when the candidate pool feeds a
selection, reranking, verification or best-of-k step -- and it is the quantity the
Spectrum phase of the Spectrum-to-Signal Principle scores checkpoints by (see
``dl_techniques.optimization.ssp``, which imports this module rather than
duplicating it).

**Plain functions on numpy arrays, not ``keras.metrics.Metric``.** Deliberate, and
structural rather than a shortcut: every quantity here is a functional of the whole
``(n_problems, n_samples)`` outcome matrix. A streaming ``update_state`` sees one
sample at a time and cannot know how many of a problem's samples were correct, which
is the entire input to the estimator. The package already carries plain functions
for the same reason (all of ``embedding_quality``, ``llm_metrics.self_bleu``,
``perplexity_metric.perplexity``).

Conventions of the outcome matrix
---------------------------------
``outcomes`` is ``(n_problems, n_samples)``, row-major by problem:

* ``outcomes[i, j] > 0`` marks sample ``j`` of problem ``i`` as **correct**;
* ``outcomes[i, j] == 0`` marks it as incorrect;
* on a continuous reward in ``[0, 1]``, ``> 0`` means "counts as solved".

So a partial-credit reward can be binarised by the caller (``outcomes > 0.5``)
before calling, and everything below is unchanged. Fractional values strictly
between 0 and 1 are accepted and treated as correct -- see
:func:`solved_counts` for the exact rule and why it is stated rather than assumed.

The two estimators
------------------
``estimator="unbiased"`` (the default) is the standard estimator of Chen et al.
(2021), "Evaluating Large Language Models Trained on Code", section 2.1: drawing
``k`` samples without replacement from the ``n`` observed,

.. math::

    \\widehat{\\mathrm{pass@}k}_i
        = 1 - \\frac{\\binom{n - c_i}{k}}{\\binom{n}{k}}

where ``c_i`` is the number of correct samples for problem ``i``. It is unbiased for
pass@k over the sampling distribution, and it is the number to quote.

``estimator="plug_in"`` is the naive plug-in: the fraction of problems for which
*any* of the ``n`` observed samples was correct, i.e. exactly ``pass@n``. It is
what most quick scripts print, it never under-reports, and for ``k < n`` it is an
**upper bound** on the unbiased value. Report one estimator, not both -- they
answer slightly different questions and quoting both invites a false comparison.

References:
    - Chen, M. et al. (2021). Evaluating Large Language Models Trained on Code.
      (https://arxiv.org/abs/2107.03374)
    - Wortsman, T. et al. (2022). Model soups: averaging weights of multiple
      fine-tuned models improves accuracy without increasing inference cost.
      (the averaging counterpart of this module; see
      ``dl_techniques.optimization.ssp.fusion``)
    - Xu, S. et al. (2025). Tiny Model, Big Logic: Diversity-Driven Optimization
      Elicits Large-Model Reasoning Ability in VibeThinker-1.5B. (the Spectrum
      phase that scores checkpoints by pass@k rather than pass@1)
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: Supported ``estimator`` values for :func:`pass_at_k`.
PASS_AT_K_ESTIMATORS: Tuple[str, ...] = ("unbiased", "plug_in")

# DECISION <ssp-2026-10-05>/D-001
# The binomial ratio is evaluated through `gammaln`, never through `math.comb` or
# integer binomials. Two reasons, and the second is the load-bearing one:
#   (a) `C(n, k)` overflows float64 above n ~ 1300, so an exact-integer path is not
#       merely slower, it RAISES on a realistic eval budget;
#   (b) `gammaln` is vectorised over the whole outcome matrix, which is what makes
#       `pass_at_k_curve` a single call instead of an n_k-length Python loop that
#       recomputes identical per-problem quantities once per k.
# The cost is that differences of `gammaln` lose absolute precision as n grows, so
# the ratio is clamped into [0, 1] before `1 - ratio`. The clamp is a BOUND, not a
# fudge: the ratio is a probability and cannot leave that interval, so clamping only
# ever repairs round-off.
_RATIO_FLOOR: float = 0.0
_RATIO_CEIL: float = 1.0

# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


def _as_outcome_matrix(outcomes) -> np.ndarray:
    """Validate and normalise an ``(n_problems, n_samples)`` outcome matrix.

    Args:
        outcomes: Anything array-like coercible to a float array. Cast to
            ``float64`` so the ``gammaln`` path never runs in reduced precision.

    Returns:
        A ``(n_problems, n_samples)`` ``float64`` array.

    Raises:
        ValueError: If the array is not 2-D, or has a zero-length axis. A
            zero-length ``n_samples`` makes every ``pass@k`` undefined rather
            than zero, so it is refused here instead of propagating a NaN.
    """
    matrix = np.asarray(outcomes, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"outcomes must be 2-D (n_problems, n_samples); got shape "
            f"{matrix.shape} with ndim={matrix.ndim}"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            f"outcomes must have non-zero extent on both axes; got shape "
            f"{matrix.shape}"
        )
    return matrix


def _validate_k(k: int, n_samples: int) -> int:
    """Validate a ``k`` against the number of observed samples.

    Args:
        k: Number of samples drawn without replacement. Must be an integer in
            ``[1, n_samples]``.
        n_samples: Observed samples per problem.

    Returns:
        ``k`` as an ``int``.

    Raises:
        ValueError: If ``k`` is not an integer, or lies outside ``[1, n_samples]``.
            ``k > n_samples`` is refused rather than clamped: the unbiased
            estimator's ratio is then not defined from the observed data, and
            clamping would hand back a confident-looking number with no data behind
            it.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer; got {k!r} of type {type(k).__name__}")
    k_int = int(k)
    if k_int < 1:
        raise ValueError(f"k must be >= 1; got {k_int}")
    if k_int > n_samples:
        raise ValueError(
            f"k must be <= n_samples={n_samples}; got {k_int}. Clamping instead "
            f"would return 1.0 for every problem, because C(n - c, k) is then zero "
            f"for every problem and the ratio carries no information."
        )
    return k_int


def _check_estimator(estimator: str) -> str:
    """Normalise and validate the ``estimator`` name.

    Args:
        estimator: ``"unbiased"`` or ``"plug_in"``, any case, surrounding
            whitespace tolerated.

    Returns:
        The lower-cased estimator name.

    Raises:
        ValueError: If the name is not a string, or not one of
            :data:`PASS_AT_K_ESTIMATORS`.
    """
    if not isinstance(estimator, str):
        raise ValueError(
            f"estimator must be a string; got {type(estimator).__name__}"
        )
    name = estimator.strip().lower()
    if name not in PASS_AT_K_ESTIMATORS:
        raise ValueError(
            f"Unknown estimator {estimator!r}. "
            f"Supported: {list(PASS_AT_K_ESTIMATORS)}"
        )
    return name


# ---------------------------------------------------------------------
# Core quantities
# ---------------------------------------------------------------------


def solved_counts(outcomes) -> np.ndarray:
    """Per-problem count of correct samples.

    The rule is ``> 0``, stated explicitly because a fractional reward is ambiguous
    otherwise: a rubric score of ``0.4`` counts as SOLVED here, not as a partial
    credit. That is the right reading for a coverage metric -- pass@k asks whether a
    usable answer exists in the pool, and whether ``0.4`` is usable is the caller's
    decision, made by thresholding *before* calling. Binarise upstream
    (``outcomes > 0.5``) if a threshold is wanted.

    Args:
        outcomes: ``(n_problems, n_samples)`` array; see the module docstring.

    Returns:
        ``(n_problems,)`` ``int64`` array of counts, each in ``[0, n_samples]``.

    Raises:
        ValueError: If ``outcomes`` is not a non-empty 2-D array.
    """
    matrix = _as_outcome_matrix(outcomes)
    return np.count_nonzero(matrix > 0.0, axis=1).astype(np.int64)


def _unbiased_pass_at_k(
        n_samples: int,
        counts: np.ndarray,
        ks: np.ndarray,
) -> np.ndarray:
    """Vectorised ``1 - C(n - c, k) / C(n, k)`` over a whole ``k`` ladder.

    The ratio is

    .. math::

        \\frac{C(n - c, k)}{C(n, k)}
            = \\exp\\big(\\mathrm{lgamma}(n - c + 1)
                        - \\mathrm{lgamma}(n - c - k + 1)
                        - \\mathrm{lgamma}(n + 1)
                        + \\mathrm{lgamma}(n - k + 1)\\big)

    Args:
        n_samples: Observed samples per problem, ``n``.
        counts: ``(n_problems,)`` correct-sample counts, ``c``.
        ks: ``(n_k,)`` positive ``k`` values, each already validated ``<= n_samples``.

    Returns:
        ``(n_problems, n_k)`` ``float64`` array of per-problem pass@k estimates.

    Notes:
        One entry class makes the expression ``nan`` rather than a number, and it
        is set to its exact value instead: when ``n - c < k`` the falling factorial
        runs past zero, so ``C(n - c, k)`` is zero, the ratio is ``0`` and pass@k is
        exactly ``1.0`` -- every ``k``-subset of the observed samples then contains
        a correct one. That covers both ``c == n`` at any ``k`` and
        ``k > n - c`` at any ``c``; letting it reach ``gammaln`` instead produced
        ``nan`` on the fully-solved case, which is the single most common row in an
        eval matrix.

        The boundary is ``< 0`` and NOT ``<= 0``. At ``n - c == k`` exactly, every
        ``k``-subset IS the full set of incorrect samples, so ``C(n - c, k) = 1``,
        the ratio is ``1 / C(n, k) > 0`` and pass@k is strictly below 1. Widening the
        test to ``<= 0`` therefore reported the never-solved ``(n=4, c=0, k=4)`` row
        as ``1.0`` -- a model that solved nothing, scored as perfect.
    """
    counts = counts.astype(np.float64)
    n = np.float64(n_samples)
    ks = ks.astype(np.float64)

    n_minus_c = (n - counts)[:, None]            # (P, 1)
    k_grid = ks[None, :]                         # (1, K)
    remainder = n_minus_c - k_grid               # (P, K)

    # C(n - c, k) == 0 wherever the falling factorial would run past zero.
    degenerate = remainder < 0.0
    safe_remainder = np.where(degenerate, 1.0, remainder)

    log_ratio = (
        gammaln(n_minus_c + 1.0)
        - gammaln(safe_remainder + 1.0)
        - gammaln(n + 1.0)
        + gammaln(n - k_grid + 1.0)
    )
    ratio = np.where(degenerate, 0.0, np.exp(log_ratio))
    ratio = np.clip(ratio, _RATIO_FLOOR, _RATIO_CEIL)
    return 1.0 - ratio


def pass_at_k(outcomes, k: int, estimator: str = "unbiased") -> float:
    """Probability that at least one of ``k`` drawn samples solves each problem.

    The pool-level coverage number. For each problem the estimator is computed from
    that problem's ``n`` observed samples, and the returned value is the mean over
    problems.

    Args:
        outcomes: ``(n_problems, n_samples)`` array of per-sample outcomes; see the
            module docstring for the layout and the ``> 0`` rule.
        k: Number of samples drawn without replacement. Must satisfy
            ``1 <= k <= n_samples``.
        estimator: ``"unbiased"`` (default; Chen et al. 2021) or ``"plug_in"``
            (the naive any-of-``n``, i.e. ``pass@n``).

    Returns:
        A float in ``[0.0, 1.0]``: the mean over problems of the per-problem
        estimate.

    Raises:
        ValueError: If ``outcomes`` is not a non-empty 2-D array, ``k`` is out of
            range, or ``estimator`` is unknown.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.metrics.pass_at_k import pass_at_k
        >>> # two problems, four samples each: problem 0 solved by 2, problem 1 by none
        >>> outcomes = np.array([[1, 0, 1, 0],
        ...                      [0, 0, 0, 0]], dtype=float)
        >>> float(np.round(pass_at_k(outcomes, k=1), 6))
        0.25
        >>> float(np.round(pass_at_k(outcomes, k=2), 6))
        0.416667

    Note:
        The ``k=2`` value is ``5/12``, not ``0.5``: for problem 0 the unbiased
        estimate is ``1 - C(2, 2)/C(4, 2) = 1 - 1/6 = 5/6``, not the plug-in
        ``2/4``. This gap is the whole reason the default estimator is the unbiased
        one, and it is why ``pass@k`` over a fixed pool is not the mean success rate
        times anything.
    """
    estimator_name = _check_estimator(estimator)
    matrix = _as_outcome_matrix(outcomes)
    _, n_samples = matrix.shape
    k_int = _validate_k(k, n_samples)

    counts = solved_counts(matrix)
    if estimator_name == "plug_in":
        per_problem = (counts > 0).astype(np.float64)
    else:
        per_problem = _unbiased_pass_at_k(
            n_samples, counts, np.asarray([k_int], dtype=np.int64)
        )[:, 0]
    return float(np.mean(per_problem))


def pass_at_k_curve(
        outcomes,
        ks: Optional[Sequence[int]] = None,
        estimator: str = "unbiased",
) -> Dict[int, float]:
    """Pool-level ``pass@k`` over a ladder of ``k`` values, from one matrix.

    Args:
        outcomes: ``(n_problems, n_samples)`` array; see the module docstring.
        ks: The ``k`` values to evaluate. Defaults to ``1, 2, 4, ..., n_samples``
            (powers of two, always including ``n_samples``). Every value must
            satisfy ``1 <= k <= n_samples``; the ladder is not silently clipped.
        estimator: ``"unbiased"`` (default) or ``"plug_in"``.

    Returns:
        A ``dict`` mapping each requested ``int`` ``k`` to its float pass@k, keys in
        ascending ``k`` order.

    Raises:
        ValueError: On any problem with :func:`pass_at_k`, plus if ``ks`` is empty.

    Note:
        One matrix, one vectorised ``gammaln`` call, every ``k``. Calling
        :func:`pass_at_k` in a loop over ``ks`` recomputes identical per-problem
        quantities ``len(ks)`` times; this returns the same numbers with the
        redundancy removed, and the test suite pins the two against each other.
    """
    estimator_name = _check_estimator(estimator)
    matrix = _as_outcome_matrix(outcomes)
    _, n_samples = matrix.shape

    if ks is None:
        powers = [2**e for e in range(0, int(np.floor(np.log2(n_samples))) + 1)]
        ks = sorted({int(k) for k in powers + [n_samples] if 1 <= k <= n_samples})
    else:
        ks = sorted({int(k) for k in ks})
        if not ks:
            raise ValueError("ks must not be empty")

    validated = [_validate_k(k, n_samples) for k in ks]
    counts = solved_counts(matrix)

    if estimator_name == "plug_in":
        per_problem = (counts > 0).astype(np.float64)
        return {k: float(np.mean(per_problem)) for k in validated}

    table = _unbiased_pass_at_k(
        n_samples, counts, np.asarray(validated, dtype=np.int64)
    )
    return {k: float(np.mean(table[:, i])) for i, k in enumerate(validated)}


def per_problem_pass_at_k(
        outcomes,
        k: int,
        estimator: str = "unbiased",
) -> np.ndarray:
    """Per-problem pass@k estimates, before averaging over problems.

    The averaged number hides which problems are wide and which are narrow, and the
    split is usually the interesting one: a high mean pass@k can be one very broad
    problem and a wall of unsolvable ones. This is the array to sort by when choosing
    what to train on.

    Args:
        outcomes: ``(n_problems, n_samples)`` array; see the module docstring.
        k: Number of samples drawn without replacement, ``1 <= k <= n_samples``.
        estimator: ``"unbiased"`` (default) or ``"plug_in"``.

    Returns:
        ``(n_problems,)`` ``float64`` array, each entry in ``[0.0, 1.0]``.

    Raises:
        ValueError: On any problem with :func:`pass_at_k`.
    """
    estimator_name = _check_estimator(estimator)
    matrix = _as_outcome_matrix(outcomes)
    _, n_samples = matrix.shape
    k_int = _validate_k(k, n_samples)

    counts = solved_counts(matrix)
    if estimator_name == "plug_in":
        return (counts > 0).astype(np.float64)
    return _unbiased_pass_at_k(
        n_samples, counts, np.asarray([k_int], dtype=np.int64)
    )[:, 0]
