"""The Spectrum phase: score a checkpoint by coverage, not by single-shot accuracy.

SSP's first claim is that the SFT stage should be selected by how BROAD a model's
solution repertoire is, not by how often its first guess is right. This module is
that selection, made independent of the paper's LLM framing.

Three pieces, in the order a pipeline uses them:

1. :func:`spectrum_profile` -- summarise one model's candidate pool. Wraps the
   ``pass@k`` estimators from :mod:`dl_techniques.metrics.pass_at_k` (imported, not
   duplicated) with the breadth diagnostics pass@k alone does not carry.
2. :func:`select_specialists` -- given a score matrix over checkpoints x subdomains,
   pick the diversity-maximising checkpoint per subdomain. This is the paper's
   "Domain-Aware Diversity Probing": a global argmax is wrong here, because the best
   overall checkpoint is frequently not the best in any particular subdomain.
3. :func:`spectrum_sampling_weights` -- turn per-example coverage into a sampling
   distribution over the training set. This is the Spectrum phase applied to DATA
   SELECTION instead of checkpoint selection, and it needs no RL machinery to be
   useful.

The unification worth knowing about
-----------------------------------
Piece 3 reuses :func:`dl_techniques.optimization.ssp.signal.max_entropy_weight`
verbatim. The argument is that "the training value of an item peaks where the model
is maximally uncertain about it" is the SAME statement whether the item is a
checkpoint scored by Pass@K over a probe set, or a training example scored by the
fraction of sampled targets the model currently gets right. One criterion, two
uses, one implementation -- so a change to the weighting changes both coherently
instead of leaving two subtly different copies of "uncertainty is valuable" in the
tree.

What this module deliberately does NOT do
-----------------------------------------
It does not build probe sets. The paper has a capable LLM construct one per
subdomain; that is a data pipeline concern, and this repository has no
LLM-backed data authoring. It also makes no claim about whether breadth is
achievable at a given scale -- that is the paper's empirical result and it is not
measured here.

References:
    - Xu, S. et al. (2025). Tiny Model, Big Logic: Diversity-Driven Optimization
      Elicits Large-Model Reasoning Ability in VibeThinker-1.5B. (Domain-Aware
      Diversity Probing; Pass@K-maximising checkpoint selection.)
    - Chen, M. et al. (2021). Evaluating Large Language Models Trained on Code.
      (the unbiased pass@k estimator, via ``metrics.pass_at_k``.)
    - Dalal, U. et al. (2025). Leveraging LLM Inconsistency to Boost Pass@k
      Performance. (the diversity / Pass@K link the paper relies on.)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import keras
import numpy as np

from dl_techniques.metrics.pass_at_k import (
    pass_at_k,
    pass_at_k_curve,
    per_problem_pass_at_k,
    solved_counts,
)
from dl_techniques.utils.logger import logger

from ..constants import (
    DEFAULT_MGPO_LAMBDA,
    DEFAULT_MGPO_P0,
    DEFAULT_SSP_ESTIMATOR,
    DEFAULT_SSP_PASS_AT_K,
)
from .signal import binary_entropy, max_entropy_weight

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: Supported ``mode`` values for :func:`spectrum_sampling_weights`.
SAMPLING_MODES: Tuple[str, ...] = (
    "max_entropy",
    "uniform",
    "proportional",
    "inverse",
)

#: How many significant digits a normalised weight vector is logged to.
_LOG_PRECISION: int = 4

# DECISION <ssp-2026-10-05>/D-005
# `select_specialists` breaks score ties toward the LOWEST checkpoint index rather
# than toward the last one seen. `np.argmax` already does this, so the rule is free
# -- but it is stated because the alternative is a real reproducibility hazard: a
# tie means two checkpoints measured identically, and a selection that depends on
# dict ordering or on which tensor happened to be scanned first is not a selection
# anybody can re-derive from a saved score matrix.


# ---------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------


def spectrum_profile(
        outcomes,
        ks: Optional[Sequence[int]] = None,
        estimator: str = DEFAULT_SSP_ESTIMATOR,
        include_per_problem: bool = False,
) -> Dict[str, object]:
    """Summarise how broad one model's candidate pool is.

    ``pass@k`` alone is not enough to read a profile. A pool can be broad on average
    because every problem is moderately covered, or because one problem is fully
    covered and the rest are not at all -- and those two call for opposite actions.
    The profile reports both, plus the outcome entropy that separates "many samples,
    all missing the same way" from "many samples, genuinely split".

    Args:
        outcomes: ``(n_problems, n_samples)`` per-sample outcome matrix; see
            :mod:`dl_techniques.metrics.pass_at_k` for the layout and the ``> 0``
            rule.
        ks: The ``k`` values for the coverage curve. Defaults to the powers of two
            up to ``n_samples``, always including ``n_samples``.
        estimator: ``"unbiased"`` (default) or ``"plug_in"``.
        include_per_problem: When ``True``, add a ``"per_problem"`` key holding the
            ``(n_problems,)`` per-problem pass@k array at the largest requested
            ``k``. Useful for sorting the training set; off by default because it is
            the one part of the profile that scales with the problem count.

    Returns:
        A ``dict`` with these keys:

        * ``"pass_at_1"`` -- float; mean single-sample success rate, the accuracy
          SSP argues is the WRONG selection criterion;
        * ``"pass_at_k_curve"`` -- ``dict[int, float]``; coverage against ``k``;
        * ``"pass_at_k"`` -- float; coverage at the largest requested ``k``, i.e.
          the headline breadth number;
        * ``"mean_solved_count"`` -- float; mean count of correct samples per
          problem;
        * ``"outcome_entropy"`` -- float in ``[0, log 2]``; mean binary entropy of
          the per-problem solved rate. This is the entropy of the OUTCOME
          distribution, not of the model's token distribution, and it is a
          diagnostic rather than a training signal;
        * ``"solved_fraction"`` -- float; fraction of problems with at least one
          correct sample;
        * ``"n_problems"``, ``"n_samples"`` -- int; the matrix shape;
        * ``"per_problem"`` -- ``(n_problems,)`` array, only when
          ``include_per_problem`` is ``True``.

    Raises:
        ValueError: On any problem with
            :func:`dl_techniques.metrics.pass_at_k.pass_at_k_curve`.

    Note:
        ``"pass_at_1"`` and ``"pass_at_k"`` are reported together deliberately.
        SSP's claim is that the two can be maximised together -- that optimising
        the spectrum does not cost single-shot accuracy -- and that claim is only
        checkable if both numbers are in front of you. Reporting one is how the
        claim gets mistaken for an established result.
    """
    matrix = np.asarray(outcomes, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"outcomes must be 2-D (n_problems, n_samples); got shape {matrix.shape}"
        )
    n_problems, n_samples = matrix.shape

    curve = pass_at_k_curve(matrix, ks=ks, estimator=estimator)
    counts = solved_counts(matrix)
    rates = counts.astype(np.float64) / float(n_samples)

    widest = max(curve)
    profile: Dict[str, object] = {
        "pass_at_1": float(curve.get(1, pass_at_k(matrix, 1, estimator=estimator))),
        "pass_at_k_curve": {int(k): float(v) for k, v in curve.items()},
        "pass_at_k": float(curve[widest]),
        "mean_solved_count": float(np.mean(counts.astype(np.float64))),
        "outcome_entropy": float(np.mean(binary_entropy(rates))),
        "solved_fraction": float(np.mean(counts > 0)),
        "n_problems": int(n_problems),
        "n_samples": int(n_samples),
    }

    if include_per_problem:
        profile["per_problem"] = per_problem_pass_at_k(
            matrix, widest, estimator=estimator
        )

    logger.info(
        f"spectrum_profile: pass@1={profile['pass_at_1']:.4f} "
        f"pass@{widest}={profile['pass_at_k']:.4f} "
        f"outcome_entropy={profile['outcome_entropy']:.4f} "
        f"over {n_problems} problems x {n_samples} samples"
    )
    return profile


# ---------------------------------------------------------------------
# Domain-aware specialist selection
# ---------------------------------------------------------------------


def select_specialists(
        score_matrix,
        subdomains: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Pick the diversity-maximising checkpoint per subdomain.

    The paper's Domain-Aware Diversity Probing: a checkpoint is evaluated on each
    subdomain's probe set, and each subdomain keeps its own ``argmax``. A single
    global argmax is the wrong rule here and demonstrably so -- the returned
    ``"spectrum_gain"`` is exactly the amount by which per-subdomain selection beats
    the best single checkpoint, and it is ``0.0`` precisely when a global argmax
    would have been correct anyway.

    Args:
        score_matrix: ``(n_checkpoints, n_subdomains)``. Any broadcastable-to-2-D
            array of scores; higher means broader. A ``(n_checkpoints,)`` vector is
            accepted and treated as one subdomain, which makes the degenerate
            single-domain case a spelling rather than a special case.
        subdomains: Optional labels, one per column, for the returned mapping. When
            ``None`` the columns are named ``"domain_0" ...``.

    Returns:
        A ``dict`` with these keys:

        * ``"checkpoint_index"`` -- ``(n_subdomains,)`` int array; the selected
          checkpoint per subdomain. Ties go to the LOWEST index.
        * ``"checkpoint_by_subdomain"`` -- ``dict[str, int]``; the same selection
          keyed by label.
        * ``"selected_scores"`` -- ``(n_subdomains,)`` array; each winner's score.
        * ``"best_overall_index"`` -- int; the ``argmax`` of the column MEANS, i.e.
          the checkpoint a non-diversity-aware pipeline would have kept.
        * ``"best_overall_score"`` -- float; its mean score.
        * ``"spectrum_gain"`` -- float; mean of ``selected_scores`` minus
          ``best_overall_score``. Non-negative by construction, and ``0.0`` exactly
          when one checkpoint dominates every subdomain.
        * ``"subdomains"`` -- ``tuple[str, ...]``; the labels used.

    Raises:
        ValueError: If the matrix is not 2-D (after the 1-D promotion), has a
            zero-length axis, contains a non-finite score, or ``subdomains`` has a
            length different from the number of columns.

    Note:
        ``"spectrum_gain"`` is the honest measure of whether the diversity-aware
        machinery earned anything on THIS matrix. A gain of zero means the extra step
        bought nothing and the simple argmax was already optimal.

        **It is also a WEAK discriminator, which is measured rather than
        asserted.** An implementation that ignored subdomains and returned the
        best-overall checkpoint for every column would report a gain of exactly
        ``0.0`` too -- the best-overall row's mean IS the mean of its own
        per-column values. So a gain of zero cannot distinguish "nothing to gain"
        from "never looked". Check ``"checkpoint_index"`` against the per-column
        maxima as well; the count of subdomains where the selection is a true column
        argmax is the statistic that does discriminate.
    """
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D (n_checkpoints, n_subdomains); got shape "
            f"{score_matrix.shape}"
        )
    n_checkpoints, n_subdomains = matrix.shape
    if n_checkpoints == 0 or n_subdomains == 0:
        raise ValueError(
            f"score_matrix must have non-zero extent on both axes; got shape "
            f"{matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "score_matrix must be finite; found "
            f"{int(np.count_nonzero(~np.isfinite(matrix)))} non-finite entries. A "
            f"NaN score would silently lose the argmax."
        )

    if subdomains is None:
        labels = tuple(f"domain_{i}" for i in range(n_subdomains))
    else:
        labels = tuple(str(label) for label in subdomains)
        if len(labels) != n_subdomains:
            raise ValueError(
                f"subdomains has {len(labels)} labels but score_matrix has "
                f"{n_subdomains} columns; one label per column."
            )

    # DECISION <ssp-2026-10-05>/D-005: np.argmax returns the FIRST maximal index, so
    # ties resolve to the lowest checkpoint deterministically.
    checkpoint_index = np.argmax(matrix, axis=0).astype(np.int64)
    selected_scores = matrix[checkpoint_index, np.arange(n_subdomains)]

    column_means = matrix.mean(axis=1)
    best_overall_index = int(np.argmax(column_means))
    best_overall_score = float(column_means[best_overall_index])
    spectrum_gain = float(np.mean(selected_scores) - best_overall_score)

    logger.info(
        f"select_specialists: {n_subdomains} subdomains x {n_checkpoints} "
        f"checkpoints -> indices {checkpoint_index.tolist()}, "
        f"spectrum_gain={spectrum_gain:.{_LOG_PRECISION}f} over best_overall="
        f"{best_overall_index} ({best_overall_score:.{_LOG_PRECISION}f})"
    )

    return {
        "checkpoint_index": checkpoint_index,
        "checkpoint_by_subdomain": {
            label: int(index) for label, index in zip(labels, checkpoint_index)
        },
        "selected_scores": selected_scores,
        "best_overall_index": best_overall_index,
        "best_overall_score": best_overall_score,
        "spectrum_gain": spectrum_gain,
        "subdomains": labels,
    }


# ---------------------------------------------------------------------
# Data-side spectrum selection
# ---------------------------------------------------------------------


def spectrum_sampling_weights(
        coverage,
        mode: str = "max_entropy",
        lam: float = DEFAULT_MGPO_LAMBDA,
        p0: float = DEFAULT_MGPO_P0,
) -> np.ndarray:
    """Turn per-item coverage into a sampling distribution over a training set.

    The Spectrum phase applied to DATA SELECTION rather than to checkpoint
    selection, and the piece that makes the principle usable without any RL. The
    argument is the same one MGPO rests on, transplanted: an item's training value
    peaks where the model is maximally uncertain about it, so weight the items whose
    coverage sits nearest 0.5 and let the already-solved and the not-yet-reachable
    ones recede.

    ``mode="max_entropy"`` calls
    :func:`~dl_techniques.optimization.ssp.signal.max_entropy_weight` directly, so
    the checkpoint-side and data-side criteria cannot drift apart.

    Args:
        coverage: Per-item coverage in ``[0, 1]``. Any 1-D array-like: pass@k per
            example, the fraction of sampled targets the model currently solves, or
            anything else that reduces to "how broad is this item's solution set".
        mode: ``"max_entropy"`` (default; ``w = exp(-lam * D_ME(coverage || p0))``),
            ``"uniform"``, ``"proportional"`` (``w = coverage``) or ``"inverse"``
            (``w = 1 - coverage``).
        lam: Sharpening, used only by ``"max_entropy"``. Larger concentrates harder
            on coverage near ``p0``.
        p0: Target coverage, used only by ``"max_entropy"``.

    Returns:
        A ``(N,)`` ``float64`` array of non-negative weights summing to ``1.0``
        within float tolerance.

    Raises:
        ValueError: If ``mode`` is unknown, ``coverage`` is not 1-D or is empty, or
            the weights sum to zero and so cannot be normalised -- which happens for
            ``"proportional"`` on an all-zero coverage vector, and is a real signal
            (nothing has been solved at all, so there is no ordering to express)
            rather than a degenerate input.

    Note:
        ``"proportional"`` and ``"inverse"`` are deliberately naive baselines, and
        the difference between them and ``"max_entropy"`` is the point.
        ``"proportional"`` up-weights the items the model already solves best -- the
        opposite of the intent. ``"inverse"`` up-weights the items it fails, which
        points at items it cannot reach at all, where the reward signal is uniform
        noise. ``"max_entropy"`` is the only one of the three that peaks in the
        middle.
    """
    if mode not in SAMPLING_MODES:
        raise ValueError(
            f"Unknown sampling mode {mode!r}. Supported: {list(SAMPLING_MODES)}"
        )

    values = np.asarray(coverage, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"coverage must be 1-D (one value per training item); got shape "
            f"{values.shape}"
        )
    if values.size == 0:
        raise ValueError("coverage must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("coverage must all be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            f"coverage must lie in [0, 1]; got range "
            f"[{float(values.min())}, {float(values.max())}]. It is read as a "
            f"success probability, so threshold a raw score upstream."
        )

    if mode == "uniform":
        weights = np.ones_like(values)
    elif mode == "proportional":
        weights = values.copy()
    elif mode == "inverse":
        weights = 1.0 - values
    else:
        # The one seam between this numpy module and the keras.ops Signal half:
        # `max_entropy_weight` is backend-agnostic because the RL use has to run in
        # a traced graph, and `spectrum.py` holds plain arrays. Converting here is
        # deliberate -- the criterion is SHARED, the container is not.
        weights = np.asarray(
            keras.ops.convert_to_numpy(max_entropy_weight(values, lam=lam, p0=p0)),
            dtype=np.float64,
        )

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"mode={mode!r} produced weights summing to {total} for this coverage "
            f"vector, so there is no distribution to normalise. For "
            f"'proportional' that means every item scored zero coverage -- nothing "
            f"has been solved yet, so coverage carries no ordering signal. Use "
            f"'uniform' or 'max_entropy'."
        )
    weights = weights / total
    positive = weights[weights > 0.0]
    # DECISION <ssp-2026-10-05>/D-008: report the max/min weight ratio only over the
    # POSITIVE weights. `proportional` on a coverage vector containing a zero yields
    # an exact zero, and the ratio against it is ~1e308 -- a log line that fills the
    # screen with digits and tells the reader nothing. "some items get exactly zero"
    # is the fact worth logging, so it is logged as such.
    if positive.size == 0:
        spread = "all weights are zero"
    elif positive.size < weights.size:
        spread = (
            f"{weights.size - positive.size} of {weights.size} weights are exactly "
            f"zero; max/min over the rest = "
            f"{float(positive.max() / positive.min()):.{_LOG_PRECISION}f}"
        )
    else:
        spread = f"{float(weights.max() / positive.min()):.{_LOG_PRECISION}f}"
    logger.info(
        f"spectrum_sampling_weights: mode={mode}, n={weights.size}, "
        f"max/min weight ratio={spread}"
    )
    return weights
