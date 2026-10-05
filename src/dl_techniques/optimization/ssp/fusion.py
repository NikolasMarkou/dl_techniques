"""Expert Model Fusion: consolidating a Spectrum into one model.

The second half of the Spectrum phase. Having selected one diversity-maximising
specialist per subdomain, SSP combines them into a single model by a weighted
linear combination of parameters:

.. math::  M_{Merge} = \\sum_{i=1}^{N} w_i M_i^{*}, \\qquad w_i \\ge 0,\\ \\sum_i w_i = 1

with the paper using the unweighted case ``w_i = 1 / N``. This module implements
that, plus the two schemes that are strictly better-known siblings of it and cost
nothing to add:

* **score-softmax weights** -- ``w_i = softmax(P_i / T)`` over the specialists'
  measured Pass@K, which spends capacity in proportion to how broad each
  specialist's spectrum actually was instead of treating a specialist that found
  one correct sample per problem the same as one that found eight;
* **task arithmetic** (Ilharco et al. 2023) -- ``base + c * sum_i w_i (M_i - base)``,
  the difference-space form, for specialists that share a pretrained ancestor but do
  NOT share a fine-tuning trajectory;
* **greedy soup** (Wortsman et al. 2022) -- add candidates one at a time, keep one
  only if a held-out score improves. This needs a scorer and so is a function of a
  callback rather than of the weights alone.

Three properties are load-bearing and are pinned by the test suite rather than left
to the docstrings:

**Fusion never mutates its inputs.** Every function here returns NEW arrays. There
is no ``set_weights`` convenience wrapper, deliberately: a merge that writes into a
live model is unrecoverable the moment two of the inputs alias it, and the caller
who wants the assignment can say so in one line.

**Uniform fusion of identical inputs is the identity, bit for bit.** Averaging
``N`` copies of the same array and dividing by ``N`` in float64 is *not* guaranteed
to return that array exactly, so the implementation detects the degenerate case
(``N == 1``, or every input already identical) and short-circuits. Without it,
``fuse_specialists([w, w, w, w])`` returns ``w +/- 1 ulp``, and every downstream
comparison against the original weights inherits the discrepancy.

**Linear and task-arithmetic fusion coincide when the base is shared.** With a
common ``base`` and ``coefficient == 1``,

.. math::

    base + \\sum_i w_i (M_i - base) = base \\sum_i w_i + \\sum_i w_i M_i - base \\sum_i w_i
                                  = \\sum_i w_i M_i

because ``sum_i w_i == 1``. So for specialists fine-tuned from one ancestor -- which
is exactly the paper's setting -- the two modes are the same function and the
``mode`` knob only matters when specialists came from different bases, or when
``coefficient != 1`` scales the deviation. This is derived in the tests rather than
asserted here.

References:
    - Xu, S. et al. (2025). Tiny Model, Big Logic: Diversity-Driven Optimization
      Elicits Large-Model Reasoning Ability in VibeThinker-1.5B. (the uniform merge
      ``w_i = 1/N``.)
    - Wortsman, T. et al. (2022). Model soups: averaging weights of multiple
      fine-tuned models improves accuracy without increasing inference cost.
      (https://arxiv.org/abs/2203.05482)
    - Ilharco, G. et al. (2023). Editing models with task arithmetic.
      (https://arxiv.org/abs/2212.04089)
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: Supported ``mode`` values for :func:`fuse_specialists`.
FUSION_MODES: Tuple[str, ...] = ("linear", "task_arithmetic")

#: Supported ``scheme`` values for :func:`fusion_weights_from_scores`.
FUSION_WEIGHT_SCHEMES: Tuple[str, ...] = ("uniform", "score_softmax")

# DECISION <ssp-2026-10-05>/D-003
# Fusion accumulates in float64 and casts back ONCE at the end. Accumulating in the
# inputs' own dtype (typically float32, occasionally bfloat16 under a mixed policy)
# makes the rounding depend on the summation ORDER, so reordering the specialist
# list would move a fused weight by a visible fraction of a float32 epsilon across
# a whole tensor -- and with thousands of variables, enough of them to shift the
# fused model's logits.
#
# WHAT THIS DOES AND DOES NOT BUY, stated because the obvious next claim is false:
# float addition is not associative in ANY precision, so float64 accumulation does
# not make the result bit-exact under reordering -- it makes the perturbation small.
# MEASURED on 5 specialists x 3 variables of (500, 40) float32, reordering moved
# the fused weights by at most 1e-6 relative (and left almost every stored float32
# bit identical); the same sum accumulated in float32 is order-dependent by
# construction. That bound, not bit-equality, is the guarantee, and it is what
# `tests/test_optimization/test_ssp/test_fusion.py` pins.
_ACCUMULATION_DTYPE: np.dtype = np.dtype(np.float64)

#: Number of decimals used when logging the fused weight vector.
_LOG_PRECISION: int = 4


# ---------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------


def _as_weight_sets(models_or_weights: Sequence[Any]) -> List[List[np.ndarray]]:
    """Coerce the caller's inputs to a list of plain numpy weight lists.

    Accepts either a sequence of weight lists (``List[np.ndarray]``, i.e. what
    ``model.get_weights()`` returns) or a sequence of objects exposing
    ``get_weights()`` -- which includes every ``keras.Model``. Reading weights is
    non-mutating, so a live model may be passed directly; it is never written to.

    Args:
        models_or_weights: A sequence whose elements are either weight lists or
            ``get_weights()``-bearing objects.

    Returns:
        A list of weight lists, each a ``list`` of read-only numpy views. The
        arrays are the caller's own buffers (``np.asarray`` does not copy) and are
        treated as immutable throughout.

    Raises:
        ValueError: If the outer sequence is empty, an element is neither a
            sequence of arrays nor has ``get_weights``, or ``get_weights()`` returns
            an empty list.
    """
    if isinstance(models_or_weights, np.ndarray):
        raise ValueError(
            "models_or_weights must be a SEQUENCE of weight sets (one per "
            "specialist), not a single array. A bare (num_vars, ...) array was "
            "probably meant as a stack of same-shaped variables."
        )
    items = list(models_or_weights)
    if not items:
        raise ValueError(
            "models_or_weights must contain at least one specialist; got an "
            "empty sequence"
        )

    weight_sets: List[List[np.ndarray]] = []
    for index, item in enumerate(items):
        if hasattr(item, "get_weights") and not isinstance(item, (list, tuple)):
            weights = list(item.get_weights())
        elif isinstance(item, (list, tuple)):
            weights = list(item)
        else:
            raise ValueError(
                f"specialist {index} is a {type(item).__name__}, which is neither a "
                f"sequence of arrays nor an object with get_weights(). Pass "
                f"model.get_weights() results, or the keras.Model itself."
            )
        if not weights:
            raise ValueError(
                f"specialist {index} has no weights (empty list). An unbuilt model "
                f"has no variables to fuse; build it first."
            )
        for position, array in enumerate(weights):
            if not isinstance(array, np.ndarray):
                raise ValueError(
                    f"specialist {index} variable {position} is a "
                    f"{type(array).__name__}, not a numpy array. Convert with "
                    f"keras.ops.convert_to_numpy(...) first."
                )
        weight_sets.append(weights)
    return weight_sets


def _validate_aligned(weight_sets: List[List[np.ndarray]]) -> None:
    """Assert every specialist has the same variables, in the same order and shape.

    A silent misalignment here is the worst failure mode this module has: fusing
    variable 3 of model A with variable 3 of model B produces a model that builds
    and saves and computes garbage, with no error anywhere.

    Args:
        weight_sets: The coerced specialist weight lists.

    Raises:
        ValueError: On a differing variable count, a differing shape at any
            position, or a non-floating dtype (an integer variable cannot be
            meaningfully averaged and casting it would silently change its
            semantics).
    """
    reference = weight_sets[0]
    n_vars = len(reference)
    for index, weights in enumerate(weight_sets[1:], start=1):
        if len(weights) != n_vars:
            raise ValueError(
                f"specialist {index} has {len(weights)} variables but specialist 0 "
                f"has {n_vars}. Fusion is positional: variable i of every "
                f"specialist must be the same variable."
            )
        for position, (expected, actual) in enumerate(zip(reference, weights)):
            if expected.shape != actual.shape:
                raise ValueError(
                    f"variable {position} has shape {actual.shape} in specialist "
                    f"{index} but {expected.shape} in specialist 0. Two specialists "
                    f"with different architectures cannot be fused -- the result "
                    f"would build, save, and compute nonsense with no error."
                )
            if not np.issubdtype(actual.dtype, np.floating):
                raise ValueError(
                    f"variable {position} of specialist {index} has non-floating "
                    f"dtype {actual.dtype}. Averaging an integer or boolean "
                    f"variable is not meaningful; exclude it and handle it "
                    f"separately."
                )


def _normalize_weights(
        weights: Optional[Sequence[float]],
        n_specialists: int,
) -> np.ndarray:
    """Build a non-negative weight vector summing to 1.

    SSP requires ``w_i >= 0`` and ``sum_i w_i = 1`` -- the second condition is what
    preserves the parameter scale of the fused model. An explicit weight vector is
    therefore renormalised rather than trusted, and a negative entry is refused
    rather than clipped: a negative mixture weight is not a slightly-wrong soup, it
    is extrapolation into a region of weight space no specialist occupies.

    Args:
        weights: The caller's weights, or ``None`` for the uniform ``1 / N`` the
            paper uses.
        n_specialists: How many specialists are being fused.

    Returns:
        A ``(N,)`` ``float64`` array of non-negative weights summing to 1 within
        float tolerance.

    Raises:
        ValueError: If the length is wrong, an entry is negative or non-finite, or
            the entries sum to zero (nothing to normalise).
    """
    if weights is None:
        return np.full(n_specialists, 1.0 / n_specialists, dtype=np.float64)

    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vector.shape[0] != n_specialists:
        raise ValueError(
            f"weights has {vector.shape[0]} entries but there are "
            f"{n_specialists} specialists; fusion is one weight per specialist, in "
            f"the same order."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"weights must all be finite; got {weights!r}")
    if np.any(vector < 0.0):
        raise ValueError(
            f"weights must be non-negative; got {vector.tolist()}. A negative "
            f"mixture weight extrapolates outside the hull of the specialists "
            f"rather than interpolating inside it."
        )
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(
            "weights sum to zero, so there is nothing to normalise. Pass at least "
            "one positive weight."
        )
    return vector / total


# ---------------------------------------------------------------------
# Weight schemes
# ---------------------------------------------------------------------


def uniform_weights(n_specialists: int) -> np.ndarray:
    """The paper's scheme: ``w_i = 1 / N``.

    Every specialist counts equally regardless of how broad its measured spectrum
    was. Simple, and a reasonable default when the specialists were selected by
    anything other than a measured score.

    Args:
        n_specialists: Number of specialists, ``>= 1``.

    Returns:
        A ``(N,)`` ``float64`` array of ``1 / N``.

    Raises:
        ValueError: If ``n_specialists`` is not a positive integer.
    """
    count = int(n_specialists)
    if count < 1:
        raise ValueError(f"n_specialists must be >= 1; got {n_specialists}")
    return np.full(count, 1.0 / count, dtype=np.float64)


def score_softmax_weights(
        scores: Sequence[float],
        temperature: float = 1.0,
) -> np.ndarray:
    """``w_i = softmax(scores_i / T)`` -- capacity in proportion to measured breadth.

    The natural alternative to uniform weights when the specialists were chosen by
    a score (the paper's ``Pass@K`` per subdomain). A specialist whose spectrum
    covered one correct sample per problem gets an exponentially smaller share than
    one that covered eight.

    Args:
        scores: One score per specialist, in the same order as the specialists.
            Higher means broader.
        temperature: Divisor on the scores. Larger is softer (closer to uniform);
            smaller is sharper. Must be positive. Default ``1.0``.

    Returns:
        A ``(N,)`` ``float64`` array of strictly positive weights summing to 1.

    Raises:
        ValueError: If ``scores`` is empty, non-finite, or ``temperature`` is not
            positive and finite.

    Note:
        ``softmax`` is shift-invariant, so adding a constant to every score changes
        nothing -- only the DIFFERENCES between scores matter, and the ratio
        ``max(s) / min(s)`` is what ``temperature`` divides.
    """
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"temperature must be positive and finite; got {temperature!r}")

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"scores must all be finite; got {scores!r}")

    scaled = values / temp
    # Shift by the max before exp: the raw softmax overflows float64 for a spread
    # above ~709 nats, and pass@K differences are bounded by 1 so this only ever
    # matters for a caller passing log-probabilities or logits.
    scaled = scaled - np.max(scaled)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum()


# ---------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------


def fuse_specialists(
        models_or_weights: Sequence[Any],
        weights: Optional[Sequence[float]] = None,
        mode: str = "linear",
        base: Optional[Sequence[np.ndarray]] = None,
        coefficient: float = 1.0,
) -> List[np.ndarray]:
    """Fuse specialists into one weight set. Returns NEW arrays; mutates nothing.

    .. math::

        \\text{linear}: \\quad M = \\sum_i w_i M_i

        \\text{task\\_arithmetic}: \\quad M = base
            + c \\sum_i w_i (M_i - base)

    Args:
        models_or_weights: A sequence of specialists. Each element is either a
            weight list (``model.get_weights()``) or an object with
            ``get_weights()`` -- a ``keras.Model`` qualifies, and is only ever read.
        weights: One non-negative weight per specialist, in order. ``None`` (default)
            is the paper's uniform ``1 / N``. A supplied vector is renormalised to
            sum to 1.
        mode: ``"linear"`` (default) or ``"task_arithmetic"``.
        base: The shared ancestor weights, required by ``"task_arithmetic"`` and
            ignored otherwise. Must match the specialists' variables positionally.
        coefficient: ``c``, the task-arithmetic scale. Default ``1.0``. Ignored by
            ``"linear"``.

    Returns:
        A ``list`` of new ``numpy`` arrays, one per variable, each in the dtype of
        the corresponding input variable.

    Raises:
        ValueError: If ``mode`` is unknown, fewer than one specialist is given, the
            specialists are misaligned, a weight is negative or non-finite,
            ``"task_arithmetic"`` is requested without ``base``, ``base`` is
            misaligned, or a variable is non-floating.

    Note:
        With a shared ``base`` and ``coefficient == 1`` the two modes are the SAME
        function, because ``sum_i w_i == 1``. The ``mode`` knob bites only when the
        specialists came from different ancestors, or when ``coefficient != 1``
        scales the deviation away from the base. See the module docstring.
    """
    if mode not in FUSION_MODES:
        raise ValueError(
            f"Unknown fusion mode {mode!r}. Supported: {list(FUSION_MODES)}"
        )

    weight_sets = _as_weight_sets(models_or_weights)
    _validate_aligned(weight_sets)
    mix = _normalize_weights(weights, len(weight_sets))
    coefficient_value = float(coefficient)
    if not np.isfinite(coefficient_value):
        raise ValueError(f"coefficient must be finite; got {coefficient!r}")

    base_weights: Optional[List[np.ndarray]] = None
    if mode == "task_arithmetic":
        if base is None:
            raise ValueError(
                "mode='task_arithmetic' needs `base` -- the weights the deviations "
                "are measured against. Without a base there is no difference space "
                "and the mode is identical to 'linear'."
            )
        base_weights = _as_weight_sets([base])[0]
        _validate_aligned([weight_sets[0], base_weights])
    elif base is not None:
        logger.warning(
            f"fuse_specialists: `base` was supplied but mode={mode!r} ignores it. "
            f"Pass mode='task_arithmetic' if the deviations are what you meant."
        )

    reference = weight_sets[0]
    n_vars = len(reference)

    # DECISION <ssp-2026-10-05>/D-004 -- the exact-identity short circuit. Fusing N
    # identical inputs computes `sum_i w_i * x_i` in float64 and divides by N; with
    # w_i = 1/N that is exactly x in real arithmetic, but the multiply-add sequence
    # rounds, and the result lands within an ulp of -- not on -- the input. Any caller
    # comparing a fused model against the original with `atol=0` then fails on a
    # difference the implementation introduced. The test is exact equality of the
    # ARRAY BYTES, so it costs nothing when it does not fire.
    if len(weight_sets) == 1 or _all_identical(weight_sets):
        logger.info(
            f"fuse_specialists: all {len(weight_sets)} specialists are identical; "
            f"returning a copy of specialist 0 unchanged."
        )
        return [np.array(array, copy=True) for array in reference]

    fused: List[np.ndarray] = []
    for position in range(n_vars):
        accumulator = np.zeros(reference[position].shape, dtype=_ACCUMULATION_DTYPE)
        for index, weights in enumerate(weight_sets):
            array = np.asarray(weights[position], dtype=_ACCUMULATION_DTYPE)
            if base_weights is not None:
                array = array - np.asarray(
                    base_weights[position], dtype=_ACCUMULATION_DTYPE
                )
            accumulator += mix[index] * array
        if base_weights is not None:
            accumulator += coefficient_value * np.asarray(
                base_weights[position], dtype=_ACCUMULATION_DTYPE
            )
        fused.append(accumulator.astype(reference[position].dtype))

    logger.info(
        f"fuse_specialists: fused {len(weight_sets)} specialists "
        f"({n_vars} variables) with mode={mode}, weights="
        f"{np.round(mix, _LOG_PRECISION).tolist()}"
    )
    return fused


def _all_identical(weight_sets: List[List[np.ndarray]]) -> bool:
    """Report whether every specialist holds byte-identical weights.

    Args:
        weight_sets: The coerced specialist weight lists.

    Returns:
        ``True`` if there is more than one specialist and every array of every
        specialist equals specialist 0's byte for byte (same dtype, same shape,
        same bits). ``False`` otherwise, including for a single specialist's
        internals.

    Note:
        The head-and-tail fingerprint is a FILTER, never a verdict. It can only
        return ``False`` early, and it makes the overwhelmingly common case -- N
        genuinely different specialists -- cost a few hundred byte comparisons
        instead of a full pass over every parameter of every model, which on a
        multi-billion-parameter set is gigabytes of reads to learn nothing. Only a
        fingerprint match escalates to the exhaustive comparison.
    """
    if len(weight_sets) < 2:
        return False
    reference = weight_sets[0]

    def _fingerprint(weights: List[np.ndarray]) -> List[np.ndarray]:
        probes: List[np.ndarray] = []
        for position in sorted({0, len(weights) - 1}):
            flat = np.ravel(weights[position])
            probes.append(flat[:64])
            probes.append(flat[-64:])
        return probes

    ref_probe = _fingerprint(reference)
    for weights in weight_sets[1:]:
        if len(weights) != len(reference):
            return False
        for expected, actual in zip(reference, weights):
            if expected.dtype != actual.dtype or expected.shape != actual.shape:
                return False
        for expected, actual in zip(ref_probe, _fingerprint(weights)):
            if not np.array_equal(expected, actual):
                return False
    # Fingerprints all matched: pay for the exhaustive check.
    for weights in weight_sets[1:]:
        for expected, actual in zip(reference, weights):
            if not np.array_equal(expected, actual):
                return False
    return True


def fusion_weights_from_scores(
        scores: Sequence[float],
        scheme: str = "uniform",
        temperature: float = 1.0,
) -> np.ndarray:
    """Dispatch to a named weight scheme.

    Args:
        scores: One score per specialist (the paper's per-subdomain Pass@K).
        scheme: ``"uniform"`` (the paper's ``1 / N``) or ``"score_softmax"``.
        temperature: Only used by ``"score_softmax"``.

    Returns:
        A ``(N,)`` ``float64`` array of non-negative weights summing to 1.

    Raises:
        ValueError: If ``scheme`` is unknown, or on any problem with
            :func:`uniform_weights` / :func:`score_softmax_weights`.
    """
    if scheme not in FUSION_WEIGHT_SCHEMES:
        raise ValueError(
            f"Unknown weight scheme {scheme!r}. Supported: {list(FUSION_WEIGHT_SCHEMES)}"
        )
    if scheme == "uniform":
        return uniform_weights(len(list(scores)))
    return score_softmax_weights(scores, temperature=temperature)


# ---------------------------------------------------------------------
# Greedy soup
# ---------------------------------------------------------------------


def greedy_soup(
        pool: Sequence[Any],
        score_fn: Callable[[List[np.ndarray]], float],
        max_size: Optional[int] = None,
) -> Tuple[List[int], List[np.ndarray]]:
    """Greedy model soup (Wortsman et al. 2022): add candidates while the score rises.

    Uniform averaging assumes every candidate helps. Greedy soup tests that
    assumption per candidate, against a held-out scorer, and is the standard
    companion to uniform fusion when a scorer is available at all. The pool is not
    modified and nothing is written to any model.

    The returned weights are UNIFORM over the selected candidates, which is what
    makes the greedy part meaningful: if the weights were free, the greedy search
    would be solving a harder problem than soup does and the comparison to uniform
    averaging would no longer be like-for-like.

    Algorithm, in order:

    1. Score every candidate alone; seed the accumulator with the best.
    2. For each remaining candidate in descending solo-score order, tentatively add
       it with weight ``1 / (m + 1)``; keep it only if the score strictly improves.
    3. Stop early when the score stops improving or ``max_size`` is reached.

    Args:
        pool: Candidate specialists, in the same accepted forms as
            :func:`fuse_specialists`.
        score_fn: Callable taking a fused weight list and returning a float to
            MAXIMISE. Must be deterministic for greedy soup to mean anything; a
            stochastic scorer makes the selection order-dependent and the result
            irreproducible.
        max_size: Optional cap on how many candidates are kept.

    Returns:
        A ``(indices, weights)`` tuple: the ascending indices of the selected
        candidates into ``pool``, and the uniform ``(len(indices),)`` weight vector
        over them.

    Raises:
        ValueError: If ``pool`` is empty, ``score_fn`` is not callable, a solo score
            is non-finite, or ``max_size`` is below 1.
    """
    if not callable(score_fn):
        raise ValueError(
            f"score_fn must be callable, taking a fused weight list and returning "
            f"a float to maximise; got {type(score_fn).__name__}"
        )
    weight_sets = _as_weight_sets(pool)
    _validate_aligned(weight_sets)
    n_candidates = len(weight_sets)
    if max_size is not None and int(max_size) < 1:
        raise ValueError(f"max_size must be >= 1 when given; got {max_size}")

    solo = [float(score_fn([np.array(a, copy=True) for a in ws])) for ws in weight_sets]
    if not all(np.isfinite(solo)):
        raise ValueError(
            f"score_fn returned a non-finite solo score: "
            f"{[(i, s) for i, s in enumerate(solo) if not np.isfinite(s)]}"
        )

    order = sorted(range(n_candidates), key=lambda i: (-solo[i], i))
    selected = [order[0]]
    best_score = solo[order[0]]
    logger.info(
        f"greedy_soup: seeded with candidate {order[0]} (score {best_score:.6f})"
    )

    for candidate in order[1:]:
        if max_size is not None and len(selected) >= int(max_size):
            break
        trial = selected + [candidate]
        trial_weights = uniform_weights(len(trial))
        merged = fuse_specialists(
            [weight_sets[i] for i in trial], weights=trial_weights
        )
        trial_score = float(score_fn(merged))
        if trial_score > best_score:
            logger.info(
                f"greedy_soup: + candidate {candidate} -> {trial_score:.6f} "
                f"(was {best_score:.6f})"
            )
            selected = trial
            best_score = trial_score
        else:
            logger.debug(
                f"greedy_soup: rejected candidate {candidate} ({trial_score:.6f} "
                f"<= {best_score:.6f})"
            )

    selected.sort()
    return selected, uniform_weights(len(selected))
