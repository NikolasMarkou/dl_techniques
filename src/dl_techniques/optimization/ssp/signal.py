"""MaxEnt-Guided Policy Optimization (MGPO): the Signal half of SSP.

This module holds the weight that turns a *group-relative advantage* into a
*group-relative advantage weighted by how uncertain the policy was on that group*.
It is deliberately not an RL algorithm, not an optimizer, and not a trainer: it is
one scalar weight function plus the arithmetic around it, so that the same array is
usable wherever a per-group learning signal is formed.

The weighting
-------------
For a query ``q``, sample ``G`` responses and score them ``r_i``. The empirical
success rate is

.. math::  p_c(q) = \\frac{1}{G} \\sum_{i=1}^{G} \\mathbb{1}[r_i = 1]

The paper's premise is that a problem's training value peaks where the policy is
maximally *uninformed* about it, i.e. at ``p_c = 0.5``, and falls off as ``p_c``
approaches either 0 or 1. The weight makes that explicit as a KL divergence from
the maximum-entropy Bernoulli ``p_0``:

.. math::

    D_{ME}(p_c \\| p_0) = D_{KL}\\big(\\mathrm{Bern}(p_c) \\,\\|\\, \\mathrm{Bern}(p_0)\\big)
        = p_c \\log\\frac{p_c}{p_0} + (1 - p_c) \\log\\frac{1 - p_c}{1 - p_0}

.. math::  w_{ME}(p_c) = \\exp\\big(-\\lambda \\cdot D_{ME}(p_c \\| p_0)\\big)

which multiplies the group-relative advantage directly, ``A'_j = w_ME(p_c) * A_j``.

The closed form that makes this cheap
------------------------------------
For a Bernoulli the KL against a Bernoulli is the ENTROPY DEFICIT, exactly:

.. math::

    D_{KL}(p \\| p_0) = -H(p) + H(p, p_0), \\quad H(p, p_0) = -p\\log p_0 - (1-p)\\log(1-p_0)

so at ``p_0 = 0.5`` the cross-entropy is the constant ``log 2`` and

.. math::

    D_{ME}(p \\| 0.5) = \\log 2 - H(p), \\qquad
    w_{ME}(p) = 2^{-\\lambda} \\cdot e^{\\lambda H(p)}

That identity is what this module's tests pin, and it is why ``w_ME`` is a strictly
MONOTONE INCREASING function of binary entropy. Three consequences follow, and all
three are asserted rather than asserted-in-prose:

* ``w_ME(0.5) == 1.0`` exactly -- maximum weight on maximum uncertainty;
* ``w_ME(0) == w_ME(1) == 2**-lambda`` -- the floor, exponentially separated from 1
  as ``lambda`` grows;
* ``lambda == 0`` gives ``w_ME == 1`` for every ``p``, i.e. the weighting vanishes
  and the objective is exactly the unweighted one.

The paper's printed distance is not this
----------------------------------------
The VibeThinker-1.5B technical report prints

    D_ME = p_c log(p_c / (1 - p_c)) + (1 - p_c) log(p0 / (1 - p0))

which is **not a KL divergence**. The second factor of the first term should be
``p0``, not ``(1 - p_c)``; as printed the expression is asymmetric, is not
minimised at ``p_c = p_0`` (it evaluates to ``0.5 log(0.5 / 0.5) = 0`` there by
accident but is negative at ``p_c = 0.1``), and cannot be the "distance from the
ideal maximum-entropy state" its own prose describes. This module implements the
Bernoulli KL above -- the reading that is zero exactly at ``p_c = p_0`` and grows
to ``log 2`` at ``p_c in {0, 1}``, which is what the prose describes -- and pins
the two against each other in ``tests/test_optimization/test_ssp/test_signal.py``.

Scope, stated plainly
---------------------
What is specified and implemented here is the ADVANTAGE REWEIGHTING, and nothing
else. The same report also claims MGPO "incentivizes the increased generation
probability of low-probability yet correct reasoning traces"; no formula is given
for that term anywhere in the paper, so it is NOT implemented and no such behaviour
should be attributed to this code.

Nothing in this module has been trained, benchmarked or A/B-tested in this
repository. The weights are arithmetic; whether focusing a policy update on
high-uncertainty groups improves a downstream metric is the paper's empirical
claim, and it is not verified here.

Scope of the generalisation
---------------------------
``w_ME`` is a per-group weight vector. It multiplies whatever advantage you already
have, so the same array serves:

* group-relative policy optimisation (advantages from reward),
* REINFORCE-with-group-baseline / RLOO / any group-relative estimator,
* a per-example curriculum weight in ordinary supervised fine-tuning, where the
  "group" is a set of sampled targets for one input and ``p_c`` is the fraction of
  them the model currently gets right (:func:`spectrum_sampling_weights` in
  ``ssp.spectrum`` is that use, pre-built).

References:
    - Xu, S., Zhou, Y., Wang, W., Min, J., Yin, Z., Dai, Y., Liu, S., Pang, L.,
      Chen, Y., & Zhang, J. (2025). Tiny Model, Big Logic: Diversity-Driven
      Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B.
      (Weibo technical report; the MGPO / max-entropy-deviation weighting.)
    - Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical
      Reasoning in Open Language Models. (GRPO; the group-relative advantage this
      reweights.)
    - Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. (the
      clipped surrogate :func:`mgpo_surrogate` implements.)
"""

from typing import Optional, Sequence, Tuple, Union

import keras
import numpy as np

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

from ..constants import (
    DEFAULT_MGPO_CLIP_EPS,
    DEFAULT_MGPO_DDOF,
    DEFAULT_MGPO_EPS,
    DEFAULT_MGPO_LAMBDA,
    DEFAULT_MGPO_NORMALIZE_WEIGHTS,
    DEFAULT_MGPO_P0,
    DEFAULT_SSP_TOKEN_REDUCTION,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: Token reductions accepted by :func:`mgpo_surrogate`.
TOKEN_REDUCTIONS: Tuple[str, ...] = (
    "per_sequence_mean",
    "per_token_mean",
    "none",
)

#: Policies for a group whose rewards have no spread.
ZERO_VARIANCE_POLICIES: Tuple[str, ...] = ("zero", "raise")

#: Channel order used by :func:`pack_mgpo_targets` on the trailing axis.
#: Index 0 carries the REFERENCE log-probabilities, index 1 the advantage.
MGPO_TARGET_CHANNELS: Tuple[str, str] = ("old_log_probs", "advantage")

#: Every tensor in this module is built in this dtype. Explicit, not inherited
#: from the global policy: an advantage is a gradient-direction selector, and
#: computing it in `mixed_float16` under an AMP policy loses the `log 1e-8`
#: resolution that keeps `binary_entropy(1e-7)` from reading as `-0.0`.
_COMPUTE_DTYPE: str = "float32"

# DECISION <ssp-2026-10-05>/D-002
# `0 * log(0)` is defined as 0, and it is evaluated as 0 -- NOT as `nan`, and not
# by clipping `p` into `[eps, 1 - eps]`. The clip is the tempting one-word version
# and it is wrong at both ends: `binary_entropy(0)` would return
# `-0 * log(eps)`-ish noise instead of exactly 0, so a fully-unsolved group would
# collect a small positive entropy, `max_entropy_weight` would treat it as slightly
# uncertain, and the "no signal here" rows would keep a non-zero weight forever.
# The `where` guard costs three extra ops and is exact at `p in {0, 1}`, which is
# where every saturated group in a real rollout batch sits.


# ---------------------------------------------------------------------
# dtype / validation helpers
# ---------------------------------------------------------------------


def _broadcast_per_row(value, target_shape, name: str) -> np.ndarray:
    """Broadcast a per-response value across a token axis.

    A ``(batch,)`` array cannot be broadcast to ``(batch, tokens)`` by either numpy
    or ``keras.ops``: broadcasting aligns axes from the TRAILING end, so it compares
    ``batch`` against ``tokens`` and fails whenever they differ. The per-response
    shape has to be promoted to ``(batch, 1)`` explicitly, which is what this does.

    Values that are already token-shaped, and ``(batch, 1)`` values, pass through.

    Args:
        value: The array to broadcast.
        target_shape: The ``(batch, tokens)`` shape to broadcast onto.
        name: Argument name, used in the error message.

    Returns:
        A ``float32`` numpy array of exactly ``target_shape``.

    Raises:
        ValueError: If the value is not broadcastable to ``target_shape`` after the
            1-D promotion.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    try:
        return np.ascontiguousarray(
            np.broadcast_to(array, tuple(target_shape)), dtype=np.float32
        )
    except ValueError as exc:
        raise ValueError(
            f"{name} of shape {array.shape} is not broadcastable to the token shape "
            f"{tuple(target_shape)}: {exc}"
        ) from exc


def _as_float(p, name: str = "p") -> keras.KerasTensor:
    """Convert to a float32 tensor for the entropy / KL arithmetic.

    Args:
        p: Anything ``keras.ops.convert_to_tensor`` accepts.
        name: Argument name, used in the error message.

    Returns:
        A ``float32`` tensor of the same shape.

    Raises:
        ValueError: If the value cannot be converted to a tensor.
    """
    try:
        return keras.ops.convert_to_tensor(p, dtype=_COMPUTE_DTYPE)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to a float tensor: {exc}") from exc


def _check_unit_interval(p: keras.KerasTensor, name: str) -> None:
    """Assert every entry of ``p`` lies in ``[0, 1]``.

    Args:
        p: The tensor to check.
        name: Argument name, used in the error message.

    Raises:
        ValueError: If any entry falls outside ``[0, 1]``.

    Note:
        The offending entry is reported with ``keras.ops.min`` over the masked
        array. Building the message by converting the whole tensor to a Python float
        -- which is the obvious one-liner -- raises ``TypeError`` for any array of
        more than one element, turning a clear argument error into an opaque backend
        one.
    """
    outside = keras.ops.logical_or(keras.ops.less(p, 0.0), keras.ops.greater(p, 1.0))
    if bool(keras.ops.any(outside)):
        offender = keras.ops.min(
            keras.ops.where(outside, p, keras.ops.ones_like(p))
        )
        raise ValueError(
            f"{name} must lie in [0, 1] -- it is read as a Bernoulli success "
            f"probability, and the most offending entry is {float(offender)}. "
            f"Threshold or normalise your rewards upstream."
        )


def _check_lambda(lam: float) -> float:
    """Validate the sharpening coefficient.

    Args:
        lam: Must be a finite, non-negative float.

    Returns:
        ``lam`` as a ``float``.

    Raises:
        ValueError: If ``lam`` is negative or not finite. A negative ``lam`` would
            INVERT the weighting -- up-weighting the groups the policy has already
            mastered -- which is a different, and unnamed, algorithm.
    """
    value = float(lam)
    if not np.isfinite(value):
        raise ValueError(f"lam must be finite; got {lam!r}")
    if value < 0.0:
        raise ValueError(
            f"lam must be >= 0; got {value}. A negative lam INVERTS the weighting "
            f"(maximum weight on p_c = 0 and p_c = 1, minimum at p_c = 0.5), which "
            f"is not the max-entropy guidance and has no name here."
        )
    return value


def _check_p0(p0: float) -> float:
    """Validate the target maximum-entropy Bernoulli parameter.

    Args:
        p0: Must lie strictly inside ``(0, 1)``.

    Returns:
        ``p0`` as a ``float``.

    Raises:
        ValueError: If ``p0`` is outside ``(0, 1)``. The endpoints define DEGENERATE
            Bernoullis, whose KL against a non-degenerate one is ``+inf``, so every
            weight would collapse to ``exp(-inf) = 0`` and the whole batch would be
            silently discarded.
    """
    value = float(p0)
    if not 0.0 < value < 1.0:
        raise ValueError(
            f"p0 must lie strictly inside (0, 1); got {value}. p0 = 0 and p0 = 1 "
            f"are degenerate Bernoullis: KL(Bern(p) || Bern(p0)) is +inf for every "
            f"p > 0 (resp. p < 1), so every weight underflows to 0."
        )
    return value


# ---------------------------------------------------------------------
# Entropy and the max-entropy deviation
# ---------------------------------------------------------------------


def binary_entropy(p, axis: Optional[int] = None) -> keras.KerasTensor:
    """Shannon entropy of a Bernoulli, in nats.

    .. math::  H(p) = -p\\log p - (1 - p)\\log(1 - p)

    Exact at both endpoints: ``H(0) == H(1) == 0``, and ``0 log 0`` is evaluated as
    ``0`` rather than as ``nan``.

    Args:
        p: Success probabilities. Any shape; every entry must lie in ``[0, 1]``.
        axis: If given, reduce over this axis before returning. ``None`` (default)
            returns the elementwise entropy.

    Returns:
        A ``float32`` tensor of entropies in ``[0, log 2]``, shaped like ``p``
        (reduced over ``axis`` when given).

    Raises:
        ValueError: If any entry of ``p`` lies outside ``[0, 1]``.

    Example:
        >>> import keras
        >>> from dl_techniques.optimization.ssp.signal import binary_entropy
        >>> float(keras.ops.convert_to_numpy(binary_entropy(0.5)))
        0.6931472
        >>> float(keras.ops.convert_to_numpy(binary_entropy(0.0)))
        0.0
    """
    probs = _as_float(p, "p")
    _check_unit_interval(probs, "p")

    # DECISION <ssp-2026-10-05>/D-002: the `where` guard, not a clip. See the
    # module-level note. `where(p > 0, p, 1.0)` keeps log() finite and multiplies
    # by p = 0 anyway, so the term is exactly 0 at both endpoints.
    log_p = keras.ops.log(keras.ops.where(probs > 0.0, probs, keras.ops.ones_like(probs)))
    log_q = keras.ops.log(
        keras.ops.where(probs < 1.0, 1.0 - probs, keras.ops.ones_like(probs))
    )
    entropy = -(probs * log_p + (1.0 - probs) * log_q)

    if axis is not None:
        entropy = keras.ops.mean(entropy, axis=axis)
    return entropy


def max_entropy_deviation(p, p0: float = DEFAULT_MGPO_P0) -> keras.KerasTensor:
    """``D_ME``: KL divergence from ``Bern(p)`` to the target ``Bern(p0)``.

    .. math::

        D_{KL}(p \\| p_0) = p \\log\\frac{p}{p_0}
                          + (1 - p) \\log\\frac{1 - p}{1 - p_0}

    At the default ``p0 = 0.5`` this is exactly the entropy deficit ``log 2 - H(p)``:
    zero precisely at maximum uncertainty, and ``log 2`` at either endpoint. The
    general two-parameter form is computed, not the ``p0 = 0.5`` special case, so
    that changing ``p0`` stays correct.

    Args:
        p: Success probabilities, in ``[0, 1]``.
        p0: Target probability, strictly inside ``(0, 1)``. Default ``0.5``, the
            maximum-entropy point.

    Returns:
        A ``float32`` tensor of divergences in ``[0, -H(p0)]``, shaped like ``p``.

    Raises:
        ValueError: If ``p`` leaves ``[0, 1]`` or ``p0`` is degenerate.

    Note:
        This is the CORRECTED form. The technical report prints
        ``p log(p / (1 - p)) + (1 - p) log(p0 / (1 - p0))``, whose first denominator
        should be ``p0``; see the module docstring for why the printed expression
        cannot be the quantity its own prose describes.
    """
    probs = _as_float(p, "p")
    target = _check_p0(p0)
    _check_unit_interval(probs, "p")

    target_t = keras.ops.convert_to_tensor(target, dtype=_COMPUTE_DTYPE)
    log_p0 = keras.ops.log(target_t)
    log_1mp0 = keras.ops.log1p(-target_t)

    log_p = keras.ops.log(keras.ops.where(probs > 0.0, probs, keras.ops.ones_like(probs)))
    # DECISION <ssp-2026-10-05>/D-006: `log` of `(1 - p)`, NOT `log1p` of `(1 - p)`.
    # `log1p(x)` is `log(1 + x)`, so `log1p(1 - p)` is `log(2 - p)`, which is neither
    # term of the Bernoulli KL. It is a one-character-looking slip with a large
    # effect and no error: at `p = 0` it returned `2 ln 2` instead of `ln 2`, and at
    # `p = 0.5` it returned `0.549` instead of `0`, so the weight peaked at the wrong
    # `p` and the "maximum weight at maximum uncertainty" property -- the entire
    # premise of the method -- silently did not hold. `binary_entropy` above uses
    # plain `log` on `(1 - p)` for the same reason; the two must agree.
    log_1mp = keras.ops.log(
        keras.ops.where(probs < 1.0, 1.0 - probs, keras.ops.ones_like(probs))
    )

    first = probs * (log_p - log_p0)
    second = (1.0 - probs) * (log_1mp - log_1mp0)
    return first + second


def max_entropy_weight(
        p,
        lam: float = DEFAULT_MGPO_LAMBDA,
        p0: float = DEFAULT_MGPO_P0,
) -> keras.KerasTensor:
    """``w_ME``: the max-entropy weight, ``exp(-lam * D_ME(p || p0))``.

    The single scalar function the Signal half is built on. At the default
    ``p0 = 0.5`` it satisfies, exactly:

    * ``w_ME(0.5) == 1.0``            -- maximum weight at maximum uncertainty;
    * ``w_ME(0) == w_ME(1) == 2**-lam`` -- the exponentially-separated floor;
    * ``lam == 0`` gives ``w_ME == 1`` everywhere, so the weighting disappears and
      the caller recovers the unweighted objective bit-for-bit.

    Equivalently, and only at ``p0 = 0.5``, ``w_ME(p) = 2**-lam * exp(lam * H(p))``
    -- a strictly increasing function of binary entropy.

    Args:
        p: Success probabilities, in ``[0, 1]``.
        lam: Sharpening coefficient, finite and ``>= 0``. ``0`` disables the
            weighting entirely. Larger values push harder toward ``p = p0``.
        p0: Target probability, strictly inside ``(0, 1)``. Default ``0.5``.

    Returns:
        A ``float32`` tensor of weights in ``[exp(-lam * -H(p0)), 1.0]``, shaped like
        ``p``.

    Raises:
        ValueError: If ``p`` leaves ``[0, 1]``, ``lam`` is negative or non-finite,
            or ``p0`` is degenerate.

    Example:
        >>> import keras
        >>> from dl_techniques.optimization.ssp.signal import max_entropy_weight
        >>> w = keras.ops.convert_to_numpy(max_entropy_weight([0.0, 0.5, 1.0], lam=2.0))
        >>> [round(float(x), 6) for x in w]        # [0.25, 1.0, 0.25]
        >>> float(keras.ops.convert_to_numpy(max_entropy_weight(0.3, lam=0.0)))
        1.0
    """
    lam_value = _check_lambda(lam)
    deviation = max_entropy_deviation(p, p0=p0)
    return keras.ops.exp(-lam_value * deviation)


# ---------------------------------------------------------------------
# Group statistics and advantages
# ---------------------------------------------------------------------


def group_success_rate(
        rewards,
        group_axis: int = -1,
) -> keras.KerasTensor:
    """``p_c``: the empirical success rate within each group of rollouts.

    For a binary verifier this is the fraction of correct rollouts. For a
    continuous reward in ``[0, 1]`` it is the mean reward, and the Bernoulli it
    induces is the object the max-entropy weighting is defined on -- no special
    case is needed anywhere downstream.

    Args:
        rewards: Rollout rewards. Any shape; the group axis is ``group_axis``.
        group_axis: The axis holding the ``G`` rollouts of one query. Default
            ``-1``.

    Returns:
        A ``float32`` tensor of per-group rates in ``[0, 1]``, with ``group_axis``
        removed.

    Raises:
        ValueError: If a reward lies outside ``[0, 1]``. Rewards outside the unit
            interval have no Bernoulli reading, and the weighted advantage would be
            a number with no interpretation.
    """
    values = _as_float(rewards, "rewards")
    _check_unit_interval(values, "rewards")
    return keras.ops.mean(values, axis=group_axis)


def group_relative_advantages(
        rewards,
        eps: float = DEFAULT_MGPO_EPS,
        ddof: int = DEFAULT_MGPO_DDOF,
        group_axis: int = -1,
        zero_variance: str = "zero",
) -> keras.KerasTensor:
    """Group-relative (GRPO-style) advantages: ``(r - mu) / (sigma + eps)``.

    No critic and no value network: the group mean is the baseline and the group
    spread is the scale. This is the advantage the Signal half reweights, and it is
    useful on its own.

    **The zero-variance group is handled by the ``eps`` in the denominator, not by
    a separate branch.** When every rollout in a group earns the same reward the
    numerator is exactly zero, so the advantage is ``0 / (0 + eps) == 0`` whatever
    the denominator does -- no division by zero, no ``nan``, and the group
    contributes no gradient. That is the whole of the ``zero_variance="zero"``
    policy, and it is why there is no masked-select in this function.
    ``zero_variance="raise"`` is the opt-in strictness for a caller that would
    rather know.

    Args:
        rewards: Rollout rewards, in ``[0, 1]``. Grouped along ``group_axis``.
        eps: Added to the standard deviation in the denominator, for stability.
            Default ``1e-6``.
        ddof: Delta degrees of freedom for the standard deviation. Default ``0``
            (population), which is defined for a group of size 1. Use ``1``
            (sample) only when every group has at least 2 rollouts.
        group_axis: The axis holding the ``G`` rollouts of one query. Default
            ``-1``.
        zero_variance: ``"zero"`` (default; degenerate groups get advantage ``0``)
            or ``"raise"``.

    Returns:
        A ``float32`` tensor shaped like ``rewards``.

    Raises:
        ValueError: If ``eps`` is not positive, ``ddof`` is negative or not smaller
            than the group size, ``zero_variance`` is unknown, or a reward leaves
            ``[0, 1]``.

    Note:
        Adding ``eps`` rather than clamping the denominator is what keeps the
        mapping continuous, but it also SHRINKS the advantage of a genuine
        low-variance group: with ``ddof=0`` and ``sigma`` genuinely near zero, the
        advantage is bounded by roughly ``G / 2``. That is a real property of the
        estimator, not a defect, and it is why ``zero_variance="raise"`` exists.
    """
    if zero_variance not in ZERO_VARIANCE_POLICIES:
        raise ValueError(
            f"Unknown zero_variance policy {zero_variance!r}. "
            f"Supported: {list(ZERO_VARIANCE_POLICIES)}"
        )
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError(
            f"eps must be > 0 so the zero-variance group cannot divide by zero; "
            f"got {eps_value}"
        )

    values = _as_float(rewards, "rewards")
    _check_unit_interval(values, "rewards")

    group_size = int(keras.ops.shape(values)[group_axis])
    ddof_value = int(ddof)
    if ddof_value < 0:
        raise ValueError(f"ddof must be >= 0; got {ddof_value}")
    if ddof_value >= group_size:
        raise ValueError(
            f"ddof={ddof_value} needs a group of at least {ddof_value + 1} "
            f"rollouts; the group axis {group_axis} has size {group_size}. Use "
            f"ddof=0 for a group of size 1."
        )

    mean = keras.ops.mean(values, axis=group_axis, keepdims=True)
    centred = values - mean
    variance = keras.ops.mean(keras.ops.square(centred), axis=group_axis, keepdims=True)
    stddev = keras.ops.sqrt(variance)

    if zero_variance == "raise":
        degenerate = keras.ops.any(stddev <= eps_value)
        if bool(degenerate):
            raise ValueError(
                f"zero_variance='raise' and at least one group has a reward "
                f"standard deviation <= eps={eps_value}. A group whose rollouts "
                f"all earn the same reward carries no relative signal; drop it "
                f"(zero_variance='zero') or raise eps."
            )

    # keepdims=True on both statistics, so the broadcast reproduces `rewards`'
    # shape and no axis bookkeeping is needed.
    return keras.ops.divide(centred, stddev + eps_value)


def mgpo_advantages(
        rewards,
        lam: float = DEFAULT_MGPO_LAMBDA,
        p0: float = DEFAULT_MGPO_P0,
        eps: float = DEFAULT_MGPO_EPS,
        ddof: int = DEFAULT_MGPO_DDOF,
        group_axis: int = -1,
        zero_variance: str = "zero",
        normalize_weights: bool = DEFAULT_MGPO_NORMALIZE_WEIGHTS,
) -> keras.KerasTensor:
    """The MGPO advantage: ``w_ME(p_c) * A_j``, token- and framework-agnostic.

    This is the Signal phase's complete contribution as a single array. It is
    deliberately independent of GRPO, of PPO, and of reinforcement learning: give
    it any group-relative advantage and it returns the reweighted one. In
    particular it is a valid per-example weight for ordinary supervised training,
    where the group is a set of sampled targets for one input.

    Args:
        rewards: Rollout rewards, in ``[0, 1]``, grouped along ``group_axis``.
        lam: Max-entropy sharpening coefficient. ``0`` reproduces the unweighted
            group-relative advantage exactly.
        p0: Target success probability, strictly inside ``(0, 1)``.
        eps: Denominator stabilizer for the advantage; see
            :func:`group_relative_advantages`.
        ddof: Delta degrees of freedom for the group standard deviation.
        group_axis: The axis holding the ``G`` rollouts of one query.
        zero_variance: ``"zero"`` or ``"raise"``; see
            :func:`group_relative_advantages`.
        normalize_weights: When ``True``, rescale the weights to mean 1 over the
            whole array before applying them, so the reweighting is purely RELATIVE
            and the expected gradient magnitude is preserved. Default ``False``,
            which is the paper's literal form: weights are ``<= 1``, so ambiguous
            groups keep full magnitude and already-decided groups are damped, and
            the batch's overall advantage scale shrinks with ``lambda``.

    Returns:
        A ``float32`` tensor shaped like ``rewards``.

    Raises:
        ValueError: On any problem with :func:`group_relative_advantages`,
            :func:`max_entropy_weight`, or a non-positive mean weight when
            ``normalize_weights`` is set.

    Note:
        With ``normalize_weights=True`` the weights are rescaled by their own mean,
        computed over EVERY element. For a batch in which most groups are saturated
        that mean is near the floor ``2**-lam``, so the rescaling multiplies by a
        large factor and the ambiguous groups -- the ones the method is about --
        end up with an advantage far above 1. Normalisation preserves the average,
        not the maximum; it is not a safety clamp.

    Note:
        **The returned advantages sum to zero within every group, always.** A
        group-relative advantage satisfies ``sum_i A_i = 0`` by construction, and the
        max-entropy weight is CONSTANT across the group (it depends only on that
        group's ``p_c``), so ``sum_i w(p_c) A_i = w(p_c) * 0 = 0``. Consequences worth
        knowing before you go looking for a bug:

        * the MEAN of this array over a complete batch is ``0.0`` for every
          ``lambda``, including ``lambda = 0``. A run whose logged loss sits at
          ``0.000`` is not broken;
        * the reweighting is therefore invisible in any batch-mean scalar and shows
          up only in the GRADIENT -- in how strongly each group pulls, relative to
          the others. Log the per-group weight, or the per-group advantage magnitude,
          not the mean loss;
        * it also means ``normalize_weights`` cannot change the batch mean, only
          the scale.
    """
    values = _as_float(rewards, "rewards")
    advantages = group_relative_advantages(
        values,
        eps=eps,
        ddof=ddof,
        group_axis=group_axis,
        zero_variance=zero_variance,
    )
    p_c = group_success_rate(values, group_axis=group_axis)
    weights = max_entropy_weight(p_c, lam=lam, p0=p0)

    if normalize_weights:
        mean_weight = keras.ops.mean(weights)
        weights = keras.ops.divide_no_nan(weights, mean_weight)

    # DECISION <ssp-2026-10-05>/D-007: re-insert the group axis as a singleton before
    # multiplying. `group_success_rate` drops it, so the weight vector is
    # `rewards.shape[:-1]` -- for a (batch, G) reward tensor that is (batch,), and
    # numpy/TF broadcast aligns TRAILING axes, so `(batch,) * (batch, G)` compares
    # `batch` against `G` and raises unless the two happen to be equal. Expanding at
    # `group_axis` puts the singleton exactly where the reduction removed it, so the
    # broadcast is correct for any group axis and any group size, including G == 1
    # (where the "group" is a single rollout and the advantage is 0 anyway).
    weights = keras.ops.expand_dims(weights, axis=group_axis)
    return weights * advantages


# ---------------------------------------------------------------------
# Clipped surrogate
# ---------------------------------------------------------------------


def mgpo_surrogate(
        log_probs,
        old_log_probs,
        advantages,
        clip_eps: float = DEFAULT_MGPO_CLIP_EPS,
        token_mask=None,
        reduction: str = DEFAULT_SSP_TOKEN_REDUCTION,
) -> Union[keras.KerasTensor, np.ndarray]:
    """The MGPO objective: a PPO clipped surrogate over MGPO-weighted advantages.

    .. math::

        r_{i,t} = \\exp\\!\\big(\\log \\pi_\\theta(y_{i,t} \\mid \\cdot)
                             - \\log \\pi_{old}(y_{i,t} \\mid \\cdot)\\big)

        \\mathcal{L} = -\\frac{1}{G} \\sum_{i=1}^{G} \\frac{1}{|y_i|}
            \\sum_{t} \\min\\big(r_{i,t} A'_{i,t},
            \\; \\mathrm{clip}(r_{i,t}, 1 \\pm \\varepsilon) A'_{i,t}\\big)

    with ``A'`` from :func:`mgpo_advantages`. Note the shape of the reduction: the
    paper averages over the tokens of each response FIRST and over the responses
    SECOND, so a long response does not dominate the batch. ``reduction`` makes that
    choice explicit; the default is the paper's.

    Args:
        log_probs: ``log pi_theta(y | q)`` under the CURRENT policy, ``(B, T)``.
        old_log_probs: ``log pi_old(y | q)``, ``(B, T)``, broadcastable to
            ``log_probs``.
        advantages: The MGPO advantage ``A'``. Either ``(B, 1)`` / ``(B,)``, which
            broadcasts one scalar advantage per response -- the standard GRPO case,
            where the advantage is constant across a response's tokens -- or
            ``(B, T)`` for a per-token advantage.
        clip_eps: The PPO clip range half-width, ``> 0``. Default ``0.2``.
        token_mask: Optional ``(B, T)`` mask; nonzero entries are summed, zero
            entries are skipped. Use it to drop padding and truncated positions.
            When ``None`` every position counts.
        reduction: ``"per_sequence_mean"`` (default; the paper's two-stage mean),
            ``"per_token_mean"`` (one flat mean over all unmasked tokens), or
            ``"none"`` (return the ``(B, T)`` token-level loss, unreduced).

    Returns:
        Under ``"per_sequence_mean"``: a ``(B,)`` tensor, one value per response.
        Under ``"per_token_mean"``: a scalar. Under ``"none"``: a ``(B, T)`` tensor.

    Raises:
        ValueError: If ``clip_eps`` is not positive, ``reduction`` is unknown,
            ``token_mask`` is not ``(B, T)``, a mask row is entirely zero (its
            ``1 / |y_i|`` would be a division by zero), or the three arrays are not
            mutually broadcastable.

    Note:
        The advantage is multiplied inside the ``min``, so it cannot be factored
        out and handed to Keras as a ``sample_weight``: the clip changes which
        branch wins depending on the SIGN of the advantage, and a factored form
        would silently drop that. :class:`MGPOObjective` therefore carries the
        advantage packed alongside the reference log-probs rather than separated
        into ``sample_weight``.

    Note:
        **A batch-mean of this objective is ``0.0`` for every ``lambda``**, and that
        is arithmetic rather than a defect. ``mgpo_advantages`` sums to zero within
        every group (see its note), so averaging the per-response losses over a
        complete batch cancels exactly. Read the per-response vector, or the
        gradient; a logged mean loss of zero carries no information about whether
        MGPO is doing anything.
    """
    if reduction not in TOKEN_REDUCTIONS:
        raise ValueError(
            f"Unknown reduction {reduction!r}. Supported: {list(TOKEN_REDUCTIONS)}"
        )
    clip_value = float(clip_eps)
    if not clip_value > 0.0:
        raise ValueError(f"clip_eps must be > 0; got {clip_value}")

    current = _as_float(log_probs, "log_probs")
    reference = _as_float(old_log_probs, "old_log_probs")

    if current.shape != reference.shape:
        raise ValueError(
            f"log_probs and old_log_probs must have the SAME shape; got "
            f"{tuple(current.shape)} and {tuple(reference.shape)}. They are read "
            f"position by position, so a broadcast here would mean comparing "
            f"different tokens."
        )
    # DECISION <ssp-2026-10-05>/D-007: `_broadcast_per_row` promotes a `(batch,)`
    # advantage to `(batch, 1)` before broadcasting. The group axis was removed by
    # `group_relative_advantages`, so the weight vector is `rewards.shape[:-1]`; for
    # a (batch, G) reward tensor that is `(batch,)`, and trailing-axis broadcasting
    # would compare `batch` against `G`. Expanding at the group axis there, and
    # promoting the trailing axis here, are the same defect in two directions.
    advantage = keras.ops.convert_to_tensor(
        _broadcast_per_row(advantages, current.shape, "advantages"),
        dtype=_COMPUTE_DTYPE,
    )

    if token_mask is None:
        mask = keras.ops.ones_like(current)
    else:
        mask = _as_float(token_mask, "token_mask")
        if tuple(mask.shape) != tuple(current.shape):
            raise ValueError(
                f"token_mask must have the token shape {tuple(current.shape)}; got "
                f"{tuple(mask.shape)}"
            )
        empty_rows = keras.ops.any(mask > 0.0, axis=-1)
        if not bool(keras.ops.all(empty_rows)):
            raise ValueError(
                f"at least one response has an all-zero token_mask. Its 1 / |y_i| "
                f"would be a division by zero, and a response with no scored token "
                f"is a caller bug (a truncated rollout with no usable prefix), not "
                f"something to average around."
            )

    ratio = keras.ops.exp(current - reference)
    unclipped = ratio * advantage
    clipped = keras.ops.clip(ratio, 1.0 - clip_value, 1.0 + clip_value) * advantage
    token_loss = -keras.ops.minimum(unclipped, clipped) * mask

    if reduction == "none":
        return token_loss

    if reduction == "per_token_mean":
        return keras.ops.sum(token_loss) / keras.ops.sum(mask)

    # "per_sequence_mean": mean over each response's scored tokens, then the caller
    # (or Keras' SUM_OVER_BATCH_SIZE) averages over responses. Returns (B,), which
    # is the shape `losses/AGENTS.md` requires of a keras.losses.Loss.call.
    lengths = keras.ops.sum(mask, axis=-1)
    return keras.ops.sum(token_loss, axis=-1) / lengths


# ---------------------------------------------------------------------
# Target packing for the Keras loss
# ---------------------------------------------------------------------


def pack_mgpo_targets(old_log_probs, advantage) -> np.ndarray:
    """Pack reference log-probs and advantage into one ``y_true`` array.

    :class:`MGPOObjective` needs three arrays (current log-probs, reference
    log-probs, advantage) but ``keras.losses.Loss.call`` is handed two plus an
    optional ``sample_weight`` -- and ``sample_weight`` cannot carry the advantage,
    because Keras multiplies by it *after* ``call`` returns, which would square it.
    So the reference log-probs and the advantage travel together on a trailing
    channel axis and ``sample_weight`` stays free for its actual job (masking whole
    responses).

    **The advantage is PER RESPONSE, i.e. one scalar for each row.** In a GRPO-shaped
    rollout the advantage ``A_i`` belongs to response ``i`` and is constant across
    that response's tokens, so a batch of ``B * G`` responses carries ``B * G``
    scalar advantages -- not a ``(B, G)`` array, which is one advantage per ROLLOUT
    and has no meaning against one response's token axis. Flatten first:

    .. code-block:: python

        adv = mgpo_advantages(rewards, lam=2.0)          # (B, G): one per rollout
        packed = pack_mgpo_targets(
            old_log_probs.reshape(B * G, T),
            np.asarray(adv).reshape(B * G),                # (B*G,): one per response
        )

    Args:
        old_log_probs: ``(batch, tokens)`` reference log-probabilities.
        advantage: ``(batch,)``, ``(batch, 1)`` or ``(batch, tokens)``. Non-singleton
            axes are broadcast against the token shape.

    Returns:
        A ``(batch, tokens, 2)`` ``float32`` array whose ``[..., 0]`` is
        ``old_log_probs`` and ``[..., 1]`` is the broadcast advantage.

    Raises:
        ValueError: If ``old_log_probs`` is not 2-D, or ``advantage`` is not
            broadcastable to ``old_log_probs``' shape. A ``(B, G)`` advantage against
            a ``(B, T)`` token shape is the common shape of that failure and is named
            in the message.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.optimization.ssp.signal import (
        ...     pack_mgpo_targets, unpack_mgpo_targets)
        >>> packed = pack_mgpo_targets(np.zeros((2, 3)), np.array([1.0, -1.0]))
        >>> packed.shape
        (2, 3, 2)
        >>> ref, adv = unpack_mgpo_targets(packed)
        >>> ref.shape, adv.shape
        ((2, 3), (2, 3))
    """
    reference = np.asarray(old_log_probs, dtype=np.float32)
    if reference.ndim != 2:
        raise ValueError(
            f"old_log_probs must be 2-D (batch, tokens); got shape "
            f"{reference.shape}"
        )
    weights = np.asarray(advantage, dtype=np.float32)
    # The per-ROLLOUT mis-shape is worth naming: it is the most likely first
    # mistake, and the generic broadcast message does not point at it. A `(B, 1)`
    # advantage is the LEGITIMATE per-response form, so the test needs `> 1`.
    if (
        weights.ndim == 2
        and weights.shape[0] == reference.shape[0]
        and weights.shape[1] > 1
        and weights.shape[1] != reference.shape[1]
    ):
        raise ValueError(
            f"advantage of shape {weights.shape} against a token shape of "
            f"{reference.shape} is the per-ROLLOUT array straight out of "
            f"mgpo_advantages, but each response needs its OWN scalar advantage. "
            f"Flatten the group on both sides: "
            f"old_log_probs.reshape(-1, {reference.shape[1]}) with "
            f"advantage.reshape(-1)."
        )
    weights = _broadcast_per_row(weights, reference.shape, "advantage")
    return np.stack([reference, weights], axis=-1).astype(np.float32)


def unpack_mgpo_targets(y_true) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_mgpo_targets`.

    Args:
        y_true: A ``(B, T, 2)`` packed array.

    Returns:
        A ``(old_log_probs, advantage)`` tuple of shapes ``(B, T)`` and ``(B, T)``.
        The advantage comes back FULLY BROADCAST to the token shape, which is the
        lossless form: the per-row rank it was packed from is not recoverable from
        the packed array, and a ``(B, T)`` advantage is a valid input to
        :func:`mgpo_surrogate` either way. Feed it back to
        :func:`pack_mgpo_targets` for an exact round trip.

    Raises:
        ValueError: If the last axis is not of length 2.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.optimization.ssp.signal import (
        ...     pack_mgpo_targets, unpack_mgpo_targets)
        >>> reference, advantage = unpack_mgpo_targets(
        ...     pack_mgpo_targets(np.zeros((2, 3)), np.array([1.0, -1.0])))
        >>> reference.shape, advantage.shape
        ((2, 3), (2, 3))
    """
    packed = np.asarray(y_true, dtype=np.float32)
    if packed.ndim != 3 or packed.shape[-1] != len(MGPO_TARGET_CHANNELS):
        raise ValueError(
            f"packed targets must be (batch, tokens, {len(MGPO_TARGET_CHANNELS)}); "
            f"got shape {packed.shape}. Build them with pack_mgpo_targets()."
        )
    return np.ascontiguousarray(packed[..., 0]), np.ascontiguousarray(packed[..., 1])


# ---------------------------------------------------------------------
# The Keras loss
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.optimization.ssp.signal")
class MGPOObjective(keras.losses.Loss):
    """MaxEnt-Guided Policy Optimization objective as a ``keras.losses.Loss``.

    The clipped surrogate of :func:`mgpo_surrogate`, shaped for
    ``model.compile(loss=MGPOObjective(...))`` inside a custom ``train_step`` that
    has already sampled rollouts, scored them and built the advantage with
    :func:`mgpo_advantages`.

    **Tensor contract.** ``y_true`` is the packed ``(B, T, 2)`` array from
    :func:`pack_mgpo_targets` (``[..., 0]`` = reference log-probs, ``[..., 1]`` =
    MGPO advantage); ``y_pred`` is ``(B, T)`` current-policy log-probs;
    ``sample_weight`` is an optional ``(B,)`` per-response weight whose zero entries
    exclude a response from the batch (a degenerate group, a truncated rollout).
    ``call`` returns ``(B,)`` -- one value per response, the shape
    ``dl_techniques/losses/AGENTS.md`` requires -- and Keras' default
    ``sum_over_batch_size`` reduction then divides by ``B``, which is exactly the
    paper's outer ``1 / G``. Because ``reduction="none"`` would leave the per-token
    ``(B, T)`` array un-reduced, this class deliberately does not expose a
    ``reduction`` knob at all.

    **``__call__`` is overridden because Keras 3 does not pass ``sample_weight`` to
    ``call``.** ``keras.losses.Loss.__call__`` invokes ``self.call(y_true, y_pred)``
    with two arguments and applies ``sample_weight`` itself, afterwards, inside
    ``reduce_weighted_values``. So a shape check written inside ``call`` is dead code
    -- which is exactly how this class shipped its first draft, with a validation
    branch that could never fire. Validating in ``__call__`` is the only place the
    argument is visible; ``dino_loss.iBOTPatchLoss`` overrides ``__call__`` for the
    same reason.

    **Calling it: ``__call__`` reduces, ``call`` does not.** ``loss(packed, new)``
    returns a SCALAR -- the batch mean. Call ``loss.call(packed, new)`` to get the
    per-response ``(B,)`` vector that ``sample_weight`` selects rows from, and which
    is the only place the max-entropy weighting is visible. See the note on the
    always-zero batch mean below.

    **No trainer ships with this.** An MGPO run needs a rollout sampler and a
    verifier, neither of which this repository has. The loss is the objective, not
    the algorithm; ``tests/test_optimization/test_ssp/test_signal.py`` checks its
    arithmetic against the pure function, not against a training run.

    Args:
        clip_eps: PPO clip range half-width, ``> 0``. Default ``0.2``.
        name: Keras object name. Default ``"mgpo_objective"``.
        dtype: Compute dtype for the internal arithmetic. The loss itself runs in
            float32 regardless, because a policy ratio is a difference of log-probs
            and half precision has no headroom for it.

    Example:
        >>> import numpy as np, keras
        >>> from dl_techniques.optimization.ssp.signal import (
        ...     MGPOObjective, mgpo_advantages, pack_mgpo_targets)
        >>> B, T, G = 2, 6, 4                       # 2 queries, 4 rollouts each
        >>> rewards = np.array([[1, 1, 0, 0],      # p_c = 0.5 -> weight 1.0
        ...                      [1, 1, 1, 1]],     # p_c = 1.0 -> weight 2**-lam
        ...                     dtype="float32")
        >>> adv = np.asarray(mgpo_advantages(rewards, lam=2.0))       # (B, G)
        >>> adv.round(4)
        array([[ 1.,  1., -1., -1.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
        >>> # Flatten the group: each (query, rollout) pair is ONE response
        >>> # carrying ONE scalar advantage.
        >>> old = np.zeros((B * G, T), dtype="float32")
        >>> new = np.full((B * G, T), -0.1, dtype="float32")         # ratio ~ 0.905
        >>> loss = MGPOObjective(clip_eps=0.2)
        >>> packed = pack_mgpo_targets(old, adv.reshape(-1))
        >>> np.asarray(loss.call(packed, new)).round(4)   # ambiguous group pulls
        array([-0.9048, -0.9048,  0.9048,  0.9048,  0.    ,  0.    ,
                0.    ,  0.    ], dtype=float32)
        >>> float(loss(packed, new))                      # batch MEAN: 0.0 always
        0.0

    Note:
        The solved group's advantage is all zeros because every rollout in it earns
        the same reward, so ``sigma_G == 0`` and the group carries no relative signal
        at any ``lambda``. The weighting's effect is the RATIO between the two
        groups' magnitudes, which the batch mean cancels exactly. Log the weights or
        the per-response vector; the mean loss is structurally incapable of showing
        you anything.

    References:
        - Xu, S. et al. (2025). Tiny Model, Big Logic (Weibo). The weighting.
        - Shao, Z. et al. (2024). DeepSeekMath. GRPO; the advantage.
        - Schulman, J. et al. (2017). PPO. The clipped surrogate.
    """

    def __init__(
            self,
            clip_eps: float = DEFAULT_MGPO_CLIP_EPS,
            name: str = "mgpo_objective",
            dtype: Optional[str] = None,
            **kwargs,
    ) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)
        clip_value = float(clip_eps)
        if not clip_value > 0.0:
            raise ValueError(f"clip_eps must be > 0; got {clip_eps!r}")
        self.clip_eps = clip_value
        # Losses never compute in half precision here: `exp(log pi - log pi_old)`
        # is a difference of two log-probs and `mixed_float16` cannot hold it.
        self._dtype_policy = keras.mixed_precision.Policy("float32")

    def __call__(self, y_true, y_pred, sample_weight=None) -> keras.KerasTensor:
        """Invoke the loss, validating ``sample_weight``'s shape on the way in.

        Overridden because ``keras.losses.Loss.__call__`` calls ``self.call(y_true,
        y_pred)`` with TWO arguments and applies ``sample_weight`` itself, afterwards.
        A shape check inside ``call`` therefore never runs. This is the only place
        the argument is visible.

        :param y_true: Packed ``(B, T, 2)`` targets from :func:`pack_mgpo_targets`.
        :param y_pred: ``(B, T)`` current-policy log-probabilities.
        :param sample_weight: Optional ``(B,)`` per-response weight; a zero entry
            excludes that response.
        :return: The reduced scalar loss -- the batch mean.
        :raises ValueError: If ``sample_weight`` is not ``(B,)``.
        """
        if sample_weight is not None:
            weights = keras.ops.convert_to_tensor(sample_weight, dtype=_COMPUTE_DTYPE)
            if int(weights.shape[-1]) != int(y_pred.shape[0]):
                raise ValueError(
                    f"sample_weight must be per-RESPONSE, shape "
                    f"({int(y_pred.shape[0])},); got {tuple(weights.shape)}. A "
                    f"token-shaped weight would be folded into the mean Keras "
                    f"already takes over responses, double-counting the padding "
                    f"mask -- pass a token_mask through mgpo_surrogate instead."
                )
        return super().__call__(y_true, y_pred, sample_weight)

    def call(self, y_true, y_pred) -> keras.KerasTensor:
        """Compute the per-response MGPO loss.

        Two arguments, deliberately: that is the contract
        ``keras.losses.Loss.__call__`` calls with, and adding a third optional
        ``sample_weight`` parameter would be a branch that can never execute.

        :param y_true: Packed ``(B, T, 2)`` targets from :func:`pack_mgpo_targets`.
        :param y_pred: ``(B, T)`` current-policy log-probabilities.
        :return: ``(B,)`` float32 tensor, one value per response.
        :raises ValueError: If ``y_true`` is not ``(B, T, 2)`` or the token shapes
            disagree.
        """
        packed = keras.ops.convert_to_tensor(y_true, dtype=_COMPUTE_DTYPE)
        if packed.ndim != 3 or int(packed.shape[-1]) != len(MGPO_TARGET_CHANNELS):
            raise ValueError(
                f"y_true must be the packed (batch, tokens, "
                f"{len(MGPO_TARGET_CHANNELS)}) array from pack_mgpo_targets(); got "
                f"shape {tuple(packed.shape)}"
            )

        reference = packed[..., 0]
        advantage = packed[..., 1]
        current = keras.ops.convert_to_tensor(y_pred, dtype=_COMPUTE_DTYPE)
        if tuple(current.shape) != tuple(reference.shape):
            raise ValueError(
                f"y_pred must have the same token shape as y_true's channels; got "
                f"{tuple(current.shape)} against {tuple(reference.shape)}"
            )

        return mgpo_surrogate(
            current,
            reference,
            advantage,
            clip_eps=self.clip_eps,
            token_mask=None,
            reduction="per_sequence_mean",
        )

    def get_config(self) -> dict:
        """Return the constructor arguments needed to rebuild this loss.

        :return: A dict with ``clip_eps``, ``name`` and ``dtype``.
        """
        config = super().get_config()
        config.update({"clip_eps": self.clip_eps})
        return config
