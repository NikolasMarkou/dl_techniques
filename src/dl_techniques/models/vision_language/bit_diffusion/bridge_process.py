"""
Data-pipeline math of the bridge: noising, score targets, weightings, time sampling.

These functions run outside the network, in a ``tf.data`` pipeline. Given the
clean endpoints, packed token embeddings ``x_0`` at ``t = 0`` and a VAE image
latent ``x_1`` at ``t = 1``, and per-sample times, they produce the noisy bridge
tensor ``x_t ~ N(mean(t), var(t) I)``, the analytic score the model regresses
onto, and the per-sample weight of the squared error. The two directions do not
mix: each target has its own weighting and its own singular endpoint, so the
time samplers keep ``t`` inside ``[TIME_EPS, 1 - TIME_EPS]``. A flow-matching
baseline (straight-line interpolant, constant velocity, no weighting) sits
alongside. All math runs in at least float32.

.. code-block:: text

    direction  generates        target                 weight              singular
    reverse    text from image  score_target_reverse   dsm_weight_reverse  t = 0
    forward    image from text  score_target_forward   dsm_weight_forward  t = 1

References:
    - Upstream ``sde_utils_loss.py`` and ``time_sampling.py``.
      ``sample_bridge_x_t`` is ``sample_p_base_x_t_cond_x_0_x_1``
      specialised to ``t_prev=0, t_next=1``.
    - ``findings/source-model-semantics.md`` sections 1 and 2, the bridge
      kernel and the derivation of the two weightings.
"""

import keras
from typing import Any, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .config import TIME_EPS
# _as_working is the package's single never-narrow cast (D-010).
from .sde import BridgeSDE, _as_working, _expand_like, bridge_math_dtype

# ---------------------------------------------------------------------

#: Logit-normal mean from upstream's launch script; its function defaults go unused.
LOGIT_NORMAL_P_MEAN: float = 0.4
#: Logit-normal standard deviation from the same launch script.
LOGIT_NORMAL_P_STD: float = 0.7

# ---------------------------------------------------------------------

def bridge_posterior_moments(
    sde: BridgeSDE, x_0: Any, x_1: Any, t: Any
) -> Tuple[Any, Any]:
    """Return the mean and variance of ``p(x_t | x_0, x_1)`` under the base process.

    The Brownian-bridge / Kalman-smoother posterior, specialised to endpoints
    anchored at ``t_prev = 0`` and ``t_next = 1``::

        mean = phi(0,t) x_0 + [C(0,t,1)/C(0,1,1)] (x_1 - phi(0,1) x_0)
        var  = C(0,t,t) - C(0,t,1)^2 / C(0,1,1)

    The variance is non-negative in exact arithmetic, but near ``t = 1`` it is a
    difference of two nearly equal numbers and goes slightly negative in
    float32 (about ``-3e-08``). It is clamped at zero, so a caller taking its
    square root gets 0, not NaN.

    :param sde: The base process supplying ``phi`` and ``C``.
    :type sde: BridgeSDE
    :param x_0: Endpoint at ``t = 0``, shape ``(B, ...)``.
    :type x_0: Any
    :param x_1: Endpoint at ``t = 1``, same shape as ``x_0``.
    :type x_1: Any
    :param t: Per-sample times, shape ``(B,)`` in ``[0, 1]``.
    :type t: Any
    :return: ``(mean, variance)``: ``mean`` shaped like ``x_0``, ``variance``
        shaped like ``t``, both in at least float32.
    :rtype: Tuple[Any, Any]
    :raises NotImplementedError: If ``sde`` has no covariance (``FlowMatchingODE``).
    """
    x_0, x_1, t = _as_working(x_0, x_1, t)
    zeros = keras.ops.zeros_like(t)
    ones = keras.ops.ones_like(t)

    c_t1 = sde.C(zeros, t, ones)
    c_11 = sde.C(zeros, ones, ones)
    gain = c_t1 / c_11

    mu_prior = _expand_like(sde.phi(zeros, t), x_0) * x_0
    mu_innovation = x_1 - _expand_like(sde.phi(zeros, ones), x_0) * x_0
    mean = mu_prior + _expand_like(gain, x_0) * mu_innovation

    variance = sde.C(zeros, t, t) - c_t1 * c_t1 / c_11
    # DECISION plan-2026-09-02T094601-77d4a04e/D-011: keep this clamp at zero.
    # float32 round-off makes the difference negative and its sqrt NaN. See decisions.md.
    variance = keras.ops.maximum(variance, keras.ops.zeros_like(variance))
    return mean, variance

# ---------------------------------------------------------------------

def sample_bridge_x_t(
    sde: BridgeSDE,
    x_0: Any,
    x_1: Any,
    t: Any,
    seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
) -> Any:
    """Draw the noisy bridge tensor the network is trained to denoise.

    ``x_t ~ N(mean(t), variance(t) I)`` with the moments of
    :func:`bridge_posterior_moments`. This is upstream's
    ``sample_p_base_x_t_cond_x_0_x_1``.

    Pass ``seed`` inside a ``tf.data`` pipeline: with ``None`` the draw uses the
    global RNG state, whose order the pipeline's parallelism decides.

    :param sde: The base process.
    :type sde: BridgeSDE
    :param x_0: Endpoint at ``t = 0``, shape ``(B, ...)``.
    :type x_0: Any
    :param x_1: Endpoint at ``t = 1``, same shape.
    :type x_1: Any
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :param seed: An integer (stateless, reproducible) or a
        :class:`keras.random.SeedGenerator` (stateful, advances per call).
    :type seed: Optional[Union[int, keras.random.SeedGenerator]]
    :return: ``x_t``, shaped like ``x_0``, in at least float32.
    :rtype: Any
    :raises NotImplementedError: If ``sde`` has no covariance (``FlowMatchingODE``).
    """
    mean, variance = bridge_posterior_moments(sde, x_0, x_1, t)
    noise = keras.random.normal(
        keras.ops.shape(mean), dtype=mean.dtype, seed=seed
    )
    return mean + _expand_like(keras.ops.sqrt(variance), mean) * noise

# ---------------------------------------------------------------------

def score_target_forward(sde: BridgeSDE, x_t: Any, t: Any, x_1: Any) -> Any:
    """Return ``grad_{x_t} log p(x_1 | x_t)``, the forward (text to image) target.

    Upstream ``grad_wrt_x_t_log_p_base_x_1_cond_x_t``::

        shrink = phi(t, 1)
        score  = shrink / C(t,1,1) * (x_1 - shrink * x_t)

    The gradient is taken with respect to the conditioning variable of the
    kernel ``N(x_1; phi(t,1) x_t, C(t,1,1))``, which gives the leading
    ``phi(t,1)``. Pair it with :func:`dsm_weight_forward` (``C/phi``), not
    :func:`dsm_weight_reverse`.

    Singular at ``t = 1``, where ``C(1,1,1) = 0``. Nothing here clamps ``t``;
    keep it at most ``1 - TIME_EPS``.

    :param sde: The base process.
    :type sde: BridgeSDE
    :param x_t: The noisy bridge tensor, shape ``(B, ...)``.
    :type x_t: Any
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :param x_1: The observed image-end endpoint, shaped like ``x_t``.
    :type x_1: Any
    :return: The target score, shaped like ``x_t``.
    :rtype: Any
    :raises NotImplementedError: If ``sde`` has no covariance (``FlowMatchingODE``).
    """
    x_t, t, x_1 = _as_working(x_t, t, x_1)
    ones = keras.ops.ones_like(t)
    shrink = sde.phi(t, ones)
    first_term = shrink / sde.C(t, ones, ones)
    second_term = x_1 - _expand_like(shrink, x_t) * x_t
    return _expand_like(first_term, x_t) * second_term

# ---------------------------------------------------------------------

def score_target_reverse(sde: BridgeSDE, x_t: Any, t: Any, x_0: Any) -> Any:
    """Return ``grad_{x_t} log p(x_t | x_0)``, the reverse (image to text) target.

    Upstream ``grad_wrt_x_t_log_p_base_x_t_cond_x_0``::

        score = -(x_t - phi(0,t) x_0) / C(0,t,t)

    This is the textbook denoising-score-matching target (Vincent 2011), the
    score of the forward marginal. Pair it with :func:`dsm_weight_reverse`, the
    perturbation variance, not :func:`dsm_weight_forward`.

    Singular at ``t = 0``, where ``C(0,0,0) = 0``. Nothing here clamps ``t``;
    keep it at least ``TIME_EPS``.

    :param sde: The base process.
    :type sde: BridgeSDE
    :param x_t: The noisy bridge tensor, shape ``(B, ...)``.
    :type x_t: Any
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :param x_0: The observed text-end endpoint, shaped like ``x_t``.
    :type x_0: Any
    :return: The target score, shaped like ``x_t``.
    :rtype: Any
    :raises NotImplementedError: If ``sde`` has no covariance (``FlowMatchingODE``).
    """
    x_t, t, x_0 = _as_working(x_t, t, x_0)
    zeros = keras.ops.zeros_like(t)
    left_term = -1.0 / sde.C(zeros, t, t)
    right_term = x_t - _expand_like(sde.phi(zeros, t), x_t) * x_0
    return _expand_like(left_term, x_t) * right_term

# ---------------------------------------------------------------------

def dsm_weight_forward(sde: BridgeSDE, t: Any) -> Any:
    """Return ``C(t,1,1) / phi(t,1)``, the weighting for :func:`score_target_forward`.

    Not :func:`dsm_weight_reverse`: the two weightings normalize two
    different conditional densities. The forward target is the gradient of
    ``p(x_1 | x_t)`` with respect to the conditioning variable, so it carries a
    leading ``phi(t,1)/C(t,1,1)``; cancelling that requires ``C(t,1,1)/phi(t,1)``.
    The reverse target is the gradient of ``p(x_t | x_0)`` with respect to the
    sampled variable and carries only ``1/C(0,t,t)``, so its weighting is the
    plain variance ``C(0,t,t)``. Neither expression can be obtained from the
    other by relabelling times; they are not interchangeable.

    The ``/ phi(t,1)`` division is a no-op for a driftless process
    (``phi == 1``), so dropping it shows only on drifting ones. At ``t = 1``
    this returns exactly 0, silently, although the target is non-finite there.

    :param sde: The base process.
    :type sde: BridgeSDE
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :return: The per-sample weighting, shaped like ``t``.
    :rtype: Any
    :raises NotImplementedError: If ``sde`` has no covariance (``FlowMatchingODE``).
    """
    (t,) = _as_working(t)
    ones = keras.ops.ones_like(t)
    return sde.C(t, ones, ones) / sde.phi(t, ones)

# ---------------------------------------------------------------------

def dsm_weight_reverse(sde: BridgeSDE, t: Any) -> Any:
    """Return ``C(0,t,t)``, the weighting for :func:`score_target_reverse`.

    Not :func:`dsm_weight_forward`. This is the perturbation variance of the
    forward marginal ``p(x_t | x_0)``, the standard denoising score-matching
    weighting; it has no ``phi`` factor because the reverse target
    differentiates the sampled variable, not the conditioning one. Pairing it
    with the forward target mis-scales the loss at every ``t`` with no NaN, no
    shape error, and a loss that still descends.

    At ``t = 0`` this returns exactly 0 and silently zeroes the loss; both time
    samplers keep ``t`` at least ``TIME_EPS``.

    :param sde: The base process.
    :type sde: BridgeSDE
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :return: The per-sample weighting, shaped like ``t``.
    :rtype: Any
    :raises NotImplementedError: If ``sde`` has no covariance (``FlowMatchingODE``).
    """
    (t,) = _as_working(t)
    zeros = keras.ops.zeros_like(t)
    return sde.C(zeros, t, t)

# ---------------------------------------------------------------------

def flow_matching_interpolant(x_0: Any, x_1: Any, t: Any) -> Any:
    """Return ``(1 - t) x_0 + t x_1``, the deterministic rectified-flow interpolant.

    The flow-matching baseline (D-002) has no base process: no noise and no
    covariance. This is its whole noising step.

    :param x_0: Endpoint at ``t = 0``, shape ``(B, ...)``.
    :type x_0: Any
    :param x_1: Endpoint at ``t = 1``, same shape.
    :type x_1: Any
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :return: ``x_t`` on the straight line between the endpoints, in at least
        float32.
    :rtype: Any
    """
    x_0, x_1, t = _as_working(x_0, x_1, t)
    t_b = _expand_like(t, x_0)
    return (1.0 - t_b) * x_0 + t_b * x_1

# ---------------------------------------------------------------------

def flow_matching_target(x_0: Any, x_1: Any) -> Any:
    """Return ``x_1 - x_0``, the rectified-flow velocity, for both directions.

    Time-independent and identical in both directions: the straight-line path
    has constant velocity, so there is no direction-specific target and no
    weighting (upstream's ``flow_matching_loss`` takes a plain unweighted mean).

    :param x_0: Endpoint at ``t = 0``, shape ``(B, ...)``.
    :type x_0: Any
    :param x_1: Endpoint at ``t = 1``, same shape.
    :type x_1: Any
    :return: The constant velocity, shaped like ``x_0``.
    :rtype: Any
    """
    x_0, x_1 = _as_working(x_0, x_1)
    return x_1 - x_0

# ---------------------------------------------------------------------

def sample_timesteps_uniform(
    batch: int,
    seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    dtype: str = "float32",
) -> Any:
    """Draw uniform times on ``[TIME_EPS, 1 - TIME_EPS]``.

    The window comes from the affine map ``u * (1 - 2 eps) + eps``, as upstream;
    there is no ``clip``. Without the window, a draw near 0 would divide
    :func:`score_target_reverse` by ``C(0,0,0) = 0``.

    :param batch: Number of times to draw.
    :type batch: int
    :param seed: Integer seed or :class:`keras.random.SeedGenerator`.
    :type seed: Optional[Union[int, keras.random.SeedGenerator]]
    :param dtype: Requested dtype; floored at ``float32`` (see D-010).
    :type dtype: str
    :return: Times, shape ``(batch,)``.
    :rtype: Any
    :raises ValueError: If ``batch`` is not positive.
    """
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    work = bridge_math_dtype(dtype)
    u = keras.random.uniform((batch,), dtype=work, seed=seed)
    # DECISION plan-2026-09-02T094601-77d4a04e/D-012: no clip on top of this map.
    # Two redundant range guards let either be deleted unnoticed. See decisions.md.
    return u * (1.0 - 2.0 * TIME_EPS) + TIME_EPS

# ---------------------------------------------------------------------

def sample_timesteps_logit_normal(
    batch: int,
    p_mean: float = LOGIT_NORMAL_P_MEAN,
    p_std: float = LOGIT_NORMAL_P_STD,
    seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    dtype: str = "float32",
) -> Any:
    """Draw logit-normal times, clipped to ``[TIME_EPS, 1 - TIME_EPS]``.

    ``t = sigmoid(z * p_std + p_mean)`` with ``z ~ N(0, 1)``, concentrating the
    training budget in the middle of the bridge. The defaults are upstream's
    launch-script values (``0.4`` / ``0.7``), not its function defaults.

    The ``clip`` is the only range guard here, as upstream. At the defaults it
    practically never fires; with a wide ``p_std`` it does. A ``p_std`` of 0
    logs a warning and every ``t`` equals ``sigmoid(p_mean)``; a negative one
    also warns but draws the same distribution as ``|p_std|``.

    :param batch: Number of times to draw.
    :type batch: int
    :param p_mean: Mean of the pre-sigmoid normal.
    :type p_mean: float
    :param p_std: Standard deviation of the pre-sigmoid normal. Not validated
        beyond the warning.
    :type p_std: float
    :param seed: Integer seed or :class:`keras.random.SeedGenerator`.
    :type seed: Optional[Union[int, keras.random.SeedGenerator]]
    :param dtype: Requested dtype; floored at ``float32`` (see D-010).
    :type dtype: str
    :return: Times, shape ``(batch,)``.
    :rtype: Any
    :raises ValueError: If ``batch`` is not positive.
    """
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if p_std <= 0.0:
        logger.warning(
            "sample_timesteps_logit_normal called with p_std=%s; a non-positive "
            "spread collapses every sampled time onto sigmoid(p_mean).",
            p_std,
        )
    work = bridge_math_dtype(dtype)
    z = keras.random.normal((batch,), dtype=work, seed=seed)
    t = keras.ops.sigmoid(z * p_std + p_mean)
    return keras.ops.clip(t, TIME_EPS, 1.0 - TIME_EPS)

# ---------------------------------------------------------------------
