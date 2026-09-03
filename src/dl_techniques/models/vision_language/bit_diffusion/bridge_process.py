"""The data-pipeline math of the bridge: noising, targets, weightings, time.

Everything in this module runs outside the network. Given the two clean
endpoints and a time, it produces the noisy bridge tensor the model sees, the
analytic score the model is regressed onto, and the scalar the squared error
is multiplied by. A ``tf.data`` pipeline is the only production caller.

.. code-block:: text

    t = 0                            t                             t = 1
    x_0                              |                               x_1
    (packed token embeddings)        |          (VAE image latent)
      o------------------------------o--------------------------------o
      |            x_t ~ N( mu(t), s^2(t) I )                          |
      |<----------- reverse direction: generate text from image ------>|
      |   target = score_target_reverse, weight = dsm_weight_reverse   |
      |            singular at t = 0                                  |
      |<----------- forward direction: generate image from text ------>|
      |   target = score_target_forward, weight = dsm_weight_forward   |
      |            singular at t = 1                                  |
      +---------------------------------------------------------------+

Both time samplers draw ``t`` in ``[TIME_EPS, 1 - TIME_EPS]`` so neither
singular endpoint is reached. The two weightings are the natural weightings
of two different conditional densities and are not interchangeable: the
reverse target differentiates the sampled variable and its weighting is a
plain variance, ``C(0,t,t)``; the forward target differentiates the
conditioning variable and carries an extra ``phi(t,1)`` factor, so its
weighting is ``C(t,1,1)/phi(t,1)`` instead. See each function's own
docstring for the derivation.

References:
    - Upstream ``sde_utils_loss.py`` and ``time_sampling.py``.
      ``sample_bridge_x_t`` is ``sample_p_base_x_t_cond_x_0_x_1``
      specialised to ``t_prev=0, t_next=1``.
    - ``findings/source-model-semantics.md`` sections 1 and 2, the bridge
      kernel and the derivation of the two weightings.
"""

from typing import Any, Optional, Tuple, Union

import keras

from dl_techniques.utils.logger import logger

from .config import TIME_EPS
# _as_working is imported, not reimplemented -- it is the one copy of the
# never-narrow cast this package uses (D-010).
from .sde import BridgeSDE, _as_working, _expand_like, bridge_math_dtype

#: Upstream's shipped logit-normal time-sampling parameters, read off the launch
#: script rather than the function's own defaults (which upstream never uses).
LOGIT_NORMAL_P_MEAN: float = 0.4
LOGIT_NORMAL_P_STD: float = 0.7


def bridge_posterior_moments(
    sde: BridgeSDE, x_0: Any, x_1: Any, t: Any
) -> Tuple[Any, Any]:
    """Mean and variance of ``p(x_t | x_0, x_1)`` under the base process.

    The Brownian-bridge / Kalman-smoother posterior, specialised to endpoints
    anchored at ``t_prev = 0`` and ``t_next = 1``::

        mean = phi(0,t) x_0 + [C(0,t,1)/C(0,1,1)] (x_1 - phi(0,1) x_0)
        var  = C(0,t,t) - C(0,t,1)^2 / C(0,1,1)

    The variance is a Gaussian conditional variance and is therefore
    non-negative in exact arithmetic, but the expression is a difference of two
    nearly equal quantities near ``t = 1`` and does go slightly negative in
    float32 -- measured at ``-2.98e-08`` for
    ``PeriodicVolatilitySDE(alpha=0.95, k=3.0, eps=1e-3)`` at ``t = 0.99875``,
    and at ``-7.45e-09`` for ``UniformVolatilitySDE(A=5.0, K=1.0)`` at ``t = 1``.
    It is clamped at zero before any square root; without the clamp the sampler
    returns NaN for those inputs. Both measured cases are asserted in
    ``test_the_bridge_process_math.py``.

    :param sde: The base process supplying ``phi`` and ``C``.
    :type sde: BridgeSDE
    :param x_0: Endpoint at ``t = 0``, shape ``(B, ...)``.
    :type x_0: Any
    :param x_1: Endpoint at ``t = 1``, same shape as ``x_0``.
    :type x_1: Any
    :param t: Per-sample times, shape ``(B,)`` in ``[0, 1]``.
    :type t: Any
    :return: ``(mean, variance)`` -- ``mean`` shaped like ``x_0``, ``variance``
        shaped like ``t``, both in the never-narrow working dtype.
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
    # DECISION plan-2026-09-02T094601-77d4a04e/D-011: keep this clamp -- float32
    # round-off measurably makes this difference negative (-2.98e-08 at k=3/eps=1e-3, t=0.99875), and the unclamped sqrt returns NaN. See decisions.md.
    variance = keras.ops.maximum(variance, keras.ops.zeros_like(variance))
    return mean, variance


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

    ``seed`` is threaded explicitly and is not optional in practice: step 9 calls
    this inside a ``tf.data`` pipeline, where global RNG state is unreliable
    because the pipeline's own parallelism decides execution order.

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
    :return: ``x_t``, shaped like ``x_0``, in the never-narrow working dtype.
    :rtype: Any
    """
    mean, variance = bridge_posterior_moments(sde, x_0, x_1, t)
    noise = keras.random.normal(
        keras.ops.shape(mean), dtype=mean.dtype, seed=seed
    )
    return mean + _expand_like(keras.ops.sqrt(variance), mean) * noise


def score_target_forward(sde: BridgeSDE, x_t: Any, t: Any, x_1: Any) -> Any:
    """``grad_{x_t} log p(x_1 | x_t)``, the forward (text -> image) target.

    Upstream ``grad_wrt_x_t_log_p_base_x_1_cond_x_t``::

        shrink = phi(t, 1)
        score  = shrink / C(t,1,1) * (x_1 - shrink * x_t)

    The gradient is taken with respect to the *conditioning* variable of the
    kernel ``N(x_1; phi(t,1) x_t, C(t,1,1))``, which is where the leading
    ``phi(t,1)`` comes from -- and why this direction's weighting is
    :func:`dsm_weight_forward` (``C/phi``) rather than a plain variance, not
    to be paired with :func:`dsm_weight_reverse`.

    Singular at ``t = 1``, where ``C(1,1,1) = 0``: clamp ``t`` to
    ``1 - TIME_EPS``.

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
    """
    x_t, t, x_1 = _as_working(x_t, t, x_1)
    ones = keras.ops.ones_like(t)
    shrink = sde.phi(t, ones)
    first_term = shrink / sde.C(t, ones, ones)
    second_term = x_1 - _expand_like(shrink, x_t) * x_t
    return _expand_like(first_term, x_t) * second_term


def score_target_reverse(sde: BridgeSDE, x_t: Any, t: Any, x_0: Any) -> Any:
    """``grad_{x_t} log p(x_t | x_0)``, the reverse (image -> text) target.

    Upstream ``grad_wrt_x_t_log_p_base_x_t_cond_x_0``::

        score = -(x_t - phi(0,t) x_0) / C(0,t,t)

    This is the textbook denoising-score-matching target (Vincent 2011): the
    score of the forward marginal, whose natural weighting is the perturbation
    variance :func:`dsm_weight_reverse`, not to be paired with
    :func:`dsm_weight_forward`.

    Singular at ``t = 0``, where ``C(0,0,0) = 0``: clamp ``t`` to ``TIME_EPS``.

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
    """
    x_t, t, x_0 = _as_working(x_t, t, x_0)
    zeros = keras.ops.zeros_like(t)
    left_term = -1.0 / sde.C(zeros, t, t)
    right_term = x_t - _expand_like(sde.phi(zeros, t), x_t) * x_0
    return _expand_like(left_term, x_t) * right_term


def dsm_weight_forward(sde: BridgeSDE, t: Any) -> Any:
    """``C(t,1,1) / phi(t,1)`` -- the weighting for :func:`score_target_forward`.

    Not :func:`dsm_weight_reverse`: the two weightings normalize two
    different conditional densities. The forward target is the gradient of
    ``p(x_1 | x_t)`` with respect to the conditioning variable, so it carries a
    leading ``phi(t,1)/C(t,1,1)``; cancelling that requires ``C(t,1,1)/phi(t,1)``.
    The reverse target is the gradient of ``p(x_t | x_0)`` with respect to the
    sampled variable and carries only ``1/C(0,t,t)``, so its weighting is the
    plain variance ``C(0,t,t)``. Neither expression can be obtained from the
    other by relabelling times; they are not interchangeable.

    Two traps this docstring exists to prevent:

    * The ``/ phi(t,1)`` division is a no-op on every driftless variant
      (``phi == 1``), so deleting it passes three of the four SDE families.
    * At ``t = 1`` this returns exactly ``0`` rather than blowing up. The
      endpoint failure is silent here even though the *target* is non-finite
      there.

    :param sde: The base process.
    :type sde: BridgeSDE
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :return: The per-sample weighting, shaped like ``t``.
    :rtype: Any
    """
    (t,) = _as_working(t)
    ones = keras.ops.ones_like(t)
    return sde.C(t, ones, ones) / sde.phi(t, ones)


def dsm_weight_reverse(sde: BridgeSDE, t: Any) -> Any:
    """``C(0,t,t)`` -- the weighting for :func:`score_target_reverse`.

    Not :func:`dsm_weight_forward`. This is the perturbation
    variance of the forward marginal ``p(x_t | x_0)``, the standard denoising
    score-matching weighting; it carries no ``phi`` factor because the reverse
    target differentiates the *sampled* variable, not the conditioning one. See
    :func:`dsm_weight_forward` for the other half of the argument. Pairing this
    with the forward target mis-scales the loss at every ``t`` without producing
    a NaN, a shape error, or a loss that fails to descend.

    At ``t = 0`` this returns exactly ``0``, silently annihilating the loss --
    which is why both time samplers clamp to ``TIME_EPS``.

    :param sde: The base process.
    :type sde: BridgeSDE
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :return: The per-sample weighting, shaped like ``t``.
    :rtype: Any
    """
    (t,) = _as_working(t)
    zeros = keras.ops.zeros_like(t)
    return sde.C(zeros, t, t)


def flow_matching_interpolant(x_0: Any, x_1: Any, t: Any) -> Any:
    """``(1 - t) x_0 + t x_1`` -- the deterministic rectified-flow interpolant.

    The flow-matching baseline (D-002) has no base process, so it never touches
    ``sigma``/``phi``/``C``: there is no noise to add and no covariance to
    condition on. This is the whole of its "noising" step.

    :param x_0: Endpoint at ``t = 0``, shape ``(B, ...)``.
    :type x_0: Any
    :param x_1: Endpoint at ``t = 1``, same shape.
    :type x_1: Any
    :param t: Per-sample times, shape ``(B,)``.
    :type t: Any
    :return: ``x_t`` on the straight line between the endpoints.
    :rtype: Any
    """
    x_0, x_1, t = _as_working(x_0, x_1, t)
    t_b = _expand_like(t, x_0)
    return (1.0 - t_b) * x_0 + t_b * x_1


def flow_matching_target(x_0: Any, x_1: Any) -> Any:
    """``x_1 - x_0``, the rectified-flow velocity, for both directions.

    Time-independent, and the same tensor in the forward and reverse
    directions: the straight-line path has constant velocity, so there is no
    direction-specific target and no weighting at all (upstream's
    ``flow_matching_loss`` takes a plain unweighted mean). Adding a ``t``
    argument here would invent a knob upstream does not have.

    :param x_0: Endpoint at ``t = 0``, shape ``(B, ...)``.
    :type x_0: Any
    :param x_1: Endpoint at ``t = 1``, same shape.
    :type x_1: Any
    :return: The constant velocity, shaped like ``x_0``.
    :rtype: Any
    """
    x_0, x_1 = _as_working(x_0, x_1)
    return x_1 - x_0


def sample_timesteps_uniform(
    batch: int,
    seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    dtype: str = "float32",
) -> Any:
    """Uniform times on ``[TIME_EPS, 1 - TIME_EPS]``.

    The window is produced by the affine map ``u * (1 - 2 eps) + eps``, exactly
    as upstream does it, not by a ``clip`` afterwards: with both mechanisms
    present, deleting either one would leave the range intact and no test
    could tell that the guard had been removed.

    A plain ``uniform(0, 1)`` draw of 200,000 samples lands below ``1e-4``
    with probability ``1 - e^-20``, and a single such sample divides
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
    # DECISION plan-2026-09-02T094601-77d4a04e/D-012
    # DECISION plan-2026-09-02T094601-77d4a04e/D-012: no keras.ops.clip on top of
    # this affine map -- two redundant range mechanisms make either one deletable with the suite still green. See decisions.md.
    return u * (1.0 - 2.0 * TIME_EPS) + TIME_EPS


def sample_timesteps_logit_normal(
    batch: int,
    p_mean: float = LOGIT_NORMAL_P_MEAN,
    p_std: float = LOGIT_NORMAL_P_STD,
    seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    dtype: str = "float32",
) -> Any:
    """Logit-normal times, clipped to ``[TIME_EPS, 1 - TIME_EPS]``.

    ``t = sigmoid(z * p_std + p_mean)`` with ``z ~ N(0, 1)``, concentrating the
    training budget in the middle of the bridge. The defaults are upstream's
    shipped launch-script values (``0.4`` / ``0.7``), not the reference
    function's signature defaults, which upstream never uses.

    The ``clip`` is the only range mechanism here (upstream uses ``clip`` for
    this sampler and an affine map for the uniform one; both are ported as-is).
    At the shipped defaults it never fires -- reaching ``t < 1e-4`` would take a
    ``-13.7 sigma`` normal draw -- so the guard test exercises it at
    ``p_std = 6``, where it is the only thing holding the range.

    :param batch: Number of times to draw.
    :type batch: int
    :param p_mean: Mean of the pre-sigmoid normal.
    :type p_mean: float
    :param p_std: Standard deviation of the pre-sigmoid normal.
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
