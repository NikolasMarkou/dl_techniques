"""
Base processes of the text-image bridge, and the never-narrow dtype helper.

A bridge diffusion needs a base process: how a sample wanders between the two
anchored endpoints when nothing conditions it. Each process defines three closed
forms: the diffusion coefficient ``sigma(t)``; the drift transition
``phi(start, end) = exp(-A (end - start))``, identically 1 when driftless; and
the covariance ``C(start, t_a, t_b) = Cov(X_{t_a}, X_{t_b} | X_start)``. Both
analytic score targets divide by ``C``, which is exactly zero at the anchored
endpoint, so a wrong constant mistrains the model without crashing. The closed
forms run in :func:`bridge_math_dtype`: float64 if any input is, else float32.
Unlike upstream, the processes hold no network; :meth:`BridgeSDE.dX_t` and
:meth:`BridgeSDE.simulate` take it as an argument (D-009).
:func:`create_bridge_sde` builds a process by name:

.. code-block:: text

    sde_type       class                        sigma(t)
    uniform        UniformVolatilitySDE         K; OU drift when A != 0
    periodic       PeriodicVolatilitySDE        alpha/2 (1 - cos 2 pi k t) + eps
    cosine_decay   CosineDecayingVolatilitySDE  alpha/2 (1 + cos pi t) + eps
    flow_matching  FlowMatchingODE              none; sigma, phi and C raise

References:
    - Upstream ``sde_utils_sde.py``.
    - ``findings/source-model-semantics.md`` section 3, the formula table
      every closed form here is transcribed from, and section 4 for the
      sampler that consumes them.
"""

import math
import keras
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------

# DECISION plan-2026-09-02T094601-77d4a04e/D-010: the package's only copy of this rule.
# Every module imports it from here; do not add another. See decisions.md.
def bridge_math_dtype(*dtypes: Any) -> str:
    """Return the never-narrowing dtype the bridge closed forms run in.

    ``C`` is ``O(1e-4)`` near either endpoint and is divided by, so half
    precision is an accuracy hazard. The rule is a floor at float32 that keeps
    float64: any narrower, integer or other dtype gives float32.

    :param dtypes: Candidate dtypes, typically the input tensors' dtypes;
        ``None`` entries are skipped.
    :type dtypes: Any
    :return: ``"float64"`` if any candidate is float64, otherwise ``"float32"``.
    :rtype: str
    """
    for dtype in dtypes:
        if dtype is None:
            continue
        # DECISION plan-2026-09-02T094601-77d4a04e/D-015: read the name, no standardize_dtype.
        # That Keras-2 call is banned across src/. See decisions.md.
        if (getattr(dtype, "name", None) or str(dtype)) == "float64":
            return "float64"
    return "float32"

# ---------------------------------------------------------------------------

def _expand_like(scalars: Any, reference: Any) -> Any:
    """Reshape a per-sample ``(B,)`` vector to broadcast against ``reference``.

    ``t`` and everything derived from it is one value per sample, while the
    bridge tensor is rank 4. Broadcasting ``(B,)`` against ``(B, H, W, C)``
    without a reshape aligns it with the channel axis: a wrong answer with the
    right shape. Returns ``scalars`` unchanged when it is rank 0 or
    ``reference`` is rank at most 1, otherwise reshaped to ``(-1, 1, 1, ...)``
    with ``rank(reference) - 1`` trailing ones. Never changes dtype.

    :param scalars: A rank-0 or rank-1 tensor.
    :type scalars: Any
    :param reference: The tensor to broadcast against.
    :type reference: Any
    :return: ``scalars`` reshaped to ``(-1, 1, 1, ...)``, or unchanged.
    :rtype: Any
    """
    rank = len(keras.ops.shape(reference))
    if len(keras.ops.shape(scalars)) == 0 or rank <= 1:
        return scalars
    return keras.ops.reshape(scalars, (-1,) + (1,) * (rank - 1))

# ---------------------------------------------------------------------------

def _require_network(score_network: Any, caller: str) -> None:
    """Raise if ``score_network`` is ``None``; no other check is made.

    Without a network the sampler would integrate the base process with a zero
    score: finite, plausible and untrained.

    :param score_network: The candidate network.
    :type score_network: Any
    :param caller: Method name, for the message.
    :type caller: str
    :raises ValueError: If ``score_network`` is ``None``.
    """
    if score_network is None:
        raise ValueError(
            f"{caller} needs a score_network; this port deliberately does not "
            f"store one on the SDE object (see decisions.md D-009). Without it "
            f"the sampler would silently integrate the base process with a zero "
            f"score -- finite, plausible and completely untrained."
        )

# ---------------------------------------------------------------------------

def _time_grid(num_steps: int, reverse: bool) -> list:
    """Return the ``num_steps + 1`` integration times, endpoints included.

    ``linspace(1, 0)`` when reversing (image to text), ``linspace(0, 1)``
    otherwise, as Python floats, because the sampler loop needs each ``dt`` as
    a Python float.

    :param num_steps: Number of integration steps; the grid has one more entry.
    :type num_steps: int
    :param reverse: Whether to integrate from ``t = 1`` down to ``t = 0``.
    :type reverse: bool
    :return: The monotonically ordered times.
    :rtype: list
    """
    if reverse:
        return [1.0 - i / num_steps for i in range(num_steps + 1)]
    return [i / num_steps for i in range(num_steps + 1)]

# ---------------------------------------------------------------------------

def _as_working(*tensors: Any) -> Any:
    """Cast every tensor to the shared never-narrow working dtype.

    :param tensors: Tensors or array-likes.
    :type tensors: Any
    :return: A tuple of tensors, all in ``bridge_math_dtype`` of the inputs.
    :rtype: Any
    """
    converted = [keras.ops.convert_to_tensor(x) for x in tensors]
    dtype = bridge_math_dtype(*[x.dtype for x in converted])
    return tuple(keras.ops.cast(x, dtype) for x in converted)

# ---------------------------------------------------------------------------

# DECISION plan-2026-09-02T094601-77d4a04e/D-009: no score_network attribute.
# Storing one makes the object unserializable; dX_t/simulate take it. See decisions.md.
@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class BridgeSDE:
    """Define a bridge base process (``sigma``, ``phi``, ``C``) and sample with it.

    A plain Python object, not a Keras layer. Subclasses supply the three closed
    forms; the base raises for all of them, since a ``0.0`` default would let a
    half-ported variant train on a degenerate score. :meth:`dX_t` takes one
    Euler-Maruyama or probability-flow step and :meth:`simulate` loops it from
    one endpoint to the other.

    Sampling schedule of :meth:`simulate`, ``num_steps = 4``, reverse:

    .. code-block:: text

        i            0        1        2        3
        t      1.00     0.75     0.50     0.25     0.00
                x_1 ────► · ─────► · ─────► · ────► x_0
        step        SDE      ODE*     ODE*     ODE*

    ``ODE*`` is the probability-flow step when ``ode=True``, else SDE. Step 0 is
    always SDE: the analytic base score divides by ``C(start, t, t)``, which is
    0 at the anchored endpoint, so an ODE first step would turn the whole
    trajectory into NaN.

    :param A: Linear drift coefficient, not validated. ``0`` for every
        driftless variant; only ``UniformVolatilitySDE`` takes it non-zero.
    :type A: float
    """

    def __init__(self, A: float = 0.0) -> None:
        self.A = float(A)

    def sigma(self, t: Any) -> Any:
        """Diffusion coefficient at time ``t``.

        :param t: Times, any shape.
        :type t: Any
        :return: ``sigma(t)``, shaped like ``t``, in the never-narrow dtype.
        :rtype: Any
        :raises NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define the diffusion coefficient sigma"
        )

    def phi(self, start: Any, end: Any) -> Any:
        """Deterministic transition factor from ``start`` to ``end``.

        :param start: Start times.
        :type start: Any
        :param end: End times.
        :type end: Any
        :return: ``phi(start, end)`` in the never-narrow dtype.
        :rtype: Any
        :raises NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define the base-process transition phi"
        )

    def C(self, start: Any, t_a: Any, t_b: Any) -> Any:
        """Covariance ``Cov(X_{t_a}, X_{t_b} | X_start)`` of the base process.

        :param start: Conditioning time.
        :type start: Any
        :param t_a: First time.
        :type t_a: Any
        :param t_b: Second time. ``C`` is symmetric in ``t_a`` and ``t_b``.
        :type t_b: Any
        :return: The covariance in the never-narrow dtype.
        :rtype: Any
        :raises NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define the base-process covariance C"
        )

    def _time_vector(self, x_t: Any, t_value: float) -> Any:
        """Return a ``(B,)`` vector of ``t_value`` in the working dtype of ``x_t``."""
        work = bridge_math_dtype(x_t.dtype)
        return keras.ops.ones(keras.ops.shape(x_t)[:1], dtype=work) * t_value

    def _evaluate_score(
        self,
        score_network: Any,
        x_t: Any,
        t: Any,
        x_cond: Any,
        y: Any,
        reverse: bool,
        cfg_scale: float,
        training: Optional[bool],
        cond_mask: Optional[Any] = None,
    ) -> Any:
        """Call the network for the score, through CFG only when ``cfg_scale > 0``.

        ``direction`` is 1 for every sample when ``reverse``, else 0.

        :param cond_mask: Optional ``(B,)`` conditioning mask forwarded to the
            network. ``None`` (the default) omits the key, which the network
            reads as all-ones. Only :class:`FlowMatchingODE` passes one, to
            force the unconditional branch.
        :type cond_mask: Optional[Any]
        :return: The predicted score, cast to ``x_t``'s dtype.
        """
        direction = keras.ops.ones_like(t) if reverse else keras.ops.zeros_like(t)
        inputs = {
            "x_t": x_t,
            "t": t,
            "y": y,
            "x_cond": x_cond,
            "direction": direction,
        }
        if cond_mask is not None:
            inputs["cond_mask"] = cond_mask
        # DECISION plan-2026-09-02T094601-77d4a04e/D-018: keep the cfg_scale > 0 gate.
        # Ungated, every step runs the network twice, and s = 0 hides it. See decisions.md.
        if cfg_scale > 0:
            score = score_network.forward_with_cfg(
                inputs, cfg_scale=cfg_scale, training=training
            )
        else:
            score = score_network(inputs, training=training)
        return keras.ops.cast(score, x_t.dtype)

    def dX_t(
        self,
        x_t: Any,
        t: Any,
        x_cond: Any,
        y: Any,
        dt: float,
        score_network: Optional[Any] = None,
        reverse: bool = False,
        cfg_scale: float = 0.0,
        ode: bool = False,
        x_start: Optional[Any] = None,
        seed: Optional[Any] = None,
        training: Optional[bool] = False,
    ) -> Any:
        """Return one Euler-Maruyama (or probability-flow) increment of the bridge.

        The stochastic branch (``ode=False``) is the standard Euler-Maruyama
        discretisation of a linear-drift SDE whose drift is corrected by the
        learned score::

            dB_Q = sqrt(dt) * N(0, I)
            dB_P = dB_Q + sigma(t) * score * dt
            A_eff = +A if reverse else -A
            dX_t  = A_eff * x_t * dt + sigma(t) * dB_P

        The ``ode=True`` branch does not integrate the full probability-flow
        drift. It integrates only the model's deviation from the analytically
        known base-process score for the endpoint the trajectory is anchored to::

            dX_t = 1/2 * sigma(t)^2 * (score - analytic_base_score) * dt

        with the analytic term swapped relative to training: ``reverse`` uses
        :func:`~...bridge_process.score_target_forward`, because a reverse
        trajectory starts at ``t = 1`` anchored to ``x_1``; the forward
        direction uses :func:`~...bridge_process.score_target_reverse`. At
        sampling time ``x_start`` is an endpoint, not a training role.

        The closed forms run in the working dtype, but the increment itself is
        computed in ``x_t``'s dtype. The network is called before the ``ode``
        checks, so a bad ``ode`` request still costs one forward pass.

        :param x_t: Current state, ``(B, ...)``.
        :type x_t: Any
        :param t: Per-sample time, ``(B,)``.
        :type t: Any
        :param x_cond: Conditioning bridge tensor, shaped like ``x_t``.
        :type x_cond: Any
        :param y: Prompt-kind labels, ``(B,)``.
        :type y: Any
        :param dt: Positive step size.
        :type dt: float
        :param score_network: The trained network. Required despite the
            ``None`` default, because this port does not store it (D-009).
        :type score_network: Optional[Any]
        :param reverse: Integrate image to text rather than text to image.
        :type reverse: bool
        :param cfg_scale: Classifier-free guidance strength. Strictly positive
            values route through ``forward_with_cfg``; ``0`` and below do not.
        :type cfg_scale: float
        :param ode: Take the probability-flow branch. Requires ``A == 0`` and
            ``x_start``.
        :type ode: bool
        :param x_start: The anchored endpoint. Required when ``ode`` is set.
        :type x_start: Optional[Any]
        :param seed: Integer or :class:`keras.random.SeedGenerator` for the
            Brownian increment; ``None`` uses the global RNG. An integer gives
            the same noise on every call.
        :type seed: Optional[Any]
        :param training: Forwarded to the network.
        :type training: Optional[bool]
        :return: The increment, shaped like and dtyped like ``x_t``.
        :rtype: Any
        :raises ValueError: If ``score_network`` is ``None``, if ``dt`` is not
            positive, or if ``ode`` is requested with ``A != 0`` or without an
            ``x_start``.
        """
        _require_network(score_network, "dX_t")
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}")

        x_t = keras.ops.convert_to_tensor(x_t)
        t = keras.ops.convert_to_tensor(t)
        score = self._evaluate_score(
            score_network, x_t, t, x_cond, y, reverse, cfg_scale, training
        )
        sigma_t = _expand_like(keras.ops.cast(self.sigma(t), x_t.dtype), x_t)

        if ode:
            if self.A != 0.0:
                raise ValueError(
                    "the probability-flow branch requires a driftless process "
                    f"(A == 0); this one has A = {self.A}"
                )
            if x_start is None:
                raise ValueError(
                    "ode=True needs the anchored endpoint x_start; the analytic "
                    "base score is conditioned on it."
                )
            # Local import: bridge_process imports BridgeSDE from this module, so a
            # top-level import would be a cycle.
            from .bridge_process import score_target_forward, score_target_reverse

            analytic = (
                score_target_forward(self, x_t, t, x_start)
                if reverse
                else score_target_reverse(self, x_t, t, x_start)
            )
            analytic = keras.ops.cast(analytic, x_t.dtype)
            return 0.5 * sigma_t ** 2 * (score - analytic) * dt

        dB_Q = keras.ops.sqrt(
            keras.ops.cast(dt, x_t.dtype)
        ) * keras.random.normal(keras.ops.shape(x_t), dtype=x_t.dtype, seed=seed)
        dB_P = dB_Q + sigma_t * score * dt
        drift = self.A if reverse else -self.A
        return drift * x_t * dt + sigma_t * dB_P

    def simulate(
        self,
        x_start: Any,
        num_steps: int,
        score_network: Optional[Any] = None,
        reverse: bool = False,
        return_all: bool = False,
        cfg_scale: float = 0.0,
        ode: bool = False,
        x_cond: Optional[Any] = None,
        y: Optional[Any] = None,
        seed: Optional[Any] = None,
        training: Optional[bool] = False,
    ) -> Any:
        """Integrate the bridge from one anchored endpoint to the other.

        Runs :meth:`dX_t` ``num_steps`` times on an evenly spaced time grid.
        Step 0 always takes the SDE branch; with ``ode=True`` the later steps
        take the probability-flow branch (see the schedule on the class). An
        integer ``seed`` is turned into one ``SeedGenerator`` so each step gets
        fresh noise. Errors raised by :meth:`dX_t` propagate; with ``ode=True``
        and ``A != 0`` that happens at step 1.

        :param x_start: The anchored endpoint, ``(B, ...)``.
        :type x_start: Any
        :param num_steps: Number of integration steps. Must be positive.
        :type num_steps: int
        :param score_network: The trained network. Required despite the
            ``None`` default.
        :type score_network: Optional[Any]
        :param reverse: ``True`` integrates ``t: 1 -> 0``, ``False`` ``0 -> 1``.
        :type reverse: bool
        :param return_all: Return the list of the state after every step
            (``x_start`` not included) instead of only the final one.
        :type return_all: bool
        :param cfg_scale: Guidance strength; see :meth:`dX_t`.
        :type cfg_scale: float
        :param ode: Take the probability-flow branch from step 1 onwards.
        :type ode: bool
        :param x_cond: Conditioning tensor; defaults to ``x_start``, as upstream.
        :type x_cond: Optional[Any]
        :param y: Prompt-kind labels, ``(B,)``. Required.
        :type y: Optional[Any]
        :param seed: Integer, :class:`keras.random.SeedGenerator`, or ``None``
            for the global RNG.
        :type seed: Optional[Any]
        :param training: Forwarded to the network; ``False`` by default so a
            sampler never triggers label dropout or stochastic depth.
        :type training: Optional[bool]
        :return: The final state, or every state when ``return_all``.
        :rtype: Any
        :raises ValueError: If ``score_network`` is ``None``, ``num_steps`` is
            not positive, ``y`` is ``None``, or :meth:`dX_t` rejects a step.
        """
        # Validated before anything is converted or allocated, so a caller who
        # forgot the network gets the useful message rather than a dtype error
        # from `convert_to_tensor(None)`.
        _require_network(score_network, "simulate")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if y is None:
            raise ValueError("y is required for simulation")

        x_t = keras.ops.convert_to_tensor(x_start)
        if x_cond is None:
            x_cond = x_t

        # DECISION plan-2026-09-02T094601-77d4a04e/D-019: promote an int seed once, here.
        # An int seed is stateless, so every step would draw identical noise. See decisions.md.
        if seed is not None and not isinstance(seed, keras.random.SeedGenerator):
            seed = keras.random.SeedGenerator(seed)

        # `simulate` is an eager sampling utility built around a Python loop, so
        # the static batch size is always available here.
        times = _time_grid(num_steps, reverse)
        trajectory = []
        for i in range(num_steps):
            dt = abs(times[i + 1] - times[i])
            x_t = x_t + self.dX_t(
                x_t=x_t,
                t=self._time_vector(x_t, times[i]),
                x_cond=x_cond,
                y=y,
                dt=dt,
                score_network=score_network,
                reverse=reverse,
                cfg_scale=cfg_scale,
                ode=ode and i > 0,
                x_start=x_start if ode else None,
                seed=seed,
                training=training,
            )
            if return_all:
                trajectory.append(x_t)
        return trajectory if return_all else x_t

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments.

        :return: ``{"A": A}``; subclasses add their own.
        :rtype: Dict[str, Any]
        """
        return {"A": self.A}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BridgeSDE":
        """Rebuild a process from :meth:`get_config`.

        :param config: Output of :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new process.
        :rtype: BridgeSDE
        """
        return cls(**config)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{type(self).__name__}({args})"

# ---------------------------------------------------------------------------

@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class UniformVolatilitySDE(BridgeSDE):
    """Base process with constant volatility ``sigma(t) = K``.

    With ``A == 0`` this is scaled Brownian motion and
    ``C = K^2 (min(t_a, t_b) - start)``; with ``A != 0`` it is an
    Ornstein-Uhlenbeck process and ``C`` gains an exponential envelope. The two
    branches agree only in the limit ``A -> 0``.

    :param A: Drift coefficient. Exactly ``0`` selects the Brownian branch. May
        be negative; the ``1/(2A)`` denominator carries the sign. A tiny
        non-zero ``A`` loses precision to cancellation in ``C``.
    :type A: float
    :param K: The constant diffusion coefficient. A non-positive value logs a
        warning and does not raise.
    :type K: float
    """

    def __init__(self, A: float = 0.0, K: float = 1.0) -> None:
        super().__init__(A=A)
        self.K = float(K)
        if self.K <= 0.0:
            logger.warning(
                "UniformVolatilitySDE built with a non-positive K=%s; the base "
                "process is degenerate and C will not be a valid covariance.",
                self.K,
            )

    def sigma(self, t: Any) -> Any:
        """``sigma(t) = K``, broadcast to the shape of ``t``."""
        (t,) = _as_working(t)
        return keras.ops.full_like(t, self.K)

    def phi(self, start: Any, end: Any) -> Any:
        """``exp(-A * (end - start))``; identically ``1`` when ``A == 0``."""
        start, end = _as_working(start, end)
        return keras.ops.exp(-self.A * (end - start))

    def C(self, start: Any, t_a: Any, t_b: Any) -> Any:
        """Brownian (``A == 0``) or Ornstein-Uhlenbeck covariance.

        ``A == 0``::

            C = K^2 * (min(t_a, t_b) - start)

        ``A != 0``::

            C = K^2 * exp(-A (t_a + t_b)) * (exp(2 A min) - exp(2 A start)) / (2 A)
        """
        start, t_a, t_b = _as_working(start, t_a, t_b)
        upper = keras.ops.minimum(t_a, t_b)
        if self.A == 0.0:
            return (self.K**2) * (upper - start)
        numerator = (self.K**2) * keras.ops.exp(-self.A * (t_a + t_b))
        window = keras.ops.exp(2.0 * self.A * upper) - keras.ops.exp(
            2.0 * self.A * start
        )
        return numerator * window / (2.0 * self.A)

    def get_config(self) -> Dict[str, Any]:
        """Return ``A`` and ``K``."""
        config = super().get_config()
        config.update({"K": self.K})
        return config

# ---------------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class PeriodicVolatilitySDE(BridgeSDE):
    """Driftless process whose volatility rises and falls as a raised cosine.

    ``sigma(t) = alpha/2 * (1 - cos(2 pi k t)) + eps``: quiet at ``t = 0``,
    loudest at the half-period, quiet again at ``t = 1`` for integer ``k``. The
    floor ``eps`` keeps the process from degenerating where the cosine is 1.

    ``C`` is the exact antiderivative of ``sigma^2``. Expanding
    ``sigma(s)^2`` with ``cos^2 x = (1 + cos 2x)/2`` gives three terms::

        sigma(s)^2 = (3a^2/8 + a e + e^2)          constant
                     - (a^2/2 + a e) cos(2 pi k s) fundamental
                     + (a^2/8)       cos(4 pi k s) second harmonic

    whose antiderivative is ``first_term - second_term + third_term`` in
    :meth:`_antiderivative`. The minus on the middle term is the sign of the
    fundamental, the easiest part to get wrong.

    :param alpha: Peak-to-trough amplitude of the volatility. Not validated.
    :type alpha: float
    :param k: Cycles over the unit interval. Must be non-zero.
    :type k: float
    :param eps: Volatility floor. Not validated.
    :type eps: float
    :raises ValueError: If ``k`` is 0.
    """

    def __init__(self, alpha: float = 0.95, k: float = 1.0, eps: float = 0.05) -> None:
        super().__init__(A=0.0)
        self.alpha = float(alpha)
        self.k = float(k)
        self.eps = float(eps)
        if self.k == 0.0:
            raise ValueError(
                "PeriodicVolatilitySDE needs a non-zero k; the antiderivative "
                "divides by `4 * pi * k` and would be undefined."
            )

    def sigma(self, t: Any) -> Any:
        """``alpha/2 * (1 - cos(2 pi k t)) + eps``."""
        (t,) = _as_working(t)
        return self.alpha / 2.0 * (
            1.0 - keras.ops.cos(2.0 * math.pi * self.k * t)
        ) + self.eps

    def phi(self, start: Any, end: Any) -> Any:
        """Return ones shaped like ``start``; the process is driftless."""
        start, end = _as_working(start, end)
        return keras.ops.ones_like(start)

    def _antiderivative(self, s: Any) -> Any:
        """``F(s)`` with ``F' = sigma^2``; ``C = F(min(t_a,t_b)) - F(start)``.

        :param s: Times, already in the working dtype.
        :type s: Any
        :return: ``F(s)``.
        :rtype: Any
        """
        alpha, k, eps = self.alpha, self.k, self.eps
        first_term = (3.0 * alpha**2 / 8.0 + alpha * eps + eps**2) * s
        second_term = (
            alpha
            * (alpha + 2.0 * eps)
            / (4.0 * math.pi * k)
            * keras.ops.sin(2.0 * math.pi * k * s)
        )
        third_term = (
            alpha**2 / (32.0 * math.pi * k) * keras.ops.sin(4.0 * math.pi * k * s)
        )
        return first_term - second_term + third_term

    def C(self, start: Any, t_a: Any, t_b: Any) -> Any:
        """``\\int_start^{min(t_a, t_b)} sigma(s)^2 ds``, in closed form."""
        start, t_a, t_b = _as_working(start, t_a, t_b)
        upper = keras.ops.minimum(t_a, t_b)
        return self._antiderivative(upper) - self._antiderivative(start)

    def get_config(self) -> Dict[str, Any]:
        """Return ``alpha``, ``k`` and ``eps``; ``A`` is fixed at 0."""
        config = super().get_config()
        config.pop("A", None)
        config.update({"alpha": self.alpha, "k": self.k, "eps": self.eps})
        return config

# ---------------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class CosineDecayingVolatilitySDE(PeriodicVolatilitySDE):
    """Base process whose volatility decays from ``alpha + eps`` to ``eps``.

    :class:`PeriodicVolatilitySDE` at ``k = 0.5`` (half a cycle, no second
    peak), with a ``t - 1`` shift applied to both ``sigma`` and ``C`` that
    turns the rising half-cosine into a falling one::

        sigma(t) = alpha/2 * (1 + cos(pi t)) + eps

    At ``t = 0.5`` the shift changes nothing, so it only shows elsewhere.

    :param alpha: Peak-to-trough amplitude of the volatility.
    :type alpha: float
    :param eps: Volatility floor, reached at ``t = 1``.
    :type eps: float
    """

    def __init__(self, alpha: float = 0.95, eps: float = 0.05) -> None:
        super().__init__(alpha=alpha, k=0.5, eps=eps)

    def sigma(self, t: Any) -> Any:
        """Return ``PeriodicVolatilitySDE.sigma(t - 1)``."""
        (t,) = _as_working(t)
        return super().sigma(t - 1.0)

    def C(self, start: Any, t_a: Any, t_b: Any) -> Any:
        """Return ``PeriodicVolatilitySDE.C(start - 1, t_a - 1, t_b - 1)``.

        The same shift as :meth:`sigma`; shifting only one would leave ``C``
        inconsistent with its integrand.
        """
        start, t_a, t_b = _as_working(start, t_a, t_b)
        return super().C(start - 1.0, t_a - 1.0, t_b - 1.0)

    def get_config(self) -> Dict[str, Any]:
        """Return ``alpha`` and ``eps``; ``k`` is fixed at 0.5."""
        config = super().get_config()
        # `k` is fixed by the definition of this variant; it is not a knob.
        config.pop("k", None)
        return config

# ---------------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class FlowMatchingODE(BridgeSDE):
    """Run deterministic rectified-flow transport through the bridge SDE interface.

    The transport is deterministic, so ``sigma``, ``phi`` and ``C`` are
    undefined and all three raise; a ``0.0`` would turn the division by ``C``
    in the score targets into a silent infinite score. :meth:`dX_t` is
    overridden so sampling never touches them. The paper uses this variant as
    a failure case (D-002).

    :param force_unconditional: Sample with the conditioning stream masked off.
        Read by :meth:`dX_t`, which then feeds the network an all-false
        ``cond_mask``, pins ``direction`` to forward, and rejects any non-zero
        ``cfg_scale``.
    :type force_unconditional: bool
    """

    def __init__(self, force_unconditional: bool = False) -> None:
        super().__init__(A=0.0)
        self.force_unconditional = bool(force_unconditional)

    @staticmethod
    def _unsupported(name: str) -> None:
        """Raise for the undefined quantity ``name``.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"FlowMatchingODE has no {name}; it is a deterministic flow, not a "
            f"diffusion. Nothing may substitute a value here -- the bridge score "
            f"targets divide by C."
        )

    def sigma(self, t: Any) -> Any:
        """:raises NotImplementedError: Always."""
        self._unsupported("diffusion coefficient sigma")

    def phi(self, start: Any, end: Any) -> Any:
        """:raises NotImplementedError: Always."""
        self._unsupported("base-process transition phi")

    def C(self, start: Any, t_a: Any, t_b: Any) -> Any:
        """:raises NotImplementedError: Always."""
        self._unsupported("base-process covariance C")

    def dX_t(
        self,
        x_t: Any,
        t: Any,
        x_cond: Any,
        y: Any,
        dt: float,
        score_network: Optional[Any] = None,
        reverse: bool = False,
        cfg_scale: float = 0.0,
        ode: bool = False,
        x_start: Optional[Any] = None,
        seed: Optional[Any] = None,
        training: Optional[bool] = False,
    ) -> Any:
        """Return one Euler increment of the deterministic flow::

            signed_dt = -dt if reverse else dt
            dX_t      = velocity * signed_dt

        No Brownian term, no drift and no call to ``sigma``, ``phi`` or ``C``.
        ``ode`` and ``x_start`` are accepted and ignored, as upstream: the flow
        already is an ODE. ``seed`` is ignored because nothing is stochastic.

        :param x_t: Current state, ``(B, ...)``.
        :type x_t: Any
        :param t: Per-sample time, ``(B,)``.
        :type t: Any
        :param x_cond: Conditioning tensor, shaped like ``x_t``.
        :type x_cond: Any
        :param y: Prompt-kind labels, ``(B,)``.
        :type y: Any
        :param dt: Positive step size. The sign is applied here, not by the
            caller: :meth:`BridgeSDE.simulate` always passes ``|t[i+1]-t[i]|``.
        :type dt: float
        :param score_network: The trained network. Required despite the
            ``None`` default (D-009).
        :type score_network: Optional[Any]
        :param reverse: Integrate image to text rather than text to image. Under
            ``force_unconditional`` it flips ``dt`` only; the network always
            sees ``direction`` 0.
        :type reverse: bool
        :param cfg_scale: Guidance strength. Must be ``0`` under
            ``force_unconditional``.
        :type cfg_scale: float
        :param ode: Accepted and ignored; see above.
        :type ode: bool
        :param x_start: Accepted and ignored; see above.
        :type x_start: Optional[Any]
        :param seed: Accepted and ignored; the flow is deterministic.
        :type seed: Optional[Any]
        :param training: Forwarded to the network.
        :type training: Optional[bool]
        :return: The increment, shaped like and dtyped like ``x_t``.
        :rtype: Any
        :raises ValueError: If ``score_network`` is ``None``, if ``dt`` is not
            positive, or if ``cfg_scale != 0`` under ``force_unconditional``.
        """
        _require_network(score_network, "dX_t")
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}")

        x_t = keras.ops.convert_to_tensor(x_t)
        t = keras.ops.convert_to_tensor(t)

        # DECISION plan-2026-09-02T094601-77d4a04e/D-029: reverse=False here, as upstream.
        # The unconditional flow is one shared velocity field. See decisions.md.
        if self.force_unconditional:
            if cfg_scale != 0:
                raise ValueError(
                    "force_unconditional=True is incompatible with CFG: there "
                    "is no conditional pass to guide towards. Got cfg_scale = "
                    f"{cfg_scale}; pass cfg_scale=0 or build the SDE with "
                    "force_unconditional=False."
                )
            velocity = self._evaluate_score(
                score_network,
                x_t,
                t,
                x_cond,
                y,
                reverse=False,
                cfg_scale=0.0,
                training=training,
                cond_mask=keras.ops.zeros_like(t),
            )
        else:
            velocity = self._evaluate_score(
                score_network, x_t, t, x_cond, y, reverse, cfg_scale, training
            )

        signed_dt = -dt if reverse else dt
        return velocity * keras.ops.cast(signed_dt, velocity.dtype)

    def get_config(self) -> Dict[str, Any]:
        """Return ``force_unconditional``; ``A`` is fixed at 0."""
        config = super().get_config()
        config.pop("A", None)
        config.update({"force_unconditional": self.force_unconditional})
        return config

# ---------------------------------------------------------------------------

#: Name -> class, for a config-driven build of the base process. Not named
#: ``SDE_VARIANTS``: see the D-015 anchor below.
SDE_TYPES: Dict[str, type] = {
    "uniform": UniformVolatilitySDE,
    "periodic": PeriodicVolatilitySDE,
    "cosine_decay": CosineDecayingVolatilitySDE,
    "flow_matching": FlowMatchingODE,
}

# ---------------------------------------------------------------------------

# DECISION plan-2026-09-02T094601-77d4a04e/D-015: no "variant" names on this builder.
# A "variant" parameter marks a model factory that needs from_variant. See decisions.md.
def create_bridge_sde(sde_type: str, **kwargs: Any) -> BridgeSDE:
    """Build one of the four base processes by name.

    :param sde_type: One of :data:`SDE_TYPES`.
    :type sde_type: str
    :param **kwargs: Forwarded to the selected class's constructor.
    :type **kwargs: Any
    :return: The constructed process.
    :rtype: BridgeSDE
    :raises ValueError: If ``sde_type`` is not registered, or the constructor
        rejects an argument (``k == 0`` for ``periodic``).
    """
    if sde_type not in SDE_TYPES:
        raise ValueError(
            f"Unknown SDE type '{sde_type}'. Available: {sorted(SDE_TYPES)}"
        )
    return SDE_TYPES[sde_type](**kwargs)

# ---------------------------------------------------------------------------
