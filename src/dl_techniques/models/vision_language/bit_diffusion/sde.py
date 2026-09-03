"""The four base processes the text<->image bridge is built on.

A bridge diffusion needs a *base* stochastic process to bridge: a law for how a
sample wanders between the two anchored endpoints when nothing is conditioning
it. Three quantities specify that law completely, and the whole port depends on
them being right:

``sigma(t)``
    The diffusion coefficient at time ``t`` -- how violently the process is
    shaken there. It multiplies the Brownian increment of every sampler step.
``phi(start, end)``
    The deterministic transition factor of the drift, ``exp(-A*(end-start))``.
    It is identically ``1`` for every driftless (``A == 0``) variant.
``C(start, t_a, t_b)``
    The covariance ``Cov(X_{t_a}, X_{t_b} | X_start)``. For a driftless process
    it is simply ``\\int_start^{min(t_a,t_b)} sigma(s)^2 ds``.

``C`` is the dangerous one. It DIVIDES both analytic score targets and scales
both direction-specific loss weightings, and it is exactly zero at the anchored
endpoint, so a wrong constant is a silently mistrained model rather than a
crash. Every closed form below is pinned to hand-derived float64 golden values
in ``tests/.../test_the_sde_closed_forms.py``.

The bridge timeline::

    t = 0                                                          t = 1
    text endpoint                                          image endpoint
    x_0 (packed token embeddings)                 x_1 (VAE image latent)
      |------------------------------------------------------------|
      |                     x_t ~ N(mu(t), s^2(t) I)                |
      |                                                             |
      |   mu(t)  = phi(0,t) x_0 + C(0,t,1)/C(0,1,1) (x_1 - phi(0,1) x_0)
      |   s^2(t) = C(0,t,t) - C(0,t,1)^2 / C(0,1,1)                 |
      |                                                             |
      |   C(0,t,t) -> 0                             C(t,1,1) -> 0   |
      |   (reverse target divides by it)   (forward target does)    |

    PeriodicVolatilitySDE, alpha = 0.95, k = 1.0, eps = 0.05:

      sigma
      1.00 |               ****
           |            ***    ***
           |          **          **
      0.50 |        **              **
           |      **                  **
           |   ***                      ***
      0.05 |***                            ****
           +-------------------------------------> t
          0.0        0.25      0.5       0.75      1.0

    CosineDecayingVolatilitySDE is the SAME curve shifted by one time unit
    (``k = 0.5``, ``sigma(t) = Periodic.sigma(t - 1)``), so it starts LOUD at
    ``alpha + eps`` and decays monotonically to ``eps``:

      sigma
      1.00 |****
           |    ****
      0.50 |        ****
           |            *****
      0.05 |                 **********
           +-------------------------------------> t
          0.0        0.25      0.5       0.75      1.0

Two deliberate divergences from upstream, both recorded in ``decisions.md``:

1. **No ``score_network`` attribute.** Upstream stores the network on the SDE
   object (``SDE.__init__(self, A, score_network)``). These classes are pure
   math objects; the network is passed in at sampling time. See D-009.
2. **``dX_t`` / ``simulate`` take the network as an argument.** Same reason:
   the sampler is a method on the math object, but the network reaches it
   through the call, and passing ``score_network=None`` raises rather than
   silently sampling from the base process.

References:
    - Upstream ``sde_utils_sde.py``, staged verbatim under the plan's
      ``reference/`` directory.
    - ``findings/source-model-semantics.md`` section 3 -- the formula table every
      closed form here is transcribed from, and section 4 for the sampler that
      will consume them.
"""

import math
from typing import Any, Dict, Optional

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Never-narrow working dtype
# ---------------------------------------------------------------------


# DECISION plan-2026-09-02T094601-77d4a04e/D-010
# Do NOT replace this with an import from `losses/brier_spiegelhalters_ztest_loss.py`
# or the two other hand-rolled copies: all three are module-private and promoting one
# is a repo-wide refactor this port has no gate for. Do NOT add a fifth copy either --
# every module in this package imports THIS one. See decisions.md D-010.
def bridge_math_dtype(*dtypes: Any) -> str:
    """Return the never-narrowing dtype the bridge closed forms run in.

    ``C`` is ``O(1e-4)`` near either endpoint and is then DIVIDED by, so a
    ``mixed_float16`` policy is an accuracy hazard rather than a speed win. The
    rule is a FLOOR, ``max(inputs, float32)``, not a hard-coded ``"float32"``:
    pinning float32 would silently narrow a float64 policy, which is the exact
    mistake the floor exists to prevent.

    :param dtypes: Candidate dtypes -- typically the input tensors' dtypes.
    :type dtypes: Any
    :return: ``"float64"`` if any candidate is float64, otherwise ``"float32"``.
    :rtype: str
    """
    for dtype in dtypes:
        if dtype is None:
            continue
        # DECISION plan-2026-09-02T094601-77d4a04e/D-015
        # `getattr(dtype, "name", None) or str(dtype)` rather than
        # `keras.backend.standardize_dtype`: the Keras-2-residue guard in
        # `tests/test_the_keras2_backend_calls_are_gone.py` forbids any
        # `keras.backend.*` call across all of `src/`, and a backend tensor's dtype
        # already carries its own `.name` (a `tf.DType` stringifies as
        # "<dtype: 'float64'>", so `str` alone is not enough). A plain-string
        # dtype has no `.name` and falls through to `str` unchanged.
        # See decisions.md D-015.
        if (getattr(dtype, "name", None) or str(dtype)) == "float64":
            return "float64"
    return "float32"


def _expand_like(scalars: Any, reference: Any) -> Any:
    """Reshape a per-sample ``(B,)`` vector to broadcast against ``reference``.

    ``t`` and every quantity derived from it (``sigma(t)``, ``phi``, ``C``) is
    one value per sample, while the bridge tensor is rank-4 in production.
    Broadcasting a ``(B,)`` against a ``(B, H, W, C)`` without an explicit
    reshape aligns it with the LAST axis, silently weighting channels instead of
    samples -- a wrong answer with the right shape.

    Interface contract: returns ``scalars`` unchanged when either operand is
    effectively rank-<=1 (nothing to broadcast against), otherwise a view of
    shape ``(-1, 1, 1, ...)`` with ``rank(reference) - 1`` trailing ones. It
    never changes dtype and never reduces.

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


def _require_network(score_network: Any, caller: str) -> None:
    """Reject a missing score network loudly.

    Interface contract: returns ``None`` when ``score_network`` is anything but
    ``None``; otherwise raises. It performs no other validation -- a network
    that is merely wrong will fail at its own call site.

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


def _time_grid(num_steps: int, reverse: bool) -> list:
    """The ``num_steps + 1`` integration times, endpoints included.

    ``linspace(1, 0)`` when reversing (image -> text), ``linspace(0, 1)``
    otherwise. Built in Python rather than with ``keras.ops.linspace`` because
    the sampler needs each ``dt`` as a Python float and reads the grid inside a
    Python loop anyway.

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


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class BridgeSDE:
    # DECISION plan-2026-09-02T094601-77d4a04e/D-009
    # Do NOT add a `score_network` attribute the way upstream does. It makes the
    # object unserializable (get_config would have to embed a whole DiTXA or drop
    # it) and untestable without a model. Pass the network to `dX_t`/`simulate`.
    # See decisions.md D-009.
    """Interface of a bridge base process: ``sigma``, ``phi`` and ``C``.

    Concrete subclasses supply the three closed forms. The base raises for all
    of them -- an abstract method that returned ``0.0`` would let a half-ported
    variant train to convergence on a degenerate score.

    :param A: Linear drift coefficient. ``0`` for every driftless variant;
        ``UniformVolatilitySDE`` is the only one that takes it non-zero.
    :type A: float
    """

    def __init__(self, A: float = 0.0) -> None:
        self.A = float(A)

    # -- the three closed forms -----------------------------------------

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

    # -- sampler surface --------------------------------------------------

    def _time_vector(self, x_t: Any, t_value: float) -> Any:
        """A ``(B,)`` constant-time vector matching ``x_t``'s batch and dtype."""
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
        """Call the network for the score, routing through CFG only when asked.

        :param cond_mask: Optional ``(B,)`` conditioning mask forwarded to the
            network. ``None`` (the default) omits the key entirely, which the
            network reads as all-ones. Only :class:`FlowMatchingODE` passes one,
            and only to force the unconditional branch.
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
        # DECISION plan-2026-09-02T094601-77d4a04e/D-018
        # Do NOT "simplify" this to always calling `forward_with_cfg` and letting
        # `cfg_scale = 0` be a no-op. Upstream gates on `if cfg_scale > 0`
        # (`sde_utils_sde.py:18`) and the gate is load-bearing for COST, not for
        # value: `forward_with_cfg` runs the network TWICE, so an ungated call
        # doubles every sampler step of every unguided run. The two formulas do
        # coincide at `s = 0` (this port's `cond + 0*(cond - uncond) == cond`),
        # which is exactly why no value-level test can see the regression.
        # See decisions.md D-018.
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
        """One Euler-Maruyama (or probability-flow) increment of the bridge.

        The stochastic branch (``ode=False``) is the standard Euler-Maruyama
        discretisation of a linear-drift SDE whose drift is corrected by the
        learned score::

            dB_Q = sqrt(dt) * N(0, I)
            dB_P = dB_Q + sigma(t) * score * dt
            A_eff = +A if reverse else -A
            dX_t  = A_eff * x_t * dt + sigma(t) * dB_P

        The ``ode=True`` branch does **not** integrate the full probability-flow
        drift. It integrates only the model's DEVIATION from the analytically
        known base-process score for the endpoint the trajectory is anchored to::

            dX_t = 1/2 * sigma(t)^2 * (score - analytic_base_score) * dt

        and the analytic term is the **swapped** one: ``reverse`` uses
        :func:`~...bridge_process.score_target_forward` (upstream's
        ``grad_wrt_x_t_log_p_base_x_1_cond_x_t``) because a reverse trajectory
        starts at ``t = 1`` and is anchored to ``x_1``; the forward direction
        uses :func:`~...bridge_process.score_target_reverse`. Those are the
        opposite pairings from the training-time targets, and that is correct --
        at sampling time ``x_start`` is an endpoint, not a training role. Verified
        against ``reference/sde_utils_sde.py:22-29``.

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
        :param score_network: The trained network. Required; keyword-only in
            practice because upstream stores it on the object and this port
            deliberately does not (D-009).
        :type score_network: Optional[Any]
        :param reverse: Integrate image -> text rather than text -> image.
        :type reverse: bool
        :param cfg_scale: Classifier-free guidance strength. Strictly positive
            values route through ``forward_with_cfg``; ``0`` does not.
        :type cfg_scale: float
        :param ode: Take the probability-flow branch. Requires ``A == 0`` and
            ``x_start``.
        :type ode: bool
        :param x_start: The anchored endpoint. Required when ``ode`` is set.
        :type x_start: Optional[Any]
        :param seed: Integer or :class:`keras.random.SeedGenerator` for the
            Brownian increment. Threaded explicitly -- never global RNG.
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
            # Imported here, not at module scope: `bridge_process` imports
            # `BridgeSDE` from this module, so a top-level import would be a
            # cycle. The alternative -- re-deriving the two analytic scores
            # inline -- would put a second copy of the formulas that
            # `test_the_bridge_process_math.py` pins somewhere it does not look.
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

        The loop, for ``num_steps = 4`` in the reverse direction::

            i      =      0         1         2         3
            t      =    1.00      0.75      0.50      0.25      0.00
                        x_1 -------> . -------> . -------> . -------> x_0
            branch =     SDE       ODE?      ODE?      ODE?
                          ^          ^
                          |          `-- ode=True from here on
                          `-- ALWAYS the SDE branch, even when ode=True

        The first-step skip is ``ode and i > 0``, and it is not a style choice.
        The analytic base score divides by ``C(start, t, t)``, which is exactly
        ``0`` at the anchored endpoint (``C(0,0,0) = 0`` forward, ``C(1,1,1) = 0``
        reverse). Taking the ODE branch on step ``0`` therefore divides by zero
        and the whole remaining trajectory is ``nan``. Reproduce it; do not
        "fix" it into a pure-ODE start.

        :param x_start: The anchored endpoint, ``(B, ...)``.
        :type x_start: Any
        :param num_steps: Number of integration steps. Must be positive.
        :type num_steps: int
        :param score_network: The trained network (required).
        :type score_network: Optional[Any]
        :param reverse: ``True`` integrates ``t: 1 -> 0``, ``False`` ``0 -> 1``.
        :type reverse: bool
        :param return_all: Return the list of every intermediate state instead
            of only the final one.
        :type return_all: bool
        :param cfg_scale: Guidance strength; see :meth:`dX_t`.
        :type cfg_scale: float
        :param ode: Take the probability-flow branch from step 1 onwards.
        :type ode: bool
        :param x_cond: Conditioning tensor; defaults to ``x_start``, as upstream.
        :type x_cond: Optional[Any]
        :param y: Prompt-kind labels, ``(B,)``. Required.
        :type y: Optional[Any]
        :param seed: Integer or :class:`keras.random.SeedGenerator`.
        :type seed: Optional[Any]
        :param training: Forwarded to the network; ``False`` by default so a
            sampler never triggers label dropout or stochastic depth.
        :type training: Optional[bool]
        :return: The final state, or every state when ``return_all``.
        :rtype: Any
        :raises ValueError: If ``num_steps`` is not positive or ``y`` is None.
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

        # DECISION plan-2026-09-02T094601-77d4a04e/D-019
        # Do NOT pass a bare integer `seed` down to the per-step
        # `keras.random.normal`. `keras.random.*` is STATELESS given an int: the
        # same integer draws the same tensor every call, so every step of the
        # trajectory would get the IDENTICAL Brownian increment. The result is
        # still finite, still shaped right, and still varies with the seed, so
        # no finiteness, shape or reproducibility arm can see it. Promoting the
        # int to one SeedGenerator here -- once, outside the loop -- is what
        # makes the increments independent. See decisions.md D-019.
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

    # -- serialization ----------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """:return: Constructor arguments. :rtype: Dict[str, Any]"""
        return {"A": self.A}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BridgeSDE":
        """:param config: Output of :meth:`get_config`. :rtype: BridgeSDE"""
        return cls(**config)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{type(self).__name__}({args})"


# ---------------------------------------------------------------------
# Uniform volatility -- Brownian motion (A = 0) or Ornstein-Uhlenbeck (A != 0)
# ---------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class UniformVolatilitySDE(BridgeSDE):
    """Constant-volatility base process.

    ``sigma(t) = K`` everywhere. With ``A == 0`` this is scaled Brownian motion
    and ``C`` is the familiar ``K^2 (min(t_a, t_b) - start)``; with ``A != 0`` it
    is an Ornstein-Uhlenbeck process and ``C`` picks up the exponential envelope.

    Both branches are live and both are golden-pinned: an ``A = 0``-only test is
    vacuous for the OU code path, because the two expressions coincide nowhere
    except in the limit.

    :param A: Drift coefficient. ``0`` selects the Brownian branch. May be
        negative -- the ``1/(2A)`` denominator carries the sign correctly.
    :type A: float
    :param K: The constant diffusion coefficient.
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
        config = super().get_config()
        config.update({"K": self.K})
        return config


# ---------------------------------------------------------------------
# Periodic volatility
# ---------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class PeriodicVolatilitySDE(BridgeSDE):
    """Driftless process whose volatility rises and falls as a raised cosine.

    ``sigma(t) = alpha/2 * (1 - cos(2 pi k t)) + eps`` -- quiet at ``t = 0``,
    loudest at the half-period, quiet again at ``t = 1`` when ``k = 1``. The
    floor ``eps`` keeps the process from degenerating where the cosine is 1.

    ``C`` is the exact antiderivative of ``sigma^2``. Expanding
    ``sigma(s)^2`` with ``cos^2 x = (1 + cos 2x)/2`` gives three terms::

        sigma(s)^2 = (3a^2/8 + a e + e^2)          constant
                     - (a^2/2 + a e) cos(2 pi k s) fundamental
                     + (a^2/8)       cos(4 pi k s) second harmonic

    whose antiderivative is the ``first_term - second_term + third_term`` below.
    The MINUS on the middle term is the sign of the fundamental and it is the
    single easiest thing to get wrong; it is golden-pinned.

    :param alpha: Peak-to-trough amplitude of the volatility.
    :type alpha: float
    :param k: Cycles over the unit interval.
    :type k: float
    :param eps: Volatility floor.
    :type eps: float
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
        """``1`` -- the process is driftless (``A == 0``)."""
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
        config = super().get_config()
        config.pop("A", None)
        config.update({"alpha": self.alpha, "k": self.k, "eps": self.eps})
        return config


# ---------------------------------------------------------------------
# Cosine-decaying volatility
# ---------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class CosineDecayingVolatilitySDE(PeriodicVolatilitySDE):
    """Periodic volatility at ``k = 0.5``, shifted so it DECAYS across the bridge.

    Two things define it, and a port that ships only the first is wrong while
    looking right:

    1. ``k = 0.5`` -- a half cycle over the unit interval, so no second peak.
    2. A ``t - 1`` time shift applied to **both** ``sigma`` and ``C``, which
       turns the rising half-cosine into a falling one::

           sigma(t) = alpha/2 * (1 + cos(pi t)) + eps
                      alpha + eps  at t = 0   ->   eps  at t = 1

    Beware ``t = 0.5``: it is exactly a half-period of the ``k = 0.5`` cosine, so
    the shift is INVISIBLE there (``sigma(0.5) = 0.525`` with or without it). Any
    guard on the shift must sample somewhere else.

    :param alpha: Peak-to-trough amplitude of the volatility.
    :type alpha: float
    :param eps: Volatility floor, reached at ``t = 1``.
    :type eps: float
    """

    def __init__(self, alpha: float = 0.95, eps: float = 0.05) -> None:
        super().__init__(alpha=alpha, k=0.5, eps=eps)

    def sigma(self, t: Any) -> Any:
        """``Periodic.sigma(t - 1)`` -- the shift is load-bearing, not cosmetic."""
        (t,) = _as_working(t)
        return super().sigma(t - 1.0)

    def C(self, start: Any, t_a: Any, t_b: Any) -> Any:
        """``Periodic.C(start - 1, t_a - 1, t_b - 1)`` -- the SAME shift.

        Shifting ``sigma`` alone leaves ``C`` inconsistent with its own
        integrand while every ``sigma`` assertion stays green.
        """
        start, t_a, t_b = _as_working(start, t_a, t_b)
        return super().C(start - 1.0, t_a - 1.0, t_b - 1.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # `k` is fixed by the definition of this variant; it is not a knob.
        config.pop("k", None)
        return config


# ---------------------------------------------------------------------
# Flow-matching baseline -- deliberately NOT a diffusion
# ---------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.sde")
class FlowMatchingODE(BridgeSDE):
    """Deterministic rectified-flow transport wearing the bridge SDE interface.

    It has no diffusion coefficient, no transition factor and no covariance --
    not "not yet", but *never*: the transport is deterministic, so the three
    quantities are undefined rather than unimplemented. All three therefore
    RAISE. Returning ``0.0`` would be worse than a crash, because the bridge
    process math divides by ``C`` and a zero there yields a plausible-looking
    infinite or degenerate score with no exception to trace it back to.

    The paper uses this variant as a deliberate failure case, which is why it is
    in scope at all (see D-002).

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
        """:raises NotImplementedError: Always; ``name`` identifies the quantity."""
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
        """One Euler increment of the deterministic flow::

            signed_dt = -dt if reverse else dt
            dX_t      = velocity * signed_dt

        There is no Brownian term, no drift term and **no call to** ``sigma``,
        ``phi`` or ``C`` anywhere on this path -- which is exactly what lets
        those three keep raising while the variant remains sampleable. The
        inherited :meth:`BridgeSDE.dX_t` cannot serve here: its very first step
        after the network call is ``self.sigma(t)``.

        ``ode`` and ``x_start`` are **accepted and ignored**, matching upstream
        (``reference/sde_utils_sde.py:69-82``, whose override takes both and
        references neither). A deterministic flow has no separate
        probability-flow branch to switch into: the transport already *is* the
        ODE, and the ``ode=True`` branch of the base class exists only to
        subtract an analytic base score that needs ``C``. ``seed`` is ignored
        for the same reason -- nothing here is stochastic.

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
        :param score_network: The trained network. Required (D-009: this port
            does not store it on the SDE).
        :type score_network: Optional[Any]
        :param reverse: Integrate image -> text rather than text -> image. Under
            ``force_unconditional`` it flips ``dt`` only -- see the anchor below.
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

        # DECISION plan-2026-09-02T094601-77d4a04e/D-029
        # Do NOT pass the outer `reverse` to the network on this branch, and do
        # NOT drop the all-false `cond_mask` as "what `cfg_scale=0` already
        # does". Upstream (reference/sde_utils_sde.py:71-76) hard-codes
        # `reverse=False` here because a forced-unconditional flow is ONE shared
        # velocity field for both directions; the outer `reverse` then only
        # flips the sign of `dt`. Threading `reverse` through instead calls the
        # reverse conditioning embedder and the reverse `t_cond`, which is a
        # different field -- finite, same shape, plausible trajectories, and
        # invisible to every shape/finiteness arm. See decisions.md D-029.
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
        config = super().get_config()
        config.pop("A", None)
        config.update({"force_unconditional": self.force_unconditional})
        return config


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

#: Name -> class, for a config-driven build of the base process.
#:
#: Deliberately NOT named ``SDE_VARIANTS``: see the D-015 anchor below.
SDE_TYPES: Dict[str, type] = {
    "uniform": UniformVolatilitySDE,
    "periodic": PeriodicVolatilitySDE,
    "cosine_decay": CosineDecayingVolatilitySDE,
    "flow_matching": FlowMatchingODE,
}


# DECISION plan-2026-09-02T094601-77d4a04e/D-015
# Do NOT rename this parameter back to `variant`, and do NOT rename `SDE_TYPES`
# back to `SDE_VARIANTS`. `variant` is a RESERVED word in this tree: the
# repo-wide sweep `_sweep_create_delegation` in
# `tests/test_models/test_package_api_contract.py` classifies every module-level
# `create_*` that takes a parameter literally named `variant` as a MODEL factory
# and requires a matching `from_variant` classmethod -- which this module can
# never have, because it builds a pure-math object that is not a `keras.Model`
# and has no `MODEL_VARIANTS` table. Spelling it `variant` here made this
# function land in the `_CREATE_WITHOUT_FROM_VARIANT` scope-exclusion pin and
# turned that set-equality assertion red. The fix is the honest one -- an SDE
# family is not a model variant -- not an entry in a shared exception list.
# `SDE_TYPES` also avoids `_LEGACY_VARIANT_TABLE_RE` (`[A-Z0-9]+_VARIANTS`) in
# the same file. See decisions.md D-015.
def create_bridge_sde(sde_type: str, **kwargs: Any) -> BridgeSDE:
    """Build one of the four base processes by name.

    :param sde_type: One of :data:`SDE_TYPES`.
    :type sde_type: str
    :param kwargs: Forwarded to the selected class's constructor.
    :type kwargs: Any
    :return: The constructed process.
    :rtype: BridgeSDE
    :raises ValueError: If ``sde_type`` is not registered.
    """
    if sde_type not in SDE_TYPES:
        raise ValueError(
            f"Unknown SDE type '{sde_type}'. Available: {sorted(SDE_TYPES)}"
        )
    return SDE_TYPES[sde_type](**kwargs)
