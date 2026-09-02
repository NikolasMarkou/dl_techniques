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
2. **``dX_t`` / ``simulate`` are declared here but raise.** They belong to the
   sampler step of the port and are written later; a stub that RAISES is the
   only honest placeholder, because one that returned a plausible tensor would
   be an untested code path under a green suite.

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
        # `keras.backend.standardize_dtype`: the repo-wide Keras-2-residue guard
        # in `tests/test_models/test_package_api_contract.py` forbids any
        # `keras.backend.*` call under `models/`, and a backend tensor's dtype
        # already carries its own `.name` (a `tf.DType` stringifies as
        # "<dtype: 'float64'>", so `str` alone is not enough). A plain-string
        # dtype has no `.name` and falls through to `str` unchanged.
        # See decisions.md D-015.
        if (getattr(dtype, "name", None) or str(dtype)) == "float64":
            return "float64"
    return "float32"


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

    # -- sampler surface, implemented in a later step ---------------------

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
    ) -> Any:
        """One Euler-Maruyama / probability-flow increment. **Not yet implemented.**

        Declared here so the interface is visible where the closed forms live,
        but deliberately left raising: the sampler is written in a later step of
        the port, and a stub returning a plausible tensor would ship an
        untested code path under a green suite.

        :raises NotImplementedError: Always, for now.
        """
        raise NotImplementedError(
            "BridgeSDE.dX_t is not implemented yet; the Euler-Maruyama step and "
            "the probability-flow ODE branch land with the sampler."
        )

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
    ) -> Any:
        """Integrate the bridge over ``num_steps``. **Not yet implemented.**

        :raises NotImplementedError: Always, for now.
        """
        raise NotImplementedError(
            "BridgeSDE.simulate is not implemented yet; it lands with the sampler, "
            "including the `ode and i > 0` first-step skip that avoids the "
            "C(0, 0, 0) = 0 singularity."
        )

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
        Consumed by the sampler in a later step; carried here so the object's
        config is complete.
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
