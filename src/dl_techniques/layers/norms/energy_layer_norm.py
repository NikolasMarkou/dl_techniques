"""
Energy Layer Normalization for Keras 3.x.

Implements the layer normalization of the Energy Transformer (ET), Hoover, Liang, Pham,
Panda, Strobelt, Zaki, Chau, Krotov, "Energy Transformer", NeurIPS 2023
(https://arxiv.org/abs/2302.07253), equations (1)-(2).

The distinguishing feature versus ``keras.layers.LayerNormalization`` is the
parameterization: the scale ``gamma`` is a **SCALAR** and the offset ``delta`` is a
**VECTOR** of dimension ``D``. Stock ``LayerNormalization`` has a per-feature (vector)
gamma, which does NOT correspond to the ET Lagrangian and therefore does NOT yield the
positive-semi-definite Hessian that makes the ET energy-descent guarantee provable.
"""

import keras
from keras import ops, initializers, constraints
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------

# DECISION plan_2026-07-13_57c9833e/D-010
# The DEFAULT lower bound on `gamma`. `gamma > 0` is not a style preference — it is the
# PRECONDITION for the PSD Hessian of the Lagrangian, i.e. the precondition for the ENTIRE
# energy-descent guarantee of the Energy Transformer:
#     dE/dt = -(dE/dg)^T (dg/dx) (dE/dg) <= 0   requires  dg/dx  PSD  requires  gamma > 0.
# Iteration 1 stated this as a plan invariant and enforced it NOWHERE: an adversarial
# reviewer set `gamma = -1.0` and the block silently performed energy ASCENT (max diff(E) =
# +1.3e4) with no error and no failing test.
#
# WHY A STRICTLY POSITIVE FLOOR AND NOT `keras.constraints.NonNeg()`:
# NonNeg permits gamma == 0 EXACTLY. At gamma == 0 the Hessian is the ZERO matrix — still
# technically PSD, so the guarantee is not violated, but it becomes VACUOUS: `g` collapses
# to the constant `delta`, the update stops depending on the token state, and dE/dt == 0.
# The descent test would stay green on a dead layer. A small strictly-positive floor keeps
# the Hessian PD on the mean-zero subspace and keeps the guarantee non-vacuous, and it also
# means gradient descent cannot park gamma at an exactly-degenerate value.
#
# WHAT NOT TO DO: do not remove this default to "match the paper" (the paper does not
# constrain gamma because it never trains gamma negative; we ship a layer other people
# train). The constraint IS overridable — pass `gamma_constraint=None` — but that must be a
# DELIBERATE act, never a silent one. See decisions.md D-010.
_GAMMA_FLOOR = 1e-3


@keras.saving.register_keras_serializable()
class EnergyLayerNorm(keras.layers.Layer):
    """Energy Transformer layer normalization (scalar gamma, vector delta).

    **Intent**: provide the ``g = dL/dx`` "activation function" of the Energy Transformer,
    whose Lagrangian ``L`` has a PSD Hessian and therefore turns the block's hand-coded
    closed-form update into a *provable* descent direction on the block's scalar energy.

    **Mathematics** (over the LAST axis ``d``, per token, per sample — no token mixing):

    .. code-block:: text

        xbar = mean_j(x_j)                                  # scalar per token
        var  = mean_j((x_j - xbar)^2)                       # scalar per token
        g_i  = gamma * (x_i - xbar) / sqrt(var + eps) + delta_i

    This ``g`` is exactly the gradient of the Lagrangian

    .. code-block:: text

        L(x) = D * gamma * sqrt(var + eps) + sum_j delta_j * x_j
        g    = dL/dx

    Its Hessian ``dg/dx`` is PSD for ``gamma > 0``, hence for the ET energy ``E``:

    .. code-block:: text

        dE/dt = -(dE/dg)^T (dg/dx) (dE/dg) <= 0

    which is the whole reason this parameterization (scalar gamma, vector delta) exists.

    **Shape contract**: gamma is a SCALAR (``shape=()``); delta is a VECTOR
    (``shape=(D,)``). A per-feature gamma would break the Lagrangian identity above.

    **Numerical note: the constant-token cliff.** ``eps`` sits INSIDE the ``sqrt``, so as a
    token approaches CONSTANT (``var -> 0``) the Jacobian ``dg/dx`` does not merely grow — it
    saturates at a hard ceiling set by ``eps``:

    .. code-block:: text

        var >> eps :  eigenvalues of dg/dx  ~  1.4 .. 3.8         (a normal token)
        var == 0   :  eigenvalues of dg/dx  ->  gamma / sqrt(eps)
                                            ~  1.7 / sqrt(1e-5)  ~  538

    i.e. a **140-350x amplification** of the local gain, reached at ``var == 0`` (all measured;
    the numbers above are at ``gamma = 1.7``, ``eps = 1e-5``). A constant token is not exotic:
    an ``Embedding`` PAD row, an all-zero conv cell, and a collapsed early-training activation
    are all exactly ``var = 0``.

    Two things this is **NOT** (both VERIFIED numerically, do not "fix" either):

    * **It is not a broken guarantee.** ``dg/dx`` stays **PSD** across the cliff — the energy
      descent still holds. This is a CONDITIONING problem, not a correctness one.
    * **It is not an fp16 flush-to-zero bug.** Under ``mixed_float16``, ``eps = 1e-5`` is
      SUBNORMAL but REPRESENTABLE (fp16 min subnormal ~ ``6e-8``); it does not flush to zero,
      so ``sqrt(0 + eps)`` does not become ``sqrt(0)``. There is no division by zero here.

    **Mitigation for a caller who sees a training-stability cliff** (loss spikes on a batch
    with heavy padding, or in the first steps before activations spread out): raise
    ``norm_epsilon`` / ``epsilon``. The ceiling is ``gamma / sqrt(eps)``, so eps ``1e-5 ->
    1e-3`` cuts the worst-case gain 10x. The alternative — masking PAD tokens so they never
    reach the norm — is what ``EnergyTransformer`` already does for the Hopfield energy.

    :param epsilon: Small positive constant for numerical stability inside the sqrt.
        Defaults to ``1e-5``.
    :type epsilon: float
    :param gamma_initializer: Initializer for the scalar ``gamma``. Defaults to ``'ones'``
        (the paper requires ``gamma > 0`` for the PSD Hessian).
    :type gamma_initializer: Union[str, initializers.Initializer]
    :param delta_initializer: Initializer for the ``(D,)`` offset ``delta``.
        Defaults to ``'zeros'``.
    :type delta_initializer: Union[str, initializers.Initializer]
    :param gamma_constraint: Constraint applied to ``gamma`` after every optimizer step.
        **Defaults to a strictly-positive floor** (``ValueRangeConstraint(min_value=1e-3)``)
        because ``gamma > 0`` is the PRECONDITION for the PSD Hessian that makes the Energy
        Transformer's descent guarantee true. **Without it the guarantee is silently FALSE**:
        a trained ``gamma < 0`` makes ``dg/dx`` negative-definite and the block performs
        energy **ASCENT** — no error, no NaN, no failing test (measured: max ``diff(E)`` =
        ``+1.3e4`` at ``gamma = -1.0``). Pass ``None`` to disable it — which is a legitimate
        thing to want, but it must be a DELIBERATE choice. See the D-010 anchor above.
    :type gamma_constraint: Optional[constraints.Constraint]

    :raises ValueError: If ``epsilon <= 0``.

    Input shape:
        Tensor of shape ``(..., D)``; normalization is over the last axis. Typically
        ``(batch, num_tokens, embed_dim)``.

    Output shape:
        Identical to the input shape.

    Example:
        >>> layer = EnergyLayerNorm(epsilon=1e-5)
        >>> g = layer(keras.random.normal((2, 16, 64)))
        >>> g.shape
        (2, 16, 64)

    References:
        - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253, eq. (1)-(2).
    """

    # DECISION plan_2026-07-13_57c9833e/D-005
    # Do NOT add a `lagrangian()` / `energy()` method to this class. It is tempting
    # (the Lagrangian L is written out above, and the sibling ET layers DO expose an
    # `energy()`/`update()` pair) — but it would have ZERO call sites: the ET block's
    # reported energy is `E_ATT + E_HN` only; the LayerNorm Lagrangian is NOT a term in
    # E. Adding it, and then summing it into the block's reported energy, would make the
    # energy-descent test assert on the WRONG quantity and silently invalidate the
    # headline guarantee. Omitted per the use-before-reuse / earned-abstraction rule.
    # See decisions.md D-005.

    # Sentinel: distinguishes "caller said nothing" (-> apply the default positivity floor)
    # from "caller explicitly said None" (-> deliberately UNCONSTRAINED gamma). Without it,
    # `gamma_constraint=None` could not turn the constraint OFF.
    _DEFAULT_CONSTRAINT = "__default__"

    def __init__(
        self,
        epsilon: float = 1e-5,
        gamma_initializer: Union[str, initializers.Initializer] = 'ones',
        delta_initializer: Union[str, initializers.Initializer] = 'zeros',
        gamma_constraint: Any = _DEFAULT_CONSTRAINT,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ----- validation -----
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError(f"epsilon must be a positive number, got {epsilon}")

        # ----- store ALL configuration -----
        self.epsilon = float(epsilon)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.delta_initializer = initializers.get(delta_initializer)

        # DECISION plan_2026-07-13_57c9833e/D-010
        # ON BY DEFAULT. Do NOT flip this default to `None` "because the paper doesn't
        # constrain gamma": the paper never trains gamma negative, and we ship a trainable
        # layer to people who will. `gamma < 0` => the Lagrangian's Hessian dg/dx is NOT PSD
        # => the block silently performs energy ASCENT while still running, still training
        # and still emitting finite output. Reused (not re-implemented) from
        # `dl_techniques.constraints.value_range_constraint`. See decisions.md D-010.
        self.gamma_constraint = (
            ValueRangeConstraint(min_value=_GAMMA_FLOOR)
            if gamma_constraint is self._DEFAULT_CONSTRAINT
            else constraints.get(gamma_constraint)
        )

        # ----- weights are created in build() -----
        self.gamma: Optional[keras.Variable] = None
        self.delta: Optional[keras.Variable] = None

        self.supports_masking = True

        logger.debug(
            f"Initialized EnergyLayerNorm with "
            f"epsilon={self.epsilon}, "
            f"gamma_initializer={gamma_initializer}, "
            f"delta_initializer={delta_initializer}"
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the scalar ``gamma`` and the ``(D,)`` vector ``delta``.

        :param input_shape: Shape of the input tensor; the last axis is the feature
            axis ``D``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the last axis of ``input_shape`` is undefined.
        """
        if self.built:
            return

        feature_dim = input_shape[-1]
        if feature_dim is None:
            raise ValueError(
                "The last axis of the input shape must be defined, got "
                f"input_shape={input_shape}"
            )

        # gamma is a SCALAR (paper eq. 1). This is NOT a bug and must NOT be
        # "fixed" to a per-feature vector: a vector gamma breaks g = dL/dx. A vector gamma
        # is not merely "different" — it makes the Jacobian dg/dx ASYMMETRIC, so it is no
        # longer the Hessian of ANY scalar Lagrangian and the descent guarantee evaporates.
        # That is guarded BEHAVIORALLY by `test_jacobian_is_symmetric` (S16), not just by
        # the shape assertion below it.
        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer=self.gamma_initializer,
            constraint=self.gamma_constraint,   # positivity floor by default (D-010)
            trainable=True,
            dtype=self.dtype,
        )

        # delta is a VECTOR of dim D (the Lagrangian's linear term).
        self.delta = self.add_weight(
            name="delta",
            shape=(int(feature_dim),),
            initializer=self.delta_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the ET layer norm: ``gamma * (x - xbar) / sqrt(var + eps) + delta``.

        :param inputs: Input tensor of shape ``(..., D)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; present for interface consistency.
        :type training: Optional[bool]

        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # Statistics over the LAST axis only — per token, per sample. No token mixing.
        x_bar = ops.mean(inputs, axis=-1, keepdims=True)
        centered = inputs - x_bar
        variance = ops.mean(ops.square(centered), axis=-1, keepdims=True)

        inv_std = ops.rsqrt(variance + self.epsilon)

        # gamma is a scalar => broadcasts over everything; delta is (D,) => broadcasts
        # over the leading (batch, token) axes.
        return self.gamma * centered * inv_std + self.delta

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape (identity — the layer is shape-preserving).

        Uses only the passed shape and stored config, never a weight shape, so it is
        valid on an UNBUILT layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'delta_initializer': initializers.serialize(self.delta_initializer),
            # Serialized EXPLICITLY (D-010). If it were dropped from get_config, a saved
            # model would silently reload with an UNCONSTRAINED gamma and could then train
            # itself into energy ascent — the exact defect this constraint exists to stop.
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
        })
        return config

    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergyLayerNorm":
        """Reconstruct the layer from its serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]

        :return: A new ``EnergyLayerNorm`` instance.
        :rtype: EnergyLayerNorm
        """
        config = dict(config)
        for key in ('gamma_initializer', 'delta_initializer'):
            if key in config:
                config[key] = initializers.deserialize(config[key])
        if 'gamma_constraint' in config:
            # NOTE: a serialized `None` round-trips as an explicit `None` (deliberately
            # unconstrained), NOT as the default floor — the sentinel is only for a caller
            # who never mentioned the argument at all.
            config['gamma_constraint'] = constraints.deserialize(
                config['gamma_constraint']
            )
        return cls(**config)

# ---------------------------------------------------------------------
