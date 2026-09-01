"""
Learn decorrelated and gated features through a structured pipeline.

This layer provides a mathematically-motivated alternative to a standard Dense
layer by enforcing a structured computational flow. The core philosophy is to
decouple the feature transformation into three distinct, interpretable steps:
a projection toward an orthonormal basis, a normalization step to stabilize
activations, and a learned gating mechanism for feature selection.

Architecturally, the layer implements a four-stage pipeline, in this order:

1.  **Orthonormal Projection:** A linear transformation whose kernel is softly
    regularized toward a partial isometry, so that every nonzero singular value
    approaches one and the projected features are mutually decorrelated.
2.  **Magnitude Stabilization:** `ZeroCenteredRMSNorm`, which subtracts the
    mean over the feature axis and divides by the RMS of the centered values.
3.  **Feature Gating:** A learnable scale vector `s`, constrained to `[0, 1]`,
    applied element-wise.
4.  **Non-Linearity:** The activation function, applied last.

Note that gating precedes the activation. An earlier version of this docstring
listed those two in the opposite order; the code order above is authoritative.

The normalization centers
-------------------------
`ZeroCenteredRMSNorm` is documented in its own source as exactly
`keras.layers.LayerNormalization(center=False)`, verified to 3.576e-07. It
subtracts the feature-axis mean. An earlier version of this docstring claimed
the opposite, that centering was omitted in order to preserve directional
information. That claim was wrong, and three consequences follow from the
correction:

*   The block output is exactly zero-mean across features, so it spans an
    `(units - 1)`-dimensional subspace. One output direction is structurally
    unavailable regardless of what the kernel learns.
*   The loss is invariant to adding any per-sample constant to every
    pre-normalization feature. That makes the mean of the bias, and the
    column-mean of the kernel, unidentifiable: `d + 1` parameter directions the
    loss cannot see.
*   Combined with the fact that an unregularized bias is the only parameter in
    this block with no restoring force at all, and therefore diffuses as
    `eta * sqrt(units * steps)` until it swamps the input-dependent part of the
    pre-normalization signal, `use_bias` defaults to False. If you want an
    additive offset, put it after the normalization where it is identifiable
    and bounded.

The orthonormality constraint
-----------------------------
The penalty actually applied is

    Loss_ortho = (lambda / sqrt(rank)) * ||G - I||^2_F

where `G` is the Gram matrix over whichever kernel axis is smaller and `rank`
is its side length:

    units <= input_dim :  G = W^T W,  rank = units
                          orthonormal output features
    units >  input_dim :  G = W W^T,  rank = input_dim
                          orthonormal input directions

Both say that every nonzero singular value of `W` equals one, and only one of
them is reachable at any given shape. The `sqrt(rank)` divisor is what makes
the achieved constraint independent of layer width; see `soft_orthogonal.py`
for the derivation.

What the constraint actually achieves
-------------------------------------
The penalty competes against the task gradient, and the balance point is
optimizer independent, since a diagonal preconditioner cannot move a stationary
point. Writing `g` for the per-coordinate task gradient magnitude and `nu` for
the column norm, the equilibrium Gram off-diagonal is approximately

    eps ~ g * sqrt(input_dim) / (4 * ortho_reg_factor * nu)

Read that carefully before reporting a final `||G - I||` number: the achieved
orthogonality is proportional to the residual task gradient. It is worst early
in training, when the gradient-flow rationale matters most, and best at the
end, when it matters least. A small final deviation is largely evidence that
the loss converged, not that the constraint did work.

The radial direction behaves differently. Weight decay and any L2 term are
exactly parallel to `W`, hence exactly tangent to the orthogonality
constraints, so they shorten the columns without changing the angles between
them. Column length is set by a three-way balance between the orthonormality
penalty, the optimizer's decay, and the Adam noise inflation term
`eta * fan_in * kappa^2 / 2` that any scale-invariant weight accumulates. With
`ortho_reg_factor = 0.01` and the corrected divisor, the penalty wins that
balance and column norms sit within a few percent of one.

Migration from the previous release
-----------------------------------
Three defaults changed underneath this layer:

*   The regularizer's size divisor went from `rank**2` to `sqrt(rank)`, so the
    same `ortho_reg_factor` is now `units**1.5` stronger. At 512 units that is
    a factor of about 11600. The old effective strength contributed roughly
    0.02 percent of the radial restoring force; the new one contributes
    essentially all of it. `ortho_reg_factor = 0.01` is retained because it is
    now correctly scaled, but any value tuned against the old behaviour should
    be divided by `units**1.5` before comparison.
*   The regularizer's `l2_coefficient` default went from `1e-4` to `0.0`. This
    layer now passes both `l1` and `l2` explicitly, so it does not depend on
    that default either way.
*   The hardcoded `l1_coefficient=1e-5` is gone. It was stronger than typical
    weight decay in gradient units, and since L1 combined with orthonormality
    is jointly minimized by signed permutation matrices, it biased the solution
    toward the trivial, axis-aligned corner of the orthogonal group. It is now
    `ortho_l1_factor`, default 0.0.

The gate
--------
Each `s_i` in `[0, 1]` acts as a differentiable gate, and the values can be
inspected after training as a rough feature-importance readout. Three caveats:

*   `scale_initial_value` defaults to 0.8, not 0.5. The binary preference
    regularizer `mu * s * (1 - s)` has zero derivative at exactly 0.5, so that
    point is the unstable maximum of a double well. Weight decay then supplies
    a one-sided downward tilt, and gates collapse toward zero systematically
    rather than sorting themselves by usefulness. 0.8 starts inside the open
    well.
*   The gate is applied before the activation. For a positively homogeneous
    activation such as ReLU the two commute and the gate is purely
    multiplicative. For GELU or SiLU they do not: since the normalized input
    has unit RMS, a gate at 0.1 places the pre-activation inside the
    near-linear region, so a partly closed gate makes its unit approximately
    linear rather than merely small. The feature-importance reading holds
    cleanly only for homogeneous activations.
*   With ReLU specifically, `s_i = 0` is absorbing. Keras defines
    `relu'(0) = 0`, so a gate that reaches zero receives no gradient and cannot
    reopen. A warning is logged if a ReLU-family activation is configured.

Because `ValueRangeConstraint` projects after the optimizer step, a pinned gate
keeps accumulating momentum in the forbidden direction, giving roughly
`1/(1 - beta_1)` steps of hysteresis when the task gradient reverses. If that
matters, disable the binary preference with `binary_preference_factor=0.0` or
reparameterize the gate outside this layer.

References:
    - Saxe et al., 2013. Exact solutions to the nonlinear dynamics of
      learning in deep linear neural networks (for orthogonal initialization).
    - Cisse et al., 2017. Parseval Networks: Improving Robustness to
      Adversarial Examples (for orthogonal regularization).
    - Zhang & Sennrich, 2019. Root Mean Square Layer Normalization.
    - Loshchilov & Hutter, 2019. Decoupled Weight Decay Regularization
      (for why the L1/L2 terms above default to zero).
"""

import keras
from typing import Optional, Union, Any, Tuple, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LayerScale
from .norms.zero_centered_rms_norm import ZeroCenteredRMSNorm
from ..constraints.value_range_constraint import ValueRangeConstraint
from ..regularizers.binary_preference import BinaryPreferenceRegularizer
from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from ..initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Activations whose derivative at zero is zero, which makes a gate that reaches
# the lower bound of ValueRangeConstraint absorbing rather than reflecting.
_ZERO_DERIVATIVE_AT_ORIGIN = frozenset({"relu", "relu6"})

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.orthoblock")
class OrthoBlock(keras.layers.Layer):
    """Structured feature learning block with orthonormal regularization and gating.

    Pipeline, in execution order: an orthonormally regularized linear
    projection, `ZeroCenteredRMSNorm`, a learnable gate in ``[0, 1]``, and then
    the activation. The kernel is softly constrained toward a partial isometry
    via ``(lambda / sqrt(rank)) * ||G - I||^2_F``, where ``G`` is the Gram over
    whichever kernel axis is smaller. See the module docstring for what that
    constraint does and does not achieve.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │  Input [..., input_dim]             │
        └─────────────────┬───────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  Dense(units) + Orthonormal Reg.    │
        │  z = xW (+ b, off by default)       │
        └─────────────────┬───────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  ZeroCenteredRMSNorm                │
        │  centers, then divides by RMS       │
        │  output is exactly zero-mean        │
        └─────────────────┬───────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  Constrained Scale s in [0,1]       │
        │  (learnable feature gates)          │
        └─────────────────┬───────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  Activation (optional)              │
        └─────────────────┬───────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  Output [..., units], zero-mean     │
        │  before gating and activation       │
        └─────────────────────────────────────┘

    :param units: Dimensionality of the output space. Must be a positive
        integer. Note that centering costs one dimension: the pre-gate
        representation spans ``units - 1`` directions.
    :type units: int
    :param activation: Activation applied after the gate. String name,
        callable, or ``None`` for linear. A ReLU-family activation makes a
        fully closed gate unrecoverable; see the module docstring.
    :type activation: Optional[Union[str, Callable]]
    :param use_bias: Whether the dense layer includes a bias vector. Defaults
        to False. Before a centering normalization the bias mean is
        unidentifiable, and an unregularized bias is the only parameter here
        with no restoring force.
    :type use_bias: bool
    :param ortho_reg_factor: Strength of the orthonormality penalty. Must be
        non-negative. Now ``units ** 1.5`` stronger than in the previous
        release at the same value; see the migration note.
    :type ortho_reg_factor: float
    :param ortho_l1_factor: Coupled L1 penalty on the kernel, applied by the
        same regularizer. Defaults to 0.0. Nonzero values bias the solution
        toward signed permutation matrices, which are orthogonal but mix
        nothing.
    :type ortho_l1_factor: float
    :param ortho_l2_factor: Coupled L2 penalty on the kernel. Defaults to 0.0.
        Prefer the optimizer's decoupled ``weight_decay``: a coupled L2 passes
        through the preconditioner and makes the equilibrium weight norm
        depend on the absolute loss scale.
    :type ortho_l2_factor: float
    :param kernel_initializer: Initializer for the dense weight matrix.
        ``None`` builds a fresh ``OrthogonalHypersphereInitializer`` per layer.
    :type kernel_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param bias_initializer: Initializer for the bias vector.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Optional regularizer for the bias vector.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param scale_initial_value: Initial gate value, in ``[0.0, 1.0]``. Defaults
        to 0.8. Avoid exactly 0.5, which is the unstable maximum of the binary
        preference well.
    :type scale_initial_value: float
    :param binary_preference_factor: Strength of the regularizer pushing gates
        toward ``{0, 1}``. Set to 0.0 to disable binarization and keep graded
        gates.
    :type binary_preference_factor: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]]] = None,
        use_bias: bool = False,
        ortho_reg_factor: float = 0.01,
        ortho_l1_factor: float = 0.0,
        ortho_l2_factor: float = 0.0,
        kernel_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        scale_initial_value: float = 0.8,
        binary_preference_factor: float = 1e-4,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate input parameters with clear error messages
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")
        if not isinstance(ortho_reg_factor, (int, float)) or ortho_reg_factor < 0:
            raise ValueError(f"ortho_reg_factor must be non-negative, got {ortho_reg_factor}")
        if not isinstance(ortho_l1_factor, (int, float)) or ortho_l1_factor < 0:
            raise ValueError(f"ortho_l1_factor must be non-negative, got {ortho_l1_factor}")
        if not isinstance(ortho_l2_factor, (int, float)) or ortho_l2_factor < 0:
            raise ValueError(f"ortho_l2_factor must be non-negative, got {ortho_l2_factor}")
        if not isinstance(binary_preference_factor, (int, float)) or binary_preference_factor < 0:
            raise ValueError(
                f"binary_preference_factor must be non-negative, got {binary_preference_factor}"
            )
        if not isinstance(scale_initial_value, (int, float)) or not (0.0 <= scale_initial_value <= 1.0):
            raise ValueError(f"scale_initial_value must be between 0.0 and 1.0, got {scale_initial_value}")

        # Store ALL configuration parameters for serialization
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.ortho_reg_factor = ortho_reg_factor
        self.ortho_l1_factor = ortho_l1_factor
        self.ortho_l2_factor = ortho_l2_factor
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.scale_initial_value = scale_initial_value
        self.binary_preference_factor = binary_preference_factor

        # A default-argument initializer instance would be constructed once at
        # import and shared by every OrthoBlock in the process. Build a fresh
        # one per layer instead.
        if kernel_initializer is None:
            self.kernel_initializer = OrthogonalHypersphereInitializer()
        else:
            self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # keras.activations.get(None) returns `linear`, not None, so a
        # `self.activation is not None` guard never fires. Compare against the
        # identity function instead. This is also stable across a
        # serialize/deserialize round trip, where None becomes "linear".
        self._is_identity_activation = self.activation is keras.activations.linear

        self._warn_if_gate_death_is_reachable()

        # CREATE orthonormal regularizer.
        # Every coefficient is passed explicitly: the library defaults for l1
        # and l2 have changed once already, and a hidden coupled L2 competing
        # with the orthonormality term is exactly the failure this layer used
        # to have.
        self.ortho_reg = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=self.ortho_reg_factor,
            l1_coefficient=self.ortho_l1_factor,
            l2_coefficient=self.ortho_l2_factor,
            use_matrix_scaling=True,
        )

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        # Dense layer with orthonormal regularization
        self.dense = keras.layers.Dense(
            units=self.units,
            activation=None,  # Activation applied separately at the end
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.ortho_reg,  # Orthonormal regularization
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            name="ortho_dense"
        )

        # Centering normalization. use_scale=False because the gate below is
        # the only magnitude parameter in this block; two of them in series
        # would be redundant and would give the block a second, unconstrained
        # scale direction.
        self.norm = ZeroCenteredRMSNorm(
            axis=-1,
            use_scale=False,
            name="zero_centered_rms_norm"
        )

        # Constrained learnable scaling for feature gating
        self.constrained_scale = LayerScale(
            multiplier_type="CHANNEL",
            initializer=keras.initializers.Constant(self.scale_initial_value),
            regularizer=(
                BinaryPreferenceRegularizer(multiplier=self.binary_preference_factor)
                if self.binary_preference_factor > 0.0
                else None
            ),
            constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0),
            name="constrained_scale"
        )

    def _warn_if_gate_death_is_reachable(self) -> None:
        """Warn when the activation makes a fully closed gate unrecoverable.

        The gate is applied before the activation, so the gradient reaching
        ``s_i`` carries a factor ``phi'(s_i * z_i)``. For a ReLU-family
        activation that factor is zero at ``s_i = 0``, and the lower bound of
        ``ValueRangeConstraint`` is therefore absorbing rather than reflecting:
        a gate that closes can never reopen. GELU, SiLU, tanh and the linear
        default all have nonzero derivative at the origin and are unaffected.
        """
        name = getattr(self.activation, "__name__", "")
        if name in _ZERO_DERIVATIVE_AT_ORIGIN:
            logger.warning(
                f"OrthoBlock configured with activation '{name}', whose "
                f"derivative at zero is zero. Combined with the [0, 1] gate "
                f"constraint this makes s_i = 0 an absorbing state: a gate "
                f"that closes receives no gradient and cannot reopen. Prefer "
                f"gelu or silu, or set binary_preference_factor=0.0 to remove "
                f"the pressure driving gates to the boundary."
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be defined for OrthoBlock")

        input_dim = int(input_shape[-1])

        # The regularizer builds the Gram over whichever kernel axis is
        # smaller, so an expansion layer gets a reachable target rather than a
        # penalty floor. The statement being enforced does change, though, so
        # say so once at build time rather than leaving it to the regularizer's
        # own log line.
        if self.units > input_dim:
            logger.info(
                f"OrthoBlock: units ({self.units}) exceeds input_dim "
                f"({input_dim}), so {self.units} mutually orthonormal output "
                f"features do not exist. The regularizer will constrain the "
                f"{input_dim} input directions instead, which is the "
                f"equivalent reachable target. Centering also removes one "
                f"output direction, so the representation spans "
                f"{self.units - 1} dimensions of which at most {input_dim} "
                f"are independent."
            )

        self.dense.build(input_shape)
        dense_output_shape = self.dense.compute_output_shape(input_shape)

        self.norm.build(dense_output_shape)
        norm_output_shape = self.norm.compute_output_shape(dense_output_shape)

        self.constrained_scale.build(norm_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward computation through the orthonormal block pipeline.

        :param inputs: Input tensor with shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training or inference mode.
        :type training: Optional[bool]
        :return: Output tensor with shape ``(..., units)``.
        :rtype: keras.KerasTensor
        """
        # Stage 1: Dense projection with orthonormal regularization
        z = self.dense(inputs, training=training)

        # Stage 2: Center over the feature axis, then divide by the RMS of the
        # centered values. The result is exactly zero-mean across features.
        z_norm = self.norm(z, training=training)

        # Stage 3: Constrained scaling for learnable feature gating
        outputs = self.constrained_scale(z_norm, training=training)

        # Stage 4: Apply activation function, last. Skipped when it is the
        # identity, which is what keras.activations.get(None) returns.
        if not self._is_identity_activation:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension replaced by ``units``.
        :rtype: Tuple[Optional[int], ...]
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)

        # Replace last dimension with units
        output_shape[-1] = self.units

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "ortho_reg_factor": self.ortho_reg_factor,
            "ortho_l1_factor": self.ortho_l1_factor,
            "ortho_l2_factor": self.ortho_l2_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "scale_initial_value": self.scale_initial_value,
            "binary_preference_factor": self.binary_preference_factor,
        })
        return config

# ---------------------------------------------------------------------
