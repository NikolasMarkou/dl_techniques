"""Decorrelated, gated dense alternative, built by :class:`OrthoBlock`.

A standard Dense layer lets features stay correlated and offers no feature
selection. This block runs a fixed four-stage pipeline instead: a linear
projection whose kernel is regularized toward a partial isometry (so
projected features decorrelate), a centering normalization
(``ZeroCenteredRMSNorm``), a learnable per-feature gate constrained to
``[0, 1]``, and the activation, applied last, after the gate.

The centering step makes the block's bias direction unidentifiable and
gives it no restoring force, so ``use_bias`` defaults to False; add an
offset after the block instead if you need one. ``scale_initial_value``
defaults to 0.8, not 0.5, because 0.5 is the unstable equilibrium of the
binary-preference regularizer on the gate. With a ReLU-family activation a
gate that reaches 0 cannot reopen (``relu'(0) = 0``); a warning is logged
in that case.

The orthonormality penalty is
``(lambda / sqrt(rank)) * ||G - I||_F^2``, where ``G`` is the Gram matrix
over whichever kernel axis is smaller. See ``soft_orthogonal.py`` for the
derivation.

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

from dl_techniques.layers.regularization.layer_scale import LayerScale
from dl_techniques.layers.norms.zero_centered_rms_norm import ZeroCenteredRMSNorm
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from dl_techniques.initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# Activations whose derivative at zero is zero, which makes a gate that reaches
# the lower bound of ValueRangeConstraint absorbing rather than reflecting.
_ZERO_DERIVATIVE_AT_ORIGIN = frozenset({"relu", "relu6"})


@register_dl_technique("dl_techniques.layers.structured_linear.orthoblock")
class OrthoBlock(keras.layers.Layer):
    """Structured feature learning block with orthonormal regularization and gating.

    Pipeline, in execution order: an orthonormally regularized linear
    projection, `ZeroCenteredRMSNorm`, a learnable gate in ``[0, 1]``, and then
    the activation. The kernel is softly constrained toward a partial isometry
    via ``(lambda / sqrt(rank)) * ||G - I||^2_F``, where ``G`` is the Gram over
    whichever kernel axis is smaller. See the module docstring for what that
    constraint does and does not achieve.

    Architecture:

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
        non-negative.
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

        self.dense = keras.layers.Dense(
            units=self.units,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.ortho_reg,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            name="ortho_dense"
        )

        self.norm = ZeroCenteredRMSNorm(
            axis=-1,
            use_scale=False,
            name="zero_centered_rms_norm"
        )

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
        z = self.dense(inputs, training=training)

        # Centers, then divides by RMS; output is exactly zero-mean.
        z_norm = self.norm(z, training=training)

        outputs = self.constrained_scale(z_norm, training=training)

        # Skipped when activation is the identity (keras.activations.get(None)).
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
        output_shape = list(input_shape)

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
