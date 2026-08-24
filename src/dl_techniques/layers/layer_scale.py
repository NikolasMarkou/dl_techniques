"""
Learnable element-wise scaling for adaptive feature and pathway weighting.

This layer embodies the principle of minimal parameterized gating, a design
paradigm that inserts the smallest possible learnable transformation into a
computation graph in order to let the network modulate signal magnitude without
altering its content. The core idea is that many architectural decisions
normally hard-coded as fixed constants, how much of a residual branch to admit,
how strongly to weight a feature map, are better left to gradient descent, and
that a single multiplicative parameter is sufficient to express them. Unlike a
`Dense` layer, no mixing occurs across positions or channels; the operation is
purely a rescaling, so the layer can only attenuate or amplify what is already
present.

Architecturally, the layer applies a trainable parameter `gamma` element-wise:

`output = gamma * input`

Two modes determine the shape of `gamma` and therefore the granularity of
control:

1.  **GLOBAL.** A single scalar broadcast across the entire tensor. All features
    are scaled uniformly, so the parameter expresses one importance score for
    the whole pathway. This is the form used to gate a residual branch, where
    the quantity being learned is how much of the branch to contribute.
2.  **CHANNEL.** A vector of length equal to the last dimension, with each
    channel scaled independently. This lets the network re-weight feature maps
    relative to one another, which amounts to a static, input-independent form
    of channel attention.

Two defaults are chosen specifically for training stability rather than
generality. Initializing `gamma` to ones makes the layer an exact identity at
step zero, so inserting it into an existing network leaves forward signal
propagation and gradient magnitudes unchanged; any deviation from identity is
something the network chose rather than something imposed at initialization.
Constraining `gamma` to be non-negative prevents the layer from inverting the
sign of a feature, which restricts it to the semantics of a soft gate: the
parameter answers how much, never in which direction. Both defaults are
overridable when a signed or non-identity scale is genuinely wanted.

One implementation detail is load-bearing under mixed precision. Keras stores
`gamma` in float32 even when the compute policy is `mixed_float16`, so the
multiplication is cast to the input's compute dtype before it is applied.
Without the cast, the float32-weight against float16-activation pairing is
rejected by XLA during gradient computation as disallowed mixed precision. The
cast is a no-op on the pure float32 path.

Conceptually this is the standalone form of the learnable `gamma` found inside
`BatchNormalization` and `LayerNormalization`, separated from any normalization
statistics so that the scaling can be placed independently of where activations
are normalized.

References:
    - Ioffe and Szegedy, 2015. Batch Normalization: Accelerating Deep Network
      Training by Reducing Internal Covariate Shift.
      (https://arxiv.org/abs/1502.03167)
    - Bachlechner et al., 2020. ReZero is All You Need: Fast Convergence at
      Large Depth. (https://arxiv.org/abs/2003.04887)
    - Touvron et al., 2021. Going Deeper with Image Transformers (LayerScale).
      (https://arxiv.org/abs/2103.17239)
    - Hu et al., 2018. Squeeze-and-Excitation Networks.
      (https://arxiv.org/abs/1709.01507)

"""

import keras
from keras import ops
from enum import Enum
from typing import Dict, Any, Optional, Union, Tuple

# ---------------------------------------------------------------------


class MultiplierType(Enum):
    """Enumeration for multiplier types (GLOBAL or CHANNEL)."""

    GLOBAL = 0
    CHANNEL = 1

    @staticmethod
    def from_string(type_str: Union[str, "MultiplierType"]) -> "MultiplierType":
        """Convert string to MultiplierType enum.

        :param type_str: String representation or MultiplierType instance.
        :type type_str: Union[str, MultiplierType]
        :return: MultiplierType enum value.
        :rtype: MultiplierType
        :raises ValueError: If type_str is invalid.
        """
        if type_str is None:
            raise ValueError("type_str must not be null")
        if isinstance(type_str, MultiplierType):
            return type_str
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")

        # Clean string and get enum value
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        try:
            return MultiplierType[type_str]
        except KeyError:
            raise ValueError(f"Invalid multiplier type: {type_str}")

    def to_string(self) -> str:
        """Convert enum to string representation.

        :return: String representation of the enum.
        :rtype: str
        """
        return self.name


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LearnableMultiplier(keras.layers.Layer):
    """Learnable element-wise multiplier for adaptive feature scaling.

    This layer introduces trainable scaling parameters applied either globally
    (single scalar) or per-channel. In global mode,
    ``output = gamma * input`` where gamma is scalar. In channel mode,
    ``output = gamma * input`` where gamma has shape ``(channels,)`` and
    multiplication is element-wise. The layer defaults to identity
    initialization (``ones``) and non-negative constraint for stable training.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │     Input (any shape)        │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  gamma * input               │
        │  (GLOBAL: scalar gamma)      │
        │  (CHANNEL: per-channel gamma)│
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │    Output (same shape)       │
        └──────────────────────────────┘

    :param multiplier_type: Type of multiplier operation: ``'GLOBAL'`` or
        ``'CHANNEL'``. Defaults to ``'CHANNEL'``.
    :type multiplier_type: Union[MultiplierType, str]
    :param initializer: Initializer for multiplier weights. Defaults to ``'ones'``.
    :type initializer: Union[str, keras.initializers.Initializer]
    :param regularizer: Optional regularizer for multiplier weights. Defaults to None.
    :type regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param constraint: Optional constraint for multiplier weights.
        Defaults to ``'non_neg'``.
    :type constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If multiplier_type is invalid or input dimensions are
        incompatible with CHANNEL mode.
    """

    def __init__(
        self,
        multiplier_type: Union[MultiplierType, str] = MultiplierType.CHANNEL,
        initializer: Union[str, keras.initializers.Initializer] = "ones",
        regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        constraint: Optional[Union[str, keras.constraints.Constraint]] = "non_neg",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate and store configuration parameters
        self.multiplier_type = MultiplierType.from_string(multiplier_type)
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)

        # Initialize weight attribute - created in build()
        self.gamma = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's trainable multiplier weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is incompatible with multiplier type.
        """
        # Determine weight shape based on multiplier type
        if self.multiplier_type == MultiplierType.GLOBAL:
            # Global multiplier: single scalar value broadcasted across entire tensor
            weight_shape = (1,)
        elif self.multiplier_type == MultiplierType.CHANNEL:
            # Per-channel multiplier: one weight per channel (last dimension)
            if len(input_shape) < 2:
                raise ValueError(
                    f"CHANNEL multiplier requires input with at least 2 dimensions, "
                    f"got shape: {input_shape}"
                )
            weight_shape = (input_shape[-1],)
        else:
            # This should never happen due to enum validation, but defensive programming
            raise ValueError(f"Invalid multiplier_type: {self.multiplier_type}")

        # Create the trainable multiplier weight
        self.gamma = self.add_weight(
            name="gamma",
            shape=weight_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            trainable=True,
            dtype=self.dtype
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Apply the learnable multipliers to inputs.

        :param inputs: Input tensor to be scaled.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API consistency.
        :type training: Optional[bool]
        :param kwargs: Additional call arguments.
        :return: Scaled tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Element-wise multiplication using Keras ops for backend compatibility.
        # Cast gamma (stored fp32, including under a mixed_float16 policy) to the input's
        # compute dtype so fp16 activations multiply an fp16 scale. Without the cast the
        # fp32-weight x fp16-activation mismatch is rejected by XLA in the gradient
        # ("mixed precision disallowed"). No-op when dtypes already match (fp32 path).
        return ops.multiply(inputs, ops.cast(self.gamma, inputs.dtype))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (identical to input shape).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "multiplier_type": self.multiplier_type.to_string(),
            "initializer": keras.initializers.serialize(self.initializer),
            "regularizer": keras.regularizers.serialize(self.regularizer),
            "constraint": keras.constraints.serialize(self.constraint),
        })
        return config


# ---------------------------------------------------------------------