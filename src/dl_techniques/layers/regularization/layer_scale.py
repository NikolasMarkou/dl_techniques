"""
``LayerScale`` multiplies its input element-wise by a trainable parameter
``gamma``: ``output = gamma * input``.

Two modes set the shape of ``gamma``. In ``GLOBAL`` mode ``gamma`` is a single
scalar, uniformly scaling the whole tensor -- the form used to gate how much
of a residual branch to admit. In ``CHANNEL`` mode ``gamma`` has one value per
channel, giving each feature map its own static, input-independent weight.
``gamma`` defaults to ones, so the layer starts as an identity and any
deviation is learned rather than imposed at initialization. It defaults to a
non-negative constraint, so it can only scale, never flip sign.

Under a ``mixed_float16`` policy, ``gamma`` is stored in float32; `call()`
casts it to the input's compute dtype before multiplying, since XLA rejects
an uncast float32-weight times float16-activation multiply during the
gradient pass. This is the standalone form of the learnable scale inside
``BatchNormalization`` and ``LayerNormalization``, usable anywhere in a graph
independent of a normalization layer.

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
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

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


@register_dl_technique("dl_techniques.layers.regularization.layer_scale")
class LayerScale(keras.layers.Layer):
    """Learnable element-wise multiplier for adaptive feature scaling.

    This layer introduces trainable scaling parameters applied either globally
    (single scalar) or per-channel. In global mode,
    ``output = gamma * input`` where gamma is scalar. In channel mode,
    ``output = gamma * input`` where gamma has shape ``(channels,)`` and
    multiplication is element-wise. The layer defaults to identity
    initialization (``ones``) and non-negative constraint for stable training.

    Architecture:

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

        self.multiplier_type = MultiplierType.from_string(multiplier_type)
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)

        self.gamma = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's trainable multiplier weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is incompatible with multiplier type.
        """
        if self.multiplier_type == MultiplierType.GLOBAL:
            weight_shape = (1,)
        elif self.multiplier_type == MultiplierType.CHANNEL:
            if len(input_shape) < 2:
                raise ValueError(
                    f"CHANNEL multiplier requires input with at least 2 dimensions, "
                    f"got shape: {input_shape}"
                )
            weight_shape = (input_shape[-1],)
        else:
            raise ValueError(f"Invalid multiplier_type: {self.multiplier_type}")

        self.gamma = self.add_weight(
            name="gamma",
            shape=weight_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            trainable=True,
            dtype=self.dtype
        )

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
        # Cast gamma to the input's compute dtype; under mixed_float16 it is
        # stored in fp32 and an uncast multiply is rejected by XLA in the gradient pass.
        return ops.multiply(inputs, ops.cast(self.gamma, inputs.dtype))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
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