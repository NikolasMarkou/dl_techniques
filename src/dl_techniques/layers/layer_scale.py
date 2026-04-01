"""
This module provides a specialized Keras layer, `LearnableMultiplier`, for implementing
learnable, element-wise scaling operations. It allows a network to adaptively scale
feature maps, either globally or on a per-channel basis, providing a flexible
building block for modern neural network architectures.

The primary purpose of this layer is to introduce a simple, data-driven scaling
factor into a model's computation graph. Instead of using a fixed scalar or a
complex transformation (like a `Dense` layer), this layer learns a parameter `gamma`
that multiplies the entire input tensor. This can be used to dynamically adjust the
magnitude of activations, effectively allowing the network to learn the importance of
certain features or pathways. It is conceptually similar to the learnable `gamma`
parameter in `BatchNormalization` or `LayerNormalization`, but offered as a
standalone layer.

Key Features and Mechanisms:

1.  **Learnable Scaling Parameter (`gamma`):** The core of the layer is a trainable
    weight `gamma` that performs the element-wise multiplication.

2.  **Two Operational Modes (`multiplier_type`):**
    -   **`GLOBAL`:** A single scalar `gamma` is learned and broadcasted across the
        entire input tensor. This uniformly scales all features, learning a global
        importance score for the entire tensor.
    -   **`CHANNEL`:** A vector `gamma` is learned with a size equal to the number of
        input channels (the last dimension). Each channel is multiplied by its own
        unique `gamma` value, allowing the network to independently re-weight each
        feature map.

3.  **Sensible Defaults for Stability:**
    -   **Initializer:** Defaults to `ones`, meaning the layer initially acts as an
        identity function (`output = 1 * input`). This is crucial for stable
        training, as it ensures that inserting the layer into a network does not
        drastically change the signal propagation at the beginning of training.
    -   **Constraint:** Defaults to `non_neg`, ensuring the learned multipliers are
        always zero or positive. This is useful for preventing the layer from
        flipping the sign of features and allows it to function as a "soft gate" that
        can only attenuate or amplify signals.

Common Use Cases:
-   **Gating Residual Connections:** Used in residual blocks to learn how much of the
    residual to add: `output = x + LearnableMultiplier(type='GLOBAL')(residual_block(x))`.
-   **Feature Re-weighting:** Dynamically adjusting the importance of different channels
    before they are fused or combined with other features.
-   **Simple Attention:** Acting as a very simple channel-wise attention mechanism.
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
        # Element-wise multiplication using Keras ops for backend compatibility
        return ops.multiply(inputs, self.gamma)

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