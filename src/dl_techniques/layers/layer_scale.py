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
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------


class MultiplierType(Enum):
    """Enumeration for multiplier types."""

    GLOBAL = 0
    CHANNEL = 1

    @staticmethod
    def from_string(type_str: Union[str, "MultiplierType"]) -> "MultiplierType":
        """
        Convert string to MultiplierType enum.

        Args:
            type_str: String representation of multiplier type or MultiplierType instance.

        Returns:
            MultiplierType enum value.

        Raises:
            ValueError: If type_str is invalid.
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
        """
        Convert enum to string representation.

        Returns:
            String representation of the enum.
        """
        return self.name


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LearnableMultiplier(keras.layers.Layer):
    """
    Layer implementing learnable multipliers.

    The multipliers can be either global (single value) or per-channel.
    This layer multiplies the input tensor with learnable parameters that
    can be constrained and regularized.

    Args:
        multiplier_type: Type of multiplier ('GLOBAL' or 'CHANNEL').
            - GLOBAL: Single multiplier applied to entire tensor
            - CHANNEL: Separate multiplier for each channel
        initializer: Weight initializer for the multiplier parameters.
            Defaults to constant 1.0.
        regularizer: Optional regularizer for the multiplier weights.
        constraint: Optional constraint for the multiplier weights.
            Defaults to NonNeg constraint (values >= 0).
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape` (tuple of integers,
        does not include the batch axis) when using this layer as the first
        layer in a model.

    Output shape:
        Same shape as input.

    Example:
        >>> # Global multiplier
        >>> layer = LearnableMultiplier(multiplier_type='GLOBAL')
        >>> x = keras.random.normal((2, 10, 10, 3))
        >>> y = layer(x)
        >>> print(y.shape)
        (2, 10, 10, 3)

        >>> # Per-channel multiplier
        >>> layer = LearnableMultiplier(multiplier_type='CHANNEL')
        >>> x = keras.random.normal((2, 10, 10, 3))
        >>> y = layer(x)
        >>> print(y.shape)
        (2, 10, 10, 3)
    """

    def __init__(
        self,
        multiplier_type: Union[MultiplierType, str] = MultiplierType.CHANNEL,
        initializer: Union[str, keras.initializers.Initializer] = "ones",
        regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        constraint: Optional[Union[str, keras.constraints.Constraint]] = "non_neg",
        **kwargs: Any
    ) -> None:
        """Initialize the LearnableMultiplier layer."""
        super().__init__(**kwargs)

        # Store configuration parameters
        self.multiplier_type = MultiplierType.from_string(multiplier_type)
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)

        # Will be initialized in build()
        self.gamma = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer by creating the multiplier weights.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Determine weight shape based on multiplier type
        if self.multiplier_type == MultiplierType.GLOBAL:
            # Global multiplier: single scalar value broadcasted
            weight_shape = (1,)
        elif self.multiplier_type == MultiplierType.CHANNEL:
            # Per-channel multiplier: shape matches last dimension (channels)
            if len(input_shape) < 2:
                raise ValueError(
                    f"Input must have at least 2 dimensions for CHANNEL multiplier, "
                    f"got shape: {input_shape}"
                )
            weight_shape = (input_shape[-1],)
        else:
            raise ValueError(f"Invalid multiplier_type: {self.multiplier_type}")

        # Create the multiplier weight
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
        """
        Apply the learnable multipliers to inputs.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.
            **kwargs: Additional call arguments.

        Returns:
            Output tensor with multipliers applied element-wise.
        """
        # Use keras.ops for backend compatibility
        return ops.multiply(inputs, self.gamma)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape).

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "multiplier_type": self.multiplier_type.to_string(),
            "initializer": keras.initializers.serialize(self.initializer),
            "regularizer": keras.regularizers.serialize(self.regularizer),
            "constraint": keras.constraints.serialize(self.constraint),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for proper serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a build configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------