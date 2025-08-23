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
from typing import Dict, Any, Optional, Union, Tuple, Literal

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
    Layer implementing learnable element-wise multipliers for adaptive feature scaling.

    This layer introduces trainable scaling parameters that can be applied either
    globally across the entire input tensor or per-channel. It provides a simple
    but effective mechanism for learning feature importance and can act as a
    differentiable gating mechanism.

    The layer implements two operational modes:
    - **Global**: Single scalar multiplier applied uniformly across all features
    - **Channel**: Individual multipliers for each channel, enabling channel-wise
      feature re-weighting

    Key architectural benefits:
    - **Identity initialization**: Starts as identity transform for stable training
    - **Non-negative constraint**: Optional constraint to ensure positive scaling
    - **Minimal overhead**: Lightweight operation with negligible computational cost
    - **Flexible integration**: Can be inserted anywhere in the network architecture

    Mathematical formulation:
        - Global mode: ``output = gamma * input`` where gamma is scalar
        - Channel mode: ``output = gamma ⊙ input`` where ⊙ is element-wise product
          and gamma has shape ``(channels,)``

    Args:
        multiplier_type: Union[MultiplierType, str], type of multiplier operation.
            Either 'GLOBAL' for uniform scaling or 'CHANNEL' for per-channel scaling.
            Accepts MultiplierType enum or string. Defaults to 'CHANNEL'.
        initializer: Union[str, keras.initializers.Initializer], initializer for
            the multiplier weights. Should typically be 'ones' to start as identity.
            Defaults to 'ones'.
        regularizer: Optional[Union[str, keras.regularizers.Regularizer]], optional
            regularizer applied to the multiplier weights. Use for preventing
            overfitting or enforcing sparsity. Defaults to None.
        constraint: Optional[Union[str, keras.constraints.Constraint]], optional
            constraint applied to the multiplier weights. Common choices include
            'non_neg' to ensure positive values or 'unit_norm' for normalization.
            Defaults to 'non_neg'.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary tensor shape. For CHANNEL mode, input must have at least 2 dimensions
        with the last dimension representing channels.

    Output shape:
        Same shape as input tensor.

    Attributes:
        gamma: The trainable multiplier weights. Shape depends on multiplier_type:
            - Global: shape ``(1,)``
            - Channel: shape ``(input_channels,)``
        multiplier_type: The type of multiplier operation being performed.

    Example:
        ```python
        # Global multiplier for residual gating
        inputs = keras.Input(shape=(32, 32, 64))
        residual = keras.layers.Conv2D(64, 3, padding='same')(inputs)
        gated_residual = LearnableMultiplier(
            multiplier_type='GLOBAL',
            initializer='zeros'  # Start with no residual
        )(residual)
        outputs = keras.layers.Add()([inputs, gated_residual])

        # Per-channel feature re-weighting
        inputs = keras.Input(shape=(224, 224, 3))
        features = keras.layers.Conv2D(256, 3)(inputs)
        reweighted = LearnableMultiplier(
            multiplier_type='CHANNEL',
            regularizer=keras.regularizers.L1(1e-4)
        )(features)

        # Custom constraint for bounded scaling
        bounded_multiplier = LearnableMultiplier(
            multiplier_type='CHANNEL',
            constraint=keras.constraints.clip(0.1, 2.0)
        )
        ```

    Raises:
        ValueError: If multiplier_type is invalid or input dimensions are
            incompatible with CHANNEL mode.
        TypeError: If initializer, regularizer, or constraint types are invalid.

    Note:
        This layer is particularly useful in attention mechanisms, gating networks,
        and architectural search where adaptive feature scaling is beneficial.
        The non-negative constraint is recommended for interpretability but can
        be removed if negative scaling is desired.
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

        # Validate and store configuration parameters
        self.multiplier_type = MultiplierType.from_string(multiplier_type)
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)

        # Initialize weight attribute - created in build()
        self.gamma = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's trainable multiplier weights.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is incompatible with multiplier type.
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
        """
        Apply the learnable multipliers to inputs.

        Args:
            inputs: Input tensor to be scaled.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                included for API consistency.
            **kwargs: Additional call arguments.

        Returns:
            Output tensor with multipliers applied element-wise. Same shape as input.
        """
        # Element-wise multiplication using Keras ops for backend compatibility
        return ops.multiply(inputs, self.gamma)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (identical to input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters needed
            for reconstruction during model loading.
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