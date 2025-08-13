"""
MLP Block for Transformers - Modern Keras 3 Implementation
==========================================================

This module implements the standard MLP (Multi-Layer Perceptron) block
commonly used in transformer architectures, following modern Keras 3 best practices.

The MLP block consists of:
1. First dense layer (expansion)
2. Activation function (typically GELU)
3. Dropout (optional)
4. Second dense layer (projection)
5. Dropout (optional)

This follows the standard transformer architecture where the MLP block
is used after the multi-head attention mechanism within each transformer layer.
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MLPBlock(keras.layers.Layer):
    """
    MLP block used in Transformers with modern Keras 3 implementation.

    This block implements the standard feed-forward network used in transformer
    architectures, consisting of two dense layers with an activation function
    and optional dropout in between.

    The implementation follows the modern Keras 3 pattern where all sub-layers
    are created in __init__ and Keras handles the build lifecycle automatically.
    This ensures proper serialization and eliminates common build errors.

    Mathematical formulation:
        output = dropout(dense2(dropout(activation(dense1(input)))))

    Where:
    - dense1: Linear expansion to hidden_dim
    - activation: Non-linear activation function
    - dense2: Linear projection to output_dim
    - dropout: Optional regularization (applied after activation and final output)

    Args:
        hidden_dim: Integer, hidden dimension for the first dense layer (expansion).
            Must be positive. This is typically larger than the input dimension
            to provide expressiveness (common ratios are 2x to 8x input dimension).
        output_dim: Integer, output dimension for the second dense layer (projection).
            Must be positive. This is typically the same as the input dimension
            in transformer architectures to maintain residual connections.
        activation: String or callable, activation function applied after the first
            dense layer. Can be string name ('gelu', 'relu', 'swish') or callable.
            Defaults to 'gelu' which is common in modern transformers.
        dropout_rate: Float, dropout rate applied after both the activation and
            the final output. Must be between 0.0 and 1.0. Set to 0.0 to disable
            dropout. Defaults to 0.0.
        use_bias: Boolean, whether the dense layers use bias vectors.
            Defaults to True.
        kernel_initializer: String or keras.initializers.Initializer, initializer
            for the dense layer kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: String or keras.initializers.Initializer, initializer
            for the bias vectors. Only used if use_bias=True. Defaults to 'zeros'.
        kernel_regularizer: Optional keras.regularizers.Regularizer, regularizer
            applied to the dense layer kernel weights. Defaults to None.
        bias_regularizer: Optional keras.regularizers.Regularizer, regularizer
            applied to the bias vectors. Only used if use_bias=True. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        Most common case is 3D input: `(batch_size, sequence_length, input_dim)`.

    Output shape:
        N-D tensor with shape: `(..., output_dim)`.
        For 3D input: `(batch_size, sequence_length, output_dim)`.

    Example:
        ```python
        # Basic usage - typical transformer MLP
        mlp = MLPBlock(hidden_dim=2048, output_dim=512)

        # Custom configuration
        mlp = MLPBlock(
            hidden_dim=1024,
            output_dim=256,
            activation='swish',
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer layer
        inputs = keras.Input(shape=(128, 512))  # (seq_len, embed_dim)
        # ... attention layer ...
        mlp_output = mlp(attention_output)

        # In a complete model
        model_input = keras.Input(shape=(128, 512))
        x = mlp(model_input)
        model = keras.Model(model_input, x)
        ```

    Note:
        This implementation creates all sub-layers in __init__ following the
        modern Keras 3 pattern. This ensures proper serialization and avoids
        build-related errors that were common with older patterns.

    Raises:
        ValueError: If hidden_dim or output_dim is not positive.
        ValueError: If dropout_rate is not between 0.0 and 1.0.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

        # Store ALL configuration arguments as instance attributes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ - this is the modern Keras 3 pattern
        # First dense layer (expansion)
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="expansion_dense"
        )

        # Activation layer
        self.activation_layer = keras.layers.Activation(
            activation=self.activation,
            name="activation"
        )

        # Dropout layer (created even if rate=0.0 for consistency)
        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )

        # Second dense layer (projection)
        self.fc2 = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="projection_dense"
        )

        logger.info(f"Initialized MLPBlock with hidden_dim={hidden_dim}, "
                   f"output_dim={output_dim}, activation={activation}, "
                   f"dropout_rate={dropout_rate}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the MLP block to input tensors.

        Args:
            inputs: Input tensor with shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. This affects dropout behavior.

        Returns:
            Output tensor with shape (..., output_dim).
        """
        # First dense layer (expansion)
        x = self.fc1(inputs, training=training)

        # Activation function
        x = self.activation_layer(x, training=training)

        # Dropout after activation (only applied during training when dropout_rate > 0)
        x = self.dropout_layer(x, training=training)

        # Second dense layer (projection)
        x = self.fc2(x, training=training)

        # Dropout after final layer (only applied during training when dropout_rate > 0)
        x = self.dropout_layer(x, training=training)

        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. Same as input except last dimension becomes output_dim.
        """
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,  # Store original activation (string or callable)
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    # NO get_build_config or build_from_config methods needed!
    # The modern Keras 3 pattern handles building automatically.

# ---------------------------------------------------------------------
