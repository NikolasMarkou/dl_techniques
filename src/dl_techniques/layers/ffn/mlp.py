"""
MLP Block Transformers
=================================

This module implements the standard MLP (Multi-Layer Perceptron) block
commonly used in transformer architectures.

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
    MLP block used in Transformers.

    This block implements the standard feed-forward network used in transformer
    architectures, consisting of two dense layers with an activation function
    and optional dropout in between.

    The layer follows the modern Keras 3 pattern: create all sub-layers in __init__
    and build them explicitly in build() for robust serialization.

    Args:
        hidden_dim: Integer, hidden dimension for the first dense layer (expansion).
            Must be positive.
        output_dim: Integer, output dimension for the second dense layer (projection).
            Must be positive.
        activation: Union[str, callable], activation function name or callable.
            Accepts string names ('gelu', 'relu', 'swish') or callable functions.
            Defaults to 'gelu'.
        dropout_rate: Float, dropout rate applied after both dense layers.
            Must be in range [0.0, 1.0). Defaults to 0.0.
        use_bias: Boolean, whether the dense layers use bias vectors.
            Defaults to True.
        kernel_initializer: Union[str, keras.initializers.Initializer], initializer
            for the dense layer kernels. Accepts string names ('glorot_uniform',
            'he_normal') or Initializer instances. Defaults to 'glorot_uniform'.
        bias_initializer: Union[str, keras.initializers.Initializer], initializer
            for the bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for the dense layer kernels.
            Can be string name ('l2') or Regularizer instance. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tensor with shape (..., input_dim) where input_dim is the size of
        the last dimension.

    Output shape:
        Tensor with shape (..., output_dim) where all dimensions except
        the last are preserved.

    Attributes:
        fc1: First dense layer (expansion) of shape (input_dim, hidden_dim).
        fc2: Second dense layer (projection) of shape (hidden_dim, output_dim).
        dropout: Dropout layer if dropout_rate > 0.0, None otherwise.
        activation_fn: Activation function used between the dense layers.

    Example:
        ```python
        # Basic usage
        mlp = MLPBlock(hidden_dim=2048, output_dim=512)

        # Advanced configuration
        mlp = MLPBlock(
            hidden_dim=3072,
            output_dim=768,
            activation='gelu',
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer layer
        inputs = keras.Input(shape=(128, 768))  # (seq_len, model_dim)
        x = MultiHeadAttention(...)(inputs)  # Attention block
        x = LayerNormalization()(x + inputs)  # Add & Norm
        ffn_output = MLPBlock(
            hidden_dim=3072,
            output_dim=768,
            dropout_rate=0.1
        )(x)
        outputs = LayerNormalization()(ffn_output + x)  # Add & Norm
        ```

    Raises:
        ValueError: If hidden_dim or output_dim is not positive.
        ValueError: If dropout_rate is not in range [0.0, 1.0).

    Note:
        This implementation follows the standard transformer MLP design where
        the hidden dimension is typically 4x the model dimension. The dropout
        is applied after both the activation and the final projection.
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
        **kwargs: Any
    ) -> None:
        """
        Initialize the MLP block.

        Args:
            hidden_dim: Hidden dimension for the first dense layer (expansion).
            output_dim: Output dimension for the second dense layer (projection).
            activation: Activation function name or callable.
            dropout_rate: Dropout rate applied after both dense layers.
            use_bias: Boolean, whether the dense layers use bias vectors.
            kernel_initializer: Initializer for the dense layer kernels.
            bias_initializer: Initializer for the bias vectors.
            kernel_regularizer: Optional regularizer for the dense layer kernels.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

        # Validate inputs immediately
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store ALL configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Get activation function once
        self.activation_fn = keras.activations.get(activation)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        # They will be unbuilt until build() is called
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="fc1"
        )

        self.fc2 = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="fc2"
        )

        # Create dropout layer if needed
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout = None

        logger.info(
            f"Initialized MLPBlock with hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, activation={activation}, "
            f"dropout_rate={dropout_rate}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        This method is called automatically when the layer first processes input.
        For robust serialization, we explicitly build each sub-layer.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build sub-layers in computational order for robust serialization
        self.fc1.build(input_shape)

        # Compute intermediate shape after first dense layer
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        intermediate_shape_tuple = tuple(intermediate_shape)

        # Dropout doesn't change shape
        if self.dropout is not None:
            self.dropout.build(intermediate_shape_tuple)

        # Build second dense layer
        self.fc2.build(intermediate_shape_tuple)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the MLP block to input tensors.

        Args:
            inputs: Input tensor of shape (..., input_dim).
            training: Boolean indicating whether the layer should behave
                in training mode. Affects dropout behavior.

        Returns:
            Output tensor of shape (..., output_dim).
        """
        # First dense layer (expansion)
        x = self.fc1(inputs)

        # Activation function
        x = self.activation_fn(x)

        # Dropout after activation if enabled
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Second dense layer (projection)
        x = self.fc2(x)

        # Final dropout if enabled
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. All dimensions preserved except last
            dimension changes to output_dim.
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)
        # Only the last dimension changes (to output_dim)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL parameters passed to __init__ for complete reconstruction.

        Returns:
            Dictionary containing the complete layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation_name,  # Store original name/callable
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------