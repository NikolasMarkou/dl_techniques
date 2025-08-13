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
from typing import Optional, Union, Any, Dict, Tuple

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

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            activation: Union[str, callable] = "gelu",
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

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Validate inputs
        self._validate_inputs()

        # These will be initialized in build()
        self.fc1 = None
        self.fc2 = None
        self.dropout = None
        self.activation_fn = None
        self._build_input_shape = None

        logger.info(f"Initialized MLPBlock with hidden_dim={hidden_dim}, "
                    f"output_dim={output_dim}, activation={activation}, "
                    f"dropout_rate={dropout_rate}")

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")

        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")

        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

    def _get_activation_fn(self, activation: Union[str, callable]) -> callable:
        """Get activation function from name or callable."""
        if callable(activation):
            return activation
        return keras.activations.get(activation)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the layer's sublayers.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Get activation function
        self.activation_fn = self._get_activation_fn(self.activation_name)

        # Create the first dense layer (expansion)
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_fc1" if self.name else None
        )

        # Create the second dense layer (projection)
        self.fc2 = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_fc2" if self.name else None
        )

        # Create dropout layer
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name=f"{self.name}_dropout" if self.name else None
            )
        else:
            self.dropout = None

        # Build sublayers with appropriate shapes
        self.fc1.build(input_shape)

        # Calculate intermediate shape after first dense layer
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        intermediate_shape = tuple(intermediate_shape)

        self.fc2.build(intermediate_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> Any:
        """
        Apply the MLP block to input tensors.

        Args:
            inputs: Input tensor of shape [..., input_dim].
            training: Boolean indicating whether the layer should behave in training mode.

        Returns:
            Output tensor of shape [..., output_dim].
        """
        x = inputs

        # First dense layer (expansion)
        x = self.fc1(x)

        # Activation function
        x = self.activation_fn(x)

        # Dropout after activation
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Second dense layer (projection)
        x = self.fc2(x)

        # Dropout after second dense layer
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Only the last dimension changes (to output_dim)
        output_shape_list = input_shape_list[:-1] + [self.output_dim]

        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
