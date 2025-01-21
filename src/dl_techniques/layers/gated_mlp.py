"""
Gated MLP Layer Implementation
=============================

This module implements a Gated Multi-Layer Perceptron (MLP) layer for use in
neural networks. The layer combines gating mechanisms with 1x1 convolutions
to create a powerful feature transformation block.

Key Features:
------------
- Gating mechanism to control information flow
- 1x1 convolutions for feature transformation
- Configurable activation functions for attention and output
- Support for kernel regularization and initialization
- Bias-optional operation

Architecture:
------------
The layer consists of three parallel branches:
1. Gate branch: Controls information flow
2. Up branch: Transforms input features
3. Down branch: Final feature projection

The computation flow is:
input -> (gate_conv, up_conv) -> activation -> multiply -> down_conv -> activation -> output

Usage Examples:
-------------
Basic usage:
```python
layer = GatedMLP(
    filters=64,
    attention_activation='relu',
    output_activation='linear'
)
```

With L2 regularization:
```python
layer = GatedMLP(
    filters=64,
    kernel_regularizer=keras.regularizers.L2(l2=0.01),
    kernel_initializer='he_normal'
)
```

Notes:
-----
- The layer maintains the input spatial dimensions
- All convolutions use 1x1 kernels
- The number of filters in internal operations matches the input channels
- The output channel dimension is determined by the 'filters' parameter
"""

import copy
import keras
from keras import layers
import tensorflow as tf
from typing import Dict, Optional, Tuple, Union, Callable


@keras.utils.register_keras_serializable()
class GatedMLP(layers.Layer):
    """Gated Multi-Layer Perceptron implementation as a Keras layer.

    This layer implements a gated MLP with configurable activation functions
    using 1x1 convolutions for transformation.

    Args:
        filters: Number of output filters.
        kernel_initializer: Initializer for the kernel weights matrix.
            Defaults to "glorot_normal".
        kernel_regularizer: Optional regularizer function applied to kernel weights.
            Defaults to None.
        use_bias: Boolean, whether the layer uses a bias vector.
            Defaults to False.
        attention_activation: Activation function for the attention branch.
            Defaults to "relu".
        output_activation: Activation function for the output.
            Defaults to "linear".

    Example:
        ```python
        # Using L2 regularization
        regularizer = keras.regularizers.L2(l2=0.01)
        layer = GatedMLP(
            filters=64,
            kernel_regularizer=regularizer,
            attention_activation='relu'
        )
        ```
    """

    def __init__(
            self,
            filters: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_bias: bool = False,
            attention_activation: str = "relu",
            output_activation: str = "linear",
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Layer parameters
        self.filters = filters
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.attention_activation = attention_activation
        self.output_activation = output_activation

        # Layer components (initialized in build)
        self.conv_gate: Optional[layers.Conv2D] = None
        self.conv_up: Optional[layers.Conv2D] = None
        self.conv_down: Optional[layers.Conv2D] = None
        self.attention_activation_fn: Optional[Callable] = None
        self.output_activation_fn: Optional[Callable] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Builds the layer with given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        from .conv2d_builder import activation_wrapper

        # Base convolution parameters
        conv_params: Dict = {
            "filters": input_shape[-1],
            "kernel_size": (1, 1),
            "strides": (1, 1),
            "padding": "same",
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
        }

        # Initialize convolution layers
        self.conv_gate = layers.Conv2D(**conv_params)
        self.conv_up = layers.Conv2D(**conv_params)
        self.conv_down = layers.Conv2D(**conv_params)

        # Initialize activation functions
        self.attention_activation_fn = activation_wrapper(
            activation=self.attention_activation
        )
        self.output_activation_fn = activation_wrapper(
            activation=self.output_activation
        )

        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Output tensor after applying gated MLP operations.
        """
        # Gate branch
        x_gate = self.attention_activation_fn(
            self.conv_gate(inputs, training=training)
        )

        # Up branch
        x_up = self.attention_activation_fn(
            self.conv_up(inputs, training=training)
        )

        # Combine branches and apply final convolution
        x_combined = x_gate * x_up
        x_gated_mlp = self.conv_down(x_combined, training=training)

        return self.output_activation_fn(x_gated_mlp)

    def compute_output_shape(
            self,
            input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Computes output shape given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.

        Returns:
            Tuple of integers defining the output shape.
        """
        return copy.deepcopy(input_shape)

    def get_config(self) -> Dict:
        """Returns the config of the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "attention_activation": self.attention_activation,
            "output_activation": self.output_activation,
        })
        return config