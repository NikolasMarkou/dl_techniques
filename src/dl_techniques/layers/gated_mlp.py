"""
Implementation of the Gated MLP (gMLP) architecture from the paper:
"Pay Attention to MLPs" by Liu et al., 2021 (https://arxiv.org/abs/2105.08050)

This module implements a Gated Multi-Layer Perceptron (MLP) layer for use in
neural networks. The layer combines gating mechanisms with 1x1 convolutions
to create a powerful feature transformation block without self-attention.

Key Features:
------------
- Spatial Gating Unit (SGU) to enable cross-token interactions
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

As described in the paper, gMLP is a simple variant of MLPs with gating that can
perform comparably to Transformers in language and vision tasks, demonstrating that
self-attention is not always necessary for achieving strong performance.

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
    kernel_regularizer=keras.regularizers.L2(1e-4),
    kernel_initializer='he_normal'
)
```
"""

import copy
import keras
import tensorflow as tf
from keras import layers
from typing import Optional, Union, Literal, Dict, Any, Tuple, Callable


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class GatedMLP(layers.Layer):
    """
    A Gated MLP layer implementation using 1x1 convolutions.

    This layer implements a gated MLP architecture where the input is processed through
    three separate 1x1 convolution paths: gate, up, and down projections. The gating
    mechanism allows the network to selectively focus on relevant features.

    Args:
        filters (int): Number of filters for the output convolution.
        use_bias (bool): Whether to use bias in the convolution layers.
        kernel_initializer (Union[str, keras.initializers.Initializer]): Initializer for the kernel
            weights matrices. Defaults to "glorot_normal".
        kernel_regularizer (Union[str, keras.regularizers.Regularizer]): Regularizer function applied
            to the kernel weights matrices. Defaults to L2(1e-4).
        attention_activation (str): Activation function for the gate and up projections.
        output_activation (str): Activation function for the output.
        **kwargs: Additional arguments passed to the parent Layer class.

    Example:
        ```python
        # Create a GatedMLP layer with 64 filters
        gated_mlp = GatedMLP(
            filters=64,
            use_bias=True,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-5),
            attention_activation="gelu",
            output_activation="linear"
        )
        ```
    """

    def __init__(
            self,
            filters: int,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
            kernel_regularizer: Union[str, keras.regularizers.Regularizer] = keras.regularizers.L2(1e-4),
            attention_activation: Literal["relu", "gelu", "swish", "linear"] = "relu",
            output_activation: Literal["relu", "gelu", "swish", "linear"] = "linear",
            **kwargs: Any
    ) -> None:
        """Initialize the GatedMLP layer."""
        super().__init__(**kwargs)

        self.filters = filters
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.attention_activation = attention_activation
        self.output_activation = output_activation

        # Will be defined in build()
        self.conv_gate = None
        self.conv_up = None
        self.conv_down = None
        self.attention_activation_fn = None
        self.output_activation_fn = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the GatedMLP layer.

        Args:
            input_shape: The input shape as a TensorShape object.
        """
        from .conv2d_builder import activation_wrapper

        # Base parameters for conv layers
        conv_params: Dict[str, Any] = {
            "filters": self.filters,
            "kernel_size": (1, 1),
            "strides": (1, 1),
            "padding": "same",
            "activation": None,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
        }

        # Create the convolution layers
        self.conv_gate = layers.Conv2D(**conv_params)
        self.conv_up = layers.Conv2D(**conv_params)
        self.conv_down = layers.Conv2D(**conv_params)

        # Set up activation functions
        self.attention_activation_fn = activation_wrapper(activation=self.attention_activation)
        self.output_activation_fn = activation_wrapper(activation=self.output_activation)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass for the GatedMLP layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (Optional[bool]): Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            tf.Tensor: The output tensor after applying the Gated MLP operations.
        """
        # Gate branch
        x_gate = self.conv_gate(inputs, training=training)
        x_gate = self.attention_activation_fn(x_gate)

        # Up branch
        x_up = self.conv_up(inputs, training=training)
        x_up = self.attention_activation_fn(x_up)

        # Combine gate and up paths with element-wise multiplication
        x_combined = x_gate * x_up

        # Down path
        x_gated_mlp = self.conv_down(x_combined, training=training)

        # Apply output activation
        output = self.output_activation_fn(x_gated_mlp)

        return output

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Compute the output shape of the layer.

        Args:
            input_shape (tf.TensorShape): Shape of the input.

        Returns:
            tf.TensorShape: Output shape, which matches the input shape for this layer.
        """
        return tf.TensorShape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dict[str, Any]: Dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "attention_activation": self.attention_activation,
            "output_activation": self.output_activation,
        })
        return config

# ---------------------------------------------------------------------
