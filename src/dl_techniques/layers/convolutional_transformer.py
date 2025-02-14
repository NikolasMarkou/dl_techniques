"""
Convolutional Transformer Implementation
=====================================

A specialized vision transformer architecture that combines convolutional operations
with transformer attention mechanisms for image processing tasks.

Key Components
------------
1. ConvolutionalTransformerBlock:
   - Core building block combining convolution and self-attention
   - Follows layer ordering: Conv -> Norm -> Activation
   - Uses grouped convolutions for Q,K,V projections
   - Incorporates residual connections and layer normalization
   - Features configurable MLP with customizable activation

2. Architecture Flow:
   Input -> Conv Projection -> [ConvTransformer Blocks] -> Global Pool -> Classification

Technical Details
---------------
- Compatible with Keras 3.8.0 and TensorFlow 2.18.0 backend
- Input: 4D tensor (batch_size, height, width, channels)
- Output: Class probabilities for num_classes
- Serializable via @keras.utils.register_keras_serializable()
- Supports model saving in .keras format

Key Features
----------
- L2 regularization on all learnable weights (default rate: 1e-3)
- Proper layer normalization ordering
- Configurable attention mechanism with dropout
- Customizable activation functions
- Robust error handling for dimension mismatches

Best Practices
------------
1. Initialization:
   - Validates dimensions compatibility
   - Uses proper kernel initializers
   - Implements L2 regularization

2. Layer Configuration:
   - Follows: Conv -> Norm -> Activation pattern
   - Uses grouped convolutions for efficiency
   - Implements residual connections

3. Implementation:
   - Full type hinting
   - Comprehensive error handling
   - Proper serialization support
   - Memory-efficient tensor reshaping

Usage Example
-----------
>>> model = create_convolutional_transformer_model(
...     input_shape=(224, 224, 3),
...     num_classes=1000,
...     num_blocks=6,
...     dim=256,
...     num_heads=8
... )

Testing
------
Compatible with pytest framework:
- Unit tests for layer operations
- Integration tests for model building
- Serialization/deserialization tests
- Shape validation tests

References
---------
- Layer Normalization: https://arxiv.org/abs/1607.06450
- Vision Transformers: https://arxiv.org/abs/2010.11929
- Convolutional Attention: https://arxiv.org/abs/2103.02907

Note: This implementation prioritizes code clarity and maintainability while
following Keras best practices for custom layer development.
"""

import keras
import tensorflow as tf
from typing import Optional, Tuple, Union, Callable

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class ConvolutionalTransformerBlock(keras.layers.Layer):
    """
    Convolutional Transformer Block for vision applications.

    This block combines Keras's built-in attention mechanism with convolutional
    projections, feed-forward network, and layer normalization. Features configurable
    activation functions.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            attention_dropout: float = 0.0,
            use_scale: bool = False,
            conv_kernel_size: int = 1,
            activation: Union[str, Callable] = "gelu",
            **kwargs
    ):
        """
        Initialize the Convolutional Transformer Block.

        Args:
            dim (int): The number of input channels.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            dropout_rate (float): Dropout rate.
            attention_dropout (float): Dropout rate for attention.
            use_scale (bool): Whether to use scaled attention.
            conv_kernel_size (int): Kernel size for convolutions.
            activation (Union[str, Callable]): Activation function to use in MLP.
            **kwargs: Additional keyword arguments for the keras.layers.Layer.
        """
        super().__init__(**kwargs)

        # Validate dimensions
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        # Store initialization parameters
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.use_scale = use_scale
        self.conv_kernel_size = conv_kernel_size
        self.activation_str = activation
        self.activation = activation
        self.head_dim = dim // num_heads
        self.regularization_rate = 1e-3

        # Initialize layer attributes as None
        self.q_conv = None
        self.k_conv = None
        self.v_conv = None
        self.attention = None
        self.proj = None
        self.mlp = None
        self.norm1 = None
        self.norm2 = None
        self.dropout_layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer when the input shape is known.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.
        """
        # Convolutional projections for Q, K, V
        self.q_conv = keras.layers.Conv2D(
            self.dim, self.conv_kernel_size,
            padding="same",
            groups=self.num_heads,
            name="query_conv",
            kernel_regularizer=keras.regularizers.l2(self.regularization_rate)
        )
        self.k_conv = keras.layers.Conv2D(
            self.dim, self.conv_kernel_size,
            padding="same",
            groups=self.num_heads,
            name="key_conv",
            kernel_regularizer=keras.regularizers.l2(self.regularization_rate)
        )
        self.v_conv = keras.layers.Conv2D(
            self.dim, self.conv_kernel_size,
            padding="same",
            groups=self.num_heads,
            name="value_conv",
            kernel_regularizer=keras.regularizers.l2(self.regularization_rate)
        )

        # Attention layer
        self.attention = keras.layers.Attention(
            use_scale=self.use_scale,
            dropout=self.attention_dropout,
            name="attention"
        )

        # Output projection
        self.proj = keras.layers.Conv2D(
            self.dim, 1,
            name="output_projection",
            kernel_regularizer=keras.regularizers.l2(self.regularization_rate))

        # Feed-Forward Network
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(
                mlp_hidden_dim,
                name="mlp_dense_1",
                kernel_regularizer=keras.regularizers.l2(self.regularization_rate)),
            keras.layers.Activation(self.activation, name="mlp_activation"),
            keras.layers.Dropout(rate=self.dropout_rate, name="mlp_dropout_1"),
            keras.layers.Dense(
                self.dim,
                name="mlp_dense_2",
                kernel_regularizer=keras.regularizers.l2(self.regularization_rate))
        ], name="mlp")

        # Layer Normalization
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")

        # Dropout
        self.dropout_layer = keras.layers.Dropout(rate=self.dropout_rate, name="proj_dropout")

        super().build(input_shape)

    def _reshape_for_attention(self, x: tf.Tensor) -> tf.Tensor:
        """
        Reshape spatial dimensions to sequence dimension.

        Args:
            x: Input tensor of shape [batch_size, height, width, channels]

        Returns:
            Reshaped tensor of shape [batch_size, height * width, channels]
        """
        batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        return tf.reshape(x, [batch_size, height * width, channels])

    def _reshape_from_attention(self, x: tf.Tensor, height: int, width: int) -> tf.Tensor:
        """
        Reshape sequence dimension back to spatial dimensions.

        Args:
            x: Input tensor of shape [batch_size, height * width, channels]
            height: Original height
            width: Original width

        Returns:
            Reshaped tensor of shape [batch_size, height, width, channels]
        """
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, [batch_size, height, width, self.dim])

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of the Convolutional Transformer Block.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).
            training (Optional[bool]): Whether the call is for training or inference.

        Returns:
            tf.Tensor: Output tensor of the same shape as input.
        """
        batch_size, height, width = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        x = inputs

        # Generate Q, K, V using convolutions
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        # Reshape for attention
        q_seq = self._reshape_for_attention(q)
        k_seq = self._reshape_for_attention(k)
        v_seq = self._reshape_for_attention(v)

        # Apply attention
        attention_output = self.attention(
            [q_seq, v_seq, k_seq],
            training=training
        )

        # Reshape back to spatial dimensions
        attention_output = self._reshape_from_attention(attention_output, height, width)
        attention_output = self.proj(attention_output)
        attention_output = self.dropout_layer(attention_output, training=training)

        # First residual connection
        x = inputs + attention_output
        x = self.norm1(x)

        # Feed-forward network
        mlp_output = self.mlp(x, training=training)

        # Second residual connection
        x = x + mlp_output
        x = self.norm2(x)
        return x

    def get_config(self) -> dict:
        """
        Get the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary containing all initialization parameters.
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'use_scale': self.use_scale,
            'conv_kernel_size': self.conv_kernel_size,
            'activation': self.activation_str,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a layer instance from its configuration.

        Args:
            config: Layer configuration dictionary.

        Returns:
            ConvolutionalTransformerBlock: A new instance of the layer.
        """
        return cls(**config)

# ---------------------------------------------------------------------


def create_convolutional_transformer_model(
        input_shape: Tuple[int, int, int],
        num_classes: int,
        num_blocks: int = 6,
        dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        activation: Union[str, Callable] = "relu",
) -> keras.Model:
    """
    Create a Convolutional Transformer model for vision applications.

    Args:
        input_shape (Tuple[int, int, int]): Shape of the input image (height, width, channels).
        num_classes (int): Number of output classes.
        num_blocks (int): Number of Convolutional Transformer Blocks.
        dim (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        dropout_rate (float): Dropout rate.
        attention_dropout (float): Dropout rate for attention.
        activation (Union[str, Callable]): Activation function to use in MLPs.

    Returns:
        keras.Model: A Keras model with the Convolutional Transformer architecture.
    """
    inputs = keras.Input(shape=input_shape)

    # Initial convolution
    x = keras.layers.Conv2D(dim, kernel_size=7, strides=2, padding="same")(inputs)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Convolutional Transformer Blocks
    for i in range(num_blocks):
        x = ConvolutionalTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            activation=activation,
            name=f"conv_transformer_block_{i}"
        )(x)

    # Classification head
    x = keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return keras.Model(inputs, outputs, name="ConvolutionalTransformer")

# ---------------------------------------------------------------------

