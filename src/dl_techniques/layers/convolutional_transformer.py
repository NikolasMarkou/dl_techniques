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

import copy
import keras
import tensorflow as tf
from typing import Optional, Union, Callable, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.layer_scale import (
    LearnableMultiplier, MultiplierType)
from dl_techniques.regularizers.soft_orthogonal import (
    DEFAULT_SOFTORTHOGONAL_STDDEV,
    SoftOrthonormalConstraintRegularizer
)

from .conv2d_builder import activation_wrapper

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
            use_bias: bool = True,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            attention_dropout: float = 0.0,
            activation: Union[str, Callable] = "relu",
            regularization_rate: float = 1e-4,
            use_gamma: bool = False,
            use_soft_orthonormal_regularization: bool = False,
            seed: int = 1,
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
            activation (Union[str, Callable]): Activation function to use in MLP.
            regularization_rate (float): L2 regularization rate
            use_gamma: Whether to use the learnable scaling factor (default: True)
            use_soft_orthonormal_regularization (bool): if True use soft_orthonormal regularization
            **kwargs: Additional keyword arguments for the keras.layers.Layer.
        """
        super().__init__(**kwargs)

        # Store initialization parameters
        self.dim = dim
        self.seed = seed
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.activation_str = activation
        self.activation = activation_wrapper(activation)
        self.regularization_rate = regularization_rate
        self.use_soft_orthonormal_regularization = use_soft_orthonormal_regularization

        # Initialize layer attributes as None
        self.attention = None
        self.mlp = None
        self.norm1 = None
        self.norm2 = None
        # Validate dimensions
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        # Setup learnable multiplier
        self.gamma = None
        self.use_gamma = use_gamma

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer components when the input shape is known.

        Args:
            input_shape: Shape of the input tensor.
        """

        logger.info(f"self.use_bias: {self.use_bias}")
        logger.info(f"self.use_soft_orthonormal_regularization: {self.use_soft_orthonormal_regularization}")

        if self.use_soft_orthonormal_regularization:
            regularizer = SoftOrthonormalConstraintRegularizer()
        else:
            regularizer = keras.regularizers.l2(self.regularization_rate)

        # Layer Normalization
        self.norm1 = (
            keras.layers.LayerNormalization(
                epsilon=1e-6,
                scale=True,
                center=self.use_bias,
                name="norm1")
        )

        self.attention = (
            keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.dim // self.num_heads,
                dropout=self.attention_dropout,
                name="attention",
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.truncated_normal(
                    stddev=DEFAULT_SOFTORTHOGONAL_STDDEV, seed=self.seed),
                kernel_regularizer=keras.regularizers.l2(self.regularization_rate)
            )
        )

        # Layer Normalization
        self.norm2 = (
            keras.layers.LayerNormalization(
                epsilon=1e-6,
                scale=True,
                center=self.use_bias,
                name="norm2")
        )

        # Feed-Forward Network
        self.mlp = keras.Sequential([
            keras.layers.Conv2D(
                filters=int(self.dim * self.mlp_ratio),
                kernel_size=1,
                name="mlp_dense_1",
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.truncated_normal(
                    stddev=DEFAULT_SOFTORTHOGONAL_STDDEV, seed=self.seed),
                kernel_regularizer=copy.deepcopy(regularizer)
            ),
            keras.layers.Activation(self.activation, name="mlp_activation_2"),
            keras.layers.Dropout(rate=self.dropout_rate, name="mlp_dropout_1"),
            keras.layers.Conv2D(
                filters=self.dim,
                kernel_size=1,
                name="mlp_dense_2",
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.truncated_normal(
                    stddev=DEFAULT_SOFTORTHOGONAL_STDDEV, seed=self.seed),
                kernel_regularizer=copy.deepcopy(regularizer)
            )
        ], name="mlp")

        # Setup learnable scaling factor (gamma)
        if self.use_gamma:
            self.gamma = LearnableMultiplier(
                name="gamma",
                regularizer=keras.regularizers.L2(1e-2),
                multiplier_type="CHANNEL"
            )
        else:
            # If no gamma, use identity function
            self.gamma = lambda x, training=None: x

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        b, h, w, c = tf.unstack(tf.shape(inputs))

        # (1) Pre-norm
        x_norm = self.norm1(inputs, training=training)

        # (2) Flatten for attention
        x_norm = tf.reshape(x_norm, [b, h * w, c])  # shape => (batch, seq=H*W, channels)

        # (3) MultiHeadAttention (now correct shape)
        attention_output = self.attention(
            query=x_norm,
            value=x_norm,
            training=training,
            use_causal_mask=False,
            return_attention_scores=False,
        )

        # (4) Reshape back
        attention_output = tf.reshape(attention_output, [b, h, w, c])

        # --- Apply learnable scaling factor (gamma)
        attention_output = self.gamma(attention_output, training=training)

        x = inputs + attention_output  # residual

        # (5) Pre-norm for MLP
        x_norm2 = self.norm2(x, training=training)

        # (6) MLP is Conv2D-based, so that shape is correct
        mlp_output = self.mlp(x_norm2, training=training)

        # --- Apply learnable scaling factor (gamma)
        mlp_output = self.gamma(mlp_output, training=training)

        return x + mlp_output

    def get_config(self) -> dict:
        """
        Get the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary containing all initialization parameters.
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'use_bias': self.use_bias,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'use_gamma': self.use_gamma,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_str,
            'attention_dropout': self.attention_dropout,
            'use_soft_orthonormal_regularization': self.use_soft_orthonormal_regularization
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

