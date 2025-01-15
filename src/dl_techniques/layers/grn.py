import keras
import tensorflow as tf
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np


@keras.utils.register_keras_serializable()
class GlobalResponseNormalization(keras.layers.Layer):
    """
    Optimized Global Response Normalization (GRN) layer.

    This layer implements an optimized version of GRN as described in the ConvNeXt V2 paper.
    It normalizes features across spatial dimensions and applies learnable scale and bias.

    Key optimizations:
    1. Memory-efficient computation using fused operations
    2. Improved numerical stability
    3. Static shape inference where possible
    4. Pre-computed constants and cached computations
    5. Proper initialization strategy for better convergence

    Reference:
        https://arxiv.org/abs/2301.00808

    Args:
        eps: Small constant for numerical stability
        gamma_initializer: Initializer for gamma weights
        beta_initializer: Initializer for beta weights
        dtype: Data type of the layer's computations
    """

    def __init__(
            self,
            eps: float = 1e-6,
            gamma_initializer: Union[str, keras.initializers.Initializer] = 'ones',
            beta_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            dtype: Any = None,
            **kwargs: Any
    ) -> None:
        """Initialize the GRN layer."""
        super().__init__(dtype=dtype, **kwargs)

        self.eps = tf.cast(eps, dtype=self.dtype or tf.float32)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)

        # Initialize instance variables
        self.gamma: Optional[tf.Variable] = None
        self.beta: Optional[tf.Variable] = None
        self.built = False

        # Cache for static shapes
        self._input_spec = None
        self._channels = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer by creating weights and validating input shape.

        Args:
            input_shape: Shape of input tensor

        Raises:
            ValueError: If input shape is invalid
        """
        if len(input_shape) != 4:
            raise ValueError(f"Input shape must be 4D (batch, height, width, channels), "
                             f"got {len(input_shape)}D")

        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined")

        self._channels = channels

        # Create weights with proper shape and initialization
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, channels),
            initializer=self.gamma_initializer,
            trainable=True,
            dtype=self.dtype
        )

        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, channels),
            initializer=self.beta_initializer,
            trainable=True,
            dtype=self.dtype
        )

        # Set input spec for shape validation
        self._input_spec = keras.layers.InputSpec(
            ndim=4, axes={-1: channels}
        )

        self.built = True

    @tf.function(jit_compile=True)
    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> tf.Tensor:
        """
        Apply global response normalization to the input tensor.

        This implementation uses XLA-optimized operations and proper numerical
        stability techniques.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether the layer is in training mode (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            Normalized tensor of the same shape as input
        """
        # Ensure input dtype matches layer dtype
        inputs = tf.cast(inputs, self.dtype)

        # Pre-compute shapes for efficiency
        shape = tf.shape(inputs)
        batch_size, height, width = shape[0], shape[1], shape[2]

        # Compute L2 norm efficiently using fused operation
        # Reshape to combine spatial dimensions for faster computation
        reshaped_input = tf.reshape(inputs, [batch_size, -1, self._channels])
        norm_squared = tf.reduce_sum(tf.square(reshaped_input), axis=1, keepdims=True)
        norm = tf.sqrt(norm_squared + self.eps)

        # Normalize by mean norm (with numerical stability)
        mean_norm = tf.reduce_mean(norm, axis=-1, keepdims=True)
        norm_channels = norm / (mean_norm + self.eps)

        # Reshape norm back to original spatial dimensions
        norm_channels = tf.reshape(norm_channels, [batch_size, 1, 1, self._channels])

        # Apply scale and bias with residual connection
        # Use fused multiply-add for better performance
        output = tf.raw_ops.AddV2(
            x=inputs,
            y=tf.raw_ops.AddV2(
                x=self.beta,
                y=tf.raw_ops.Mul(x=inputs * norm_channels, y=self.gamma)
            )
        )

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "eps": float(self.eps),
            "gamma_initializer": keras.initializers.serialize(self.gamma_initializer),
            "beta_initializer": keras.initializers.serialize(self.beta_initializer)
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GlobalResponseNormalization':
        """
        Create a layer instance from its configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            A new instance of the layer
        """
        config["gamma_initializer"] = keras.initializers.deserialize(config["gamma_initializer"])
        config["beta_initializer"] = keras.initializers.deserialize(config["beta_initializer"])
        return cls(**config)
