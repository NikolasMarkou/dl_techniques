import keras
import tensorflow as tf
from typing import Any, Dict, Optional, Union, Tuple


# Type aliases for improved readability
Shape = Tuple[Optional[int], ...]
DType = Union[str, tf.dtypes.DType]
Initializer = Union[str, keras.initializers.Initializer]


@keras.utils.register_keras_serializable()
class GlobalResponseNormalization(keras.layers.Layer):
    """
    Optimized Global Response Normalization (GRN) layer.

    This layer implements an optimized version of GRN with improved numerical stability,
    memory efficiency, and XLA optimization. It normalizes features across spatial
    dimensions and applies learnable scale and bias.

    Attributes:
        eps: Small constant for numerical stability
        gamma: Learnable scale parameter
        beta: Learnable bias parameter
        _channels: Number of input channels (set during build)
        _eps_tensor: Cached epsilon tensor for improved performance
        _input_spec: Input shape specification for validation

    Args:
        eps: Float value for numerical stability (default: 1e-6)
        gamma_initializer: Initializer for gamma weights (default: 'ones')
        beta_initializer: Initializer for beta weights (default: 'zeros')
        dtype: Data type for layer computations
        name: Optional name for the layer
    """

    def __init__(
            self,
            eps: float = 1e-6,
            gamma_initializer: Initializer = 'ones',
            beta_initializer: Initializer = 'zeros',
            dtype: Optional[DType] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the GRN layer with specified parameters."""
        super().__init__(dtype=dtype, name=name, **kwargs)

        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)

        # Initialize instance variables
        self.gamma: Optional[tf.Variable] = None
        self.beta: Optional[tf.Variable] = None
        self._channels: Optional[int] = None
        self._eps_tensor: Optional[tf.Tensor] = None
        self._input_spec: Optional[keras.layers.InputSpec] = None

    def build(self, input_shape: Shape) -> None:
        """
        Build the layer by creating weights and validating input shape.

        Args:
            input_shape: Tuple of integers defining the input shape

        Raises:
            ValueError: If input shape is invalid or channel dimension is undefined
            TypeError: If input_shape is not a tuple or list
        """
        if self.built:
            return

        if not isinstance(input_shape, (tuple, list)):
            raise TypeError(f"Expected tuple or list, got {type(input_shape)}")

        if len(input_shape) != 4:
            raise ValueError(f"Input shape must be 4D (batch, height, width, channels), "
                           f"got {len(input_shape)}D")

        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined")

        self._channels = channels
        self._eps_tensor = tf.constant(self.eps, dtype=self.dtype)

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

    @tf.function
    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> tf.Tensor:
        """
        Apply global response normalization to the input tensor.

        This implementation uses XLA-optimized operations and proper numerical
        stability techniques. It includes memory optimizations and improved error checking.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether the layer is in training mode (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            tf.Tensor: Normalized tensor of the same shape as input

        Raises:
            ValueError: If layer has not been built
            tf.errors.InvalidArgumentError: If input rank is not 4
        """
        if not self.built:
            raise ValueError("Layer has not been built. Call 'build()' first.")

        tf.debugging.assert_rank(inputs, 4, "Input tensor must be 4-dimensional")

        # Ensure input dtype matches layer dtype and has correct shape
        inputs = tf.cast(inputs, self.dtype)
        tf.debugging.assert_shapes([
            (inputs, ('N', 'H', 'W', self._channels))
        ])

        # Pre-compute shapes for efficiency
        shape = tf.shape(inputs)
        batch_size, height, width = shape[0], shape[1], shape[2]

        with tf.control_dependencies([inputs]):
            # Compute L2 norm efficiently using fused operation
            # Reshape to combine spatial dimensions for faster computation
            reshaped_input = tf.reshape(inputs, [batch_size, -1, self._channels])
            norm_squared = tf.reduce_sum(tf.square(reshaped_input), axis=1, keepdims=True)
            norm = tf.sqrt(norm_squared + self._eps_tensor)

            # Normalize by mean norm (with numerical stability)
            mean_norm = tf.reduce_mean(norm, axis=-1, keepdims=True)
            norm_channels = norm / (mean_norm + self._eps_tensor)

            # Reshape norm back to original spatial dimensions
            norm_channels = tf.reshape(norm_channels, [batch_size, 1, 1, self._channels])

            # Apply scale and bias with residual connection using fused operations
            scaled = tf.raw_ops.Mul(x=inputs * norm_channels, y=self.gamma)
            biased = tf.raw_ops.AddV2(x=scaled, y=self.beta)
            output = tf.raw_ops.AddV2(x=inputs, y=biased)

            return output

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Tuple of integers or None defining the input shape

        Returns:
            Tuple defining the output shape
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
        config = config.copy()  # Prevent modifications to original config
        config["gamma_initializer"] = keras.initializers.deserialize(
            config["gamma_initializer"])
        config["beta_initializer"] = keras.initializers.deserialize(
            config["beta_initializer"])
        return cls(**config)