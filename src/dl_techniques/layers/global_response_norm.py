import keras
import tensorflow as tf
from typing import Any, Dict, Optional, Union, Tuple

# ---------------------------------------------------------------------

# Type aliases for improved readability
Shape = Tuple[Optional[int], ...]
DType = Union[str, tf.dtypes.DType]
Initializer = Union[str, keras.initializers.Initializer]

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class GlobalResponseNormalization(keras.layers.Layer):
    """
    Global Response Normalization (GRN) layer using standard Keras operations.

    This layer implements the GRN operation from the ConvNeXt V2 paper, which enhances
    inter-channel feature competition. It normalizes features across spatial dimensions
    and applies learnable scale and bias parameters.

    The operation flow is:
    1. Compute L2 norm across spatial dimensions for each channel
    2. Normalize by the mean of the L2 norm across channels
    3. Apply learnable scaling (gamma) and bias (beta)
    4. Add the result to the input (residual connection)

    Args:
        eps: Small constant for numerical stability (default: 1e-6)
        gamma_initializer: Initializer for gamma weights (default: 'ones')
        beta_initializer: Initializer for beta weights (default: 'zeros')
        dtype: Data type for layer computations
        name: Name for the layer

    References:
        - ConvNeXt V2 paper: https://arxiv.org/abs/2301.00808
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

        # Validate epsilon
        if eps <= 0:
            raise ValueError(f"epsilon must be positive, got {eps}")

        # Store configuration
        self.epsilon = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)

        # Initialize instance variables (will be set in build)
        self.gamma = None
        self.beta = None
        self._channels = None
        self._reshape_to_pixels = None
        self._reshape_to_spatial = None

    def build(self, input_shape: Shape) -> None:
        """
        Build the layer by creating weights and layers.

        Args:
            input_shape: Tuple of integers defining the input shape

        Raises:
            TypeError: If input_shape is not a tuple or list
            ValueError: If input shape is invalid
        """
        if self.built:
            return

        # Input validation
        if not isinstance(input_shape, (tuple, list)):
            raise TypeError(f"Expected tuple or list, got {type(input_shape)}")

        if len(input_shape) != 4:
            raise ValueError(f"Input shape must be 4D (batch, height, width, channels), "
                             f"got {len(input_shape)}D")

        # Get number of channels
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined")

        self._channels = channels

        # Create reshape layers (reused across calls for efficiency)
        self._reshape_to_pixels = keras.layers.Reshape((-1, channels))
        self._reshape_to_spatial = keras.layers.Reshape((1, 1, channels))

        # Create trainable parameters
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

        # Set input spec for automatic shape validation
        self.input_spec = keras.layers.InputSpec(
            ndim=4, axes={-1: channels}
        )

        self.built = True

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Apply global response normalization to the input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode (unused, included for API compatibility)

        Returns:
            Normalized tensor of the same shape as input
        """
        # Cast inputs to layer dtype for consistency
        inputs = keras.ops.cast(inputs, self.dtype)

        # Step 1: Reshape to (batch_size, pixels, channels) for efficient norm calculation
        reshaped = self._reshape_to_pixels(inputs)

        # Step 2: Compute L2 norm across spatial dimensions (axis=1)
        norm_squared = keras.ops.sum(keras.ops.square(reshaped), axis=1, keepdims=True)
        norm = keras.ops.sqrt(norm_squared + self.epsilon)  # Add epsilon for numerical stability

        # Step 3: Normalize by mean norm across channels
        mean_norm = keras.ops.mean(norm, axis=-1, keepdims=True)
        norm_channels = norm / (mean_norm + self.epsilon)  # Add epsilon for numerical stability

        # Step 4: Reshape norm back to (batch, 1, 1, channels) for broadcasting
        norm_spatial = self._reshape_to_spatial(norm_channels)

        # Step 5: Apply scale and bias with residual connection
        # Original formula: output = x + gamma * (x * normalized) + beta
        scaled = inputs * norm_spatial * self.gamma
        biased = scaled + self.beta
        output = inputs + biased

        return output

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        """Return output shape (same as input shape)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "epsilon": float(self.epsilon),
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

        # Deserialize initializers
        config["gamma_initializer"] = keras.initializers.deserialize(
            config.pop("gamma_initializer"))
        config["beta_initializer"] = keras.initializers.deserialize(
            config.pop("beta_initializer"))

        # Handle backward compatibility with old config key "eps"
        if "eps" in config:
            config["epsilon"] = config.pop("eps")

        return cls(**config)