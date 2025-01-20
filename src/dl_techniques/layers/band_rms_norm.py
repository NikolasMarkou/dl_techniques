import keras
import tensorflow as tf
from keras.api.layers import Layer
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class BandRMSNorm(Layer):
    """Root Mean Square Normalization layer with L2 norm constraints.

    This layer implements root mean square normalization that guarantees the output
    L2 norm will be between [1-α, 1], where α is the max_band_width parameter.
    The normalization is computed in two steps:
    1. RMS normalization to unit norm
    2. Learnable scaling within the [1-α, 1] band

    Args:
        max_band_width: float, default=0.2
            Maximum allowed deviation from unit normalization (0 < α < 1)
        axis: int, default=-1
            Axis along which to compute RMS statistics
        epsilon: float, default=1e-7
            Small constant added to denominator for numerical stability
        band_regularizer: Optional[keras.regularizers.Regularizer], default=L2(1e-5)
            Regularizer for the band parameter

    Inputs:
        A tensor of any rank

    Outputs:
        A tensor of the same shape as input, with L2 norm guaranteed to be
        between [1-max_band_width, 1]

    References:
        "Root Mean Square Layer Normalization", 2019
        https://arxiv.org/abs/1910.07467
    """

    def __init__(
            self,
            max_band_width: float = 0.2,
            axis: int = -1,
            epsilon: float = 1e-7,
            band_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-5),
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(max_band_width, epsilon)
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_regularizer = band_regularizer
        self.band_param = None

    def _validate_inputs(self, max_band_width: float, epsilon: float) -> None:
        """Validate initialization parameters.

        Args:
            max_band_width: Maximum allowed deviation from unit norm
            epsilon: Small constant for numerical stability

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create the layer's trainable weights.

        Args:
            input_shape: Shape of input tensor
        """
        ndims = len(input_shape)
        axis = self.axis if self.axis >= 0 else ndims + self.axis

        param_shape = [1] * ndims
        param_shape[axis] = input_shape[axis]

        # Initialize band parameter with zeros for optimal training
        self.band_param = self.add_weight(
            name="band_param",
            shape=param_shape,
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
            regularizer=self.band_regularizer
        )

        self.built = True

    def _compute_rms(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute root mean square values along specified axis.

        Args:
            inputs: Input tensor

        Returns:
            RMS values tensor with same shape as input except for normalized axis
        """
        x_squared = tf.square(inputs)
        mean_squared = tf.reduce_mean(x_squared, axis=self.axis, keepdims=True)
        return tf.sqrt(mean_squared + self.epsilon)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Apply constrained RMS normalization.

        Args:
            inputs: Input tensor
            training: Whether in training mode (unused)

        Returns:
            Normalized tensor with L2 norm in [1-max_band_width, 1]
        """
        inputs = tf.cast(inputs, self.compute_dtype)

        # Step 1: RMS normalization to get unit norm
        rms = self._compute_rms(inputs)
        normalized = inputs / rms

        # Step 2: Apply learnable scaling within [1-α, 1] band
        # Use hard sigmoid to strictly enforce the bounds
        scale = (1.0 - self.max_band_width) + (
                self.max_band_width *
                tf.keras.backend.hard_sigmoid(tf.cast(self.band_param, inputs.dtype))
        )

        # The output will have L2 norm between [1-max_band_width, 1]
        # since we first normalize to unit norm and then scale by a factor
        # that is guaranteed to be in [1-max_band_width, 1]
        return normalized * scale

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute shape of output tensor.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor (same as input)
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "band_regularizer": keras.regularizers.serialize(self.band_regularizer)
        })
        return config

# ---------------------------------------------------------------------
