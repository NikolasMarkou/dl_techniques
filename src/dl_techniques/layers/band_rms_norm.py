import keras
import tensorflow as tf
from keras.api.layers import Layer
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class BandRMSNorm(Layer):
    """Root Mean Square Normalization layer for classification tasks.

    This layer implements root mean square normalization by normalizing inputs
    by their RMS value within a learnable band [1-α, 1]. The normalization is
    computed as:
        scale_factor = (1 - α) + α * sigmoid(learnable_param)
        output = (input / sqrt(mean(input^2) + epsilon)) * scale_factor

    Args:
        max_band_width: float, default=0.2
            Maximum allowed deviation from unit normalization
        constant: float, default=1.0
            Scaling factor applied after normalization.
        axis: int, default=-1
            Axis along which to compute RMS statistics.
        epsilon: float, default=1e-7
            Small constant added to denominator for numerical stability.

    Inputs:
        A tensor of any rank

    Outputs:
        A tensor of the same shape as the input, normalized by RMS values

    References:
        "Root Mean Square Layer Normalization", 2019
        https://arxiv.org/abs/1910.07467
    """

    def __init__(
            self,
            max_band_width: float = 0.2,
            constant: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(max_band_width, constant, epsilon)
        self.max_band_width = max_band_width
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon
        self.band_param = None

    def _validate_inputs(
            self,
            max_band_width: float,
            constant: float,
            epsilon: float
    ) -> None:
        """Validate initialization parameters."""
        if max_band_width <= 0 or max_band_width >= 1:
            raise ValueError(f"max_band_width must be between 0 and 1, got {max_band_width}")
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create the layer's state."""
        ndims = len(input_shape)
        axis = self.axis if self.axis >= 0 else ndims + self.axis

        param_shape = [1] * ndims
        param_shape[axis] = input_shape[axis]

        self.band_param = self.add_weight(
            name="band_param",
            shape=param_shape,
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
            regularizer=keras.regularizers.l2(0.00001)
        )

        self.built = True

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Apply logit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode (unused)

        Returns:
            Normalized logits tensor
        """
        # Cast inputs to float
        inputs = tf.cast(inputs, self.compute_dtype)

        # Compute RMS values
        x_squared = tf.square(inputs)
        rms = tf.sqrt(
            tf.reduce_mean(x_squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute learnable band scale factor
        band_scale = ((1.0 - self.max_band_width) +
                      (self.max_band_width *
                       tf.sigmoid(tf.cast(self.band_param, inputs.dtype))))

        # Apply normalization with learnable band
        return ((inputs / rms) *
                (band_scale / self.constant))

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config

# ---------------------------------------------------------------------
