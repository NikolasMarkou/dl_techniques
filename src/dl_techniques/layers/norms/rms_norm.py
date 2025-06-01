import keras
from keras import ops
from typing import Any, Dict, Optional

@keras.utils.register_keras_serializable()
class RMSNorm(keras.layers.Layer):
    """
    Root Mean Square Normalization layer for classification tasks.

    This layer implements root mean square normalization by normalizing inputs by their
    RMS value. Unlike LogitNorm which uses L2 normalization, RMSNorm uses root mean
    square for normalization, which can help stabilize training and improve model
    robustness.

    The normalization is computed as:
        output = input / sqrt(mean(input^2) + epsilon) * constant

    This implementation differs from LogitNorm in that it:
    - Uses mean of squared values rather than sum
    - Applies a constant scaling factor instead of temperature
    - Does not specifically target logit calibration

    Args:
        constant: float, default=1.0
            Scaling factor applied after normalization. Higher values produce
            outputs with larger magnitudes.
        axis: int, default=-1
            Axis along which to compute RMS statistics. The default (-1)
            computes RMS over the last dimension.
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
            constant: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(constant, epsilon)
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, constant: float, epsilon: float) -> None:
        """Validate initialization parameters."""
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply logit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode (unused)

        Returns:
            Normalized logits tensor
        """
        # Compute L2 norm
        x_squared = ops.square(inputs)
        x_norm = ops.sqrt(
            ops.sum(x_squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Normalize logits
        return inputs / (x_norm * self.constant)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config