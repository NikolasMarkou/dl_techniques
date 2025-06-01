import keras
from keras import ops
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class LogitNorm(keras.layers.Layer):
    """
    LogitNorm layer for classification tasks.

    This layer implements logit normalization by applying L2 normalization with a learned temperature
    parameter. This helps stabilize training and can improve model calibration.

    Args:
        temperature: Float, temperature scaling parameter. Higher values produce more spread-out logits.
        axis: Integer, axis along which to perform normalization.
        epsilon: Float, small constant for numerical stability.

    References:
        - Paper: "Mitigating Neural Network Overconfidence with Logit Normalization"
    """

    def __init__(
            self,
            temperature: float = 0.04,  # Default from paper for CIFAR-10
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(temperature, epsilon)
        self.temperature = temperature
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, temperature: float, epsilon: float) -> None:
        """Validate initialization parameters."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
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
        # Compute L2 norm along specified axis
        norm = ops.sqrt(
            ops.maximum(
                ops.sum(ops.square(inputs), axis=self.axis, keepdims=True),
                self.epsilon
            )
        )

        # Normalize logits and scale by temperature
        return inputs / (norm * self.temperature)

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config