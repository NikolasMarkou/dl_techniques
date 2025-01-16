import tensorflow as tf
from keras.api.layers import Layer
from typing import Optional, Union, Tuple, Dict, Any


@tf.keras.utils.register_keras_serializable()
class CoupledLogitNorm(Layer):
    """
    Coupled LogitNorm layer for multi-label classification.

    This layer implements a modified version of LogitNorm that deliberately couples
    label predictions through normalization, creating a form of "confidence budget"
    across labels.

    Args:
        constant: Scaling factor for normalization. Higher values reduce coupling.
        coupling_strength: Additional factor to control coupling strength (1.0 = normal LogitNorm).
        axis: Axis along which to perform normalization.
        epsilon: Small constant for numerical stability.
    """

    def __init__(
            self,
            constant: float = 1.0,
            coupling_strength: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(constant, coupling_strength, epsilon)

        self.constant = constant
        self.coupling_strength = coupling_strength
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, constant: float, coupling_strength: float, epsilon: float) -> None:
        """Validate initialization parameters."""
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if coupling_strength < 0:
            raise ValueError(f"coupling_strength must be positive, got {coupling_strength}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply coupled logit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode (unused)

        Returns:
            Tuple of (normalized_logits, normalizing_factor)
        """
        # Cast inputs to float
        inputs = tf.cast(inputs, self.compute_dtype)

        # Compute L2 norm with coupling strength
        x_squared = tf.square(inputs)
        x_norm = tf.reduce_sum(x_squared, axis=self.axis, keepdims=True)
        x_norm = tf.pow(x_norm + self.epsilon, self.coupling_strength / 2.0)

        # Normalize logits
        normalized_logits = inputs / (x_norm * self.constant)

        return normalized_logits, x_norm

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "coupling_strength": self.coupling_strength,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


class CoupledMultiLabelHead(Layer):
    """
    Multi-label classification head with coupled logit normalization.

    This head applies coupled LogitNorm followed by sigmoid activation,
    creating interdependence between label predictions.
    """

    def __init__(
            self,
            constant: float = 1.0,
            coupling_strength: float = 1.0,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.logit_norm = CoupledLogitNorm(
            constant=constant,
            coupling_strength=coupling_strength
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply coupled normalization and sigmoid activation."""
        normalized_logits, _ = self.logit_norm(inputs)
        return tf.sigmoid(normalized_logits)
