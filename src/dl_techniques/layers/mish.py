import keras
from keras import ops
from typing import Any, Dict, Optional, Union
import numpy as np

# ---------------------------------------------------------------------

def mish(inputs):
    # Calculate softplus: log(1 + exp(x))
    softplus = ops.softplus(inputs)
    # Calculate tanh of softplus
    tanh_softplus = ops.tanh(softplus)
    # Return x * tanh(softplus(x))
    return inputs * tanh_softplus

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Mish(keras.layers.Layer):
    """Mish activation function.

    Implementation of the Mish activation function from the paper
    "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    (https://arxiv.org/abs/1908.08681).

    The function is computed as: f(x) = x * tanh(softplus(x))
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Mish activation layer.

        Args:
            **kwargs: Additional layer keyword arguments.
        """
        super().__init__(**kwargs)
        # This layer has no weights or state to initialize

    def build(self, input_shape: tuple) -> None:
        """Build the layer (no-op for this activation layer).

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples,
                indicating the input shape of the layer.
        """
        # No weights to build for this activation layer
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass computation.

        Args:
            inputs: Input tensor.
            training: Whether in training mode. Not used in this layer but
                included for API consistency.

        Returns:
            A tensor with the same shape as input after applying the
            Mish activation function.
        """
        return mish(inputs)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Same shape as input.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        return super().get_config()

# ---------------------------------------------------------------------


def saturated_mish(inputs, alpha:float = 3.0, beta:float = 0.5, mish_at_alpha:float = 1.0):
    tmp_mish = mish(inputs)

    # Create a smooth sigmoid-based blending factor
    # Note: Using ops.sigmoid for backend compatibility
    blend_factor = ops.sigmoid((inputs - alpha) / beta)

    # Combine both regions with smooth blending
    # For x <= alpha: mostly standard Mish
    # For x > alpha: gradually approach mish_at_alpha
    return tmp_mish * (1.0 - blend_factor) + mish_at_alpha * blend_factor

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SaturatedMish(keras.layers.Layer):
    """SaturatedMish activation function with continuous transition at alpha.

    The function behaves as follows:
    - For x <= alpha: f(x) = x * tanh(softplus(x)) (standard Mish)
    - For x > alpha: f(x) smoothly blends between the Mish value at alpha and
      a slightly higher asymptotic value, creating a continuous transition

    Args:
        alpha: The saturation threshold. Defaults to 3.0.
        beta: Controls the steepness of the transition. A smaller beta
              makes the transition sharper, while a larger beta makes it
              smoother. Defaults to 0.5.
        **kwargs: Additional layer keyword arguments.
    """

    def __init__(
            self,
            alpha: float = 3.0,
            beta: float = 0.5,
            **kwargs: Any
    ) -> None:
        """Initialize the SaturatedMish activation layer.

        Args:
            alpha: The saturation threshold. Must be greater than 0.
            beta: Controls the steepness of the transition. Must be greater than 0.
            **kwargs: Additional layer keyword arguments.

        Raises:
            ValueError: If alpha or beta is not greater than 0.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if alpha <= 0.0:
            raise ValueError("alpha must be greater than 0.")
        if beta <= 0.0:
            raise ValueError("beta must be greater than 0.")

        # Store parameters
        self.alpha = alpha
        self.beta = beta

        # Pre-compute mish value at alpha for efficiency
        self._build_input_shape = None

    def build(self, input_shape: tuple) -> None:
        """Build the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples,
                indicating the input shape of the layer.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Pre-compute mish value at alpha
        # Note: Using numpy for initialization since we don't need gradients here
        alpha_tensor = np.float32(self.alpha)
        softplus_alpha = np.log(1.0 + np.exp(alpha_tensor))
        tanh_softplus_alpha = np.tanh(softplus_alpha)
        self.mish_at_alpha = alpha_tensor * tanh_softplus_alpha

        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Forward pass computation.

        Args:
            inputs: Input tensor.

        Returns:
            A tensor with the same shape as input after applying the
            SaturatedMish activation function.
        """
        return saturated_mish(inputs, alpha=self.alpha, beta=self.beta, mish_at_alpha=self.mish_at_alpha)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Same shape as input.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'alpha': float(self.alpha),
            'beta': float(self.beta)
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])