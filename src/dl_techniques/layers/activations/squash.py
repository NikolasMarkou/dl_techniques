import keras
from keras import ops, backend
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SquashLayer(keras.layers.Layer):
    """Applies squashing non-linearity to vectors (capsules).

    The squashing function ensures that:
    1. Short vectors get shrunk to almost zero length
    2. Long vectors get shrunk to a length slightly below 1
    3. Vector orientation is preserved

    This is commonly used in Capsule Networks to ensure capsule outputs
    have meaningful magnitudes while preserving their directional information.
    The squashing function is defined as:

    .. math::
        \\text{squash}(\\mathbf{v}) = \\frac{||\\mathbf{v}||^2}{1 + ||\\mathbf{v}||^2} \\cdot \\frac{\\mathbf{v}}{||\\mathbf{v}||}

    Args:
        axis: Integer, axis along which to compute the norm. Defaults to -1.
        epsilon: Float, small constant for numerical stability. If None, uses keras.backend.epsilon().
        **kwargs: Additional keyword arguments to pass to the Layer base class.

    Input shape:
        Arbitrary tensor of rank >= 1.

    Output shape:
        Same as input shape.

    Example:
        >>> layer = SquashLayer()
        >>> x = keras.random.normal((32, 10, 16))
        >>> y = layer(x)
        >>> print(y.shape)
        (32, 10, 16)

        >>> # Custom axis for squashing
        >>> layer = SquashLayer(axis=1)
        >>> x = keras.random.normal((32, 10, 16))
        >>> y = layer(x)
        >>> print(y.shape)
        (32, 10, 16)
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon if epsilon is not None else backend.epsilon()

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.debug(f"Initialized SquashLayer with axis={axis}, epsilon={self.epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        logger.debug(f"Building SquashLayer with input_shape={input_shape}")

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> Any:
        """Apply squashing non-linearity.

        Args:
            inputs: Input tensor to be squashed.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Squashed vectors with norm between 0 and 1.
        """
        # Compute squared norm along specified axis
        squared_norm = ops.sum(
            ops.square(inputs),
            axis=self.axis,
            keepdims=True
        )

        # Safe norm computation to avoid division by zero
        safe_norm = ops.sqrt(squared_norm + self.epsilon)

        # Compute scale factor: ||v||^2 / (1 + ||v||^2)
        scale = squared_norm / (1.0 + squared_norm)

        # Compute unit vector
        unit_vector = inputs / safe_norm

        # Apply squashing: scale * unit_vector
        return scale * unit_vector

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

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

# ---------------------------------------------------------------------
