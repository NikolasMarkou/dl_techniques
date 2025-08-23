import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BasisFunction(keras.layers.Layer):
    """Basis function layer implementing b(x) = x / (1 + e^(-x)).

    This layer implements the basis function branch of PowerMLP, which enhances
    the expressiveness of neural networks by capturing complex non-linear
    relationships. The function is mathematically equivalent to the Swish
    activation function (x * sigmoid(x)) and provides smooth, differentiable
    transformations that can help with gradient flow.

    The basis function is defined as:
        b(x) = x / (1 + e^(-x)) = x * sigmoid(x)

    This activation function has several desirable properties:
    - Smooth and differentiable everywhere
    - Non-monotonic (unlike ReLU)
    - Bounded below by a linear function
    - Self-gated (the function gates itself)

    Args:
        **kwargs: Additional keyword arguments passed to the Layer parent class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape` (tuple of integers,
        does not include the batch axis) when using this layer as the first
        layer in a model.

    Output shape:
        Same shape as the input.

    Example:
        >>> # Create basis function layer
        >>> basis_layer = BasisFunction()
        >>>
        >>> # Apply to some test data
        >>> x = keras.utils.normalize(np.random.randn(32, 10))
        >>> output = basis_layer(x)
        >>> print(f"Input shape: {x.shape}, Output shape: {output.shape}")
        >>>
        >>> # Use in a model (PowerMLP architecture)
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, input_shape=(784,)),
        ...     BasisFunction(),
        ...     keras.layers.Dense(32),
        ...     BasisFunction(),
        ...     keras.layers.Dense(10, activation='softmax')
        ... ])

    References:
        - Ramachandran et al. "Searching for Activation Functions" (2017)
        - The basis function is equivalent to Swish: f(x) = x * sigmoid(x)
        - Used in PowerMLP architectures for enhanced expressiveness

    Note:
        - This function is smooth and differentiable everywhere
        - It's self-gated, meaning the function modulates its own output
        - Provides better gradient flow than ReLU-based activations
        - Computationally efficient due to the sigmoid's properties
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the BasisFunction activation layer.

        Args:
            **kwargs: Additional keyword arguments for the Layer parent class.
        """
        super().__init__(**kwargs)
        self._build_input_shape = None

        logger.info(f"Initialized BasisFunction layer")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and initialize internal state.

        This method is called automatically when the layer is first used.
        For activation layers, this typically just stores the input shape
        for serialization purposes.

        Args:
            input_shape: Shape tuple of the input tensor, including the batch
                dimension as None or an integer.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        logger.debug(f"Built BasisFunction layer with input shape: {input_shape}")
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the basis function activation.

        Applies the basis function: b(x) = x / (1 + e^(-x)) = x * sigmoid(x)

        Args:
            inputs: Input tensor of any shape.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                kept for API consistency.

        Returns:
            Output tensor with the same shape as inputs, after applying
            the basis function transformation.
        """
        # Compute b(x) = x / (1 + e^(-x))
        # This is mathematically equivalent to x * sigmoid(x)
        return inputs / (1.0 + ops.exp(-inputs))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        For activation layers, the output shape is identical to the input shape.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration from the parent class.
        """
        config = super().get_config()
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for serialization.

        This method returns the configuration needed to properly rebuild
        the layer after deserialization.

        Returns:
            Dictionary containing the build configuration, specifically
            the input shape used during the build process.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a build configuration.

        This method is used during model loading to properly rebuild
        the layer's internal state.

        Args:
            config: Dictionary containing the build configuration,
                as returned by get_build_config().
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including the layer name.
        """
        return f"BasisFunction(name='{self.name}')"

# ---------------------------------------------------------------------
