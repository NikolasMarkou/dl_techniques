import keras
from typing import Optional, Dict, Any, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    """Implements Stochastic Depth for deep networks.

    This layer implements batch-wise dropping of residual paths as described in the
    Stochastic Depth paper. Unlike sample-wise dropping methods (like DropPath in timm),
    this implementation drops the same paths for all samples in a batch.

    Args:
        drop_path_rate: Float between 0 and 1, probability of dropping the residual path.
            Default is 0.5.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Raises:
        ValueError: If drop_path_rate is not in the interval [0, 1).

    References:
        - Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
        - timm library implementation (https://github.com/rwightman/pytorch-image-models)

    Example:
        >>> import keras
        >>> import numpy as np
        >>> layer = StochasticDepth(drop_path_rate=0.2)
        >>> x = np.random.rand(4, 32, 32, 64)
        >>> output = layer(x, training=True)
        >>> print(output.shape)  # (4, 32, 32, 64)
    """

    def __init__(
            self,
            drop_path_rate: float = 0.5,
            **kwargs: Any
    ) -> None:
        """Initialize the StochasticDepth layer.

        Args:
            drop_path_rate: Probability of dropping the residual path.
                Must be in the range [0, 1).
            **kwargs: Additional keyword arguments passed to the parent Layer class.

        Raises:
            TypeError: If drop_path_rate is not a number.
            ValueError: If drop_path_rate is not in [0, 1).
        """
        super().__init__(**kwargs)

        # Validate drop_path_rate
        if not isinstance(drop_path_rate, (int, float)):
            raise TypeError("drop_path_rate must be a number")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )

        self.drop_path_rate = float(drop_path_rate)

        # Initialize sublayer placeholder - will be created in build()
        self.dropout: Optional[keras.layers.Dropout] = None

        # Store build input shape for serialization
        self._build_input_shape: Optional[Tuple[Optional[int], ...]] = None

    def build(self, input_shape: Union[Tuple[Optional[int], ...], keras.KerasTensor]) -> None:
        """Build the layer's internal state.

        Creates a Dropout layer with appropriate noise shape based on input dimensions.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            TypeError: If input_shape is not a valid shape.
            ValueError: If input has insufficient dimensions.
        """
        # Store input shape for serialization
        if hasattr(input_shape, 'shape'):
            # Handle KerasTensor
            self._build_input_shape = tuple(input_shape.shape)
        else:
            # Handle tuple/list
            self._build_input_shape = tuple(input_shape)

        # Validate input shape
        if not isinstance(input_shape, (tuple, list)) and not hasattr(input_shape, 'shape'):
            raise TypeError(f"Expected tuple, list, or KerasTensor, got {type(input_shape)}")

        shape_to_use = self._build_input_shape
        dims = len(shape_to_use)

        if dims < 2:
            raise ValueError(
                f"Input must have at least 2 dimensions, got {dims}"
            )

        # Create noise shape: (batch_size, 1, 1, ..., 1)
        # This ensures the same dropout mask is applied to all spatial locations
        # but can differ across batch samples
        noise_shape = (shape_to_use[0],) + (1,) * (dims - 1)

        # Initialize dropout layer in build() as per best practices
        self.dropout = keras.layers.Dropout(
            rate=self.drop_path_rate,
            noise_shape=noise_shape,
            name=f"{self.name}_dropout"
        )

        # Build the dropout layer
        self.dropout.build(input_shape)

        logger.info(
            f"Built StochasticDepth layer '{self.name}' with "
            f"drop_path_rate={self.drop_path_rate}, "
            f"input_shape={shape_to_use}, "
            f"noise_shape={noise_shape}"
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode. If None,
                the layer's default behavior is used.

        Returns:
            Output tensor with same shape as input. During training, the tensor
            may be zeroed out with probability drop_path_rate. During inference,
            the tensor is returned unchanged.

        Raises:
            RuntimeError: If the layer has not been built.
        """
        if self.dropout is None:
            raise RuntimeError(
                f"Layer '{self.name}' has not been built. "
                "Call build() or pass data through the layer first."
            )

        # Apply stochastic depth only during training
        if training is False:
            return inputs

        return self.dropout(inputs, training=training)

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], keras.KerasTensor]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Tuple representing output shape (same as input shape).
        """
        if hasattr(input_shape, 'shape'):
            return tuple(input_shape.shape)
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for layer serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        base_config = super().get_config()
        config = {
            "drop_path_rate": self.drop_path_rate,
        }
        return {**base_config, **config}

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for proper serialization.

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

