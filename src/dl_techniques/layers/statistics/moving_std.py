import keras
from keras import ops
from typing import Tuple, Union, List, Optional, Any, Dict

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class MovingStd(keras.layers.Layer):
    """Applies a 2D moving standard deviation filter to input images.

    This layer computes the standard deviation over a sliding window using the
    formula: `sqrt(E[X^2] - (E[X])^2)`. It achieves this by using average
    pooling for the expectation `E`. Each channel is processed independently.

    The layer is useful for texture analysis, edge detection, and as a feature
    extractor that captures local variability in images. It can be particularly
    effective in computer vision tasks where local variance information is
    important.

    Args:
        pool_size: Tuple of two integers specifying the height and width of the
            2D pooling window. Default is (3, 3).
        strides: Tuple or list of 2 integers specifying the strides of the pooling
            operation along height and width. Default is (1, 1).
        padding: String, either "valid" or "same" (case-insensitive). "valid"
            means no padding, "same" results in padding to preserve input size.
            Default is "same".
        data_format: String, either "channels_last" or "channels_first". The
            ordering of the dimensions in the inputs. "channels_last" corresponds
            to inputs with shape (batch_size, height, width, channels) while
            "channels_first" corresponds to inputs with shape
            (batch_size, channels, height, width). If None, defaults to the
            image_data_format value found in your Keras config file.
        epsilon: A small float added to the variance to avoid taking the square
            root of a negative number due to floating-point inaccuracies.
            Default is 1e-7.
        **kwargs: Additional keyword arguments passed to the Layer base class.

    Input shape:
        4D tensor with shape:
        - If data_format="channels_last": (batch_size, height, width, channels)
        - If data_format="channels_first": (batch_size, channels, height, width)

    Output shape:
        4D tensor with shape:
        - If data_format="channels_last":
            (batch_size, new_height, new_width, channels)
        - If data_format="channels_first":
            (batch_size, channels, new_height, new_width)

        Where `new_height` and `new_width` depend on the pool_size, strides,
        and padding parameters.

    Returns:
        A 4D tensor representing the local standard deviation at each spatial
        location.

    Raises:
        ValueError: If pool_size is not a tuple/list of length 2.
        ValueError: If strides is not a tuple/list of length 2.
        ValueError: If padding is not "valid" or "same".
        ValueError: If data_format is not "channels_first" or "channels_last".

    Example:
        >>> import numpy as np
        >>> x = np.random.rand(4, 32, 32, 3).astype("float32")
        >>> layer = MovingStd(pool_size=(5, 5), padding="same")
        >>> y = layer(x)
        >>> print(y.shape)
        (4, 32, 32, 3)

        >>> # With different parameters
        >>> layer2 = MovingStd(pool_size=(3, 3), strides=(2, 2), padding="valid")
        >>> y2 = layer2(x)
        >>> print(y2.shape)
        (4, 15, 15, 3)
    """

    def __init__(
            self,
            pool_size: Tuple[int, int] = (3, 3),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            padding: str = "same",
            data_format: Optional[str] = None,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate and store pool size
        if not isinstance(pool_size, (tuple, list)) or len(pool_size) != 2:
            raise ValueError(
                f"pool_size must be a tuple or list of length 2, got {pool_size}"
            )
        if not all(isinstance(x, int) and x > 0 for x in pool_size):
            raise ValueError(
                f"pool_size values must be positive integers, got {pool_size}"
            )
        self.pool_size = tuple(pool_size)

        # Validate and store strides
        if not isinstance(strides, (tuple, list)) or len(strides) != 2:
            raise ValueError(
                f"strides must be a tuple or list of length 2, got {strides}"
            )
        if not all(isinstance(x, int) and x > 0 for x in strides):
            raise ValueError(
                f"strides values must be positive integers, got {strides}"
            )
        self.strides = tuple(strides)

        # Process padding
        if not isinstance(padding, str):
            raise ValueError(f"padding must be a string, got {type(padding)}")
        self.padding = padding.lower()
        if self.padding not in {"valid", "same"}:
            raise ValueError(
                f"padding must be 'valid' or 'same', got '{padding}'"
            )

        # Process data_format
        if data_format is None:
            self.data_format = keras.config.image_data_format()
        else:
            self.data_format = data_format.lower()

        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                f"data_format must be 'channels_first' or 'channels_last', "
                f"got '{data_format}'"
            )

        # Validate epsilon
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError(f"epsilon must be a non-negative number, got {epsilon}")
        self.epsilon = float(epsilon)

        # Initialize sublayer to None - will be created in build()
        self.pooler = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the internal pooling layer.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Input must be a 4D tensor, got shape {input_shape}"
            )

        # Create the average pooling layer
        self.pooler = keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dtype=self.compute_dtype
        )

        # Build the pooling layer
        self.pooler.build(input_shape)

        logger.debug(
            f"MovingStd layer built with pool_size={self.pool_size}, "
            f"strides={self.strides}, padding={self.padding}, "
            f"data_format={self.data_format}"
        )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply the moving standard deviation filter to the input tensor.

        Args:
            inputs: Input tensor of shape:
                - If data_format="channels_last": (batch_size, height, width, channels)
                - If data_format="channels_first": (batch_size, channels, height, width)
            training: Boolean indicating whether in training mode (unused for this layer).

        Returns:
            Filtered tensor representing the local standard deviation.
        """
        # E[X] - mean of inputs over the pooling window
        mean_x = self.pooler(inputs)

        # E[X^2] - mean of squared inputs over the pooling window
        mean_x_sq = self.pooler(ops.square(inputs))

        # Var(X) = E[X^2] - (E[X])^2
        variance = mean_x_sq - ops.square(mean_x)

        # Ensure variance is non-negative for numerical stability
        # This handles potential floating-point precision issues
        variance = ops.maximum(variance, 0.0)

        # Std(X) = sqrt(Var(X) + epsilon)
        stddev = ops.sqrt(variance + self.epsilon)

        return stddev

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Use the pooling layer's compute_output_shape method
        # Create a temporary pooling layer if not built yet
        if self.pooler is None:
            temp_pooler = keras.layers.AveragePooling2D(
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format
            )
            output_shape = temp_pooler.compute_output_shape(input_shape)
        else:
            output_shape = self.pooler.compute_output_shape(input_shape)

        # Return as tuple for consistency
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "epsilon": self.epsilon,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the configuration needed to build the layer.

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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MovingStd":
        """Create a layer from its configuration.

        Args:
            config: Dictionary containing the layer configuration.

        Returns:
            MovingStd layer instance.
        """
        return cls(**config)