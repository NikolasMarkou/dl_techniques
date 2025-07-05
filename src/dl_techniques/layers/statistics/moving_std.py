import keras
import numpy as np
from keras import ops
from typing import Tuple, Union, List, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MovingStd(keras.layers.Layer):
    """Applies a 2D moving standard deviation filter to input images.

    This layer computes the standard deviation over a sliding window using the
    formula: `sqrt(E[X^2] - (E[X])^2)`. It achieves this by using average
    pooling for the expectation `E`. Each channel is processed independently.

    Args:
        pool_size: Tuple of two integers specifying the height and width of the
            2D pooling window.
        strides: Tuple/list of 2 integers specifying the strides of the pooling
            operation along height and width.
        padding: String, either "valid" or "same" (case-insensitive).
        data_format: String, either "channels_last" or "channels_first".
        epsilon: A small float added to the variance to avoid taking the square
            root of a negative number due to floating-point inaccuracies.
        trainable: Boolean, if True allow the weights to change (not applicable
            for this layer, but kept for consistency).
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

    Example:
        >>> x = np.random.rand(4, 32, 32, 3).astype("float32") # Input images
        >>> layer = MovingStd(pool_size=(5, 5))
        >>> y = layer(x)
        >>> print(y.shape)
        (4, 32, 32, 3)
    """

    def __init__(
            self,
            pool_size: Tuple[int, int] = (3, 3),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            padding: str = "same",
            data_format: Optional[str] = None,
            epsilon: float = 1e-7,
            trainable: bool = False,
            **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        # Validate and store pool size
        if len(pool_size) != 2:
            raise ValueError("pool_size must be length 2")
        self.pool_size = pool_size

        # Validate and store strides
        if not isinstance(strides, (tuple, list)) or len(strides) != 2:
            raise ValueError("strides must be a tuple/list of length 2")
        self.strides = strides

        # Process padding
        self.padding = padding.lower()
        if self.padding not in {"valid", "same"}:
            raise ValueError(f"padding must be 'valid' or 'same', got {padding}")

        # Process data_format
        self.data_format = keras.backend.image_data_format() if data_format is None else data_format
        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"data_format must be 'channels_first' or 'channels_last', got {data_format}")

        self.epsilon = epsilon
        # Internal pooling layer will be created in build()
        self.pooler = None

    def build(self, input_shape):
        """Build the internal pooling layer.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # We only need one AveragePooling2D layer, which we can apply twice.
        self.pooler = keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dtype=self.compute_dtype
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Apply the moving standard deviation filter to the input tensor.

        Args:
            inputs: Input tensor of shape:
                - If data_format="channels_last": (batch_size, height, width, channels)
                - If data_format="channels_first": (batch_size, channels, height, width)
            training: Boolean indicating whether in training mode (unused).

        Returns:
            Filtered tensor representing the local standard deviation.
        """
        # E[X]
        mean_x = self.pooler(inputs)

        # E[X^2]
        mean_x_sq = self.pooler(ops.square(inputs))

        # Var(X) = E[X^2] - (E[X])^2
        variance = mean_x_sq - ops.square(mean_x)

        # Ensure variance is non-negative for numerical stability
        variance = ops.maximum(variance, 0.0)

        # Std(X) = sqrt(Var(X))
        stddev = ops.sqrt(variance + self.epsilon)

        return stddev

    def get_config(self):
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
            "epsilon": self.epsilon
        })
        return config

# ---------------------------------------------------------------------