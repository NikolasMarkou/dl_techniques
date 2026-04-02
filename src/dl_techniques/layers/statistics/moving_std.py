"""
This module provides a `MovingStd` layer that applies a 2D moving standard deviation
filter to input images using a sliding window approach.

The layer computes local standard deviation by applying the mathematical formula:
`sqrt(E[X^2] - (E[X])^2)` where E represents the expectation (mean) over a sliding
window. This is efficiently implemented using average pooling operations to compute
both E[X] and E[X^2].

This layer is particularly valuable for:
- **Texture analysis**: Capturing local texture patterns and roughness
- **Edge detection**: Highlighting regions with high local variability  
- **Feature extraction**: Providing variance-based features for classification
- **Noise characterization**: Analyzing spatial noise patterns
- **Medical imaging**: Detecting tissue boundaries and abnormalities

The implementation processes each channel independently, making it suitable for
both grayscale and multi-channel images while maintaining computational efficiency
through vectorized operations.
"""

import keras
from keras import ops
from typing import Tuple, Union, List, Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MovingStd(keras.layers.Layer):
    """
    Applies a 2D moving standard deviation filter to input images for texture analysis.

    Computes local standard deviation over sliding windows using the stable formula
    ``std = sqrt(E[X^2] - (E[X])^2 + epsilon)``. Average pooling operations
    efficiently compute the local expectations ``E[X]`` and ``E[X^2]``, and each
    channel is processed independently. The variance is clamped to be non-negative
    before taking the square root to handle floating-point precision errors.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────┐
        │  Input (batch, H, W, C)            │
        └──────────┬─────────┬───────────────┘
                   │         │
                   ▼         ▼
        ┌──────────────┐ ┌──────────────────┐
        │ AvgPool(X)   │ │ AvgPool(X^2)     │
        │ ─► E[X]      │ │ ─► E[X^2]        │
        └──────┬───────┘ └────────┬─────────┘
               │                  │
               ▼                  ▼
        ┌──────────────────────────────────┐
        │  Var = E[X^2] - (E[X])^2         │
        │  Var = max(0, Var)               │
        │  Std = sqrt(Var + epsilon)       │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output (batch, H', W', C)       │
        └──────────────────────────────────┘

    :param pool_size: Size of the 2D pooling window as ``(height, width)``.
        Defaults to ``(3, 3)``.
    :type pool_size: tuple[int, int]
    :param strides: Strides for the pooling operation. Defaults to ``(1, 1)``.
    :type strides: tuple[int, int] | list[int]
    :param padding: Padding mode, ``'valid'`` or ``'same'``. Defaults to ``'same'``.
    :type padding: str
    :param data_format: Data layout format. If ``None``, uses
        ``keras.config.image_data_format()``. Defaults to ``None``.
    :type data_format: str | None
    :param epsilon: Small value added to variance before square root.
        Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Initialize the MovingStd layer."""
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

        # Process and validate padding
        if not isinstance(padding, str):
            raise ValueError(f"padding must be a string, got {type(padding)}")
        self.padding = padding.lower()
        if self.padding not in {"valid", "same"}:
            raise ValueError(
                f"padding must be 'valid' or 'same', got '{padding}'"
            )

        # Process and validate data_format
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

        # CREATE the average pooling sub-layer in __init__ (modern Keras 3 pattern)
        self.pooler = keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dtype=self.compute_dtype,
            name='internal_pooler'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its internal average pooling component.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Input must be a 4D tensor, got shape {input_shape}"
            )

        # BUILD the internal pooling layer (critical for serialization)
        self.pooler.build(input_shape)

        logger.debug(
            f"MovingStd layer built with pool_size={self.pool_size}, "
            f"strides={self.strides}, padding={self.padding}, "
            f"data_format={self.data_format}"
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the moving standard deviation filter to the input tensor.

        :param inputs: Input tensor with shape determined by ``data_format``.
        :type inputs: keras.KerasTensor
        :param training: Boolean for training mode (unused, for API consistency).
        :type training: bool | None
        :return: Local standard deviation at each spatial location.
        :rtype: keras.KerasTensor
        """
        # Compute E[X] - local mean over the pooling window
        mean_x = self.pooler(inputs, training=training)

        # Compute E[X²] - local mean of squared values over the pooling window
        mean_x_sq = self.pooler(ops.square(inputs), training=training)

        # Calculate local variance: Var(X) = E[X²] - (E[X])²
        variance = mean_x_sq - ops.square(mean_x)

        # Ensure variance is non-negative for numerical stability
        # This handles potential floating-point precision issues
        variance = ops.maximum(variance, 0.0)

        # Compute local standard deviation: Std(X) = sqrt(Var(X) + epsilon)
        stddev = ops.sqrt(variance + self.epsilon)

        return stddev

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: Output shape tuple.
        :rtype: tuple[int | None, ...]
        """
        # Use the pooling layer's compute_output_shape method
        # Create a temporary pooling layer if not built yet
        if self.pooler is None or not hasattr(self.pooler, '_build_input_shape'):
            temp_pooler = keras.layers.AveragePooling2D(
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format
            )
            output_shape = temp_pooler.compute_output_shape(input_shape)
        else:
            output_shape = self.pooler.compute_output_shape(input_shape)

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
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

# ---------------------------------------------------------------------
