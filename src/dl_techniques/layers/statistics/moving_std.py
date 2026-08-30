"""
A 2D moving standard deviation filter for images.

`MovingStd` slides a window over the input and reports the standard deviation of
the pixels inside that window. Channels are processed independently, so the layer
works on grayscale and multi-channel images alike. All the work is done by pooling
ops, so it stays vectorized.

The variance behind that standard deviation is not computed the naive way. See the
`MovingStd` class docstring for the shifted two-pass form and why it is used.

This layer is particularly valuable for:
- **Texture analysis**: local texture patterns and roughness
- **Edge detection**: regions with high local variability
- **Feature extraction**: variance-based features for classification
- **Noise characterization**: spatial noise patterns
- **Medical imaging**: tissue boundaries and abnormalities
"""

import keras
from keras import ops
from typing import Tuple, Union, List, Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.statistics.moving_std")
class MovingStd(keras.layers.Layer):
    """
    Applies a 2D moving standard deviation filter to images for texture analysis.

    Every window position gets the standard deviation of the pixels inside it.
    Each channel is handled on its own.

    The variance is computed over a shifted copy of the input, not over the input
    itself. Before pooling, the layer subtracts ``c``, the per-channel spatial mean
    of the batch, wrapped in ``stop_gradient`` so it acts as a constant. Both pooled
    terms are then taken over ``Y = X - c`` and combined as ``E[Y^2] - (E[Y])^2``.
    ``Var(X) == Var(X - c)``, so this is still the exact windowed variance.

    The shift is what keeps the result accurate. It holds both pooled terms near the
    size of the variance instead of the size of ``mean^2``, so the subtraction no
    longer cancels two nearly equal large numbers. The naive ``E[X^2] - (E[X])^2``
    form loses most of its precision on large-mean inputs such as pixel values
    around 200.

    The shift has to be a per-channel constant rather than a per-window mean.
    ``AveragePooling2D`` returns means on the pooled grid, not on the input grid, so
    there is no per-window mean available to subtract from ``inputs`` before pooling.

    Variance is clamped at zero from below before the square root. That absorbs any
    leftover floating-point error.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────┐
        │ inputs  (batch, H, W, C)                        │
        └───────────┬─────────────────────────┬───────────┘
                    │                         ▼
                    │             ┌───────────────────────┐
                    │             │ mean over spatial_axes│
                    │             │ shift:  (batch,1,1,C) │
                    │             └───────────┬───────────┘
                    │                         │  stop_gradient
                    ▼                         ▼
        ┌─────────────────────────────────────────────────┐
        │ centered = inputs - shift                       │
        └───────────┬─────────────────────────┬───────────┘
                    │                         ▼
                    │             ┌───────────────────────┐
                    │             │ square(centered)      │
                    │             └───────────┬───────────┘
                    ▼                         ▼
        ┌───────────────────────┐ ┌───────────────────────┐
        │ pooler (AvgPool2D)    │ │ pooler (AvgPool2D)    │
        │ -> mean_x             │ │ -> mean_x_sq          │
        └───────────┬───────────┘ └───────────┬───────────┘
                    │  (batch, H', W', C)     │
                    └────────────┬────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │ variance = mean_x_sq - square(mean_x)           │
        │ variance = maximum(variance, 0.0)               │
        │ stddev   = sqrt(variance + epsilon)             │
        └────────────────────────┬────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │ output  (batch, H', W', C)                      │
        └─────────────────────────────────────────────────┘

    The two ``pooler`` boxes are the same ``AveragePooling2D`` instance, called
    twice. It holds no weights.

    :param pool_size: Size of the 2D pooling window as (height, width).
        Defaults to (3, 3).
    :type pool_size: tuple[int, int]
    :param strides: Strides for the pooling operation. Defaults to (1, 1).
    :type strides: tuple[int, int] | list[int]
    :param padding: Padding mode, "valid" or "same". Defaults to "same".
    :type padding: str
    :param data_format: Data layout format. If None, uses
        keras.config.image_data_format(). Defaults to None.
    :type data_format: str | None
    :param epsilon: Small value added to variance before the square root.
        Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar pooler: The shared average pooling sub-layer used for both pooled terms.
    :vartype pooler: keras.layers.AveragePooling2D

    :raises ValueError: If pool_size or strides is not a length-2 tuple or list of
        positive integers, if padding is not a string or not "valid"/"same", if
        data_format is not "channels_first"/"channels_last", or if epsilon is
        negative.

    Input shape:
        4D tensor. (batch, H, W, C) for "channels_last",
        (batch, C, H, W) for "channels_first".

    Output shape:
        4D tensor with the spatial dims set by the pooling window, strides and
        padding. With the defaults (3, 3), (1, 1) and "same", the shape is
        unchanged.

    Example:
        >>> layer = MovingStd(pool_size=(5, 5), padding="valid")
        >>> texture = layer(images)

    Note:
        ``training`` is accepted for API consistency. The layer behaves the same
        in both modes.
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

        # Create the pooling sub-layer here, in __init__, per the Keras 3 pattern.
        # One instance serves both pooled terms.
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
        :raises ValueError: If ``input_shape`` is not 4D.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Input must be a 4D tensor, got shape {input_shape}"
            )

        # Build the pooling sub-layer explicitly. Serialization depends on it.
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
        # Subtract a per-channel constant before pooling, then use the two-pass
        # form. The class docstring explains why the shift is needed.
        spatial_axes = (
            (2, 3) if self.data_format == "channels_first" else (1, 2)
        )
        shift = ops.stop_gradient(
            ops.mean(inputs, axis=spatial_axes, keepdims=True)
        )
        centered = inputs - shift

        # E[Y]: local mean of the shifted values over the pooling window
        mean_x = self.pooler(centered, training=training)

        # E[Y^2]: local mean of the squared shifted values over the same window
        mean_x_sq = self.pooler(ops.square(centered), training=training)

        # Local variance: Var(X) = Var(Y) = E[Y^2] - (E[Y])^2
        variance = mean_x_sq - ops.square(mean_x)

        # Clamp away any small negative value left by floating-point error
        variance = ops.maximum(variance, 0.0)

        # Local standard deviation: Std(X) = sqrt(Var(X) + epsilon)
        stddev = ops.sqrt(variance + self.epsilon)

        return stddev

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        The pooling sub-layer owns the shape arithmetic, so ask it. If it has not
        been built yet, a throwaway pooler with the same settings answers instead.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: Output shape tuple.
        :rtype: tuple[int | None, ...]
        """
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
