"""
This module provides a `MovingStd` layer that applies a 2D moving standard deviation
filter to input images using a sliding window approach.

The layer computes local standard deviation as the windowed variance over a
sliding window. To avoid catastrophic cancellation for large-mean inputs (e.g.
pixel values ~200), the variance is computed in a shift-invariant two-pass form:
the input is first shifted by a per-channel constant (its spatial mean, treated as
a constant), then average pooling computes E[Y] and E[Y^2] of the shifted signal.
Because Var(X) == Var(X - c), this yields the exact windowed variance while keeping
the pooled terms O(variance) instead of O(mean^2).

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

    Computes local standard deviation over sliding windows. To avoid catastrophic
    cancellation for large-mean inputs, the variance is computed in a shift-invariant
    two-pass form: the input is first shifted by a per-channel spatial-mean constant
    (``stop_gradient``), then average pooling computes ``E[Y]`` and ``E[Y^2]`` of the
    shifted signal ``Y = X - c``. Since ``Var(X) == Var(X - c)``, the windowed
    variance ``E[Y^2] - (E[Y])^2`` is exact while both pooled terms stay
    ``O(variance)`` rather than ``O(mean^2)``, eliminating the precision loss of the
    naive ``E[X^2] - (E[X])^2`` shortcut. Each channel is processed independently.
    The variance is clamped to be non-negative before taking ``sqrt`` to absorb any
    residual floating-point error.

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
        # Two-pass, shift-invariant variance to avoid catastrophic cancellation.
        #
        # Resolution caveat: ``AveragePooling2D`` produces windowed means at the
        # POOLED grid resolution, not the per-pixel grid, so we cannot subtract a
        # per-window mean from ``inputs`` before pooling. Instead we exploit the
        # shift-invariance of variance: Var(X) == Var(X - c) for any constant c.
        # We subtract a per-channel constant ``c`` (the spatial mean of the batch,
        # stop-gradient so it is treated as a constant) to shrink the magnitudes
        # fed into the pooled E[Y] and E[Y²] terms. This keeps both pooled terms
        # O(variance) rather than O(mean²), so the ``E[Y²] - E[Y]²`` subtraction no
        # longer cancels two nearly-equal large numbers (the failure mode for
        # large-mean inputs such as pixel values ~200). The result is exactly the
        # windowed variance because the global per-channel shift is window-invariant.
        spatial_axes = (
            (2, 3) if self.data_format == "channels_first" else (1, 2)
        )
        shift = ops.stop_gradient(
            ops.mean(inputs, axis=spatial_axes, keepdims=True)
        )
        centered = inputs - shift

        # Compute E[Y] - local mean of the shifted values over the pooling window
        mean_x = self.pooler(centered, training=training)

        # Compute E[Y²] - local mean of squared shifted values over the window
        mean_x_sq = self.pooler(ops.square(centered), training=training)

        # Calculate local variance: Var(X) = Var(Y) = E[Y²] - (E[Y])²
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
