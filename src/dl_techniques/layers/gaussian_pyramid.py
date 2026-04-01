"""
Construct a multi-scale Gaussian pyramid representation of an image.

Implements the classic Gaussian pyramid by iteratively applying Gaussian blur
(anti-aliasing low-pass filter) followed by downsampling. Each level is a
smoothed, lower-resolution version of its predecessor. The Gaussian function
G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2)) prevents aliasing
artifacts per the Nyquist-Shannon sampling theorem by limiting signal
bandwidth before subsampling.

References:
    - Burt, P. J. and Adelson, E. H. "The Laplacian Pyramid as a Compact
      Image Code". https://doi.org/10.1109/T-C.1983.225452
"""

import keras
from keras import ops
from typing import Tuple, Union, Optional, Sequence, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .gaussian_filter import GaussianFilter


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GaussianPyramid(keras.layers.Layer):
    """
    Gaussian Pyramid layer for multi-scale image representation.

    Creates a Gaussian pyramid by repeatedly applying Gaussian blur and
    downsampling to produce multiple scales of the input image. Each level
    is a low-pass filtered and spatially reduced version of the previous
    level, forming a hierarchical representation for scale-invariant feature
    detection, image registration, and texture analysis.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input [batch, H, W, C]                  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Level 0: GaussianFilter ──▶ output[0]   │
        │           [batch, H, W, C]               │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  AvgPool (downsample by scale_factor)    │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Level 1: GaussianFilter ──▶ output[1]   │
        │           [batch, H/2, W/2, C]           │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  AvgPool (downsample by scale_factor)    │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Level 2: GaussianFilter ──▶ output[2]   │
        │           [batch, H/4, W/4, C]           │
        └──────────────────────────────────────────┘

    :param levels: Number of pyramid levels to generate. Must be >= 1.
        Defaults to 3.
    :type levels: int
    :param kernel_size: Height and width of the 2D Gaussian kernel.
    :type kernel_size: Tuple[int, int]
    :param sigma: Standard deviation of the Gaussian. If a single value, same
        for both dimensions. If tuple, (sigma_h, sigma_w). If -1 or None,
        calculated from kernel size.
    :type sigma: Union[float, Tuple[float, float]]
    :param scale_factor: Downsampling factor between levels. Defaults to 2.
    :type scale_factor: int
    :param padding: Either "valid" or "same" (case-insensitive).
    :type padding: str
    :param data_format: Either "channels_last" or "channels_first".
    :type data_format: Optional[str]
    :param trainable: If True allow the Gaussian filter weights to change.
    :type trainable: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            levels: int = 3,
            kernel_size: Tuple[int, int] = (5, 5),
            sigma: Union[float, Tuple[float, float]] = -1,
            scale_factor: int = 2,
            padding: str = "same",
            data_format: Optional[str] = None,
            trainable: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(trainable=trainable, **kwargs)

        # Validate levels
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")

        # Validate kernel size
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be length 2")

        # Validate scale factor
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")

        # Store ALL configuration parameters
        self.levels = levels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Process sigma
        if (sigma is None or
                (isinstance(sigma, (float, int)) and sigma <= 0)):
            # Default sigma based on kernel size
            self.sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            self.sigma = (float(sigma[0]), float(sigma[1]))
        elif isinstance(sigma, (float, int)):
            self.sigma = (float(sigma), float(sigma))
        else:
            raise ValueError(f"Invalid sigma value: {sigma}")

        self.padding = padding.lower()
        if self.padding not in {"valid", "same"}:
            raise ValueError(f"padding must be 'valid' or 'same', got {padding}")

        # Process data_format
        self.data_format = keras.backend.image_data_format() if data_format is None else data_format
        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"data_format must be 'channels_first' or 'channels_last', got {data_format}")

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.gaussian_filters: List[GaussianFilter] = []
        for i in range(self.levels):
            gaussian_filter = GaussianFilter(
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                padding=self.padding,
                data_format=self.data_format,
                trainable=self.trainable,
                name=f"gaussian_filter_{i}"
            )
            self.gaussian_filters.append(gaussian_filter)

        logger.info(
            f"GaussianPyramid: levels={self.levels}, "
            f"kernel_size={self.kernel_size}, "
            f"scale_factor={self.scale_factor}, "
            f"sigma={self.sigma}, "
            f"data_format={self.data_format}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all Gaussian filters with appropriate input shapes.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build sub-layers in computational order with correct shapes
        current_shape = input_shape

        for i, gaussian_filter in enumerate(self.gaussian_filters):
            # Build the gaussian filter with current shape
            gaussian_filter.build(current_shape)

            # Update shape for next level (downsampled)
            if i < self.levels - 1:  # Don't update for the last level
                current_shape = self._compute_downsampled_shape(current_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _compute_downsampled_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the shape after downsampling.

        :param input_shape: Shape tuple before downsampling.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape tuple after downsampling.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape_list = list(input_shape)

        if self.data_format == "channels_last":
            # (batch_size, height, width, channels)
            if input_shape_list[1] is not None:
                input_shape_list[1] = max(1, input_shape_list[1] // self.scale_factor)
            if input_shape_list[2] is not None:
                input_shape_list[2] = max(1, input_shape_list[2] // self.scale_factor)
        else:  # channels_first
            # (batch_size, channels, height, width)
            if input_shape_list[2] is not None:
                input_shape_list[2] = max(1, input_shape_list[2] // self.scale_factor)
            if input_shape_list[3] is not None:
                input_shape_list[3] = max(1, input_shape_list[3] // self.scale_factor)

        return tuple(input_shape_list)

    def _downsample(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Downsample the input tensor by the scale factor.

        :param inputs: Input tensor to downsample.
        :type inputs: keras.KerasTensor
        :return: Downsampled tensor.
        :rtype: keras.KerasTensor
        """
        if self.scale_factor == 1:
            return inputs

        # Use average pooling for downsampling
        return ops.nn.average_pool(
            inputs=inputs,
            pool_size=(self.scale_factor, self.scale_factor),
            strides=(self.scale_factor, self.scale_factor),
            padding="valid",
            data_format=self.data_format
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """Apply Gaussian pyramid decomposition.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: List of tensors representing the Gaussian pyramid levels.
        :rtype: List[keras.KerasTensor]
        """
        results = []
        x = inputs

        for i, gaussian_filter in enumerate(self.gaussian_filters):
            # Apply Gaussian blur to current level
            x_blurred = gaussian_filter(x, training=training)
            results.append(x_blurred)

            # Downsample for next level (except for the last level)
            if i < self.levels - 1:
                x = self._downsample(x_blurred)

        return results

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> List[Tuple[Optional[int], ...]]:
        """Compute the output shapes for all pyramid levels.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: List of shape tuples for each pyramid level.
        :rtype: List[Tuple[Optional[int], ...]]
        """
        output_shapes = []
        current_shape = input_shape

        for i in range(self.levels):
            output_shapes.append(current_shape)
            if i < self.levels - 1:  # Don't compute for the last level
                current_shape = self._compute_downsampled_shape(current_shape)

        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "levels": self.levels,
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "scale_factor": self.scale_factor,
            "padding": self.padding,
            "data_format": self.data_format,
        })
        return config


# ---------------------------------------------------------------------


def gaussian_pyramid(
        inputs: keras.KerasTensor,
        levels: int = 3,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Union[float, Tuple[float, float]] = -1,
        scale_factor: int = 2,
        padding: str = "same",
        data_format: Optional[str] = None,
        name: Optional[str] = None
) -> List[keras.KerasTensor]:
    """Functional interface for Gaussian pyramid decomposition.

    :param inputs: Input tensor.
    :type inputs: keras.KerasTensor
    :param levels: Number of pyramid levels. Must be >= 1. Defaults to 3.
    :type levels: int
    :param kernel_size: Height and width of the Gaussian kernel. Defaults to
        (5, 5).
    :type kernel_size: Tuple[int, int]
    :param sigma: Standard deviation of the Gaussian. If -1, calculated from
        kernel size.
    :type sigma: Union[float, Tuple[float, float]]
    :param scale_factor: Downsampling factor between levels. Defaults to 2.
    :type scale_factor: int
    :param padding: Either "valid" or "same". Defaults to "same".
    :type padding: str
    :param data_format: Either "channels_last" or "channels_first". If None,
        uses Keras default.
    :type data_format: Optional[str]
    :param name: Optional name for the operation.
    :type name: Optional[str]
    :return: List of tensors representing the Gaussian pyramid levels.
    :rtype: List[keras.KerasTensor]
    """
    layer = GaussianPyramid(
        levels=levels,
        kernel_size=kernel_size,
        sigma=sigma,
        scale_factor=scale_factor,
        padding=padding,
        data_format=data_format,
        name=name
    )
    return layer(inputs)

# ---------------------------------------------------------------------
