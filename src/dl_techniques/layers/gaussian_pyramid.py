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
    """Gaussian Pyramid layer for multi-scale image representation.

    This layer creates a Gaussian pyramid by repeatedly applying Gaussian blur
    and downsampling to create multiple scales of the input image. Each level
    is a downsampled version of the previous level, creating a multi-scale
    representation suitable for various computer vision tasks.

    Args:
        levels: Integer, number of pyramid levels to generate. Must be >= 1.
            The first level is the original (possibly blurred) image.
        kernel_size: Tuple of two integers specifying the height and width of the
            2D Gaussian kernel used for blurring before downsampling.
        sigma: Standard deviation of the Gaussian distribution. If a single value,
            the same sigma is used for both dimensions. If a tuple, (sigma_height,
            sigma_width) are used. If -1 or None, sigma is calculated based on
            kernel size.
        scale_factor: Integer, downsampling factor between levels. Default is 2.
        padding: String, either "valid" or "same" (case-insensitive).
        data_format: String, either "channels_last" or "channels_first".
        trainable: Boolean, if True allow the Gaussian filter weights to change,
            otherwise they remain static.
        **kwargs: Additional keyword arguments passed to the Layer base class.

    Input shape:
        4D tensor with shape:
        - If data_format="channels_last": (batch_size, height, width, channels)
        - If data_format="channels_first": (batch_size, channels, height, width)

    Output shape:
        List of 4D tensors, each with progressively smaller spatial dimensions:
        - If data_format="channels_last":
            [(batch_size, height, width, channels),
             (batch_size, height//2, width//2, channels),
             (batch_size, height//4, width//4, channels), ...]
        - If data_format="channels_first":
            [(batch_size, channels, height, width),
             (batch_size, channels, height//2, width//2),
             (batch_size, channels, height//4, width//4), ...]

    Returns:
        List of tensors representing the Gaussian pyramid levels.

    Raises:
        ValueError: If levels < 1, kernel_size is not length 2, scale_factor < 1,
            or invalid padding/data_format values.

    Example:
        ```python
        # Basic usage
        layer = GaussianPyramid(levels=3, kernel_size=(5, 5), sigma=1.0)

        # Advanced configuration
        layer = GaussianPyramid(
            levels=4,
            kernel_size=(7, 7),
            sigma=(1.5, 1.5),
            scale_factor=2,
            padding="same"
        )

        # In a model
        inputs = keras.Input(shape=(64, 64, 3))
        pyramid = GaussianPyramid(levels=3)(inputs)
        # pyramid is a list of 3 tensors with shapes:
        # [(None, 64, 64, 3), (None, 32, 32, 3), (None, 16, 16, 3)]
        ```

    Note:
        This implementation follows modern Keras 3 patterns where all sub-layers
        are created in __init__ and explicitly built in build() for robust
        serialization support.
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

        This method explicitly builds each sub-layer with the correct input shape
        for robust serialization support.

        Args:
            input_shape: Shape tuple of the input tensor.
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

        Args:
            input_shape: Shape tuple before downsampling.

        Returns:
            Shape tuple after downsampling.
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

        Args:
            inputs: Input tensor to downsample.

        Returns:
            Downsampled tensor.
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

        Args:
            inputs: Input tensor of shape:
                - If data_format="channels_last": (batch_size, height, width, channels)
                - If data_format="channels_first": (batch_size, channels, height, width)
            training: Boolean indicating whether in training mode.

        Returns:
            List of tensors representing the Gaussian pyramid levels.
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

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            List of shape tuples for each pyramid level.
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

        Returns ALL __init__ parameters for complete serialization support.

        Returns:
            Dictionary containing the layer configuration.
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

    This is a convenience function that creates a Gaussian pyramid of the input.

    Args:
        inputs: Input tensor of shape:
            - If data_format="channels_last": (batch_size, height, width, channels)
            - If data_format="channels_first": (batch_size, channels, height, width)
        levels: Integer, number of pyramid levels to generate. Must be >= 1.
        kernel_size: Tuple of two integers specifying the height and width of the
            2D Gaussian kernel. Defaults to (5, 5).
        sigma: Standard deviation of the Gaussian distribution. If a single value,
            the same sigma is used for both dimensions. If a tuple, (sigma_height,
            sigma_width) are used. If -1, sigma is calculated based on kernel size.
        scale_factor: Integer, downsampling factor between levels. Default is 2.
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "same".
        data_format: String, either "channels_last" or "channels_first". If None,
            uses the default format from Keras configuration.
        name: Optional name for the operation.

    Returns:
        List of tensors representing the Gaussian pyramid levels.

    Example:
        ```python
        import numpy as np

        # Create sample input
        x = keras.random.normal(shape=(4, 32, 32, 3))

        # Generate 3-level pyramid
        pyramid = gaussian_pyramid(x, levels=3, kernel_size=(5, 5), sigma=1.0)

        print(f"Number of levels: {len(pyramid)}")
        print(f"Shapes: {[p.shape for p in pyramid]}")
        # Output: Number of levels: 3
        # Output: Shapes: [(4, 32, 32, 3), (4, 16, 16, 3), (4, 8, 8, 3)]
        ```
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
