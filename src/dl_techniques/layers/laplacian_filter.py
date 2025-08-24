"""
This module provides Keras layers for applying Laplacian filters to image data,
a fundamental technique in classical image processing for edge detection.

The Laplacian operator is a second-order derivative filter that is highly sensitive
to rapid changes in intensity. In the context of images, this makes it an excellent
tool for edge detection, as it highlights regions like edges, corners, and lines
where pixel values change abruptly. The output of a Laplacian filter is typically
zero in areas of constant intensity, positive on one side of an edge, and negative on
the other side, creating a characteristic 'zero-crossing' at the edge itself.

Since the second derivative is very sensitive to noise, the Laplacian operator is
almost always combined with a Gaussian smoothing filter first to reduce noise before
edge detection. This module implements this concept through several common approximations:

-   **Difference of Gaussians (DoG):** A simple and efficient approximation where a
    Gaussian-blurred version of the image is subtracted from the original.
-   **Laplacian of Gaussian (LoG):** A more direct approach where a single kernel that
    combines both the Gaussian smoothing and the Laplacian differentiation is convolved
    with the image.
-   **Discrete Kernel:** The use of a small, fixed integer kernel (e.g., a 3x3 matrix)
    that directly approximates the Laplacian operator.

This module offers two layers to leverage these techniques:

1.  **`LaplacianFilter`:** A straightforward implementation based exclusively on the
    Difference of Gaussians (DoG) method. It uses an internal `GaussianFilter` layer
    and is simple to use for general-purpose edge detection.

2.  **`AdvancedLaplacianFilter`:** A more flexible and powerful layer that allows the user
    to explicitly choose between the three different implementation methods ('dog', 'log',
    or 'kernel'). This enables experimentation with different trade-offs between
    computational efficiency, filter accuracy, and kernel size.

It's important to note that these layers implement fixed, non-trainable filters. Their
parameters (the kernel shapes) are determined by mathematical formulas, not learned
from data. They serve as classical feature extractors that can be seamlessly integrated
into a modern deep learning pipeline.
"""

import keras
import numpy as np
from typing import Tuple, Union, List, Optional, Sequence, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .gaussian_filter import GaussianFilter

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LaplacianFilter(keras.layers.Layer):
    """Laplacian filter layer that detects edges by approximating the second derivative.

    This filter highlights areas of rapid intensity change in an image and is
    commonly used for edge detection. It works by applying a Gaussian blur
    and then computing the difference between the blurred image and the original image.

    The implementation uses a difference of Gaussians (DoG) approach, which
    is a common approximation of the Laplacian of Gaussian.

    Args:
        kernel_size: Tuple of two integers specifying the height and width of the 2D kernel.
            Must be positive odd integers for symmetric filtering.
        strides: Tuple of two integers specifying the strides of the convolution.
            Defaults to (1, 1). Note: Strides are ignored and forced to (1, 1) for this
            layer to preserve shape for the subtraction operation.
        sigma: Standard deviation for the Gaussian kernel. If float, same sigma is used for
            both dimensions. If tuple, (sigma_height, sigma_width) are used.
            If None or <= 0, calculated automatically from kernel_size.
        scale_factor: Float, scaling factor for the Laplacian response. Controls the
            strength of edge detection. Defaults to 1.0.
        kernel_initializer: Initializer for the kernel weights. Used for compatibility
            but not actively used since this is a fixed filter.
        kernel_regularizer: Optional regularizer for the kernel weights.
        **kwargs: Additional keyword arguments passed to the parent class constructor.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with same shape as input: `(batch_size, height, width, channels)`

    Returns:
        Tensor with highlighted edges, same shape as input tensor.

    Raises:
        ValueError: If kernel_size is not length 2, or if sigma is invalid.

    Example:
        ```python
        # Basic edge detection
        layer = LaplacianFilter(kernel_size=(5, 5), sigma=1.0)

        # Fine-tuned parameters
        layer = LaplacianFilter(
            kernel_size=(7, 7),
            sigma=(1.5, 1.5),
            scale_factor=2.0
        )

        # In a model
        inputs = keras.Input(shape=(224, 224, 3))
        edges = LaplacianFilter(kernel_size=(5, 5))(inputs)
        # edges will have shape (None, 224, 224, 3) with highlighted edges
        ```

    Note:
        This implementation follows modern Keras 3 patterns where sub-layers
        are created in __init__ and explicitly built in build() for robust
        serialization support.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 5),
        strides: Union[Tuple[int, int], List[int]] = (1, 1),
        sigma: Optional[Union[float, Tuple[float, float]]] = 1.0,
        scale_factor: float = 1.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the LaplacianFilter layer."""
        super().__init__(**kwargs)

        # Validate kernel_size
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be length 2")

        # Validate and store parameters
        self.kernel_size = kernel_size
        self.strides = tuple(strides) if isinstance(strides, list) else strides
        self.scale_factor = scale_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Handle sigma parameter
        if (sigma is None or
                (isinstance(sigma, (float, int)) and sigma <= 0.0)):
            # Default sigma based on kernel size
            self.sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            self.sigma = (float(sigma[0]), float(sigma[1]))
        elif isinstance(sigma, (float, int)):
            self.sigma = (float(sigma), float(sigma))
        else:
            raise ValueError(f"Invalid sigma value: {sigma}")

        # CREATE sub-layer in __init__ (following modern Keras 3 pattern)
        # Strides must be (1, 1) to preserve shape for the subtraction operation.
        self.gaussian_filter = GaussianFilter(
            kernel_size=self.kernel_size,
            strides=(1, 1),
            sigma=self.sigma,
            name="gaussian_filter"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and explicitly build the Gaussian filter.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build the sub-layer with the input shape
        if not self.gaussian_filter.built:
            self.gaussian_filter.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Apply the Laplacian filter to the input tensor.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating whether the layer is in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor with highlighted edges, same shape as input tensor.
        """
        # Apply Gaussian blur
        blurred = self.gaussian_filter(inputs, training=training)

        # Compute Laplacian as difference between blurred and original image.
        # This order (blurred - original) is consistent with the standard
        # Laplacian operator, which yields a negative response for a bright spot.
        laplacian = keras.ops.multiply(
            self.scale_factor,
            keras.ops.subtract(blurred, inputs)
        )

        return laplacian

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input for this layer).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for the layer.

        Returns ALL __init__ parameters for complete serialization support.

        Returns:
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "sigma": self.sigma,
            "scale_factor": self.scale_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AdvancedLaplacianFilter(keras.layers.Layer):
    """Advanced Laplacian filter with multiple implementation options.

    This layer offers different methods to compute the Laplacian:
    - 'dog': Difference of Gaussians (subtracting the original from a blurred image)
    - 'log': Laplacian of Gaussian (applying a LoG kernel directly)
    - 'kernel': Using a discrete Laplacian kernel

    Each method has different characteristics and performance implications.

    Args:
        method: String, the method to use. Must be one of 'dog', 'log', or 'kernel'.
        kernel_size: Tuple of two integers specifying the height and width of the 2D kernel.
        strides: Tuple of two integers specifying the strides of the convolution.
        sigma: Standard deviation for the Gaussian kernel. If float, same value for both
            dimensions. If tuple, (sigma_height, sigma_width) are used.
        scale_factor: Float, scaling factor for the Laplacian response.
        kernel_initializer: Initializer for the kernel weights (compatibility parameter).
        kernel_regularizer: Optional regularizer for the kernel weights.
        **kwargs: Additional keyword arguments passed to the parent class constructor.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor whose shape depends on the method and strides. For 'dog' method,
        output shape is the same as input shape. For 'log' and 'kernel' methods,
        the spatial dimensions are downsampled according to strides.

    Returns:
        Tensor with highlighted edges.

    Raises:
        ValueError: If method is not supported, or if parameters are invalid.

    Example:
        ```python
        # Difference of Gaussians (fastest)
        dog_layer = AdvancedLaplacianFilter(method='dog', kernel_size=(5, 5))

        # Laplacian of Gaussian (most accurate)
        log_layer = AdvancedLaplacianFilter(method='log', kernel_size=(7, 7), sigma=1.5)

        # Discrete kernel (traditional)
        kernel_layer = AdvancedLaplacianFilter(method='kernel', kernel_size=(3, 3))

        # In a model
        inputs = keras.Input(shape=(224, 224, 3))
        edges = AdvancedLaplacianFilter(method='log', kernel_size=(5, 5))(inputs)
        ```

    Note:
        The 'dog' method creates a GaussianFilter sub-layer in __init__ for proper
        serialization. The 'log' and 'kernel' methods create fixed kernels during
        build() since they depend on the number of input channels.
    """

    def __init__(
        self,
        method: Literal['dog', 'log', 'kernel'] = 'dog',
        kernel_size: Tuple[int, int] = (5, 5),
        strides: Union[Tuple[int, int], List[int]] = (1, 1),
        sigma: Union[float, Tuple[float, float]] = 1.0,
        scale_factor: float = 1.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the AdvancedLaplacianFilter layer."""
        super().__init__(**kwargs)

        # Validate method
        if method not in ['dog', 'log', 'kernel']:
            raise ValueError(f"Method '{method}' not supported. Use 'dog', 'log', or 'kernel'.")

        # Store ALL configuration parameters
        self.method = method
        self.kernel_size = kernel_size
        self.strides = tuple(strides) if isinstance(strides, list) else strides
        self.scale_factor = scale_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Handle sigma parameter
        if isinstance(sigma, (int, float)):
            self.sigma = (float(sigma), float(sigma))
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            self.sigma = (float(sigma[0]), float(sigma[1]))
        else:
            raise ValueError(f"Invalid sigma value: {sigma}")

        # CREATE sub-layer in __init__ for 'dog' method (following modern Keras 3 pattern)
        if self.method == 'dog':
            # For DoG, strides must be (1, 1) to ensure blurred image has the
            # same shape as the input for the subtraction operation.
            self.gaussian_filter = GaussianFilter(
                kernel_size=self.kernel_size,
                strides=(1, 1),
                sigma=self.sigma,
                name="gaussian_filter"
            )
        else:
            self.gaussian_filter = None

        # Kernel weights (created in build() since they depend on input channels)
        self.filter_kernel = None

    def _create_laplacian_kernel(self, channels: int) -> keras.KerasTensor:
        """Create a discrete Laplacian kernel.

        Args:
            channels: Number of input channels.

        Returns:
            Laplacian kernel tensor.
        """
        # Simple discrete Laplacian kernel
        if self.kernel_size == (3, 3):
            # Standard 3x3 Laplacian kernel
            kernel_2d = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=np.float32)
        else:
            # For other sizes, use a LoG approximation
            kernel_2d = self._create_log_kernel()

        # Create the depthwise kernel
        kernel = np.zeros((*self.kernel_size, channels, 1), dtype=np.float32)
        for i in range(channels):
            kernel[:, :, i, 0] = kernel_2d

        return keras.ops.convert_to_tensor(kernel, dtype=self.compute_dtype)

    def _create_log_kernel(self) -> np.ndarray:
        """Create a Laplacian of Gaussian (LoG) kernel.

        Returns:
            LoG kernel as numpy array.
        """
        sigma_x, sigma_y = self.sigma
        height, width = self.kernel_size

        y, x = np.mgrid[-(height // 2):((height + 1) // 2), -(width // 2):((width + 1) // 2)]

        # LoG equation
        x_squared_norm = x ** 2 / (2 * sigma_x ** 2)
        y_squared_norm = y ** 2 / (2 * sigma_y ** 2)

        # Normalized distance
        r_squared = x_squared_norm + y_squared_norm

        # LoG
        log_kernel = -1.0 / (np.pi * sigma_x * sigma_y) * (1.0 - r_squared) * np.exp(-r_squared)

        # Normalize to ensure kernel sums to zero (important for Laplacian)
        log_kernel = log_kernel - np.mean(log_kernel)

        return log_kernel

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and initialize components based on the selected method.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Last dimension (channels) of input must be defined")

        if self.method == 'dog':
            # Build the Gaussian filter sub-layer
            if not self.gaussian_filter.built:
                self.gaussian_filter.build(input_shape)
        elif self.method == 'log':
            # Create a LoG kernel
            log_kernel = self._create_log_kernel().reshape(*self.kernel_size, 1, 1)
            kernel_tensor = keras.ops.convert_to_tensor(log_kernel, dtype=self.compute_dtype)
            # Repeat for each channel
            self.filter_kernel = keras.ops.tile(kernel_tensor, [1, 1, channels, 1])
        else:  # 'kernel'
            # Create a discrete Laplacian kernel
            self.filter_kernel = self._create_laplacian_kernel(channels)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Apply the Laplacian filter to the input tensor.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating whether the layer is in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor with highlighted edges, same shape as input tensor.
        """
        if self.method == 'dog':
            # Difference of Gaussians approach
            blurred = self.gaussian_filter(inputs, training=training)
            result = keras.ops.subtract(blurred, inputs)
            return keras.ops.multiply(self.scale_factor, result)
        else:
            # Direct convolution with LoG or Laplacian kernel
            conv_result = keras.ops.depthwise_conv(
                inputs=inputs,
                kernel=self.filter_kernel,
                strides=self.strides,
                padding="same"
            )
            return keras.ops.multiply(self.scale_factor, conv_result)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        if self.method == 'dog':
            return input_shape

        # For 'log' and 'kernel' with 'same' padding
        output_h = (input_shape[1] + self.strides[0] - 1) // self.strides[0]
        output_w = (input_shape[2] + self.strides[1] - 1) // self.strides[1]
        return input_shape[0], output_h, output_w, input_shape[3]

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for the layer.

        Returns ALL __init__ parameters for complete serialization support.

        Returns:
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update({
            "method": self.method,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "sigma": self.sigma,
            "scale_factor": self.scale_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

# ---------------------------------------------------------------------