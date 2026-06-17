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

Beyond these single-scale DoG edge-detector filters, this module also provides
``LaplacianPyramidLevel``, an exactly-invertible multi-scale split/merge pyramid LEVEL.
This is a distinct operation from the edge-detector filters above: rather than producing
a single edge map, it downsamples a low-frequency residual to the next pyramid level and
keeps the complementary high-frequency band, such that ``merge(split(x)) == x`` to float
precision. It is channel-preserving and signal-level (no learnable channel change), which
is exactly what makes its reconstruction identity hold.
"""

import keras
import numpy as np
from typing import Tuple, Union, List, Optional, Sequence, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .gaussian_filter import GaussianFilter
from .blur_pool import BlurPool2D
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LaplacianFilter(keras.layers.Layer):
    """Laplacian filter layer using Difference of Gaussians for edge detection.

    This filter highlights areas of rapid intensity change in an image by
    applying a Gaussian blur and computing the difference between the blurred
    and original image: ``laplacian = scale_factor * (blurred - input)``. This
    Difference of Gaussians (DoG) approach is a common and efficient
    approximation of the Laplacian of Gaussian operator. The filter is
    non-trainable with parameters determined by mathematical formulas.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────┐
        │  Input [B, H, W, C]           │
        └───────┬───────────────┬───────┘
                │               │
                ▼               │
        ┌───────────────┐       │
        │ GaussianFilter│       │
        │(kernel, sigma)│       │
        └───────┬───────┘       │
                │               │
                ▼               ▼
        ┌───────────────────────────────┐
        │  scale * (blurred - input)    │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Output [B, H, W, C]          │
        └───────────────────────────────┘

    :param kernel_size: Height and width of the 2D kernel. Must be positive
        odd integers. Defaults to ``(5, 5)``.
    :type kernel_size: Tuple[int, int]
    :param strides: Strides of the convolution. Forced to ``(1, 1)`` to
        preserve shape for the subtraction. Defaults to ``(1, 1)``.
    :type strides: Union[Tuple[int, int], List[int]]
    :param sigma: Standard deviation for the Gaussian kernel. If float, same
        for both dimensions. If tuple, ``(sigma_h, sigma_w)``. If None or <= 0,
        calculated from kernel_size.
    :type sigma: Optional[Union[float, Tuple[float, float]]]
    :param scale_factor: Scaling factor for the Laplacian response. Defaults to 1.0.
    :type scale_factor: float
    :param kernel_initializer: Initializer for compatibility (not actively used).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the parent class.

    :raises ValueError: If kernel_size is not length 2 or sigma is invalid.
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

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
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

        :param inputs: Input tensor of shape ``[batch_size, height, width, channels]``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :param kwargs: Additional keyword arguments.
        :return: Tensor with highlighted edges, same shape as input.
        :rtype: keras.KerasTensor
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

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for the layer.

        :return: Dictionary containing configuration parameters.
        :rtype: Dict[str, Any]
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
    """Advanced Laplacian filter with multiple implementation methods.

    This layer offers three methods to compute the Laplacian operator for edge
    detection: ``'dog'`` (Difference of Gaussians), ``'log'`` (Laplacian of
    Gaussian kernel convolution), and ``'kernel'`` (discrete Laplacian kernel).
    Each method trades off between computational efficiency and filter accuracy.
    All filters are non-trainable with fixed mathematical kernels.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │    Input [B, H, W, C]             │
        └───────────────┬───────────────────┘
                        │
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
        ┌────────┐  ┌────────┐  ┌────────┐
        │  DoG   │  │  LoG   │  │Discrete│
        │ method │  │ method │  │ kernel │
        └───┬────┘  └───┬────┘  └───┬────┘
            │           │           │
            └───────────┼───────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │  scale_factor * result            │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │  Output [B, H', W', C]            │
        └───────────────────────────────────┘

    :param method: Method to use: ``'dog'``, ``'log'``, or ``'kernel'``.
        Defaults to ``'dog'``.
    :type method: Literal['dog', 'log', 'kernel']
    :param kernel_size: Height and width of the 2D kernel. Defaults to ``(5, 5)``.
    :type kernel_size: Tuple[int, int]
    :param strides: Strides of the convolution. Defaults to ``(1, 1)``.
    :type strides: Union[Tuple[int, int], List[int]]
    :param sigma: Standard deviation for the Gaussian kernel. Defaults to 1.0.
    :type sigma: Union[float, Tuple[float, float]]
    :param scale_factor: Scaling factor for the Laplacian response. Defaults to 1.0.
    :type scale_factor: float
    :param kernel_initializer: Initializer for compatibility (not actively used).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the parent class.

    :raises ValueError: If method is not supported or parameters are invalid.
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

        :param channels: Number of input channels.
        :type channels: int
        :return: Laplacian kernel tensor.
        :rtype: keras.KerasTensor
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

        :return: LoG kernel as numpy array.
        :rtype: np.ndarray
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

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
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

        :param inputs: Input tensor of shape ``[batch_size, height, width, channels]``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :param kwargs: Additional keyword arguments.
        :return: Tensor with highlighted edges.
        :rtype: keras.KerasTensor
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

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        if self.method == 'dog':
            return input_shape

        # For 'log' and 'kernel' with 'same' padding
        output_h = (input_shape[1] + self.strides[0] - 1) // self.strides[0]
        output_w = (input_shape[2] + self.strides[1] - 1) // self.strides[1]
        return input_shape[0], output_h, output_w, input_shape[3]

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for the layer.

        :return: Dictionary containing configuration parameters.
        :rtype: Dict[str, Any]
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


@keras.saving.register_keras_serializable()
class LaplacianPyramidLevel(keras.layers.Layer):
    """Explicit, signal-level Laplacian pyramid split/merge for one level.

    **Intent**: Provide an exactly-invertible, auditable multi-scale frequency
    decomposition for a single pyramid level. Unlike :class:`LaplacianFilter`
    (a single-scale edge detector), this layer splits an input into a
    downsampled low-frequency band and a same-resolution high-frequency
    residual, such that merging them reconstructs the input to float precision.
    The split is purely signal-level (channel-preserving, NO learnable channel
    change), which is exactly what makes the reconstruction identity hold.

    **Architecture**:
    ```
    split(x):
        x (B, H, W, C)
           |
           +-----------------------------+
           |                             |
           v                             |
        blur = GaussianFilter(x)         |
           |                             |
           v                             |
        low  = BlurPool2D(blur)          |  (B, H/2, W/2, C)
           |                             |
           v                             |
        up   = UpSampling2D(low)         |  (B, H, W, C)
           |                             |
           +------------> high = x - up(low)   (B, H, W, C)

    merge(low, high):
        x_rec = high + up(low)  ==  x   (exact reconstruction identity)
    ```

    Args:
        blur_kernel_size: Height/width of the Gaussian blur kernel as a
            length-2 sequence of positive ints. Defaults to ``(5, 5)``.
        blur_sigma: Gaussian sigma; ``-1`` derives it from the kernel size.
            Defaults to ``-1``.
        blur_trainable: If ``True`` the blur kernel is learnable; the default
            ``False`` keeps the split a fixed, auditable signal operation.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor: `(batch_size, height, width, channels)`.

    Output shape:
        Tuple of two 4D tensors ``(low, high)`` where
        ``low`` is `(batch_size, height / 2, width / 2, channels)` and
        ``high`` is `(batch_size, height, width, channels)`.

    The decomposition is channel-preserving and operates at the raw signal
    level, so ``merge(split(x)) == x`` holds to float precision independent of
    blur quality (``high`` is DEFINED as the residual ``x - up(low)``).

    See Also:
        :class:`LaplacianFilter` -- a single-scale Difference-of-Gaussians edge
        detector. It produces one same-resolution edge map (no downsampling and
        no inverse). ``LaplacianPyramidLevel`` is a different operation: a
        multi-scale, exactly-invertible split/merge that downsamples a
        low-frequency residual to the next level and keeps the complementary
        high-frequency band.

    :raises ValueError: If ``blur_kernel_size`` is not a length-2 sequence of
        positive ints, ``blur_sigma`` is not a number, or ``blur_trainable`` is
        not a bool.
    """

    def __init__(
        self,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: float = -1,
        blur_trainable: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate blur_kernel_size: length-2 sequence of positive ints.
        if (not isinstance(blur_kernel_size, Sequence)
                or isinstance(blur_kernel_size, str)
                or len(blur_kernel_size) != 2):
            raise ValueError(
                "blur_kernel_size must be a length-2 sequence (height, width), "
                f"got {blur_kernel_size!r}"
            )
        if not all(isinstance(k, int) and not isinstance(k, bool) and k > 0
                   for k in blur_kernel_size):
            raise ValueError(
                "blur_kernel_size entries must be positive ints, "
                f"got {blur_kernel_size!r}"
            )

        # Validate blur_sigma: numeric (bool excluded).
        if isinstance(blur_sigma, bool) or not isinstance(blur_sigma, (int, float)):
            raise ValueError(
                f"blur_sigma must be a number, got {blur_sigma!r}"
            )

        # Validate blur_trainable: bool.
        if not isinstance(blur_trainable, bool):
            raise ValueError(
                f"blur_trainable must be a bool, got {blur_trainable!r}"
            )

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.blur_trainable = blur_trainable

        # Sublayers created in __init__ (built explicitly in build()).
        self.blur = GaussianFilter(
            kernel_size=blur_kernel_size,
            strides=(1, 1),
            sigma=blur_sigma,
            padding="same",
            trainable=blur_trainable,
        )
        self.down = BlurPool2D(strides=2)
        self.up = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

    def build(self, input_shape) -> None:
        # Build sublayers in computational order; super().build() LAST.
        self.blur.build(input_shape)
        blur_out = self.blur.compute_output_shape(input_shape)
        self.down.build(blur_out)
        low_shape = self.down.compute_output_shape(blur_out)
        self.up.build(low_shape)
        super().build(input_shape)

    def split(self, x):
        """Decompose ``x`` into ``(low, high)`` signal bands.

        :param x: Input tensor ``(B, H, W, C)``.
        :return: ``(low, high)`` where ``low`` is ``(B, H/2, W/2, C)`` and
            ``high`` is ``(B, H, W, C)``.
        """
        low = self.down(self.blur(x))
        high = keras.ops.subtract(x, self.up(low))
        return low, high

    def merge(self, low, high):
        """Reconstruct the level: ``high + up(low)`` (exact inverse of split).

        :param low: Low band ``(B, H/2, W/2, C)``.
        :param high: High band ``(B, H, W, C)``.
        :return: Reconstructed tensor ``(B, H, W, C)``.
        """
        return keras.ops.add(high, self.up(low))

    def call(self, inputs):
        return self.split(inputs)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        low_h = None if h is None else h // 2
        low_w = None if w is None else w // 2
        low_shape = (batch, low_h, low_w, c)
        high_shape = (batch, h, w, c)
        return low_shape, high_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "blur_kernel_size": self.blur_kernel_size,
                "blur_sigma": self.blur_sigma,
                "blur_trainable": self.blur_trainable,
            }
        )
        return config


# ---------------------------------------------------------------------
