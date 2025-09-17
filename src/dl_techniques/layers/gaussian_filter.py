"""Apply a 2D Gaussian blur using a depthwise convolution.

This layer implements a Gaussian filter, a fundamental low-pass filter used
for smoothing images and reducing high-frequency noise. The filtering is
achieved by convolving the input image with a kernel whose weights are derived
from a 2D Gaussian distribution. This operation is a cornerstone of many
computer vision algorithms, often serving as a preprocessing step to enhance
feature stability.

Architectural and Mathematical Foundations:
The core of this layer is the 2D Gaussian function, which is used to generate
the convolution kernel. The function is defined as:

    G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

The parameter `sigma` (standard deviation) controls the "spread" of the bell-
shaped curve. A larger `sigma` results in a wider curve, leading to a kernel
that averages pixels over a larger neighborhood, producing a more pronounced
blur. Conversely, a smaller `sigma` yields a sharper kernel with less blur.

To create the discrete convolution kernel, this continuous function is sampled
at integer coordinates `(x, y)` over a grid defined by `kernel_size`. The
resulting values are then normalized to sum to 1, ensuring that the overall
brightness of the image is preserved after filtering.

The filtering operation is implemented as a **depthwise convolution**. This is a
critical architectural choice. In image processing, blurring is typically
performed independently on each color channel (e.g., R, G, B). A depthwise
convolution naturally enforces this by applying a separate filter to each input
channel. In this implementation, the *same* Gaussian kernel is replicated for
each channel, ensuring that all channels are blurred consistently. This approach
prevents unnatural color mixing ("bleeding") between channels and is both
conceptually sound and computationally efficient.

References:
    - Gonzalez, R. C., & Woods, R. E. "Digital Image Processing". This text
      provides a comprehensive background on Gaussian filtering and its
      mathematical properties in the context of image processing.

    - Canny, J. "A Computational Approach to Edge Detection". This seminal
      paper demonstrates the use of Gaussian smoothing as a critical first
      step to reduce noise before computing image gradients for edge detection.
      https://doi.org/10.1109/TPAMI.1986.4767851
"""

import keras
from keras import ops
from typing import Tuple, Union, List, Optional, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..utils.tensors import depthwise_gaussian_kernel

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GaussianFilter(keras.layers.Layer):
    """Applies Gaussian blur filter to input images.

    This layer creates a depthwise convolution with Gaussian kernel to perform
    filtering on images. Each channel is processed independently using the same
    Gaussian filter.

    Args:
        kernel_size: Tuple of two integers specifying the height and width of the
            2D Gaussian kernel.
        strides: Tuple/list of 2 integers specifying the strides of the convolution
            along height and width.
        sigma: Standard deviation of the Gaussian distribution. If a single value,
            the same sigma is used for both dimensions. If a tuple, (sigma_height,
            sigma_width) are used. If -1 or None, sigma is calculated based on
            kernel size.
        padding: String, either "valid" or "same" (case-insensitive).
        data_format: String, either "channels_last" or "channels_first".
        trainable: Boolean, if True allow the weights to change, otherwise static
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
        >>> x = np.random.rand(4, 32, 32, 3)  # Input images
        >>> layer = GaussianFilter(kernel_size=(5, 5), sigma=1.5)
        >>> y = layer(x)
        >>> print(y.shape)
        (4, 32, 32, 3)
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            sigma: Union[float, Tuple[float, float]] = -1,
            padding: str = "same",
            data_format: Optional[str] = None,
            trainable: bool = False,
            **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        # Validate and store kernel size
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be length 2")
        self.kernel_size = kernel_size

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

        # Will be set in build()
        self.kernel = None

        logger.info(
            f"kernel_size: {self.kernel_size}, "
            f"padding: {self.padding}, "
            f"sigma: {self.sigma}"
        )

    def build(self, input_shape):
        """Build the Gaussian kernel weights based on input shape."""
        if self.built:
            return

        # Get number of channels
        if self.data_format == "channels_last":
            channels = input_shape[-1]
        else:  # channels_first
            channels = input_shape[1]

        if channels is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                "The 'channels' argument is 'None'."
            )

        # Create the Gaussian kernel as a numpy array first
        kernel_np = depthwise_gaussian_kernel(
            channels=channels,
            kernel_size=self.kernel_size,
            nsig=self.sigma,
            dtype=self.compute_dtype,
        )

        # Use add_weight to create a properly managed state variable.
        # This ensures the kernel can be used across different tf.function scopes.
        self.kernel = self.add_weight(
            name="gaussian_kernel",
            shape=kernel_np.shape,
            dtype=self.compute_dtype,
            initializer=keras.initializers.Constant(kernel_np),
            trainable=self.trainable,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Apply the Gaussian filter to the input tensor.

        Args:
            inputs: Input tensor of shape:
                - If data_format="channels_last": (batch_size, height, width, channels)
                - If data_format="channels_first": (batch_size, channels, height, width)
            training: Boolean indicating whether in training mode (unused).

        Returns:
            Filtered tensor with the same shape as the input tensor.
        """
        # Apply depthwise convolution
        outputs = ops.nn.depthwise_conv(
            inputs=inputs,
            kernel=self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format
        )

        return outputs

    def get_config(self):
        """Return the configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "sigma": self.sigma,
            "padding": self.padding,
            "data_format": self.data_format
        })
        return config

# ---------------------------------------------------------------------
