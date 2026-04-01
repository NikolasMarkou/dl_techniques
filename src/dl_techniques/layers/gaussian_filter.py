"""
2D Gaussian blur using a depthwise convolution.

Implements a Gaussian filter for smoothing images and reducing high-frequency
noise. The filtering is achieved by convolving the input with a kernel derived
from the 2D Gaussian function G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2) /
(2*sigma^2)). The kernel is applied via depthwise convolution so each channel
is filtered independently with the same kernel, preventing color bleeding.

References:
    - Gonzalez, R. C., & Woods, R. E. "Digital Image Processing".
    - Canny, J. "A Computational Approach to Edge Detection".
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
    """
    Apply Gaussian blur filter to input images via depthwise convolution.

    Creates a depthwise convolution with Gaussian kernel weights derived from
    G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2)), where sigma
    controls the blur spread. Each channel is processed independently using the
    same normalized kernel to preserve brightness and prevent channel mixing.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input [batch, H, W, C]                  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Depthwise Conv2D with Gaussian kernel   │
        │  (same kernel replicated per channel)    │
        │  kernel_size, strides, padding           │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output [batch, H', W', C]               │
        └──────────────────────────────────────────┘

    :param kernel_size: Height and width of the 2D Gaussian kernel.
    :type kernel_size: Tuple[int, int]
    :param strides: Strides of the convolution along height and width.
    :type strides: Union[Tuple[int, int], List[int]]
    :param sigma: Standard deviation of the Gaussian distribution. If a single
        value, same sigma for both dimensions. If a tuple, (sigma_h, sigma_w).
        If -1 or None, sigma is calculated from kernel size.
    :type sigma: Union[float, Tuple[float, float]]
    :param padding: Either "valid" or "same" (case-insensitive).
    :type padding: str
    :param data_format: Either "channels_last" or "channels_first".
    :type data_format: Optional[str]
    :param trainable: If True allow the weights to change, otherwise static.
    :type trainable: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Build the Gaussian kernel weights based on input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
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

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (unused).
        :type training: Optional[bool]
        :return: Filtered tensor with the same shape as the input.
        :rtype: keras.KerasTensor
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

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
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
