import keras
import numpy as np
from keras import ops
from typing import Tuple, Union, List, Optional, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.tensors import depthwise_gaussian_kernel

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

    def build(self, input_shape):
        """Build the Gaussian kernel weights based on input shape.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Get number of channels
        if self.data_format == "channels_last":
            channels = input_shape[-1]
        else:  # channels_first
            channels = input_shape[1]

        # Create the Gaussian kernel using the provided utility function
        kernel_np = depthwise_gaussian_kernel(
            channels=channels,
            kernel_size=self.kernel_size,
            nsig=self.sigma,
            dtype=np.float32
        )

        # Convert numpy array to tensor with proper dtype
        self.kernel = keras.ops.convert_to_tensor(kernel_np, dtype=self.compute_dtype)
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


def gaussian_filter(
        inputs,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Union[float, Tuple[float, float]] = 1.0,
        strides: Tuple[int, int] = (1, 1),
        padding: str = "same",
        data_format: Optional[str] = None,
        name: Optional[str] = None
):
    """Functional interface for Gaussian filtering.

    This is a convenience function that applies a Gaussian filter to the input.

    Args:
        inputs: Input tensor of shape:
            - If data_format="channels_last": (batch_size, height, width, channels)
            - If data_format="channels_first": (batch_size, channels, height, width)
        kernel_size: Tuple of two integers specifying the height and width of the
            2D Gaussian kernel. Defaults to (5, 5).
        sigma: Standard deviation of the Gaussian distribution. If a single value,
            the same sigma is used for both dimensions. If a tuple, (sigma_height,
            sigma_width) are used. Defaults to 1.0.
        strides: Tuple of 2 integers specifying the strides of the convolution
            along height and width. Defaults to (1, 1).
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "same".
        data_format: String, either "channels_last" or "channels_first". If None,
            uses the default format from Keras configuration.
        name: Optional name for the operation.

    Returns:
        Filtered tensor with the same structure as the input tensor.

    Example:
        >>> x = np.random.rand(4, 32, 32, 3)  # Input images
        >>> y = gaussian_filter(x, kernel_size=(5, 5), sigma=1.5)
        >>> print(y.shape)
        (4, 32, 32, 3)
    """
    layer = GaussianFilter(
        kernel_size=kernel_size,
        strides=strides,
        sigma=sigma,
        padding=padding,
        data_format=data_format,
        name=name
    )
    return layer(inputs)

# ---------------------------------------------------------------------
