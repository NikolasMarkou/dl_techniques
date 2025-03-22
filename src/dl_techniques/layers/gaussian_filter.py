import keras
import numpy as np
import tensorflow as tf
from collections.abc import Sequence
from typing import Tuple, Union, List, Optional

# ---------------------------------------------------------------------

from dl_techniques.utils.tensors import depthwise_gaussian_kernel


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class GaussianFilter(keras.layers.Layer):
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            sigma: Union[float, Tuple[float, float]] = -1,
            **kwargs):
        """
        Initialize the GaussianFilter layer.

        Args:
            kernel_size: Tuple of two integers specifying the height and width of the 2D gaussian kernel.
            strides: Tuple of two integers specifying the strides of the convolution along the height and width.
            sigma: sigma of the gaussian
            **kwargs: Additional keyword arguments passed to the parent class constructor.
        """
        super().__init__(**kwargs)
        if len(kernel_size) != 2:
            raise ValueError("kernel size must be length 2")
        if len(strides) == 2:
            strides = [1] + list(strides) + [1]
        self._kernel_size = kernel_size
        self._strides = strides
        self._kernel = None
        if (sigma is None or
                (isinstance(sigma, float) and sigma <= 0.0) or
                (isinstance(sigma, int) and sigma <= 0)):
            self._sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
        elif isinstance(sigma, Sequence):
            self._sigma = (float(sigma[0]), float(sigma[1]))
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = (float(sigma), float(sigma))
        else:
            raise ValueError(f"don't know how to handle sigma [{sigma}]")

    def build(self, input_shape: tf.TensorShape):
        """
        Build the layer and create the gaussian kernel.

        Args:
            input_shape: TensorShape of the input tensor.
        """
        self._kernel = depthwise_gaussian_kernel(
            channels=input_shape[-1],
            kernel_size=self._kernel_size,
            nsig=self._sigma,
            dtype=np.float32
        )
        self._kernel = tf.constant(self._kernel, dtype=self.compute_dtype)

    def call(self, inputs, training=False, **kwargs):
        """
        Apply the gaussian filter to the input tensor.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating whether the layer is in training mode (unused in this layer).

        Returns:
            Filtered tensor of the same shape as the input tensor.
        """
        return tf.nn.depthwise_conv2d(
            input=inputs,
            filter=self._kernel,
            strides=self._strides,
            padding="SAME"
        )

    def get_config(self):
        """Return the config dictionary for the layer."""
        tmp_config = super().get_config()
        tmp_config.update({
            "kernel_size": self._kernel_size,
            "strides": self._strides[1:3],  # Only return the height and width strides
            "sigma": self._sigma
        })
        return tmp_config


# ---------------------------------------------------------------------

def create_gaussian_kernel(
        kernel_size: Tuple[int, int],
        sigma: Union[float, Tuple[float, float]],
        channels: int,
        dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """
    Creates a 2D Gaussian kernel for multiple channels.

    Args:
        kernel_size: Tuple[int, int]
            The size of the kernel in (height, width) format.
        sigma: Union[float, Tuple[float, float]]
            The standard deviation for the Gaussian kernel.
            If float, same sigma is used for both dimensions.
            If tuple, (sigma_height, sigma_width) are used.
        channels: int
            Number of input channels.
        dtype: tf.DType, optional
            Data type of the output kernel, defaults to tf.float32.

    Returns:
        tf.Tensor: A 4D tensor of shape (kernel_height, kernel_width, channels, 1)
            representing the depthwise Gaussian kernel.

    Raises:
        ValueError: If kernel_size is not a tuple of length 2.
        ValueError: If sigma is negative or invalid type.
    """
    if len(kernel_size) != 2:
        raise ValueError("kernel_size must be a tuple of length 2")

    if isinstance(sigma, (int, float)):
        sigma = (float(sigma), float(sigma))
    elif isinstance(sigma, (tuple, list)) and len(sigma) == 2:
        sigma = (float(sigma[0]), float(sigma[1]))
    else:
        raise ValueError(f"Invalid sigma value: {sigma}")

    if sigma[0] <= 0 or sigma[1] <= 0:
        raise ValueError("sigma values must be positive")

    # Create meshgrid for kernel computation
    x = tf.range(-(kernel_size[1] - 1) / 2, (kernel_size[1] + 1) / 2)
    y = tf.range(-(kernel_size[0] - 1) / 2, (kernel_size[0] + 1) / 2)
    y_grid, x_grid = tf.meshgrid(y, x)

    # Compute 2D Gaussian kernel
    gaussian = tf.exp(-(
            tf.square(x_grid) / (2 * tf.square(sigma[1])) +
            tf.square(y_grid) / (2 * tf.square(sigma[0]))
    ))

    # Normalize the kernel
    gaussian = gaussian / tf.reduce_sum(gaussian)

    # Expand dimensions for channels
    gaussian = tf.expand_dims(gaussian, axis=-1)
    gaussian = tf.expand_dims(gaussian, axis=-1)

    # Tile for multiple channels
    kernel = tf.tile(gaussian, [1, 1, channels, 1])

    return tf.cast(kernel, dtype)

# ---------------------------------------------------------------------


@tf.function
def gaussian_filter(
        inputs: tf.Tensor,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Union[float, Tuple[float, float]] = 1.0,
        strides: Tuple[int, int] = (1, 1),
        padding: str = 'SAME',
        name: Optional[str] = None
) -> tf.Tensor:
    """
    Applies Gaussian filtering to the input tensor.

    Args:
        inputs: tf.Tensor
            Input tensor of shape [batch_size, height, width, channels].
        kernel_size: Tuple[int, int], optional
            Size of the Gaussian kernel (height, width), defaults to (5, 5).
        sigma: Union[float, Tuple[float, float]], optional
            Standard deviation for Gaussian kernel, defaults to 1.0.
        strides: Tuple[int, int], optional
            Stride of the sliding window for each dimension, defaults to (1, 1).
        padding: str, optional
            The type of padding algorithm, either 'SAME' or 'VALID', defaults to 'SAME'.
        name: Optional[str], optional
            Name for the operation, defaults to None.

    Returns:
        tf.Tensor: Filtered tensor of shape [batch_size, height', width', channels],
            where dimensions depend on padding and stride values.

    Raises:
        ValueError: If inputs tensor has incorrect rank or invalid parameters.
    """
    with tf.name_scope(name or "gaussian_filter"):
        inputs = tf.convert_to_tensor(inputs)

        if inputs.shape.rank != 4:
            raise ValueError(
                f"Expected input tensor of rank 4, got shape {inputs.shape}"
            )

        # Create Gaussian kernel
        kernel = create_gaussian_kernel(
            kernel_size=kernel_size,
            sigma=sigma,
            channels=inputs.shape[-1],
            dtype=inputs.dtype
        )

        # Apply depthwise convolution
        return tf.nn.depthwise_conv2d(
            input=inputs,
            filter=kernel,
            strides=[1, strides[0], strides[1], 1],
            padding=padding
        )

# ---------------------------------------------------------------------

