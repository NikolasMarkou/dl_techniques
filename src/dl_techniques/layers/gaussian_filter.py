import keras
import numpy as np
import tensorflow as tf
from collections.abc import Sequence
from typing import Tuple, Union, List, Optional


# ---------------------------------------------------------------------

def gaussian_kernel(
        kernel_size: Tuple[int, int],
        nsig: Tuple[float, float]
) -> np.ndarray:
    """
    Build a 2D Gaussian kernel array.

    Args:
        kernel_size (Tuple[int, int]): Size of the grid (height, width).
        nsig (Tuple[float, float]): Standard deviation for x and y dimensions.

    Returns:
        np.ndarray: 2D Gaussian kernel.
    """
    if len(nsig) != 2 or len(kernel_size) != 2:
        raise ValueError("Both kernel_size and nsig must be tuples of length 2.")

    x = np.linspace(-nsig[0], nsig[0], kernel_size[0])
    y = np.linspace(-nsig[1], nsig[1], kernel_size[1])
    x, y = np.meshgrid(x, y)

    kernel = np.exp(-(x ** 2 + y ** 2) / 2)
    return kernel / np.sum(kernel)


# ---------------------------------------------------------------------


def depthwise_gaussian_kernel(
        channels: int = 3,
        kernel_size: Tuple[int, int] = (5, 5),
        nsig: Tuple[float, float] = (2.0, 2.0),
        dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    Create a depthwise Gaussian kernel.

    Args:
        channels (int): Number of input channels.
        kernel_size (Tuple[int, int]): Size of the kernel (height, width).
        nsig (Tuple[float, float]): Standard deviation for x and y dimensions.
        dtype (Optional[np.dtype]): Data type of the output kernel.

    Returns:
        np.ndarray: Depthwise Gaussian kernel of shape (kernel_height, kernel_width, in_channels, 1).
    """
    # Generate the 2D Gaussian kernel
    kernel_2d = gaussian_kernel(kernel_size, nsig)

    # Create the depthwise kernel
    kernel = np.zeros((*kernel_size, channels, 1))
    for i in range(channels):
        kernel[:, :, i, 0] = kernel_2d

    # Set the data type
    if dtype is not None:
        kernel = kernel.astype(dtype)

    return kernel


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


if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create a sample image
    image = np.random.rand(1, 100, 100, 3).astype(np.float32)

    # Create and apply GaussianFilter with different kernel sizes
    gaussian_filter_3x3 = GaussianFilter(kernel_size=(3, 3))
    gaussian_filter_5x5 = GaussianFilter(kernel_size=(5, 5))
    gaussian_filter_7x7 = GaussianFilter(kernel_size=(7, 7))

    # Apply filters
    filtered_3x3 = gaussian_filter_3x3(image)
    filtered_5x5 = gaussian_filter_5x5(image)
    filtered_7x7 = gaussian_filter_7x7(image)

    # Test serialization and deserialization
    config = gaussian_filter_5x5.get_config()
    reconstructed_layer = GaussianFilter.from_config(config)

    print("Original layer config:", config)
    print("Reconstructed layer config:", reconstructed_layer.get_config())
    print("Configs are identical:", config == reconstructed_layer.get_config())

    # Verify that the reconstructed layer produces the same output
    reconstructed_output = reconstructed_layer(image)
    is_equal = tf.reduce_all(tf.equal(filtered_5x5, reconstructed_output))
    print("Original and reconstructed layer outputs are identical:", is_equal.numpy())

# ---------------------------------------------------------------------
