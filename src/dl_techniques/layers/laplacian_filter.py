import keras
import numpy as np
import tensorflow as tf
from typing import Tuple, Union, List, Optional, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .gaussian_filter import GaussianFilter

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class LaplacianFilter(keras.layers.Layer):
    """
    Laplacian filter layer that detects edges by approximating the second derivative.

    This filter highlights areas of rapid intensity change in an image and is
    commonly used for edge detection. It works by applying a Gaussian blur
    and then computing the difference between the original image and the blurred image.

    The implementation uses a difference of Gaussians (DoG) approach, which
    is a common approximation of the Laplacian of Gaussian.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            sigma: Union[float, Tuple[float, float]] = 1.0,
            scale_factor: float = 1.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        """
        Initialize the LaplacianFilter layer.

        Args:
            kernel_size: Tuple of two integers specifying the height and width of the 2D kernel.
            strides: Tuple of two integers specifying the strides of the convolution.
            sigma: Standard deviation for the Gaussian kernel. If float, same sigma is used for
                  both dimensions. If tuple, (sigma_height, sigma_width) are used.
            scale_factor: Scaling factor for the Laplacian response.
            kernel_initializer: Initializer for the kernel weights.
            kernel_regularizer: Regularizer for the kernel weights.
            **kwargs: Additional keyword arguments passed to the parent class constructor.
        """
        super().__init__(**kwargs)

        if len(kernel_size) != 2:
            raise ValueError("kernel size must be length 2")

        if len(strides) == 2:
            strides = [1] + list(strides) + [1]

        self._kernel_size = kernel_size
        self._strides = strides
        self._scale_factor = scale_factor
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._kernel_regularizer = kernel_regularizer

        # Handle sigma parameter
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

        # Initialize the Gaussian filter
        self.gaussian_filter = None

    def build(self, input_shape: tf.TensorShape):
        """
        Build the layer and initialize the Gaussian filter.

        Args:
            input_shape: TensorShape of the input tensor.
        """
        # Create a Gaussian filter for blurring
        self.gaussian_filter = GaussianFilter(
            kernel_size=self._kernel_size,
            strides=self._strides[1:3],  # Only pass height and width strides
            sigma=self._sigma
        )

        # Build the Gaussian filter
        self.gaussian_filter.build(input_shape)

        self.built = True

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        """
        Apply the Laplacian filter to the input tensor.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating whether the layer is in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor with highlighted edges, same shape as input tensor.
        """
        # Apply Gaussian blur
        blurred = self.gaussian_filter(inputs, training=training)

        # Compute Laplacian as difference between original and blurred image
        # Multiplying by scale_factor to control the strength of the edge detection
        laplacian = self._scale_factor * (inputs - blurred)

        return laplacian

    def get_config(self):
        """
        Return the config dictionary for the layer.

        Returns:
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update({
            "kernel_size": self._kernel_size,
            "strides": self._strides[1:3],  # Only return the height and width strides
            "sigma": self._sigma,
            "scale_factor": self._scale_factor,
            "kernel_initializer": keras.initializers.serialize(self._kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self._kernel_regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a layer from its config.

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
            A new instance of LaplacianFilter configured from the input dictionary.
        """
        return cls(**config)


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class AdvancedLaplacianFilter(keras.layers.Layer):
    """
    Advanced Laplacian filter with multiple implementation options.

    This layer offers different methods to compute the Laplacian:
    - 'dog': Difference of Gaussians (subtracting a blurred image from the original)
    - 'log': Laplacian of Gaussian (applying a LoG kernel directly)
    - 'kernel': Using a discrete Laplacian kernel

    Each method has different characteristics and performance implications.
    """

    def __init__(
            self,
            method: str = 'dog',
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            sigma: Union[float, Tuple[float, float]] = 1.0,
            scale_factor: float = 1.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        """
        Initialize the AdvancedLaplacianFilter layer.

        Args:
            method: The method to use ('dog', 'log', or 'kernel').
            kernel_size: Tuple of two integers specifying the height and width of the 2D kernel.
            strides: Tuple of two integers specifying the strides of the convolution.
            sigma: Standard deviation for the Gaussian kernel.
            scale_factor: Scaling factor for the Laplacian response.
            kernel_initializer: Initializer for the kernel weights.
            kernel_regularizer: Regularizer for the kernel weights.
            **kwargs: Additional keyword arguments passed to the parent class constructor.
        """
        super().__init__(**kwargs)

        if method not in ['dog', 'log', 'kernel']:
            raise ValueError(f"Method '{method}' not supported. Use 'dog', 'log', or 'kernel'.")

        self._method = method
        self._kernel_size = kernel_size
        self._strides = [1] + list(strides) + [1] if len(strides) == 2 else strides
        self._scale_factor = scale_factor
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._kernel_regularizer = kernel_regularizer

        # Handle sigma parameter
        if isinstance(sigma, (int, float)):
            self._sigma = (float(sigma), float(sigma))
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            self._sigma = (float(sigma[0]), float(sigma[1]))
        else:
            raise ValueError(f"Invalid sigma value: {sigma}")

        # Initialize layer components
        self.gaussian_filter = None
        self._kernel = None

    def _create_laplacian_kernel(self, channels: int) -> tf.Tensor:
        """
        Create a discrete Laplacian kernel.

        Args:
            channels: Number of input channels.

        Returns:
            Laplacian kernel tensor.
        """
        # Simple discrete Laplacian kernel
        if self._kernel_size == (3, 3):
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
        kernel = np.zeros((*self._kernel_size, channels, 1), dtype=np.float32)
        for i in range(channels):
            kernel[:, :, i, 0] = kernel_2d

        return tf.constant(kernel, dtype=self.compute_dtype)

    def _create_log_kernel(self) -> np.ndarray:
        """
        Create a Laplacian of Gaussian (LoG) kernel.

        Returns:
            LoG kernel as numpy array.
        """
        sigma_x, sigma_y = self._sigma
        height, width = self._kernel_size

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

    def build(self, input_shape: tf.TensorShape):
        """
        Build the layer and initialize components based on the selected method.

        Args:
            input_shape: TensorShape of the input tensor.
        """
        channels = input_shape[-1]

        if self._method == 'dog':
            # Create a Gaussian filter for the DoG approach
            self.gaussian_filter = GaussianFilter(
                kernel_size=self._kernel_size,
                strides=self._strides[1:3],
                sigma=self._sigma
            )
            self.gaussian_filter.build(input_shape)
        elif self._method == 'log':
            # Create a LoG kernel
            self._kernel = tf.constant(
                self._create_log_kernel().reshape(*self._kernel_size, 1, 1),
                dtype=self.compute_dtype
            )
            # Repeat for each channel
            self._kernel = tf.tile(self._kernel, [1, 1, channels, 1])
        else:  # 'kernel'
            # Create a discrete Laplacian kernel
            self._kernel = self._create_laplacian_kernel(channels)

        self.built = True

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        """
        Apply the Laplacian filter to the input tensor.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean indicating whether the layer is in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor with highlighted edges, same shape as input tensor.
        """
        if self._method == 'dog':
            # Difference of Gaussians approach
            blurred = self.gaussian_filter(inputs, training=training)
            return self._scale_factor * (inputs - blurred)
        else:
            # Direct convolution with LoG or Laplacian kernel
            return self._scale_factor * tf.nn.depthwise_conv2d(
                input=inputs,
                filter=self._kernel,
                strides=self._strides,
                padding="SAME"
            )

    def get_config(self):
        """
        Return the config dictionary for the layer.

        Returns:
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update({
            "method": self._method,
            "kernel_size": self._kernel_size,
            "strides": self._strides[1:3],
            "sigma": self._sigma,
            "scale_factor": self._scale_factor,
            "kernel_initializer": keras.initializers.serialize(self._kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self._kernel_regularizer)
        })
        return config

# ---------------------------------------------------------------------
