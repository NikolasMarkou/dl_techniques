"""
Convolutional Kolmogorov-Arnold Networks Implementation
====================================================

This module implements Convolutional Kolmogorov-Arnold Networks (KANs) as described
in the paper by Bodner et al. (2024). KANs integrate the principles of
Kolmogorov-Arnold representation into convolutional layers, using learnable
non-linear functions based on B-splines.

Classes
-------
    KANConvolution
        A custom Keras layer implementing the KAN convolution operation
    ConvolutionalKAN
        A complete model architecture using KAN convolution layers

References
----------
    [1] Bodner et al. (2024). "Convolutional Kolmogorov-Arnold Networks"
    [2] Kolmogorov, A. N. (1957). "On the representation of continuous functions"
"""

import keras
import tensorflow as tf
from typing import Tuple, Optional, Union, List

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class KANConvolution(keras.layers.Layer):
    """
    Implements a Kolmogorov-Arnold Network convolution layer.

    This layer replaces traditional convolution operations with learnable non-linear
    functions based on B-splines for each element of the convolutional kernel.

    Parameters
    ----------
    filters : int
        Number of output filters in the convolution
    kernel_size : Union[int, Tuple[int, int]]
        Size of the convolution kernel
    grid_size : int, optional
        Number of control points for the spline interpolation (default: 16)
    strides : Union[int, Tuple[int, int]], optional
        Stride length of the convolution (default: (1, 1))
    padding : str, optional
        One of 'valid' or 'same' (default: 'same')
    kernel_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the kernel weights (default: 'glorot_uniform')
    kernel_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer function for the kernel weights (default: None)

    Attributes
    ----------
    control_points : tf.Variable
        Learnable control points for the spline interpolation
    w1 : tf.Variable
        Weights for combining spline output
    w2 : tf.Variable
        Weights for combining SiLU activation
    grid : tf.Tensor
        Fixed grid points for spline interpolation
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]],
            grid_size: int = 16,
            strides: Union[int, Tuple[int, int]] = (1, 1),
            padding: str = 'same',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        """Initialize the KANConvolution layer."""
        super(KANConvolution, self).__init__(**kwargs)

        # Convert kernel_size to tuple if integer
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.filters = filters
        self.grid_size = grid_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.padding = padding.upper()
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape: Tuple) -> None:
        """
        Build the layer's weights.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor
        """
        input_channels = input_shape[-1]

        # Initialize spline control points
        self.control_points = self.add_weight(
            shape=(self.filters, input_channels, *self.kernel_size, self.grid_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='control_points'
        )

        # Initialize combination weights
        self.w1 = self.add_weight(
            shape=(self.filters, input_channels, *self.kernel_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='w1'
        )

        self.w2 = self.add_weight(
            shape=(self.filters, input_channels, *self.kernel_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='w2'
        )

        # Initialize fixed grid
        self.grid = tf.linspace(-1.0, 1.0, self.grid_size)

        self.built = True

    def _apply_spline(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply spline interpolation to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor

        Returns
        -------
        tf.Tensor
            Interpolated values
        """
        # Clip input values to grid range
        x_clipped = tf.clip_by_value(x, -1.0, 1.0)

        # Find nearest grid points
        idx = tf.searchsorted(self.grid, x_clipped) - 1
        idx = tf.clip_by_value(idx, 0, self.grid_size - 2)

        # Calculate interpolation weights
        grid_diff = self.grid[idx + 1] - self.grid[idx]
        frac = (x_clipped - self.grid[idx]) / grid_diff

        # Get control points and interpolate
        y0 = tf.gather(self.control_points, idx, axis=-1)
        y1 = tf.gather(self.control_points, idx + 1, axis=-1)

        return y0 + frac[..., tf.newaxis] * (y1 - y0)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor
        training : bool, optional
            Whether the layer is in training mode

        Returns
        -------
        tf.Tensor
            Output tensor after KAN convolution
        """

        def kan_conv(x: tf.Tensor) -> tf.Tensor:
            """Apply KAN convolution operation to input."""
            spline_output = self._apply_spline(x)
            return self.w1 * spline_output + self.w2 * tf.nn.silu(x)

        outputs = tf.nn.convolution(
            inputs,
            kan_conv,
            strides=self.strides,
            padding=self.padding
        )

        return outputs

    def get_config(self) -> dict:
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Layer configuration
        """
        config = super(KANConvolution, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'grid_size': self.grid_size,
            'strides': self.strides,
            'padding': self.padding.lower(),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

# ---------------------------------------------------------------------


class ConvolutionalKAN(keras.Model):
    """
    Complete Convolutional KAN model architecture.

    This model combines KANConvolution layers with batch normalization and pooling
    for image classification tasks.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes (default: 10)
    """

    def __init__(self,
                 num_classes: int = 10,
                 kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
                 kernel_regularizer: Union[str, keras.regularizers.Regularizer] = "l2") -> None:
        """Initialize the ConvolutionalKAN model."""
        super(ConvolutionalKAN, self).__init__()

        # First KAN block
        self.conv1 = KANConvolution(
            filters=32,
            kernel_size=3,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.bn1 = keras.layers.BatchNormalization()

        # Second KAN block
        self.conv2 = KANConvolution(
            filters=64,
            kernel_size=3,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.bn2 = keras.layers.BatchNormalization()

        # Fully connected layers
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(
            128,
            activation='relu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.dense2 = keras.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='he_normal'
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor
        training : bool, optional
            Whether the model is in training mode

        Returns
        -------
        tf.Tensor
            Model predictions
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.max_pool2d(x, 2, 2, padding='SAME')

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.max_pool2d(x, 2, 2, padding='SAME')

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# ---------------------------------------------------------------------
