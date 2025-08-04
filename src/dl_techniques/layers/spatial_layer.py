"""
This module provides a `SpatialLayer`, a custom Keras layer that dynamically
generates spatial coordinate grids and injects them into a model.

In many computer vision tasks, particularly those involving complex spatial reasoning
or generative models, it is beneficial for the network to have explicit knowledge of
pixel locations. Standard convolutional networks build this understanding implicitly
through their local receptive fields, but providing coordinate information directly
can enhance performance. This layer serves as a "coordinate encoder," generating a
normalized `(x, y)` coordinate for every pixel position.

The layer is non-trainable; it does not learn any parameters from data. Instead, it
functions as a pre-defined feature generator that operates in two main stages:

1.  **Grid Creation (`build` phase):**
    -   A low-resolution coordinate grid is created once when the layer is built.
    -   This grid represents normalized `x` and `y` coordinates, typically ranging
        from -0.5 to 0.5.
    -   Crucially, these coordinates are then standardized (normalized to have zero
        mean and unit standard deviation). This ensures that the coordinate features
        are on a similar scale to the activations of other layers, promoting stable
        training.

2.  **Dynamic Resizing (`call` phase):**
    -   During the forward pass, the layer takes an input tensor (e.g., a feature map
        from a preceding layer).
    -   It dynamically resizes its internal, low-resolution coordinate grid to match
        the spatial dimensions (height and width) of the input tensor. This is done
        using a specified interpolation method (`'nearest'` or `'bilinear'`).
    -   The resized grid is then tiled to match the batch size of the input, producing
        a final output tensor of shape `(batch_size, height, width, 2)`.

This output tensor can then be concatenated with the original feature map, providing
every subsequent layer with explicit information about the absolute position of each
feature vector in the grid. This is a common technique in architectures like
CoordConv and is also used in various generative models and attention mechanisms.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Any

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SpatialLayer(keras.layers.Layer):
    """A custom Keras layer that generates and resizes spatial coordinate grids.

    This layer creates normalized coordinate grids (x, y) and resizes them to match
    the input tensor dimensions. The coordinates range from -0.5 to +0.5 and are
    normalized to have zero mean and unit standard deviation.

    Args:
        resolution: Tuple of integers specifying the initial grid resolution (height, width).
            Defaults to (4, 4).
        resize_method: String specifying the interpolation method for resizing.
            Options: 'nearest', 'bilinear'. Defaults to 'nearest'.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        4D tensor with shape: (batch_size, height, width, channels)

    Output shape:
        4D tensor with shape: (batch_size, height, width, 2)
        The last dimension contains the (x, y) coordinates.

    Example:
        >>> spatial_layer = SpatialLayer(resolution=(8, 8), resize_method='bilinear')
        >>> input_tensor = keras.ops.ones((2, 64, 64, 3))
        >>> coords = spatial_layer(input_tensor)
        >>> print(coords.shape)  # (2, 64, 64, 2)
    """

    def __init__(
            self,
            resolution: Tuple[int, int] = (4, 4),
            resize_method: str = 'nearest',
            **kwargs: Any
    ) -> None:
        super().__init__(trainable=False, **kwargs)
        self.resolution = resolution
        self.resize_method = resize_method
        self.xy_grid = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Creates the coordinate grid during layer building.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Create coordinate grids using keras.ops
        x_grid = ops.linspace(
            start=-0.5,
            stop=0.5,
            num=self.resolution[0]
        )
        y_grid = ops.linspace(
            start=-0.5,
            stop=0.5,
            num=self.resolution[1]
        )

        # Create meshgrid
        xx_grid, yy_grid = ops.meshgrid(x_grid, y_grid)

        # Normalize the grids to have zero mean and unit standard deviation
        xx_grid = (xx_grid - ops.mean(xx_grid)) / (ops.std(xx_grid) + 1e-7)
        yy_grid = (yy_grid - ops.mean(yy_grid)) / (ops.std(yy_grid) + 1e-7)

        # Prepare grids for concatenation
        xx_grid = ops.expand_dims(xx_grid, axis=2)
        yy_grid = ops.expand_dims(yy_grid, axis=2)

        # Combine x and y grids
        self.xy_grid = ops.concatenate([xx_grid, yy_grid], axis=2)

        # Add batch dimension
        self.xy_grid = ops.expand_dims(self.xy_grid, axis=0)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Coordinate grid tensor with shape (batch_size, height, width, 2).
        """
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        # Resize the coordinate grid to match input spatial dimensions
        xy_grid_resized = ops.image.resize(
            image=self.xy_grid,
            size=(height, width),
            interpolation=self.resize_method,
            data_format='channels_last'
        )

        # Repeat grid to match batch size
        xy_grid_batched = ops.repeat(
            xy_grid_resized,
            repeats=batch_size,
            axis=0
        )

        return xy_grid_batched

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        return input_shape[:-1] + (2,)

    def get_config(self) -> dict:
        """Returns the configuration of the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'resolution': self.resolution,
            'resize_method': self.resize_method
        })
        return config

# ---------------------------------------------------------------------

