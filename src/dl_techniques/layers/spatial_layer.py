import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SpatialLayer(keras.layers.Layer):
    """
    A custom Keras layer that generates and resizes spatial coordinate grids.

    This layer creates a normalized meshgrid of spatial coordinates that can be
    resized and batched to match input tensor dimensions.

    Attributes:
        resolution (Tuple[int, int]): The initial resolution of the spatial grid.
        resize_method (tf.image.ResizeMethod): Method used for resizing the grid.
        xy_grid (Optional[tf.Tensor]): Preprocessed spatial coordinate grid.

    Args:
        resolution (Tuple[int, int], optional): Grid resolution. Defaults to (4, 4).
        resize_method (tf.image.ResizeMethod, optional): Resize interpolation method.
            Defaults to NEAREST_NEIGHBOR.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (4, 4),
        resize_method: tf.image.ResizeMethod = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        **kwargs
    ):
        """
        Initialize the SpatialLayer.

        Args:
            resolution (Tuple[int, int], optional): Grid resolution. Defaults to (4, 4).
            resize_method (tf.image.ResizeMethod, optional): Resize interpolation method.
                Defaults to NEAREST_NEIGHBOR.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(trainable=False, **kwargs)
        self.xy_grid: Optional[tf.Tensor] = None
        self.resolution: Tuple[int, int] = resolution
        self.resize_method: tf.image.ResizeMethod = resize_method

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer by creating and normalizing the spatial grid.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.
        """
        # Create coordinate grids
        x_grid = ops.linspace(start=-0.5, stop=0.5, num=self.resolution[0])
        y_grid = ops.linspace(start=-0.5, stop=0.5, num=self.resolution[1])

        # Create meshgrid
        xx_grid, yy_grid = ops.meshgrid(x_grid, y_grid)

        # Normalize the grids
        xx_grid = (xx_grid - ops.mean(xx_grid)) / (ops.std(xx_grid) + 1e-7)
        yy_grid = (yy_grid - ops.mean(yy_grid)) / (ops.std(yy_grid) + 1e-7)

        # Validate grid dimensions
        tf.debugging.assert_rank(x=xx_grid, rank=2)
        tf.debugging.assert_rank(x=yy_grid, rank=2)

        # Prepare grids for later use
        xx_grid = ops.expand_dims(xx_grid, axis=2)
        yy_grid = ops.expand_dims(yy_grid, axis=2)
        self.xy_grid = ops.concatenate([xx_grid, yy_grid], axis=2)
        self.xy_grid = ops.expand_dims(self.xy_grid, axis=0)

        # Validate final grid dimensions
        tf.debugging.assert_rank(x=self.xy_grid, rank=4)

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        **kwargs
    ) -> tf.Tensor:
        """
        Generate resized and batched spatial coordinate grids.

        Args:
            inputs (tf.Tensor): Input tensor to match grid dimensions.
            training (Optional[bool], optional): Training mode flag. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tf.Tensor: Resized and batched spatial coordinate grid.
        """
        shape = ops.shape(inputs)
        batch_size, height, width = shape[0], shape[1], shape[2]

        # Resize grid to match input dimensions
        xy_grid = tf.image.resize(
            images=self.xy_grid,
            size=(height, width),
            method=self.resize_method
        )

        # Repeat grids to match batch size
        xy_grid_batched = ops.repeat(xy_grid, axis=0, repeats=batch_size)

        return xy_grid_batched

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Compute the output shape of the layer.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.

        Returns:
            tf.TensorShape: Shape of the output tensor.
        """
        return input_shape[:-1] + (input_shape[-1] + 2,)

    def get_config(self) -> dict:
        """
        Get the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary containing layer parameters.
        """
        base_config = super().get_config()
        base_config.update({
            'resolution': self.resolution,
            'resize_method': self.resize_method
        })
        return base_config

# ---------------------------------------------------------------------
