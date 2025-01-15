import keras
import tensorflow as tf

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SpatialLayer(keras.layers.Layer):
    def __init__(self, resolution=(4, 4), resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, **kwargs):
        super(SpatialLayer, self).__init__(trainable=False, **kwargs)
        self.xy_grid = None
        self.resolution = resolution
        self.resize_method = resize_method

    def build(self, input_shape):
        x_grid = tf.linspace(start=-0.5, stop=+0.5, num=self.resolution[0])
        y_grid = tf.linspace(start=-0.5, stop=+0.5, num=self.resolution[1])

        # create the meshgrid
        xx_grid, yy_grid = tf.meshgrid(x_grid, y_grid)

        # Normalize the grids
        xx_grid = (xx_grid - tf.reduce_mean(xx_grid)) / (tf.math.reduce_std(xx_grid) + 1e-7)
        yy_grid = (yy_grid - tf.reduce_mean(yy_grid)) / (tf.math.reduce_std(yy_grid) + 1e-7)

        tf.debugging.assert_rank(x=xx_grid, rank=2)
        tf.debugging.assert_rank(x=yy_grid, rank=2)

        # Prepare grids for later use
        xx_grid = tf.expand_dims(xx_grid, axis=2)
        yy_grid = tf.expand_dims(yy_grid, axis=2)
        self.xy_grid = tf.concat([xx_grid, yy_grid], axis=2)
        self.xy_grid = tf.expand_dims(self.xy_grid, axis=0)

        tf.debugging.assert_rank(x=self.xy_grid, rank=4)

        super(SpatialLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        # resize
        xy_grid = (
            tf.image.resize(
                images=self.xy_grid,
                size=(height, width),
                method=self.resize_method)
        )

        # repeat grids to match batch size
        xy_grid_batched = tf.repeat(xy_grid, axis=0, repeats=batch_size)

        return xy_grid_batched

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + 2,)

    def get_config(self):
        base_config = super(SpatialLayer, self).get_config()
        base_config.update({'resolution': self.resolution})
        base_config.update({'resize_method': self.resize_method})
        return base_config

# ---------------------------------------------------------------------
