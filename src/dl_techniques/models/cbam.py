import keras
from typing import Optional, Union, Tuple
from keras.api import layers, regularizers, initializers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..layers.attention.convolutional_block_attention import CBAM

# ---------------------------------------------------------------------

def create_cbam_model(
        input_shape: Tuple[int, int, int],
        num_classes: int,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[regularizers.Regularizer] = None
) -> keras.Model:
    """Create a CNN model with CBAM modules.

    Args:
        input_shape: Shape of the input images (height, width, channels).
        num_classes: Number of output classes.
        kernel_initializer: Initializer for all kernels in the model.
        kernel_regularizer: Regularizer for all kernels in the model.

    Returns:
        A Keras model with CBAM attention modules.
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        64, 3,
        activation='relu',
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = CBAM(
        64,
        channel_kernel_initializer=kernel_initializer,
        spatial_kernel_initializer=kernel_initializer,
        channel_kernel_regularizer=kernel_regularizer,
        spatial_kernel_regularizer=kernel_regularizer
    )(x)

    x = layers.Conv2D(
        128, 3,
        activation='relu',
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(x)
    x = layers.BatchNormalization()(x)
    x = CBAM(
        128,
        channel_kernel_initializer=kernel_initializer,
        spatial_kernel_initializer=kernel_initializer,
        channel_kernel_regularizer=kernel_regularizer,
        spatial_kernel_regularizer=kernel_regularizer
    )(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(x)

    return keras.Model(inputs, outputs)

# ---------------------------------------------------------------------
