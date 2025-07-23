"""
Bias-Free CNN Denoiser Model

Implements a ResNet-style denoising CNN where all additive constants (bias terms)
have been removed to enable better generalization across different noise levels.

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020).
"""

import keras
from keras import layers
from typing import Optional, Union, Tuple


from dl_techniques.utils.logger import logger
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock

def create_bfcnn_denoiser(
        input_shape: Tuple[int, int, int],
        num_blocks: int = 8,
        filters: int = 64,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        activation: Union[str, callable] = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        model_name: str = 'bfcnn_denoiser'
) -> keras.Model:
    """
    Create a bias-free CNN model for image denoising using ResNet architecture.

    This function creates a complete Keras model using bias-free residual blocks.
    The model implements the scaling-invariant property described in the paper:
    if you scale the input by α, the output is scaled by α as well.

    Architecture:
    - Initial bias-free convolution
    - Multiple bias-free residual blocks
    - Final bias-free convolution to output channels

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        num_blocks: Integer, number of residual blocks. Defaults to 8.
        filters: Integer, number of filters in residual blocks. Defaults to 64.
        kernel_size: Integer or tuple, size of convolutional kernels. Defaults to 3.
        activation: String or callable, activation function. Defaults to 'relu'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'glorot_uniform'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        model_name: String, name for the model. Defaults to 'bfcnn_denoiser'.

    Returns:
        keras.Model: Compiled Keras model ready for training.

    Example:
        >>> # Create model for grayscale images
        >>> model = create_bfcnn_denoiser(
        ...     input_shape=(None, None, 1),
        ...     num_blocks=10,
        ...     filters=64
        ... )
        >>> model.compile(optimizer='adam', loss='mse', metrics=['psnr'])
        >>>
        >>> # The model exhibits scaling invariance
        >>> # If input is scaled by α, output is also scaled by α
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_images')

    # Initial convolution to project to feature space
    x = BiasFreeConv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,  # First layer typically no batch norm
        name='initial_conv'
    )(inputs)

    # Stack of bias-free residual blocks
    for i in range(num_blocks):
        x = BiasFreeResidualBlock(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f'residual_block_{i}'
        )(x)

    # Final convolution to output channels (no activation, no batch norm)
    outputs = layers.Conv2D(
        filters=input_shape[-1],  # Match input channels
        kernel_size=kernel_size,
        padding='same',
        use_bias=False,  # Key: no bias for scaling invariance
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name='final_conv'
    )(x)

    # Create the model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name
    )

    logger.info(f"Created bias-free CNN model with {num_blocks} residual blocks and {filters} filters")

    return model


# Predefined model configurations
def create_bfcnn_light(input_shape: Tuple[int, int, int]) -> keras.Model:
    """Create a lightweight bias-free CNN (fewer blocks, good for quick experiments)."""
    return create_bfcnn_denoiser(
        input_shape=input_shape,
        num_blocks=4,
        filters=32,
        model_name='bfcnn_light'
    )


def create_bfcnn_standard(input_shape: Tuple[int, int, int]) -> keras.Model:
    """Create a standard bias-free CNN (balanced performance and speed)."""
    return create_bfcnn_denoiser(
        input_shape=input_shape,
        num_blocks=8,
        filters=64,
        model_name='bfcnn_standard'
    )


def create_bfcnn_deep(input_shape: Tuple[int, int, int]) -> keras.Model:
    """Create a deep bias-free CNN (more blocks for complex denoising)."""
    return create_bfcnn_denoiser(
        input_shape=input_shape,
        num_blocks=16,
        filters=128,
        model_name='bfcnn_deep'
    )