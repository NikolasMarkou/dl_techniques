"""
Bias-Free CNN Denoiser Model with Variants

Implements a ResNet-style denoising CNN where all additive constants (bias terms)
have been removed to enable better generalization across different noise levels.
Provides multiple model variants (tiny, small, base, large, xlarge) for different
computational requirements and performance targets.

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020).
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock

# ---------------------------------------------------------------------
# Model Variant Configurations
# ---------------------------------------------------------------------

BFCNN_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'num_blocks': 2,
        'filters': 32,
        'description': 'Tiny BFCNN (~ResNet-5) for quick experiments and resource-constrained environments'
    },
    'small': {
        'num_blocks': 5,
        'filters': 48,
        'description': 'Small BFCNN (~ResNet-10) with minimal capacity'
    },
    'base': {
        'num_blocks': 12,
        'filters': 64,
        'description': 'Base BFCNN (~ResNet-25) with standard configuration'
    },
    'large': {
        'num_blocks': 25,
        'filters': 96,
        'description': 'Large BFCNN (~ResNet-50) with high capacity'
    },
    'xlarge': {
        'num_blocks': 50,
        'filters': 128,
        'description': 'Extra-Large BFCNN (~ResNet-100) for maximum performance'
    }
}

# ---------------------------------------------------------------------
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_bfcnn_denoiser(
        input_shape: Tuple[int, int, int],
        num_blocks: int = 8,
        filters: int = 64,
        initial_kernel_size: Union[int, Tuple[int, int]] = 5,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        activation: Union[str, callable] = 'relu',
        final_activation: Union[str, callable] = 'linear',
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
        initial_kernel_size: Integer or tuple, size of the first convolutional kernels. Defaults to 5.
        kernel_size: Integer or tuple, size of convolutional kernels. Defaults to 3.
        activation: String or callable, activation function. Defaults to 'relu'.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'glorot_uniform'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        model_name: String, name for the model. Defaults to 'bfcnn_denoiser'.

    Returns:
        keras.Model: Compiled Keras model ready for training.

    Raises:
        ValueError: If num_blocks is negative or filters is zero or negative.
        TypeError: If input_shape is not a tuple of 3 integers.

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

    # Input validation
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise TypeError("input_shape must be a tuple of 3 integers (height, width, channels)")

    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")

    if filters <= 0:
        raise ValueError(f"filters must be positive, got {filters}")

    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_images')

    # Initial convolution to project to feature space
    x = BiasFreeConv2D(
        filters=filters,
        kernel_size=initial_kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,  # First layer typically no batch norm
        name='stem'
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
    # Output same number of channels as input
    output_channels = input_shape[-1]
    outputs = BiasFreeConv2D(
        filters=output_channels,
        kernel_size=1,
        activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,  # Last layer typically no batch norm
        name='final_conv'
    )(x)

    # Create the model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name
    )

    logger.info(f"Created bias-free CNN model '{model_name}' with {num_blocks} residual blocks and {filters} filters")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")

    return model

# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_bfcnn_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        **kwargs
) -> keras.Model:
    """
    Create a BFCNN model with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        **kwargs: Additional keyword arguments to override default parameters.

    Returns:
        keras.Model: BFCNN model with the specified variant configuration.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        >>> model = create_bfcnn_variant('base', (256, 256, 3))
        >>> model.summary()
    """
    if variant not in BFCNN_CONFIGS:
        available_variants = list(BFCNN_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = BFCNN_CONFIGS[variant].copy()
    description = config.pop('description')

    # Override config with any provided kwargs
    config.update(kwargs)

    # Set model name if not provided
    if 'model_name' not in config:
        config['model_name'] = f'bfcnn_{variant}'

    logger.info(f"Creating BFCNN variant '{variant}': {description}")

    return create_bfcnn_denoiser(
        input_shape=input_shape,
        **config
    )

# ---------------------------------------------------------------------
