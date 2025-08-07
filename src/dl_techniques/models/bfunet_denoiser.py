"""
Bias-Free U-Net Model with Variants

Implements a U-Net architecture where all additive constants (bias terms)
have been removed to enable better generalization across different noise levels
and improved scaling invariance properties. Provides multiple model variants
(tiny, small, base, large, xlarge) for different computational requirements
and performance targets.

Based on the bias-free principles from "Robust and Interpretable Blind Image
Denoising via Bias-Free Convolutional Neural Networks" (Mohan et al., ICLR 2020)
applied to the U-Net architecture from "U-Net: Convolutional Networks for
Biomedical Image Segmentation" (Ronneberger et al., MICCAI 2015).
"""

import keras
from typing import Optional, Union, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock

# ---------------------------------------------------------------------
# Model Variant Configurations
# ---------------------------------------------------------------------

BFUNET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'depth': 3,
        'initial_filters': 16,
        'blocks_per_level': 2,
        'description': 'Tiny BF-UNet (depth=3) for quick experiments.'
    },
    'small': {
        'depth': 3,
        'initial_filters': 24,
        'blocks_per_level': 2,
        'description': 'Small BF-UNet (depth=3) with minimal capacity.'
    },
    'base': {
        'depth': 4,
        'initial_filters': 32,
        'blocks_per_level': 3,
        'description': 'Base BF-UNet (depth=4) with standard configuration.'
    },
    'large': {
        'depth': 4,
        'initial_filters': 48,
        'blocks_per_level': 4,
        'description': 'Large BF-UNet (depth=4) with high capacity.'
    },
    'xlarge': {
        'depth': 5,
        'initial_filters': 64,
        'blocks_per_level': 5,
        'description': 'Extra-Large BF-UNet (depth=5) for maximum performance.'
    }
}

# ---------------------------------------------------------------------
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_bfunet_denoiser(
        input_shape: Tuple[int, int, int],
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: int = 2,
        blocks_per_level: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        initial_kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: Union[str, callable] = 'leaky_relu',
        final_activation: Union[str, callable] = 'linear',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_residual_blocks: bool = True,
        model_name: str = 'bias_free_unet'
) -> keras.Model:
    """
    Create a bias-free U-Net model with configurable depth.

    This function creates a complete U-Net architecture using bias-free layers.
    The model exhibits scaling-invariant properties: if the input is scaled by α,
    the output is also scaled by α. The U-Net consists of an encoder (contracting)
    path, a bottleneck, and a decoder (expanding) path with skip connections.

    Architecture:
    - Encoder: Bias-free conv blocks + downsampling at each level
    - Bottleneck: Bias-free conv blocks at the lowest resolution
    - Decoder: Upsampling + skip connections + bias-free conv blocks
    - Skip connections preserve high-resolution features

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        depth: Integer, depth of the U-Net (number of downsampling levels). Defaults to 4.
        initial_filters: Integer, number of filters in the first level. Defaults to 64.
        filter_multiplier: Integer, multiplier for filters at each level. Defaults to 2.
        blocks_per_level: Integer, number of conv blocks per level. Defaults to 2.
        kernel_size: Integer or tuple, size of convolutional kernels. Defaults to 3.
        initial_kernel_size: Integer or tuple, size of first convolutional kernels. Defaults to 5.
        activation: String or callable, activation function. Defaults to 'relu'.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'glorot_uniform'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        use_residual_blocks: Boolean, whether to use residual blocks. Defaults to True.
        model_name: String, name for the model. Defaults to 'bias_free_unet'.

    Returns:
        keras.Model: Bias-free U-Net model ready for training.

    Raises:
        ValueError: If depth is less than 3, initial_filters is non-positive,
                   filter_multiplier is less than 1, or blocks_per_level is non-positive.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create standard bias-free U-Net
        >>> model = create_bfunet_denoiser(
        ...     input_shape=(256, 256, 3),
        ...     depth=4,
        ...     initial_filters=64
        ... )
        >>> model.compile(optimizer='adam', loss='mse')
        >>>
        >>> # Model exhibits scaling invariance
        >>> # If input is scaled by α, output is also scaled by α
    """

    # Input validation
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise TypeError("input_shape must be a tuple of 3 integers (height, width, channels)")

    if depth < 3:
        raise ValueError(f"depth must be at least 3, got {depth}")

    if initial_filters <= 0:
        raise ValueError(f"initial_filters must be positive, got {initial_filters}")

    if filter_multiplier < 1:
        raise ValueError(f"filter_multiplier must be at least 1, got {filter_multiplier}")

    if blocks_per_level <= 0:
        raise ValueError(f"blocks_per_level must be positive, got {blocks_per_level}")

    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_images')

    # Calculate filter sizes for each level
    filter_sizes = [initial_filters * (filter_multiplier ** i) for i in range(depth + 1)]

    # Storage for skip connections
    skip_connections: List[keras.layers.Layer] = []

    # =========================================================================
    # ENCODER PATH (Contracting)
    # =========================================================================

    x = inputs
    logger.info(f"Building encoder path with {depth} levels")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"Encoder level {level}: {current_filters} filters")

        # Convolution blocks at current resolution
        for block_idx in range(blocks_per_level):
            if level == 0 and block_idx == 0:
                # first level
                x = BiasFreeConv2D(
                    filters=current_filters,
                    kernel_size=initial_kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    use_batch_norm=True,
                    name=f'encoder_level_{level}_conv_{block_idx}'
                )(x)
            else:
                if use_residual_blocks:
                    x = BiasFreeResidualBlock(
                        filters=current_filters,
                        kernel_size=kernel_size,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        name=f'encoder_level_{level}_residual_block_{block_idx}'
                    )(x)
                else:
                    x = BiasFreeConv2D(
                        filters=current_filters,
                        kernel_size=kernel_size,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        use_batch_norm=True,
                        name=f'encoder_level_{level}_conv_{block_idx}'
                    )(x)

        # Store skip connection before downsampling
        skip_connections.append(x)

        # Downsampling (except for the last level which goes to bottleneck)
        if level < depth - 1:
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'encoder_downsample_{level}'
            )(x)

    # =========================================================================
    # BOTTLENECK
    # =========================================================================

    bottleneck_filters = filter_sizes[depth]
    logger.info(f"Building bottleneck with {bottleneck_filters} filters")

    # Downsample to bottleneck
    x = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='bottleneck_downsample'
    )(x)

    # Bottleneck convolution blocks
    for block_idx in range(blocks_per_level):
        if use_residual_blocks:
            x = BiasFreeResidualBlock(
                filters=bottleneck_filters,
                kernel_size=kernel_size,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'bottleneck_residual_block_{block_idx}'
            )(x)
        else:
            x = BiasFreeConv2D(
                filters=bottleneck_filters,
                kernel_size=kernel_size,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=True,
                name=f'bottleneck_conv_{block_idx}'
            )(x)

    # =========================================================================
    # DECODER PATH (Expanding)
    # =========================================================================

    logger.info(f"Building decoder path with {depth} levels")

    for level in range(depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        logger.info(f"Decoder level {level}: {current_filters} filters")

        # Upsampling
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation='bilinear',
            name=f'decoder_upsample_{level}'
        )(x)

        # Get corresponding skip connection
        skip = skip_connections[level]

        # Ensure spatial dimensions match for concatenation
        # Handle potential size mismatches due to pooling/upsampling
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            # Crop or pad to match dimensions
            target_height, target_width = skip.shape[1], skip.shape[2]
            x = keras.layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation='bilinear',
                name=f'decoder_resize_{level}'
            )(x)

        # Merge skip connection
        x = keras.layers.Concatenate(
            axis=-1,
            name=f'decoder_concat_{level}'
        )([skip, x])

        # Convolution blocks after merging
        for block_idx in range(blocks_per_level):
            if use_residual_blocks:
                x = BiasFreeResidualBlock(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name=f'decoder_level_{level}_residual_block_{block_idx}'
                )(x)
            else:
                x = BiasFreeConv2D(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    use_batch_norm=True,
                    name=f'decoder_level_{level}_conv_{block_idx}'
                )(x)

    # =========================================================================
    # FINAL OUTPUT LAYER
    # =========================================================================

    # Final convolution to output channels (no batch norm, custom activation)
    output_channels = input_shape[-1]
    outputs = BiasFreeConv2D(
        filters=output_channels,
        kernel_size=1,
        activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,
        name='final_conv'
    )(x)

    # Create the model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name
    )

    logger.info(f"Created bias-free U-Net model '{model_name}' with depth {depth}")
    logger.info(f"Filter progression: {filter_sizes}")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model

# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_bfunet_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        **kwargs
) -> keras.Model:
    """
    Create a bias-free U-Net model with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        **kwargs: Additional keyword arguments to override default parameters.

    Returns:
        keras.Model: Bias-free U-Net model with the specified variant configuration.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        >>> # Standard usage
        >>> model = create_bfunet_variant('base', (256, 256, 3))
        >>> model.summary()
        >>>
        >>> # With custom parameters
        >>> model = create_bfunet_variant('large', (224, 224, 1),
        ...                                     activation='gelu',
        ...                                     use_residual_blocks=False)
        >>>
        >>> # All available variants
        >>> for variant in ['tiny', 'small', 'base', 'large', 'xlarge']:
        ...     model = create_bfunet_variant(variant, (128, 128, 3))
    """
    if variant not in BFUNET_CONFIGS:
        available_variants = list(BFUNET_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = BFUNET_CONFIGS[variant].copy()
    description = config.pop('description')

    # Override config with any provided kwargs
    config.update(kwargs)

    # Set model name if not provided
    if 'model_name' not in config:
        config['model_name'] = f'bias_free_unet_{variant}'

    logger.info(f"Creating bias-free U-Net variant '{variant}': {description}")

    return create_bfunet_denoiser(
        input_shape=input_shape,
        **config
    )

# ---------------------------------------------------------------------
