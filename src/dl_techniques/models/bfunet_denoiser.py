"""
Bias-Free U-Net Model

Implements a U-Net architecture where all additive constants (bias terms)
have been removed to enable better generalization across different noise levels
and improved scaling invariance properties.

Based on the bias-free principles from "Robust and Interpretable Blind Image
Denoising via Bias-Free Convolutional Neural Networks" (Mohan et al., ICLR 2020)
applied to the U-Net architecture from "U-Net: Convolutional Networks for
Biomedical Image Segmentation" (Ronneberger et al., MICCAI 2015).
"""

import keras
from typing import Optional, Union, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock


# ---------------------------------------------------------------------

def create_bias_free_unet(
        input_shape: Tuple[int, int, int],
        depth: int = 3,
        initial_filters: int = 64,
        filter_multiplier: int = 2,
        blocks_per_level: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        activation: Union[str, callable] = 'relu',
        final_activation: Union[str, callable] = 'linear',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
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
        depth: Integer, depth of the U-Net (number of downsampling levels). Defaults to 3.
        initial_filters: Integer, number of filters in the first level. Defaults to 64.
        filter_multiplier: Integer, multiplier for filters at each level. Defaults to 2.
        blocks_per_level: Integer, number of conv blocks per level. Defaults to 2.
        kernel_size: Integer or tuple, size of convolutional kernels. Defaults to 3.
        activation: String or callable, activation function. Defaults to 'relu'.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'glorot_uniform'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        use_residual_blocks: Boolean, whether to use residual blocks. Defaults to True.
        model_name: String, name for the model. Defaults to 'bias_free_unet'.

    Returns:
        keras.Model: Bias-free U-Net model ready for training.

    Raises:
        ValueError: If depth is less than 1, initial_filters is non-positive,
                   filter_multiplier is less than 1, or blocks_per_level is non-positive.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create standard bias-free U-Net
        >>> model = create_bias_free_unet(
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

    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")

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
# Predefined model configurations
# ---------------------------------------------------------------------

def create_bias_free_unet_light(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Create a lightweight bias-free U-Net (shallow depth, fewer filters).

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).

    Returns:
        keras.Model: Lightweight bias-free U-Net with depth=2 and 32 initial filters.
    """
    return create_bias_free_unet(
        input_shape=input_shape,
        depth=2,
        initial_filters=32,
        blocks_per_level=1,
        model_name='bias_free_unet_light'
    )


def create_bias_free_unet_standard(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Create a standard bias-free U-Net (balanced performance and speed).

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).

    Returns:
        keras.Model: Standard bias-free U-Net with depth=3 and 64 initial filters.
    """
    return create_bias_free_unet(
        input_shape=input_shape,
        depth=3,
        initial_filters=64,
        blocks_per_level=2,
        model_name='bias_free_unet_standard'
    )


def create_bias_free_unet_deep(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Create a deep bias-free U-Net (more depth for complex tasks).

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).

    Returns:
        keras.Model: Deep bias-free U-Net with depth=4 and 64 initial filters.
    """
    return create_bias_free_unet(
        input_shape=input_shape,
        depth=4,
        initial_filters=64,
        blocks_per_level=2,
        model_name='bias_free_unet_deep'
    )


def create_bias_free_unet_large(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Create a large bias-free U-Net (high capacity for complex tasks).

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).

    Returns:
        keras.Model: Large bias-free U-Net with depth=4 and 128 initial filters.
    """
    return create_bias_free_unet(
        input_shape=input_shape,
        depth=4,
        initial_filters=128,
        blocks_per_level=3,
        model_name='bias_free_unet_large'
    )


# ---------------------------------------------------------------------
# Additional utility functions
# ---------------------------------------------------------------------

def create_bias_free_unet_segmentation(
        input_shape: Tuple[int, int, int],
        num_classes: int,
        depth: int = 3
) -> keras.Model:
    """
    Create a bias-free U-Net specifically configured for segmentation tasks.

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        num_classes: Integer, number of segmentation classes.
        depth: Integer, depth of the U-Net. Defaults to 3.

    Returns:
        keras.Model: Bias-free U-Net configured for segmentation with sigmoid/softmax output.
    """
    # Determine final activation based on number of classes
    if num_classes == 1:
        final_activation = 'sigmoid'  # Binary segmentation
        output_channels = 1
    else:
        final_activation = 'softmax'  # Multi-class segmentation
        output_channels = num_classes

    # Override input shape for output channels
    modified_input_shape = (input_shape[0], input_shape[1], input_shape[2])

    model = create_bias_free_unet(
        input_shape=modified_input_shape,
        depth=depth,
        initial_filters=64,
        final_activation=final_activation,
        model_name=f'bias_free_unet_segmentation_{num_classes}classes'
    )

    # Modify the final layer to output correct number of classes
    # Get all layers except the last one
    x = model.layers[-2].output

    # Create new final layer with correct number of output channels
    outputs = BiasFreeConv2D(
        filters=output_channels,
        kernel_size=1,
        activation=final_activation,
        use_batch_norm=False,
        name='segmentation_output'
    )(x)

    # Create new model with modified output
    segmentation_model = keras.Model(
        inputs=model.input,
        outputs=outputs,
        name=model.name
    )

    logger.info(f"Created bias-free U-Net for segmentation with {num_classes} classes")
    logger.info(f"Final activation: {final_activation}, Output channels: {output_channels}")

    return segmentation_model

# ---------------------------------------------------------------------