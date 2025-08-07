"""
Bias-Free U-Net Model with Deep Supervision and Variants

Implements a U-Net architecture with deep supervision where all additive constants
(bias terms) have been removed to enable better generalization across different
noise levels and improved scaling invariance properties. The deep supervision
outputs intermediate predictions at multiple scales during training, allowing
for better gradient flow and more stable training.

Deep supervision provides several benefits:
- Better gradient flow to deeper layers during training
- Multi-scale feature learning and supervision
- More stable training for very deep networks
- Curriculum learning capabilities through weight scheduling

The model outputs multiple scales during training:
- Output 0: Final inference output (highest resolution, primary output)
- Output 1-N: Intermediate supervision outputs at progressively lower resolutions

Based on the bias-free principles from "Robust and Interpretable Blind Image
Denoising via Bias-Free Convolutional Neural Networks" (Mohan et al., ICLR 2020)
applied to the U-Net architecture with deep supervision.
"""

import keras
import tensorflow as tf
from typing import Optional, Union, Tuple, List, Dict, Any
from pathlib import Path


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock

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
        enable_deep_supervision: bool = False,
        model_name: str = 'bias_free_unet'
) -> keras.Model:
    """
    Create a bias-free U-Net model with optional deep supervision.

    This function creates a complete U-Net architecture using bias-free layers with
    deep supervision capabilities. The model exhibits scaling-invariant properties:
    if the input is scaled by α, the output is also scaled by α.

    During training with deep supervision enabled, the model outputs multiple scales:
    - Output 0: Final inference output (full resolution)
    - Output 1: Second-to-last decoder level output
    - Output N: Deepest supervision level output

    During inference, only the final output (index 0) is typically used.

    Architecture:
    - Encoder: Bias-free conv blocks + downsampling at each level
    - Bottleneck: Bias-free conv blocks at the lowest resolution
    - Decoder: Upsampling + skip connections + bias-free conv blocks
    - Deep Supervision: Additional outputs at intermediate decoder levels
    - Skip connections preserve high-resolution features

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        depth: Integer, depth of the U-Net (number of downsampling levels). Defaults to 4.
        initial_filters: Integer, number of filters in the first level. Defaults to 64.
        filter_multiplier: Integer, multiplier for filters at each level. Defaults to 2.
        blocks_per_level: Integer, number of conv blocks per level. Defaults to 2.
        kernel_size: Integer or tuple, size of convolutional kernels. Defaults to 3.
        initial_kernel_size: Integer or tuple, size of first convolutional kernels. Defaults to 5.
        activation: String or callable, activation function. Defaults to 'leaky_relu'.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'he_normal'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        use_residual_blocks: Boolean, whether to use residual blocks. Defaults to True.
        enable_deep_supervision: Boolean, whether to add deep supervision outputs. Defaults to True.
        model_name: String, name for the model. Defaults to 'bias_free_unet'.

    Returns:
        keras.Model: Bias-free U-Net model ready for training.
                    - If deep_supervision=False: Single output tensor
                    - If deep_supervision=True: List of output tensors [final_output, intermediate_outputs...]

    Raises:
        ValueError: If depth is less than 3, initial_filters is non-positive,
                   filter_multiplier is less than 1, or blocks_per_level is non-positive.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create standard bias-free U-Net with deep supervision
        >>> model = create_bfunet_denoiser(
        ...     input_shape=(256, 256, 3),
        ...     depth=4,
        ...     initial_filters=64,
        ...     enable_deep_supervision=True
        ... )
        >>> # Model outputs: [final_output, supervision_output_1, supervision_output_2, ...]
        >>>
        >>> # Create inference-only model (single output)
        >>> inference_model = create_bfunet_denoiser(
        ...     input_shape=(None, None, 3),  # Flexible spatial dimensions
        ...     depth=4,
        ...     initial_filters=64,
        ...     enable_deep_supervision=False  # Single output for inference
        ... )
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

    # Storage for skip connections and deep supervision outputs
    skip_connections: List[keras.layers.Layer] = []
    deep_supervision_outputs: List[keras.layers.Layer] = []

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
    # DECODER PATH (Expanding) with Deep Supervision
    # =========================================================================

    logger.info(f"Building decoder path with {depth} levels")
    output_channels = input_shape[-1]

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

        # =====================================================================
        # DEEP SUPERVISION OUTPUT (if enabled and not the final level)
        # =====================================================================

        if enable_deep_supervision and level > 0:  # Skip final level (it will be the main output)
            # Create supervision output at current scale
            supervision_output = BiasFreeConv2D(
                filters=output_channels,
                kernel_size=1,
                activation=final_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=False,
                name=f'supervision_output_level_{level}'
            )(x)

            deep_supervision_outputs.append(supervision_output)

            logger.info(f"Added deep supervision output at level {level} "
                       f"with shape: {supervision_output.shape}")

    # =========================================================================
    # FINAL OUTPUT LAYER (Primary inference output)
    # =========================================================================

    # Final convolution to output channels (no batch norm, custom activation)
    final_output = BiasFreeConv2D(
        filters=output_channels,
        kernel_size=1,
        activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,
        name='final_output'
    )(x)

    # =========================================================================
    # MODEL CREATION
    # =========================================================================

    if enable_deep_supervision and deep_supervision_outputs:
        # Return multiple outputs: [final_output, supervision_outputs...]
        # The final output (index 0) is the primary inference output
        # Supervision outputs (indices 1+) are for training only
        all_outputs = [final_output] + deep_supervision_outputs

        logger.info(f"Created deep supervision model with {len(all_outputs)} outputs:")
        logger.info(f"  - Final output (index 0): {final_output.shape}")
        for i, sup_output in enumerate(deep_supervision_outputs):
            logger.info(f"  - Supervision output {i+1} (index {i+1}): {sup_output.shape}")

        # Create model with multiple outputs
        model = keras.Model(
            inputs=inputs,
            outputs=all_outputs,
            name=model_name
        )

    else:
        # Single output model (standard U-Net or inference model)
        model = keras.Model(
            inputs=inputs,
            outputs=final_output,
            name=model_name
        )

        logger.info(f"Created single-output model")

    logger.info(f"Created bias-free U-Net model '{model_name}' with depth {depth}")
    logger.info(f"Filter progression: {filter_sizes}")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")
    logger.info(f"Deep supervision enabled: {enable_deep_supervision}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model

# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_bfunet_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        enable_deep_supervision: bool = True,
        **kwargs
) -> keras.Model:
    """
    Create a bias-free U-Net model with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        enable_deep_supervision: Boolean, whether to enable deep supervision outputs.
        **kwargs: Additional keyword arguments to override default parameters.

    Returns:
        keras.Model: Bias-free U-Net model with the specified variant configuration.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        >>> # Standard usage with deep supervision
        >>> model = create_bfunet_variant('base', (256, 256, 3), enable_deep_supervision=True)
        >>> model.summary()
        >>>
        >>> # Inference model (single output)
        >>> inference_model = create_bfunet_variant('base', (None, None, 3), enable_deep_supervision=False)
        >>>
        >>> # With custom parameters
        >>> model = create_bfunet_variant('large', (224, 224, 1),
        ...                                     enable_deep_supervision=True,
        ...                                     activation='gelu',
        ...                                     use_residual_blocks=False)
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
        ds_suffix = '_ds' if enable_deep_supervision else ''
        config['model_name'] = f'bias_free_unet_{variant}{ds_suffix}'

    # Set deep supervision
    config['enable_deep_supervision'] = enable_deep_supervision

    logger.info(f"Creating bias-free U-Net variant '{variant}': {description}")
    logger.info(f"Deep supervision: {'enabled' if enable_deep_supervision else 'disabled'}")

    return create_bfunet_denoiser(
        input_shape=input_shape,
        **config
    )

# ---------------------------------------------------------------------
# Utility Functions for Deep Supervision
# ---------------------------------------------------------------------

def get_model_output_info(model: keras.Model) -> Dict[str, Any]:
    """
    Get information about model outputs for deep supervision models.

    Args:
        model: Keras model to analyze

    Returns:
        Dictionary containing output information:
        - 'num_outputs': Number of outputs
        - 'has_deep_supervision': Whether model has multiple outputs
        - 'output_shapes': List of output shapes
        - 'primary_output_index': Index of the primary inference output (always 0)

    Example:
        >>> model = create_bfunet_variant('base', (256, 256, 3), enable_deep_supervision=True)
        >>> info = get_model_output_info(model)
        >>> print(f"Number of outputs: {info['num_outputs']}")
        >>> print(f"Primary output shape: {info['output_shapes'][info['primary_output_index']]}")
    """
    # Handle both single output and multi-output models
    if isinstance(model.output, list):
        num_outputs = len(model.output)
        output_shapes = [output.shape for output in model.output]
        has_deep_supervision = True
    else:
        num_outputs = 1
        output_shapes = [model.output.shape]
        has_deep_supervision = False

    return {
        'num_outputs': num_outputs,
        'has_deep_supervision': has_deep_supervision,
        'output_shapes': output_shapes,
        'primary_output_index': 0  # Primary output is always at index 0
    }

def create_inference_model_from_training_model(training_model: keras.Model) -> keras.Model:
    """
    Create a single-output inference model from a multi-output training model.

    Args:
        training_model: Multi-output training model with deep supervision

    Returns:
        Single-output model using only the primary output (index 0)

    Example:
        >>> # Create training model with deep supervision
        >>> training_model = create_bfunet_variant('base', (256, 256, 3), enable_deep_supervision=True)
        >>>
        >>> # Create inference model (single output)
        >>> inference_model = create_inference_model_from_training_model(training_model)
        >>>
        >>> # Inference model accepts flexible input shapes
        >>> inference_model = keras.Model(
        ...     inputs=keras.Input(shape=(None, None, 3)),
        ...     outputs=inference_model.layers[-1].output  # Get final layer output
        ... )
    """
    model_info = get_model_output_info(training_model)

    if not model_info['has_deep_supervision']:
        logger.info("Model already has single output, returning as-is")
        return training_model

    # Extract only the primary output (index 0)
    primary_output = training_model.output[model_info['primary_output_index']]

    # Create new model with single output
    inference_model = keras.Model(
        inputs=training_model.input,
        outputs=primary_output,
        name=f"{training_model.name}_inference"
    )

    logger.info(f"Created inference model with single output shape: {primary_output.shape}")

    return inference_model
