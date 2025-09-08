"""
ConvUNext: Modern Bias-Free U-Net with ConvNeXt-Inspired Architecture

Implements a ConvUNext architecture with deep supervision leveraging existing
ConvNeXt V1/V2 blocks while maintaining bias-free properties for better
generalization across different noise levels and improved scaling invariance.

ConvUNext combines the best of U-Net and ConvNeXt architectures:
- U-Net's encoder-decoder structure with skip connections
- ConvNeXt's modern architectural innovations via existing implementations
- Bias-free design for scaling invariance (use_bias=False)

Key modern improvements over standard U-Net:
- Reuses existing ConvNeXt V1/V2 blocks with bias-free configuration
- Depthwise separable convolutions for efficiency
- Inverted bottleneck design (channel expansion then contraction)
- Global Response Normalization (GRN) for V2 blocks
- GELU activation functions for smoother gradients
- Larger kernel sizes (7x7) for better receptive fields
- Layer scaling for training stability
- Optional stochastic depth for regularization

The architecture maintains the bias-free principle: if input is scaled by α,
output is also scaled by α, enabling better generalization across noise levels.

Deep supervision provides several benefits:
- Better gradient flow to deeper layers during training
- Multi-scale feature learning and supervision
- More stable training for very deep networks
- Curriculum learning capabilities through weight scheduling

The model outputs multiple scales during training:
- Output 0: Final inference output (highest resolution, primary output)
- Output 1-N: Intermediate supervision outputs at progressively lower resolutions

Based on ConvNeXt innovations from "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
and "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
(Woo et al., CVPR 2023) applied to bias-free U-Net architecture.
"""

import keras
from typing import Optional, Union, Tuple, List, Dict, Any


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization

# ---------------------------------------------------------------------
# ConvUNext Bias-Free Building Blocks (Simple Stem)
# ---------------------------------------------------------------------

class ConvUNextStem(keras.layers.Layer):
    """
    ConvUNext stem block for initial feature extraction using bias-free design.

    Simple stem that uses a single large kernel convolution followed by
    Global Response Normalization and GELU activation, while keeping
    channel count conservative to avoid OOM issues.

    Args:
        filters: Integer, number of output filters.
        kernel_size: Integer or tuple, size of convolution kernel. Defaults to 7.
        kernel_initializer: String or Initializer, weight initializer.
        kernel_regularizer: String or Regularizer, weight regularizer.
        **kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Sublayers initialized in build()
        self.conv = None
        self.grn = None

    def build(self, input_shape):
        """Build the stem layers."""
        super().build(input_shape)

        # Large kernel convolution (bias-free)
        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,  # Bias-free for scaling invariance
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stem_conv'
        )

        # Global Response Normalization (consistent with ConvNeXt V2)
        self.grn = GlobalResponseNormalization(name='stem_grn')

    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.conv(inputs)
        x = self.grn(x)
        x = keras.activations.gelu(x)
        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
# ConvUNext Model Variant Configurations
# ---------------------------------------------------------------------

CONVUNEXT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'depth': 3,
        'initial_filters': 32,  # Start conservative to avoid OOM
        'blocks_per_level': 2,
        'convnext_version': 'v2',  # Use V2 by default for GRN
        'drop_path_rate': 0.0,
        'description': 'Tiny ConvUNext (depth=3) for quick experiments.'
    },
    'small': {
        'depth': 3,
        'initial_filters': 48,
        'blocks_per_level': 2,
        'convnext_version': 'v2',
        'drop_path_rate': 0.1,
        'description': 'Small ConvUNext (depth=3) with minimal capacity.'
    },
    'base': {
        'depth': 4,
        'initial_filters': 64,
        'blocks_per_level': 3,
        'convnext_version': 'v2',
        'drop_path_rate': 0.1,
        'description': 'Base ConvUNext (depth=4) with standard configuration.'
    },
    'large': {
        'depth': 4,
        'initial_filters': 96,
        'blocks_per_level': 4,
        'convnext_version': 'v2',
        'drop_path_rate': 0.2,
        'description': 'Large ConvUNext (depth=4) with high capacity.'
    },
    'xlarge': {
        'depth': 5,
        'initial_filters': 128,
        'blocks_per_level': 5,
        'convnext_version': 'v2',
        'drop_path_rate': 0.3,
        'description': 'Extra-Large ConvUNext (depth=5) for maximum performance.'
    }
}

# ---------------------------------------------------------------------
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_convunext_denoiser(
        input_shape: Tuple[int, int, int],
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: int = 2,
        blocks_per_level: int = 2,
        convnext_version: str = 'v2',
        stem_kernel_size: Union[int, Tuple[int, int]] = 7,
        block_kernel_size: Union[int, Tuple[int, int]] = 7,
        drop_path_rate: float = 0.1,
        final_activation: Union[str, callable] = 'linear',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        enable_deep_supervision: bool = False,
        model_name: str = 'convunext'
) -> keras.Model:
    """
    Create a ConvUNext model using existing ConvNeXt V1/V2 blocks with bias-free configuration.

    This function creates a complete ConvUNext architecture using existing ConvNeXt blocks
    with bias-free design (`use_bias=False`) and deep supervision capabilities. The model
    exhibits scaling-invariant properties: if the input is scaled by α, the output is also
    scaled by α.

    ConvUNext leverages existing implementations:
    - U-Net's encoder-decoder structure with skip connections
    - ConvNeXt V1/V2 blocks with bias-free configuration
    - Deep supervision for better training

    Key features:
    - Reuses existing ConvNeXt V1/V2 block implementations
    - Bias-free design via use_bias=False parameter
    - Depthwise separable convolutions for efficiency
    - Global Response Normalization (V2) or LayerNorm (V1)
    - GELU activation for smoother gradients
    - Layer scaling for training stability
    - Optional stochastic depth for regularization
    - Larger kernels (7x7) for better receptive fields

    During training with deep supervision enabled, the model outputs multiple scales:
    - Output 0: Final inference output (full resolution)
    - Output 1: Second-to-last decoder level output
    - Output N: Deepest supervision level output

    Architecture:
    - Encoder: ConvNeXt blocks + downsampling at each level
    - Bottleneck: ConvNeXt blocks at the lowest resolution
    - Decoder: Upsampling + skip connections + ConvNeXt blocks
    - Deep Supervision: Additional outputs at intermediate decoder levels

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        depth: Integer, depth of the U-Net (number of downsampling levels). Defaults to 4.
        initial_filters: Integer, number of filters in the first level. Defaults to 64.
        filter_multiplier: Integer, multiplier for filters at each level. Defaults to 2.
        blocks_per_level: Integer, number of blocks per level. Defaults to 2.
        convnext_version: String, 'v1' or 'v2' to choose ConvNeXt version. Defaults to 'v2'.
        stem_kernel_size: Integer or tuple, size of stem convolution kernels. Defaults to 7.
        block_kernel_size: Integer or tuple, size of block kernels. Defaults to 7.
        drop_path_rate: Float, stochastic depth drop probability. Defaults to 0.1.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'he_normal'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        enable_deep_supervision: Boolean, whether to add deep supervision outputs. Defaults to False.
        model_name: String, name for the model. Defaults to 'convunext'.

    Returns:
        keras.Model: ConvUNext model ready for training.
                    - If deep_supervision=False: Single output tensor
                    - If deep_supervision=True: List of output tensors [final_output, intermediate_outputs...]

    Raises:
        ValueError: If depth is less than 3, initial_filters is non-positive,
                   filter_multiplier is less than 1, blocks_per_level is non-positive,
                   or convnext_version is not 'v1' or 'v2'.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create ConvUNext with ConvNeXt V2 blocks and deep supervision
        >>> model = create_convunext_denoiser(
        ...     input_shape=(256, 256, 3),
        ...     depth=4,
        ...     initial_filters=64,
        ...     convnext_version='v2',
        ...     enable_deep_supervision=True
        ... )
        >>>
        >>> # Create inference-only model with V1 blocks
        >>> inference_model = create_convunext_denoiser(
        ...     input_shape=(None, None, 3),  # Flexible spatial dimensions
        ...     depth=4,
        ...     initial_filters=64,
        ...     convnext_version='v1',
        ...     enable_deep_supervision=False
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

    if convnext_version not in ['v1', 'v2']:
        raise ValueError(f"convnext_version must be 'v1' or 'v2', got {convnext_version}")

    # Select ConvNeXt block type
    ConvNextBlock = ConvNextV2Block if convnext_version == 'v2' else ConvNextV1Block

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
    logger.info(f"Building ConvUNext encoder path with {depth} levels using ConvNeXt {convnext_version.upper()}")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"Encoder level {level}: {current_filters} filters")

        # First level uses stem block with conservative channel count
        if level == 0:
            # Use stem block for initial feature extraction
            x = ConvUNextStem(
                filters=current_filters,
                kernel_size=stem_kernel_size,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'encoder_level_{level}_stem'
            )(x)
        else:
            # Channel adjustment if needed (bias-free)
            if x.shape[-1] != current_filters:
                x = keras.layers.Conv2D(
                    filters=current_filters,
                    kernel_size=1,
                    use_bias=False,  # Bias-free
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name=f'encoder_level_{level}_channel_adjust'
                )(x)

        # ConvNeXt blocks at current resolution (bias-free)
        for block_idx in range(blocks_per_level):
            # Calculate progressive drop path rate
            current_drop_path = drop_path_rate * (level * blocks_per_level + block_idx) / (depth * blocks_per_level)

            x = ConvNextBlock(
                kernel_size=block_kernel_size,
                filters=current_filters,
                activation='gelu',
                use_bias=False,  # Bias-free for scaling invariance
                dropout_rate=current_drop_path,  # Use as stochastic depth
                spatial_dropout_rate=0.0,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'encoder_level_{level}_convnext_{convnext_version}_block_{block_idx}'
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
    logger.info(f"Building ConvUNext bottleneck with {bottleneck_filters} filters")

    # Downsample to bottleneck
    x = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='bottleneck_downsample'
    )(x)

    # Channel adjustment for bottleneck (bias-free)
    if x.shape[-1] != bottleneck_filters:
        x = keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=1,
            use_bias=False,  # Bias-free
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='bottleneck_channel_adjust'
        )(x)

    # Bottleneck ConvNeXt blocks (bias-free)
    for block_idx in range(blocks_per_level):
        x = ConvNextBlock(
            kernel_size=block_kernel_size,
            filters=bottleneck_filters,
            activation='gelu',
            use_bias=False,  # Bias-free for scaling invariance
            dropout_rate=drop_path_rate,
            spatial_dropout_rate=0.0,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f'bottleneck_convnext_{convnext_version}_block_{block_idx}'
        )(x)

    # =========================================================================
    # DECODER PATH (Expanding) with Deep Supervision
    # =========================================================================

    logger.info(f"Building ConvUNext decoder path with {depth} levels")
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
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
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

        # Channel adjustment after concatenation (bias-free)
        if x.shape[-1] != current_filters:
            x = keras.layers.Conv2D(
                filters=current_filters,
                kernel_size=1,
                use_bias=False,  # Bias-free
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'decoder_level_{level}_channel_adjust'
            )(x)

        # ConvNeXt blocks after merging (bias-free)
        for block_idx in range(blocks_per_level):
            # Calculate progressive drop path rate
            current_drop_path = drop_path_rate * (level * blocks_per_level + block_idx) / (depth * blocks_per_level)

            x = ConvNextBlock(
                kernel_size=block_kernel_size,
                filters=current_filters,
                activation='gelu',
                use_bias=False,  # Bias-free for scaling invariance
                dropout_rate=current_drop_path,
                spatial_dropout_rate=0.0,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'decoder_level_{level}_convnext_{convnext_version}_block_{block_idx}'
            )(x)

        # =====================================================================
        # DEEP SUPERVISION OUTPUT (if enabled and not the final level)
        # =====================================================================

        if enable_deep_supervision and level > 0:
            # Create supervision output at current scale (bias-free)
            supervision_branch = keras.layers.Conv2D(
                filters=current_filters // 2,
                kernel_size=1,
                use_bias=False,  # Bias-free
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'supervision_intermediate_level_{level}'
            )(x)

            supervision_branch = GlobalResponseNormalization(
                name=f'supervision_grn_level_{level}'
            )(supervision_branch)

            supervision_branch = keras.layers.Activation(
                'gelu',
                name=f'supervision_activation_level_{level}'
            )(supervision_branch)

            supervision_output = keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=1,
                activation=final_activation,
                use_bias=False,  # Bias-free
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'supervision_output_level_{level}'
            )(supervision_branch)

            deep_supervision_outputs.append(supervision_output)

            logger.info(f"Added deep supervision output at level {level} "
                       f"with shape: {supervision_output.shape}")

    # =========================================================================
    # FINAL OUTPUT LAYER (Primary inference output)
    # =========================================================================

    # Final convolution to output channels (bias-free)
    final_output = keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=1,
        activation=final_activation,
        use_bias=False,  # Bias-free
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name='final_output'
    )(x)

    # =========================================================================
    # MODEL CREATION
    # =========================================================================

    if enable_deep_supervision and deep_supervision_outputs:
        # Return multiple outputs: [final_output, supervision_outputs...]
        # Order supervision outputs from shallowest to deepest (by resolution)
        ordered_supervision_outputs = list(reversed(deep_supervision_outputs))
        all_outputs = [final_output] + ordered_supervision_outputs

        logger.info(f"Created ConvUNext deep supervision model with {len(all_outputs)} outputs:")
        logger.info(f"  - Final output (index 0): {final_output.shape}")
        for i, sup_output in enumerate(ordered_supervision_outputs):
            level = i + 1
            logger.info(f"  - Supervision output {i + 1} (index {i + 1}, level {level}): {sup_output.shape}")

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

        logger.info(f"Created single-output ConvUNext model")

    logger.info(f"Created ConvUNext model '{model_name}' with depth {depth}")
    logger.info(f"ConvNeXt version: {convnext_version.upper()}")
    logger.info(f"Filter progression: {filter_sizes}")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")
    logger.info(f"Deep supervision enabled: {enable_deep_supervision}")
    logger.info(f"Drop path rate: {drop_path_rate}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model

# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_convunext_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        enable_deep_supervision: bool = True,
        **kwargs
) -> keras.Model:
    """
    Create a ConvUNext model with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        enable_deep_supervision: Boolean, whether to enable deep supervision outputs.
        **kwargs: Additional keyword arguments to override default parameters.

    Returns:
        keras.Model: ConvUNext model with the specified variant configuration.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        >>> # Standard usage with ConvNeXt V2 blocks and deep supervision
        >>> model = create_convunext_variant('base', (256, 256, 3), enable_deep_supervision=True)
        >>> model.summary()
        >>>
        >>> # Inference model with ConvNeXt V1 blocks
        >>> inference_model = create_convunext_variant('base', (None, None, 3),
        ...                                                   enable_deep_supervision=False,
        ...                                                   convnext_version='v1')
        >>>
        >>> # Custom parameters
        >>> model = create_convunext_variant('large', (224, 224, 1),
        ...                                         enable_deep_supervision=True,
        ...                                         convnext_version='v2',
        ...                                         drop_path_rate=0.3)
    """
    if variant not in CONVUNEXT_CONFIGS:
        available_variants = list(CONVUNEXT_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = CONVUNEXT_CONFIGS[variant].copy()
    description = config.pop('description')

    # Override config with any provided kwargs
    config.update(kwargs)

    # Set model name if not provided
    if 'model_name' not in config:
        ds_suffix = '_ds' if enable_deep_supervision else ''
        convnext_version = config.get('convnext_version', 'v2')
        config['model_name'] = f'convunext_{variant}_{convnext_version}{ds_suffix}'

    # Set deep supervision
    config['enable_deep_supervision'] = enable_deep_supervision

    logger.info(f"Creating ConvUNext variant '{variant}': {description}")
    logger.info(f"ConvNeXt version: {config.get('convnext_version', 'v2').upper()}")
    logger.info(f"Deep supervision: {'enabled' if enable_deep_supervision else 'disabled'}")

    return create_convunext_denoiser(
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
        >>> model = create_convunext_variant('base', (256, 256, 3), enable_deep_supervision=True)
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
        >>> training_model = create_convunext_variant('base', (256, 256, 3), enable_deep_supervision=True)
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