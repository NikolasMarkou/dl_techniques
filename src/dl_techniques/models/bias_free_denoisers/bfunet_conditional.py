"""
Conditional Bias-Free U-Net Model with Deep Supervision

Extends the bias-free U-Net architecture to support class-conditional denoising
following Miyasawa's theorem for conditionals. The model learns the conditional
score function ∇log p(y|c) where c is a class label.

Key features:
- Class-conditional denoising via embedding layer
- Maintains bias-free properties for scaling invariance (Homogeneity of Degree 1)
- Supports Classifier-Free Guidance (CFG) training
- Compatible with deep supervision
- Optional unconditional token for CFG sampling

Based on:
- Miyasawa's theorem for conditional denoisers
- Bias-free CNN principles (Mohan et al., ICLR 2020)
- Classifier-Free Guidance (Ho & Salimans, 2022)
"""

import keras
from typing import Optional, Union, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock
from .bfunet import BFUNET_CONFIGS

# ---------------------------------------------------------------------

def create_conditional_bfunet_denoiser(
        input_shape: Tuple[int, int, int],
        num_classes: int,
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
        class_embedding_dim: int = 128,
        class_injection_method: str = 'spatial_broadcast',
        enable_cfg_training: bool = True,
        model_name: str = 'conditional_bias_free_unet'
) -> keras.Model:
    """
    Create a class-conditional bias-free U-Net model with optional deep supervision.

    This model extends the bias-free U-Net to support class-conditional denoising,
    learning the conditional score function ∇log p(y|c). The conditioning is achieved
    through class embeddings that are injected into the network architecture while
    maintaining bias-free properties.

    The model follows Miyasawa's theorem for conditionals:
        x̂(y, c) = E[x|y, c] = y + σ² ∇_y log p(y|c)

    CRITICAL NOTE ON BIAS-FREE CONSTRAINT:
    To strictly satisfy the Bias-Free property f(αx) = αf(x), the class conditioning
    must be Multiplicative (Gating) rather than Additive. Additive injection would
    introduce a static bias term that does not scale with the input intensity α,
    breaking generalization to unseen noise levels. This implementation uses
    multiplicative modulation.

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        num_classes: Integer, number of class labels (includes unconditional token if CFG enabled).
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
        enable_deep_supervision: Boolean, whether to add deep supervision outputs. Defaults to False.
        class_embedding_dim: Integer, dimension of class embedding vectors. Defaults to 128.
        class_injection_method: String, method for injecting class information.
                               Options: 'spatial_broadcast', 'channel_concat'. Defaults to 'spatial_broadcast'.
        enable_cfg_training: Boolean, whether to support CFG by including unconditional token. Defaults to True.
        model_name: String, name for the model. Defaults to 'conditional_bias_free_unet'.

    Returns:
        keras.Model: Conditional bias-free U-Net model with inputs [image, class_label].
                    - If deep_supervision=False: Single output tensor
                    - If deep_supervision=True: List of output tensors [final_output, intermediate_outputs...]

    Raises:
        ValueError: If parameters are invalid.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create conditional model for 10 classes
        >>> model = create_conditional_bfunet_denoiser(
        ...     input_shape=(256, 256, 3),
        ...     num_classes=10,
        ...     depth=4,
        ...     enable_deep_supervision=True,
        ...     enable_cfg_training=True
        ... )
        >>> # Model expects inputs: [images, class_labels]
        >>> # class_labels shape: (batch_size,) with integers in [0, num_classes-1]
        >>> # For CFG: reserve class num_classes-1 as unconditional token
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

    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    if class_injection_method not in ['spatial_broadcast', 'channel_concat']:
        raise ValueError(f"Invalid class_injection_method: {class_injection_method}")

    # Input layers
    image_input = keras.Input(shape=input_shape, name='image_input')
    class_input = keras.Input(shape=(1,), dtype='int32', name='class_input')

    # Class embedding layer
    # Maps discrete class labels to dense embedding vectors
    class_embedding = keras.layers.Embedding(
        input_dim=num_classes,
        output_dim=class_embedding_dim,
        embeddings_initializer=kernel_initializer,
        name='class_embedding'
    )(class_input)
    class_embedding = keras.layers.Flatten(name='class_embedding_flatten')(class_embedding)

    logger.info(f"Class embedding: {num_classes} classes -> {class_embedding_dim}D")
    if enable_cfg_training:
        logger.info(f"CFG training enabled: class {num_classes - 1} reserved as unconditional token")

    # Calculate filter sizes for each level
    filter_sizes = [initial_filters * (filter_multiplier ** i) for i in range(depth + 1)]

    # Storage for skip connections and deep supervision outputs
    skip_connections: List[keras.layers.Layer] = []
    deep_supervision_outputs: List[keras.layers.Layer] = []

    # =========================================================================
    # CLASS CONDITIONING INJECTION
    # =========================================================================

    def inject_class_conditioning(
            x: keras.layers.Layer,
            class_emb: keras.layers.Layer,
            level: int,
            stage: str
    ) -> keras.layers.Layer:
        """
        Inject class conditioning into feature maps.

        Uses Multiplicative Modulation to ensure the Bias-Free property:
        f(alpha * x) = alpha * f(x).

        Additive injection is strictly avoided as it introduces static biases.

        Args:
            x: Feature tensor (batch, height, width, channels)
            class_emb: Class embedding vector (batch, embedding_dim)
            level: Current network level
            stage: 'encoder', 'bottleneck', or 'decoder'

        Returns:
            Conditioned feature tensor
        """
        spatial_shape = keras.ops.shape(x)
        height, width = spatial_shape[1], spatial_shape[2]
        feature_channels = x.shape[-1] # Static shape access if available

        if class_injection_method == 'spatial_broadcast':
            # Project embedding to match feature channels
            projected = keras.layers.Dense(
                feature_channels if feature_channels is not None else keras.ops.shape(x)[-1],
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'{stage}_level_{level}_class_project'
            )(class_emb)

            # Reshape to (batch, 1, 1, channels) for broadcasting
            projected = keras.layers.Reshape((1, 1, -1))(projected)

            # Multiplicative Modulation (Gating)
            # x_out = x * (1 + P(c)) would be ideal for initialization,
            # but x * P(c) is strictly homogeneous if P(c) acts as a scaler.
            # We use Multiply. The network learns the scaling factor.
            # To preserve signal flow at init, P(c) should optimally be close to 1 (or we can adding 1).
            # Here we rely on standard Multiply, assuming proper training dynamics.
            conditioned = keras.layers.Multiply(
                name=f'{stage}_level_{level}_class_gate'
            )([x, projected])

        elif class_injection_method == 'channel_concat':
            # To maintain Bias-Free property with concatenation, the concatenated
            # features must also scale with the input intensity.
            # We achieve this by scaling the broadcasted embedding by the average
            # energy of the input features.

            # 1. Project embedding
            projected = keras.layers.Dense(
                initial_filters // 2,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'{stage}_level_{level}_class_project'
            )(class_emb)

            # 2. Tile to spatial dimensions
            tiled = keras.layers.RepeatVector(height * width)(projected)
            tiled = keras.layers.Reshape(
                (height, width, -1),
                name=f'{stage}_level_{level}_class_reshape'
            )(tiled)

            # 3. Compute Scaling Factor from Input (Mean Absolute Amplitude)
            # This ensures that if x scales by alpha, the scale factor also scales by alpha
            # preserving homogeneity.
            # shape: (batch, height, width, 1) or (batch, 1, 1, 1) depending on granularity.
            # We use global spatial average for stability.
            scale_factor = keras.layers.Lambda(
                lambda t: keras.ops.mean(keras.ops.abs(t), axis=[1, 2, 3], keepdims=True),
                name=f'{stage}_level_{level}_energy_calc'
            )(x)

            # 4. Modulate the tiled embedding
            tiled_scaled = keras.layers.Multiply(
                name=f'{stage}_level_{level}_emb_scale'
            )([tiled, scale_factor])

            # 5. Concatenate with features
            conditioned = keras.layers.Concatenate(
                axis=-1,
                name=f'{stage}_level_{level}_class_concat'
            )([x, tiled_scaled])

        return conditioned

    # =========================================================================
    # ENCODER PATH (Contracting)
    # =========================================================================

    x = image_input
    logger.info(f"Building conditional encoder path with {depth} levels")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"Encoder level {level}: {current_filters} filters")

        # Inject class conditioning at each level
        x = inject_class_conditioning(x, class_embedding, level, 'encoder')

        # Convolution blocks at current resolution
        for block_idx in range(blocks_per_level):
            if level == 0 and block_idx == 0:
                # First block with larger kernel
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

        # Downsampling (except for the last level)
        if level < depth - 1:
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'encoder_downsample_{level}'
            )(x)

    # =========================================================================
    # BOTTLENECK
    # =========================================================================

    bottleneck_filters = filter_sizes[depth]
    logger.info(f"Building conditional bottleneck with {bottleneck_filters} filters")

    # Downsample to bottleneck
    x = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='bottleneck_downsample'
    )(x)

    # Inject class conditioning at bottleneck
    x = inject_class_conditioning(x, class_embedding, depth, 'bottleneck')

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

    logger.info(f"Building conditional decoder path with {depth} levels")
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

        # Ensure spatial dimensions match
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

        # Inject class conditioning after concatenation
        x = inject_class_conditioning(x, class_embedding, level, 'decoder')

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

        # Deep supervision output (if enabled and not the final level)
        if enable_deep_supervision and level > 0:
            supervision_branch = BiasFreeConv2D(
                filters=initial_filters,
                kernel_size=3,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=True,
                name=f'supervision_intermediate_level_{level}'
            )(x)

            supervision_output = BiasFreeConv2D(
                filters=output_channels,
                kernel_size=1,
                activation=final_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=False,
                name=f'supervision_output_level_{level}'
            )(supervision_branch)

            deep_supervision_outputs.append(supervision_output)

            logger.info(f"Added deep supervision output at level {level}")

    # =========================================================================
    # FINAL OUTPUT LAYER
    # =========================================================================

    # Final convolution to output channels (bias-free, maintains scaling invariance)
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
        # Order supervision outputs from shallowest to deepest (by resolution)
        # The final output (index 0) is the primary inference output
        # Supervision outputs (indices 1+) are ordered by decreasing resolution

        # Reverse the supervision outputs so they go from shallow to deep
        # deep_supervision_outputs was built as [level_3, level_2, level_1] (deep to shallow)
        # We want [level_1, level_2, level_3] (shallow to deep)
        ordered_supervision_outputs = list(reversed(deep_supervision_outputs))
        all_outputs = [final_output] + ordered_supervision_outputs

        logger.info(f"Created conditional deep supervision model with {len(all_outputs)} outputs:")
        logger.info(f"  - Final output (index 0): {final_output.shape}")
        for i, sup_output in enumerate(ordered_supervision_outputs):
            # Calculate the actual level based on reversed order
            level = i + 1  # levels 1, 2, 3 for indices 1, 2, 3
            logger.info(f"  - Supervision output {i + 1} (index {i + 1}, level {level}): {sup_output.shape}")

        # Create model with multiple outputs
        model = keras.Model(
            inputs=[image_input, class_input],
            outputs=all_outputs,
            name=model_name
        )
    else:
        # Single output model (standard U-Net or inference model)
        model = keras.Model(
            inputs=[image_input, class_input],
            outputs=final_output,
            name=model_name
        )

    logger.info(f"Created conditional bias-free U-Net '{model_name}' with depth {depth}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class embedding dimension: {class_embedding_dim}")
    logger.info(f"Injection method: {class_injection_method}")
    logger.info(f"Total parameters: {model.count_params():,}")

    return model

# ---------------------------------------------------------------------

def create_conditional_bfunet_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        enable_deep_supervision: bool = False,
        enable_cfg_training: bool = True,
        **kwargs
) -> keras.Model:
    """
    Create a conditional bias-free U-Net with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        num_classes: Integer, number of class labels.
        enable_deep_supervision: Boolean, whether to enable deep supervision.
        enable_cfg_training: Boolean, whether to enable CFG training support.
        **kwargs: Additional keyword arguments to override defaults.

    Returns:
        keras.Model: Conditional bias-free U-Net model.

    Example:
        >>> # Create conditional model for 10 classes
        >>> model = create_conditional_bfunet_variant(
        ...     'base',
        ...     (256, 256, 3),
        ...     num_classes=10,
        ...     enable_deep_supervision=True,
        ...     enable_cfg_training=True
        ... )
    """

    if variant not in BFUNET_CONFIGS:
        available_variants = list(BFUNET_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available: {available_variants}")

    config = BFUNET_CONFIGS[variant].copy()
    description = config.pop('description')

    config.update(kwargs)

    if 'model_name' not in config:
        ds_suffix = '_ds' if enable_deep_supervision else ''
        cfg_suffix = '_cfg' if enable_cfg_training else ''
        config['model_name'] = f'conditional_bfunet_{variant}{ds_suffix}{cfg_suffix}'

    config['enable_deep_supervision'] = enable_deep_supervision
    config['enable_cfg_training'] = enable_cfg_training

    logger.info(f"Creating conditional BFU-Net variant '{variant}': {description}")

    return create_conditional_bfunet_denoiser(
        input_shape=input_shape,
        num_classes=num_classes,
        **config
    )

# ---------------------------------------------------------------------
