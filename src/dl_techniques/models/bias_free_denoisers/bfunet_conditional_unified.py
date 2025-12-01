"""
Unified Conditional Bias-Free U-Net Model

Implements a generalized conditional denoising architecture supporting multiple
conditioning modalities under Miyasawa's theorem:

    x̂(y, c) = E[x|y, c] = y + σ² ∇_y log p(y|c)

Supported conditioning types:
- Dense conditioning: High-dimensional signals (RGB images → depth maps)
- Discrete conditioning: Low-dimensional signals (class labels → images)
- Hybrid conditioning: Multiple modalities simultaneously (RGB + class → depth)

Key features:
- Unified mathematical framework for all conditioning types
- Modular conditioning injection mechanisms
- Bias-free architecture maintaining scaling invariance
- Deep supervision support
- Classifier-Free Guidance (CFG) for discrete conditioning

THEORETICAL CONSTRAINT:
To maintain the Bias-Free property f(αx) = αf(x), all conditioning injections
must be Multiplicative (Gating/FiLM) or Energy-Scaled Concatenation. 
Purely additive injections are strictly prohibited as they introduce static biases.
"""

import keras
from typing import Optional, Union, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.bias_free_conv2d import BiasFreeConv2D, BiasFreeResidualBlock

# ---------------------------------------------------------------------
# CONDITIONING ENCODERS
# ---------------------------------------------------------------------

def create_dense_conditioning_encoder(
        input_layer: keras.layers.Layer,
        num_levels: int,
        base_filters: int = 64,
        encoder_type: str = 'custom',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activation: Union[str, callable] = 'relu'
) -> List[keras.layers.Layer]:
    """
    Create multi-scale feature encoder for dense conditioning signals.

    Extracts hierarchical features from dense inputs (e.g., RGB images) at
    multiple spatial scales for injection into the denoising U-Net.

    Args:
        input_layer: Input tensor for dense signal.
        num_levels: Number of hierarchical levels to extract.
        base_filters: Base number of filters for first level.
        encoder_type: Encoder architecture ('custom', 'simple').
        kernel_initializer: Weight initializer.
        kernel_regularizer: Weight regularizer.
        activation: Activation function.

    Returns:
        List of feature tensors at different scales, ordered from high to low resolution:
        [level_0 (H,W), level_1 (H/2,W/2), level_2 (H/4,W/4), ...]
    """

    logger.info(f"Building dense conditioning encoder: {encoder_type}, {num_levels} levels")

    features = []
    x = input_layer

    for level in range(num_levels):
        current_filters = base_filters * (2 ** level)

        # Convolution blocks at current scale
        for block_idx in range(2):
            x = BiasFreeConv2D(
                filters=current_filters,
                kernel_size=3,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_batch_norm=True,
                name=f'dense_encoder_level_{level}_conv_{block_idx}'
            )(x)

        # Store features at this scale
        features.append(x)

        logger.info(f"  Level {level}: {x.shape} with {current_filters} filters")

        # Downsample for next level (except last)
        if level < num_levels - 1:
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'dense_encoder_downsample_{level}'
            )(x)

    return features


# ---------------------------------------------------------------------
# CONDITIONING INJECTION MECHANISMS
# ---------------------------------------------------------------------

class DenseConditioningInjection(keras.layers.Layer):
    """
    Inject dense conditioning features into target features.

    Supports multiple injection methods compatible with Bias-Free constraints:
    - 'film': Scale-only Feature-wise Linear Modulation (x * (1 + P(c))).
    - 'multiplication': Direct gating (x * P(c)).
    - 'concatenation': Energy-scaled concatenation to preserve homogeneity.
    
    NOTE: 'addition' is not supported as it violates bias-free constraints.
    """

    def __init__(
            self,
            method: str = 'film',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: str = 'dense_injection',
            **kwargs
    ):
        """
        Initialize dense conditioning injection layer.

        Args:
            method: Injection method ('film', 'multiplication', 'concatenation').
            kernel_initializer: Weight initializer.
            kernel_regularizer: Weight regularizer.
            name: Layer name.
        """
        super().__init__(name=name, **kwargs)
        if method == 'addition':
            logger.warning(f"Method 'addition' violates bias-free constraints. Promoting to 'film'.")
            method = 'film'
            
        self.method = method
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.projection_layer = None

    def build(self, input_shape):
        """Build layer based on input shapes."""
        target_shape, conditioning_shape = input_shape
        target_channels = target_shape[-1]
        conditioning_channels = conditioning_shape[-1]

        if self.method == 'multiplication':
            # Project conditioning to match target channels
            if conditioning_channels != target_channels:
                self.projection_layer = BiasFreeConv2D(
                    filters=target_channels,
                    kernel_size=1,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    use_batch_norm=False,
                    name=f'{self.name}_projection'
                )

        elif self.method == 'film':
            # Scale only (no shift to maintain bias-free)
            self.projection_layer = BiasFreeConv2D(
                filters=target_channels,
                kernel_size=1,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_batch_norm=False,
                name=f'{self.name}_scale'
            )

        super().build(input_shape)

    def call(self, inputs):
        """
        Inject conditioning features.

        Args:
            inputs: Tuple of (target_features, conditioning_features)

        Returns:
            Conditioned target features
        """
        target_features, conditioning_features = inputs

        if self.method == 'multiplication':
            if self.projection_layer is not None:
                conditioning_projected = self.projection_layer(conditioning_features)
            else:
                conditioning_projected = conditioning_features

            return keras.layers.Multiply()([target_features, conditioning_projected])

        elif self.method == 'concatenation':
            # For concatenation to be bias-free, the conditioning signal must scale
            # with the target signal's energy.
            # Calculate mean absolute amplitude of target features
            scale_factor = keras.layers.Lambda(
                lambda t: keras.ops.mean(keras.ops.abs(t), axis=[1, 2, 3], keepdims=True),
                name=f'{self.name}_energy_calc'
            )(target_features)
            
            # Scale the conditioning features
            conditioning_scaled = keras.layers.Multiply()([conditioning_features, scale_factor])
            
            return keras.layers.Concatenate(axis=-1)([target_features, conditioning_scaled])

        elif self.method == 'film':
            # FiLM without bias: y = x * (1 + scale)
            scale = self.projection_layer(conditioning_features)
            return keras.layers.Multiply()([target_features, scale + 1.0])

        else:
            raise ValueError(f"Unknown injection method: {self.method}")

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'method': self.method,
            'kernel_initializer': keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(
                keras.regularizers.get(self.kernel_regularizer)
            ) if self.kernel_regularizer else None,
        })
        return config


class DiscreteConditioningInjection(keras.layers.Layer):
    """
    Inject discrete conditioning (embeddings) into target features.

    Methods:
    - 'spatial_broadcast': Scale-only FiLM (x * (1 + P(emb))).
    - 'channel_concat': Energy-scaled concatenation.
    """

    def __init__(
            self,
            method: str = 'spatial_broadcast',
            projected_channels: Optional[int] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: str = 'discrete_injection',
            **kwargs
    ):
        """
        Initialize discrete conditioning injection layer.

        Args:
            method: Injection method ('spatial_broadcast', 'channel_concat').
            projected_channels: Number of channels to project to (defaults to target channels).
            kernel_initializer: Weight initializer.
            kernel_regularizer: Weight regularizer.
            name: Layer name.
        """
        super().__init__(name=name, **kwargs)
        if method == 'addition': # Backwards compatibility/correction
             method = 'spatial_broadcast' # mapped to multiplicative broadcast below

        self.method = method
        self.projected_channels = projected_channels
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.projection_layer = None

    def build(self, input_shape):
        """Build layer based on input shapes."""
        target_shape, embedding_shape = input_shape
        target_channels = target_shape[-1]
        
        # Determine projection size
        if self.projected_channels is None:
            if self.method == 'spatial_broadcast':
                self.projected_channels = target_channels
            elif self.method == 'channel_concat':
                self.projected_channels = target_channels // 4  # Smaller to balance parameters

        # Create projection layer
        self.projection_layer = keras.layers.Dense(
            self.projected_channels,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f'{self.name}_projection'
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        Inject discrete conditioning.

        Args:
            inputs: Tuple of (target_features, embedding_vector)

        Returns:
            Conditioned target features
        """
        target_features, embedding = inputs

        # Project embedding
        projected = self.projection_layer(embedding)

        if self.method == 'spatial_broadcast':
            # Reshape to (batch, 1, 1, channels) for broadcasting
            projected = keras.layers.Reshape((1, 1, self.projected_channels))(projected)

            # Multiplicative Modulation (Scale-only FiLM)
            # x_out = x * (1 + P(c))
            return keras.layers.Multiply()([target_features, projected + 1.0])

        elif self.method == 'channel_concat':
            # Get spatial dimensions from target
            spatial_h = keras.ops.shape(target_features)[1]
            spatial_w = keras.ops.shape(target_features)[2]

            # Tile spatially
            tiled = keras.layers.RepeatVector(spatial_h * spatial_w)(projected)
            tiled = keras.layers.Reshape((spatial_h, spatial_w, self.projected_channels))(tiled)

            # Calculate mean absolute amplitude of target features for scaling
            scale_factor = keras.layers.Lambda(
                lambda t: keras.ops.mean(keras.ops.abs(t), axis=[1, 2, 3], keepdims=True),
                name=f'{self.name}_energy_calc'
            )(target_features)
            
            # Scale the tiled embeddings
            tiled_scaled = keras.layers.Multiply()([tiled, scale_factor])

            # Concatenate along channel dimension
            return keras.layers.Concatenate(axis=-1)([target_features, tiled_scaled])

        else:
            raise ValueError(f"Unknown injection method: {self.method}")

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'method': self.method,
            'projected_channels': self.projected_channels,
            'kernel_initializer': keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(
                keras.regularizers.get(self.kernel_regularizer)
            ) if self.kernel_regularizer else None,
        })
        return config


# ---------------------------------------------------------------------
# UNIFIED CONDITIONAL BFUNET MODEL
# ---------------------------------------------------------------------

def create_unified_conditional_bfunet(
        target_shape: Tuple[int, int, int],
        dense_conditioning_shape: Optional[Tuple[int, int, int]] = None,
        num_classes: Optional[int] = None,
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: int = 2,
        blocks_per_level: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        activation: Union[str, callable] = 'relu',
        final_activation: Union[str, callable] = 'linear',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_residual_blocks: bool = True,
        dense_conditioning_encoder_filters: int = 64,
        dense_injection_method: str = 'film',
        class_embedding_dim: int = 128,
        discrete_injection_method: str = 'spatial_broadcast',
        enable_cfg_training: bool = False,
        enable_deep_supervision: bool = False,
        model_name: str = 'unified_conditional_bfunet'
) -> keras.Model:
    """
    Create unified conditional bias-free U-Net supporting multiple conditioning modalities.

    This model implements the generalized conditional Miyasawa's theorem:
        x̂(y, c) = E[x|y, c] = y + σ² ∇_y log p(y|c)

    where c can be:
    - Dense signal (e.g., RGB image for depth estimation)
    - Discrete signal (e.g., class label for generation)
    - Both simultaneously (hybrid conditioning)

    Args:
        target_shape: Shape of target signal (H, W, C) - e.g., depth map or image.
        dense_conditioning_shape: Shape of dense conditioning signal (H, W, C) - e.g., RGB image.
                                  None to disable dense conditioning.
        num_classes: Number of discrete classes. None to disable discrete conditioning.
        depth: Number of downsampling levels in U-Net.
        initial_filters: Number of filters at first level.
        filter_multiplier: Multiplier for filters at each level.
        blocks_per_level: Number of residual blocks per level.
        kernel_size: Size of convolutional kernels.
        activation: Activation function.
        final_activation: Final layer activation.
        kernel_initializer: Weight initializer.
        kernel_regularizer: Weight regularizer.
        use_residual_blocks: Whether to use residual blocks.
        dense_conditioning_encoder_filters: Base filters for dense encoder.
        dense_injection_method: Method for dense injection ('film', 'concatenation').
        class_embedding_dim: Dimension of class embeddings.
        discrete_injection_method: Method for discrete injection ('spatial_broadcast', 'channel_concat').
        enable_cfg_training: Enable classifier-free guidance training for discrete conditioning.
        enable_deep_supervision: Enable multi-scale deep supervision.
        model_name: Name for the model.

    Returns:
        keras.Model with appropriate inputs based on conditioning configuration:
        - Dense only: inputs=[noisy_target, dense_condition]
        - Discrete only: inputs=[noisy_target, class_label]
        - Hybrid: inputs=[noisy_target, dense_condition, class_label]
        - Outputs: Single tensor or list (if deep supervision enabled)

    Raises:
        ValueError: If neither dense nor discrete conditioning is specified.

    Example:
        >>> # Depth estimation (dense conditioning only)
        >>> model = create_unified_conditional_bfunet(
        ...     target_shape=(256, 256, 1),  # Depth map
        ...     dense_conditioning_shape=(256, 256, 3),  # RGB image
        ...     num_classes=None
        ... )
    """

    # =========================================================================
    # VALIDATION
    # =========================================================================

    has_dense = dense_conditioning_shape is not None
    has_discrete = num_classes is not None

    if not has_dense and not has_discrete:
        raise ValueError("At least one conditioning modality must be specified")

    if depth < 3:
        raise ValueError(f"depth must be at least 3, got {depth}")

    if initial_filters <= 0:
        raise ValueError(f"initial_filters must be positive, got {initial_filters}")

    # =========================================================================
    # INPUT LAYERS
    # =========================================================================

    logger.info(f"Building unified conditional BFU-Net: depth={depth}")
    logger.info(f"  Dense conditioning: {has_dense}")
    logger.info(f"  Discrete conditioning: {has_discrete}")

    # Target input (noisy signal to denoise)
    noisy_target_input = keras.Input(
        shape=target_shape,
        name='noisy_target_input'
    )

    inputs = [noisy_target_input]
    dense_features_per_level = None
    class_embedding = None

    # Dense conditioning input and encoder
    if has_dense:
        dense_condition_input = keras.Input(
            shape=dense_conditioning_shape,
            name='dense_condition_input'
        )
        inputs.append(dense_condition_input)

        logger.info(f"  Building dense conditioning encoder")
        dense_features_per_level = create_dense_conditioning_encoder(
            input_layer=dense_condition_input,
            num_levels=depth + 1,
            base_filters=dense_conditioning_encoder_filters,
            encoder_type='custom',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation=activation
        )

    # Discrete conditioning input and embedding
    if has_discrete:
        class_label_input = keras.Input(
            shape=(1,),
            dtype='int32',
            name='class_label_input'
        )
        inputs.append(class_label_input)

        logger.info(f"  Building discrete conditioning embedding: {num_classes} classes")

        # Class embedding layer
        class_embedding = keras.layers.Embedding(
            input_dim=num_classes,
            output_dim=class_embedding_dim,
            embeddings_initializer=kernel_initializer,
            name='class_embedding'
        )(class_label_input)

        class_embedding = keras.layers.Flatten(name='class_embedding_flatten')(class_embedding)

        if enable_cfg_training:
            logger.info(f"  CFG training enabled: class {num_classes - 1} reserved as unconditional")

    # =========================================================================
    # DENOISING U-NET WITH CONDITIONAL INJECTION
    # =========================================================================

    # Calculate filter sizes for each level
    filter_sizes = [initial_filters * (filter_multiplier ** i) for i in range(depth + 1)]

    # Storage for skip connections and deep supervision outputs
    skip_connections: List[keras.layers.Layer] = []
    deep_supervision_outputs: List[keras.layers.Layer] = []

    # Helper function for conditional injection
    def inject_conditioning(
            features: keras.layers.Layer,
            level: int,
            stage: str
    ) -> keras.layers.Layer:
        """
        Inject all available conditioning signals into features.

        Args:
            features: Current feature tensor.
            level: Current network level.
            stage: Stage identifier ('encoder', 'bottleneck', 'decoder').

        Returns:
            Conditioned feature tensor.
        """
        x = features

        # Inject dense conditioning (if available)
        if has_dense and dense_features_per_level is not None:
            x = DenseConditioningInjection(
                method=dense_injection_method,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'{stage}_level_{level}_dense_injection'
            )([x, dense_features_per_level[level]])

        # Inject discrete conditioning (if available)
        if has_discrete and class_embedding is not None:
            x = DiscreteConditioningInjection(
                method=discrete_injection_method,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f'{stage}_level_{level}_discrete_injection'
            )([x, class_embedding])

        return x

    # =========================================================================
    # ENCODER PATH
    # =========================================================================

    x = noisy_target_input
    logger.info(f"Building encoder path with {depth} levels")

    for level in range(depth):
        current_filters = filter_sizes[level]
        logger.info(f"  Encoder level {level}: {current_filters} filters")

        # Inject conditioning at this level
        x = inject_conditioning(x, level, 'encoder')

        # Convolution blocks
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

        # Store skip connection
        skip_connections.append(x)

        # Downsample (except for last level)
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

    # Inject conditioning at bottleneck
    x = inject_conditioning(x, depth, 'bottleneck')

    # Bottleneck blocks
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
    # DECODER PATH WITH DEEP SUPERVISION
    # =========================================================================

    logger.info(f"Building decoder path with {depth} levels")
    output_channels = target_shape[-1]

    for level in range(depth - 1, -1, -1):
        current_filters = filter_sizes[level]
        logger.info(f"  Decoder level {level}: {current_filters} filters")

        # Upsample
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation='bilinear',
            name=f'decoder_upsample_{level}'
        )(x)

        # Get skip connection
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

        # Concatenate skip connection
        x = keras.layers.Concatenate(
            axis=-1,
            name=f'decoder_concat_{level}'
        )([skip, x])

        # Inject conditioning after concatenation
        x = inject_conditioning(x, level, 'decoder')

        # Convolution blocks
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

        # Deep supervision output (if enabled and not final level)
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
            logger.info(f"  Added deep supervision output at level {level}")

    # =========================================================================
    # FINAL OUTPUT LAYER
    # =========================================================================

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
        ordered_supervision_outputs = list(reversed(deep_supervision_outputs))
        all_outputs = [final_output] + ordered_supervision_outputs

        logger.info(f"Created model with {len(all_outputs)} outputs (deep supervision)")

        model = keras.Model(
            inputs=inputs,
            outputs=all_outputs,
            name=model_name
        )
    else:
        model = keras.Model(
            inputs=inputs,
            outputs=final_output,
            name=model_name
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    conditioning_str = []
    if has_dense:
        conditioning_str.append(f"dense ({dense_conditioning_shape})")
    if has_discrete:
        conditioning_str.append(f"discrete ({num_classes} classes)")

    logger.info(f"Created unified conditional BFU-Net '{model_name}'")
    logger.info(f"  Conditioning: {', '.join(conditioning_str)}")
    logger.info(f"  Depth: {depth}")
    logger.info(f"  Filter progression: {filter_sizes}")
    logger.info(f"  Target shape: {target_shape}")
    logger.info(f"  Deep supervision: {enable_deep_supervision}")
    logger.info(f"  Total parameters: {model.count_params():,}")

    return model


# ---------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ---------------------------------------------------------------------

def create_depth_estimation_bfunet(
        depth_shape: Tuple[int, int, int] = (256, 256, 1),
        rgb_shape: Tuple[int, int, int] = (256, 256, 3),
        **kwargs
) -> keras.Model:
    """
    Create bias-free U-Net for monocular depth estimation.

    Convenience wrapper for depth estimation task using dense conditioning only.

    Args:
        depth_shape: Shape of depth maps (H, W, 1).
        rgb_shape: Shape of RGB images (H, W, 3).
        **kwargs: Additional arguments passed to create_unified_conditional_bfunet.

    Returns:
        Model with inputs [noisy_depth, rgb_image].

    Example:
        >>> model = create_depth_estimation_bfunet(
        ...     depth_shape=(256, 256, 1),
        ...     rgb_shape=(256, 256, 3),
        ...     depth=4,
        ...     initial_filters=64
        ... )
    """
    return create_unified_conditional_bfunet(
        target_shape=depth_shape,
        dense_conditioning_shape=rgb_shape,
        num_classes=None,
        model_name='depth_estimation_bfunet',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_class_conditional_bfunet(
        image_shape: Tuple[int, int, int] = (256, 256, 3),
        num_classes: int = 10,
        enable_cfg_training: bool = True,
        **kwargs
) -> keras.Model:
    """
    Create bias-free U-Net for class-conditional image generation/denoising.

    Convenience wrapper for class-conditional task using discrete conditioning only.

    Args:
        image_shape: Shape of images (H, W, C).
        num_classes: Number of classes.
        enable_cfg_training: Enable classifier-free guidance training.
        **kwargs: Additional arguments passed to create_unified_conditional_bfunet.

    Returns:
        Model with inputs [noisy_image, class_label].

    Example:
        >>> model = create_class_conditional_bfunet(
        ...     image_shape=(256, 256, 3),
        ...     num_classes=10,
        ...     enable_cfg_training=True,
        ...     depth=4,
        ...     initial_filters=64
        ... )
    """
    return create_unified_conditional_bfunet(
        target_shape=image_shape,
        dense_conditioning_shape=None,
        num_classes=num_classes,
        enable_cfg_training=enable_cfg_training,
        model_name='class_conditional_bfunet',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_semantic_depth_bfunet(
        depth_shape: Tuple[int, int, int] = (256, 256, 1),
        rgb_shape: Tuple[int, int, int] = (256, 256, 3),
        num_classes: int = 20,
        **kwargs
) -> keras.Model:
    """
    Create bias-free U-Net for semantic-aware depth estimation.

    Convenience wrapper for hybrid conditioning (RGB + semantic class).

    Args:
        depth_shape: Shape of depth maps (H, W, 1).
        rgb_shape: Shape of RGB images (H, W, 3).
        num_classes: Number of semantic classes.
        **kwargs: Additional arguments passed to create_unified_conditional_bfunet.

    Returns:
        Model with inputs [noisy_depth, rgb_image, class_label].

    Example:
        >>> model = create_semantic_depth_bfunet(
        ...     depth_shape=(256, 256, 1),
        ...     rgb_shape=(256, 256, 3),
        ...     num_classes=20,  # e.g., PASCAL VOC classes
        ...     depth=4,
        ...     initial_filters=64
        ... )
    """
    return create_unified_conditional_bfunet(
        target_shape=depth_shape,
        dense_conditioning_shape=rgb_shape,
        num_classes=num_classes,
        enable_cfg_training=False,  # CFG not typically used for depth estimation
        model_name='semantic_depth_bfunet',
        **kwargs
    )

# ---------------------------------------------------------------------
