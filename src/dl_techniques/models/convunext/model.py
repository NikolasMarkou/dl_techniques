"""
ConvUNext Model: Modern U-Net with ConvNeXt-Inspired Architecture

Refined implementation adhering to Keras 3 strict serialization, layer tracking,
and explicit build patterns following the Complete Guide to Modern Keras 3 Custom
Layers and Models.

Key features:
- Complete Sphinx-style documentation
- Proper type hints throughout
- Explicit sub-layer building in build()
- Graph-safe operations in call()
- Full serialization support
- Configurable bias for standard or restoration tasks
- include_top parameter for feature extraction vs full model
- **Weight compatibility: train with include_top=True, reuse with include_top=False**
"""

import os
import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization


# ---------------------------------------------------------------------
# ConvUNext Stem Block
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextStem(keras.layers.Layer):
    """
    ConvUNext stem block for initial feature extraction.

    Uses LayerNormalization instead of GRN to ensure proper gradient flow
    at initialization (GRN zero-init is for residual blocks). This stem
    performs spatial downsampling via large kernel convolution followed
    by channel-wise normalization.

    **Intent**: Provide efficient initial feature extraction from raw inputs,
    replacing traditional aggressive pooling with learned feature extraction
    via large-kernel convolution.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)
           ↓
    LayerNormalization(epsilon=1e-6)
           ↓
    Output(shape=[batch, height, width, filters])
    ```

    Args:
        filters: Integer, number of output filters. Must be positive.
            Determines the channel dimensionality after stem processing.
        kernel_size: Integer or tuple of 2 integers, specifying the spatial
            dimensions of the convolution kernel. Defaults to 7.
        use_bias: Boolean, whether the convolution layer uses a bias vector.
            Defaults to True. Set to False for restoration tasks.
        kernel_initializer: String or initializer instance for convolution weights.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional string or regularizer instance for convolution
            weights. Defaults to None.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, filters)`.
        Spatial dimensions are preserved due to 'same' padding.
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 7,
            use_bias: bool = True,
            kernel_initializer: str = 'he_normal',
            kernel_regularizer: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize ConvUNextStem layer.

        Parameters
        ----------
        filters : int
            Number of output filters.
        kernel_size : int or tuple of 2 ints, optional
            Size of the convolution kernel, by default 7.
        use_bias : bool, optional
            Whether to use bias in convolution, by default True.
        kernel_initializer : str, optional
            Initializer for kernel weights, by default 'he_normal'.
        kernel_regularizer : str or None, optional
            Regularizer for kernel weights, by default None.
        **kwargs : Any
            Additional keyword arguments for Layer base class.
        """
        super().__init__(**kwargs)

        # Store configuration for serialization
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Create sub-layers in __init__ (Golden Rule #1)
        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stem_conv'
        )

        # Use LayerNorm for stem (standard ConvNeXt design)
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name='stem_norm'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build layer by creating weights for sub-layers.

        Parameters
        ----------
        input_shape : tuple of int or None
            Shape of input tensor (batch_size, height, width, channels).
        """
        # Explicitly build sub-layers (Golden Rule #2)
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(conv_output_shape)

        # Mark layer as built
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the stem layer.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, height, width, channels).
        training : bool or None, optional
            Whether in training mode, by default None.

        Returns
        -------
        keras.KerasTensor
            Output tensor of shape (batch_size, height, width, filters).
        """
        x = self.conv(inputs)
        x = self.norm(x)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape for given input shape.

        Parameters
        ----------
        input_shape : tuple of int or None
            Shape of input tensor.

        Returns
        -------
        tuple of int or None
            Shape of output tensor.
        """
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ConvUNextStem':
        """
        Create layer from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from get_config().

        Returns
        -------
        ConvUNextStem
            Reconstructed layer instance.
        """
        # Deserialize initializers and regularizers
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )
        if 'kernel_regularizer' in config:
            config['kernel_regularizer'] = regularizers.deserialize(
                config['kernel_regularizer']
            )
        return cls(**config)


# ---------------------------------------------------------------------
# ConvUNext Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextModel(keras.Model):
    """
    ConvUNext Model: Modern U-Net with ConvNeXt-inspired blocks and deep supervision.

    This model implements a U-Net-style encoder-decoder architecture using ConvNeXt
    blocks for feature extraction and transformation. It supports multiple model
    variants (tiny to xlarge), optional deep supervision for multi-scale training,
    flexible architecture configuration including bias control, and can be used
    as a feature extractor with include_top=False.

    **Weight Compatibility**: Models are designed so that weights trained with
    include_top=True can be loaded into models with include_top=False. All layers
    are created with identical names regardless of include_top setting.

    **Intent**: Provide a modern, efficient U-Net-style architecture for dense
    prediction tasks (segmentation, super-resolution, etc.) with state-of-the-art
    ConvNeXt blocks and proper skip connections.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C])
           ↓
    Stem: ConvUNextStem(initial_filters)
           ↓
    Encoder: [ConvNeXt Blocks + Downsample] × depth
           ↓ (skip connections)
    Bottleneck: ConvNeXt Blocks
           ↓
    Decoder: [Upsample + Concat + ConvNeXt Blocks] × depth
           ↓ (optional deep supervision)
    Output: Conv2D(output_channels, 1×1) [only used if include_top=True]
           ↓
    Output(shape=[batch, H, W, output_channels or filters])
    ```

    The model uses stochastic depth (drop path) for regularization and supports
    both ConvNeXt v1 and v2 block variants.

    Args:
        input_shape: Tuple of 3 integers (height, width, channels) specifying
            input dimensions. Height and width can be None for variable size.
            Defaults to (None, None, 3).
        depth: Integer, number of encoder/decoder levels. Must be >= 2.
            Controls model capacity and receptive field. Defaults to 4.
        initial_filters: Integer, number of filters in first encoder stage.
            Subsequent stages multiply this by filter_multiplier. Defaults to 64.
        filter_multiplier: Integer, multiplicative factor for filters between
            stages. Defaults to 2 (doubling each stage).
        blocks_per_level: Integer, number of ConvNeXt blocks per encoder/decoder
            level. More blocks = more capacity. Defaults to 2.
        convnext_version: String, either 'v1' or 'v2'. Selects ConvNeXt block
            variant. V2 includes GRN normalization. Defaults to 'v2'.
        stem_kernel_size: Integer or tuple of 2 integers, kernel size for stem
            convolution. Larger kernels = larger initial receptive field.
            Defaults to 7.
        block_kernel_size: Integer or tuple of 2 integers, kernel size for
            depthwise convolutions in ConvNeXt blocks. Defaults to 7.
        drop_path_rate: Float in [0, 1], maximum drop path rate for stochastic
            depth. Linearly increases with depth. Defaults to 0.1.
        final_activation: String or callable, activation for final output layer.
            Common choices: 'sigmoid' (binary), 'softmax' (multi-class),
            'linear' (regression). Only used if include_top=True. Defaults to 'linear'.
        use_bias: Boolean, whether to use bias in convolutions. Defaults to True.
            Set to False for bias-free restoration tasks.
        kernel_initializer: String or initializer instance for convolution weights.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional string or regularizer instance for convolution
            weights. Defaults to None.
        include_top: Boolean, whether to include the final classification/prediction
            head in forward pass. If False, returns feature maps from decoder.
            If True, returns predictions with output_channels.
            **Important**: All layers are always created, but include_top controls
            which ones are used in the forward pass. This ensures weight compatibility.
            Defaults to True.
        enable_deep_supervision: Boolean, whether to output predictions/features at
            multiple scales for deep supervision training. When True, returns list of
            outputs. Works with both include_top=True and include_top=False.
            Defaults to False.
        output_channels: Optional integer, number of output channels. If None and
            include_top=True, uses same as input_channels. Required when creating
            models for training. Defaults to None.
        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        Height and width can be variable if specified as None in input_shape.

    Output shape:
        If include_top=True and enable_deep_supervision=False:
            4D tensor with shape: `(batch_size, height, width, output_channels)`.
        If include_top=True and enable_deep_supervision=True:
            List of 4D tensors, with primary output first followed by
            intermediate supervision outputs at decreasing resolutions.
        If include_top=False and enable_deep_supervision=False:
            4D tensor with shape: `(batch_size, height, width, filters)`.
            Returns decoder features without final projection.
        If include_top=False and enable_deep_supervision=True:
            List of 4D tensors with decoder features at multiple scales.

    Attributes:
        MODEL_VARIANTS: Dictionary mapping variant names to configurations.
        PRETRAINED_WEIGHTS: Dictionary of pretrained weight URLs (placeholder).

    Example:
        ```python
        # Train with predictions
        model = ConvUNextModel.from_variant(
            'base',
            input_shape=(256, 256, 3),
            output_channels=10,
            include_top=True
        )
        model.compile(...)
        model.fit(...)
        model.save('trained.keras')

        # Reuse as feature extractor
        encoder = ConvUNextModel.from_variant(
            'base',
            input_shape=(256, 256, 3),
            output_channels=10,  # Must match for weight loading
            include_top=False
        )
        encoder.load_weights('trained.keras')
        features = encoder(inputs)
        ```
    """

    MODEL_VARIANTS = {
        'tiny': {
            'depth': 3,
            'initial_filters': 32,
            'blocks_per_level': 2,
            'convnext_version': 'v2',
            'drop_path_rate': 0.0,
            'description': 'Tiny variant for resource-constrained environments'
        },
        'small': {
            'depth': 3,
            'initial_filters': 48,
            'blocks_per_level': 2,
            'convnext_version': 'v2',
            'drop_path_rate': 0.1,
            'description': 'Small variant balancing efficiency and performance'
        },
        'base': {
            'depth': 4,
            'initial_filters': 64,
            'blocks_per_level': 3,
            'convnext_version': 'v2',
            'drop_path_rate': 0.1,
            'description': 'Base variant with good general performance'
        },
        'large': {
            'depth': 4,
            'initial_filters': 96,
            'blocks_per_level': 4,
            'convnext_version': 'v2',
            'drop_path_rate': 0.2,
            'description': 'Large variant for high-capacity requirements'
        },
        'xlarge': {
            'depth': 5,
            'initial_filters': 128,
            'blocks_per_level': 5,
            'convnext_version': 'v2',
            'drop_path_rate': 0.3,
            'description': 'Extra-large variant for maximum performance'
        }
    }

    PRETRAINED_WEIGHTS = {
        'tiny': {'imagenet': 'https://example.com/convunext_tiny_imagenet.keras'},
        'small': {'imagenet': 'https://example.com/convunext_small_imagenet.keras'},
        'base': {'imagenet': 'https://example.com/convunext_base_imagenet.keras'},
        'large': {'imagenet': 'https://example.com/convunext_large_imagenet.keras'},
        'xlarge': {'imagenet': 'https://example.com/convunext_xlarge_imagenet.keras'},
    }

    def __init__(
            self,
            input_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 3),
            depth: int = 4,
            initial_filters: int = 64,
            filter_multiplier: int = 2,
            blocks_per_level: int = 2,
            convnext_version: str = 'v2',
            stem_kernel_size: Union[int, Tuple[int, int]] = 7,
            block_kernel_size: Union[int, Tuple[int, int]] = 7,
            drop_path_rate: float = 0.1,
            final_activation: str = 'linear',
            use_bias: bool = True,
            kernel_initializer: str = 'he_normal',
            kernel_regularizer: Optional[str] = None,
            include_top: bool = True,
            enable_deep_supervision: bool = False,
            output_channels: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize ConvUNextModel.

        Parameters
        ----------
        input_shape : tuple of int, optional
            Input shape (height, width, channels), by default (None, None, 3).
        depth : int, optional
            Number of encoder/decoder levels, by default 4.
        initial_filters : int, optional
            Number of filters in first stage, by default 64.
        filter_multiplier : int, optional
            Filter multiplication factor between stages, by default 2.
        blocks_per_level : int, optional
            Number of ConvNeXt blocks per level, by default 2.
        convnext_version : str, optional
            ConvNeXt block version ('v1' or 'v2'), by default 'v2'.
        stem_kernel_size : int or tuple of 2 ints, optional
            Stem convolution kernel size, by default 7.
        block_kernel_size : int or tuple of 2 ints, optional
            ConvNeXt block kernel size, by default 7.
        drop_path_rate : float, optional
            Maximum drop path rate for stochastic depth, by default 0.1.
        final_activation : str, optional
            Activation for output layer (only applied if include_top=True), by default 'linear'.
        use_bias : bool, optional
            Whether to use bias in convolutions, by default True.
        kernel_initializer : str, optional
            Kernel weight initializer, by default 'he_normal'.
        kernel_regularizer : str or None, optional
            Kernel weight regularizer, by default None.
        include_top : bool, optional
            Whether to use final prediction head in forward pass, by default True.
        enable_deep_supervision : bool, optional
            Whether to enable deep supervision outputs, by default False.
        output_channels : int or None, optional
            Number of output channels (None = same as input, required for training),
            by default None.
        **kwargs : Any
            Additional keyword arguments for Model base class.

        Raises
        ------
        ValueError
            If depth < 2.
        """
        super().__init__(**kwargs)

        # Validate configuration
        if depth < 2:
            raise ValueError(f"Depth must be >= 2, got {depth}")

        # Store configuration for serialization
        self.input_shape_config = input_shape
        self.depth = depth
        self.initial_filters = initial_filters
        self.filter_multiplier = filter_multiplier
        self.blocks_per_level = blocks_per_level
        self.convnext_version = convnext_version
        self.stem_kernel_size = stem_kernel_size
        self.block_kernel_size = block_kernel_size
        self.drop_path_rate = drop_path_rate
        self.final_activation = final_activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.include_top = include_top
        self.enable_deep_supervision = enable_deep_supervision

        # Parse input shape
        self.input_height, self.input_width, self.input_channels = input_shape
        self.output_channels = (
            output_channels if output_channels is not None else self.input_channels
        )

        # Compute filter sizes for each stage
        self.filter_sizes = [
            initial_filters * (filter_multiplier ** i)
            for i in range(depth + 1)
        ]

        # Select ConvNeXt block class
        self.ConvNextBlock = (
            ConvNextV2Block if convnext_version == 'v2' else ConvNextV1Block
        )

        # Build model architecture (Golden Rule #1: Create in __init__)
        # CRITICAL: Always create all layers to ensure weight compatibility
        self._build_stem()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()

        if self.enable_deep_supervision:
            self._build_deep_supervision()

        # CRITICAL: Always create final output layer for weight compatibility
        # Whether it's used depends on include_top in call()
        self.final_output_layer = keras.layers.Conv2D(
            filters=self.output_channels,
            kernel_size=1,
            activation=self.final_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='final_output'
        )

    def _build_stem(self) -> None:
        """Create stem layer for initial feature extraction."""
        self.stem = ConvUNextStem(
            filters=self.filter_sizes[0],
            kernel_size=self.stem_kernel_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='encoder_stem'
        )

    def _build_encoder(self) -> None:
        """Create encoder stages with downsampling."""
        self.encoder_stages = []
        self.encoder_downsamples = []

        for level in range(self.depth):
            current_filters = self.filter_sizes[level]
            stage_layers = []

            # Create ConvNeXt blocks for this stage
            for block_idx in range(self.blocks_per_level):
                # Compute drop path rate (increases with depth)
                current_drop_path = (
                        self.drop_path_rate
                        * (level * self.blocks_per_level + block_idx)
                        / (self.depth * self.blocks_per_level)
                )

                block = self.ConvNextBlock(
                    kernel_size=self.block_kernel_size,
                    filters=current_filters,
                    activation='gelu',
                    use_bias=self.use_bias,
                    dropout_rate=current_drop_path,
                    spatial_dropout_rate=0.0,
                    name=f'enc_L{level}_blk{block_idx}'
                )
                stage_layers.append(block)

            self.encoder_stages.append(stage_layers)

            # Create downsampling layer (if not last stage)
            if level < self.depth - 1:
                downsample_ops = []

                # Spatial downsampling via striding
                next_filters = self.filter_sizes[level + 1]
                downsample_ops.append(
                    keras.layers.Conv2D(
                        filters=next_filters,
                        kernel_size=2,
                        strides=2,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name=f'enc_down_{level}'
                    )
                )

                self.encoder_downsamples.append(
                    keras.Sequential(
                        downsample_ops,
                        name=f'downsample_seq_{level}'
                    )
                )

    def _build_bottleneck(self) -> None:
        """Create bottleneck stage at deepest level."""
        bottleneck_filters = self.filter_sizes[self.depth]
        prev_filters = self.filter_sizes[self.depth - 1]

        # Entry to bottleneck: downsample + channel adjust
        bn_ops = []
        bn_ops.append(
            keras.layers.Conv2D(
                filters=bottleneck_filters,
                kernel_size=2,
                strides=2,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='bn_down'
            )
        )

        self.bottleneck_entry = keras.Sequential(bn_ops, name='bottleneck_entry')

        # Bottleneck processing blocks
        self.bottleneck_blocks = []
        for block_idx in range(self.blocks_per_level):
            self.bottleneck_blocks.append(
                self.ConvNextBlock(
                    kernel_size=self.block_kernel_size,
                    filters=bottleneck_filters,
                    activation='gelu',
                    use_bias=self.use_bias,
                    dropout_rate=self.drop_path_rate,
                    name=f'bn_blk_{block_idx}'
                )
            )

    def _build_decoder(self) -> None:
        """Create decoder stages with upsampling and skip connections."""
        self.decoder_upsamples = []
        self.decoder_blocks = []

        for level in range(self.depth - 1, -1, -1):
            current_filters = self.filter_sizes[level]

            # Upsampling layer
            self.decoder_upsamples.append(
                keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation='bilinear',
                    name=f'dec_up_{level}'
                )
            )

            # Processing blocks after skip concatenation
            stage_blocks = []

            # Channel adjustment after concatenation
            stage_blocks.append(
                keras.layers.Conv2D(
                    filters=current_filters,
                    kernel_size=1,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'dec_L{level}_adjust'
                )
            )

            # ConvNeXt blocks for this decoder stage
            for block_idx in range(self.blocks_per_level):
                # Compute drop path rate (mirrors encoder)
                current_drop_path = (
                        self.drop_path_rate
                        * (level * self.blocks_per_level + block_idx)
                        / (self.depth * self.blocks_per_level)
                )

                stage_blocks.append(
                    self.ConvNextBlock(
                        kernel_size=self.block_kernel_size,
                        filters=current_filters,
                        activation='gelu',
                        use_bias=self.use_bias,
                        dropout_rate=current_drop_path,
                        name=f'dec_L{level}_blk_{block_idx}'
                    )
                )

            self.decoder_blocks.append(stage_blocks)

    def _build_deep_supervision(self) -> None:
        """
        Create deep supervision output heads at intermediate decoder stages.

        CRITICAL: Always creates prediction heads with output_channels to ensure
        weight compatibility between include_top=True and include_top=False modes.
        The include_top flag controls whether these are used in forward pass.
        """
        self.supervision_heads = []

        # Create supervision heads for all but the final (highest resolution) level
        for level in range(self.depth - 1, 0, -1):
            # Always create prediction heads for weight compatibility
            head = keras.Sequential([
                keras.layers.Conv2D(
                    filters=self.filter_sizes[level] // 2,
                    kernel_size=1,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer
                ),
                GlobalResponseNormalization(),
                keras.layers.Activation('gelu'),
                keras.layers.Conv2D(
                    filters=self.output_channels,
                    kernel_size=1,
                    activation=self.final_activation,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer
                )
            ], name=f'deep_sup_L{level}')

            self.supervision_heads.append(head)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build model by creating weights for all sub-layers.

        This method explicitly builds all sub-layers in the correct order,
        tracking shape transformations through the network to ensure proper
        weight variable creation before weight loading during deserialization.

        Parameters
        ----------
        input_shape : tuple of int or None
            Shape of input tensor (batch_size, height, width, channels).
        """
        # Parse input shape
        shape = tuple(input_shape)
        if len(shape) == 4:
            b, h, w, c = shape
        else:
            h, w, c = self.input_shape_config

        current_shape = (None, h, w, c)

        # Build stem
        self.stem.build(current_shape)
        current_shape = self.stem.compute_output_shape(current_shape)

        # Track skip connection shapes
        skip_shapes = [current_shape]

        # Build encoder stages
        for level in range(self.depth):
            # Build ConvNeXt blocks for this stage
            for layer in self.encoder_stages[level]:
                layer.build(current_shape)
                current_shape = layer.compute_output_shape(current_shape)

            # Store skip shape before downsampling
            skip_shapes[level] = current_shape

            # Build downsampling (if not last encoder stage)
            if level < self.depth - 1:
                self.encoder_downsamples[level].build(current_shape)
                current_shape = self.encoder_downsamples[level].compute_output_shape(
                    current_shape
                )
                skip_shapes.append(current_shape)

        # Build bottleneck
        self.bottleneck_entry.build(current_shape)
        current_shape = self.bottleneck_entry.compute_output_shape(current_shape)

        for layer in self.bottleneck_blocks:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        # Build decoder stages
        levels_rev = list(range(self.depth - 1, -1, -1))

        for idx, level in enumerate(levels_rev):
            # Build upsampling
            self.decoder_upsamples[idx].build(current_shape)
            current_shape = self.decoder_upsamples[idx].compute_output_shape(
                current_shape
            )

            # Simulate concatenation with skip connection
            skip_shape = skip_shapes[level]
            concat_channels = current_shape[-1] + skip_shape[-1]
            current_shape = (
                current_shape[0],
                current_shape[1],
                current_shape[2],
                concat_channels
            )

            # Build decoder blocks
            for layer in self.decoder_blocks[idx]:
                layer.build(current_shape)
                current_shape = layer.compute_output_shape(current_shape)

            # Build deep supervision head if enabled
            if self.enable_deep_supervision and idx < len(self.supervision_heads):
                self.supervision_heads[idx].build(current_shape)

        # CRITICAL: Always build final output layer for weight compatibility
        self.final_output_layer.build(current_shape)

        # Mark model as built
        super().build(input_shape)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """
        Compute output shape(s) for the model.

        Parameters
        ----------
        input_shape : tuple of int or None
            Shape of input tensor (batch_size, height, width, channels).

        Returns
        -------
        tuple or list of tuples
            If include_top=True and enable_deep_supervision=False:
                Single tuple (batch_size, height, width, output_channels).
            If include_top=True and enable_deep_supervision=True:
                List of tuples [main_shape, aux_shape_1, ..., aux_shape_n].
            If include_top=False and enable_deep_supervision=False:
                Single tuple (batch_size, height, width, decoder_filters).
            If include_top=False and enable_deep_supervision=True:
                List of tuples with feature shapes at multiple scales.
        """
        # Parse input shape
        if len(input_shape) == 4:
            batch_size, h, w, c = input_shape
        else:
            # Handle cases where input_shape might not include batch dimension explicitly
            # though Keras layers usually receive inputs with batch dim
            batch_size = input_shape[0] if len(input_shape) > 0 else None
            h, w = self.input_shape_config[0], self.input_shape_config[1]

        # Helper to compute spatial dimension at specific level
        def get_dim_at_level(dim: Optional[int], level: int) -> Optional[int]:
            if dim is None:
                return None
            # Each level represents a factor of 2 downsampling relative to input
            return dim // (2 ** level)

        # Determine output channels based on include_top
        if self.include_top:
            out_channels = self.output_channels
        else:
            # Output decoder features (highest resolution decoder features)
            out_channels = self.filter_sizes[0]

        # Main output is at level 0 (full resolution)
        main_output_shape = (
            batch_size,
            h,
            w,
            out_channels
        )

        if not self.enable_deep_supervision:
            return main_output_shape

        # Calculate auxiliary shapes for deep supervision
        # Corresponds to levels: depth-1 (coarsest) down to 1 (finest aux)
        aux_output_shapes = []
        for level in range(self.depth - 1, 0, -1):
            if self.include_top:
                aux_channels = self.output_channels
            else:
                aux_channels = self.filter_sizes[level]

            shape = (
                batch_size,
                get_dim_at_level(h, level),
                get_dim_at_level(w, level),
                aux_channels
            )
            aux_output_shapes.append(shape)

        # The call() method returns: [main] + list(reversed(aux_outputs))
        # aux_outputs are collected from coarse (level depth-1) to fine (level 1)
        # Reversing them puts them in order: Level 1 (fine) -> Level depth-1 (coarse)
        return [main_output_shape] + list(reversed(aux_output_shapes))

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, List[keras.KerasTensor]]:
        """
        Forward pass of ConvUNext model.

        Implements U-Net-style encoder-decoder with skip connections. If deep
        supervision is enabled, returns multiple outputs at different scales.
        The include_top parameter controls whether final prediction layers are applied.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, height, width, channels).
        training : bool or None, optional
            Whether in training mode (affects dropout/normalization), by default None.

        Returns
        -------
        keras.KerasTensor or list of keras.KerasTensor
            If include_top=True and enable_deep_supervision=False:
                Single prediction tensor.
            If include_top=True and enable_deep_supervision=True:
                List of prediction tensors at multiple scales.
            If include_top=False and enable_deep_supervision=False:
                Single feature tensor from decoder.
            If include_top=False and enable_deep_supervision=True:
                List of feature tensors at multiple scales.
        """
        skip_connections = []
        deep_supervision_outputs = []

        # Stem
        x = self.stem(inputs, training=training)

        # Encoder with skip connections
        for level in range(self.depth):
            # Process through ConvNeXt blocks
            for layer in self.encoder_stages[level]:
                x = layer(x, training=training)

            # Save skip connection
            skip_connections.append(x)

            # Downsample (if not last stage)
            if level < self.depth - 1:
                x = self.encoder_downsamples[level](x, training=training)

        # Bottleneck
        x = self.bottleneck_entry(x, training=training)
        for layer in self.bottleneck_blocks:
            x = layer(x, training=training)

        # Decoder with skip connections
        for idx, level in enumerate(range(self.depth - 1, -1, -1)):
            # Upsample
            x = self.decoder_upsamples[idx](x)

            # Get skip connection
            skip = skip_connections[level]

            # Ensure spatial dimensions match (graph-safe resize if needed)
            x_shape = ops.shape(x)
            skip_shape = ops.shape(skip)

            # Check if resize is needed
            needs_resize = ops.logical_or(
                ops.not_equal(x_shape[1], skip_shape[1]),
                ops.not_equal(x_shape[2], skip_shape[2])
            )

            # Conditionally resize using ops.cond (graph-safe)
            def resize_fn() -> keras.KerasTensor:
                """Resize x to match skip dimensions."""
                return ops.image.resize(
                    x,
                    size=(skip_shape[1], skip_shape[2]),
                    interpolation='bilinear'
                )

            def identity_fn() -> keras.KerasTensor:
                """Return x unchanged."""
                return x

            x = ops.cond(needs_resize, resize_fn, identity_fn)

            # Concatenate with skip connection
            x = keras.layers.Concatenate(axis=-1)([skip, x])

            # Process through decoder blocks
            for layer in self.decoder_blocks[idx]:
                x = layer(x, training=training)

            # Deep supervision output (if enabled and not final level)
            if self.enable_deep_supervision and idx < len(self.supervision_heads):
                # Apply supervision head if include_top=True
                if self.include_top:
                    ds_out = self.supervision_heads[idx](x, training=training)
                else:
                    # Return features without supervision head for include_top=False
                    ds_out = x
                deep_supervision_outputs.append(ds_out)

        # Final output: apply projection head if include_top=True
        if self.include_top:
            final_output = self.final_output_layer(x)
        else:
            # Return decoder features without projection
            final_output = x

        # Return outputs
        if self.enable_deep_supervision:
            # Return [main_output, supervision_outputs from coarse to fine]
            return [final_output] + list(reversed(deep_supervision_outputs))

        return final_output

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all constructor parameters.
            Includes serialized initializers and regularizers.
        """
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_config,
            'depth': self.depth,
            'initial_filters': self.initial_filters,
            'filter_multiplier': self.filter_multiplier,
            'blocks_per_level': self.blocks_per_level,
            'convnext_version': self.convnext_version,
            'stem_kernel_size': self.stem_kernel_size,
            'block_kernel_size': self.block_kernel_size,
            'drop_path_rate': self.drop_path_rate,
            'final_activation': self.final_activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'include_top': self.include_top,
            'enable_deep_supervision': self.enable_deep_supervision,
            'output_channels': self.output_channels
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ConvUNextModel':
        """
        Create model from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from get_config().

        Returns
        -------
        ConvUNextModel
            Reconstructed model instance.
        """
        # Deserialize initializers and regularizers
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )
        if 'kernel_regularizer' in config:
            config['kernel_regularizer'] = regularizers.deserialize(
                config['kernel_regularizer']
            )
        return cls(**config)

    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = 'imagenet',
            cache_dir: Optional[str] = None
    ) -> str:
        """
        Download pretrained weights for a model variant.

        Parameters
        ----------
        variant : str
            Model variant name ('tiny', 'small', 'base', 'large', 'xlarge').
        dataset : str, optional
            Dataset name for pretrained weights, by default 'imagenet'.
        cache_dir : str or None, optional
            Directory to cache downloaded weights, by default None.

        Returns
        -------
        str
            Path to downloaded weights file.

        Raises
        ------
        ValueError
            If no pretrained weights available for variant.
        """
        if variant not in ConvUNextModel.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(ConvUNextModel.PRETRAINED_WEIGHTS.keys())}"
            )

        # Placeholder implementation
        logger.warning("Pretrained weight download not implemented")
        return "dummy_path.keras"

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 3),
            include_top: bool = True,
            enable_deep_supervision: bool = False,
            output_channels: Optional[int] = None,
            use_bias: bool = True,
            **kwargs: Any
    ) -> 'ConvUNextModel':
        """
        Create model from predefined variant configuration.

        Parameters
        ----------
        variant : str
            Model variant name. Must be one of: 'tiny', 'small', 'base',
            'large', 'xlarge'.
        input_shape : tuple of int, optional
            Input shape (height, width, channels), by default (None, None, 3).
        include_top : bool, optional
            Whether to use final prediction head in forward pass, by default True.
        enable_deep_supervision : bool, optional
            Whether to enable deep supervision, by default False.
        output_channels : int or None, optional
            Number of output channels (required when include_top=True for training),
            by default None.
        use_bias : bool, optional
            Whether to use bias in convolutions, by default True.
        **kwargs : Any
            Additional arguments to override variant defaults.

        Returns
        -------
        ConvUNextModel
            Model instance with variant configuration.

        Raises
        ------
        ValueError
            If variant is not recognized.

        Example
        -------
        ```python
        # Train model with predictions
        model = ConvUNextModel.from_variant(
            'base',
            input_shape=(256, 256, 3),
            output_channels=10,
            include_top=True
        )
        model.fit(...)
        model.save('trained.keras')

        # Reuse as feature extractor (same output_channels for weight loading)
        encoder = ConvUNextModel.from_variant(
            'base',
            input_shape=(256, 256, 3),
            output_channels=10,
            include_top=False
        )
        encoder.load_weights('trained.keras')
        ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available variants: {list(cls.MODEL_VARIANTS.keys())}"
            )

        # Get variant configuration
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop('description', None)  # Remove description field

        # Override with user-provided kwargs
        config.update(kwargs)

        # Set critical parameters
        config['include_top'] = include_top
        config['enable_deep_supervision'] = enable_deep_supervision
        config['use_bias'] = use_bias

        # Create model
        return cls(
            input_shape=input_shape,
            output_channels=output_channels,
            **config
        )


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_convunext_variant(
        variant: str,
        input_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 3),
        include_top: bool = True,
        enable_deep_supervision: bool = False,
        output_channels: Optional[int] = None,
        use_bias: bool = True,
        **kwargs: Any
) -> ConvUNextModel:
    """
    Factory function to create ConvUNext model from variant name.

    This is a convenience wrapper around ConvUNextModel.from_variant().

    Parameters
    ----------
    variant : str
        Model variant name ('tiny', 'small', 'base', 'large', 'xlarge').
    input_shape : tuple of int, optional
        Input shape (height, width, channels), by default (None, None, 3).
    include_top : bool, optional
        Whether to use final prediction head in forward pass, by default True.
    enable_deep_supervision : bool, optional
        Whether to enable deep supervision, by default False.
    output_channels : int or None, optional
        Number of output channels, by default None.
    use_bias : bool, optional
        Whether to use bias in convolutions, by default True.
    **kwargs : Any
        Additional arguments to override variant defaults.

    Returns
    -------
    ConvUNextModel
        Configured model instance.

    Example
    -------
    ```python
    # Create full model for segmentation
    model = create_convunext_variant(
        'base',
        input_shape=(256, 256, 3),
        output_channels=1,
        final_activation='sigmoid',
        use_bias=True
    )

    # Create feature extractor
    encoder = create_convunext_variant(
        'base',
        input_shape=(256, 256, 3),
        output_channels=1,  # Must match for weight loading
        include_top=False
    )
    ```
    """
    return ConvUNextModel.from_variant(
        variant=variant,
        input_shape=input_shape,
        include_top=include_top,
        enable_deep_supervision=enable_deep_supervision,
        output_channels=output_channels,
        use_bias=use_bias,
        **kwargs
    )


# ---------------------------------------------------------------------

def create_inference_model_from_training_model(
        training_model: ConvUNextModel,
        disable_deep_supervision: bool = True
) -> ConvUNextModel:
    """
    Create inference model from training model.

    This function converts a training model to an inference model by:
    1. Optionally disabling deep supervision
    2. Extracting the configuration
    3. Creating a new model
    4. Transferring weights

    IMPORTANT: For weight compatibility, output_channels must match between
    training and inference models.

    Parameters
    ----------
    training_model : ConvUNextModel
        Training model (potentially with enable_deep_supervision=True).
    disable_deep_supervision : bool, optional
        Whether to disable deep supervision in inference model, by default True.

    Returns
    -------
    ConvUNextModel
        Inference model with transferred weights.
        If training_model already has the desired configuration, returns it unchanged.

    Example
    -------
    ```python
    # Train with deep supervision
    train_model = ConvUNextModel.from_variant(
        'base',
        enable_deep_supervision=True,
        output_channels=1
    )
    train_model.compile(...)
    train_model.fit(...)

    # Convert to inference model (no deep supervision)
    inference_model = create_inference_model_from_training_model(train_model)
    inference_model.save('final_model.keras')
    ```
    """
    # If already in desired configuration, return as-is
    if disable_deep_supervision and not training_model.enable_deep_supervision:
        logger.info("Model already has deep supervision disabled")
        return training_model

    if not disable_deep_supervision:
        logger.info("Model configuration unchanged")
        return training_model

    # Extract configuration and modify as needed
    config = training_model.get_config()

    if disable_deep_supervision:
        config['enable_deep_supervision'] = False

    # Create new inference model
    inference_model = ConvUNextModel.from_config(config)

    # Build inference model with same input shape as training model
    dummy_shape = (1,) + tuple(training_model.input_shape_config)
    inference_model.build(dummy_shape)

    # Transfer weights (skip_mismatch handles deep supervision heads)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as f:
        temp_path = f.name

    try:
        training_model.save_weights(temp_path)
        inference_model.load_weights(
            filepath=temp_path,
            skip_mismatch=True
        )
        logger.info("Successfully transferred weights to inference model")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return inference_model

# ---------------------------------------------------------------------