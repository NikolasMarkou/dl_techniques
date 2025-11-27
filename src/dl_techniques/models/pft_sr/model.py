"""
Progressive Focused Transformer for Single Image Super-Resolution (PFT-SR).

This module implements the complete PFT-SR architecture for image super-resolution,
combining shallow feature extraction, deep feature extraction with PFT blocks,
and high-quality image reconstruction.

References:
    Long, Wei, et al. "Progressive Focused Transformer for Single Image Super-Resolution."
    CVPR 2025. https://arxiv.org/abs/2503.20337
"""

import keras
from typing import Optional, List, Literal
from dl_techniques.layers.transformers.progressive_focused_transformer_block import PFTBlock


@keras.saving.register_keras_serializable()
class PFTSR(keras.Model):
    """
    Progressive Focused Transformer for Single Image Super-Resolution.

    A state-of-the-art transformer-based super-resolution model that achieves
    excellent performance through progressive focused attention mechanism.

    The architecture consists of:
    1. Shallow feature extraction (single conv layer)
    2. Deep feature extraction (multiple stages of PFT blocks)
    3. Reconstruction module (conv + pixel shuffle upsampling)

    Key innovations:
    - Progressive Focused Attention (PFA) that inherits attention maps across layers
    - Windowed attention with shifted windows for efficient computation
    - LePE (Locally-Enhanced Positional Encoding) for better spatial modeling

    Args:
        scale: Integer, upsampling scale factor (2, 3, or 4). Default: 4.
        in_channels: Integer, number of input image channels. Default: 3.
        embed_dim: Integer, embedding dimension. Default: 60.
        num_blocks: List of integers, number of PFT blocks in each stage.
            Default: [4, 4, 4, 6, 6, 6].
        num_heads: Integer, number of attention heads. Default: 6.
        window_size: Integer, size of the attention window. Default: 8.
        mlp_ratio: Float, expansion ratio for MLP. Default: 2.0.
        qkv_bias: Boolean, whether to use bias in QKV projections. Default: True.
        attention_dropout: Float, dropout rate for attention. Default: 0.0.
        projection_dropout: Float, dropout rate for projections. Default: 0.0.
        drop_path_rate: Float, stochastic depth rate. Default: 0.0.
        norm_type: String, normalization type ('layer_norm' or 'rms_norm'). Default: 'layer_norm'.
        use_lepe: Boolean, whether to use LePE. Default: True.
        upsampler: String, upsampling method ('pixelshuffle', 'pixelshuffledirect',
            or 'nearest+conv'). Default: 'pixelshuffle'.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, in_channels)`.
        Height and width should be divisible by window_size.

    Output shape:
        4D tensor with shape: `(batch_size, height * scale, width * scale, in_channels)`.

    Example:
        >>> import keras
        >>> # Create PFT-SR model for 4x super-resolution
        >>> model = PFTSR(scale=4, embed_dim=60, num_blocks=[4, 4, 4, 6, 6, 6])
        >>>
        >>> # Low-resolution input (48x48)
        >>> lr_image = keras.random.normal((1, 48, 48, 3))
        >>>
        >>> # Super-resolve to high-resolution (192x192)
        >>> sr_image = model(lr_image)
        >>> print(sr_image.shape)
        (1, 192, 192, 3)
        >>>
        >>> # Create lightweight variant
        >>> model_light = PFTSR(scale=4, embed_dim=48, num_blocks=[4, 4, 4, 4])
        >>> sr_image_light = model_light(lr_image)
    """

    def __init__(
            self,
            scale: int = 4,
            in_channels: int = 3,
            embed_dim: int = 60,
            num_blocks: List[int] = None,
            num_heads: int = 6,
            window_size: int = 8,
            mlp_ratio: float = 2.0,
            qkv_bias: bool = True,
            attention_dropout: float = 0.0,
            projection_dropout: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
            use_lepe: bool = True,
            upsampler: Literal['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'] = 'pixelshuffle',
            **kwargs
    ):
        super().__init__(**kwargs)

        if num_blocks is None:
            num_blocks = [4, 4, 4, 6, 6, 6]

        self.scale = scale
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.drop_path_rate = drop_path_rate
        self.norm_type = norm_type
        self.use_lepe = use_lepe
        self.upsampler = upsampler

        # Calculate total number of blocks for stochastic depth
        self.total_blocks = sum(num_blocks)

        # Stochastic depth decay rule
        if drop_path_rate > 0.0:
            dpr = [
                x.item()
                for x in keras.ops.linspace(0.0, drop_path_rate, self.total_blocks)
            ]
        else:
            dpr = [0.0] * self.total_blocks

        self.dpr = dpr

    def build(self, input_shape):
        """
        Build model layers.

        Args:
            input_shape: Shape tuple of the input.
        """
        # 1. Shallow feature extraction
        self.conv_first = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_first"
        )

        # 2. Deep feature extraction with PFT blocks
        self.stages = []
        block_idx = 0

        for stage_idx, num_blocks_in_stage in enumerate(self.num_blocks):
            stage_blocks = []

            for block_idx_in_stage in range(num_blocks_in_stage):
                # Alternate between regular and shifted window attention
                shift_size = 0 if (block_idx_in_stage % 2 == 0) else self.window_size // 2

                block = PFTBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    attention_dropout=self.attention_dropout,
                    projection_dropout=self.projection_dropout,
                    drop_path=self.dpr[block_idx],
                    norm_type=self.norm_type,
                    use_lepe=self.use_lepe,
                    name=f"stage{stage_idx}_block{block_idx_in_stage}"
                )
                stage_blocks.append(block)
                block_idx += 1

            self.stages.append(stage_blocks)

        # 3. Reconstruction
        self.conv_after_body = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_after_body"
        )

        # 4. Upsampling
        if self.upsampler == 'pixelshuffle':
            # Traditional upsampling with pixel shuffle
            self.upsample = self._build_pixelshuffle_upsampler()
        elif self.upsampler == 'pixelshuffledirect':
            # Direct pixel shuffle
            self.upsample = self._build_pixelshuffledirect_upsampler()
        else:  # nearest+conv
            self.upsample = self._build_nearest_upsampler()

        # 5. Final reconstruction
        self.conv_last = keras.layers.Conv2D(
            filters=self.in_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_last"
        )

        super().build(input_shape)

    def _build_pixelshuffle_upsampler(self) -> keras.Sequential:
        """
        Build pixel shuffle upsampler.

        Returns:
            Sequential model for upsampling.
        """
        layers = []

        if self.scale == 2 or self.scale == 3:
            layers.append(
                keras.layers.Conv2D(
                    self.embed_dim * (self.scale ** 2),
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name=f"upsample_conv"
                )
            )
            layers.append(
                keras.layers.Lambda(
                    lambda x: keras.ops.nn.depth_to_space(x, self.scale),
                    name="pixel_shuffle"
                )
            )
        elif self.scale == 4:
            # 4x = 2x + 2x
            layers.append(
                keras.layers.Conv2D(
                    self.embed_dim * 4,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name="upsample_conv1"
                )
            )
            layers.append(
                keras.layers.Lambda(
                    lambda x: keras.ops.nn.depth_to_space(x, 2),
                    name="pixel_shuffle1"
                )
            )
            layers.append(
                keras.layers.Conv2D(
                    self.embed_dim * 4,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name="upsample_conv2"
                )
            )
            layers.append(
                keras.layers.Lambda(
                    lambda x: keras.ops.nn.depth_to_space(x, 2),
                    name="pixel_shuffle2"
                )
            )
        else:
            raise ValueError(f"Unsupported scale: {self.scale}")

        return keras.Sequential(layers, name="upsampler")

    def _build_pixelshuffledirect_upsampler(self) -> keras.Sequential:
        """
        Build direct pixel shuffle upsampler.

        Returns:
            Sequential model for upsampling.
        """
        layers = [
            keras.layers.Conv2D(
                self.embed_dim * (self.scale ** 2),
                kernel_size=3,
                strides=1,
                padding='same',
                name="upsample_conv"
            ),
            keras.layers.Lambda(
                lambda x: keras.ops.nn.depth_to_space(x, self.scale),
                name="pixel_shuffle"
            )
        ]

        return keras.Sequential(layers, name="upsampler")

    def _build_nearest_upsampler(self) -> keras.Sequential:
        """
        Build nearest neighbor + conv upsampler.

        Returns:
            Sequential model for upsampling.
        """
        layers = []

        for i in range(int(keras.ops.log2(self.scale))):
            layers.extend([
                keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
                keras.layers.Conv2D(
                    self.embed_dim,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name=f"upsample_conv{i}"
                )
            ])

        return keras.Sequential(layers, name="upsampler")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of PFT-SR.

        Args:
            inputs: Input low-resolution images of shape (batch_size, height, width, channels).
            training: Boolean or boolean tensor, whether in training mode.

        Returns:
            Super-resolved high-resolution images.
        """
        # 1. Shallow feature extraction
        x = self.conv_first(inputs)
        residual = x

        # 2. Deep feature extraction with progressive focused attention
        prev_attn_map = None

        for stage_blocks in self.stages:
            for block in stage_blocks:
                x, prev_attn_map = block((x, prev_attn_map), training=training)

        # 3. Reconstruction
        x = self.conv_after_body(x)
        x = x + residual  # Global residual connection

        # 4. Upsampling
        x = self.upsample(x)

        # 5. Final reconstruction
        output = self.conv_last(x)

        return output

    def get_config(self):
        """Return model configuration."""
        config = super().get_config()
        config.update({
            "scale": self.scale,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "attention_dropout": self.attention_dropout,
            "projection_dropout": self.projection_dropout,
            "drop_path_rate": self.drop_path_rate,
            "norm_type": self.norm_type,
            "use_lepe": self.use_lepe,
            "upsampler": self.upsampler,
        })
        return config


def create_pft_sr(
        scale: int = 4,
        variant: Literal['base', 'light', 'large'] = 'base'
) -> PFTSR:
    """
    Factory function to create PFT-SR models with predefined configurations.

    Args:
        scale: Integer, upsampling scale factor (2, 3, or 4).
        variant: String, model variant:
            - 'light': Lightweight model (48 channels, [4, 4, 4, 4] blocks)
            - 'base': Base model (60 channels, [4, 4, 4, 6, 6, 6] blocks)
            - 'large': Large model (80 channels, [6, 6, 6, 8, 8, 8] blocks)

    Returns:
        PFTSR model instance.

    Example:
        >>> # Create base model for 4x SR
        >>> model = create_pft_sr(scale=4, variant='base')
        >>>
        >>> # Create lightweight model for 2x SR
        >>> model_light = create_pft_sr(scale=2, variant='light')
        >>>
        >>> # Create large model for 4x SR
        >>> model_large = create_pft_sr(scale=4, variant='large')
    """
    configs = {
        'light': {
            'embed_dim': 48,
            'num_blocks': [4, 4, 4, 4],
            'num_heads': 6,
            'mlp_ratio': 2.0,
        },
        'base': {
            'embed_dim': 60,
            'num_blocks': [4, 4, 4, 6, 6, 6],
            'num_heads': 6,
            'mlp_ratio': 2.0,
        },
        'large': {
            'embed_dim': 80,
            'num_blocks': [6, 6, 6, 8, 8, 8],
            'num_heads': 8,
            'mlp_ratio': 2.0,
        }
    }

    if variant not in configs:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Available variants: {list(configs.keys())}"
        )

    config = configs[variant]

    return PFTSR(
        scale=scale,
        in_channels=3,
        embed_dim=config['embed_dim'],
        num_blocks=config['num_blocks'],
        num_heads=config['num_heads'],
        window_size=8,
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=True,
        attention_dropout=0.0,
        projection_dropout=0.0,
        drop_path_rate=0.0,
        norm_type='layer_norm',
        use_lepe=True,
        upsampler='pixelshuffle',
    )