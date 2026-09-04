"""
PFTSR performs single-image super-resolution with a chain of windowed
transformer blocks and pixel-shuffle upsampling.

Each block multiplies the previous block's attention map into its own raw
attention scores before the softmax, so a token pair a lower layer ruled out
stays suppressed rather than being recomputed from scratch. Windows keep
attention cost linear in image size rather than quadratic, using the same
window-and-shift scheme as Swin Transformer.

The published method also skips attention for pairs the inherited map has
ruled out. This implementation always runs dense attention: any
``sparsity_mode`` other than ``'none'`` raises ``NotImplementedError`` at
construction. Height and width must be divisible by ``window_size`` and must
be statically known (no ``None`` dims), because the shifted-window mask is
built once at build time.

References:
    - Long et al., 2025. Progressive Focused Transformer for Single Image
      Super-Resolution. CVPR. (https://arxiv.org/abs/2503.20337)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. ICCV. (https://arxiv.org/abs/2103.14030)
    - Liang et al., 2021. SwinIR: Image Restoration Using Swin Transformer.
      (https://arxiv.org/abs/2108.10257)
    - Shi et al., 2016. Real-Time Single Image and Video Super-Resolution Using an
      Efficient Sub-Pixel Convolutional Neural Network.
      (https://arxiv.org/abs/1609.05158)
    - Dong et al., 2022. CSWin Transformer: A General Vision Transformer Backbone
      with Cross-Shaped Windows. (https://arxiv.org/abs/2107.00652)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from typing import Any, Dict, Optional, List, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.pooling.pixel_unshuffle import PixelShuffle2D
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.transformers.progressive_focused_transformer import PFTBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.pft_sr.model")
class PFTSR(keras.Model):
    """
    Progressive Focused Transformer for single image super-resolution.

    A Keras port of Long et al.'s PFT-SR (CVPR 2025). It has not been
    trained or benchmarked in this codebase, so no performance claim is made
    for it here.

    Architecture:

    .. code-block:: text

        x [B, H, W, C]
          |
          v
        +----------------+
        |  conv_first    |  3x3, shallow features
        +----------------+
          |----------------------------------+
          v                                  |
        +-----------------------------+      |
        |  PFT block stages           |      |
        |  (attention map carried     |      |
        |   forward across stages)    |      |
        +-----------------------------+      |
          |                                  |
          v                                  |
        +------------------+                 |
        |  conv_after_body |                 |
        +------------------+                 |
          |                                  |
          v                                  |
         (+) <-------------------------------+   long skip
          |
          v
        +----------------+
        |  upsample      |  pixelshuffle / pixelshuffledirect / nearest+conv
        +----------------+
          |
          v
        +----------------+
        |  conv_last     |  3x3
        +----------------+
          |
          v
        output [B, H*scale, W*scale, C]

    Named variants:

    .. code-block:: text

        variant       embed_dim  num_blocks       heads  mlp_ratio  window
        light         52         [2,4,6,6,6]      4      1.0        32
        base          240        [4,4,4,6,6,6]    6      2.0        32
        repo_medium   80         [6,6,6,8,8,8]    8      2.0        8

    :param scale: Upsampling scale factor (2, 3, or 4).
    :type scale: int
    :param in_channels: Number of input image channels.
    :type in_channels: int
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param num_blocks: Number of PFT blocks in each stage.
    :type num_blocks: list[int]
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param window_size: Attention window size. The ``light`` and ``base``
        variants override this to 32; 8 is only the constructor's own
        fallback.
    :type window_size: int
    :param mlp_ratio: Expansion ratio for the block's MLP.
    :type mlp_ratio: float
    :param qkv_bias: Whether to use bias in the QKV projections.
    :type qkv_bias: bool
    :param attention_dropout_rate: Dropout rate applied to attention weights.
    :type attention_dropout_rate: float
    :param projection_dropout_rate: Dropout rate applied to output projections.
    :type projection_dropout_rate: float
    :param drop_path_rate: Stochastic depth rate.
    :type drop_path_rate: float
    :param norm_type: Normalization type, ``'layer_norm'`` or ``'rms_norm'``.
    :type norm_type: str
    :param use_lepe: Whether to add locally-enhanced positional encoding.
    :type use_lepe: bool
    :param upsampler: Upsampling method: ``'pixelshuffle'``,
        ``'pixelshuffledirect'``, or ``'nearest+conv'``.
    :type upsampler: str
    :param kwargs: Additional keyword arguments for the Keras ``Model`` base class.

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
        >>> # The paper's PFT_light config, spelled out explicitly
        >>> model_light = PFTSR(scale=4, embed_dim=52, num_blocks=[2, 4, 6, 6, 6],
        ...                     num_heads=4, mlp_ratio=1.0, window_size=32)
        >>> sr_image_light = model_light(lr_image)
    """

    # 'light' and 'base' mirror the paper's two released training configs;
    # window_size=32 matches those published models, not the constructor default.
    # 'repo_medium' has no published counterpart.
    # DECISION plan-2026-08-23T091307-9a110062/D-463: keep window_size explicit per
    # variant here; collapsing it to the constructor default silently changes the architecture.
    # See decisions.md.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # PFT_light, 101_PFT_light_SRx2_scratch.yml network_g
        'light': {
            'embed_dim': 52,
            'num_blocks': [2, 4, 6, 6, 6],
            'num_heads': 4,
            'mlp_ratio': 1.0,
            'window_size': 32,
        },
        # PFT, 001_PFT_SRx2_scratch.yml network_g
        'base': {
            'embed_dim': 240,
            'num_blocks': [4, 4, 4, 6, 6, 6],
            'num_heads': 6,
            'mlp_ratio': 2.0,
            'window_size': 32,
        },
        # Repo-original. Not in the paper, not in the official repo, not a rung
        # above ``base``. Kept because it is a cheap mid-size model, not because
        # anything published looks like it.
        'repo_medium': {
            'embed_dim': 80,
            'num_blocks': [6, 6, 6, 8, 8, 8],
            'num_heads': 8,
            'mlp_ratio': 2.0,
            'window_size': 8,
        }
    }

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
            attention_dropout_rate: float = 0.0,
            projection_dropout_rate: float = 0.0,
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
        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.norm_type = norm_type
        self.use_lepe = use_lepe
        self.upsampler = upsampler

        # Calculate total number of blocks for stochastic depth
        self.total_blocks = sum(num_blocks)

        # linear_drop_path_rates returns plain floats; a tensor-based linspace
        # raised AttributeError under TF eager and left drop_path_rate > 0 dead.
        self.dpr = linear_drop_path_rates(self.total_blocks, drop_path_rate)

    def build(self, input_shape):
        """
        Build model layers.

        :param input_shape: Shape tuple of the input.
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
                    attention_dropout_rate=self.attention_dropout_rate,
                    projection_dropout_rate=self.projection_dropout_rate,
                    drop_path_rate=self.dpr[block_idx],
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

        # Force sublayers to build now: build_from_config reloads them unbuilt,
        # and weight loading fails on layers "never built" without this step.
        if all(d is not None for d in input_shape[1:]):
            dummy = keras.ops.zeros((1,) + tuple(input_shape[1:]))
            self.call(dummy, training=False)

        super().build(input_shape)

    def _build_pixelshuffle_upsampler(self) -> keras.Sequential:
        """
        Build pixel shuffle upsampler.

        :return: Sequential model for upsampling.
        :rtype: keras.Sequential
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
            # DECISION plan_2026-06-15_39a31d4a/D-003: use PixelShuffle2D, not
            # keras.ops.nn.depth_to_space (absent in Keras 3.8) or a Lambda wrapping it.
            # See decisions.md.
            layers.append(
                PixelShuffle2D(block_size=self.scale, name="pixel_shuffle")
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
                PixelShuffle2D(block_size=2, name="pixel_shuffle1")
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
                PixelShuffle2D(block_size=2, name="pixel_shuffle2")
            )
        else:
            raise ValueError(f"Unsupported scale: {self.scale}")

        return keras.Sequential(layers, name="upsampler")

    def _build_pixelshuffledirect_upsampler(self) -> keras.Sequential:
        """
        Build direct pixel shuffle upsampler.

        :return: Sequential model for upsampling.
        :rtype: keras.Sequential
        """
        layers = [
            keras.layers.Conv2D(
                self.embed_dim * (self.scale ** 2),
                kernel_size=3,
                strides=1,
                padding='same',
                name="upsample_conv"
            ),
            PixelShuffle2D(block_size=self.scale, name="pixel_shuffle")
        ]

        return keras.Sequential(layers, name="upsampler")

    def _build_nearest_upsampler(self) -> keras.Sequential:
        """
        Build nearest neighbor + conv upsampler.

        Only power-of-two scales work here: the stage count is
        ``int(log2(scale))`` doubling stages, so any other scale emits the
        wrong resolution (``scale=3`` gives one stage, a 2x output for a 3x
        request).

        :return: Sequential model for upsampling.
        :rtype: keras.Sequential
        :raises ValueError: If ``scale`` is not a power of two.
        """
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-078: reject non-power-of-two
        # scales here instead of silently under-upsampling. See decisions.md.
        if self.scale < 1 or (self.scale & (self.scale - 1)) != 0:
            raise ValueError(
                f"Unsupported scale for upsampler='nearest+conv': {self.scale}. "
                f"This upsampler stacks int(log2(scale)) doubling stages, so it can "
                f"only realize powers of two (1, 2, 4, 8, ...); scale={self.scale} "
                f"would silently emit a "
                f"{2 ** max(self.scale.bit_length() - 1, 0) if self.scale > 0 else 0}x "
                f"image. "
                f"Use upsampler='pixelshuffle' (supports 2, 3 and 4) instead."
            )

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

        :param inputs: Input low-resolution images of shape
            (batch_size, height, width, channels).
        :type inputs: keras.KerasTensor
        :param training: Whether the call runs in training mode.
        :type training: bool or None
        :return: Super-resolved high-resolution images.
        :rtype: keras.KerasTensor
        """
        # 1. Shallow feature extraction
        x = self.conv_first(inputs)
        residual = x

        # 2. Deep feature extraction with progressive focused attention
        prev_attn_map = None

        for stage_blocks in self.stages:
            for block in stage_blocks:
                # DECISION plan_2026-06-15_39a31d4a/D-003: the first block gets a bare
                # tensor (no prior map); later blocks get [x, prev_attn_map] as a list,
                # never a tuple, or Keras' shape machinery misreads the input. See decisions.md.
                if prev_attn_map is None:
                    x, prev_attn_map = block(x, training=training)
                else:
                    x, prev_attn_map = block(
                        [x, prev_attn_map], training=training
                    )

        # 3. Reconstruction
        x = self.conv_after_body(x)
        # Global residual connection.
        x = x + residual

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
            "attention_dropout_rate": self.attention_dropout_rate,
            "projection_dropout_rate": self.projection_dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "norm_type": self.norm_type,
            "use_lepe": self.use_lepe,
            "upsampler": self.upsampler,
        })
        return config


def create_pft_sr(
        scale: int = 4,
        variant: Literal['base', 'light', 'repo_medium'] = 'base',
        **kwargs: Any,
) -> PFTSR:
    """
    Build a PFT-SR model from a named variant.

    :param scale: Upsampling scale factor (2, 3, or 4).
    :type scale: int
    :param variant: Model variant: ``'light'`` and ``'base'`` mirror the
        paper's two released configs (from its official training YAMLs);
        ``'repo_medium'`` is a repo-original mid-size tier with no
        published counterpart.
    :type variant: str
    :param kwargs: Any :class:`PFTSR` constructor argument, overriding the
        variant table. This is the only route to ``window_size``,
        ``drop_path_rate``, ``upsampler``, ``norm_type``, ``use_lepe`` and
        the dropout rates.
    :return: Configured :class:`PFTSR` instance.
    :rtype: PFTSR

    Example:
        >>> # Create base model for 4x SR
        >>> model = create_pft_sr(scale=4, variant='base')
        >>>
        >>> # Create lightweight model for 2x SR
        >>> model_light = create_pft_sr(scale=2, variant='light')
        >>>
        >>> # Create the repo-original mid-size model for 4x SR
        >>> model_medium = create_pft_sr(scale=4, variant='repo_medium')
    """
    if variant not in PFTSR.MODEL_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Available variants: {list(PFTSR.MODEL_VARIANTS.keys())}"
        )

    # DECISION plan-2026-08-19T163559-499b6f0e/D-118: copy the variant dict before
    # updating with kwargs, or a caller's override mutates PFTSR.MODEL_VARIANTS itself.
    # See decisions.md.
    config = PFTSR.MODEL_VARIANTS[variant].copy()
    config['num_blocks'] = list(config['num_blocks'])
    config.update(kwargs)

    return PFTSR(scale=scale, **config)
