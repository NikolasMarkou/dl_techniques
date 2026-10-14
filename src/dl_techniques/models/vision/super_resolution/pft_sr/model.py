"""
PFTSR performs single-image super-resolution with a chain of windowed
transformer blocks and pixel-shuffle upsampling. Each block multiplies the
previous block's attention map into its own raw scores before the softmax, so
a token pair a lower layer ruled out stays suppressed instead of being
recomputed. Windows keep attention cost linear in image size, using the same
window-and-shift scheme as Swin Transformer. The published method also skips
attention for pairs the inherited map has ruled out; this implementation always
runs dense attention, and any ``sparsity_mode`` other than ``'none'`` raises
``NotImplementedError`` at construction. Height and width must be divisible by
``window_size`` and must be statically known, with no ``None`` dims, because
the shifted-window mask is built once at build time.

The default output (``include_top=True``) ends in the official reconstruction
tail of CVL-UESTC/PFT-SR ``basicsr/archs/pft_arch.py``, selected by
``upsampler`` exactly as there: ``'pixelshuffle'`` (the PFT config:
``conv_before_upsample`` 3x3 to 64 + LeakyReLU(0.01), then a 3x3 conv and a
pixel shuffle per factor of 2, or one of each for 3, then ``conv_last``),
``'pixelshuffledirect'`` (the PFT_light config: one 3x3 conv to
``scale**2 * in_channels`` and one pixel shuffle) and ``'nearest+conv'`` (x4
only: two nearest x2 stages with LeakyReLU(0.2), ``conv_hr``, ``conv_last``).
The model builds this tail itself: the shared ``EnhancementHead`` upsamples
with a transposed conv and cannot express any of the three.
``include_top=False`` returns the backbone features the tail reads,
``conv_after_body(body(x)) + x``, at the input resolution.

The body is a simplified port, not the official code; the README lists the
differences (no patch-embed or final LayerNorm, no per-stage residual conv, an
MLP instead of the ConvFFN, dense instead of top-k attention, no RGB mean
shift).

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
from typing import Any, Dict, Optional, List, Literal, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.pooling.pixel_unshuffle import PixelShuffle2D
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.transformers.progressive_focused_transformer import PFTBlock
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: The three reconstruction tails of the official ``PFT`` class.
UPSAMPLERS: Tuple[str, ...] = ('pixelshuffle', 'pixelshuffledirect', 'nearest+conv')

#: The official tail's working width, ``num_feat = 64`` in ``PFT.__init__``,
#: used by ``'pixelshuffle'`` and ``'nearest+conv'`` whatever ``embed_dim`` is.
NUM_FEAT = 64

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.pft_sr.model")
class PFTSR(keras.Model):
    """
    Progressive Focused Transformer for single image super-resolution.

    A Keras port of Long et al.'s PFT-SR (CVPR 2025). Shallow features from a
    3x3 convolution pass through the PFT stages, which thread one attention map
    through every block, and a long skip carries the shallow features around
    the stages to the upsampler. The port has not been trained or benchmarked
    in this codebase, so no performance claim is made for it here.

    Architecture:

    .. code-block:: text

              x [B, H, W, in_channels]
                          │
                          ▼
                ┌───────────────────┐
                │    conv_first     │
                └───────────────────┘
                          │ [B, H, W, embed_dim]
            ┌─────────────┤
            │             ▼
            │ ┌───────────────────────┐
            │ │   pft block stages    │
            │ └───────────────────────┘
            │             │
            │             ▼
            │   ┌───────────────────┐
            │   │  conv_after_body  │
            │   └───────────────────┘
            │             │
            └─────────────┤
                          ▼
                         add ──────────► include_top=False returns this,
                          │              [B, H, W, embed_dim]
                          ▼
                ┌───────────────────┐
                │  upsampler tail   │  include_top=True, see below
                └───────────────────┘
                          │
                          ▼
        output [B, H*scale, W*scale, in_channels]

    Attention map chaining:

    .. code-block:: text

                  x
                  │
                  ▼
          ┌───────────────────────────┐
          │     block 0: block(x)     │
          └───────────────────────────┘
                  │             │
                  ▼             ▼
                  x         attn_map
                  │             │
                  ▼             ▼
          ┌───────────────────────────┐
          │ block 1: block([x, attn]) │
          └───────────────────────────┘
                  │             │
                  ▼             ▼
                  x         attn_map

    The map crosses stage boundaries; stages only group the blocks.

    Upsampler tails (the official ``PFT.forward`` branches; F = 64):

    .. code-block:: text

        upsampler           layers                                     scale
        pixelshuffle        conv_before_upsample 3x3 -> F, LReLU(0.01) 2^n, 3
                            (conv 3x3 -> 4F, shuffle 2) x n,
                            or conv 3x3 -> 9F, shuffle 3
                            conv_last 3x3 -> in_channels
        pixelshuffledirect  conv 3x3 -> scale^2 * in_channels,         any
                            shuffle scale
        nearest+conv        conv_before_upsample 3x3 -> F, LReLU(0.01) 4 only
                            (nearest 2x, conv 3x3 -> F, LReLU(0.2)) x 2
                            conv_hr 3x3 -> F, LReLU(0.2)
                            conv_last 3x3 -> in_channels

    Named variants:

    .. code-block:: text

        variant     embed_dim  num_blocks     heads  mlp_ratio  window  upsampler
        light       52         [2,4,6,6,6]    4      1.0        32      pixelshuffledirect
        base        240        [4,4,4,6,6,6]  6      2.0        32      pixelshuffle
        repo_medium 80         [6,6,6,8,8,8]  8      2.0        8       pixelshuffle

    `window` is each variant's own `window_size`, not the constructor default.

    :param scale: Upsampling scale factor. ``'pixelshuffle'`` takes a power
        of two or 3, ``'pixelshuffledirect'`` any positive integer,
        ``'nearest+conv'`` only 4.
    :type scale: int
    :param in_channels: Number of input image channels.
    :type in_channels: int
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param num_blocks: Number of PFT blocks in each stage. ``None`` means
        ``[4, 4, 4, 6, 6, 6]``.
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
    :param drop_path_rate: Stochastic depth rate. The per-block rates rise
        linearly from 0 to this value across all blocks.
    :type drop_path_rate: float
    :param norm_type: Normalization type, ``'layer_norm'`` or ``'rms_norm'``.
    :type norm_type: str
    :param use_lepe: Whether to add locally-enhanced positional encoding.
    :type use_lepe: bool
    :param upsampler: Which official reconstruction tail ``include_top``
        builds: ``'pixelshuffle'``, ``'pixelshuffledirect'`` or
        ``'nearest+conv'`` (see the table above). The ``light`` variant uses
        ``'pixelshuffledirect'`` and ``base`` ``'pixelshuffle'``, as their
        official configs do.
    :type upsampler: str
    :param include_top: When ``True`` (default), the model builds and applies
        the ``upsampler`` tail and outputs the super-resolved image. When
        ``False``, no tail is built and the model returns the backbone
        features at the input resolution. Defaults to True.
    :type include_top: bool
    :param kwargs: Additional keyword arguments for the Keras ``Model`` base class.
    :raises ValueError: At construction, if ``upsampler`` is not one of
        :data:`UPSAMPLERS`, or ``scale`` is one that upsampler does not
        support.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, in_channels)`.
        Height and width must be divisible by window_size.

    Output shape:
        - ``include_top=True``: 4D tensor with shape
          `(batch_size, height * scale, width * scale, in_channels)`.
        - ``include_top=False``: 4D tensor with shape
          `(batch_size, height, width, embed_dim)`, the features the tail
          reads.

    Example:
        >>> import keras
        >>> model = PFTSR(scale=4, embed_dim=60, num_blocks=[4, 4, 4, 6, 6, 6])
        >>> lr_image = keras.random.normal((1, 48, 48, 3))
        >>> sr_image = model(lr_image)
        >>> print(sr_image.shape)
        (1, 192, 192, 3)
        >>>
        >>> # The paper's PFT_light config, spelled out. Its window_size of 32
        >>> # needs a 64x64 input, since 48 is not divisible by 32.
        >>> model_light = PFTSR(scale=4, embed_dim=52, num_blocks=[2, 4, 6, 6, 6],
        ...                     num_heads=4, mlp_ratio=1.0, window_size=32)
        >>> sr_image_light = model_light(keras.random.normal((1, 64, 64, 3)))
    """

    # DECISION plan-2026-08-23T091307-9a110062/D-463: window_size stays explicit
    # per variant; the constructor default changes the architecture. See decisions.md.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # PFT_light, 101_PFT_light_SRx2_scratch.yml network_g
        'light': {
            'embed_dim': 52,
            'num_blocks': [2, 4, 6, 6, 6],
            'num_heads': 4,
            'mlp_ratio': 1.0,
            'window_size': 32,
            'upsampler': 'pixelshuffledirect',
        },
        # PFT, 001_PFT_SRx2_scratch.yml network_g
        'base': {
            'embed_dim': 240,
            'num_blocks': [4, 4, 4, 6, 6, 6],
            'num_heads': 6,
            'mlp_ratio': 2.0,
            'window_size': 32,
            'upsampler': 'pixelshuffle',
        },
        # Repo-original mid-size tier, with no published counterpart.
        'repo_medium': {
            'embed_dim': 80,
            'num_blocks': [6, 6, 6, 8, 8, 8],
            'num_heads': 8,
            'mlp_ratio': 2.0,
            'window_size': 8,
            'upsampler': 'pixelshuffle',
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
            include_top: bool = True,
            **kwargs
    ):
        """Validate the tail choice, store the configuration and the per-block
        drop-path schedule.

        :raises ValueError: If ``upsampler`` is unknown or ``scale`` is one it
            does not support.
        """
        super().__init__(**kwargs)

        if num_blocks is None:
            num_blocks = [4, 4, 4, 6, 6, 6]
        self._validate_upsampler(upsampler, scale)

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
        self.include_top = include_top

        self.total_blocks = sum(num_blocks)

        # Plain floats, since a tensor linspace breaks under TF eager.
        self.dpr = linear_drop_path_rates(self.total_blocks, drop_path_rate)

    @staticmethod
    def _validate_upsampler(upsampler: str, scale: int) -> None:
        """
        Refuse a tail the official code does not build.

        :param upsampler: One of :data:`UPSAMPLERS`.
        :type upsampler: str
        :param scale: The requested upsampling factor.
        :type scale: int
        :raises ValueError: If ``upsampler`` is unknown, ``scale`` is not a
            positive integer, or ``scale`` is one the upsampler cannot realize.
        """
        if upsampler not in UPSAMPLERS:
            raise ValueError(
                f"upsampler must be one of {UPSAMPLERS}, got {upsampler!r}"
            )
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale!r}")
        if upsampler == 'pixelshuffle':
            # The official `Upsample`: `scale & (scale - 1) == 0`, or 3.
            if (scale & (scale - 1)) != 0 and scale != 3:
                raise ValueError(
                    f"Unsupported scale for upsampler='pixelshuffle': {scale}. "
                    f"The official Upsample supports powers of two and 3."
                )
        elif upsampler == 'nearest+conv':
            # DECISION plan-2026-10-14T062616-81938e81/D-022: x4 ONLY, as the
            # official `assert self.upscale == 4`; do NOT stack log2(scale)
            # stages, an invented generalization. See decisions.md.
            if scale != 4:
                raise ValueError(
                    f"Unsupported scale for upsampler='nearest+conv': {scale}. "
                    f"The official tail has exactly two nearest 2x stages "
                    f"(conv_up1, conv_up2), so it realizes 4 only. Use "
                    f"upsampler='pixelshuffle' (powers of two and 3) or "
                    f"'pixelshuffledirect' (any scale) instead."
                )

    def build(self, input_shape):
        """
        Build model layers, the ``upsampler`` tail only with ``include_top``.

        :param input_shape: Shape tuple of the input.
        """
        self.conv_first = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_first"
        )

        self.stages = []
        block_idx = 0

        for stage_idx, num_blocks_in_stage in enumerate(self.num_blocks):
            stage_blocks = []

            for block_idx_in_stage in range(num_blocks_in_stage):
                # Even blocks use plain windows, odd blocks shifted ones, so a
                # window boundary in one block sits mid-window in the next.
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

        self.conv_after_body = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=3,
            strides=1,
            padding='same',
            name="conv_after_body"
        )

        # The tail is the model's own, not an EnhancementHead (the SqueezeNet
        # D-017 precedent): that head upsamples with a transposed conv, which
        # no official tail uses. Built only with include_top, so nothing built
        # here goes unapplied.
        self.tail_layers = self._build_tail() if self.include_top else []

        # Force sublayers to build now: build_from_config reloads them unbuilt,
        # and weight loading fails on layers "never built".
        if all(d is not None for d in input_shape[1:]):
            dummy = keras.ops.zeros((1,) + tuple(input_shape[1:]))
            self.call(dummy, training=False)

        super().build(input_shape)

    @staticmethod
    def _conv3x3(filters: int, name: str) -> keras.layers.Conv2D:
        """
        A 3x3, stride-1, ``'same'`` conv with bias: the official
        ``nn.Conv2d(in, out, 3, 1, 1)`` every tail layer uses.

        :param filters: Output channels.
        :type filters: int
        :param name: Layer name.
        :type name: str
        :return: The conv layer.
        :rtype: keras.layers.Conv2D
        """
        return keras.layers.Conv2D(
            filters=filters, kernel_size=3, strides=1, padding='same', name=name
        )

    def _build_tail(self) -> List[keras.layers.Layer]:
        """
        Build the official reconstruction tail for ``self.upsampler``, as a
        flat list applied in order.

        A flat list, not a nested ``keras.Sequential``: a nested sub-model's
        variable paths carry its own name in the donor and lose it after a
        ``.keras`` reload.

        :return: The tail's layers, in application order.
        :rtype: List[keras.layers.Layer]
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-024: pixel shuffle after
        # forward convs, NOT a transposed conv. Besides not being the paper,
        # the 2026-10-10 Conv2DTranspose tail ran cuDNN backward-data on GPU,
        # whose autotuner picked a non-deterministic engine once warm (a 1-ulp
        # call-to-call spread). See decisions.md.
        tail: List[keras.layers.Layer] = []
        if self.upsampler == 'pixelshuffledirect':
            # UpsampleOneStep: one conv straight to the output channels.
            tail.append(self._conv3x3(
                (self.scale ** 2) * self.in_channels, "upsample_conv"))
            tail.append(PixelShuffle2D(block_size=self.scale, name="pixel_shuffle"))
            return tail

        # DECISION plan-2026-10-14T062616-81938e81/D-022: negative_slope=0.01 is
        # torch nn.LeakyReLU()'s default, which the official tail uses; do NOT
        # fall back to Keras' LeakyReLU default of 0.3. See decisions.md.
        tail.append(self._conv3x3(NUM_FEAT, "conv_before_upsample"))
        tail.append(keras.layers.LeakyReLU(
            negative_slope=0.01, name="conv_before_upsample_lrelu"))

        if self.upsampler == 'pixelshuffle':
            # The official `Upsample(scale, num_feat)`.
            factors = [3] if self.scale == 3 else [2] * (self.scale.bit_length() - 1)
            for i, factor in enumerate(factors):
                tail.append(self._conv3x3(
                    (factor ** 2) * NUM_FEAT, f"upsample_conv_{i}"))
                # DECISION plan_2026-06-15_39a31d4a/D-003: PixelShuffle2D, not
                # keras.ops.nn.depth_to_space, absent in Keras 3.8. See decisions.md.
                tail.append(PixelShuffle2D(block_size=factor, name=f"pixel_shuffle_{i}"))
        else:  # 'nearest+conv', x4 (validated in __init__)
            for i in (1, 2):
                tail.append(keras.layers.UpSampling2D(
                    size=(2, 2), interpolation='nearest', name=f"nearest_up{i}"))
                tail.append(self._conv3x3(NUM_FEAT, f"conv_up{i}"))
                tail.append(keras.layers.LeakyReLU(
                    negative_slope=0.2, name=f"conv_up{i}_lrelu"))
            tail.append(self._conv3x3(NUM_FEAT, "conv_hr"))
            tail.append(keras.layers.LeakyReLU(negative_slope=0.2, name="conv_hr_lrelu"))

        tail.append(self._conv3x3(self.in_channels, "conv_last"))
        return tail

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
        :return: The super-resolved image (include_top=True) or the backbone
            features at the input resolution (include_top=False).
        :rtype: keras.KerasTensor
        """
        x = self.conv_first(inputs)
        residual = x

        prev_attn_map = None

        for stage_blocks in self.stages:
            for block in stage_blocks:
                # DECISION plan_2026-06-15_39a31d4a/D-003: the first block takes a bare
                # tensor, later blocks a list, never a tuple. See decisions.md.
                if prev_attn_map is None:
                    x, prev_attn_map = block(x, training=training)
                else:
                    x, prev_attn_map = block(
                        [x, prev_attn_map], training=training
                    )

        x = self.conv_after_body(x)
        x = x + residual

        for layer in self.tail_layers:
            x = layer(x)
        return x

    def get_config(self):
        """Return model configuration.

        :return: Dictionary containing all constructor arguments.
        """
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
            "include_top": self.include_top,
        })
        return config


def create_pft_sr(
        scale: int = 4,
        variant: Literal['base', 'light', 'repo_medium'] = 'base',
        include_top: bool = True,
        **kwargs: Any,
) -> PFTSR:
    """
    Build a PFT-SR model from a named variant.

    :param scale: Upsampling scale factor; see :class:`PFTSR` for what each
        upsampler supports.
    :type scale: int
    :param variant: Model variant: ``'light'`` and ``'base'`` mirror the
        paper's two released configs (from its official training YAMLs),
        upsampler included; ``'repo_medium'`` is a repo-original mid-size tier
        with no published counterpart.
    :type variant: str
    :param include_top: When True (default), the model ends in the variant's
        official upsampler tail and outputs the super-resolved image. When
        False, it returns the backbone features at the input resolution.
    :type include_top: bool
    :param kwargs: Any :class:`PFTSR` constructor argument, overriding the
        variant table. This is the only route to ``window_size``,
        ``drop_path_rate``, ``norm_type``, ``use_lepe`` and the dropout rates,
        and to an ``upsampler`` other than the variant's.
    :return: Configured :class:`PFTSR` instance.
    :rtype: PFTSR
    :raises ValueError: If ``variant`` is not a known variant name, or the
        upsampler cannot realize ``scale``.

    Example:
        >>> model = create_pft_sr(scale=4, variant='base')
        >>> model_light = create_pft_sr(scale=2, variant='light')
        >>> model_medium = create_pft_sr(scale=4, variant='repo_medium')
        >>> # Backbone features for a custom tail
        >>> backbone = create_pft_sr(scale=4, variant='base', include_top=False)
    """
    if variant not in PFTSR.MODEL_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Available variants: {list(PFTSR.MODEL_VARIANTS.keys())}"
        )

    # DECISION plan-2026-08-19T163559-499b6f0e/D-118: copy the variant dict before
    # updating with kwargs, or an override mutates MODEL_VARIANTS. See decisions.md.
    config = PFTSR.MODEL_VARIANTS[variant].copy()
    config['num_blocks'] = list(config['num_blocks'])
    config.update(kwargs)

    return PFTSR(
        scale=scale,
        include_top=include_top,
        **config
    )

# ---------------------------------------------------------------------