"""
PFTSR performs single-image super-resolution with the Progressive Focused
Transformer of Long et al. (CVPR 2025), ported from the official code,
CVL-UESTC/PFT-SR ``basicsr/archs/pft_arch.py`` (``PFT``).

The forward is the official one: mirror-pad the image to a multiple of
``window_size``, subtract the DIV2K RGB mean and scale by ``img_range``,
``conv_first``, a patch-embed LayerNorm, the PFT stages (each a chain of
:class:`~.blocks.PFTLayer` then its own 3x3 conv and a residual around the
stage), a final LayerNorm, ``conv_after_body`` plus the shallow features, the
reconstruction tail, then divide by ``img_range``, add the mean back and crop.
Every layer runs windowed attention whose post-softmax map is multiplied by the
map of the previous layer of its chain (plain-window layers and shifted layers
form two chains across all stages), and the per-layer ``num_topk`` schedule
narrows the keys each chain attends to. The official code does that with
sparse CUDA kernels; this port computes the same numbers on a dense map under
an inherited boolean mask (see :mod:`.blocks`).

The default output (``include_top=True``) ends in the official reconstruction
tail, selected by ``upsampler`` exactly as there: ``'pixelshuffle'`` (the PFT
config: ``conv_before_upsample`` 3x3 to 64 + LeakyReLU(0.01), then a 3x3 conv
and a pixel shuffle per factor of 2, or one of each for 3, then
``conv_last``), ``'pixelshuffledirect'`` (the PFT_light config: one 3x3 conv to
``scale**2 * in_channels`` and one pixel shuffle) and ``'nearest+conv'`` (x4
only: two nearest x2 stages with LeakyReLU(0.2), ``conv_hr``, ``conv_last``).
The model builds this tail itself: the shared ``EnhancementHead`` upsamples
with a transposed conv and cannot express any of the three.
``include_top=False`` returns the features the tail reads,
``conv_after_body(body(x)) + x``, cropped to the input size (the mean is not
added back to features).

Height and width must be statically known (the shifted-window mask is built
from them) and each at least half of its padded size (the official mirror pad
flips the image once). The README section 4.3 lists what still differs from
the official code.

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
import numpy as np
from typing import Any, Dict, Optional, List, Literal, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.pooling.pixel_unshuffle import PixelShuffle2D
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.keras_registration import register_dl_technique
from .blocks import LAYER_NORM_EPSILON, PFTLayer, mirror_pad

# ---------------------------------------------------------------------

#: The three reconstruction tails of the official ``PFT`` class.
UPSAMPLERS: Tuple[str, ...] = ('pixelshuffle', 'pixelshuffledirect', 'nearest+conv')

#: The official tail's working width, ``num_feat = 64`` in ``PFT.__init__``,
#: used by ``'pixelshuffle'`` and ``'nearest+conv'`` whatever ``embed_dim`` is.
NUM_FEAT = 64

#: ``PFT.__init__`` (:915): the DIV2K RGB mean, subtracted from a 3-channel
#: input and added back to the output; a model of any other channel count
#: shifts by 0, as there.
RGB_MEAN: Tuple[float, float, float] = (0.4488, 0.4371, 0.4040)

#: ``num_topk`` of the PFT config (001_PFT_SRx2_scratch.yml, 30 layers).
PFT_NUM_TOPK: Tuple[int, ...] = (
    1024, 1024, 1024, 1024,
    256, 256, 256, 256,
    128, 128, 128, 128,
    64, 64, 64, 64, 64, 64,
    32, 32, 32, 32, 32, 32,
    16, 16, 16, 16, 16, 16,
)

#: ``num_topk`` of the PFT_light config (101_PFT_light_SRx2_scratch.yml, 24 layers).
PFT_LIGHT_NUM_TOPK: Tuple[int, ...] = (
    1024, 1024,
    256, 256, 256, 256,
    128, 128, 128, 128, 128, 128,
    64, 64, 64, 64, 64, 64,
    32, 32, 32, 32, 32, 32,
)

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.pft_sr.model")
class PFTSR(keras.Model):
    """
    Progressive Focused Transformer for single image super-resolution.

    A Keras port of Long et al.'s PFT-SR (CVPR 2025), the official ``PFT``
    network. The port has not been trained or benchmarked in this codebase,
    so no performance claim is made for it here.

    Architecture:

    .. code-block:: text

        x [B, H, W, in_channels]
              │  mirror pad to a multiple of window_size
              │  (x - mean) * img_range
              ▼
        conv_first 3x3 ───────────────────────────────┐ shallow
              ▼                                       │
        patch-embed LayerNorm                         │
              ▼                                       │
        stage s (PFTB):  t ─► PFTLayer x num_blocks[s] ─► conv 3x3 ─► + t
              ▼          (two attention chains thread through all stages)
        LayerNorm                                     │
              ▼                                       │
        conv_after_body 3x3 ─────────────────────────(+)
                                                      │ include_top=False:
                                                      │ crop, return
                                                      ▼
        upsampler tail ─► / img_range + mean ─► crop
              ▼
        output [B, H*scale, W*scale, in_channels]

    Attention chains (the official ``pfa_values[shift]``): the layers with a
    plain window (even index within their stage) form one chain and the
    shifted ones another; each layer multiplies its attention by the map of
    the previous layer of its chain and, when its ``num_topk`` entry is below
    the chain's width, keeps only its top ``num_topk`` keys per query.

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

        variant     embed_dim  num_blocks     heads  mlp  window  kernel  num_topk           upsampler
        light       52         [2,4,6,6,6]    4      1.0  32      7       1024 .. 32 (24)    pixelshuffledirect
        base        240        [4,4,4,6,6,6]  6      2.0  32      7       1024 .. 16 (30)    pixelshuffle
        repo_medium 80         [6,6,6,8,8,8]  8      2.0  8       7       None (every key)   pixelshuffle

    `window` is each variant's own `window_size`, not the constructor default.

    :param scale: Upsampling scale factor. ``'pixelshuffle'`` takes a power
        of two or 3, ``'pixelshuffledirect'`` any positive integer,
        ``'nearest+conv'`` only 4.
    :type scale: int
    :param in_channels: Number of input image channels. With 3, the RGB mean
        is shifted out and back; with any other count, nothing is.
    :type in_channels: int
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param num_blocks: Number of PFT layers in each stage (the official
        ``depths``). ``None`` means ``[4, 4, 4, 6, 6, 6]``.
    :type num_blocks: list[int]
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param window_size: Attention window size. The ``light`` and ``base``
        variants override this to 32; 8 is only the constructor's own
        fallback.
    :type window_size: int
    :param mlp_ratio: ConvFFN hidden width as a multiple of ``embed_dim``.
    :type mlp_ratio: float
    :param convffn_kernel_size: Kernel of the depthwise convs in the LePE and
        the ConvFFN (official default 5; both released configs use 7). Odd.
    :type convffn_kernel_size: int
    :param num_topk: Keys kept per query by each layer, one entry per layer in
        order (``len == sum(num_blocks)``). A layer whose entry is below its
        chain's current width keeps that many; otherwise it keeps the width.
        ``None`` keeps every key in every layer (progressive focusing without
        sparsification).
    :type num_topk: Optional[list[int]]
    :param qkv_bias: Whether to use bias in the QKV projections.
    :type qkv_bias: bool
    :param img_range: The official ``img_range``: the mean-shifted input is
        multiplied by it and the output divided by it (1.0 for ``[0, 1]``
        images, as both released configs).
    :type img_range: float
    :param drop_path_rate: Stochastic depth rate, rising linearly from 0 to
        this value over all layers. Not in the official PFT, which has no
        stochastic depth; the default 0.0 builds none.
    :type drop_path_rate: float
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
        :data:`UPSAMPLERS`, ``scale`` is one that upsampler does not support,
        ``num_topk`` does not have one positive entry per layer, or another
        argument is out of range; at call, if the input's height or width is
        unknown or too small for the official mirror pad.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, in_channels)`, with
        static height and width (any size from half a padded window up).

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
        >>> # The paper's PFT_light config: its 32x32 windows mirror-pad a
        >>> # 48x48 input to 64x64 and crop the output back to 192x192.
        >>> model_light = create_pft_sr(scale=4, variant='light')
        >>> sr_image_light = model_light(keras.random.normal((1, 48, 48, 3)))
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
            'convffn_kernel_size': 7,
            'num_topk': list(PFT_LIGHT_NUM_TOPK),
            'upsampler': 'pixelshuffledirect',
        },
        # PFT, 001_PFT_SRx2_scratch.yml network_g
        'base': {
            'embed_dim': 240,
            'num_blocks': [4, 4, 4, 6, 6, 6],
            'num_heads': 6,
            'mlp_ratio': 2.0,
            'window_size': 32,
            'convffn_kernel_size': 7,
            'num_topk': list(PFT_NUM_TOPK),
            'upsampler': 'pixelshuffle',
        },
        # Repo-original mid-size tier, with no published counterpart: the
        # released configs' kernel 7, and no top-k schedule (none was ever
        # published for an 8x8 window; every key is kept).
        'repo_medium': {
            'embed_dim': 80,
            'num_blocks': [6, 6, 6, 8, 8, 8],
            'num_heads': 8,
            'mlp_ratio': 2.0,
            'window_size': 8,
            'convffn_kernel_size': 7,
            'upsampler': 'pixelshuffle',
        }
    }

    def __init__(
            self,
            scale: int = 4,
            in_channels: int = 3,
            embed_dim: int = 60,
            num_blocks: Optional[List[int]] = None,
            num_heads: int = 6,
            window_size: int = 8,
            mlp_ratio: float = 2.0,
            convffn_kernel_size: int = 5,
            num_topk: Optional[Sequence[int]] = None,
            qkv_bias: bool = True,
            img_range: float = 1.0,
            drop_path_rate: float = 0.0,
            upsampler: Literal['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'] = 'pixelshuffle',
            include_top: bool = True,
            **kwargs
    ):
        """Validate the configuration and create every layer: the body, the
        per-layer top-k schedule and, with ``include_top``, the tail.

        :raises ValueError: If ``upsampler`` is unknown, ``scale`` is one it
            does not support, ``num_topk`` is not one positive integer per
            layer, or ``img_range`` is not positive.
        """
        super().__init__(**kwargs)

        if num_blocks is None:
            num_blocks = [4, 4, 4, 6, 6, 6]
        self._validate_upsampler(upsampler, scale)
        total_blocks = sum(num_blocks)
        if num_topk is not None:
            num_topk = [int(k) for k in num_topk]
            if len(num_topk) != total_blocks or any(k < 1 for k in num_topk):
                raise ValueError(
                    f"num_topk must hold one positive integer per layer "
                    f"(sum(num_blocks) = {total_blocks}), got {len(num_topk)} "
                    f"entries: {num_topk}")
        if not img_range > 0:
            raise ValueError(f"img_range must be positive, got {img_range}")

        self.scale = scale
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_blocks = list(num_blocks)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        self.img_range = img_range
        self.drop_path_rate = drop_path_rate
        self.upsampler = upsampler
        self.include_top = include_top

        self.total_blocks = total_blocks
        # Plain floats, since a tensor linspace breaks under TF eager.
        self.dpr = linear_drop_path_rates(self.total_blocks, drop_path_rate)
        # float64 here, cast to the compute dtype in call: a float32 copy would
        # round 0.4488 before a float64 model ever saw it.
        self.mean = np.array(RGB_MEAN if in_channels == 3 else (0.0,), dtype="float64")

        self.conv_first = self._conv3x3(embed_dim, "conv_first")
        self.patch_norm = create_normalization_layer(
            "layer_norm", epsilon=LAYER_NORM_EPSILON, name="patch_norm")
        self.pft_layers: List[PFTLayer] = []
        self.stage_convs: List[keras.layers.Conv2D] = []
        # The official top-k rule (:307): a layer narrows its chain only when
        # its num_topk entry is below the chain's current width.
        # DECISION plan-2026-10-14T062616-81938e81/D-215: two chains keyed by the
        # STAGE-LOCAL shift parity (the official pfa_values[shift]), each running
        # across stages. Do NOT chain each layer to the one just before it: a
        # shifted window's map does not align with a plain one. See decisions.md.
        width = {0: window_size * window_size, 1: window_size * window_size}
        layer_id = 0
        for stage_idx, depth in enumerate(self.num_blocks):
            for i in range(depth):
                chain = i % 2
                keep = None
                if num_topk is not None and num_topk[layer_id] < width[chain]:
                    keep = width[chain] = num_topk[layer_id]
                self.pft_layers.append(PFTLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if chain == 0 else window_size // 2,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    keep_topk=keep,
                    drop_path_rate=self.dpr[layer_id],
                    name=f"stage{stage_idx}_layer{i}",
                ))
                layer_id += 1
            self.stage_convs.append(self._conv3x3(embed_dim, f"stage{stage_idx}_conv"))
        self.norm = create_normalization_layer(
            "layer_norm", epsilon=LAYER_NORM_EPSILON, name="norm")
        self.conv_after_body = self._conv3x3(embed_dim, "conv_after_body")

        # The tail is the model's own, not an EnhancementHead (the SqueezeNet
        # D-017 precedent): that head upsamples with a transposed conv, which
        # no official tail uses. Built only with include_top, so nothing built
        # here goes unapplied.
        self.tail_layers = self._build_tail() if self.include_top else []

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
        Build every layer from the channel counts alone, so an explicit
        ``build`` with unknown height and width creates every weight.

        :param input_shape: Shape tuple of the input.
        """
        features = (None, None, None, self.embed_dim)
        self.conv_first.build((None, None, None, self.in_channels))
        self.patch_norm.build(features)
        for layer in self.pft_layers:
            layer.build(features)
        for conv in self.stage_convs:
            conv.build(features)
        self.norm.build(features)
        self.conv_after_body.build(features)
        shape = features
        for layer in self.tail_layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
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
        Forward pass of PFT-SR (the official ``PFT.forward``, :1102).

        :param inputs: Input low-resolution images of shape
            (batch_size, height, width, channels), height and width static.
        :type inputs: keras.KerasTensor
        :param training: Whether the call runs in training mode.
        :type training: bool or None
        :return: The super-resolved image (include_top=True) or the backbone
            features at the input resolution (include_top=False).
        :rtype: keras.KerasTensor
        """
        height, width = inputs.shape[1], inputs.shape[2]
        x = mirror_pad(inputs, self.window_size)
        mean = keras.ops.cast(self.mean, x.dtype)
        x = (x - mean) * self.img_range
        x = self.conv_first(x)
        shallow = x

        # forward_features (:1047): two progressive-focusing chains,
        # pfa_values[shift] and pfa_indices[shift], across every stage (the
        # chains of __init__, D-215).
        values: List[Optional[keras.KerasTensor]] = [None, None]
        masks: List[Optional[keras.KerasTensor]] = [None, None]
        t = self.patch_norm(x)
        layer_id = 0
        for stage_idx, depth in enumerate(self.num_blocks):
            stage_input = t
            for i in range(depth):
                chain = i % 2
                t, values[chain], masks[chain] = self.pft_layers[layer_id](
                    t, prev_values=values[chain], prev_mask=masks[chain],
                    training=training)
                layer_id += 1
            # PFTB.forward (:713): patch_embed(conv(patch_unembed(x))) + x
            t = self.stage_convs[stage_idx](t) + stage_input
        x = self.conv_after_body(self.norm(t)) + shallow

        if not self.include_top:
            return x[:, :height, :width]
        for layer in self.tail_layers:
            x = layer(x)
        x = x / self.img_range + mean
        return x[:, :height * self.scale, :width * self.scale]

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
            "convffn_kernel_size": self.convffn_kernel_size,
            "num_topk": self.num_topk,
            "qkv_bias": self.qkv_bias,
            "img_range": self.img_range,
            "drop_path_rate": self.drop_path_rate,
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
        variant table. This is the only route to ``drop_path_rate``,
        ``img_range`` and ``qkv_bias``, and to a ``window_size``,
        ``num_topk``, ``convffn_kernel_size`` or ``upsampler`` other than the
        variant's.
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
    if config.get('num_topk') is not None:
        config['num_topk'] = list(config['num_topk'])
    config.update(kwargs)

    return PFTSR(
        scale=scale,
        include_top=include_top,
        **config
    )

# ---------------------------------------------------------------------