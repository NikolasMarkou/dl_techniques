"""ConvNeXtV1, a pure convolutional network modernized toward transformer design.

The architecture tests how much of the Vision Transformer's success came from
attention versus everything else transformer papers changed at the same time: a
patchified stem, a large receptive field per operation, an inverted bottleneck, sparse
normalization and activation, LayerNorm instead of BatchNorm, and separate downsampling
stages. Applying those changes to a ResNet-50, with no attention anywhere, matches a
Swin Transformer's accuracy at the same FLOPs. Each block is a depthwise `KxK`
convolution (default 7) followed by an inverted bottleneck MLP (1x1 expand to `4F`
channels, GELU, 1x1 reduce back to `F`), with one normalization and one activation per
block rather than one per convolution. `ConvNextV1Block` is transform-only: it returns
`F(x)`, not `x + F(x)`; the residual add and the drop-path gate live in this model's
`call`. The drop-path rate ramps linearly across the whole network, not per stage, so
the schedule does not reset at each stage boundary. Downsampling uses `padding="same"`
between stages so small CIFAR-scale inputs do not collapse to a 0x0 feature map; the
stem keeps `"valid"` padding at `strides > 1` to preserve the spatial geometry
existing checkpoints were trained with. With `include_top=True` the model returns
the pre-softmax logits of the shared vision `ClassificationHead`; compile with
`from_logits=True`. No pretrained weights are distributed:
`pretrained=True` raises `NotImplementedError`.
Local checkpoints load by path; every layer outside the classification head must be
restored, and only a head of another `num_classes` is skipped.

References:
    - Liu et al., 2022. A ConvNet for the 2020s. (https://arxiv.org/abs/2201.03545)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. (https://arxiv.org/abs/2103.14030)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Touvron et al., 2021. Going deeper with Image Transformers. (LayerScale)
      (https://arxiv.org/abs/2103.17239)
"""

from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.conv_blocks.convnext_v1_block import ConvNextV1Block
from dl_techniques.models.vision.convnext.common import _ConvNeXtBase
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.convnext.convnext_v1")
class ConvNeXtV1(_ConvNeXtBase):
    """ConvNeXt V1: a modernized ConvNet with optional local-checkpoint loading.

    A pure convolutional architecture that reaches Vision Transformer accuracy
    at matched FLOPs without any attention. A patchified stem feeds
    ``len(depths)`` stages of :class:`ConvNextV1Block` -- depthwise ``KxK``
    convolution, LayerNorm, inverted-bottleneck MLP (``F -> 4F -> F``) with a
    single GELU, and a learnable per-channel ``gamma`` -- separated by
    LayerNorm + strided-convolution downsample layers rather than a stride
    inside a block. The block is transform-only: it returns ``F(x)``, and the
    residual add plus the optional drop-path are owned by :meth:`call`. The
    drop-path ramp is global across ``sum(depths)`` blocks, not per stage. The
    model is fully convolutional and global-pools before the head, so the
    spatial dims of ``input_shape`` may be ``None``.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │       Input [B, H, W, C_in]          │
        │  (H, W may be None: fully conv)      │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Stem: Conv S×S /S → LayerNorm       │
        │  padding "valid" (S > 1)             │
        │          "same"  (S == 1)            │
        │  S = strides (default 4, patchify)   │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 0: D₀ × Block(dim₀)           │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Downsample: LayerNorm → Conv S×S /S │
        │  padding "same" (never 0×0)          │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 1: D₁ × Block(dim₁)           │
        └───────────────┬──────────────────────┘
                        ▼
                  Downsample
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 2: D₂ × Block(dim₂)           │
        └───────────────┬──────────────────────┘
                        ▼
                  Downsample
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 3: D₃ × Block(dim₃)           │
        └───────────────┬──────────────────────┘
                        │
            ┌───────────┴──────────────────────┐
            │  Residual wiring (owned here,    │
            │  not by the block)               │
            │    residual = x                  │
            │    x = ConvNextV1Block(x)   F(x) │
            │    x = DropPath(x)     if rate>0 │
            │    x = add([residual, x])        │
            └───────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  ClassificationHead (include_top):   │
        │  GAP → LayerNorm → Dense, logits     │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Output: logits [B, num_classes]     │
        │   include_top=False → [B, H', W', d₃]│
        └──────────────────────────────────────┘

    Block internals (ConvNextV1Block, transform only, returns F(x)):

    .. code-block:: text

        x ──► DWConv K×K (K=7) ──► LayerNorm
              └─ large receptive field, stands in for attention
                        │
                        ▼
              Dense/1×1 → 4F ──► GELU ──► Dense/1×1 → F
              └─ inverted bottleneck; one act, one norm per block
                        │
                        ▼
              × gamma  (layer scale, use_gamma; floored at 1e-6)
                        │
                        ▼
                      F(x)   ── residual added by the caller

    Drop-path ramp (global, not per stage):

    .. code-block:: text

        rate[i] = drop_path_rate · i / (sum(depths) - 1)

        block i:  0 ────────────────────────────► sum(depths)-1
        rate:    0.0 ──────────────────────────► drop_path_rate
                 └ stage 0 ┘└ 1 ┘└── 2 ──┘└ 3 ┘

        stochastic_mode='depth'    → StochasticDepth (drops branch)
        stochastic_mode='gradient' → StochasticGradient (fwd identity)

    :param num_classes: Number of output classes. Only used when
        ``include_top=True``, where it must be positive. Defaults to 1000.
    :type num_classes: int
    :param depths: Number of ConvNeXt blocks in each stage. Must have the same
        length as ``dims``. ``None`` resolves to ``[3, 3, 9, 3]``
        (ConvNeXt-Tiny). A length other than 4 that matches no variant logs a
        warning.
    :type depths: Optional[List[int]]
    :param dims: Channel count per stage. ``None`` resolves to
        ``[96, 192, 384, 768]``.
    :type dims: Optional[List[int]]
    :param drop_path_rate: Terminal stochastic-depth rate. The per-block rate
        ramps linearly from 0.0 at the first block of stage 0 to this value at
        the last block of the last stage, indexed globally over
        ``sum(depths)`` blocks. Must lie in ``[0, 1)``. Defaults to 0.0
        (disabled).
    :type drop_path_rate: float
    :param stochastic_mode: What the per-block regularizer does. ``'depth'``
        (default) is standard stochastic depth and drops the whole residual
        branch at training time; ``'gradient'`` is forward-identity and only
        perturbs the backward pass. ``'depth'`` is behaviour-preserving.
    :type stochastic_mode: str
    :param kernel_size: Depthwise kernel size inside each block. Defaults to 7,
        following the paper.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param activation: Activation used inside each block, resolved through
        :func:`deserialize_activation`. Defaults to ``"gelu"``.
    :type activation: str
    :param use_bias: Whether convolutions use a bias, and whether every
        backbone LayerNormalization centers. The head's classifier and its
        pooled LayerNorm always carry a bias, as the reference's do. Defaults to True.
    :type use_bias: bool
    :param kernel_regularizer: Optional regularizer applied to the stem, the
        downsampling layers, the blocks and the head's classifier kernel. A
        regularizer, its serialized config or a name.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param dropout_rate: Dropout rate applied within blocks, in ``[0, 1)``.
        Defaults to 0.0.
    :type dropout_rate: float
    :param spatial_dropout_rate: Spatial dropout rate applied within blocks,
        in ``[0, 1)``. Defaults to 0.0.
    :type spatial_dropout_rate: float
    :param strides: Patchify factor. Used as BOTH the kernel size and the
        stride of the stem convolution and of every inter-stage downsample
        convolution. Defaults to 4.
    :type strides: int
    :param use_gamma: Whether blocks apply the learnable per-channel layer
        scale. Defaults to True.
    :type use_gamma: bool
    :param use_softorthonormal_regularizer: Whether blocks use soft orthonormal
        regularization. Defaults to False.
    :type use_softorthonormal_regularizer: bool
    :param include_top: Whether to include the GAP + LayerNorm + Dense
        classification head. When False the final stage's feature maps are
        returned. A ``bool`` or ``numpy.bool_``; anything else, a string
        included, raises ``TypeError``. Defaults to True.
    :type include_top: bool
    :param input_shape: Input shape ``(height, width, channels)`` excluding the
        batch dimension. ``None`` resolves to ``(None, None, 3)`` -- the model
        is fully convolutional and global-pools before the head, so a concrete
        spatial size is optional. It is required only where a downstream
        consumer needs static spatial dims; a checkpoint load builds the model
        at this shape, spatial dims left ``None`` included.
    :type input_shape: Optional[Tuple[int, ...]]
    :param head_config_overrides: Optional dict merged into the
        ``ClassificationHead`` configuration (e.g.,
        ``{'block_dims': (512,), 'dropout_rate': 0.2}`` adds a hidden
        block). ``name``, ``num_classes`` and ``task_type`` are owned by the
        model and raise ``ValueError``. Every value must be JSON data (a
        regularizer is passed as its serialized dict). Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises TypeError: If ``include_top`` is not a ``bool`` or ``numpy.bool_``.
    :raises ValueError: If ``include_top`` is True and ``num_classes`` is not
        positive, if ``drop_path_rate``, ``dropout_rate`` or
        ``spatial_dropout_rate`` is outside ``[0, 1)`` (the message names the
        argument and its value), if ``depths`` and ``dims`` differ in length, if
        ``strides`` is not positive, if ``stochastic_mode`` is not ``'depth'``
        or ``'gradient'``, if ``input_shape`` is not 3D, or if
        ``head_config_overrides`` names ``name``, ``num_classes`` or
        ``task_type`` or holds a value that is not JSON data.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_top=True``: 2D logits tensor ``(batch_size, num_classes)``.
        - ``include_top=False``: 4D tensor ``(batch_size, H', W', dims[-1])``.

    Example::

        # ConvNeXt-Tiny for CIFAR-10
        model = ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

        # Feature extractor
        model = ConvNeXtV1.from_variant("base", include_top=False)

        # Warm start from a local checkpoint
        model = ConvNeXtV1.from_variant("large", pretrained="/path/to.keras")

    Note:
        No pretrained ConvNeXt V1 weights are distributed with
        ``dl_techniques``. ``pretrained=True`` raises ``NotImplementedError``
        rather than warning and returning a randomly-initialized model; pass a
        local checkpoint via ``pretrained='/path/to/weights.keras'`` instead.
    """

    MODEL_VARIANTS = {
        "cifar10": {"depths": [5, 5], "dims": [96, 192]},
        "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
        "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
        "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
        "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
        "xlarge": {"depths": [3, 3, 27, 3], "dims": [256, 512, 1024, 2048]},
    }
    BLOCK_CLASS = ConvNextV1Block


# ---------------------------------------------------------------------


def create_convnext_v1(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs
) -> ConvNeXtV1:
    """Create a ConvNeXt V1 model.

    :param variant: Model variant: `"cifar10"`, `"tiny"`, `"small"`, `"base"`, `"large"`,
        `"xlarge"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; `None` resolves to `(None, None, 3)`.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param kwargs: Passthrough to `ConvNeXtV1.from_variant`, e.g.
        `head_config_overrides`, `include_top`, `drop_path_rate`, `depths`.
    :return: A configured `ConvNeXtV1` instance.
    :raises NotImplementedError: If `pretrained` is `True`.
    :raises ValueError: If `variant` is not recognized.

    Example::

        model = create_convnext_v1("tiny")
        backbone = create_convnext_v1("base", include_top=False)
        tuned = create_convnext_v1(
            "small", num_classes=10, input_shape=(32, 32, 3),
            pretrained="path/to/weights.keras")
    """
    return ConvNeXtV1.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )


# ---------------------------------------------------------------------
