"""ConvNeXtV2, ConvNeXt with Global Response Normalization, co-designed with masked autoencoding.

V1 was designed for supervised training, and carrying it directly into masked-autoencoder
pretraining underperforms: many channels of the inverted bottleneck's expanded `4F`
representation collapse onto near-duplicates of each other. Global Response Normalization
(GRN) is the fix, and the only structural change from V1. GRN sits after the GELU inside
the expanded part of the block: it computes each channel's L2 norm over the spatial
dimensions, divides by the mean across channels, and rescales each channel by that ratio,
suppressing channels that are quiet relative to their peers and amplifying ones that carry
signal. It is computed from a global statistic, so it costs almost nothing and adds only
two learnable per-channel parameters. Everything else matches V1: a depthwise `KxK`
convolution, a 1x1 expansion to `4F`, GELU, now GRN, a 1x1 reduction back to `F`, and a
learnable `gamma` layer scale, with the block transform-only and the residual/drop-path
wiring owned by `call`. The variant table spans a wider range than V1's, from Atto (3.7M
parameters) to Huge (660M), since GRN's benefit is most visible at the small end. With
`include_top=True` the model returns the pre-softmax logits of the shared vision
`ClassificationHead`; compile with `from_logits=True`. No pretrained weights are
distributed: `pretrained=True` raises `NotImplementedError`.
Local checkpoints load by path; every layer outside the classification head must be
restored, and only a head of another `num_classes` is skipped.

References:
    - Woo et al., 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with
      Masked Autoencoders. (https://arxiv.org/abs/2301.00808)
    - Liu et al., 2022. A ConvNet for the 2020s. (https://arxiv.org/abs/2201.03545)
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      (https://arxiv.org/abs/2111.06377)
"""

from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.conv_blocks.convnext_v2_block import ConvNextV2Block
from dl_techniques.models.vision.convnext.common import _ConvNeXtBase
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.convnext.convnext_v2")
class ConvNeXtV2(_ConvNeXtBase):
    """ConvNeXt V2: ConvNeXt plus Global Response Normalization.

    A modern ConvNet that adds Global Response Normalization inside the
    inverted bottleneck, an explicit inter-channel competition that prevents
    the feature collapse V1 exhibits under masked-autoencoder pretraining. GRN
    is the only structural change from :class:`ConvNeXtV1`: a patchify stem
    feeds ``len(depths)`` stages of :class:`ConvNextV2Block` -- depthwise
    ``KxK`` convolution, LayerNorm, ``F -> 4F`` expansion, GELU, GRN, ``4F ->
    F`` reduction, and a learnable ``gamma`` -- separated by LayerNorm +
    strided-convolution downsample layers. The block is transform-only: it
    returns ``F(x)``, and the residual add plus the optional drop-path are
    owned by :meth:`call`. The drop-path ramp is global across ``sum(depths)``
    blocks, not per stage. The model is fully convolutional and global-pools
    before the head, so the spatial dims of ``input_shape`` may be ``None``.

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
            │    x = ConvNextV2Block(x)   F(x) │
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
        ``sum(depths)`` blocks. Defaults to 0.0 (disabled).
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
    :param dropout_rate: Dropout rate applied within blocks. Defaults to 0.0.
    :type dropout_rate: float
    :param spatial_dropout_rate: Spatial dropout rate applied within blocks.
        Defaults to 0.0.
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
        returned. Defaults to True.
    :type include_top: bool
    :param input_shape: Input shape ``(height, width, channels)`` excluding the
        batch dimension. ``None`` resolves to ``(None, None, 3)`` -- the model
        is fully convolutional and global-pools before the head, so a concrete
        spatial size is optional. It is required only where a downstream
        consumer needs static spatial dims; a checkpoint load with unspecified
        spatial dims materializes weights at ``PRETRAINED_BUILD_SPATIAL`` (224).
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

    :raises ValueError: If ``include_top`` is True and ``num_classes`` is not
        positive, if ``depths`` and ``dims`` differ in length, if
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

        # ConvNeXt V2-Tiny for CIFAR-10
        model = ConvNeXtV2.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

        # Feature extractor
        model = ConvNeXtV2.from_variant("base", include_top=False)

        # Warm start from a local checkpoint
        model = ConvNeXtV2.from_variant("large", pretrained="/path/to.keras")

    Note:
        No pretrained ConvNeXt V2 weights are distributed with
        ``dl_techniques``. ``pretrained=True`` raises ``NotImplementedError``
        rather than warning and returning a randomly-initialized model; pass a
        local checkpoint via ``pretrained='/path/to/weights.keras'`` instead.
    """

    MODEL_VARIANTS = {
        "cifar10": {"depths": [5, 5], "dims": [96, 192]},
        "atto": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},
        "femto": {"depths": [2, 2, 6, 2], "dims": [48, 96, 192, 384]},
        "pico": {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512]},
        "nano": {"depths": [2, 2, 8, 2], "dims": [80, 160, 320, 640]},
        "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
        "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
        "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
        "huge": {"depths": [3, 3, 27, 3], "dims": [352, 704, 1408, 2816]},
    }
    BLOCK_CLASS = ConvNextV2Block


# ---------------------------------------------------------------------


def create_convnext_v2(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs
) -> ConvNeXtV2:
    """Create a ConvNeXt V2 model.

    :param variant: Model variant: `"cifar10"`, `"atto"`, `"femto"`, `"pico"`, `"nano"`,
        `"tiny"`, `"base"`, `"large"`, `"huge"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; `None` resolves to `(None, None, 3)`.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param kwargs: Passthrough to `ConvNeXtV2.from_variant`, e.g.
        `head_config_overrides`, `include_top`, `drop_path_rate`, `depths`.
    :return: A configured `ConvNeXtV2` instance.
    :raises NotImplementedError: If `pretrained` is `True`.
    :raises ValueError: If `variant` is not recognized.

    Example::

        model = create_convnext_v2("tiny")
        backbone = create_convnext_v2("base", include_top=False)
        tuned = create_convnext_v2(
            "pico", num_classes=10, input_shape=(32, 32, 3),
            pretrained="path/to/weights.keras")
    """
    return ConvNeXtV2.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )


# ---------------------------------------------------------------------
