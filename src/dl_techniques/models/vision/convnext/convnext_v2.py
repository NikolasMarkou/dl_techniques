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
parameters) to Huge (660M), since GRN's benefit is most visible at the small end. No
pretrained weights are distributed: `pretrained=True` raises `NotImplementedError`.

References:
    - Woo et al., 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with
      Masked Autoencoders. (https://arxiv.org/abs/2301.00808)
    - Liu et al., 2022. A ConvNet for the 2020s. (https://arxiv.org/abs/2201.03545)
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      (https://arxiv.org/abs/2111.06377)
"""

import os
import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.conv_blocks.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.regularization.stochastic_gradient import StochasticGradient
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.convnext.convnext_v2")
class ConvNeXtV2(keras.Model):
    """ConvNeXt V2: ConvNeXt plus Global Response Normalization.

    A modern ConvNet that adds Global Response Normalization inside the
    inverted bottleneck, an explicit inter-channel competition that prevents
    the feature collapse V1 exhibits under masked-autoencoder pretraining. GRN is the only structural change from :class:`ConvNeXtV1`: a patchify stem
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
            │  Residual wiring (owned HERE,    │
            │  not by the block)               │
            │    residual = x                  │
            │    x = ConvNextV2Block(x)   F(x) │
            │    x = DropPath(x)     if rate>0 │
            │    x = add([residual, x])        │
            └───────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  GAP → LayerNorm → Dense(num_classes)│
        │  (if include_top)                    │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Output: [B, num_classes]            │
        │   include_top=False → [B, H', W', d₃]│
        │   num_classes=0     → [B, d₃] pooled │
        └──────────────────────────────────────┘

    Variants:

    .. code-block:: text

        cifar10   [5, 5]         [96, 192]
        atto      [2, 2, 6, 2]   [40, 80, 160, 320]
        femto     [2, 2, 6, 2]   [48, 96, 192, 384]
        pico      [2, 2, 6, 2]   [64, 128, 256, 512]
        nano      [2, 2, 8, 2]   [80, 160, 320, 640]
        tiny      [3, 3, 9, 3]   [96, 192, 384, 768]
        base      [3, 3, 27, 3]  [128, 256, 512, 1024]
        large     [3, 3, 27, 3]  [192, 384, 768, 1536]
        huge      [3, 3, 27, 3]  [352, 704, 1408, 2816]

    :param num_classes: Number of output classes. Only used when
        ``include_top=True``. A value of 0 returns pooled, normalized features
        from the head. Defaults to 1000.
    :type num_classes: int
    :param depths: Number of ConvNeXt blocks in each stage. Must have the same
        length as ``dims``. ``None`` resolves to ``[3, 3, 9, 3]``
        (ConvNeXt-Tiny). A length other than 4 is permitted but logs a warning.
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
        following the original paper.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param activation: Activation used inside each block, resolved through
        :func:`deserialize_activation`. Defaults to ``"gelu"``.
    :type activation: str
    :param use_bias: Whether convolutions and the classifier use a bias, and
        whether every LayerNormalization centers. Defaults to True.
    :type use_bias: bool
    :param kernel_regularizer: Optional regularizer applied to all convolution
        and dense kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
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
    :type input_shape: Tuple[int, ...]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``depths`` and ``dims`` differ in length, if
        ``strides`` is not positive, if ``stochastic_mode`` is not ``'depth'``
        or ``'gradient'``, or if ``input_shape`` is not 3D.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_top=True``: 2D tensor ``(batch_size, num_classes)``.
        - ``include_top=True, num_classes=0``: 2D tensor
          ``(batch_size, dims[-1])`` of pooled, normalized features.
        - ``include_top=False``: 4D tensor ``(batch_size, H', W', dims[-1])``.

    Example:
        >>> # ConvNeXt V2-Tiny for CIFAR-10
        >>> model = ConvNeXtV2.from_variant("tiny", num_classes=10,
        ...                                 input_shape=(32, 32, 3))
        >>>
        >>> # Feature extractor
        >>> model = ConvNeXtV2.from_variant("base", include_top=False)
        >>>
        >>> # Warm start from a local checkpoint
        >>> model = ConvNeXtV2.from_variant("large",
        ...                                 pretrained="/path/to.keras")

    Note:
        No pretrained ConvNeXt V2 weights are distributed with
        ``dl_techniques``. ``pretrained=True`` raises ``NotImplementedError``
        rather than warning and returning a randomly-initialized model; pass a
        local checkpoint via ``pretrained='/path/to/weights.keras'`` instead.
    """

    # Model variant configurations
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

    # Architecture constants
    LAYERNORM_EPSILON = 1e-6
    STEM_INITIALIZER = "truncated_normal"
    HEAD_INITIALIZER = "truncated_normal"

    def __init__(
            self,
            num_classes: int = 1000,
            depths: Optional[List[int]] = None,
            dims: Optional[List[int]] = None,
            drop_path_rate: float = 0.0,
            stochastic_mode: str = 'depth',
            kernel_size: Union[int, Tuple[int, int]] = 7,
            activation: str = "gelu",
            use_bias: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            strides: int = 4,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            include_top: bool = True,
            input_shape: Tuple[int, ...] = (None, None, 3),
            **kwargs
    ):
        super().__init__(**kwargs)

        depths = list(depths) if depths is not None else [3, 3, 9, 3]
        dims = list(dims) if dims is not None else [96, 192, 384, 768]

        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )

        if len(depths) != 4:
            logger.warning(
                f"ConvNeXt typically uses 4 stages, got {len(depths)} stages"
            )

        if strides <= 0:
            raise ValueError(
                f"Strides {strides} must be positive."
            )

        if stochastic_mode not in ('depth', 'gradient'):
            raise ValueError(
                f"stochastic_mode must be 'depth' or 'gradient', got {stochastic_mode!r}"
            )

        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.stochastic_mode = stochastic_mode
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer
        self.include_top = include_top
        self.strides = strides
        self.input_shape = input_shape

        self.input_height, self.input_width, self.input_channels = input_shape
        if self.input_channels not in [1, 3]:
            logger.warning(
                f"Unusual number of channels: {self.input_channels}. ConvNeXt typically uses 3 channels")

        self._build_stem()

        self.downsample_layers_list = []
        self.stages_list = []
        for i in range(len(self.depths)):
            if i > 0:
                self._build_downsample_layer(i)
            self._build_stage(i)

        if self.include_top:
            self._build_head()

        logger.info(
            f"Created ConvNeXt V2 model for input {input_shape} "
            f"with {sum(depths)} blocks"
        )

    def _build_stem(self) -> None:
        """Build and assign the patchify stem.

        A ``strides x strides`` convolution at stride ``strides`` followed by
        LayerNorm. The padding rule is deliberately NOT the downsample layers'
        unconditional ``"same"``; see the D-125 anchor in the body.
        """
        stem_kernel_size = self.strides
        stem_stride = self.strides
        self.stem_conv = keras.layers.Conv2D(
            filters=self.dims[0],
            kernel_size=stem_kernel_size,
            strides=stem_stride,
            # DECISION plan-2026-08-19T163559-499b6f0e/D-125: not the unconditional
            # "same" the downsample layers use — "same" here would silently shift the spatial geometry of checkpoints trained at a non-divisible input size. See decisions.md.
            padding="same" if stem_stride == 1 else "valid",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        self.stem_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="stem_norm"
        )

    def _build_downsample_layer(self, stage_idx: int) -> None:
        """Build and assign the LayerNorm + strided convolution before a stage.

        :param stage_idx: Index of the stage this downsample layer feeds. Must
            be greater than 0; the pair is stored at ``stage_idx - 1`` in
            ``downsample_layers_list``.
        :type stage_idx: int
        """
        downsample_kernel_size, downsample_stride = self.strides, self.strides
        downsample_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_norm_{stage_idx - 1}"
        )
        downsample_conv = keras.layers.Conv2D(
            filters=self.dims[stage_idx],
            kernel_size=downsample_kernel_size,
            strides=downsample_stride,
            # DECISION plan_2026-06-15_e6a0391c/D-003: "same" not "valid" — at
            # kernel==stride, "valid" collapses small CIFAR-scale inputs to 0x0 and produced NaN output. Identical to "valid" when the spatial dim divides the stride. See decisions.md.
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name=f"downsample_conv_{stage_idx - 1}"
        )
        self.downsample_layers_list.append([downsample_norm, downsample_conv])

    def _build_stage(self, stage_idx: int) -> None:
        """Build and assign one stage of ConvNeXt V2 blocks.

        Each entry appended to ``stages_list`` is a
        ``{"block": ..., "drop_path": ...}`` mapping; ``drop_path`` is ``None``
        when this block's ramped rate is 0.

        :param stage_idx: Index of the stage to build.
        :type stage_idx: int
        """
        stage_blocks = []
        depth = self.depths[stage_idx]
        dim = self.dims[stage_idx]
        total_blocks = sum(self.depths)
        block_start_idx = sum(self.depths[:stage_idx])
        # The drop-path ramp is global across stages, not per-stage: the index is
        # `block_start_idx + block_idx` over `sum(self.depths)` blocks, so stage 0
        # starts at 0.0 and the last block of the last stage reaches drop_path_rate.
        # `linear_drop_path_rates` already handles total_blocks <= 1 (all-zero).
        drop_path_rates = linear_drop_path_rates(total_blocks, self.drop_path_rate)

        for block_idx in range(depth):
            current_block_idx = block_start_idx + block_idx
            drop_rate = drop_path_rates[current_block_idx]

            block = ConvNextV2Block(
                kernel_size=self.kernel_size,
                filters=dim,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                spatial_dropout_rate=self.spatial_dropout_rate,
                use_gamma=self.use_gamma,
                use_softorthonormal_regularizer=self.use_softorthonormal_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )
            drop_path_cls = StochasticDepth if self.stochastic_mode == 'depth' else StochasticGradient
            drop_path = drop_path_cls(
                drop_path_rate=drop_rate,
                name=f"stage_{stage_idx}_block_{block_idx}_drop_path"
            ) if drop_rate > 0 else None
            stage_blocks.append({"block": block, "drop_path": drop_path})
        self.stages_list.append(stage_blocks)

    def _build_head(self) -> None:
        """Build and assign the GAP + LayerNorm + classifier head.

        ``self.classifier`` is ``None`` when ``num_classes == 0``, in which case
        the head returns pooled, normalized features.
        """
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        if self.num_classes > 0:
            self.classifier = keras.layers.Dense(
                units=self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=self.HEAD_INITIALIZER,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )
        else:
            self.classifier = None

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer by tracing ``call`` on a symbolic input.

        :param input_shape: Shape of the input to ``call``, with or without the
            batch dimension. A 3D shape (as ``summary()`` may pass) is given a
            dummy batch axis so sub-layers build correctly.
        :type input_shape: Any
        """
        # The summary() method might call build with a 3D shape (without batch dim).
        # We add a dummy batch dimension if that's the case to ensure layers build correctly.
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = input_shape
        # A dummy forward pass with a KerasTensor will correctly build all sub-layers.
        dummy_input = keras.KerasTensor(build_shape)
        _ = self.call(dummy_input)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the model.

        The residual add and the drop-path are applied HERE, not inside the
        block: :class:`ConvNextV2Block` is transform-only and returns ``F(x)``.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Output tensor. ``(batch_size, num_classes)`` with
            ``include_top=True``, ``(batch_size, dims[-1])`` when additionally
            ``num_classes == 0``, otherwise the final stage's feature maps
            ``(batch_size, H', W', dims[-1])``.
        :rtype: keras.KerasTensor
        """
        x = self.stem_conv(inputs)
        x = self.stem_norm(x)

        for stage_idx, stage_blocks in enumerate(self.stages_list):
            if stage_idx > 0:
                norm_layer, conv_layer = self.downsample_layers_list[stage_idx - 1]
                x = norm_layer(x)
                x = conv_layer(x)

            for block_info in stage_blocks:
                residual = x
                x = block_info["block"](x, training=training)
                if block_info["drop_path"]:
                    x = block_info["drop_path"](x, training=training)
                x = keras.layers.add([residual, x])

        if self.include_top:
            x = self.gap(x)
            x = self.head_norm(x)
            if self.classifier:
                x = self.classifier(x)

        return x

    # ---------------------------------------------------------------------

    # Spatial size used to materialize weights before a checkpoint load when the
    # model was constructed with the default, fully-shape-agnostic
    # `input_shape=(None, None, 3)`. 224 is the ImageNet size these checkpoints
    # are trained at, and it is the size the factory docstrings already promise.
    PRETRAINED_BUILD_SPATIAL = 224

    def _pretrained_build_shape(self) -> Tuple[int, ...]:
        """Resolve a concrete ``(H, W, C)`` for the pre-load dummy forward.

        :return: ``(height, width, channels)`` with any ``None`` spatial dim replaced by `PRETRAINED_BUILD_SPATIAL`.
        :rtype: Tuple[int, ...]
        :raises ValueError: If `input_shape` has no channel count.
        """
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-067: resolve None spatial dims
        # here rather than pass self.input_shape through — the default (None, None, 3) built (1, None, None, 3) and broke pretrained=<path> loads. See decisions.md.
        height, width, channels = self.input_shape
        if channels is None:
            raise ValueError(
                "Cannot materialize weights for a checkpoint load: input_shape "
                f"{self.input_shape} has no channel count. Pass a concrete "
                "input_shape=(height, width, channels) to the constructor."
            )
        height = self.PRETRAINED_BUILD_SPATIAL if height is None else height
        width = self.PRETRAINED_BUILD_SPATIAL if width is None else width
        return (int(height), int(width), int(channels))

    # ---------------------------------------------------------------------

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load pretrained weights into the model from a local checkpoint.

        Handles mismatches gracefully, which is what makes a checkpoint usable
        when the number of classes differs or when only the backbone is wanted.
        Weights are transferred layer-by-layer via
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
        the canonical replacement for ``self.load_weights(by_name=True)`` (which
        raises on ``.keras`` files in Keras 3.8+).

        :param weights_path: Path to the weights file (``.keras`` format).
        :type weights_path: str
        :param skip_mismatch: Whether to skip layers with mismatched shapes.
            Useful when loading weights with a different ``num_classes``. Maps
            to ``strict=not skip_mismatch``.
        :type skip_mismatch: bool
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If weights cannot be loaded. The original exception
            is chained with ``from e``.

        Example:
            >>> model = ConvNeXtV2.from_variant("tiny", num_classes=10)
            >>> model.load_pretrained_weights(
            ...     "convnext_v2_tiny_imagenet.keras", skip_mismatch=True)
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # Build model if not already built (weight transfer needs a built target)
            if not self.built:
                dummy_input = keras.random.normal(
                    (1,) + self._pretrained_build_shape()
                )
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )

            logger.info(report.summary_string())
            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped (e.g., classifier layer)."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            # `from e` so the real cause survives; this wrapper used to be the
            # only thing a caller saw.
            raise ValueError(
                f"Failed to load weights from {weights_path}: "
                f"{type(e).__name__}: {e}"
            ) from e

    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public ConvNeXt V2 weights ship with
        ``dl_techniques``. Always raises. Kept to mirror the house factory
        recipe (see ``models/vision/resnet/model.py``) and to give an explicit failure
        mode instead of a silent random-init fallback.

        :param variant: Variant name (unused).
        :type variant: str
        :param dataset: Dataset name (unused).
        :type dataset: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained ConvNeXt V2 weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: ConvNeXtV2.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: Union[bool, str] = False,
            weights_dataset: str = "imagenet",
            weights_input_shape: Optional[Tuple[int, ...]] = None,
            cache_dir: Optional[str] = None,
            **kwargs
    ) -> "ConvNeXtV2":
        """Create a ConvNeXt V2 model from a predefined variant.

        :param variant: One of ``"cifar10"``, ``"atto"``, ``"femto"``,
            ``"pico"``, ``"nano"``, ``"tiny"``, ``"base"``, ``"large"``,
            ``"huge"``.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input shape. ``None`` resolves to
            ``(None, None, 3)``; a pretrained load then materializes weights at
            224x224.
        :type input_shape: Optional[Tuple[int, ...]]
        :param pretrained: If a string, a path to a local weights file to load.
            If True, raises ``NotImplementedError`` -- no public ConvNeXt V2
            weights ship with ``dl_techniques``. If False (default), returns a
            randomly-initialized model.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset the checkpoint was trained on, one of
            ``"imagenet"`` or ``"imagenet22k"``. Selects the class count
            (1000 / 21841) that ``num_classes`` is compared against when
            deciding whether to skip the classifier.
        :type weights_dataset: str
        :param weights_input_shape: Input shape used during weight pretraining.
            Only needed when loading pretrained weights at a different
            ``input_shape``; a mismatch sets ``skip_mismatch``.
        :type weights_input_shape: Optional[Tuple[int, ...]]
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :param kwargs: Additional arguments passed to the constructor.
        :return: ConvNeXtV2 model instance.
        :rtype: ConvNeXtV2
        :raises ValueError: If ``variant`` is not recognized.
        :raises NotImplementedError: If ``pretrained`` is True.

        Example:
            >>> # Feature extractor for fine-tuning
            >>> model = ConvNeXtV2.from_variant("base", include_top=False)
            >>>
            >>> # Fine-tune on a custom dataset at a different input size
            >>> model = ConvNeXtV2.from_variant(
            ...     "pico",
            ...     num_classes=10,
            ...     input_shape=(32, 32, 3),
            ...     weights_input_shape=(224, 224, 3)
            ... )
            >>>
            >>> # Load from a local weights file
            >>> model = ConvNeXtV2.from_variant("large",
            ...                                 pretrained="path/to/weights.keras")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt V2-{variant.upper()} model")
        logger.info(f"from_variant received input_shape: {input_shape}")

        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant=variant,
                    dataset=weights_dataset,
                    cache_dir=cache_dir
                )

            # Determine if we need to skip mismatches
            include_top = kwargs.get("include_top", True)
            if include_top:
                # Check if num_classes matches pretrained weights
                pretrained_classes = 1000 if weights_dataset == "imagenet" else 21841
                if num_classes != pretrained_classes:
                    skip_mismatch = True
                    logger.info(
                        f"num_classes ({num_classes}) differs from pretrained "
                        f"({pretrained_classes}). Will skip classifier weights."
                    )

            # Handle different input shapes
            if weights_input_shape and input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Loading weights pretrained on {weights_input_shape} "
                    f"for model with input shape {input_shape}. "
                    f"Only backbone weights will be loaded."
                )
                skip_mismatch = True

        model = cls(
            num_classes=num_classes,
            depths=config["depths"],
            dims=config["dims"],
            input_shape=input_shape,
            **kwargs
        )

        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the model.

        Crucial for using this subclassed model within the Keras Functional API
        or anywhere static shape inference is needed. The blocks inside a stage
        are shape-preserving, so only the stem, the inter-stage downsamples and
        the head move the shape.

        :param input_shape: Input shape, channels-last.
        :type input_shape: Tuple[int, ...]
        :return: The corresponding output shape.
        :rtype: Tuple[int, ...]
        """
        # This assumes channels_last data format
        current_shape = input_shape

        # 1. Stem
        current_shape = self.stem_conv.compute_output_shape(current_shape)
        current_shape = self.stem_norm.compute_output_shape(current_shape)

        # 2. Stages
        for i in range(len(self.depths)):
            # Downsample layer
            if i > 0:
                norm_layer, conv_layer = self.downsample_layers_list[i - 1]
                current_shape = norm_layer.compute_output_shape(current_shape)
                current_shape = conv_layer.compute_output_shape(current_shape)

            # The blocks within a stage do not change the shape, so we can skip them.

        # 3. Head
        if self.include_top:
            current_shape = self.gap.compute_output_shape(current_shape)
            current_shape = self.head_norm.compute_output_shape(current_shape)
            if self.classifier:
                current_shape = self.classifier.compute_output_shape(current_shape)

        return current_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "dims": self.dims,
            "drop_path_rate": self.drop_path_rate,
            "stochastic_mode": self.stochastic_mode,
            "kernel_size": self.kernel_size,
            "activation": serialize_activation(self.activation),
            "use_bias": self.use_bias,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_softorthonormal_regularizer": self.use_softorthonormal_regularizer,
            "include_top": self.include_top,
            "input_shape": self.input_shape,
            "strides": self.strides
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtV2":
        """Create a model instance from its configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: ConvNeXtV2 model instance.
        :rtype: ConvNeXtV2
        """
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional ConvNeXt-specific information.

        Builds the model first if it is not already built, so the summary is
        never printed against an unmaterialized weight tree.

        :param kwargs: Additional arguments passed to ``keras.Model.summary``.
        """
        if not self.built:
            dummy_input = keras.KerasTensor(self.input_shape)
            self.build(dummy_input.shape)

        super().summary(**kwargs)

        total_blocks = sum(self.depths)
        logger.info("ConvNeXt V2 configuration:")
        logger.info(f"  - Input shape: ({self.input_height}, {self.input_width}, {self.input_channels})")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Original dimensions: {self.dims}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Kernel size: {self.kernel_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")


# ---------------------------------------------------------------------

def create_convnext_v2(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = (None, None, 3),
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        weights_input_shape: Optional[Tuple[int, ...]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
) -> ConvNeXtV2:
    """Convenience function to create ConvNeXt V2 models.

    Thin wrapper around :meth:`ConvNeXtV2.from_variant` exposing the most
    common construction arguments at module level.

    :param variant: Model variant, one of ``"cifar10"``, ``"atto"``,
        ``"femto"``, ``"pico"``, ``"nano"``, ``"tiny"``, ``"base"``,
        ``"large"``, ``"huge"``.
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Input shape. Defaults to ``(None, None, 3)``.
    :type input_shape: Optional[Tuple[int, ...]]
    :param pretrained: If a string, a path to a local weights file. If True,
        raises ``NotImplementedError`` -- no public ConvNeXt V2 weights ship
        with ``dl_techniques``. If False (default), random initialization.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for pretrained weights, ``"imagenet"`` or
        ``"imagenet22k"``.
    :type weights_dataset: str
    :param weights_input_shape: Input shape used during weight pretraining.
        Only needed when loading pretrained weights at a different
        ``input_shape``.
    :type weights_input_shape: Optional[Tuple[int, ...]]
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param kwargs: Additional arguments passed to the model constructor.
    :return: ConvNeXtV2 model instance.
    :rtype: ConvNeXtV2
    :raises NotImplementedError: If ``pretrained`` is True.
    :raises ValueError: If ``variant`` is not recognized.

    Example:
        >>> # Create ConvNeXt V2-Tiny (randomly initialized; no weights ship here)
        >>> model = create_convnext_v2("tiny")
        >>>
        >>> # Create ConvNeXt V2-Base as feature extractor
        >>> model = create_convnext_v2("base", include_top=False)
        >>>
        >>> # Fine-tune on CIFAR-10 from a local checkpoint
        >>> model = create_convnext_v2(
        ...     "pico",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3),
        ...     pretrained="path/to/weights.keras",
        ...     weights_input_shape=(224, 224, 3)
        ... )
        >>>
        >>> # `pretrained=True` raises NotImplementedError -- pass a local path.
    """
    model = ConvNeXtV2.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------