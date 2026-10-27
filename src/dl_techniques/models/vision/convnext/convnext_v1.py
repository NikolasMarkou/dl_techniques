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

import copy
import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import (
    materialize_sublayers,
    plain_head_config_overrides,
)
from dl_techniques.utils.weight_transfer import (
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.conv_blocks.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.regularization.stochastic_gradient import StochasticGradient
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head
from dl_techniques.utils.keras_registration import register_dl_technique

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (the regularizers, the
# dropout and drop-path rates, the input size). `load_pretrained_checkpoint` compares
# every other key the checkpoint and the model both hold, so a changed `kernel_size`,
# `activation`, `strides` or `stochastic_mode` is refused even when every weight shape
# still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "input_shape",
    "kernel_regularizer",
    "use_softorthonormal_regularizer",
    "dropout_rate",
    "spatial_dropout_rate",
    "drop_path_rate",
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.convnext.convnext_v1")
class ConvNeXtV1(keras.Model):
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

    # Architecture constants
    LAYERNORM_EPSILON = 1e-6
    # DECISION plan-2026-10-14T062616-81938e81/D-076
    # The reference's _init_weights draws every Conv2d, the stem and the
    # downsample convs included, from trunc_normal_(std=.02). Do NOT go back to
    # the "truncated_normal" string: Keras' default stddev is 0.05, 2.5x the
    # reference. A config dict, so each conv resolves its own instance.
    # See decisions.md D-076.
    STEM_INITIALIZER: Dict[str, Any] = {
        "class_name": "TruncatedNormal",
        "config": {"mean": 0.0, "stddev": 0.02},
    }
    # DECISION plan-2026-10-14T062616-81938e81/D-013
    # The reference initializes the head Linear with trunc_normal_(std=.02).
    # Do NOT revert to the "truncated_normal" string: Keras' default stddev is
    # 0.05. A config dict, not an instance: a shared instance replays one draw.
    # See decisions.md D-013.
    HEAD_INITIALIZER: Dict[str, Any] = {
        "class_name": "TruncatedNormal",
        "config": {"mean": 0.0, "stddev": 0.02},
    }
    # Spatial size used to materialize weights before a checkpoint load when the
    # model was constructed with the default, fully-shape-agnostic
    # `input_shape=(None, None, 3)`. 224 is the ImageNet size these checkpoints
    # are trained at, and it is the size the factory docstrings already promise.
    PRETRAINED_BUILD_SPATIAL = 224

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
            kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
            dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            strides: int = 4,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = (None, None, 3),
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )

        depths = list(depths) if depths is not None else [3, 3, 9, 3]
        dims = list(dims) if dims is not None else [96, 192, 384, 768]

        if len(depths) != len(dims):
            raise ValueError(
                f"depths and dims must have the same length, got depths={depths} and dims={dims}"
            )

        # A built-in variant's depths (the 2-stage ``cifar10``) are deliberate, and they
        # arrive here as explicit arguments both from ``from_variant`` and after
        # ``from_config`` or a ``.keras`` reload, so the check compares the values.
        if len(depths) != 4 and not any(
                depths == list(v["depths"]) for v in self.MODEL_VARIANTS.values()):
            logger.warning(f"ConvNeXt typically uses 4 stages, got {len(depths)} stages")

        if strides <= 0:
            raise ValueError(f"strides must be positive, got {strides}")

        if stochastic_mode not in ('depth', 'gradient'):
            raise ValueError(
                f"stochastic_mode must be 'depth' or 'gradient', got {stochastic_mode!r}"
            )

        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")
        if input_shape[2] not in (1, 3):
            logger.warning(
                f"Unusual number of channels: {input_shape[2]}. ConvNeXt typically uses 3 channels")

        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.stochastic_mode = stochastic_mode
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer
        self.include_top = include_top
        self.strides = strides
        self.input_shape_config = input_shape

        self._head_config_overrides = plain_head_config_overrides(
            head_config_overrides, type(self).__name__
        )

        self._build_stem()

        self.downsamplers = []
        self.stages = []
        for stage_idx in range(len(self.depths)):
            if stage_idx > 0:
                self._build_downsampler(stage_idx)
            self._build_stage(stage_idx)

        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created {type(self).__name__} for input {input_shape} "
            f"with {sum(depths)} blocks"
        )

    def _build_stem(self) -> None:
        """Build the patchify stem via :class:`ConvStem`.

        A ``strides x strides`` convolution at stride ``strides`` followed by
        LayerNorm. The padding rule is deliberately NOT the downsamplers'
        unconditional ``"same"``; see the D-125 anchor in the body.
        """
        self.stem = ConvStem(
            filters=self.dims[0],
            kernel_size=self.strides,
            strides=self.strides,
            # DECISION plan-2026-08-19T163559-499b6f0e/D-125: not the unconditional
            # "same" the downsample layers use — "same" here would silently shift the spatial geometry of checkpoints trained at a non-divisible input size. See decisions.md.
            padding="same" if self.strides == 1 else "valid",
            use_bias=self.use_bias,
            num_convs=1,
            normalization_type="layer_norm",
            normalization_kwargs={
                "epsilon": self.LAYERNORM_EPSILON,
                "center": self.use_bias,
                "scale": True,
            },
            activation_type="linear",
            use_pooling=False,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name="stem",
        )

    def _build_downsampler(self, stage_idx: int) -> None:
        """Build the LayerNorm and strided convolution that feed a stage.

        :param stage_idx: Index of the stage the downsampler feeds; it must be
            greater than 0. The pair is appended to ``downsamplers``, so it sits
            at ``stage_idx - 1`` and is named ``downsample_{stage_idx - 1}_*``.
        """
        norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_{stage_idx - 1}_norm"
        )
        conv = keras.layers.Conv2D(
            filters=self.dims[stage_idx],
            kernel_size=self.strides,
            strides=self.strides,
            # DECISION plan_2026-06-15_e6a0391c/D-003: "same" not "valid" — at
            # kernel==stride, "valid" collapses small CIFAR-scale inputs to 0x0 and produced NaN output. Identical to "valid" when the spatial dim divides the stride. See decisions.md.
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name=f"downsample_{stage_idx - 1}_conv"
        )
        self.downsamplers.append({"norm": norm, "conv": conv})

    def _build_stage(self, stage_idx: int) -> None:
        """Build one stage of blocks.

        Each entry appended to ``stages`` is a list of ``{"block": ..., "drop_path": ...}``
        mappings; ``drop_path`` is ``None`` when the block's ramped rate is 0.

        :param stage_idx: Index of the stage to build.
        """
        stage_blocks = []
        depth = self.depths[stage_idx]
        dim = self.dims[stage_idx]
        block_start_idx = sum(self.depths[:stage_idx])
        # The drop-path ramp is global across stages, not per stage: the index is
        # `block_start_idx + block_idx` over `sum(self.depths)` blocks, so stage 0
        # starts at 0.0 and the last block of the last stage reaches drop_path_rate.
        # `linear_drop_path_rates` already handles a single block (all zero).
        drop_path_rates = linear_drop_path_rates(sum(self.depths), self.drop_path_rate)
        drop_path_cls = StochasticDepth if self.stochastic_mode == 'depth' else StochasticGradient

        for block_idx in range(depth):
            drop_rate = drop_path_rates[block_start_idx + block_idx]
            block_name = f"stage{stage_idx + 1}_block{block_idx + 1}"

            block = ConvNextV1Block(
                kernel_size=self.kernel_size,
                filters=dim,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                spatial_dropout_rate=self.spatial_dropout_rate,
                use_gamma=self.use_gamma,
                use_softorthonormal_regularizer=self.use_softorthonormal_regularizer,
                name=block_name
            )
            drop_path = drop_path_cls(
                drop_path_rate=drop_rate,
                name=f"{block_name}_drop_path"
            ) if drop_rate > 0 else None
            stage_blocks.append({"block": block, "drop_path": drop_path})
        self.stages.append(stage_blocks)

    def _build_head(self) -> None:
        """Build the classification head through the vision heads factory.

        The reference head (facebookresearch/ConvNeXt{,-V2} models/convnext*.py:
        ``self.norm(x.mean([-2, -1]))`` then ``self.head``): global average
        pooling, LayerNorm at the reference's epsilon 1e-6 (the factory's
        ``layer_norm`` default) and one linear layer, no hidden block.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                'num_classes': self.num_classes,
                'dropout_rate': self.dropout_rate,
                'normalization_type': 'layer_norm',
                'activation_type': 'gelu',
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
                'block_dims': (),
                'pooled_normalization_type': 'layer_norm',
                'kernel_initializer': self.HEAD_INITIALIZER,
                'kernel_regularizer': self.kernel_regularizer,
                # An unnamed head is named `classification_head` in whichever
                # process creates it first and `classification_head_1`, `_2`, ...
                # afterwards. Weight transfer matches layers BY NAME, so a
                # checkpoint written by one process and loaded by another left
                # the classifier at its initialization while the report still
                # read as a success. Naming it makes the path stable.
                'name': 'classification_head',
            },
            self._head_config_overrides,
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from `input_shape` by tracing `call`.

        `materialize_sublayers` traces `call` on symbolic inputs, so what
        gets built cannot drift from what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to `call`.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the stem, the stages with their downsamplers, and the head.

        The residual add and the drop-path are applied HERE, not inside the
        block: :class:`ConvNextV1Block` is transform-only and returns ``F(x)``.

        :param inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        :param training: Whether dropout and stochastic depth run in training mode.
        :return: The logits `(batch_size, num_classes)` (`include_top=True`) or
            the final stage's feature maps `(batch_size, H', W', dims[-1])`
            (`include_top=False`).
        """
        x = self.stem(inputs, training=training)

        for stage_idx, stage_blocks in enumerate(self.stages):
            if stage_idx > 0:
                downsampler = self.downsamplers[stage_idx - 1]
                x = downsampler["norm"](x)
                x = downsampler["conv"](x)

            for entry in stage_blocks:
                residual = x
                x = entry["block"](x, training=training)
                if entry["drop_path"]:
                    x = entry["drop_path"](x, training=training)
                x = keras.layers.add([residual, x])

        if self.include_top:
            return self.classification_head(x, training=training)['logits']

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Report the shape `call` returns, without building the model.

        The blocks inside a stage are shape-preserving, so only the stem, the
        downsamplers and the head move the shape. This is what lets the model sit in
        the Keras Functional API, where static shape inference is needed.

        :param input_shape: `(batch_size, height, width, channels)`, channels-last;
            the spatial axes may be `None`.
        :return: The logits shape `(batch_size, num_classes)` (`include_top=True`)
            or the final feature map `(batch_size, H', W', dims[-1])`
            (`include_top=False`).
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-069: this method exists because
        # ConvNeXtV2 implements it and ConvNeXtV1 previously raised NotImplementedError for the same call. Mirrors convnext_v2.py's body. See decisions.md.
        shape = self.stem.compute_output_shape(input_shape)

        for downsampler in self.downsamplers:
            shape = downsampler["norm"].compute_output_shape(shape)
            shape = downsampler["conv"].compute_output_shape(shape)

        if self.include_top:
            # The head reports both keys; `call` returns the 'logits' one.
            shape = self.classification_head.compute_output_shape(shape)['logits']

        return shape

    def _pretrained_build_shape(self) -> Tuple[int, ...]:
        """Resolve a concrete `(H, W, C)` for the build that precedes a checkpoint load.

        :return: `(height, width, channels)` with any `None` spatial dim replaced by
            `PRETRAINED_BUILD_SPATIAL`.
        :raises ValueError: If `input_shape` has no channel count.
        """
        # (The anchor below says `self.input_shape`; the attribute is `self.input_shape_config`.)
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-067: resolve None spatial dims
        # here rather than pass self.input_shape through — the default (None, None, 3) built (1, None, None, 3) and broke pretrained=<path> loads. See decisions.md.
        height, width, channels = self.input_shape_config
        if channels is None:
            raise ValueError(
                "Cannot materialize weights for a checkpoint load: input_shape "
                f"{self.input_shape_config} has no channel count. Pass a concrete "
                "input_shape=(height, width, channels) to the constructor."
            )
        height = self.PRETRAINED_BUILD_SPATIAL if height is None else height
        width = self.PRETRAINED_BUILD_SPATIAL if width is None else width
        return (int(height), int(width), int(channels))

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load a local `.keras` checkpoint into the model, layer by layer.

        The transfer is `load_pretrained_checkpoint`: it builds the model at
        `input_shape` (a `None` spatial dim at `PRETRAINED_BUILD_SPATIAL`) if it
        is unbuilt and takes a headed checkpoint into a headless model. Every
        layer outside the classification head must be restored, so a checkpoint
        of another variant, depth, width or kernel size raises and leaves the
        model untouched. Only a head of another `num_classes` is skipped: it
        keeps its initialization and a warning names it. The archive's own
        config is compared too: a constructor argument that both hold and that
        is not in `_NON_STRUCTURAL_CONFIG_KEYS` must be equal, so a checkpoint
        of another `activation` or `stochastic_mode` is refused even when every
        weight shape fits. An argument only one side holds is not compared, and
        an archive without a config of this class is checked by layers only.

        :param weights_path: Path to the weights file (`.keras` format).
        :param skip_mismatch: `True` (default) lets the head layers differ;
            `False` allows no difference anywhere. The inverse of the transfer
            helper's `strict`.
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If a layer outside the head is not restored, or
            the weights cannot be loaded.
        """
        load_pretrained_checkpoint(
            self, weights_path, self._pretrained_build_shape(), skip_mismatch,
            ignore_config_keys=_NON_STRUCTURAL_CONFIG_KEYS,
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: Union[bool, str] = False,
            **kwargs
    ) -> "ConvNeXtV1":
        """Create a ConvNeXt V1 model from a predefined variant.

        :param variant: One of `"cifar10"`, `"tiny"`, `"small"`, `"base"`, `"large"`, `"xlarge"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape. `None` resolves to `(None, None, 3)`; a
            checkpoint load then materializes weights at 224x224.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no public ConvNeXt V1 weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; an architecture field of
            the variant (e.g. `depths`, `dims`) is replaced by the value given here.
        :return: A configured ConvNeXtV1 instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

        Example::

            model = ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
            backbone = ConvNeXtV1.from_variant("base", include_top=False)
            shallow = ConvNeXtV1.from_variant("tiny", depths=[1, 1, 1, 1])
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # Raises for True instead of falling back to random init; do not reinstate a warn-and-return branch.
        weights_path = pretrained_path_or_raise(
            pretrained, f"{cls.__name__} variant '{variant}'"
        )

        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)

        model = cls(num_classes=num_classes, input_shape=input_shape, **config)

        if weights_path:
            model.load_pretrained_weights(weights_path)

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization."""
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
            "input_shape": self.input_shape_config,
            "strides": self.strides,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self, **kwargs) -> None:
        """Print the model summary; an unbuilt model is materialized first.

        :param kwargs: Passed to `keras.Model.summary`.
        """
        if not self.built:
            self.build((None, *self.input_shape_config))

        super().summary(**kwargs)


# ---------------------------------------------------------------------

def create_convnext_v1(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs
) -> ConvNeXtV1:
    """Create a ConvNeXt V1 model.

    :param variant: Model variant: `"cifar10"`, `"tiny"`, `"small"`, `"base"`, `"large"`, `"xlarge"`.
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
        tuned = create_convnext_v1("small", num_classes=10, input_shape=(32, 32, 3),
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
