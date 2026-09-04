"""``MobileNetV4``, a seven-stage `UniversalInvertedBottleneck` tower with optional Mobile Multi-Query Attention, plus the ``create_mobilenetv4`` factory.

The Universal Inverted Bottleneck makes depthwise-convolution placement a
per-stage choice: an optional depthwise before the expansion and another
after it, so one block specializes to an inverted bottleneck (`"IB"`,
middle depthwise only), a ConvNeXt-style block (`"ConvNext"`, pre-expansion
depthwise, `7x7` kernel), a transformer FFN (`"FFN"`, no depthwise), or
both (`"ExtraDW"`). Mobile MQA shares one key/value head across all query
heads, which favors mobile accelerators since K/V loading, not the matmul,
dominates their cost; it strides keys and values down 2x spatially, and
enters the residual through a learnable scalar gate initialized to one.

`MODEL_VARIANTS` here are hand-written depth/width ladders, not the paper's
NAS-found MNv4-Conv-S/M/L tables, so paper accuracy and latency numbers do
not apply to models built here. Variant keys are `"small"`, `"medium"`,
`"large"`, `"hybrid_medium"`, `"hybrid_large"`. Only the hybrid variants
carry attention, appended once at the end of stages 5 and 6; `MobileMQA`
has no positional encoding, so it depends entirely on the convolutional
stack for spatial ordering, and its `num_heads=8` default means a stage
width not divisible by 8 raises at construction. `width_multiplier` scales
`dims` by plain truncation, with no round-to-multiple-of-8 rule. No
pretrained checkpoints ship with this package; `pretrained=True` raises
`NotImplementedError`.

References:
    - Qin et al., 2024. MobileNetV4: Universal Models for the Mobile Ecosystem.
      (https://arxiv.org/abs/2404.10518)
    - Shazeer, 2019. Fast Transformer Decoding: One Write-Head is All You Need.
      (https://arxiv.org/abs/1911.02150) — the multi-query attention Mobile MQA
      specializes.
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear Bottlenecks.
      (https://arxiv.org/abs/1801.04381)
    - Liu et al., 2022. A ConvNet for the 2020s.
      (https://arxiv.org/abs/2201.03545) — the ConvNeXt block the UIB search space
      subsumes.
"""

import keras
from keras import layers, regularizers
from typing import List, Tuple, Optional, Dict, Any, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention.mobile_mqa import MobileMQA
from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.vision.mobilenet.common import (
    REFERENCE_BN_EPSILON,
    REFERENCE_BN_MOMENTUM,
    materialize_for_summary,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mobilenet.mobilenet_v4")
class MobileNetV4(keras.Model):
    """Seven-stage Universal Inverted Bottleneck tower, optionally hybrid with Mobile MQA.

    Architecture:

    .. code-block:: text

        image  [H, W, 3]
           |
           v
        Conv2D 3x3 s2 -> BN -> ReLU   (dims[0])
           |
           v
        7x stage:
           UniversalInvertedBottleneck x depths[i]  (block_types[i] shape)
           MobileMQA (only stages in attention_stages, hybrid variants)
           |
           v
        GlobalAvgPool -> Dense(1280) -> ReLU -> Dropout -> Dense (softmax)
           |
           v
        class probabilities  [num_classes]

    :param num_classes: Number of output classes; used only if `include_top=True`.
    :param depths: UIB block count per stage.
    :param dims: Channel count per stage.
    :param block_types: UIB structure per stage: `"IB"`, `"ConvNext"`, `"ExtraDW"`, `"FFN"`.
    :param strides: Stride for the first block of each stage.
    :param width_multiplier: Channel-count multiplier, applied by truncation.
    :param use_attention: Add Mobile MQA to `attention_stages`.
    :param attention_stages: Stage indices to add attention to, when `use_attention=True`.
    :param dropout_rate: Dropout rate in the classifier head.
    :param weight_decay: L2 regularization factor for all layers.
    :param kernel_initializer: Weight initialization strategy.
    :param include_top: Whether to include the classification head.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If `depths`, `dims`, `block_types`, or `strides` have
        different lengths, an invalid block type is given, or an attention
        stage index is out of range.

    Example::

        model = MobileNetV4.from_variant("medium", num_classes=1000)
        small = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "small": {
            "depths": [1, 1, 2, 3, 2, 2, 1],
            "dims": [16, 24, 32, 64, 96, 128, 160],
            "block_types": ["IB", "IB", "IB", "ExtraDW", "IB", "IB", "IB"],
            "use_attention": False,
        },
        "medium": {
            "depths": [1, 2, 3, 4, 3, 3, 1],
            "dims": [16, 24, 40, 80, 112, 192, 320],
            "block_types": ["IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"],
            "use_attention": False,
        },
        "large": {
            "depths": [1, 2, 4, 5, 4, 4, 1],
            "dims": [24, 32, 48, 96, 136, 224, 384],
            "block_types": ["IB", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "IB"],
            "use_attention": False,
        },
        "hybrid_medium": {
            "depths": [1, 2, 3, 4, 3, 3, 1],
            "dims": [16, 24, 40, 80, 112, 192, 320],
            "block_types": ["IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"],
            "use_attention": True,
            "attention_stages": [5, 6],
        },
        "hybrid_large": {
            "depths": [1, 2, 4, 5, 4, 4, 1],
            "dims": [24, 32, 48, 96, 136, 224, 384],
            "block_types": ["IB", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "IB"],
            "use_attention": True,
            "attention_stages": [5, 6],
        },
    }

    # Architecture constants
    STEM_KERNEL_SIZE = 3
    STEM_STRIDE = 2
    DEFAULT_STRIDES = [1, 2, 2, 2, 1, 2, 1]
    DEFAULT_ATTENTION_STAGES = [5, 6]
    HEAD_HIDDEN_DIM = 1280
    LAYERNORM_EPSILON = 1e-6

    def __init__(
        self,
        num_classes: int = 1000,
        depths: Sequence[int] = (1, 2, 3, 4, 3, 3, 1),
        dims: Sequence[int] = (16, 24, 40, 80, 112, 192, 320),
        block_types: Sequence[str] = ("IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"),
        strides: Sequence[int] = (1, 2, 2, 2, 1, 2, 1),
        width_multiplier: float = 1.0,
        use_attention: bool = False,
        attention_stages: Sequence[int] = (5, 6),
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-5,
        kernel_initializer: str = "he_normal",
        include_top: bool = True,
        input_shape: Tuple[int, ...] = (224, 224, 3),
        **kwargs
    ):
        super().__init__(**kwargs)

        valid_block_types = {"IB", "ConvNext", "ExtraDW", "FFN"}
        for block_type in block_types:
            if block_type not in valid_block_types:
                raise ValueError(
                    f"Invalid block type '{block_type}'. "
                    f"Must be one of {valid_block_types}"
                )

        stage_configs = [depths, dims, block_types, strides]
        stage_lengths = [len(config) for config in stage_configs]
        if not all(length == stage_lengths[0] for length in stage_lengths):
            raise ValueError(
                f"All stage configurations must have same length. Got: "
                f"depths={len(depths)}, dims={len(dims)}, "
                f"block_types={len(block_types)}, strides={len(strides)}"
            )

        if use_attention:
            max_stage_idx = len(depths) - 1
            for stage_idx in attention_stages:
                if not (0 <= stage_idx <= max_stage_idx):
                    raise ValueError(
                        f"Attention stage index {stage_idx} out of range. "
                        f"Must be in [0, {max_stage_idx}]"
                    )

        if input_shape and len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        if input_shape:
            height, width, channels = input_shape
            if channels not in [1, 3]:
                logger.warning(f"Unusual number of channels: {channels}")

        self.num_classes = num_classes
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: store as a list even though the default is a tuple.
        # get_config has always emitted a list, so keeping this conversion matches every saved config's JSON shape. See decisions.md.
        self.depths = list(depths)
        self.dims = list(dims)
        self.block_types = list(block_types)
        self.strides = list(strides)
        self.width_multiplier = width_multiplier
        self.use_attention = use_attention
        self.attention_stages = list(attention_stages)
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self._input_shape = input_shape

        self.actual_dims = [int(dim * width_multiplier) for dim in dims]

        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        self.stem_conv, self.stem_bn, self.stem_activation = self._build_stem()

        self.stages = []
        for stage_idx in range(len(self.depths)):
            stage_layers = self._build_stage(stage_idx)
            self.stages.append(stage_layers)

        if self.include_top:
            self.head_layers = self._build_head()

    def call(self, x, training=None):
        """Run the stem, stages, and optional head.

        :param x: Input images, shape `(B, H, W, C)`.
        :param training: Passed to batch norm and dropout.
        :return: Class probabilities, or the 4-D feature map if `include_top=False`.
        """
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_activation(x)

        for stage_layers in self.stages:
            for layer in stage_layers:
                x = layer(x, training=training)

        if self.include_top:
            for layer in self.head_layers:
                if isinstance(layer, layers.Dropout):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        return x

    def _build_stem(self):
        """Build the stem's conv, batch norm, and activation layers."""
        stem_conv = layers.Conv2D(
            filters=self.actual_dims[0],
            kernel_size=self.STEM_KERNEL_SIZE,
            strides=self.STEM_STRIDE,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        stem_bn = layers.BatchNormalization(
            momentum=REFERENCE_BN_MOMENTUM,
            epsilon=REFERENCE_BN_EPSILON,
            name="stem_bn",
        )
        stem_activation = layers.ReLU(name="stem_relu")
        return stem_conv, stem_bn, stem_activation

    def _build_stage(self, stage_idx: int):
        """Build the UIB blocks, and optional Mobile MQA, for one stage.

        :param stage_idx: Index into `depths`/`dims`/`block_types`/`strides`.
        :return: List of layers for this stage, in call order.
        """
        stage_layers = []
        depth = self.depths[stage_idx]
        dim = self.actual_dims[stage_idx]
        block_type = self.block_types[stage_idx]
        stage_stride = self.strides[stage_idx]

        logger.info(
            f"Building stage {stage_idx}: {depth} blocks, {dim} dims, "
            f"type={block_type}, stride={stage_stride}"
        )

        # DECISION plan-2026-08-14T183218-f4c612aa/D-011: map block_type to which depthwise POSITION is occupied, not to use_dw1/use_dw2.
        # Both use_dw1/use_dw2 are post-expansion, so that mapping cannot express a pre-expansion start depthwise and built a plain IB for every stage. See decisions.md.
        block_structure = {
            "IB": dict(use_start_dw=False, use_dw1=True, use_dw2=False),
            "ConvNext": dict(
                use_start_dw=True, start_dw_kernel_size=7,
                use_dw1=False, use_dw2=False,
            ),
            "ExtraDW": dict(use_start_dw=True, use_dw1=True, use_dw2=False),
            "FFN": dict(use_start_dw=False, use_dw1=False, use_dw2=False),
        }[block_type]

        for block_idx in range(depth):
            block_stride = stage_stride if block_idx == 0 else 1
            block = UniversalInvertedBottleneck(
                filters=dim,
                stride=block_stride,
                block_type=block_type,
                # DECISION plan-2026-08-22T035419-a11304c8/D-203: pass epsilon explicitly; see mobilenet_v1.py for the measured impact. See decisions.md.
                normalization_args={'epsilon': REFERENCE_BN_EPSILON},
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}",
                **block_structure
            )
            stage_layers.append(block)

        if self.use_attention and stage_idx in self.attention_stages:
            logger.info(f"Adding Mobile MQA to stage {stage_idx}")
            mqa_layer = MobileMQA(
                dim=dim,
                use_downsampling=True,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"stage_{stage_idx}_mqa"
            )
            stage_layers.append(mqa_layer)

        return stage_layers

    def _build_head(self):
        """Build the pooling, hidden dense, and classifier layers."""
        head_layers_list = []
        gap = layers.GlobalAveragePooling2D(name="global_avg_pool")
        head_layers_list.append(gap)

        if self.HEAD_HIDDEN_DIM > 0:
            hidden_dense = layers.Dense(
                self.HEAD_HIDDEN_DIM,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="head_hidden"
            )
            hidden_activation = layers.ReLU(name="head_hidden_relu")
            hidden_dropout = layers.Dropout(self.dropout_rate, name="head_dropout")
            head_layers_list.extend([hidden_dense, hidden_activation, hidden_dropout])

        if self.num_classes > 0:
            classifier = layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                activation='softmax',
                name="classifier"
            )
            head_layers_list.append(classifier)

        return head_layers_list

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        width_multiplier: float = 1.0,
        **kwargs
    ) -> "MobileNetV4":
        """Create a MobileNetV4 model from a predefined variant.

        :param variant: One of `"small"`, `"medium"`, `"large"`, `"hybrid_medium"`, `"hybrid_large"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param width_multiplier: Channel-count multiplier.
        :param kwargs: Passthrough to the constructor.
        :return: A configured `MobileNetV4` instance.
        :raises ValueError: If `variant` is not recognized.

        Example::

            model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
            hybrid = MobileNetV4.from_variant("hybrid_medium", num_classes=1000)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        if input_shape is None:
            input_shape = (224, 224, 3)

        logger.info(f"Creating MobileNetV4-{variant} model")
        logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

        return cls(
            num_classes=num_classes,
            depths=config["depths"],
            dims=config["dims"],
            block_types=config["block_types"],
            strides=config.get("strides", cls.DEFAULT_STRIDES),
            width_multiplier=width_multiplier,
            use_attention=config["use_attention"],
            attention_stages=config.get("attention_stages", cls.DEFAULT_ATTENTION_STAGES),
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "dims": self.dims,
            "block_types": self.block_types,
            "strides": self.strides,
            "width_multiplier": self.width_multiplier,
            "use_attention": self.use_attention,
            "attention_stages": self.attention_stages,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": self.kernel_initializer,
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileNetV4":
        """Create a model from its `get_config()` output."""
        return cls(**config)

    def summary(self, **kwargs):
        """Print the model summary, plus configuration and parameter count."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: materialize with a real forward pass.
        # self.build(self._input_shape) passed an unbatched shape where Model.build() expects a batch shape, and even fixed would materialize no sub-layer weights. See decisions.md.
        materialize_for_summary(self, self._input_shape)

        super().summary(**kwargs)

        total_blocks = sum(self.depths)
        total_params = self.count_params()

        logger.info("MobileNetV4 Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Original dimensions: {self.dims}")
        logger.info(f"  - Actual dimensions: {self.actual_dims}")
        logger.info(f"  - Block types: {self.block_types}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Width multiplier: {self.width_multiplier}")
        logger.info(f"  - Use attention: {self.use_attention}")
        if self.use_attention:
            logger.info(f"  - Attention stages: {self.attention_stages}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv4(
    variant: str = "medium",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = None,
    width_multiplier: float = 1.0,
    pretrained: bool = False,
    **kwargs
) -> MobileNetV4:
    """Create a MobileNetV4 model.

    :param variant: Model variant: `"small"`, `"medium"`, `"large"`, `"hybrid_medium"`, `"hybrid_large"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param width_multiplier: Channel-count multiplier.
    :param pretrained: Must be `False`; `True` raises `NotImplementedError`
        since no MobileNetV4 checkpoints ship with this package.
    :param kwargs: Passthrough to the constructor.
    :return: A configured `MobileNetV4` instance.

    Example::

        model = create_mobilenetv4("small", num_classes=10, input_shape=(32, 32, 3))
        hybrid = create_mobilenetv4("hybrid_medium", num_classes=1000)
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise on pretrained=True, do not warn and return random weights.
    # See decisions.md.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained MobileNetV4 weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_mobilenetv4('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )

    model = MobileNetV4.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=width_multiplier,
        **kwargs
    )

    return model

# ------------------------------------------------------------------------
