"""
A MobileNetV4-shaped classifier: a seven-stage `UniversalInvertedBottleneck` tower
with optional Mobile Multi-Query Attention in the last two stages.

The generation's two published contributions are a *search space* and an attention
block. The Universal Inverted Bottleneck generalizes V2's inverted residual by
making the placement of depthwise convolutions a searchable choice rather than a
fixed one: with an optional depthwise before the expansion and another after it,
one parameterized block specializes to the inverted bottleneck (`dw` after
expansion only), to a ConvNeXt-style block (`dw` before expansion, large kernel),
to a transformer FFN (no depthwise at all), and to ExtraDW (both), so a NAS can
trade spatial mixing against channel mixing per stage without changing block
implementations. Mobile MQA is the second: multi-query attention with a *single*
shared key/value head, which matters on mobile accelerators because those are
bandwidth-bound rather than FLOP-bound, and K/V loading — not the matmul — is what
dominates. It additionally strides the keys and values down by a factor of two
spatially before attending, so the score matrix is `N x N/4` instead of `N x N`,
and it enters the residual through a learnable scalar, `x + lambda * Attn(x)`,
initialized to one.

The per-stage `block_types` entries — `"IB"`, `"ConvNext"`, `"ExtraDW"`, `"FFN"` —
select real structure. `_build_stage` maps each one to which of the block's two
optional depthwise POSITIONS is occupied: `IB` takes the middle depthwise only,
`ConvNext` the pre-expansion start depthwise only (with a `7x7` kernel, the
convention that position exists for), `ExtraDW` both, `FFN` neither. **The
selector is the position, not the count** — `IB` and `ConvNext` each own exactly
one depthwise convolution and are different architectures, distinguishable only
by the channel count it operates on (`ConvNext`'s sees the unexpanded input,
`IB`'s the expanded tensor).

This was worth doing carefully because the obvious mapping is wrong. These entries
used to be inert labels, and the fix this docstring itself prescribed was to pass
`use_dw1` / `use_dw2` through from `_build_stage`. That would not have produced the
paper's structures: both of those depthwise convolutions sit AFTER the expansion,
so toggling them varies how many *middle* depthwise convs a block has (0, 1 or 2)
and can never place one before the expansion. Implementing `ConvNext` and
`ExtraDW` faithfully required a genuinely new third slot on
`UniversalInvertedBottleneck` (`use_start_dw`, with its own `start_dw_kernel_size`),
which is what `_build_stage` now drives. `use_dw2` — a second middle depthwise —
survives as a layer-level knob but is no longer any block type's meaning.

**What this module builds is still a simplification of the paper, and the
remaining gap is worth stating plainly before the tables below are read as the
paper's.** The stage tables in `MODEL_VARIANTS` are hand-written depth/width ladders,
not the NAS-found MNv4-Conv-S/M/L specifications: there are no per-block kernel
sizes and no per-block expansion ratios, which is exactly the freedom the UIB
search space exists to exploit. Accuracy or latency numbers from the paper should
not be attributed to models built here, and the variant keys are `"small"`,
`"medium"`, `"large"`, `"hybrid_medium"`, `"hybrid_large"` — there is no
`"conv_small"` or `"conv_medium"`.

What the code does build is coherent on its own terms. A `3x3` stride-2 ReLU stem
into `dims[0]`, then seven stages following `DEFAULT_STRIDES = [1, 2, 2, 2, 1, 2, 1]`
with the stride applied to each stage's first block only, giving a `/32` final
grid. The residual inside each block is added only where `stride == 1` and the
input width already equals the output width, so a stage's first block is
feed-forward and the rest are residual. The head is `GlobalAveragePooling ->
Dense(1280) -> ReLU -> dropout -> Dense(num_classes, softmax)`. `width_multiplier`
scales `dims` by plain truncation, with no round-to-multiple-of-8 rule.

The hybrid variants append one `MobileMQA` layer to the *end* of stages 5 and 6,
after that stage's convolutional blocks, with `use_downsampling=True`. Two
consequences are non-obvious. The attention layer carries no positional encoding
of any kind — RoPE is hard-disabled in `MobileMQA` — so all spatial ordering
information reaching it is whatever the convolutional stack has induced; this is
by design in the paper, but it means the block is not usable as a standalone
mixer. And its `num_heads` defaults to 8, so a stage width that is not divisible
by 8 raises; `width_multiplier` values that break that divisibility on
`dims[5]`/`dims[6]` will fail at construction rather than silently degrade.

`pretrained=True` on `create_mobilenetv4` raises `NotImplementedError`. No
checkpoints ship with this package; combined with the hand-written stage tables
above, this model should be treated as an architecture sketch to train from
scratch, not as a MobileNetV4 reimplementation. It used to log a warning and
return a randomly initialized model; the contract in `resnet/model.py` now holds
here too. Warm-start from a local file with `model.load_weights(path)`.
The mutable list defaults on
`__init__` (`depths`, `dims`, `block_types`, `strides`, `attention_stages`) are a
known defect inherited by copy-paste across this package: they are never mutated
in place, so they are currently harmless, but they should not be copied into new
code.

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
from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.models.mobilenet.common import (
    REFERENCE_BN_EPSILON,
    REFERENCE_BN_MOMENTUM,
    materialize_for_summary,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MobileNetV4(keras.Model):
    """MobileNetV4 model implementation with Universal Inverted Bottleneck blocks.

    A modern efficient architecture combining the best of MobileNets with new
    Universal Inverted Bottleneck (UIB) blocks that unify different architectural
    patterns. Supports both pure convolutional and hybrid variants with Mobile MQA.

    Args:
        num_classes: Integer, number of output classes for classification.
            Only used if include_top=True.
        depths: List of integers, number of UIB blocks in each stage.
            Default is [1, 2, 3, 4, 3, 3, 1] for MobileNetV4-ConvMedium.
        dims: List of integers, number of channels in each stage.
            Default is [16, 24, 40, 80, 112, 192, 320] for MobileNetV4-ConvMedium.
        block_types: List of strings, UIB block type for each stage.
            Options: "IB", "ConvNext", "ExtraDW", "FFN". Default optimized per stage.
        strides: List of integers, stride for the first block of each stage.
            Default is [1, 2, 2, 2, 1, 2, 1].
        width_multiplier: Float, multiplier for the number of filters.
            Values like 0.5, 0.75, 1.0, 1.25 control model capacity.
        use_attention: Boolean, whether to use Mobile MQA in later stages.
            Creates hybrid MobileNetV4-Hybrid variant when True.
        attention_stages: List of integers, which stages to add attention.
            Default is [5, 6] (last two stages) when use_attention=True.
        dropout_rate: Float, dropout rate for regularization in classifier head.
        weight_decay: Float, L2 regularization factor for all layers.
        kernel_initializer: String or initializer, weight initialization strategy.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape. If None, defaults to (224, 224, 3).
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths, dims, block_types, or strides have different lengths.
        ValueError: If invalid block type is specified.
        ValueError: If invalid attention stage indices are provided.

    Example:
        >>> # Create MobileNetV4-Medium for ImageNet. The variant keys are
        >>> # "small", "medium", "large", "hybrid_medium", "hybrid_large" --
        >>> # there is no "conv_medium"/"conv_small"; see the module docstring.
        >>> model = MobileNetV4.from_variant("medium", num_classes=1000)
        >>>
        >>> # Create MobileNetV4-Small for CIFAR-10
        >>> model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create MobileNetV4-Hybrid with custom configuration
        >>> model = MobileNetV4(
        ...     num_classes=100,
        ...     depths=[1, 2, 3, 4, 3, 3, 1],
        ...     dims=[16, 24, 40, 80, 112, 192, 320],
        ...     use_attention=True,
        ...     attention_stages=[5, 6],
        ...     input_shape=(128, 128, 3)
        ... )
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

        # Validate block types first to ensure tests get the expected error
        valid_block_types = {"IB", "ConvNext", "ExtraDW", "FFN"}
        for block_type in block_types:
            if block_type not in valid_block_types:
                raise ValueError(
                    f"Invalid block type '{block_type}'. "
                    f"Must be one of {valid_block_types}"
                )

        # Validate configuration lengths
        stage_configs = [depths, dims, block_types, strides]
        stage_lengths = [len(config) for config in stage_configs]
        if not all(length == stage_lengths[0] for length in stage_lengths):
            raise ValueError(
                f"All stage configurations must have same length. Got: "
                f"depths={len(depths)}, dims={len(dims)}, "
                f"block_types={len(block_types)}, strides={len(strides)}"
            )

        # Validate attention stages
        if use_attention:
            max_stage_idx = len(depths) - 1
            for stage_idx in attention_stages:
                if not (0 <= stage_idx <= max_stage_idx):
                    raise ValueError(
                        f"Attention stage index {stage_idx} out of range. "
                        f"Must be in [0, {max_stage_idx}]"
                    )

        # Validate input shape
        if input_shape and len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        if input_shape:
            height, width, channels = input_shape
            if channels not in [1, 3]:
                logger.warning(f"Unusual number of channels: {channels}")


        # Store configuration
        self.num_classes = num_classes
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: the DEFAULT is a
        # tuple (R-009 S1) and the STORED attribute is a list. Keeping the
        # store as `list(...)` is what makes the conversion invisible: it is
        # the type `get_config` has always emitted, so a saved config's JSON
        # shape and every `== [..]` assertion in the suites are unchanged.
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

        # Apply width multiplier to dimensions
        self.actual_dims = [int(dim * width_multiplier) for dim in dims]

        # Create regularizer
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None
        
        # Instantiate layers in __init__ for proper tracking and serialization
        self.stem_conv, self.stem_bn, self.stem_activation = self._build_stem()

        self.stages = []
        for stage_idx in range(len(self.depths)):
            stage_layers = self._build_stage(stage_idx)
            self.stages.append(stage_layers)

        if self.include_top:
            self.head_layers = self._build_head()

    def call(self, x, training=None):
        """Forward pass of the MobileNetV4 model."""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_activation(x)

        # Body (Stages)
        for stage_layers in self.stages:
            for layer in stage_layers:
                x = layer(x, training=training)

        # Head
        if self.include_top:
            for layer in self.head_layers:
                if isinstance(layer, layers.Dropout):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        return x

    def _build_stem(self):
        """Build and return the stem layers."""
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
        """Build and return layers for a single stage."""
        stage_layers = []
        depth = self.depths[stage_idx]
        dim = self.actual_dims[stage_idx]
        block_type = self.block_types[stage_idx]
        stage_stride = self.strides[stage_idx]

        logger.info(
            f"Building stage {stage_idx}: {depth} blocks, {dim} dims, "
            f"type={block_type}, stride={stage_stride}"
        )

        # DECISION plan-2026-08-14T183218-f4c612aa/D-011
        # `block_type` used to be passed to the block as a LABEL only, so every
        # stage — whatever its table said — got the layer's defaults and built a
        # plain IB. What selects a structure is which of the two optional
        # depthwise POSITIONS is occupied, so that is what this maps to. Do NOT
        # "simplify" this to `use_dw1`/`use_dw2`, as this module's own docstring
        # used to advise: both of those are POST-expansion, so that mapping
        # varies the number of middle depthwise convs and cannot express a
        # pre-expansion start DW at all. `use_dw2` is deliberately absent here —
        # a second middle DW is the layer's own extra axis, not one of the four
        # named structures. See decisions.md D-011.
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
        """Build and return the head layers."""
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

        Args:
            variant: String, one of "small", "medium", "large",
                "hybrid_medium", "hybrid_large"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (224, 224, 3)
            width_multiplier: Float, multiplier for filter dimensions
            **kwargs: Additional arguments passed to the constructor

        Returns:
            MobileNetV4 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
            >>> model = MobileNetV4.from_variant("hybrid_medium", num_classes=1000)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        # Set default input shape if not provided
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
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
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
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            MobileNetV4 model instance
        """
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: a real forward pass. The
        # old `self.build(self._input_shape)` passed the UNBATCHED 3-tuple where
        # Model.build() expects the batch shape, and even the batch-shaped form
        # materializes no sub-layer weights on a subclassed model. See
        # decisions.md D-065.
        materialize_for_summary(self, self._input_shape)

        super().summary(**kwargs)

        # Print additional model information
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
    """Convenience function to create MobileNetV4 models.

    Args:
        variant: String, model variant ("small", "medium", "large",
            "hybrid_medium", "hybrid_large")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        width_multiplier: Float, multiplier for filter dimensions
        pretrained: Boolean, must be False. `True` raises `NotImplementedError` —
            no MobileNetV4 checkpoints ship with this package.
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV4 model instance

    Example:
        >>> model = create_mobilenetv4("small", num_classes=10, input_shape=(32, 32, 3))
        >>> model = create_mobilenetv4("hybrid_medium", num_classes=1000)
        >>> model = create_mobilenetv4("medium", width_multiplier=0.75)
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
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
