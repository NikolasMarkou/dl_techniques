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
`dims` by plain truncation, with no round-to-multiple-of-8 rule. Every
convolution kernel draws from `kernel_initializer` through its own copy of
the stored spec, so a name, a dict or a seedless instance gives every kernel
its own sample; a seeded instance (or a dict with an explicit `seed`) reuses
its seed at every site by contract (`HeNormal(seed=7)` ties 351 of 861 kernel
pairs at `small`). No pretrained
checkpoints ship with this package; `pretrained=True` raises
`NotImplementedError`, and a local `.keras` path loads by layer.

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

import copy
import keras
from typing import Tuple, Optional, Dict, Any, Sequence, Union

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
from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.layers.attention.mobile_mqa import MobileMQA
from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.mobilenet.common import (
    materialize_for_summary,
    reference_bn_kwargs,
)
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head
from dl_techniques.utils.keras_registration import register_dl_technique

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (the initializer, the
# dropout rate, the weight decay, the input size). `load_pretrained_checkpoint` compares
# every other key the checkpoint and the model both hold, so a changed stage depth,
# width, block type or stride is refused even when every weight shape still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "input_shape",
    "kernel_initializer",
    "dropout_rate",
    "weight_decay",
)

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
        GlobalAvgPool -> Dense(1280) -> BatchNorm -> ReLU -> Dropout -> Dense (linear)
           |
           v
        logits  [num_classes]

    :param num_classes: Number of output classes; used only if `include_top=True`,
        where it must be positive.
    :type num_classes: int
    :param depths: UIB block count per stage.
    :type depths: Sequence[int]
    :param dims: Channel count per stage.
    :type dims: Sequence[int]
    :param block_types: UIB structure per stage: `"IB"`, `"ConvNext"`, `"ExtraDW"`, `"FFN"`.
    :type block_types: Sequence[str]
    :param strides: Stride for the first block of each stage.
    :type strides: Sequence[int]
    :param width_multiplier: Channel-count multiplier, applied by truncation. Must be positive.
    :type width_multiplier: float
    :param use_attention: Add Mobile MQA to `attention_stages`.
    :type use_attention: bool
    :param attention_stages: Stage indices to add attention to, when `use_attention=True`.
    :type attention_stages: Sequence[int]
    :param dropout_rate: Dropout rate in the classifier head, in `[0, 1)`.
    :type dropout_rate: float
    :param weight_decay: L2 regularization factor for all layers, the head's
        included.
    :type weight_decay: float
    :param kernel_initializer: Weight initialization strategy, the head's
        included: a name, a serialized config dict, or an `Initializer`
        instance. It is stored as its serialized spec and every layer draws
        from its own copy of it, so each kernel has its own draw unless the spec
        carries an explicit `seed`, which every site then reuses by contract.
    :type kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param input_shape: Input shape `(height, width, channels)`; `None` means
        `(224, 224, 3)`.
    :type input_shape: Optional[Tuple[int, ...]]
    :param head_config_overrides: Optional dict merged into the head
        configuration. `name`, `num_classes` and `task_type` raise
        `ValueError`.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Passthrough to `keras.Model`.
    :raises ValueError: If `depths`, `dims`, `block_types`, or `strides` have
        different lengths, an invalid block type is given, an attention
        stage index is out of range, `num_classes` is not positive with a
        head, `width_multiplier` is not positive or truncates a stage width to
        0, `dropout_rate` is outside `[0, 1)`, or `input_shape` is not 3-D.
    :raises TypeError: If `kernel_initializer` is not a string, a dict or an
        `Initializer` instance.

    Input shape:
        4-D tensor `(batch_size, height, width, channels)`.

    Output shape:
        - `include_top=True`: logits `(batch_size, num_classes)`.
        - `include_top=False`: the last stage's feature map
          `(batch_size, H', W', actual_dims[-1])`.

    Example::

        model = MobileNetV4.from_variant("medium", num_classes=1000)
        small = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        shallow = MobileNetV4.from_variant("small", depths=[1, 1, 1, 1, 1, 1, 1])
    """

    MODEL_VARIANTS = {
        "small": {
            "depths": [1, 1, 2, 3, 2, 2, 1],
            "dims": [16, 24, 32, 64, 96, 128, 160],
            "block_types": ["IB", "IB", "IB", "ExtraDW", "IB", "IB", "IB"],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "use_attention": False,
        },
        "medium": {
            "depths": [1, 2, 3, 4, 3, 3, 1],
            "dims": [16, 24, 40, 80, 112, 192, 320],
            "block_types": ["IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "use_attention": False,
        },
        "large": {
            "depths": [1, 2, 4, 5, 4, 4, 1],
            "dims": [24, 32, 48, 96, 136, 224, 384],
            "block_types": ["IB", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "IB"],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "use_attention": False,
        },
        "hybrid_medium": {
            "depths": [1, 2, 3, 4, 3, 3, 1],
            "dims": [16, 24, 40, 80, 112, 192, 320],
            "block_types": ["IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "use_attention": True,
            "attention_stages": [5, 6],
        },
        "hybrid_large": {
            "depths": [1, 2, 4, 5, 4, 4, 1],
            "dims": [24, 32, 48, 96, 136, 224, 384],
            "block_types": ["IB", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "IB"],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "use_attention": True,
            "attention_stages": [5, 6],
        },
    }

    STEM_KERNEL_SIZE = 3
    STEM_STRIDE = 2
    HEAD_HIDDEN_DIM = 1280

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
        kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = "he_normal",
        include_top: bool = True,
        input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )

        valid_block_types = ("ConvNext", "ExtraDW", "FFN", "IB")
        for stage_idx, block_type in enumerate(block_types):
            if block_type not in valid_block_types:
                raise ValueError(
                    f"block_types[{stage_idx}] must be one of "
                    f"{list(valid_block_types)}, got '{block_type}'"
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
                        f"attention_stages entries must be in [0, {max_stage_idx}], "
                        f"got {stage_idx}"
                    )

        if width_multiplier <= 0:
            raise ValueError(
                f"width_multiplier must be positive, got {width_multiplier}"
            )

        if any(int(dim * width_multiplier) <= 0 for dim in dims):
            raise ValueError(
                f"width_multiplier={width_multiplier} truncates a width of "
                f"dims={list(dims)} to 0; every stage width must stay positive"
            )

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )

        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")
        if input_shape[2] not in (1, 3):
            logger.warning(f"Unusual number of channels: {input_shape[2]}")

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
        # DECISION plan-2026-10-26T125959-7af851e5/D-033: store the initializer as a spec and hand that spec to the stem, every block, every Mobile MQA and the head; never keep the caller's Initializer instance.
        # A seedless instance given to two layers replays one draw at both (the stem and the Mobile MQA downsample did, 3 of 2701 kernel pairs at hybrid_medium), and an instance in get_config is not JSON-safe. See decisions.md.
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.include_top = include_top
        self.input_shape_config = input_shape

        self._head_config_overrides = plain_head_config_overrides(
            head_config_overrides, "MobileNetV4"
        )

        self.actual_dims = [int(dim * width_multiplier) for dim in dims]

        self.kernel_regularizer = keras.regularizers.L2(weight_decay) if weight_decay > 0 else None

        self._build_stem()

        self.stages = []
        for stage_idx in range(len(self.depths)):
            self._build_stage(stage_idx)

        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created MobileNetV4 with {sum(self.depths)} blocks in "
            f"{len(self.depths)} stages for input {input_shape}"
        )

    def _build_stem(self) -> None:
        """Build the stem: conv, batch norm and ReLU in one :class:`ConvStem`."""
        self.stem = ConvStem(
            filters=self.actual_dims[0],
            kernel_size=self.STEM_KERNEL_SIZE,
            strides=self.STEM_STRIDE,
            padding='same',
            use_bias=False,
            num_convs=1,
            normalization_type='batch_norm',
            normalization_kwargs=reference_bn_kwargs(),
            activation_type='relu',
            use_pooling=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='stem',
        )

    def _build_stage(self, stage_idx: int) -> None:
        """Build the UIB blocks, and optional Mobile MQA, for one stage.

        The blocks are named ``stage{S}_block{B}`` and the attention layer
        ``stage{S}_mqa``, with S and B counted from 1.

        :param stage_idx: Index into `depths`/`dims`/`block_types`/`strides`.
        """
        stage_layers = []
        depth = self.depths[stage_idx]
        dim = self.actual_dims[stage_idx]
        block_type = self.block_types[stage_idx]
        stage_stride = self.strides[stage_idx]

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
                normalization_args=reference_bn_kwargs(),
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"stage{stage_idx + 1}_block{block_idx + 1}",
                **block_structure
            )
            stage_layers.append(block)

        if self.use_attention and stage_idx in self.attention_stages:
            mqa_layer = MobileMQA(
                dim=dim,
                use_downsampling=True,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"stage{stage_idx + 1}_mqa"
            )
            stage_layers.append(mqa_layer)

        self.stages.append(stage_layers)

    def _build_head(self) -> None:
        """Build the reference classifier on the shared ``ClassificationHead``.

        The paper's classifier (Qin et al. 2024: global pool, then a ``1x1``
        conv to 1280 with BatchNorm and ReLU, then the classifier), as timm's
        ``mobilenetv4`` builds it (``head_norm=True``: ``conv_head`` 1280,
        ``norm_head`` BatchNorm + ReLU, dropout before ``classifier``): global
        average pool, a bias-free ``Dense(HEAD_HIDDEN_DIM)``, BatchNorm at the family's
        reference epsilon and momentum, ReLU, ``Dropout(dropout_rate)``, one
        linear ``Dense``. The model returns logits.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                'num_classes': self.num_classes,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
                # One hidden block: Dense(1280) -> BatchNorm -> ReLU -> dropout.
                'block_dims': (self.HEAD_HIDDEN_DIM,),
                'normalization_type': 'batch_norm',
                # The family rule (common.py, D-111 of plan-2026-08-22): every
                # BatchNorm runs at the reference epsilon, the head's too.
                'normalization_kwargs': reference_bn_kwargs(),
                # timm's conv_head is "never a bias": the BatchNorm after it would
                # leave a bias with an identically-zero gradient (measured).
                'block_use_bias': False,
                'activation_type': 'relu',
                'dropout_rate': self.dropout_rate,
                'kernel_initializer': self.kernel_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                # Deterministic layer name: an unnamed head takes Keras'
                # auto suffix in process-creation order, which breaks the
                # by-name matching that weight transfer relies on.
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

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the stem, stages, and optional head.

        :param inputs: Input images, shape `(B, H, W, C)`.
        :param training: Passed to batch norm and dropout.
        :return: Logits, or the 4-D feature map if `include_top=False`.
        """
        x = self.stem(inputs, training=training)

        for stage_layers in self.stages:
            for layer in stage_layers:
                x = layer(x, training=training)

        if self.include_top:
            return self.classification_head(x, training=training)['logits']

        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Report the shape `call` returns, without building the model.

        :param input_shape: `(batch_size, height, width, channels)`; the
            spatial axes may be `None`.
        :return: The logits shape `(batch_size, num_classes)`
            (`include_top=True`) or the last stage's feature map shape.
        """
        shape = self.stem.compute_output_shape(input_shape)

        for stage_layers in self.stages:
            for layer in stage_layers:
                shape = layer.compute_output_shape(shape)

        if not self.include_top:
            return shape

        # The head reports both keys; the model returns the logits.
        return self.classification_head.compute_output_shape(shape)['logits']

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True
    ) -> None:
        """Load a local `.keras` checkpoint into the model, layer by layer.

        The transfer is `load_pretrained_checkpoint`: it builds the model at
        `input_shape` if it is unbuilt and takes a headed checkpoint into a
        headless model. Every layer outside the classification head must be
        restored, so a checkpoint of another variant, block type, width or stem
        raises and leaves the model untouched. Only a head of another
        `num_classes` is skipped: it keeps its initialization and a warning
        names it. The skipped head is the whole `classification_head`, so its
        pretrained hidden layer (V4 `small`: 209,920 parameters, `medium`: 414,720) is discarded with the
        classifier. The archive's own config is compared too: a constructor
        argument that both hold and that is not in `_NON_STRUCTURAL_CONFIG_KEYS`
        must be equal, so a checkpoint of another `width_multiplier` or stride
        pattern is refused even when every weight shape fits. An argument only
        one side holds is not compared, and an archive without a config of this
        class is checked by layers only.

        :param weights_path: Path to the weights file (`.keras` format).
        :param skip_mismatch: `True` (default) lets the head layers differ;
            `False` allows no difference anywhere. The inverse of the transfer
            helper's `strict`.
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If a layer outside the head is not restored, or
            the weights cannot be loaded.
        """
        load_pretrained_checkpoint(
            self, weights_path, self.input_shape_config, skip_mismatch,
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
    ) -> "MobileNetV4":
        """Create a MobileNetV4 model from a predefined variant.

        :param variant: One of `"small"`, `"medium"`, `"large"`, `"hybrid_medium"`, `"hybrid_large"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no MobileNetV4 weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; a field of the variant
            (e.g. `depths`, `use_attention`, `width_multiplier`) is replaced
            by the value given here.
        :return: A configured `MobileNetV4` instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

        Example::

            model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
            hybrid = MobileNetV4.from_variant("hybrid_medium", num_classes=1000)
            wide = MobileNetV4.from_variant("small", width_multiplier=0.5, use_attention=True)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise on pretrained=True, do not warn and return random weights.
        # See decisions.md.
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
            "input_shape": self.input_shape_config,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self, **kwargs):
        """Print the model summary; an unbuilt model is materialized first."""
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-065: materialize with a real forward pass.
        # self.build(self._input_shape) passed an unbatched shape where Model.build() expects a batch shape, and even fixed would materialize no sub-layer weights. See decisions.md.
        materialize_for_summary(self, self.input_shape_config)

        super().summary(**kwargs)


# ---------------------------------------------------------------------

def create_mobilenetv4(
    variant: str = "medium",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = None,
    pretrained: Union[bool, str] = False,
    **kwargs
) -> MobileNetV4:
    """Create a MobileNetV4 model.

    :param variant: Model variant: `"small"`, `"medium"`, `"large"`, `"hybrid_medium"`, `"hybrid_large"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param kwargs: Passthrough to `MobileNetV4.from_variant`, e.g.
        `width_multiplier`, `head_config_overrides`, `include_top`.
    :return: A configured `MobileNetV4` instance.
    :raises NotImplementedError: If `pretrained` is `True`.

    Example::

        model = create_mobilenetv4("small", num_classes=10, input_shape=(32, 32, 3))
        hybrid = create_mobilenetv4("hybrid_medium", num_classes=1000)
    """
    return MobileNetV4.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )

# ------------------------------------------------------------------------
