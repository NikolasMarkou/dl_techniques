"""``ResNet``, a configurable residual network, plus the ``create_resnet`` factory.

Each block learns a residual `F(x)` added to an identity shortcut,
`y = F(x) + x`, instead of learning the full mapping directly, which makes
the identity easy for gradient descent to fall back on at any depth. A
basic block (two 3x3 convolutions) or a bottleneck block (1x1 reduce, 3x3,
1x1 restore) trades depth against width; five presets span ResNet-18
through ResNet-152. The stem is selectable: `stem_type='imagenet'` (7x7
stride-2 + max pool) downsamples 4x before stage 1, while
`stem_type='cifar'` (single 3x3 stride-1, no pooling) does not, so the
choice matters on small inputs. A shortcut carries a projection only where
identity cannot apply directly (a stride-2 stage boundary, or stage 0
under the bottleneck design); everywhere else it is parameter-free.

With deep supervision on, every stage but the last gets its own pooling
and classification head, and the model returns
`[final_output, stage3, stage2, stage1]` (deepest head first). BatchNorm
momentum defaults to `0.9` (Keras convention) to match torchvision's
`momentum=0.1` (`keras_momentum = 1 - torch_momentum`), a training-only
constant that does not affect the `training=False` forward pass. No
pretrained weights ship with this package; `pretrained=True` raises
`NotImplementedError`. Local checkpoints load by path, skipping
shape-mismatched layers by name.

References:
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
    - He et al., 2016. Identity Mappings in Deep Residual Networks.
      (https://arxiv.org/abs/1603.05027)
    - Lee et al., 2015. Deeply-Supervised Nets. AISTATS 2015.
      (https://arxiv.org/abs/1409.5185)
    - Veit et al., 2016. Residual Networks Behave Like Ensembles of Relatively
      Shallow Networks. (https://arxiv.org/abs/1605.06431)
    - torchvision `nn.BatchNorm2d` (`momentum=0.1`), the de facto reference
      reimplementation this port's BatchNorm momentum is matched against.
      (https://docs.pytorch.org/docs/2.13/generated/torch.nn.BatchNorm2d.html)
"""

import os
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.activations import create_activation_layer
from dl_techniques.layers.standard_blocks import (
    BasicBlock,
    BottleneckBlock,
)
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.resnet.model")
class ResNet(keras.Model):
    """
    Deep residual network with configurable blocks and optional deep supervision.

    Implements the ResNet family, in which each block learns a residual
    ``F(x)`` added to an identity shortcut (``y = F(x) + x``) so that depth does
    not obstruct optimization. A stem -- 7x7 stride-2 plus max pool by default,
    or a 3x3 stride-1 CIFAR stem via ``stem_type='cifar'`` -- feeds four stages
    that each halve spatial resolution and double channel width, built from
    either
    ``BasicBlock`` (two 3x3 convolutions) or ``BottleneckBlock`` (1x1 reduce,
    3x3, 1x1 expand). Shortcut projections are inserted only where the identity
    cannot be taken verbatim: at the first block of every stage after the first
    (stride-2 shape change), plus stage 0's first block under the bottleneck
    design (channel widening at stride 1). With ``enable_deep_supervision=True``
    every stage but the last receives its own GAP + Dense head (the last stage is
    the main head's own), and the model returns
    ``[final_output, stage3, stage2, stage1]``.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │       Input [B, H, W, C_in]          │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Stem: Conv 7×7 /2 → Norm → Act      │
        │        → MaxPool 3×3 /2              │
        │  (stem_type='cifar': Conv 3×3 /1,    │
        │   no MaxPool)                        │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 1: N₁ × Block(f₁, stride 1)   │──────┐
        └───────────────┬──────────────────────┘      │
                        ▼                             │
        ┌──────────────────────────────────────┐      │
        │  Stage 2: N₂ × Block(f₂, stride 2)   │───┐  │
        └───────────────┬──────────────────────┘   │  │
                        ▼                          │  │
        ┌──────────────────────────────────────┐   │  │
        │  Stage 3: N₃ × Block(f₃, stride 2)   │┐  │  │
        └───────────────┬──────────────────────┘│  │  │
                        ▼                       │  │  │
        ┌──────────────────────────────────────┐│  │  │
        │  Stage 4: N₄ × Block(f₄, stride 2)   ││  │  │
        └───────────────┬──────────────────────┘│  │  │
                        │                       │  │  │
            ┌───────────┴──────────┐            │  │  │
            │  Block (residual)    │            │  │  │
            │  x ──► F(x) ──► (+)  │            │  │  │
            │  └──── shortcut ──┘  │            │  │  │
            └───────────┬──────────┘            │  │  │
                        ▼                       ▼  ▼  ▼ (deep supervision)
        ┌──────────────────────────────────────┐ ┌──────────────┐
        │  GAP → Dense(num_classes)            │ │ GAP → Dense  │
        │  (if include_top)                    │ │ per stage    │
        └───────────────┬──────────────────────┘ └──────┬───────┘
                        │                               │
                        ▼                               ▼
        ┌───────────────────────────────────────────────────────┐
        │  Output: [B, num_classes]                             │
        │   include_top=False        → [B, H', W', f₄·expansion]│
        │   deep_supervision=True    → [final, s3, s2, s1]      │
        └───────────────────────────────────────────────────────┘

    Variants:

    .. code-block:: text

        resnet18   [2, 2, 2, 2]   [64, 128, 256, 512]   basic
        resnet34   [3, 4, 6, 3]   [64, 128, 256, 512]   basic
        resnet50   [3, 4, 6, 3]   [64, 128, 256, 512]   bottleneck
        resnet101  [3, 4, 23, 3]  [64, 128, 256, 512]   bottleneck
        resnet152  [3, 8, 36, 3]  [64, 128, 256, 512]   bottleneck

    :param num_classes: Number of output classes. Only used when
        ``include_top=True``. A value of 0 returns pooled features from the head.
    :type num_classes: int
    :param blocks_per_stage: Number of residual blocks in each stage. Must have
        the same length as ``filters_per_stage``. Defaults to ``[3, 4, 6, 3]``.
    :type blocks_per_stage: Optional[List[int]]
    :param filters_per_stage: Base filter count per stage. Defaults to
        ``[64, 128, 256, 512]``.
    :type filters_per_stage: Optional[List[int]]
    :param block_type: Residual block design, ``'basic'`` or ``'bottleneck'``.
        Defaults to ``'bottleneck'``.
    :type block_type: Literal['basic', 'bottleneck']
    :param kernel_regularizer: Optional regularizer applied to all convolution
        and dense kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param normalization_type: Normalization layer identifier passed to
        ``create_normalization_layer``. Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param normalization_kwargs: Optional kwargs forwarded to every
        normalization factory call in the stem and in every block. ``None``
        resolves to ``{}``. When ``normalization_type == 'batch_norm'`` and no
        ``momentum`` is supplied, ``momentum=0.9`` is injected to match
        torchvision's ``BatchNorm2d`` (see the class body); pass an explicit
        ``momentum`` to override. Weight shapes and the inference forward pass
        are unaffected, so existing checkpoints stay bit-exact.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param activation_type: Activation identifier passed to
        ``create_activation_layer``. Defaults to ``'relu'``.
    :type activation_type: str
    :param include_top: Whether to include the classification head. When False
        the final stage's feature maps are returned. Defaults to True.
    :type include_top: bool
    :param enable_deep_supervision: Whether to attach auxiliary classification
        heads. Requires ``include_top=True``. One head per stage except the last,
        whose features the main head already consumes; stage 0 is included.
        Defaults to False.
    :type enable_deep_supervision: bool
    :param input_shape: Input shape ``(height, width, channels)`` excluding the
        batch dimension. Defaults to ``(224, 224, 3)``.
    :type input_shape: Tuple[int, ...]
    :param stem_type: Which input stem to build. ``'imagenet'`` (default) is the
        published 7x7 stride-2 convolution followed by a 3x3 stride-2 max pool,
        downsampling by 4x before stage 1; it is bit-identical to the model
        before this argument existed. ``'cifar'`` is He et al.'s own CIFAR
        configuration -- a single 3x3 stride-1 convolution with no pooling --
        which preserves the input resolution into stage 1. Use it for small
        inputs: on ``(32, 32, 3)`` the ImageNet stem leaves ``resnet18`` with a
        ``(1, 1, 1, 512)`` feature map before the global average pool, so the
        last two stages stride an already-collapsed map. Both stems take their
        width from ``filters_per_stage[0]``.
    :type stem_type: Literal['imagenet', 'cifar']
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.

    :raises ValueError: If ``blocks_per_stage`` and ``filters_per_stage`` differ
        in length, if ``block_type`` is not ``'basic'`` or ``'bottleneck'``, if
        ``stem_type`` is not ``'imagenet'`` or ``'cifar'``, or if
        ``input_shape`` is not 3D.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_top=True``: 2D tensor ``(batch_size, num_classes)``.
        - ``include_top=False``: 4D tensor ``(batch_size, H', W', channels)``.
        - ``enable_deep_supervision=True``: list of 2D tensors, ordered
          ``[final_output, stage3, stage2, stage1]`` (reversed, matching the
          BFUNet output convention). Inference typically uses index 0 alone.

    Example:
        >>> # ResNet-50 for ImageNet
        >>> model = ResNet.from_variant("resnet50", num_classes=1000)
        >>>
        >>> # Deep supervision for training
        >>> model = ResNet.from_variant("resnet50", enable_deep_supervision=True)
        >>>
        >>> # Feature extractor from a local checkpoint
        >>> model = ResNet.from_variant("resnet34", pretrained="/path/to.keras",
        ...                             include_top=False)

    Note:
        No pretrained ResNet weights are distributed with ``dl_techniques``.
        ``pretrained=True`` raises ``NotImplementedError`` rather than warning
        and returning a randomly-initialized model; pass a local checkpoint via
        ``pretrained='/path/to/weights.keras'`` instead.
    """

    MODEL_VARIANTS = {
        "resnet18": {
            "blocks_per_stage": [2, 2, 2, 2],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "basic"
        },
        "resnet34": {
            "blocks_per_stage": [3, 4, 6, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "basic"
        },
        "resnet50": {
            "blocks_per_stage": [3, 4, 6, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "bottleneck"
        },
        "resnet101": {
            "blocks_per_stage": [3, 4, 23, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "bottleneck"
        },
        "resnet152": {
            "blocks_per_stage": [3, 8, 36, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "bottleneck"
        },
    }

    def __init__(
            self,
            num_classes: int = 1000,
            blocks_per_stage: Optional[List[int]] = None,
            filters_per_stage: Optional[List[int]] = None,
            block_type: Literal["basic", "bottleneck"] = "bottleneck",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: str = "batch_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_type: str = "relu",
            include_top: bool = True,
            enable_deep_supervision: bool = False,
            input_shape: Tuple[int, ...] = (224, 224, 3),
            stem_type: Literal["imagenet", "cifar"] = "imagenet",
            **kwargs
    ):
        super().__init__(**kwargs)

        blocks_per_stage = list(blocks_per_stage) if blocks_per_stage is not None else [3, 4, 6, 3]
        filters_per_stage = list(filters_per_stage) if filters_per_stage is not None else [64, 128, 256, 512]

        if len(blocks_per_stage) != len(filters_per_stage):
            raise ValueError(
                f"Length of blocks_per_stage ({len(blocks_per_stage)}) must equal "
                f"length of filters_per_stage ({len(filters_per_stage)})"
            )

        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(
                f"block_type must be 'basic' or 'bottleneck', got '{block_type}'"
            )

        if stem_type not in ["imagenet", "cifar"]:
            raise ValueError(
                f"stem_type must be 'imagenet' or 'cifar', got '{stem_type}'"
            )

        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.blocks_per_stage = blocks_per_stage
        self.filters_per_stage = filters_per_stage
        self.block_type = block_type
        self.stem_type = stem_type
        self.kernel_regularizer = kernel_regularizer
        self.normalization_type = normalization_type
        # DECISION plan_2026-05-18_6776f8ba/D-003: forward normalization_kwargs to every factory call in the stem and every block.
        # A None default resolves to {}, so every call stays byte-identical to the pre-plumbing version and existing checkpoints stay bit-exact. See decisions.md.
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}

        # DECISION plan-2026-08-23T091307-9a110062/D-480: inject momentum=0.9 for batch_norm; do not fall back to Keras' bare default or "correct" it to 0.1.
        # Keras and torchvision define momentum oppositely (keras_momentum = 1 - torch_momentum), so torch's canonical 0.1 is Keras' 0.9, not 0.99 or 0.1. See decisions.md.
        if self.normalization_type == "batch_norm":
            self.normalization_kwargs.setdefault("momentum", 0.9)

        self.activation_type = activation_type
        self.include_top = include_top
        self.enable_deep_supervision = enable_deep_supervision
        self.input_shape_config = input_shape
        self.input_height, self.input_width, self.input_channels = input_shape

        self._build_stem()

        self.stages = []
        for stage_idx in range(len(blocks_per_stage)):
            self._build_stage(stage_idx)

        if self.include_top:
            self._build_head()

        self.supervision_heads = []
        if self.enable_deep_supervision and self.include_top:
            self._build_supervision_heads()

        logger.info(
            f"Created ResNet with {sum(blocks_per_stage)} blocks for input "
            f"{input_shape} (deep supervision: {enable_deep_supervision})"
        )

    def _build_stem(self) -> None:
        """Build the initial convolution stem.

        Both stem variants take their width from `filters_per_stage[0]`,
        not a literal 64, so a `basic`-block model can run at any stage-0
        width.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-041: stem width must follow filters_per_stage[0], never a literal 64.
        # A basic block's stage-0 shortcut is unconditionally identity, so a mismatched stem width raised a forward-pass error 2 frames deep. See decisions.md.
        # DECISION plan-2026-08-23T203721-009b7ccf/D-019: stem_type='cifar' must stay a literal single-conv/no-pool alternative, never derived from input_shape.
        # An input-shape-derived stride would silently change the weight tree of every small-input checkpoint; 'imagenet' stays the bit-identical default. See decisions.md (this is a different plan's D-019 than _build_supervision_heads').
        self.stem_conv = keras.layers.Conv2D(
            filters=self.filters_per_stage[0],
            kernel_size=7 if self.stem_type == "imagenet" else 3,
            strides=2 if self.stem_type == "imagenet" else 1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        self.stem_bn = create_normalization_layer(
            self.normalization_type,
            name="stem_bn",
            **self.normalization_kwargs,
        )
        self.stem_act = create_activation_layer(
            self.activation_type,
            name="stem_act"
        )
        self.stem_pool = keras.layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            padding="same",
            name="stem_pool"
        ) if self.stem_type == "imagenet" else None

    def _build_stage(self, stage_idx: int) -> None:
        """Build one residual stage's blocks.

        :param stage_idx: Index of the stage to build.
        """
        num_blocks = self.blocks_per_stage[stage_idx]
        base_filters = self.filters_per_stage[stage_idx]

        BlockClass = BasicBlock if self.block_type == "basic" else BottleneckBlock

        stage_blocks = []

        for block_idx in range(num_blocks):
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1

            # Stage 0's first block projects only to widen channels (bottleneck);
            # every later stage's first block projects to match the stride-2 shortcut.
            use_projection = False
            if block_idx == 0:
                use_projection = (
                    self.block_type == "bottleneck" if stage_idx == 0 else True
                )

            block = BlockClass(
                filters=base_filters,
                stride=stride,
                use_projection=use_projection,
                kernel_regularizer=self.kernel_regularizer,
                normalization_type=self.normalization_type,
                normalization_kwargs=dict(self.normalization_kwargs),
                activation_type=self.activation_type,
                name=f"stage{stage_idx+1}_block{block_idx+1}"
            )
            stage_blocks.append(block)

        self.stages.append(stage_blocks)

    def _build_head(self) -> None:
        """Build classification head."""
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")

        if self.num_classes > 0:
            self.classifier = keras.layers.Dense(
                units=self.num_classes,
                kernel_initializer="he_normal",
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )
        else:
            self.classifier = None

    def _build_supervision_heads(self) -> None:
        """Build one GAP + Dense head per stage except the last.

        The final stage is already served by the main head; stage 0 is
        supervised since it is the one stage where deep supervision
        actually shortens the backpropagation path.
        """
        # DECISION plan-2026-08-17T183311-79c63e38/D-019: range(0, N-1), not range(1, N) or range(1, N-1).
        # Supervising the final stage duplicates the main head; skipping stage 0 loses the one stage the technique helps most. See decisions.md.
        for stage_idx in range(0, len(self.blocks_per_stage) - 1):
            gap_layer = keras.layers.GlobalAveragePooling2D(
                name=f"supervision_gap_stage{stage_idx+1}"
            )

            if self.num_classes > 0:
                classifier_layer = keras.layers.Dense(
                    units=self.num_classes,
                    kernel_initializer="he_normal",
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"supervision_classifier_stage{stage_idx+1}"
                )
            else:
                classifier_layer = None

            self.supervision_heads.append({
                "gap": gap_layer,
                "classifier": classifier_layer,
                "stage_idx": stage_idx
            })


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
    ) -> Union[keras.KerasTensor, List[keras.KerasTensor]]:
        """Run the stem, stages, and optional main and supervision heads.

        :param inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        :param training: Whether batch norm and dropout run in training mode.
        :return:
            A single tensor `(batch_size, num_classes)` (`include_top=True`)
            or `(batch_size, H', W', channels)` (`include_top=False`), or,
            with deep supervision, the list
            `[final_output, stage3, stage2, stage1]`.
        """
        x = self.stem_conv(inputs)
        x = self.stem_bn(x, training=training)
        x = self.stem_act(x)
        if self.stem_pool is not None:
            x = self.stem_pool(x)

        stage_features = []
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x, training=training)
            if self.enable_deep_supervision and self.include_top:
                stage_features.append(x)

        if self.include_top:
            final_features = self.gap(x)
            final_output = (
                self.classifier(final_features) if self.classifier else final_features
            )
        else:
            final_output = x

        if self.enable_deep_supervision and self.include_top and self.supervision_heads:
            # Reversed (stage 3, 2, 1) to match the BFUNet output convention.
            supervision_outputs = []
            for sup_head in reversed(self.supervision_heads):
                feat = sup_head["gap"](stage_features[sup_head["stage_idx"]])
                supervision_outputs.append(
                    sup_head["classifier"](feat) if sup_head["classifier"] else feat
                )
            return [final_output] + supervision_outputs

        return final_output

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load pretrained weights into the model from a local checkpoint.

        Transfer is layer-by-layer via
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
        not `self.load_weights(..., by_name=True)`: Keras 3 removed `by_name`
        from `Model.load_weights`, so that call raises `ValueError` for
        every caller.

        :param weights_path: Path to the weights file (`.keras` format).
        :param skip_mismatch: Whether to skip layers with mismatched shapes;
            the inverse of the transfer helper's `strict`.
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            if not self.built:
                dummy_input = keras.random.normal((1,) + tuple(self.input_shape_config))
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")
            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )
            logger.info(f"Weight transfer complete: {report}")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    # Raises instead of falling back to random init; do not reinstate a warn-and-return branch here or in from_variant.
    # No public ResNet weights are distributed with dl_techniques; pass a local path via pretrained="/path/to/file.keras".
    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None
    ) -> str:
        """Always raise; no public ResNet weights ship with `dl_techniques`.

        Kept to mirror the BERT / GPT-2 / WaveFieldLLM factory recipe and
        give an explicit failure mode instead of a silent random-init
        fallback.

        :param variant: Variant name (unused).
        :param dataset: Dataset name (unused).
        :param cache_dir: Cache directory (unused).
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained ResNet weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: ResNet.from_variant('{variant}', "
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
    ) -> "ResNet":
        """Create a ResNet model from a predefined variant.

        :param variant: One of `"resnet18"`, `"resnet34"`, `"resnet50"`,
            `"resnet101"`, `"resnet152"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no public ResNet weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param weights_dataset: Dataset the pretrained weights were trained on.
        :param weights_input_shape: Input shape used during pretraining.
        :param cache_dir: Directory to cache downloaded weights.
        :param kwargs: Passthrough to the constructor.
        :return: A configured `ResNet` instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.

        Example::

            model = ResNet.from_variant("resnet50", enable_deep_supervision=True)
            small = ResNet.from_variant("resnet34", num_classes=10, input_shape=(32, 32, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127: copy the preset, drop metadata, then config.update(kwargs) -- do not splat preset fields alongside **kwargs.
        # Splatting raised TypeError on every overridden field; skipping .copy() would permanently poison the shared MODEL_VARIANTS[variant] dict. See decisions.md.
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        config.update(kwargs)

        if input_shape is None:
            input_shape = (224, 224, 3)

        logger.info(f"Creating {variant.upper()} model")

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

            # The ImageNet head is 1000-wide; a different num_classes or a
            # different input shape than the checkpoint was trained at means the
            # affected layers must be skipped rather than refused.
            if kwargs.get("include_top", True) and num_classes != 1000:
                skip_mismatch = True
                logger.info(
                    f"num_classes ({num_classes}) differs from the pretrained 1000; "
                    f"classifier weights will be skipped."
                )

            if weights_input_shape and input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Loading weights pretrained on {weights_input_shape} "
                    f"for model with input shape {input_shape}."
                )
                skip_mismatch = True

        model = cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **config
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

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "blocks_per_stage": self.blocks_per_stage,
            "filters_per_stage": self.filters_per_stage,
            "block_type": self.block_type,
            "stem_type": self.stem_type,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
            "activation_type": self.activation_type,
            "include_top": self.include_top,
            "enable_deep_supervision": self.enable_deep_supervision,
            "input_shape": self.input_shape_config,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResNet":
        """Create a model from its `get_config()` output."""
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)


# ---------------------------------------------------------------------


def create_resnet(
        variant: str = "resnet50",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        weights_input_shape: Optional[Tuple[int, ...]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
) -> ResNet:
    """Create a ResNet model.

    :param variant: Model variant: `"resnet18"`, `"resnet34"`, `"resnet50"`,
        `"resnet101"`, `"resnet152"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param weights_dataset: Dataset the pretrained weights were trained on.
    :param weights_input_shape: Input shape used during pretraining.
    :param cache_dir: Directory to cache downloaded weights.
    :param kwargs: Passthrough to the constructor.
    :return: A configured `ResNet` instance.
    :raises NotImplementedError: If `pretrained` is `True`.

    Example::

        model = create_resnet("resnet34", include_top=False)
        supervised = create_resnet("resnet18", num_classes=10, input_shape=(32, 32, 3), enable_deep_supervision=True)
    """
    return ResNet.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        **kwargs
    )


# ---------------------------------------------------------------------