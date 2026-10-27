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

The head is the shared vision `ClassificationHead` configured as the
paper's: global average pool, then one linear `Dense(num_classes)`
(`block_dims=()`). The model returns its pre-softmax LOGITS: compile with
`from_logits=True` and apply `keras.ops.softmax` to read probabilities. With
deep supervision on, every stage but the last gets its own head of the same
kind, and the model returns the logits list
`[final_output, stage3, stage2, stage1]` (deepest head first).

Every convolution kernel draws from `he_normal` unless `kernel_initializer`
names another; each convolution resolves its own initializer from that spec,
so two kernels do not replay one sample when it is a name, a config dict or a
seedless instance. A seeded `Initializer` instance (or a dict with an explicit
`seed`) reuses its seed at every site by contract, so same-shaped kernels start
equal (`HeNormal(seed=7)` ties 190 of 210 kernel pairs at `resnet18`).
BatchNorm momentum defaults to `0.9` (Keras
convention) to match torchvision's `momentum=0.1`
(`keras_momentum = 1 - torch_momentum`), a training-only constant that does
not affect the `training=False` forward pass. No pretrained weights ship with
this package; `pretrained=True` raises `NotImplementedError`. Local
checkpoints load by path; every layer outside the classification head must
be restored, and only a head of another `num_classes` is skipped.

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

import copy
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import (
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.layers.conv_blocks.basic_block import BasicBlock
from dl_techniques.layers.conv_blocks.bottleneck_block import BottleneckBlock
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head
from dl_techniques.utils.model_build import (
    materialize_sublayers,
    plain_head_config_overrides,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (the initializer and the
# regularizer, the training-only deep-supervision heads, the input size).
# `load_pretrained_checkpoint` compares every other key the checkpoint and the model
# both hold, so a changed block type, width, stem or activation is refused even when
# every weight shape still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "enable_deep_supervision",
    "input_shape",
    "kernel_initializer",
    "kernel_regularizer",
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.resnet.model")
class ResNet(keras.Model):
    """Deep residual network with configurable blocks and optional deep supervision.

    Implements the ResNet family, in which each block learns a residual
    ``F(x)`` added to an identity shortcut (``y = F(x) + x``) so that depth does
    not obstruct optimization. A stem -- 7x7 stride-2 plus max pool by default,
    or a 3x3 stride-1 CIFAR stem via ``stem_type='cifar'`` -- feeds four stages
    that each halve spatial resolution and double channel width, built from
    either ``BasicBlock`` (two 3x3 convolutions) or ``BottleneckBlock`` (1x1
    reduce, 3x3, 1x1 expand). Shortcut projections are inserted only where the
    identity cannot be taken verbatim: at the first block of every stage after
    the first (stride-2 shape change), plus stage 0's first block under the
    bottleneck design (channel widening at stride 1). The head is the shared
    ``ClassificationHead`` configured as the paper's (He et al., Sec. 3.4:
    "global average pooling and a 1000-way fully-connected layer"): GAP, then
    a linear ``Dense(num_classes)`` with no hidden block (``block_dims=()``),
    and the model returns its logits. With ``enable_deep_supervision=True``
    every stage but the last receives its own head of the same kind (the last
    stage is the main head's own), and the model returns the logits list
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
        │  ClassificationHead (if include_top) │ │ Classific.   │
        │  GAP → Dense, logits                 │ │ Head / stage │
        └───────────────┬──────────────────────┘ └──────┬───────┘
                        │                               │
                        ▼                               ▼
        ┌───────────────────────────────────────────────────────┐
        │  Output: logits [B, num_classes]                      │
        │   include_top=False        → [B, H', W', f₄·expansion]│
        │   deep_supervision=True    → [final, s3, s2, s1]      │
        └───────────────────────────────────────────────────────┘

    :param num_classes: Number of output classes. Only used when
        ``include_top=True``, where it must be positive.
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
    :param kernel_regularizer: Optional regularizer applied to every convolution
        kernel of the stem and the blocks, and to the classifier kernel of the
        main and every supervision head. A regularizer, its serialized config or
        a name.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param kernel_initializer: Initializer of every convolution kernel in the stem
        and the blocks (the classifier heads keep the head's own initializer). A
        name or a serialized config is handed to each convolution, which draws
        from it independently; an ``Initializer`` instance is stored as its
        serialized config. A seedless instance, name or dict gives every
        convolution its own draw; an instance or dict carrying an explicit
        ``seed`` reuses that seed at every site by contract, so same-shaped
        kernels start equal (``HeNormal(seed=7)`` ties 190 of 210 kernel pairs
        at ``resnet18``). Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
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
    :type input_shape: Optional[Tuple[int, ...]]
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
    :param head_config_overrides: Optional dict merged into the configuration of
        the main and every supervision ``ClassificationHead`` (e.g.,
        ``{'block_dims': (512,), 'dropout_rate': 0.2}`` adds a hidden block).
        ``name``, ``num_classes`` and ``task_type`` are owned by the model and
        raise ``ValueError``. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.

    :raises ValueError: If ``include_top`` is True and ``num_classes`` is not
        positive, if ``blocks_per_stage`` and ``filters_per_stage`` differ
        in length, if ``block_type`` is not ``'basic'`` or ``'bottleneck'``, if
        ``stem_type`` is not ``'imagenet'`` or ``'cifar'``, if
        ``input_shape`` is not 3D, if ``kernel_initializer`` is not a known
        initializer, or if ``head_config_overrides`` names ``name``,
        ``num_classes`` or ``task_type``.
    :raises TypeError: If ``kernel_initializer`` is not a string, a dict or an
        ``Initializer`` instance.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_top=True``: 2D logits tensor ``(batch_size, num_classes)``.
        - ``include_top=False``: 4D tensor ``(batch_size, H', W', channels)``.
        - ``enable_deep_supervision=True``: list of 2D logits tensors, ordered
          ``[final_output, stage3, stage2, stage1]`` (reversed, matching the
          BFUNet output convention). Inference typically uses index 0 alone.

    Example::

        # ResNet-50 for ImageNet
        model = ResNet.from_variant("resnet50", num_classes=1000)

        # Deep supervision for training
        model = ResNet.from_variant("resnet50", enable_deep_supervision=True)

        # Feature extractor from a local checkpoint
        model = ResNet.from_variant("resnet34", pretrained="/path/to.keras",
                                    include_top=False)

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
            kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
            kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = "he_normal",
            normalization_type: str = "batch_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_type: str = "relu",
            include_top: bool = True,
            enable_deep_supervision: bool = False,
            input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
            stem_type: Literal["imagenet", "cifar"] = "imagenet",
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )

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
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        # DECISION plan-2026-10-26T125959-7af851e5/D-033: store the initializer as a spec and hand that spec to the stem and to every block; never keep the caller's Initializer instance.
        # A seedless instance given to two layers replays one random draw at both, and an instance in get_config is not JSON-safe. See decisions.md.
        self.kernel_initializer = initializer_spec(kernel_initializer)
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

        self._head_config_overrides = plain_head_config_overrides(
            head_config_overrides, "ResNet"
        )

        self._build_stem()

        self.stages = []
        for stage_idx in range(len(blocks_per_stage)):
            self._build_stage(stage_idx)

        self.classification_head = None
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
        """Build the initial convolution stem via :class:`ConvStem`.

        Both stem variants take their width from `filters_per_stage[0]`,
        not a literal 64, so a `basic`-block model can run at any stage-0
        width.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-041: stem width must follow filters_per_stage[0], never a literal 64.
        # A basic block's stage-0 shortcut is unconditionally identity, so a mismatched stem width raised a forward-pass error 2 frames deep. See decisions.md.
        # DECISION plan-2026-08-23T203721-009b7ccf/D-019: stem_type='cifar' must stay a literal single-conv/no-pool alternative, never derived from input_shape.
        # An input-shape-derived stride would silently change the weight tree of every small-input checkpoint; 'imagenet' stays the bit-identical default. See decisions.md (this is a different plan's D-019 than _build_supervision_heads').
        _use_pooling = self.stem_type == "imagenet"
        self.stem = ConvStem(
            filters=self.filters_per_stage[0],
            kernel_size=7 if self.stem_type == "imagenet" else 3,
            strides=2 if self.stem_type == "imagenet" else 1,
            padding="same",
            use_bias=False,
            num_convs=1,
            normalization_type=self.normalization_type,
            normalization_kwargs=dict(self.normalization_kwargs),
            activation_type=self.activation_type,
            use_pooling=_use_pooling,
            pool_type="max",
            pool_size=3,
            # DECISION plan-2026-10-26T125959-7af851e5/D-019 and plan-2026-10-26T125959-7af851e5/D-020 (D-020 corrects D-019): the original ResNet passed strides=2, padding="same" explicitly; ConvStem's defaults (stride = pool_size, "valid") would drift from them, so state both. Do not drop these arguments. See decisions.md.
            pool_strides=2,
            pool_padding="same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stem",
        )

    def _build_stage(self, stage_idx: int) -> None:
        """Build one residual stage's blocks.

        :param stage_idx: Index of the stage to build.
        """
        num_blocks = self.blocks_per_stage[stage_idx]
        base_filters = self.filters_per_stage[stage_idx]

        BlockClass = BasicBlock if self.block_type == "basic" else BottleneckBlock

        # The initializer is the stored spec (a name or a config dict, never an
        # Initializer instance), so each block, and each convolution inside it,
        # resolves its own draw from it.
        kernel_kwargs = {
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_initializer": self.kernel_initializer,
        }

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
                **kernel_kwargs,
                normalization_type=self.normalization_type,
                normalization_kwargs=dict(self.normalization_kwargs),
                activation_type=self.activation_type,
                name=f"stage{stage_idx+1}_block{block_idx+1}"
            )
            stage_blocks.append(block)

        self.stages.append(stage_blocks)

    def _head_config(self, name: str) -> Dict[str, Any]:
        """Configuration of one `ClassificationHead`, before `head_config_overrides`.

        The paper's head: global average pooling, then one linear layer and no
        hidden block. The name is deterministic because an unnamed head takes
        Keras' auto suffix in process-creation order, which breaks the by-name
        matching that weight transfer relies on.

        :param name: Layer name of the head.
        :return: The configuration passed to `build_model_vision_head`.
        """
        return {
            'num_classes': self.num_classes,
            'dropout_rate': 0.0,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'use_global_pooling': True,
            'pooling_type': 'avg',
            'use_attention': False,
            'block_dims': (),
            'kernel_regularizer': self.kernel_regularizer,
            'name': name,
        }

    def _build_head(self) -> None:
        """Build the main classification head through the vision heads factory."""
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            self._head_config('classification_head'),
            self._head_config_overrides,
        )

    def _build_supervision_heads(self) -> None:
        """Build one ClassificationHead per stage except the last; each emits logits.

        The final stage is already served by the main head; stage 0 is
        supervised since it is the one stage where deep supervision
        actually shortens the backpropagation path. Each head is named by its
        stage, since this model builds several.
        """
        # DECISION plan-2026-08-17T183311-79c63e38/D-019: range(0, N-1), not range(1, N) or range(1, N-1).
        # Supervising the final stage duplicates the main head; skipping stage 0 loses the one stage the technique helps most. See decisions.md.
        for stage_idx in range(0, len(self.blocks_per_stage) - 1):
            sup_head = build_model_vision_head(
                VisionTaskType.CLASSIFICATION,
                self._head_config(f'supervision_head_{stage_idx}'),
                self._head_config_overrides,
            )
            self.supervision_heads.append({
                "head": sup_head,
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
            The logits `(batch_size, num_classes)` (`include_top=True`)
            or `(batch_size, H', W', channels)` (`include_top=False`), or,
            with deep supervision, the logits list
            `[final_output, stage3, stage2, stage1]`.
        """
        x = self.stem(inputs, training=training)

        stage_features = []
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x, training=training)
            if self.enable_deep_supervision and self.include_top:
                stage_features.append(x)

        if self.include_top:
            head_output = self.classification_head(x, training=training)
            final_output = head_output['logits']
        else:
            final_output = x

        if self.enable_deep_supervision and self.include_top and self.supervision_heads:
            # Reversed (stage 3, 2, 1) to match the BFUNet output convention.
            supervision_outputs = []
            for sup_head in reversed(self.supervision_heads):
                feat = stage_features[sup_head["stage_idx"]]
                sup_output = sup_head["head"](feat, training=training)
                supervision_outputs.append(sup_output['logits'])
            return [final_output] + supervision_outputs

        return final_output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """Report the shape `call` returns, without building the model.

        :param input_shape: `(batch_size, height, width, channels)`; the
            spatial axes may be `None`.
        :return: The logits shape `(batch_size, num_classes)`
            (`include_top=True`), the final feature map
            `(batch_size, H', W', channels)` (`include_top=False`), or, with
            deep supervision, one logits shape per head in the order `call`
            returns them.
        """
        shape = self.stem.compute_output_shape(input_shape)
        stage_shapes = []
        for stage_blocks in self.stages:
            for block in stage_blocks:
                shape = block.compute_output_shape(shape)
            stage_shapes.append(shape)

        if not self.include_top:
            return shape

        logits = self.classification_head.compute_output_shape(shape)['logits']
        if not self.supervision_heads:
            return logits
        return [logits] + [
            sup_head["head"].compute_output_shape(
                stage_shapes[sup_head["stage_idx"]]
            )['logits']
            for sup_head in reversed(self.supervision_heads)
        ]

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load a local `.keras` checkpoint into the model, layer by layer.

        The transfer is `load_pretrained_checkpoint`: it builds the model at
        `input_shape` if it is unbuilt and takes a headed checkpoint into a
        headless model. Every layer outside the classification head (and the deep-supervision heads) must be
        restored, so a checkpoint of another variant, block type, width or stem
        raises and leaves the model untouched. Only a head of another
        `num_classes` is skipped: it keeps its initialization and a warning
        names it. The archive's own config is compared too: a constructor
        argument that both hold and that is not in `_NON_STRUCTURAL_CONFIG_KEYS`
        must be equal, so a checkpoint of another `activation_type` or
        `stem_type` is refused even when every weight shape fits. An argument
        only one side holds is not compared, and an archive without a config of
        this class is checked by layers only.

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
    ) -> "ResNet":
        """Create a ResNet model from a predefined variant.

        :param variant: One of `"resnet18"`, `"resnet34"`, `"resnet50"`,
            `"resnet101"`, `"resnet152"`.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape; defaults to `(224, 224, 3)`.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no public ResNet weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; an architecture field
            of the variant (e.g. `blocks_per_stage`) overrides the preset.
        :return: A configured `ResNet` instance.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

        Example::

            model = ResNet.from_variant("resnet50", enable_deep_supervision=True)
            small = ResNet.from_variant("resnet34", num_classes=10, input_shape=(32, 32, 3))
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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127: copy the preset, drop metadata, then config.update(kwargs) -- do not splat preset fields alongside **kwargs.
        # Splatting raised TypeError on every overridden field; skipping .copy() would permanently poison the shared MODEL_VARIANTS[variant] dict. See decisions.md.
        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)

        model = cls(num_classes=num_classes, input_shape=input_shape, **config)

        if weights_path:
            model.load_pretrained_weights(weights_path)

        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "blocks_per_stage": self.blocks_per_stage,
            "filters_per_stage": self.filters_per_stage,
            "block_type": self.block_type,
            "stem_type": self.stem_type,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_initializer": self.kernel_initializer,
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
            "activation_type": self.activation_type,
            "include_top": self.include_top,
            "enable_deep_supervision": self.enable_deep_supervision,
            "input_shape": self.input_shape_config,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}


# ---------------------------------------------------------------------


def create_resnet(
        variant: str = "resnet50",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs
) -> ResNet:
    """Create a ResNet model.

    :param variant: Model variant: `"resnet18"`, `"resnet34"`, `"resnet50"`,
        `"resnet101"`, `"resnet152"`.
    :param num_classes: Number of output classes.
    :param input_shape: Input shape; defaults to `(224, 224, 3)`.
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :param kwargs: Passthrough to `ResNet.from_variant`, e.g.
        `head_config_overrides`, `enable_deep_supervision`, `include_top`.
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
        **kwargs
    )


# ---------------------------------------------------------------------