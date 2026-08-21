"""
Residual networks with configurable blocks and optional deep supervision.

This model embodies the principle of residual learning, a design paradigm that
reformulates what each layer is asked to learn rather than changing its capacity.
The core idea addresses the degradation problem: beyond a certain depth, adding
layers to a plain convolutional stack makes *training* error worse, not merely
test error. Since a deeper network can always represent a shallower one by setting
the extra layers to identity, the failure is one of optimization rather than
expressiveness. Making the identity the default instead of something the layers
must discover resolves it:

`y = F(x) + x`

A block now learns only the residual `F(x)`, and driving `F` toward zero is far
easier for gradient descent than fitting an identity map through a stack of
nonlinear convolutions. The additive shortcut also gives gradients an unobstructed
path to earlier layers, so signal reaches the network's start without being
attenuated by every intervening weight matrix.

Two block designs trade depth against width. The basic block stacks two 3x3
convolutions and is used for the shallower variants. The bottleneck block
sandwiches a 3x3 convolution between 1x1 projections that reduce and then restore
channel count, which cuts the cost of the spatial convolution by roughly a factor
of the reduction ratio and is what makes 101 and 152 layer configurations
tractable. Five preset variants span the standard family, from ResNet-18
(`[2, 2, 2, 2]`, basic) through ResNet-152 (`[3, 8, 36, 3]`, bottleneck), all over
the same `[64, 128, 256, 512]` filter progression.

Architecturally the model is a 7x7 stride-2 stem with max pooling, followed by
four stages that each halve resolution and double width. Shortcut projections are
inserted only where the identity cannot be taken verbatim: at the first block of
every stage after the first, where the stride-2 shortcut changes spatial shape,
and additionally at stage 0's first block under the bottleneck design, where the
channel count must widen even though the stride is 1. Everywhere else the shortcut
is parameter-free, which is what keeps the identity path exactly an identity.

Deep supervision is available as an optional training aid. Intermediate stages are
given their own pooling and classification heads, so gradient enters the network
at several depths rather than only at the output. This shortens the effective
backpropagation distance for early layers and pressures intermediate
representations to be linearly discriminative on their own. Every stage but the
last gets a head: the final stage is already served by the main head, and stage 0
— the shallowest — is exactly where the shortened path is worth having. For the
four-stage default that is three supervision heads, on stages 1, 2 and 3 counting
from one. When enabled the model returns `[final_output, stage3, stage2, stage1]`,
reversed so the deepest supervision head comes first; inference typically consumes
index 0 alone.

Normalization and activation are supplied through factories rather than
hard-coded, and an optional `normalization_kwargs` dict is forwarded to every
construction site in both the stem and each block. Its default of `None` resolves
to an empty dict, leaving all factory calls byte-identical to the pre-plumbing
version so existing checkpoints remain bit-exact.

No pretrained weights are distributed with this package. `pretrained=True` raises
`NotImplementedError` rather than warning and returning a randomly initialized
model, which is a deliberate choice: the previous behaviour made an unavailable
download silently indistinguishable from a successful one. Local checkpoints are
loaded by path, with shape mismatches in the classifier or input-dependent layers
skipped by name when the target task differs from the checkpoint's.

References:
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
    - He et al., 2016. Identity Mappings in Deep Residual Networks.
      (https://arxiv.org/abs/1603.05027)
    - Lee et al., 2015. Deeply-Supervised Nets. AISTATS 2015.
      (https://arxiv.org/abs/1409.5185)
    - Veit et al., 2016. Residual Networks Behave Like Ensembles of Relatively
      Shallow Networks. (https://arxiv.org/abs/1605.06431)
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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ResNet(keras.Model):
    """
    Deep residual network with configurable blocks and optional deep supervision.

    Implements the ResNet family, in which each block learns a residual
    ``F(x)`` added to an identity shortcut (``y = F(x) + x``) so that depth does
    not obstruct optimization. A 7x7 stride-2 stem feeds four stages that each
    halve spatial resolution and double channel width, built from either
    ``BasicBlock`` (two 3x3 convolutions) or ``BottleneckBlock`` (1x1 reduce,
    3x3, 1x1 expand). Shortcut projections are inserted only where the identity
    cannot be taken verbatim: at the first block of every stage after the first
    (stride-2 shape change), plus stage 0's first block under the bottleneck
    design (channel widening at stride 1). With ``enable_deep_supervision=True``
    every stage but the last receives its own GAP + Dense head (the last stage is
    the main head's own), and the model returns
    ``[final_output, stage3, stage2, stage1]``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │       Input [B, H, W, C_in]          │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Stem: Conv 7×7 /2 → Norm → Act      │
        │        → MaxPool 3×3 /2              │
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

    **Variants:**

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
        resolves to ``{}``, keeping all calls byte-identical to the
        pre-plumbing version so existing checkpoints stay bit-exact.
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
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.

    :raises ValueError: If ``blocks_per_stage`` and ``filters_per_stage`` differ
        in length, if ``block_type`` is not ``'basic'`` or ``'bottleneck'``, or
        if ``input_shape`` is not 3D.

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

        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.blocks_per_stage = blocks_per_stage
        self.filters_per_stage = filters_per_stage
        self.block_type = block_type
        self.kernel_regularizer = kernel_regularizer
        self.normalization_type = normalization_type
        # DECISION plan_2026-05-18_6776f8ba/D-003
        # Optional `normalization_kwargs` forwarded to every
        # `create_normalization_layer` call inside the stem AND inside every
        # BasicBlock/BottleneckBlock. Default `None` -> `{}` -> all factory
        # calls byte-identical to the pre-plumbing version, preserving
        # bit-exactness for every existing ResNet checkpoint. Used by
        # `src/train/rms_variants_train/experiments/e2_resnet_cifar100.py`
        # in `--mode param_matched` to pass `use_scale=False` so the
        # gamma-removal contrast in the headline E2 result becomes a
        # pure 1-vs-d parameter-count confound rather than a norm choice.
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}
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
        """Build initial convolution stem.

        # DECISION plan-2026-08-19T163559-499b6f0e/D-041
        The stem width FOLLOWS ``filters_per_stage[0]``; it is not the literal
        64 it used to be. Do not put the constant back. A `basic` block's
        stage-0 first block is given ``use_projection=False``
        unconditionally -- correctly, because stage 0 does not stride -- so its
        identity shortcut requires the stem to emit exactly
        ``filters_per_stage[0]`` channels. With the stem pinned at 64,
        ``ResNet(block_type='basic')`` could not run a forward pass for ANY
        other stage-0 width: MEASURED 8, 16, 32 and 128 all raise while 64
        works, `bottleneck` works at every width (its stage-0 block projects),
        and `from_variant` works because every shipped variant uses 64. The
        error surfaced two frames deep in `layers/standard_blocks.py` naming
        neither `filters_per_stage` nor 64.
        This is checkpoint-safe by construction: it changes the stem's weight
        SHAPE only for configurations that previously RAISED.
        See decisions.md D-041.
        """
        self.stem_conv = keras.layers.Conv2D(
            filters=self.filters_per_stage[0],
            kernel_size=7,
            strides=2,
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
        )

    def _build_stage(self, stage_idx: int) -> None:
        """Build a residual stage.

        Args:
            stage_idx: Index of the stage to build.
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
        """Build deep supervision classification heads.

        One GAP + Dense head per stage EXCEPT the last. The final stage is
        already served by the main head — ``call`` appends one entry to
        ``stage_features`` per stage, so a head at ``stage_idx ==
        len(blocks_per_stage) - 1`` would read the very tensor ``self.gap`` /
        ``self.classifier`` consume and inject no gradient the main head does
        not already carry. Stage 0 IS supervised: it is the only stage for which
        deep supervision actually shortens the backpropagation path, which is
        the entire point of the technique.
        """
        # DECISION plan-2026-08-17T183311-79c63e38/D-019
        # range(0, N-1), NOT range(1, N) and NOT range(1, N-1). The old bound
        # supervised the FINAL stage (a duplicate of the main head) and skipped
        # stage 0 (the only one that shortens backprop). Do NOT "restore" the
        # `Stage 0 is skipped as too shallow` rule that the old prose asserted:
        # skipping stage 0 while also excluding the final stage would leave a
        # 4-stage ResNet with two supervision heads and no shallow supervision
        # at all. See decisions.md D-019.
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


    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, List[keras.KerasTensor]]:
        """Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating training mode.

        Returns:
            Output tensor or list of tensors depending on configuration:
            - If deep_supervision=False: Single output tensor
              - If include_top=True: (batch_size, num_classes)
              - If include_top=False: (batch_size, H', W', channels)
            - If deep_supervision=True: List of output tensors
              [final_output, supervision_output_stage3, supervision_output_stage2, supervision_output_stage1]
        """
        x = self.stem_conv(inputs)
        x = self.stem_bn(x, training=training)
        x = self.stem_act(x)
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
        not ``self.load_weights(..., by_name=True)``. Keras 3 removed ``by_name``
        from ``Model.load_weights`` — its signature is
        ``(filepath, skip_mismatch=False, **kwargs)`` and it rejects the unknown
        keyword — so the old call raised ``ValueError: Invalid keyword arguments:
        {'by_name': True}`` for every caller. Nothing noticed, because the only
        route to it was ``pretrained=<path>`` and the surrounding ``except``
        turned the failure into a warning that continued with random weights.

        Args:
            weights_path: String, path to the weights file (.keras format).
            skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
                Maps to the inverse of the transfer helper's ``strict``.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If weights cannot be loaded.
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

    # `_download_weights` raises instead of falling back to random init. The
    # previous version held a `PRETRAINED_WEIGHTS` table of placeholder URLs
    # pointing at a non-existent host; `from_variant` caught the download failure,
    # logged a warning and returned a randomly-initialized model, so
    # `pretrained=True` silently produced untrained weights. Do NOT reinstate a
    # warn-and-return branch here or in `from_variant`. No public ResNet weights
    # are distributed with dl_techniques; pass a local path via
    # `pretrained="/path/to/file.keras"` or use `pretrained=False` (default).
    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public ResNet weights ship with ``dl_techniques``.
        Always raises. Kept to mirror the BERT / GPT-2 / WaveFieldLLM factory
        recipe and to give an explicit failure mode instead of a silent
        random-init fallback.

        Args:
            variant: Variant name (unused).
            dataset: Dataset name (unused).
            cache_dir: Cache directory (unused).

        Raises:
            NotImplementedError: Always.
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

        Args:
            variant: String, one of "resnet18", "resnet34", "resnet50",
                "resnet101", "resnet152".
            num_classes: Integer, number of output classes.
            input_shape: Tuple, input shape. If None, uses (224, 224, 3).
            pretrained: If a string, a path to a local weights file to load.
                If True, raises NotImplementedError — no public ResNet weights
                ship with dl_techniques. If False (default), returns a
                randomly-initialized model.
            weights_dataset: String, dataset for pretrained weights.
            weights_input_shape: Tuple, input shape used during weight pretraining.
            cache_dir: Optional string, directory to cache downloaded weights.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            ResNet model instance.

        Raises:
            ValueError: If variant is not recognized.
            NotImplementedError: If pretrained is True.

        Example:
            >>> # Create with deep supervision for training
            >>> model = ResNet.from_variant("resnet50", enable_deep_supervision=True)
            >>>
            >>> # Fine-tune on custom dataset
            >>> model = ResNet.from_variant("resnet34", num_classes=10,
            ...                             input_shape=(32, 32, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127
        # House style (`wave_field/model.py`): copy the preset, drop the
        # metadata key, then `config.update(kwargs)`. Do NOT go back to
        # splatting named preset fields alongside `**kwargs` -- every
        # documented override of one of those fields raised
        # `TypeError: got multiple values for keyword argument`
        # (MEASURED at all six sites). The `.copy()` is NOT optional and
        # NOT cosmetic: `config.update(kwargs)` on the shared
        # `MODEL_VARIANTS[variant]` dict would permanently poison the
        # class-level table for every later caller. See decisions.md D-127.
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
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ResNet model instance.
        """
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)


# ---------------------------------------------------------------------
# Utility Functions for Deep Supervision
# ---------------------------------------------------------------------

# Re-exported so callers of `models.resnet` get the deep-supervision helpers
# from the same module as the model itself — `src/train/resnet/train_resnet.py`
# imports `create_resnet` and `get_model_output_info` in one statement. The
# import sits here rather than at the top because it is an API re-export, not a
# dependency of the class above.
from dl_techniques.utils.deep_supervision import (  # noqa: E402,F401
    get_model_output_info,
    create_inference_model_from_training_model,
)


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
    """Convenience function to create ResNet models.

    Args:
        variant: String, model variant ("resnet18", "resnet34", "resnet50",
            "resnet101", "resnet152").
        num_classes: Integer, number of output classes.
        input_shape: Tuple, input shape.
        pretrained: If a string, a path to a local weights file. If True, raises
            NotImplementedError — no public ResNet weights ship with
            dl_techniques. If False (default), random initialization.
        weights_dataset: String, dataset for pretrained weights.
        weights_input_shape: Tuple, input shape used during weight pretraining.
        cache_dir: Optional string, directory to cache downloaded weights.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        ResNet model instance.

    Raises:
        NotImplementedError: If pretrained is True.

    Example:
        >>> # Create ResNet-34 as a feature extractor
        >>> model = create_resnet("resnet34", include_top=False)
        >>>
        >>> # Fine-tune on CIFAR-10 with deep supervision
        >>> model = create_resnet("resnet18", num_classes=10,
        ...                       input_shape=(32, 32, 3),
        ...                       enable_deep_supervision=True)
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