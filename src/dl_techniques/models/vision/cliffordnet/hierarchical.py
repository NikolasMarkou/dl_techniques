"""HierarchicalCliffordNet, a CliffordNet image classifier with stages at falling resolution.

:class:`dl_techniques.models.vision.cliffordnet.model.CliffordNet` is isotropic: one
stem and ``depth`` identical blocks at constant width. This model keeps the same
geometric-algebra block (:class:`CliffordNetBlock`, a detail stream and a
depthwise-convolved context stream combined by the sparse rolling Clifford
product) but arranges it in stages, as a classical CNN pyramid does. Stage ``i``
runs ``stage_depths[i]`` blocks at width ``stage_dims[i]``; between two stages a
stride-2 downsampler halves the resolution and moves to the next width. The
architecture follows ``model_hier.py`` of the authors' repository
(github.com/ParaMind2025/CAN), whose constructor defaults are the one preset here,
``hier_345``: three stages of 3, 4 and 5 blocks at widths 32, 64 and 128, a
patch-1 stem (stage 1 runs at full input resolution), shifts ``[1, 2]``, full
interaction and differential context.

The downsampler between stages is a depthwise 3x3 convolution with stride 2, a
batch norm, SiLU and a 1x1 convolution to the next width (left out when the two
widths are equal). Stochastic depth ramps linearly from 0 to
``stochastic_depth_rate`` over ALL blocks, indexed globally and not reset per
stage. As in :class:`CliffordNet`, the blocks return only their gamma-scaled
update, so the residual add and the stochastic-depth gate sit in this model's
forward pass, and the head pools, normalizes, then applies one linear layer. The
model returns logits: compile with
``SparseCategoricalCrossentropy(from_logits=True)``.

Not ported from the authors' class, on purpose: ``embed_dim`` apart from
``stage_dims[0]`` (the stem emits ``stage_dims[0]`` channels directly), the
``dim_policy`` switch (``stage_dims=None`` doubles from 32; any list is accepted),
the ``patch`` and ``avgpool`` downsample modes, patch sizes other than 1 and 2, and
``enable_cuda``. The authors' hierarchical model has no global-context branch, and
neither does this one. No accuracy and no parameter count of this model is
published, so nothing here is compared against a paper value. The architecture was
built from a summary of the authors' file, not from a byte-checked read. No
pretrained weights are distributed, so ``pretrained=True`` raises
``NotImplementedError``; pass a local ``.keras`` path instead.

References:
    - Ji, Z., 2026. CliffordNet: All You Need is Geometric Algebra.
      (https://arxiv.org/abs/2601.06793)
    - Authors' code, ``model_hier.py``. (https://github.com/ParaMind2025/CAN)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

from __future__ import annotations

import copy
import os
import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CliffordNetBlock,
)
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head

# trunc_normal_(std=0.02) for every Conv2d and Linear, as in the reference.
_DEFAULT_KERNEL_INIT = keras.initializers.TruncatedNormal(stddev=0.02)

# Keras momentum 0.9 is torch momentum 0.1, the reference's BatchNorm default.
_BN_MOMENTUM = 0.9

# Stage widths when ``stage_dims`` is not given: doubling from this width.
_BASE_WIDTH = 32
_DEFAULT_STAGE_DEPTHS = (3, 4, 5)
_SUPPORTED_PATCH_SIZES = (1, 2)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.cliffordnet.hierarchical")
class HierarchicalCliffordNet(keras.Model):
    """Classify images with CliffordNet blocks arranged in downsampling stages.

    Architecture:

    .. code-block:: text

        input [B, H, W, C_in]
              │
              ▼
        ┌──────────────────────────────┐
        │ stem, patch_size 1 or 2      │
        │ ending in batch norm         │
        └──────────────────────────────┘
              │ [B, H/p, W/p, D_0]
              ▼
        ┌──────────────────────────────┐
        │ stage 0                      │
        │ x + drop_path(block(x))      │  x stage_depths[0]
        └──────────────────────────────┘
              │ [B, H/p, W/p, D_0]
              ▼
        ┌──────────────────────────────┐
        │ depthwise 3x3 stride 2, BN,  │
        │ SiLU, 1x1 conv to D_1        │  repeated between stages
        └──────────────────────────────┘
              │ [B, H/2p, W/2p, D_1]
              ▼
              ...  one stage per entry of stage_depths  ...
              │ [B, h, w, D_last]
              ▼
        ┌──────────────────────────────┐
        │ global average pool          │
        └──────────────────────────────┘
              │ [B, D_last]
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dense: num_classes           │
        └──────────────────────────────┘
              │
              ▼
        logits [B, num_classes]

    Stems: ``patch_size=1`` is a 3x3 convolution to ``D_0 // 2``, batch norm,
    SiLU, then a 3x3 convolution to ``D_0``; ``patch_size=2`` is one 3x3
    convolution with stride 2. Both end in a batch norm. The convolutions of the
    two-conv stem are bias-free.

    Variants:

    .. code-block:: text

        variant    stage_depths  stage_dims    patch  shifts  cli_mode
        hier_345   3, 4, 5       32, 64, 128   1      1, 2    full

    :param num_classes: Number of output classes. Must be positive.
    :param stage_depths: Blocks per stage, one entry per stage; each must be
        positive. Defaults to ``[3, 4, 5]``.
    :param stage_dims: Width of each stage, same length as ``stage_depths``; each
        must be positive. Defaults to 32 doubling per stage. The stem emits
        ``stage_dims[0]`` channels.
    :param patch_size: Total stride of the stem, 1 or 2.
    :param shifts: Channel offsets of the rolling interaction, shared by every
        stage. Defaults to ``[1, 2]``. An offset not below a stage's width is
        dropped by the block.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"`` (default).
    :param ctx_mode: ``"diff"`` (default) | ``"abs"``.
    :param layer_scale_init: Initial LayerScale gamma of every block.
    :param stochastic_depth_rate: Maximum DropPath rate, ramped linearly over all
        blocks. 0 disables it.
    :param dropout_rate: Dropout before the classifier's linear layer.
    :param use_bias: Whether the block convolutions and the downsampler's 1x1
        convolution carry a bias.
    :param kernel_initializer: Initializer of every kernel.
    :param bias_initializer: Initializer of every bias.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    :param head_config_overrides: Optional overrides merged into the head config.
    :raises ValueError: If an argument is out of range, ``stage_dims`` and
        ``stage_depths`` differ in length, or ``cli_mode`` is unknown.
    """

    # The head's pooled LayerNorm, torch's nn.LayerNorm default eps.
    LAYERNORM_EPSILON: float = 1e-5

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "hier_345": dict(
            stage_depths=[3, 4, 5],
            stage_dims=[32, 64, 128],
            patch_size=1,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    def __init__(
        self,
        num_classes: int,
        stage_depths: Optional[List[int]] = None,
        stage_dims: Optional[List[int]] = None,
        patch_size: int = 1,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        depths = list(stage_depths) if stage_depths is not None else list(_DEFAULT_STAGE_DEPTHS)
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not depths:
            raise ValueError("stage_depths must not be empty")
        if any(d <= 0 for d in depths):
            raise ValueError(f"every stage depth must be positive, got {depths}")
        dims = (
            list(stage_dims)
            if stage_dims is not None
            else [_BASE_WIDTH * 2 ** i for i in range(len(depths))]
        )
        if len(dims) != len(depths):
            raise ValueError(
                f"stage_dims {dims} and stage_depths {depths} must have the same length"
            )
        if any(d <= 0 for d in dims):
            raise ValueError(f"every stage width must be positive, got {dims}")
        if patch_size not in _SUPPORTED_PATCH_SIZES:
            raise ValueError(
                f"patch_size must be one of {_SUPPORTED_PATCH_SIZES}, got {patch_size}"
            )
        if patch_size == 1 and dims[0] < 2:
            raise ValueError(
                f"patch_size=1 halves the first width for its first convolution, "
                f"so stage_dims[0] must be at least 2, got {dims[0]}"
            )

        self.num_classes = num_classes
        self.stage_depths = depths
        self.stage_dims = dims
        self.patch_size = patch_size
        self.shifts = list(shifts) if shifts is not None else [1, 2]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self._head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides else None
        )

        self._build_stem()
        self._build_blocks()
        self._build_downsamplers()
        self._build_head()

        logger.info(
            f"Created HierarchicalCliffordNet (stage_depths={depths}, "
            f"stage_dims={dims}, patch_size={patch_size}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode})"
        )

    def _conv_kwargs(self) -> Dict[str, Any]:
        """Initializers and regularizers shared by every convolution.

        :return: Keyword arguments for ``keras.layers.Conv2D``.
        """
        return dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

    def _build_stem(self) -> None:
        """Build the stem via :class:`ConvStem`, emitting ``stage_dims[0]``.

        :return: Nothing.
        :rtype: None
        """
        width = self.stage_dims[0]
        if self.patch_size == 1:
            # Conv(D/2) → BN(momentum=0.9) → SiLU → Conv(D) → BN(momentum=0.9)
            self.stem = ConvStem(
                filters=width // 2,
                conv2_filters=width,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                num_convs=2,
                normalization_type="batch_norm",
                normalization_kwargs={"momentum": _BN_MOMENTUM},
                activation_type="silu",
                final_activation=False,
                use_pooling=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="stem",
            )
        else:
            # Conv(D, stride=2) → BN(momentum=0.9)
            self.stem = ConvStem(
                filters=width,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=self.use_bias,
                num_convs=1,
                normalization_type="batch_norm",
                normalization_kwargs={"momentum": _BN_MOMENTUM},
                activation_type="linear",
                use_pooling=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="stem",
            )

        # Keep aliases so existing references keep working.
        self.stem_conv1 = self.stem.conv1
        self.stem_conv2 = self.stem.conv2
        self.stem_conv = self.stem.conv1
        self.stem_bn1 = self.stem.norm1
        self.stem_norm = self.stem.norm2 if self.stem.norm2 is not None else self.stem.norm1

    def _build_blocks(self) -> None:
        """Build every block with its drop-path gate, in one flat list.

        The list is flat on purpose (a list of lists of layers has lost weights
        in this repository before); :attr:`stages` groups it. The drop-path rates
        ramp linearly over ALL blocks, indexed globally.

        :return: Nothing.
        :rtype: None
        """
        total = sum(self.stage_depths)
        drop_rates = linear_drop_path_rates(total, self.stochastic_depth_rate)

        self.blocks_list: List[Dict[str, Any]] = []
        index = 0
        for stage, (depth, width) in enumerate(zip(self.stage_depths, self.stage_dims)):
            for j in range(depth):
                block = CliffordNetBlock(
                    channels=width,
                    shifts=self.shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    layer_scale_init=self.layer_scale_init,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name=f"stage{stage}_block_{j}",
                )
                drop_path = StochasticDepth(
                    drop_path_rate=drop_rates[index],
                    name=f"stage{stage}_drop_path_{j}",
                )
                self.blocks_list.append(
                    {"stage": stage, "block": block, "drop_path": drop_path}
                )
                index += 1

    def _build_downsamplers(self) -> None:
        """Build the downsampler between each pair of consecutive stages.

        Depthwise 3x3 stride 2 (bias-free), batch norm, SiLU, then a 1x1
        convolution to the next width, or ``None`` when the widths are equal.

        :return: Nothing.
        :rtype: None
        """
        self.downsamplers: List[Dict[str, Any]] = []
        for i in range(len(self.stage_dims) - 1):
            pointwise = None
            if self.stage_dims[i + 1] != self.stage_dims[i]:
                pointwise = keras.layers.Conv2D(
                    filters=self.stage_dims[i + 1], kernel_size=1,
                    use_bias=self.use_bias, name=f"downsample_{i}_pointwise",
                    **self._conv_kwargs(),
                )
            self.downsamplers.append({
                "depthwise": keras.layers.DepthwiseConv2D(
                    kernel_size=3, strides=2, padding="same", use_bias=False,
                    depthwise_initializer=self.kernel_initializer,
                    depthwise_regularizer=self.kernel_regularizer,
                    name=f"downsample_{i}_depthwise",
                ),
                "bn": keras.layers.BatchNormalization(
                    name=f"downsample_{i}_bn", momentum=_BN_MOMENTUM
                ),
                "pointwise": pointwise,
            })

    def _build_head(self) -> None:
        """Build the pooled-LayerNorm classifier head as a shared ``ClassificationHead``.

        :return: Nothing.
        :rtype: None
        :raises ValueError: If ``head_config_overrides`` names a reserved key.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                'num_classes': self.num_classes,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
                'block_dims': (),
                'pooled_normalization_type': 'layer_norm',
                'normalization_kwargs': {'epsilon': self.LAYERNORM_EPSILON},
                'classifier_dropout_rate': self.dropout_rate,
                'dropout_rate': 0.0,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'bias_regularizer': self.bias_regularizer,
                # Fixed name: an unnamed head gets a counter suffix that differs
                # after a .keras round trip and breaks by-name weight transfer.
                'name': 'classification_head',
            },
            self._head_config_overrides,
        )

    @property
    def stages(self) -> List[List[Dict[str, Any]]]:
        """The blocks grouped by stage, each entry ``{"block", "drop_path", "stage"}``.

        :return: One list per stage, in order.
        """
        grouped: List[List[Dict[str, Any]]] = [[] for _ in self.stage_depths]
        for entry in self.blocks_list:
            grouped[entry["stage"]].append(entry)
        return grouped

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model via a symbolic forward pass.

        A 3-tuple is read as an unbatched shape, with the batch axis prepended.

        :param input_shape: Input tensor shape ``(B, H, W, C_in)``.
        :return: Nothing.
        :rtype: None
        """
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = tuple(input_shape)
        _ = self.call(keras.KerasTensor(build_shape))
        super().build(input_shape)

    def _apply_stem(
        self, inputs: keras.KerasTensor, training: Optional[bool]
    ) -> keras.KerasTensor:
        """Apply the stem via :class:`ConvStem`.

        :param inputs: Raw image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode (batch norm).
        :return: Embedded feature map ``(B, H/p, W/p, stage_dims[0])``.
        """
        return self.stem(inputs, training=training)

    def _downsample(
        self, index: int, x: keras.KerasTensor, training: Optional[bool]
    ) -> keras.KerasTensor:
        """Apply the downsampler that follows stage ``index``.

        :param index: Stage the downsampler follows (0-based).
        :param x: Feature map ``(B, h, w, stage_dims[index])``.
        :param training: Whether in training mode (batch norm).
        :return: ``(B, ceil(h/2), ceil(w/2), stage_dims[index + 1])``.
        """
        entry = self.downsamplers[index]
        x = entry["depthwise"](x)
        x = keras.activations.silu(entry["bn"](x, training=training))
        if entry["pointwise"] is not None:
            x = entry["pointwise"](x)
        return x

    def extract_stage_features(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> List[keras.KerasTensor]:
        """Run the stem and every stage, returning each stage's output.

        :param inputs: Image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode.
        :return: One feature map per stage, taken before that stage's
            downsampler: ``(B, h_i, w_i, stage_dims[i])``.
        """
        x = self._apply_stem(inputs, training=training)
        features: List[keras.KerasTensor] = []
        for stage, stage_blocks in enumerate(self.stages):
            for entry in stage_blocks:
                x = x + entry["drop_path"](
                    entry["block"](x, training=training), training=training
                )
            features.append(x)
            if stage < len(self.downsamplers):
                x = self._downsample(stage, x, training=training)
        return features

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the forward pass.

        :param inputs: Image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode.
        :return: Logits ``(B, num_classes)``, no softmax.
        """
        last = self.extract_stage_features(inputs, training=training)[-1]
        return self.classification_head(last, training=training)["logits"]

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: ``(B, H, W, C_in)``.
        :return: ``(B, num_classes)``.
        """
        return (input_shape[0], self.num_classes)

    def load_pretrained_weights(
        self, weights_path: str, skip_mismatch: bool = True
    ) -> None:
        """Load weights from a local ``.keras`` checkpoint into this model.

        Transfers layer by layer through
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.

        :param weights_path: Path to the ``.keras`` weights file.
        :param skip_mismatch: Skip layers whose shapes differ, which lets a
            checkpoint with another ``num_classes`` load.
        :return: Nothing.
        :rtype: None
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If the weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        try:
            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )
            logger.info(report.summary_string())
        except Exception as exc:
            raise ValueError(
                f"Failed to load weights from {weights_path}: {exc}"
            ) from exc

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "HierarchicalCliffordNet":
        """Create a :class:`HierarchicalCliffordNet` from a predefined variant.

        :param variant: A key of :attr:`MODEL_VARIANTS`.
        :param num_classes: Number of output classes.
        :param pretrained: A local path to a ``.keras`` weights file. ``True``
            raises, since no checkpoints are hosted.
        :param kwargs: Override any variant default.
        :return: Configured instance.
        :raises ValueError: If ``variant`` is not recognised.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        if pretrained is True:
            raise NotImplementedError(
                "No public HierarchicalCliffordNet checkpoints are distributed. "
                "To load local weights, pass `pretrained='/path/to/weights.keras'`."
            )
        # Deep copy: the table holds lists, and a shallow copy let a mutation of
        # the built model reach the shared table.
        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)

        logger.info(f"Creating HierarchicalCliffordNet-{variant}")
        model = cls(num_classes=num_classes, **config)

        if isinstance(pretrained, str):
            model.load_pretrained_weights(weights_path=pretrained)
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the serialisable configuration.

        :return: Dictionary with every constructor argument.
        """
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "stage_depths": self.stage_depths,
                "stage_dims": self.stage_dims,
                "patch_size": self.patch_size,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "layer_scale_init": self.layer_scale_init,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
                "head_config_overrides": self._head_config_overrides,
            }
        )
        return config


def create_hierarchical_cliffordnet(
    variant: str = "hier_345",
    num_classes: int = 100,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> "HierarchicalCliffordNet":
    """Create a HierarchicalCliffordNet from a variant name.

    A thin factory that delegates to :meth:`HierarchicalCliffordNet.from_variant`
    and adds no logic of its own. The model builds on first call, or with
    ``model.build((None, H, W, C))``.

    :param variant: A key of ``HierarchicalCliffordNet.MODEL_VARIANTS``.
    :param num_classes: Number of output classes.
    :param pretrained: A local ``.keras`` weights path, or ``True`` to raise
        ``NotImplementedError`` (no checkpoints are hosted).
    :param kwargs: Forwarded to ``from_variant`` (for example
        ``stochastic_depth_rate``, ``stage_depths``).
    :return: A :class:`HierarchicalCliffordNet`.
    :raises ValueError: If ``variant`` is not recognised.
    :raises NotImplementedError: If ``pretrained`` is ``True``.
    """
    return HierarchicalCliffordNet.from_variant(
        variant, num_classes=num_classes, pretrained=pretrained, **kwargs
    )
