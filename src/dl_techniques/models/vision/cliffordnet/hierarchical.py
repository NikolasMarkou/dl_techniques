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
``SparseCategoricalCrossentropy(from_logits=True)``. With ``include_top=False`` it
returns the last stage's feature map instead.

Not ported from the authors' class, on purpose: ``embed_dim`` apart from
``stage_dims[0]`` (the stem emits ``stage_dims[0]`` channels directly), the
``dim_policy`` switch (``stage_dims=None`` doubles from 32; any list is accepted),
the ``patch`` and ``avgpool`` downsample modes, patch sizes other than 1 and 2, and
``enable_cuda``. The authors' hierarchical model has no global-context branch, and
neither does this one. No accuracy and no parameter count of this model is
published, so nothing here is compared against a paper value. The architecture was
built from a summary of the authors' file, not from a byte-checked read. No
pretrained weights are distributed, so ``pretrained=True`` raises
``NotImplementedError``; pass a local ``.keras`` path instead. The stem, head and
argument checks are CliffordNet's own helpers, imported from
:mod:`dl_techniques.models.vision.cliffordnet.model`.

References:
    - Ji, Z., 2026. CliffordNet: All You Need is Geometric Algebra.
      (https://arxiv.org/abs/2601.06793)
    - Authors' code, ``model_hier.py``. (https://github.com/ParaMind2025/CAN)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import copy
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
from dl_techniques.models.vision.cliffordnet.model import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_KERNEL_INITIALIZER,
    STEM_BN_MOMENTUM,
    build_clifford_stem,
    clifford_head_config,
    coerce_include_top,
    validate_rates,
)
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.model_build import (
    materialize_sublayers,
    plain_head_config_overrides,
)
from dl_techniques.utils.weight_transfer import (
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head

# Stage widths when ``stage_dims`` is not given: doubling from this width.
_BASE_WIDTH = 32
_DEFAULT_STAGE_DEPTHS = (3, 4, 5)
_SUPPORTED_PATCH_SIZES = (1, 2)

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (initializers,
# regularizers, dropout and stochastic-depth rates, the layer-scale starting value).
# `load_pretrained_checkpoint` compares every other key the checkpoint and the model
# both hold, so a changed `shifts`, `ctx_mode`, `cli_mode` or width is refused even when
# every weight shape still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "kernel_initializer",
    "bias_initializer",
    "kernel_regularizer",
    "bias_regularizer",
    "dropout_rate",
    "stochastic_depth_rate",
    "layer_scale_init",
)

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

    With ``include_top=False`` the pooling, layer norm, dropout and dense layers
    of the diagram are not built and ``call`` returns the ``[B, h, w, D_last]``
    map of the last stage, not a pooled vector.

    Stems: ``patch_size=1`` is a 3x3 convolution to ``D_0 // 2``, batch norm,
    SiLU, then a 3x3 convolution to ``D_0``, then a second batch norm;
    ``patch_size=2`` is one 3x3 convolution with stride 2 and a batch norm. The
    stem is one :class:`~dl_techniques.layers.conv_blocks.stem.ConvStem` named
    ``stem`` whose batch norms use momentum 0.9 and epsilon 1e-3. The
    convolutions of the two-conv stem are bias-free. The stem and the head are
    CliffordNet's (``build_clifford_stem`` and ``clifford_head_config``), so the two
    models cannot drift.

    ``MODEL_VARIANTS`` holds the one preset, ``hier_345``, the authors' constructor
    default.

    :param num_classes: Number of output classes. Must be positive when
        ``include_top`` is true and is otherwise ignored, so ``0`` is accepted
        headless. Defaults to ``1000``.
    :type num_classes: int
    :param stage_depths: Blocks per stage, one entry per stage; each must be
        positive. Defaults to ``[3, 4, 5]``.
    :type stage_depths: Optional[List[int]]
    :param stage_dims: Width of each stage, same length as ``stage_depths``; each
        must be positive. Defaults to 32 doubling per stage. The stem emits
        ``stage_dims[0]`` channels.
    :type stage_dims: Optional[List[int]]
    :param patch_size: Total stride of the stem, 1 or 2.
    :type patch_size: int
    :param shifts: Channel offsets of the rolling interaction, shared by every
        stage. Defaults to ``[1, 2]``. An offset not below a stage's width is
        dropped by the block.
    :type shifts: Optional[List[int]]
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"`` (default).
    :type cli_mode: CliMode
    :param ctx_mode: ``"diff"`` (default) | ``"abs"``.
    :type ctx_mode: CtxMode
    :param layer_scale_init: Initial LayerScale gamma of every block.
    :type layer_scale_init: float
    :param stochastic_depth_rate: Maximum DropPath rate, ramped linearly over all
        blocks, in ``[0, 1)``. 0 disables it.
    :type stochastic_depth_rate: float
    :param dropout_rate: Dropout before the classifier's linear layer, in
        ``[0, 1)``.
    :type dropout_rate: float
    :param use_bias: Whether the block convolutions and the downsampler's 1x1
        convolution carry a bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer of every kernel: a name, a config dict
        or an ``Initializer`` instance. It is stored as a name or config dict (an
        instance is serialized) and that spec is handed to every layer, so each
        layer draws its own kernel when the spec is a name, a dict or a seedless
        instance. An instance or dict carrying an explicit ``seed`` reuses that
        seed at every layer by contract, so same-shaped kernels start equal.
    :type kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param bias_initializer: Initializer of every bias; stored and handed out like
        ``kernel_initializer``.
    :type bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param kernel_regularizer: Optional kernel regularizer.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional bias regularizer.
    :type bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param head_config_overrides: Optional overrides merged into the head config.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param include_top: Whether to build the classification head. When
        ``False`` there is no head, ``classification_head`` is ``None`` and
        the model returns the last stage's feature map
        ``(B, h, w, stage_dims[-1])`` instead of logits. Must be a real
        ``bool`` or ``numpy.bool_`` (stored as a Python ``bool``; ``[1, 1]``
        raises ``TypeError``); it is the last named
        parameter, after ``head_config_overrides``, so it cannot capture a
        positional ``stage_depths``. Defaults to ``True``.
    :type include_top: bool
    :param kwargs: Passed to :class:`keras.Model`.
    :raises TypeError: If ``include_top`` is not a ``bool``.
    :raises ValueError: If an argument is out of range (``num_classes`` only
        while ``include_top`` is true), ``dropout_rate`` or
        ``stochastic_depth_rate`` is outside ``[0, 1)``, ``stage_dims`` and
        ``stage_depths`` differ in length, or ``cli_mode`` is unknown.

    Input shape:
        ``(batch_size, height, width, channels_in)``; the spatial sizes are free.

    Output shape:
        ``(batch_size, num_classes)`` logits with ``include_top=True``;
        ``(batch_size, h, w, stage_dims[-1])`` with ``include_top=False``, where
        ``h`` and ``w`` are the stem's output sizes halved (rounding up) once per
        downsampler.

    Example::

        model = HierarchicalCliffordNet.from_variant("hier_345", num_classes=100)
        small = HierarchicalCliffordNet(stage_depths=[2, 2], stage_dims=[16, 32])
    """

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "hier_345": dict(
            stage_depths=[3, 4, 5],
            stage_dims=[32, 64, 128],
            patch_size=1,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
    }

    def __init__(
        self,
        num_classes: int = 1000,
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
        kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        include_top: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # A bool or numpy.bool_ only: plain `bool(...)` coercion would accept the
        # `(10, [1, 1])` style misbinding that the old second slot let through.
        include_top = coerce_include_top(include_top)
        depths = list(stage_depths) if stage_depths is not None else list(_DEFAULT_STAGE_DEPTHS)
        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )
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
        validate_rates(dropout_rate, stochastic_depth_rate)

        self.num_classes = num_classes
        self.include_top = include_top
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
        # DECISION plan-2026-10-26T125959-7af851e5/D-031: store each initializer as a spec and hand that spec, never a live instance, to the stem, every block, every downsampler and the head.
        # A block resolves one instance per sub-layer from a spec. An instance given to the stem, the blocks and the downsamplers replays one draw at all of them (820 of 2211 kernel pairs at `hier_345`), and a clone built once outside the block loop and splatted into every block still shares it. See decisions.md.
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.bias_initializer = initializer_spec(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self._head_config_overrides = plain_head_config_overrides(
            head_config_overrides, "HierarchicalCliffordNet"
        )

        self._build_stem()

        self.blocks: List[Dict[str, Any]] = []
        self._build_blocks()

        self.downsamplers: List[Dict[str, Any]] = []
        self._build_downsamplers()

        # Always defined, so `model.classification_head` is safe to read.
        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created HierarchicalCliffordNet (stage_depths={depths}, "
            f"stage_dims={dims}, patch_size={patch_size}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, include_top={include_top})"
        )

    def _conv_kwargs(self) -> Dict[str, Any]:
        """Initializers and regularizers shared by every convolution.

        :return: Keyword arguments for ``keras.layers.Conv2D`` and ``ConvStem``.
        """
        return dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

    def _build_stem(self) -> None:
        """Build the stem with :func:`build_clifford_stem`, emitting ``stage_dims[0]``.

        :return: Nothing.
        :rtype: None
        """
        self.stem = build_clifford_stem(
            channels=self.stage_dims[0],
            patch_size=self.patch_size,
            use_bias=self.use_bias,
            **self._conv_kwargs(),
        )

    def _build_blocks(self) -> None:
        """Build every block with its drop-path gate, in one flat list.

        The list is flat so that the blocks are created, tracked and saved in one
        order, stage after stage; the :attr:`stages` property groups it by each
        entry's ``stage``, so the partition is derived from ``stage_depths`` and
        is not a second container to keep in step with the first. The drop-path
        rates ramp linearly over ALL blocks, indexed globally.

        :return: Nothing.
        :rtype: None
        """
        total = sum(self.stage_depths)
        drop_rates = linear_drop_path_rates(total, self.stochastic_depth_rate)
        conv_kw = self._conv_kwargs()

        index = 0
        for stage, (depth, width) in enumerate(zip(self.stage_depths, self.stage_dims)):
            for j in range(depth):
                name = f"stage{stage + 1}_block{j + 1}"
                block = CliffordNetBlock(
                    channels=width,
                    shifts=self.shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    layer_scale_init=self.layer_scale_init,
                    use_bias=self.use_bias,
                    name=name,
                    **conv_kw,
                )
                drop_path = StochasticDepth(
                    drop_path_rate=drop_rates[index],
                    name=f"{name}_drop_path",
                )
                self.blocks.append(
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
        conv_kw = self._conv_kwargs()
        # The initializer is the stored spec, so each layer resolves its own draw.
        depthwise_kw = dict(
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
        )
        for i in range(len(self.stage_dims) - 1):
            pointwise = None
            if self.stage_dims[i + 1] != self.stage_dims[i]:
                pointwise = keras.layers.Conv2D(
                    filters=self.stage_dims[i + 1], kernel_size=1,
                    use_bias=self.use_bias, name=f"downsample_{i}_pointwise",
                    **conv_kw,
                )
            depthwise = keras.layers.DepthwiseConv2D(
                kernel_size=3, strides=2, padding="same", use_bias=False,
                name=f"downsample_{i}_depthwise", **depthwise_kw,
            )
            # The downsampler's BatchNormalization takes the stem's momentum.
            # DECISION plan-2026-10-26T125959-7af851e5/D-007: the stem's BatchNormalization
            # epsilon is 1e-3, the Keras default these layers had before the stem moved onto
            # ConvStem. It is the PRE-MIGRATION value, not a published reference (no paper
            # citation was found). The norm factory's own default is 1e-6, so leaving it
            # unstated moved every stem BN by 1000x with no shape symptom. The downsamplers
            # below still use the bare Keras default. Guard:
            # tests/test_models/test_the_conv_stem_replicates_every_hand_rolled_stem.py.
            norm = keras.layers.BatchNormalization(
                name=f"downsample_{i}_norm", momentum=STEM_BN_MOMENTUM
            )
            self.downsamplers.append({
                "depthwise": depthwise,
                "norm": norm,
                "pointwise": pointwise,
            })

    def _build_head(self) -> None:
        """Build the pooled-LayerNorm classifier head from :func:`clifford_head_config`.

        :return: Nothing.
        :rtype: None
        :raises ValueError: If ``head_config_overrides`` names a reserved key.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            clifford_head_config(
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
                head_config_overrides=self._head_config_overrides,
                **self._conv_kwargs(),
            ),
            self._head_config_overrides,
        )

    @property
    def stages(self) -> List[List[Dict[str, Any]]]:
        """The blocks grouped by stage, each entry ``{"block", "drop_path", "stage"}``.

        :return: One list per stage, in order.
        """
        grouped: List[List[Dict[str, Any]]] = [[] for _ in self.stage_depths]
        for entry in self.blocks:
            grouped[entry["stage"]].append(entry)
        return grouped

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Materialize every sub-layer from ``input_shape`` by tracing ``call``.

        :param input_shape: Input tensor shape ``(B, H, W, C_in)``.
        :return: Nothing.
        :rtype: None
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

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
        x = keras.activations.silu(entry["norm"](x, training=training))
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
        x = self.stem(inputs, training=training)
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
        :return: Logits ``(B, num_classes)``, no softmax, when ``include_top``;
            otherwise the last stage's feature map ``(B, h, w, stage_dims[-1])``.
        """
        last = self.extract_stage_features(inputs, training=training)[-1]
        if self.include_top:
            return self.classification_head(last, training=training)["logits"]
        return last

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: ``(B, H, W, C_in)``.
        :return: ``(B, num_classes)`` when ``include_top``, the head's own
            ``compute_output_shape``; otherwise the last stage's map
            ``(B, h, w, stage_dims[-1])``: the stem's output shape chained
            through each downsampler's own ``compute_output_shape``, an unknown
            size staying unknown.
        """
        shape = tuple(self.stem.compute_output_shape(input_shape))
        for entry in self.downsamplers:
            shape = entry["depthwise"].compute_output_shape(shape)
            if entry["pointwise"] is not None:
                shape = entry["pointwise"].compute_output_shape(shape)
        if not self.include_top:
            return tuple(shape)
        return tuple(self.classification_head.compute_output_shape(shape)["logits"])

    def load_pretrained_weights(
        self, weights_path: str, skip_mismatch: bool = True
    ) -> None:
        """Load a local ``.keras`` checkpoint into the model, layer by layer.

        The transfer is
        :func:`~dl_techniques.utils.weight_transfer.load_pretrained_checkpoint`:
        it builds the model at :data:`DEFAULT_INPUT_SHAPE` if it is unbuilt and
        takes a headed checkpoint into a headless model. Every layer outside
        ``classification_head`` must be restored, so a checkpoint of another
        width, depth or stem raises and copies no weight. Only a head of
        another ``num_classes`` is skipped: it keeps its initialization and a
        warning names it. The archive's own config is compared too: a
        constructor argument that both hold and that is not in
        ``_NON_STRUCTURAL_CONFIG_KEYS`` must be equal, so a checkpoint of
        another ``ctx_mode``, ``cli_mode`` or ``shifts`` is refused even when
        every weight shape fits. An argument only one side holds is not
        compared, and an archive in which no single model of this class can be
        found is checked by layers only (the load warns). Where the head is
        copied, its `head_config_overrides` must be equal too (dropout,
        initializers and regularizers aside).

        :param weights_path: Path to the ``.keras`` weights file.
        :param skip_mismatch: ``True`` (default) lets the head differ, which
            lets a checkpoint with another ``num_classes`` load; ``False``
            allows no difference anywhere. The inverse of the transfer
            helper's ``strict``.
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If a layer outside the head is not restored, or
            the weights cannot be loaded.
        """
        load_pretrained_checkpoint(
            self, weights_path, DEFAULT_INPUT_SHAPE, skip_mismatch,
            ignore_config_keys=_NON_STRUCTURAL_CONFIG_KEYS,
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "HierarchicalCliffordNet":
        """Create a :class:`HierarchicalCliffordNet` from a predefined variant.

        :param variant: A key of :attr:`MODEL_VARIANTS`.
        :param num_classes: Number of output classes. Ignored when
            ``include_top=False`` is passed in ``kwargs``.
        :param input_shape: Unbatched ``(H, W, C_in)`` the model is built at.
            The constructor takes no ``input_shape``, so unlike ResNet, which
            only stores it, ``from_variant`` builds the model whenever one is
            given, with or without ``pretrained``. Without it the model stays
            unbuilt until its first call or ``build``, and a pretrained load
            builds it at :data:`DEFAULT_INPUT_SHAPE` ``(32, 32, 3)``. Weight
            shapes do not depend on ``H`` and ``W``, only on ``C_in``.
        :param pretrained: A path to a local ``.keras`` weights file, ``True``
            (raises ``NotImplementedError``, since no checkpoints are hosted)
            or ``False`` (default, random init).
        :param kwargs: Passthrough to the constructor; an argument of the
            variant (e.g. ``stage_depths``) overrides the preset.
        :return: Configured instance.
        :raises ValueError: If ``variant`` is not recognised.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        :raises FileNotFoundError: If ``pretrained`` is a path that does not exist.

        Example::

            model = HierarchicalCliffordNet.from_variant("hier_345", num_classes=100)
            backbone = HierarchicalCliffordNet.from_variant(
                "hier_345", include_top=False, pretrained="weights.keras"
            )
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

        # Deep copy: the table holds lists, and a shallow copy let a mutation of
        # the built model reach the shared table.
        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)

        model = cls(num_classes=num_classes, **config)

        if input_shape is not None:
            model.build((None,) + tuple(input_shape))

        if weights_path:
            model.load_pretrained_weights(weights_path)

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the serialisable configuration.

        :return: Dictionary with every constructor argument.
        """
        config = {
            "num_classes": self.num_classes,
            "include_top": self.include_top,
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
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def create_hierarchical_cliffordnet(
    variant: str = "hier_345",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = None,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> HierarchicalCliffordNet:
    """Create a HierarchicalCliffordNet from a variant name.

    A thin factory that delegates to :meth:`HierarchicalCliffordNet.from_variant`
    and adds no logic of its own.

    :param variant: A key of ``HierarchicalCliffordNet.MODEL_VARIANTS``.
    :param num_classes: Number of output classes.
    :param input_shape: Unbatched ``(H, W, C_in)`` to build the model at, or
        ``None`` to leave it unbuilt.
    :param pretrained: A local ``.keras`` weights path, or ``True`` to raise
        ``NotImplementedError`` (no checkpoints are hosted).
    :param kwargs: Forwarded to ``from_variant`` (for example
        ``stochastic_depth_rate``, ``stage_depths``).
    :return: A :class:`HierarchicalCliffordNet`.
    :raises ValueError: If ``variant`` is not recognised.
    :raises NotImplementedError: If ``pretrained`` is ``True``.

    Example::

        model = create_hierarchical_cliffordnet("hier_345", num_classes=100)
    """
    return HierarchicalCliffordNet.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs,
    )
