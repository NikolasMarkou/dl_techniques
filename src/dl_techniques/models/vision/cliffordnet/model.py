"""CliffordNet, an isotropic image classifier built from geometric-algebra blocks and no FFN.

This file holds :class:`CliffordNet`, its ``MODEL_VARIANTS`` table and the
``create_cliffordnet`` factory, plus the stem, head-config and argument helpers that
``HierarchicalCliffordNet`` imports from here. Where a standard vision block separates a token
mixer from a channel-mixing MLP, a CliffordNet block does both with the Clifford
geometric product `u v = u . v + u ^ v` of two per-pixel channel streams: the
symmetric part acts as an attention score and the antisymmetric part carries the
disagreement between the streams. The product is bilinear, so channels mix
without an FFN. Each block takes a pointwise detail stream and a
depthwise-convolved context stream and samples only a few channel offsets of
their pairwise product (`shifts`), costing `O(N * D * |shifts|)` instead of
`O(D^2)`. The model is isotropic: one patch-embedding stem, then `depth`
identical blocks at constant width, with no stage structure. `CliffordNetBlock`
returns the gamma-scaled update alone, so the residual add and the
stochastic-depth gate sit in this model's `call`. The head pools before it
normalizes. No pretrained weights are distributed, so `pretrained=True` raises
`NotImplementedError`; pass a local `.keras` path instead. The model returns
logits: compile with ``SparseCategoricalCrossentropy(from_logits=True)``.
With ``include_top=False`` it returns the final feature map instead.

Every initializer is stored as a spec (a name or a config dict) and that spec,
never a live instance, is handed to the stem, each block and the head, so no two
layers replay one random draw and the preset initializer carries no seed fixed at
import time.

References:
    - Ji, Z., 2026. CliffordNet: All You Need is Geometric Algebra.
      (https://arxiv.org/abs/2601.06793)
    - Brandstetter et al., 2023. Clifford Neural Layers for PDE Modeling.
      (https://arxiv.org/abs/2209.04934)
    - Ruhe et al., 2023. Geometric Clifford Algebra Networks.
      (https://arxiv.org/abs/2302.06594)
    - Yu et al., 2022. MetaFormer Is Actually What You Need for Vision.
      (https://arxiv.org/abs/2111.11418)
    - Touvron et al., 2021. Going Deeper with Image Transformers (CaiT).
      (LayerScale) (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

from __future__ import annotations

import copy
import keras
import numpy as np
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
from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.weight_transfer import (
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head

# Matches the reference: trunc_normal_(std=0.02) for all Conv2d and Linear. A
# config dict and not an Initializer instance: an instance would draw its seed
# once, at import, and replay that draw at every layer it reaches.
DEFAULT_KERNEL_INITIALIZER: Dict[str, Any] = {
    "class_name": "TruncatedNormal",
    "config": {"stddev": 0.02},
}

# The head's pooled LayerNorm. The reference's ``nn.LayerNorm(embed_dim)`` takes
# torch's default eps, 1e-5, not the 1e-6 of its LayerNorm2d blocks.
LAYERNORM_EPSILON = 1e-5

# DECISION plan-2026-08-23T091307-9a110062/D-480: momentum stays 0.9; Keras and
# torch define it oppositely, so the reference's 0.1 is this value. See
# decisions.md.
STEM_BN_MOMENTUM = 0.9

# DECISION plan-2026-10-26T125959-7af851e5/D-007: the stem's BatchNormalization
# epsilon is 1e-3, the Keras default these layers had before the stem moved onto
# ConvStem. It is the PRE-MIGRATION value, not a published reference (no paper
# citation was found). The norm factory's own default is 1e-6, so leaving it
# unstated moved every stem BN by 1000x with no shape symptom. Guard:
# tests/test_models/test_the_conv_stem_replicates_every_hand_rolled_stem.py.
STEM_BN_EPSILON = 1e-3

# Unbatched probe shape a pretrained load builds at when the model is not built
# yet (shared with HierarchicalCliffordNet, which imports it from here). Every
# patch size accepts it (patch 4 leaves 8x8, the hierarchical stride chain 4x4),
# and the weights do not depend on it.
DEFAULT_INPUT_SHAPE = (32, 32, 3)


def coerce_include_top(include_top: Any) -> bool:
    """Return ``include_top`` as a Python ``bool``, refusing anything else.

    Shared by :class:`CliffordNet` and ``HierarchicalCliffordNet`` (which imports
    it from here, like :data:`DEFAULT_INPUT_SHAPE`), so the two cannot drift.

    Interface contract: ``bool`` and ``numpy.bool_`` are accepted (the type every
    comparison on an array yields) and coerced, so ``get_config`` stays
    JSON-serializable. Everything else is refused, including ``0``/``1``, a
    0-d array and ``None``: ``include_top`` was once the second positional
    parameter, and ``bool(...)`` would have let a misbound ``CliffordNet(10, 64)``
    through. The message prints the repr and the module-qualified type, because
    ``type(np.False_).__name__`` is ``"bool"`` under numpy 2 and a bare name
    would contradict the refusal.

    :param include_top: The value passed as ``include_top``.
    :return: ``bool(include_top)``.
    :raises TypeError: If ``include_top`` is not a ``bool`` or ``numpy.bool_``.
    """
    if not isinstance(include_top, (bool, np.bool_)):
        kind = type(include_top)
        raise TypeError(
            f"include_top must be a bool or numpy.bool_, got {include_top!r} "
            f"of type {kind.__module__}.{kind.__qualname__}"
        )
    return bool(include_top)


def validate_rates(dropout_rate: float, stochastic_depth_rate: float) -> None:
    """Refuse a head dropout or drop-path rate outside ``[0, 1)``.

    Shared by :class:`CliffordNet` and ``HierarchicalCliffordNet``. Without it
    the sub-layer that first meets the value raises with its own argument name
    (``classifier_dropout_rate``, ``drop_path_rate``), which is not the name the
    caller wrote.

    :param dropout_rate: The model's ``dropout_rate``.
    :param stochastic_depth_rate: The model's ``stochastic_depth_rate``.
    :raises ValueError: If either rate is outside ``[0, 1)``; the message names
        the model's argument and the value.
    """
    for name, value in (
        ("dropout_rate", dropout_rate),
        ("stochastic_depth_rate", stochastic_depth_rate),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}")


def build_clifford_stem(
    channels: int,
    patch_size: int,
    use_bias: bool,
    kernel_initializer: Union[str, Dict[str, Any]],
    bias_initializer: Union[str, Dict[str, Any]],
    kernel_regularizer: Optional[keras.regularizers.Regularizer],
    bias_regularizer: Optional[keras.regularizers.Regularizer],
) -> ConvStem:
    """Build the patch-embedding stem of :class:`CliffordNet` and ``HierarchicalCliffordNet``.

    Interface contract: ``patch_size`` 1 and 4 give the two-conv stem
    ``Conv(channels // 2) -> BN -> SiLU -> Conv(channels) -> BN`` with both
    convolutions bias-free (3x3, stride 1 for patch 1 and 2 for patch 4, so patch
    4 reaches its total stride with two stride-2 convolutions). Any other
    ``patch_size`` gives ``Conv(channels) -> BN`` with the bias per ``use_bias``:
    3x3 stride 2 for patch 2, kernel and stride ``patch_size`` otherwise. Every
    BatchNorm takes :data:`STEM_BN_MOMENTUM` and :data:`STEM_BN_EPSILON`. The
    returned :class:`~dl_techniques.layers.conv_blocks.stem.ConvStem` is named
    ``stem``. The initializers are specs; the stem draws the second kernel of the
    two-conv stem independently of the first. Argument validation is the
    caller's: a non-positive ``channels`` or ``patch_size`` is not checked here.

    :param channels: Width of the stem's output.
    :param patch_size: Total stride of the stem and the selector for which stem
        is built.
    :param use_bias: Whether a single-conv stem's convolution has a bias.
    :param kernel_initializer: Kernel initializer spec for every convolution.
    :param bias_initializer: Bias initializer spec for every convolution.
    :param kernel_regularizer: Kernel regularizer for every convolution.
    :param bias_regularizer: Bias regularizer for every convolution.
    :return: The stem layer.
    """
    norm_kwargs = {"momentum": STEM_BN_MOMENTUM, "epsilon": STEM_BN_EPSILON}
    conv_kwargs: Dict[str, Any] = dict(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    )

    if patch_size in (1, 4):
        return ConvStem(
            filters=channels // 2,
            conv2_filters=channels,
            kernel_size=3,
            strides=1 if patch_size == 1 else 2,
            padding="same",
            use_bias=False,
            num_convs=2,
            normalization_type="batch_norm",
            normalization_kwargs=norm_kwargs,
            activation_type="silu",
            final_activation=False,
            use_pooling=False,
            name="stem",
            **conv_kwargs,
        )

    is_patch2 = patch_size == 2
    return ConvStem(
        filters=channels,
        kernel_size=3 if is_patch2 else patch_size,
        strides=2 if is_patch2 else patch_size,
        padding="same",
        use_bias=use_bias,
        num_convs=1,
        normalization_type="batch_norm",
        normalization_kwargs=norm_kwargs,
        activation_type="linear",
        use_pooling=False,
        name="stem",
        **conv_kwargs,
    )


def clifford_head_config(
    num_classes: int,
    dropout_rate: float,
    kernel_initializer: Union[str, Dict[str, Any]],
    bias_initializer: Union[str, Dict[str, Any]],
    kernel_regularizer: Optional[keras.regularizers.Regularizer],
    bias_regularizer: Optional[keras.regularizers.Regularizer],
) -> Dict[str, Any]:
    """Return the head configuration shared by :class:`CliffordNet` and ``HierarchicalCliffordNet``.

    Interface contract: the dict is the config a model hands to the vision head
    factory (``build_model_vision_head`` with ``VisionTaskType.CLASSIFICATION``,
    followed by the model's ``head_config_overrides``). It describes the
    reference's ``self.head(self.norm(x.mean(dim=[-2, -1])))`` as the shared
    ``ClassificationHead`` with no hidden block: global average pooling, a
    ``pooled_norm`` LayerNorm at :data:`LAYERNORM_EPSILON`, the optional dropout and
    one linear ``classifier``, named ``classification_head``. Every call returns a
    fresh dict. Each model builds the head itself, so each owns its head.

    :param num_classes: Width of the classifier.
    :param dropout_rate: Dropout before the classifier; 0 builds no Dropout.
    :param kernel_initializer: Kernel initializer spec of the classifier.
    :param bias_initializer: Bias initializer spec of the classifier.
    :param kernel_regularizer: Kernel regularizer of the classifier.
    :param bias_regularizer: Bias regularizer of the classifier.
    :return: The head configuration.
    """
    return {
        'num_classes': num_classes,
        'use_global_pooling': True,
        'pooling_type': 'avg',
        'use_attention': False,
        'block_dims': (),
        'pooled_normalization_type': 'layer_norm',
        'normalization_kwargs': {'epsilon': LAYERNORM_EPSILON},
        'classifier_dropout_rate': dropout_rate,
        # No hidden block, so no block dropout either.
        'dropout_rate': 0.0,
        'kernel_initializer': kernel_initializer,
        'bias_initializer': bias_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        # An unnamed head is `classification_head` in whichever process
        # builds it first and `classification_head_1`, `_2`, ... after.
        # A `.keras` round trip re-enters the naming counter, so the loaded
        # model came back as `classification_head_2` against the
        # original's `classification_head_1` -- every weight path under the
        # head differed, and by-name weight transfer could not match it.
        'name': 'classification_head',
    }

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.cliffordnet.model")
class CliffordNet(keras.Model):
    """Classify images with a stack of identical geometric-algebra blocks.

    Constant width throughout, so the only spatial reduction is the stem's.

    Architecture:

    .. code-block:: text

        input [B, H, W, C_in]
              │
              ▼
        ┌──────────────────────────────┐
        │ stem, patch_size dependent   │
        │ ending in batch norm         │
        └──────────────────────────────┘
              │ [B, H/p, W/p, D]
              ▼
        ┌──────────────────────────────┐
        │ x + drop_path(block(x))      │  x depth
        └──────────────────────────────┘
              │ [B, H/p, W/p, D]
              ▼
        ┌──────────────────────────────┐
        │ global average pool          │
        └──────────────────────────────┘
              │ [B, D]
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dropout                      │  (dropout_rate > 0)
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dense to num_classes         │
        └──────────────────────────────┘
              │
              ▼
        logits [B, num_classes]

    Pooling comes before the normalization, the reverse of the usual order.
    That is the reference head (``ParaMind2025/CAN`` ``model.py``:
    ``x.mean(dim=[-2, -1])``, ``nn.LayerNorm(embed_dim)``,
    ``nn.Linear(embed_dim, num_classes)``), configured by :func:`clifford_head_config`
    as the shared ``ClassificationHead`` with no hidden block: global average
    pooling, a ``pooled_norm`` LayerNorm at the torch default epsilon ``1e-5``
    (:data:`LAYERNORM_EPSILON`), the optional dropout, then the linear
    ``classifier``. ``call`` returns that classifier's output, the logits, as
    one tensor. The model's initializers and regularizers reach the head.

    With ``include_top=False`` the pooling, layer norm, dropout and dense layers
    of the diagram are not built and ``call`` returns the ``[B, H/p, W/p, D]``
    map the blocks produce, not a pooled vector.

    Block wiring:

    .. code-block:: text

        x
        ├──────────────────────────────┐
        ▼                              │
        clifford block, gamma-scaled   │
        │                              │
        ▼                              │
        stochastic depth               │
        │                              │
        ▼                              │
       (+)◄────────────────────────────┘
        │
        ▼
        out

    ``StochasticDepth(0.0)`` is the identity, so a rate of 0 gives exactly
    ``x + block(x)``.

    Stem variants:

    .. code-block:: text

        patch_size   layers                                     stride
        1            conv3 D/2, bn, silu, conv3 D, bn           1
        2            conv3 D stride 2, bn                       2
        4            conv3 D/2 /2, bn, silu, conv3 D /2, bn     4
        other p      conv p D stride p, bn                      p

    The stem is one :class:`~dl_techniques.layers.conv_blocks.stem.ConvStem`
    named ``stem`` (``stem_conv1``, ``stem_norm1``, ``stem_act1`` and, for the
    two-conv variants, ``stem_conv2``, ``stem_norm2``), built by
    :func:`build_clifford_stem`. Every convolution is followed by its own batch
    norm (momentum 0.9, epsilon 1e-3), so the two-conv stems hold two and the
    single-conv stems one; the SiLU sits only between the two convolutions. The
    two-conv stems are always bias-free; the single-conv stems follow
    ``use_bias``.

    ``MODEL_VARIANTS`` holds nine presets. ``nano``, ``lite`` and ``nano_g`` /
    ``lite_g`` are CliffordNet-Nano, -Lite and their +gFFN-G rows of the paper
    (arXiv:2601.06793v2, table 3). ``cn_<depth>_<shifts>`` are the factories of
    the authors' ``networks.py`` (github.com/ParaMind2025/CAN, read through a fetch
    summary): ``cn_32_3`` is the paper's CliffordNet-32 and ``cn_64_5`` its
    CliffordNet-64. The authors train ``cn_64_5`` with drop path 0.4 and every
    other preset with 0.3; that rate is a trainer argument here, not part of the
    preset. ``cn_12_3`` uses ``patch_size=1``, whose stem was not compared with the
    authors'. Parameter counts at 100 classes are pinned in
    ``tests/test_models/test_cliffordnet/test_paper_v2_variants.py``.

    Every preset uses ``ctx_mode='diff'``, ``layer_scale_init=1e-5`` and
    ``TruncatedNormal(0.02)`` kernels.

    Input shape:
        ``(batch_size, height, width, channels_in)``; the spatial sizes are free.

    Output shape:
        ``(batch_size, num_classes)`` logits with ``include_top=True``;
        ``(batch_size, ceil(height / patch_size), ceil(width / patch_size), channels)``
        with ``include_top=False``.

    :param num_classes: Number of output classes. Must be positive when
        ``include_top`` is true and is otherwise ignored, so ``0`` is accepted
        headless. Defaults to ``1000``.
    :param channels: Feature dimensionality ``D`` (constant throughout). Must
        be positive.
    :param depth: Number of CliffordNet blocks ``L``. Must be positive.
    :param patch_size: Total stride of the patch-embedding stem, and the
        selector for which stem is built. Must be positive. ``patch_size=2``
        is optimal for CIFAR-scale inputs.
    :param shifts: Channel-shift offsets for the sparse rolling product.
        ``None`` resolves to ``[1, 2]``.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"`` (default). Not
        validated here; an unknown value reaches the block.
    :param ctx_mode: ``"diff"`` (default) | ``"abs"``. Not validated here.
    :param use_global_context: Add the global-average-pool gFFN-G branch.
    :param layer_scale_init: Initial LayerScale value. Defaults to ``1e-5``.
    :param stochastic_depth_rate: Maximum DropPath rate (linearly scheduled
        across blocks), in ``[0, 1)``. Defaults to ``0.0``.
    :param dropout_rate: Pre-classifier head dropout, in ``[0, 1)``. At ``0.0``,
        the default, no Dropout layer is created.
    :param use_bias: Whether the blocks' Dense layers and the single-conv
        stems use a bias. The classifier always has one, as the reference's
        ``nn.Linear`` does.
    :param kernel_initializer: Kernel initializer for the stem, the blocks and
        the head's classifier: a name, a config dict or an ``Initializer``
        instance. It is stored as a name or config dict (an instance is
        serialized) and that spec is handed to every layer, so each layer draws
        its own kernel. Defaults to Keras' ``glorot_uniform``, while every
        ``MODEL_VARIANTS`` entry overrides it with ``TruncatedNormal(0.02)`` to
        match the reference.
    :param bias_initializer: Bias initializer, the classifier's included; stored
        and handed out like ``kernel_initializer``.
    :param kernel_regularizer: Kernel regularizer, the classifier's included.
    :param bias_regularizer: Bias regularizer, the classifier's included.
    :param head_config_overrides: Optional dict merged into the head
        configuration (for example ``{'block_dims': (256,)}``). ``name``,
        ``num_classes`` and ``task_type`` are refused with ``ValueError``.
        Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param include_top: Whether to build the classification head. When
        ``False`` there is no head, ``classification_head`` is ``None`` and
        the model returns the final feature map ``(B, H/p, W/p, channels)``
        instead of logits. Must be a real ``bool`` or ``numpy.bool_`` (stored as
        a Python ``bool``; ``64`` or ``[1, 1]`` raises ``TypeError``); it is the last named parameter, after
        ``head_config_overrides``, so it cannot capture a positional
        ``channels``. Defaults to ``True``.
    :param kwargs: Passed to :class:`keras.Model`.

    :raises TypeError: If ``include_top`` is not a ``bool``.
    :raises ValueError: If ``num_classes`` is not positive while
        ``include_top`` is true, if ``channels``, ``depth`` or ``patch_size``
        is not positive, if ``dropout_rate`` or ``stochastic_depth_rate`` is
        outside ``[0, 1)``, or if ``head_config_overrides`` names a reserved key.

    Example::

        model = CliffordNet.from_variant("nano", num_classes=100)
        backbone = CliffordNet(channels=64, depth=4, include_top=False)
    """

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "lite": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "lite_g": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "nano_g": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "cn_12_3": dict(
            channels=160,
            depth=12,
            patch_size=1,
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "cn_18_5": dict(
            channels=128,
            depth=18,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "cn_32_3": dict(
            channels=128,
            depth=32,
            patch_size=2,
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "cn_32_5": dict(
            channels=128,
            depth=32,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="inner",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
        "cn_64_5": dict(
            channels=128,
            depth=64,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="inner",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
        ),
    }

    def __init__(
        self,
        num_classes: int = 1000,
        channels: int = 128,
        depth: int = 12,
        patch_size: int = 2,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        include_top: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # A bool or numpy.bool_ only: plain `bool(...)` coercion would accept the
        # `CliffordNet(10, 64, ...)` style misbinding (64, [1, 1]) that the old
        # second slot let through.
        include_top = coerce_include_top(include_top)
        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        validate_rates(dropout_rate, stochastic_depth_rate)

        self.num_classes = num_classes
        self.include_top = include_top
        self.channels = channels
        self.depth = depth
        self.patch_size = patch_size
        self.shifts = shifts if shifts is not None else [1, 2]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        # DECISION plan-2026-10-26T125959-7af851e5/D-031: store each initializer as a spec and hand that spec, never a live instance, to the stem, every block and the head.
        # A block resolves one instance per sub-layer from a spec. An instance given to the stem and the blocks replays one draw at all of them (666 of 1891 kernel pairs at `nano`), and a clone built once outside the block loop and splatted into every block still shares it. See decisions.md.
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.bias_initializer = initializer_spec(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self._head_config_overrides = dict(head_config_overrides) if head_config_overrides else None

        self._build_stem()
        self._build_blocks()
        # Always defined, so `model.classification_head` is safe to read.
        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created CliffordNet (channels={channels}, depth={depth}, "
            f"patch_size={patch_size}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, "
            f"use_global_context={use_global_context}, include_top={include_top})"
        )

    def _build_stem(self) -> None:
        """Build the patch-embedding stem with :func:`build_clifford_stem`.

        Sizes 1, 2 and 4 get their own stems and anything else falls back to a
        single conv whose kernel and stride both equal ``patch_size``. See the
        stem table in the class docstring.

        :return: Nothing.
        :rtype: None
        """
        self.stem = build_clifford_stem(
            channels=self.channels,
            patch_size=self.patch_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

    def _build_blocks(self) -> None:
        """Build the block list, pairing each block with its drop-path gate.

        The rates ramp linearly from 0 to ``stochastic_depth_rate`` across the
        ``depth`` blocks.

        :return: Nothing.
        :rtype: None
        """
        drop_rates = linear_drop_path_rates(self.depth, self.stochastic_depth_rate)

        _block_kw: Dict[str, Any] = dict(
            channels=self.channels,
            shifts=self.shifts,
            cli_mode=self.cli_mode,
            ctx_mode=self.ctx_mode,
            use_global_context=self.use_global_context,
            layer_scale_init=self.layer_scale_init,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        self.blocks_list: List[Dict[str, Any]] = []
        for i in range(self.depth):
            block = CliffordNetBlock(
                name=f"block_{i}",
                **_block_kw,
            )
            # The blocks are transform-only, so the residual and the gate are
            # applied around them in call().
            drop_path = StochasticDepth(
                drop_path_rate=drop_rates[i],
                name=f"block_{i}_drop_path",
            )
            self.blocks_list.append({"block": block, "drop_path": drop_path})

    def _build_head(self) -> None:
        """Build the reference classifier head from :func:`clifford_head_config`.

        :return: Nothing.
        :rtype: None
        :raises ValueError: If ``head_config_overrides`` names a reserved key.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            clifford_head_config(
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
            ),
            self._head_config_overrides,
        )

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

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the forward pass.

        :param inputs: Image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode.
        :return: Logits ``(B, num_classes)``, no softmax, when ``include_top``;
            otherwise the final feature map ``(B, h, w, channels)``.
        """
        x = self.stem(inputs, training=training)

        for block_info in self.blocks_list:
            x = x + block_info["drop_path"](
                block_info["block"](x, training=training), training=training
            )

        if self.include_top:
            return self.classification_head(x, training=training)["logits"]
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: ``(B, H, W, C_in)``
        :return: ``(B, num_classes)`` when ``include_top``; otherwise the
            stem's output shape ``(B, h, w, channels)``, since the blocks
            preserve it.
        """
        if self.include_top:
            return (input_shape[0], self.num_classes)
        return tuple(self.stem.compute_output_shape(input_shape))

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True
    ) -> None:
        """Load a local ``.keras`` checkpoint into the model, layer by layer.

        The transfer is
        :func:`~dl_techniques.utils.weight_transfer.load_pretrained_checkpoint`:
        it builds the model at :data:`DEFAULT_INPUT_SHAPE` if it is unbuilt,
        takes a headed checkpoint into a headless model, and refuses a
        checkpoint that restores no layer.

        :param weights_path: Path to the ``.keras`` weights file.
        :param skip_mismatch: Whether to skip layers with mismatched shapes,
            which lets a checkpoint with another ``num_classes`` load; the
            inverse of the transfer helper's ``strict``.
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If weights cannot be loaded or none are restored.
        """
        load_pretrained_checkpoint(
            self, weights_path, DEFAULT_INPUT_SHAPE, skip_mismatch
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "CliffordNet":
        """Create a :class:`CliffordNet` from a predefined variant.

        :param variant: A key of :attr:`MODEL_VARIANTS` (see the class docstring).
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
            variant (e.g. ``depth``) overrides the preset.
        :return: Configured :class:`CliffordNet` instance.
        :raises ValueError: If ``variant`` is not recognised.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        :raises FileNotFoundError: If ``pretrained`` is a path that does not exist.

        Example::

            model = CliffordNet.from_variant("lite", num_classes=100)
            tuned = CliffordNet.from_variant("nano", num_classes=10, stochastic_depth_rate=0.1)
            warm = CliffordNet.from_variant("lite", num_classes=100, pretrained="weights.keras")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-10-26T125959-7af851e5/D-031: the download branch, `_download_weights` and the `except` clause around it are gone. `pretrained=True` raises inside `pretrained_path_or_raise` and nothing degrades to random init; the two D-001 notes below are kept verbatim.
        # DECISION plan_2026-05-11_0090b0b8/D-001: raise rather than return a
        # random-init model; NotImplementedError is outside from_variant's except
        # clause, so it reaches the caller. See decisions.md.
        # DECISION plan_2026-05-11_0090b0b8/D-001: only transport errors
        # degrade to random init here; do not widen this to
        # NotImplementedError. See decisions.md.
        weights_path = pretrained_path_or_raise(
            pretrained, f"{cls.__name__} variant '{variant}'"
        )

        # Deep copy: the table holds lists, and a shallow copy lets a mutation of
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
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        """
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "include_top": self.include_top,
                "channels": self.channels,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "layer_scale_init": self.layer_scale_init,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
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


def create_cliffordnet(
        variant: str = "lite",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
) -> CliffordNet:
    """Create a CliffordNet model from a variant name.

    A thin module-level factory that delegates to
    :meth:`CliffordNet.from_variant`, mirroring
    :func:`dl_techniques.models.vision.resnet.model.create_resnet`.

    :param variant: Model variant, a key of ``CliffordNet.MODEL_VARIANTS``.
    :param num_classes: Number of output classes.
    :param input_shape: Unbatched ``(H, W, C_in)`` to build the model at, or
        ``None`` to leave it unbuilt.
    :param pretrained: A local ``.keras`` weights path, or ``True`` to raise
        ``NotImplementedError`` (no public CliffordNet weights are hosted).
    :param kwargs: Passthrough to ``CliffordNet.from_variant`` (e.g.
        ``stochastic_depth_rate``, ``dropout_rate``, ``head_config_overrides``).
    :return: A ``CliffordNet`` instance.
    :raises ValueError: If ``variant`` is not recognised.
    :raises NotImplementedError: If ``pretrained`` is ``True``.

    Example::

        model = create_cliffordnet("lite", num_classes=100)
        small = create_cliffordnet("nano", num_classes=10, stochastic_depth_rate=0.1)
        warm = create_cliffordnet("lite", num_classes=100, pretrained="weights.keras")
    """
    return CliffordNet.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs,
    )

# ---------------------------------------------------------------------
