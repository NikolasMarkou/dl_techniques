"""FastViT (MCi) image tower for MobileCLIP2.

Builds :class:`FastVitImageEncoder`, a channels-last Keras 3 port of timm's
``FastVit`` class restricted to the five MCi configurations (mci0 through
mci4), assembled from the block primitives in ``layers/fastvit/``.

Self-attention mixes tokens globally but its cost grows with the square of
the token count, so it is too expensive in the early, high-resolution stages
where a convolutional network spends most of its time. FastViT keeps
attention only in the last one or two stages, where the feature map is
small, and mixes tokens everywhere else with RepMixer, a depthwise
convolution written as an affine residual that folds into a single
convolution at inference. Every convolution in the reference is likewise a
sum of parallel branches (a k x k conv-BN beside a 1x1 scale branch beside
an identity) that also collapses to one convolution once batch
normalization is folded in.

This port implements only the train-time multi-branch form: there is no
``reparameterize()`` / branch-fusion path anywhere under ``layers/fastvit/``
or ``layers/mobile_one_block.py``, so a built model always runs every
branch and the latency half of the paper's claim does not apply here. This
matches how the MobileCLIP2 reference weights are shipped and evaluated
(always with ``inference_mode=False``). The tower works standalone and also
as the vision branch of MobileCLIP2 — ``mobile_clip_v2.py`` imports
:class:`FastVitImageEncoder` from here. No pretrained weights are included
and this package makes no accuracy claim.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. ICCV.
      (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2022. MobileOne: An Improved One millisecond Mobile Backbone.
      (https://arxiv.org/abs/2206.04040)
    - Ding et al., 2021. RepVGG: Making VGG-style ConvNets Great Again.
      (https://arxiv.org/abs/2101.03697)
    - Hu et al., 2017. Squeeze-and-Excitation Networks.
      (https://arxiv.org/abs/1709.01507)
    - Vasu et al., 2024. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. CVPR. (https://arxiv.org/abs/2311.17049)
    - Faghri et al., 2025. MobileCLIP2: Improving Multi-Modal Reinforced
      Training. (https://arxiv.org/abs/2508.20691)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
# `keras.layers` is not imported under the bare name `layers`: this module has
# a constructor argument and a variant field named `layers` (per-stage depths).
from keras import initializers, regularizers, activations
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.conv_blocks.mobile_one_block import MobileOneBlock
from dl_techniques.layers.fastvit import FastVitStage
from dl_techniques.layers.fastvit.reference import (
    REFERENCE_NORM_EPSILON,
    REFERENCE_PADDING_MODE,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# reference constants
# ---------------------------------------------------------------------

#: Width multiplier applied to the last stage's channel count by ``final_conv``.
#: timm's ``FastVit`` default and the value every MCi variant uses.
_REFERENCE_CLS_RATIO = 2.0

#: Squeeze-and-Excitation bottleneck ratio inside ``final_conv``: timm's
#: unoverridden ``SqueezeExcite`` default. ``ReparamLargeKernelConv`` uses a
#: different ratio (0.25) elsewhere in the same network; both are correct.
_FINAL_CONV_SE_REDUCTION_RATIO = 1.0 / 16.0

#: timm's ``SqueezeExcite`` uses biased 1x1 convolutions.
_FINAL_CONV_SE_USE_BIAS = True

#: The reference applies SE before the activation
#: (``return self.act(self.se(out))`` in ``MobileOneBlock.forward``).
_FINAL_CONV_SE_POSITION = 'pre_act'

#: LayerScale gamma initialization used by every MCi block.
_REFERENCE_LAYER_SCALE_INIT = 1e-5

#: Per-head width of the attention token mixer; ``num_heads = dim // head_dim``.
_REFERENCE_HEAD_DIM = 32

#: Downsample (``FastVitPatchEmbed``) geometry, identical in every MCi variant.
_REFERENCE_DOWN_PATCH_SIZE = 7
_REFERENCE_DOWN_STRIDE = 2

#: RepMixer depthwise kernel size, identical in every MCi variant.
_REFERENCE_REPMIXER_KERNEL_SIZE = 3

#: Depthwise kernel of ``RepConditionalPosEnc``.
_REFERENCE_POS_EMB_SPATIAL_SHAPE = (7, 7)

# ---------------------------------------------------------------------
# variant table
# ---------------------------------------------------------------------

# Fields common to every row and not tabulated: cls_ratio=2.0, down_patch_size=7,
# down_stride=2, repmixer_kernel_size=3, layer_scale_init_value=1e-5, head_dim=32,
# activation=GELU. `pos_embs` entries are the RepCPE spatial shape (7, 7) or None.
#
# DECISION plan-2026-08-13T183738-24486492/D-004: mci3/mci4 rows come from the
# committed reference file (a real oracle); mci0-mci2 come from a timm fetch with
# no local oracle here. Do not fix an mci0-mci2 row by reasoning from mci3/mci4 —
# different provenance. See decisions.md.
#
# DECISION plan-2026-08-19T163559-499b6f0e/D-079: this is a MappingProxyType, not
# a plain dict, because `MODEL_VARIANTS` aliases the same object and a plain dict
# let a write through either name mutate the table for every later caller. See
# decisions.md.
MCI_VARIANTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
{
    'mci0': {
        'layers': (2, 6, 10, 2),
        'embed_dims': (64, 128, 256, 512),
        'mlp_ratios': (3.0, 3.0, 3.0, 3.0),
        'se_downsamples': (False, False, True, True),
        'downsamples': (False, True, True, True),
        'pos_embs': (None, None, None, (7, 7)),
        'token_mixers': ('repmixer', 'repmixer', 'repmixer', 'attention'),
        'stem_use_scale_branch': True,
        'norm_layer': 'batch_norm',
        'lkc_use_act': True,
    },
    'mci1': {
        'layers': (4, 12, 20, 4),
        'embed_dims': (64, 128, 256, 512),
        'mlp_ratios': (3.0, 3.0, 3.0, 3.0),
        'se_downsamples': (False, False, True, True),
        'downsamples': (False, True, True, True),
        'pos_embs': (None, None, None, (7, 7)),
        'token_mixers': ('repmixer', 'repmixer', 'repmixer', 'attention'),
        'stem_use_scale_branch': True,
        'norm_layer': 'batch_norm',
        'lkc_use_act': True,
    },
    'mci2': {
        'layers': (4, 12, 24, 4),
        'embed_dims': (80, 160, 320, 640),
        'mlp_ratios': (3.0, 3.0, 3.0, 3.0),
        'se_downsamples': (False, False, True, True),
        'downsamples': (False, True, True, True),
        'pos_embs': (None, None, None, (7, 7)),
        'token_mixers': ('repmixer', 'repmixer', 'repmixer', 'attention'),
        'stem_use_scale_branch': True,
        'norm_layer': 'batch_norm',
        'lkc_use_act': True,
    },
    'mci3': {
        'layers': (2, 12, 24, 4, 2),
        'embed_dims': (96, 192, 384, 768, 1536),
        'mlp_ratios': (4.0, 4.0, 4.0, 4.0, 4.0),
        'se_downsamples': (False, False, False, False, False),
        'downsamples': (False, True, True, True, True),
        'pos_embs': (None, None, None, (7, 7), (7, 7)),
        'token_mixers': (
            'repmixer', 'repmixer', 'repmixer', 'attention', 'attention'),
        'stem_use_scale_branch': False,
        'norm_layer': 'layer_norm',
        'lkc_use_act': True,
    },
    'mci4': {
        'layers': (2, 12, 24, 4, 4),
        'embed_dims': (128, 256, 512, 1024, 2048),
        'mlp_ratios': (4.0, 4.0, 4.0, 4.0, 4.0),
        'se_downsamples': (False, False, False, False, False),
        'downsamples': (False, True, True, True, True),
        'pos_embs': (None, None, None, (7, 7), (7, 7)),
        'token_mixers': (
            'repmixer', 'repmixer', 'repmixer', 'attention', 'attention'),
        'stem_use_scale_branch': False,
        'norm_layer': 'layer_norm',
        'lkc_use_act': True,
    },
}
)

#: The per-variant fields that must all agree in length (one entry per stage).
_PER_STAGE_FIELDS = (
    'layers',
    'embed_dims',
    'mlp_ratios',
    'se_downsamples',
    'downsamples',
    'pos_embs',
    'token_mixers',
)

# ---------------------------------------------------------------------


def _resolve_mci_variant(variant: str) -> Dict[str, Any]:
    """Look up a variant name in :data:`MCI_VARIANTS`.

    :param variant: A key of :data:`MCI_VARIANTS` (``'mci0'`` ... ``'mci4'``). The
        ``'fastvit_'`` prefix used by timm's model names is accepted and
        stripped.

    :return: A shallow copy of the variant's configuration dictionary.

    :raises ValueError: If ``variant`` is not a known variant name.
    """
    key = str(variant)
    if key.startswith('fastvit_'):
        key = key[len('fastvit_'):]
    if key not in MCI_VARIANTS:
        raise ValueError(
            f"Unknown MCi variant {variant!r}. Available: {sorted(MCI_VARIANTS)} "
            f"(a 'fastvit_' prefix is also accepted)."
        )
    return dict(MCI_VARIANTS[key])


def _stagewise_drop_path_rates(
        depths: Sequence[int],
        drop_path_rate: float,
) -> List[List[float]]:
    """Split one global linear stochastic-depth ramp into per-stage slices.

    The ramp is computed across ``sum(depths)`` blocks — the whole network — and
    then cut at the cumulative stage boundaries, so stage ``i`` starts where stage
    ``i - 1`` ended. This reproduces timm's
    ``calculate_drop_path_rates(drop_path_rate, layers, stagewise=True)``.

    :param depths: Number of blocks in each stage.
    :param drop_path_rate: Maximum (last block of the last stage) drop probability.

    :return: One list of floats per stage; concatenating them in order reproduces
        ``linear_drop_path_rates(sum(depths), drop_path_rate)`` exactly.
    """
    flat = linear_drop_path_rates(int(sum(depths)), float(drop_path_rate))
    slices: List[List[float]] = []
    cursor = 0
    for depth in depths:
        slices.append(flat[cursor:cursor + int(depth)])
        cursor += int(depth)
    return slices


@register_dl_technique("dl_techniques.models.fastvit.model")
class FastVitImageEncoder(keras.Model):
    """MobileCLIP2's FastViT (MCi) image tower.

    A convolutional stem at stride 4, then ``N`` :class:`FastVitStage` stages
    (the shallow ones mixing tokens with the convolutional RepMixer, the deepest
    one or two with global self-attention), a wide depthwise ``final_conv`` with
    squeeze-and-excitation, and a pooled projection head.

    Architecture:

    .. code-block:: text

        image [B, H, W, 3]
            |
        stem: 3x MobileOneBlock       (k3/s2 dense, k3/s2 depthwise, k1/s1) -> /4
            |
        stage_0 .. stage_{N-1}        FastVitStage (downsample? RepCPE? blocks)
            |
        final_conv: MobileOneBlock    k3, depthwise, SE, -> embed_dims[-1]*cls_ratio
            |
        GlobalAveragePooling2D
            |
        Dropout(head_dropout_rate)
            |
        Dense(projection_dim)         (the CLIP image projection, optional)
            |
        [B, projection_dim] or [B, final_features]

    The head ``Dense`` is the CLIP image projection, not a classifier.
    MobileCLIP's open_clip configs use ``timm_pool="avg"`` with
    ``timm_proj=null``, which makes the trunk's own classifier linear the
    projection into the joint image-text embedding space. Do not add a second
    projection on top of this model; pass ``projection_dim`` and use its
    output directly as the image embedding. Set ``projection_dim=None`` to
    get the pooled ``embed_dims[-1] * cls_ratio`` features instead, useful
    for backbone reuse and not for CLIP.

    :param variant: Optional key of :data:`MCI_VARIANTS` (``'mci0'`` ... ``'mci4'``,
        with an optional ``'fastvit_'`` prefix). When given, every
        architecture field left as ``None`` is filled from that variant's
        row. When ``None``, all seven per-stage tuples must be supplied
        explicitly.
    :param layers: Blocks per stage, e.g. ``(2, 6, 10, 2)``.
    :param embed_dims: Output channels per stage.
    :param mlp_ratios: ConvMlp expansion ratio per stage.
    :param se_downsamples: Whether each stage's downsample uses squeeze-and-excitation.
    :param downsamples: Whether each stage begins with a downsample. Stage 0 is
        ``False`` in every MCi variant — the stem has already done the /4.
    :param pos_embs: Per stage, the RepCPE depthwise kernel shape, or ``None`` for
        no positional encoding in that stage.
    :param token_mixers: ``'repmixer'`` or ``'attention'`` per stage.
    :param stem_use_scale_branch: Whether the three stem ``MobileOneBlock``s keep
        their 1x1 scale branch. ``False`` for mci3/mci4.
    :param norm_layer: Normalization key for the attention stages' pre-norm, either
        ``'batch_norm'`` or ``'layer_norm'``. Ignored by RepMixer stages,
        which have no ``norm_layer`` parameter in the reference.
    :param lkc_use_act: Whether each downsample's large-kernel convolution applies
        its activation.
    :param input_shape: Image shape ``(H, W, C)``. Defaults to ``(256, 256, 3)``,
        MobileCLIP's fastvit input resolution.
    :param projection_dim: Width of the CLIP image projection. ``None`` skips the
        projection and returns the pooled features. Defaults to 512.
    :param cls_ratio: ``final_conv`` widens the last stage by this factor. Defaults
        to 2.0 (the reference value).
    :param drop_path_rate: Maximum stochastic-depth rate of the single global linear
        ramp spanning every block of every stage. Defaults to 0.0.
    :param dropout_rate: Dropout inside every block's ConvMlp. Defaults to 0.0.
    :param head_dropout_rate: Dropout between the pooling and the projection.
        Defaults to 0.0 (``timm_drop`` is 0.0 in all four MobileCLIP configs).
    :param layer_scale_init_value: LayerScale gamma initialization in every block, or
        ``None`` to omit LayerScale. Defaults to ``1e-5``.
    :param head_dim: Per-head width of the attention token mixer. Defaults to 32.
    :param down_patch_size: Downsample large-kernel size. Defaults to 7.
    :param down_stride: Downsample stride. Defaults to 2.
    :param repmixer_kernel_size: RepMixer depthwise kernel size. Defaults to 3.
    :param activation: Activation used throughout. Defaults to ``'gelu'``.
    :param kernel_initializer: Initializer for every convolution / projection kernel.
        Defaults to ``'he_normal'``.
    :param kernel_regularizer: Optional regularizer applied to every kernel.
    :param **kwargs: Forwarded to :class:`keras.Model`.

    :raises ValueError: If ``variant`` is unknown; if any per-stage tuple is missing
        when no variant is given; if the seven per-stage tuples do not all
        have the same length; if ``cls_ratio`` is not positive; or if a rate
        lies outside ``[0, 1)``.

    Input shape:
        4D tensor ``(batch, height, width, channels)``.

    Output shape:
        ``(batch, projection_dim)``, or
        ``(batch, int(embed_dims[-1] * cls_ratio))`` when ``projection_dim`` is
        ``None``.

    Example:
        >>> import numpy as np
        >>> encoder = FastVitImageEncoder.from_variant('mci0', projection_dim=512)
        >>> y = encoder(np.zeros((1, 256, 256, 3), dtype='float32'), training=False)
        >>> y.shape
        (1, 512)
    """

    # `MODEL_VARIANTS` is a class-level alias of `MCI_VARIANTS`, the same object,
    # not a copy, per models/CLAUDE.md — tooling resolves a variant registry via
    # getattr(cls, 'MODEL_VARIANTS').
    MODEL_VARIANTS = MCI_VARIANTS

    def __init__(
            self,
            variant: Optional[str] = None,
            layers: Optional[Sequence[int]] = None,
            embed_dims: Optional[Sequence[int]] = None,
            mlp_ratios: Optional[Sequence[float]] = None,
            se_downsamples: Optional[Sequence[bool]] = None,
            downsamples: Optional[Sequence[bool]] = None,
            pos_embs: Optional[Sequence[Any]] = None,
            token_mixers: Optional[Sequence[str]] = None,
            stem_use_scale_branch: Optional[bool] = None,
            norm_layer: Optional[str] = None,
            lkc_use_act: Optional[bool] = None,
            input_shape: Tuple[int, int, int] = (256, 256, 3),
            projection_dim: Optional[int] = 512,
            cls_ratio: float = _REFERENCE_CLS_RATIO,
            drop_path_rate: float = 0.0,
            dropout_rate: float = 0.0,
            head_dropout_rate: float = 0.0,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            head_dim: int = _REFERENCE_HEAD_DIM,
            down_patch_size: int = _REFERENCE_DOWN_PATCH_SIZE,
            down_stride: int = _REFERENCE_DOWN_STRIDE,
            repmixer_kernel_size: int = _REFERENCE_REPMIXER_KERNEL_SIZE,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # ---- resolve the variant row (explicit arguments always win) -----
        row: Dict[str, Any] = (
            _resolve_mci_variant(variant) if variant is not None else {}
        )
        self.variant = None if variant is None else str(variant)

        def _pick(name: str, supplied: Any) -> Any:
            if supplied is not None:
                return supplied
            if name in row:
                return row[name]
            raise ValueError(
                f"{name!r} must be supplied explicitly when no `variant` is "
                f"given. Either pass variant=<one of {sorted(MCI_VARIANTS)}> or "
                f"pass every one of {list(_PER_STAGE_FIELDS)} plus "
                f"'stem_use_scale_branch', 'norm_layer', 'lkc_use_act'."
            )

        self.layers_per_stage = tuple(int(v) for v in _pick('layers', layers))
        self.embed_dims = tuple(int(v) for v in _pick('embed_dims', embed_dims))
        self.mlp_ratios = tuple(float(v) for v in _pick('mlp_ratios', mlp_ratios))
        self.se_downsamples = tuple(
            bool(v) for v in _pick('se_downsamples', se_downsamples))
        self.downsamples = tuple(bool(v) for v in _pick('downsamples', downsamples))
        self.pos_embs = tuple(
            None if v is None else tuple(v) if isinstance(v, (list, tuple))
            else (int(v), int(v))
            for v in _pick('pos_embs', pos_embs)
        )
        self.token_mixers = tuple(str(v) for v in _pick('token_mixers', token_mixers))
        self.stem_use_scale_branch = bool(
            _pick('stem_use_scale_branch', stem_use_scale_branch))
        self.norm_layer = str(_pick('norm_layer', norm_layer))
        self.lkc_use_act = bool(_pick('lkc_use_act', lkc_use_act))

        # ---- store the remaining configuration --------------------------
        self.input_shape_config = tuple(int(v) for v in input_shape)
        self.projection_dim = None if projection_dim is None else int(projection_dim)
        self.cls_ratio = float(cls_ratio)
        self.drop_path_rate = float(drop_path_rate)
        self.dropout_rate = float(dropout_rate)
        self.head_dropout_rate = float(head_dropout_rate)
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.head_dim = int(head_dim)
        self.down_patch_size = int(down_patch_size)
        self.down_stride = int(down_stride)
        self.repmixer_kernel_size = int(repmixer_kernel_size)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self._validate_config()

        # ---- derived geometry -------------------------------------------
        self.num_stages = len(self.layers_per_stage)
        #: Channel count entering ``final_conv`` and (times ``cls_ratio``) the
        #: pooled feature width.
        self.final_features = int(self.embed_dims[-1] * self.cls_ratio)
        #: A single global ramp, cut at the stage boundaries, not a per-stage ramp.
        self.drop_path_rates: List[List[float]] = _stagewise_drop_path_rates(
            self.layers_per_stage, self.drop_path_rate
        )

        # ---- sub-layers, created unbuilt ---------------------------------
        # timm's `convolutional_stem`: dense k3/s2, depthwise k3/s2, pointwise
        # k1/s1, net stride /4. `use_scale_branch` is off for mci3/mci4.
        stem_dim = self.embed_dims[0]
        self.stem: List[MobileOneBlock] = [
            MobileOneBlock(
                out_channels=stem_dim,
                kernel_size=3,
                stride=2,
                use_scale_branch=self.stem_use_scale_branch,
                activation=self.activation,
                norm_epsilon=REFERENCE_NORM_EPSILON,
                padding_mode=REFERENCE_PADDING_MODE,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='stem_0',
            ),
            MobileOneBlock(
                out_channels=stem_dim,
                kernel_size=3,
                stride=2,
                group_size=1,  # depthwise
                use_scale_branch=self.stem_use_scale_branch,
                activation=self.activation,
                norm_epsilon=REFERENCE_NORM_EPSILON,
                padding_mode=REFERENCE_PADDING_MODE,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='stem_1',
            ),
            MobileOneBlock(
                out_channels=stem_dim,
                kernel_size=1,
                stride=1,
                use_scale_branch=self.stem_use_scale_branch,
                activation=self.activation,
                norm_epsilon=REFERENCE_NORM_EPSILON,
                padding_mode=REFERENCE_PADDING_MODE,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='stem_2',
            ),
        ]

        # A flat list: a nested List[List[Layer]] restores fresh kernels on a
        # `.keras` round trip even though the layer count and parameter total match.
        self.stages: List[FastVitStage] = [
            FastVitStage(
                dim=self.embed_dims[i],
                depth=self.layers_per_stage[i],
                token_mixer=self.token_mixers[i],
                downsample=self.downsamples[i],
                se_downsample=self.se_downsamples[i],
                use_pos_emb=self.pos_embs[i] is not None,
                pos_emb_spatial_shape=(
                    self.pos_embs[i] if self.pos_embs[i] is not None
                    else _REFERENCE_POS_EMB_SPATIAL_SHAPE
                ),
                mlp_ratio=self.mlp_ratios[i],
                repmixer_kernel_size=self.repmixer_kernel_size,
                head_dim=self.head_dim,
                normalization_type=self.norm_layer,
                down_patch_size=self.down_patch_size,
                down_stride=self.down_stride,
                lkc_use_act=self.lkc_use_act,
                dropout_rate=self.dropout_rate,
                drop_path_rates=self.drop_path_rates[i],
                layer_scale_init_value=self.layer_scale_init_value,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'stage_{i}',
            )
            for i in range(self.num_stages)
        ]

        # `final_conv` is depthwise (group_size=1) and widens by cls_ratio, so its
        # Conv2D has groups=embed_dims[-1] and filters=final_features. Its SE ratio
        # is 1/16 (timm's default), different from the 0.25 ReparamLargeKernelConv uses.
        self.final_conv = MobileOneBlock(
            out_channels=self.final_features,
            kernel_size=3,
            stride=1,
            group_size=1,
            use_se=True,
            se_reduction_ratio=_FINAL_CONV_SE_REDUCTION_RATIO,
            se_use_bias=_FINAL_CONV_SE_USE_BIAS,
            se_position=_FINAL_CONV_SE_POSITION,
            num_conv_branches=1,
            activation=self.activation,
            norm_epsilon=REFERENCE_NORM_EPSILON,
            padding_mode=REFERENCE_PADDING_MODE,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='final_conv',
        )

        self.pool = keras.layers.GlobalAveragePooling2D(name='pool')
        self.head_dropout = keras.layers.Dropout(
            self.head_dropout_rate, name='head_dropout')
        # This is the CLIP image projection; see the class docstring.
        self.projection = (
            keras.layers.Dense(
                self.projection_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='projection',
            )
            if self.projection_dim is not None else None
        )

        logger.info(
            f"FastVitImageEncoder: variant={self.variant}, "
            f"{self.num_stages} stages, embed_dims={self.embed_dims}, "
            f"depths={self.layers_per_stage}, final_features={self.final_features}, "
            f"projection_dim={self.projection_dim}"
        )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the resolved configuration.

        :raises ValueError: If the per-stage tuples disagree in length, if any size is
            non-positive, or if a rate lies outside ``[0, 1)``.
        """
        # A 4-stage vs 5-stage mixup is the most likely transcription error, and
        # every downstream symptom of it is an obscure shape or channel error
        # deep inside a stage. Name the mismatched lengths here.
        lengths = {
            name: len(getattr(self, attr))
            for name, attr in (
                ('layers', 'layers_per_stage'),
                ('embed_dims', 'embed_dims'),
                ('mlp_ratios', 'mlp_ratios'),
                ('se_downsamples', 'se_downsamples'),
                ('downsamples', 'downsamples'),
                ('pos_embs', 'pos_embs'),
                ('token_mixers', 'token_mixers'),
            )
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "All per-stage tuples must have one entry per stage, but their "
                f"lengths disagree: "
                + ", ".join(f"{k}={v}" for k, v in lengths.items())
                + ". A 4-stage variant needs 4 entries in every tuple and a "
                  "5-stage variant needs 5."
            )
        if lengths['layers'] == 0:
            raise ValueError("A FastVitImageEncoder needs at least one stage.")

        for depth in self.layers_per_stage:
            if depth <= 0:
                raise ValueError(
                    f"Every stage depth must be positive, got "
                    f"{self.layers_per_stage}"
                )
        for dim in self.embed_dims:
            if dim <= 0:
                raise ValueError(
                    f"Every embed_dim must be positive, got {self.embed_dims}")

        if self.cls_ratio <= 0.0:
            raise ValueError(f"cls_ratio must be positive, got {self.cls_ratio}")
        if self.projection_dim is not None and self.projection_dim <= 0:
            raise ValueError(
                f"projection_dim must be positive or None, got {self.projection_dim}"
            )
        if len(self.input_shape_config) != 3:
            raise ValueError(
                f"input_shape must be (height, width, channels), got "
                f"{self.input_shape_config}"
            )
        for rate_name, rate in (
                ('drop_path_rate', self.drop_path_rate),
                ('dropout_rate', self.dropout_rate),
                ('head_dropout_rate', self.head_dropout_rate),
        ):
            if not 0.0 <= rate < 1.0:
                raise ValueError(f"{rate_name} must be in [0, 1), got {rate}")

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    def stem_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Shape after the three stem blocks (net stride 4).

        :param input_shape: ``(B, H, W, C)``.

        :return: ``(B, ceil(H / 4), ceil(W / 4), embed_dims[0])``.
        """
        shape = tuple(input_shape)
        for block in self.stem:
            shape = block.compute_output_shape(shape)
        return shape

    def stage_output_shapes(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> List[Tuple[Optional[int], ...]]:
        """Per-stage output shapes, composed from stored config alone.

        Valid before the model is built. The first element is the shape after
        stage 0, not after the stem — use :meth:`stem_output_shape` for that.

        :param input_shape: ``(B, H, W, C)`` of the image entering the stem.

        :return: One shape tuple per stage, in order.
        """
        shape = self.stem_output_shape(input_shape)
        shapes = []
        for stage in self.stages:
            shape = stage.compute_output_shape(shape)
            shapes.append(shape)
        return shapes

    # ------------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Build every sub-layer explicitly, in forward order.

        Each stage changes both the spatial dimensions and the channel count, so
        the next stage must be built on the previous stage's output shape, not
        on ``input_shape``.

        :param input_shape: ``(B, H, W, C)``. When it carries no spatial information
            the model's ``input_shape`` config is used instead.
        """
        if self.built:
            return

        shape = tuple(input_shape) if input_shape is not None else None
        if shape is None or len(shape) != 4:
            shape = (None,) + self.input_shape_config

        for block in self.stem:
            block.build(shape)
            shape = block.compute_output_shape(shape)

        for stage in self.stages:
            stage.build(shape)
            shape = stage.compute_output_shape(shape)

        self.final_conv.build(shape)
        shape = self.final_conv.compute_output_shape(shape)

        self.pool.build(shape)
        pooled_shape = self.pool.compute_output_shape(shape)
        self.head_dropout.build(pooled_shape)
        if self.projection is not None:
            self.projection.build(pooled_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode a batch of images.

        :param inputs: Image tensor ``(B, H, W, C)``.
        :param training: Keras training flag. Pass ``False`` explicitly for a
            deterministic forward; ``training=None`` runs the stochastic
            path of every stochastic-depth branch and uses batch statistics
            in every BatchNormalization.

        :return: ``(B, projection_dim)``, the CLIP image embedding (un-normalized), or
            ``(B, final_features)`` when ``projection_dim`` is ``None``.
        """
        x = inputs
        for block in self.stem:
            x = block(x, training=training)
        for stage in self.stages:
            x = stage(x, training=training)
        x = self.final_conv(x, training=training)
        x = self.pool(x)
        x = self.head_dropout(x, training=training)
        if self.projection is not None:
            x = self.projection(x)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Output shape from stored config — valid before the model is built.

        :param input_shape: ``(B, H, W, C)``.

        :return: ``(B, projection_dim)`` or ``(B, final_features)``.
        """
        shape = tuple(input_shape)
        batch = shape[0] if len(shape) == 4 else None
        width = (
            self.projection_dim if self.projection_dim is not None
            else self.final_features
        )
        return (batch, width)

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full configuration for serialization.

        :return: A dictionary containing every constructor parameter, with the
            architecture stored explicitly, not only as a variant name, so a
            checkpoint keeps describing the network it was trained with even if
            the variant table is later corrected.
        """
        config = super().get_config()
        config.update({
            'variant': self.variant,
            'layers': list(self.layers_per_stage),
            'embed_dims': list(self.embed_dims),
            'mlp_ratios': list(self.mlp_ratios),
            'se_downsamples': list(self.se_downsamples),
            'downsamples': list(self.downsamples),
            'pos_embs': [
                None if p is None else list(p) for p in self.pos_embs
            ],
            'token_mixers': list(self.token_mixers),
            'stem_use_scale_branch': self.stem_use_scale_branch,
            'norm_layer': self.norm_layer,
            'lkc_use_act': self.lkc_use_act,
            'input_shape': list(self.input_shape_config),
            'projection_dim': self.projection_dim,
            'cls_ratio': self.cls_ratio,
            'drop_path_rate': self.drop_path_rate,
            'dropout_rate': self.dropout_rate,
            'head_dropout_rate': self.head_dropout_rate,
            'layer_scale_init_value': self.layer_scale_init_value,
            'head_dim': self.head_dim,
            'down_patch_size': self.down_patch_size,
            'down_stride': self.down_stride,
            'repmixer_kernel_size': self.repmixer_kernel_size,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitImageEncoder":
        """Rebuild the model from a serialized configuration.

        :param config: A dictionary produced by :meth:`get_config`.

        :return: A new :class:`FastVitImageEncoder`.
        """
        config = dict(config)
        config['input_shape'] = tuple(config['input_shape'])
        config['activation'] = activations.deserialize(config['activation'])
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Tuple[int, int, int] = (256, 256, 3),
            projection_dim: Optional[int] = 512,
            **kwargs: Any,
    ) -> "FastVitImageEncoder":
        """Create an encoder from an MCi variant name.

        :param variant: A key of :data:`MCI_VARIANTS` (``'mci0'`` ... ``'mci4'``),
            optionally prefixed ``'fastvit_'``.
        :param input_shape: Image shape ``(H, W, C)``.
        :param projection_dim: CLIP image-projection width, or ``None``.
        :param **kwargs: Forwarded to the constructor.

        :return: The configured image tower.

        :raises ValueError: If ``variant`` is not recognized.
        """
        return cls(
            variant=variant,
            input_shape=input_shape,
            projection_dim=projection_dim,
            **kwargs,
        )


# ---------------------------------------------------------------------


def create_fastvit_image_encoder(
        variant: str = 'mci0',
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        projection_dim: Optional[int] = 512,
        **overrides: Any,
) -> FastVitImageEncoder:
    """Create a MobileCLIP2 FastViT image tower.

    :param variant: ``'mci0'`` ... ``'mci4'`` (an optional ``'fastvit_'`` prefix is
        accepted). Defaults to ``'mci0'``.
    :param input_shape: Image shape ``(H, W, C)``. Defaults to ``(256, 256, 3)``.
    :param projection_dim: Width of the CLIP image projection (the head ``Dense``),
        or ``None`` to return pooled features. Defaults to 512.
    :param **overrides: Any :class:`FastVitImageEncoder` constructor keyword, e.g.
        ``drop_path_rate``, ``head_dropout_rate``, ``kernel_regularizer``.

    :return: The configured image tower.

    :raises ValueError: If ``variant`` is not recognized.
    """
    return FastVitImageEncoder(
        variant=variant,
        input_shape=input_shape,
        projection_dim=projection_dim,
        **overrides,
    )

# ---------------------------------------------------------------------
