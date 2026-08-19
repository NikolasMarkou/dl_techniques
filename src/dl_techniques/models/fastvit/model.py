"""FastViT (MCi) image tower assembled from the ``layers/fastvit/`` primitives.

FastViT is built around a tension every efficient vision backbone has to
resolve: self-attention mixes tokens globally but costs quadratically in their
number, so it is unaffordable exactly where a convolutional network spends most
of its time — the early, high-resolution stages. FastViT keeps attention only in
the deepest one or two stages, where the feature map is small, and mixes tokens
everywhere else with RepMixer, a depthwise convolution written as
``y = x + gamma * (Mixer(x) - Norm(x))``. The subtraction is what makes that
expression collapse: ``Norm`` is a deliberately degenerate block whose only
surviving branch is an identity BatchNormalization, so the whole residual is
affine at inference and fuses into a single depthwise convolution.

That fusability is the architecture's second idea, structural
reparameterization. Wherever the reference could have used one convolution it
uses a sum of parallel branches — a ``k x k`` conv-BN beside a ``1x1`` scale
branch beside an identity, or a ``7x7`` beside a ``3x3`` — which is a strictly
better-conditioned thing to optimize, and which collapses back into one
convolution once the BatchNormalizations are folded, at zero inference cost.

**This port implements the train-time multi-branch form only.** There is no
``reparameterize()`` / branch-fusion path anywhere under ``layers/fastvit/`` or
``layers/mobile_one_block.py``, so a model built here always runs every branch,
and the *latency* half of the paper's claim is not realized. That matches how
the MobileCLIP2 reference weights are shipped and evaluated (always with
``inference_mode=False``); it does mean this tower is a faithful *functional*
transcription, not a fast one.

The tower is a standalone :class:`keras.Model` and is usable on its own. It is
also the vision branch of MobileCLIP2 specifically —
``models/mobile_clip/mobile_clip_v2.py`` imports :class:`FastVitImageEncoder`
from here, while the deliberately non-faithful ``mobile_clip_v1.py`` does not.
The one place that dual role shows through is the head ``Dense``.

**Architecture**::

    image (B, H, W, 3)
        |
    stem: 3 x MobileOneBlock          (k3/s2 dense, k3/s2 depthwise, k1/s1)  -> /4
        |
    stage_0 .. stage_{N-1}            FastVitStage
        |                             (downsample? RepCPE? depth x token mixer)
    final_conv: MobileOneBlock        k3, depthwise, SE, -> embed_dims[-1] * cls_ratio
        |
    GlobalAveragePooling2D
        |
    Dropout(head_dropout_rate)
        |
    Dense(projection_dim)             <- THIS IS THE CLIP IMAGE PROJECTION
        |
    (B, projection_dim)

**The head ``Dense`` is the CLIP image projection, not a classifier.** It is
named ``projection_dim`` rather than ``num_classes`` for exactly this reason.
All four MobileCLIP / MobileCLIP2 fastvit configs set ``"timm_proj": null`` with
``"timm_pool": "avg"``. In open_clip's ``TimmModel`` a non-attention pool asserts
that the trunk itself does the projecting and instantiates the trunk with
``num_classes=embed_dim``; the timm ``ClassifierHead``'s linear layer therefore
*is* the image-side projection into the joint embedding space. There is no
separate projection layer to add, and adding one would be a second, unfaithful
projection. ``timm_drop`` is ``0.0`` in all four configs, so the head dropout is
inert at the reference settings. Passing ``projection_dim=None`` skips it and
returns the pooled ``embed_dims[-1] * cls_ratio`` features, which is useful for
backbone reuse and wrong for CLIP.

**The stochastic-depth schedule is GLOBAL, then sliced.** The reference computes
one linear ramp across ``sum(layers)`` blocks — every block of every stage — and
hands each stage its contiguous slice. Computing a fresh ``0 -> drop_path_rate``
ramp per stage would be a different function (stage 1 of a ``(2, 12, 24, 4)``
model must start where stage 0 ended, not at zero) and produces an
identically-shaped, identically-parameterized, subtly-wrong model. See
:func:`_stagewise_drop_path_rates`.

Two more asymmetries in the reference are reproduced rather than tidied away.
Squeeze-and-Excitation appears at two different reduction ratios in the same
network — ``1/16`` inside ``final_conv`` (timm's ``SqueezeExcite`` default, never
overridden there) against ``0.25`` at ``ReparamLargeKernelConv``'s call site.
And the stages are stored as a FLAT list of :class:`FastVitStage`: a nested
``List[List[Layer]]`` restores fresh kernels on a ``.keras`` round trip while the
layer count, the variable paths and the parameter total all still match, so the
damage is invisible to every check except a value comparison.

The variant table's provenance is uneven and the comment above it says so in
detail — ``mci3``/``mci4`` are cross-checked against a committed reference file
by a real oracle, while ``mci0``/``mci1``/``mci2`` come from a timm fetch with no
local oracle, since timm is not installed here. ``get_config`` therefore stores
the resolved architecture explicitly rather than only the variant name, so a
checkpoint keeps describing the network it was trained with even if a table row
is later corrected.

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
# NOTE: `keras.layers` is deliberately NOT imported under the bare name `layers`
# — this module has a constructor parameter and a variant field called `layers`
# (the reference's name for the per-stage depths), and `keras.Model` also owns a
# `.layers` property. Sub-layers are always spelled `keras.layers.X`.
from keras import initializers, regularizers, activations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.mobile_one_block import MobileOneBlock
from dl_techniques.layers.fastvit import FastVitStage
from dl_techniques.layers.fastvit.reference import (
    REFERENCE_NORM_EPSILON,
    REFERENCE_PADDING_MODE,
)

# ---------------------------------------------------------------------
# reference constants
# ---------------------------------------------------------------------

#: Width multiplier applied to the last stage's channel count by ``final_conv``.
#: timm's ``FastVit`` default and the value every MCi variant uses.
_REFERENCE_CLS_RATIO = 2.0

#: Squeeze-and-Excitation bottleneck ratio inside ``final_conv``. timm builds it
#: as ``SqueezeExcite(out_chs, rd_divisor=1)`` and never overrides ``rd_ratio``,
#: whose ``SqueezeExcite`` default is ``1/16``. This is DELIBERATELY different
#: from the ``0.25`` that ``ReparamLargeKernelConv`` passes at ITS call site —
#: the reference really does use two different ratios in the same network.
_FINAL_CONV_SE_REDUCTION_RATIO = 1.0 / 16.0

#: timm's ``SqueezeExcite`` uses biased 1x1 convolutions.
_FINAL_CONV_SE_USE_BIAS = True

#: The reference applies SE BEFORE the activation
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

# DECISION plan-2026-08-13T183738-24486492/D-004
# PROVENANCE OF THIS TABLE — read before changing a single number.
#
# DO NOT "fix" an mci0/mci1/mci2 row by reasoning from the mci3/mci4 rows, and
# do not edit the committed reference files to make a test pass. The two groups
# have DIFFERENT provenance and different levels of evidence; conflating them is
# the deviation X-3 this block exists to keep visible. See decisions.md D-004.
#
# (i)   `mci3` and `mci4` are transcribed from the USER-SUPPLIED
#       `mobileclip2.py`, which defines `fastvit_mci3` and `fastvit_mci4` and
#       nothing else. They are the only two rows with a local provenance. That
#       file is COMMITTED VERBATIM at `research/mobileclip2_reference/`, and
#       `tests/test_models/test_fastvit/
#       test_model.py::test_mci3_mci4_match_supplied_source` PARSES it
#       (with `ast` — it is PyTorch/timm code and cannot be imported here) and
#       cross-checks these two rows field by field. That is a real oracle.
#
# (ii)  `mci0`, `mci1` and `mci2` are transcribed from TIMM UPSTREAM
#       (`timm/models/fastvit.py`, fetched 2026-08-13), NOT from the supplied
#       files. The supplied source does not define them at all.
#
# (iii) `timm` is NOT INSTALLED in this environment, so THERE IS NO LOCAL ORACLE
#       for (ii) — constraint H-7. The tests in this repo can only check that the
#       table matches a SECOND hand transcription of the same fetch; they cannot
#       check it against timm itself. Anyone changing an mci0/mci1/mci2 row must
#       re-derive it from timm upstream and say so, not reason from the mci3/mci4
#       rows (which differ structurally: 5 stages, no SE, LayerNorm, mlp_ratio 4).
#
# Fields common to every row and therefore NOT tabulated: cls_ratio=2.0,
# down_patch_size=7, down_stride=2, repmixer_kernel_size=3,
# layer_scale_init_value=1e-5, head_dim=32, activation=GELU.
#
# `pos_embs` entries are the RepCPE spatial shape (7, 7) or None. A stage gets a
# RepConditionalPosEnc iff its entry is not None.
MCI_VARIANTS: Dict[str, Dict[str, Any]] = {
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

    Args:
        variant: A key of :data:`MCI_VARIANTS` (``'mci0'`` ... ``'mci4'``). The
            ``'fastvit_'`` prefix used by timm's model names is accepted and
            stripped.

    Returns:
        A shallow copy of the variant's configuration dictionary.

    Raises:
        ValueError: If ``variant`` is not a known variant name.
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
    """Split ONE global linear stochastic-depth ramp into per-stage slices.

    The ramp is computed across ``sum(depths)`` blocks — the whole network — and
    then cut at the cumulative stage boundaries, so stage ``i`` starts where stage
    ``i - 1`` ended. This reproduces timm's
    ``calculate_drop_path_rates(drop_path_rate, layers, stagewise=True)``.

    Args:
        depths: Number of blocks in each stage.
        drop_path_rate: Maximum (last block of the last stage) drop probability.

    Returns:
        One list of floats per stage; concatenating them in order reproduces
        ``linear_drop_path_rates(sum(depths), drop_path_rate)`` exactly.
    """
    flat = linear_drop_path_rates(int(sum(depths)), float(drop_path_rate))
    slices: List[List[float]] = []
    cursor = 0
    for depth in depths:
        slices.append(flat[cursor:cursor + int(depth)])
        cursor += int(depth)
    return slices


@keras.saving.register_keras_serializable()
class FastVitImageEncoder(keras.Model):
    """MobileCLIP2's FastViT (MCi) image tower.

    A convolutional stem at stride 4, then ``N`` :class:`FastVitStage` stages
    (the shallow ones mixing tokens with the convolutional RepMixer, the deepest
    one or two with global self-attention), a wide depthwise ``final_conv`` with
    squeeze-and-excitation, and a pooled projection head.

    **The head ``Dense`` IS the CLIP image projection.** MobileCLIP's open_clip
    configs use ``timm_pool="avg"`` with ``timm_proj=null``, which makes the
    trunk's own classifier linear the projection into the joint image-text
    embedding space. Do not add a second projection on top of this model; pass
    ``projection_dim`` and use its output directly as the image embedding. Set
    ``projection_dim=None`` to get the pooled ``embed_dims[-1] * cls_ratio``
    features instead (useful for dense/backbone reuse, NOT for CLIP).

    Args:
        variant: Optional key of :data:`MCI_VARIANTS` (``'mci0'`` ... ``'mci4'``,
            with an optional ``'fastvit_'`` prefix). When given, every
            architecture field left as ``None`` is filled from that variant's
            row. When ``None``, all seven per-stage tuples must be supplied
            explicitly.
        layers: Blocks per stage, e.g. ``(2, 6, 10, 2)``.
        embed_dims: Output channels per stage.
        mlp_ratios: ConvMlp expansion ratio per stage.
        se_downsamples: Whether each stage's downsample uses squeeze-and-excitation.
        downsamples: Whether each stage begins with a downsample. Stage 0 is
            ``False`` in every MCi variant — the stem has already done the /4.
        pos_embs: Per stage, the RepCPE depthwise kernel shape, or ``None`` for
            no positional encoding in that stage.
        token_mixers: ``'repmixer'`` or ``'attention'`` per stage.
        stem_use_scale_branch: Whether the three stem ``MobileOneBlock``s keep
            their 1x1 scale branch. ``False`` for mci3/mci4.
        norm_layer: Normalization key for the attention stages' pre-norm, either
            ``'batch_norm'`` or ``'layer_norm'``. Ignored by RepMixer stages,
            which have no ``norm_layer`` parameter in the reference.
        lkc_use_act: Whether each downsample's large-kernel convolution applies
            its activation.
        input_shape: Image shape ``(H, W, C)``. Defaults to ``(256, 256, 3)``,
            MobileCLIP's fastvit input resolution.
        projection_dim: Width of the CLIP image projection. ``None`` skips the
            projection and returns the pooled features. Defaults to 512.
        cls_ratio: ``final_conv`` widens the last stage by this factor. Defaults
            to 2.0 (the reference value).
        drop_path_rate: Maximum stochastic-depth rate of the SINGLE global linear
            ramp spanning every block of every stage. Defaults to 0.0.
        dropout_rate: Dropout inside every block's ConvMlp. Defaults to 0.0.
        head_dropout_rate: Dropout between the pooling and the projection.
            Defaults to 0.0 (``timm_drop`` is 0.0 in all four MobileCLIP configs).
        layer_scale_init_value: LayerScale gamma initialization in every block, or
            ``None`` to omit LayerScale. Defaults to ``1e-5``.
        head_dim: Per-head width of the attention token mixer. Defaults to 32.
        down_patch_size: Downsample large-kernel size. Defaults to 7.
        down_stride: Downsample stride. Defaults to 2.
        repmixer_kernel_size: RepMixer depthwise kernel size. Defaults to 3.
        activation: Activation used throughout. Defaults to ``'gelu'``.
        kernel_initializer: Initializer for every convolution / projection kernel.
            Defaults to ``'he_normal'``.
        kernel_regularizer: Optional regularizer applied to every kernel.
        **kwargs: Forwarded to :class:`keras.Model`.

    Raises:
        ValueError: If ``variant`` is unknown; if any per-stage tuple is missing
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

    # `MCI_VARIANTS` is this package's only variant table, so the canonical
    # `MODEL_VARIANTS` spelling is exposed as a class-level ALIAS to the same
    # dict -- not a copy -- per models/CLAUDE.md. Tooling that resolves a
    # variant registry via getattr(cls, 'MODEL_VARIANTS') got AttributeError
    # here until 2026-08-19, while CLAUDE.md asserted fastvit carried it.
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
        #: ONE global ramp, cut at the stage boundaries. Never a per-stage ramp.
        self.drop_path_rates: List[List[float]] = _stagewise_drop_path_rates(
            self.layers_per_stage, self.drop_path_rate
        )

        # ---- CREATE all sub-layers in __init__ (unbuilt) ----------------
        # timm's `convolutional_stem`: dense k3/s2, DEPTHWISE k3/s2
        # (`group_size=1` means depthwise in timm's `num_groups` mapping), then a
        # pointwise k1/s1. Net stride /4. `use_scale_branch` is per-variant: the
        # supplied mobileclip2.py monkey-patches the stem specifically to turn it
        # off for mci3/mci4.
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

        # FLAT list. A nested List[List[Layer]] restores FRESH kernels on a
        # `.keras` round trip while the layer count, the variable paths and the
        # parameter total all still match (measured repo-wide).
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

        # `final_conv` is DEPTHWISE (group_size=1) and widens by cls_ratio, so
        # its Conv2D has groups=embed_dims[-1] and filters=final_features.
        # Its SE ratio is 1/16 (timm's SqueezeExcite default, unoverridden here)
        # with BIASED convolutions, applied BEFORE the activation. That is NOT
        # the 0.25 used by ReparamLargeKernelConv — the reference uses two
        # different SE ratios in the same network on purpose.
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
        # THE CLIP IMAGE PROJECTION. See the module docstring.
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

        Raises:
            ValueError: If the per-stage tuples disagree in length, if any size is
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

        Args:
            input_shape: ``(B, H, W, C)``.

        Returns:
            ``(B, ceil(H / 4), ceil(W / 4), embed_dims[0])``.
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

        Args:
            input_shape: ``(B, H, W, C)`` of the image entering the stem.

        Returns:
            One shape tuple per stage, in order.
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
        the next stage must be built on the PREVIOUS stage's output shape, never
        on ``input_shape``.

        Args:
            input_shape: ``(B, H, W, C)``. When it carries no spatial information
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

        Args:
            inputs: Image tensor ``(B, H, W, C)``.
            training: Keras training flag. Pass ``False`` EXPLICITLY for a
                deterministic forward — ``training=None`` runs the stochastic
                path of every stochastic-depth branch and uses batch statistics
                in every BatchNormalization.

        Returns:
            ``(B, projection_dim)``, the CLIP image embedding (un-normalized), or
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

        Args:
            input_shape: ``(B, H, W, C)``.

        Returns:
            ``(B, projection_dim)`` or ``(B, final_features)``.
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

        Returns:
            A dictionary containing every constructor parameter, with the
            architecture stored EXPLICITLY (not only as a variant name) so a
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

        Args:
            config: A dictionary produced by :meth:`get_config`.

        Returns:
            A new :class:`FastVitImageEncoder`.
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

        Args:
            variant: A key of :data:`MCI_VARIANTS` (``'mci0'`` ... ``'mci4'``),
                optionally prefixed ``'fastvit_'``.
            input_shape: Image shape ``(H, W, C)``.
            projection_dim: CLIP image-projection width, or ``None``.
            **kwargs: Forwarded to the constructor.

        Returns:
            The configured image tower.

        Raises:
            ValueError: If ``variant`` is not recognized.
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

    Args:
        variant: ``'mci0'`` ... ``'mci4'`` (an optional ``'fastvit_'`` prefix is
            accepted). Defaults to ``'mci0'``.
        input_shape: Image shape ``(H, W, C)``. Defaults to ``(256, 256, 3)``.
        projection_dim: Width of the CLIP image projection (the head ``Dense``),
            or ``None`` to return pooled features. Defaults to 512.
        **overrides: Any :class:`FastVitImageEncoder` constructor keyword, e.g.
            ``drop_path_rate``, ``head_dropout_rate``, ``kernel_regularizer``.

    Returns:
        The configured image tower.

    Raises:
        ValueError: If ``variant`` is not recognized.
    """
    return FastVitImageEncoder(
        variant=variant,
        input_shape=input_shape,
        projection_dim=projection_dim,
        **overrides,
    )

# ---------------------------------------------------------------------
