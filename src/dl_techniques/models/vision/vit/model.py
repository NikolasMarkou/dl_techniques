"""
The Vision Transformer, which treats an image as a short sequence of patch
tokens and hands it to a plain transformer encoder.

This model embodies the principle that the convolutional priors -- locality,
translation equivariance, a hierarchy of scales -- are useful shortcuts rather
than requirements, and that with enough data a general architecture can learn
whatever spatial structure the task actually needs. A ConvNet builds those
priors into its weight sharing, which is what makes it sample-efficient on
small datasets and what caps it on large ones. ViT removes them entirely: the
image is cut into non-overlapping `patch_size` squares, each is flattened and
linearly projected to `embed_dim`, and from that point the network has no
notion of two dimensions at all. Every layer sees a set of tokens, and any
geometry it uses it must learn.

Two consequences follow directly. Because the patch grid is discarded, spatial
position must be re-injected explicitly, which is what the learned positional
embedding does -- remove it and the model becomes permutation-invariant over
patches, seeing an image and its shuffled version identically. And because
self-attention is global from the first layer, receptive field is not something
that grows with depth; a token at one corner can attend to the opposite corner
immediately, which is precisely the long-range interaction a ConvNet needs many
downsampling stages to reach.

The sequence carries a prepended learnable CLS token, a position with no image
content whose only job is to accumulate a whole-image summary through
attention. That gives the classification head a single vector to read without
imposing any pooling rule on the patch tokens. Pooled feature extraction is
also available and is where the code does something non-obvious: `mean` and
`max` pooling exclude position 0, because averaging the CLS token into the
patch statistics mixes a summary vector into the thing it is summarizing. Only
`cls` pooling reads position 0, and it reads it alone.

Cost is quadratic in the number of patches, which is `(H/P) x (W/P)`, so patch
size is the architecture's central efficiency knob -- halving it quadruples the
sequence and roughly sixteen-times the attention cost. Five scales span tiny
(192d, 3 heads, 12 layers) through huge (1280d, 16 heads, 32 layers). Block
internals -- attention type, FFN type, normalization type and position -- are
supplied through the `dl_techniques` factories rather than hard-coded, so a
variant can be swapped in without forking the file. The scale table
(`ViT.MODEL_VARIANTS`) reproduces the published widths, depths and head counts;
the BLOCK defaults do not, in one named respect.

References:
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Touvron et al., 2021. Training data-efficient image transformers &
      distillation through attention. (https://arxiv.org/abs/2012.12877)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
"""


import os
import keras
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

PoolingMode = Literal['cls', 'mean', 'max']
VitScale = Literal['pico', 'tiny', 'small', 'base', 'large', 'huge']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']

# ---------------------------------------------------------------------

# DECISION plan-2026-08-23T091307-9a110062/D-503
#: Kernel initializer for every layer, matching the ViT convention this port
#: follows: HuggingFace's ``ViTConfig.initializer_range`` defaults to ``0.02``
#: and is applied as ``TruncatedNormal(std=0.02)`` to every weight matrix:
#:   https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/configuration_vit.py
#: Do NOT revert to ``"he_normal"`` (what this parameter used to be):
#: ``he_normal`` is ``VarianceScaling(scale=2.0, mode='fan_in')``, i.e. a
#: FAN-DEPENDENT scale, so it disagrees with the reference by a different factor
#: in every layer rather than by a constant. TRAINING-ONLY -- an initializer is
#: overwritten by any weight load, so no checkpoint changes meaning.
#:
#: A config DICT, not an ``Initializer`` instance: a seedless instance bakes its
#: seed at construction and REPLAYS the identical draw (MEASURED: two calls of
#: one instance at the same shape differ by exactly 0.0), so an instance used as
#: a default argument -- evaluated once at import -- would hand every model in
#: the process the same weights. Same hazard as D-072 / D-481.
#: See decisions.md D-503.
REFERENCE_KERNEL_INITIALIZER: Dict[str, Any] = {
    "class_name": "TruncatedNormal",
    "config": {"stddev": 0.02},
}

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ViT(keras.Model):
    """Vision Transformer: an image as a short sequence of patch tokens.

    Cuts the image into non-overlapping ``patch_size`` squares, linearly
    projects each to ``embed_dim``, prepends a learnable CLS token, adds a
    learned positional embedding and runs a plain transformer encoder. After
    the patch projection the model has NO notion of two dimensions: the
    positional embedding is the only source of geometry, and removing it would
    make the model permutation-invariant over patches. Attention is global from
    layer one, so receptive field does not grow with depth. Block internals --
    attention type, FFN type, normalization type and position -- come from the
    ``dl_techniques`` factories rather than being hard-coded.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, H, W, C]                  │
        │  H % patch_h == 0, W % patch_w == 0  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  PatchEmbedding2D                    │
        │  Conv P×P /P → [B, N, D]             │
        │  N = (H/Pₕ)·(W/Pw)                   │
        │  LINEAR: the ViT stem has no         │
        │  activation (see the D-022 anchor)   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  prepend CLS token                   │
        │  ONE (1, 1, D) weight, zero-init,    │
        │  broadcast over the batch            │
        │  → [B, N+1, D]                       │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  + learned positional embedding      │
        │  → Dropout(pos_dropout_rate)         │
        │  WITHOUT this the model is           │
        │  PERMUTATION-INVARIANT over patches  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  TransformerLayer × num_layers       │
        │    MHA → FFN, normalization_position │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  final norm over the WHOLE sequence  │
        └───────────────┬──────────────────────┘
                        │
            ┌───────────┴────────────┐
            ▼                        ▼
        include_top=True        include_top=False
            │                        │
        ┌───────────────┐   ┌────────────────────────────┐
        │ x_norm[:, 0]  │   │ pooling='cls'  → x[:, 0]   │
        │ [Dropout]     │   │ pooling='mean' → mean over │
        │ Dense(classes)│   │ pooling='max'  → max  over │
        │               │   │   positions 1.. ONLY       │
        │ → [B, classes]│   │ pooling=None → [B,N+1,D]   │
        └───────────────┘   └────────────────────────────┘

    **CLS exclusion in mean/max pooling (deliberate):**

    .. code-block:: text

        sequence:   [ CLS | p₀ | p₁ | p₂ | ... | p_{N-1} ]
                       ▲     └──────────────────────────┘
                       │            patch tokens
                       │
              pooling='cls' reads THIS alone

              pooling='mean'/'max' read the patch tokens ONLY
              (SequencePooling(exclude_positions=[0]))

        averaging CLS into the patch statistics would mix a
        SUMMARY vector into the thing it summarizes. The
        siblings vit_siglip / vit_hmlp include it; here the
        divergence is EXPLICIT in exclude_positions, not silent.

    **Normalization position (default diverges from the paper):**

    .. code-block:: text

        'post' (DEFAULT here)          'pre' (published ViT)

        x ──┬─► MHA ─┐                 x ──┬─► Norm ─► MHA ─┐
            │        ▼                     │                ▼
            └─────► (+) ─► Norm            └──────────────► (+)
                                                            │
        x ──┬─► FFN ─┐                 x ──┬─► Norm ─► FFN ─┐
            │        ▼                     │                ▼
            └─────► (+) ─► Norm            └──────────────► (+)

        Dosovitskiy et al. 2020 use PRE. The default is 'post'
        and is NOT flipped, because every vit checkpoint and
        training script in this repo was fitted under it and a
        flipped default would silently rebuild only the models
        that stored no value. Pass normalization_position='pre'
        for the paper. See the D-047 anchor in __init__.

    **Scales:**

    .. code-block:: text

        scale     embed_dim   heads   layers   mlp_ratio
        pico         192        3        6        4.0
        tiny         192        3       12        4.0
        small        384        6       12        4.0
        base         768       12       12        4.0
        large       1024       16       24        4.0
        huge        1280       16       32        4.0

        variant keys for from_variant: "vit_pico" .. "vit_huge"

        cost is QUADRATIC in N = (H/P)·(W/P), so patch_size is
        the central efficiency knob: halving P quadruples the
        sequence and ~16×s the attention cost.

    :param input_shape: Input image shape ``(height, width, channels)``. All
        dimensions must be positive and the spatial dims must be divisible by
        the corresponding patch dimensions. Defaults to ``(224, 224, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param num_classes: Number of output classes. Must be positive. Only used
        when ``include_top=True``. Defaults to 1000.
    :type num_classes: int
    :param scale: Model scale, one of ``'pico'``, ``'tiny'``, ``'small'``,
        ``'base'``, ``'large'``, ``'huge'``; fixes ``embed_dim``, ``num_heads``,
        ``num_layers`` and ``mlp_ratio``. Defaults to ``'base'``.
    :type scale: VitScale
    :param patch_size: Patch size; an int gives square patches. The image
        dimensions must be divisible by it. Defaults to 16.
    :type patch_size: Union[int, Tuple[int, int]]
    :param include_top: Whether to include the classification head. When False
        the model is a feature extractor. Defaults to True.
    :type include_top: bool
    :param pooling: Pooling strategy, used ONLY when ``include_top=False``:
        ``'cls'`` reads position 0 alone, ``'mean'`` and ``'max'`` pool the
        patch tokens EXCLUDING position 0, and ``None`` returns the full
        normalized sequence. Defaults to None.
    :type pooling: Optional[PoolingMode]
    :param dropout_rate: General dropout rate, applied in the transformer
        layers and before the classification head. Must be in ``[0, 1]``.
        Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate for attention weights. Must be
        in ``[0, 1]``. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param pos_dropout_rate: Dropout rate after the positional embedding. Must
        be in ``[0, 1]``. Defaults to 0.0.
    :type pos_dropout_rate: float
    :param kernel_initializer: Weight initializer for every layer. ``None``
        resolves to :data:`REFERENCE_KERNEL_INITIALIZER`,
        ``TruncatedNormal(stddev=0.02)``, the ViT convention. Each sub-layer
        receives its own ``clone_initializer`` copy; see the D-540 anchor.
    :type kernel_initializer: Optional[Union[str, Dict[str, Any], keras.initializers.Initializer]]
    :param kernel_regularizer: Weight regularizer for every layer. Defaults to
        None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: Bias initializer for every layer. Defaults to
        ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Bias regularizer for every layer. Defaults to None.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param normalization_type: Normalization identifier passed to
        ``create_normalization_layer``: ``'layer_norm'`` (default),
        ``'rms_norm'``, ``'band_rms'``, ``'dynamic_tanh'`` and so on.
    :type normalization_type: NormalizationType
    :param normalization_kwargs: Optional kwargs forwarded to the final norm's
        factory call and to every transformer layer as both
        ``attention_norm_args`` and ``ffn_norm_args``. ``None`` resolves to
        ``{}``, which makes every factory call byte-identical to the
        pre-plumbing version and keeps existing checkpoints bit-exact.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param normalization_position: ``'post'`` (default) or ``'pre'``. The
        default is NOT the published ViT configuration; pass ``'pre'`` to
        reproduce the paper or to load a checkpoint ported from its release.
        See the module docstring for why the default is deliberately not
        flipped.
    :type normalization_position: Literal['pre', 'post']
    :param ffn_type: Feed-forward network identifier passed to the factory:
        ``'mlp'`` (default), ``'swiglu'``, ``'geglu'`` and so on.
    :type ffn_type: FFNType
    :param activation: Activation for the FFN. Defaults to ``'gelu'``. Note it
        is NOT forwarded to the patch projection, which is linear.
    :type activation: Union[str, callable]
    :param use_layer_scale: Whether transformer layers apply layer scale.
        Defaults to False.
    :type use_layer_scale: bool
    :param layer_scale_init_value: Initial layer-scale value. Defaults to 1e-5.
    :type layer_scale_init_value: float
    :param name: Model name; auto-generated as ``vision_transformer_<scale>``
        when None.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If ``input_shape`` is not a positive 3-tuple, if
        ``patch_size`` is invalid or does not divide the image dimensions, if
        ``num_classes`` is not positive, if ``scale`` or ``pooling`` is
        unrecognized, or if any dropout rate leaves ``[0, 1]``.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``, with height and
        width divisible by the corresponding patch dimensions.

    Output shape:
        - ``include_top=True``: ``(batch_size, num_classes)``, LOGITS with no
          softmax applied.
        - ``include_top=False, pooling='cls'|'mean'|'max'``:
          ``(batch_size, embed_dim)``.
        - ``include_top=False, pooling=None``:
          ``(batch_size, num_patches + 1, embed_dim)``.

    :ivar embed_dim: Embedding dimension, fixed by ``scale``.
    :vartype embed_dim: int
    :ivar num_heads: Attention head count, fixed by ``scale``.
    :vartype num_heads: int
    :ivar num_layers: Transformer depth, fixed by ``scale``.
    :vartype num_layers: int
    :ivar num_patches: ``(H // patch_h) * (W // patch_w)``.
    :vartype num_patches: int
    :ivar max_seq_len: ``num_patches + 1``, counting the CLS token.
    :vartype max_seq_len: int
    :ivar cls_token: Learnable ``(1, 1, embed_dim)`` token, created in ``build``.
    :vartype cls_token: keras.Variable
    :ivar transformer_layers: The encoder stack.
    :vartype transformer_layers: list[TransformerLayer]

    Example:
        .. code-block:: python

            # Standard ViT-Base for ImageNet classification
            model = ViT(
                input_shape=(224, 224, 3),
                num_classes=1000,
                scale='base'
            )

            # Feature extractor with CLS token
            feature_model = ViT(
                input_shape=(224, 224, 3),
                scale='base',
                include_top=False,
                pooling='cls'
            )

            # Published architecture plus modern components
            custom_model = ViT(
                input_shape=(384, 384, 3),
                num_classes=10,
                scale='small',
                patch_size=16,
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=0.1,
                attention_dropout_rate=0.1
            )

            model.compile(
                optimizer='adamw',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

    Note:
        The head emits LOGITS, so compile with ``from_logits=True``.
    """

    # Scale configurations: [embed_dim, num_heads, num_layers, mlp_ratio]
    SCALE_CONFIGS: Dict[str, Tuple[int, int, int, float]] = {
        "pico": (192, 3, 6, 4.0),  # ViT-Pico
        "tiny": (192, 3, 12, 4.0),  # ViT-Tiny
        "small": (384, 6, 12, 4.0),  # ViT-Small
        "base": (768, 12, 12, 4.0),  # ViT-Base
        "large": (1024, 16, 24, 4.0),  # ViT-Large
        "huge": (1280, 16, 32, 4.0),  # ViT-Huge
    }

    # ResNet-template variant registry. Thin wrapper over SCALE_CONFIGS so
    # callers can use `ViT.from_variant("vit_pico", ...)` exactly like
    # `ResNet.from_variant("resnet50", ...)`.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "vit_pico":  {"scale": "pico"},
        "vit_tiny":  {"scale": "tiny"},
        "vit_small": {"scale": "small"},
        "vit_base":  {"scale": "base"},
        "vit_large": {"scale": "large"},
        "vit_huge":  {"scale": "huge"},
    }

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            num_classes: int = 1000,
            scale: VitScale = "base",
            patch_size: Union[int, Tuple[int, int]] = 16,
            include_top: bool = True,
            pooling: Optional[PoolingMode] = None,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            pos_dropout_rate: float = 0.0,
            kernel_initializer: Optional[Union[str, Dict[str, Any], keras.initializers.Initializer]] = None,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: NormalizationType = "layer_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            normalization_position: Literal['pre', 'post'] = "post",
            ffn_type: FFNType = "mlp",
            activation: Union[str, callable] = "gelu",
            use_layer_scale: bool = False,
            layer_scale_init_value: float = 1e-5,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the Vision Transformer and create every sub-layer.

        :param input_shape: Input image shape ``(height, width, channels)``.
        :type input_shape: Tuple[int, int, int]
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param scale: Model scale key into :attr:`SCALE_CONFIGS`.
        :type scale: VitScale
        :param patch_size: Square or ``(h, w)`` patch size.
        :type patch_size: Union[int, Tuple[int, int]]
        :param include_top: Whether to build the classification head.
        :type include_top: bool
        :param pooling: Feature-extraction pooling strategy.
        :type pooling: Optional[PoolingMode]
        :param dropout_rate: General dropout rate.
        :type dropout_rate: float
        :param attention_dropout_rate: Attention-weight dropout rate.
        :type attention_dropout_rate: float
        :param pos_dropout_rate: Post-positional-embedding dropout rate.
        :type pos_dropout_rate: float
        :param kernel_initializer: Weight initializer; ``None`` resolves to the
            ViT reference.
        :type kernel_initializer: Optional[Union[str, Dict[str, Any], keras.initializers.Initializer]]
        :param kernel_regularizer: Weight regularizer.
        :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
        :param bias_initializer: Bias initializer.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param bias_regularizer: Bias regularizer.
        :type bias_regularizer: Optional[keras.regularizers.Regularizer]
        :param normalization_type: Normalization identifier.
        :type normalization_type: NormalizationType
        :param normalization_kwargs: Kwargs forwarded to every norm factory call.
        :type normalization_kwargs: Optional[Dict[str, Any]]
        :param normalization_position: ``'pre'`` or ``'post'``.
        :type normalization_position: Literal['pre', 'post']
        :param ffn_type: Feed-forward network identifier.
        :type ffn_type: FFNType
        :param activation: FFN activation.
        :type activation: Union[str, callable]
        :param use_layer_scale: Whether layers apply layer scale.
        :type use_layer_scale: bool
        :param layer_scale_init_value: Initial layer-scale value.
        :type layer_scale_init_value: float
        :param name: Model name; auto-generated when None.
        :type name: Optional[str]
        :param kwargs: Additional keyword arguments for ``keras.Model``.
        :raises ValueError: If any configuration value is invalid.
        """
        # Auto-generate name if not provided
        if name is None:
            name = f"vision_transformer_{scale}"

        super().__init__(name=name, **kwargs)

        # Validate and store input_shape
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
            raise ValueError(f"input_shape must be a 3-tuple (height, width, channels), got {input_shape}")

        img_h, img_w, img_c = input_shape
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")

        # Validate and normalize patch_size
        if isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {patch_size}")
            patch_h = patch_w = patch_size
        else:
            if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
                raise ValueError(f"patch_size must be int or tuple of 2 ints, got {patch_size}")
            patch_h, patch_w = patch_size
            if patch_h <= 0 or patch_w <= 0:
                raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")

        # Validate divisibility for patch extraction
        if img_h % patch_h != 0:
            raise ValueError(f"Image height ({img_h}) must be divisible by patch height ({patch_h})")
        if img_w % patch_w != 0:
            raise ValueError(f"Image width ({img_w}) must be divisible by patch width ({patch_w})")

        # Validate other parameters
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        if scale not in self.SCALE_CONFIGS:
            raise ValueError(f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}")

        if pooling not in [None, "cls", "mean", "max"]:
            raise ValueError(f"Unsupported pooling: {pooling}. Choose from [None, 'cls', 'mean', 'max']")

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(f"pos_dropout_rate must be between 0 and 1, got {pos_dropout_rate}")

        # Store ALL configuration parameters for serialization
        self.input_shape_config = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.scale = str(scale)
        self.patch_size = (patch_h, patch_w)
        self.include_top = bool(include_top)
        self.pooling = pooling
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.pos_dropout_rate = float(pos_dropout_rate)
        # A module-level dict is a MUTABLE DEFAULT: bound once at def time and shared
        # by every caller. Resolved from a None sentinel instead, which also keeps
        # `initializers.get` producing a FRESH instance per model.
        if kernel_initializer is None:
            kernel_initializer = REFERENCE_KERNEL_INITIALIZER
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = bias_regularizer
        self.normalization_type = str(normalization_type)
        # DECISION plan_2026-05-18_6776f8ba/D-003
        # Optional, additive `normalization_kwargs` plumbed into both the final
        # `self.norm` factory call and every TransformerLayer (as
        # `attention_norm_args` + `ffn_norm_args`). Default `None` -> `{}` -> the
        # factory call is byte-identical to the pre-plumbing version, preserving
        # bit-exactness for ALL existing serialized ViT checkpoints. This is the
        # multi-flag-plumbing pattern from LESSONS L72; the ViT path here mirrors
        # the ResNet path in `dl_techniques/models/vision/resnet/model.py` for the
        # rms_variants_train Phase 3 `param_matched` mode (use_scale=False).
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}
        # DECISION plan-2026-08-18T140459-7991552f/D-047
        # `normalization_position` DEFAULTS TO `"post"` while this file's
        # `References` block cites Dosovitskiy et al. 2020, which is PRE-LN.
        # The mismatch is real and is documented in the module docstring; the
        # resolution taken was to correct the documentation, NOT to flip the
        # default (see the plan's Assumption A2).
        # WHAT NOT TO DO: do not "fix" this by changing the default here or in
        # `create_vit` to `"pre"` to match the sibling `vit_hmlp/model.py`.
        # `normalization_position` selects between two different functions
        # (`TransformerLayer.call`'s pre-LN and post-LN branches), and every
        # `vit` checkpoint, training script and result in this repository was
        # fitted under `"post"`. A saved model records the value in its config,
        # so a flipped default silently rebuilds only the models that DID NOT
        # store one -- i.e. every fresh construction -- while old artifacts keep
        # loading as post-LN, which is the worst possible split. The flip
        # remains a one-line change plus one test edit
        # (`tests/test_models/test_vit/test_model.py:46`) if it is ever taken
        # deliberately. Related: `ViT.call`'s final `self.norm` over the whole
        # sequence is the pre-LN idiom and is redundant (not wrong) under the
        # post-LN default. See decisions.md D-047.
        self.normalization_position = str(normalization_position)
        self.ffn_type = str(ffn_type)
        self.activation = activation
        self.use_layer_scale = bool(use_layer_scale)
        self.layer_scale_init_value = float(layer_scale_init_value)

        # Get model configuration from scale
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate derived parameters
        self.intermediate_size = int(self.embed_dim * self.mlp_ratio)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate derived parameters
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Using factories for consistent component creation

        # Patch embedding using factory
        # DECISION plan-2026-08-18T140459-7991552f/D-022
        # The four initializer/regularizer knobs are forwarded; `activation` is
        # deliberately NOT, although 'patch_2d' declares one and `self.activation`
        # exists. ViT's `activation` is the FFN activation (see this class's
        # docstring and the TransformerLayer construction below); forwarding its
        # default 'gelu' into the patch projection would make the stem
        # nonlinear, which no ViT is. It is a name collision, not a dropped
        # knob, and is recorded as one in
        # tests/test_models/test_package_api_contract.py::_NAME_COLLISIONS.
        # See D-022 in plans/plan-2026-08-18T140459-7991552f/decisions.md.
        # DECISION plan-2026-08-23T091307-9a110062/D-540
        # Every sub-layer gets its OWN `clone_initializer(...)` copy. Do NOT
        # "simplify" this back to passing `self.kernel_initializer` directly:
        # a single seedless initializer INSTANCE replays its draw, so every
        # same-shape kernel it reaches is bit-identical. MEASURED at HEAD
        # before this change, on a seeded ViT-tiny/12L: all 12
        # `transformer_layer_*/attention/cross_attention/qkv/kernel` were
        # pairwise `max|delta| = 0.0` (66 pairs), likewise the 12 `.../proj/kernel`
        # (66 pairs) -- i.e. the 12 blocks started as 12 COPIES of one block.
        # `seed=` is not the discriminator; instance identity is. See
        # decisions.md D-540 and initializers/clone.py.
        self.patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="patch_embed"
        )

        # Positional embedding using factory
        self.pos_embed = create_embedding_layer(
            'positional_learned',
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout_rate=self.pos_dropout_rate,
            name="pos_embed"
        )

        # Transformer layers using existing TransformerLayer
        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type="multi_head",
                normalization_type=self.normalization_type,
                attention_norm_args=dict(self.normalization_kwargs),
                ffn_norm_args=dict(self.normalization_kwargs),
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_layer_scale=self.use_layer_scale,
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Final normalization using factory
        self.norm = create_normalization_layer(
            self.normalization_type,
            name="norm",
            **self.normalization_kwargs,
        )

        # Classification components (if include_top)
        self.head_dropout = None
        self.head = None
        if self.include_top:
            if self.dropout_rate > 0.0:
                self.head_dropout = keras.layers.Dropout(self.dropout_rate, name="head_dropout")

            self.head = keras.layers.Dense(
                self.num_classes,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="head"
            )

        # Feature-extraction pooling via the shared SequencePooling layer.
        # DECISION plan-2026-07-15T144225-5b25d9f1/D-001: pool via SequencePooling(exclude_positions=[0])
        # — preserves vit's CLS-excluded mean/max (byte-identical); the CLS-inclusion divergence vs
        # vit_siglip/vit_hmlp is now EXPLICIT in exclude_positions, not silent. Do NOT drop
        # exclude_positions (would start averaging the CLS token).
        self.pool = None
        if self.pooling == "cls":
            self.pool = SequencePooling(strategy="cls", name="seq_pool")
        elif self.pooling in ("mean", "max"):
            self.pool = SequencePooling(
                strategy=self.pooling, exclude_positions=[0], name="seq_pool"
            )

        # CLS token weight (created in build())
        self.cls_token = None

        logger.info(f"Created VisionTransformer-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")
        logger.info(
            f"Image shape: {self.input_shape_config}, Patch size: {self.patch_size}, Num patches: {self.num_patches}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the CLS token and explicitly build every sub-layer.

        Each sub-layer is built in computational order rather than left to a
        lazy first call, so the weight tree materializes on ``.keras`` reload.

        :param input_shape: Input shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not 4D.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        # Create CLS token weight
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )

        # Build all sub-layers in computational order
        # Patch embedding
        dummy_input_shape = (None,) + self.input_shape_config
        self.patch_embed.build(dummy_input_shape)

        # Positional embedding
        pos_input_shape = (None, self.max_seq_len, self.embed_dim)
        self.pos_embed.build(pos_input_shape)

        # Transformer layers
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)

        # Final normalization
        self.norm.build(pos_input_shape)

        # Classification head components
        if self.include_top:
            head_input_shape = (None, self.embed_dim)
            if self.head_dropout is not None:
                self.head_dropout.build(head_input_shape)
            self.head.build(head_input_shape)

        # Feature-extraction pooling
        if self.pool is not None:
            self.pool.build(pos_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the Vision Transformer.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Model output. ``(batch, num_classes)`` logits with
            ``include_top=True``; otherwise the pooled features
            ``(batch, embed_dim)`` or, when ``pooling is None``, the full
            normalized sequence ``(batch, max_seq_len, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        # 1. Convert image to a sequence of patch embeddings
        x = self.patch_embed(inputs, training=training)

        # 2. Prepend the CLS token
        batch_size = keras.ops.shape(x)[0]
        cls_tokens = keras.ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = keras.ops.concatenate([cls_tokens, x], axis=1)

        # 3. Add learned positional embeddings
        x = self.pos_embed(x, training=training)

        # 4. Process through the Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # 5. Apply final normalization to the entire sequence for architectural consistency
        x_norm = self.norm(x, training=training)

        # 6. Handle the output based on the model's configuration
        if self.include_top:
            # --- Classification Head Logic ---
            # Extract the CLS token from the *normalized* sequence.
            cls_token = x_norm[:, 0, :]

            # Pass through the final classification head
            if self.head_dropout is not None:
                cls_token = self.head_dropout(cls_token, training=training)

            return self.head(cls_token)
        else:
            # --- Feature Extraction Logic ---
            # cls / mean / max all route through SequencePooling. For mean/max the
            # pool is built with exclude_positions=[0], so the CLS token is dropped
            # before pooling (byte-identical to the previous GAP1D/GMP1D over x[:, 1:]).
            if self.pool is not None:
                return self.pool(x_norm)
            # pooling is None -> return the full, normalized sequence
            return x_norm

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape for a given input shape.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape, depending on ``include_top`` and ``pooling``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not 4D.
        """
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        batch_size = input_shape[0]

        if self.include_top:
            return (batch_size, self.num_classes)
        else:
            if self.pooling in ["cls", "mean", "max"]:
                return (batch_size, self.embed_dim)
            else:
                return (batch_size, self.max_seq_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        Includes EVERY ``__init__`` parameter, which is what makes the
        round trip lossless.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes,
            "scale": self.scale,
            "patch_size": self.patch_size,
            "include_top": self.include_top,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "pos_dropout_rate": self.pos_dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            # DECISION plan-2026-08-23T091307-9a110062/D-400
            # D-205 inlined this as a 17-line `isinstance(str)`-guarded
            # expression at three sites. It is now the single shared pair in
            # `dl_techniques.utils.activation_serialization`, which ~50 other
            # classes in this tree also call. Do NOT re-inline it: the string
            # passthrough is load-bearing (`keras.activations.serialize` REJECTS
            # a bare string, and many callers store a dl_techniques
            # activation-factory key such as 'mish' that is not a Keras
            # activation at all), and a second copy of that rule is exactly the
            # kind of hand-maintained lockstep this centralisation removes.
            "activation": serialize_activation(self.activation),
            "use_layer_scale": self.use_layer_scale,
            "layer_scale_init_value": self.layer_scale_init_value,
        })
        return config

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
            custom_objects: Optional[Dict[str, Any]] = None
    ) -> "ViT":
        """Recreate a model from its serialized configuration.

        ViT had NO ``from_config`` at all before D-205. That was survivable only
        because every other serialized key (``initializers``, ``regularizers``)
        is resolved by ``keras.initializers.get`` / ``keras.regularizers.get``
        inside ``__init__``. ``activation`` is not: it is handed straight to
        ``TransformerLayer``, so a serialized callable has to be turned back
        into one here.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :param custom_objects: Optional mapping of names to custom callables,
            used to resolve an activation that is not registered with
            ``keras.saving.register_keras_serializable``.
        :type custom_objects: Optional[Dict[str, Any]]
        :return: A new ``ViT`` instance.
        :rtype: ViT
        """
        config = dict(config)
        activation = config.get("activation")
        if activation is not None and not isinstance(activation, str):
            config["activation"] = deserialize_activation(
                activation, custom_objects=custom_objects
            )
        return cls(**config)

    def get_feature_extractor(self) -> "ViT":
        """Return a feature-extractor twin of this model.

        Copies every configuration value, sets ``include_top=False`` and
        ``pooling='cls'``. Note this constructs a NEW, randomly-initialized
        model; it does not transfer this instance's weights.

        :return: New ViT instance configured for CLS-token feature extraction.
        :rtype: ViT
        :raises ValueError: If the model was not properly initialized.
        """
        if not hasattr(self, 'input_shape_config') or not self.input_shape_config:
            raise ValueError("Model must be properly initialized before creating feature extractor")

        return ViT(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            pos_dropout_rate=self.pos_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            normalization_type=self.normalization_type,
            normalization_kwargs=dict(self.normalization_kwargs),
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            activation=self.activation,
            use_layer_scale=self.use_layer_scale,
            layer_scale_init_value=self.layer_scale_init_value,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Log a detailed architecture summary, beyond ``keras.Model.summary``."""
        logger.info("Vision Transformer Model Summary")
        logger.info(f"Scale: {self.scale}")
        logger.info(f"Input Shape: {self.input_shape_config}")
        logger.info(f"Patch Size: {self.patch_size}")
        logger.info(f"Number of Patches: {self.num_patches}")
        logger.info(f"Sequence Length: {self.max_seq_len}")
        logger.info(f"Embedding Dimension: {self.embed_dim}")
        logger.info(f"Number of Heads: {self.num_heads}")
        logger.info(f"Number of Layers: {self.num_layers}")
        logger.info(f"MLP Ratio: {self.mlp_ratio}")
        logger.info(f"Intermediate Size: {self.intermediate_size}")
        logger.info(f"Dropout Rate: {self.dropout_rate}")
        logger.info(f"Attention Dropout Rate: {self.attention_dropout_rate}")
        logger.info(f"Positional Dropout Rate: {self.pos_dropout_rate}")
        logger.info(f"Normalization Type: {self.normalization_type}")
        logger.info(f"Normalization Position: {self.normalization_position}")
        logger.info(f"FFN Type: {self.ffn_type}")
        logger.info(f"Activation: {self.activation}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self.built:
            logger.info(f"Total Parameters: {self.count_params():,}")

        # Additional architecture information
        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        logger.info(f"Patches per dimension: {img_h // patch_h} x {img_w // patch_w}")

    # -----------------------------------------------------------------
    # Pretrained-weights API (resnet-template parity)
    # -----------------------------------------------------------------

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True,
    ) -> None:
        """Load pretrained weights from a local ``.keras`` file.

        # DECISION plan_2026-05-12_f2d29729/D-001
        Diverges from the resnet template's ``self.load_weights(..., by_name=True)``
        path because Keras 3.8 raises ``ValueError("Invalid keyword arguments:
        {'by_name': True}")`` when ``by_name=True`` is passed to
        ``.keras`` files (LESSONS L71). Route through
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`
        which does a full-model load + layer-by-layer ``set_weights``.

        :param weights_path: Path to the ``.keras`` weights file.
        :type weights_path: str
        :param skip_mismatch: Whether to skip layers with mismatched shapes,
            forwarded as the inverse of ``strict``. The ``head`` prefix is
            skipped unconditionally.
        :type skip_mismatch: bool
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If the file is not ``.keras`` or strict transfer
            fails.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Build the model if not already built (probe forward pass).
        if not self.built:
            dummy_input = keras.random.normal((1,) + tuple(self.input_shape_config))
            self(dummy_input, training=False)

        logger.info(f"Loading pretrained weights from {weights_path}")
        report = load_weights_from_checkpoint(
            target=self,
            ckpt_path=weights_path,
            skip_prefixes=("head",),
            strict=(not skip_mismatch),
        )
        logger.info(f"Weight transfer report: {report}")

    # `_download_weights` raises instead of falling back to random init. A
    # vestigial `PRETRAINED_WEIGHTS` table of placeholder URLs on a non-existent
    # host used to sit on the class; it was never read, because this method has
    # raised since D-002, but it advertised
    # downloads that could never happen. Do NOT reinstate it, and do NOT widen
    # the except clause in `from_variant` into a warn-and-return branch -- that
    # combination is what makes `pretrained=True` silently return an untrained
    # model. Pass a local path via `pretrained="/path/to/file.keras"` instead.
    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None,
    ) -> str:
        """Download pretrained weights for ``variant``; always raises.

        # DECISION plan_2026-05-12_f2d29729/D-002
        Diverges from the resnet template which actually attempts to download
        from a placeholder URL. No public ViT checkpoints in the
        ``dl_techniques`` weight format are distributed at this time
        (LESSONS L53). Calling this method always raises
        :class:`NotImplementedError` so failures are loud, not silent
        404s that produce HTML payloads masquerading as ``.keras`` files.

        :param variant: Model variant (e.g. ``"vit_base"``). Unused; reserved
            for signature parity with :class:`ResNet`.
        :type variant: str
        :param dataset: Pretraining dataset name. Unused; same reason.
        :type dataset: str
        :param cache_dir: Cache directory. Unused; same reason.
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        del variant, dataset, cache_dir  # silence unused-arg lint
        raise NotImplementedError(
            "No public ViT checkpoints are distributed for this implementation. "
            "Pass a local .keras path to `pretrained=` to load custom weights, "
            "e.g. `ViT.from_variant('vit_base', pretrained='/path/to/weights.keras')`."
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, int, int]] = None,
            pretrained: Union[bool, str] = False,
            weights_dataset: str = "imagenet",
            weights_input_shape: Optional[Tuple[int, int, int]] = None,
            cache_dir: Optional[str] = None,
            **kwargs: Any,
    ) -> "ViT":
        """Create a ViT model from a predefined variant.

        A user who sets ``pretrained=True`` without providing a local checkpoint
        path gets a clear :class:`NotImplementedError` straight out of
        ``_download_weights``; see the ``# DECISION ... D-122`` note at that call
        site for why no ``except`` clause guards it.

        :param variant: One of the keys in :attr:`MODEL_VARIANTS`
            (``"vit_pico".."vit_huge"``).
        :type variant: str
        :param num_classes: Number of output classes for the classification
            head.
        :type num_classes: int
        :param input_shape: Input image shape ``(H, W, C)``. ``None`` resolves
            to ``(224, 224, 3)``.
        :type input_shape: Optional[Tuple[int, int, int]]
        :param pretrained: Either a boolean (``True`` attempts a download and
            will raise) or a string path to a local ``.keras`` checkpoint.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset name for pretrained-weight routing.
        :type weights_dataset: str
        :param weights_input_shape: Optional input shape the pretrained weights
            were trained at. A difference from ``input_shape`` enables mismatch
            skipping during transfer, since the positional embeddings differ.
        :type weights_input_shape: Optional[Tuple[int, int, int]]
        :param cache_dir: Optional cache directory for downloads.
        :type cache_dir: Optional[str]
        :param kwargs: Forwarded to the :class:`ViT` constructor.
        :type kwargs: Any
        :return: Configured (and possibly weight-loaded) :class:`ViT` instance.
        :rtype: ViT
        :raises ValueError: If ``variant`` is not recognized.
        :raises NotImplementedError: If ``pretrained`` is the boolean ``True``;
            no public ViT weights are hosted.
        :raises FileNotFoundError: If ``pretrained`` is a path that is missing.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        variant_cfg = cls.MODEL_VARIANTS[variant]
        scale = variant_cfg["scale"]

        if input_shape is None:
            input_shape = (224, 224, 3)

        logger.info(f"Creating ViT model variant '{variant}' (scale='{scale}')")

        # Resolve pretrained source --------------------------------------
        load_weights_path: Optional[str] = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                # DECISION plan-2026-08-19T163559-499b6f0e/D-122
                # `_download_weights` is called BARE on purpose. It raises
                # `NotImplementedError` unconditionally, and `NotImplementedError`
                # inherits from `RuntimeError`, NOT from `OSError`/`ValueError`
                # (MEASURED: `issubclass(NotImplementedError, OSError)` is False,
                # `issubclass(NotImplementedError, ValueError)` is False; `IOError`
                # IS `OSError` on Python 3, so the deleted 3-tuple was really a
                # 2-tuple). The former
                # `except (IOError, OSError, ValueError): warn + random init`
                # therefore could not fire from ANY reachable state -- a
                # warn-and-continue branch whose own `# DECISION` comment argued it
                # was "narrow on purpose" while closing nothing. Do NOT reinstate it
                # and do NOT broaden the tuple to `RuntimeError`/`Exception` "to make
                # it work": broadening is the ONE edit that turns this into a silent
                # fallback that hands the caller a randomly initialised model when
                # `pretrained=True` (LESSONS L53), and the repo-wide guard added at
                # step 6 fires on exactly that shape. See decisions.md D-122.
                load_weights_path = cls._download_weights(
                    variant=variant,
                    dataset=weights_dataset,
                    cache_dir=cache_dir,
                )

            # Decide whether to enable shape-mismatch skipping.
            include_top = kwargs.get("include_top", True)
            if include_top and num_classes != 1000:
                skip_mismatch = True
                logger.info(
                    f"num_classes ({num_classes}) != pretrained head (1000); "
                    f"head weights will be skipped."
                )
            if weights_input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Pretrained weights trained on {weights_input_shape} but "
                    f"model uses {input_shape}; positional embeddings may differ."
                )
                skip_mismatch = True

        # Build model ----------------------------------------------------
        model = cls(
            input_shape=input_shape,
            num_classes=num_classes,
            scale=scale,
            **kwargs,
        )

        # Load weights if a path was resolved ----------------------------
        if load_weights_path:
            model.load_pretrained_weights(
                weights_path=load_weights_path,
                skip_mismatch=skip_mismatch,
            )

        return model


# ---------------------------------------------------------------------
# Factory Functions for Convenient Model Creation
# ---------------------------------------------------------------------


def create_vit(
        variant: str = "vit_base",
        num_classes: int = 1000,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        weights_input_shape: Optional[Tuple[int, int, int]] = None,
        cache_dir: Optional[str] = None,
        include_top: bool = True,
        pooling: Optional[PoolingMode] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        pos_dropout_rate: float = 0.0,
        kernel_initializer: Optional[Union[str, Dict[str, Any], keras.initializers.Initializer]] = None,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        normalization_type: NormalizationType = "layer_norm",
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        normalization_position: Literal['pre', 'post'] = "post",
        ffn_type: FFNType = "mlp",
        activation: Union[str, callable] = "gelu",
        **kwargs: Any
) -> ViT:
    """Create a Vision Transformer with the specified configuration.

    ResNet-template factory: ``variant`` is the canonical key and routes
    through :meth:`ViT.from_variant`, so callers get the pretrained-weights
    handling and the variant registry for free. This function validates
    NOTHING itself; see the D-078 anchor in the body.

    :param variant: Variant key, ``"vit_pico".."vit_huge"``. Defaults to
        ``"vit_base"``.
    :type variant: str
    :param num_classes: Number of output classes. Only used when
        ``include_top=True``.
    :type num_classes: int
    :param input_shape: Input image shape ``(height, width, channels)``; the
        spatial dims must be divisible by ``patch_size``.
    :type input_shape: Tuple[int, int, int]
    :param patch_size: Patch size; an int gives square patches.
    :type patch_size: Union[int, Tuple[int, int]]
    :param pretrained: Local ``.keras`` path, or the boolean ``True`` which
        raises ``NotImplementedError``.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset name for pretrained-weight routing.
    :type weights_dataset: str
    :param weights_input_shape: Input shape the pretrained weights were trained
        at.
    :type weights_input_shape: Optional[Tuple[int, int, int]]
    :param cache_dir: Optional cache directory for downloads.
    :type cache_dir: Optional[str]
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param pooling: Feature-extraction pooling when ``include_top=False``:
        ``'cls'``, ``'mean'``, ``'max'`` or None.
    :type pooling: Optional[PoolingMode]
    :param dropout_rate: General dropout rate.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-weight dropout rate.
    :type attention_dropout_rate: float
    :param pos_dropout_rate: Post-positional-embedding dropout rate.
    :type pos_dropout_rate: float
    :param kernel_initializer: Weight initializer; ``None`` resolves to
        :data:`REFERENCE_KERNEL_INITIALIZER`, ``TruncatedNormal(stddev=0.02)``.
    :type kernel_initializer: Optional[Union[str, Dict[str, Any], keras.initializers.Initializer]]
    :param kernel_regularizer: Weight regularizer.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: Bias initializer.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Bias regularizer.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param normalization_type: Normalization identifier.
    :type normalization_type: NormalizationType
    :param normalization_kwargs: Kwargs forwarded to every norm factory call.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param normalization_position: ``'post'`` (default) or ``'pre'``. The
        default is NOT the published ViT configuration; see the module
        docstring.
    :type normalization_position: Literal['pre', 'post']
    :param ffn_type: Feed-forward network identifier.
    :type ffn_type: FFNType
    :param activation: FFN activation.
    :type activation: Union[str, callable]
    :param kwargs: Additional arguments forwarded to the :class:`ViT`
        constructor.
    :type kwargs: Any
    :return: ViT model instance.
    :rtype: ViT
    :raises ValueError: Propagated from :class:`ViT`'s own validation.

    Example:
        .. code-block:: python

            # ViT-Base for ImageNet
            model = create_vit(
                variant='vit_base',
                input_shape=(224, 224, 3),
                num_classes=1000
            )

            # Feature extractor with modern components
            feature_model = create_vit(
                variant='vit_small',
                input_shape=(384, 384, 3),
                include_top=False,
                pooling='cls',
                normalization_type='rms_norm',
                ffn_type='swiglu'
            )
    """
    # DECISION plan-2026-08-19T163559-499b6f0e/D-078: this factory validates
    # NOTHING. It used to carry eight `raise ValueError` branches -- num_classes
    # sign, input_shape arity, input_shape sign, patch_size sign, patch_size
    # type, patch_size element sign, and the two divisibility checks -- and
    # `ViT.__init__` was MEASURED to raise its own `ValueError` for every one of
    # the eight, with a near-identical message. They were dead duplication and
    # the exact shape R-051 names. Do NOT re-add a check here "for a clearer
    # message": two copies of a rule drift, and the copy the caller hits is the
    # one that is NOT next to the code it constrains. Add it to `ViT.__init__`.
    #
    # `num_patches <= 0` was additionally UNREACHABLE: it followed the positive
    # -dimension and divisibility checks, so the product was already >= 1.
    #
    # Delegate to from_variant for unified pretrained-weights handling.
    model = ViT.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        patch_size=patch_size,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        pos_dropout_rate=pos_dropout_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        normalization_type=normalization_type,
        normalization_kwargs=normalization_kwargs,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        activation=activation,
        **kwargs,
    )

    # Reporting only -- derived AFTER construction, so an invalid `patch_size`
    # has already been rejected by `ViT.__init__` and never reaches this line.
    patch_h, patch_w = (
        (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    )
    img_h, img_w = input_shape[:2]
    num_patches = (img_h // patch_h) * (img_w // patch_w)
    if num_patches > 10000:  # Reasonable upper limit
        logger.warning(f"Large number of patches ({num_patches}) may cause memory issues")

    logger.info(f"ViT variant '{variant}' created successfully")
    logger.info(f"Configuration: {num_patches} patches ({img_h // patch_h}x{img_w // patch_w}), {num_classes} classes")
    return model

# ---------------------------------------------------------------------
