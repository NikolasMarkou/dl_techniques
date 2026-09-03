"""A BEiT image trunk with a masked-image-modeling head and a classification head.

BEiT adapts BERT's masked-token pre-training to images by borrowing a
vocabulary: a discrete variational autoencoder, trained separately and
frozen, maps every patch to one of 8192 code ids, and the model predicts
that id at each masked position instead of regressing raw pixels. Three
classes share one trunk: :class:`BeitModel` is the patch-embedding and
transformer-block encoder; :class:`BeitForMaskedImageModeling` adds a
projection over the codebook; :class:`BeitForImageClassification` adds a
pooling and classification head and emits logits (compile with
``from_logits=True``). Masked positions are substituted before the class
token is prepended and before any block runs, so BEiT processes the whole
sequence and never drops tokens, unlike MAE. Position information comes
from a learnable relative-position-bias table per block rather than an
absolute embedding, so ``use_absolute_position_embeddings`` defaults to
``False``.

The tokenizer is not part of this module: code ids arrive as tensors
alongside the patch mask, and the mask is applied through ``sample_weight``
rather than inside these models, so none of them define ``train_step`` or
``compute_loss``. ``layer_norm_eps`` defaults to ``1e-12`` (HF's
``BeitConfig``), tighter than a generic ViT's ``1e-6``. ``tiny`` and
``small`` variants are repo inventions for cheap tests; ``base`` and
``large`` reproduce the fetched HF configs verbatim.

References:
    - Bao et al., 2022. BEiT: BERT Pre-Training of Image Transformers. ICLR.
      (https://arxiv.org/abs/2106.08254)
    - Devlin et al., 2019. BERT: Pre-training of Deep Bidirectional Transformers
      for Language Understanding. (https://arxiv.org/abs/1810.04805)
    - Ramesh et al., 2021. Zero-Shot Text-to-Image Generation. (the DALL-E dVAE
      whose 8192-entry codebook supplies the visual tokens)
      (https://arxiv.org/abs/2102.12092)
    - Dosovitskiy et al., 2021. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Shaw et al., 2018. Self-Attention with Relative Position Representations.
      (https://arxiv.org/abs/1803.02155)
    - Touvron et al., 2021. Going Deeper with Image Transformers (CaiT).
      (LayerScale) (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - ``microsoft/beit-base-patch16-224`` / ``microsoft/beit-large-patch16-224``
      ``config.json`` — the hyperparameters reproduced in :data:`SCALE_CONFIGS`
      and the constructor defaults.
"""

# DECISION plan-2026-08-24T074054-247151fd/D-012: this module is Sphinx/reST
# throughout, a named exception to the repo's usual "match the file" rule.
# Do not convert it back to Google style. See decisions.md.

import keras
from keras.saving import serialize_keras_object, deserialize_keras_object
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.embedding.mask_token import MaskTokenApply
from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

BeitScale = Literal['tiny', 'small', 'base', 'large']

# load_weights_from_checkpoint matches layers by name, so the MIM model and
# the classifier must name their backbone identically.
BACKBONE_NAME = "beit_backbone"

# The DALL-E dVAE codebook size BEiT v1 predicts over (HF BeitConfig.vocab_size).
DEFAULT_VOCAB_SIZE = 8192

# SCALE_CONFIGS maps a scale to its architecture; MODEL_VARIANTS maps a public
# name to a scale; _resolve_scale accepts either spelling.
# DECISION plan-2026-08-24T074054-247151fd/D-009: keep these two tables
# separate rather than merged into one, per the repo-wide CLAUDE.md rule that
# SCALE_CONFIGS is not a stale spelling of MODEL_VARIANTS. See decisions.md.

# DECISION plan-2026-08-11T012340-f63796dc/D-003: layer_scale_init_value
# follows timm's split (0.1 base, 1e-5 large), not HF's uniform 0.1 config.
# Both ports of the official checkpoints disagree; this pick is pinned. See decisions.md.
#
# `tiny` and `small` are REPO INVENTIONS for cheap tests — no BEiT of either size
# exists in the paper, in HF, or in timm. `base` and `large` reproduce the fetched HF
# config.json verbatim.
SCALE_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'hidden_size': 192,
        'num_layers': 12,
        'num_heads': 3,
        'intermediate_size': 768,
        'layer_scale_init_value': 0.1,
    },
    'small': {
        'hidden_size': 384,
        'num_layers': 12,
        'num_heads': 6,
        'intermediate_size': 1536,
        'layer_scale_init_value': 0.1,
    },
    'base': {
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'intermediate_size': 3072,
        'layer_scale_init_value': 0.1,
    },
    'large': {
        'hidden_size': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'intermediate_size': 4096,
        'layer_scale_init_value': 1e-5,
    },
}

# Variant registry (house convention, mirrors ViT.MODEL_VARIANTS / ET's MODEL_VARIANTS).
MODEL_VARIANTS: Dict[str, Dict[str, str]] = {
    'beit_tiny': {'scale': 'tiny'},
    'beit_small': {'scale': 'small'},
    'beit_base': {'scale': 'base'},
    'beit_large': {'scale': 'large'},
}


def _resolve_scale(variant: str) -> str:
    """Accept either a scale key (``'base'``) or a variant key (``'beit_base'``).

    :param variant: A key of :data:`SCALE_CONFIGS` or of :data:`MODEL_VARIANTS`.
    :type variant: str
    :returns: The resolved :data:`SCALE_CONFIGS` key.
    :rtype: str
    :raises ValueError: If ``variant`` is neither spelling.
    """
    if variant in SCALE_CONFIGS:
        return variant
    if variant in MODEL_VARIANTS:
        return MODEL_VARIANTS[variant]['scale']
    raise ValueError(
        f"Unknown variant '{variant}'. Available: "
        f"{sorted(SCALE_CONFIGS)} or {sorted(MODEL_VARIANTS)}"
    )


def _as_pair(value: Union[int, Tuple[int, int]], name: str) -> Tuple[int, int]:
    """Coerce an int or 2-sequence to a validated positive integer pair.

    :param value: A scalar edge length, or an explicit ``(h, w)`` pair.
    :type value: Union[int, Tuple[int, int]]
    :param name: Field name, used only in the error messages.
    :type name: str
    :returns: The validated ``(h, w)`` pair.
    :rtype: Tuple[int, int]
    :raises ValueError: If ``value`` is neither an ``int`` nor a length-2 sequence,
        or if either component is non-positive.
    """
    if isinstance(value, int):
        pair = (int(value), int(value))
    else:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{name} must be an int or a 2-tuple, got {value!r}")
        pair = (int(value[0]), int(value[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} components must be positive, got {value!r}")
    return pair


def _split_inputs(inputs: Any, owner: str) -> Tuple[Any, Optional[Any]]:
    """Split ``image`` / ``(image, bool_mask)`` / ``{'images': ..., 'mask': ...}``.

    This is a TRACE-TIME structural choice (did the caller supply a second tensor?),
    not a runtime ``ops.where`` on tensor values: the MIM model always passes a mask,
    the classifier never does.

    :param inputs: The raw ``call`` argument, in any of the three accepted forms.
    :type inputs: Any
    :param owner: Calling class name, used only in the error messages.
    :type owner: str
    :returns: ``(image, bool_mask)``, where ``bool_mask`` is ``None`` when the caller
        supplied only an image.
    :rtype: Tuple[Any, Optional[Any]]
    :raises ValueError: If a dict has no ``'images'`` key, or a sequence is not of
        length 2.
    """
    if isinstance(inputs, dict):
        try:
            image = inputs['images']
        except KeyError:
            raise ValueError(
                f"{owner} received a dict without an 'images' key; got keys "
                f"{sorted(inputs)}"
            ) from None
        return image, inputs.get('mask', inputs.get('bool_masked_pos'))
    if isinstance(inputs, (tuple, list)):
        if len(inputs) != 2:
            raise ValueError(
                f"{owner} accepts either `image` or `(image, bool_mask)`; got a "
                f"sequence of length {len(inputs)}"
            )
        return inputs[0], inputs[1]
    return inputs, None


def _image_shape_of(input_shape: Any) -> Any:
    """Pull the image entry out of a possibly-nested ``input_shape``.

    The shape counterpart of :func:`_split_inputs`, for ``compute_output_shape``.

    :param input_shape: A bare image shape, a ``(image_shape, mask_shape)`` pair, or
        a dict carrying an ``'images'`` key.
    :type input_shape: Any
    :returns: The image shape, or ``None`` if a dict carried no ``'images'`` key.
    :rtype: Any
    """
    if isinstance(input_shape, dict):
        return input_shape.get('images')
    if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (tuple, list))
    ):
        return input_shape[0]
    return input_shape


# DECISION plan-2026-08-24T074054-247151fd/D-010: keep the "BeitModel" return
# annotation quoted; BeitModel is not yet defined at this point in the module. See decisions.md.


def _coerce_backbone(backbone: Any) -> "BeitModel":
    """Accept a live backbone or its serialized config dict (the ``from_config`` path).

    :param backbone: A :class:`BeitModel`, or the config dict a saved head carries.
    :type backbone: Any
    :returns: The live backbone.
    :rtype: BeitModel
    :raises TypeError: If ``backbone`` is neither a :class:`BeitModel` nor a config
        dict that deserializes to one.
    """
    if isinstance(backbone, BeitModel):
        return backbone
    if isinstance(backbone, dict):
        obj = deserialize_keras_object(backbone)
        if not isinstance(obj, BeitModel):
            raise TypeError(
                f"Deserialized backbone is a {type(obj).__name__}, expected BeitModel"
            )
        return obj
    raise TypeError(
        "backbone must be a BeitModel (or its serialized config dict), got "
        f"{type(backbone).__name__}"
    )


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.beit.model")
class BeitModel(keras.Model):
    """
    BEiT trunk: patch-embed -> [mask token] -> cls token -> N x BEiT block -> tokens.

    A single, separately-checkpointable image trunk that BOTH the masked-image-modeling
    model and the classifier compose under the same name, so a pre-trained encoder
    transfers into the classifier layer-for-layer.

    **Call signature** — ``backbone(image)``, ``backbone((image, bool_mask))``, or
    ``backbone({'images': image, 'mask': bool_mask})``. ``bool_mask`` is
    ``(B, N)`` boolean with ``True`` at positions whose patch embedding is replaced by
    the learnable mask token, BEFORE the cls token is prepended and before any
    transformer block — BEiT processes the full sequence and never drops tokens.

    **``MaskTokenApply`` is ALWAYS created and ALWAYS built**, even for the classifier
    that never calls it (authoring guide §9). That is what makes the two trunks
    weight-identical so the warm start is complete. Do not "optimize" it away.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐   ┌────────────────────┐
        │  Input image [B, H, W, C]            │   │ bool_mask [B, N]   │
        └───────────────┬──────────────────────┘   └─────────┬──────────┘
                        ▼                                    │ (optional)
        ┌──────────────────────────────────────┐             │
        │  patch_embed: Conv p×p stride p      │             │
        │    → [B, N, D],  N = (H/pₕ)·(W/p_w)  │             │
        └───────────────┬──────────────────────┘             │
                        ▼                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  mask_token: x[mask] ← learnable mask_token                  │
        │    ALWAYS created and built; applied only when a mask is     │
        │    passed. No token is ever dropped.                         │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  cls_token: prepend  → [B, N+1, D]   │
        │  [+ pos_embed, absolute, off by      │
        │     default]                         │
        │  [+ embed_dropout]                   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  × num_layers   BEiT block (PRE-norm)                        │
        │                                                              │
        │    x ──► LN ──► BeitAttention ──► g₁· ──► DropPath ──┐       │
        │    └────────────────────────────────────────────────►(+)     │
        │                                                       │      │
        │    x ──► LN ──► MLP(GELU) ──────► g₂· ──► DropPath ──┐│      │
        │    └────────────────────────────────────────────────►(+)     │
        │                                                              │
        │  relative position bias: ONE TABLE PER BLOCK                 │
        │    (use_shared_relative_position_bias=True raises)           │
        │  drop-path: linear ramp 0 → drop_path_rate                   │
        │  g₁, g₂: LayerScale, init layer_scale_init_value             │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  final_norm — created only when use_mean_pooling is False.   │
        │    When True the pooling head owns the norm instead (D-007), │
        │    and the trunk emits unnormalized tokens.                  │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, N+1, D] — cls at index 0 │
        └──────────────────────────────────────┘

    **Variants:**

    .. code-block:: text

        scale     hidden   layers   heads   ffn     layer_scale_init
        tiny        192      12       3      768    0.1     (repo invention)
        small       384      12       6     1536    0.1     (repo invention)
        base        768      12      12     3072    0.1     (HF config.json)
        large      1024      24      16     4096    1e-5    (timm's split, X-2)

        MODEL_VARIANTS: 'beit_tiny' | 'beit_small' | 'beit_base' | 'beit_large'
                        → the matching SCALE_CONFIGS key

    :param input_shape: Image shape ``(height, width, channels)``. Defaults to
        ``(224, 224, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param patch_size: Patch size; ``int`` for square patches or ``(h, w)``. Defaults
        to ``16``.
    :type patch_size: Union[int, Tuple[int, int]]
    :param scale: One of ``'tiny'``, ``'small'``, ``'base'``, ``'large'`` (see
        :data:`SCALE_CONFIGS`); a variant spelling such as ``'beit_base'`` is also
        accepted.
    :type scale: BeitScale
    :param hidden_size: Override the scale's model width ``D``. ``None`` -> from
        ``scale``.
    :type hidden_size: Optional[int]
    :param num_layers: Override the scale's block count. ``None`` -> from ``scale``.
    :type num_layers: Optional[int]
    :param num_heads: Override the scale's head count. ``None`` -> from ``scale``.
    :type num_heads: Optional[int]
    :param intermediate_size: Override the scale's FFN width. ``None`` -> from
        ``scale``.
    :type intermediate_size: Optional[int]
    :param layer_scale_init_value: Override the scale's LayerScale init. ``None`` ->
        from ``scale``. See :data:`SCALE_CONFIGS` for why the value is scale-dependent.
    :type layer_scale_init_value: Optional[float]
    :param layer_norm_eps: Epsilon at EVERY normalization site. Defaults to ``1e-12``
        (HF ``BeitConfig``), which is 6 orders of magnitude tighter than a generic
        ViT's ``1e-6`` — it is passed explicitly and never inherited from a
        constructor default.
    :type layer_norm_eps: float
    :param drop_path_rate: Maximum stochastic-depth rate. The per-block rates are the
        linear ramp ``0 -> drop_path_rate`` across ``num_layers``. Defaults to ``0.1``.
    :type drop_path_rate: float
    :param hidden_dropout_rate: Dropout after the embedding stage and on each block's
        FFN output. Defaults to ``0.0`` (HF ``BeitConfig``). Upstream the field is
        ``BeitConfig.hidden_dropout_prob``; it is spelled ``_rate`` here because every
        dropout rate in this repository is (D-130), and the value and meaning are
        unchanged.
    :type hidden_dropout_rate: float
    :param attention_probs_dropout_rate: Dropout on the attention probabilities.
        Defaults to ``0.0`` (HF ``BeitConfig``). Upstream:
        ``BeitConfig.attention_probs_dropout_prob``.
    :type attention_probs_dropout_rate: float
    :param use_absolute_position_embeddings: Add a learnable absolute position
        embedding over the ``N + 1`` token sequence. Defaults to ``False`` — BEiT uses
        RELATIVE position bias instead, and no shipped BEiT/BEiTv2 variant in HF or
        timm enables this.
    :type use_absolute_position_embeddings: bool
    :param use_relative_position_bias: Give every block's ``BeitAttention`` its own
        learnable relative-position-bias table. Defaults to ``True``.
    :type use_relative_position_bias: bool
    :param use_shared_relative_position_bias: One table shared by every block (BEiT's
        pre-training-only mode). Only ``False`` is supported; ``True`` raises.
    :type use_shared_relative_position_bias: bool
    :param use_mean_pooling: BEiT's classification convention — mean over the final
        patch tokens (cls excluded) with a SEPARATE LayerNorm on the pooled mean. It
        also controls whether this trunk applies its own final LayerNorm; see the note
        on :attr:`final_norm`. Defaults to ``True`` (HF ``BeitConfig``).
    :type use_mean_pooling: bool
    :param initializer_range: Stddev of the ``TruncatedNormal`` used for every
        projection kernel. Defaults to ``0.02`` (HF ``BeitConfig``).
    :type initializer_range: float
    :param name: Model name. Defaults to :data:`BACKBONE_NAME` — do not change it for
        a model that must participate in the MIM -> classifier warm start.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.
    :type kwargs: Any
    :raises ValueError: If any size is non-positive, ``hidden_size % num_heads != 0``,
        the image dims are not divisible by the patch dims, a dropout rate lies outside
        ``[0, 1]``, ``scale`` is unknown, or ``use_shared_relative_position_bias`` is
        ``True``.

    Input shape:
        - Unmasked: 4D tensor ``(batch, H, W, C)``.
        - Masked, tuple form: ``[(batch, H, W, C), (batch, N)]``, the second entry
          boolean.
        - Masked, dict form: ``{'images': (batch, H, W, C), 'mask': (batch, N)}``;
          the ``'mask'`` key is optional.

    Output shape:
        3D tensor ``(batch, N + 1, hidden_size)`` — the FULL sequence, cls token at
        index 0. There is only ONE output mode: this trunk never pools and never
        drops a token, so both the masked and the unmasked call return the same
        shape.

    Example:
        >>> # By variant name, the house entry point
        >>> backbone = BeitModel.from_variant('beit_base', (224, 224, 3), 16)
        >>>
        >>> # Deterministic forward: training=False, NOT training=None
        >>> images = keras.random.normal((2, 224, 224, 3))
        >>> tokens = backbone(images, training=False)   # (2, 197, 768)
        >>>
        >>> # Masked forward for BEiT pre-training
        >>> mask = keras.ops.zeros((2, 196), dtype='bool')
        >>> tokens = backbone((images, mask), training=False)
        >>>
        >>> # cls-token pooling instead of BEiT's mean pooling: the trunk then owns
        >>> # the final LayerNorm itself
        >>> backbone = BeitModel(scale='tiny', use_mean_pooling=False)

    Note:
        No pretrained BEiT weights are distributed with ``dl_techniques``. Train it
        and measure, or load a checkpoint you produced yourself; the numbers in the
        BEiT paper are not numbers about this code.

    Attributes:
        patch_embed: Strided convolution turning the image into ``N`` patch tokens.
        mask_token: ``MaskTokenApply``; always created and built, used only when a
            mask is passed.
        cls_token: ``ClassTokenPrepend``, producing the ``N + 1`` sequence.
        pos_embed: Absolute position embedding, or ``None`` when disabled.
        embed_dropout: The single embedding-stage dropout.
        encoder_layers: The ``num_layers`` blocks, stored FLAT.
        drop_path_rates: The per-block stochastic-depth ramp.
        final_norm: Trunk-level LayerNorm, or ``None`` when ``use_mean_pooling``.
        grid_size: The patch grid ``(Wh, Ww)``, which is ``BeitAttention``'s
            ``window_size``.
        num_patches: ``Wh * Ww``.
        seq_len: ``num_patches + 1``.
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            patch_size: Union[int, Tuple[int, int]] = 16,
            scale: BeitScale = 'base',
            hidden_size: Optional[int] = None,
            num_layers: Optional[int] = None,
            num_heads: Optional[int] = None,
            intermediate_size: Optional[int] = None,
            layer_scale_init_value: Optional[float] = None,
            layer_norm_eps: float = 1e-12,
            drop_path_rate: float = 0.1,
            hidden_dropout_rate: float = 0.0,
            attention_probs_dropout_rate: float = 0.0,
            use_absolute_position_embeddings: bool = False,
            use_relative_position_bias: bool = True,
            use_shared_relative_position_bias: bool = False,
            use_mean_pooling: bool = True,
            initializer_range: float = 0.02,
            name: Optional[str] = BACKBONE_NAME,
            **kwargs: Any,
    ) -> None:
        """Resolve the scale, store all configuration, validate, and create sub-layers.

        The five ``_build_*`` / ``_resolve_*`` helpers below run from here, once each,
        in the order written. See the class docstring for the parameter reference.
        """
        super().__init__(name=name, **kwargs)

        scale = _resolve_scale(str(scale))
        cfg = SCALE_CONFIGS[scale]

        # ----- store ALL configuration (serialization contract) -----
        self.input_shape_config = tuple(int(v) for v in input_shape) \
            if isinstance(input_shape, (tuple, list)) else input_shape
        self.patch_size = patch_size
        self.scale = scale
        self.hidden_size = int(hidden_size) if hidden_size is not None else cfg['hidden_size']
        self.num_layers = int(num_layers) if num_layers is not None else cfg['num_layers']
        self.num_heads = int(num_heads) if num_heads is not None else cfg['num_heads']
        self.intermediate_size = (
            int(intermediate_size) if intermediate_size is not None
            else cfg['intermediate_size']
        )
        self.layer_scale_init_value = (
            float(layer_scale_init_value) if layer_scale_init_value is not None
            else cfg['layer_scale_init_value']
        )
        self.layer_norm_eps = float(layer_norm_eps)
        self.drop_path_rate = float(drop_path_rate)
        self.hidden_dropout_rate = float(hidden_dropout_rate)
        self.attention_probs_dropout_rate = float(attention_probs_dropout_rate)
        self.use_absolute_position_embeddings = bool(use_absolute_position_embeddings)
        self.use_relative_position_bias = bool(use_relative_position_bias)
        self.use_shared_relative_position_bias = bool(use_shared_relative_position_bias)
        self.use_mean_pooling = bool(use_mean_pooling)
        self.initializer_range = float(initializer_range)

        self._validate_config()
        self._resolve_geometry()
        self._build_embeddings()
        self._build_tokens()
        self._build_encoder()
        self._build_final_norm()

        logger.info(
            f"Created BeitModel-{scale}: {self.hidden_size}d, {self.num_layers}L, "
            f"{self.num_heads}h, ffn={self.intermediate_size}, grid={self.grid_size}, "
            f"N={self.num_patches}, eps={self.layer_norm_eps}, "
            f"layer_scale={self.layer_scale_init_value}"
        )

    # -----------------------------------------------------------------
    # Construction helpers. Each is called EXACTLY ONCE, from `__init__`, in the
    # order written there -- that is what a constructor decomposition is, and it is
    # why they take no arguments and return nothing: they read and write `self`.
    #
    # WHAT NOT TO DO: do not rename these after the ORDER they run in
    # (`_build_step1`, `_build_phase2`). A temporal name tells the reader when the
    # method is called, which `__init__` already shows, and hides what it owns, which
    # is the only thing a reader coming from a weight name or a traceback needs.

    def _resolve_geometry(self) -> None:
        """Derive the patch grid, the patch count and the token-sequence length.

        Runs AFTER `_validate_config`, which is what guarantees the divisions below
        are exact: the image dims are checked divisible by the patch dims there.
        """
        img_h, img_w, _ = self.input_shape_config
        patch_h, patch_w = _as_pair(self.patch_size, "patch_size")
        # The PATCH GRID, which is `BeitAttention`'s `window_size` — NOT the image size
        # and NOT a scalar edge length.
        self.grid_size: Tuple[int, int] = (img_h // patch_h, img_w // patch_w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.seq_len = self.num_patches + 1  # + cls
        # Kept so `_build_embeddings` can pass the RESOLVED pair without re-running
        # `_as_pair`; private, because `self.patch_size` stays the serialization field.
        self._patch_size_pair: Tuple[int, int] = (patch_h, patch_w)

        # DECISION plan-2026-08-24T074054-247151fd/D-017: keep one shared
        # TruncatedNormal instance; N per-layer instances draw from the global
        # RNG in a different pattern and silently change every weight. See decisions.md.
        self._kernel_init = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range
        )

    def _build_embeddings(self) -> None:
        """Create the embedding stage: patch projection, optional absolute position, dropout.

        Sub-layers are CREATED here and left UNBUILT; `build()` builds every one of them
        explicitly from stored config.
        """
        self.patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=self._patch_size_pair,
            embed_dim=self.hidden_size,
            kernel_initializer=self._kernel_init,
            name="patch_embed",
        )

        # Absolute position embedding is created ONLY when enabled. Unlike the mask
        # token, this is not a warm-start hazard: it is static backbone config, so the
        # MIM trunk and the classifier trunk agree on it by construction. Creating it
        # unconditionally would add a `(1, N+1, D)` weight that BEiT's own default
        # config (`use_absolute_position_embeddings=False`, every shipped HF/timm
        # variant) never reads — dead weight in every checkpoint.
        self.pos_embed = None
        if self.use_absolute_position_embeddings:
            self.pos_embed = create_embedding_layer(
                'positional_learned',
                max_seq_len=self.seq_len,
                dim=self.hidden_size,
                # The registry key is `dropout_rate`; `dropout=` would be SILENTLY
                # dropped. It is held at 0.0 on purpose — `embed_dropout` below is the
                # single embedding-stage dropout, and routing it here too would double
                # it.
                dropout_rate=0.0,
                name="pos_embed",
            )

        self.embed_dropout = keras.layers.Dropout(
            self.hidden_dropout_rate, name="embed_dropout"
        )

    def _build_tokens(self) -> None:
        """Create the two token-manipulating layers: the mask token and the cls token."""
        # Guide §9: ALWAYS CREATE, CONDITIONALLY USE. The classifier never calls this,
        # but it MUST own the weight or its trunk stops matching the MIM trunk and the
        # warm start silently transfers a different set of layers. This is the
        # DELIBERATE exception to "build only what call() runs" — see build().
        self.mask_token = MaskTokenApply(name="mask_token")

        self.cls_token = ClassTokenPrepend(name="cls_token")

    def _build_encoder(self) -> None:
        """Create the stochastic-depth ramp and the `num_layers` transformer blocks."""
        # The stochastic-depth LINEAR RAMP is a MODEL-level responsibility:
        # `TransformerLayer` holds one float per instance and has no intra-layer
        # schedule. `linear_drop_path_rates` is the repo's single definition of this
        # schedule (utils/drop_path.py) and also handles num_layers == 1 without
        # dividing by zero.
        self.drop_path_rates: List[float] = linear_drop_path_rates(
            self.num_layers, self.drop_path_rate
        )

        # Stored FLAT. A `List[List[Layer]]` restores FRESH kernels on a `.keras` round
        # trip while the layer count, the weight paths and the parameter total ALL
        # still match — a measured framework trap, invisible to every structural
        # assertion.
        self.encoder_layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            self.encoder_layers.append(
                TransformerLayer(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_size=self.intermediate_size,
                    attention_type='beit',
                    # The PATCH GRID (Wh, Ww). `BeitAttention` expects exactly
                    # Wh*Ww + 1 tokens (the +1 being the cls token).
                    window_size=self.grid_size,
                    attention_args={
                        'use_relative_position_bias': self.use_relative_position_bias,
                    },
                    normalization_type='layer_norm',
                    normalization_position='pre',
                    # 1e-12 is passed EXPLICITLY at every norm site; the factory's own
                    # default is 1e-6 and inheriting it would be a silent architecture
                    # change.
                    attention_norm_args={'epsilon': self.layer_norm_eps},
                    ffn_norm_args={'epsilon': self.layer_norm_eps},
                    ffn_type='mlp',
                    activation='gelu',
                    dropout_rate=self.hidden_dropout_rate,
                    attention_dropout_rate=self.attention_probs_dropout_rate,
                    use_layer_scale=True,
                    layer_scale_init_value=self.layer_scale_init_value,
                    use_stochastic_depth=True,
                    stochastic_depth_rate=self.drop_path_rates[i],
                    kernel_initializer=self._kernel_init,
                    name=f"encoder_layer_{i}",
                )
            )

    def _build_final_norm(self) -> None:
        """Create the trunk's final LayerNorm — only on the `use_mean_pooling is False` fork."""
        # DECISION plan-2026-08-11T012340-f63796dc/D-007: create final_norm only
        # when use_mean_pooling is False; always applying it double-norms the
        # pooled output in front of both heads. See decisions.md.
        self.final_norm = None
        if not self.use_mean_pooling:
            self.final_norm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_eps, name="final_norm"
            )

    # -----------------------------------------------------------------

    def _validate_config(self) -> None:
        """Reject every invalid configuration at CONSTRUCTION time, not at first call.

        :raises ValueError: On any invalid geometry, size, rate or unsupported mode;
            see the class docstring's ``:raises:`` for the full list.
        """
        if (
                not isinstance(self.input_shape_config, tuple)
                or len(self.input_shape_config) != 3
        ):
            raise ValueError(
                "input_shape must be a 3-tuple (height, width, channels), got "
                f"{self.input_shape_config!r}"
            )
        img_h, img_w, img_c = self.input_shape_config
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(
                f"All input_shape dims must be positive, got {self.input_shape_config}"
            )

        patch_h, patch_w = _as_pair(self.patch_size, "patch_size")
        if img_h % patch_h != 0:
            raise ValueError(
                f"Image height ({img_h}) must be divisible by patch height ({patch_h})"
            )
        if img_w % patch_w != 0:
            raise ValueError(
                f"Image width ({img_w}) must be divisible by patch width ({patch_w})"
            )

        for size_name, size in (
                ("hidden_size", self.hidden_size),
                ("num_layers", self.num_layers),
                ("num_heads", self.num_heads),
                ("intermediate_size", self.intermediate_size),
        ):
            if size <= 0:
                raise ValueError(f"{size_name} must be positive, got {size}")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads "
                f"({self.num_heads})"
            )

        for rate_name, rate in (
                ("drop_path_rate", self.drop_path_rate),
                ("hidden_dropout_rate", self.hidden_dropout_rate),
                ("attention_probs_dropout_rate", self.attention_probs_dropout_rate),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{rate_name} must be in [0, 1], got {rate}")

        if self.layer_norm_eps <= 0.0:
            raise ValueError(
                f"layer_norm_eps must be positive, got {self.layer_norm_eps}"
            )
        if self.initializer_range <= 0.0:
            raise ValueError(
                f"initializer_range must be positive, got {self.initializer_range}"
            )

        if self.use_shared_relative_position_bias:
            raise ValueError(
                "use_shared_relative_position_bias=True is not implemented. Every "
                "shipped BEiT/BEiTv2 variant in HF and timm uses per-layer tables "
                "(use_shared_relative_position_bias=False); the shared-table mode "
                "would require threading a per-forward bias tensor through "
                "TransformerLayer.call(), a shared-block signature change that is "
                "deliberately out of scope."
            )

    def build(self, input_shape: Any) -> None:
        """Explicitly build EVERY sub-layer from stored config.

        The shapes come from the CONFIG, never from ``input_shape``'s optional mask
        entry, so ``mask_token`` is built identically whether or not the caller ever
        passes a mask. A lazily-built sub-layer silently drops its weights on a
        ``.keras`` round trip.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``. Only
            forwarded to ``super().build``; every sub-layer shape is derived from
            config.
        :type input_shape: Any
        """
        if self.built:
            return

        image_shape = (None,) + self.input_shape_config
        patch_shape = (None, self.num_patches, self.hidden_size)
        mask_shape = (None, self.num_patches)
        seq_shape = (None, self.seq_len, self.hidden_size)

        self.patch_embed.build(image_shape)
        # ALWAYS built — even in the classifier, which never calls it.
        self.mask_token.build([patch_shape, mask_shape])
        self.cls_token.build(patch_shape)
        if self.pos_embed is not None:
            self.pos_embed.build(seq_shape)
        self.embed_dropout.build(seq_shape)
        for layer in self.encoder_layers:
            layer.build(seq_shape)
        if self.final_norm is not None:
            self.final_norm.build(seq_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: ``image (B, H, W, C)``, ``(image, bool_mask (B, N))``, or a
            dict with an ``'images'`` key and an optional ``'mask'`` key.
        :type inputs: Any
        :param training: Keras training flag. Note that ``training=None`` is NOT
            inference for the blocks' stochastic depth — pass ``training=False``
            explicitly for a deterministic forward.
        :type training: Optional[bool]
        :returns: ``(B, N + 1, hidden_size)`` — the full token sequence, cls first.
        :rtype: keras.KerasTensor
        """
        image, bool_mask = _split_inputs(inputs, "BeitModel")

        x = self.patch_embed(image, training=training)

        # Python `if` on a TRACE-TIME structural fact (did the caller pass a mask?),
        # the sanctioned "ALWAYS CREATE / CONDITIONALLY USE" pattern — NOT a Python
        # `if` on a tensor VALUE. The layer stays built either way.
        if bool_mask is not None:
            x = self.mask_token([x, bool_mask])

        x = self.cls_token(x)

        if self.pos_embed is not None:
            x = self.pos_embed(x, training=training)

        x = self.embed_dropout(x, training=training)

        for layer in self.encoder_layers:
            x = layer(x, training=training)

        # D-007: present only when use_mean_pooling is False. See the anchor above.
        if self.final_norm is not None:
            x = self.final_norm(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Output shape from stored config — valid UNBUILT.

        :param input_shape: A bare image shape, an ``(image, mask)`` pair, or a dict;
            only the batch dimension is read from it.
        :type input_shape: Any
        :returns: ``(batch, seq_len, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        image_shape = _image_shape_of(input_shape)
        batch = (
            image_shape[0]
            if image_shape is not None and len(image_shape) == 4
            else None
        )
        return (batch, self.seq_len, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        :returns: Every constructor argument, so the trunk is reconstructed from
            config rather than from serialized sub-layers.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "patch_size": self.patch_size,
            "scale": self.scale,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "layer_scale_init_value": self.layer_scale_init_value,
            "layer_norm_eps": self.layer_norm_eps,
            "drop_path_rate": self.drop_path_rate,
            "hidden_dropout_rate": self.hidden_dropout_rate,
            "attention_probs_dropout_rate": self.attention_probs_dropout_rate,
            "use_absolute_position_embeddings": self.use_absolute_position_embeddings,
            "use_relative_position_bias": self.use_relative_position_bias,
            "use_shared_relative_position_bias": self.use_shared_relative_position_bias,
            "use_mean_pooling": self.use_mean_pooling,
            "initializer_range": self.initializer_range,
        })
        return config

    # DECISION plan-2026-08-24T074054-247151fd/D-008: only patch_size needs
    # normalizing here; input_shape already coerces to a tuple in __init__, so
    # normalizing it too would be dead code. See decisions.md.
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BeitModel":
        """Create a trunk from configuration, normalising ``patch_size`` only.

        :param config: Configuration dictionary, possibly JSON round-tripped.
        :type config: Dict[str, Any]
        :returns: The reconstructed backbone.
        :rtype: BeitModel
        """
        config = dict(config)
        patch_size = config.get("patch_size")
        if isinstance(patch_size, (list, tuple)):
            config["patch_size"] = tuple(int(v) for v in patch_size)
        return cls(**config)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            patch_size: Union[int, Tuple[int, int]] = 16,
            **kwargs: Any,
    ) -> "BeitModel":
        """Create a :class:`BeitModel` from a variant name.

        :param variant: A key of :data:`MODEL_VARIANTS` (``'beit_tiny'`` ...
            ``'beit_large'``) or the bare scale (``'tiny'`` ... ``'large'``).
        :type variant: str
        :param input_shape: Image shape ``(H, W, C)``.
        :type input_shape: Tuple[int, int, int]
        :param patch_size: ``int`` or ``(h, w)``.
        :type patch_size: Union[int, Tuple[int, int]]
        :param kwargs: Forwarded to the constructor.
        :type kwargs: Any
        :returns: The configured backbone, named :data:`BACKBONE_NAME`.
        :rtype: BeitModel
        :raises ValueError: If ``variant`` is not recognized.
        """
        return cls(
            input_shape=input_shape,
            patch_size=patch_size,
            scale=_resolve_scale(variant),
            **kwargs,
        )


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.beit.model")
class BeitForMaskedImageModeling(keras.Model):
    """
    BEiT MIM model: trunk -> ``decoder_norm`` -> ``decoder_head`` -> patch logits.

    BEiT's pre-training objective — predict the frozen tokenizer's discrete
    visual-token id at every MASKED patch position. The head is a single affine
    projection over the codebook; the loss is restricted to the masked set by the
    ``sample_weight`` carried in the ``tf.data`` batch, NOT by anything in this class
    (no ``train_step``, no ``compute_loss``, and none may be added).

    **The cls position is excluded from the output.** The trunk emits ``N + 1`` tokens;
    this head slices ``[:, 1:, :]`` before projecting, so the output is ``(B, N, vocab)``
    and lines up index-for-index with an ``(B, N)`` target-id tensor produced from the
    patch grid. Emitting ``N + 1`` logits would put every target off by one position
    with no error anywhere.

    **The head emits LOGITS** (no softmax). Compile with
    ``SparseCategoricalCrossentropy(from_logits=True)`` — the house convention.

    All head sub-layers carry a ``decoder_`` prefix, disjoint from the classifier's
    ``head_`` prefix, so ``skip_prefixes=("decoder_", "head_")`` transfers the trunk
    and nothing else.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐   ┌────────────────────┐
        │  Input image [B, H, W, C]            │   │ bool_mask [B, N]   │
        └───────────────┬──────────────────────┘   └─────────┬──────────┘
                        └──────────────┬───────────────────--┘
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  BeitModel trunk (BACKBONE_NAME)              → [B, N+1, D]  │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  slice [:, 1:, :] — DROP the cls position                     │
        │    → [B, N, D], index-aligned with an [B, N] target-id        │
        │    tensor. Emitting N+1 logits would put every target off     │
        │    by one, silently.                                          │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  decoder_norm: LayerNorm(trunk eps)  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  decoder_head: Dense(vocab_size)     │
        │    NO activation                     │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, N, vocab_size] — LOGITS  │
        └──────────────────────────────────────┘

    :param backbone: A :class:`BeitModel` (must be named :data:`BACKBONE_NAME` for the
        warm start to match by name), or its serialized config dict.
    :type backbone: Union[BeitModel, Dict[str, Any]]
    :param vocab_size: Size of the discrete visual-token codebook. Defaults to
        :data:`DEFAULT_VOCAB_SIZE` (8192, the DALL-E dVAE codebook).
    :type vocab_size: int
    :param name: Model name.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.
    :type kwargs: Any
    :raises ValueError: If ``vocab_size`` is not a positive integer.
    :raises TypeError: If ``backbone`` is neither a :class:`BeitModel` nor a config
        dict that deserializes to one.

    Input shape:
        - Masked (the training form): ``[(batch, H, W, C), (batch, N) bool]`` — image
          plus patch mask; or the dict form
          ``{'images': ..., 'mask': ...}``.
        - Unmasked: a bare ``(batch, H, W, C)`` image is also accepted, in which case
          no token is replaced.

    Output shape:
        3D tensor ``(batch, N, vocab_size)`` — logits over the codebook, cls position
        excluded. One output mode only.

    Example:
        >>> model = create_beit_mim('tiny', (224, 224, 3), 16)
        >>> model.compile(
        ...     optimizer='adamw',
        ...     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ... )  # the sample_weight in the tf.data element restricts the loss to the
        ...    # masked positions -- this class has no train_step and adds none
        >>>
        >>> images = keras.random.normal((2, 224, 224, 3))
        >>> mask = keras.ops.zeros((2, 196), dtype='bool')
        >>> logits = model((images, mask), training=False)   # (2, 196, 8192)

    Attributes:
        backbone: The shared :class:`BeitModel` trunk.
        decoder_norm: LayerNorm over the patch tokens, at the trunk's epsilon.
        decoder_head: Affine projection onto the codebook. No activation.
    """

    def __init__(
            self,
            backbone: BeitModel,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
            name: Optional[str] = "beit_mim",
            **kwargs: Any,
    ) -> None:
        """Coerce and store the trunk, then create the ``decoder_``-prefixed head.

        See the class docstring for the parameter reference.
        """
        super().__init__(name=name, **kwargs)

        backbone = _coerce_backbone(backbone)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")

        self.backbone = backbone
        self.vocab_size = int(vocab_size)
        self.num_patches = backbone.num_patches
        self.hidden_size = backbone.hidden_size

        # `decoder_` prefix: distinct from `head_`, so a warm start skips exactly these.
        self.decoder_norm = keras.layers.LayerNormalization(
            epsilon=backbone.layer_norm_eps, name="decoder_norm"
        )
        self.decoder_head = keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=backbone.initializer_range
            ),
            name="decoder_head",
        )

    def build(self, input_shape: Any) -> None:
        """Build the trunk and both head sub-layers from stored config.

        The head shapes are the PATCH shape ``(None, N, D)``, not the token shape:
        the cls position is sliced off before ``decoder_norm`` ever sees the sequence.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        :type input_shape: Any
        """
        if self.built:
            return
        self.backbone.build(input_shape)
        patch_shape = (None, self.num_patches, self.hidden_size)
        self.decoder_norm.build(patch_shape)
        self.decoder_head.build(patch_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass: trunk, drop the cls position, norm, project.

        :param inputs: ``image``, ``(image, bool_mask)``, or the dict form; the
            training form always carries a mask.
        :type inputs: Any
        :param training: Keras training flag. Pass ``training=False`` explicitly for
            a deterministic forward.
        :type training: Optional[bool]
        :returns: ``(B, N, vocab_size)`` LOGITS over the codebook.
        :rtype: keras.KerasTensor
        """
        tokens = self.backbone(inputs, training=training)
        # DECISION plan-2026-08-11T012340-f63796dc/D-012: drop the cls position
        # with tokens[:, 1:, :] before the head, not [:, :-1, :] or [:, :, :] —
        # both produce the same shape but attribute targets to the wrong patch. See decisions.md.
        patch_tokens = tokens[:, 1:, :]
        x = self.decoder_norm(patch_tokens, training=training)
        return self.decoder_head(x)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Output shape from stored config — valid UNBUILT.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        :type input_shape: Any
        :returns: ``(batch, num_patches, vocab_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_patches, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        The trunk is serialized as a nested config object, so the head and its
        backbone round-trip together.

        :returns: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "vocab_size": self.vocab_size,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BeitForMaskedImageModeling":
        """Create the MIM model from configuration, deserializing the nested trunk.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :returns: The reconstructed model.
        :rtype: BeitForMaskedImageModeling
        """
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.beit.model")
class BeitForImageClassification(keras.Model):
    """
    BEiT classifier: the SAME trunk -> pool -> LayerNorm -> Dropout -> logits.

    Fine-tunes a MIM-pre-trained trunk. The backbone is composed under the identical
    name (:data:`BACKBONE_NAME`) and identical config path, so
    ``load_weights_from_checkpoint(model, mim_ckpt, skip_prefixes=("decoder_", "head_"))``
    moves the whole trunk and nothing else.

    **Pooling follows the backbone's ``use_mean_pooling``**, which is BEiT's own
    classification convention and differs from plain ViT's cls-only default:

    * ``use_mean_pooling=True`` (default): pooled = mean of the final PATCH tokens with
      the cls position EXCLUDED (``SequencePooling(strategy='mean',
      exclude_positions=[0])``), followed by this head's own ``head_norm`` — the trunk
      applies no final norm in this mode (D-007).
    * ``use_mean_pooling=False``: pooled = the cls hidden state, already normed by the
      trunk's ``final_norm``; ``head_norm`` is then the reference's no-op and is not
      created.

    **The head emits LOGITS** (no softmax). Compile with
    ``SparseCategoricalCrossentropy(from_logits=True)`` — the house convention.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input image [B, H, W, C]            │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  BeitModel trunk (BACKBONE_NAME)              → [B, N+1, D]  │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌───────────────────────────────┬──────────────────────────────┐
        │  use_mean_pooling = True      │  use_mean_pooling = False    │
        │                               │                              │
        │  head_pool: mean over         │  take the cls state          │
        │    [:, 1:, :] (cls EXCLUDED)  │    [:, 0, :]                 │
        │  head_norm: LayerNorm         │  already normed by the       │
        │    (the trunk applies none    │    trunk's final_norm;       │
        │     in this mode — D-007)     │    head_norm is not created  │
        └───────────────┬───────────────┴──────────────────────────────┘
                        ▼  [B, D]
        ┌──────────────────────────────────────┐
        │  head_dropout                        │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  head_classifier: Dense(num_classes) │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, num_classes] — LOGITS    │
        └──────────────────────────────────────┘

    :param backbone: A :class:`BeitModel` (named :data:`BACKBONE_NAME`), or its
        serialized config dict.
    :type backbone: Union[BeitModel, Dict[str, Any]]
    :param num_classes: Number of output classes. Must be positive.
    :type num_classes: int
    :param dropout_rate: Dropout before the final Dense. Defaults to ``0.0``.
    :type dropout_rate: float
    :param name: Model name.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.
    :type kwargs: Any
    :raises ValueError: If ``num_classes <= 0`` or ``dropout_rate`` is outside
        ``[0, 1]``.
    :raises TypeError: If ``backbone`` is neither a :class:`BeitModel` nor a config
        dict that deserializes to one.

    Input shape:
        4D tensor ``(batch, H, W, C)``. This head never takes a mask.

    Output shape:
        2D tensor ``(batch, num_classes)`` — LOGITS, in both pooling modes. The
        ``use_mean_pooling`` fork changes which tokens are pooled and where the
        LayerNorm lives, never the output shape.

    Example:
        >>> model = create_beit_classifier('tiny', (224, 224, 3), 16, num_classes=10)
        >>> model.build((None, 224, 224, 3))   # BEFORE any warm-start transfer
        >>> model.compile(
        ...     optimizer='adamw',
        ...     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ... )
        >>>
        >>> # cls-token pooling instead: the trunk owns the final norm in that mode
        >>> backbone = BeitModel(scale='tiny', use_mean_pooling=False)
        >>> model = BeitForImageClassification(backbone, num_classes=10)

    Attributes:
        backbone: The shared :class:`BeitModel` trunk.
        head_pool: ``SequencePooling`` over the patch tokens, or ``None`` on the
            cls-token fork.
        head_norm: LayerNorm on the pooled mean, or ``None`` on the cls-token fork.
        head_dropout: Dropout before the classifier; created at every rate.
        head_classifier: Final Dense. No activation.
    """

    # DECISION plan-2026-08-24T074054-247151fd/D-007: keep this constructor
    # flat rather than decomposed into _build_* helpers like BeitModel's; at
    # 40 lines building 4 attributes with no reuse, a helper would be classitis. See decisions.md.
    def __init__(
            self,
            backbone: BeitModel,
            num_classes: int,
            dropout_rate: float = 0.0,
            name: Optional[str] = "beit_classifier",
            **kwargs: Any,
    ) -> None:
        """Coerce and store the trunk, then create the ``head_``-prefixed head.

        The pooling fork is read off the backbone's ``use_mean_pooling``, so the two
        halves of the head cannot disagree with the trunk about where the final norm
        lives. See the class docstring for the parameter reference.
        """
        super().__init__(name=name, **kwargs)

        backbone = _coerce_backbone(backbone)
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
        if not 0.0 <= float(dropout_rate) <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)
        self.use_mean_pooling = backbone.use_mean_pooling
        self.seq_len = backbone.seq_len
        self.hidden_size = backbone.hidden_size

        # `head_` prefix: distinct from `decoder_`, so it is never transferred.
        self.head_pool = None
        self.head_norm = None
        if self.use_mean_pooling:
            # exclude_positions=[0] drops the cls token before the mean — BEiT pools
            # the PATCH tokens only. BEiT's patch sequence is fixed-length and
            # unpadded, so the historical positional-mode leak in SequencePooling is
            # not reachable here.
            self.head_pool = SequencePooling(
                strategy='mean', exclude_positions=[0], name="head_pool"
            )
            self.head_norm = keras.layers.LayerNormalization(
                epsilon=backbone.layer_norm_eps, name="head_norm"
            )

        # ALWAYS CREATE / CONDITIONALLY USE (guide §9): the Dropout exists at every
        # rate so the layer structure does not depend on a numeric value.
        self.head_dropout = keras.layers.Dropout(self.dropout_rate, name="head_dropout")
        self.head_classifier = keras.layers.Dense(
            self.num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=backbone.initializer_range
            ),
            name="head_classifier",
        )

    def build(self, input_shape: Any) -> None:
        """Build the trunk and every head sub-layer that exists on this fork.

        :param input_shape: Shape of the input image to ``call``.
        :type input_shape: Any
        """
        if self.built:
            return
        self.backbone.build(input_shape)
        seq_shape = (None, self.seq_len, self.hidden_size)
        pooled_shape = (None, self.hidden_size)
        if self.head_pool is not None:
            self.head_pool.build(seq_shape)
        if self.head_norm is not None:
            self.head_norm.build(pooled_shape)
        self.head_dropout.build(pooled_shape)
        self.head_classifier.build(pooled_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass: trunk, pool per the fork, dropout, classify.

        :param inputs: Image tensor ``(B, H, W, C)``. This head never takes a mask.
        :type inputs: Any
        :param training: Keras training flag. Pass ``training=False`` explicitly for
            a deterministic forward.
        :type training: Optional[bool]
        :returns: ``(B, num_classes)`` LOGITS.
        :rtype: keras.KerasTensor
        """
        tokens = self.backbone(inputs, training=training)

        if self.use_mean_pooling:
            pooled = self.head_pool(tokens, training=training)
            pooled = self.head_norm(pooled, training=training)
        else:
            # The trunk's final_norm already normed the sequence in this mode (D-007).
            pooled = tokens[:, 0, :]

        pooled = self.head_dropout(pooled, training=training)
        return self.head_classifier(pooled)  # logits — no softmax

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Output shape from stored config — valid UNBUILT.

        :param input_shape: Shape of the input image to ``call``.
        :type input_shape: Any
        :returns: ``(batch, num_classes)``.
        :rtype: Tuple[Optional[int], ...]
        """
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        The trunk is serialized as a nested config object, so the head and its
        backbone round-trip together.

        :returns: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BeitForImageClassification":
        """Create the classifier from configuration, deserializing the nested trunk.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :returns: The reconstructed model.
        :rtype: BeitForImageClassification
        """
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------


def create_beit_backbone(
        variant: str = 'base',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        **overrides: Any,
) -> BeitModel:
    """Create a standalone :class:`BeitModel` trunk.

    :param variant: ``'tiny'`` / ``'small'`` / ``'base'`` / ``'large'`` (or
        ``'beit_base'`` ...).
    :type variant: str
    :param input_shape: ``(H, W, C)``.
    :type input_shape: Tuple[int, int, int]
    :param patch_size: ``int`` or ``(h, w)``.
    :type patch_size: Union[int, Tuple[int, int]]
    :param overrides: Any :class:`BeitModel` constructor kwarg (e.g.
        ``drop_path_rate``, ``use_mean_pooling``, ``layer_norm_eps``).
    :type overrides: Any
    :returns: The backbone, named :data:`BACKBONE_NAME`.
    :rtype: BeitModel

    Example:
        >>> backbone = create_beit_backbone('base', (224, 224, 3), 16)
        >>> backbone.build((None, 224, 224, 3))
    """
    return BeitModel(
        input_shape=input_shape,
        patch_size=patch_size,
        scale=_resolve_scale(variant),
        name=BACKBONE_NAME,
        **overrides,
    )


def create_beit_mim(
        variant: str = 'base',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        **overrides: Any,
) -> BeitForMaskedImageModeling:
    """Create the masked-image-modeling model.

    :param variant: ``'tiny'`` / ``'small'`` / ``'base'`` / ``'large'``.
    :type variant: str
    :param input_shape: ``(H, W, C)``.
    :type input_shape: Tuple[int, int, int]
    :param patch_size: ``int`` or ``(h, w)``.
    :type patch_size: Union[int, Tuple[int, int]]
    :param vocab_size: Discrete visual-token codebook size.
    :type vocab_size: int
    :param overrides: Backbone constructor kwargs.
    :type overrides: Any
    :returns: A :class:`BeitForMaskedImageModeling` whose trunk is named
        :data:`BACKBONE_NAME`.
    :rtype: BeitForMaskedImageModeling

    Example:
        >>> model = create_beit_mim('tiny', (224, 224, 3), 16, vocab_size=8192)
        >>> model.compile(
        ...     optimizer='adamw',
        ...     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ... )  # sample_weight in the tf.data element does the masking
    """
    backbone = create_beit_backbone(
        variant=variant,
        input_shape=input_shape,
        patch_size=patch_size,
        **overrides,
    )
    return BeitForMaskedImageModeling(backbone=backbone, vocab_size=vocab_size)


def create_beit_classifier(
        variant: str = 'base',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_classes: int = 1000,
        dropout_rate: float = 0.0,
        **overrides: Any,
) -> BeitForImageClassification:
    """Create the classifier (logits head; warm-startable from an MIM checkpoint).

    :param variant: ``'tiny'`` / ``'small'`` / ``'base'`` / ``'large'``.
    :type variant: str
    :param input_shape: ``(H, W, C)``.
    :type input_shape: Tuple[int, int, int]
    :param patch_size: ``int`` or ``(h, w)``.
    :type patch_size: Union[int, Tuple[int, int]]
    :param num_classes: Number of classes.
    :type num_classes: int
    :param dropout_rate: Dropout before the final Dense.
    :type dropout_rate: float
    :param overrides: Backbone constructor kwargs.
    :type overrides: Any
    :returns: A :class:`BeitForImageClassification` whose trunk is named
        :data:`BACKBONE_NAME` and is weight-identical to :func:`create_beit_mim`'s at
        the same backbone config.
    :rtype: BeitForImageClassification

    Example:
        >>> model = create_beit_classifier('tiny', (224, 224, 3), 16, num_classes=10)
        >>> model.build((None, 224, 224, 3))   # BEFORE the transfer — H-12
        >>> model.compile(
        ...     optimizer='adamw',
        ...     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ... )
    """
    backbone = create_beit_backbone(
        variant=variant,
        input_shape=input_shape,
        patch_size=patch_size,
        **overrides,
    )
    return BeitForImageClassification(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


# ---------------------------------------------------------------------