"""A BEiT image trunk with a masked-image-modeling head and a classification head.

BEiT asks what BERT's pre-training objective becomes when the input is an image.
Masking a word and predicting it works because language arrives pre-discretized:
the target is one symbol out of a finite vocabulary, and the loss is a clean
classification. Pixels have neither property. Regressing raw pixels at masked
positions spends the model's capacity on exactly the high-frequency detail that
carries the least semantic content, and short-range pixel correlation makes much
of the task solvable by interpolation rather than understanding. BEiT's answer is
to borrow a vocabulary: a discrete variational autoencoder, trained separately
and frozen, maps every patch to one code id out of 8192, and the model predicts
*that* id at each masked position. The objective becomes classification over a
codebook, the targets sit at the abstraction level of appearance rather than
intensity, and the pixel-interpolation shortcut disappears because neighbouring
patches do not determine each other's code.

The tokenizer is deliberately not part of this module. Code ids are targets, and
they arrive as tensors in the ``tf.data`` pipeline alongside the patch mask. Nor
is the masking part of the loss computation here: the objective is restricted to
masked positions by the ``sample_weight`` carried in the batch, so these models
define no ``train_step`` and no ``compute_loss``, and none may be added.

Three classes share one trunk. :class:`BeitModel` is the encoder: patch embedding,
optional mask-token substitution, a prepended class token, ``num_layers``
transformer blocks with BEiT attention, and a full ``(B, N + 1, D)`` sequence out.
:class:`BeitForMaskedImageModeling` puts a ``decoder_``-prefixed norm and a single
affine projection over the codebook on top. :class:`BeitForImageClassification`
puts a ``head_``-prefixed pool, norm, dropout and classifier on top, and emits
logits (compile with ``from_logits=True``). Masked positions are replaced *before*
the class token is prepended and before any block runs — BEiT processes the whole
sequence and never drops tokens, which is the structural difference from MAE and
the reason the mask is a substitution rather than a gather.

Position information is relative, not absolute. Each block owns a learnable
relative-position-bias table indexed by the patch grid, so
``use_absolute_position_embeddings`` defaults to ``False`` and the absolute
embedding is not even created when it is off — an unread ``(1, N+1, D)`` weight
in every checkpoint would be pure dead weight. The shared-table variant
(one bias table for all blocks, a pre-training-only mode) is not implemented and
raises rather than silently falling back to per-layer tables: supporting it would
require threading a per-forward bias tensor through ``TransformerLayer.call()``.

Two details of this implementation exist to protect things that fail silently
when they are "cleaned up".

``MaskTokenApply`` is created *and built* by every backbone, including the
classifier's, which never calls it. Removing that apparently dead weight from the
classifier would leave the two trunks with different weight sets, and the warm
start ``load_weights_from_checkpoint(target, mim_ckpt,
skip_prefixes=("decoder_", "head_"))`` — which matches by name, hence the fixed
``BACKBONE_NAME`` — would quietly transfer a different set of layers with no
error anywhere.

The trunk's final ``LayerNormalization`` exists only when ``use_mean_pooling`` is
``False``. That is BEiT's own fork, not a simplification: at the default the
pooler applies its own norm to the mean of the patch tokens, so a trunk-level
norm would insert an extra normalization in front of both heads that the
reference does not have — no error, no shape change, and a perfectly plausible
loss curve. Likewise the MIM head slices ``[:, 1:, :]`` to drop the class
position before projecting, so output index ``i`` is patch ``i``; every other
length-``N`` window of the sequence produces the same output shape and the same
finite logits while attributing every code-id target to the wrong patch.

``layer_norm_eps`` defaults to ``1e-12`` (HF's ``BeitConfig``), six orders of
magnitude tighter than a generic ViT's ``1e-6``, and is passed explicitly at
every normalization site rather than inherited from any factory default.
Stochastic depth is a linear ramp from ``0`` to ``drop_path_rate`` across the
blocks, computed at model level because a block holds only its own rate.

Two deliberate deviations, also recorded in this package's ``README.md``:
``layer_scale_init_value`` follows timm's split (``0.1`` for tiny/small/base,
``1e-5`` for large) rather than HF's uniform ``0.1`` — the two ports of the same
official checkpoints disagree, and layer-scale init is training-time-only, so
neither is wrong but the pick is pinned. And ``tiny`` and ``small`` are repo
inventions for cheap tests; no BEiT of either size exists in the paper, in HF or
in timm, while ``base`` and ``large`` reproduce the fetched HF configs verbatim.

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

import keras
from keras import layers
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

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

BeitScale = Literal['tiny', 'small', 'base', 'large']

# The stable sub-model name. `load_weights_from_checkpoint` matches layer BY NAME, so
# the MIM model and the classifier MUST name their backbone identically or the
# warm-start transfers zero layers.
BACKBONE_NAME = "beit_backbone"

# The DALL-E dVAE codebook size BEiT v1 predicts over; HF's `BeitConfig.vocab_size`.
# It is a property of the MIM TARGET, not of the trunk, so it lives on the MIM head
# rather than on the backbone (a backbone field nothing reads is a config-shaped lie).
DEFAULT_VOCAB_SIZE = 8192

# DECISION plan-2026-08-11T012340-f63796dc/D-003
# `layer_scale_init_value` DIVERGES between the two primary sources, and the split
# below is timm's, not HF's. Measured (both fetched verbatim, 2026-08-11):
#   * HF `config.json` for microsoft/beit-base-patch16-224 AND
#     microsoft/beit-large-patch16-224 both report "layer_scale_init_value": 0.1.
#   * timm `models/beit.py` uses init_values=0.1 for every `beit_base_patch16_*` and
#     init_values=1e-5 for every `beit_large_patch16_*` (and for both BEiTv2 sizes).
# WHAT NOT TO DO: do NOT "correct" the large entry to 0.1 to make this table agree with
# HF's config.json field-for-field. The disagreement is between two ports of the SAME
# official checkpoints, it is real, and it has been decided deliberately in favour of
# timm's. Layer-scale init is a training-time-only hyperparameter, so neither value is
# wrong; an unrecorded pick is what gets re-litigated. Pinned by
# `TestBeitScaleConfigs::test_layer_scale_init_value_split_is_timms`.
# See decisions.md D-003 (deviations X-2 and X-3).
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

    Args:
        variant: A key of :data:`SCALE_CONFIGS` or of :data:`MODEL_VARIANTS`.

    Returns:
        The resolved :data:`SCALE_CONFIGS` key.

    Raises:
        ValueError: If ``variant`` is neither spelling.
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
    """Coerce an int or 2-sequence to a validated positive integer pair."""
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
    """Pull the image entry out of a possibly-nested ``input_shape``."""
    if isinstance(input_shape, dict):
        return input_shape.get('images')
    if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (tuple, list))
    ):
        return input_shape[0]
    return input_shape


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BeitModel(keras.Model):
    """BEiT trunk: patch-embed -> [mask token] -> cls token -> N x BEiT block -> tokens.

    **Intent**: a single, separately-checkpointable image trunk that BOTH the
    masked-image-modeling model and the classifier compose under the same name, so a
    pre-trained encoder transfers into the classifier layer-for-layer.

    **Call signature** — ``backbone(image)``, ``backbone((image, bool_mask))``, or
    ``backbone({'images': image, 'mask': bool_mask})``. ``bool_mask`` is
    ``(B, N)`` boolean with ``True`` at positions whose patch embedding is replaced by
    the learnable mask token, BEFORE the cls token is prepended and before any
    transformer block — BEiT processes the full sequence and never drops tokens.

    **``MaskTokenApply`` is ALWAYS created and ALWAYS built**, even for the classifier
    that never calls it (authoring guide §9). That is what makes the two trunks
    weight-identical so the warm start is complete. Do not "optimize" it away.

    Args:
        input_shape: Image shape ``(height, width, channels)``. Defaults to
            ``(224, 224, 3)``.
        patch_size: Patch size; ``int`` for square patches or ``(h, w)``. Defaults to
            ``16``.
        scale: One of ``'tiny'``, ``'small'``, ``'base'``, ``'large'`` (see
            :data:`SCALE_CONFIGS`); a variant spelling such as ``'beit_base'`` is also
            accepted.
        hidden_size: Override the scale's model width ``D``. ``None`` -> from ``scale``.
        num_layers: Override the scale's block count. ``None`` -> from ``scale``.
        num_heads: Override the scale's head count. ``None`` -> from ``scale``.
        intermediate_size: Override the scale's FFN width. ``None`` -> from ``scale``.
        layer_scale_init_value: Override the scale's LayerScale init. ``None`` -> from
            ``scale``. See :data:`SCALE_CONFIGS` for why the value is scale-dependent.
        layer_norm_eps: Epsilon at EVERY normalization site. Defaults to ``1e-12``
            (HF ``BeitConfig``), which is 6 orders of magnitude tighter than a generic
            ViT's ``1e-6`` — it is passed explicitly and never inherited from a
            constructor default.
        drop_path_rate: Maximum stochastic-depth rate. The per-block rates are the
            linear ramp ``0 -> drop_path_rate`` across ``num_layers``. Defaults to
            ``0.1``.
        hidden_dropout_prob: Dropout after the embedding stage and on each block's FFN
            output. Defaults to ``0.0`` (HF ``BeitConfig``).
        attention_probs_dropout_prob: Dropout on the attention probabilities. Defaults
            to ``0.0`` (HF ``BeitConfig``).
        use_absolute_position_embeddings: Add a learnable absolute position embedding
            over the ``N + 1`` token sequence. Defaults to ``False`` — BEiT uses
            RELATIVE position bias instead, and no shipped BEiT/BEiTv2 variant in HF or
            timm enables this.
        use_relative_position_bias: Give every block's ``BeitAttention`` its own
            learnable relative-position-bias table. Defaults to ``True``.
        use_shared_relative_position_bias: One table shared by every block (BEiT's
            pre-training-only mode). Only ``False`` is supported; ``True`` raises.
        use_mean_pooling: BEiT's classification convention — mean over the final patch
            tokens (cls excluded) with a SEPARATE LayerNorm on the pooled mean. It also
            controls whether this trunk applies its own final LayerNorm; see the note
            on :attr:`final_norm`. Defaults to ``True`` (HF ``BeitConfig``).
        initializer_range: Stddev of the ``TruncatedNormal`` used for every projection
            kernel. Defaults to ``0.02`` (HF ``BeitConfig``).
        name: Model name. Defaults to :data:`BACKBONE_NAME` — do not change it for a
            model that must participate in the MIM -> classifier warm start.

    Raises:
        ValueError: If any size is non-positive, ``hidden_size % num_heads != 0``, the
            image dims are not divisible by the patch dims, a dropout rate lies outside
            ``[0, 1]``, ``scale`` is unknown, or ``use_shared_relative_position_bias``
            is ``True``.

    Input shape:
        ``(batch, H, W, C)``; or a 2-tuple ``[(batch, H, W, C), (batch, N)]`` whose
        second entry is the boolean patch mask.

    Output shape:
        ``(batch, N + 1, hidden_size)`` — the FULL sequence, cls token first.
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
            hidden_dropout_prob: float = 0.0,
            attention_probs_dropout_prob: float = 0.0,
            use_absolute_position_embeddings: bool = False,
            use_relative_position_bias: bool = True,
            use_shared_relative_position_bias: bool = False,
            use_mean_pooling: bool = True,
            initializer_range: float = 0.02,
            name: Optional[str] = BACKBONE_NAME,
            **kwargs: Any,
    ) -> None:
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
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.attention_probs_dropout_prob = float(attention_probs_dropout_prob)
        self.use_absolute_position_embeddings = bool(use_absolute_position_embeddings)
        self.use_relative_position_bias = bool(use_relative_position_bias)
        self.use_shared_relative_position_bias = bool(use_shared_relative_position_bias)
        self.use_mean_pooling = bool(use_mean_pooling)
        self.initializer_range = float(initializer_range)

        self._validate_config()

        # ----- derived geometry -----
        img_h, img_w, _ = self.input_shape_config
        patch_h, patch_w = _as_pair(self.patch_size, "patch_size")
        # The PATCH GRID, which is `BeitAttention`'s `window_size` — NOT the image size
        # and NOT a scalar edge length.
        self.grid_size: Tuple[int, int] = (img_h // patch_h, img_w // patch_w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.seq_len = self.num_patches + 1  # + cls

        kernel_init = keras.initializers.TruncatedNormal(stddev=self.initializer_range)

        # ----- CREATE all sub-layers in __init__ (unbuilt) -----
        self.patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=(patch_h, patch_w),
            embed_dim=self.hidden_size,
            kernel_initializer=kernel_init,
            name="patch_embed",
        )

        # Guide §9: ALWAYS CREATE, CONDITIONALLY USE. The classifier never calls this,
        # but it MUST own the weight or its trunk stops matching the MIM trunk and the
        # warm start silently transfers a different set of layers. This is the
        # DELIBERATE exception to "build only what call() runs" — see build().
        self.mask_token = MaskTokenApply(name="mask_token")

        self.cls_token = ClassTokenPrepend(name="cls_token")

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

        self.embed_dropout = layers.Dropout(
            self.hidden_dropout_prob, name="embed_dropout"
        )

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
                    dropout_rate=self.hidden_dropout_prob,
                    attention_dropout_rate=self.attention_probs_dropout_prob,
                    use_layer_scale=True,
                    layer_scale_init_value=self.layer_scale_init_value,
                    use_stochastic_depth=True,
                    stochastic_depth_rate=self.drop_path_rates[i],
                    kernel_initializer=kernel_init,
                    name=f"encoder_layer_{i}",
                )
            )

        # DECISION plan-2026-08-11T012340-f63796dc/D-007
        # The trunk's final LayerNorm EXISTS ONLY WHEN `use_mean_pooling is False`.
        # This is not a simplification, it is BEiT's own fork. In HF's port
        # `BeitModel.layernorm` is `nn.Identity()` when `use_mean_pooling=True`, and
        # `BeitPooler` then applies its OWN LayerNorm to the mean of the patch tokens;
        # when `use_mean_pooling=False` the trunk applies the LayerNorm and the pooled
        # output is the raw cls hidden state with no further norm.
        # WHAT NOT TO DO: do NOT "clean this up" by always applying a final norm here.
        # At the default `use_mean_pooling=True` that inserts an extra normalization
        # the reference does not have, in front of BOTH heads (the classifier's
        # `head_norm` and the MIM head's `decoder_norm` would each be norming an
        # already-normed sequence). It raises no error, changes no shape, and produces
        # a perfectly plausible loss curve. Equally, do NOT create the layer
        # unconditionally and skip it in `call()`: an unused-but-built sub-layer is
        # dead weight in every checkpoint, and the two heads share this backbone config
        # so there is no warm-start asymmetry to protect against (unlike `mask_token`,
        # whose use depends on the CALL, not on the config).
        # Pinned by `TestBeitArchitectureValidation::test_final_norm_follows_the_mean_pooling_fork`.
        # See decisions.md D-007.
        self.final_norm = None
        if not self.use_mean_pooling:
            self.final_norm = layers.LayerNormalization(
                epsilon=self.layer_norm_eps, name="final_norm"
            )

        logger.info(
            f"Created BeitModel-{scale}: {self.hidden_size}d, {self.num_layers}L, "
            f"{self.num_heads}h, ffn={self.intermediate_size}, grid={self.grid_size}, "
            f"N={self.num_patches}, eps={self.layer_norm_eps}, "
            f"layer_scale={self.layer_scale_init_value}"
        )

    # -----------------------------------------------------------------

    def _validate_config(self) -> None:
        """Reject every invalid configuration at CONSTRUCTION time, not at first call."""
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
                ("hidden_dropout_prob", self.hidden_dropout_prob),
                ("attention_probs_dropout_prob", self.attention_probs_dropout_prob),
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

        Args:
            inputs: ``image (B, H, W, C)``, ``(image, bool_mask (B, N))``, or a dict
                with an ``'images'`` key and an optional ``'mask'`` key.
            training: Keras training flag. Note that ``training=None`` is NOT inference
                for the blocks' stochastic depth — pass ``training=False`` explicitly
                for a deterministic forward.

        Returns:
            ``(B, N + 1, hidden_size)`` — the full token sequence, cls first.
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
        """Output shape from stored config — valid UNBUILT."""
        image_shape = _image_shape_of(input_shape)
        batch = (
            image_shape[0]
            if image_shape is not None and len(image_shape) == 4
            else None
        )
        return (batch, self.seq_len, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
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
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "use_absolute_position_embeddings": self.use_absolute_position_embeddings,
            "use_relative_position_bias": self.use_relative_position_bias,
            "use_shared_relative_position_bias": self.use_shared_relative_position_bias,
            "use_mean_pooling": self.use_mean_pooling,
            "initializer_range": self.initializer_range,
        })
        return config

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            patch_size: Union[int, Tuple[int, int]] = 16,
            **kwargs: Any,
    ) -> "BeitModel":
        """Create a :class:`BeitModel` from a variant name.

        Args:
            variant: A key of :data:`MODEL_VARIANTS` (``'beit_tiny'`` ...
                ``'beit_large'``) or the bare scale (``'tiny'`` ... ``'large'``).
            input_shape: Image shape ``(H, W, C)``.
            patch_size: ``int`` or ``(h, w)``.
            **kwargs: Forwarded to the constructor.

        Returns:
            The configured backbone, named :data:`BACKBONE_NAME`.

        Raises:
            ValueError: If ``variant`` is not recognized.
        """
        return cls(
            input_shape=input_shape,
            patch_size=patch_size,
            scale=_resolve_scale(variant),
            **kwargs,
        )


# ---------------------------------------------------------------------


def _coerce_backbone(backbone: Any) -> BeitModel:
    """Accept a live backbone or its serialized config dict (the ``from_config`` path)."""
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


@keras.saving.register_keras_serializable()
class BeitForMaskedImageModeling(keras.Model):
    """BEiT MIM model: trunk -> ``decoder_norm`` -> ``decoder_head`` -> patch logits.

    **Intent**: BEiT's pre-training objective — predict the frozen tokenizer's discrete
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

    Args:
        backbone: A :class:`BeitModel` (must be named :data:`BACKBONE_NAME` for the
            warm start to match by name), or its serialized config dict.
        vocab_size: Size of the discrete visual-token codebook. Defaults to
            :data:`DEFAULT_VOCAB_SIZE` (8192, the DALL-E dVAE codebook).
        name: Model name.

    Raises:
        ValueError: If ``vocab_size`` is not a positive integer.

    Input shape:
        ``[(batch, H, W, C), (batch, N) bool]`` — image + patch mask. A bare
        ``(batch, H, W, C)`` image is also accepted (no tokens are replaced).

    Output shape:
        ``(batch, N, vocab_size)`` — logits over the codebook, cls position excluded.
    """

    def __init__(
            self,
            backbone: BeitModel,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
            name: Optional[str] = "beit_mim",
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        backbone = _coerce_backbone(backbone)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")

        self.backbone = backbone
        self.vocab_size = int(vocab_size)
        self.num_patches = backbone.num_patches
        self.hidden_size = backbone.hidden_size

        # `decoder_` prefix: distinct from `head_`, so a warm start skips exactly these.
        self.decoder_norm = layers.LayerNormalization(
            epsilon=backbone.layer_norm_eps, name="decoder_norm"
        )
        self.decoder_head = layers.Dense(
            self.vocab_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=backbone.initializer_range
            ),
            name="decoder_head",
        )

    def build(self, input_shape: Any) -> None:
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
        tokens = self.backbone(inputs, training=training)
        # DECISION plan-2026-08-11T012340-f63796dc/D-012
        # Drop the cls position BEFORE the head so output index i is patch i.
        # Do NOT change this slice. `tokens` is `(B, N+1, D)` with cls at index 0, and
        # EVERY length-N window of it produces the same `(B, N, vocab)` output shape,
        # the same finite logits and the same plausible loss curve:
        #   - `[:, :-1, :]`  drops the LAST PATCH and feeds cls in as patch 0 — every
        #     code-id target is then attributed to the wrong patch, silently;
        #   - `[:, :, :]`    keeps cls and emits N+1 logits, which only fails loudly if
        #     the loss refuses to broadcast.
        # Do NOT "verify" this with a shape assertion — a shape cannot see it (README
        # §14 Issue 2). It is pinned by IDENTITY in
        # `TestBeitForMaskedImageModeling::test_the_head_reads_the_patch_tokens_not_a_
        # shifted_window`, which was demonstrated RED under the `[:, :-1, :]` mutation.
        patch_tokens = tokens[:, 1:, :]
        x = self.decoder_norm(patch_tokens, training=training)
        return self.decoder_head(x)  # logits — no softmax

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_patches, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "vocab_size": self.vocab_size,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BeitForMaskedImageModeling":
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BeitForImageClassification(keras.Model):
    """BEiT classifier: the SAME trunk -> pool -> LayerNorm -> Dropout -> logits.

    **Intent**: fine-tune a MIM-pre-trained trunk. The backbone is composed under the
    identical name (:data:`BACKBONE_NAME`) and identical config path, so
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

    Args:
        backbone: A :class:`BeitModel` (named :data:`BACKBONE_NAME`), or its serialized
            config dict.
        num_classes: Number of output classes. Must be positive.
        dropout_rate: Dropout before the final Dense. Defaults to ``0.0``.
        name: Model name.

    Raises:
        ValueError: If ``num_classes <= 0`` or ``dropout_rate`` is outside ``[0, 1]``.

    Input shape:
        ``(batch, H, W, C)``.

    Output shape:
        ``(batch, num_classes)`` — logits.
    """

    def __init__(
            self,
            backbone: BeitModel,
            num_classes: int,
            dropout_rate: float = 0.0,
            name: Optional[str] = "beit_classifier",
            **kwargs: Any,
    ) -> None:
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
            self.head_norm = layers.LayerNormalization(
                epsilon=backbone.layer_norm_eps, name="head_norm"
            )

        # ALWAYS CREATE / CONDITIONALLY USE (guide §9): the Dropout exists at every
        # rate so the layer structure does not depend on a numeric value.
        self.head_dropout = layers.Dropout(self.dropout_rate, name="head_dropout")
        self.head_classifier = layers.Dense(
            self.num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=backbone.initializer_range
            ),
            name="head_classifier",
        )

    def build(self, input_shape: Any) -> None:
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
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BeitForImageClassification":
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

    Args:
        variant: ``'tiny'`` / ``'small'`` / ``'base'`` / ``'large'`` (or ``'beit_base'``
            ...).
        input_shape: ``(H, W, C)``.
        patch_size: ``int`` or ``(h, w)``.
        **overrides: Any :class:`BeitModel` constructor kwarg (e.g. ``drop_path_rate``,
            ``use_mean_pooling``, ``layer_norm_eps``).

    Returns:
        The backbone, named :data:`BACKBONE_NAME`.
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

    Args:
        variant: ``'tiny'`` / ``'small'`` / ``'base'`` / ``'large'``.
        input_shape: ``(H, W, C)``.
        patch_size: ``int`` or ``(h, w)``.
        vocab_size: Discrete visual-token codebook size.
        **overrides: Backbone constructor kwargs.

    Returns:
        A :class:`BeitForMaskedImageModeling` whose trunk is named
        :data:`BACKBONE_NAME`.

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

    Args:
        variant: ``'tiny'`` / ``'small'`` / ``'base'`` / ``'large'``.
        input_shape: ``(H, W, C)``.
        patch_size: ``int`` or ``(h, w)``.
        num_classes: Number of classes.
        dropout_rate: Dropout before the final Dense.
        **overrides: Backbone constructor kwargs.

    Returns:
        A :class:`BeitForImageClassification` whose trunk is named
        :data:`BACKBONE_NAME` and is weight-identical to :func:`create_beit_mim`'s at
        the same backbone config.

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
