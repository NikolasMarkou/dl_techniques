"""MobileCLIP2 dual-encoder model.

Pairs the faithful FastViT (MCi) image tower of :mod:`.image_encoder` with the
OpenCLIP-shaped text transformer already shipped by ``models/mobile_clip`` and
adds the CLIP epilogue: L2-normalized features, a learnable temperature, and the
symmetric logits matrix.

**Architecture**::

    {'image': (B, 256, 256, 3), 'text': (B, 77)}
        |                             |
    FastVitImageEncoder          MobileClipTextEncoder
        |                             |
    (B, embed_dim)                (B, embed_dim)
        |                             |
    L2 normalize                 L2 normalize
        \\___________  _______________/
                    \\/
        scale = clip(exp(logit_scale), 0, 100)
                    |
    {'image_features', 'text_features',
     'logits_per_image', 'logits_per_text', 'logit_scale'}

**There is no separate image projection.** The image tower's terminal ``Dense``
IS the CLIP image projection (open_clip's ``TimmModel`` with ``timm_proj=null``
builds the trunk at ``num_classes=embed_dim``), so ``projection_dim`` is passed
straight to :class:`FastVitImageEncoder` and its output is the image embedding.

**Both towers see raw, un-normalized features from their own ``call``.**
Normalization happens in :meth:`MobileClipV2Model.encode_image` /
:meth:`MobileClipV2Model.encode_text`, because
:func:`dl_techniques.utils.clip_utils.compute_clip_logits` documents that it
expects ALREADY L2-normalized inputs and does not normalize internally.

References:
    - Vasu et al., 2024. "MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training." CVPR. arXiv:2311.17049.
    - Faghri et al., 2025. "MobileCLIP2: Improving Multi-Modal Reinforced
      Training." arXiv:2508.20691.
    - Radford et al., 2021. "Learning Transferable Visual Models From Natural
      Language Supervision." ICML. arXiv:2103.00020.
"""

import math
import keras
from keras import ops, initializers
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.clip_utils import compute_clip_logits
from dl_techniques.models.mobile_clip_v2.image_encoder import (
    FastVitImageEncoder,
)

# DECISION plan-2026-08-13T183738-24486492/D-001
# The text tower is IMPORTED from the v1 package rather than re-implemented.
# Cross-package precedent: `models/lewm/model.py` imports `ViT` from
# `models/vit/model.py`.
#
# DO NOT re-implement a text transformer in this package, and do not copy
# `MobileClipTextEncoder` here "to remove the cross-package import". Either move
# would create a THIRD block->keep causal-mask adapter site, which SYSTEM.md:171
# records as a mandatory trigger for promoting that adapter into a keep-polarity
# `MaskFactory` variant — an unrelated refactor with its own blast radius.
# See decisions.md D-001.
#
# WHY AN IMPORT AND NOT A COPY: `MobileClipTextEncoder` owns one of exactly TWO
# block->keep causal-mask adapter sites in `src/` (the other is
# `layers/heads/vlm/factory.py`). A THIRD site triggers a mandatory promotion of
# that adapter into a keep-polarity `MaskFactory` variant — an unrelated
# refactor with its own blast radius. Re-implementing the text tower here would
# create that third site for no architectural gain: the layer is already
# dimension-generic (MEASURED at 768/12/3072 and 512/8/2048) and already carries
# the graph-safe `MaskFactory.create_causal_mask` path that `ops.tril` cannot
# provide on this stack.
from dl_techniques.models.mobile_clip.components import MobileClipTextEncoder

# ---------------------------------------------------------------------
# reference constants
# ---------------------------------------------------------------------

#: Stable sub-model names. `load_weights_from_checkpoint` matches layers BY NAME,
#: so a tower that is ever warm-started independently must be named identically
#: in every model that holds it.
IMAGE_TOWER_NAME = "image_encoder"
TEXT_TOWER_NAME = "text_encoder"

#: OpenAI CLIP's initial temperature: ``log(1 / 0.07)``.
_DEFAULT_LOGIT_SCALE_INIT = math.log(1.0 / 0.07)

#: Upper bound applied to ``exp(logit_scale)`` on every use. This is v1's
#: convention (`models/mobile_clip/mobile_clip_v1.py`) and OpenCLIP's, and it is
#: NOT cosmetic: without it a diverging temperature produces `inf` logits and a
#: `nan` contrastive loss with no other symptom.
_LOGIT_SCALE_MAX = 100.0

# ---------------------------------------------------------------------
# variant table
# ---------------------------------------------------------------------

# PROVENANCE — one row per SUPPLIED JSON config file, keyed by that file's own
# name, so each row is a checkable transcription rather than a re-derivation.
#
# `use_causal_mask` is the NEGATION of the JSON's `no_causal_mask` field. The
# MobileCLIP2 series (`mobileclip2_s*`) sets `"no_causal_mask": true`, i.e. a
# BIDIRECTIONAL text tower; the earlier MobileCLIP-S3/S4 configs leave it false,
# i.e. the classic causal CLIP text tower. That single flag is the only reason
# both families appear in this table — do NOT "simplify" it away.
#
# Fields identical in every row and therefore NOT tabulated:
#   vocab_size = 49408 (OpenAI BPE), context_length = 77, image_size = 256,
#   text_intermediate = 4 * text_width.
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    'mobileclip2_s0': {
        'embed_dim': 512,
        'image_backbone': 'mci0',
        'text_width': 512,
        'text_heads': 8,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip2_s2': {
        'embed_dim': 512,
        'image_backbone': 'mci2',
        'text_width': 512,
        'text_heads': 8,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip2_s3': {
        'embed_dim': 768,
        'image_backbone': 'mci3',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip2_s4': {
        'embed_dim': 768,
        'image_backbone': 'mci4',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': False,
    },
    'mobileclip_s3': {
        'embed_dim': 768,
        'image_backbone': 'mci3',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': True,
    },
    'mobileclip_s4': {
        'embed_dim': 768,
        'image_backbone': 'mci4',
        'text_width': 768,
        'text_heads': 12,
        'text_layers': 12,
        'use_causal_mask': True,
    },
}

#: Shared by every row of :data:`MODEL_VARIANTS`.
DEFAULT_VOCAB_SIZE = 49408
DEFAULT_CONTEXT_LENGTH = 77
DEFAULT_IMAGE_SIZE = 256

#: The reference text tower's FFN expansion.
_TEXT_MLP_RATIO = 4

# ---------------------------------------------------------------------


def _resolve_model_variant(variant: str) -> Dict[str, Any]:
    """Look up a variant name in :data:`MODEL_VARIANTS`.

    Args:
        variant: A key of :data:`MODEL_VARIANTS`, e.g. ``'mobileclip2_s0'``.

    Returns:
        A shallow copy of the variant's configuration dictionary.

    Raises:
        ValueError: If ``variant`` is not a known variant name.
    """
    key = str(variant)
    if key not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown MobileCLIP2 variant {variant!r}. Available: "
            f"{sorted(MODEL_VARIANTS)}."
        )
    return dict(MODEL_VARIANTS[key])


@keras.saving.register_keras_serializable()
class MobileClipV2Model(keras.Model):
    """MobileCLIP2 dual encoder — FastViT (MCi) image tower + CLIP text tower.

    Holds both towers as attributes and owns a single learnable scalar
    ``logit_scale`` (created in :meth:`build`, following the ``models/clip``
    precedent). :meth:`call` takes a dict and returns a dict.

    Args:
        embed_dim: Width of the joint image-text embedding space. Both towers
            project into it.
        image_backbone: MCi variant name for the image tower (``'mci0'`` ...
            ``'mci4'``), forwarded to :meth:`FastVitImageEncoder.from_variant`.
        image_size: Square input resolution of the image tower. Defaults to 256.
        vocab_size: Token vocabulary size. Defaults to 49408 (OpenAI BPE).
        context_length: Maximum text sequence length. Defaults to 77.
        text_width: Hidden width of the text transformer.
        text_heads: Attention heads per text transformer layer. Must divide
            ``text_width``.
        text_layers: Number of text transformer layers.
        text_intermediate: FFN width of the text transformer. Defaults to
            ``4 * text_width`` when ``None``.
        use_causal_mask: Whether the text tower attends causally. ``False`` for
            the MobileCLIP2 series, ``True`` for MobileCLIP-S3/S4.
        logit_scale_init: Initial value of the RAW ``logit_scale`` weight (a log
            temperature). Defaults to ``log(1 / 0.07)``.
        logit_scale_max: Upper clip applied to ``exp(logit_scale)`` on use.
            Defaults to 100.0.
        dropout_rate: Dropout inside both towers. Defaults to 0.0.
        attention_dropout_rate: Attention dropout in the text tower. Defaults
            to 0.0.
        image_encoder_kwargs: Extra keyword arguments forwarded to
            :class:`FastVitImageEncoder` (e.g. ``layers``, ``embed_dims``,
            ``drop_path_rate``) — used to build reduced-depth towers for tests.
        image_encoder: An already-constructed image tower to install instead of
            building one from the scalar fields. Used by :meth:`from_config`;
            supplying it is what makes a non-default tower round-trip as itself.
        text_encoder: An already-constructed text tower, as above.
        variant: Optional variant name recorded for provenance. Set
            automatically by :meth:`from_variant`.
        **kwargs: Forwarded to :class:`keras.Model`.

    Raises:
        ValueError: If any dimension is non-positive, if ``text_heads`` does not
            divide ``text_width``, or if ``logit_scale_max`` is not positive.

    Input shape:
        A dict ``{'image': (B, H, W, 3), 'text': (B, context_length)}``.

    Output shape:
        A dict with ``image_features`` / ``text_features`` of shape
        ``(B, embed_dim)``, ``logits_per_image`` / ``logits_per_text`` of shape
        ``(B, B)``, and a scalar ``logit_scale``.

    Example:
        >>> import numpy as np
        >>> model = MobileClipV2Model.from_variant('mobileclip2_s0')
        >>> out = model(
        ...     {'image': np.zeros((2, 256, 256, 3), dtype='float32'),
        ...      'text': np.zeros((2, 77), dtype='int32')},
        ...     training=False,
        ... )
        >>> sorted(out)
        ['image_features', 'logit_scale', 'logits_per_image', 'logits_per_text', 'text_features']
    """

    def __init__(
            self,
            embed_dim: int = 512,
            image_backbone: str = 'mci0',
            image_size: int = DEFAULT_IMAGE_SIZE,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
            context_length: int = DEFAULT_CONTEXT_LENGTH,
            text_width: int = 512,
            text_heads: int = 8,
            text_layers: int = 12,
            text_intermediate: Optional[int] = None,
            use_causal_mask: bool = False,
            logit_scale_init: float = _DEFAULT_LOGIT_SCALE_INIT,
            logit_scale_max: float = _LOGIT_SCALE_MAX,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            image_encoder_kwargs: Optional[Dict[str, Any]] = None,
            image_encoder: Optional[FastVitImageEncoder] = None,
            text_encoder: Optional[MobileClipTextEncoder] = None,
            variant: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = int(embed_dim)
        self.image_backbone = str(image_backbone)
        self.image_size = int(image_size)
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.text_width = int(text_width)
        self.text_heads = int(text_heads)
        self.text_layers = int(text_layers)
        self.text_intermediate = (
            int(_TEXT_MLP_RATIO * self.text_width) if text_intermediate is None
            else int(text_intermediate)
        )
        self.use_causal_mask = bool(use_causal_mask)
        self.logit_scale_init = float(logit_scale_init)
        self.logit_scale_max = float(logit_scale_max)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.image_encoder_kwargs = dict(image_encoder_kwargs or {})
        self.variant = None if variant is None else str(variant)

        self._validate_config()

        # ---- CREATE both towers in __init__ (unbuilt) --------------------
        # A tower passed in explicitly is installed as-is. `from_config` uses
        # this route; the towers are NEVER substituted after construction,
        # because Keras refuses a post-build sub-layer swap and a pre-build one
        # leaves the discarded tower's variables reachable through tracking.
        self.image_encoder = (
            image_encoder if image_encoder is not None
            else FastVitImageEncoder.from_variant(
                self.image_backbone,
                input_shape=(self.image_size, self.image_size, 3),
                projection_dim=self.embed_dim,
                dropout_rate=self.dropout_rate,
                name=IMAGE_TOWER_NAME,
                **self.image_encoder_kwargs,
            )
        )
        self.text_encoder = (
            text_encoder if text_encoder is not None
            else MobileClipTextEncoder(
                vocab_size=self.vocab_size,
                max_seq_len=self.context_length,
                embed_dim=self.text_width,
                num_layers=self.text_layers,
                num_heads=self.text_heads,
                intermediate_size=self.text_intermediate,
                projection_dim=self.embed_dim,
                use_causal_mask=self.use_causal_mask,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=TEXT_TOWER_NAME,
            )
        )

        # The weight itself is created in `build()` (the `models/clip` CLIP
        # precedent), not here.
        self.logit_scale = None

        logger.info(
            f"MobileClipV2Model: variant={self.variant}, embed_dim="
            f"{self.embed_dim}, image_backbone={self.image_backbone}@"
            f"{self.image_size}px, text={self.text_width}w/{self.text_heads}h/"
            f"{self.text_layers}L/{self.text_intermediate}ffn, "
            f"use_causal_mask={self.use_causal_mask}"
        )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the resolved configuration.

        Raises:
            ValueError: If a dimension is non-positive, if ``text_heads`` does
                not divide ``text_width``, or if ``logit_scale_max`` is not
                positive.
        """
        for name, value in (
                ('embed_dim', self.embed_dim),
                ('image_size', self.image_size),
                ('vocab_size', self.vocab_size),
                ('context_length', self.context_length),
                ('text_width', self.text_width),
                ('text_heads', self.text_heads),
                ('text_layers', self.text_layers),
                ('text_intermediate', self.text_intermediate),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        # Enforced downstream by the attention sub-layer, but its error message
        # names neither this model's parameter names nor this model's values.
        if self.text_width % self.text_heads != 0:
            raise ValueError(
                f"text_heads must divide text_width, got text_width="
                f"{self.text_width}, text_heads={self.text_heads} "
                f"(remainder {self.text_width % self.text_heads})"
            )

        if self.logit_scale_max <= 0.0:
            raise ValueError(
                f"logit_scale_max must be positive, got {self.logit_scale_max}"
            )
        for name, rate in (
                ('dropout_rate', self.dropout_rate),
                ('attention_dropout_rate', self.attention_dropout_rate),
        ):
            if not 0.0 <= rate < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {rate}")

    # ------------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Create ``logit_scale`` and build both towers.

        Args:
            input_shape: Either a dict with ``'image'`` and/or ``'text'`` keys
                mapping to per-tower shapes, or anything else — in which case
                each tower is built on the shape implied by this model's own
                configuration.
        """
        if self.built:
            return

        # This model's ONLY own weight. Created here, not in `__init__`.
        self.logit_scale = self.add_weight(
            name='logit_scale',
            shape=(),
            initializer=initializers.Constant(self.logit_scale_init),
            trainable=True,
        )

        image_shape: Any = None
        text_shape: Any = None
        if isinstance(input_shape, dict):
            image_shape = input_shape.get('image')
            text_shape = input_shape.get('text')

        if image_shape is None:
            image_shape = (None, self.image_size, self.image_size, 3)
        if text_shape is None:
            text_shape = (None, self.context_length)

        # `built` is checked PER TOWER, not just on self. On a `.keras` load the
        # towers arrive ALREADY BUILT (each carries its own build config) and
        # `MobileClipTextEncoder.build` has no idempotence guard of its own —
        # calling it a second time re-enters `LayerNormalization.build`, which
        # tries to `add_weight` on a locked tracker and raises
        # "You cannot add new elements of state to a layer that is already
        # built". Do NOT "fix" that by editing the v1 text encoder.
        if not self.image_encoder.built:
            self.image_encoder.build(tuple(image_shape))
        if not self.text_encoder.built:
            self.text_encoder.build(tuple(text_shape))

        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        """Return the shapes needed to rebuild this model's state on load.

        Keras' generic implementation cannot round-trip a DICT input-shape
        spec — it warns ``the model cannot be built automatically in
        build_from_config`` and leaves the restored model unbuilt. Both towers'
        shapes are fully determined by this model's own config, so they are
        stated explicitly here.

        Returns:
            A dictionary with per-tower shapes.
        """
        return {
            'image_shape': [None, self.image_size, self.image_size, 3],
            'text_shape': [None, self.context_length],
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild this model's state from :meth:`get_build_config`'s output.

        Args:
            config: A dictionary produced by :meth:`get_build_config`.
        """
        self.build({
            'image': tuple(config['image_shape']),
            'text': tuple(config['text_shape']),
        })

    # ------------------------------------------------------------------
    # encoders
    # ------------------------------------------------------------------

    def encode_image(
            self,
            image: keras.KerasTensor,
            normalize: bool = True,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode a batch of images into the joint embedding space.

        Args:
            image: Image tensor ``(B, H, W, 3)``.
            normalize: Whether to L2-normalize the features. Leave ``True`` for
                anything that feeds :func:`compute_clip_logits` or a contrastive
                loss — that helper does NOT normalize internally.
            training: Keras training flag. Pass ``False`` explicitly for a
                deterministic forward.

        Returns:
            ``(B, embed_dim)`` image features.
        """
        features = self.image_encoder(image, training=training)
        if normalize:
            features = ops.normalize(features, axis=-1)
        return features

    def encode_text(
            self,
            text: keras.KerasTensor,
            normalize: bool = True,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode a batch of tokenized captions into the joint embedding space.

        Args:
            text: Integer token-id tensor ``(B, context_length)``.
            normalize: Whether to L2-normalize the features. See
                :meth:`encode_image`.
            training: Keras training flag.

        Returns:
            ``(B, embed_dim)`` text features.
        """
        features = self.text_encoder(text, training=training)
        if normalize:
            features = ops.normalize(features, axis=-1)
        return features

    def compute_logit_scale(self) -> keras.KerasTensor:
        """Return the temperature actually used to scale the logits.

        ``exp`` of the raw learnable weight, clipped to
        ``[0, logit_scale_max]``. The clip is load-bearing: an unbounded
        temperature turns a diverging run into ``inf`` logits and a ``nan``
        loss with no other observable symptom.

        Returns:
            A scalar tensor.
        """
        return ops.clip(
            ops.exp(self.logit_scale), 0.0, self.logit_scale_max
        )

    def call(
            self,
            inputs: Union[
                Dict[str, keras.KerasTensor],
                Tuple[keras.KerasTensor, ...],
            ],
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Encode both modalities and compute the symmetric CLIP logits.

        Args:
            inputs: A dict with keys ``'image'`` and/or ``'text'``, or a
                ``(images, texts)`` tuple.
            training: Keras training flag. Pass ``False`` explicitly for a
                deterministic forward — the image tower's stochastic-depth
                branches run their stochastic path at ``training=None``.

        Returns:
            A dict holding ``image_features`` ``(B, embed_dim)`` when images were
            given, ``text_features`` ``(B, embed_dim)`` when text was given, and,
            when BOTH were given, ``logits_per_image`` ``(B, B)``,
            ``logits_per_text`` ``(B, B)`` and the scalar ``logit_scale``.
        """
        if isinstance(inputs, dict):
            images = inputs.get('image')
            texts = inputs.get('text')
        else:
            images = inputs[0] if len(inputs) > 0 else None
            texts = inputs[1] if len(inputs) > 1 else None

        results: Dict[str, keras.KerasTensor] = {}
        image_features = None
        text_features = None

        if images is not None:
            image_features = self.encode_image(
                images, normalize=True, training=training)
            results['image_features'] = image_features
        if texts is not None:
            text_features = self.encode_text(
                texts, normalize=True, training=training)
            results['text_features'] = text_features

        if image_features is not None and text_features is not None:
            logit_scale = self.compute_logit_scale()
            # `compute_clip_logits` documents PRE-NORMALIZED inputs; both
            # features above went through `ops.normalize`.
            logits_per_image, logits_per_text = compute_clip_logits(
                image_features, text_features, logit_scale
            )
            results.update({
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text,
                'logit_scale': logit_scale,
            })

        return results

    def compute_output_shape(
            self,
            input_shape: Any,
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Output shapes from stored config — valid before the model is built.

        Args:
            input_shape: A dict with ``'image'`` and/or ``'text'`` keys.

        Returns:
            A dict mirroring :meth:`call`'s keys.
        """
        image_shape = None
        text_shape = None
        if isinstance(input_shape, dict):
            image_shape = input_shape.get('image')
            text_shape = input_shape.get('text')

        shapes: Dict[str, Tuple[Optional[int], ...]] = {}
        image_batch = None if image_shape is None else tuple(image_shape)[0]
        text_batch = None if text_shape is None else tuple(text_shape)[0]

        if image_shape is not None:
            shapes['image_features'] = (image_batch, self.embed_dim)
        if text_shape is not None:
            shapes['text_features'] = (text_batch, self.embed_dim)
        if image_shape is not None and text_shape is not None:
            shapes['logits_per_image'] = (image_batch, text_batch)
            shapes['logits_per_text'] = (text_batch, image_batch)
            shapes['logit_scale'] = ()
        return shapes

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full configuration for serialization.

        Returns:
            A dictionary containing every constructor parameter. The two towers
            are serialized EXPLICITLY (not merely as a variant name) so a
            checkpoint keeps describing the network it was trained with.
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'image_backbone': self.image_backbone,
            'image_size': self.image_size,
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'text_width': self.text_width,
            'text_heads': self.text_heads,
            'text_layers': self.text_layers,
            'text_intermediate': self.text_intermediate,
            'use_causal_mask': self.use_causal_mask,
            'logit_scale_init': self.logit_scale_init,
            'logit_scale_max': self.logit_scale_max,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            # Kept even though the serialized towers below already carry the
            # architecture it produced: `get_config()` must name EVERY
            # constructor parameter, so a reader can reconstruct the call that
            # was made, not merely a network that behaves like it.
            'image_encoder_kwargs': dict(self.image_encoder_kwargs),
            'variant': self.variant,
            # The towers are reconstructed from their OWN serialized configs
            # rather than from this model's scalars, so a reduced-depth or
            # otherwise-overridden image tower round-trips as itself.
            'image_encoder': keras.saving.serialize_keras_object(
                self.image_encoder),
            'text_encoder': keras.saving.serialize_keras_object(
                self.text_encoder),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileClipV2Model":
        """Rebuild the model from a serialized configuration.

        Both towers are reconstructed with
        :func:`keras.saving.deserialize_keras_object` and handed to
        ``__init__`` as objects, so the restored model holds exactly the towers
        the checkpoint describes rather than fresh ones re-derived from the
        scalar fields.

        Args:
            config: A dictionary produced by :meth:`get_config`.

        Returns:
            A new :class:`MobileClipV2Model`.
        """
        config = dict(config)
        config['image_encoder'] = keras.saving.deserialize_keras_object(
            config.get('image_encoder'))
        config['text_encoder'] = keras.saving.deserialize_keras_object(
            config.get('text_encoder'))
        return cls(**config)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            **kwargs: Any,
    ) -> "MobileClipV2Model":
        """Create a model from a :data:`MODEL_VARIANTS` key.

        Args:
            variant: One of ``'mobileclip2_s0'``, ``'mobileclip2_s2'``,
                ``'mobileclip2_s3'``, ``'mobileclip2_s4'``, ``'mobileclip_s3'``,
                ``'mobileclip_s4'``.
            **kwargs: Any constructor keyword, overriding the variant row.

        Returns:
            The configured dual encoder.

        Raises:
            ValueError: If ``variant`` is not recognized.
        """
        row = _resolve_model_variant(variant)
        config: Dict[str, Any] = {
            'embed_dim': row['embed_dim'],
            'image_backbone': row['image_backbone'],
            'image_size': DEFAULT_IMAGE_SIZE,
            'vocab_size': DEFAULT_VOCAB_SIZE,
            'context_length': DEFAULT_CONTEXT_LENGTH,
            'text_width': row['text_width'],
            'text_heads': row['text_heads'],
            'text_layers': row['text_layers'],
            'text_intermediate': _TEXT_MLP_RATIO * row['text_width'],
            'use_causal_mask': row['use_causal_mask'],
            'variant': variant,
        }
        config.update(kwargs)
        return cls(**config)


# ---------------------------------------------------------------------


def create_mobile_clip_v2(
        variant: str = 'mobileclip2_s0',
        **overrides: Any,
) -> MobileClipV2Model:
    """Create a MobileCLIP2 dual-encoder model.

    Args:
        variant: A key of :data:`MODEL_VARIANTS`. Defaults to
            ``'mobileclip2_s0'``.
        **overrides: Any :class:`MobileClipV2Model` constructor keyword, e.g.
            ``dropout_rate``, ``logit_scale_init``, ``image_encoder_kwargs``.

    Returns:
        The configured dual encoder.

    Raises:
        ValueError: If ``variant`` is not recognized.
    """
    return MobileClipV2Model.from_variant(variant, **overrides)

# ---------------------------------------------------------------------
