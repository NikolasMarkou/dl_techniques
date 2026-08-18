"""
MobileCLIP2 dual encoder — the faithful port, pairing the FastViT MCi image
tower with the shared CLIP text transformer.

This is the weights-faithful half of the package. Its sibling
``mobile_clip_v1.py`` deliberately substitutes ``keras.applications`` CNNs for
the MCi trunk under its own D-001; this class instead builds the real tower from
``models/fastvit/``. Neither deprecates the other.

MobileCLIP's efficiency comes from what the image tower does at training time
versus what it costs at inference. FastViT is structurally reparameterizable: a
block is trained with several parallel branches — an over-parameterized
convolution, a scale branch, an implicit skip — and every one of them is an
affine map over the same input, so at inference they collapse algebraically into
a single convolution of the same kernel size. The network is therefore wide and
easy to optimize during training and narrow and cheap afterwards, with no
approximation in between. The second lever is where attention is spent: token
mixing in the early, high-resolution stages is done by a depthwise RepMixer,
which is itself reparameterizable, and self-attention appears only in the last
stage where the token count has already been reduced enough for its quadratic
cost to be affordable. MobileCLIP2's own advance over MobileCLIP is a training
recipe — a stronger captioner and teacher ensemble for multi-modal reinforced
training — not an architectural change, which is precisely why an
architecture-only port such as this one can be structurally faithful and still
make no accuracy claim.

The contrastive epilogue is standard CLIP: L2-normalized features from both
towers, a learnable temperature, and the symmetric logits matrix. The one
structural subtlety is that **there is no separate image projection layer**.
MobileCLIP's open_clip configs set ``timm_pool: avg`` with ``timm_proj: null``,
so the trunk is instantiated at ``num_classes=embed_dim`` and its terminal
``Dense`` *is* the CLIP image projection. ``embed_dim`` is injected as
``projection_dim`` into both sub-configs for that reason and must never be
tabulated inside them; ``_validate_config`` rejects a ``projection_dim`` that
appears there, because stacking another projection on top of the trunk's would
be a second, unfaithful one. Two more naming hazards sit one nesting level apart
and are worth stating outright: ``text_config['embed_dim']`` is the text
transformer's width, not the joint space, and ``image_config['variant']``
(``'mci0'``) is FastViT's own kwarg, a different thing from the model-level
variant name (``'mobileclip2_s0'``).

Normalization is placed at the model, not in the towers. Both encoders return
raw features from their own ``call``; ``encode_image`` and ``encode_text``
normalize, because ``compute_clip_logits`` expects already-normalized inputs and
does not normalize internally. The temperature is a single scalar weight holding
a log, created in ``build`` rather than ``__init__``, and read through
``compute_logit_scale``, which exponentiates and clips to ``logit_scale_max``.
That clip carries weight: without it a diverging temperature yields ``inf``
logits and a ``nan`` loss and nothing else observably wrong.

Causal masking in the text tower is the sole reason two families are tabulated.
The ``mobileclip2_s*`` rows are non-causal (their JSON configs set
``no_causal_mask: true``) while the earlier ``mobileclip_s3``/``mobileclip_s4``
rows are causal over the same image backbones. Every row is transcribed from one
supplied JSON file and keyed by that file's name, so a row is a checkable
transcription rather than a re-derivation of itself.

Serialization is where most of this class's non-obvious code lives. Both towers
are serialized as objects in ``get_config`` and handed back to ``__init__``
already constructed, so a reduced-depth or otherwise-overridden tower round-trips
as itself instead of being rebuilt from its config; the towers are never
substituted after construction, since Keras refuses a post-build sub-layer swap
and a pre-build one leaves the discarded tower's variables reachable through
tracking. ``build`` checks ``built`` per tower rather than only on ``self``,
because on a ``.keras`` load the towers arrive already built and the shared v1
text encoder has no idempotence guard of its own — a second ``build`` re-enters
``LayerNormalization.build`` and raises. ``get_build_config`` states both shapes
explicitly because Keras' generic implementation cannot round-trip a dict
input-shape spec and would leave the restored model unbuilt. And the sequence
fields of ``image_config`` are coerced back to tuples on the way in, since JSON
returns them as lists and a restored model would otherwise compare unequal to
the one it was saved from.

Two behavioural choices differ from v1 on purpose. With ``output_dict=False``
this model returns a five-tuple rather than v1's three, because dropping to
``(image, text, logit_scale)`` would silently discard both logits matrices —
which v1 never computes and this class always does. And a deterministic forward
requires passing ``training=False`` explicitly: the image tower's stochastic-depth
branches take their stochastic path at ``training=None``.

No pretrained weights are ported. ``create_mobile_clip_v2(pretrained=True)``
raises ``NotImplementedError`` rather than handing back a randomly initialized
model. See the package ``README.md`` §16 for the deviations that void any
comparison against published numbers.

References:
    - Vasu et al., 2023. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. (https://arxiv.org/abs/2311.17049)
    - Faghri et al., 2025. MobileCLIP2: Improving Multi-Modal Reinforced
      Training. (https://arxiv.org/abs/2508.20691)
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Ding et al., 2021. RepVGG: Making VGG-style ConvNets Great Again.
      (https://arxiv.org/abs/2101.03697)
    - Radford et al., 2021. Learning Transferable Visual Models From Natural
      Language Supervision. (https://arxiv.org/abs/2103.00020)
"""

import copy
import math
import keras
from keras import ops, initializers
from typing import Optional, Union, Tuple, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.clip_utils import compute_clip_logits
# The image tower is a standalone backbone package, not a private component of
# this model: `models/fastvit/` owns the faithful timm `FastVit` MCi
# transcription and is usable on its own. Its terminal `Dense(projection_dim)`
# IS the CLIP image projection — do NOT stack another projection on top of it.
from dl_techniques.models.fastvit import FastVitImageEncoder

# DECISION plan-2026-08-13T183738-24486492/D-001
# DECISION plan-2026-08-14T135600-mcsplit/D-001  (amends the above: this used to
# be a CROSS-package import from `models/mobile_clip_v2/`; the packages have
# since been merged, so it is now an in-package one. The rule below is unchanged.)
# The text tower is SHARED with v1 (`components.py`) rather than re-implemented.
#
# DO NOT re-implement a text transformer in this module, and do not copy
# `MobileClipTextEncoder` here "so v2 stands on its own". Either move would
# create a THIRD block->keep causal-mask adapter site, which SYSTEM.md:171
# records as a mandatory trigger for promoting that adapter into a keep-polarity
# `MaskFactory` variant — an unrelated refactor with its own blast radius.
# See decisions.md D-001.
#
# WHY A SHARED IMPORT AND NOT A COPY: `MobileClipTextEncoder` owns one of
# exactly TWO block->keep causal-mask adapter sites in `src/` (the other is
# `layers/heads/vlm/factory.py`). Re-implementing the text tower here would
# create that third site for no architectural gain: the layer is already
# dimension-generic (MEASURED at 768/12/3072 and 512/8/2048) and already carries
# the graph-safe `MaskFactory.create_causal_mask` path that `ops.tril` cannot
# provide on this stack.
#
# Note that sharing it does NOT make v2 inherit v1's non-fidelity: v1's
# deliberate `keras.applications` substitution (its own D-001) is confined to
# the IMAGE branch in `components.py`. The text tower is a plain CLIP
# transformer and is faithful for both.
from .components import MobileClipTextEncoder

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

#: Shared by every row of :attr:`MobileClipV2Model.MODEL_VARIANTS`.
DEFAULT_VOCAB_SIZE = 49408
DEFAULT_CONTEXT_LENGTH = 77
DEFAULT_IMAGE_SIZE = 256

#: The reference text tower's FFN expansion. Used ONLY to fill
#: ``text_config['intermediate_size']`` when a caller omits it — every tabulated
#: row states the width as a literal, so the JSON oracle checks a transcription
#: rather than agreeing with a re-derivation of itself.
_TEXT_MLP_RATIO = 4

#: Sub-dict keys `_validate_config` requires. Absent ones would otherwise
#: surface as a bare `KeyError` from inside validation, or as a `TypeError`
#: raised by the encoder several frames away from the mistake.
_REQUIRED_IMAGE_CONFIG_KEYS = ('variant', 'input_shape')
_REQUIRED_TEXT_CONFIG_KEYS = (
    'vocab_size', 'max_seq_len', 'embed_dim', 'num_layers', 'num_heads',
)

#: `image_config` entries that arrive as tuples but come back from JSON as
#: lists. Coerced on the way in so `get_config()` is a FIXED POINT and a
#: restored model compares equal to the one it was saved from.
_IMAGE_CONFIG_SEQUENCE_KEYS = (
    'input_shape', 'layers', 'embed_dims', 'mlp_ratios', 'token_mixers',
    'se_downsamples', 'pos_embs',
)


# ---------------------------------------------------------------------


def _resolve_model_variant(variant: str) -> Dict[str, Any]:
    """Look up a variant name in :attr:`MobileClipV2Model.MODEL_VARIANTS`.

    The table lives on the class (v1's convention); this helper stays
    module-level and reads it at call time, so there is exactly ONE place that
    formats the "unknown variant" error.

    Args:
        variant: A key of the variant table, e.g. ``'mobileclip2_s0'``.

    Returns:
        A DEEP copy of the variant's configuration dictionary. Deep, because a
        shallow one would hand out the class table's own nested sub-dicts and
        the caller would then mutate them process-wide.

    Raises:
        ValueError: If ``variant`` is not a known variant name.
    """
    key = str(variant)
    table = MobileClipV2Model.MODEL_VARIANTS
    if key not in table:
        raise ValueError(
            f"Unknown MobileCLIP2 variant {variant!r}. Available: "
            f"{sorted(table)}."
        )
    return copy.deepcopy(table[key])


@keras.saving.register_keras_serializable()
class MobileClipV2Model(keras.Model):
    """
    MobileCLIP2 dual encoder — FastViT (MCi) image tower + CLIP text tower.

    The faithful MobileCLIP port (Faghri et al., 2025, arXiv:2508.20691). Pairs
    the FastViT MCi image tower of :mod:`dl_techniques.models.fastvit` with the
    OpenCLIP-shaped text transformer of :mod:`.components` — the same one v1
    uses — and adds the CLIP epilogue: L2-normalized features, a learnable
    temperature, and the symmetric logits matrix.

    Its sibling :class:`~.mobile_clip_v1.MobileClipModel` is deliberately
    non-faithful on the image side (it substitutes ``keras.applications``
    backbones for the MCi tower under its own D-001) and is neither deprecated
    nor changed by this class. See the package ``README.md`` §17.

    **Architecture**:
    ```
    {'image': (B, 256, 256, 3)}      {'text': (B, 77)}
              ↓                            ↓
     FastVitImageEncoder          MobileClipTextEncoder
              ↓                            ↓
        (B, embed_dim)               (B, embed_dim)
              ↓                            ↓
        L2 Normalization            L2 Normalization
              └────────── Similarity ──────┘
                 scale = clip(exp(logit_scale), 0, logit_scale_max)
    ```

    Model Variants:
    --------------
    - MobileCLIP2-S0/S2/S3/S4: the ``mobileclip2_s*`` rows, **non-causal** text
      towers (their JSON configs set ``"no_causal_mask": true``).
    - MobileCLIP-S3/S4: the earlier ``mobileclip_s3``/``mobileclip_s4`` rows,
      **causal** text towers over the same image backbones.

    That single flag is the only reason both families are tabulated — do NOT
    "simplify" it away.

    .. note::
        **There is no separate image projection.** The image tower's terminal
        ``Dense`` IS the CLIP image projection: MobileCLIP's open_clip configs
        set ``"timm_pool": "avg"`` with ``"timm_proj": null``, so the trunk is
        instantiated at ``num_classes=embed_dim``. ``embed_dim`` is therefore
        injected as ``projection_dim`` into both sub-configs and must never be
        tabulated inside them. Stacking another projection on top would be a
        second, unfaithful one.

    .. note::
        This class is architecture-only. No pretrained weights are ported and it
        makes **no accuracy claim**; see ``README.md`` §16 (deviations X-1..X-5)
        before quoting it against any published number.

    Args:
        embed_dim: Integer, width of the joint image-text embedding space. Both
            towers project into it. Must be positive.
        image_config: Dictionary of :class:`FastVitImageEncoder` constructor
            keywords. Requires ``'variant'`` (an MCi name such as ``'mci0'``)
            and ``'input_shape'``; may carry any tower override such as
            ``'layers'`` or ``'drop_path_rate'``. ``projection_dim`` is injected
            from ``embed_dim`` and must not appear here.
        text_config: Dictionary of :class:`MobileClipTextEncoder` constructor
            keywords. Requires ``'vocab_size'``, ``'max_seq_len'``,
            ``'embed_dim'``, ``'num_layers'``, ``'num_heads'``;
            ``'intermediate_size'`` is filled with ``4 * embed_dim`` when
            omitted. ``projection_dim`` is injected, as above.
        logit_scale_init: Float, initial value of the RAW ``logit_scale`` weight
            (a log temperature). Defaults to ln(1/0.07) ≈ 2.66.
        output_dict: Boolean, whether to return outputs as a dictionary.
            Defaults to True.
        logit_scale_max: Float, upper clip applied to ``exp(logit_scale)`` on
            use. Defaults to 100.0.
        image_encoder: An already-constructed image tower to install instead of
            building one from ``image_config``. Used by :meth:`from_config`;
            supplying it is what makes a non-default tower round-trip as itself.
        text_encoder: An already-constructed text tower, as above.
        variant: Optional variant name recorded for provenance. Set
            automatically by :meth:`from_variant`.
        **kwargs: Additional arguments for the Model base class.

    Input shape:
        Dictionary with keys:
        - 'image': 4D tensor `(batch_size, height, width, 3)`
        - 'text': 2D tensor `(batch_size, sequence_length)`

        A `(images, texts)` tuple is also accepted.

    Output shape:
        If output_dict=True: Dictionary with keys 'image_features',
        'text_features', and — when BOTH modalities were given —
        'logits_per_image', 'logits_per_text', 'logit_scale'.
        If output_dict=False: 5-tuple in that same key order, with None for
        anything absent.

    Attributes:
        image_encoder: FastVitImageEncoder instance.
        text_encoder: MobileClipTextEncoder instance.
        logit_scale: Learnable temperature, created in `build()`.

    Example:
        ```python
        # Create from variant
        model = MobileClipV2Model.from_variant('mobileclip2_s0')

        # Override ONE sub-config field. `from_variant` replaces a sub-dict
        # wholesale (it does a top-level `config.update(kwargs)`), so merge:
        row = MobileClipV2Model.MODEL_VARIANTS['mobileclip2_s0']
        model = MobileClipV2Model.from_variant(
            'mobileclip2_s0',
            text_config={**row['text_config'], 'num_layers': 2},
        )

        # A reduced-depth image tower, for tests on a small card
        model = MobileClipV2Model.from_variant(
            'mobileclip2_s3',
            image_config={**row['image_config'], 'layers': (1, 1, 1, 1, 1)},
        )

        # Use model
        outputs = model(
            {'image': keras.random.normal((2, 256, 256, 3)),
             'text': keras.random.randint((2, 77), 0, 49408)},
            training=False,
        )
        ```

    Note:
        Both towers return RAW, un-normalized features from their own ``call``.
        Normalization happens in :meth:`encode_image` / :meth:`encode_text`,
        because :func:`~dl_techniques.utils.clip_utils.compute_clip_logits`
        expects already-normalized inputs and does not normalize internally.
    """

    # PROVENANCE — one row per SUPPLIED JSON config file, keyed by that file's
    # own name, so each row is a checkable transcription rather than a
    # re-derivation. `use_causal_mask` is the NEGATION of the JSON's
    # `no_causal_mask` field.
    #
    # TWO NAMING HAZARDS, both one nesting level apart:
    #   * `text_config['embed_dim']` is the TEXT WIDTH, not the joint space.
    #     The joint space is the row's top-level `embed_dim`.
    #   * `image_config['variant']` ('mci0') is FastViT's kwarg name and is a
    #     DIFFERENT "variant" from the model-level one ('mobileclip2_s0').
    MODEL_VARIANTS = {
        "mobileclip2_s0": {
            "embed_dim": 512,
            "image_config": {
                "variant": "mci0",  # timm_model_name minus the 'fastvit_' prefix
                "input_shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
            },
            "text_config": {
                "vocab_size": DEFAULT_VOCAB_SIZE,
                "max_seq_len": DEFAULT_CONTEXT_LENGTH,  # context_length
                "embed_dim": 512,  # text_cfg.width — the TEXT width
                "num_layers": 12,  # text_cfg.layers
                "num_heads": 8,  # text_cfg.heads
                "intermediate_size": 2048,  # 4 * width, stated as a literal
                "use_causal_mask": False,  # not no_causal_mask
            },
        },
        "mobileclip2_s2": {
            "embed_dim": 512,
            "image_config": {
                "variant": "mci2",
                "input_shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
            },
            "text_config": {
                "vocab_size": DEFAULT_VOCAB_SIZE,
                "max_seq_len": DEFAULT_CONTEXT_LENGTH,
                "embed_dim": 512,
                "num_layers": 12,
                "num_heads": 8,
                "intermediate_size": 2048,
                "use_causal_mask": False,
            },
        },
        "mobileclip2_s3": {
            "embed_dim": 768,
            "image_config": {
                "variant": "mci3",
                "input_shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
            },
            "text_config": {
                "vocab_size": DEFAULT_VOCAB_SIZE,
                "max_seq_len": DEFAULT_CONTEXT_LENGTH,
                "embed_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "intermediate_size": 3072,
                "use_causal_mask": False,
            },
        },
        "mobileclip2_s4": {
            "embed_dim": 768,
            "image_config": {
                "variant": "mci4",
                "input_shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
            },
            "text_config": {
                "vocab_size": DEFAULT_VOCAB_SIZE,
                "max_seq_len": DEFAULT_CONTEXT_LENGTH,
                "embed_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "intermediate_size": 3072,
                "use_causal_mask": False,
            },
        },
        "mobileclip_s3": {
            "embed_dim": 768,
            "image_config": {
                "variant": "mci3",
                "input_shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
            },
            "text_config": {
                "vocab_size": DEFAULT_VOCAB_SIZE,
                "max_seq_len": DEFAULT_CONTEXT_LENGTH,
                "embed_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "intermediate_size": 3072,
                "use_causal_mask": True,  # the CAUSAL family
            },
        },
        "mobileclip_s4": {
            "embed_dim": 768,
            "image_config": {
                "variant": "mci4",
                "input_shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
            },
            "text_config": {
                "vocab_size": DEFAULT_VOCAB_SIZE,
                "max_seq_len": DEFAULT_CONTEXT_LENGTH,
                "embed_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "intermediate_size": 3072,
                "use_causal_mask": True,
            },
        },
    }

    def __init__(
            self,
            embed_dim: int,
            image_config: Dict[str, Any],
            text_config: Dict[str, Any],
            logit_scale_init: float = _DEFAULT_LOGIT_SCALE_INIT,
            output_dict: bool = True,
            logit_scale_max: float = _LOGIT_SCALE_MAX,
            image_encoder: Optional[FastVitImageEncoder] = None,
            text_encoder: Optional[MobileClipTextEncoder] = None,
            variant: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(image_config, dict):
            raise TypeError("image_config must be a dictionary")
        if not isinstance(text_config, dict):
            raise TypeError("text_config must be a dictionary")

        self.embed_dim = int(embed_dim)
        # DEEP copies. A shallow `.copy()` would leave the nested sub-dicts
        # shared with MODEL_VARIANTS, so mutating `model.image_config['layers']`
        # would rewrite the class table for the whole process.
        self.image_config = self._normalize_image_config(image_config)
        self.text_config = copy.deepcopy(dict(text_config))
        self.logit_scale_init = float(logit_scale_init)
        self.output_dict = bool(output_dict)
        self.logit_scale_max = float(logit_scale_max)
        self.variant = None if variant is None else str(variant)

        # `text_intermediate`'s old `None` default has no dict equivalent, so
        # the fill is applied to the STORED copy, before `get_config()` can
        # observe it. That way it round-trips instead of being re-derived.
        self.text_config.setdefault(
            'intermediate_size',
            _TEXT_MLP_RATIO * int(self.text_config.get('embed_dim', 0)),
        )

        self._validate_config()

        # ---- CREATE both towers in __init__ (unbuilt) --------------------
        # A tower passed in explicitly is installed as-is. `from_config` uses
        # this route; the towers are NEVER substituted after construction,
        # because Keras refuses a post-build sub-layer swap and a pre-build one
        # leaves the discarded tower's variables reachable through tracking.
        # Do NOT "simplify" this to always-build-then-swap because the configs
        # now look self-sufficient — that breaks the reduced-tower round trip.
        image_constructor_config = copy.deepcopy(self.image_config)
        text_constructor_config = copy.deepcopy(self.text_config)

        image_constructor_config['projection_dim'] = self.embed_dim
        text_constructor_config['projection_dim'] = self.embed_dim

        self.image_encoder = (
            image_encoder if image_encoder is not None
            else FastVitImageEncoder.from_variant(
                **image_constructor_config, name=IMAGE_TOWER_NAME)
        )
        self.text_encoder = (
            text_encoder if text_encoder is not None
            else MobileClipTextEncoder(
                **text_constructor_config, name=TEXT_TOWER_NAME)
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
    # Read-only views over the two sub-configs.
    #
    # v1 has no equivalent; these exist because the scalars they expose are read
    # by `_validate_config`, `get_build_config`, `compute_output_shape` and
    # `summary`, and by ~15 test assertions. Without them each of those becomes
    # a hand-written two-level dict lookup — which is where transcription errors
    # get in. There is deliberately NO `dropout_rate` property: dropout is
    # per-tower now, and a single accessor would have to pick one and would lie.
    # ------------------------------------------------------------------

    @property
    def image_backbone(self) -> str:
        """MCi backbone name of the image tower, e.g. ``'mci0'``."""
        return self.image_config['variant']

    @property
    def image_size(self) -> int:
        """Square input resolution of the image tower."""
        return self.image_config['input_shape'][0]

    @property
    def vocab_size(self) -> int:
        """Token vocabulary size of the text tower."""
        return self.text_config['vocab_size']

    @property
    def context_length(self) -> int:
        """Maximum text sequence length."""
        return self.text_config['max_seq_len']

    @property
    def text_width(self) -> int:
        """Hidden width of the text transformer (NOT :attr:`embed_dim`)."""
        return self.text_config['embed_dim']

    @property
    def text_heads(self) -> int:
        """Attention heads per text transformer layer."""
        return self.text_config['num_heads']

    @property
    def text_layers(self) -> int:
        """Number of text transformer layers."""
        return self.text_config['num_layers']

    @property
    def text_intermediate(self) -> int:
        """FFN width of the text transformer."""
        return self.text_config['intermediate_size']

    @property
    def use_causal_mask(self) -> bool:
        """Whether the text tower attends causally."""
        return self.text_config.get('use_causal_mask', True)

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_image_config(image_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-copy ``image_config`` and coerce its sequence fields to tuples.

        ``input_shape``, ``layers`` and friends are written as tuples but come
        back from a ``.keras`` round trip as LISTS, which would make a restored
        model's config compare unequal to the one it was saved from. Coercing on
        the way in makes :meth:`get_config` a fixed point.

        Args:
            image_config: The caller's image-tower configuration.

        Returns:
            A normalized deep copy.
        """
        normalized = copy.deepcopy(dict(image_config))
        for key in _IMAGE_CONFIG_SEQUENCE_KEYS:
            value = normalized.get(key)
            if isinstance(value, list):
                normalized[key] = tuple(value)
        input_shape = normalized.get('input_shape')
        if isinstance(input_shape, tuple):
            normalized['input_shape'] = tuple(
                None if dim is None else int(dim) for dim in input_shape
            )
        return normalized

    def _validate_config(self) -> None:
        """Validate the resolved configuration.

        Raises:
            ValueError: If a required sub-config key is missing, if a dimension
                is non-positive, if ``input_shape`` is malformed, if
                ``text_config['num_heads']`` does not divide
                ``text_config['embed_dim']``, or if ``logit_scale_max`` is not
                positive.
        """
        for label, config, required in (
                ('image_config', self.image_config, _REQUIRED_IMAGE_CONFIG_KEYS),
                ('text_config', self.text_config, _REQUIRED_TEXT_CONFIG_KEYS),
        ):
            missing = [key for key in required if key not in config]
            if missing:
                raise ValueError(
                    f"{label} is missing required key(s) {missing}. It must be "
                    f"a dict of the encoder's constructor keywords; see "
                    f"MobileClipV2Model.MODEL_VARIANTS for a worked example."
                )

        if 'projection_dim' in self.image_config or 'projection_dim' in self.text_config:
            raise ValueError(
                "projection_dim must NOT appear in image_config/text_config — "
                "it is injected from embed_dim, and the image tower's terminal "
                "Dense IS the CLIP image projection."
            )

        input_shape = self.image_config['input_shape']
        if len(input_shape) != 3 or any(
                dim is None or dim <= 0 for dim in input_shape):
            raise ValueError(
                f"image_config['input_shape'] must be a positive (H, W, C) "
                f"triple, got {input_shape!r}"
            )

        for name, value in (
                ('embed_dim', self.embed_dim),
                ("text_config['vocab_size']", self.vocab_size),
                ("text_config['max_seq_len']", self.context_length),
                ("text_config['embed_dim']", self.text_width),
                ("text_config['num_heads']", self.text_heads),
                ("text_config['num_layers']", self.text_layers),
                ("text_config['intermediate_size']", self.text_intermediate),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        # Enforced downstream by the attention sub-layer, but its error message
        # names neither this model's parameter names nor this model's values.
        if self.text_width % self.text_heads != 0:
            raise ValueError(
                f"text_config['num_heads'] must divide text_config['embed_dim'],"
                f" got embed_dim={self.text_width}, num_heads={self.text_heads} "
                f"(remainder {self.text_width % self.text_heads})"
            )

        if self.logit_scale_max <= 0.0:
            raise ValueError(
                f"logit_scale_max must be positive, got {self.logit_scale_max}"
            )
        for label, config in (
                ('image_config', self.image_config),
                ('text_config', self.text_config),
        ):
            for key in ('dropout_rate', 'attention_dropout_rate'):
                if key in config and not 0.0 <= config[key] < 1.0:
                    raise ValueError(
                        f"{label}['{key}'] must be in [0, 1), got {config[key]}"
                    )

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
            image_shape = (None,) + tuple(self.image_config['input_shape'])
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
        """
        return {
            'image_shape': [None] + list(self.image_config['input_shape']),
            'text_shape': [None, self.context_length],
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild this model's state from :meth:`get_build_config`'s output."""
        self.build({
            'image': tuple(config['image_shape']),
            'text': tuple(config['text_shape']),
        })

    def encode_image(
            self,
            image: keras.KerasTensor,
            normalize: bool = True,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode images to embedding vectors.

        Leave ``normalize=True`` for anything that feeds
        :func:`compute_clip_logits` or a contrastive loss — that helper does NOT
        normalize internally.
        """
        features = self.image_encoder(image, training=training)
        if normalize:
            features = ops.normalize(features, axis=-1)
        return features

    def encode_text(
            self,
            text: keras.KerasTensor,
            normalize: bool = True,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode text tokens to embedding vectors.
        """
        features = self.text_encoder(text, training=training)
        if normalize:
            features = ops.normalize(features, axis=-1)
        return features

    def compute_logit_scale(self) -> keras.KerasTensor:
        """
        Return the temperature actually used to scale the logits.

        ``exp`` of the raw learnable weight, clipped to
        ``[0, logit_scale_max]``. The clip is load-bearing: an unbounded
        temperature turns a diverging run into ``inf`` logits and a ``nan``
        loss with no other observable symptom.
        """
        return ops.clip(ops.exp(self.logit_scale), 0.0, self.logit_scale_max)

    def call(
            self,
            inputs: Union[
                Dict[str, keras.KerasTensor],
                Tuple[keras.KerasTensor, ...],
            ],
            training: Optional[bool] = None
    ) -> Union[Dict[str, keras.KerasTensor], Tuple[Any, ...]]:
        """
        Forward pass for the MobileCLIP2 model.

        Pass ``training=False`` explicitly for a deterministic forward — the
        image tower's stochastic-depth branches run their stochastic path at
        ``training=None``.
        """
        if isinstance(inputs, dict):
            images = inputs.get('image')
            texts = inputs.get('text')
        else:
            images = inputs[0] if len(inputs) > 0 else None
            texts = inputs[1] if len(inputs) > 1 else None

        image_features = (
            self.encode_image(images, normalize=True, training=training)
            if images is not None else None
        )
        text_features = (
            self.encode_text(texts, normalize=True, training=training)
            if texts is not None else None
        )

        logits_per_image = None
        logits_per_text = None
        logit_scale = None
        if image_features is not None and text_features is not None:
            logit_scale = self.compute_logit_scale()
            # `compute_clip_logits` documents PRE-NORMALIZED inputs; both
            # features above went through `ops.normalize`.
            logits_per_image, logits_per_text = compute_clip_logits(
                image_features, text_features, logit_scale
            )

        if not self.output_dict:
            # A 5-tuple in the documented key order, NOT v1's 3-tuple: dropping
            # to `(image, text, logit_scale)` would silently discard both logits
            # matrices, which v1 never computes and v2 always does.
            return (
                image_features, text_features,
                logits_per_image, logits_per_text, logit_scale,
            )

        results: Dict[str, keras.KerasTensor] = {}
        if image_features is not None:
            results['image_features'] = image_features
        if text_features is not None:
            results['text_features'] = text_features
        if logit_scale is not None:
            results.update({
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text,
                'logit_scale': logit_scale,
            })
        return results

    def compute_output_shape(
            self,
            input_shape: Any
    ) -> Union[Dict[str, Tuple[Optional[int], ...]], Tuple[Any, ...]]:
        """Output shapes from stored config — valid before the model is built.

        Mirrors :meth:`call`, INCLUDING its ``output_dict`` branch: returning a
        dict unconditionally would contradict a model built with
        ``output_dict=False``.
        """
        image_shape = None
        text_shape = None
        if isinstance(input_shape, dict):
            image_shape = input_shape.get('image')
            text_shape = input_shape.get('text')

        image_batch = None if image_shape is None else tuple(image_shape)[0]
        text_batch = None if text_shape is None else tuple(text_shape)[0]

        features_image = (
            None if image_shape is None else (image_batch, self.embed_dim))
        features_text = (
            None if text_shape is None else (text_batch, self.embed_dim))
        both = image_shape is not None and text_shape is not None
        logits_image = (image_batch, text_batch) if both else None
        logits_text = (text_batch, image_batch) if both else None
        scale = () if both else None

        if not self.output_dict:
            return (
                features_image, features_text,
                logits_image, logits_text, scale,
            )

        shapes: Dict[str, Tuple[Optional[int], ...]] = {}
        if features_image is not None:
            shapes['image_features'] = features_image
        if features_text is not None:
            shapes['text_features'] = features_text
        if both:
            shapes['logits_per_image'] = logits_image
            shapes['logits_per_text'] = logits_text
            shapes['logit_scale'] = scale
        return shapes

    @classmethod
    def from_variant(
            cls,
            variant: str,
            **kwargs: Any
    ) -> "MobileClipV2Model":
        """
        Create a MobileCLIP2 model from a predefined variant.

        ``kwargs`` override the row at the TOP level, so passing
        ``text_config=`` REPLACES the row's sub-dict wholesale. To change one
        field, merge explicitly — see the class docstring's example.
        """
        # `_resolve_model_variant` validates and returns a DEEP copy — a shallow
        # one would hand out the class table's own nested sub-dicts, and the
        # model would then mutate them process-wide.
        config = _resolve_model_variant(variant)
        config['variant'] = variant
        logger.info(f"Creating MobileCLIP2 model from variant '{variant}'")
        config.update(kwargs)
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Contains every constructor parameter. The two towers are serialized
        EXPLICITLY (not merely as a variant name) so a checkpoint keeps
        describing the network it was trained with, and so a reduced-depth or
        otherwise-overridden tower round-trips as itself.
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'image_config': copy.deepcopy(self.image_config),
            'text_config': copy.deepcopy(self.text_config),
            'logit_scale_init': self.logit_scale_init,
            'output_dict': self.output_dict,
            'logit_scale_max': self.logit_scale_max,
            'variant': self.variant,
            'image_encoder': keras.saving.serialize_keras_object(
                self.image_encoder),
            'text_encoder': keras.saving.serialize_keras_object(
                self.text_encoder),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileClipV2Model":
        """Create model from configuration.

        Both towers are reconstructed with
        :func:`keras.saving.deserialize_keras_object` and handed to ``__init__``
        as objects, so the restored model holds exactly the towers the
        checkpoint describes rather than fresh ones re-derived from the configs.
        """
        config = dict(config)
        config['image_encoder'] = keras.saving.deserialize_keras_object(
            config.get('image_encoder'))
        config['text_encoder'] = keras.saving.deserialize_keras_object(
            config.get('text_encoder'))
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)
        logger.info(f"MobileCLIP2 configuration:")
        logger.info(f"  - Variant: {self.variant or 'custom'}")
        logger.info(f"  - Embed dimension: {self.embed_dim}")
        logger.info(f"  - Image backbone: {self.image_backbone}")
        logger.info(f"  - Image size: {self.image_size}")
        logger.info(f"  - Text vocab size: {self.vocab_size}")
        logger.info(f"  - Text max seq len: {self.context_length}")
        logger.info(f"  - Text width: {self.text_width}")
        logger.info(f"  - Text layers: {self.text_layers}")
        logger.info(f"  - Text heads: {self.text_heads}")
        logger.info(f"  - Causal masking: {self.use_causal_mask}")
        logger.info(f"  - Output format: {'Dictionary' if self.output_dict else 'Tuple'}")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_mobile_clip_v2(
        variant: str = "mobileclip2_s0",
        pretrained: bool = False,
        **kwargs: Any
) -> MobileClipV2Model:
    """
    Convenience function to create MobileCLIP2 models.

    :raises NotImplementedError: If ``pretrained=True`` — no MobileCLIP2
        checkpoints ship with this package.
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained MobileCLIP2 weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_mobile_clip_v2('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )
    model = MobileClipV2Model.from_variant(variant, **kwargs)
    return model

# ---------------------------------------------------------------------
