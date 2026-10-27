"""``ImageTextMatchingHead``: a contrastive branch and a fine-grained fusion branch."""

import math
import numbers
import keras
from typing import Dict, List, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...ffn.factory import (
    FFN_REGISTRY,
    assemble_ffn_config,
    create_ffn_layer,
    FFNType,
)
from ...norms import NormalizationType
from ...norms.factory import create_normalization_layer
from ...sequence_pooling import SequencePooling
from .task_types import VLMTaskConfig
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _ffn_width_kwargs, _check_task_config_fields, _pool_sequence
from .base import BaseVLMHead


@register_dl_technique("dl_techniques.layers.heads.vlm.image_text_matching")
class ImageTextMatchingHead(BaseVLMHead):
    """
    Runs a contrastive branch and a fine-grained branch in parallel.

    This head answers two questions about the same pair of inputs. The
    contrastive branch projects both modalities into one L2-normalized space
    and returns a CLIP-style similarity matrix scaled by a learnable
    temperature. The fine-grained branch fuses the two pooled vectors and
    returns one match logit per pair. Both branches read the same pooled
    tensors, and ``call`` returns five keys covering both.

    No output is squashed. ``'logits'`` is the similarity matrix divided by
    the temperature, and ``'match_score'`` comes from a linear ``Dense(1)``,
    so it is a pre-sigmoid LOGIT, not confined to ``[0, 1]``. Train it with
    ``BinaryCrossentropy(from_logits=True)``, or apply ``keras.ops.sigmoid``
    yourself for a probability.

    It inherits from :class:`BaseVLMHead` and defaults ``fusion_strategy`` to
    ``"concatenation"``. It is the only head in this module that runs a
    post-fusion stack, so it owns that stack and its four arguments.

    **Architecture Overview:**

    .. code-block:: text

        vision_features            text_features
        (B, N_vis, D_vis)          (B, S, D_txt)
                │                          │
                ▼ mean axis 1              ▼ mean axis 1
          vision_pooled                 text_pooled
            (B, D_vis)                  (B, D_txt)
                │                          │
                └────────────┬─────────────┘
                             │ both feed both branches
                   ┌─────────┴─────────────┐
                   ▼                       ▼
        ┌─────────────────────┐ ┌─────────────────────┐
        │ vision_proj Dense   │ │ expand to (B, 1, D) │
        │ text_proj Dense     │ │ fusion              │
        │ L2 normalize both   │ │ squeeze axis 1      │
        │ v_norm @ t_norm.T   │ │ post_fusion_norm    │
        │ scale by temperature│ │ post_fusion_ffn     │
        │                     │ │ (optional)          │
        │                     │ │ similarity_head     │
        │                     │ │ Dense 1, linear     │
        └──────────┬──────────┘ └──────────┬──────────┘
                   ▼                       ▼
         'similarity_matrix' (B, B)  'match_score' (B,)
         'logits'            (B, B)
         'vision_embeddings' (B, projection_dim)
         'text_embeddings'   (B, projection_dim)

    The fused branch applies no dropout of its own after the fusion: the
    fusion layer and the post-fusion FFN each apply ``task_config.dropout_rate``
    inside themselves.

    The similarity matrix is computed across the batch, so ``similarity_matrix``
    and ``logits`` are ``(batch, batch)``. Row ``i``, column ``j`` is image
    ``i`` against caption ``j``.

    Each mean honours an optional keep mask (``'vision_mask'``,
    ``'text_mask'``, ``(batch, length)``, 1 = real): it averages the real
    positions only.

    Input shape:
        ``{'vision_features': (batch, num_vision, vision_width),
        'text_features': (batch, seq_len, text_width)}``, plus the optional
        masks. Rank-2 inputs are accepted and used unpooled; a mask for one
        is refused.

    Output shape:
        ``{'similarity_matrix': (batch, batch), 'logits': (batch, batch),
        'match_score': (batch,), 'vision_embeddings': (batch,
        projection_dim), 'text_embeddings': (batch, projection_dim)}``.
        ``'match_score'`` is one logit per pair.

    :param task_config: Configuration object for the task.
    :type task_config: VLMTaskConfig
    :param projection_dim: Width of the shared contrastive embedding space.
    :type projection_dim: int
    :param temperature: Starting value of the learnable temperature, a
        positive finite number. The logits are the similarity matrix divided
        by it. It is stored in the config as the starting value; the trained
        value is a weight and travels with the weights.
    :type temperature: float
    :param normalization_type: Which registered normalization to build for
        ``post_fusion_norm``.
    :type normalization_type: NormalizationType
    :param use_post_fusion_ffn: Whether to run an FFN after the
        post-fusion norm.
    :type use_post_fusion_ffn: bool
    :param ffn_type: Which registered FFN type to build after the fusion.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: Width multiplier for the post-fusion FFN.

        It reaches the FFN by one of two channels, chosen by ``ffn_type``
        (D-008 and D-020). A type that requires an explicit ``hidden_dim``
        receives ``hidden_dim * ffn_expansion_factor``. A type that derives
        its own width receives the factor itself and applies its own rule. Of
        the 8 registry types in the second group, only ``swiglu`` accepts it.
        For the other 7 (``kan``, ``mixer``, ``tversky``, ...) the concept does
        not apply, and the value does not reach them. It is never passed both
        ways.

        The default is ``ffn_type="mlp"``, which takes the first channel, so
        the default post-fusion width is
        ``hidden_dim * ffn_expansion_factor``. :class:`ImageCaptioningHead`
        defaults to ``swiglu`` and therefore takes the second.
    :type ffn_expansion_factor: int
    :param kwargs: Arguments for :class:`BaseVLMHead`. ``fusion_strategy``
        defaults to ``"concatenation"`` here.

    :ivar projection_dim: Width of the contrastive embedding space.
    :vartype projection_dim: int
    :ivar vision_proj: ``Dense`` into the contrastive space.
    :vartype vision_proj: keras.layers.Dense
    :ivar text_proj: ``Dense`` into the contrastive space.
    :vartype text_proj: keras.layers.Dense
    :ivar similarity_head: Linear ``Dense`` producing the fine-grained match
        logit.
    :vartype similarity_head: keras.layers.Dense
    :ivar post_fusion_norm: Normalization after the fusion.
    :vartype post_fusion_norm: keras.layers.Layer
    :ivar post_fusion_ffn: FFN after the norm, created when
        ``use_post_fusion_ffn`` is set.
    :vartype post_fusion_ffn: keras.layers.Layer
    :ivar temperature: Learnable scalar weight. A ``.keras`` round-trip
        restores its trained value with the other weights; the config holds
        the starting value (``initial_temperature``).
    :vartype temperature: keras.Variable
    :ivar initial_temperature: The constructor's ``temperature``.
    :vartype initial_temperature: float
    :ivar vision_pooling: Weightless ``SequencePooling('mean')``.
    :vartype vision_pooling: SequencePooling
    :ivar text_pooling: Weightless ``SequencePooling('mean')``.
    :vartype text_pooling: SequencePooling
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        projection_dim: int = 256,
        temperature: float = 0.07,
        normalization_type: NormalizationType = "layer_norm",
        use_post_fusion_ffn: bool = True,
        ffn_type: FFNType = "mlp",
        ffn_expansion_factor: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Default the fusion strategy, then create the task layers and weight.

        :param task_config: Configuration object for the task.
        :type task_config: VLMTaskConfig
        :param projection_dim: Width of the contrastive embedding space.
        :type projection_dim: int
        :param temperature: Starting value of the learnable temperature.
        :type temperature: float
        :param normalization_type: Which normalization to build after fusion.
        :type normalization_type: NormalizationType
        :param use_post_fusion_ffn: Whether to create and run the post-fusion
            FFN.
        :type use_post_fusion_ffn: bool
        :param ffn_type: Which registered FFN type to build.
        :type ffn_type: FFNType
        :param ffn_expansion_factor: Post-fusion FFN width multiplier.
        :type ffn_expansion_factor: int
        :param kwargs: Arguments for :class:`BaseVLMHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``temperature`` is not a positive finite
            number, or ``task_config`` sets ``vocab_size`` or ``num_classes``.
        """
        _check_task_config_fields(task_config, "ImageTextMatchingHead")
        if (isinstance(temperature, bool)
                or not isinstance(temperature, numbers.Real)
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError(
                f"ImageTextMatchingHead: temperature must be a positive finite "
                f"number (the logits are divided by it), got {temperature!r}"
            )
        kwargs.setdefault("fusion_strategy", "concatenation")
        super().__init__(task_config=task_config, **kwargs)

        self.normalization_type = normalization_type
        self.use_post_fusion_ffn = use_post_fusion_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self._create_post_fusion_layers()

        self.projection_dim = projection_dim
        self.vision_proj = keras.layers.Dense(projection_dim, name=f"{self.name}_vision_proj")
        self.text_proj = keras.layers.Dense(projection_dim, name=f"{self.name}_text_proj")
        # DECISION plan-2026-10-14T062616-81938e81/D-003
        # Linear: 'match_score' carries a match logit under its old name. Do
        # NOT put activation='sigmoid' back: a from_logits=True loss would
        # squash it twice, as for every class-like output of layers/heads.
        # See decisions.md D-003.
        self.similarity_head = keras.layers.Dense(
            1, name=f"{self.name}_similarity"
        )
        # DECISION plan-2026-10-14T062616-81938e81/D-057
        # Keep the starting value in the config. Do NOT drop it again on the
        # grounds that the weight is saved: the weight restores the TRAINED
        # value only where weights are loaded, and from_config(get_config())
        # of a head built with temperature=0.5 started at 0.07. See
        # decisions.md D-057.
        self.initial_temperature = float(temperature)
        self.temperature = self.add_weight(
            name=f"{self.name}_temperature",
            shape=(),
            initializer=keras.initializers.Constant(self.initial_temperature),
            trainable=True,
        )
        self.vision_pooling = SequencePooling(strategy="mean", name=f"{self.name}_vision_pooling")
        self.text_pooling = SequencePooling(strategy="mean", name=f"{self.name}_text_pooling")

    def _create_post_fusion_layers(self) -> None:
        """
        Create the post-fusion norm, and the FFN when it is asked for.

        ``post_fusion_norm`` is always created; ``post_fusion_ffn`` only when
        ``use_post_fusion_ffn`` is set. Nothing is built here; the shapes are
        not known until :meth:`build`. Called from ``__init__``, following
        the package rule that layers are created in ``__init__``.

        This is one of the two FFN construction sites in the module. The other
        is :class:`ImageCaptioningHead`'s per-layer decoder loop. The two must
        keep the same three rules. Otherwise the same ``ffn_type`` means two
        different things in one file. The rules are:

        * Pass ``hidden_dim`` only where the registry marks it required.
        * Forward ``ffn_expansion_factor`` to a type that derives its own width.
        * Read the output-width parameter name from the registry, using
          the helper :func:`_ffn_width_kwargs`.

        ``create_ffn_layer`` raises on a keyword the type does not declare, so
        the keyword set is filtered by ``assemble_ffn_config`` before it is
        sent.

        The D-034 note below records an open question about the ``hidden_dim``
        rule. Its training-quality impact is unmeasured. It stays unmeasurable
        only while no checkpoint holds a VLM head, so one such checkpoint makes
        it answerable as written. Re-derive that count rather than trusting it:
        ``find . -name '*.keras' -not -path './.git/*'``, then read each file's
        ``config.json`` for a VLM head class name.

        Do NOT transfer D-033's closure to this case. D-033 closes its question
        by unreachability, which is a proof no instrument can overturn. There
        is no unreachability argument here at all, because this path is the
        default one.

        :return: None.
        :rtype: None
        """
        self.post_fusion_norm = create_normalization_layer(
            self.normalization_type, name=f"{self.name}_post_fusion_norm"
        )

        if self.use_post_fusion_ffn:
            # The default path is unaffected by the rule below. `ffn_type`
            # defaults to "mlp", and `mlp` marks `hidden_dim` required, so the
            # conditional makes the identical call. The types that stay closed
            # here, and why, are listed once in the `ffn_type` docstring of
            # `ImageCaptioningHead`.
            #
            # DECISION plan-2026-07-30T081929-1645aa52/D-008
            # Pass `hidden_dim` only to an FFN type that requires it. Do NOT go
            # back to an unconditional keyword: that overrides a type's own
            # width rule, measured on swiglu as post-fusion FFN parameters of
            # 55296 / 110592 / 221184 at factor 2 / 4 / 8. See decisions.md D-008.

            # DECISION plan-2026-07-30T140922-8af1028f/D-034
            # STILL OPEN: the training-quality impact of D-008 and D-020 is
            # unmeasured. 0 of the 44 repo `.keras` files hold a VLM head, so
            # nothing can answer it yet. This is a DIFFERENT case from D-033's
            # closure; do NOT transfer it. See decisions.md D-034 and above.
            ffn_kwargs = {
                "dropout_rate": self.task_config.dropout_rate,
                "name": f"{self.name}_post_fusion_ffn",
                **_ffn_width_kwargs(self.ffn_type, self.hidden_dim),
            }
            _entry = FFN_REGISTRY.get(self.ffn_type, {})
            if "hidden_dim" in _entry.get("required_params", ()):
                ffn_kwargs["hidden_dim"] = (
                    self.hidden_dim * self.ffn_expansion_factor
                )
            elif "ffn_expansion_factor" in set(
                _entry.get("required_params", ())
            ) | set(_entry.get("optional_params", {})):
                # Of the 8 registry types that treat `hidden_dim` as optional,
                # only `swiglu` accepts this parameter. The other 7 (`kan`,
                # `mixer`, `tversky`, ...) size themselves from unrelated
                # parameters, so for them the factor does not apply at all.
                #
                # DECISION plan-2026-07-30T081929-1645aa52/D-020
                # Forward the factor itself to a type that derives its own
                # width. Do NOT drop it: `hidden_dim` was its only channel to
                # the FFN, and without this swiglu's post-fusion width sat at
                # 73728 for factor 2, 4 AND 8. See decisions.md D-020.
                ffn_kwargs["ffn_expansion_factor"] = self.ffn_expansion_factor

            # Filtering belongs here because `dropout_rate` is this head's own
            # default, not the caller's intent. This head exposes no `ffn_args`
            # surface, so the third argument is None. If one is ever added it
            # goes there, never into `ffn_kwargs`. See `assemble_ffn_config`
            # and its D-017 contract.
            #
            # Re-derive the count in the anchor below rather than trusting it.
            # Call `assemble_ffn_config(k, {"dropout_rate": 0.1, "name": "x"})`
            # for every key `k` in `FFN_REGISTRY` and count the results that
            # have no `dropout_rate`. Measured 6 of 21 on 2026-08-31.
            #
            # DECISION plan-2026-07-30T140922-8af1028f/D-019
            # Pre-filter the keyword set. `dropout_rate` is injected above, and
            # `assemble_ffn_config` drops it for 6 of the 21 registry types:
            # counting, gated_mlp, kan, logic, power_mlp, tversky. Do NOT drop
            # this filter; the factory raises on them. See decisions.md D-019.
            ffn_kwargs = assemble_ffn_config(self.ffn_type, ffn_kwargs)
            self.post_fusion_ffn = create_ffn_layer(self.ffn_type, **ffn_kwargs)
    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        ``call()`` mean-pools each modality to 2-D and projects both. It then
        expands the pooled tensors to ``(B, 1, D)``, because the fusion requires
        3-D. It squeezes axis 1 back off for the 2-D post-fusion stack.
        ``_build_fusion_stack`` is told about that squeeze, so the norm and FFN
        are built at the rank they actually see.

        :raises ValueError: If the post-fusion FFN produces a width other than
            ``hidden_dim``.

        ``temperature`` is already created by ``add_weight`` in ``__init__``, so
        it needs nothing here.

        :param input_shape: Shape dict with ``vision_features`` and
            ``text_features``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        vision_shape = tuple(input_shape["vision_features"])
        text_shape = tuple(input_shape["text_features"])
        batch = vision_shape[0]

        # Pooled to 2-D when the input is 3-D, otherwise passed through as-is.
        vision_pooled_shape = (batch, vision_shape[-1])
        text_pooled_shape = (batch, text_shape[-1])

        self.vision_proj.build(vision_pooled_shape)
        self.text_proj.build(text_pooled_shape)
        self.vision_pooling.build(vision_shape)
        self.text_pooling.build(text_shape)

        processed_shape = self._build_fusion_stack(
            [(batch, 1, vision_shape[-1]), (batch, 1, text_shape[-1])],
            squeeze_axis=1,
        )
        self.post_fusion_norm.build(processed_shape)
        if self.use_post_fusion_ffn:
            self.post_fusion_ffn.build(processed_shape)
            processed_shape = self.post_fusion_ffn.compute_output_shape(
                processed_shape)

            # The oracle is `compute_output_shape`, the very method the
            # wiring above consumes, so the assertion cannot pass while the
            # wiring is wrong. That required fixing
            # `SwiGLUFFN.compute_output_shape`, which reported its input
            # width (D-013). `ImageCaptioningHead` needs no such check: its
            # unprojected `x + ffn_output` residual add raises on the same
            # mistake.
            #
            # DECISION plan-2026-07-30T140922-8af1028f/D-015
            # Assert the post-fusion width. Do NOT drop this check or swap
            # its oracle: `similarity_head` is a Dense that accepts any input
            # width, so a wrong FFN width rewires the head silently and it
            # still trains. See decisions.md D-015.
            if processed_shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"post-fusion FFN ffn_type={self.ffn_type!r} produces "
                    f"output width {processed_shape[-1]}, but "
                    f"{type(self).__name__} requires exactly "
                    f"hidden_dim={self.hidden_dim}. Nothing downstream would "
                    f"raise on this -- the task head is built from the FFN's "
                    f"derived shape and would silently accept the wrong "
                    f"width. Check FFN_REGISTRY[{self.ffn_type!r}]"
                    f"['output_dim_param'] and the type's own width rule."
                )
        self.similarity_head.build(tuple(processed_shape))

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run both branches on the same pooled tensors and return five keys.

        A rank-3 input is mean-pooled over axis 1. A rank-2 input is used as
        it is.

        :param inputs: Dict with ``vision_features`` and ``text_features``,
            and optionally ``vision_mask`` and ``text_mask``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to the fusion, norm
            and FFN layers.
        :type training: Optional[bool]
        :return: Dict with ``similarity_matrix``, ``logits``, ``match_score``
            (one match logit per pair), ``vision_embeddings`` and
            ``text_embeddings``.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If an input's rank is not 2 or 3 (or unknown), or
            a mask is given for a rank-2 input.
        """
        vision_pooled = _pool_sequence(
            self.vision_pooling, inputs, "vision_features", "ImageTextMatchingHead")
        text_pooled = _pool_sequence(
            self.text_pooling, inputs, "text_features", "ImageTextMatchingHead")

        # 1. Contrastive Alignment part
        vision_projected = self.vision_proj(vision_pooled)
        text_projected = self.text_proj(text_pooled)
        vision_norm = keras.ops.normalize(vision_projected, axis=-1)
        text_norm = keras.ops.normalize(text_projected, axis=-1)
        similarity_matrix = keras.ops.matmul(vision_norm, keras.ops.transpose(text_norm))
        logits = similarity_matrix / self.temperature

        # 2. Fine-grained matching logit.
        # MultiModalFusion with the concatenation strategy needs rank-3
        # (B, S, D) inputs, and the pooled features are rank-2 (B, D). Expand
        # to (B, 1, D) for the fusion, then squeeze the (B, 1, F) result back
        # to (B, F) for the rank-2 post-fusion norm, FFN and similarity head.
        vision_pooled_3d = keras.ops.expand_dims(vision_pooled, axis=1)
        text_pooled_3d = keras.ops.expand_dims(text_pooled, axis=1)
        fused = self.fusion([vision_pooled_3d, text_pooled_3d], training=training)
        fused = keras.ops.squeeze(fused, axis=1)
        processed = self.post_fusion_norm(fused, training=training)
        if self.use_post_fusion_ffn:
            processed = self.post_fusion_ffn(processed, training=training)
        match_score = self.similarity_head(processed)

        return {
            "similarity_matrix": similarity_matrix,
            "logits": logits,
            "match_score": keras.ops.squeeze(match_score, axis=-1),
            "vision_embeddings": vision_norm,
            "text_embeddings": text_norm,
        }

    def compute_output_shape(self, input_shape: Dict) -> Dict[str, Tuple[Optional[int], ...]]:
        """Returns output-shape dict mirroring call() outputs.

        :param input_shape: Dict with ``vision_features`` and ``text_features``.
        :return: Shapes for the five output tensors.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        vision_shape = input_shape["vision_features"]
        batch = vision_shape[0]
        return {
            "similarity_matrix": (batch, batch),
            "logits": (batch, batch),
            "match_score": (batch,),
            "vision_embeddings": (batch, self.projection_dim),
            "text_embeddings": (batch, self.projection_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this head.

        ``temperature`` is the starting value the head was built with. The
        trained value is a weight and is restored with the other weights.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "temperature": self.initial_temperature,
                "normalization_type": self.normalization_type,
                "use_post_fusion_ffn": self.use_post_fusion_ffn,
                "ffn_type": self.ffn_type,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config
