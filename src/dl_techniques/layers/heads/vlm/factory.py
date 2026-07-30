"""
VLM Task Head Factory

A comprehensive factory for building configurable head networks for Visual Language
Model tasks. Designed to be model-agnostic and work with any VLM foundation
model (CLIP, BLIP, Flamingo, etc.).
"""

import inspect

import keras
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from ...activations import ActivationType
from ...attention.factory import create_attention_layer
from ...ffn.factory import (
    FFN_REGISTRY,
    create_ffn_from_config,
    create_ffn_layer,
    FFNType,
)
from ...fusion.multimodal_fusion import FusionStrategy, MultiModalFusion
from ...norms import NormalizationType
from ...norms.factory import create_normalization_layer
from ....utils.logger import logger
from ....utils.masking import MaskFactory
from .task_types import VLMTaskConfig, VLMTaskType


# Keras `Layer.__init__` arguments. These belong to a layer itself and must never
# be forwarded into a child layer's constructor -- `name` collides with a child's
# own name, and `trainable`/`dtype` silently reconfigure it.
_KERAS_BASE_LAYER_KWARGS = frozenset(
    {"name", "trainable", "dtype", "dynamic", "autocast", "activity_regularizer"}
)

# Arguments `MultiTaskVLMHead` supplies to every head itself. A caller passing one
# of these through would collide with the wrapper's own positional forwarding.
_WRAPPER_OWNED_HEAD_KWARGS = frozenset({"task_config", "vision_dim", "text_dim"})


def _accepted_constructor_kwargs(head_class: type) -> frozenset:
    """Return the constructor argument names ``head_class`` actually accepts.

    The five VLM head classes do NOT share a signature: ``ImageCaptioningHead``
    and ``VQAHead`` derive straight from ``keras.layers.Layer``, while
    ``VisualGroundingHead`` and ``ImageTextMatchingHead`` derive from
    ``BaseVLMHead`` and inherit its fusion arguments. So a kwarg meaningful to
    one head (``fusion_strategy``) is a hard error on another.

    Walks the MRO and unions each class's OWN explicitly-named parameters,
    stopping at ``keras.layers.Layer``. ``**kwargs`` is deliberately excluded:
    every head declares it, but only to forward to ``Layer.__init__``, which
    rejects unknown keys -- so treating ``**kwargs`` as "accepts anything" would
    invert the very check this function exists to perform.

    :param head_class: A VLM head class.
    :type head_class: type
    :return: Constructor argument names the class accepts, including inherited.
    :rtype: frozenset
    """
    accepted = set()
    for klass in head_class.__mro__:
        if klass in (keras.layers.Layer, object):
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, param in inspect.signature(init).parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            accepted.add(name)
    return frozenset(accepted)


def _is_single_shape(shape: Any) -> bool:
    """Is this ONE shape, rather than a collection of shapes?

    ``compute_output_shape`` may legitimately return either, and both are
    sequences, so ``isinstance(shape, (list, tuple))`` cannot tell them apart.
    The discriminator is the ELEMENT type: a single shape holds ints (or
    ``None`` for a dynamic axis), a collection holds sequences.

    Used by :meth:`BaseVLMHead._build_fusion_stack` to reject a fusion strategy
    whose output is per-modality before the shape reaches Keras and surfaces as
    ``ValueError: Invalid dtype: tuple`` (D-011).

    :param shape: A shape tuple, or a list/tuple of shape tuples.
    :type shape: Any
    :return: ``True`` for a single shape, ``False`` for a collection of them.
    :rtype: bool
    """
    if not isinstance(shape, (list, tuple)):
        return False
    return not any(isinstance(entry, (list, tuple)) for entry in shape)


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------


def _serialize_task_config(task_config: VLMTaskConfig) -> Dict[str, Any]:
    """Serialize a ``VLMTaskConfig`` to a JSON-safe dict.

    The dataclass stores ``task_type`` as a ``VLMTaskType`` enum, which is not
    JSON-serializable; this converts it to its string value so the layer config
    survives a ``.keras`` round-trip.

    :param task_config: The task configuration to serialize.
    :type task_config: VLMTaskConfig
    :return: JSON-serializable configuration dict.
    :rtype: Dict[str, Any]
    """
    cfg = dict(task_config.__dict__)
    cfg["task_type"] = task_config.task_type.value
    return cfg


def _deserialize_task_config(
    config: Union[VLMTaskConfig, Dict[str, Any]]
) -> VLMTaskConfig:
    """Reconstruct a ``VLMTaskConfig`` from a serialized dict.

    :param config: Serialized config dict (or an existing ``VLMTaskConfig``,
        returned unchanged).
    :type config: Union[VLMTaskConfig, Dict[str, Any]]
    :return: A ``VLMTaskConfig`` instance.
    :rtype: VLMTaskConfig
    """
    if isinstance(config, VLMTaskConfig):
        return config
    cfg = dict(config)
    task_type = cfg.get("task_type")
    if isinstance(task_type, str):
        cfg["task_type"] = VLMTaskType.from_string(task_type)
    return VLMTaskConfig(**cfg)


# ---------------------------------------------------------------------
# Base VLM Head Class
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BaseVLMHead(keras.layers.Layer):
    """
    Base class for all VLM task heads, using an advanced fusion module.

    Provides common functionality for multi-modal tasks, delegating complex
    fusion logic to the dedicated MultiModalFusion layer.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────┐  ┌──────────────┐
        │Vision Features│  │Text Features │
        └──────┬────────┘  └──────┬───────┘
               └──────┬───────────┘
                      ▼
            ┌──────────────────┐
            │ MultiModalFusion │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │ Post-Fusion Norm │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │ Post-Fusion FFN  │
            │    (optional)    │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │ Task-Specific    │
            │    Output Head   │
            └──────────────────┘

    :param task_config: VLMTaskConfig object with task configuration.
    :type task_config: VLMTaskConfig
    :param vision_dim: Dimension of vision features.
    :type vision_dim: int
    :param text_dim: Dimension of text features.
    :type text_dim: int
    :param fusion_strategy: The fusion strategy for the MultiModalFusion layer.
    :type fusion_strategy: FusionStrategy
    :param fusion_config: Configuration parameters for the MultiModalFusion layer.
    :type fusion_config: Optional[Dict[str, Any]]
    :param normalization_type: Type of normalization for post-fusion blocks.
    :type normalization_type: NormalizationType
    :param activation_type: Type of activation function for post-fusion blocks.
    :type activation_type: ActivationType
    :param use_post_fusion_ffn: If True, includes an FFN block after fusion.
    :type use_post_fusion_ffn: bool
    :param ffn_type: Type of FFN to use in the post-fusion block.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: Expansion factor for the post-fusion FFN.
    :type ffn_expansion_factor: int
    :param kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        fusion_strategy: FusionStrategy = "cross_attention",
        fusion_config: Optional[Dict[str, Any]] = None,
        normalization_type: NormalizationType = "layer_norm",
        activation_type: ActivationType = "gelu",
        use_post_fusion_ffn: bool = True,
        ffn_type: FFNType = "mlp",
        ffn_expansion_factor: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        # Store configuration
        self.task_config = task_config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_strategy = fusion_strategy
        self.fusion_config = fusion_config or {}
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.use_post_fusion_ffn = use_post_fusion_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor

        self.hidden_dim = task_config.hidden_size or max(vision_dim, text_dim)
        self._build_common_layers()

    def _build_common_layers(self) -> None:
        """Builds common layers used across different heads."""
        self.fusion = MultiModalFusion(
            dim=self.hidden_dim,
            fusion_strategy=self.fusion_strategy,
            dropout_rate=self.task_config.dropout_rate,
            name=f"{self.name}_fusion",
            **self.fusion_config,
        )

        # These blocks process the output of the fusion layer.
        self.post_fusion_norm = create_normalization_layer(
            self.normalization_type, name=f"{self.name}_post_fusion_norm"
        )
        self.post_fusion_dropout = layers.Dropout(
            self.task_config.dropout_rate, name=f"{self.name}_post_fusion_dropout"
        )

        if self.use_post_fusion_ffn:
            # DECISION plan-2026-07-30T081929-1645aa52/D-008
            # Supply `hidden_dim` only to FFN types that REQUIRE it, matching
            # `ImageCaptioningHead` (see the longer note at its own FFN site).
            # This site used to pass it UNCONDITIONALLY while that one passed it
            # conditionally -- two contradictory rules for the same question in
            # one file. The unconditional form silently OVERRIDES the internal
            # width derivation of any type that lists `hidden_dim` as optional:
            # measured on `ImageTextMatchingHead` with `ffn_type="swiglu"`, the
            # post-fusion FFN's parameter count tracked `ffn_expansion_factor`
            # (55296 / 110592 / 221184 at factor 2 / 4 / 8) instead of staying
            # invariant, because swiglu derives its own hidden width by the 2/3
            # rule and had that derivation overwritten.
            #
            # The DEFAULT path is unaffected: `ffn_type` defaults to "mlp" here,
            # and `mlp` lists `hidden_dim` as required, so the conditional makes
            # the identical call. Do NOT "simplify" this back to an
            # unconditional kwarg -- the swiglu parameter-count invariance test
            # is what fails when you do.
            ffn_kwargs = {
                "output_dim": self.hidden_dim,
                "dropout_rate": self.task_config.dropout_rate,
                "name": f"{self.name}_post_fusion_ffn",
            }
            if "hidden_dim" in FFN_REGISTRY.get(self.ffn_type, {}).get(
                "required_params", ()
            ):
                ffn_kwargs["hidden_dim"] = (
                    self.hidden_dim * self.ffn_expansion_factor
                )
            self.post_fusion_ffn = create_ffn_layer(self.ffn_type, **ffn_kwargs)

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Builds the layer.

        Deliberately does NOT build the common sub-layers created in
        ``_build_common_layers()``. It cannot: the shapes they see depend on how
        each subclass drives them (``VisualGroundingHead`` fuses per-region
        ``(B, N, D)`` tensors, ``ImageTextMatchingHead`` fuses pooled
        ``(B, 1, D)`` tensors and then squeezes), and one subclass does not use
        the post-fusion stack at all. Subclasses call
        :meth:`_build_fusion_stack` from their own ``build()`` with their actual
        shapes -- see the note there about not building unused sub-layers.
        """
        super().build(input_shape)

    def _build_fusion_stack(
        self,
        fusion_input_shapes: List[Tuple[Optional[int], ...]],
        squeeze_axis: Optional[int] = None,
        build_post_fusion: bool = True,
    ) -> Tuple[Optional[int], ...]:
        """Build ``fusion`` and (optionally) the post-fusion stack; return its shape.

        Shapes are DERIVED via each sub-layer's ``compute_output_shape`` rather
        than recomputed from ``hidden_dim``, so this stays correct if a fusion
        strategy changes its output width.

        :param fusion_input_shapes: ``[vision_shape, text_shape]`` exactly as the
            subclass passes them to ``self.fusion``.
        :type fusion_input_shapes: List[Tuple[Optional[int], ...]]
        :param squeeze_axis: Axis the subclass squeezes off the fusion output
            before the post-fusion stack, or ``None`` if it does not squeeze.
        :type squeeze_axis: Optional[int]
        :param build_post_fusion: Whether this subclass's ``call()`` actually
            runs the post-fusion norm/dropout/FFN. **Pass ``False`` when it does
            not**: building an unused sub-layer would create weights that the
            lazy path never created, changing the layer's weight count and its
            ``.keras`` layout for no benefit.
        :type build_post_fusion: bool
        :return: Shape entering the post-fusion stack (post-squeeze).
        :rtype: Tuple[Optional[int], ...]
        """
        self.fusion.build(fusion_input_shapes)
        fused_shape = self.fusion.compute_output_shape(fusion_input_shapes)

        # DECISION plan-2026-07-30T081929-1645aa52/D-011
        # The post-fusion stack (norm -> dropout -> FFN -> task head) consumes ONE
        # tensor, and each subclass's `call()` squeezes a known axis off it. Two of
        # `MultiModalFusion`'s eight strategies do not meet that contract:
        #
        #   * `cross_attention` returns a TUPLE, one tensor per modality
        #     (`multimodal_fusion.py::_call_cross_attention` -> `tuple(outputs)`).
        #   * `attention_pooling` returns rank-2 `(batch, dim)`, having already
        #     pooled away the axis the caller intends to squeeze.
        #
        # Both were reaching Keras as a shape and dying with an error that named
        # neither the strategy nor the reason: `ValueError: Invalid dtype: tuple`
        # for the first, and a squeeze/fully-defined-shape complaint for the
        # second. Measured: 6 of the 8 strategies work on both heads, those 2 fail
        # IDENTICALLY on the lazy and the explicit-build path -- so this is a
        # pre-existing capability gap, not a `build()`-contract bug.
        #
        # Raise here, at the point of wiring, rather than teaching the post-fusion
        # stack to consume a per-modality tuple: that would be a new fan-in
        # mechanism (which tensor feeds the task head? both? concatenated?) and
        # the answer is a modelling decision nobody has asked for. The check is
        # derived from the fusion layer's declared OUTPUT CONTRACT rather than
        # from a hardcoded strategy list, so a future strategy that returns a
        # tuple is rejected automatically.
        if not _is_single_shape(fused_shape):
            raise ValueError(
                f"fusion_strategy={self.fusion_strategy!r} produces one output "
                f"per modality (compute_output_shape returned "
                f"{fused_shape!r}), but {type(self).__name__}'s post-fusion "
                f"stack consumes a single fused tensor. Use one of the "
                f"single-tensor strategies (concatenation, addition, "
                f"multiplication, gated, bilinear, tensor_fusion)."
            )

        # Rank must be PRESERVED. Both heads hand `self.fusion` rank-3
        # `(batch, seq_or_region, dim)` tensors and consume a rank-3 result --
        # `ImageTextMatchingHead` squeezes its length-1 axis back off,
        # `VisualGroundingHead` keeps the region axis and scores per region. A
        # strategy that pools an axis away (`attention_pooling` -> rank 2) breaks
        # both, but at different places and with different errors: ITM died in the
        # squeeze, VG only later inside an ArgMax ("Expected dimension in the
        # range [-1, 1)"). One rank check upstream covers both heads and any
        # future pooling strategy, and needs no `squeeze_axis` special-casing.
        if len(fused_shape) != len(tuple(fusion_input_shapes[0])):
            raise ValueError(
                f"fusion_strategy={self.fusion_strategy!r} returns a "
                f"rank-{len(fused_shape)} output {tuple(fused_shape)!r} for "
                f"rank-{len(tuple(fusion_input_shapes[0]))} inputs, so it pools "
                f"away an axis that {type(self).__name__} still needs. Use a "
                f"rank-preserving strategy (concatenation, addition, "
                f"multiplication, gated, bilinear, tensor_fusion)."
            )

        if squeeze_axis is not None:
            fused_shape = tuple(
                d for i, d in enumerate(fused_shape) if i != squeeze_axis
            )

        if build_post_fusion:
            self.post_fusion_norm.build(fused_shape)
            self.post_fusion_dropout.build(fused_shape)
            if self.use_post_fusion_ffn:
                self.post_fusion_ffn.build(fused_shape)
                fused_shape = self.post_fusion_ffn.compute_output_shape(fused_shape)

        return tuple(fused_shape)

    def compute_output_shape(
        self, input_shape: Union[Dict, Tuple, List]
    ) -> Tuple[Optional[int], ...]:
        """Compute the pooled fused-representation output shape.

        The common post-fusion stack emits a ``(batch, hidden_dim)`` tensor;
        subclasses with richer ``call()`` outputs override this.

        :param input_shape: Input shape(s); a dict keyed by feature name, a list
            of shapes, or a single shape tuple.
        :return: Output shape ``(batch, hidden_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, dict):
            ref = input_shape.get("vision_features") or next(iter(input_shape.values()))
        elif (
            isinstance(input_shape, (list, tuple))
            and input_shape
            and isinstance(input_shape[0], (list, tuple))
        ):
            ref = input_shape[0]
        else:
            ref = input_shape
        return (ref[0], self.hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """Gets layer configuration."""
        config = super().get_config()
        config.update(
            {
                "task_config": _serialize_task_config(self.task_config),
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "fusion_strategy": self.fusion_strategy,
                "fusion_config": self.fusion_config,
                "normalization_type": self.normalization_type,
                "activation_type": self.activation_type,
                "use_post_fusion_ffn": self.use_post_fusion_ffn,
                "ffn_type": self.ffn_type,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseVLMHead":
        """Reconstruct the head, deserializing the ``task_config`` dataclass.

        The layer name is regenerated from ``task_config.name`` in ``__init__``,
        so the stored ``name`` is dropped to avoid a duplicate-keyword conflict.

        :param config: Serialized layer configuration.
        :type config: Dict[str, Any]
        :return: A reconstructed head instance.
        :rtype: BaseVLMHead
        """
        config = dict(config)
        config.pop("name", None)
        config["task_config"] = _deserialize_task_config(config["task_config"])
        return cls(**config)


# ---------------------------------------------------------------------
# Image Captioning Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ImageCaptioningHead(keras.layers.Layer):
    """
    An autoregressive decoder head for generating text conditioned on vision features.

    Implements a multi-layer Transformer decoder adapted for image captioning,
    generating descriptive text one token at a time conditioned on static visual
    features. Each layer uses causal self-attention for text modeling and
    cross-attention to incorporate visual information.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌──────────────┐
        │Vision Features│  │Text Embeddings│
        └──────┬───────┘  └──────┬───────┘
               │                 ▼
               │        ┌────────────────┐
               │        │ Self-Attention  │
               │        │ (causal mask)   │
               │        └───────┬────────┘
               │                ▼
               └───────►┌────────────────┐
                        │Cross-Attention  │
                        └───────┬────────┘
                                ▼
                        ┌────────────────┐
                        │     FFN        │
                        └───────┬────────┘
                                ▼
                          (x num_layers)
                                ▼
                        ┌────────────────┐
                        │ Output Proj.   │
                        │ (vocab_size)   │
                        └────────────────┘

    :param task_config: Configuration object for the task.
    :type task_config: VLMTaskConfig
    :param vision_dim: Dimension of vision features.
    :type vision_dim: int
    :param text_dim: Dimension of text features.
    :type text_dim: int
    :param num_layers: Number of decoder layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param ffn_type: Type of feed-forward network in decoder blocks.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: Width multiplier for the decoder FFN's hidden
        layer, i.e. ``hidden_dim * ffn_expansion_factor``. Defaults to 4.

        Read ONLY by the 13 registry FFN types that require an explicit
        ``hidden_dim`` (``mlp``, ``geglu``, ``glu``, ``reglu``, ``lowrank``, ...).
        The default ``swiglu`` derives its own hidden width and IGNORES this, so
        changing it does not affect the default configuration.
    :type ffn_expansion_factor: int
    :param kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_type: FFNType = "swiglu",
        ffn_expansion_factor: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{task_config.name}_head", **kwargs)
        self.task_config = task_config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = task_config.hidden_size or text_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # CREATE all sub-layers in __init__
        self.cross_attention_layers = []
        self.self_attention_layers = []
        self.ffn_layers = []
        self.norm_layers = []

        for i in range(self.num_layers):
            cross_attn = create_attention_layer(
                "multi_head_cross",
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name=f"cross_attention_{i}",
            )
            self.cross_attention_layers.append(cross_attn)

            self_attn = create_attention_layer(
                "multi_head",
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name=f"self_attention_{i}",
            )
            self.self_attention_layers.append(self_attn)

            # Supply `hidden_dim` only to FFN types that REQUIRE it. 13 of the
            # registry's 21 types do (`mlp`, `geglu`, `glu`, `reglu`, `lowrank`,
            # ...), and omitting it raised
            # "Required parameters missing for mlp: ['hidden_dim']" -- so every
            # one of those `ffn_type` values was unusable on this head.
            #
            # It is passed CONDITIONALLY rather than always, because the default
            # `swiglu` lists `hidden_dim` as OPTIONAL and derives it internally
            # (2/3 rule from `ffn_expansion_factor`, rounded to
            # `ffn_multiple_of`). Passing it explicitly would override that
            # derivation and silently change the default path's widths.
            ffn_config = {
                "type": self.ffn_type,
                "output_dim": self.hidden_dim,
                "name": f"ffn_{i}",
            }
            if "hidden_dim" in FFN_REGISTRY.get(self.ffn_type, {}).get(
                "required_params", ()
            ):
                ffn_config["hidden_dim"] = self.hidden_dim * self.ffn_expansion_factor
            ffn = create_ffn_from_config(ffn_config)
            self.ffn_layers.append(ffn)

            norm1 = create_normalization_layer("rms_norm", name=f"norm1_{i}")
            norm2 = create_normalization_layer("rms_norm", name=f"norm2_{i}")
            norm3 = create_normalization_layer("rms_norm", name=f"norm3_{i}")
            self.norm_layers.extend([norm1, norm2, norm3])

        # Final projection to vocabulary
        self.output_proj = layers.Dense(
            self.task_config.vocab_size, name=f"{self.name}_output_proj"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer, in computational order.

        Without this, the sub-layers created in ``__init__`` stay unbuilt until
        Keras traces ``call()``, and a ``.keras`` round-trip through a Functional
        model then fails to restore them -- measured as
        ``ValueError: A total of 12 objects could not be loaded`` with
        ``<Dense name=kv, built=False>`` as the example. This is the repo's
        documented lazy-sublayer serialization trap.

        Note the shape contract this head already relies on: ``call()`` adds each
        sub-block's output straight onto its input (``x + attn_output``) with no
        projection anywhere, so the text feature width must equal
        ``hidden_dim``. The builds below therefore use ``hidden_dim`` for the
        whole decoder stack. ``vision_dim`` is independent -- it only enters as
        the cross-attention's key/value width.

        :param input_shape: Shape dict with ``vision_features`` and
            ``text_features`` entries, as passed to ``call()``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        text_shape = input_shape["text_features"]
        vision_shape = input_shape["vision_features"]
        batch = text_shape[0]
        seq_len = text_shape[1]

        # The decoder stream carries `hidden_dim` throughout (see the residual
        # note above), regardless of the declared `text_dim`.
        stream_shape = (batch, seq_len, self.hidden_dim)

        for i in range(self.num_layers):
            # Self-attention takes one shape; cross-attention takes [query, kv].
            self.self_attention_layers[i].build(stream_shape)
            self.norm_layers[i * 3].build(stream_shape)
            self.cross_attention_layers[i].build([stream_shape, tuple(vision_shape)])
            self.norm_layers[i * 3 + 1].build(stream_shape)
            self.ffn_layers[i].build(stream_shape)
            self.norm_layers[i * 3 + 2].build(stream_shape)

        self.output_proj.build(stream_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]
        text_features = inputs["text_features"]  # Assumes pre-embedded text

        x = text_features
        seq_len = ops.shape(x)[1]
        # Lower-triangular KEEP mask: 1 = attend (current+past), 0 = future.
        #
        # Built from `MaskFactory.create_causal_mask` (an arange index comparison)
        # rather than `ops.tril`. `ops.tril` routes through a `tf.cond` that
        # rejects a Python-bool predicate once traced, raising
        # `TypeError: pred must not be a Python bool` -- it works EAGERLY and
        # fails on every graph path (`tf.function`, `Model.predict`, `.keras`
        # save/load, `jit_compile=True`), for both static and symbolic sequence
        # lengths. The same trap is documented at
        # `models/sd3_mmdit/text_encoders.py`. Note that Keras downgrades such a
        # `call()` crash during build-tracing to a UserWarning, so this was
        # invisible in a green test suite.
        #
        # MaskFactory returns the BLOCK polarity (True where a position must be
        # suppressed, i.e. j > i); this site needs the complementary KEEP mask as
        # a float, hence `logical_not` + cast. The cast target is
        # `backend.floatx()` because that is what the previous `ops.ones(...)`
        # defaulted to -- keeping the fix numerically inert.
        causal_mask = ops.cast(
            ops.logical_not(MaskFactory.create_causal_mask(seq_len, dtype="bool")),
            keras.backend.floatx(),
        )
        causal_mask = ops.expand_dims(causal_mask, 0)           # (1, S, S) full-mask form

        for i in range(self.num_layers):
            # Self-attention with causal mask
            attn_output = self.self_attention_layers[i](
                x, attention_mask=causal_mask, training=training
            )
            x = self.norm_layers[i * 3](x + attn_output)

            # Cross-attention to vision features
            cross_attn_output = self.cross_attention_layers[i](
                x, kv_input=vision_features, training=training
            )
            x = self.norm_layers[i * 3 + 1](x + cross_attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.norm_layers[i * 3 + 2](x + ffn_output)

        logits = self.output_proj(x)
        return {"logits": logits, "hidden_states": x}

    def compute_output_shape(self, input_shape):
        """Returns output-shape dict mirroring call() outputs."""
        text_shape = input_shape["text_features"]
        batch, seq_len = text_shape[0], text_shape[1]
        return {
            "logits": (batch, seq_len, self.task_config.vocab_size),
            "hidden_states": (batch, seq_len, self.hidden_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "task_config": _serialize_task_config(self.task_config),
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ffn_type": self.ffn_type,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImageCaptioningHead":
        """Reconstruct the head, deserializing the ``task_config`` dataclass."""
        config = dict(config)
        config.pop("name", None)
        config["task_config"] = _deserialize_task_config(config["task_config"])
        return cls(**config)


# ---------------------------------------------------------------------
# Visual Question Answering Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VQAHead(keras.layers.Layer):
    """
    A multimodal fusion and classification head for Visual Question Answering.

    Fuses vision and text representations into a joint vector via configurable
    pooling strategies, then classifies through a multi-layer MLP to predict
    the final answer from a fixed answer vocabulary.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌────────────────┐
        │Vision Features│  │Question Features│
        └──────┬───────┘  └──────┬─────────┘
               └──────┬──────────┘
                      ▼
            ┌──────────────────┐
            │ Pooling Strategy │
            │(mean/max/attn.)  │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │  Concatenation   │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │  Hidden Layers   │
            │  + Dropout       │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │  Output Layer    │
            │  (num_classes)   │
            └──────────────────┘

    :param task_config: Configuration object for the task.
    :type task_config: VLMTaskConfig
    :param vision_dim: Dimension of vision features.
    :type vision_dim: int
    :param text_dim: Dimension of text features.
    :type text_dim: int
    :param hidden_dims: List of hidden layer dimensions for the classifier MLP.
    :type hidden_dims: List[int]
    :param pooling_strategy: Strategy for pooling features ("mean", "max", "attention").
    :type pooling_strategy: str
    :param kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        pooling_strategy: str = "attention",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        if task_config.num_classes is None or task_config.num_classes <= 0:
            raise ValueError("VQAHead requires a positive num_classes in task_config.")
        if pooling_strategy not in ["mean", "max", "attention"]:
            raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")

        # Store configuration
        self.task_config = task_config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dims = hidden_dims
        self.pooling_strategy = pooling_strategy
        self.embed_dim = self.task_config.hidden_size or max(vision_dim, text_dim)

        # CREATE sub-layers
        if self.pooling_strategy == "attention":
            self.attention_pooling = create_attention_layer(
                "multi_head_cross",
                dim=self.embed_dim,
                num_heads=8,
                dropout_rate=self.task_config.dropout_rate,
                name="attention_pooling",
            )
        else:
            self.attention_pooling = None

        self.hidden_layers = []
        self.dropout_layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                layers.Dense(hidden_dim, activation="gelu", name=f"hidden_{i}")
            )
            self.dropout_layers.append(
                layers.Dropout(self.task_config.dropout_rate, name=f"dropout_{i}")
            )

        self.output_layer = layers.Dense(
            self.task_config.num_classes, name="output_layer"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        Only the sub-layers this head's ``pooling_strategy`` actually exercises
        are built: ``attention_pooling`` exists only for ``"attention"``, and
        building it otherwise would create weights the lazy path never created.

        The classifier's input width is derived from the ACTUAL input shapes
        rather than from ``vision_dim``/``text_dim``, because the two pooling
        branches produce different widths -- ``mean``/``max`` keep each
        modality's own channel count, while ``attention`` maps both to
        ``embed_dim``.

        :param input_shape: Shape dict with ``vision_features`` and
            ``question_features``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        vision_shape = tuple(input_shape["vision_features"])
        question_shape = tuple(input_shape["question_features"])
        batch = vision_shape[0]

        if self.pooling_strategy == "attention":
            # One cross-attention layer is reused in BOTH directions
            # (vision<-question and question<-vision), so it is built once with
            # [query, kv] = [vision, question]; the reversed call requires the
            # two feature widths to match, which is a pre-existing contract of
            # this head, not something introduced here.
            self.attention_pooling.build([vision_shape, question_shape])
            pooled_width = self.embed_dim * 2
        else:
            pooled_width = vision_shape[-1] + question_shape[-1]

        x_shape = (batch, pooled_width)
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            hidden_layer.build(x_shape)
            x_shape = hidden_layer.compute_output_shape(x_shape)
            dropout_layer.build(x_shape)

        self.output_layer.build(x_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]
        question_features = inputs["question_features"]

        if self.pooling_strategy == "mean":
            vision_pooled = ops.mean(vision_features, axis=1)
            text_pooled = ops.mean(question_features, axis=1)
        elif self.pooling_strategy == "max":
            vision_pooled = ops.max(vision_features, axis=1)
            text_pooled = ops.max(question_features, axis=1)
        elif self.pooling_strategy == "attention":
            vision_attended = self.attention_pooling(
                vision_features, kv_input=question_features, training=training
            )
            text_attended = self.attention_pooling(
                question_features, kv_input=vision_features, training=training
            )
            vision_pooled = ops.mean(vision_attended, axis=1)
            text_pooled = ops.mean(text_attended, axis=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        x = ops.concatenate([vision_pooled, text_pooled], axis=-1)

        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x, training=training)

        logits = self.output_layer(x)
        return {"answer_logits": logits}

    def compute_output_shape(self, input_shape):
        """Returns output-shape dict mirroring call() outputs."""
        vision_shape = input_shape["vision_features"]
        return {"answer_logits": (vision_shape[0], self.task_config.num_classes)}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "task_config": _serialize_task_config(self.task_config),
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "hidden_dims": self.hidden_dims,
                "pooling_strategy": self.pooling_strategy,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VQAHead":
        """Reconstruct the head, deserializing the ``task_config`` dataclass."""
        config = dict(config)
        config.pop("name", None)
        config["task_config"] = _deserialize_task_config(config["task_config"])
        return cls(**config)


# ---------------------------------------------------------------------
# Visual Grounding Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisualGroundingHead(BaseVLMHead):
    """
    Head for visual grounding tasks.

    Localizes image regions matching a text query by fusing per-region visual
    features with the pooled text query, scoring each region, and regressing
    a bounding box from the top-scoring region.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌──────────────┐
        │Vision Regions │  │Text Features │
        │ [B, N, D_vis]│  │              │
        └──────┬───────┘  └──────┬───────┘
               │                 ▼
               │        ┌────────────────┐
               │        │ Mean Pooling   │
               │        └───────┬────────┘
               │                ▼
               └───────►┌────────────────┐
                        │ Gated Fusion   │
                        │ (per region)   │
                        └───┬────────┬───┘
                            ▼        ▼
                    ┌──────────┐ ┌────────┐
                    │Confidence│ │  BBox  │
                    │ Scorer   │ │Regress.│
                    └──────────┘ └────────┘
    """

    def __init__(self, **kwargs: Any) -> None:
        # A strategy that scores per-region interactions is best.
        kwargs.setdefault("fusion_strategy", "gated")
        super().__init__(**kwargs)

        self.bbox_regressor = layers.Dense(4, name=f"{self.name}_bbox_regressor")
        self.confidence_scorer = layers.Dense(
            1, activation="sigmoid", name=f"{self.name}_confidence"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        ``call()`` pools the text features to one query vector and TILES it
        across the vision regions, so the fusion sees two ``(B, N_regions, D)``
        tensors -- the text side keeps its own channel count, not a pooled-away
        one.

        The post-fusion norm/dropout/FFN are NOT built: this head's ``call()``
        never runs them (it goes straight from ``fusion`` to
        ``confidence_scorer``). Building them would create weights the lazy path
        never created, inflating the weight count and changing the ``.keras``
        layout.

        :param input_shape: Shape dict with ``vision_features`` (3-D, per-region)
            and ``text_features``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        vision_shape = tuple(input_shape["vision_features"])
        text_shape = tuple(input_shape["text_features"])
        batch, num_regions = vision_shape[0], vision_shape[1]

        # Text is mean-pooled over its sequence axis then tiled to num_regions.
        text_tiled_shape = (batch, num_regions, text_shape[-1])

        fused_shape = self._build_fusion_stack(
            [vision_shape, text_tiled_shape], build_post_fusion=False
        )

        # confidence_scorer sees the per-region fused tensor; bbox_regressor sees
        # only the single top-scoring region, i.e. the same width without the
        # region axis.
        self.confidence_scorer.build(fused_shape)
        self.bbox_regressor.build((batch, fused_shape[-1]))

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]  # [B, N_regions, D_vis]
        text_features = inputs["text_features"]

        if len(ops.shape(vision_features)) != 3:
            raise ValueError("VisualGrounding requires spatial vision features.")

        # Pool text features to a single query vector.
        text_query = (
            ops.mean(text_features, axis=1)
            if len(ops.shape(text_features)) == 3
            else text_features
        )

        # Align text query with each visual region for fusion.
        num_regions = ops.shape(vision_features)[1]
        text_expanded = ops.expand_dims(text_query, axis=1)
        text_expanded = ops.tile(text_expanded, [1, num_regions, 1])

        # Fuse each region with the text query. The output is [B, N_regions, D_fused].
        fused_per_region = self.fusion([vision_features, text_expanded], training=training)

        # Score each aligned region's features.
        region_scores = self.confidence_scorer(fused_per_region)
        region_scores = ops.squeeze(region_scores, axis=-1)  # [B, N_regions]

        # Regress bounding box from the top-scoring region's features.
        #
        # Gather one region per batch element with `ops.take_along_axis`, NOT with
        # NumPy-style fancy indexing (`fused[batch_indices, top_indices]`). That
        # form is a NumPy idiom that TF tensors reject outright --
        # "Only integers, slices (`:`), ellipsis, tf.newaxis and scalar
        # tf.int32/tf.int64 tensors are valid indices" -- so it failed EAGERLY,
        # not merely under tracing, leaving this head dead on its forward pass.
        top_indices = ops.argmax(region_scores, axis=1)
        gather_index = ops.reshape(ops.cast(top_indices, "int32"), (-1, 1, 1))
        gather_index = ops.broadcast_to(
            gather_index, (ops.shape(fused_per_region)[0], 1, ops.shape(fused_per_region)[2])
        )
        top_features = ops.squeeze(
            ops.take_along_axis(fused_per_region, gather_index, axis=1), axis=1
        )
        bbox = self.bbox_regressor(top_features)

        return {
            "bbox": ops.sigmoid(bbox),
            "confidence": region_scores,
            "grounded_features": top_features,
        }

    def compute_output_shape(self, input_shape: Dict) -> Dict[str, Tuple[Optional[int], ...]]:
        """Returns output-shape dict mirroring call() outputs.

        :param input_shape: Dict with ``vision_features`` ``(B, N_regions, D_vis)``.
        :return: Shapes for ``bbox``, ``confidence`` and ``grounded_features``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        vision_shape = input_shape["vision_features"]
        batch, num_regions = vision_shape[0], vision_shape[1]
        return {
            "bbox": (batch, 4),
            "confidence": (batch, num_regions),
            "grounded_features": (batch, self.hidden_dim),
        }


# ---------------------------------------------------------------------
# Image-Text Matching Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ImageTextMatchingHead(BaseVLMHead):
    """
    A projection head for contrastive image-text alignment and fine-grained matching.

    Performs two functions: (1) projects vision and text features into a shared
    L2-normalized embedding space for CLIP-style contrastive loss scaled by a
    learnable temperature, and (2) fuses features to produce a fine-grained
    matching score indicating semantic correspondence.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌──────────────┐
        │Vision Features│  │Text Features │
        └──┬───────┬───┘  └──┬───────┬───┘
           │       │         │       │
           ▼       │         ▼       │
        ┌──────┐   │      ┌──────┐   │
        │V Proj│   │      │T Proj│   │
        └──┬───┘   │      └──┬───┘   │
           ▼       │         ▼       │
        ┌──────┐   │      ┌──────┐   │
        │L2Norm│   │      │L2Norm│   │
        └──┬───┘   │      └──┬───┘   │
           └──┬────┘─────────┘       │
              ▼                      │
        ┌───────────┐    ┌───────────┘
        │Similarity │    │
        │Matrix/τ   │    ▼
        └───────────┘  ┌──────────┐
                       │  Fusion  │
                       └────┬─────┘
                            ▼
                       ┌──────────┐
                       │Match Scr.│
                       └──────────┘

    :param task_config: Configuration object for the task.
    :type task_config: VLMTaskConfig
    :param vision_dim: Dimension of vision features.
    :type vision_dim: int
    :param text_dim: Dimension of text features.
    :type text_dim: int
    :param projection_dim: Projection dimension for contrastive learning.
    :type projection_dim: int
    :param temperature: Initial temperature for contrastive loss.
    :type temperature: float
    :param kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("fusion_strategy", "concatenation")
        super().__init__(
            task_config=task_config, vision_dim=vision_dim, text_dim=text_dim, **kwargs
        )

        self.projection_dim = projection_dim
        self.vision_proj = layers.Dense(projection_dim, name=f"{self.name}_vision_proj")
        self.text_proj = layers.Dense(projection_dim, name=f"{self.name}_text_proj")
        self.similarity_head = layers.Dense(
            1, activation="sigmoid", name=f"{self.name}_similarity"
        )
        self.temperature = self.add_weight(
            name=f"{self.name}_temperature",
            shape=(),
            initializer=keras.initializers.Constant(temperature),
            trainable=True,
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        ``call()`` mean-pools each modality to 2-D, projects both, and separately
        expands the pooled tensors to ``(B, 1, D)`` for the fusion (which
        requires 3-D) before squeezing axis 1 back off for the 2-D post-fusion
        stack. ``_build_fusion_stack`` is told about that squeeze so the norm/FFN
        are built at the rank they actually see.

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

        processed_shape = self._build_fusion_stack(
            [(batch, 1, vision_shape[-1]), (batch, 1, text_shape[-1])],
            squeeze_axis=1,
            build_post_fusion=True,
        )
        self.similarity_head.build(processed_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]
        text_features = inputs["text_features"]

        vision_pooled = (
            ops.mean(vision_features, axis=1)
            if len(ops.shape(vision_features)) == 3
            else vision_features
        )
        text_pooled = (
            ops.mean(text_features, axis=1)
            if len(ops.shape(text_features)) == 3
            else text_features
        )

        # 1. Contrastive Alignment part
        vision_projected = self.vision_proj(vision_pooled)
        text_projected = self.text_proj(text_pooled)
        vision_norm = ops.normalize(vision_projected, axis=-1)
        text_norm = ops.normalize(text_projected, axis=-1)
        similarity_matrix = ops.matmul(vision_norm, ops.transpose(text_norm))
        logits = similarity_matrix / self.temperature

        # 2. Fine-grained Matching Score part
        # MultiModalFusion (concatenation) requires 3-D (B, S, D) inputs, but the
        # pooled features are 2-D (B, D). Expand to (B, 1, D) for fusion, then
        # squeeze the (B, 1, F) output back to (B, F) for the 2-D post-fusion
        # norm / FFN / similarity head (D-001 scope expansion).
        vision_pooled_3d = ops.expand_dims(vision_pooled, axis=1)
        text_pooled_3d = ops.expand_dims(text_pooled, axis=1)
        fused = self.fusion([vision_pooled_3d, text_pooled_3d], training=training)
        fused = ops.squeeze(fused, axis=1)
        processed = self.post_fusion_norm(fused, training=training)
        if self.use_post_fusion_ffn:
            processed = self.post_fusion_ffn(processed, training=training)
        match_score = self.similarity_head(processed)

        return {
            "similarity_matrix": similarity_matrix,
            "logits": logits,
            "match_score": ops.squeeze(match_score, axis=-1),
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
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                # temperature is a weight, will be saved automatically
            }
        )
        return config


# ---------------------------------------------------------------------
# Multi-Task VLM Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiTaskVLMHead(keras.layers.Layer):
    """
    Multi-task head combining multiple VLM task-specific heads.

    Routes shared vision and text features to independently configured
    task-specific heads, enabling joint multi-task VLM training.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌──────────────┐
        │Vision Features│  │Text Features │
        └──────┬───────┘  └──────┬───────┘
               └──────┬──────────┘
                      ▼
            ┌──────────────────┐
            │  Task Router     │
            └──┬───┬───┬───┬──┘
               ▼   ▼   ▼   ▼
            ┌───┐┌───┐┌───┐┌───┐
            │Cap││VQA││Grd││ITM│...
            └─┬─┘└─┬─┘└─┬─┘└─┬─┘
              ▼    ▼    ▼    ▼
            ┌──────────────────┐
            │ Task Output Dict │
            └──────────────────┘
    """

    def __init__(
        self,
        task_configs: Dict[str, VLMTaskConfig],
        shared_vision_dim: int = 768,
        shared_text_dim: int = 768,
        task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Build one head per task, from shared settings plus per-task overrides.

        :param task_configs: Mapping of task name to its ``VLMTaskConfig``.
        :type task_configs: Dict[str, VLMTaskConfig]
        :param shared_vision_dim: Vision feature width handed to every head.
        :type shared_vision_dim: int
        :param shared_text_dim: Text feature width handed to every head.
        :type shared_text_dim: int
        :param task_specific_kwargs: Per-task constructor overrides, keyed by the
            same task names as ``task_configs`` -- e.g.
            ``{"cap": {"num_heads": 4, "num_layers": 1}}``. Merged over the
            shared kwargs, so a task-specific value wins.

            This is an EXPLICIT parameter rather than something read back out of
            ``**kwargs``. It used to be the latter, which made it unusable: it
            stayed in ``kwargs``, was forwarded to ``Layer.__init__``, and was
            rejected with ``ValueError: Unrecognized keyword arguments``. So the
            documented feature always raised and every head was forced onto the
            shared defaults.
        :type task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]]
        :param kwargs: Shared per-head constructor settings plus this layer's own
            Keras base arguments. The two are SEPARATED below: base arguments go
            only to ``Layer.__init__``, never into a head constructor, where
            ``name`` would collide with the head's own auto-generated name.

            A shared kwarg is routed to the heads whose class ACCEPTS it and
            skipped for the rest, because the five head classes do not share one
            signature -- ``ImageCaptioningHead`` and ``VQAHead`` are plain
            ``Layer`` subclasses, so a ``BaseVLMHead`` argument such as
            ``fusion_strategy`` is a hard error on them, and requiring every head
            to accept every shared argument would make this wrapper unusable for
            any mixed set of tasks.

            Two guards keep that from being silent: a shared kwarg accepted by
            NO head raises, and partial application is reported via
            ``logger.info``. Use ``task_specific_kwargs`` to target one head
            exactly -- those ARE validated strictly, since they name a head.
        :raises ValueError: If ``task_specific_kwargs`` names a task absent from
            ``task_configs``.
        """
        # Keras base arguments belong to THIS layer; everything else is intended
        # for the per-task heads. Forwarding `name` into a head raises (each head
        # sets its own `name=f"{task_config.name}_head"`), and forwarding
        # `trainable`/`dtype` silently reconfigures the children.
        base_kwargs = {
            k: v for k, v in kwargs.items() if k in _KERAS_BASE_LAYER_KWARGS
        }
        shared_head_kwargs = {
            k: v for k, v in kwargs.items() if k not in _KERAS_BASE_LAYER_KWARGS
        }
        super().__init__(**base_kwargs)

        self.task_configs = task_configs
        self.shared_vision_dim = shared_vision_dim
        self.shared_text_dim = shared_text_dim
        self.shared_head_kwargs = shared_head_kwargs
        self.task_specific_kwargs = dict(task_specific_kwargs or {})

        # A typo'd task name would otherwise apply to nothing at all, which is
        # exactly the silent-no-op class this package's conventions forbid.
        unknown_tasks = set(self.task_specific_kwargs) - set(task_configs)
        if unknown_tasks:
            raise ValueError(
                f"task_specific_kwargs names task(s) absent from task_configs: "
                f"{sorted(unknown_tasks)}. Known tasks: {sorted(task_configs)}."
            )

        head_classes = {
            task_name: get_head_class(task_config.task_type)
            for task_name, task_config in task_configs.items()
        }
        accepted = {
            task_name: _accepted_constructor_kwargs(head_class)
            for task_name, head_class in head_classes.items()
        }

        # The wrapper supplies these itself; a caller passing one through would
        # collide with that and raise an opaque duplicate-argument TypeError.
        for source, mapping in (
            ("shared kwargs", self.shared_head_kwargs),
            *(
                (f"task_specific_kwargs[{t!r}]", o)
                for t, o in self.task_specific_kwargs.items()
            ),
        ):
            reserved = set(mapping) & _WRAPPER_OWNED_HEAD_KWARGS
            if reserved:
                raise ValueError(
                    f"{source} may not set {sorted(reserved)} — "
                    f"MultiTaskVLMHead supplies these to every head itself. Use "
                    f"shared_vision_dim / shared_text_dim, and task_configs."
                )

        # A per-task override names ONE head explicitly, so an argument that head
        # cannot accept is a caller error and is rejected outright.
        for task_name, overrides in self.task_specific_kwargs.items():
            unusable = set(overrides) - accepted[task_name]
            if unusable:
                raise ValueError(
                    f"task_specific_kwargs[{task_name!r}] sets {sorted(unusable)}, "
                    f"which {head_classes[task_name].__name__} does not accept. "
                    f"It accepts: {sorted(accepted[task_name] - _WRAPPER_OWNED_HEAD_KWARGS)}."
                )

        # A SHARED kwarg is best-effort by design: the head classes do not share
        # a signature, so requiring every head to accept every shared argument
        # would make this wrapper unusable for any mixed set of tasks (which is
        # its entire purpose). It is routed to the heads that accept it and
        # skipped for the rest.
        #
        # The guard that keeps that from becoming a silent no-op: a shared kwarg
        # accepted by NO head is a typo or a mistake, and raises. Partial
        # application is reported through the logger so it is discoverable.
        for key in sorted(self.shared_head_kwargs):
            takers = [t for t, acc in accepted.items() if key in acc]
            if not takers:
                every = sorted(
                    set().union(*accepted.values()) - _WRAPPER_OWNED_HEAD_KWARGS
                )
                raise ValueError(
                    f"shared kwarg {key!r} is not accepted by ANY head in this "
                    f"multi-task wrapper "
                    f"({ {t: c.__name__ for t, c in head_classes.items()} }). "
                    f"Accepted by at least one head: {every}."
                )
            skipped = sorted(set(accepted) - set(takers))
            if skipped:
                logger.info(
                    f"MultiTaskVLMHead: shared kwarg '{key}' applied to "
                    f"{sorted(takers)}; skipped for {skipped} "
                    f"(their head classes do not accept it). Use "
                    f"task_specific_kwargs to configure those explicitly."
                )

        self.task_heads = {}
        for task_name, task_config in task_configs.items():
            head_class = head_classes[task_name]

            # Shared settings, filtered to what THIS head accepts, then this
            # task's overrides on top (already validated as acceptable above).
            head_kwargs = {
                k: v
                for k, v in self.shared_head_kwargs.items()
                if k in accepted[task_name]
            }
            head_kwargs.update(self.task_specific_kwargs.get(task_name, {}))

            head = head_class(
                task_config=task_config,
                vision_dim=self.shared_vision_dim,
                text_dim=self.shared_text_dim,
                **head_kwargs,
            )
            self.task_heads[task_name] = head
            setattr(self, f"head_{task_name}", head)

    def build(self, input_shape: Dict) -> None:
        """Build every task head against the shared input shapes.

        ``call()`` hands the SAME ``inputs`` dict to every head, so each head is
        built from the same shapes. A head that needs a key the caller did not
        supply raises here rather than at first use -- the same failure ``call()``
        would produce, just earlier, and deliberately not swallowed.

        :param input_shape: Shared input shapes, keyed by feature name.
        :type input_shape: Dict
        """
        if self.built:
            return

        for head in self.task_heads.values():
            head.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        task_name: Optional[str] = None,
        training: Optional[bool] = None,
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if task_name:
            if task_name not in self.task_heads:
                raise ValueError(f"Unknown task: {task_name}")
            return self.task_heads[task_name](inputs, training=training)

        outputs = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(inputs, training=training)
        return outputs

    def compute_output_shape(self, input_shape: Dict) -> Dict[str, Any]:
        """Returns a dict of per-task output shapes.

        Mirrors the no-``task_name`` ``call()`` path, which routes the shared
        inputs to every task head.

        :param input_shape: Shared input shapes (dict keyed by feature name).
        :return: Mapping of task name to that head's output shape(s).
        :rtype: Dict[str, Any]
        """
        return {
            name: head.compute_output_shape(input_shape)
            for name, head in self.task_heads.items()
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "task_configs": {
                    name: _serialize_task_config(tc)
                    for name, tc in self.task_configs.items()
                },
                "shared_vision_dim": self.shared_vision_dim,
                "shared_text_dim": self.shared_text_dim,
                "task_specific_kwargs": self.task_specific_kwargs,
            }
        )
        # `shared_head_kwargs` no longer contains this layer's Keras base
        # arguments, so this cannot clobber the `name`/`dtype` that
        # `super().get_config()` already wrote.
        config.update(self.shared_head_kwargs)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiTaskVLMHead":
        """Reconstruct the multi-task head, deserializing each ``task_config``.

        Keras base kwargs (``name``/``trainable``/``dtype``) are dropped so they
        are not forwarded into per-task head construction (where ``name`` would
        collide with each head's auto-generated name).
        """
        config = dict(config)
        for base_key in ("name", "trainable", "dtype"):
            config.pop(base_key, None)
        config["task_configs"] = {
            name: _deserialize_task_config(tc)
            for name, tc in config["task_configs"].items()
        }
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------


def get_head_class(task_type: VLMTaskType) -> type:
    """
    Get the appropriate head class for a VLM task type.

    :param task_type: The VLM task type to look up.
    :type task_type: VLMTaskType
    :return: The head class corresponding to the task type.
    :rtype: type
    """
    head_mapping = {
        VLMTaskType.IMAGE_CAPTIONING: ImageCaptioningHead,
        VLMTaskType.DENSE_CAPTIONING: BaseVLMHead,  # Placeholder
        VLMTaskType.VISUAL_QUESTION_ANSWERING: VQAHead,
        VLMTaskType.VISUAL_GROUNDING: VisualGroundingHead,
        VLMTaskType.IMAGE_TEXT_MATCHING: ImageTextMatchingHead,
        VLMTaskType.VISUAL_DIALOGUE: BaseVLMHead,  # Placeholder
    }
    # NO SILENT FALLBACK, AND `BaseVLMHead` IS NOT A USABLE HEAD.
    #
    # This used to be `return head_mapping.get(task_type, BaseVLMHead)`, so 41 of the 47
    # VLMTaskType members silently returned `BaseVLMHead`. That class has NO `call()`
    # method -- it is fusion + norm + optional FFN and nothing else -- so the factory
    # handed back an object that CONSTRUCTS fine and then dies the moment it is used:
    #
    #     create_vlm_head(VLMTaskConfig(task_type=VIDEO_CAPTIONING), ...)   # no error
    #     head({'vision_features': v, 'text_features': t})
    #     -> NotImplementedError: Layer BaseVLMHead does not have a call() method
    #
    # The error names BaseVLMHead, not the task, so the caller has no idea their task type
    # was never implemented. The same is true of the two entries above that map to
    # BaseVLMHead explicitly and are commented "# Placeholder": a placeholder that cannot
    # be called is not a head, so they are rejected here too rather than deferred.
    #
    # Only FOUR VLM task types have a real head today (image captioning, VQA, visual
    # grounding, image-text matching). Say so, at construction time, instead of pretending.
    # The sibling `heads/vision/factory.py` already raises on an unsupported task; this
    # matches it. To add support, implement the head and map it -- do not restore the
    # fallback.
    head_class = head_mapping.get(task_type)
    if head_class is None or head_class is BaseVLMHead:
        implemented = sorted(
            t.name for t, c in head_mapping.items() if c is not BaseVLMHead
        )
        raise ValueError(
            f"No VLM head is implemented for task type '{task_type.name}'. "
            f"Implemented task types: {implemented}. "
            f"(This previously returned a bare BaseVLMHead, which has no call() method and "
            f"raised NotImplementedError only once the head was actually used.)"
        )

    return head_class


def create_vlm_head(
    task_config: Union[VLMTaskConfig, Dict[str, Any]], **kwargs: Any
) -> Union[BaseVLMHead, keras.layers.Layer]:
    """
    Factory function to create VLM task heads.

    :param task_config: VLMTaskConfig object or dict with task configuration.
    :type task_config: Union[VLMTaskConfig, Dict[str, Any]]
    :param kwargs: Additional configuration parameters for the head, including
        ``vision_dim``, ``text_dim``, ``fusion_strategy``, etc.
    :return: A configured VLM head for the specified task.
    :rtype: Union[BaseVLMHead, keras.layers.Layer]
    """
    if isinstance(task_config, dict):
        task_config = VLMTaskConfig(**task_config)

    head_class = get_head_class(task_config.task_type)
    return head_class(task_config=task_config, **kwargs)


def create_multi_task_vlm_head(
    task_configs: Union[List[VLMTaskConfig], Dict[str, VLMTaskConfig]],
    **kwargs: Any,
) -> MultiTaskVLMHead:
    """
    Create a multi-task VLM head from task configurations.

    :param task_configs: List or dict of VLMTaskConfig objects.
    :type task_configs: Union[List[VLMTaskConfig], Dict[str, VLMTaskConfig]]
    :param kwargs: Shared configuration for all heads, such as ``vision_dim``,
        ``text_dim``, ``fusion_strategy``. Can also include
        ``task_specific_kwargs`` to override settings for specific tasks.
    :return: Configured multi-task VLM head instance.
    :rtype: MultiTaskVLMHead
    """
    if isinstance(task_configs, list):
        task_configs = {config.name: config for config in task_configs}

    return MultiTaskVLMHead(task_configs=task_configs, **kwargs)