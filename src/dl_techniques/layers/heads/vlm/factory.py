"""Vision-language task heads, and the factory that picks one.

This module holds six layer classes and three module-level helpers. The
classes turn vision features and text features into task predictions. The
helpers pick a class and configure it.

The heads take feature tensors, not images or token ids. Nothing here knows
which foundation model produced them, so the same head works on CLIP, BLIP or
Flamingo features.

Classes
-------
* :class:`BaseVLMHead` -- shared construction for the fusion stack. Two heads
  inherit from it. It has no ``call`` and refuses to be constructed on its
  own.
* :class:`ImageCaptioningHead` -- a transformer decoder over text, cross-
  attending to vision. It does NOT inherit from :class:`BaseVLMHead`.
* :class:`VQAHead` -- pools each modality, concatenates, classifies. It does
  NOT inherit from :class:`BaseVLMHead` either.
* :class:`VisualGroundingHead` -- scores image regions against a text query
  and regresses a box from the top-scoring one.
* :class:`ImageTextMatchingHead` -- a contrastive branch and a fine-grained
  fusion branch, run in parallel.
* :class:`MultiTaskVLMHead` -- several heads behind one layer. It does not
  inherit from :class:`BaseVLMHead`.

Helpers
-------
* :data:`VLM_HEAD_DISPATCH` -- the one table from task type to head class.
  Four of the 47 ``VLMTaskType`` members have a head.
* :func:`get_head_class` -- reads that table; every other member raises
  ``ValueError``.
* :func:`create_vlm_head` -- build one head from a ``VLMTaskConfig``.
* :func:`create_multi_task_vlm_head` -- build a :class:`MultiTaskVLMHead`
  from a list or a dict of configurations.

The task types and the configuration dataclass live in ``vlm/task_types.py``.

Every feature width comes from the input shapes handed to ``build``; no head
takes a declared ``vision_dim`` or ``text_dim``. Each head owns exactly the
sub-layers its ``call`` runs, and takes no argument its ``call`` does not use.

Padding masks ride in the input dict next to the features they describe:
``vision_mask`` for ``vision_features``, ``text_mask`` for ``text_features``
and ``question_mask`` for ``question_features``. Each is a ``(batch, length)``
keep predicate, 1 for a real token and 0 for padding, the convention every
attention layer in the package uses. A head given no mask treats every
position as real, exactly as before masks existed.

No model in ``dl_techniques.models`` and no trainer in ``src/train``
constructs a VLM head today; the heads are a library for callers to compose.

Example
-------
>>> from dl_techniques.layers.heads.vlm.factory import create_vlm_head
>>> from dl_techniques.layers.heads.vlm.task_types import (
...     VLMTaskConfig, VLMTaskType)
>>> head = create_vlm_head(VLMTaskConfig(
...     name='vqa', task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
...     num_classes=100))
>>> out = head({'vision_features': v, 'question_features': q})
>>> sorted(out)
['answer_logits']
"""

import inspect
import math
import numbers

import keras
from typing import Dict, List, Optional, Union, Tuple, Any, Sequence

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from ...attention.factory import create_attention_layer
from ...ffn.factory import (
    FFN_REGISTRY,
    assemble_ffn_config,
    create_ffn_from_config,
    create_ffn_layer,
    FFNType,
)
from ...fusion.multimodal_fusion import FusionStrategy, MultiModalFusion
from ...norms import NormalizationType
from ...norms.factory import create_normalization_layer
from ...sequence_pooling import SequencePooling
from ....utils.dtype_policy import mask_sentinel
from ....utils.logger import logger
from ....utils.masking import MaskFactory
from ....utils.tensors import is_single_shape
from ..validation import positive_int, static_rank
from .task_types import VLMTaskConfig, VLMTaskType
from dl_techniques.utils.keras_registration import register_dl_technique


# Keras `Layer.__init__` arguments. These belong to a layer itself and must never
# be forwarded into a child layer's constructor -- `name` collides with a child's
# own name, and `trainable`/`dtype` silently reconfigure it.
_KERAS_BASE_LAYER_KWARGS = frozenset(
    {"name", "trainable", "dtype", "dynamic", "autocast", "activity_regularizer"}
)

# Arguments `MultiTaskVLMHead` supplies to every head itself. A caller passing one
# of these through would collide with the wrapper's own positional forwarding.
_WRAPPER_OWNED_HEAD_KWARGS = frozenset({"task_config"})


def _accepted_constructor_kwargs(head_class: type) -> frozenset:
    """Return the constructor argument names ``head_class`` actually accepts.

    The five VLM head classes do not share a signature. ``ImageCaptioningHead``
    and ``VQAHead`` derive straight from ``keras.layers.Layer``.
    ``VisualGroundingHead`` and ``ImageTextMatchingHead`` derive from
    ``BaseVLMHead`` and inherit its fusion arguments. So a keyword that is
    meaningful to one head, such as ``fusion_strategy``, is a hard error on
    another.

    The function walks the MRO and unions each class's own explicitly named
    parameters, stopping at ``keras.layers.Layer``.

    ``**kwargs`` is excluded from the result. Every head declares it, but only
    to forward to ``Layer.__init__``, which rejects unknown keys. Counting it
    as "accepts anything" would invert the very check this function performs.

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


def _ffn_width_kwargs(ffn_type: str, width: int) -> Dict[str, int]:
    """The kwargs that set ``ffn_type``'s OUTPUT width to ``width``.

    Interface contract. There are 2 call sites, and keeping them identical is
    the point of the helper existing at all. They are
    :meth:`ImageTextMatchingHead._create_post_fusion_layers` and the per-layer FFN loop in
    :meth:`ImageCaptioningHead.__init__`. The contract:

    * Returns ``{<the type's own width-parameter name>: width}``, read from
      ``FFN_REGISTRY[ffn_type]['output_dim_param']``.
    * Returns ``{}`` when that field is ``None``. The type then has no
      output-width concept and must be passed NO width key. ``mixer`` is the
      case: its output shape is structurally its input shape.
    * Returns ``{}`` for a type absent from ``FFN_REGISTRY``, leaving
      ``create_ffn_layer`` to raise its own unknown-type error rather than
      inventing a second one here.
    * Raises ``KeyError`` if a registry entry is missing the field entirely. A
      silent default there would reintroduce the defect below.

    Both call sites once hardcoded ``"output_dim": self.hidden_dim``. For the
    four types named in the anchor the key was dropped by
    ``create_ffn_layer``'s parameter filter. The type then died inside
    ``validate_ffn_config`` on its own width parameter, which nobody had
    supplied. ``tests/test_layers/test_heads/test_vlm.py::
    TestFFNOutputWidthParamRouting`` pins the routing.

    # DECISION plan-2026-07-30T140922-8af1028f/D-014
    Read the width parameter's NAME from the registry. Do NOT spell it as the
    literal ``"output_dim"``. That variant is a no-op for exactly the 4 types
    named differently: ``gated_mlp`` (``filters``), ``kan`` (``features``),
    ``power_mlp`` and ``tversky`` (``units``). See decisions.md D-014.

    :param ffn_type: An ``FFN_REGISTRY`` key.
    :type ffn_type: str
    :param width: The output width the FFN must produce.
    :type width: int
    :return: Zero or one kwarg for ``create_ffn_layer``.
    :rtype: Dict[str, int]
    """
    entry = FFN_REGISTRY.get(ffn_type)
    if entry is None:
        return {}
    width_param = entry["output_dim_param"]
    if width_param is None:
        return {}
    return {width_param: width}


# ---------------------------------------------------------------------
# Padding masks and per-head config fields
# ---------------------------------------------------------------------

#: The mask key for each feature key a head reads.
_MASK_KEYS = {
    "vision_features": "vision_mask",
    "text_features": "text_mask",
    "question_features": "question_mask",
}

#: The ``VLMTaskConfig`` fields only one head reads, and that head.
_PER_HEAD_CONFIG_FIELDS = {
    "vocab_size": "ImageCaptioningHead",
    "num_classes": "VQAHead",
}


def _check_task_config_fields(
    task_config: VLMTaskConfig,
    owner: str,
    required: Tuple[str, ...] = (),
) -> None:
    """Refuse a per-head config field this head does not read, and require its own.

    Interface contract, shared by the four task heads' ``__init__``:

    * Each field of :data:`_PER_HEAD_CONFIG_FIELDS` named in ``required`` must
      be set; ``None`` raises ``ValueError`` naming ``owner`` and the field.
    * Each other field of that table must be ``None``; a set value raises
      ``ValueError`` naming the field and the one head that reads it.
    * Returns ``None``. ``hidden_size`` and ``dropout_rate`` are read by every
      head and are not checked here; ``VLMTaskConfig`` validates all values.

    # DECISION plan-2026-10-14T062616-81938e81/D-058
    Refuse, do not ignore. Do NOT let a head accept a ``num_classes`` or a
    ``vocab_size`` it never reads: the old config carried both for every head
    (``vocab_size`` defaulted to 50000) and a grounding or matching head
    silently dropped them. See decisions.md D-058.

    :param task_config: The head's validated task config.
    :type task_config: VLMTaskConfig
    :param owner: Head class name, for the error message.
    :type owner: str
    :param required: The per-head fields this head reads and needs.
    :type required: Tuple[str, ...]
    :return: None.
    :rtype: None
    :raises ValueError: If a required field is unset, or another one is set.
    """
    for field_name, reader in _PER_HEAD_CONFIG_FIELDS.items():
        value = getattr(task_config, field_name)
        if field_name in required:
            if value is None:
                raise ValueError(
                    f"{owner} requires task_config.{field_name}, a positive "
                    f"integer; got None."
                )
        elif value is not None:
            raise ValueError(
                f"{owner}: task_config.{field_name}={value!r} is not read by "
                f"this head; only {reader} reads it. Leave it unset."
            )


def _weightless_pooling(strategy: str, name: str) -> SequencePooling:
    """Build the weightless ``SequencePooling`` a head reduces one modality with.

    Interface contract, shared by :class:`VQAHead`,
    :class:`VisualGroundingHead` and :class:`ImageTextMatchingHead` (five
    sites): ``strategy`` is ``'mean'`` or ``'max'``, the layer owns no weight,
    and constructing it consumes no random draw.

    # DECISION plan-2026-10-14T062616-81938e81/D-055
    Pass ``kernel_initializer='zeros'``. Do NOT drop it as a no-op: the mean
    and max strategies read no initializer, but ``SequencePooling.__init__``
    resolves its ``'glorot_uniform'`` default, and a seedless Keras random
    initializer takes a seed from the global Python RNG when it is created.
    Measured: that one draw moved the fixed-seed initial weights of every VQA,
    grounding and matching head (and every multi-task head after them).
    ``'zeros'`` draws nothing, so a fixed seed builds the same weights as
    before masks existed. See decisions.md D-055.

    :param strategy: ``'mean'`` or ``'max'``.
    :type strategy: str
    :param name: Layer name.
    :type name: str
    :return: The pooling layer.
    :rtype: SequencePooling
    """
    return SequencePooling(strategy=strategy, kernel_initializer="zeros", name=name)


def _pool_sequence(
    pooling: keras.layers.Layer,
    inputs: Dict[str, Any],
    features_key: str,
    owner: str,
) -> keras.KerasTensor:
    """Pool one modality over its sequence axis, honouring its padding mask.

    Interface contract, shared by :class:`VQAHead`,
    :class:`VisualGroundingHead` and :class:`ImageTextMatchingHead`:

    * A rank-3 ``inputs[features_key]`` is reduced over axis 1 by ``pooling``,
      a weightless ``SequencePooling`` (``'mean'`` or ``'max'``), with the
      mask under ``_MASK_KEYS[features_key]`` when the dict holds one. The
      masked mean divides by the number of real positions (at least 1), and
      the masked max never selects a padded position; that rule lives in
      ``SequencePooling`` and is not restated here. With no mask the result is
      exactly ``keras.ops.mean``/``keras.ops.max`` over axis 1.
    * A rank-2 input is already pooled and is returned unchanged; a mask for
      it raises ``ValueError``, because there is nothing to mask.
    * Any other rank, or an unknown one, raises ``ValueError`` naming
      ``owner`` and ``features_key``.

    # DECISION plan-2026-10-14T062616-81938e81/D-055
    Pool through ``SequencePooling``. Do NOT write a masked mean here: the
    package already has one rule for it (count clamped to 1, a dtype-aware
    max sentinel), and a second copy is a lockstep invariant. See
    decisions.md D-055.

    :param pooling: The head's weightless ``SequencePooling`` for this input.
    :type pooling: keras.layers.Layer
    :param inputs: The head's input dict.
    :type inputs: Dict[str, Any]
    :param features_key: Which features to pool.
    :type features_key: str
    :param owner: Head class name, for the error message.
    :type owner: str
    :return: ``(batch, width)``.
    :rtype: keras.KerasTensor
    :raises ValueError: On a mask for a rank-2 input, or a rank other than 2
        or 3.
    """
    features = inputs[features_key]
    mask_key = _MASK_KEYS[features_key]
    mask = inputs.get(mask_key)
    rank = static_rank(features, features_key, owner)
    if rank == 2:
        if mask is not None:
            raise ValueError(
                f"{owner}: {mask_key} was given for a rank-2 {features_key}. "
                f"A rank-2 input is already pooled, so there is nothing to "
                f"mask; drop {mask_key} or pass the rank-3 sequence."
            )
        return features
    if rank != 3:
        raise ValueError(
            f"{owner}: {features_key} must be rank 2 (pooled) or rank 3 "
            f"(batch, length, width), got rank {rank}."
        )
    return pooling(features, mask=mask)


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------


def _serialize_task_config(task_config: VLMTaskConfig) -> Dict[str, Any]:
    """Serialize a ``VLMTaskConfig`` to a JSON-safe dict.

    The dataclass stores ``task_type`` as a ``VLMTaskType`` enum, which is not
    JSON-serializable. This converts it to its string value, so the layer
    config survives a ``.keras`` round-trip.

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
    # VLMTaskConfig parses the string task type itself.
    return VLMTaskConfig(**dict(config))


# ---------------------------------------------------------------------
# Base VLM Head Class
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vlm.factory")
class BaseVLMHead(keras.layers.Layer):
    """
    Shared construction for the fusion layer two VLM heads use.

    This class creates the ``MultiModalFusion`` layer and stores the
    configuration it needs. It has no ``call``, so it is not a head:
    constructing it directly raises ``TypeError``.
    :class:`VisualGroundingHead` and :class:`ImageTextMatchingHead` inherit
    from it and supply their own forward pass. :class:`ImageCaptioningHead`,
    :class:`VQAHead` and :class:`MultiTaskVLMHead` do not inherit from it at
    all.

    It owns exactly one sub-layer, ``fusion``, because that is the one layer
    both subclasses run. Each subclass calls :meth:`_build_fusion_stack` from
    its own ``build`` with the shapes it will really pass.

    **Architecture Overview:**

    .. code-block:: text

        vision_features        text_features
        (B, N, D_vis)          (B, N, D_txt)
                  │                 │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────────┐
                  │ fusion              │
                  │ MultiModalFusion    │
                  └──────────┬──────────┘
                             ▼ (B, N, hidden_dim)
                       squeeze_axis (optional)
                             ▼
                    the subclass task layers

    Only :class:`ImageTextMatchingHead` runs a post-fusion norm and FFN, so
    that head owns them and takes their arguments (``normalization_type``,
    ``use_post_fusion_ffn``, ``ffn_type``, ``ffn_expansion_factor``).
    :class:`VisualGroundingHead` goes from ``fusion`` straight to its scorer
    and refuses those arguments, as an unrecognized keyword.

    Every feature width comes from the shapes handed to ``build``. The head
    takes no declared ``vision_dim``/``text_dim``; ``task_config`` carries
    ``hidden_size``, the working width.

    Two fusion strategies are rejected up front. ``cross_attention`` returns
    one tensor per modality and ``attention_pooling`` returns a rank-2 tensor,
    and both subclasses consume a single rank-preserving tensor. Six of the
    eight strategies pass. See :meth:`_build_fusion_stack`.

    :param task_config: Task configuration. ``hidden_size`` becomes
        ``hidden_dim``, the working width of the fusion.
    :type task_config: VLMTaskConfig
    :param fusion_strategy: Which ``MultiModalFusion`` strategy to build. It
        has no default here; each subclass supplies its own.
    :type fusion_strategy: FusionStrategy
    :param fusion_config: Extra keyword arguments for ``MultiModalFusion``.
    :type fusion_config: Optional[Dict[str, Any]]
    :param kwargs: Additional arguments for the base Layer.

    :ivar hidden_dim: Working width, taken from ``task_config.hidden_size``.
    :vartype hidden_dim: int
    :ivar fusion: The ``MultiModalFusion`` layer.
    :vartype fusion: MultiModalFusion
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        fusion_strategy: FusionStrategy,
        fusion_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration and create the fusion layer.

        The layer name is derived from ``task_config.name``, so a caller must
        not pass ``name``.

        :param task_config: Task configuration for this head.
        :type task_config: VLMTaskConfig
        :param fusion_strategy: Which ``MultiModalFusion`` strategy to build.
        :type fusion_strategy: FusionStrategy
        :param fusion_config: Extra keyword arguments for ``MultiModalFusion``.
        :type fusion_config: Optional[Dict[str, Any]]
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises TypeError: If this class is constructed directly rather than
            through a subclass.
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-059
        # Refuse direct construction. Do NOT hand back a bare BaseVLMHead: it
        # has no call(), so it constructed fine and raised
        # NotImplementedError, naming the base class, on first use. See
        # decisions.md D-059.
        if type(self) is BaseVLMHead:
            raise TypeError(
                "BaseVLMHead is the shared base of VisualGroundingHead and "
                "ImageTextMatchingHead and has no call(); construct one of "
                "those, or create_vlm_head() for a task type."
            )
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        # Store configuration
        self.task_config = task_config
        self.fusion_strategy = fusion_strategy
        self.fusion_config = fusion_config or {}

        # `VLMTaskConfig.__post_init__` in task_types.py refuses any
        # `hidden_size` that is not a positive integer, so a head never sees
        # None here. Three sites had drifted to two different fallback
        # expressions for a value that was never missing.
        #
        # DECISION plan-2026-07-30T081929-1645aa52/D-014
        # Read `hidden_size` straight. Do NOT re-add an `or <fallback>` in any
        # spelling: the fallback was dead code, proved by constructing with
        # `hidden_size=None` and dims 64/96, which yielded 768, NOT 96.
        # See decisions.md D-014.
        self.hidden_dim = task_config.hidden_size

        # DECISION plan-2026-10-14T062616-81938e81/D-032
        # The base owns the fusion and nothing else. Do NOT move the
        # post-fusion norm, dropout or FFN back here: VisualGroundingHead
        # never runs them (it created all three and built none), the dropout
        # ran in no call at all, and use_post_fusion_ffn was silently ignored
        # by the grounding head. Only ImageTextMatchingHead runs the stack,
        # so it owns it. See decisions.md D-032.
        self.fusion = MultiModalFusion(
            dim=self.hidden_dim,
            fusion_strategy=self.fusion_strategy,
            dropout_rate=self.task_config.dropout_rate,
            name=f"{self.name}_fusion",
            **self.fusion_config,
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Mark the layer built. It builds no sub-layer.

        It cannot build ``fusion``, because its shapes depend on how each
        subclass drives it. :class:`VisualGroundingHead` fuses per-region
        ``(B, N, D)`` tensors. :class:`ImageTextMatchingHead` fuses pooled
        ``(B, 1, D)`` tensors and then squeezes. Each subclass calls
        :meth:`_build_fusion_stack` from its own ``build`` with its real
        shapes.

        :param input_shape: Input shape(s), passed straight to the base class.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        super().build(input_shape)

    def _build_fusion_stack(
        self,
        fusion_input_shapes: List[Tuple[Optional[int], ...]],
        squeeze_axis: Optional[int] = None,
    ) -> Tuple[Optional[int], ...]:
        """Build ``fusion`` and return the shape the subclass consumes.

        Shapes are DERIVED via the fusion's ``compute_output_shape`` rather
        than recomputed from ``hidden_dim``, so this stays correct if a fusion
        strategy changes its output width.

        :param fusion_input_shapes: ``[vision_shape, text_shape]`` exactly as the
            subclass passes them to ``self.fusion``.
        :type fusion_input_shapes: List[Tuple[Optional[int], ...]]
        :param squeeze_axis: Axis the subclass squeezes off the fusion output,
            or ``None`` if it does not squeeze.
        :type squeeze_axis: Optional[int]
        :return: The fusion output shape, after the squeeze.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If the fusion strategy returns one shape per
            modality, or if it does not preserve rank.
        """
        self.fusion.build(fusion_input_shapes)
        fused_shape = self.fusion.compute_output_shape(fusion_input_shapes)

        # Two strategies are rejected here, for two different reasons.
        # `cross_attention` returns one tensor per modality. `attention_pooling`
        # returns a single rank-2 tensor, having already pooled away the axis the
        # caller means to squeeze. Both used to reach Keras as a shape and die
        # with an error naming neither the strategy nor the reason. The two fail
        # identically on the lazy and the explicit-build path, so this is a
        # capability gap, not a `build()` defect. The check reads the fusion
        # layer's declared output contract, not a hardcoded strategy list, so a
        # future strategy that returns a tuple is rejected on its own terms.
        #
        # Do NOT instead teach the subclasses to fan in a per-modality tuple.
        # Which tensor feeds the task head is a modelling decision nobody has
        # asked for.
        #
        # Measured over all 8 strategies at input shapes [(2,4,32),(2,4,32)]:
        # 7 return a single tensor, and 6 of those also preserve rank 3.
        #
        # DECISION plan-2026-07-30T081929-1645aa52/D-011
        # Raise twice. Once on a strategy whose output is not one tensor
        # (cross_attention). Once on one that drops rank (attention_pooling,
        # checked below and stated in this method's `:raises` field). 6 of the
        # 8 strategies pass both checks. See decisions.md D-011.
        if not is_single_shape(fused_shape):
            raise ValueError(
                f"fusion_strategy={self.fusion_strategy!r} produces one output "
                f"per modality (compute_output_shape returned "
                f"{fused_shape!r}), but {type(self).__name__} consumes a "
                f"single fused tensor. Use one of the "
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
        return tuple(fused_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this head.

        ``task_config`` is converted to a JSON-safe dict, because its
        ``task_type`` field holds an enum.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_config": _serialize_task_config(self.task_config),
                "fusion_strategy": self.fusion_strategy,
                "fusion_config": self.fusion_config,
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

@register_dl_technique("dl_techniques.layers.heads.vlm.factory")
class ImageCaptioningHead(keras.layers.Layer):
    """
    A transformer decoder over text, cross-attending to vision features.

    This head generates a caption one token at a time. Each decoder layer runs
    three sub-blocks in order: causal self-attention over the text stream,
    cross-attention onto the vision features, then an FFN. Every sub-block adds
    its output back onto its input and normalizes afterwards, so this is a
    post-norm decoder with three residual adds per layer.

    It does NOT inherit from :class:`BaseVLMHead` and has no fusion layer. It
    derives from ``keras.layers.Layer`` directly.

    The residual adds carry no projection, so the text feature width must
    already equal ``hidden_dim``. The vision width is independent: the vision
    features only enter as the cross-attention key and value.

    **Architecture Overview:**

    .. code-block:: text

        vision_features            text_features
        (B, N_vis, D_vis)          (B, S, hidden_dim)
               │                          │
               │                          ▼ x
               │              ┌──────────────────────┐
               │              │ self_attention[i]    │
               │              │ causal keep mask     │
               │              └───────────┬──────────┘
               │                          ▼
               │              ┌──────────────────────┐
               │              │ norm[3i](x + attn)   │
               │              └───────────┬──────────┘
               │                          ▼
               │              ┌──────────────────────┐
               └──── kv ────► │ cross_attention[i]   │
                              └───────────┬──────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ norm[3i+1](x + cross)│
                              └───────────┬──────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ ffn_layers[i]        │
                              └───────────┬──────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ norm[3i+2](x + ffn)  │
                              └───────────┬──────────┘
                                          ▼
                                repeat x num_layers
                                          ▼
                              ┌──────────────────────┐
                              │ output_proj Dense    │
                              └───────────┬──────────┘
                              ┌────────┴────────┐
                              ▼                 ▼
                          'logits'      'hidden_states'
                    (B, S, vocab_size)  (B, S, hidden_dim)

    The causal mask comes from ``MaskFactory.create_causal_mask``, not from
    ``keras.ops.tril``. See the note in ``call``.

    Input shape:
        ``{'vision_features': (batch, num_vision, vision_width),
        'text_features': (batch, seq_len, hidden_dim)}``, plus the optional
        keep masks ``'vision_mask': (batch, num_vision)`` and
        ``'text_mask': (batch, seq_len)``. The vision mask is the
        cross-attention's key mask. The text mask is combined with the causal
        mask in each self-attention, so a real token never attends to a
        padded one, which matters for left padding; the outputs at padded
        text positions are not meaningful and belong out of the loss.

    Output shape:
        ``{'logits': (batch, seq_len, vocab_size), 'hidden_states':
        (batch, seq_len, hidden_dim)}``.

    :param task_config: Configuration object for the task.
        ``vocab_size`` is required; ``num_classes`` must be unset.
    :type task_config: VLMTaskConfig
    :param num_layers: Number of decoder layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param ffn_type: Type of feed-forward network in decoder blocks.

        **Not every registry type is usable, and this is the ONE place that says
        which.** Re-derive the list rather than trusting it. Run this::

            CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
              tests/test_layers/test_heads/test_vlm.py \
              -k "OutputWidthParamRouting" -v

        Both FFN construction sites supply ``hidden_dim`` conditionally from
        ``FFN_REGISTRY`` (D-008/D-020). They supply the OUTPUT width by the
        type's own parameter name, through ``_ffn_width_kwargs`` (D-014). The
        two sites are this head's per-layer loop and
        ``ImageTextMatchingHead._create_post_fusion_layers``. What that closed, and what remains
        closed and why:

        * ``kan`` and ``power_mlp`` -- newly usable on BOTH sites. Their width
          parameter was simply named something else (``features`` / ``units``)
          and was the only thing they were missing.
        * ``tversky`` -- still closed on BOTH sites, for two independent
          reasons. It also requires ``num_features``, a feature-bank size with
          no source in ``VLMTaskConfig``. And ``TverskyProjectionLayer.build()``
          hard-raises on anything but rank-2, which this head's rank-3 decoder
          stream can never satisfy.
        * ``gated_mlp`` -- a permanent capability gap. ``GatedMLP`` is 1x1-conv
          based and needs a rank-4 ``(B, H, W, C)`` input; this head passes
          rank-3 and ``BaseVLMHead`` rank-2. Its ``filters`` width parameter now
          reaches it, which is exactly why the failure moved from a missing
          parameter to a kernel-rank error.
        * ``counting``, ``logic``, ``mixer`` -- closed by a hyperparameter
          ``VLMTaskConfig`` does not carry and cannot derive: ``count_dim``,
          ``logic_dim``, and ``tokens_mlp_dim`` + ``channels_mlp_dim``
          respectively. Inventing defaults for those is a modelling decision,
          not a lookup, so it was not done. ``mixer`` also has no output-width
          concept at all. Its output shape is its input shape, so its
          ``output_dim_param`` is ``None``.

        The superseded claim, recorded so it is not re-derived: this docstring
        used to say the failures were caused by ``output_dim`` being hardcoded.
        Execution disproved that. ``output_dim`` was either accepted and
        present, or dropped by the factory's parameter filter, which raises as
        of D-023. Every failing type died earlier inside ``validate_ffn_config``
        on a DIFFERENT required key.

        Related, and also measured rather than assumed: several combinations
        once built and then failed inside ``call()``. Each is now refused in
        ``build()``, by name (D-011, D-056): ``VisualGroundingHead`` with
        ``fusion_strategy="attention_pooling"``, ``ImageCaptioningHead`` with a
        text width other than ``hidden_dim`` or a 2-D vision or text input,
        and ``VQAHead`` attention pooling with a width other than
        ``hidden_dim``. ``VQAHead`` ``"mean"``/``"max"`` on 2-D inputs, the
        last of them, now works: a 2-D input is taken as already pooled.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: Width multiplier for the decoder FFN's hidden
        layer. Defaults to 4.

        Read by every FFN type that can use it, via one of two channels. Types
        that REQUIRE an explicit ``hidden_dim`` receive
        ``hidden_dim * ffn_expansion_factor``. Types that derive their own width
        receive the factor itself and apply their own rule to it. Of the 8 such
        registry types only ``swiglu`` does, and its rule is 2/3 of
        ``output_dim * factor``, rounded up to ``ffn_multiple_of`` (D-020).

        The remaining 7 optional-``hidden_dim`` types (``kan``, ``mixer``,
        ``tversky``, ...) have no expansion concept. They size themselves from
        unrelated parameters, so this value does not reach them.

    :type ffn_expansion_factor: int
    :param kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_type: FFNType = "swiglu",
        ffn_expansion_factor: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration and create every decoder sub-layer.

        The loop creates, per decoder layer, one self-attention layer, one
        cross-attention layer, one FFN and three normalization layers. The
        norms are kept in one flat list, so layer ``i`` owns indices ``3i``,
        ``3i + 1`` and ``3i + 2``. One final ``Dense`` projects to the
        vocabulary.

        The layer name is derived from ``task_config.name``, so a caller must
        not pass ``name``.

        :param task_config: Configuration object for the task.
        :type task_config: VLMTaskConfig
        :param num_layers: Number of decoder layers.
        :type num_layers: int
        :param num_heads: Number of attention heads. ``hidden_dim`` must divide
            by it.
        :type num_heads: int
        :param ffn_type: Which registered FFN type each decoder layer builds.
        :type ffn_type: FFNType
        :param ffn_expansion_factor: FFN width multiplier.
        :type ffn_expansion_factor: int
        :param kwargs: Additional arguments for the base Layer class.
        :raises ValueError: If ``hidden_dim`` is not divisible by
            ``num_heads``, ``task_config.vocab_size`` is unset, or
            ``task_config.num_classes`` is set.
        :return: None.
        :rtype: None
        """
        _check_task_config_fields(
            task_config, "ImageCaptioningHead", required=("vocab_size",))
        super().__init__(name=f"{task_config.name}_head", **kwargs)
        self.task_config = task_config
        # D-014: dead fallback removed (see BaseVLMHead.__init__).
        self.hidden_dim = task_config.hidden_size
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
            #
            # The OUTPUT width is routed by name, not hardcoded -- same helper,
            # same arguments as `ImageTextMatchingHead._create_post_fusion_layers` (I-4). See
            # `_ffn_width_kwargs` (D-014) for why the literal string
            # `"output_dim"` must not reappear here, and the `ffn_type` docstring
            # above for which types this does and does not make reachable.
            ffn_config = {
                "type": self.ffn_type,
                "name": f"ffn_{i}",
                **_ffn_width_kwargs(self.ffn_type, self.hidden_dim),
            }
            _entry = FFN_REGISTRY.get(self.ffn_type, {})
            if "hidden_dim" in _entry.get("required_params", ()):
                ffn_config["hidden_dim"] = self.hidden_dim * self.ffn_expansion_factor
            elif "ffn_expansion_factor" in set(
                _entry.get("required_params", ())
            ) | set(_entry.get("optional_params", {})):
                # D-020: forward the factor to types that derive their own width,
                # or this head's `ffn_expansion_factor` is inert for them. Same
                # rule as `ImageTextMatchingHead._create_post_fusion_layers` -- keeping the two
                # sites identical is the entire point of D-008.
                ffn_config["ffn_expansion_factor"] = self.ffn_expansion_factor

            # D-019: same pre-filter as `ImageTextMatchingHead._create_post_fusion_layers`, and
            # for the same reason (keeping the two sites identical is I-4).
            # MEASURED difference from site 1, recorded so nobody "fixes" a
            # symmetry that is not there: at HEAD this loop emitted ZERO dropped
            # keys over all 21 types, because -- unlike site 1 -- it never
            # injects `dropout_rate`, and every other key it sends is already
            # gated on the registry. The pre-filter is therefore a no-op here
            # TODAY; it is present so that adding one unconditional convenience
            # cannot silently re-arm the hazard.
            ffn_config = assemble_ffn_config(self.ffn_type, ffn_config)
            ffn = create_ffn_from_config(ffn_config)
            self.ffn_layers.append(ffn)

            norm1 = create_normalization_layer("rms_norm", name=f"norm1_{i}")
            norm2 = create_normalization_layer("rms_norm", name=f"norm2_{i}")
            norm3 = create_normalization_layer("rms_norm", name=f"norm3_{i}")
            self.norm_layers.extend([norm1, norm2, norm3])

        # Final projection to vocabulary
        self.output_proj = keras.layers.Dense(
            self.task_config.vocab_size, name=f"{self.name}_output_proj"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer, in computational order.

        Without this, the sub-layers created in ``__init__`` stay unbuilt until
        Keras traces ``call()``. A ``.keras`` round-trip through a Functional
        model then fails to restore them. Measured as ``ValueError: A total of
        12 objects could not be loaded``, with ``<Dense name=kv, built=False>``
        as the example. This is the repo's documented lazy-sublayer
        serialization trap.

        Note the shape contract this head already relies on. ``call()`` adds
        each sub-block's output straight onto its input (``x + attn_output``),
        with no projection anywhere. So the text feature width must equal
        ``hidden_dim``, and the builds below use ``hidden_dim`` for the whole
        decoder stack. The vision width is independent. It only enters as the
        cross-attention's key/value width.

        :param input_shape: Shape dict with ``vision_features`` and
            ``text_features`` entries, as passed to ``call()``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        text_shape = input_shape["text_features"]
        vision_shape = input_shape["vision_features"]
        for key, shape in (("text_features", text_shape),
                           ("vision_features", vision_shape)):
            if len(shape) != 3:
                raise ValueError(
                    f"ImageCaptioningHead: {key} must be rank 3, (batch, "
                    f"length, width), got shape {tuple(shape)}. The decoder "
                    f"attends over both sequences; a pooled input has none."
                )
        batch = text_shape[0]
        seq_len = text_shape[1]

        # DECISION plan-2026-10-14T062616-81938e81/D-056
        # Refuse a text width other than hidden_dim here, at build. Do NOT
        # rely on the residual add: before this check the head built and
        # then failed inside MultiHeadCrossAttention.call, naming neither the
        # head nor text_features. See decisions.md D-056.
        if text_shape[-1] is not None and text_shape[-1] != self.hidden_dim:
            raise ValueError(
                f"ImageCaptioningHead: text_features is {text_shape[-1]} wide, "
                f"but the decoder adds every sub-block back onto it without a "
                f"projection, so it must be task_config.hidden_size="
                f"{self.hidden_dim} wide. Project the text embeddings first; "
                f"vision_features may be any width."
            )

        # The decoder stream carries `hidden_dim` throughout (see the residual
        # note above).
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
        """
        Run the decoder stack and project to the vocabulary.

        ``text_features`` must already be embedded. This head holds no
        embedding table, and the residual adds carry no projection, so the text
        width must equal ``hidden_dim``.

        :param inputs: Dict with ``vision_features`` and ``text_features``,
            and optionally ``vision_mask`` and ``text_mask``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``logits`` and ``hidden_states``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        vision_features = inputs["vision_features"]
        # The text is assumed pre-embedded; this head holds no embedding table.
        text_features = inputs["text_features"]
        vision_mask = inputs.get("vision_mask")
        text_mask = inputs.get("text_mask")

        x = text_features
        seq_len = keras.ops.shape(x)[1]
        # Lower-triangular KEEP mask: 1 = attend to current and past, 0 = future.
        #
        # Build it from `MaskFactory.create_causal_mask`, an arange index
        # comparison, and not from `keras.ops.tril`. `keras.ops.tril` and `keras.ops.triu` share an
        # implementation that routes through a `tf.cond` rejecting a Python-bool
        # predicate once traced, raising `TypeError: pred must not be a Python
        # bool`. That works eagerly and fails on every graph path: `tf.function`,
        # `Model.predict`, `.keras` save and load, and `jit_compile=True`. It
        # fails for both static and symbolic sequence lengths. Keras downgrades
        # such a crash during build tracing to a UserWarning, so a green test
        # suite could not see it. The same trap is documented in the SD3 MMDiT
        # text encoders.
        #
        # MaskFactory returns the BLOCK polarity, True where a position must be
        # suppressed. This site needs the complementary KEEP mask as a float,
        # hence `logical_not` plus a cast.
        #
        # This block-to-keep adapter now exists at two sites: here and in the
        # MobileCLIP components module. Two is not enough to earn a shared
        # helper. A third consumer should promote it into a keep-polarity
        # variant in `utils/masking/factory.py` rather than copying it again.
        # Promoting it is deferred because it touches two shipped,
        # mixed-precision-sensitive call sites and must preserve the cast target
        # below. That target is `backend.floatx()` because it is what the
        # previous `keras.ops.ones(...)` defaulted to, which keeps the change
        # numerically inert.
        causal_mask = keras.ops.cast(
            keras.ops.logical_not(MaskFactory.create_causal_mask(seq_len, dtype="bool")),
            keras.config.floatx(),
        )
        # Expand to the (1, S, S) full-mask form the attention layers expect.
        causal_mask = keras.ops.expand_dims(causal_mask, 0)
        if text_mask is not None:
            # A padded token is no key for any query: (1, S, S) * (B, 1, S).
            # A query row left with no key (a left-padded position) is
            # rescued by the attention layer; its output is padding anyway.
            causal_mask = causal_mask * keras.ops.expand_dims(
                keras.ops.cast(text_mask, causal_mask.dtype), 1)

        for i in range(self.num_layers):
            # Self-attention with causal mask
            attn_output = self.self_attention_layers[i](
                x, attention_mask=causal_mask, training=training
            )
            x = self.norm_layers[i * 3](x + attn_output)

            # Cross-attention to vision features. `vision_mask` (B, N_vis) is
            # a key-padding mask, shared by every text query.
            cross_attn_output = self.cross_attention_layers[i](
                x, kv_input=vision_features, attention_mask=vision_mask,
                training=training
            )
            x = self.norm_layers[i * 3 + 1](x + cross_attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.norm_layers[i * 3 + 2](x + ffn_output)

        logits = self.output_proj(x)
        return {"logits": logits, "hidden_states": x}

    def compute_output_shape(self, input_shape):
        """
        Return the output shapes, mirroring the ``call`` return value.

        :param input_shape: Shape dict; only ``text_features`` is read.
        :type input_shape: Dict
        :return: Shapes for ``logits`` and ``hidden_states``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        text_shape = input_shape["text_features"]
        batch, seq_len = text_shape[0], text_shape[1]
        return {
            "logits": (batch, seq_len, self.task_config.vocab_size),
            "hidden_states": (batch, seq_len, self.hidden_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this head.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_config": _serialize_task_config(self.task_config),
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ffn_type": self.ffn_type,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImageCaptioningHead":
        """
        Reconstruct the head, deserializing the ``task_config`` dataclass.

        The layer name is regenerated from ``task_config.name`` in
        ``__init__``, so the stored ``name`` is dropped to avoid a duplicate
        keyword.

        :param config: Serialized layer configuration.
        :type config: Dict[str, Any]
        :return: A reconstructed head instance.
        :rtype: ImageCaptioningHead
        """
        config = dict(config)
        config.pop("name", None)
        config["task_config"] = _deserialize_task_config(config["task_config"])
        return cls(**config)


# ---------------------------------------------------------------------
# Visual Question Answering Head
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.heads.vlm.factory")
class VQAHead(keras.layers.Layer):
    """
    Pools each modality, concatenates them, and classifies the answer.

    This head picks one answer from a fixed vocabulary. It pools the vision
    features and the question features to one vector each, joins them end to
    end, and runs the result through an MLP.

    It does NOT inherit from :class:`BaseVLMHead` and has no fusion layer. It
    derives from ``keras.layers.Layer`` directly.

    It is also the one head in this module that reads ``question_features``
    rather than ``text_features``. :class:`MultiTaskVLMHead` fans one input
    dict to every head, so mixing this head with any other needs both keys.

    **Architecture Overview:**

    .. code-block:: text

        vision_features           question_features
        (B, N_vis, D_vis)         (B, S, D_txt)
                │                          │
                ▼                          ▼
        ┌────────────────┐        ┌────────────────┐
        │ pool vision    │        │ pool question  │
        │ mean/max/attn  │        │ mean/max/attn  │
        └───────┬────────┘        └───────┬────────┘
                ▼ (B, D_v)                ▼ (B, D_q)
                └────────────┬────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ concatenate axis -1 │
                  └──────────┬──────────┘
                             ▼ (B, pooled_width)
                  ┌─────────────────────┐
                  │ hidden_layers[i]    │
                  │ Dense gelu          │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ dropout_layers[i]   │
                  └──────────┬──────────┘
                             ▼
                   repeat x len(hidden_dims)
                             ▼
                  ┌─────────────────────┐
                  │ output_layer Dense  │
                  └──────────┬──────────┘
                             ▼
                   'answer_logits' (B, num_classes)

    The ``"attention"`` strategy is not two pooling layers. One
    cross-attention layer runs in both directions, vision attending to the
    question and the question attending to vision, and each attended result is
    then mean-pooled. Running it in reverse needs the two feature widths to
    match, which is a standing contract of this head.

    ``pooled_width`` depends on the strategy. ``mean`` and ``max`` keep each
    modality's own channel count, so it is ``D_vis + D_txt``. ``attention``
    maps both to ``embed_dim``, so it is ``2 * embed_dim``.

    Every pooling honours the optional keep masks ``'vision_mask'`` and
    ``'question_mask'`` (``(batch, length)``, 1 = real token). ``mean`` and
    ``max`` reduce over the real positions only. ``attention`` passes each
    mask as the key mask of the direction that attends to that modality, then
    mean-pools each attended result over its own real positions.

    Input shape:
        ``{'vision_features': (batch, num_vision, vision_width),
        'question_features': (batch, seq_len, text_width)}``, plus the
        optional masks. ``mean`` and ``max`` also take rank-2 (already
        pooled) features, and refuse a mask for them. ``attention`` needs
        both inputs rank 3 and ``hidden_size`` wide, checked in ``build``.

    Output shape:
        ``{'answer_logits': (batch, num_classes)}``.

    :param task_config: Configuration object for the task. ``num_classes``
        is required and ``vocab_size`` must be unset. ``hidden_size`` becomes
        ``embed_dim``; only the ``"attention"`` strategy reads it.
    :type task_config: VLMTaskConfig
    :param hidden_dims: Widths of the classifier MLP's hidden layers.
    :type hidden_dims: Sequence[int]
    :param pooling_strategy: One of ``"mean"``, ``"max"`` or ``"attention"``.
    :type pooling_strategy: str
    :param num_heads: Head count of the ``"attention"`` strategy's
        cross-attention. ``None`` means 8. It must divide ``hidden_size``.
        The other two strategies run no attention and refuse a value.
    :type num_heads: Optional[int]
    :param kwargs: Additional arguments for the base Layer.

    :ivar embed_dim: Attention width, taken from ``task_config.hidden_size``.
    :vartype embed_dim: int
    :ivar num_heads: The resolved head count, or ``None`` without attention.
    :vartype num_heads: Optional[int]
    :ivar attention_pooling: Cross-attention layer, or ``None`` when the
        strategy is not ``"attention"``.
    :vartype attention_pooling: Optional[keras.layers.Layer]
    :ivar vision_pooling: Weightless ``SequencePooling`` over the vision
        sequence (``'max'`` for the max strategy, ``'mean'`` otherwise).
    :vartype vision_pooling: SequencePooling
    :ivar question_pooling: The same, over the question sequence.
    :vartype question_pooling: SequencePooling
    :ivar hidden_layers: The classifier MLP's ``Dense`` layers.
    :vartype hidden_layers: List[keras.layers.Dense]
    :ivar dropout_layers: One ``Dropout`` per hidden layer.
    :vartype dropout_layers: List[keras.layers.Dropout]
    :ivar output_layer: Final ``Dense`` to ``num_classes``.
    :vartype output_layer: keras.layers.Dense
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        hidden_dims: Sequence[int] = (512, 256),
        pooling_strategy: str = "attention",
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration and create the pooling and MLP sub-layers.

        ``attention_pooling`` is created only for the ``"attention"``
        strategy. The two weightless ``SequencePooling`` layers are created
        for every strategy. One ``Dense`` and one ``Dropout`` are created per
        entry in ``hidden_dims``, plus the output ``Dense``.

        The layer name is derived from ``task_config.name``, so a caller must
        not pass ``name``.

        :param task_config: Configuration object for the task.
        :type task_config: VLMTaskConfig
        :param hidden_dims: Widths of the classifier MLP's hidden layers.
        :type hidden_dims: Sequence[int]
        :param pooling_strategy: One of ``"mean"``, ``"max"``, ``"attention"``.
        :type pooling_strategy: str
        :param num_heads: Attention head count; ``None`` means 8. Only the
            ``"attention"`` strategy takes one.
        :type num_heads: Optional[int]
        :param kwargs: Additional arguments for the base Layer class.
        :raises ValueError: If ``task_config.num_classes`` is unset or
            ``task_config.vocab_size`` is set, if ``pooling_strategy`` is not
            one of the three, if ``num_heads`` is not a positive integer or
            does not divide ``hidden_size``, or if ``num_heads`` is given
            without the ``"attention"`` strategy.
        :return: None.
        :rtype: None
        """
        _check_task_config_fields(task_config, "VQAHead", required=("num_classes",))
        if pooling_strategy not in ["mean", "max", "attention"]:
            raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")

        # DECISION plan-2026-10-14T062616-81938e81/D-056
        # The head count is an argument, resolved to 8 only where it is read.
        # Do NOT hardcode it again (hidden_size=36 died inside the attention
        # factory naming neither VQAHead nor a knob the caller could change),
        # and do NOT accept it for mean/max, which run no attention (a
        # default-valued dead knob is the D-030 defect). See decisions.md
        # D-056.
        if pooling_strategy == "attention":
            num_heads = positive_int(
                8 if num_heads is None else num_heads, "num_heads", "VQAHead")
            if task_config.hidden_size % num_heads != 0:
                raise ValueError(
                    f"VQAHead: task_config.hidden_size={task_config.hidden_size} "
                    f"must be divisible by num_heads={num_heads} for the "
                    f"attention pooling; pass num_heads=... to match."
                )
        elif num_heads is not None:
            raise ValueError(
                f"VQAHead: num_heads={num_heads!r} is read only by "
                f"pooling_strategy='attention'; pooling_strategy="
                f"{pooling_strategy!r} runs no attention. Leave it unset."
            )
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        # Store configuration
        self.task_config = task_config
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: the DEFAULT is a
        # tuple (R-009 S1) and the STORED attribute is a list. Keeping the
        # store as `list(...)` is what makes the conversion invisible: it is
        # the type `get_config` has always emitted, so a saved config's JSON
        # shape and every `== [..]` assertion in the suites are unchanged.
        self.hidden_dims = list(hidden_dims)
        self.pooling_strategy = pooling_strategy
        self.num_heads = num_heads
        # D-014: dead fallback removed (see BaseVLMHead.__init__).
        self.embed_dim = self.task_config.hidden_size

        # CREATE sub-layers
        if self.pooling_strategy == "attention":
            self.attention_pooling = create_attention_layer(
                "multi_head_cross",
                dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name="attention_pooling",
            )
        else:
            self.attention_pooling = None

        # Weightless; `max` reduces with `max`, the other two with `mean`.
        reduction = "max" if self.pooling_strategy == "max" else "mean"
        self.vision_pooling = _weightless_pooling(reduction, "vision_pooling")
        self.question_pooling = _weightless_pooling(reduction, "question_pooling")

        self.hidden_layers = []
        self.dropout_layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                keras.layers.Dense(hidden_dim, activation="gelu", name=f"hidden_{i}")
            )
            self.dropout_layers.append(
                keras.layers.Dropout(self.task_config.dropout_rate, name=f"dropout_{i}")
            )

        self.output_layer = keras.layers.Dense(
            self.task_config.num_classes, name="output_layer"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        Only the sub-layers this head's ``pooling_strategy`` actually exercises
        are built. ``attention_pooling`` exists only for ``"attention"``.
        Building it otherwise would create weights the lazy path never created.

        The classifier's input width is derived from the ACTUAL input shapes,
        not from any declared width. The two pooling branches produce
        different widths. ``mean``/``max`` keep each modality's own channel
        count, while ``attention`` maps both to ``embed_dim``.

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
            # (vision<-question and question<-vision), so each modality is a
            # query once and must be `embed_dim` wide. The attention layer
            # checks only the query of the direction it is built with, so
            # the question width used to pass build and fail in call; both
            # are checked here (D-056).
            for key, shape in (("vision_features", vision_shape),
                               ("question_features", question_shape)):
                if shape[-1] is not None and shape[-1] != self.embed_dim:
                    raise ValueError(
                        f"VQAHead: {key} is {shape[-1]} wide, but "
                        f"pooling_strategy='attention' runs one cross-attention "
                        f"in both directions, so both inputs must be "
                        f"task_config.hidden_size={self.embed_dim} wide. Project "
                        f"them first, or use pooling_strategy='mean' or 'max', "
                        f"which take any widths."
                    )
            self.attention_pooling.build([vision_shape, question_shape])
            pooled_width = self.embed_dim * 2
        else:
            pooled_width = vision_shape[-1] + question_shape[-1]

        # The poolers are weightless; building them only marks them built.
        attended = self.pooling_strategy == "attention"
        self.vision_pooling.build(
            vision_shape[:-1] + (self.embed_dim,) if attended else vision_shape)
        self.question_pooling.build(
            question_shape[:-1] + (self.embed_dim,) if attended else question_shape)

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
        """
        Pool both modalities, concatenate them, and classify.

        ``"mean"`` and ``"max"`` reduce over axis 1, so they need rank-3
        inputs. ``"attention"`` runs one cross-attention layer in both
        directions and mean-pools each attended result.

        :param inputs: Dict with ``vision_features`` and ``question_features``,
            and optionally ``vision_mask`` and ``question_mask``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with a single ``answer_logits`` entry.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``pooling_strategy`` is not one of the three,
            or a mask is given for a rank-2 input.
        """
        if self.pooling_strategy in ("mean", "max"):
            vision_pooled = _pool_sequence(
                self.vision_pooling, inputs, "vision_features", "VQAHead")
            text_pooled = _pool_sequence(
                self.question_pooling, inputs, "question_features", "VQAHead")
        elif self.pooling_strategy == "attention":
            vision_features = inputs["vision_features"]
            question_features = inputs["question_features"]
            vision_mask = inputs.get("vision_mask")
            question_mask = inputs.get("question_mask")
            # Each direction masks the modality it attends TO, then the
            # attended sequence is mean-pooled over its own real positions.
            vision_attended = self.attention_pooling(
                vision_features, kv_input=question_features,
                attention_mask=question_mask, training=training
            )
            text_attended = self.attention_pooling(
                question_features, kv_input=vision_features,
                attention_mask=vision_mask, training=training
            )
            vision_pooled = self.vision_pooling(vision_attended, mask=vision_mask)
            text_pooled = self.question_pooling(text_attended, mask=question_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        x = keras.ops.concatenate([vision_pooled, text_pooled], axis=-1)

        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x, training=training)

        logits = self.output_layer(x)
        return {"answer_logits": logits}

    def compute_output_shape(self, input_shape):
        """
        Return the output shape, mirroring the ``call`` return value.

        ``call`` concatenates the two pooled vectors, so the batch axis is
        whichever of the two inputs states it (VLM-4 review item: this read
        the vision batch alone and lost a batch only the question stated).

        :param input_shape: Shape dict with ``vision_features`` and
            ``question_features``.
        :type input_shape: Dict
        :return: Shape for ``answer_logits``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch = input_shape["vision_features"][0]
        if batch is None:
            batch = input_shape["question_features"][0]
        return {"answer_logits": (batch, self.task_config.num_classes)}

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this head.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_config": _serialize_task_config(self.task_config),
                "hidden_dims": self.hidden_dims,
                "pooling_strategy": self.pooling_strategy,
                "num_heads": self.num_heads,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VQAHead":
        """
        Reconstruct the head, deserializing the ``task_config`` dataclass.

        The layer name is regenerated from ``task_config.name`` in
        ``__init__``, so the stored ``name`` is dropped to avoid a duplicate
        keyword.

        :param config: Serialized layer configuration.
        :type config: Dict[str, Any]
        :return: A reconstructed head instance.
        :rtype: VQAHead
        """
        config = dict(config)
        config.pop("name", None)
        config["task_config"] = _deserialize_task_config(config["task_config"])
        return cls(**config)


# ---------------------------------------------------------------------
# Visual Grounding Head
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.heads.vlm.factory")
class VisualGroundingHead(BaseVLMHead):
    """
    Scores image regions against a text query and boxes the best one.

    This head fuses each visual region with the pooled text query, scores
    every region, then regresses a bounding box from the single top-scoring
    region. It inherits from :class:`BaseVLMHead` and uses its ``fusion``
    layer, defaulting the strategy to ``"gated"``.

    It runs no post-fusion stack. ``call`` goes straight from ``fusion`` to
    ``confidence_scorer``, so the head owns no post-fusion norm or FFN and
    takes none of their arguments: ``normalization_type``,
    ``use_post_fusion_ffn``, ``ffn_type`` and ``ffn_expansion_factor`` raise
    ``ValueError`` (unrecognized keyword) here. They belong to
    :class:`ImageTextMatchingHead`, the one head that runs them.

    ``vision_features`` must be rank 3. The regions are whatever the caller
    put on axis 1; this head does not propose them.

    Both inputs must also be ``hidden_dim`` wide. ``D_txt`` and ``D_vis`` are
    not independent, despite the separate symbols on the diagram edges below.
    With ``hidden_size=64``, a 32-wide text stream raises ``ValueError:
    Modality 1 dimension 32 doesn't match expected dim 64`` from the fusion
    layer. Project both streams to one width before this head.

    The two outputs are treated differently, and the diagram says which is
    which. ``confidence_scorer`` is a linear ``Dense(1)``, so ``'confidence'``
    holds one pre-sigmoid LOGIT per region: it is not confined to ``[0, 1]``.
    Train it with ``BinaryCrossentropy(from_logits=True)``, or apply
    ``keras.ops.sigmoid`` yourself for a score. The region pick is
    unaffected, because the sigmoid is monotonic and ``argmax`` of the
    logits is ``argmax`` of their sigmoid. The box is a range map, not a
    class score: ``bbox_regressor`` is linear, and ``call`` applies
    ``keras.ops.sigmoid`` to its output so ``'bbox'`` lies in ``(0, 1)``.

    **Architecture Overview:**

    .. code-block:: text

        text_features (B, S, D_txt)
                             │
                             ▼
                  ┌─────────────────────┐
                  │ mean over axis 1    │
                  └──────────┬──────────┘
                             ▼ (B, D_txt)
                  ┌─────────────────────┐
                  │ tile to N_regions   │
                  └──────────┬──────────┘
                             ▼
        vision_features ─────┤ (B, N_regions, D_vis)
                             ▼
                  ┌─────────────────────┐
                  │ fusion, gated       │
                  └──────────┬──────────┘
                             ▼ (B, N_regions, hidden_dim)
                  ┌─────────────────────┐
                  │ confidence_scorer   │
                  │ Dense 1, linear     │
                  └──────────┬──────────┘
                             ▼ squeeze
                    ┌────────┴────────┐
                    ▼                 ▼
             'confidence'      argmax over regions
            (B, N_regions)     take_along_axis
            region logits
                                  │
                                  ▼ (B, hidden_dim)
                          ┌───────┴─────────┐
                          ▼                 ▼
               ┌─────────────────────┐  'grounded_features'
               │ bbox_regressor      │
               │ Dense 4, linear     │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │ keras.ops.sigmoid in call │
               └──────────┬──────────┘
                          ▼
                    'bbox' (B, 4)

    The region is picked with ``keras.ops.argmax`` and ``keras.ops.take_along_axis``. See
    the note in ``call`` about why NumPy fancy indexing cannot be used.

    Two optional keep masks (``(batch, length)``, 1 = real): ``'text_mask'``
    restricts the query mean to the real text tokens, and ``'vision_mask'``
    keeps a padded region out of the pick, so ``'bbox'`` and
    ``'grounded_features'`` come from a real region. ``'confidence'`` still
    holds one logit for every region, padded ones included; leave those out
    of the loss.

    Input shape:
        ``{'vision_features': (batch, num_regions, hidden_dim),
        'text_features': (batch, seq_len, hidden_dim)}``, plus the optional
        masks. A rank-2 ``text_features`` is accepted and used as the query
        directly; a ``'text_mask'`` for it is refused.

    Output shape:
        ``{'bbox': (batch, 4), 'confidence': (batch, num_regions),
        'grounded_features': (batch, hidden_dim)}``. ``'bbox'`` is in
        ``(0, 1)``; ``'confidence'`` is region logits.

    :param kwargs: Arguments for :class:`BaseVLMHead` (``task_config``,
        ``fusion_strategy``, ``fusion_config`` and the Keras base arguments).
        ``fusion_strategy`` defaults to ``"gated"`` here rather than to the
        base class value.

    :ivar bbox_regressor: Linear ``Dense`` producing the four box values;
        ``call`` applies the sigmoid.
    :vartype bbox_regressor: keras.layers.Dense
    :ivar confidence_scorer: Linear ``Dense`` producing one logit per region.
    :vartype confidence_scorer: keras.layers.Dense
    :ivar text_pooling: Weightless ``SequencePooling('mean')`` for the query.
    :vartype text_pooling: SequencePooling
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Default the fusion strategy, then create the task layers.

        :param kwargs: Arguments for :class:`BaseVLMHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_config`` sets ``vocab_size`` or
            ``num_classes``, which this head does not read.
        """
        if "task_config" in kwargs:
            _check_task_config_fields(kwargs["task_config"], "VisualGroundingHead")
        # A gated strategy scores per-region interactions, which is what this
        # head needs; the caller can still override it.
        kwargs.setdefault("fusion_strategy", "gated")
        super().__init__(**kwargs)

        self.text_pooling = _weightless_pooling("mean", f"{self.name}_text_pooling")

        self.bbox_regressor = keras.layers.Dense(4, name=f"{self.name}_bbox_regressor")
        # DECISION plan-2026-10-14T062616-81938e81/D-003
        # Linear: 'confidence' carries region logits under its old name. Do
        # NOT put activation='sigmoid' back: the sigmoid belongs to the loss
        # (from_logits=True) or the caller, as for every class-like output of
        # layers/heads, and the argmax region pick is the same either way.
        # The bbox sigmoid in call() is a range map and stays. See
        # decisions.md D-003.
        self.confidence_scorer = keras.layers.Dense(
            1, name=f"{self.name}_confidence"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        ``call()`` pools the text features to one query vector and TILES it
        across the vision regions. The fusion therefore sees two
        ``(B, N_regions, D)`` tensors. The text side keeps its own channel
        count, not a pooled-away one.

        The head has no post-fusion layers to build: its ``call()`` goes
        straight from ``fusion`` to ``confidence_scorer``.

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

        fused_shape = self._build_fusion_stack([vision_shape, text_tiled_shape])
        self.text_pooling.build(text_shape)

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
        """
        Score every region, then box the top-scoring one.

        :param inputs: Dict with ``vision_features``, shaped
            ``(B, N_regions, D_vis)``, and ``text_features``, and optionally
            ``vision_mask`` and ``text_mask``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to the fusion layer.
        :type training: Optional[bool]
        :return: Dict with ``bbox`` (sigmoid range map), ``confidence``
            (region logits) and ``grounded_features``.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``vision_features`` is not rank 3, or a
            ``text_mask`` is given for a rank-2 ``text_features``.
        """
        vision_features = inputs["vision_features"]
        vision_mask = inputs.get("vision_mask")

        if static_rank(vision_features, "vision_features", "VisualGroundingHead") != 3:
            raise ValueError(
                "VisualGroundingHead: vision_features must be rank 3, "
                "(batch, num_regions, width); the regions are scored one by one."
            )

        # Pool text features to a single query vector, over the real tokens.
        text_query = _pool_sequence(
            self.text_pooling, inputs, "text_features", "VisualGroundingHead")

        # Align text query with each visual region for fusion.
        num_regions = keras.ops.shape(vision_features)[1]
        text_expanded = keras.ops.expand_dims(text_query, axis=1)
        text_expanded = keras.ops.tile(text_expanded, [1, num_regions, 1])

        # Fuse each region with the text query. The output is [B, N_regions, D_fused].
        fused_per_region = self.fusion([vision_features, text_expanded], training=training)

        # Score each aligned region's features: one logit per region.
        region_scores = self.confidence_scorer(fused_per_region)
        # Drop the trailing size-1 axis, leaving [B, N_regions].
        region_scores = keras.ops.squeeze(region_scores, axis=-1)

        # Pick among the real regions only. The padded regions keep their
        # logits in 'confidence'; only the pick and what follows it ignore
        # them. Without a mask the scores are used as they are.
        pick_scores = region_scores
        if vision_mask is not None:
            pick_scores = keras.ops.where(
                keras.ops.cast(vision_mask, "bool"),
                region_scores,
                keras.ops.cast(mask_sentinel(region_scores.dtype), region_scores.dtype),
            )

        # Regress the bounding box from the top-scoring region's features.
        #
        # Gather one region per batch element with `keras.ops.take_along_axis`, not
        # with NumPy-style fancy indexing such as
        # `fused[batch_indices, top_indices]`. TF tensors reject that form
        # outright: "Only integers, slices (`:`), ellipsis, tf.newaxis and
        # scalar tf.int32/tf.int64 tensors are valid indices". It failed
        # eagerly, not only under tracing, which left this head dead on its
        # forward pass.
        top_indices = keras.ops.argmax(pick_scores, axis=1)
        gather_index = keras.ops.reshape(keras.ops.cast(top_indices, "int32"), (-1, 1, 1))
        gather_index = keras.ops.broadcast_to(
            gather_index, (keras.ops.shape(fused_per_region)[0], 1, keras.ops.shape(fused_per_region)[2])
        )
        top_features = keras.ops.squeeze(
            keras.ops.take_along_axis(fused_per_region, gather_index, axis=1), axis=1
        )
        bbox = self.bbox_regressor(top_features)

        return {
            "bbox": keras.ops.sigmoid(bbox),
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


@register_dl_technique("dl_techniques.layers.heads.vlm.factory")
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
        self.vision_pooling = _weightless_pooling("mean", f"{self.name}_vision_pooling")
        self.text_pooling = _weightless_pooling("mean", f"{self.name}_text_pooling")

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


# ---------------------------------------------------------------------
# Multi-Task VLM Head
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.heads.vlm.factory")
class MultiTaskVLMHead(keras.layers.Layer):
    """
    Several VLM heads behind one layer, fed from one shared input dict.

    This wrapper builds one head per entry in ``task_configs`` and hands every
    head the SAME ``inputs`` dict. There is no per-task projection stage, so
    the heads see identical tensors and differ only in what they compute.

    It does not inherit from :class:`BaseVLMHead`. It derives from
    ``keras.layers.Layer`` directly.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │ inputs, one shared dict                │
        │  {'vision_features': ...,              │
        │   'text_features': ...,                │
        │   'question_features': ...}            │
        └─────────────────────┬──────────────────┘
                              ▼
        ┌──────────────────────────────────────────────┐
        │ call(inputs, training=None, *, task_name)    │
        └──────────┬───────────────────────┬───────────┘
                   │ yes                   │ no
                   ▼                       ▼
        ┌─────────────────────┐ ┌─────────────────────┐
        │ task_heads[name]    │ │ every task head     │
        │ that head's output  │ │ {name: output, ...} │
        └─────────────────────┘ └─────────────────────┘

    ``task_name`` is keyword-only, so the second positional slot is
    ``training``, as on every other head. A ``task_name`` that is not in
    ``task_heads`` raises ``ValueError``, on the eager and on the symbolic
    path, and the symbolic output and ``compute_output_shape`` follow
    ``task_name`` as the eager call does.

    Keyword routing has three tiers. Wrapper-owned arguments
    (``task_configs``, ``task_specific_kwargs``) are consumed here. ``task_specific_kwargs``
    names one head, so it is checked strictly. Everything else is shared and
    routed to whichever heads accept it, because the head classes do not
    share a signature. A shared keyword no head accepts raises; partial
    application is reported through ``logger.info``.

    :class:`VQAHead` reads ``question_features`` while the other four heads
    read ``text_features``. One dict goes to all of them, so a wrapper mixing
    VQA with any other task raises ``KeyError`` unless the caller supplies
    both keys. Duplicating the same tensor under both keys is fine.

    Input shape:
        One shape dict, keyed by feature name, holding every key any of the
        configured heads reads.

    Output shape:
        With ``task_name``, that head's own output. Without it, a dict mapping
        each task name to that head's output.

    :param task_configs: Mapping of task name to its ``VLMTaskConfig``. One
        head is built per entry. At least one entry.
    :type task_configs: Dict[str, VLMTaskConfig]
    :param task_specific_kwargs: Per-task constructor overrides, keyed by the
        same task names as ``task_configs``. Merged over the shared keywords,
        so a per-task value wins.
    :type task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]]
    :param kwargs: Shared per-head constructor settings plus this layer's own
        Keras base arguments. The two are separated in ``__init__``.

    :ivar task_heads: Task name to head instance. Each head is also set as an
        attribute named ``head_<task_name>``.
    :vartype task_heads: Dict[str, keras.layers.Layer]
    :ivar shared_head_kwargs: The shared keywords, with the Keras base
        arguments removed.
    :vartype shared_head_kwargs: Dict[str, Any]
    """

    def __init__(
        self,
        task_configs: Dict[str, VLMTaskConfig],
        task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Build one head per task, from shared settings plus per-task overrides.

        :param task_configs: Mapping of task name to its ``VLMTaskConfig``.
        :type task_configs: Dict[str, VLMTaskConfig]
        :param task_specific_kwargs: Per-task constructor overrides, keyed by the
            same task names as ``task_configs`` -- e.g.
            ``{"cap": {"num_heads": 4, "num_layers": 1}}``. Merged over the
            shared kwargs, so a task-specific value wins.

            This is an EXPLICIT parameter rather than something read back out of
            ``**kwargs``. It used to be the latter, which made it unusable. The
            dict stayed in ``kwargs`` and was forwarded to ``Layer.__init__``.
            Keras rejected it with ``ValueError: Unrecognized keyword
            arguments``. So the documented feature always raised, and every head
            was forced onto the shared defaults.
        :type task_specific_kwargs: Optional[Dict[str, Dict[str, Any]]]
        :param kwargs: Shared per-head constructor settings plus this layer's own
            Keras base arguments. The two are SEPARATED below. Base arguments
            go only to ``Layer.__init__``, never into a head constructor. There
            ``name`` would collide with the head's own auto-generated name.

            A shared kwarg goes to the heads whose class ACCEPTS it. The rest
            skip it. The five head classes do not share one signature.
            ``ImageCaptioningHead`` and ``VQAHead`` are plain ``Layer``
            subclasses, so a ``BaseVLMHead`` argument such as
            ``fusion_strategy`` is a hard error on them. Requiring every head to
            accept every shared argument would make this wrapper unusable for
            any mixed set of tasks.

            Two guards keep that from being silent. A shared kwarg accepted by
            NO head raises. Partial application is reported via
            ``logger.info``. Use ``task_specific_kwargs`` to target one head
            exactly. Those ARE validated strictly, since they name a head.
        :raises ValueError: If ``task_configs`` is empty, or
            ``task_specific_kwargs`` names a task absent from ``task_configs``.
        """
        if not task_configs:
            raise ValueError(
                "MultiTaskVLMHead needs at least one task in task_configs; an "
                "empty wrapper returns {} for any input."
            )
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
                    f"MultiTaskVLMHead supplies these to every head itself, "
                    f"from task_configs."
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

            head = head_class(task_config=task_config, **head_kwargs)
            self.task_heads[task_name] = head
            setattr(self, f"head_{task_name}", head)

    def build(self, input_shape: Dict) -> None:
        """Build every task head against the shared input shapes.

        ``call`` hands the same ``inputs`` dict to every head, so each head is
        built from the same shapes. A head that needs a key the caller did not
        supply raises here rather than at first use. That is the same failure
        ``call`` would produce, only earlier, and it is not swallowed.

        :param input_shape: Shared input shapes, keyed by feature name.
        :type input_shape: Dict
        :return: None.
        :rtype: None
        :raises KeyError: If a head needs an input key the caller omitted.
        """
        if self.built:
            return

        for head in self.task_heads.values():
            head.build(input_shape)

        super().build(input_shape)

    def _selected_heads(self, task_name: Optional[str]) -> Dict[str, keras.layers.Layer]:
        """Return the heads a call runs: one for ``task_name``, else all of them.

        :param task_name: A configured task name, or ``None``.
        :type task_name: Optional[str]
        :return: Task name to head.
        :rtype: Dict[str, keras.layers.Layer]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        if task_name is None:
            return dict(self.task_heads)
        if task_name not in self.task_heads:
            raise ValueError(
                f"Unknown task: {task_name!r}. Configured tasks: "
                f"{sorted(self.task_heads)}."
            )
        return {task_name: self.task_heads[task_name]}

    # DECISION plan-2026-10-14T062616-81938e81/D-060
    # `task_name` is keyword-only and `training` keeps the second positional
    # slot. Do NOT move `task_name` back in front of `training`: a positional
    # flag then landed in `task_name` (`.call(inputs, True)` raised "Unknown
    # task: True", a tensor flag "Tensor is unhashable"). See decisions.md
    # D-060.
    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
        *,
        task_name: Optional[str] = None,
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Run one task head, or every one of them.

        :param inputs: One shared input dict, handed to each head unchanged.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to each head.
        :type training: Optional[bool]
        :param task_name: Keyword only. Which head to run; ``None`` runs them
            all.
        :type task_name: Optional[str]
        :return: That head's output, or a dict of task name to output.
        :rtype: Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        outputs = {
            name: head(inputs, training=training)
            for name, head in self._selected_heads(task_name).items()
        }
        return outputs[task_name] if task_name is not None else outputs

    def compute_output_shape(
        self, input_shape: Dict, *, task_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return the output shapes ``call`` returns for the same ``task_name``.

        :param input_shape: Shared input shapes (dict keyed by feature name).
        :type input_shape: Dict
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output shapes, or task name to output shapes.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        shapes = {
            name: head.compute_output_shape(input_shape)
            for name, head in self._selected_heads(task_name).items()
        }
        return shapes[task_name] if task_name is not None else shapes

    def compute_output_spec(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
        *,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the symbolic outputs ``call`` returns for the same ``task_name``.

        Keras' default routes a symbolic call through
        ``compute_output_shape`` with the input shapes only, so ``task_name``
        never reached it and a symbolic ``head(inputs, task_name='itm')``
        returned every task's outputs. Each selected head reports its own
        spec instead.

        :param inputs: Symbolic shared inputs.
        :type inputs: Dict[str, Any]
        :param training: Unused; accepted because ``call`` takes it.
        :type training: Optional[bool]
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output spec, or task name to output spec.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        specs = {
            name: head.compute_output_spec(inputs)
            for name, head in self._selected_heads(task_name).items()
        }
        return specs[task_name] if task_name is not None else specs

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this wrapper.

        The shared per-head keywords are written back out flat, alongside the
        wrapper's own arguments, which is the shape ``__init__`` expects.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_configs": {
                    name: _serialize_task_config(tc)
                    for name, tc in self.task_configs.items()
                },
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

        This used to pop ``name``, ``trainable`` and ``dtype``, so that they
        could not leak into per-task head construction. ``__init__`` already
        stops that leak: it partitions ``**kwargs`` on
        ``_KERAS_BASE_LAYER_KWARGS`` and passes the base ones to
        ``super().__init__`` only, never to a sub-head. The pop prevented
        nothing and discarded the values instead.

        # DECISION plan-2026-07-30T081929-1645aa52/D-012
        Keep and forward the Keras base kwargs. Do NOT re-add a ``config.pop``
        loop for them: it discarded the values, so a head saved with
        ``name='mt'``/``trainable=False`` reloaded as
        ``name='multi_task_vlm_head'``/``trainable=True``. See decisions.md D-012.

        :param config: Serialized layer configuration.
        :type config: Dict[str, Any]
        :return: A reconstructed multi-task head.
        :rtype: MultiTaskVLMHead
        """
        config = dict(config)
        config["task_configs"] = {
            name: _deserialize_task_config(tc)
            for name, tc in config["task_configs"].items()
        }
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------


# DECISION plan-2026-10-14T062616-81938e81/D-059
# One table, read by get_head_class and therefore by create_vlm_head and
# MultiTaskVLMHead. Do NOT add a row that maps a task type to BaseVLMHead as a
# placeholder (DENSE_CAPTIONING and VISUAL_DIALOGUE were, only to be rejected
# a line later), and do NOT restore a default: a task type with no head
# raises. See decisions.md D-059 (and D-052, the vision table).
VLM_HEAD_DISPATCH: Dict[VLMTaskType, type] = {
    VLMTaskType.IMAGE_CAPTIONING: ImageCaptioningHead,
    VLMTaskType.VISUAL_QUESTION_ANSWERING: VQAHead,
    VLMTaskType.VISUAL_GROUNDING: VisualGroundingHead,
    VLMTaskType.IMAGE_TEXT_MATCHING: ImageTextMatchingHead,
}


def get_head_class(task_type: Union[VLMTaskType, str]) -> type:
    """
    Return the head class for a VLM task type, or raise.

    The class comes from :data:`VLM_HEAD_DISPATCH`. Four task types have a
    head: image captioning, visual question answering, visual grounding, and
    image-text matching. Every other member of ``VLMTaskType`` raises. There
    is no silent fallback, matching the sibling ``heads/vision/factory.py``.

    To add support for a task type, implement its head and add it to the
    table. Do not restore a default.

    This used to be ``head_mapping.get(task_type, BaseVLMHead)``, so 41 of the
    47 members quietly returned ``BaseVLMHead``, which has no ``call()``: the
    object constructed fine and died with ``NotImplementedError``, naming the
    base class rather than the task, the moment it was used.

    :param task_type: The VLM task type, as a member or its string value.
    :type task_type: Union[VLMTaskType, str]
    :return: The head class for that task type.
    :rtype: type
    :raises ValueError: If no head is implemented for ``task_type``, or the
        string names no task type.
    """
    if isinstance(task_type, str):
        task_type = VLMTaskType.from_string(task_type)
    head_class = VLM_HEAD_DISPATCH.get(task_type)
    if head_class is None:
        implemented = sorted(t.name for t in VLM_HEAD_DISPATCH)
        raise ValueError(
            f"No VLM head is implemented for task type '{task_type.name}'. "
            f"Implemented task types: {implemented}."
        )
    return head_class


def create_vlm_head(
    task_config: Union[VLMTaskConfig, Dict[str, Any]], **kwargs: Any
) -> keras.layers.Layer:
    """
    Build one VLM head from a task configuration.

    The task type in ``task_config`` picks the class. An unsupported task type
    raises rather than returning a default head.

    :param task_config: VLMTaskConfig object, or a dict of its fields. A dict
        may give ``task_type`` as its string value, as ``VLMTaskConfig``
        itself accepts.
    :type task_config: Union[VLMTaskConfig, Dict[str, Any]]
    :param kwargs: Additional configuration for the head, such as
        ``fusion_strategy`` or ``num_layers``. Which keywords are accepted
        depends on the head class the task type selects; any other raises
        ``ValueError``.
    :return: A configured VLM head for the given task.
    :rtype: keras.layers.Layer
    :raises ValueError: If no head is implemented for the task type.
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
    Build a :class:`MultiTaskVLMHead` from task configurations.

    A list is keyed by each configuration's ``name``, and two configurations
    with one name are refused (keying would keep only the last). A dict is
    used as it is. An empty list or dict is refused.

    :param task_configs: List or dict of VLMTaskConfig objects.
    :type task_configs: Union[List[VLMTaskConfig], Dict[str, VLMTaskConfig]]
    :param kwargs: Shared per-head configuration, plus this wrapper's own Keras
        base kwargs. No head takes a declared ``vision_dim`` / ``text_dim``,
        so passing either raises (no head accepts it).

        Kwarg routing is three-tiered:

        1. **Wrapper-owned** (``task_config``) -- supplied to every head from
           ``task_configs``; passing it raises.
        2. **``task_specific_kwargs``** -- a dict keyed by task name, applied to
           that head only. STRICT: an unknown task name raises.
        3. **Everything else** -- shared, and routed BEST-EFFORT to each head
           whose constructor accepts it. A key no head accepts raises, so a
           misspelling raises too. But a key that is real for SOME head, and
           merely meant for a DIFFERENT one, is applied wherever it fits. Only
           a ``logger.info`` records where it landed. Use
           ``task_specific_kwargs`` when you mean exactly one head.

        The heads also disagree on their input key. ``VQAHead`` reads
        ``question_features`` while the other four read ``text_features``, and
        this wrapper fans one input dict to all of them. A wrapper mixing VQA
        with any other task raises ``KeyError`` unless the caller supplies both
        keys. Duplicating the same tensor under both is fine.
    :return: Configured multi-task VLM head instance.
    :rtype: MultiTaskVLMHead
    :raises ValueError: If ``task_configs`` is empty or a list repeats a
        name, or a keyword is wrapper-owned, is accepted by no head, or names
        a task absent from ``task_configs``.
    """
    if isinstance(task_configs, list):
        names = [config.name for config in task_configs]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"Duplicate task name(s) {', '.join(repr(n) for n in repeated)} "
                f"in task_configs: a list is keyed by each config's name, so "
                f"all but the last would be dropped. Give each task its own "
                f"name."
            )
        task_configs = {config.name: config for config in task_configs}

    return MultiTaskVLMHead(task_configs=task_configs, **kwargs)