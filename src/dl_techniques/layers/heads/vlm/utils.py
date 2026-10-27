"""Shared helpers for the VLM heads."""

import inspect
import keras
from typing import Dict, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...ffn.factory import FFN_REGISTRY
from ..validation import pool_or_passthrough
from .task_types import VLMTaskConfig


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
    mask_key = _MASK_KEYS[features_key]
    return pool_or_passthrough(
        inputs[features_key], inputs.get(mask_key),
        lambda features, mask: pooling(features, mask=mask),
        owner=owner, features_name=features_key, mask_name=mask_key,
    )
