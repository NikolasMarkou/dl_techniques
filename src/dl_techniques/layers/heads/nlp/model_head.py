"""Glue between a language backbone and one in-model NLP task head.

A language model that owns its task head (``include_top=True``) needs the
same few things whatever its architecture: its head knobs checked for
consistency, a head built from a task config that may arrive as a dict out of
``get_config``, that config written back as plain JSON data, a padding mask
for the head to pool under, and the head's output dict reduced to something
``model.compile`` accepts. This module holds
those pieces so the models do not each carry a copy.

Functions
---------
* :func:`validate_task_head_knobs` -- refuse ``include_top=True`` without a
  task config, and a task config or head overrides without ``include_top``.
* :func:`build_model_task_head` -- build one head, named ``name``, from a task
  config plus optional head keyword overrides.
* :func:`task_config_to_dict` -- turn a task config into a JSON-serializable
  dict that :func:`build_model_task_head` accepts back.
* :func:`finalize_head_outputs` -- drop the outputs derived from ``logits`` and
  unwrap a single-key dict to a bare tensor.
* :func:`head_output_shape` and :func:`ids_shape_of` -- the shapes a headed
  model returns, for its ``compute_output_shape``.
* :func:`head_attention_mask` -- the mask a head pools under: the caller's
  ``attention_mask``, else ``input_ids != pad_token_id``, else all ones, in a
  given dtype.

Nothing here touches the backbone. The head is the only layer created, and all
of its weights live under its own name.

Example
-------
>>> from dl_techniques.layers.heads.nlp import (
...     NLPTaskConfig, NLPTaskType, build_model_task_head, finalize_head_outputs,
... )
>>> cfg = NLPTaskConfig(name="sst2", task_type=NLPTaskType.TEXT_CLASSIFICATION,
...                     num_classes=2)
>>> head = build_model_task_head(cfg, input_dim=64, default_pooling="last")
>>> logits = finalize_head_outputs(head({"hidden_states": h, "attention_mask": m}))
"""

import dataclasses
from typing import Any, Dict, Optional, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .factory import BaseNLPHead, create_nlp_head
from .task_types import NLPTaskConfig, NLPTaskType

# ---------------------------------------------------------------------

# Keys a head adds next to ``logits`` that are pure functions of it:
# ``probabilities`` is ``softmax(logits)`` and ``predictions`` is
# ``argmax(logits, axis=-1)``, which no loss can take as a target.
_DERIVED_FROM_LOGITS = ("probabilities", "predictions")

#: Layer name of the head a model owns. Every head weight path starts with it,
#: so a model's weights split into backbone and head by this one prefix; models
#: pass it as ``skip_prefixes`` to keep a pretrained transfer backbone-only.
TASK_HEAD_NAME = "task_head"


#: The task types whose head is a ``TextGenerationHead``: logits over a
#: vocabulary, whose size the task config carries as ``vocabulary_size``.
_GENERATION_HEAD_TASKS = frozenset({
    NLPTaskType.TEXT_GENERATION,
    NLPTaskType.MASKED_LANGUAGE_MODELING,
    NLPTaskType.TEXT_COMPLETION,
    NLPTaskType.TEXT_SUMMARIZATION,
})


def _as_task_config(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        vocab_size: Optional[int] = None,
) -> NLPTaskConfig:
    """
    Return ``task_config`` as an :class:`NLPTaskConfig`.

    A dict may carry ``task_type`` as an :class:`NLPTaskType` member or as its
    string value, which is the form :func:`task_config_to_dict` and the head's
    own ``get_config`` write. The caller's dict is not modified. A dict for a
    generation-type task that sets no ``vocabulary_size`` gets ``vocab_size``
    (the model's) when one is given; an :class:`NLPTaskConfig` instance has
    already been given the dataclass default 32000 by then, which cannot be told
    from an explicit 32000, so it is returned as is.

    :param task_config: A config instance, or a dict of its fields.
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
    :param vocab_size: The owning model's vocabulary size, or None.
    :type vocab_size: Optional[int]
    :return: The config as a dataclass instance.
    :rtype: NLPTaskConfig
    :raises TypeError: If ``task_config`` is neither a dict nor an
        :class:`NLPTaskConfig`, or the dict holds an unknown field.
    :raises ValueError: If a string ``task_type`` names no task.
    """
    if isinstance(task_config, NLPTaskConfig):
        return task_config
    if not isinstance(task_config, dict):
        raise TypeError(
            "task_config must be an NLPTaskConfig or a dict of its fields, "
            f"got {type(task_config).__name__}."
        )
    fields = dict(task_config)
    if isinstance(fields.get("task_type"), str):
        fields["task_type"] = NLPTaskType.from_string(fields["task_type"])
    if (
            vocab_size is not None
            and fields.get("task_type") in _GENERATION_HEAD_TASKS
            and fields.get("vocabulary_size") is None
    ):
        fields["vocabulary_size"] = int(vocab_size)
    return NLPTaskConfig(**fields)


def validate_task_head_knobs(
        include_top: bool,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]],
        head_config_overrides: Optional[Dict[str, Any]],
) -> None:
    """
    Refuse an inconsistent set of a model's three task-head knobs.

    Shared by every language model that takes ``include_top``,
    ``task_config`` and ``head_config_overrides``, so the rule and its
    messages exist once. Call it before building anything. Valid sets are
    ``include_top=True`` with a ``task_config`` (overrides optional), or
    neither of the other two.

    :param include_top: Whether the model is to own a task head.
    :type include_top: bool
    :param task_config: The head's task, or None.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Head constructor overrides, or None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :raises ValueError: If ``include_top`` is True without a
        ``task_config``, or if ``task_config`` or ``head_config_overrides``
        is given with ``include_top`` False (a dead knob).
    """
    if include_top and task_config is None:
        raise ValueError(
            "include_top=True needs a task_config (an NLPTaskConfig or a "
            "dict of its fields) to know which head to build."
        )
    if not include_top and (
        task_config is not None or head_config_overrides is not None
    ):
        raise ValueError(
            "task_config and head_config_overrides only take effect with "
            "include_top=True; with include_top=False no head is built, so "
            "they would be silently ignored."
        )


def task_config_to_dict(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        vocab_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Serialize a task config to a plain dict for a model's ``get_config``.

    Every dataclass field is kept, so the dict rebuilds an equal
    :class:`NLPTaskConfig`. ``task_type`` is written as its string value, which
    makes the dict JSON-serializable. A dict input is normalized first, so the
    function is safe to call on whatever the model was constructed with.

    :param task_config: A config instance, or a dict of its fields.
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
    :param vocab_size: The owning model's vocabulary size: fills an unset
        ``vocabulary_size`` of a generation-type dict (see
        :func:`build_model_task_head`).
    :type vocab_size: Optional[int]
    :return: A new dict of all :class:`NLPTaskConfig` fields.
    :rtype: Dict[str, Any]
    :raises TypeError: As :func:`_as_task_config`.
    :raises ValueError: As :func:`_as_task_config`.
    """
    fields = dataclasses.asdict(_as_task_config(task_config, vocab_size))
    fields["task_type"] = fields["task_type"].value
    return fields


def build_model_task_head(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        input_dim: int,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        default_pooling: Optional[str] = None,
        name: str = TASK_HEAD_NAME,
        vocab_size: Optional[int] = None,
) -> BaseNLPHead:
    """
    Build the task head a language model owns when ``include_top=True``.

    The head is created by :func:`create_nlp_head`, so task-type dispatch and
    its refusal of unimplemented task types are unchanged. This function adds
    three rules on top.

    **Architecture Overview:**

    .. code-block:: text

        task_config (NLPTaskConfig or dict)
                │
                ▼
        ┌────────────────────────────────┐
        │ dict -> NLPTaskConfig          │
        └───────────────┬────────────────┘
                        ▼
        ┌────────────────────────────────┐        ┌────────────┐
        │ MULTIPLE_CHOICE?               │─ yes ─►│ ValueError │
        └───────────────┬────────────────┘        └────────────┘
                        │ no
                        ▼
        ┌────────────────────────────────┐
        │ overrides lack pooling_type?   │─ yes ─► pooling_type =
        └───────────────┬────────────────┘         default_pooling
                        ▼                          (if given)
        ┌────────────────────────────────┐
        │ create_nlp_head(task_config,   │
        │   input_dim, name=name, ...)   │
        └────────────────────────────────┘

    ``MULTIPLE_CHOICE`` is refused because a model hands its head the token
    sequence ``(batch, seq_len, hidden)``, while ``MultipleChoiceHead`` scores
    a ``(batch, num_choices, hidden)`` tensor. Fed the sequence, it returns one
    score per token and ignores ``num_classes``.

    ``vocab_size`` is the model's own vocabulary. A generation-type head
    (``TEXT_GENERATION``, ``MASKED_LANGUAGE_MODELING``, ``TEXT_COMPLETION``,
    ``TEXT_SUMMARIZATION``) emits logits over ``task_config.vocabulary_size``,
    which defaulted to 32000 whatever the model: BERT with vocabulary 64 and a
    summarization head returned ``(batch, seq_len, 32000)``. With ``vocab_size``
    given, a dict config that sets no ``vocabulary_size`` takes it from the
    model, and any different value raises ``ValueError``. The dataclass default
    32000 cannot be told from an explicit 32000, so an
    :class:`NLPTaskConfig` that leaves it unset on a model of another
    vocabulary raises too and must state ``vocabulary_size``. Other task types
    ignore ``vocab_size``. With ``vocab_size=None`` nothing is checked.

    ``default_pooling`` lets a backbone state the pooling its states need, for
    example ``'last'`` for a causal decoder, where position 0 has seen only
    itself. An explicit ``pooling_type`` in ``head_config_overrides`` always
    wins. With neither, the head's own default applies.

    :param task_config: The task, as an :class:`NLPTaskConfig` or a dict of
        its fields (``task_type`` as a member or its string value).
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
    :param input_dim: Width of the hidden states the backbone emits.
    :type input_dim: int
    :param head_config_overrides: Extra keyword arguments for the head
        constructor. Not modified. Must not contain ``name``.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param default_pooling: Pooling type used when the overrides set none.
    :type default_pooling: Optional[str]
    :param name: Layer name of the head. Every head weight path starts with
        it. Defaults to ``"task_head"``.
    :type name: str
    :param vocab_size: The model's vocabulary size, checked against a
        generation-type head's ``vocabulary_size`` (see above). Defaults to
        None: no check.
    :type vocab_size: Optional[int]
    :return: The head layer, not yet built.
    :rtype: BaseNLPHead
    :raises ValueError: If the task type is ``MULTIPLE_CHOICE``, if the
        overrides contain ``name``, if the task type has no implemented
        head, or if a generation-type head's ``vocabulary_size`` differs from
        ``vocab_size``.
    :raises TypeError: If ``task_config`` is neither a dict nor an
        :class:`NLPTaskConfig`.
    """
    task_config = _as_task_config(task_config, vocab_size)

    # DECISION plan-2026-10-11T170958-dc4d2202/D-009: the head's vocabulary is the model's,
    # checked here, not in the head classes (they stay untouched); do NOT drop the check for an
    # NLPTaskConfig that kept the 32000 default -- it is indistinguishable from an explicit 32000.
    if (
            vocab_size is not None
            and task_config.task_type in _GENERATION_HEAD_TASKS
            and task_config.vocabulary_size != vocab_size
    ):
        raise ValueError(
            f"The {task_config.task_type.value!r} head would emit logits over "
            f"vocabulary_size={task_config.vocabulary_size}, but the model's "
            f"vocabulary is {vocab_size}. Set vocabulary_size={vocab_size} in "
            "the task config (an NLPTaskConfig that leaves it unset carries the "
            "default 32000; a dict that leaves it out gets the model's)."
        )

    # DECISION plan-2026-08-23T203721-009b7ccf/D-024: raise on MULTIPLE_CHOICE rather than
    # build a wrong graph; this factory can only feed the head a rank-3 token sequence, not choices. See decisions.md.
    if task_config.task_type == NLPTaskType.MULTIPLE_CHOICE:
        raise ValueError(
            "A model cannot own a MULTIPLE_CHOICE head. MultipleChoiceHead "
            "scores a (batch, num_choices, hidden) tensor, but the model feeds "
            "its head the token sequence (batch, seq_len, hidden), which "
            "yields one score per TOKEN and ignores num_classes entirely. "
            "Build the encoder and the head separately: encode each choice, "
            "stack the pooled representations into "
            "(batch, num_choices, hidden), and call "
            "create_nlp_head(task_config=...) on that."
        )

    head_kwargs = dict(head_config_overrides or {})
    if "name" in head_kwargs:
        raise ValueError(
            "head_config_overrides must not set 'name'; the head is named "
            f"{name!r} so its weights stay under one known prefix."
        )
    if default_pooling is not None and "pooling_type" not in head_kwargs:
        head_kwargs["pooling_type"] = default_pooling

    return create_nlp_head(
        task_config=task_config,
        input_dim=input_dim,
        name=name,
        **head_kwargs,
    )


def finalize_head_outputs(
        outputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
    """
    Reduce a head's output dict to what ``model.compile`` can take.

    Two rules, in order. First, if the dict holds ``logits``, the derived
    ``probabilities`` and ``predictions`` entries are dropped. Second, a dict
    left with one key is unwrapped to that bare tensor, whatever the key.

    So classification, token classification and generation heads give a bare
    logits tensor, a single-sequence similarity head gives a bare embeddings
    tensor, and question answering keeps
    ``{"start_logits": ..., "end_logits": ...}``, two independent tensors. A
    bare tensor is what lets ``loss="sparse_categorical_crossentropy"`` and
    ``metrics=["accuracy"]`` compile. A non-dict input is returned unchanged,
    and the input dict is never modified.

    :param outputs: The head output, normally a dict of tensors.
    :type outputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
    :return: A bare tensor, or a new dict of two or more tensors.
    :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
    """
    if not isinstance(outputs, dict):
        return outputs

    # DECISION plan-2026-08-23T203721-009b7ccf/D-018: drop probabilities/predictions when
    # logits is present, then unwrap any remaining single-key dict; a dict output made this factory uncompilable. See decisions.md.
    if "logits" in outputs:
        outputs = {
            key: value
            for key, value in outputs.items()
            if key not in _DERIVED_FROM_LOGITS
        }
    else:
        outputs = dict(outputs)

    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return outputs


def head_output_shape(
        task_head: keras.layers.Layer,
        input_shape: Any,
        hidden_size: int,
) -> Union[tuple, Dict[str, tuple]]:
    """
    Return the shape(s) a model with a task head returns, from its input shape.

    Interface contract (shared by the ``compute_output_shape`` of the language
    models that own a head): the headed counterpart of what ``call`` does with
    :func:`finalize_head_outputs`, applied to shapes. Keras builds a symbolic
    call of a model from ``compute_output_shape``, never from ``call``, so the
    two must describe the same structure.

    :param task_head: The model's head (``TASK_HEAD_NAME``).
    :type task_head: keras.layers.Layer
    :param input_shape: Shape of ``input_ids`` ``(batch, seq_len)``, or a dict
        holding it under ``'input_ids'``, or a list whose first entry is it.
    :type input_shape: Any
    :param hidden_size: Width of the hidden states the head pools.
    :type hidden_size: int
    :return: One shape (a tuple), or a dict of two or more shapes (question
        answering), as ``finalize_head_outputs`` would give for tensors.
    :rtype: Union[tuple, Dict[str, tuple]]
    :raises ValueError: if the head cannot report its output shape.
    """
    ids_shape = ids_shape_of(input_shape)
    return finalize_head_outputs(task_head.compute_output_shape({
        "hidden_states": (*ids_shape, hidden_size),
        "attention_mask": ids_shape,
    }))


def ids_shape_of(input_shape: Any) -> tuple:
    """
    Return the ``(batch, seq_len)`` shape out of the forms a model is given.

    :param input_shape: A shape, a dict holding it under ``'input_ids'`` (the
        other entries are ignored), or a list of shapes whose first is the ids.
    :type input_shape: Any
    :return: The ids shape as a tuple.
    :rtype: tuple
    """
    if isinstance(input_shape, dict):
        input_shape = input_shape["input_ids"]
    if (
            isinstance(input_shape, (list, tuple))
            and input_shape
            and isinstance(input_shape[0], (list, tuple))
    ):
        input_shape = input_shape[0]
    return tuple(input_shape)


def head_attention_mask(
        input_ids: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
        pad_token_id: Optional[int],
        dtype: str,
) -> keras.KerasTensor:
    """
    Return the padding mask a model's task head pools under.

    For a backbone whose own forward pass may run without a mask (a causal
    decoder needs none), the head still must not pool padding: under
    ``'last'`` pooling an unmasked head reads the final position, which is
    padding for every row shorter than the batch. The caller's mask wins;
    without one, padding is every ``pad_token_id`` position. With neither a
    mask nor a pad id nothing is padding and the mask is all ones: id 0 is a
    real token in most vocabularies, so no id is assumed to be padding. A
    padded batch must then pass ``attention_mask`` (or the model be built with
    the tokenizer's ``pad_token_id``), else the head pools the padding.

    :param input_ids: Token ids ``(batch, seq_len)``.
    :type input_ids: keras.KerasTensor
    :param attention_mask: The caller's mask ``(batch, seq_len)``, 1 = keep,
        or None.
    :type attention_mask: Optional[keras.KerasTensor]
    :param pad_token_id: Id treated as padding when no mask is given, or None
        for no padding id.
    :type pad_token_id: Optional[int]
    :param dtype: Dtype of the result, normally the hidden states' dtype, so a
        float16 hidden state never meets an int mask inside the head.
    :type dtype: str
    :return: The mask, ``(batch, seq_len)``, in ``dtype``.
    :rtype: keras.KerasTensor
    """
    if attention_mask is None:
        if pad_token_id is None:
            attention_mask = keras.ops.ones_like(input_ids)
        else:
            attention_mask = keras.ops.not_equal(input_ids, pad_token_id)
    return keras.ops.cast(attention_mask, dtype)
