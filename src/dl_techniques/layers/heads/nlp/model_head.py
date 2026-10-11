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
* :func:`head_attention_mask` -- the mask a head pools under: the caller's
  ``attention_mask``, else ``input_ids != pad_token_id``, in a given dtype.

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


def _as_task_config(
        task_config: Union[NLPTaskConfig, Dict[str, Any]]
) -> NLPTaskConfig:
    """
    Return ``task_config`` as an :class:`NLPTaskConfig`.

    A dict may carry ``task_type`` as an :class:`NLPTaskType` member or as its
    string value, which is the form :func:`task_config_to_dict` and the head's
    own ``get_config`` write. The caller's dict is not modified.

    :param task_config: A config instance, or a dict of its fields.
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
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
        task_config: Union[NLPTaskConfig, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Serialize a task config to a plain dict for a model's ``get_config``.

    Every dataclass field is kept, so the dict rebuilds an equal
    :class:`NLPTaskConfig`. ``task_type`` is written as its string value, which
    makes the dict JSON-serializable. A dict input is normalized first, so the
    function is safe to call on whatever the model was constructed with.

    :param task_config: A config instance, or a dict of its fields.
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
    :return: A new dict of all :class:`NLPTaskConfig` fields.
    :rtype: Dict[str, Any]
    :raises TypeError: As :func:`_as_task_config`.
    :raises ValueError: As :func:`_as_task_config`.
    """
    fields = dataclasses.asdict(_as_task_config(task_config))
    fields["task_type"] = fields["task_type"].value
    return fields


def build_model_task_head(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        input_dim: int,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        default_pooling: Optional[str] = None,
        name: str = TASK_HEAD_NAME,
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
    :return: The head layer, not yet built.
    :rtype: BaseNLPHead
    :raises ValueError: If the task type is ``MULTIPLE_CHOICE``, if the
        overrides contain ``name``, or if the task type has no implemented
        head.
    :raises TypeError: If ``task_config`` is neither a dict nor an
        :class:`NLPTaskConfig`.
    """
    task_config = _as_task_config(task_config)

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


def head_attention_mask(
        input_ids: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
        pad_token_id: int,
        dtype: str,
) -> keras.KerasTensor:
    """
    Return the padding mask a model's task head pools under.

    For a backbone whose own forward pass may run without a mask (a causal
    decoder needs none), the head still must not pool padding: under
    ``'last'`` pooling an unmasked head reads the final position, which is
    padding for every row shorter than the batch. The caller's mask wins;
    without one, padding is every ``pad_token_id`` position.

    :param input_ids: Token ids ``(batch, seq_len)``.
    :type input_ids: keras.KerasTensor
    :param attention_mask: The caller's mask ``(batch, seq_len)``, 1 = keep,
        or None.
    :type attention_mask: Optional[keras.KerasTensor]
    :param pad_token_id: Id treated as padding when no mask is given.
    :type pad_token_id: int
    :param dtype: Dtype of the result, normally the hidden states' dtype, so a
        float16 hidden state never meets an int mask inside the head.
    :type dtype: str
    :return: The mask, ``(batch, seq_len)``, in ``dtype``.
    :rtype: keras.KerasTensor
    """
    if attention_mask is None:
        attention_mask = keras.ops.not_equal(input_ids, pad_token_id)
    return keras.ops.cast(attention_mask, dtype)

# ---------------------------------------------------------------------
