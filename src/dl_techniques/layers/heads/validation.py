"""Argument and input checks shared by the head domains.

Six helpers, each with one rule that more than one head domain needs:

* :func:`positive_int` -- a count argument (``num_classes``, ``num_heads``,
  ``hidden_size``, ...) must be a positive integer. The vision heads and the
  VLM heads and their task config call it. It is defined once, in
  ``dl_techniques.utils.validation`` (shared with the adapters and the MoE
  config), and bound here.
* :func:`static_rank` -- a head that branches on an input's rank reads that
  rank once, statically, and refuses an input whose rank is unknown. The VLM
  and NLP heads call it.
* :func:`resolve_stage_knobs` -- a knob that configures an optional stage
  (attention, FFN) is refused when that stage is off, and resolves to its
  default when the stage is on and the knob was not given (D-030's rule,
  D-210). The vision and NLP heads call it, each with its own table.

* :func:`lookup_head` -- a task type (member or string) names the head a
  dispatch table serves, or the lookup raises. No fallback head exists in any
  domain. The NLP, VLM and vision dispatchers call it.
* :func:`pool_or_passthrough` -- a head that takes either a sequence or an
  already pooled vector pools the sequence under its padding mask and passes
  the vector through, refusing a mask for the vector. The NLP and VLM heads
  call it.
* :func:`key_by_name` -- a sequence of task configs is keyed by name, and a
  name given twice is refused rather than collapsed. The three multi-task
  factories call it.

None builds a layer. Each raises ``ValueError`` naming the head (``owner``)
and the argument or input, so the refusal points at the caller's mistake
rather than at whatever layer would have failed later.
"""

import numbers
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import keras

# The count rule has one home, shared with the adapters and the MoE config
# (D-088); the head domains import it from here as before.
from dl_techniques.utils.validation import positive_int  # noqa: F401


def static_rank(x: Any, name: str, owner: str) -> int:
    """
    Return the static rank of a tensor, or refuse a tensor whose rank is unknown.

    Interface contract, shared by every head that branches on an input's
    rank (``VisualGroundingHead``, ``ImageTextMatchingHead``,
    ``TextClassificationHead``, ``TextSimilarityHead``,
    ``MultipleChoiceHead``):

    * A tensor of known rank (eager, traced with a known rank, a
      ``keras.Input``, or a backend variable) returns its rank as a plain
      ``int``.
    * A tensor of unknown rank raises ``ValueError`` naming ``owner`` and
      ``name``. ``keras.ops.ndim`` reports such a tensor as ``-1`` under the
      TensorFlow backend (``None`` elsewhere), and a caller comparing that to
      3 would take its rank-2 branch without a word; ``len(ops.shape(x))``
      raised a backend error naming no head.

    The rank is read from ``x.shape`` (a tuple, or a ``TensorShape`` whose
    ``rank`` is ``None`` when unknown), and from ``keras.ops.ndim`` only for
    an input with no ``shape``. ``keras.ops.ndim`` alone raised
    ``AttributeError`` for a ``tf.Variable``: the TensorFlow backend hands a
    variable back unconverted and reads its missing ``.ndim`` (measured in
    plan step 11.3, a float32 ``keras.ops.cast`` of a variable is the
    variable itself).

    :param x: The input tensor.
    :type x: Any
    :param name: The input's key or argument name, for the error message.
    :type name: str
    :param owner: Head class name, for the error message.
    :type owner: str
    :return: The rank.
    :rtype: int
    :raises ValueError: If the rank is not statically known.
    """
    shape = getattr(x, "shape", None)
    if shape is None:
        rank = keras.ops.ndim(x)
    elif isinstance(shape, (tuple, list)):
        rank = len(shape)
    else:
        rank = getattr(shape, "rank", None)
    if (isinstance(rank, bool) or not isinstance(rank, numbers.Integral)
            or rank < 0):
        raise ValueError(
            f"{owner}: {name} needs a statically known rank, got rank "
            f"{rank!r}. Trace the head with a known-rank input signature."
        )
    return int(rank)


def resolve_stage_knobs(
        owner: str,
        table: Dict[str, Dict[str, Any]],
        flags: Dict[str, bool],
        given: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve the knobs of each optional stage against the flag that gates it.

    Interface contract, shared by ``BaseVisionHead`` and ``BaseNLPHead``
    (each passes its own ``table``):

    * ``table`` maps each stage flag's name to ``{knob: default}``;
      ``flags`` maps each of those flag names to the caller's value;
      ``given`` maps every knob to the caller's value, ``None`` when not
      given.
    * For a stage that is on, a ``None`` knob takes its default and a given
      one is kept. For a stage that is off, every knob resolves to ``None``,
      and a given value raises ``ValueError`` naming ``owner``, the knob and
      the flag (a default given explicitly counts: stored, it would still be
      dead).
    * Returns a new dict of every knob; nothing is mutated.

    :param owner: Head class name, for the message.
    :type owner: str
    :param table: Stage flag name to that stage's ``{knob: default}``.
    :type table: Dict[str, Dict[str, Any]]
    :param flags: Each stage flag's value.
    :type flags: Dict[str, bool]
    :param given: Each knob's caller value, or ``None``.
    :type given: Dict[str, Any]
    :return: Each knob's resolved value.
    :rtype: Dict[str, Any]
    :raises ValueError: If a knob is given for a stage that is off.
    """
    resolved = {}
    for flag, defaults in table.items():
        if flags[flag]:
            for knob, default in defaults.items():
                resolved[knob] = default if given[knob] is None else given[knob]
            continue
        dead = {knob: given[knob] for knob in defaults if given[knob] is not None}
        if dead:
            raise ValueError(
                f"{owner}: {dead} only configure the stage {flag}=True builds, "
                f"and {flag} is off, so no layer would read them. Pass "
                f"{flag}=True, or leave {sorted(dead)} unset."
            )
        resolved.update(dict.fromkeys(defaults))
    return resolved


def lookup_head(
        task_type: Any,
        table: Dict[Any, Any],
        enum_cls: type,
        domain: str,
        owner: str = '',
) -> Tuple[Any, Any]:
    """
    Resolve a task type to its row of a dispatch table, or refuse it.

    Interface contract, shared by ``nlp.dispatch.get_head_class``,
    ``vlm.dispatch.get_head_class`` and the vision dispatch
    (``create_vision_head``, ``HeadConfiguration.get_config``,
    ``MultiTaskHead``):

    * ``task_type`` is a member of ``enum_cls`` or its string value (parsed by
      ``enum_cls.from_string``, which raises for an unknown value).
    * A member with no row in ``table`` raises ``ValueError`` naming it and
      listing every member that has one. There is no fallback head: a task
      with no implementation fails loudly instead of returning a plausible
      wrong head (the NLP and VLM dispatchers used to end in
      ``table.get(task_type, <default>)``).
    * Anything else raises ``ValueError``.

    :param task_type: A task type member or its string value.
    :type task_type: Any
    :param table: The domain's dispatch table, keyed by task type member.
    :type table: Dict[Any, Any]
    :param enum_cls: The domain's task type enum.
    :type enum_cls: type
    :param domain: Domain label for the message (``'NLP'``, ``'VLM'``, ...).
    :type domain: str
    :param owner: Optional caller name, prefixed to the message.
    :type owner: str
    :return: The member and its table row.
    :rtype: Tuple[Any, Any]
    :raises ValueError: On a value that is no task type, or a member with no
        head.
    """
    prefix = f"{owner}: " if owner else ""
    if isinstance(task_type, str):
        task_type = enum_cls.from_string(task_type)
    elif not isinstance(task_type, enum_cls):
        raise ValueError(
            f"{prefix}task_type must be a {enum_cls.__name__} or its string "
            f"value, got {task_type!r}"
        )
    if task_type not in table:
        supported = sorted(t.name for t in table)
        raise ValueError(
            f"{prefix}no {domain} head is implemented for task type "
            f"'{task_type.name}'. The {len(table)} task types with a head "
            f"are: {supported}."
        )
    return task_type, table[task_type]


def pool_or_passthrough(
        features: Any,
        mask: Any,
        pool: Callable[[Any, Any], Any],
        *,
        owner: str,
        features_name: str,
        mask_name: str,
) -> Any:
    """
    Pool a rank-3 sequence under its mask, or pass a rank-2 vector through.

    Interface contract, shared by ``BaseNLPHead._pool_input`` and the VLM
    heads' ``_pool_sequence``. The rank is read once, statically
    (:func:`static_rank`):

    * rank 2 is already pooled and is returned unchanged; a mask for it
      raises, because there is nothing to mask;
    * rank 3 is handed to ``pool(features, mask)``;
    * any other rank raises.

    :param features: ``(batch, length, width)`` or ``(batch, width)``.
    :type features: Any
    :param mask: ``(batch, length)`` keep mask, or ``None``.
    :type mask: Any
    :param pool: Reduces a rank-3 ``features`` under ``mask`` to rank 2.
    :type pool: Callable[[Any, Any], Any]
    :param owner: Head class name, for the message.
    :type owner: str
    :param features_name: The input's key or argument name, for the message.
    :type features_name: str
    :param mask_name: The mask's key or argument name, for the message.
    :type mask_name: str
    :return: ``(batch, width)``.
    :rtype: Any
    :raises ValueError: On an unknown rank, a rank other than 2 or 3, or a mask
        for a rank-2 input.
    """
    rank = static_rank(features, features_name, owner)
    if rank == 2:
        if mask is not None:
            raise ValueError(
                f"{owner}: {mask_name} was given for a rank-2 {features_name}. "
                f"A rank-2 input is already pooled, so there is nothing to "
                f"mask; drop {mask_name} or pass the rank-3 sequence."
            )
        return features
    if rank != 3:
        raise ValueError(
            f"{owner}: {features_name} must be rank 2 (pooled) or rank 3 "
            f"(batch, length, width), got rank {rank}."
        )
    return pool(features, mask)


def key_by_name(
        items: Union[Dict[str, Any], Sequence[Any]],
        name_of: Callable[[Any], str],
        owner: str,
        argument: str = 'task_configs',
) -> Dict[str, Any]:
    """
    Key a list or tuple of task entries by name; pass a dict through.

    Interface contract, shared by ``create_multi_task_nlp_head``,
    ``create_multi_task_vlm_head`` and ``create_multi_task_head``:

    * A ``dict`` is returned unchanged (the caller chose the keys).
    * A ``list`` or a ``tuple`` (the same semantics) becomes a dict keyed by
      ``name_of(entry)`` in order. A name that appears twice raises, because
      keying would keep only the last entry.
    * Any other type raises ``ValueError`` naming the accepted forms.

    The caller coerces each entry first, so ``name_of`` reads a built object.

    :param items: A dict, or a list or tuple of entries.
    :type items: Union[Dict[str, Any], Sequence[Any]]
    :param name_of: Name of one entry.
    :type name_of: Callable[[Any], str]
    :param owner: Factory name, for the message.
    :type owner: str
    :param argument: The argument's name, for the message.
    :type argument: str
    :return: Name to entry.
    :rtype: Dict[str, Any]
    :raises ValueError: On any other container, or a repeated name.
    """
    if isinstance(items, dict):
        return items
    if not isinstance(items, (list, tuple)):
        raise ValueError(
            f"{owner}: {argument} must be one of: list, tuple, dict; got "
            f"{type(items).__name__}."
        )
    names = [name_of(item) for item in items]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(
            f"{owner}: names {repeated} appear more than once; a sequence is "
            "keyed by name, so all but the last would be dropped. Give each "
            "entry its own name, or pass a dict."
        )
    return dict(zip(names, items))
