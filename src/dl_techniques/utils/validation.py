"""Argument checks shared across packages.

Three helpers: :func:`positive_int`, the single rule for a count argument,
:func:`positive_float`, its sibling for a scale argument such as a softmax
temperature, and :func:`non_negative_float`, the same rule for a weight that may
be exactly 0. ``positive_int`` replaced three private copies with three
different refusal sets (``layers/adapters/factory.py``, ``layers/moe/config.py``
and ``layers/heads/validation.py``; plan-2026-10-14T062616-81938e81 D-061,
D-088). ``positive_float`` replaced three missing checks in
``layers/sequence_pooling`` (D-203). ``non_negative_float`` replaced the private
``_check_real`` of ``losses/confidence_penalty_loss.py``.
"""

import math
import numbers
from typing import Any, Optional

import numpy as np

#: The largest value an int32 tensor dimension can hold. A count becomes a
#: tensor dimension, a one-hot depth or a layer width, all of which
#: TensorFlow represents in int32.
MAX_TENSOR_DIM = 2 ** 31 - 1


def positive_int(value: Any, name: str, owner: Optional[str] = None) -> int:
    """
    Check that a count argument is a positive integer, and return it as one.

    Interface contract (callers, from ``grep -rn "positive_int(" src``: every head
    domain of ``layers/heads`` (``nlp``, ``vision``, ``vlm``) through
    ``layers/heads/validation.py``, ``layers/adapters/factory.py``,
    ``layers/moe/config.py``, ``AttentionPooling``, ``WeightedPooling``,
    ``SequencePooling`` and the sequence-pooling factory in
    ``layers/sequence_pooling``, and the constructors of ``DaSiamRPN``,
    ``DeepSortAppearanceNet``, ``DeepSortResidualBlock``, ``CosineClassifier``,
    ``DepthAnything``, ``RBFProtoNet``, ``CliffordRBFProtoNet``, ``SiamFC``,
    ``SqueezeNetV1``, ``SqueezeNoduleNetV2``, ``VQVAERotationTrick`` and SAM1's
    ``PromptEncoder``):

    * An ``int`` or a numpy integer (a count read from an array shape, a
      ``.npy`` or ``np.argmax``) in ``[1, MAX_TENSOR_DIM]`` is returned as a
      plain Python ``int``. Assign the return value back: a stored
      ``np.int64`` reaches ``get_config`` and ``json.dumps`` raises at save
      time, after training.
    * Anything else raises ``ValueError`` naming ``owner`` (when given) and
      ``name``: ``bool`` and ``np.bool_`` (``True`` would build a size-1
      layer; YAML's unquoted ``true`` loads as one), a float (``16.0``
      included), a string, ``None``, ``0``, negatives, and values above the
      int32 ceiling.

    # DECISION plan-2026-08-26T155709-fb07cf4e/D-008
    Three properties are load-bearing and each looks removable. (1) The bool
    branch names ``np.bool_`` explicitly; do not delete it because the
    integral test below happens to reject it today (measured on numpy 2.0.2:
    ``np.bool_`` is not an ``np.integer``), since that is numpy's hierarchy,
    not this module's. (2) The integral test admits numpy integers
    (``np.int64(4)`` is not an ``int``); a narrow ``int`` check rejected every
    count read from an array. (3) The value is returned coerced with
    ``int(...)``: accepting a numpy integer and storing it verbatim only moves
    the failure to ``model.save()``. Moved here verbatim in substance from
    ``layers/moe/config.py`` by D-088.

    :param value: The caller's value.
    :type value: Any
    :param name: Argument name, for the error message.
    :type name: str
    :param owner: Class or function name, for the error message.
    :type owner: Optional[str]
    :return: ``value`` as a Python ``int``.
    :rtype: int
    :raises ValueError: If ``value`` is not a positive integer within the
        int32 tensor-dimension ceiling.
    """
    who = f"{owner}: " if owner else ""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{who}{name} must be a positive integer, got bool ({value!r}). "
            f"YAML's unquoted `true`/`false` load as bools; quote the value "
            f"or write a number."
        )
    if not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{who}{name} must be a positive integer, got "
            f"{type(value).__name__} ({value!r})"
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"{who}{name} must be a positive integer, got {value}")
    if value > MAX_TENSOR_DIM:
        raise ValueError(
            f"{who}{name} must be <= {MAX_TENSOR_DIM} (the int32 "
            f"tensor-dimension ceiling), got {value}"
        )
    return value


def positive_float(value: Any, name: str, owner: Optional[str] = None) -> float:
    """
    Check that a scale argument is a positive finite number, and return it as one.

    Interface contract (callers: ``AttentionPooling``, ``WeightedPooling`` and
    ``SequencePooling`` in ``layers/sequence_pooling``, for ``temperature``):

    * An ``int``, ``float`` or numpy real number that is finite and ``> 0`` is
      returned as a plain Python ``float`` (so ``get_config`` stays JSON-safe).
    * Anything else raises ``ValueError`` naming ``owner`` (when given) and
      ``name``, in the message family of :func:`positive_int`: ``bool`` and
      ``np.bool_``, a string, ``None``, NaN, an infinity, ``0`` and negatives.
      A temperature of 0 divides the logits by zero (all NaN), and a negative
      one silently inverts the weighting.

    :param value: The caller's value.
    :type value: Any
    :param name: Argument name, for the error message.
    :type name: str
    :param owner: Class or function name, for the error message.
    :type owner: Optional[str]
    :return: ``value`` as a Python ``float``.
    :rtype: float
    :raises ValueError: If ``value`` is not a positive finite number.
    """
    return non_negative_float(value, name, owner, strict=True)


def non_negative_float(
        value: Any, name: str, owner: Optional[str] = None, *, strict: bool = False
) -> float:
    """
    Check that a weight argument is a non-negative finite number, and return it as one.

    The sibling of :func:`positive_float` for an argument where ``0`` is a
    meaningful setting (``entropy_weight=0`` switches a term off,
    ``label_smoothing=0`` is no smoothing).

    Interface contract (callers: ``ConfidencePenaltyLoss`` in
    ``losses/confidence_penalty_loss`` for ``label_smoothing``, ``entropy_weight``
    and ``epsilon``; its range checks stay at the caller):

    * An ``int``, ``float`` or numpy real number that is finite and ``>= 0`` is
      returned as a plain Python ``float``, so ``get_config`` stays JSON-safe (a
      stored ``np.float32`` makes ``json.dumps`` raise at save time).
    * Anything else raises ``ValueError`` naming ``owner`` (when given) and
      ``name``, in the message family of :func:`positive_float`: ``bool`` and
      ``np.bool_`` (``True`` would pass as ``1.0``), a string, ``None``, a list,
      NaN, an infinity and negatives.
    * ``strict=True`` refuses ``0`` as well and says "positive" instead of
      "non-negative" in every message. It is the whole of :func:`positive_float`,
      which calls this function with it, so the rule lives in one body; call
      :func:`positive_float` rather than passing ``strict`` at a call site.

    :param value: The caller's value.
    :type value: Any
    :param name: Argument name, for the error message.
    :type name: str
    :param owner: Class or function name, for the error message.
    :type owner: Optional[str]
    :param strict: Refuse ``0`` too (the :func:`positive_float` rule). Defaults to
        ``False``.
    :type strict: bool
    :return: ``value`` as a Python ``float``.
    :rtype: float
    :raises ValueError: If ``value`` is not a non-negative (``strict``: positive)
        finite number.
    """
    who = f"{owner}: " if owner else ""
    word = "positive" if strict else "non-negative"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{who}{name} must be a {word} number, got bool ({value!r}).")
    if not isinstance(value, numbers.Real):
        raise ValueError(
            f"{who}{name} must be a {word} number, got "
            f"{type(value).__name__} ({value!r})"
        )
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (strict and value == 0.0):
        raise ValueError(f"{who}{name} must be a {word} finite number, got {value}")
    return value
