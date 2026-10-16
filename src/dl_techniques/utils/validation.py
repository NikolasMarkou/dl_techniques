"""Argument checks shared across packages.

One helper today, :func:`positive_int`, the single rule for a count argument.
It replaced three private copies with three different refusal sets
(``layers/adapters/factory.py``, ``layers/moe/config.py`` and
``layers/heads/validation.py``; plan-2026-10-14T062616-81938e81 D-061, D-088).
"""

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

    Interface contract (callers: the head domains through
    ``layers/heads/validation.py``, ``layers/adapters/factory.py``,
    ``layers/moe/config.py``):

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
