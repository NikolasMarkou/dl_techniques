"""Argument and input checks shared by the head domains.

Two helpers, each with one rule that more than one head domain needs:

* :func:`positive_int` -- a count argument (``num_classes``, ``num_heads``,
  ``hidden_size``, ...) must be a positive integer. The vision heads and the
  VLM heads and their task config call it.
* :func:`static_rank` -- a head that branches on an input's rank reads that
  rank once, statically, and refuses an input whose rank is unknown. The VLM
  and NLP heads call it.

Neither builds a layer. Both raise ``ValueError`` naming the head (``owner``)
and the argument or input (``name``), so the refusal points at the caller's
mistake rather than at whatever layer would have failed later.
"""

import numbers
from typing import Any

import keras


def positive_int(value: Any, name: str, owner: str) -> int:
    """
    Check that a count argument is a positive integer, and return it as one.

    Interface contract, shared by the vision heads' count arguments
    (``hidden_dim``, ``num_classes``, ``num_anchors``, ``bbox_dims``,
    ``num_instances``, ``output_channels``, ``scale_factor``), ``VQAHead``'s
    ``num_heads`` and ``VLMTaskConfig``'s ``hidden_size``, ``vocab_size`` and
    ``num_classes``:

    * An ``int`` above 0, or a numpy integer above 0 (a count read from an
      array or a ``.npy`` config), is returned as a plain ``int``.
    * Anything else raises ``ValueError`` naming ``owner`` and ``name``:
      ``bool`` (an ``int`` subclass, so ``True`` would build a one-class
      layer), a float such as ``2.5`` (``16.0`` included), a string, ``None``,
      ``0`` and negatives.

    Call it in ``__init__`` before any layer is created, so the refusal names
    the argument instead of surfacing later as a ``Conv2D`` ``filters=0``.

    :param value: The caller's value.
    :type value: Any
    :param name: Argument name, for the error message.
    :type name: str
    :param owner: Head or config class name, for the error message.
    :type owner: str
    :return: ``value`` as an ``int``.
    :rtype: int
    :raises ValueError: If ``value`` is not a positive integer.
    """
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value <= 0):
        raise ValueError(
            f"{owner}: {name} must be a positive integer, got {value!r}"
        )
    return int(value)


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
