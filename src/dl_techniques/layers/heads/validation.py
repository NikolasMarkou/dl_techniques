"""Argument and input checks shared by the head domains.

Two helpers, each with one rule that more than one head domain needs:

* :func:`positive_int` -- a count argument (``num_classes``, ``num_heads``,
  ``hidden_size``, ...) must be a positive integer. The vision heads and the
  VLM heads and their task config call it. It is defined once, in
  ``dl_techniques.utils.validation`` (shared with the adapters and the MoE
  config), and bound here.
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
