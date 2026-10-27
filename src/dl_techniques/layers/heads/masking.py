"""Padding-mask selection shared by the head domains.

One helper, :func:`fill_masked`: replace the masked positions of a score tensor
by the dtype's finite mask sentinel, by *selection*. The NLP attention pooler,
the NLP question-answering head and the VLM grounding head each need exactly
this, and each wrote it out.

It lives here, not in ``dl_techniques.utils.dtype_policy``, because that module
is pure numeric policy by design: it owns the number (``mask_sentinel``) and
defines no tensor operation. The softmax contract (``keep`` predicate,
``mask_dtype`` floor, fully-masked-row rescue) is
``layers.attention.common.apply_attention_mask``'s and is not repeated.
"""

from typing import Any, Optional

import keras

from dl_techniques.utils.dtype_policy import mask_sentinel


def fill_masked(scores: Any, keep: Any, sentinel_dtype: Optional[str] = None) -> Any:
    """
    Return ``scores`` with every position where ``keep`` is false set to the sentinel.

    Interface contract, shared by ``BaseNLPHead._pool_sequence``'s attention
    pooling, ``QuestionAnsweringHead.call`` and ``VisualGroundingHead.call``:

    * ``keep`` is a keep predicate of ``scores``' shape (1 or ``True`` for a
      real position); any numeric or boolean dtype is accepted.
    * The sentinel is ``mask_sentinel(sentinel_dtype)`` cast to ``scores``'
      dtype. ``sentinel_dtype`` defaults to ``scores``' own dtype; a layer that
      derives its sentinel from its compute dtype passes ``self.compute_dtype``.
    * The result is a ``keras.ops.where`` selection, never
      ``scores + (1 - keep) * sentinel``: under ``float16`` that product is
      ``0 * -inf = NaN`` at every KEPT position once the sentinel overflows.

    :param scores: Scores to mask.
    :type scores: Any
    :param keep: Keep predicate, same shape as ``scores``.
    :type keep: Any
    :param sentinel_dtype: Dtype the sentinel is derived for, or ``None`` for
        ``scores``' dtype.
    :type sentinel_dtype: Optional[str]
    :return: Scores with masked positions replaced.
    :rtype: Any
    """
    sentinel = keras.ops.cast(
        mask_sentinel(scores.dtype if sentinel_dtype is None else sentinel_dtype),
        scores.dtype,
    )
    return keras.ops.where(keras.ops.cast(keep, dtype="bool"), scores, sentinel)
