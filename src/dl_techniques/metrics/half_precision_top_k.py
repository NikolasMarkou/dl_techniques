"""Sparse top-k accuracy for logits in any float dtype.

Under a ``mixed_float16`` or ``mixed_bfloat16`` policy a classifier emits
half-precision logits, and the stock :class:`keras.metrics.SparseTopKCategoricalAccuracy`
rejects them: its ``in_top_k`` op takes ``float32`` only (measured:
``TypeError: Input 'predictions' of 'InTopKV2' Op has type float16``).

Interface contract
------------------
:class:`HalfPrecisionTopKAccuracy`
    A :class:`keras.metrics.SparseTopKCategoricalAccuracy` whose ``update_state``
    casts ``y_pred`` to ``float32`` first. Casting up changes no ranking, so the
    value equals the stock metric's on the same logits cast to ``float32``.

    :param k: number of top predictions that count (default 5).
    :param name: metric name, as the stock metric's default when omitted.
    :param dtype: state dtype, as the stock metric.
    :param from_sorted_ids: as the stock metric.

    Registered with ``@register_dl_technique``, so a ``.keras`` model compiled with
    it reloads by default once ``dl_techniques`` is importable. A float32 model
    should use the stock metric instead, which needs no registration at all.

    Failure mode: those of the stock metric (a ``y_true`` that is not a class
    index, a ``k`` above the class count).
"""

import keras
from typing import Any, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.metrics.half_precision_top_k")
class HalfPrecisionTopKAccuracy(keras.metrics.SparseTopKCategoricalAccuracy):
    """Sparse top-k accuracy of logits in any float dtype; see the module contract."""

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        # DECISION plan-2026-10-30T132432-a865fa08/D-041: cast the logits up HERE. Do NOT
        # use the stock ``SparseTopKCategoricalAccuracy`` with ``precision=mixed_float16``
        # (its in_top_k op raises on half precision) and do NOT cast the model's output
        # (model code is out of scope and a cast layer would change every saved model).
        return super().update_state(y_true, keras.ops.cast(y_pred, "float32"), sample_weight)

# ---------------------------------------------------------------------
