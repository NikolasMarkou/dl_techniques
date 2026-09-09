"""H-Net's ratio (load-balancing) loss over the routing records.

Dynamic chunking has no supervision signal of its own: nothing in the language-modelling
objective asks the router to emit *any* particular number of boundaries, and the
degenerate solutions at both ends (every position a boundary, or only position 0) are both
reachable. The ratio loss is the term that pins the realised compression to a target
downsampling factor ``N``. It is minimised when the fraction of positions selected as
boundaries AND the mean boundary probability both equal ``1 / N``.

The formula is transcribed from the reference's ``load_balancing_loss``
(``hnet/utils/train.py:13-40``) -- the shipped code is ground truth here and the paper is
secondary confirmation::

    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]                   # train.py:30-31
    boundary_mask = router_output.boundary_mask               # train.py:32

    true_ratio   = boundary_mask.float().mean()               # train.py:34
    average_prob = tokenized_prob.float().mean()              # train.py:35

    return (
        (1 - true_ratio) * (1 - average_prob) +
        (true_ratio) * (average_prob) * (N - 1)
    ) * N / (N - 1)                                           # train.py:37-40

Three properties of that transcription are load-bearing and each is a way a re-derivation
from the paper would silently differ from the code:

* Both means are over the WHOLE tensor -- batch and sequence axes together, ONE scalar per
  call. ``train.py:20-21`` says so explicitly: the loss is computed per minibatch and then
  averaged over minibatches, never per example.
* ``.float()`` is not decorative. Under ``mixed_float16`` a mean over a long sequence of
  small probabilities accumulates in float16 and loses the low bits; the reference casts
  and so does this port.
* ``N > 1`` is a precondition (``train.py:25``), not an omission. ``N = 1`` divides by
  zero.

Two deliberate divergences from the reference, both recorded in
``plans/plan-2026-09-09T042752-6d66ac56/decisions.md``:

1. **An optional padding mask.** Upstream never needs one: it runs the loss on the packed
   (``cu_seqlens``) layout where padding does not exist. This port is padded-only (D-007),
   so counting padded positions would make the measured compression ratio a function of
   how full the batch happened to be. ``mask=None`` reproduces the reference exactly.
2. **``target_ratio <= 1`` raises** instead of returning a non-finite number.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
    - Reference implementation: ``hnet/utils/train.py:13-40``.
"""

from typing import Any, Dict, Optional, Sequence

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

# (none -- this module reads only tensors and floats.)

__all__ = [
    "DEFAULT_TARGET_RATIO",
    "ratio_loss",
    "total_ratio_loss",
]

#: Target downsampling factor used when a caller names none. The reference *code* carries
#: no value for ``N`` at all -- it lives in the training script, which is not part of the
#: released repository -- so this is a documented default of this port rather than a
#: transcribed constant, and every training entry point is expected to set it per
#: chunking level. ``6.0`` is the compression the paper reports targeting for its
#: single-stage byte models.
DEFAULT_TARGET_RATIO: float = 6.0


def _compute_dtype(tensor: Any) -> str:
    """Pick the accumulation dtype for a mean over a whole ``(B, L)`` tensor.

    Mirrors the reference's ``.float()``: anything narrower than float32 accumulates in
    float32, while a float64 input is honoured so a float64 parity arm can measure this
    function against a float64 oracle without the cast dominating the comparison.

    :param tensor: The tensor whose dtype is being widened.
    :type tensor: Any
    :returns: ``"float64"`` or ``"float32"``.
    :rtype: str
    """
    # `keras.backend.standardize_dtype` is a banned Keras-2 backend call
    # (`tests/test_the_keras2_backend_calls_are_gone.py`); the documented
    # replacement for the single-dtype form is the `.name`-or-`str` reduction
    # below, which normalises a `tf.DType`, a `np.dtype` and a plain string alike.
    dtype_name = getattr(tensor.dtype, "name", None) or str(tensor.dtype)
    return "float64" if dtype_name == "float64" else "float32"


def ratio_loss(
        boundary_prob: Any,
        boundary_mask: Any,
        mask: Optional[Any] = None,
        target_ratio: float = DEFAULT_TARGET_RATIO,
) -> Any:
    """The load-balancing loss for ONE chunking level.

    :param boundary_prob: The routing distribution, either ``(B, L, 2)`` as
        :class:`~dl_techniques.layers.dynamic_chunking.routing_module.RoutingModule`
        emits it (``[..., -1]`` is ``p``) or ``(B, L)`` holding ``p`` directly.
    :type boundary_prob: Any
    :param boundary_mask: ``(B, L)`` boolean; ``True`` where the hard threshold selected
        a boundary.
    :type boundary_mask: Any
    :param mask: Optional ``(B, L)`` validity mask, truthy at a real token. When given,
        both means are taken over the valid positions only. When ``None`` both means are
        over the whole tensor, which is the reference's behaviour exactly.
    :type mask: Optional[Any]
    :param target_ratio: ``N``, the target downsampling factor. Must be ``> 1``.
    :type target_ratio: float
    :returns: A scalar tensor.
    :rtype: Any
    :raises ValueError: if ``target_ratio <= 1``. Upstream instead divides by zero
        (``train.py:39-40``); this port refuses the input rather than propagating a
        non-finite loss into a training run.
    """
    n = float(target_ratio)
    if not n > 1.0:
        raise ValueError(
            f"target_ratio must be > 1 (it is a downsampling factor, and the loss "
            f"divides by N - 1), got {target_ratio!r}"
        )

    prob = keras.ops.convert_to_tensor(boundary_prob)
    dtype = _compute_dtype(prob)
    prob = keras.ops.cast(prob, dtype)

    # `train.py:30-31`. The 2-class form stacks `[1 - p, p]`, so the LAST entry is `p`.
    if len(prob.shape) == 3:
        prob = prob[..., -1]

    selected = keras.ops.cast(keras.ops.convert_to_tensor(boundary_mask), dtype)

    if mask is None:
        valid = keras.ops.ones_like(selected)
    else:
        valid = keras.ops.cast(keras.ops.convert_to_tensor(mask), dtype)

    # `train.py:34-35`, as `sum / count` rather than `mean` so that the masked and
    # unmasked paths are ONE expression: with an all-true mask `count` is the element
    # count and this IS the mean.
    denominator = keras.ops.sum(valid)
    true_ratio = keras.ops.sum(selected * valid) / denominator
    average_prob = keras.ops.sum(prob * valid) / denominator

    # `train.py:37-40`. The three constants stay PYTHON floats and are combined with the
    # float64/float32 reductions directly: `keras.ops.cast(<python float>, "float64")`
    # rounds through float32 first, which was MEASURED here to put a 4.6e-08 relative
    # error into `N / (N - 1)` for every N whose ratio is not a binary fraction (N = 4
    # and N = 6 failed the float64 parity arm; N = 2 and N = 9 passed only because 2.0
    # and 1.125 are exact in float32). A Python scalar multiplied against a tensor adopts
    # the tensor's dtype with no float32 way-station.
    return (
        (1.0 - true_ratio) * (1.0 - average_prob)
        + true_ratio * average_prob * (n - 1.0)
    ) * (n / (n - 1.0))


def total_ratio_loss(
        routing_records: Sequence[Dict[str, Any]],
        target_ratios: Sequence[float],
) -> Any:
    """Sum :func:`ratio_loss` over every chunking level of a hierarchy.

    One term per record, in the order the records arrive (outermost first, as
    :meth:`~dl_techniques.models.language.hnet.stage.HNetStage.call` emits them), so
    ``target_ratios[k]`` is the target for the ``k``-th chunking level counted from the
    outside. The sum -- not the mean -- is what a per-stage target means: adding a
    hierarchy level must add its own pressure, not dilute the levels above it.

    :param routing_records: The routing records, each carrying ``boundary_prob``,
        ``boundary_mask`` and ``padding_mask``.
    :type routing_records: Sequence[Dict[str, Any]]
    :param target_ratios: One target downsampling factor per record.
    :type target_ratios: Sequence[float]
    :returns: A scalar tensor. ``0.0`` when there are no chunking levels.
    :rtype: Any
    :raises ValueError: if the two sequences have different lengths.
    """
    if len(routing_records) != len(target_ratios):
        raise ValueError(
            f"one target ratio is needed per chunking level: got "
            f"{len(target_ratios)} target_ratios for {len(routing_records)} routing "
            f"records"
        )
    if not routing_records:
        return keras.ops.cast(0.0, "float32")

    terms = [
        ratio_loss(
            record["boundary_prob"],
            record["boundary_mask"],
            mask=record.get("padding_mask"),
            target_ratio=target,
        )
        for record, target in zip(routing_records, target_ratios)
    ]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total
