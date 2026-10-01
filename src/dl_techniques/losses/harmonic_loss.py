"""
Harmonic loss for language modeling.

Implements the *harmonic max* output unit and loss of Baek, Liu, Tegmark et al.,
"Harmonic Loss Trains Interpretable AI Models" (arXiv:2502.01628).

A standard classification head scores class ``i`` by the inner product
``w_i . x`` and normalizes with a softmax. The harmonic head instead scores
class ``i`` by the Euclidean *distance* between the query ``x`` and the class
weight ``w_i``:

.. code-block:: text

    d_i = || w_i - x ||_2
    p_i = d_i^(-n) / sum_j d_j^(-n)          (harmonic max)
    loss = -log p_target

``n`` is the *harmonic exponent*: it controls how heavy-tailed the distribution
is (large ``n`` approaches a hard nearest-neighbour assignment). The unit has a
finite-distance singularity (``p -> 1`` as ``x -> w_target``), is scale
invariant in ``(x, w)`` jointly, and its class weights live in the same space as
the query, which is what makes them directly interpretable.

Log-space form
--------------
``d_i^(-n) = exp(-(n/2) log d_i^2)``, so harmonic max is exactly a softmax over

.. code-block:: text

    logit_i = -(n / 2) * log(d_i^2 + eps)

Softmax is invariant to a constant shift of every logit, so the rescalings the
reference implementation applies for floating-point safety (division by the row
minimum and by the width) are unnecessary here, and an ordinary sparse
cross-entropy over these logits *is* the harmonic loss.
:class:`HarmonicCausalLMLoss` is therefore a masked cross-entropy over the output
of :func:`harmonic_logits`.

References
----------
- Baek, Liu, Tegmark et al. (2025). "Harmonic Loss Trains Interpretable AI
  Models." arXiv:2502.01628. Reference code: ``KindXiaoming/grow-crystals``.
"""

from typing import Any, Dict

import keras

from dl_techniques.losses.masked_causal_lm_loss import MaskedCausalLMLoss
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def harmonic_logits(
        hidden_states: keras.KerasTensor,
        weights: keras.KerasTensor,
        exponent: float,
        eps: float = 1e-6,
) -> keras.KerasTensor:
    """Log of the unnormalized harmonic-max probabilities.

    ``softmax(harmonic_logits(x, w, n))_i == d_i^-n / sum_j d_j^-n`` with
    ``d_i = ||w_i - x||``. The squared distance is expanded as
    ``|x|^2 + |w|^2 - 2 x.w`` so the ``(..., V, H)`` difference tensor is never
    materialized (the cost is one matmul, exactly like a linear head).

    The computation runs in at least ``float32``: the expansion cancels
    catastrophically in half precision, and the logits scale with ``exponent``
    (hundreds to thousands for GPT-2 widths). A ``float64`` input stays
    ``float64``. The expanded squared distance can come out slightly negative
    from rounding, so it is clamped at zero before ``eps`` is added; ``eps``
    also bounds the logit at ``x == w_i`` (where ``d = 0``).

    :param hidden_states: Queries ``x``, shape ``(..., hidden_size)``.
    :type hidden_states: keras.KerasTensor
    :param weights: Class weights ``w``, shape ``(num_classes, hidden_size)``
        (e.g. a token embedding table).
    :type weights: keras.KerasTensor
    :param exponent: Harmonic exponent ``n`` of ``p ~ d^-n``. Must be positive.
    :type exponent: float
    :param eps: Positive floor added to the squared distance.
    :type eps: float
    :return: Logits of shape ``(..., num_classes)``, at least ``float32``.
    :rtype: keras.KerasTensor
    :raises ValueError: If ``exponent`` or ``eps`` is not positive.
    """
    if exponent <= 0:
        raise ValueError(f"exponent must be positive, got {exponent}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    in_dtype = getattr(hidden_states.dtype, "name", None) or str(hidden_states.dtype)
    compute_dtype = "float64" if in_dtype == "float64" else "float32"
    x = keras.ops.cast(hidden_states, compute_dtype)
    w = keras.ops.cast(weights, compute_dtype)

    xx = keras.ops.sum(x * x, axis=-1, keepdims=True)
    ww = keras.ops.sum(w * w, axis=-1)
    xw = keras.ops.matmul(x, keras.ops.transpose(w))
    dist_sq = keras.ops.maximum(xx + ww - 2.0 * xw, 0.0) + eps
    return -0.5 * exponent * keras.ops.log(dist_sq)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.harmonic_loss")
class HarmonicCausalLMLoss(MaskedCausalLMLoss):
    """Masked next-token loss over harmonic logits: ``-log p_target``.

    Expects ``y_pred`` to be the output of :func:`harmonic_logits` (for example
    the ``logits`` of a ``GPT2(head_type="harmonic")``). Because harmonic max is
    a softmax over those logits, the arithmetic is the parent's masked sparse
    cross-entropy, inherited unchanged together with its per-sample ``(batch,)``
    return shape and ``sample_weight`` behaviour. The subclass exists so the
    objective is named in logs and checkpoints, and so ``from_logits`` cannot be
    switched off: harmonic logits are not probabilities.

    :param ignore_index: Label value excluded from the loss. Default ``-1``.
    :param label_smoothing: Smoothing factor in ``[0, 1)``. Default ``0.0``.
    :param name: Loss name.
    """

    def __init__(
            self,
            ignore_index: int = -1,
            label_smoothing: float = 0.0,
            name: str = "harmonic_causal_lm_loss",
            **kwargs: Any,
    ) -> None:
        kwargs.pop("from_logits", None)
        super().__init__(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            from_logits=True,
            name=name,
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.pop("from_logits", None)
        return config
