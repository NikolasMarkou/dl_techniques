"""
Single-tower symmetric InfoNCE: two views of one batch, positives on the diagonal.

This module ships :class:`SymmetricInfoNCELoss`, the objective used when **one** encoder
produces **two** views of the same batch -- SimCSE's two dropout masks, or any augmentation
pair -- and row *i* of view A is the only positive for row *i* of view B. The similarity
matrix is ``view_a @ view_b.T / temperature``; the targets are ``arange(batch)``; the loss
is the **average of the two cross-entropy directions**, rows-as-anchors and
columns-as-anchors.

**The batch size is the negative count.** With batch ``N`` each anchor sees ``N - 1``
negatives, drawn from the same step's other examples. There is no memory bank, no negative
queue and no cross-device gather here: shrinking the batch directly weakens the objective,
which is why the degenerate ``N == 1`` case (one positive, zero negatives) is rejected
rather than trained on.

**Inputs are expected to be L2-normalized upstream.** That is what makes the entries of the
logit matrix cosine similarities in ``[-1, 1]``, which in turn is what makes a temperature
of 0.05 a sensible scale. The default ``normalize_inputs=False`` preserves the common split
where the *model* ends in a normalization layer; ``normalize_inputs=True`` is the opt-in
defensive path for callers who cannot guarantee that.

Relationship to the losses it resembles -- it is none of them exactly:

``InfoNCE`` (Oord et al. 2018)
    The family this belongs to: a categorical cross-entropy over one positive and ``N - 1``
    noise samples. CPC's formulation is **asymmetric** -- one direction only.

``CLIP`` (Radford et al. 2021, Figure 3)
    The **symmetric** form implemented here: ``(cross_entropy(logits, labels) +
    cross_entropy(logits.T, labels)) / 2``. CLIP applies it across two *different* towers
    (image, text); this class applies the identical arithmetic to two views from a
    *single* tower.

``SimCSE`` (official ``princeton-nlp/SimCSE``, ``simcse/models.py``)
    **One-directional.** The shipped reference computes ``CrossEntropyLoss`` once, with
    ``z1`` as rows and ``z2`` as columns, and never adds the transposed direction. So this
    class is *not* the official SimCSE loss despite the family resemblance; it is the
    symmetric variant, and the citation for that symmetry is CLIP, not SimCSE.

``NT-Xent`` (SimCLR)
    Builds a ``2N x 2N`` similarity matrix over the concatenation of both views, masks the
    diagonal self-similarities, and normalizes over a ``2N - 1`` denominator -- so each
    anchor's negatives include the *other view's* non-matching entries as well. This class
    uses an ``N x N`` matrix with an ``N``-term denominator. The two objectives are close
    but not equal, and the difference is in the denominator, not the temperature.

.. note::

    **No dtype story is needed here.** :meth:`keras.losses.Loss.__init__` sets ``dtype``
    from ``keras.backend.floatx()`` (``"float32"``) irrespective of the global
    mixed-precision policy, and :meth:`keras.losses.Loss.__call__` casts both ``y_true``
    and ``y_pred`` to it before :meth:`call` runs (``keras/src/losses/loss.py:60-66`` and
    ``:100-101``, read 2026-08-30). Under ``mixed_bfloat16`` this loss therefore already
    receives float32 operands; any claim that it casts to avoid an fp16/bf16 overflow
    would be fabricated.

References:
    - van den Oord, A., Li, Y. and Vinyals, O. (2018). "Representation Learning with
      Contrastive Predictive Coding." https://arxiv.org/abs/1807.03748 (InfoNCE).
    - Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural
      Language Supervision." https://arxiv.org/abs/2103.00020 -- Figure 3 gives the
      symmetric two-direction pseudocode this class implements.
    - Gao, T., Yao, X. and Chen, D. (2021). "SimCSE: Simple Contrastive Learning of
      Sentence Embeddings." https://arxiv.org/abs/2104.08821; reference implementation
      ``princeton-nlp/SimCSE``, ``simcse/models.py`` (one-directional; read 2026-08-30).
    - Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of Visual
      Representations." https://arxiv.org/abs/2002.05709 (NT-Xent, ``2N - 1``
      denominator).
"""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Dictionary keys accepted by :func:`_split_views` for the two-view dict form.
_VIEW_KEYS: Tuple[str, str] = ("view_a", "view_b")

# ---------------------------------------------------------------------


def _validate_temperature(temperature: float) -> float:
    """Validate the softmax temperature.

    Interface contract (called by :meth:`SymmetricInfoNCELoss.__init__`):

    :param temperature: Positive scale the similarity matrix is divided by.
    :returns: ``temperature`` as a plain ``float``.
    :raises ValueError: If ``temperature`` is a ``bool``, is not a real number, or is not
        strictly positive. Zero divides by zero; a negative value flips the sign of every
        logit, turning the objective into its own opposite without raising.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(
            f"temperature must be a positive real number, got "
            f"{type(temperature).__name__}: {temperature!r}"
        )
    if not temperature > 0.0:
        raise ValueError(
            f"temperature must be > 0 (it divides the similarity matrix; zero divides by "
            f"zero and a negative value inverts the objective), got {temperature!r}"
        )
    return float(temperature)


def _split_views(y_pred: Any) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Split a two-view prediction into its two ``(batch, dim)`` halves.

    Interface contract (the single entry point :meth:`SymmetricInfoNCELoss.call` uses to
    normalize every accepted input form into one pair of tensors):

    :param y_pred: One of three forms --

        * a stacked rank-3 tensor of shape ``(batch, 2, dim)``, split on axis 1;
        * a 2-tuple or 2-list of rank-2 ``(batch, dim)`` tensors;
        * a dict with keys ``"view_a"`` and ``"view_b"`` holding rank-2 tensors.

    :returns: ``(view_a, view_b)``, both rank-2 tensors.
    :raises ValueError: If the rank-3 form's axis 1 is not exactly 2; if a rank-2 (or any
        non-rank-3) tensor is passed; if a tuple/list has a length other than 2; if the
        dict is missing either key; if either view is not rank 2; or if the batch axis is
        **statically** 1.

    .. warning::

        The batch-of-one guard is **partial by construction**. Under ``fit()`` the batch
        axis is ``None`` (dynamic), so ``shape[0]`` carries no ``1`` to test and the guard
        cannot fire. Use ``drop_remainder=True`` on the training dataset: a trailing batch
        of one produces a ``1x1`` logit matrix whose softmax is identically ``1.0``, so the
        loss is exactly ``0.0`` and the gradient is exactly zero -- a healthy-looking step
        that measures nothing.
    """
    if isinstance(y_pred, dict):
        missing = [key for key in _VIEW_KEYS if key not in y_pred]
        if missing:
            raise ValueError(
                f"y_pred dict is missing the key(s) {missing}; the two-view dict form "
                f"requires exactly the keys {list(_VIEW_KEYS)}, got "
                f"{sorted(y_pred.keys())}."
            )
        view_a, view_b = y_pred[_VIEW_KEYS[0]], y_pred[_VIEW_KEYS[1]]
    elif isinstance(y_pred, (tuple, list)):
        if len(y_pred) != 2:
            raise ValueError(
                f"y_pred sequence must hold exactly 2 views (view_a, view_b), got "
                f"{len(y_pred)} elements."
            )
        view_a, view_b = y_pred[0], y_pred[1]
    else:
        stacked = keras.ops.convert_to_tensor(y_pred)
        shape = tuple(stacked.shape)
        if len(shape) != 3:
            raise ValueError(
                f"y_pred tensor must be the stacked rank-3 form (batch, 2, dim), got "
                f"rank {len(shape)} with shape {shape}. Accepted alternatives are a "
                f"2-tuple, a 2-list, or a dict with keys {list(_VIEW_KEYS)}."
            )
        if shape[1] != 2:
            raise ValueError(
                f"y_pred stacked tensor must have exactly 2 views on axis 1, got "
                f"shape[1]={shape[1]} (full shape {shape})."
            )
        view_a, view_b = stacked[:, 0, :], stacked[:, 1, :]

    view_a = keras.ops.convert_to_tensor(view_a)
    view_b = keras.ops.convert_to_tensor(view_b)

    for name, view in zip(_VIEW_KEYS, (view_a, view_b)):
        if len(view.shape) != 2:
            raise ValueError(
                f"{name} must be a rank-2 (batch, dim) tensor, got rank "
                f"{len(view.shape)} with shape {tuple(view.shape)}."
            )

    # DECISION plan-2026-08-30T191258-29fae917/D-004
    # REJECT a statically-known batch of 1; do NOT downgrade this to a warning or to a
    # silent 0.0. A 1x1 logit matrix softmaxes to exactly 1.0, so the loss is 0.0 and the
    # gradient is 0 for ANY embeddings -- a step that looks healthy and measures nothing.
    # The guard is deliberately partial: a dynamic (None) batch axis cannot be checked.
    # See decisions.md D-004; same defect class colbert_loss.py rejects for nway < 2.
    batch = view_a.shape[0]
    if batch is not None and int(batch) == 1:
        raise ValueError(
            "y_pred has a batch size of 1, which makes this loss a no-op: the 1x1 "
            "similarity matrix softmaxes to identically 1.0, so the loss is exactly 0.0 "
            "and the gradient is exactly zero regardless of the embeddings. Use a batch "
            "of at least 2 (with batch N each anchor gets N-1 negatives), and set "
            "drop_remainder=True on the training dataset."
        )

    return view_a, view_b


@register_dl_technique("dl_techniques.losses.infonce_loss")
class SymmetricInfoNCELoss(keras.losses.Loss):
    """Symmetric InfoNCE over two views of one batch, positives on the diagonal.

    Per batch, with ``a`` and ``b`` the two views and ``t`` the temperature::

        logits   = a @ b.T / t                       # (batch, batch)
        targets  = arange(batch)                     # the positive is on the diagonal
        forward  = sparse_categorical_crossentropy(targets, logits,   from_logits=True)
        backward = sparse_categorical_crossentropy(targets, logits.T, from_logits=True)
        per_sample = (forward + backward) / 2        # (batch,)

    :meth:`call` returns that ``(batch,)`` **vector**, not a scalar. The parent
    :class:`keras.losses.Loss` applies the reduction; under the default
    ``sum_over_batch_size`` the reported value is ``sum(per_sample) / batch``, which by
    linearity of the mean equals ``mean(forward + backward) / 2`` exactly.

    .. warning::

        **``y_true`` is ignored entirely.** The positives are *positional*: row ``i`` of
        view A pairs with row ``i`` of view B. It is stated here rather than left implicit
        because a silently-unused argument is indistinguishable from a wiring bug. Note
        also that ``y_true`` must still be a real tensor -- :meth:`keras.losses.Loss.__call__`
        converts it before dispatching, so ``loss_fn(None, y_pred)`` raises inside Keras.
        Pass ``keras.ops.zeros((batch,))`` or any same-dtype placeholder.

    .. note::

        Constructing this loss with a non-default ``reduction=`` moves it off the
        ``sum(per_sample) / batch`` semantics described above. That is permitted and
        unguarded; do not do it unless you mean to.

    :param temperature: Positive scale the similarity matrix is divided by. Small values
        sharpen the softmax and weight the hardest negatives more heavily. Must be ``> 0``.
    :type temperature: float
    :param normalize_inputs: If ``True``, L2-normalize both views along their last axis
        before the matmul, making the logits cosine similarities regardless of what the
        caller passed. Leave ``False`` (the default) when the model already ends in a
        normalization step -- the temperature is only meaningful on unit-norm embeddings,
        and an unnormalized caller otherwise gets a plausible, wrong number with no error.
    :type normalize_inputs: bool
    :param kwargs: Standard :class:`keras.losses.Loss` keyword arguments (``name``,
        ``reduction``, ``dtype``).
    :raises ValueError: If ``temperature`` is not a positive real number.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.losses.infonce_loss import SymmetricInfoNCELoss

        loss_fn = SymmetricInfoNCELoss(temperature=0.05)
        views = keras.random.normal((8, 2, 256))          # (batch, 2, dim)
        views = keras.ops.normalize(views, axis=-1)
        loss_fn(keras.ops.zeros((8,)), views)
    """

    def __init__(
        self,
        temperature: float = 0.05,
        normalize_inputs: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.temperature = _validate_temperature(temperature)
        self.normalize_inputs = bool(normalize_inputs)

    def call(
        self,
        y_true: Optional[keras.KerasTensor],
        y_pred: Any,
    ) -> keras.KerasTensor:
        """Compute the per-sample symmetric InfoNCE loss.

        :param y_true: **Unused.** See the class-level warning: the positives are
            positional. Accepted for the Keras loss API contract only.
        :type y_true: keras.KerasTensor or None
        :param y_pred: Two views, in any of the forms :func:`_split_views` accepts:
            a stacked ``(batch, 2, dim)`` tensor, a 2-tuple/2-list of ``(batch, dim)``
            tensors, or a dict with keys ``"view_a"`` / ``"view_b"``.
        :type y_pred: Any
        :returns: Per-sample losses of shape ``(batch,)``.
        :rtype: keras.KerasTensor
        :raises ValueError: Propagated from :func:`_split_views` for any malformed
            ``y_pred`` or a statically-1 batch axis.
        """
        del y_true  # Positional positives; see the class docstring.

        view_a, view_b = _split_views(y_pred)

        if self.normalize_inputs:
            view_a = keras.ops.normalize(view_a, axis=-1)
            view_b = keras.ops.normalize(view_b, axis=-1)

        logits = keras.ops.matmul(
            view_a, keras.ops.transpose(view_b)
        ) / self.temperature
        targets = keras.ops.arange(keras.ops.shape(logits)[0])

        forward = keras.losses.sparse_categorical_crossentropy(
            targets, logits, from_logits=True
        )
        backward = keras.losses.sparse_categorical_crossentropy(
            targets, keras.ops.transpose(logits), from_logits=True
        )

        # DECISION plan-2026-08-30T191258-29fae917/D-002
        # RETURN THE (batch,) VECTOR. Do NOT restore the reference's scalar
        # `keras.ops.mean(forward + backward) / 2.0`. The reduced value is identical
        # either way when no weights are passed (measured bit-exact) -- that is NOT the
        # reason. The reason is what a scalar does to sample_weight, MEASURED:
        #   * with no sample_weight, a rank-0 return hits the early return in
        #     `reduce_values` (keras/src/losses/loss.py:143-147) and skips reduction --
        #     and with it `scale_loss_for_distribution`;
        #   * with a sample_weight, `reduce_weighted_values` computes
        #     `values * sample_weight` BEFORE reducing, so the scalar BROADCASTS to
        #     (batch,) and is then divided by batch. The result is
        #     `whole_batch_loss * mean(sample_weight)`: every sample is charged the
        #     batch aggregate and WHICH samples were weighted is silently discarded.
        # Measured on 8 rows with a one-hot weight keeping only row 0: this vector form
        # gives 0.5340447 (= per_sample[0]/8, correct); the scalar form gives 0.7308553
        # (= 5.8468428/8, the unweighted batch loss scaled by mean(sw)). A plausible
        # wrong number, not an error. See decisions.md D-002.
        return (forward + backward) / 2.0

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration.

        :returns: Base :class:`keras.losses.Loss` config plus ``temperature`` and
            ``normalize_inputs``.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "temperature": self.temperature,
                "normalize_inputs": self.normalize_inputs,
            }
        )
        return config


def create_symmetric_infonce_loss(
    temperature: float = 0.05,
    normalize_inputs: bool = False,
    **kwargs: Any,
) -> SymmetricInfoNCELoss:
    """Build a :class:`SymmetricInfoNCELoss`.

    :param temperature: Positive softmax temperature over the similarity matrix.
    :type temperature: float
    :param normalize_inputs: L2-normalize both views inside the loss before the matmul.
    :type normalize_inputs: bool
    :param kwargs: Standard :class:`keras.losses.Loss` keyword arguments.
    :returns: The configured loss instance.
    :rtype: SymmetricInfoNCELoss
    :raises ValueError: If ``temperature`` is not a positive real number.
    """
    return SymmetricInfoNCELoss(
        temperature=temperature,
        normalize_inputs=normalize_inputs,
        **kwargs,
    )


logger.debug("Symmetric InfoNCE loss registered (single tower, two views, N-1 negatives)")
