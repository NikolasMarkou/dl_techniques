"""
Shared ``__call__`` guard of the per-sample losses
==================================================

``DiceLossPerChannel``, ``PerChannelBinaryLoss``, ``ClusteringLoss``,
``GNNUtilizationLoss`` and ``MANNUtilizationLoss`` return one value per sample,
shape ``(batch,)``. ``keras.losses.Loss.__call__`` then multiplies two things into
those rows that a ``(batch,)`` vector cannot absorb:

* the Keras mask of ``y_pred`` (set by ``Masking``, ``Embedding(mask_zero=True)``, an
  ``LSTM(return_sequences=True)`` and so on), whose shape is ``y_pred.shape[:-1]``;
* a ``sample_weight`` with an axis beyond the batch (a per-timestep weight).

Against ``(batch,)`` rows either one is a shape error, or, when the batch equals the
other length, a silent mis-weighting of the rows. The scalar-returning losses these
five replaced ignored both without saying so.

``PerSampleCallMixin`` closes that hole once, for all five, before Keras sees the
call. Put it FIRST in the bases: ``class X(PerSampleCallMixin, keras.losses.Loss)``.
"""

import numpy as np

# ---------------------------------------------------------------------


def split_keras_mask(y_pred):
    """Return ``(tensor, mask)`` from the argument of a mask-honoring ``call``.

    ``PerSampleCallMixin.__call__`` hands such a ``call`` the pair ``(y_pred, mask)``
    (the mask already cast to the loss dtype); a direct ``call(y_true, y_pred)`` on a
    tensor that carries a ``(batch, axis)`` ``_keras_mask`` is read the same way.

    Args:
        y_pred: A tensor, or the ``(tensor, mask)`` pair built by ``__call__``.

    Returns:
        ``(tensor, mask)``; ``mask`` is ``None`` unless a rank-2 mask is present.
    """
    if isinstance(y_pred, tuple):
        return y_pred
    mask = getattr(y_pred, "_keras_mask", None)
    if mask is not None and len(mask.shape) != 2:
        mask = None
    return y_pred, mask


def _shape_of(value):
    shape = getattr(value, "shape", None)
    return tuple(np.shape(value) if shape is None else shape)


# DECISION plan-2026-11-11T223825-64d1b401/D-014
# One mixin, not five copies, and a REFUSAL rather than a silent default: a mask the
# loss cannot honor must not be dropped (the old scalars did, unannounced) and must not
# reach Keras (a shape error, or wrong weights when the batch equals the masked axis).
# Do NOT clear ``_keras_mask`` on the caller's tensor to "consume" it: that mutates the
# model output in place. The pair ``(y_pred, mask)`` carries the mask through Keras's own
# ``map_structure`` instead, and a pair has no ``_keras_mask`` of its own.
class PerSampleCallMixin:
    """Make ``__call__`` refuse, or hand over, what ``(batch,)`` rows cannot absorb.

    Contract (``keras.losses.Loss`` subclasses whose ``call`` returns ``(batch,)``):

    * ``sample_weight`` must be a scalar, ``(batch,)`` or ``(batch, 1)``. Any other
      static shape raises ``ValueError`` naming ``sample_weight``.
    * A Keras mask on ``y_pred`` of rank 1 (one flag per sample) is left to Keras: it
      is a 0/1 sample weight, and ``sum(rows * m) / sum(m)`` is what comes out.
    * A mask of rank 2 or more raises ``ValueError`` naming the Keras mask, unless the
      subclass sets ``honored_mask_rank`` to that rank. Then ``call`` receives
      ``(y_pred, mask)`` (see ``split_keras_mask``), the mask is NOT visible to Keras
      any more, and the subclass must apply it inside its per-row means.

    The check reads the mask in ``__call__``, before Keras casts ``y_pred`` (a cast under
    ``mixed_float16`` returns a new tensor without the mask attribute). A direct
    ``call`` is not guarded. The mask is read with ``getattr(y_pred, "_keras_mask")``,
    the attribute Keras documents, because ``keras.backend`` exports no accessor.
    """

    #: Rank of the Keras mask the subclass applies itself, or ``None`` for "refuse all".
    honored_mask_rank = None

    #: Names the axis of an honored mask, for the error text ("time", "entities").
    honored_mask_axis = ""

    def __call__(self, y_true, y_pred, sample_weight=None):
        name = type(self).__name__
        if sample_weight is not None:
            weight_shape = _shape_of(sample_weight)
            if len(weight_shape) > 2 or (
                len(weight_shape) == 2 and weight_shape[1] not in (1, None)
            ):
                raise ValueError(
                    f"{name} returns one value per sample, so sample_weight must be a "
                    f"scalar, (batch,) or (batch, 1); got shape {weight_shape}. A weight "
                    "with an extra axis has no row to attach to (Keras would broadcast it "
                    "against the (batch,) rows: a shape error, or silent mis-weighting "
                    "when the lengths coincide). Reduce the extra axes yourself first."
                )
        mask = getattr(y_pred, "_keras_mask", None)
        if mask is not None and len(mask.shape) != 1:
            if len(mask.shape) != self.honored_mask_rank:
                mask_shape = tuple(mask.shape)
                if self.honored_mask_rank is None:
                    way_out = f"{name} gives a mask no meaning."
                else:
                    way_out = (
                        f"{name} honors only a rank-{self.honored_mask_rank} "
                        f"(batch, {self.honored_mask_axis}) mask."
                    )
                raise ValueError(
                    f"{name} returns one value per sample, so it cannot take a Keras mask "
                    f"of shape {mask_shape} on y_pred: Keras would multiply the mask into "
                    "the (batch,) rows (a shape error, or silent mis-weighting when the "
                    f"batch equals the mask's second axis). {way_out} Turn the padding "
                    "into a sample_weight of shape (batch,), or end the model with a "
                    "layer whose compute_mask returns None. A (batch,) mask is accepted "
                    "and acts as a 0/1 sample weight."
                )
            y_pred = (y_pred, mask)
        return super().__call__(y_true, y_pred, sample_weight)
