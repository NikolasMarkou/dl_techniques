import keras
from keras import ops
from typing import Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TabMLoss(keras.losses.Loss):
    """Custom loss for TabM ensemble training.

    ``call`` returns one value **per input row** -- the same batch axis Keras
    uses for ``sample_weight`` and ``class_weight``. What that axis is depends
    on ``share_training_batches``, because the two modes feed the model
    differently:

    * ``share_training_batches=True`` (default, and always at inference): the
      caller supplies ``B`` rows, the model tiles them ``k`` ways, and every
      member scores the same row. The ``k`` member losses are AVERAGED, so the
      returned shape is ``(B,)`` -- one value per supplied row.
    * ``share_training_batches=False``: the caller supplies ``B * k`` rows and
      the model reshapes them so each member gets a disjoint slice. Each row is
      then a distinct sample scored by exactly one member, there is nothing to
      average over, and the returned shape is ``(B * k,)`` -- again one value
      per supplied row.

    In both cases the returned axis matches ``y_true``'s leading axis, so a
    ``sample_weight`` / ``class_weight`` sized like the labels lines up.

    Args:
        base_loss: Base loss function to use.
        share_training_batches: Whether batches are shared across ensemble members.
        name: Loss name.
    """

    def __init__(
            self,
            base_loss: Union[str, keras.losses.Loss] = 'mse',
            share_training_batches: bool = True,
            name: str = 'tabm_loss',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.base_loss = keras.losses.get(base_loss)
        self.share_training_batches = share_training_batches

    def _unreduced_base_loss(self, y_true, y_pred):
        """Evaluate ``base_loss`` WITHOUT its own batch reduction.

        ``keras.losses.get`` returns a plain function for a string spec (which is
        already per-sample) but the object itself for a ``Loss`` instance (whose
        ``__call__`` applies ``sum_over_batch_size`` and collapses to a scalar).
        ``Loss.call`` is the unreduced half of that object, so it is what this
        class needs in both cases.

        Args:
            y_true: True labels, batch axis already expanded to ``B * k``.
            y_pred: Predictions, batch axis already flattened to ``B * k``.

        Returns:
            Per-element loss with shape ``(B * k,)``.
        """
        if isinstance(self.base_loss, keras.losses.Loss):
            return self.base_loss.call(y_true, y_pred)
        return self.base_loss(y_true, y_pred)

    def call(self, y_true, y_pred):
        """Compute loss for TabM ensemble predictions.

        Args:
            y_true: True labels with shape (batch_size,) or (batch_size, n_classes).
            y_pred: Ensemble predictions with shape (batch_size, k, n_outputs).

        Returns:
            Per-input-row loss: shape ``(B,)`` when ``share_training_batches``
            (the ensemble members' losses averaged over ``k``), shape
            ``(B * k,)`` when not (each row is one member's own sample).
        """
        # Flatten ensemble predictions: (batch_size, k, n_outputs) -> (batch_size * k, n_outputs)
        k = ops.shape(y_pred)[1]
        y_pred_flat = ops.reshape(y_pred, (-1, ops.shape(y_pred)[-1]))

        if not self.share_training_batches:
            # DECISION plan-2026-08-18T140459-7991552f/D-064: in the UNSHARED
            # arrangement the caller supplied `B * k` rows and
            # `TabM.call` reshaped them `(B * k, D) -> (B, k, D)` so each member
            # got a disjoint slice. Every row is therefore a distinct sample
            # scored by exactly ONE member.
            #
            # WHAT NOT TO DO, and why:
            #   * Do NOT fall through to the shared branch's `reshape((-1, k))`
            #     + `mean(axis=-1)`. There is no ensemble axis to average here;
            #     doing it anyway returns `(B,)` against labels of `(B * k,)`,
            #     which is what shipped: MEASURED at 9d71a8c4d,
            #     `TabMLoss('binary_crossentropy', share_training_batches=False)
            #     (y_true_(12,1), y_pred_(4,3,1), sample_weight=ones(12))` ->
            #     `InvalidArgumentError: Incompatible shapes: [4] vs. [12]
            #     [Op:Mul]`, i.e. `sample_weight`/`class_weight` was an
            #     unconditional crash in this mode while the class docstring
            #     claimed the axis lined up.
            #   * Do NOT reorder `y_pred_flat`. `reshape` is the exact inverse
            #     of the model's own `(B * k, D) -> (B, k, D)`, so flat row `i`
            #     is input row `i`; a `transpose` here would silently pair each
            #     label with another member's prediction.
            # See decisions.md D-064.
            return self._unreduced_base_loss(y_true, y_pred_flat)

        # Repeat true labels for each ensemble member
        y_true_expanded = ops.repeat(y_true, k, axis=0)

        # DECISION plan-2026-08-18T140459-7991552f/D-027: reduce the ENSEMBLE axis
        # here and return one value per input ROW. Do NOT `return
        # self.base_loss(...)` on the flattened tensor. That tensor's leading axis
        # is `B * k`, not `B`, so it is not the axis `Loss.__call__` then multiplies
        # `sample_weight` / `class_weight` against, and it is not the axis the model
        # was fit on. MEASURED at HEAD with the shipped spelling `TabMLoss(
        # 'binary_crossentropy')`, B=6, k=3: `call()` returned shape (18,) and
        # `loss(y_true, y_pred, sample_weight=<shape (6,)>)` raised
        # `InvalidArgumentError: required broadcastable shapes [Op:Mul]` -- so
        # `class_weight={0: 1.0, 1: 100.0}` in `fit()` was a hard CRASH, not the
        # silent global rescale it was reported as. The scalar-and-therefore-silent
        # shape is real too, but only for a `Loss` INSTANCE base loss (measured
        # shape ()); every call site in this repo passes a string. Averaging over k
        # (rather than summing) keeps the loss magnitude independent of ensemble
        # size, so `k` and the learning rate stay decoupled. See decisions.md D-027.
        per_member = ops.reshape(
            self._unreduced_base_loss(y_true_expanded, y_pred_flat), (-1, k)
        )
        return ops.mean(per_member, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'base_loss': keras.losses.serialize(self.base_loss),
            'share_training_batches': self.share_training_batches,
        })
        return config

# ---------------------------------------------------------------------
