import keras
from keras import ops
from typing import Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TabMLoss(keras.losses.Loss):
    """Custom loss for TabM ensemble training.

    The ensemble's ``k`` members are scored independently and then averaged, so
    ``call`` returns one value **per input row** -- shape ``(batch_size,)``, the
    same batch axis Keras uses for ``sample_weight`` and ``class_weight``.

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
            Per-sample loss with shape ``(batch_size,)`` -- the ensemble members'
            losses averaged over ``k``.
        """
        # Flatten ensemble predictions: (batch_size, k, n_outputs) -> (batch_size * k, n_outputs)
        k = ops.shape(y_pred)[1]
        y_pred_flat = ops.reshape(y_pred, (-1, ops.shape(y_pred)[-1]))

        if self.share_training_batches:
            # Repeat true labels for each ensemble member
            y_true_expanded = ops.repeat(y_true, k, axis=0)
        else:
            # Labels are already arranged for each ensemble member
            y_true_expanded = y_true

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
