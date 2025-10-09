import keras
from keras import ops
from typing import Union

# ---------------------------------------------------------------------
# lolca imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


class TabMLoss(keras.losses.Loss):
    """Custom loss for TabM ensemble training.

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

    def call(self, y_true, y_pred):
        """Compute loss for TabM ensemble predictions.

        Args:
            y_true: True labels with shape (batch_size,) or (batch_size, n_classes).
            y_pred: Ensemble predictions with shape (batch_size, k, n_outputs).

        Returns:
            Computed loss value.
        """
        # Flatten ensemble predictions: (batch_size, k, n_outputs) -> (batch_size * k, n_outputs)
        y_pred_flat = ops.reshape(y_pred, (-1, ops.shape(y_pred)[-1]))

        if self.share_training_batches:
            # Repeat true labels for each ensemble member
            k = ops.shape(y_pred)[1]
            if len(ops.shape(y_true)) == 1:
                y_true_expanded = ops.repeat(y_true, k, axis=0)
            else:
                y_true_expanded = ops.repeat(y_true, k, axis=0)
        else:
            # Labels are already arranged for each ensemble member
            y_true_expanded = y_true

        return self.base_loss(y_true_expanded, y_pred_flat)

    def get_config(self):
        config = super().get_config()
        config.update({
            'base_loss': keras.losses.serialize(self.base_loss),
            'share_training_batches': self.share_training_batches,
        })
        return config

# ---------------------------------------------------------------------
