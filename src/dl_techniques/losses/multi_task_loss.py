"""
Multi-Task Loss
===============

Weighted combination of per-task MSE losses for multi-output models that
produce dictionary outputs. Each task's loss contribution is scaled by a
configurable weight.

Example::

    from dl_techniques.losses import MultiTaskLoss

    loss_fn = MultiTaskLoss(loss_weights={
        "pressure": 1.0,
        "velocity": 2.0,
        "vorticity": 0.5,
    })
    model.compile(optimizer='adam', loss=loss_fn)
"""

import keras
from keras import ops
from typing import Any, Dict, Optional


@keras.saving.register_keras_serializable(package="dl_techniques.losses")
class MultiTaskLoss(keras.losses.Loss):
    """Per-task weighted MSE loss for dictionary-based multi-output models.

    Iterates over the keys of ``y_true`` (a dictionary), computes MSE
    against the corresponding entry in ``y_pred``, and returns the weighted
    sum of all task losses.

    Args:
        loss_weights: Dictionary mapping task name to its loss weight.
            Tasks not present default to weight 1.0.
        name: Loss name for logging.
        **kwargs: Additional arguments passed to ``keras.losses.Loss``.
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        name: str = "multi_task_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.loss_weights = loss_weights or {}

    def call(
        self,
        y_true: Dict[str, keras.KerasTensor],
        y_pred: Dict[str, keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Compute weighted sum of per-task MSE losses.

        Args:
            y_true: Dictionary mapping task names to ground truth tensors.
            y_pred: Dictionary mapping task names to prediction tensors.

        Returns:
            Scalar total loss value.
        """
        total_loss = ops.convert_to_tensor(0.0)
        for key in y_true:
            if key in y_pred:
                task_true = ops.cast(y_true[key], y_pred[key].dtype)
                task_loss = ops.mean(ops.square(task_true - y_pred[key]))
                weight = self.loss_weights.get(key, 1.0)
                total_loss = total_loss + task_loss * weight
        return total_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "loss_weights": self.loss_weights,
        })
        return config
