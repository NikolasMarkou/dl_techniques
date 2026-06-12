"""
Flow-Matching Velocity Loss
===========================

Rectified-flow / flow-matching velocity regression loss.

In rectified flow (flow matching), a model is trained to predict a velocity
field that transports a data sample ``x0`` to a noise sample ``x1`` along a
straight path. The regression target is the constant velocity::

    v_target = x1 - x0

and the loss is the plain mean-squared error between the predicted velocity
``y_pred`` and the target velocity ``y_true``::

    L = mean((y_pred - y_true) ** 2)

The loss itself is convention-agnostic: it simply compares the predicted
velocity to whatever velocity target is supplied as ``y_true``. Constructing
``v_target = x1 - x0`` (and any logit-normal time sampling / time-dependent
sample weighting) is the responsibility of the data/sampling pipeline in the
trainer, NOT this loss. An optional scalar ``loss_weight`` multiplier is
provided for convenience when blending multiple loss terms; time-dependent
weighting is deliberately NOT implemented here (it belongs at the sampling
level, where the per-sample time is available).

Example::

    from dl_techniques.losses import FlowMatchingVelocityLoss

    model.compile(
        optimizer="adam",
        loss=FlowMatchingVelocityLoss(),
    )
"""

import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable(package="dl_techniques.losses")
class FlowMatchingVelocityLoss(keras.losses.Loss):
    """Velocity MSE loss for rectified-flow / flow-matching training.

    Computes the mean squared error between a predicted velocity and a target
    velocity, reducing over the feature axis and letting the configured Keras
    ``reduction`` handle the batch axis. The target velocity is typically
    ``v_target = x1 - x0`` (noise minus data), but this loss only compares
    ``y_pred`` against the supplied ``y_true``.

    Args:
        loss_weight: Scalar multiplier applied to the loss. Useful when
            blending this term with others. Defaults to ``1.0``.
        time_weighting: Documentation-only hook. Logit-normal time weighting in
            flow matching is applied at the data/sampling level in the trainer
            (where the per-sample diffusion time is available), NOT inside the
            loss. Keras ``Loss.call`` only receives ``(y_true, y_pred)``, so no
            per-sample time is available here. When ``True``, a one-time info
            log reminds the caller of this contract; the numerical loss is
            unchanged. Defaults to ``False``.
        name: Loss name for logging. Defaults to
            ``'flow_matching_velocity_loss'``.
        reduction: Reduction over the batch axis (``'sum_over_batch_size'``,
            ``'sum'``, ``'none'``). Defaults to ``'sum_over_batch_size'``.
        **kwargs: Additional arguments passed to ``keras.losses.Loss``.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        time_weighting: bool = False,
        name: str = "flow_matching_velocity_loss",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.loss_weight = loss_weight
        self.time_weighting = time_weighting

        if self.time_weighting:
            logger.info(
                "FlowMatchingVelocityLoss(time_weighting=True): time-dependent "
                "weighting is a no-op in the loss; apply logit-normal time "
                "weighting at the data/sampling level in the trainer."
            )

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute per-sample velocity MSE (reduced over the feature axis).

        Args:
            y_true: Target velocity tensor of shape ``(..., C)``.
            y_pred: Predicted velocity tensor of shape ``(..., C)``.

        Returns:
            Per-sample loss tensor of shape ``(...)`` (feature axis reduced).
            The configured Keras reduction then collapses the remaining axes.
        """
        y_true = ops.cast(y_true, y_pred.dtype)
        per_sample = ops.mean(ops.square(y_pred - y_true), axis=-1)
        return self.loss_weight * per_sample

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "loss_weight": self.loss_weight,
            "time_weighting": self.time_weighting,
        })
        return config
