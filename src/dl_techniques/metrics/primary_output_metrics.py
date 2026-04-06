"""
Primary Output Metrics
======================

Metrics that evaluate only the primary (first) output of multi-output
models. Useful for deep supervision architectures where the model produces
outputs at multiple scales but only the highest-resolution output should
be tracked for reporting.

Example::

    from dl_techniques.metrics.primary_output_metrics import (
        PrimaryOutputAccuracy,
        PrimaryOutputTopKAccuracy,
    )

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=[
            PrimaryOutputAccuracy(),
            PrimaryOutputTopKAccuracy(k=5),
        ],
    )
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Union


@keras.saving.register_keras_serializable(package="dl_techniques.metrics")
class PrimaryOutputAccuracy(keras.metrics.Metric):
    """Classification accuracy for the primary output of multi-output models.

    Extracts the first element when ``y_pred`` is a list, then computes
    standard top-1 accuracy. Supports both one-hot encoded and integer
    class label formats for ``y_true``.

    Args:
        name: Metric name for logging.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        name: str = "primary_accuracy",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: Union[keras.KerasTensor, List[keras.KerasTensor]],
        y_pred: Union[keras.KerasTensor, List[keras.KerasTensor]],
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update accuracy state.

        Args:
            y_true: Ground truth — single tensor or list (first used).
            y_pred: Predictions — single tensor or list (first used).
            sample_weight: Optional per-sample weighting (unused).
        """
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0] if isinstance(y_true, list) else y_true
        else:
            primary_pred = y_pred
            primary_true = y_true

        primary_pred = ops.cast(primary_pred, "float32")
        primary_true = ops.cast(primary_true, "float32")

        predicted_classes = ops.argmax(primary_pred, axis=-1)

        # Handle both one-hot and integer labels
        true_rank = len(ops.shape(primary_true))
        if true_rank > 1 and ops.shape(primary_true)[-1] > 1:
            true_classes = ops.argmax(primary_true, axis=-1)
        else:
            true_classes = ops.cast(
                ops.squeeze(primary_true, axis=-1) if true_rank > 1 else primary_true,
                "int64",
            )

        predicted_classes = ops.cast(predicted_classes, "int64")
        true_classes = ops.cast(true_classes, "int64")

        matches = ops.cast(ops.equal(predicted_classes, true_classes), "float32")
        self.total.assign_add(ops.sum(matches))
        self.count.assign_add(ops.cast(ops.shape(matches)[0], "float32"))

    def result(self) -> keras.KerasTensor:
        """Compute the mean accuracy."""
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


@keras.saving.register_keras_serializable(package="dl_techniques.metrics")
class PrimaryOutputTopKAccuracy(keras.metrics.Metric):
    """Top-K accuracy for the primary output of multi-output models.

    Checks whether the true class is among the top ``k`` predicted classes
    for the primary (first) output.

    Args:
        k: Number of top predictions to consider.
        name: Metric name for logging.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        k: int = 5,
        name: str = "primary_top5_accuracy",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: Union[keras.KerasTensor, List[keras.KerasTensor]],
        y_pred: Union[keras.KerasTensor, List[keras.KerasTensor]],
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update top-K accuracy state.

        Args:
            y_true: Ground truth — single tensor or list (first used).
            y_pred: Predictions — single tensor or list (first used).
            sample_weight: Optional per-sample weighting (unused).
        """
        if isinstance(y_pred, list):
            primary_pred = y_pred[0]
            primary_true = y_true[0] if isinstance(y_true, list) else y_true
        else:
            primary_pred = y_pred
            primary_true = y_true

        primary_pred = ops.cast(primary_pred, "float32")
        primary_true = ops.cast(primary_true, "float32")

        # Get top-k prediction indices
        top_k_indices = ops.top_k(primary_pred, k=self.k).indices

        # Handle both one-hot and integer labels
        true_rank = len(ops.shape(primary_true))
        if true_rank > 1 and ops.shape(primary_true)[-1] > 1:
            true_classes = ops.argmax(primary_true, axis=-1)
        else:
            true_classes = ops.cast(
                ops.squeeze(primary_true, axis=-1) if true_rank > 1 else primary_true,
                "int64",
            )

        true_classes = ops.cast(true_classes, top_k_indices.dtype)

        # Check if true class is in top-k
        matches = ops.any(
            ops.equal(top_k_indices, ops.expand_dims(true_classes, axis=-1)),
            axis=-1,
        )
        matches = ops.cast(matches, "float32")

        self.total.assign_add(ops.sum(matches))
        self.count.assign_add(ops.cast(ops.shape(matches)[0], "float32"))

    def result(self) -> keras.KerasTensor:
        """Compute the mean top-K accuracy."""
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "k": self.k,
        })
        return config
