import keras
import numpy as np
from typing import Union, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiLabelMetrics(keras.metrics.Metric):
    """
    Comprehensive multi-label classification metrics.

    Computes precision, recall, and F1-score across all classes with support
    for excluding background class and configurable averaging strategies.

    :param num_classes: Number of output classes.
    :type num_classes: int
    :param threshold: Classification threshold for converting probabilities to binary predictions.
    :type threshold: float
    :param exclude_background: Whether to exclude class 0 (background) from metric computation.
    :type exclude_background: bool
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :param name: Name of the metric.
    :type name: str

    :example:
        >>> metric = MultiLabelMetrics(num_classes=5, exclude_background=True)
        >>> metric.update_state(y_true, y_pred)
        >>> f1_score = metric.result()
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        exclude_background: bool = False,
        epsilon: float = 1e-7,
        name: str = "multilabel_metrics",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.exclude_background = exclude_background
        self.epsilon = epsilon

        self.true_positives = self.add_weight(
            name="tp",
            shape=(num_classes,),
            initializer="zeros",
        )
        self.false_positives = self.add_weight(
            name="fp",
            shape=(num_classes,),
            initializer="zeros",
        )
        self.false_negatives = self.add_weight(
            name="fn",
            shape=(num_classes,),
            initializer="zeros",
        )
        self.true_negatives = self.add_weight(
            name="tn",
            shape=(num_classes,),
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: Union[keras.KerasTensor, np.ndarray],
        y_pred: Union[keras.KerasTensor, np.ndarray],
        sample_weight: Optional[Union[keras.KerasTensor, np.ndarray]] = None
    ) -> None:
        """
        Accumulate confusion matrix statistics.

        :param y_true: Ground truth labels, shape (..., num_classes), binary.
        :type y_true: Union[keras.KerasTensor, np.ndarray]
        :param y_pred: Predicted probabilities, shape (..., num_classes).
        :type y_pred: Union[keras.KerasTensor, np.ndarray]
        :param sample_weight: Optional per-sample weights, shape (...,) or (..., num_classes).
        :type sample_weight: Optional[Union[keras.KerasTensor, np.ndarray]]
        """
        y_pred_binary = keras.ops.cast(y_pred >= self.threshold, "float32")
        y_true = keras.ops.cast(y_true, "float32")

        y_true_flat = keras.ops.reshape(y_true, (-1, self.num_classes))
        y_pred_flat = keras.ops.reshape(y_pred_binary, (-1, self.num_classes))

        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, "float32")
            sample_weight = keras.ops.reshape(sample_weight, (-1,))
            if keras.ops.ndim(sample_weight) == 1:
                sample_weight = keras.ops.expand_dims(sample_weight, axis=-1)
            y_true_flat = y_true_flat * sample_weight
            y_pred_flat = y_pred_flat * sample_weight

        tp = keras.ops.sum(y_true_flat * y_pred_flat, axis=0)
        fp = keras.ops.sum((1.0 - y_true_flat) * y_pred_flat, axis=0)
        fn = keras.ops.sum(y_true_flat * (1.0 - y_pred_flat), axis=0)
        tn = keras.ops.sum((1.0 - y_true_flat) * (1.0 - y_pred_flat), axis=0)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
        self.true_negatives.assign_add(tn)

    def _get_class_slice(
        self,
        tensor: Union[keras.KerasTensor, np.ndarray]
    ) -> Union[keras.KerasTensor, np.ndarray]:
        """
        Slice tensor to exclude background class if configured.

        :param tensor: Per-class tensor of shape (num_classes,).
        :type tensor: Union[keras.KerasTensor, np.ndarray]
        :return: Sliced tensor.
        :rtype: Union[keras.KerasTensor, np.ndarray]
        """
        if self.exclude_background:
            return tensor[1:]
        return tensor

    def compute_precision(self) -> Union[keras.KerasTensor, np.ndarray]:
        """
        Compute per-class precision.

        :return: Precision per class, shape (num_classes,) or (num_classes-1,).
        :rtype: Union[keras.KerasTensor, np.ndarray]
        """
        tp = self._get_class_slice(self.true_positives)
        fp = self._get_class_slice(self.false_positives)
        return tp / (tp + fp + self.epsilon)

    def compute_recall(self) -> Union[keras.KerasTensor, np.ndarray]:
        """
        Compute per-class recall.

        :return: Recall per class, shape (num_classes,) or (num_classes-1,).
        :rtype: Union[keras.KerasTensor, np.ndarray]
        """
        tp = self._get_class_slice(self.true_positives)
        fn = self._get_class_slice(self.false_negatives)
        return tp / (tp + fn + self.epsilon)

    def compute_f1(self) -> Union[keras.KerasTensor, np.ndarray]:
        """
        Compute per-class F1 score.

        :return: F1 score per class, shape (num_classes,) or (num_classes-1,).
        :rtype: Union[keras.KerasTensor, np.ndarray]
        """
        precision = self.compute_precision()
        recall = self.compute_recall()
        return 2.0 * precision * recall / (precision + recall + self.epsilon)

    def result(self) -> Union[keras.KerasTensor, np.ndarray]:
        """
        Compute macro-averaged F1 score.

        :return: Mean F1 score across classes.
        :rtype: Union[keras.KerasTensor, np.ndarray]
        """
        return keras.ops.mean(self.compute_f1())

    def reset_state(self) -> None:
        """Reset all accumulated statistics to zero."""
        zeros = keras.ops.zeros((self.num_classes,), dtype="float32")
        self.true_positives.assign(zeros)
        self.false_positives.assign(zeros)
        self.false_negatives.assign(zeros)
        self.true_negatives.assign(zeros)

    def get_config(self) -> dict:
        """
        Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "threshold": self.threshold,
            "exclude_background": self.exclude_background,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
