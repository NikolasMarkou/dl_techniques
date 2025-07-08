import keras
from keras import ops
from typing import Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FeatureAlignmentLoss(keras.losses.Loss):
    """Feature alignment loss for semantic prior transfer.

    Implements a cosine similarity based loss with a margin threshold.
    Features below the margin contribute to the loss proportionally.
    Uses a cosine similarity metric to measure feature alignment between
    target and predicted features.

    Args:
        margin: Float, similarity threshold for feature alignment. Features
            with similarity above this threshold do not contribute to the loss.
            Should be in range [0, 1]. Defaults to 0.85.
        name: String, name of the loss function. Defaults to 'feature_alignment_loss'.
        **kwargs: Additional keyword arguments passed to the parent Loss class.

    Example:
        >>> import keras
        >>> import tensorflow as tf
        >>> loss_fn = FeatureAlignmentLoss(margin=0.8)
        >>> y_true = tf.random.normal([32, 128])
        >>> y_pred = tf.random.normal([32, 128])
        >>> loss = loss_fn(y_true, y_pred)
        >>> print(loss.shape)
        ()
    """

    def __init__(
            self,
            margin: float = 0.85,
            name: str = 'feature_alignment_loss',
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        if not 0.0 <= margin <= 1.0:
            raise ValueError(f"Margin must be in range [0, 1], got {margin}")

        self.margin = margin
        logger.info(f"Initialized FeatureAlignmentLoss with margin={margin}")

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute feature alignment loss.

        Args:
            y_true: Target features from frozen encoder. Shape: (batch_size, feature_dim)
            y_pred: Predicted features from trainable encoder. Shape: (batch_size, feature_dim)

        Returns:
            Computed loss value as a scalar tensor.
        """
        # Normalize features to unit vectors for cosine similarity
        y_true_norm = ops.divide(
            y_true,
            ops.maximum(ops.norm(y_true, axis=-1, keepdims=True), ops.cast(1e-12, y_true.dtype))
        )
        y_pred_norm = ops.divide(
            y_pred,
            ops.maximum(ops.norm(y_pred, axis=-1, keepdims=True), ops.cast(1e-12, y_pred.dtype))
        )

        # Compute cosine similarity
        similarity = ops.sum(ops.multiply(y_true_norm, y_pred_norm), axis=-1)

        # Compute loss: 1 - similarity for features below margin
        loss = ops.subtract(ops.cast(1.0, similarity.dtype), similarity)

        # Apply margin threshold - only penalize features below margin
        loss = ops.where(
            ops.greater(similarity, ops.cast(self.margin, similarity.dtype)),
            ops.cast(0.0, loss.dtype),
            loss
        )

        return ops.mean(loss)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the loss function.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "margin": self.margin,
        })
        return config

# ---------------------------------------------------------------------
