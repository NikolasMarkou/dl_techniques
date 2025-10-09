"""
A margin-based cosine similarity loss for feature alignment.

This loss function is designed for knowledge distillation or transfer learning
scenarios where a "student" model is trained to mimic the feature
representations of a pre-trained, frozen "teacher" model. Its goal is to
transfer the semantic knowledge embedded in the teacher's feature space to
the student, often when the student has a different or smaller architecture.

Conceptual Overview:
    The core principle is to align the feature vectors produced by the student
    with those from the teacher for the same input. Instead of forcing an
    exact match of the feature vectors (e.g., with L2 loss), this loss
    focuses on aligning their *direction* in the high-dimensional embedding
    space. The direction of a feature vector is often a better proxy for
    semantic meaning than its magnitude. By maximizing the cosine similarity,
    the student learns to place its representations in the same conceptual
    direction as the teacher.

Architectural Design:
    The loss architecture is built upon two key components:
    1.  Cosine Similarity: The primary metric for alignment. Both the teacher
        (y_true) and student (y_pred) feature vectors are first L2-normalized
        to project them onto the unit hypersphere. The loss is then derived
        from the dot product of these normalized vectors, which is equivalent
        to their cosine similarity. A similarity of 1 indicates perfect
        alignment, while -1 indicates perfect opposition.
    2.  Margin Threshold: The loss incorporates a margin `m`. The model is
        only penalized if the cosine similarity between a pair of feature
        vectors falls *below* this margin. If the alignment is already "good
        enough" (i.e., similarity > `m`), the loss for that sample is zero.
        This prevents the model from overfitting to the teacher's exact
        representations and focuses the training effort on the most poorly
        aligned examples, acting as a form of regularization.

Mathematical Formulation:
    For a given pair of teacher and student feature vectors, `v_true` and
    `v_pred`, the loss is calculated as follows:

    1.  Normalize the vectors to unit length:
        `v̂_true = v_true / ||v_true||₂`
        `v̂_pred = v_pred / ||v_pred||₂`

    2.  Compute the cosine similarity:
        `sim = v̂_true ⋅ v̂_pred`

    3.  Apply the margin `m` to compute the final loss for the sample:
        Loss = max(0, 1 - sim) if sim < m, else 0.
        This can be expressed more concisely as:
        Loss = max(0, m - sim) and then scaled, but the code's approach
        of `(1-sim)` for `sim < m` is a common variant.

References:
    The technique of aligning intermediate feature representations is a form
    of knowledge distillation, extending the original concept.
    -   Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge
        in a Neural Network." https://arxiv.org/abs/1503.02531
    -   Romero, A., et al. (2014). "FitNets: Hints for Thin Deep Nets."
        (Introduced the idea of aligning intermediate feature maps).
        https://arxiv.org/abs/1412.6550
"""

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
