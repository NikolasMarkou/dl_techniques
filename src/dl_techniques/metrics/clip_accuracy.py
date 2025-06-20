"""
CLIP Accuracy Metric Implementation

This module implements accuracy metrics for CLIP (Contrastive Language-Image Pre-training)
models, measuring how well the model aligns image and text representations. The metric
evaluates the percentage of correct image-text pairs that are ranked highest in the
similarity matrix, providing insights into the model's multimodal understanding.

Mathematical Formulation:
    CLIP accuracy operates on similarity matrices and measures retrieval performance:

    For a batch of N image-text pairs with similarity matrix S[i,j]:
    1. Image-to-text accuracy: argmax_j(S[i,:]) == i for each image i
    2. Text-to-image accuracy: argmax_i(S[:,j]) == j for each text j
    3. Overall accuracy: (I2T_accuracy + T2I_accuracy) / 2

    Where the diagonal elements S[i,i] represent correct image-text pairs.

Evaluation Variants:
    - Top-1 Accuracy: Exact match (argmax equals target)
    - Top-K Accuracy: Target is within top-K predictions
    - Directional Accuracy: Separate tracking of I2T and T2I performance
    - Recall@K: Fraction of correct pairs in top-K retrievals

Key Benefits:
    - Direct measure of multimodal alignment quality
    - Interpretable metric for zero-shot retrieval tasks
    - Enables monitoring of bidirectional performance
    - Useful for model selection and hyperparameter tuning

Implementation Features:
    - Supports both averaged and directional accuracy tracking
    - Configurable top-K evaluation (K=1 by default)
    - Handles multiple input formats (dict, tuple, list)
    - Sample weighting support for imbalanced evaluation
    - Numerically stable computation with proper type handling

References:
    - Radford, A., et al. (2021). "Learning Transferable Visual Representations
      from Natural Language Supervision." https://arxiv.org/abs/2103.00020

    - Jia, C., et al. (2021). "Scaling Up Visual and Vision-Language Representation
      Learning With Noisy Text Supervision." https://arxiv.org/abs/2102.05918

    - Li, J., et al. (2022). "BLIP: Bootstrapping Language-Image Pre-training for
      Unified Vision-Language Understanding and Generation."
      https://arxiv.org/abs/2201.12086

Usage Examples:
    Basic usage:
    >>> accuracy_metric = CLIPAccuracy()
    >>> y_pred = {
    ...     'logits_per_image': similarity_matrix_i2t,
    ...     'logits_per_text': similarity_matrix_t2i
    ... }
    >>> accuracy_metric.update_state(None, y_pred)
    >>> print(f"Accuracy: {accuracy_metric.result():.3f}")

    With directional tracking:
    >>> accuracy_metric = CLIPAccuracy(
    ...     track_directions=True,
    ...     top_k=5  # Top-5 accuracy
    ... )
    >>> print(f"I2T Accuracy: {accuracy_metric.i2t_accuracy:.3f}")
    >>> print(f"T2I Accuracy: {accuracy_metric.t2i_accuracy:.3f}")

    In model compilation:
    >>> model.compile(
    ...     optimizer='adam',
    ...     loss=CLIPContrastiveLoss(),
    ...     metrics=[CLIPAccuracy(name='clip_acc')]
    ... )
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CLIPAccuracy(keras.metrics.Metric):
    """
    Accuracy metric for CLIP contrastive learning.

    This metric measures the retrieval accuracy in both image-to-text and text-to-image
    directions, providing a comprehensive evaluation of multimodal alignment quality.

    Args:
        top_k: int, default=1
            Compute top-K accuracy. K=1 gives exact match accuracy.
        track_directions: bool, default=False
            Whether to separately track image-to-text and text-to-image accuracies.
            When True, provides additional properties for directional metrics.
        average_directions: bool, default=True
            Whether to average I2T and T2I accuracies. If False, reports them
            separately (requires track_directions=True).
        name: str, default="clip_accuracy"
            Name of the metric.
        **kwargs: Additional keyword arguments for the Metric parent class.

    Input format:
        y_true: Not used (can be None) as accuracy is computed from similarity matrices
        y_pred: Dictionary containing:
            - 'logits_per_image': Tensor of shape (batch_size, batch_size)
              representing image-to-text similarities
            - 'logits_per_text': Tensor of shape (batch_size, batch_size)
              representing text-to-image similarities

    Properties (when track_directions=True):
        i2t_accuracy: Image-to-text accuracy
        t2i_accuracy: Text-to-image accuracy

    Raises:
        ValueError: If top_k is not positive, or if configuration is invalid.
    """

    def __init__(
            self,
            top_k: int = 1,
            track_directions: bool = False,
            average_directions: bool = True,
            name: str = "clip_accuracy",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate inputs
        self._validate_inputs(top_k, track_directions, average_directions)

        self.top_k = top_k
        self.track_directions = track_directions
        self.average_directions = average_directions

        # Main accuracy tracking
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

        # Directional tracking (if enabled)
        if self.track_directions:
            self.i2t_total = self.add_weight(name="i2t_total", initializer="zeros")
            self.t2i_total = self.add_weight(name="t2i_total", initializer="zeros")
            self.i2t_count = self.add_weight(name="i2t_count", initializer="zeros")
            self.t2i_count = self.add_weight(name="t2i_count", initializer="zeros")

        logger.info(f"CLIPAccuracy initialized: top_k={top_k}, "
                    f"track_directions={track_directions}, average_directions={average_directions}")

    def _validate_inputs(
            self,
            top_k: int,
            track_directions: bool,
            average_directions: bool
    ) -> None:
        """Validate initialization parameters."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not average_directions and not track_directions:
            raise ValueError(
                "If average_directions=False, track_directions must be True "
                "to enable separate directional reporting"
            )

    def update_state(
            self,
            y_true: Optional[keras.KerasTensor],
            y_pred: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
            sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """
        Update accuracy state for both image-to-text and text-to-image directions.

        Args:
            y_true: Not used (can be None)
            y_pred: Dictionary containing logits_per_image and logits_per_text
            sample_weight: Optional sample weights
        """
        # Parse predictions
        logits_per_image, logits_per_text = self._parse_predictions(y_pred)

        # Validate logits
        self._validate_logits(logits_per_image, logits_per_text)

        batch_size = ops.shape(logits_per_image)[0]

        # Create target labels (diagonal indices)
        labels = ops.arange(batch_size, dtype='int32')

        # Compute top-K accuracy for both directions
        i2t_accuracy = self._compute_topk_accuracy(logits_per_image, labels)
        t2i_accuracy = self._compute_topk_accuracy(logits_per_text, labels)

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            # Ensure sample_weight has correct shape
            if len(ops.shape(sample_weight)) == 0:
                sample_weight = ops.broadcast_to(sample_weight, (batch_size,))
            elif ops.shape(sample_weight)[0] != batch_size:
                raise ValueError(
                    f"sample_weight batch size ({ops.shape(sample_weight)[0]}) "
                    f"doesn't match predictions batch size ({batch_size})"
                )

            i2t_accuracy = i2t_accuracy * sample_weight
            t2i_accuracy = t2i_accuracy * sample_weight

        # Update directional metrics if tracking enabled
        if self.track_directions:
            self.i2t_total.assign_add(ops.sum(i2t_accuracy))
            self.t2i_total.assign_add(ops.sum(t2i_accuracy))
            self.i2t_count.assign_add(ops.cast(batch_size, self.dtype))
            self.t2i_count.assign_add(ops.cast(batch_size, self.dtype))

        # Update main metrics
        if self.average_directions:
            # Average both directions
            total_accuracy = (i2t_accuracy + t2i_accuracy) / 2.0
            self.total.assign_add(ops.sum(total_accuracy))
            self.count.assign_add(ops.cast(batch_size, self.dtype))
        else:
            # Report directional accuracies separately (handled by result() method)
            pass

    def _compute_topk_accuracy(
            self,
            logits: keras.KerasTensor,
            labels: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute top-K accuracy for given logits and labels.

        Args:
            logits: Similarity logits of shape (batch_size, batch_size)
            labels: True labels (diagonal indices)

        Returns:
            Per-sample accuracy scores
        """
        if self.top_k == 1:
            # Efficient path for top-1 accuracy
            predictions = ops.argmax(logits, axis=-1)
            matches = ops.equal(predictions, labels)
        else:
            # Top-K accuracy: check if true label is in top-K predictions
            _, top_k_indices = ops.top_k(logits, k=self.top_k, sorted=False)

            # Check if labels are in top-K predictions
            labels_expanded = ops.expand_dims(labels, axis=1)  # (batch_size, 1)
            matches = ops.any(
                ops.equal(top_k_indices, labels_expanded),
                axis=1
            )

        return ops.cast(matches, self.dtype)

    def _parse_predictions(
            self,
            y_pred: Union[Dict[str, keras.KerasTensor], keras.KerasTensor, tuple, list]
    ) -> tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Parse different prediction formats into logit tensors.

        Args:
            y_pred: Predictions in various formats

        Returns:
            Tuple of (logits_per_image, logits_per_text)
        """
        if isinstance(y_pred, dict):
            if 'logits_per_image' not in y_pred or 'logits_per_text' not in y_pred:
                raise ValueError(
                    "y_pred dict must contain 'logits_per_image' and 'logits_per_text' keys. "
                    f"Got keys: {list(y_pred.keys())}"
                )
            return y_pred['logits_per_image'], y_pred['logits_per_text']

        elif isinstance(y_pred, (list, tuple)):
            if len(y_pred) != 2:
                raise ValueError(
                    f"y_pred list/tuple must have exactly 2 elements, got {len(y_pred)}"
                )
            return y_pred[0], y_pred[1]

        else:
            raise ValueError(
                f"Unsupported y_pred format: {type(y_pred)}. "
                "Expected dict with keys ['logits_per_image', 'logits_per_text'] "
                "or list/tuple with 2 elements."
            )

    def _validate_logits(
            self,
            logits_per_image: keras.KerasTensor,
            logits_per_text: keras.KerasTensor
    ) -> None:
        """Validate logit tensor shapes and properties."""
        img_shape = ops.shape(logits_per_image)
        txt_shape = ops.shape(logits_per_text)

        # Both should be 2D tensors
        if len(img_shape) != 2 or len(txt_shape) != 2:
            raise ValueError(
                f"Logits must be 2D tensors, got shapes: "
                f"logits_per_image={img_shape}, logits_per_text={txt_shape}"
            )

        # Check top_k feasibility
        if self.top_k > img_shape[1] or self.top_k > txt_shape[1]:
            raise ValueError(
                f"top_k ({self.top_k}) cannot be larger than number of candidates. "
                f"Got shapes: logits_per_image={img_shape}, logits_per_text={txt_shape}"
            )

    def result(self) -> keras.KerasTensor:
        """
        Compute and return the final accuracy result.

        Returns:
            Scalar tensor with accuracy value
        """
        if not self.average_directions and self.track_directions:
            # Return tuple of (i2t_accuracy, t2i_accuracy)
            i2t_acc = self.i2t_total / self.i2t_count
            t2i_acc = self.t2i_total / self.t2i_count
            return ops.stack([i2t_acc, t2i_acc])
        else:
            # Return averaged accuracy
            return self.total / self.count

    def reset_state(self) -> None:
        """Reset all metric state variables."""
        self.total.assign(0.0)
        self.count.assign(0.0)

        if self.track_directions:
            self.i2t_total.assign(0.0)
            self.t2i_total.assign(0.0)
            self.i2t_count.assign(0.0)
            self.t2i_count.assign(0.0)

    @property
    def i2t_accuracy(self) -> Optional[keras.KerasTensor]:
        """Get image-to-text accuracy (only available when track_directions=True)."""
        if not self.track_directions:
            logger.warning("i2t_accuracy not available. Set track_directions=True during initialization.")
            return None
        return self.i2t_total / self.i2t_count

    @property
    def t2i_accuracy(self) -> Optional[keras.KerasTensor]:
        """Get text-to-image accuracy (only available when track_directions=True)."""
        if not self.track_directions:
            logger.warning("t2i_accuracy not available. Set track_directions=True during initialization.")
            return None
        return self.t2i_total / self.t2i_count

    @property
    def sample_count(self) -> keras.KerasTensor:
        """Get total number of samples processed."""
        return self.count

    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration for serialization."""
        config = super().get_config()
        config.update({
            "top_k": self.top_k,
            "track_directions": self.track_directions,
            "average_directions": self.average_directions,
        })
        return config

# ---------------------------------------------------------------------
