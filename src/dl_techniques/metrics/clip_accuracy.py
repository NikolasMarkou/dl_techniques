"""
CLIP Contrastive Learning Metrics Implementation

This module implements evaluation metrics for CLIP (Contrastive Language-Image Pre-training)
models, measuring retrieval performance in multimodal image-text alignment tasks. The metrics
quantify how effectively the model learns a shared embedding space where semantically related
images and texts are positioned close together, enabling zero-shot transfer and cross-modal
retrieval capabilities.

Mathematical Formulation:
    CLIP metrics operate on similarity matrices from dual-encoder architectures:

    Given N image-text pairs, encoders produce embeddings I ∈ ℝ^(N×D) and T ∈ ℝ^(N×D).
    After L2 normalization, the similarity matrix is S = I @ T^T ∈ ℝ^(N×N), where:
    - S[i,j] = cosine_similarity(image_i, text_j)
    - Diagonal S[i,i] represents correct image-text pairs
    - Off-diagonal elements represent N²-N negative pairs

    Accuracy Metric:
        For image-to-text (I2T): acc = (1/N) Σ 1[argmax_j(S[i,:]) == i]
        For text-to-image (T2I): acc = (1/N) Σ 1[argmax_i(S[:,j]) == j]

        Measures exact top-1 retrieval performance (percentage of queries where
        the correct match has the highest similarity score).

    Recall@K Metric:
        For I2T: R@K = (1/N) Σ 1[i ∈ top_k_j(S[i,:])]
        For T2I: R@K = (1/N) Σ 1[j ∈ top_k_i(S[:,j])]

        Measures whether the correct match appears in the top-K most similar
        candidates, providing a more lenient evaluation for practical retrieval.

Evaluation Variants:
    - Accuracy (Top-1): Strict exact match requirement
    - Recall@K (K=1,5,10,50): Practical retrieval evaluation at different cutoffs
    - Bidirectional: Separate I2T and T2I performance tracking
    - Symmetric: Average of both directions for overall performance
    - Per-batch: Cumulative tracking across epochs for training monitoring

Key Benefits:
    - Direct measure of multimodal alignment quality in shared embedding space
    - Interpretable metrics for zero-shot retrieval and transfer learning
    - Enables diagnosis of directional biases (I2T vs T2I performance gaps)
    - Standard benchmarking metrics for vision-language models
    - Efficient computation using vectorized operations on similarity matrices
    - Compatible with contrastive learning objectives (CLIP, ALIGN, BLIP)

Implementation Features:
    - Keras 3 native implementation with proper serialization support
    - Stateful accumulation for epoch-level metric tracking
    - Support for variable batch sizes with automatic K capping
    - Numerically stable computations with epsilon for division safety
    - Dictionary-based input format matching CLIP model outputs
    - Comprehensive validation with actionable error messages
    - Zero memory overhead (computes on-the-fly from logits)
    - TensorFlow graph compatibility via keras.ops API

Architecture Compatibility:
    - CLIP (OpenAI): Dual ViT/ResNet + Transformer encoders
    - ALIGN (Google): EfficientNet + BERT encoders
    - BLIP (Salesforce): ViT + BERT with multimodal mixing
    - CoCa (Google): Contrastive captioner architecture
    - Any dual-encoder model producing similarity matrices

References:
    - Radford, A., et al. (2021). "Learning Transferable Visual Models From
      Natural Language Supervision." International Conference on Machine Learning.
      https://arxiv.org/abs/2103.00020
      [Original CLIP paper introducing contrastive image-text learning]

    - Jia, C., et al. (2021). "Scaling Up Visual and Vision-Language Representation
      Learning With Noisy Text Supervision." International Conference on Machine Learning.
      https://arxiv.org/abs/2102.05918
      [ALIGN model demonstrating scale and noisy data benefits]

    - Li, J., et al. (2022). "BLIP: Bootstrapping Language-Image Pre-training for
      Unified Vision-Language Understanding and Generation." International Conference
      on Machine Learning. https://arxiv.org/abs/2201.12086
      [BLIP model with improved pre-training and captioning]

    - Yu, J., et al. (2022). "CoCa: Contrastive Captioners are Image-Text Foundation
      Models." Transactions on Machine Learning Research.
      https://arxiv.org/abs/2205.01917
      [CoCa combining contrastive and captioning objectives]

Usage Examples:
    Basic accuracy tracking:
    >>> import keras
    >>> from clip_metrics_refined import CLIPAccuracy
    >>>
    >>> # Create metric
    >>> accuracy = CLIPAccuracy(direction='i2t', name='i2t_accuracy')
    >>>
    >>> # Simulate CLIP outputs
    >>> batch_size = 32
    >>> logits_i2t = keras.random.normal((batch_size, batch_size))
    >>> logits_t2i = keras.ops.transpose(logits_i2t)
    >>>
    >>> outputs = {
    ...     'logits_per_image': logits_i2t,
    ...     'logits_per_text': logits_t2i
    ... }
    >>>
    >>> # Update and retrieve
    >>> accuracy.update_state(y_pred=outputs)
    >>> print(f"I2T Accuracy: {float(accuracy.result()):.3f}")
    >>> accuracy.reset_state()  # Reset for next epoch

    Recall@K evaluation:
    >>> from clip_metrics_refined import CLIPRecallAtK
    >>>
    >>> # Create recall@5 and recall@10 metrics
    >>> recall5 = CLIPRecallAtK(k=5, direction='i2t', name='i2t_r@5')
    >>> recall10 = CLIPRecallAtK(k=10, direction='i2t', name='i2t_r@10')
    >>>
    >>> # Evaluate on batch
    >>> recall5.update_state(y_pred=outputs)
    >>> recall10.update_state(y_pred=outputs)
    >>>
    >>> print(f"Recall@5: {float(recall5.result()):.3f}")
    >>> print(f"Recall@10: {float(recall10.result()):.3f}")

    Training integration:
    >>> from clip_metrics_refined import CLIPAccuracy, CLIPRecallAtK
    >>>
    >>> # Create comprehensive metric set
    >>> metrics = [
    ...     CLIPAccuracy(direction='i2t', name='i2t_acc'),
    ...     CLIPAccuracy(direction='t2i', name='t2i_acc'),
    ...     CLIPRecallAtK(k=5, direction='i2t', name='i2t_r@5'),
    ...     CLIPRecallAtK(k=10, direction='i2t', name='i2t_r@10'),
    ...     CLIPRecallAtK(k=5, direction='t2i', name='t2i_r@5'),
    ...     CLIPRecallAtK(k=10, direction='t2i', name='t2i_r@10'),
    ... ]
    >>>
    >>> # Use in custom training loop
    >>> for batch in dataset:
    ...     outputs = model(batch, training=True)
    ...     for metric in metrics:
    ...         metric.update_state(y_pred=outputs)
    >>>
    >>> # Log epoch metrics
    >>> for metric in metrics:
    ...     print(f"{metric.name}: {float(metric.result()):.4f}")
    ...     metric.reset_state()

    Model serialization:
    >>> # Metrics are fully serializable with Keras 3
    >>> model.compile(
    ...     optimizer='adamw',
    ...     loss=clip_contrastive_loss,
    ...     metrics=[
    ...         CLIPAccuracy(direction='i2t'),
    ...         CLIPRecallAtK(k=5, direction='i2t')
    ...     ]
    ... )
    >>>
    >>> # Save and load model with metrics
    >>> model.save('clip_model.keras')
    >>> loaded_model = keras.models.load_model('clip_model.keras')
    >>> # Metrics are preserved and continue tracking correctly

Performance Considerations:
    - Metrics computed in O(N²) time where N is batch size
    - Larger batches provide more negative samples but increase compute
    - Use batch_size >= 32 for stable metric estimates
    - Recall@K converges faster than accuracy for small K
    - I2T and T2I should be similar; large gaps indicate training issues
    - Consider tracking multiple K values (1, 5, 10) for comprehensive evaluation

Troubleshooting:
    - Low accuracy (<10%): Check data quality, verify text tokenization
    - I2T >> T2I or vice versa: Indicates encoder imbalance, adjust learning rates
    - Recall@5 < 0.5: Increase training time, larger batch size, or check loss
    - NaN values: Check for gradient explosion, reduce learning rate
    - No improvement: Verify similarity scaling (temperature parameter in loss)
"""

from typing import Optional, Dict, Any
import keras
from keras import ops


@keras.saving.register_keras_serializable(package="CLIPMetrics")
class CLIPAccuracy(keras.metrics.Metric):
    """
    Accuracy metric for CLIP contrastive learning.

    Measures the percentage of correct matches in the similarity matrix.
    For a batch of N pairs, accuracy is the percentage of samples where
    the highest similarity score corresponds to the correct pair (diagonal).

    **Intent**: Evaluate CLIP's ability to correctly match images with their
    corresponding texts (or vice versa) by measuring top-1 retrieval accuracy.

    **Mathematical Definition**:
    Given a batch of N (image, text) pairs, the model produces an N×N
    similarity matrix S where S[i,j] = similarity(image_i, text_j).

    For image-to-text (I2T):
        prediction_i = argmax_j(S[i,j])
        accuracy = (1/N) * Σ 1[prediction_i == i]

    For text-to-image (T2I):
        prediction_j = argmax_i(S[i,j])
        accuracy = (1/N) * Σ 1[prediction_j == j]

    Args:
        direction: String, either 'i2t' (image-to-text) or 't2i' (text-to-image).
            Determines which similarity matrix to use for evaluation.
        name: String name for the metric instance. If None, defaults to
            'clip_{direction}_accuracy'.
        dtype: Data type for metric computations. Defaults to None (uses default).
        **kwargs: Additional keyword arguments passed to parent Metric class.

    Attributes:
        direction: Stored direction parameter.
        correct: Weight tracking number of correct predictions (cumulative).
        total: Weight tracking total number of predictions (cumulative).

    Input Format (update_state):
        y_true: Not used (can be None). Ground truth is implicit from batch structure.
        y_pred: Dictionary with keys:
            - 'logits_per_image': Tensor of shape (batch_size, batch_size)
              representing I2T similarities
            - 'logits_per_text': Tensor of shape (batch_size, batch_size)
              representing T2I similarities
        sample_weight: Optional sample weights (not currently used).

    Output:
        Scalar tensor representing accuracy as a value in [0, 1].

    Example:
        ```python
        import keras
        from keras import ops

        # Create metric
        metric = CLIPAccuracy(direction='i2t', name='i2t_accuracy')

        # Simulate CLIP model output
        batch_size = 4
        logits_per_image = keras.random.normal((batch_size, batch_size))
        logits_per_text = ops.transpose(logits_per_image)

        outputs = {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }

        # Update metric state
        metric.update_state(outputs)

        # Get result
        accuracy = metric.result()
        print(f"I2T Accuracy: {float(accuracy):.4f}")

        # Reset for next epoch
        metric.reset_state()
        ```

    References:
        Radford, A., et al. (2021). Learning Transferable Visual Models From
        Natural Language Supervision. ICML.
    """

    def __init__(
        self,
        direction: str = 'i2t',
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize CLIP accuracy metric.

        Following Keras 3 best practices: store all parameters as instance
        attributes before calling super().__init__().
        """
        # Validate direction parameter
        if direction not in ['i2t', 't2i']:
            raise ValueError(
                f"direction must be 'i2t' or 't2i', got '{direction}'. "
                "Use 'i2t' for image-to-text accuracy or 't2i' for text-to-image."
            )

        # Store all configuration parameters as instance attributes
        # This MUST happen before super().__init__() for proper serialization
        self.direction = direction

        # Set default name if not provided
        if name is None:
            name = f'clip_{direction}_accuracy'

        # Call parent constructor
        super().__init__(name=name, dtype=dtype, **kwargs)

        # Create metric state variables
        # These track cumulative correct predictions and total predictions
        self.correct = self.add_weight(
            name='correct',
            shape=(),
            initializer='zeros',
            dtype='float32'
        )
        self.total = self.add_weight(
            name='total',
            shape=(),
            initializer='zeros',
            dtype='float32'
        )

    def update_state(
        self,
        y_true: Optional[Any] = None,
        y_pred: Optional[Dict[str, keras.KerasTensor]] = None,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """
        Update metric state with batch predictions.

        For Keras metrics, the signature should accept y_true and y_pred.
        However, for CLIP, y_true is not used as ground truth is implicit
        (diagonal of similarity matrix).

        Args:
            y_true: Ignored. Ground truth is implicit from batch structure
                (diagonal elements of similarity matrix).
            y_pred: Dictionary containing model outputs with keys:
                - 'logits_per_image': Image-to-text similarity logits (N, N)
                - 'logits_per_text': Text-to-image similarity logits (N, N)
            sample_weight: Optional sample weights. Not currently used.

        Raises:
            ValueError: If y_pred is not a dictionary or missing required keys.
            ValueError: If logit tensors have incompatible shapes.
        """
        # Validate y_pred format
        if y_pred is None:
            raise ValueError("y_pred cannot be None")

        if not isinstance(y_pred, dict):
            raise ValueError(
                f"y_pred must be a dictionary, got {type(y_pred).__name__}"
            )

        # Get appropriate logits based on direction
        if self.direction == 'i2t':
            if 'logits_per_image' not in y_pred:
                raise ValueError(
                    "y_pred must contain 'logits_per_image' key for I2T accuracy"
                )
            logits = y_pred['logits_per_image']
        else:  # 't2i'
            if 'logits_per_text' not in y_pred:
                raise ValueError(
                    "y_pred must contain 'logits_per_text' key for T2I accuracy"
                )
            logits = y_pred['logits_per_text']

        # Validate logits shape (should be square matrix)
        logits_shape = ops.shape(logits)
        if len(logits.shape) != 2:
            raise ValueError(
                f"logits must be 2D tensor, got shape {logits.shape}"
            )

        # Get batch size
        batch_size = logits_shape[0]

        # Predictions: argmax along the last axis gives predicted match index
        # For each query (row), find which candidate (column) has highest similarity
        predictions = ops.argmax(logits, axis=-1)  # Shape: (batch_size,)

        # Ground truth labels: diagonal indices [0, 1, 2, ..., batch_size-1]
        # This means query_i should match with candidate_i
        labels = ops.arange(batch_size, dtype='int32')

        # Count correct predictions (where prediction matches ground truth)
        correct = ops.sum(ops.cast(ops.equal(predictions, labels), 'float32'))

        # Update cumulative state
        self.correct.assign_add(correct)
        self.total.assign_add(ops.cast(batch_size, 'float32'))

    def result(self) -> keras.KerasTensor:
        """
        Compute current accuracy value.

        Returns accuracy as the ratio of correct predictions to total predictions.
        Uses maximum operation to avoid division by zero.

        Returns:
            Scalar tensor with accuracy value in [0, 1].
        """
        return self.correct / ops.maximum(self.total, 1e-7)

    def reset_state(self) -> None:
        """
        Reset metric state for new epoch or evaluation.

        Sets both correct and total counters back to zero.
        This should be called at the start of each epoch during training.
        """
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        Following Keras 3 best practices: include ALL constructor parameters
        that are needed to recreate this metric instance.

        Returns:
            Dictionary containing all configuration parameters including
            'direction' and parent class configuration.
        """
        # Get parent class configuration
        config = super().get_config()

        # Add custom parameters
        config.update({
            'direction': self.direction,
        })

        return config


@keras.saving.register_keras_serializable(package="CLIPMetrics")
class CLIPRecallAtK(keras.metrics.Metric):
    """
    Recall@K metric for CLIP contrastive learning.

    Measures the percentage of samples where the correct match appears in the
    top-K predictions. This is a more lenient metric than accuracy (which only
    considers top-1) and is commonly used for retrieval tasks.

    **Intent**: Evaluate CLIP's retrieval performance by measuring whether the
    correct match appears within the top-K most similar candidates. This is
    more practical for real-world retrieval scenarios where users examine
    multiple results.

    **Mathematical Definition**:
    Given a batch of N (image, text) pairs and similarity matrix S:

    For image-to-text (I2T):
        top_k_i = top_k(S[i,:])  # K highest similarities for image i
        recall@k = (1/N) * Σ 1[i ∈ top_k_i]

    For text-to-image (T2I):
        top_k_j = top_k(S[:,j])  # K highest similarities for text j
        recall@k = (1/N) * Σ 1[j ∈ top_k_j]

    Args:
        k: Integer, number of top predictions to consider. Must be positive.
            Common values are 1, 5, 10, or 50 depending on application.
        direction: String, either 'i2t' (image-to-text) or 't2i' (text-to-image).
            Determines which similarity matrix to use for evaluation.
        name: String name for the metric instance. If None, defaults to
            'clip_{direction}_recall@{k}'.
        dtype: Data type for metric computations. Defaults to None (uses default).
        **kwargs: Additional keyword arguments passed to parent Metric class.

    Attributes:
        k: Stored k parameter (number of top predictions to consider).
        direction: Stored direction parameter.
        correct: Weight tracking number of correct retrievals (cumulative).
        total: Weight tracking total number of retrievals (cumulative).

    Input Format (update_state):
        y_true: Not used (can be None). Ground truth is implicit from batch structure.
        y_pred: Dictionary with keys:
            - 'logits_per_image': Tensor of shape (batch_size, batch_size)
              representing I2T similarities
            - 'logits_per_text': Tensor of shape (batch_size, batch_size)
              representing T2I similarities
        sample_weight: Optional sample weights (not currently used).

    Output:
        Scalar tensor representing recall@k as a value in [0, 1].

    Example:
        ```python
        import keras
        from keras import ops

        # Create metric for recall@5
        metric = CLIPRecallAtK(k=5, direction='i2t', name='i2t_recall@5')

        # Simulate CLIP model output
        batch_size = 8
        logits_per_image = keras.random.normal((batch_size, batch_size))
        logits_per_text = ops.transpose(logits_per_image)

        outputs = {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }

        # Update metric state
        metric.update_state(outputs)

        # Get result
        recall = metric.result()
        print(f"I2T Recall@5: {float(recall):.4f}")

        # Reset for next epoch
        metric.reset_state()
        ```

    Notes:
        - Recall@K is always >= Accuracy (since K >= 1)
        - As K increases, Recall@K approaches 1.0
        - Common benchmarks use K=1, 5, 10 for evaluation
        - For small batches, K is automatically capped at batch_size

    References:
        Radford, A., et al. (2021). Learning Transferable Visual Models From
        Natural Language Supervision. ICML.
    """

    def __init__(
        self,
        k: int = 5,
        direction: str = 'i2t',
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize CLIP recall@k metric.

        Following Keras 3 best practices: store all parameters as instance
        attributes before calling super().__init__().
        """
        # Validate parameters
        if k <= 0:
            raise ValueError(
                f"k must be positive integer, got {k}. "
                "Common values are 1, 5, 10, or 50."
            )

        if direction not in ['i2t', 't2i']:
            raise ValueError(
                f"direction must be 'i2t' or 't2i', got '{direction}'. "
                "Use 'i2t' for image-to-text recall or 't2i' for text-to-image."
            )

        # Store all configuration parameters as instance attributes
        # This MUST happen before super().__init__() for proper serialization
        self.k = k
        self.direction = direction

        # Set default name if not provided
        if name is None:
            name = f'clip_{direction}_recall@{k}'

        # Call parent constructor
        super().__init__(name=name, dtype=dtype, **kwargs)

        # Create metric state variables
        # These track cumulative correct retrievals and total retrievals
        self.correct = self.add_weight(
            name='correct',
            shape=(),
            initializer='zeros',
            dtype='float32'
        )
        self.total = self.add_weight(
            name='total',
            shape=(),
            initializer='zeros',
            dtype='float32'
        )

    def update_state(
        self,
        y_true: Optional[Any] = None,
        y_pred: Optional[Dict[str, keras.KerasTensor]] = None,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """
        Update metric state with batch predictions.

        For Keras metrics, the signature should accept y_true and y_pred.
        However, for CLIP, y_true is not used as ground truth is implicit
        (diagonal of similarity matrix).

        Args:
            y_true: Ignored. Ground truth is implicit from batch structure
                (diagonal elements of similarity matrix).
            y_pred: Dictionary containing model outputs with keys:
                - 'logits_per_image': Image-to-text similarity logits (N, N)
                - 'logits_per_text': Text-to-image similarity logits (N, N)
            sample_weight: Optional sample weights. Not currently used.

        Raises:
            ValueError: If y_pred is not a dictionary or missing required keys.
            ValueError: If logit tensors have incompatible shapes.
        """
        # Validate y_pred format
        if y_pred is None:
            raise ValueError("y_pred cannot be None")

        if not isinstance(y_pred, dict):
            raise ValueError(
                f"y_pred must be a dictionary, got {type(y_pred).__name__}"
            )

        # Get appropriate logits based on direction
        if self.direction == 'i2t':
            if 'logits_per_image' not in y_pred:
                raise ValueError(
                    "y_pred must contain 'logits_per_image' key for I2T recall@k"
                )
            logits = y_pred['logits_per_image']
        else:  # 't2i'
            if 'logits_per_text' not in y_pred:
                raise ValueError(
                    "y_pred must contain 'logits_per_text' key for T2I recall@k"
                )
            logits = y_pred['logits_per_text']

        # Validate logits shape (should be square matrix)
        logits_shape = ops.shape(logits)
        if len(logits.shape) != 2:
            raise ValueError(
                f"logits must be 2D tensor, got shape {logits.shape}"
            )

        # Get batch size
        batch_size = logits_shape[0]

        # Get top-k predictions for each sample
        # ops.top_k returns the k largest values, which is what we want
        # (highest similarities = best matches)
        # Cap k at batch_size to avoid errors when k > batch_size
        k_effective = ops.minimum(self.k, batch_size)

        # Get indices of top-k predictions (k highest similarities)
        # Shape: (batch_size, k_effective)
        top_k_indices = ops.top_k(logits, k=k_effective)[1]

        # Ground truth labels: diagonal indices [0, 1, 2, ..., batch_size-1]
        # Shape: (batch_size,)
        labels = ops.arange(batch_size, dtype='int32')

        # Expand labels for broadcasting comparison
        # Shape: (batch_size, 1)
        labels_expanded = ops.expand_dims(labels, axis=-1)

        # Check if correct label is in top-k predictions
        # Broadcasting: (batch_size, 1) == (batch_size, k_effective)
        # Result shape: (batch_size, k_effective)
        matches = ops.equal(labels_expanded, top_k_indices)

        # Any match in top-k counts as correct
        # Shape: (batch_size,) -> scalar after sum
        correct = ops.sum(ops.cast(ops.any(matches, axis=-1), 'float32'))

        # Update cumulative state
        self.correct.assign_add(correct)
        self.total.assign_add(ops.cast(batch_size, 'float32'))

    def result(self) -> keras.KerasTensor:
        """
        Compute current recall@k value.

        Returns recall@k as the ratio of correct retrievals to total retrievals.
        Uses maximum operation to avoid division by zero.

        Returns:
            Scalar tensor with recall@k value in [0, 1].
        """
        return self.correct / ops.maximum(self.total, 1e-7)

    def reset_state(self) -> None:
        """
        Reset metric state for new epoch or evaluation.

        Sets both correct and total counters back to zero.
        This should be called at the start of each epoch during training.
        """
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        Following Keras 3 best practices: include ALL constructor parameters
        that are needed to recreate this metric instance.

        Returns:
            Dictionary containing all configuration parameters including
            'k', 'direction', and parent class configuration.
        """
        # Get parent class configuration
        config = super().get_config()

        # Add custom parameters
        config.update({
            'k': self.k,
            'direction': self.direction,
        })

        return config