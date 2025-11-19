"""
Symmetric Contrastive Loss for Multimodal Training (CLIP).

This module implements the core training objective from the CLIP (Contrastive
Language-Image Pre-training) paper, designed to learn a joint multimodal embedding
space where semantically similar image and text vectors are located close together.

The loss performs contrastive learning on batches of (image, text) pairs, creating
N correct pairings (positive samples) and N×(N-1) incorrect pairings (negative samples)
for the model to contrast against.
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Union, Tuple
import warnings

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CLIPContrastiveLoss(keras.losses.Loss):
    """
    CLIP Symmetric Contrastive Loss with configurable temperature and smoothing.

    This loss function implements the bidirectional contrastive learning objective
    used in CLIP, encouraging matching image-text pairs to have high similarity while
    pushing non-matching pairs apart. The loss is computed symmetrically in both
    directions (image→text and text→image) to ensure balanced multimodal alignment.

    **Intent**: Learn a joint embedding space for vision and language where semantically
    related image-text pairs are positioned close together through contrastive learning.
    This enables zero-shot transfer, retrieval, and other cross-modal tasks.

    **Architecture & Loss Flow**:
    ```
    Input: Batch of N (image, text) pairs
    =====================================

    Image Embeddings [N×D]          Text Embeddings [N×D]
           ↓                                 ↓
    ┌──────────────────────────────────────────────────┐
    │         Compute Similarity Matrix                │
    │         S = I @ T^T  [N×N]                       │
    │                                                  │
    │  Each S[i,j] = similarity(image_i, text_j)       │
    │  Diagonal S[i,i] = positive pairs                │
    │  Off-diagonal = negative pairs                   │
    └──────────────────────────────────────────────────┘
           ↓
    Optional Temperature Scaling: S' = S / τ
           ↓
    ┌─────────────────────┬─────────────────────┐
    │  logits_per_image   │  logits_per_text    │
    │  [N×N]              │  [N×N]              │
    │  (I→T direction)    │  (T→I direction)    │
    └─────────────────────┴─────────────────────┘
           ↓                        ↓
    ┌─────────────────────┐  ┌─────────────────────┐
    │ Ground Truth Labels │  │ Ground Truth Labels │
    │ [0, 1, 2, ..., N-1] │  │ [0, 1, 2, ..., N-1] │
    │ (Diagonal matches)  │  │ (Diagonal matches)  │
    └─────────────────────┘  └─────────────────────┘
           ↓                        ↓
    ┌─────────────────────┐  ┌─────────────────────┐
    │ CrossEntropy Loss   │  │ CrossEntropy Loss   │
    │ with label_smoothing│  │ with label_smoothing│
    └─────────────────────┘  └─────────────────────┘
           ↓                        ↓
       loss_i2t [N]            loss_t2i [N]
           ↓                        ↓
    ┌─────────────────────────────────────────┐
    │  Weighted Combination (per-sample)      │
    │  L = w_i2t × loss_i2t + w_t2i × loss_t2i│
    └─────────────────────────────────────────┘
           ↓
    Per-Sample Loss [N]
           ↓
    Parent Loss.reduction → Final Scalar Loss
    ```

    **Conceptual Overview**:

    For a batch of N pairs, the model must correctly predict which of the N texts
    corresponds to each of the N images, and vice-versa. This creates:
    - N correct pairings (positive samples, diagonal of similarity matrix)
    - N × (N-1) incorrect pairings (negative samples, off-diagonal elements)

    By maximizing similarity of positive pairs while minimizing similarity of
    negative pairs, the model learns robust multimodal representations.

    **Mathematical Formulation**:

    1. **Similarity Matrix**:
       Given N normalized image embeddings I ∈ ℝ^(N×D) and N normalized text
       embeddings T ∈ ℝ^(N×D), compute similarity matrix:

       S = I @ T^T  ∈ ℝ^(N×N)

       where S[i,j] represents the similarity between image_i and text_j

    2. **Temperature Scaling** (optional):
       S' = S / τ

       Temperature τ controls the sharpness of the softmax distribution.
       Lower τ → sharper distribution → higher penalty for hard negatives

    3. **Cross-Entropy in Both Directions**:
       For ground truth labels y = [0, 1, 2, ..., N-1] (diagonal matches):

       L_i2t = CrossEntropy(y, S'_image→text)  [N]
       L_t2i = CrossEntropy(y, S'_text→image)  [N]

    4. **Symmetric Weighted Loss**:
       L_total = w_i2t × L_i2t + w_t2i × L_t2i  [N]

       Default: w_i2t = w_t2i = 0.5 for equal weighting

    5. **Final Reduction**:
       The parent Loss class applies reduction (typically mean) to get scalar

    **Why Symmetric?**:

    Bidirectional training prevents mode collapse and ensures the embedding space
    is well-aligned from both modalities. Single-direction training can lead to
    one modality dominating the embedding space.

    Args:
        temperature: Temperature parameter τ for scaling logits. Lower values
            create sharper distributions, increasing penalties for hard negatives.
            Only applied if `apply_temperature=True`. Must be positive.
            Defaults to 0.07 (CLIP paper default).
        label_smoothing: Label smoothing factor ∈ [0, 1]. Softens hard targets
            by distributing some probability mass to negative classes. Higher values
            provide more regularization. 0.0 = no smoothing (hard targets).
            Defaults to 0.0.
        apply_temperature: Whether to apply temperature scaling to input logits.
            Set to `True` if logits haven't been temperature-scaled in the model.
            Set to `False` if the model already applies temperature scaling.
            Defaults to False.
        loss_weight_i2t: Weight for image→text loss component. Must be non-negative.
            Defaults to 0.5 for equal weighting with text→image.
        loss_weight_t2i: Weight for text→image loss component. Must be non-negative.
            Defaults to 0.5 for equal weighting with image→text.
        name: String name for the loss instance. Defaults to "clip_contrastive_loss".
        **kwargs: Additional keyword arguments passed to parent Loss class
            (e.g., reduction strategy, dtype).

    Input Format:
        y_true: Not used (can be None). Contrastive loss is self-supervised;
            ground truth is derived from batch structure (diagonal correspondence).
            However, Keras Loss API requires a tensor input for y_true. Pass a
            dummy tensor of shape (batch_size,) or similar.

        y_pred: Predictions in one of the following formats:

            1. **Dictionary** (recommended):
               {
                   'logits_per_image': Tensor [N, N],  # Image→Text similarities
                   'logits_per_text': Tensor [N, N]    # Text→Image similarities
               }

            2. **Tuple/List**:
               (logits_per_image, logits_per_text)
               Both tensors must be [N, N]

        Where:
        - logits_per_image[i, j] = similarity(image_i, text_j)
        - logits_per_text[i, j] = similarity(text_i, image_j)
        - N = batch_size
        - Logits should be unnormalized (from_logits=True is used internally)

    Output:
        Per-sample loss tensor of shape [batch_size] or scalar after reduction.
        The parent Loss class handles the final reduction based on its reduction
        strategy (typically 'sum_over_batch_size' → mean).

    Shape:
        - Input logits_per_image: (batch_size, batch_size)
        - Input logits_per_text: (batch_size, batch_size)
        - Output (before parent reduction): (batch_size,)
        - Output (after parent reduction): Scalar

    Attributes:
        temperature: Stored temperature parameter value.
        label_smoothing: Stored label smoothing factor.
        apply_temperature: Whether temperature scaling is enabled.
        loss_weight_i2t: Weight for image→text loss component.
        loss_weight_t2i: Weight for text→image loss component.

    Raises:
        ValueError: If temperature ≤ 0, label_smoothing ∉ [0, 1], loss weights < 0,
            y_pred format is invalid, required keys are missing, or logit shapes
            are incompatible.

    Examples:
        **Basic Usage**:
        ```python
        import keras
        from keras import ops
        import numpy as np

        # Initialize loss with default parameters
        loss_fn = CLIPContrastiveLoss()

        # Simulate similarity logits for batch_size=4
        batch_size = 4
        logits_per_image = keras.random.normal((batch_size, batch_size))
        logits_per_text = ops.transpose(logits_per_image)

        # Create predictions dictionary
        predictions = {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }

        # Compute loss (pass dummy y_true as Keras requires a tensor)
        dummy_y_true = ops.zeros((batch_size,))
        loss_value = loss_fn(y_true=dummy_y_true, y_pred=predictions)
        print(f"CLIP Loss: {loss_value:.4f}")
        ```

        **With Temperature Scaling**:
        ```python
        # If model doesn't apply temperature, enable it in loss
        loss_fn = CLIPContrastiveLoss(
            temperature=0.07,
            apply_temperature=True
        )
        ```

        **With Label Smoothing**:
        ```python
        # Add regularization via label smoothing
        loss_fn = CLIPContrastiveLoss(
            label_smoothing=0.1  # 10% smoothing
        )
        ```

        **Custom Direction Weights**:
        ```python
        # Weight image→text more heavily
        loss_fn = CLIPContrastiveLoss(
            loss_weight_i2t=0.7,
            loss_weight_t2i=0.3
        )
        ```

        **In Model Training**:
        ```python
        # Compile model with CLIP loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=CLIPContrastiveLoss(temperature=0.07, apply_temperature=True)
        )

        # Train model
        model.fit(train_dataset, epochs=10)
        ```

        **Dynamic Temperature Updates**:
        ```python
        # Create loss with initial temperature
        loss_fn = CLIPContrastiveLoss(temperature=0.1, apply_temperature=True)

        # Update temperature during training
        loss_fn.update_temperature(0.05)
        ```

    Notes:
        - **Temperature Scaling**: The original CLIP paper uses τ=0.07. Temperature
          can be learned during training or kept fixed. Lower temperatures improve
          alignment but may cause training instability.

        - **Batch Size Impact**: Larger batches provide more negative samples,
          improving contrastive learning. CLIP was trained with very large batches
          (32K). Consider gradient accumulation for smaller GPU memory.

        - **Label Smoothing**: Acts as regularization, preventing overconfidence.
          Useful when training with smaller batches or noisy data.

        - **Symmetry**: Equal weights (0.5, 0.5) ensure balanced training. Unequal
          weights can be used if one modality is more reliable or important.

        - **Numerical Stability**: Uses categorical cross-entropy with
          from_logits=True for numerical stability. No manual softmax needed.

    References:
        - Radford et al. "Learning Transferable Visual Models From Natural Language
          Supervision" (CLIP), ICML 2021. https://arxiv.org/abs/2103.00020
        - Original CLIP implementation: https://github.com/openai/CLIP
    """

    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        apply_temperature: bool = False,
        loss_weight_i2t: float = 0.5,
        loss_weight_t2i: float = 0.5,
        name: str = "clip_contrastive_loss",
        **kwargs: Any
    ) -> None:
        """
        Initialize CLIP Contrastive Loss with validation.

        All parameters are stored as instance attributes for serialization.
        Comprehensive validation ensures parameter correctness before training.

        Args:
            temperature: Temperature scaling parameter (must be positive).
            label_smoothing: Label smoothing factor in [0, 1].
            apply_temperature: Whether to apply temperature to input logits.
            loss_weight_i2t: Weight for image→text loss (non-negative).
            loss_weight_t2i: Weight for text→image loss (non-negative).
            name: Name for this loss instance.
            **kwargs: Additional arguments for parent Loss class.

        Raises:
            ValueError: If any parameter violates its constraints.
        """
        super().__init__(name=name, **kwargs)

        # Validate all inputs before storing
        self._validate_inputs(temperature, label_smoothing, loss_weight_i2t, loss_weight_t2i)

        # Store all configuration parameters for serialization
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.apply_temperature = apply_temperature
        self.loss_weight_i2t = loss_weight_i2t
        self.loss_weight_t2i = loss_weight_t2i

        # Log initialization for debugging and monitoring
        logger.info(
            f"CLIPContrastiveLoss initialized: temperature={temperature}, "
            f"label_smoothing={label_smoothing}, apply_temperature={apply_temperature}, "
            f"loss_weight_i2t={loss_weight_i2t}, loss_weight_t2i={loss_weight_t2i}"
        )

    def _validate_inputs(
        self,
        temperature: float,
        label_smoothing: float,
        loss_weight_i2t: float,
        loss_weight_t2i: float
    ) -> None:
        """
        Validate initialization parameters for correctness.

        Ensures all parameters meet their mathematical and practical constraints
        before the loss is used in training. Provides clear error messages for
        invalid configurations.

        Args:
            temperature: Temperature parameter to validate.
            label_smoothing: Label smoothing factor to validate.
            loss_weight_i2t: Image→text loss weight to validate.
            loss_weight_t2i: Text→image loss weight to validate.

        Raises:
            ValueError: If any parameter violates its constraints with detailed
                explanation of the violation.

        Validation Rules:
            - temperature must be positive (> 0)
            - label_smoothing must be in [0, 1]
            - loss_weight_i2t must be non-negative (≥ 0)
            - loss_weight_t2i must be non-negative (≥ 0)
            - Warning if weights don't sum to 1.0 (affects loss magnitude)
        """
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive, got {temperature}. "
                "Temperature controls the sharpness of the softmax distribution; "
                "values close to 0 create very sharp distributions."
            )

        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1], got {label_smoothing}. "
                "0.0 = no smoothing (hard targets), 1.0 = uniform distribution."
            )

        if loss_weight_i2t < 0.0:
            raise ValueError(
                f"loss_weight_i2t must be non-negative, got {loss_weight_i2t}. "
                "Negative weights are not supported."
            )

        if loss_weight_t2i < 0.0:
            raise ValueError(
                f"loss_weight_t2i must be non-negative, got {loss_weight_t2i}. "
                "Negative weights are not supported."
            )

        # Warning for non-normalized weights (common configuration issue)
        weight_sum = loss_weight_i2t + loss_weight_t2i
        if abs(weight_sum - 1.0) > 1e-6:
            msg = (
                f"Loss weights sum to {weight_sum}, not 1.0. "
                "This may affect the loss magnitude and learning rate tuning. "
                f"Consider normalizing: w_i2t={loss_weight_i2t/weight_sum:.3f}, "
                f"w_t2i={loss_weight_t2i/weight_sum:.3f}"
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning)

    def call(
        self,
        y_true: Optional[keras.KerasTensor],
        y_pred: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor], list]
    ) -> keras.KerasTensor:
        """
        Compute symmetric contrastive loss for a batch.

        This method performs the core loss computation:
        1. Parse and validate input predictions
        2. Optionally apply temperature scaling
        3. Generate diagonal ground truth labels
        4. Compute cross-entropy in both directions
        5. Combine with configured weights
        6. Return per-sample losses (parent class handles reduction)

        The implementation uses categorical cross-entropy for numerical
        stability and support for label smoothing, treating contrastive learning
        as an N-way classification problem where the correct class is the diagonal.

        Args:
            y_true: Ignored for contrastive loss. Ground truth is automatically
                generated from batch structure (diagonal correspondence).
                Can be None or any value.
            y_pred: Predictions in one of these formats:
                - Dict: {'logits_per_image': Tensor, 'logits_per_text': Tensor}
                - Tuple/List: (logits_per_image, logits_per_text)
                Both logits tensors must be [batch_size, batch_size].

        Returns:
            Per-sample loss tensor of shape [batch_size]. The parent Loss class
            applies its reduction strategy (typically mean) to produce the final
            scalar loss value used for training.

        Raises:
            ValueError: If y_pred format is invalid, required keys are missing,
                or logit shapes are incompatible.

        Shape:
            - Input logits: (N, N) where N = batch_size
            - Output: (N,) per-sample losses before parent reduction

        Implementation Notes:
            - Uses keras.losses.categorical_crossentropy for label smoothing support.
            - from_logits=True: expects unnormalized logits (no softmax)
            - Per-sample losses allow custom reduction strategies in parent class
            - Temperature scaling is optional and controlled by apply_temperature flag
        """
        # Parse input predictions into separate logit tensors
        logits_per_image, logits_per_text = self._parse_predictions(y_pred)

        # Validate that logit shapes are compatible
        self._validate_logits(logits_per_image, logits_per_text)

        # Get batch size for label generation
        batch_size = ops.shape(logits_per_image)[0]

        # Apply temperature scaling if configured
        # Temperature scales logits before softmax: logits' = logits / τ
        # Lower τ → sharper distribution → higher confidence/penalty
        if self.apply_temperature:
            logits_per_image = logits_per_image / self.temperature
            logits_per_text = logits_per_text / self.temperature

        # Generate ground truth labels: diagonal elements are correct pairs
        # For batch_size=N, labels = [0, 1, 2, ..., N-1]
        # This means: image_i matches text_i (i-th diagonal element)
        labels = ops.arange(batch_size, dtype='int32')

        # Convert to one-hot encoding to support label smoothing
        # categorical_crossentropy with from_logits=True works with one-hot targets
        # and supports label_smoothing argument.
        y_true_one_hot = ops.one_hot(labels, num_classes=batch_size)

        # Compute image→text loss: "given image, predict matching text"
        # Each image should correctly classify its corresponding text from N options
        loss_i2t = keras.losses.categorical_crossentropy(
            y_true_one_hot,
            logits_per_image,
            from_logits=True,  # Input is unnormalized logits
            label_smoothing=self.label_smoothing  # Optional regularization
        )  # Shape: [batch_size]

        # Compute text→image loss: "given text, predict matching image"
        # Each text should correctly classify its corresponding image from N options
        loss_t2i = keras.losses.categorical_crossentropy(
            y_true_one_hot,
            logits_per_text,
            from_logits=True,
            label_smoothing=self.label_smoothing
        )  # Shape: [batch_size]

        # Combine losses with configured weights
        # Default (0.5, 0.5) gives equal importance to both directions
        # Custom weights allow emphasizing one modality over the other
        total_loss_per_sample = (
            self.loss_weight_i2t * loss_i2t +
            self.loss_weight_t2i * loss_t2i
        )  # Shape: [batch_size]

        return total_loss_per_sample

    def _parse_predictions(
        self,
        y_pred: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor], list]
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Parse different prediction formats into standardized logit tensors.

        Supports multiple input formats for flexibility in model design:
        - Dictionary (recommended): Explicit key-based access
        - Tuple/List: Positional access for simpler models

        Args:
            y_pred: Predictions in dictionary, tuple, or list format containing
                logits_per_image and logits_per_text tensors.

        Returns:
            Tuple of (logits_per_image, logits_per_text), both [N, N] tensors.

        Raises:
            ValueError: If format is unsupported, required data is missing, or
                tuple/list doesn't have exactly 2 elements.

        Examples:
            # Dictionary format (recommended)
            pred = {'logits_per_image': img_logits, 'logits_per_text': txt_logits}
            img, txt = _parse_predictions(pred)

            # Tuple format
            pred = (img_logits, txt_logits)
            img, txt = _parse_predictions(pred)
        """
        if isinstance(y_pred, dict):
            # Dictionary format: explicit key-based access
            required_keys = {'logits_per_image', 'logits_per_text'}
            if not required_keys.issubset(y_pred.keys()):
                missing = required_keys - set(y_pred.keys())
                raise ValueError(
                    f"y_pred dict must contain {required_keys} keys. "
                    f"Missing keys: {missing}. Got keys: {list(y_pred.keys())}"
                )
            return y_pred['logits_per_image'], y_pred['logits_per_text']

        elif isinstance(y_pred, (list, tuple)):
            # Tuple/List format: positional access
            if len(y_pred) != 2:
                raise ValueError(
                    f"y_pred list/tuple must have exactly 2 elements "
                    f"(logits_per_image, logits_per_text), got {len(y_pred)} elements"
                )
            return y_pred[0], y_pred[1]

        else:
            raise ValueError(
                f"Unsupported y_pred format: {type(y_pred).__name__}. "
                "Expected dict with keys ['logits_per_image', 'logits_per_text'] "
                "or list/tuple with 2 elements."
            )

    def _validate_logits(
        self,
        logits_per_image: keras.KerasTensor,
        logits_per_text: keras.KerasTensor
    ) -> None:
        """
        Validate logit tensor shapes and properties for correctness.

        Ensures that logit tensors have compatible shapes for contrastive loss
        computation. For standard CLIP training, both tensors should be square
        matrices [N, N] where N is batch_size.

        Args:
            logits_per_image: Image→text similarity logits.
            logits_per_text: Text→image similarity logits.

        Raises:
            ValueError: If logits are not 2D, have incompatible shapes, or other
                structural issues that prevent loss computation.

        Warnings:
            Logs warning if logit matrices are not square, which is unusual for
            standard CLIP but may be valid for custom use cases.

        Validation Checks:
            1. Both tensors must be 2D (rank 2)
            2. Both tensors must have identical shapes
            3. Warning if not square (batch_size × batch_size)
        """
        img_shape = ops.shape(logits_per_image)
        txt_shape = ops.shape(logits_per_text)

        # Check tensor rank (must be 2D matrices)
        if len(img_shape) != 2 or len(txt_shape) != 2:
            raise ValueError(
                f"Logits must be 2D tensors (similarity matrices), got: "
                f"logits_per_image.shape={img_shape}, logits_per_text.shape={txt_shape}. "
                "Expected shape: [batch_size, batch_size] for both."
            )

        # Check shape compatibility (must be identical)
        if img_shape != txt_shape:
            raise ValueError(
                f"Logit matrix shapes must be identical. "
                f"Got logits_per_image.shape={img_shape} and "
                f"logits_per_text.shape={txt_shape}. "
                "Both should be [batch_size, batch_size]."
            )

        # Check if matrices are square (expected for standard CLIP)
        if img_shape[0] != img_shape[1] or txt_shape[0] != txt_shape[1]:
            logger.warning(
                f"Logit matrices are not square. This is unusual for standard CLIP "
                f"contrastive loss. Shapes: logits_per_image={img_shape}, "
                f"logits_per_text={txt_shape}. Expected: [N, N] where N=batch_size. "
                "Proceeding with non-square matrices - ensure this is intentional."
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        This method is called during model saving to capture all constructor
        arguments needed to recreate this loss instance. It enables proper
        Keras model serialization and deserialization.

        Returns:
            Dictionary containing all configuration parameters passed to __init__.
            Includes both custom parameters and base class configuration.

        Configuration Keys:
            - temperature: Temperature scaling parameter
            - label_smoothing: Label smoothing factor
            - apply_temperature: Temperature application flag
            - loss_weight_i2t: Image→text loss weight
            - loss_weight_t2i: Text→image loss weight
            - Plus base Loss class config (name, reduction, etc.)

        Example:
            ```python
            # Create loss with custom config
            loss = CLIPContrastiveLoss(
                temperature=0.05,
                label_smoothing=0.1,
                apply_temperature=True
            )

            # Get configuration
            config = loss.get_config()
            # {'temperature': 0.05, 'label_smoothing': 0.1, ...}

            # Recreate from config
            new_loss = CLIPContrastiveLoss.from_config(config)
            # new_loss is functionally identical to original loss
            ```

        Note:
            All parameters that can be passed to __init__ must be included
            in get_config() for proper serialization. This is critical for
            model saving and loading.
        """
        # Get base class configuration (name, reduction, dtype, etc.)
        config = super().get_config()

        # Add all custom parameters
        config.update({
            "temperature": self.temperature,
            "label_smoothing": self.label_smoothing,
            "apply_temperature": self.apply_temperature,
            "loss_weight_i2t": self.loss_weight_i2t,
            "loss_weight_t2i": self.loss_weight_t2i,
        })

        return config

    @property
    def temperature_value(self) -> float:
        """
        Get the current temperature parameter value.

        Provides read-only access to the temperature parameter for monitoring
        or conditional logic during training.

        Returns:
            Current temperature value as float.

        Example:
            ```python
            loss = CLIPContrastiveLoss(temperature=0.07)
            print(f"Current temperature: {loss.temperature_value}")
            # Output: Current temperature: 0.07
            ```
        """
        return self.temperature

    def update_temperature(self, new_temperature: float) -> None:
        """
        Update the temperature parameter dynamically during training.

        Allows temperature to be adjusted between training epochs or based on
        validation metrics. This can be useful for curriculum learning or
        adaptive training strategies.

        Args:
            new_temperature: New temperature value. Must be positive.

        Raises:
            ValueError: If new_temperature is not positive.

        Examples:
            ```python
            # Create loss with initial temperature
            loss = CLIPContrastiveLoss(temperature=0.1, apply_temperature=True)

            # Train for some epochs...

            # Decrease temperature for harder training
            loss.update_temperature(0.05)

            # Continue training with new temperature
            ```

            ```python
            # Temperature annealing schedule
            loss = CLIPContrastiveLoss(temperature=0.2, apply_temperature=True)

            for epoch in range(num_epochs):
                # Gradually decrease temperature
                if epoch % 10 == 0:
                    new_temp = max(0.05, 0.2 * (0.9 ** (epoch // 10)))
                    loss.update_temperature(new_temp)

                # Train for one epoch
                model.fit(train_dataset, epochs=1)
            ```

        Note:
            Temperature updates are logged for monitoring. Changes take effect
            immediately for all subsequent forward passes.
        """
        if new_temperature <= 0.0:
            raise ValueError(
                f"Temperature must be positive, got {new_temperature}. "
                "Temperature controls softmax sharpness and must be > 0."
            )

        old_temp = self.temperature
        self.temperature = new_temperature

        logger.info(
            f"Temperature updated: {old_temp:.6f} → {new_temperature:.6f} "
            f"(change: {new_temperature - old_temp:+.6f})"
        )