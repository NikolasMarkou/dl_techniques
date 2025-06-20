"""
CLIP Contrastive Loss Implementation

This module implements the contrastive loss function used in CLIP (Contrastive Language-Image
Pre-training) for learning joint representations of images and text. The loss encourages
matching image-text pairs to have high similarity while pushing non-matching pairs apart.

Mathematical Formulation:
    CLIP uses a symmetric contrastive loss that operates on similarity matrices:

    For a batch of N image-text pairs:
    1. Compute similarity matrix S[i,j] = (image_i · text_j) / τ
    2. Create diagonal target matrix where correct pairs are on the diagonal
    3. Apply cross-entropy loss in both directions:
       - L_i2t = CrossEntropy(S_image_to_text, diagonal_labels)
       - L_t2i = CrossEntropy(S_text_to_image, diagonal_labels)
    4. Final loss = (L_i2t + L_t2i) / 2

    Where τ (tau) is a learnable temperature parameter that controls the sharpness
    of the softmax distribution.

Architecture Details:
    - Operates on pre-computed logit matrices (similarity scores)
    - Temperature scaling is typically applied during similarity computation
    - Symmetric loss ensures both image→text and text→image alignments
    - Label smoothing can be applied for regularization
    - Supports both fixed and learnable temperature parameters

Key Benefits:
    - Learns rich multimodal representations without explicit supervision
    - Scalable to large datasets through batch-wise contrastive learning
    - Enables zero-shot transfer to downstream tasks
    - Robust to noisy web-scale data through contrastive objective

Implementation Details:
    - Handles temperature scaling if not pre-applied to logits
    - Supports label smoothing for improved generalization
    - Numerically stable computation with proper logit handling
    - Configurable reduction and loss weighting options

References:
    - Radford, A., et al. (2021). "Learning Transferable Visual Representations
      from Natural Language Supervision." https://arxiv.org/abs/2103.00020

    - Jia, C., et al. (2021). "Scaling Up Visual and Vision-Language Representation
      Learning With Noisy Text Supervision." https://arxiv.org/abs/2102.05918

    - Li, J., et al. (2022). "BLIP: Bootstrapping Language-Image Pre-training for
      Unified Vision-Language Understanding and Generation."
      https://arxiv.org/abs/2201.12086

Usage Examples:
    Basic usage with pre-computed logits:
    >>> loss_fn = CLIPContrastiveLoss(temperature=0.07)
    >>> y_pred = {
    ...     'logits_per_image': image_text_similarities,
    ...     'logits_per_text': text_image_similarities
    ... }
    >>> loss = loss_fn(None, y_pred)  # y_true not needed

    With label smoothing:
    >>> loss_fn = CLIPContrastiveLoss(
    ...     temperature=0.07,
    ...     label_smoothing=0.1,
    ...     apply_temperature=True
    ... )

    In training loop:
    >>> model.compile(
    ...     optimizer='adam',
    ...     loss=CLIPContrastiveLoss(temperature=0.07),
    ...     metrics=['accuracy']
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
class CLIPContrastiveLoss(keras.losses.Loss):
    """
    Contrastive loss for CLIP training with numerical stability.

    This loss function implements the symmetric contrastive objective used in CLIP,
    encouraging matching image-text pairs to have high similarity while pushing
    non-matching pairs apart through cross-entropy loss in both directions.

    Args:
        temperature: float, default=0.07
            Temperature parameter for scaling logits. Lower values make the
            distribution sharper. If apply_temperature=False, this parameter
            is ignored.
        label_smoothing: float, default=0.0
            Label smoothing factor in [0, 1]. Higher values provide more
            regularization by softening the hard targets.
        apply_temperature: bool, default=False
            Whether to apply temperature scaling to the input logits.
            Set to True if logits haven't been temperature-scaled yet.
        loss_weight_i2t: float, default=0.5
            Weight for image-to-text loss component.
        loss_weight_t2i: float, default=0.5
            Weight for text-to-image loss component.
        name: str, default="clip_contrastive_loss"
            Name of the loss function.
        **kwargs: Additional keyword arguments for the Loss parent class.

    Input format:
        y_true: Not used (can be None) as contrastive loss is self-supervised
        y_pred: Dictionary containing:
            - 'logits_per_image': Tensor of shape (batch_size, batch_size)
              representing image-to-text similarities
            - 'logits_per_text': Tensor of shape (batch_size, batch_size)
              representing text-to-image similarities

    Returns:
        Scalar loss value representing the symmetric contrastive loss.

    Raises:
        ValueError: If temperature, label_smoothing, or loss weights are invalid,
                   or if required keys are missing from y_pred.
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
        super().__init__(name=name, **kwargs)

        # Validate inputs
        self._validate_inputs(temperature, label_smoothing, loss_weight_i2t, loss_weight_t2i)

        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.apply_temperature = apply_temperature
        self.loss_weight_i2t = loss_weight_i2t
        self.loss_weight_t2i = loss_weight_t2i

        logger.info(f"CLIPContrastiveLoss initialized: temperature={temperature}, "
                    f"label_smoothing={label_smoothing}, apply_temperature={apply_temperature}")

    def _validate_inputs(
            self,
            temperature: float,
            label_smoothing: float,
            loss_weight_i2t: float,
            loss_weight_t2i: float
    ) -> None:
        """Validate initialization parameters."""
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1], got {label_smoothing}")
        if loss_weight_i2t < 0.0:
            raise ValueError(f"loss_weight_i2t must be non-negative, got {loss_weight_i2t}")
        if loss_weight_t2i < 0.0:
            raise ValueError(f"loss_weight_t2i must be non-negative, got {loss_weight_t2i}")
        if abs(loss_weight_i2t + loss_weight_t2i - 1.0) > 1e-6:
            logger.warning(f"Loss weights sum to {loss_weight_i2t + loss_weight_t2i}, "
                           f"not 1.0. This may affect loss magnitude.")

    def call(
            self,
            y_true: Optional[keras.KerasTensor],
            y_pred: Union[Dict[str, keras.KerasTensor], keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Compute contrastive loss for CLIP.

        Args:
            y_true: Not used for contrastive loss (can be None)
            y_pred: Dictionary containing logits_per_image and logits_per_text
                   or tuple/list with (logits_per_image, logits_per_text)

        Returns:
            Scalar tensor representing the contrastive loss

        Raises:
            ValueError: If y_pred format is invalid or required keys are missing
        """
        # Handle different input formats
        logits_per_image, logits_per_text = self._parse_predictions(y_pred)

        # Validate logit shapes
        self._validate_logits(logits_per_image, logits_per_text)

        batch_size = ops.shape(logits_per_image)[0]

        # Apply temperature scaling if requested
        if self.apply_temperature:
            logits_per_image = logits_per_image / self.temperature
            logits_per_text = logits_per_text / self.temperature

        # Create target labels (diagonal matrix - correct pairs)
        labels = ops.arange(batch_size, dtype='int32')

        # Compute cross-entropy loss for both directions
        # Image-to-text: each image should match its corresponding text
        loss_i2t = keras.losses.sparse_categorical_crossentropy(
            labels,
            logits_per_image,
            from_logits=True,
            label_smoothing=self.label_smoothing
        )

        # Text-to-image: each text should match its corresponding image
        loss_t2i = keras.losses.sparse_categorical_crossentropy(
            labels,
            logits_per_text,
            from_logits=True,
            label_smoothing=self.label_smoothing
        )

        # Reduce losses to scalars
        loss_i2t_mean = ops.mean(loss_i2t)
        loss_t2i_mean = ops.mean(loss_t2i)

        # Combine losses with weights
        total_loss = (self.loss_weight_i2t * loss_i2t_mean +
                      self.loss_weight_t2i * loss_t2i_mean)

        return total_loss

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

        Raises:
            ValueError: If format is unsupported or required data is missing
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
        """
        Validate logit tensor shapes and properties.

        Args:
            logits_per_image: Image-to-text similarity logits
            logits_per_text: Text-to-image similarity logits

        Raises:
            ValueError: If logit shapes are incompatible
        """
        img_shape = ops.shape(logits_per_image)
        txt_shape = ops.shape(logits_per_text)

        # Both should be square matrices of same size
        if len(img_shape) != 2 or len(txt_shape) != 2:
            raise ValueError(
                f"Logits must be 2D tensors, got shapes: "
                f"logits_per_image={img_shape}, logits_per_text={txt_shape}"
            )

        # Check if matrices are square (batch_size x batch_size)
        if img_shape[0] != img_shape[1] or txt_shape[0] != txt_shape[1]:
            logger.warning(
                f"Logit matrices are not square. This is unusual for CLIP contrastive loss. "
                f"Shapes: logits_per_image={img_shape}, logits_per_text={txt_shape}"
            )

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization."""
        config = super().get_config()
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
        """Get the current temperature value."""
        return self.temperature

    def update_temperature(self, new_temperature: float) -> None:
        """
        Update the temperature parameter.

        Args:
            new_temperature: New temperature value (must be positive)

        Raises:
            ValueError: If new_temperature is not positive
        """
        if new_temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {new_temperature}")

        old_temp = self.temperature
        self.temperature = new_temperature
        logger.info(f"Temperature updated from {old_temp} to {new_temperature}")

# ---------------------------------------------------------------------
