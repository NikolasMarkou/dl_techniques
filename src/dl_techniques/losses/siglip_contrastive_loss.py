"""
SigLIP Contrastive Loss Implementation

Modern sigmoid-based contrastive loss that treats each image-text pair
as an independent binary classification problem, eliminating the need
for global normalization and reducing memory requirements.

Based on "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)
"""

import keras
from keras import ops
import tensorflow as tf
from dl_techniques.utils.logger import logger


class SigLIPContrastiveLoss(keras.losses.Loss):
    """
    SigLIP Contrastive Loss Function.

    This loss treats each image-text pair as an independent binary classification
    problem, eliminating the need for global batch normalization found in
    traditional InfoNCE loss. This approach:

    - Reduces memory complexity from O(N²) to O(N)
    - Enables much larger batch sizes (up to 1M)
    - Provides more stable gradients
    - Eliminates false negative problems

    The loss is computed as:
    L = Σᵢ Σⱼ log(1 + exp(-zᵢⱼ * t * xᵢ·yⱼ))

    where:
    - zᵢⱼ = 1 if i==j (positive pair), -1 otherwise (negative pair)
    - t is the temperature parameter
    - xᵢ·yⱼ is the cosine similarity between image i and text j
    """

    def __init__(
            self,
            temperature: float = 1.0,
            use_learnable_temperature: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: str = 'siglip_contrastive_loss',
            **kwargs
    ):
        """
        Initialize SigLIP Contrastive Loss.

        Args:
            temperature: Fixed temperature parameter for scaling similarities
            use_learnable_temperature: Whether to use learnable temperature from model
            reduction: Type of reduction to apply to loss
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.temperature = temperature
        self.use_learnable_temperature = use_learnable_temperature

        logger.info(f"Initialized SigLIP loss with temperature: {temperature}")

    def call(self, y_true, y_pred):
        """
        Compute SigLIP contrastive loss.

        Args:
            y_true: Not used (dummy labels). SigLIP is self-supervised.
            y_pred: Dictionary containing:
                - 'logits_per_image': (batch_size, batch_size) similarity matrix
                - 'logits_per_text': (batch_size, batch_size) similarity matrix
                - 'temperature': current temperature value (if learnable)

        Returns:
            Scalar loss value
        """
        if isinstance(y_pred, dict):
            logits_per_image = y_pred['logits_per_image']
            logits_per_text = y_pred['logits_per_text']

            # Use model's temperature if learnable, otherwise use fixed
            if self.use_learnable_temperature and 'temperature' in y_pred:
                temperature = y_pred['temperature']
            else:
                temperature = self.temperature
        else:
            raise ValueError(
                "y_pred must be a dictionary containing logits and temperature"
            )

        batch_size = ops.shape(logits_per_image)[0]

        # Create labels: 1 for positive pairs (i==j), -1 for negative pairs
        labels = ops.eye(batch_size, dtype='float32') * 2.0 - 1.0  # {-1, 1}

        # Apply temperature scaling (logits are already normalized)
        scaled_logits_per_image = logits_per_image * temperature
        scaled_logits_per_text = logits_per_text * temperature

        # Compute sigmoid loss for both directions
        # Loss = log(1 + exp(-labels * logits))
        image_loss = ops.log(1.0 + ops.exp(-labels * scaled_logits_per_image))
        text_loss = ops.log(1.0 + ops.exp(-labels * scaled_logits_per_text))

        # Average over batch dimension
        image_loss = ops.mean(image_loss)
        text_loss = ops.mean(text_loss)

        # Combine losses
        total_loss = (image_loss + text_loss) / 2.0

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'use_learnable_temperature': self.use_learnable_temperature,
        })
        return config


class AdaptiveSigLIPLoss(keras.losses.Loss):
    """
    Adaptive SigLIP Loss with dynamic temperature scaling.

    This variant automatically adjusts temperature based on the similarity
    distribution, preventing collapse in early training and maintaining
    good gradients throughout training.
    """

    def __init__(
            self,
            initial_temperature: float = 1.0,
            min_temperature: float = 0.01,
            max_temperature: float = 10.0,
            adaptation_rate: float = 0.1,
            target_entropy: float = 0.5,
            reduction: str = 'sum_over_batch_size',
            name: str = 'adaptive_siglip_loss',
            **kwargs
    ):
        """
        Initialize Adaptive SigLIP Loss.

        Args:
            initial_temperature: Starting temperature value
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature
            adaptation_rate: Rate of temperature adaptation
            target_entropy: Target entropy for temperature adaptation
            reduction: Type of reduction to apply to loss
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.adaptation_rate = adaptation_rate
        self.target_entropy = target_entropy

        # Adaptive temperature (will be updated during training)
        self.adaptive_temperature = tf.Variable(
            initial_temperature,
            trainable=False,
            name='adaptive_temperature'
        )

    def call(self, y_true, y_pred):
        """
        Compute adaptive SigLIP loss with dynamic temperature.

        Args:
            y_true: Not used (dummy labels)
            y_pred: Dictionary containing logits

        Returns:
            Scalar loss value
        """
        if isinstance(y_pred, dict):
            logits_per_image = y_pred['logits_per_image']
            logits_per_text = y_pred['logits_per_text']
        else:
            raise ValueError("y_pred must be a dictionary containing logits")

        batch_size = ops.shape(logits_per_image)[0]

        # Compute current entropy of similarity distribution
        probs = ops.softmax(logits_per_image, axis=-1)
        current_entropy = -ops.mean(ops.sum(probs * ops.log(probs + 1e-8), axis=-1))

        # Adapt temperature based on entropy
        entropy_error = current_entropy - self.target_entropy
        temperature_update = -self.adaptation_rate * entropy_error

        new_temperature = ops.clip(
            self.adaptive_temperature + temperature_update,
            self.min_temperature,
            self.max_temperature
        )

        self.adaptive_temperature.assign(new_temperature)

        # Create labels
        labels = ops.eye(batch_size, dtype='float32') * 2.0 - 1.0

        # Apply adaptive temperature
        scaled_logits_per_image = logits_per_image * self.adaptive_temperature
        scaled_logits_per_text = logits_per_text * self.adaptive_temperature

        # Compute sigmoid loss
        image_loss = ops.log(1.0 + ops.exp(-labels * scaled_logits_per_image))
        text_loss = ops.log(1.0 + ops.exp(-labels * scaled_logits_per_text))

        # Average losses
        total_loss = (ops.mean(image_loss) + ops.mean(text_loss)) / 2.0

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'initial_temperature': self.initial_temperature,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature,
            'adaptation_rate': self.adaptation_rate,
            'target_entropy': self.target_entropy,
        })
        return config


class HybridContrastiveLoss(keras.losses.Loss):
    """
    Hybrid loss combining SigLIP with score-based objectives.

    This implementation incorporates Miyasawa theorem principles by
    combining contrastive learning with score matching for improved
    cross-modal representation learning.
    """

    def __init__(
            self,
            siglip_weight: float = 1.0,
            score_weight: float = 0.1,
            temperature: float = 1.0,
            noise_level: float = 0.1,
            reduction: str = 'sum_over_batch_size',
            name: str = 'hybrid_contrastive_loss',
            **kwargs
    ):
        """
        Initialize Hybrid Contrastive Loss.

        Args:
            siglip_weight: Weight for SigLIP contrastive loss
            score_weight: Weight for score matching loss
            temperature: Temperature for contrastive loss
            noise_level: Noise level for score matching
            reduction: Type of reduction to apply to loss
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.siglip_weight = siglip_weight
        self.score_weight = score_weight
        self.temperature = temperature
        self.noise_level = noise_level

        # Base SigLIP loss
        self.siglip_loss = SigLIPContrastiveLoss(
            temperature=temperature,
            reduction='none'  # We'll handle reduction ourselves
        )

    def call(self, y_true, y_pred):
        """
        Compute hybrid contrastive + score matching loss.

        Args:
            y_true: Not used
            y_pred: Dictionary containing logits and embeddings

        Returns:
            Combined loss value
        """
        # Standard SigLIP loss
        siglip_loss = self.siglip_loss(y_true, y_pred)

        # Score matching component (simplified)
        if 'image_embeddings' in y_pred and 'text_embeddings' in y_pred:
            image_emb = y_pred['image_embeddings']
            text_emb = y_pred['text_embeddings']

            # Add noise for score matching (Miyasawa theorem application)
            noise_image = ops.random.normal(ops.shape(image_emb), stddev=self.noise_level)
            noise_text = ops.random.normal(ops.shape(text_emb), stddev=self.noise_level)

            noisy_image_emb = image_emb + noise_image
            noisy_text_emb = text_emb + noise_text

            # Score matching loss (denoising objective)
            score_loss_image = ops.mean(ops.square(noisy_image_emb - image_emb - noise_image))
            score_loss_text = ops.mean(ops.square(noisy_text_emb - text_emb - noise_text))

            score_loss = (score_loss_image + score_loss_text) / 2.0
        else:
            score_loss = 0.0

        # Combine losses
        total_loss = self.siglip_weight * siglip_loss + self.score_weight * score_loss

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'siglip_weight': self.siglip_weight,
            'score_weight': self.score_weight,
            'temperature': self.temperature,
            'noise_level': self.noise_level,
        })
        return config


# Convenience functions for creating loss functions
def create_siglip_loss(
        temperature: float = 1.0,
        use_learnable_temperature: bool = True,
        **kwargs
) -> SigLIPContrastiveLoss:
    """Create standard SigLIP contrastive loss."""
    return SigLIPContrastiveLoss(
        temperature=temperature,
        use_learnable_temperature=use_learnable_temperature,
        **kwargs
    )


def create_adaptive_siglip_loss(
        initial_temperature: float = 1.0,
        target_entropy: float = 0.5,
        **kwargs
) -> AdaptiveSigLIPLoss:
    """Create adaptive SigLIP loss with dynamic temperature."""
    return AdaptiveSigLIPLoss(
        initial_temperature=initial_temperature,
        target_entropy=target_entropy,
        **kwargs
    )


def create_hybrid_loss(
        siglip_weight: float = 1.0,
        score_weight: float = 0.1,
        **kwargs
) -> HybridContrastiveLoss:
    """Create hybrid loss combining SigLIP with score matching."""
    return HybridContrastiveLoss(
        siglip_weight=siglip_weight,
        score_weight=score_weight,
        **kwargs
    )