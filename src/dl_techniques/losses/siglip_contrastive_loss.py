"""
Sigmoid Loss for Language Image Pre-training (SigLIP).

This loss function presents a paradigm shift from the conventional softmax-
based contrastive learning used in models like CLIP. Instead of framing
the task as a multi-class classification problem over a batch of samples,
SigLIP treats each image-text pair as an independent binary
classification problem. This architectural change eliminates the need for
a global normalization term (i.e., the denominator in the softmax),
which is computationally expensive and memory-intensive.

The primary advantage of this design is scalability. By removing the
inter-sample dependency required for normalization, the computational
complexity with respect to negative samples is significantly reduced. This
allows for training with much larger batch sizes, which is crucial for
effective representation learning. Furthermore, this approach inherently
avoids the "false negative" problem where semantically similar pairs are
incorrectly pushed apart, as it does not rely on a global view of
negatives.

Foundational Mathematics
------------------------
Contrastive losses like InfoNCE (used in CLIP) are based on a softmax
cross-entropy formulation. For an image embedding `x_i` and text
embedding `y_j`, the probability of `y_j` being the correct caption for
`x_i` among `N` candidates is modeled as:

    p_ij = exp(s_ij / t) / Σ_k exp(s_ik / t)

where `s_ij` is the cosine similarity and `t` is a temperature parameter.

SigLIP replaces this with a simpler, pairwise sigmoid cross-entropy. For
any given pair `(x_i, y_j)`, the goal is to predict a binary label `z_ij`,
where `z_ij = 1` if `i=j` (a positive pair) and `z_ij = -1` otherwise (a
negative pair). The loss for a single pair is the negative log-likelihood:

    L_ij = -log(sigmoid(z_ij * s_ij * t))

Using the identity `log(sigmoid(a)) = -log(1 + exp(-a))`, this can be
rewritten in the numerically stable form used in the implementation:

    L_ij = log(1 + exp(-z_ij * s_ij * t))

The total loss is the sum (or mean) of these pairwise losses across all
possible pairs in the batch, computed symmetrically for both image-to-text
and text-to-image directions. This formulation effectively trains a
classifier to distinguish between correct and incorrect pairings on a
case-by-case basis.

References
----------
-   Zhai, X., et al. (2023). "Sigmoid Loss for Language Image
    Pre-Training". *International Conference on Computer Vision (ICCV)*.
"""

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.losses.siglip_contrastive_loss")
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
            use_learnable_temperature: Whether to read ``temperature`` from ``y_pred``.
                Defaults to ``False`` here. NOTE: :func:`create_siglip_loss` defaults it
                to ``True`` -- a deliberate, documented divergence, not an oversight.
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
            Per-sample loss values of shape `(batch_size,)`.
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
        # Loss = softplus(-labels * logits), numerically stable form of log(1 + exp(x))
        image_loss = ops.softplus(-labels * scaled_logits_per_image)
        text_loss = ops.softplus(-labels * scaled_logits_per_text)

        # Reduce over the PAIR axis only, leaving the batch axis intact, so
        # that `call()` returns one value per sample. `keras.losses.Loss.__call__`
        # multiplies by `sample_weight` BEFORE reducing: a scalar returned here
        # would broadcast and charge every row the batch aggregate, which makes
        # both `sample_weight` and `reduction=` dead knobs.
        image_loss = keras.ops.mean(image_loss, axis=-1)
        text_loss = keras.ops.mean(text_loss, axis=-1)

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

@register_dl_technique("dl_techniques.losses.siglip_contrastive_loss")
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
        self.adaptive_temperature = keras.Variable(
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
            Per-sample loss values of shape `(batch_size,)`.
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

        # Compute sigmoid loss (softplus for numerical stability)
        image_loss = ops.softplus(-labels * scaled_logits_per_image)
        text_loss = ops.softplus(-labels * scaled_logits_per_text)

        # Reduce over the PAIR axis only, leaving the batch axis intact, so that
        # `call()` returns one value per sample (see `SigLIPContrastiveLoss.call`).
        # The temperature adaptation above is UNAFFECTED: it still runs exactly one
        # `assign` per call, on a batch-global entropy estimate.
        total_loss = (
            keras.ops.mean(image_loss, axis=-1) + keras.ops.mean(text_loss, axis=-1)
        ) / 2.0

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

@register_dl_technique("dl_techniques.losses.siglip_contrastive_loss")
class HybridContrastiveLoss(keras.losses.Loss):
    """
    SigLIP plus a cross-modal denoising penalty.

    **What the second term actually is.** Gaussian noise is added to each
    modality's embeddings, and the noisy embeddings of one modality are pulled
    toward the CLEAN embeddings of the other::

        score_term = ||image_emb + n_i - text_emb||^2 + ||text_emb + n_t - image_emb||^2

    That is a squared-error regularizer with noise injection. It is **not score
    matching**, and Miyasawa's theorem is **not** applied: nothing here estimates a
    score ``grad log p(x)``, there is no ``sigma^2`` scaling relating the residual to
    a score, and the regression target is the OTHER modality rather than the clean
    signal of the same variable. Denoising score matching requires all three.

    An earlier version of this docstring claimed it "incorporates Miyasawa theorem
    principles by combining contrastive learning with score matching". That claim did
    not describe the code and was removed (2026-08-31); the math was left exactly as
    it is, because the term IS a useful noise-robustness regularizer under its real
    name. If genuine denoising score matching is wanted here, it is a new feature, not
    a docstring correction.

    For an implementation that DOES rest on Miyasawa, see
    :mod:`dl_techniques.losses.jacobian_symmetry`.
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
            score_weight: Weight for the cross-modal denoising penalty
            temperature: Temperature for contrastive loss
            noise_level: Std-dev of the Gaussian noise added to each modality
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
        Compute the SigLIP term plus the cross-modal denoising penalty.

        Args:
            y_true: Not used
            y_pred: Dictionary containing logits and embeddings

        Returns:
            Per-sample combined loss values of shape `(batch_size,)`.
        """
        # Standard SigLIP loss
        siglip_loss = self.siglip_loss(y_true, y_pred)

        # Cross-modal denoising penalty (NOT score matching -- see class docstring)
        if 'image_embeddings' in y_pred and 'text_embeddings' in y_pred:
            image_emb = y_pred['image_embeddings']
            text_emb = y_pred['text_embeddings']

            # Additive Gaussian noise. No score is estimated; see class docstring.
            noise_image = keras.random.normal(ops.shape(image_emb), stddev=self.noise_level)
            noise_text = keras.random.normal(ops.shape(text_emb), stddev=self.noise_level)

            noisy_image_emb = image_emb + noise_image
            noisy_text_emb = text_emb + noise_text

            # Cross-modal denoising objective: noisy embeddings from one
            # modality should remain close to clean embeddings of the other.
            # This regularizes the embedding space for noise robustness.
            # Sum over the FEATURE axis only: one denoising-penalty value per
            # sample, matching the per-sample SigLIP term above. The inner loss
            # is constructed with `reduction='none'`, so it already hands back
            # `(batch_size,)`.
            score_loss_image = keras.ops.sum(
                keras.ops.square(noisy_image_emb - text_emb), axis=-1
            )
            score_loss_text = keras.ops.sum(
                keras.ops.square(noisy_text_emb - image_emb), axis=-1
            )

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


# ---------------------------------------------------------------------

def create_siglip_loss(
        temperature: float = 1.0,
        use_learnable_temperature: bool = True,
        **kwargs
) -> SigLIPContrastiveLoss:
    """Create a standard SigLIP contrastive loss.

    .. warning::

        **This factory's ``use_learnable_temperature`` default is ``True``, while
        :class:`SigLIPContrastiveLoss`'s own default is ``False``.** The divergence is
        DELIBERATE and is documented rather than aligned (2026-08-31): the factory
        expresses the "wired to a model that emits a ``temperature`` key" recipe, the
        bare class expresses the standalone one. Aligning them would silently change
        behaviour for every factory caller, and no measured defect motivates it.

        The practical consequence: ``create_siglip_loss()`` reads ``temperature`` out of
        ``y_pred`` when the key is present, and ``SigLIPContrastiveLoss()`` ignores it.
        If you want the class default, pass ``use_learnable_temperature=False``
        explicitly.

    :param temperature: Fixed temperature, used when the learnable path is off or when
        ``y_pred`` carries no ``temperature`` key.
    :param use_learnable_temperature: See the warning above. Defaults to ``True`` HERE,
        unlike the class.
    :returns: A configured :class:`SigLIPContrastiveLoss`.
    """
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
    """Create a hybrid loss: SigLIP plus a cross-modal denoising penalty."""
    return HybridContrastiveLoss(
        siglip_weight=siglip_weight,
        score_weight=score_weight,
        **kwargs
    )

# ---------------------------------------------------------------------
