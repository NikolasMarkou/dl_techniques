"""Wasserstein loss functions for GANs.

This module implements various Wasserstein loss functions commonly used in
Wasserstein GANs (WGANs) and WGAN-GP (Wasserstein GAN with Gradient Penalty).
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Any
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class WassersteinLoss(keras.losses.Loss):
    """Wasserstein loss for GANs.

    This loss implements the Wasserstein distance (Earth Mover's Distance) for
    training Wasserstein GANs. The loss is computed as:

    - For critic (discriminator): L_critic = E[D(fake)] - E[D(real)]
    - For generator: L_generator = -E[D(fake)]

    The loss expects predictions from a critic network and labels indicating
    whether the samples are real (1) or fake (0).

    Args:
        for_critic: Boolean, whether this loss is for the critic (discriminator)
            or generator. If True, computes critic loss. If False, computes
            generator loss.
        reduction: Type of reduction to apply to loss. Defaults to 'sum_over_batch_size'.
        name: Optional name for the loss instance.

    Input shape:
        - y_true: A tensor of shape (batch_size,) with values 0 (fake) or 1 (real).
        - y_pred: A tensor of shape (batch_size,) with critic predictions.

    Returns:
        Wasserstein loss value.

    Example:
        >>> # For critic training
        >>> critic_loss = WassersteinLoss(for_critic=True)
        >>> # For generator training
        >>> generator_loss = WassersteinLoss(for_critic=False)
        >>>
        >>> # Usage in model compilation
        >>> critic.compile(optimizer='adam', loss=critic_loss)
        >>> generator.compile(optimizer='adam', loss=generator_loss)
    """

    def __init__(
            self,
            for_critic: bool = True,
            reduction: str = "sum_over_batch_size",
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.for_critic = for_critic

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the Wasserstein loss.

        Args:
            y_true: Ground truth labels (0 for fake, 1 for real).
            y_pred: Predicted values from the critic.

        Returns:
            Computed Wasserstein loss.
        """
        # Convert to float32 to ensure proper computation
        y_true = ops.cast(y_true, dtype="float32")
        y_pred = ops.cast(y_pred, dtype="float32")

        if self.for_critic:
            # Critic loss: E[D(fake)] - E[D(real)]
            # y_true: 1 for real, 0 for fake
            # We want to maximize D(real) and minimize D(fake)
            # So we minimize: -D(real) + D(fake)
            real_loss = y_true * y_pred  # D(real) when y_true=1, 0 when y_true=0
            fake_loss = (1 - y_true) * y_pred  # D(fake) when y_true=0, 0 when y_true=1

            # Wasserstein critic loss: E[D(fake)] - E[D(real)]
            loss = fake_loss - real_loss
        else:
            # Generator loss: -E[D(fake)]
            # For generator, we only care about fake samples (y_true should be 0)
            # We want to maximize D(fake), so minimize -D(fake)
            loss = -y_pred

        return loss

    def get_config(self) -> dict:
        """Get the configuration of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "for_critic": self.for_critic,
        })
        return config


@keras.saving.register_keras_serializable()
class WassersteinGradientPenaltyLoss(keras.losses.Loss):
    """Wasserstein loss with gradient penalty for improved training stability.

    This loss implements the WGAN-GP (Wasserstein GAN with Gradient Penalty) loss
    which adds a gradient penalty term to the standard Wasserstein loss to enforce
    the Lipschitz constraint more effectively than weight clipping.

    The total loss is:
    L = Wasserstein_loss + λ * gradient_penalty

    where gradient_penalty = (||∇D(x̂)||₂ - 1)²
    and x̂ is sampled uniformly along straight lines between real and fake samples.

    Args:
        for_critic: Boolean, whether this loss is for the critic or generator.
        lambda_gp: Float, gradient penalty coefficient. Default is 10.0.
        reduction: Type of reduction to apply to loss.
        name: Optional name for the loss instance.

    Note:
        This loss requires access to both real and fake samples to compute the
        gradient penalty. It's typically used with a custom training loop.

    Example:
        >>> loss = WassersteinGradientPenaltyLoss(for_critic=True, lambda_gp=10.0)
        >>> # Used in custom training loop with gradient penalty computation
    """

    def __init__(
            self,
            for_critic: bool = True,
            lambda_gp: float = 10.0,
            reduction: str = "sum_over_batch_size",
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.for_critic = for_critic
        self.lambda_gp = lambda_gp
        self.wasserstein_loss = WassersteinLoss(for_critic=for_critic)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the Wasserstein loss with gradient penalty.

        Note: This method computes only the Wasserstein component.
        The gradient penalty must be computed separately in the training loop
        and added to this loss.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted values from the critic.

        Returns:
            Wasserstein loss component.
        """
        return self.wasserstein_loss(y_true, y_pred)

    def get_config(self) -> dict:
        """Get the configuration of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "for_critic": self.for_critic,
            "lambda_gp": self.lambda_gp,
        })
        return config


def compute_gradient_penalty(
        critic: keras.Model,
        real_samples: tf.Tensor,
        fake_samples: tf.Tensor,
        lambda_gp: float = 10.0
) -> tf.Tensor:
    """Compute gradient penalty for WGAN-GP.

    This function computes the gradient penalty term used in WGAN-GP to enforce
    the Lipschitz constraint. It samples points along straight lines between
    real and fake samples and computes the gradient norm penalty.

    Args:
        critic: The critic (discriminator) model.
        real_samples: Tensor of real samples.
        fake_samples: Tensor of fake samples.
        lambda_gp: Gradient penalty coefficient.

    Returns:
        Gradient penalty loss.

    Example:
        >>> # In training loop
        >>> gp_loss = compute_gradient_penalty(critic, real_batch, fake_batch)
        >>> total_critic_loss = wasserstein_loss + gp_loss
    """
    # Get batch size
    batch_size = ops.shape(real_samples)[0]

    # Sample random points along lines between real and fake samples
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)

    # Ensure alpha has the same number of dimensions as the samples
    while len(alpha.shape) < len(real_samples.shape):
        alpha = ops.expand_dims(alpha, axis=-1)

    # Interpolate between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples

    # Compute gradients of critic output w.r.t. interpolated samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        critic_output = critic(interpolated, training=True)

    gradients = tape.gradient(critic_output, interpolated)

    # Compute gradient penalty
    gradient_norm = ops.sqrt(ops.sum(ops.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = lambda_gp * ops.mean(ops.square(gradient_norm - 1.0))

    return gradient_penalty


@keras.saving.register_keras_serializable()
class WassersteinDivergence(keras.losses.Loss):
    """Wasserstein divergence loss for comparing distributions.

    This loss computes the Wasserstein divergence between two probability
    distributions, which can be useful for distribution matching tasks
    beyond standard GAN training.

    Args:
        smooth_eps: Small epsilon value for numerical stability.
        reduction: Type of reduction to apply to loss.
        name: Optional name for the loss instance.

    Example:
        >>> loss = WassersteinDivergence()
        >>> # Use for distribution matching tasks
    """

    def __init__(
            self,
            smooth_eps: float = 1e-7,
            reduction: str = "sum_over_batch_size",
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.smooth_eps = smooth_eps

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the Wasserstein divergence.

        Args:
            y_true: Target distribution.
            y_pred: Predicted distribution.

        Returns:
            Wasserstein divergence loss.
        """
        # Ensure non-negative values and normalize
        y_true = ops.maximum(y_true, self.smooth_eps)
        y_pred = ops.maximum(y_pred, self.smooth_eps)

        # Normalize to ensure they sum to 1
        y_true = y_true / (ops.sum(y_true, axis=-1, keepdims=True) + self.smooth_eps)
        y_pred = y_pred / (ops.sum(y_pred, axis=-1, keepdims=True) + self.smooth_eps)

        # Compute cumulative sums (simplified 1D Wasserstein)
        y_true_cumsum = ops.cumsum(y_true, axis=-1)
        y_pred_cumsum = ops.cumsum(y_pred, axis=-1)

        # Compute L1 distance between cumulative sums
        wasserstein_dist = ops.mean(ops.abs(y_true_cumsum - y_pred_cumsum), axis=-1)

        return wasserstein_dist

    def get_config(self) -> dict:
        """Get the configuration of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "smooth_eps": self.smooth_eps,
        })
        return config


# Utility functions for WGAN training
def create_wgan_losses(lambda_gp: float = 10.0) -> tuple[WassersteinLoss, WassersteinLoss]:
    """Create critic and generator losses for WGAN.

    Args:
        lambda_gp: Gradient penalty coefficient (not used in basic WGAN).

    Returns:
        Tuple of (critic_loss, generator_loss).

    Example:
        >>> critic_loss, generator_loss = create_wgan_losses()
        >>> critic.compile(optimizer='adam', loss=critic_loss)
        >>> generator.compile(optimizer='adam', loss=generator_loss)
    """
    critic_loss = WassersteinLoss(for_critic=True)
    generator_loss = WassersteinLoss(for_critic=False)

    logger.info("Created WGAN losses (critic and generator)")
    return critic_loss, generator_loss


def create_wgan_gp_losses(lambda_gp: float = 10.0) -> tuple[
    WassersteinGradientPenaltyLoss, WassersteinGradientPenaltyLoss]:
    """Create critic and generator losses for WGAN-GP.

    Args:
        lambda_gp: Gradient penalty coefficient.

    Returns:
        Tuple of (critic_loss, generator_loss).

    Example:
        >>> critic_loss, generator_loss = create_wgan_gp_losses(lambda_gp=10.0)
        >>> # Use in custom training loop with gradient penalty
    """
    critic_loss = WassersteinGradientPenaltyLoss(for_critic=True, lambda_gp=lambda_gp)
    generator_loss = WassersteinGradientPenaltyLoss(for_critic=False, lambda_gp=lambda_gp)

    logger.info(f"Created WGAN-GP losses with lambda_gp={lambda_gp}")
    return critic_loss, generator_loss