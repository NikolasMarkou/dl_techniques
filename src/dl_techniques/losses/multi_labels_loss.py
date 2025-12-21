"""
Per-Channel Loss Wrapper for Multi-Label Segmentation
======================================================

Wraps any binary loss to apply it independently per channel,
then averages the results. This is essential for multi-label
segmentation where each channel represents an independent
binary classification problem.
"""

import keras
from keras import ops
from typing import Optional, List
import tensorflow as tf


@keras.saving.register_keras_serializable()
class PerChannelBinaryLoss(keras.losses.Loss):
    """
    Wrapper that applies a binary loss function independently per channel.

    For multi-label segmentation with C channels, this:
    1. Splits predictions and targets into C separate channels
    2. Applies the binary loss to each channel independently
    3. Averages (or weights) the per-channel losses

    This is critical for multi-label tasks where each channel is an
    independent binary classification problem.

    Parameters
    ----------
    base_loss : keras.losses.Loss or str
        The binary loss to apply per channel.
        Can be 'binary_crossentropy', 'binary_focal_crossentropy', etc.
    channel_weights : Optional[List[float]]
        Optional weights for each channel. If None, all channels weighted equally.
    reduction : str
        Reduction method ('sum_over_batch_size', 'sum', 'none')
    name : str
        Name for this loss

    Examples
    --------
    >>> # Binary cross-entropy per channel
    >>> loss = PerChannelBinaryLoss('binary_crossentropy')
    >>>
    >>> # Focal loss per channel
    >>> focal = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    >>> loss = PerChannelBinaryLoss(focal)
    >>>
    >>> # With channel weights (e.g., weight rare classes more)
    >>> weights = [1.0] * 80
    >>> weights[rare_class_idx] = 5.0
    >>> loss = PerChannelBinaryLoss('binary_crossentropy', channel_weights=weights)
    """

    def __init__(
            self,
            base_loss='binary_crossentropy',
            channel_weights: Optional[List[float]] = None,
            reduction: str = 'sum_over_batch_size',
            name: str = 'per_channel_loss',
            **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)

        # Get the base loss
        if isinstance(base_loss, str):
            self.base_loss = keras.losses.get(base_loss)
            self.base_loss_name = base_loss
        else:
            self.base_loss = base_loss
            self.base_loss_name = base_loss.__class__.__name__

        self.channel_weights = channel_weights

    def call(self, y_true, y_pred):
        """
        Compute loss independently per channel.

        Parameters
        ----------
        y_true : tensor, shape (batch, height, width, channels)
            Ground truth multi-label masks
        y_pred : tensor, shape (batch, height, width, channels)
            Predicted multi-label probabilities

        Returns
        -------
        tensor, scalar
            Average loss across all channels
        """
        # Get number of channels
        num_channels = tf.shape(y_pred)[-1]

        # Split into per-channel tensors
        y_true_channels = tf.unstack(y_true, axis=-1)
        y_pred_channels = tf.unstack(y_pred, axis=-1)

        # Compute loss per channel
        channel_losses = []
        for c, (y_t, y_p) in enumerate(zip(y_true_channels, y_pred_channels)):
            # Ensure shapes are correct (B, H, W) -> (B, H, W, 1)
            y_t = tf.expand_dims(y_t, axis=-1)
            y_p = tf.expand_dims(y_p, axis=-1)

            # Compute loss for this channel
            loss_c = self.base_loss(y_t, y_p)

            # Apply channel weight if provided
            if self.channel_weights is not None:
                weight = self.channel_weights[c]
                loss_c = loss_c * weight

            channel_losses.append(loss_c)

        # Average across channels
        total_loss = tf.reduce_mean(tf.stack(channel_losses))

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'base_loss': keras.saving.serialize_keras_object(self.base_loss),
            'channel_weights': self.channel_weights
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['base_loss'] = keras.saving.deserialize_keras_object(config['base_loss'])
        return cls(**config)


@keras.saving.register_keras_serializable()
class WeightedBinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss with class weighting for imbalanced segmentation.

    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Where:
    - alpha: Weight for positive class (higher = focus more on positive examples)
    - gamma: Focusing parameter (higher = focus more on hard examples)
    - p_t: Predicted probability for true class

    Parameters
    ----------
    alpha : float, default=0.25
        Weight for positive class. Use 0.25-0.75 for imbalanced data.
    gamma : float, default=2.0
        Focusing parameter. Higher values (2-5) focus more on hard examples.
    from_logits : bool, default=False
        Whether predictions are logits or probabilities
    reduction : str
        Reduction method
    name : str
        Loss name

    Examples
    --------
    >>> # Standard focal loss
    >>> loss = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
    >>>
    >>> # Strong focus on hard examples
    >>> loss = WeightedBinaryFocalLoss(alpha=0.25, gamma=5.0)
    >>>
    >>> # More weight to positive class (rare objects)
    >>> loss = WeightedBinaryFocalLoss(alpha=0.75, gamma=2.0)
    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            from_logits: bool = False,
            reduction: str = 'sum_over_batch_size',
            name: str = 'weighted_binary_focal_loss',
            **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Compute focal loss.

        Parameters
        ----------
        y_true : tensor
            Ground truth binary labels
        y_pred : tensor
            Predicted probabilities or logits

        Returns
        -------
        tensor
            Focal loss value
        """
        # Convert to probabilities if needed
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        # Clip to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute focal loss
        y_true = tf.cast(y_true, tf.float32)

        # For positive examples
        p_t_pos = y_pred
        alpha_t_pos = self.alpha
        loss_pos = -alpha_t_pos * tf.pow(1.0 - p_t_pos, self.gamma) * tf.math.log(p_t_pos)

        # For negative examples
        p_t_neg = 1.0 - y_pred
        alpha_t_neg = 1.0 - self.alpha
        loss_neg = -alpha_t_neg * tf.pow(1.0 - p_t_neg, self.gamma) * tf.math.log(p_t_neg)

        # Combine based on ground truth
        loss = y_true * loss_pos + (1.0 - y_true) * loss_neg

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


# Factory function for easy creation
def create_multilabel_segmentation_loss(
        loss_type: str = 'focal',
        alpha: float = 0.75,
        gamma: float = 2.0,
        channel_weights: Optional[List[float]] = None
):
    """
    Factory function to create appropriate loss for multi-label segmentation.

    Parameters
    ----------
    loss_type : str
        Type of loss: 'focal', 'bce', 'weighted_bce'
    alpha : float
        For focal loss, weight for positive class (0.5-0.9 for rare objects)
    gamma : float
        For focal loss, focusing parameter (2-5 for hard examples)
    channel_weights : Optional[List[float]]
        Per-channel weights (e.g., weight rare classes more)

    Returns
    -------
    loss : keras.losses.Loss
        Configured loss function

    Examples
    --------
    >>> # Focal loss per channel (RECOMMENDED for class imbalance)
    >>> loss = create_multilabel_segmentation_loss('focal', alpha=0.75, gamma=2.0)
    >>>
    >>> # Weighted BCE per channel
    >>> loss = create_multilabel_segmentation_loss('bce', channel_weights=weights)
    >>>
    >>> # Strong focal loss for severe imbalance
    >>> loss = create_multilabel_segmentation_loss('focal', alpha=0.9, gamma=5.0)
    """
    if loss_type == 'focal':
        base_loss = WeightedBinaryFocalLoss(alpha=alpha, gamma=gamma, from_logits=False)
        return PerChannelBinaryLoss(base_loss, channel_weights=channel_weights)

    elif loss_type == 'bce':
        return PerChannelBinaryLoss('binary_crossentropy', channel_weights=channel_weights)

    elif loss_type == 'weighted_bce':
        # Use tensorflow's built-in weighted BCE
        def weighted_bce(y_true, y_pred):
            # Compute pos_weight from alpha
            pos_weight = alpha / (1.0 - alpha) if alpha != 1.0 else 10.0
            return tf.nn.weighted_cross_entropy_with_logits(
                y_true, tf.math.log(y_pred / (1.0 - y_pred + 1e-7)), pos_weight
            )

        return PerChannelBinaryLoss(weighted_bce, channel_weights=channel_weights)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


if __name__ == '__main__':
    # Example usage
    import numpy as np

    # Create dummy data: 4 samples, 32x32, 3 classes
    y_true = np.random.randint(0, 2, (4, 32, 32, 3)).astype('float32')
    y_pred = np.random.rand(4, 32, 32, 3).astype('float32')

    # Test focal loss
    print("Testing Focal Loss per channel...")
    loss_fn = create_multilabel_segmentation_loss('focal', alpha=0.75, gamma=2.0)
    loss_value = loss_fn(y_true, y_pred)
    print(f"Loss value: {loss_value:.4f}")

    # Test with channel weights
    print("\nTesting with channel weights...")
    weights = [1.0, 2.0, 5.0]  # Weight class 2 heavily
    loss_fn = create_multilabel_segmentation_loss('focal', alpha=0.75, gamma=2.0, channel_weights=weights)
    loss_value = loss_fn(y_true, y_pred)
    print(f"Loss value: {loss_value:.4f}")

    print("\nAll tests passed!")