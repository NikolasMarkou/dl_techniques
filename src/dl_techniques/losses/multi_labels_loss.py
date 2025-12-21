"""
Per-Channel Loss Wrapper for Multi-Label Segmentation
======================================================

Wraps any binary loss to apply it independently per channel,
then averages the results. This is essential for multi-label
segmentation where each channel represents an independent
binary classification problem.
"""

import keras
from typing import Optional, List, Union
import tensorflow as tf


@keras.saving.register_keras_serializable()
class PerChannelBinaryLoss(keras.losses.Loss):
    """
    Wrapper that applies a binary loss function independently per channel.

    For multi-label segmentation with C channels, this:
    1. Computes binary loss for all channels (vectorized)
    2. Applies optional per-channel weights
    3. Averages the result

    This avoids implicit averaging over channels that standard losses might do,
    and allows for specific per-channel weighting.

    Parameters
    ----------
    base_loss : keras.losses.Loss or str
        The binary loss to apply per channel.
        Can be 'binary_crossentropy', 'binary_focal_crossentropy', etc.
        Note: If passing a custom Loss instance, ensure it returns element-wise
        losses or per-sample losses, not a scalar.
    channel_weights : Optional[List[float]]
        Optional weights for each channel. If None, all channels weighted equally.
    reduction : str
        Reduction method ('sum_over_batch_size', 'sum', 'none')
    name : str
        Name for this loss
    """

    def __init__(
            self,
            base_loss: Union[keras.losses.Loss, str] = 'binary_crossentropy',
            channel_weights: Optional[List[float]] = None,
            reduction: str = 'sum_over_batch_size',
            name: str = 'per_channel_loss',
            **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)

        # Get the base loss
        if isinstance(base_loss, str):
            self.base_loss_name = base_loss
            # We don't instantiate the string loss here to allow for
            # specific functional implementations in call()
            self.base_loss = None
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
        tensor
            Loss tensor. Shape depends on reduction, usually scalar if
            sum_over_batch_size.
        """
        y_true = tf.cast(y_true, y_pred.dtype)

        # 1. Compute Element-wise Loss (B, H, W, C)
        # -----------------------------------------
        if self.base_loss_name == 'binary_crossentropy':
            # Manual element-wise BCE to ensure no implicit reduction over channels
            # Standard keras.losses.binary_crossentropy averages over the last dim
            epsilon = keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
            loss = -bce

        elif isinstance(self.base_loss, keras.losses.Loss):
            # If it's a Loss instance, we call its call() method directly.
            # This bypasses the wrapper's __call__ reduction logic, assuming
            # the subclass's call() returns element-wise or per-sample data.
            # Our WeightedBinaryFocalLoss below is designed for this.
            loss = self.base_loss.call(y_true, y_pred)
            
        elif callable(self.base_loss):
            # For functional losses (like the custom weighted_bce in factory)
            loss = self.base_loss(y_true, y_pred)
            
        else:
            # Fallback for string names (e.g. using backend functions)
            fn = keras.losses.get(self.base_loss_name)
            loss = fn(y_true, y_pred)

        # 2. Apply Per-Channel Weights
        # ----------------------------
        if self.channel_weights is not None:
            # weights: (C,)
            weights = tf.convert_to_tensor(self.channel_weights, dtype=loss.dtype)
            # Broadcast: (B, H, W, C) * (C,) -> (B, H, W, C)
            loss = loss * weights

        # 3. Return Element-wise Loss
        # ---------------------------
        # We return the full map (or per-sample map). 
        # The Keras Loss wrapper (super().__call__) handles the final reduction
        # (e.g. averaging over batch) based on self.reduction.
        # However, we want to ensure we average over H, W, and C here if we want
        # "per-channel" semantics compressed into the standard Keras contract,
        # OR we leave it to Keras. 
        
        # Typically, Keras expects losses of shape (Batch,) or (Batch, d0, ..).
        # It then averages. 
        # Since we want to normalize by the number of pixels AND channels, 
        # we can just return the map. Keras will mean() it.
        
        return loss

    def get_config(self):
        config = super().get_config()
        # Serialize base_loss if it's an object, otherwise store name
        base_loss_config = (
            keras.saving.serialize_keras_object(self.base_loss)
            if self.base_loss is not None else self.base_loss_name
        )
        config.update({
            'base_loss': base_loss_config,
            'channel_weights': self.channel_weights
        })
        return config

    @classmethod
    def from_config(cls, config):
        if 'base_loss' in config and isinstance(config['base_loss'], dict):
            config['base_loss'] = keras.saving.deserialize_keras_object(config['base_loss'])
        return cls(**config)


@keras.saving.register_keras_serializable()
class WeightedBinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss with class weighting for imbalanced segmentation.

    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Returns element-wise loss (no reduction) to allow for external
    weighting and flexible reduction strategies.
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
        Compute focal loss element-wise.

        Parameters
        ----------
        y_true : tensor
            Ground truth binary labels
        y_pred : tensor
            Predicted probabilities or logits

        Returns
        -------
        tensor
            Focal loss value (element-wise), same shape as input
        """
        # Convert to probabilities if needed
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        # Clip to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        y_true = tf.cast(y_true, y_pred.dtype)

        # For positive examples
        # loss_pos = -alpha * (1 - p)^gamma * log(p)
        p_t_pos = y_pred
        alpha_t_pos = self.alpha
        loss_pos = -alpha_t_pos * tf.pow(1.0 - p_t_pos, self.gamma) * tf.math.log(p_t_pos)

        # For negative examples
        # loss_neg = -(1-alpha) * p^gamma * log(1-p)
        p_t_neg = 1.0 - y_pred
        alpha_t_neg = 1.0 - self.alpha
        loss_neg = -alpha_t_neg * tf.pow(1.0 - p_t_neg, self.gamma) * tf.math.log(p_t_neg)

        # Combine based on ground truth
        loss = y_true * loss_pos + (1.0 - y_true) * loss_neg

        # Return element-wise loss (no reduce_mean)
        return loss

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
    """
    if loss_type == 'focal':
        # from_logits=False because model ends with sigmoid
        base_loss = WeightedBinaryFocalLoss(alpha=alpha, gamma=gamma, from_logits=False)
        return PerChannelBinaryLoss(base_loss, channel_weights=channel_weights)

    elif loss_type == 'bce':
        return PerChannelBinaryLoss('binary_crossentropy', channel_weights=channel_weights)

    elif loss_type == 'weighted_bce':
        # Use tensorflow's built-in weighted BCE
        def weighted_bce(y_true, y_pred):
            # Compute pos_weight from alpha
            # alpha is "balance" -> pos_weight adjusts trade-off
            # if alpha=0.75 (3x weight on pos), pos_weight approx 3.0
            pos_weight = alpha / (1.0 - alpha) if alpha != 1.0 else 1.0
            
            # This function expects logits, but model outputs sigmoid.
            # We must inverse sigmoid or use standard BCE if inputs are probs.
            # However, standard weighted_cross_entropy_with_logits is numerically stable.
            # Since model has sigmoid, we can use epsilon inverse? 
            # Or better: implement weighted BCE for probabilities.
            
            y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
            
            # Loss = -pos_w * y_true * log(y_pred) - (1-y_true) * log(1-y_pred)
            loss = -pos_weight * y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            return loss

        return PerChannelBinaryLoss(weighted_bce, channel_weights=channel_weights)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


