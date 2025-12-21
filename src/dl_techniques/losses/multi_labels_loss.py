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
from typing import Optional, List, Union
import tensorflow as tf


@keras.saving.register_keras_serializable()
class PerChannelBinaryLoss(keras.losses.Loss):
    """
    Wrapper that applies a binary loss function independently per channel.
    
    For multi-label segmentation with C channels, this:
    1. Computes element-wise binary loss for all channels (Vectorized)
    2. Computes adaptive weights based on class presence (Vectorized)
    3. Averages the weighted per-channel losses
    
    This implementation avoids loops and tf.unstack to ensure compatibility
    with dynamic input shapes (None dimensions) in Keras.
    
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
            self.base_loss = None
        else:
            self.base_loss = base_loss
            self.base_loss_name = base_loss.__class__.__name__
        
        self.channel_weights = channel_weights
        
    def call(self, y_true, y_pred):
        """
        Compute loss independently per channel with adaptive weighting.
        
        Vectorized implementation of:
        Weight = sqrt(presence_ratio) + 0.1
        
        Parameters
        ----------
        y_true : tensor, shape (batch, height, width, channels)
            Ground truth multi-label masks
        y_pred : tensor, shape (batch, height, width, channels)
            Predicted multi-label probabilities
        
        Returns
        -------
        tensor, scalar
            Weighted average loss
        """
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # 1. Compute Element-wise Loss (B, H, W, C)
        # -----------------------------------------
        if self.base_loss_name == 'binary_crossentropy':
            # Manual element-wise BCE to ensure no implicit reduction
            epsilon = keras.backend.epsilon()
            y_pred_safe = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            bce = y_true * tf.math.log(y_pred_safe) + (1 - y_true) * tf.math.log(1 - y_pred_safe)
            loss_tensor = -bce
            
        elif isinstance(self.base_loss, keras.losses.Loss):
            # Assume custom loss class returns element-wise or per-sample structure
            # (WeightedBinaryFocalLoss is designed to do this)
            loss_tensor = self.base_loss.call(y_true, y_pred)
            
        elif callable(self.base_loss):
            loss_tensor = self.base_loss(y_true, y_pred)
            
        else:
            # Fallback
            fn = keras.losses.get(self.base_loss_name)
            loss_tensor = fn(y_true, y_pred)

        # 2. Compute Adaptive Channel Weights (Vectorized)
        # ----------------------------------------------
        # Sum over Batch, H, W to get counts per channel -> Shape (C,)
        # Note: We use shape of y_true to handle dynamic dims safely
        positive_counts = tf.reduce_sum(y_true, axis=[0, 1, 2])
        
        # Calculate total pixels (B*H*W)
        input_shape = tf.shape(y_true)
        # prod(B, H, W)
        total_pixels_per_channel = tf.cast(
            tf.reduce_prod(input_shape[:-1]), 
            dtype=positive_counts.dtype
        )
        
        # Adaptive Logic: Weight = sqrt(presence_ratio) + 0.1
        presence_ratio = positive_counts / (total_pixels_per_channel + 1e-7)
        adaptive_weights = tf.sqrt(presence_ratio) + 0.1
        
        # Apply user-provided channel weights if present
        if self.channel_weights is not None:
            user_weights = tf.convert_to_tensor(self.channel_weights, dtype=adaptive_weights.dtype)
            adaptive_weights = adaptive_weights * user_weights

        # Normalize weights
        weights_normalized = adaptive_weights / (tf.reduce_sum(adaptive_weights) + 1e-7)
        
        # 3. Aggregate Loss
        # ----------------------------------------------
        # First, compute mean loss for each channel (average over B,H,W) -> Shape (C,)
        channel_mean_losses = tf.reduce_mean(loss_tensor, axis=[0, 1, 2])
        
        # Then compute weighted sum over channels
        total_loss = tf.reduce_sum(channel_mean_losses * weights_normalized)
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
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
    
    Returns element-wise loss (no reduction) to allow the PerChannel wrapper
    to handle adaptive weighting and reduction.
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
        """
        # Convert to probabilities if needed
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        
        # Clip to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        y_true = tf.cast(y_true, y_pred.dtype)
        
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
        
        # Return element-wise loss (Do not reduce here, wrapper handles it)
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


@keras.saving.register_keras_serializable()
class DiceLossPerChannel(keras.losses.Loss):
    """
    Dice Loss applied per channel for multi-label segmentation.
    
    Dice Coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice Coefficient
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'sum_over_batch_size',
        name: str = 'dice_loss_per_channel',
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Flatten spatial dimensions but keep batch and channels separate
        # Shape: (batch, height*width, channels)
        # Using tf.shape handles dynamic dimensions correctly
        input_shape = tf.shape(y_true)
        B, C = input_shape[0], input_shape[-1]
        
        y_true_flat = tf.reshape(y_true, [B, -1, C])
        y_pred_flat = tf.reshape(y_pred, [B, -1, C])
        
        # Compute per-channel Dice coefficient
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)  # (batch, channels)
        union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
        
        smooth_tensor = tf.constant(self.smooth, dtype=tf.float32)
        dice = (2.0 * intersection + smooth_tensor) / (union + smooth_tensor)
        
        # Average over batch and channels
        dice_mean = tf.reduce_mean(dice)
        
        return 1.0 - dice_mean
    
    def get_config(self):
        config = super().get_config()
        config['smooth'] = self.smooth
        return config


def create_multilabel_segmentation_loss(
    loss_type: str = 'focal',
    alpha: float = 0.75,
    gamma: float = 2.0,
    channel_weights: Optional[List[float]] = None,
    smooth: float = 1.0
):
    """
    Factory function to create appropriate loss for multi-label segmentation.
    """
    if loss_type == 'focal':
        base_loss = WeightedBinaryFocalLoss(alpha=alpha, gamma=gamma, from_logits=False)
        return PerChannelBinaryLoss(base_loss, channel_weights=channel_weights)
    
    elif loss_type == 'dice':
        return DiceLossPerChannel(smooth=smooth)
    
    elif loss_type == 'bce':
        return PerChannelBinaryLoss('binary_crossentropy', channel_weights=channel_weights)
    
    elif loss_type == 'weighted_bce':
        def weighted_bce(y_true, y_pred):
            pos_weight = alpha / (1.0 - alpha) if alpha != 1.0 else 10.0
            # Clip for numerical stability in log
            y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
            loss = -pos_weight * y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            return loss
        return PerChannelBinaryLoss(weighted_bce, channel_weights=channel_weights)
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


