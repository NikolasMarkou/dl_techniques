"""
Per-Channel Loss Wrapper for Multi-Label Segmentation
======================================================

This module provides a robust loss wrapper and specific loss implementations
designed for multi-label segmentation tasks. In such tasks, each channel
represents an independent binary classification problem (e.g., overlapping
objects, independent attributes).

The core component is ``PerChannelBinaryLoss``, which allows computing
losses independently per channel and aggregating them using adaptive weights
based on class presence, stabilizing training for imbalanced datasets.
"""

import keras
from keras import ops
from typing import Optional, List, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PerChannelBinaryLoss(keras.losses.Loss):
    """
    Wrapper that applies a binary loss function independently per channel.

    For multi-label segmentation with :math:`C` channels, this layer:

    1. Computes element-wise binary loss for all channels (Vectorized).
    2. Computes adaptive weights based on class presence within the batch.
    3. Averages the weighted per-channel losses.

    The adaptive weighting formula is:

    .. math::
        w_c = \\sqrt{\\frac{\\text{count}_c}{\\text{total\_pixels}}} + 0.1

    This ensures that rare classes are not completely overwhelmed but also
    prevents common classes from dominating the gradient purely by volume.

    Parameters
    ----------
    base_loss : Union[keras.losses.Loss, str]
        The binary loss to apply per channel. Can be a string alias
        (e.g., 'binary_crossentropy') or a Keras Loss instance.
    channel_weights : Optional[List[float]], default=None
        Static manual weights for each channel. If provided, these are
        multiplied with the adaptive weights. Useful for enforcing
        domain-specific importance.
    reduction : str, default='sum_over_batch_size'
        Type of reduction to apply to the loss. Note that this wrapper
        computes a scalar loss internally based on batch statistics,
        so the framework reduction usually acts as an identity.
    name : str, default='per_channel_loss'
        The name of the loss instance.
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

        # Handle base_loss string alias or instance
        if isinstance(base_loss, str):
            self.base_loss_name = base_loss
            self.base_loss = None
        else:
            self.base_loss = base_loss
            self.base_loss_name = base_loss.__class__.__name__

        self.channel_weights = channel_weights

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute loss independently per channel with adaptive weighting.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth multi-label masks.
            Shape: ``(batch_size, height, width, channels)``.
        y_pred : keras.KerasTensor
            Predicted multi-label probabilities.
            Shape: ``(batch_size, height, width, channels)``.

        Returns
        -------
        keras.KerasTensor
            Scalar tensor representing the weighted average loss.
        """
        # Ensure consistent dtypes
        y_true = ops.cast(y_true, y_pred.dtype)

        # 1. Compute Element-wise Loss (Batch, Height, Width, Channels)
        # -------------------------------------------------------------
        if self.base_loss_name == 'binary_crossentropy':
            # Manual element-wise BCE to ensure no implicit reduction occurs
            # before we want it to.
            epsilon = keras.backend.epsilon()
            y_pred_safe = ops.clip(y_pred, epsilon, 1.0 - epsilon)

            # BCE = -[y * log(p) + (1-y) * log(1-p)]
            bce = y_true * ops.log(y_pred_safe) + (1.0 - y_true) * ops.log(1.0 - y_pred_safe)
            loss_tensor = -bce

        elif self.base_loss is not None:
            # Assume custom loss class returns element-wise structure
            # (WeightedBinaryFocalLoss is designed to do this)
            loss_tensor = self.base_loss.call(y_true, y_pred)

        else:
            # Fallback for string aliases using standard factory
            # Note: Standard Keras losses might reduce output. We rely on
            # them usually returning (B, d0, ..) if reduction='none' isn't set,
            # but usually, we want the element-wise tensor here.
            fn = keras.losses.get(self.base_loss_name)
            loss_tensor = fn(y_true, y_pred)

        # 2. Compute Adaptive Channel Weights (Vectorized)
        # ------------------------------------------------
        # We compute statistics over the batch to handle imbalance dynamically.
        # Sum over Batch(0), Height(1), Width(2) -> Shape (Channels,)
        positive_counts = ops.sum(y_true, axis=[0, 1, 2])

        # Calculate total pixels (B * H * W)
        input_shape = ops.shape(y_true)
        # Calculate product of spatial dimensions + batch
        # Note: input_shape is a tensor in symbolic execution
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        total_pixels = ops.cast(batch_size * height * width, positive_counts.dtype)

        # Adaptive Logic: Weight = sqrt(presence_ratio) + 0.1
        # Add epsilon to denominator to prevent division by zero
        presence_ratio = positive_counts / (total_pixels + 1e-7)
        adaptive_weights = ops.sqrt(presence_ratio) + 0.1

        # Apply user-provided static channel weights if present
        if self.channel_weights is not None:
            user_weights = ops.convert_to_tensor(self.channel_weights, dtype=adaptive_weights.dtype)
            adaptive_weights = adaptive_weights * user_weights

        # Normalize weights so they sum to 1
        weights_sum = ops.sum(adaptive_weights) + 1e-7
        weights_normalized = adaptive_weights / weights_sum

        # 3. Aggregate Loss
        # -----------------
        # First, compute mean loss for each channel separately (average over B, H, W)
        # Result Shape: (Channels,)
        channel_mean_losses = ops.mean(loss_tensor, axis=[0, 1, 2])

        # Then compute weighted sum over channels using the normalized adaptive weights
        # Result Shape: Scalar
        total_loss = ops.sum(channel_mean_losses * weights_normalized)

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()

        # Serialize the base loss if it's an object
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
    def from_config(cls, config: Dict[str, Any]) -> 'PerChannelBinaryLoss':
        # Deserialize the base loss if it exists as a dictionary config
        if 'base_loss' in config and isinstance(config['base_loss'], dict):
            config['base_loss'] = keras.saving.deserialize_keras_object(config['base_loss'])
        return cls(**config)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class WeightedBinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss with class weighting for imbalanced segmentation.

    The Focal Loss formula is defined as:

    .. math::
        FL(p_t) = -\\alpha (1 - p_t)^{\\gamma} \\log(p_t)

    This implementation returns the element-wise loss (no reduction) to allow
    wrappers (like :class:`PerChannelBinaryLoss`) to handle adaptive weighting
    and reduction logic.

    Parameters
    ----------
    alpha : float, default=0.25
        The balancing factor for positive vs negative classes.
        Commonly used ranges are 0.25 to 0.75.
    gamma : float, default=2.0
        The focusing parameter. Higher values reduce the loss contribution
        of easy-to-classify examples.
    from_logits : bool, default=False
        Whether the input predictions are logits or probabilities.
    reduction : str, default='sum_over_batch_size'
        Standard Keras loss reduction argument.
    name : str, default='weighted_binary_focal_loss'
        The name of the loss.
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

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute focal loss element-wise.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth tensor.
        y_pred : keras.KerasTensor
            Prediction tensor.

        Returns
        -------
        keras.KerasTensor
            Element-wise loss tensor with the same shape as inputs.
        """
        # Convert logits to probabilities if necessary
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)

        # Clip predictions to prevent log(0) and log(1)
        epsilon = keras.backend.epsilon()
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # Ensure consistent types
        y_true = ops.cast(y_true, y_pred.dtype)

        # Calculate loss for positive examples (y_true = 1)
        # Loss = -alpha * (1 - p)^gamma * log(p)
        p_t_pos = y_pred
        alpha_t_pos = self.alpha
        loss_pos = -alpha_t_pos * ops.power(1.0 - p_t_pos, self.gamma) * ops.log(p_t_pos)

        # Calculate loss for negative examples (y_true = 0)
        # Loss = -(1-alpha) * p^gamma * log(1-p)
        p_t_neg = 1.0 - y_pred
        alpha_t_neg = 1.0 - self.alpha
        loss_neg = -alpha_t_neg * ops.power(1.0 - p_t_neg, self.gamma) * ops.log(p_t_neg)

        # Combine losses based on ground truth masks
        loss = y_true * loss_pos + (1.0 - y_true) * loss_neg

        # Return element-wise loss (Do not reduce here, wrapper handles it)
        return loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DiceLossPerChannel(keras.losses.Loss):
    """
    Dice Loss applied per channel for multi-label segmentation.

    The Dice Coefficient is defined as:

    .. math::
        Dice = \\frac{2 |X \\cap Y|}{|X| + |Y| + \\epsilon}

    The loss is defined as :math:`1 - Dice`. This implementation computes
    the metric for each channel over the spatial dimensions and the batch,
    then averages the results.

    Parameters
    ----------
    smooth : float, default=1.0
        Smoothing factor (Laplace smoothing) to avoid division by zero
        and reduce overfitting.
    reduction : str, default='sum_over_batch_size'
        Standard Keras loss reduction argument.
    name : str, default='dice_loss_per_channel'
        Name of the loss.
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

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute Dice loss.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth tensor. Shape: (Batch, H, W, C)
        y_pred : keras.KerasTensor
            Prediction tensor. Shape: (Batch, H, W, C)

        Returns
        -------
        keras.KerasTensor
            Scalar loss value.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        # Get shape info
        input_shape = ops.shape(y_true)
        batch_size = input_shape[0]
        num_channels = input_shape[-1]

        # Flatten spatial dimensions but keep batch and channels separate
        # Target Shape: (batch, height*width, channels)
        y_true_flat = ops.reshape(y_true, (batch_size, -1, num_channels))
        y_pred_flat = ops.reshape(y_pred, (batch_size, -1, num_channels))

        # Compute per-channel intersection and union
        # Sum over the spatial dimension (axis 1)
        intersection = ops.sum(y_true_flat * y_pred_flat, axis=1)

        # Denominator: sum(y_true) + sum(y_pred)
        union = ops.sum(y_true_flat, axis=1) + ops.sum(y_pred_flat, axis=1)

        # Compute Dice Coefficient
        smooth_tensor = ops.convert_to_tensor(self.smooth, dtype="float32")
        dice = (2.0 * intersection + smooth_tensor) / (union + smooth_tensor)

        # Average over batch and channels to get scalar loss
        dice_mean = ops.mean(dice)

        return 1.0 - dice_mean

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['smooth'] = self.smooth
        return config

# ---------------------------------------------------------------------

def create_multilabel_segmentation_loss(
    loss_type: str = 'focal',
    alpha: float = 0.75,
    gamma: float = 2.0,
    channel_weights: Optional[List[float]] = None,
    smooth: float = 1.0
) -> keras.losses.Loss:
    """
    Factory function to create appropriate loss for multi-label segmentation.

    Parameters
    ----------
    loss_type : str
        Type of loss to create. Options:
        - 'focal': Weighted Binary Focal Loss (default)
        - 'dice': Dice Loss
        - 'bce': Standard Binary Crossentropy
        - 'weighted_bce': Binary Crossentropy with fixed positive class weighting
    alpha : float, default=0.75
        Balancing factor for Focal Loss or Weighted BCE.
    gamma : float, default=2.0
        Focusing parameter for Focal Loss.
    channel_weights : Optional[List[float]], default=None
        Per-channel weights for the PerChannel wrapper.
    smooth : float, default=1.0
        Smoothing factor for Dice Loss.

    Returns
    -------
    keras.losses.Loss
        The configured loss instance.

    Raises
    ------
    ValueError
        If an unknown ``loss_type`` is provided.
    """
    if loss_type == 'focal':
        base_loss = WeightedBinaryFocalLoss(alpha=alpha, gamma=gamma, from_logits=False)
        return PerChannelBinaryLoss(base_loss, channel_weights=channel_weights)

    elif loss_type == 'dice':
        return DiceLossPerChannel(smooth=smooth)

    elif loss_type == 'bce':
        return PerChannelBinaryLoss('binary_crossentropy', channel_weights=channel_weights)

    elif loss_type == 'weighted_bce':
        # Define internal function using pure Keras ops
        def weighted_bce_fn(y_true, y_pred):
            # Calculate weight for positive class
            # if alpha=0.5 -> weight=1.0, if alpha=0.75 -> weight=3.0
            if alpha != 1.0:
                pos_weight = alpha / (1.0 - alpha)
            else:
                pos_weight = 10.0  # Fallback cap for stability

            # Clip for numerical stability
            epsilon = keras.backend.epsilon()
            y_pred_safe = ops.clip(y_pred, epsilon, 1.0 - epsilon)

            # Weighted BCE formula
            loss = -pos_weight * y_true * ops.log(y_pred_safe) - (1.0 - y_true) * ops.log(1.0 - y_pred_safe)
            return loss

        # Return wrapped function
        return PerChannelBinaryLoss(weighted_bce_fn, channel_weights=channel_weights)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

# ---------------------------------------------------------------------
