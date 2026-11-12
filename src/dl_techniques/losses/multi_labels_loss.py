"""
Per-Channel Loss Wrapper for Multi-Label Segmentation
======================================================

This module provides a robust loss wrapper and specific loss implementations
designed for multi-label segmentation tasks. In such tasks, each channel
represents an independent binary classification problem (e.g., overlapping
objects, independent attributes).

The core component is ``PerChannelBinaryLoss``, which allows computing
losses independently per channel and aggregating them using adaptive weights
based on class presence, stabilizing training for imbalanced datasets. ``call``
returns one value per image; the weights are a batch statistic, so the loss is
batch-coupled (see its docstring).

``WeightedBinaryFocalLoss`` is a two-sided focal loss (positive + negative
terms) that returns element-wise values for the wrapper to reduce.

``WeightedBinaryCrossEntropyLoss`` is the element-wise positive-weighted BCE
behind the factory's ``weighted_bce`` option (a registered class, so it saves
and loads).

``DiceLossPerChannel`` is a thin serializable ``Loss`` over the shared backend in
``segmentation_metrics.dice_per_sample``; ``call`` returns one value per image and the
framework reduction takes the batch mean. It does not go through
``PerChannelBinaryLoss``.
"""

import keras
from keras import ops
from typing import Optional, List, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.losses._per_sample import PerSampleCallMixin
from dl_techniques.losses.segmentation_metrics import dice_per_sample


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.multi_labels_loss")
class PerChannelBinaryLoss(PerSampleCallMixin, keras.losses.Loss):
    """
    Wrapper that applies a binary loss function independently per channel.

    For multi-label segmentation with :math:`C` channels, this layer:

    1. Computes element-wise binary loss for all channels (Vectorized).
    2. Computes adaptive weights based on class presence within the batch.
    3. Averages each image's loss over its pixels, per channel, then sums the
       channels with the adaptive weights: one value per image.

    The adaptive weighting formula is:

    .. math::
        w_c = \sqrt{\frac{\text{count}_c}{\text{total\_pixels}}} + 0.1

    This ensures that rare classes are not completely overwhelmed but also
    prevents common classes from dominating the gradient purely by volume.

    ``call`` returns ``(batch,)``, so ``reduction`` and ``sample_weight`` act on
    images. The result is **batch-coupled**: the channel weights derive from the
    class-presence counts of the WHOLE batch (they are not per-image weights, which
    would be a different objective), and ``call`` never sees ``sample_weight``. A zero
    ``sample_weight`` therefore removes an image's loss but not its class-presence
    counts from the weights, so ``loss(w=[1, ..., 0]) * B / (B - 1)`` is NOT
    ``loss(batch[:-1])`` (measured gap 6.6e-4 on a 4-image batch with a ragged
    foreground, 1.55e-3 when the zero-weighted image is all-positive in one
    channel). The mean over images equals the scalar this class returned before it
    was made per-sample.

    A Keras mask on ``y_pred`` (a model output left masked by a ``Masking`` or a
    recurrent layer) has no meaning here: ``__call__`` raises ``ValueError`` for a
    mask of rank 2 or more, because Keras would multiply it into the ``(batch,)``
    rows (a shape error, or silent mis-weighting when the batch equals the mask's
    second axis). A ``(batch,)`` mask acts as a 0/1 ``sample_weight``. A
    ``sample_weight`` with an axis beyond the batch also raises ``ValueError``. A
    direct ``call`` reads neither.

    Parameters
    ----------
    base_loss : Union[keras.losses.Loss, str]
        The binary loss to apply per channel. Either the string
        ``'binary_crossentropy'`` (computed element-wise here) or a ``Loss``
        whose ``call`` returns one value per element, with the same rank as
        ``y_true`` (e.g. :class:`WeightedBinaryFocalLoss`,
        :class:`WeightedBinaryCrossEntropyLoss`). Any other string, and any
        ``call`` that reduces the channel axis, is rejected.
    channel_weights : Optional[List[float]], default=None
        Static manual weights for each channel. If provided, these are
        multiplied with the adaptive weights. Useful for enforcing
        domain-specific importance.
    reduction : str, default='sum_over_batch_size'
        Type of reduction applied to the ``(batch,)`` per-image values (after any
        ``sample_weight``): ``'sum_over_batch_size'`` averages them, ``'sum'`` adds
        them, ``'none'`` returns them.
    name : str, default='per_channel_loss'
        The name of the loss instance.

    Raises
    ------
    ValueError
        At construction, if ``base_loss`` is a string other than
        ``'binary_crossentropy'``; at ``call``, if the base loss returns a tensor
        whose rank differs from ``y_true``.
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
            # DECISION plan-2026-11-11T131349-13b28b37/D-009
            # A stock string alias ("mse", "binary_focal_crossentropy", ...) maps to
            # a function that reduces the channel axis, so loss_tensor is (B, H, W):
            # the per-channel mean below collapses to one scalar, the normalized
            # weights (which sum to 1) multiply it back to itself, and
            # channel_weights changes nothing (measured: 0.33678 for [10, 1, .1] and
            # for None). Do NOT "support" strings again by falling back to
            # keras.losses.get; pass a Loss that returns element-wise values.
            # Guard: tests/test_losses/test_multi_labels_loss.py::
            # TestPerChannelBinaryLossRefusesABaseThatCollapsesChannels
            if base_loss != 'binary_crossentropy':
                raise ValueError(
                    f"base_loss string must be 'binary_crossentropy' (the only alias "
                    f"computed element-wise here), got {base_loss!r}; pass a Loss "
                    f"whose call returns one value per element instead."
                )
            self.base_loss_name = base_loss
            self.base_loss = None
        else:
            self.base_loss = base_loss
            self.base_loss_name = base_loss.__class__.__name__

        self.channel_weights = channel_weights

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute loss independently per channel with adaptive weighting.

        One value per sample: each image's pixel-mean loss per channel, summed
        over channels with weights derived from the whole batch (batch-coupled).

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
            One value per sample, shape ``(batch_size,)``: the channel-weighted sum
            of that image's per-channel pixel-mean loss.
        """
        # Ensure consistent dtypes
        y_true = ops.cast(y_true, y_pred.dtype)

        # 1. Compute Element-wise Loss (Batch, Height, Width, Channels)
        # -------------------------------------------------------------
        if self.base_loss is None:
            # Manual element-wise BCE to ensure no implicit reduction occurs
            # before we want it to.
            epsilon = keras.config.epsilon()
            y_pred_safe = ops.clip(y_pred, epsilon, 1.0 - epsilon)

            # BCE = -[y * log(p) + (1-y) * log(1-p)]
            bce = y_true * ops.log(y_pred_safe) + (1.0 - y_true) * ops.log(1.0 - y_pred_safe)
            loss_tensor = -bce

        else:
            # Handle both Loss instances and plain callables
            if hasattr(self.base_loss, 'call'):
                loss_tensor = self.base_loss.call(y_true, y_pred)
            else:
                loss_tensor = self.base_loss(y_true, y_pred)

            # Static ranks, not tensor values: a base that reduced an axis would be
            # averaged over the wrong axes below and silently ignore channel_weights.
            if len(loss_tensor.shape) != len(y_true.shape):
                raise ValueError(
                    f"base_loss {self.base_loss_name} returned rank "
                    f"{len(loss_tensor.shape)} (shape {tuple(loss_tensor.shape)}) for "
                    f"y_true of rank {len(y_true.shape)}; it must return one value per "
                    f"element so the per-channel weights can be applied."
                )
            # A size-1 channel axis has the right rank but broadcasts against the
            # per-channel weights below, which sum to 1, so channel_weights is a
            # no-op again (measured: 0.29935 for [10, 1, .1] and for None). Only
            # static sizes are compared; a symbolic width cannot be judged here.
            loss_width, label_width = loss_tensor.shape[-1], y_true.shape[-1]
            if (
                isinstance(loss_width, int)
                and isinstance(label_width, int)
                and loss_width != label_width
            ):
                raise ValueError(
                    f"base_loss {self.base_loss_name} returned channel axis of size "
                    f"{loss_width} for y_true with {label_width} channels (shape "
                    f"{tuple(loss_tensor.shape)}); it must keep one value per channel "
                    f"so the per-channel weights can be applied."
                )

        # 2. Compute Adaptive Channel Weights (Vectorized)
        # ------------------------------------------------
        # DECISION plan-2026-11-11T223825-64d1b401/D-002
        # These weights are deliberately a BATCH statistic: the class-presence counts
        # are summed over the whole batch, so the loss is batch-coupled and a zero
        # sample_weight does not equal dropping the row. Do NOT make them per-sample
        # (sqrt of each image's own presence) to "decouple" the rows: that is a
        # different objective, not the same loss per row, and it changes the value of
        # the default reduction. Do NOT recompute them over the rows a sample_weight
        # keeps either; call() never sees sample_weight.
        # Guard: tests/test_losses/test_multi_labels_loss.py::
        # TestPerChannelBinaryLossIsPerSample::
        # test_a_zero_sample_weight_keeps_the_row_s_class_counts_in_the_weights
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
        # First, compute each image's mean loss per channel (average over H, W)
        # Result Shape: (Batch, Channels)
        per_sample_channel_losses = ops.mean(loss_tensor, axis=[1, 2])

        # Then compute the weighted sum over channels using the normalized adaptive
        # weights; the mean over the batch of this is the old batch scalar.
        # Result Shape: (Batch,)
        return ops.sum(per_sample_channel_losses * weights_normalized, axis=-1)

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
        config = dict(config)  # never mutate the caller's dict
        # Deserialize the base loss if it exists as a dictionary config
        if 'base_loss' in config and isinstance(config['base_loss'], dict):
            config['base_loss'] = keras.saving.deserialize_keras_object(config['base_loss'])
        return cls(**config)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.multi_labels_loss")
class WeightedBinaryCrossEntropyLoss(keras.losses.Loss):
    r"""
    Element-wise binary cross-entropy with a weight on the positive term.

    .. math::
        \ell = -w_{pos}\, y \log(p) - (1 - y) \log(1 - p)

    Returns the element-wise loss (no reduction), like
    :class:`WeightedBinaryFocalLoss`, so :class:`PerChannelBinaryLoss` can apply its
    per-channel weighting. It exists as a registered class because the closure it
    replaced in ``create_multilabel_segmentation_loss`` serialized as an unloadable
    builtin; the factory is its only caller.

    Parameters
    ----------
    pos_weight : float, default=1.0
        Multiplier on the positive-class term. Must be positive.
    reduction : str, default='sum_over_batch_size'
        Standard Keras loss reduction argument.
    name : str, default='weighted_binary_crossentropy'
        The name of the loss.

    Raises
    ------
    ValueError
        If ``pos_weight`` is not positive.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        reduction: str = 'sum_over_batch_size',
        name: str = 'weighted_binary_crossentropy',
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        if pos_weight <= 0:
            raise ValueError(f"pos_weight must be positive, got {pos_weight}")
        self.pos_weight = float(pos_weight)

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute the weighted BCE element-wise (probabilities in, same shape out).

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth tensor.
        y_pred : keras.KerasTensor
            Predicted probabilities, same shape as ``y_true``.

        Returns
        -------
        keras.KerasTensor
            Element-wise loss tensor with the same shape as the inputs.
        """
        y_true = ops.cast(y_true, y_pred.dtype)
        epsilon = keras.config.epsilon()
        y_pred_safe = ops.clip(y_pred, epsilon, 1.0 - epsilon)
        return (
            -self.pos_weight * y_true * ops.log(y_pred_safe)
            - (1.0 - y_true) * ops.log(1.0 - y_pred_safe)
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['pos_weight'] = self.pos_weight
        return config

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.multi_labels_loss")
class WeightedBinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss with class weighting for imbalanced segmentation.

    The Focal Loss formula is defined as:

    .. math::
        FL(p_t) = -\alpha (1 - p_t)^{\gamma} \log(p_t)

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
        epsilon = keras.config.epsilon()
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


@register_dl_technique("dl_techniques.losses.multi_labels_loss")
class DiceLossPerChannel(PerSampleCallMixin, keras.losses.Loss):
    """
    Dice Loss for multi-label segmentation using the shared backend.

    This is a thin wrapper around ``segmentation_metrics.dice_per_sample`` that
    provides a serializable Keras Loss class. ``call`` returns one value per image
    (the mean over channels of the per-channel Dice loss), so ``sample_weight`` and
    ``reduction`` act on images; nothing in it is a batch statistic, so zeroing an
    image's weight equals dropping that image.

    A Keras mask on ``y_pred`` (a model output left masked by a ``Masking`` or a
    recurrent layer) has no meaning here: ``__call__`` raises ``ValueError`` for a
    mask of rank 2 or more, because Keras would multiply it into the ``(batch,)``
    rows (a shape error, or silent mis-weighting when the batch equals the mask's
    second axis). A ``(batch,)`` mask acts as a 0/1 ``sample_weight``. A
    ``sample_weight`` with an axis beyond the batch also raises ``ValueError``. A
    direct ``call`` reads neither.

    Parameters
    ----------
    smooth : float, default=1.0
        Smoothing factor (Laplace smoothing) to avoid division by zero.
    reduction : str, default='sum_over_batch_size'
        Standard Keras loss reduction argument. It acts on the per-image values
        returned by ``call``.
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
        Compute Dice loss using shared per-sample backend.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Ground truth tensor. Shape: (Batch, H, W, C)
        y_pred : keras.KerasTensor
            Prediction tensor. Shape: (Batch, H, W, C)

        Returns
        -------
        keras.KerasTensor
            Dice loss per image, shape ``(Batch,)``, float32 whatever the input dtype
            (a direct call on float16, bfloat16 or float64 input used to raise).
        """
        # DECISION plan-2026-11-11T223825-64d1b401/D-015
        # ``dice_per_sample`` computes in float32 (it casts ``y_true``): meet it there, so a
        # direct call on half or double precision input no longer raises. A no-op for
        # float32. Do NOT drop the cast to "keep the caller's dtype": that is the raise.
        y_pred = ops.cast(y_pred, "float32")
        return dice_per_sample(y_true, y_pred, from_logits=False, smooth=self.smooth)

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
        - 'dice': Dice Loss (uses shared per-sample backend)
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
        # alpha=0.5 -> pos_weight 1.0, alpha=0.75 -> 3.0; alpha=1 is capped at 10.0
        pos_weight = alpha / (1.0 - alpha) if alpha != 1.0 else 10.0
        return PerChannelBinaryLoss(
            WeightedBinaryCrossEntropyLoss(pos_weight=pos_weight),
            channel_weights=channel_weights,
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

# ---------------------------------------------------------------------