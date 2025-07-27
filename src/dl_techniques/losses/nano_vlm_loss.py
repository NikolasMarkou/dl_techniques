"""NanoVLM Loss Function for Vision-Language Model Training.

This module implements the autoregressive language modeling loss function
specifically designed for nanoVLM training with proper masking and label smoothing.
"""

import keras
from keras import ops
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class NanoVLMLoss(keras.losses.Loss):
    """Autoregressive language modeling loss for vision-language training.

    This loss function implements the standard autoregressive language modeling
    objective with proper token masking for vision-language models. It computes
    the cross-entropy loss between predicted and target tokens while ignoring
    padding tokens and applying optional label smoothing.

    The loss shifts the prediction and target sequences to implement the
    autoregressive objective: predict the next token given all previous tokens.

    Args:
        ignore_index: Token index to ignore in loss computation. Typically
            used for padding tokens. Default: 0.
        label_smoothing: Label smoothing factor between 0.0 and 1.0.
            Higher values apply more smoothing. Default: 0.0.
        from_logits: Whether predictions are logits or probabilities.
            Default: True.
        reduction: Type of reduction to apply to loss. One of 'sum_over_batch_size',
            'sum', or 'none'. Default: 'sum_over_batch_size'.
        name: Name of the loss function. Default: "nanovlm_loss".
        **kwargs: Additional keyword arguments for Loss base class.

    Example:
        >>> loss_fn = NanoVLMLoss(ignore_index=0, label_smoothing=0.1)
        >>> # y_true: [batch_size, seq_len] - target token IDs
        >>> # y_pred: [batch_size, seq_len, vocab_size] - predicted logits
        >>> loss = loss_fn(y_true, y_pred)

    Note:
        The input sequences are automatically shifted for autoregressive training:
        - Predictions use tokens [0:seq_len-1] to predict [1:seq_len]
        - This implements the standard language modeling objective
    """

    def __init__(
            self,
            ignore_index: int = 0,
            label_smoothing: float = 0.0,
            from_logits: bool = True,
            reduction: str = "sum_over_batch_size",
            name: str = "nanovlm_loss",
            **kwargs
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)

        # Validate parameters
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(
                f"label_smoothing must be between 0.0 and 1.0, got {label_smoothing}"
            )

        if ignore_index < 0:
            logger.warning(
                f"ignore_index is negative ({ignore_index}). "
                "This may cause unexpected behavior."
            )

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits

        # Initialize the underlying sparse categorical crossentropy loss
        self.sparse_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            reduction='none',  # We handle reduction manually for masking
            label_smoothing=label_smoothing
        )

        logger.info(
            f"Initialized {self.__class__.__name__} with ignore_index={ignore_index}, "
            f"label_smoothing={label_smoothing}, from_logits={from_logits}"
        )

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute the autoregressive language modeling loss.

        Args:
            y_true: Target token IDs of shape [batch_size, seq_len].
                Should contain integer token indices.
            y_pred: Predicted logits or probabilities of shape
                [batch_size, seq_len, vocab_size].

        Returns:
            Scalar loss value averaged over non-masked tokens.

        Raises:
            ValueError: If input shapes are incompatible.
        """
        # Validate input shapes
        y_true_shape = ops.shape(y_true)
        y_pred_shape = ops.shape(y_pred)

        # Check that sequence lengths match
        if len(y_true_shape) != 2 or len(y_pred_shape) != 3:
            raise ValueError(
                f"Expected y_true shape [batch, seq_len] and y_pred shape "
                f"[batch, seq_len, vocab_size], got {y_true_shape} and {y_pred_shape}"
            )

        seq_len_true = y_true_shape[1]
        seq_len_pred = y_pred_shape[1]

        if seq_len_true != seq_len_pred:
            raise ValueError(
                f"Sequence length mismatch: y_true has {seq_len_true}, "
                f"y_pred has {seq_len_pred}"
            )

        # Ensure minimum sequence length for autoregressive training
        if seq_len_true < 2:
            logger.warning(
                f"Sequence length is {seq_len_true}, which is too short for "
                "autoregressive training. Consider using longer sequences."
            )
            return ops.zeros([], dtype=y_pred.dtype)

        # Shift sequences for autoregressive prediction
        # Predict next token given previous tokens
        y_pred_shifted = y_pred[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        y_true_shifted = y_true[:, 1:]  # [batch, seq_len-1]

        # Flatten for loss computation
        batch_size = y_pred_shape[0]
        vocab_size = y_pred_shape[2]
        seq_len_shifted = seq_len_true - 1

        y_pred_flat = ops.reshape(
            y_pred_shifted,
            [batch_size * seq_len_shifted, vocab_size]
        )
        y_true_flat = ops.reshape(
            y_true_shifted,
            [batch_size * seq_len_shifted]
        )

        # Compute per-token loss
        loss_per_token = self.sparse_ce(y_true_flat, y_pred_flat)

        # Create mask to ignore specified tokens (e.g., padding)
        mask = ops.cast(
            ops.not_equal(y_true_flat, self.ignore_index),
            dtype=loss_per_token.dtype
        )

        # Apply mask to loss
        masked_loss = loss_per_token * mask

        # Compute final loss based on reduction type
        total_loss = ops.sum(masked_loss)
        total_valid_tokens = ops.sum(mask)

        # Avoid division by zero
        mean_loss = ops.where(
            ops.greater(total_valid_tokens, 0),
            total_loss / ops.maximum(total_valid_tokens, 1.0),
            ops.zeros_like(total_loss)
        )

        # Handle different reduction types
        if self.reduction == "sum":
            return total_loss
        elif self.reduction == "none":
            # Return per-sample loss (reshape back to [batch_size, seq_len-1])
            return ops.reshape(masked_loss, [batch_size, seq_len_shifted])
        else:  # sum_over_batch_size
            return mean_loss

    def get_config(self) -> dict:
        """Get the loss function configuration.

        Returns:
            Dictionary containing the configuration parameters.
        """
        config = super().get_config()
        config.update({
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "NanoVLMLoss":
        """Create loss function from configuration.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New NanoVLMLoss instance.
        """
        return cls(**config)