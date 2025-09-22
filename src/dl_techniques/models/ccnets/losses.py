import keras
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------

class LossFunction(ABC):
    """Abstract base class for loss functions used in CCNet."""

    @abstractmethod
    def __call__(self, y_true: keras.ops.array, y_pred: keras.ops.array) -> keras.ops.array:
        """
        Compute loss between true and predicted values.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Scalar loss value.
        """
        pass

# ---------------------------------------------------------------------

class L1Loss(LossFunction):
    """L1 (Mean Absolute Error) loss function."""

    def __call__(self, y_true: keras.ops.array, y_pred: keras.ops.array) -> keras.ops.array:
        """Compute L1 loss."""
        return keras.ops.mean(keras.ops.abs(y_true - y_pred))

# ---------------------------------------------------------------------

class L2Loss(LossFunction):
    """L2 (Mean Squared Error) loss function."""

    def __call__(self, y_true: keras.ops.array, y_pred: keras.ops.array) -> keras.ops.array:
        """Compute L2 loss."""
        return keras.ops.mean(keras.ops.square(y_true - y_pred))

# ---------------------------------------------------------------------

class HuberLoss(LossFunction):
    """Huber loss function (smooth L1)."""

    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss.

        Args:
            delta: Threshold at which to switch from L2 to L1 loss.
        """
        self.delta = delta

    def __call__(self, y_true: keras.ops.array, y_pred: keras.ops.array) -> keras.ops.array:
        """Compute Huber loss."""
        error = y_true - y_pred
        abs_error = keras.ops.abs(error)
        quadratic = keras.ops.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        return keras.ops.mean(0.5 * keras.ops.square(quadratic) + self.delta * linear)

# ---------------------------------------------------------------------