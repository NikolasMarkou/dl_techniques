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

# ADDED: New PolynomialLoss class for flexible error penalization
class PolynomialLoss(LossFunction):
    """
    A generalized loss function that computes the mean of the absolute error
    raised to a power 'p'. This provides a flexible way to control the
    system's aversion to large errors.
    - p=1.0 recovers L1 loss.
    - p=2.0 recovers L2 loss.
    - p>2.0 creates hyper-aversion to large errors.
    - 0<p<1.0 creates hyper-tolerance to large errors.
    """
    def __init__(self, p: float = 2.0):
        """
        Initialize Polynomial loss.

        Args:
            p: The exponent to which the absolute error is raised. Must be > 0.
        """
        if p <= 0:
            raise ValueError("Exponent 'p' must be positive for PolynomialLoss.")
        self.p = p

    def __call__(self, y_true: keras.ops.array, y_pred: keras.ops.array) -> keras.ops.array:
        """Compute Polynomial loss."""
        error = keras.ops.abs(y_true - y_pred)
        # Add a small epsilon for numerical stability if p is close to 0 or negative
        # although we already guard for p > 0.
        epsilon = 1e-8
        powered_error = keras.ops.power(error + epsilon, self.p)
        return keras.ops.mean(powered_error)

# ---------------------------------------------------------------------
