import tensorflow as tf
from typing import Dict
from abc import ABC, abstractmethod

class ConvergenceControlStrategy(ABC):
    """Abstract base class for Reasoner training control strategies."""

    @abstractmethod
    def update_state(self, metrics: Dict[str, tf.Tensor]):
        """Update the internal state of the strategy with the latest metrics."""
        pass

    @abstractmethod
    def should_train_reasoner(self, metrics: Dict[str, tf.Tensor]) -> bool:
        """
        Determine if the Reasoner should be trained in the current step.

        Args:
            metrics: A dictionary containing at least 'batch_accuracy'.

        Returns:
            True if the Reasoner should be trained, False otherwise.
        """
        pass


class StaticThresholdStrategy(ConvergenceControlStrategy):
    """
    The default strategy: trains the Reasoner as long as its accuracy
    is below a fixed threshold. This encapsulates the original behavior.
    """

    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    def update_state(self, metrics: Dict[str, tf.Tensor]):
        """This strategy is stateless and does not require updates."""
        pass

    def should_train_reasoner(self, metrics: Dict[str, tf.Tensor]) -> bool:
        """Decision is based solely on batch accuracy."""
        return metrics['batch_accuracy'] < self.threshold


class AdaptiveDivergenceStrategy(ConvergenceControlStrategy):
    """
    An adaptive strategy that throttles the Reasoner if it converges
    significantly faster than the other modules. This version is
    graph-compatible, using tf.Variables for state.
    """

    def __init__(
            self,
            ema_alpha: float = 0.1,
            divergence_threshold: float = 0.7,
            patience: int = 10,
            static_ceiling: float = 0.99
    ):
        self.ema_alpha = tf.constant(ema_alpha, dtype=tf.float32)
        self.divergence_threshold = tf.constant(divergence_threshold, dtype=tf.float32)
        self.patience_limit = tf.constant(patience, dtype=tf.int32)
        self.static_ceiling = tf.constant(static_ceiling, dtype=tf.float32)

        # Use tf.Variable to manage state in a graph-compatible way
        self.error_emas = {
            'explainer_error': tf.Variable(1.0, dtype=tf.float32, trainable=False),
            'reasoner_error': tf.Variable(1.0, dtype=tf.float32, trainable=False),
            'producer_error': tf.Variable(1.0, dtype=tf.float32, trainable=False)
        }
        self.divergence_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

    def update_state(self, metrics: Dict[str, tf.Tensor]):
        """
        Update the EMAs. This method should be called from an EAGER context
        (e.g., the Python training loop in the Trainer).
        """
        for key, ema_var in self.error_emas.items():
            current_val = metrics[key]
            new_ema = (self.ema_alpha * current_val) + (1.0 - self.ema_alpha) * ema_var
            ema_var.assign(new_ema)

    def should_train_reasoner(self, metrics: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Decision logic written with pure TensorFlow ops to be graph-compatible.
        """
        # Enforce the hard static ceiling first
        if metrics['batch_accuracy'] >= self.static_ceiling:
            self.divergence_counter.assign(0)  # Reset counter
            return tf.constant(False)

        epsilon = 1e-8
        non_reasoner_avg_error = (self.error_emas['explainer_error'] + self.error_emas['producer_error']) / 2.0
        divergence_ratio = self.error_emas['reasoner_error'] / (non_reasoner_avg_error + epsilon)

        # Update divergence counter using tf.cond for graph compatibility
        def increment_counter():
            return self.divergence_counter.assign_add(1)

        def reset_counter():
            return self.divergence_counter.assign(0)

        tf.cond(divergence_ratio < self.divergence_threshold, increment_counter, reset_counter)

        # Return True if the patience limit has not been reached
        return self.divergence_counter < self.patience_limit