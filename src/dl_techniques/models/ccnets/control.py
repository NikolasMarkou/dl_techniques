import tensorflow as tf
from typing import Dict
from abc import ABC, abstractmethod
from collections import defaultdict

# ---------------------------------------------------------------------

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
            metrics: A dictionary containing at least 'batch_accuracy' and model errors.

        Returns:
            True if the Reasoner should be trained, False otherwise.
        """
        pass

# ---------------------------------------------------------------------

class StaticThresholdStrategy(ConvergenceControlStrategy):
    """
    The default strategy: trains the Reasoner as long as its accuracy
    is below a fixed threshold. This encapsulates the original behavior.
    """
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    def update_state(self, metrics: Dict[str, tf.Tensor]):
        """This strategy is stateless."""
        pass

    def should_train_reasoner(self, metrics: Dict[str, tf.Tensor]) -> bool:
        """Decision is based solely on batch accuracy."""
        return metrics['batch_accuracy'] < self.threshold

# ---------------------------------------------------------------------

class AdaptiveDivergenceStrategy(ConvergenceControlStrategy):
    """
    An adaptive strategy that throttles the Reasoner if it converges
    significantly faster than the other modules, preventing the system
    from settling in a suboptimal, low-complexity equilibrium.
    """
    def __init__(
            self,
            ema_alpha: float = 0.1,
            divergence_threshold: float = 0.7,
            patience: int = 10,
            static_ceiling: float = 0.99
    ):
        """
        Args:
            ema_alpha: Smoothing factor for the exponential moving average of errors.
            divergence_threshold: If Reasoner EMA error is below this ratio of the
                                  other modules' average error, it is considered divergent.
            patience: Number of consecutive steps the divergence condition must be
                      met before throttling begins.
            static_ceiling: A hard accuracy ceiling to always enforce.
        """
        self.ema_alpha = ema_alpha
        self.divergence_threshold = divergence_threshold
        self.patience = patience
        self.static_ceiling = static_ceiling
        self.error_emas = defaultdict(lambda: 1.0)
        self.divergence_counter = 0

    def update_state(self, metrics: Dict[str, tf.Tensor]):
        """Update the exponential moving averages of the model errors."""
        for key in ['explainer_error', 'reasoner_error', 'producer_error']:
            current_val = metrics[key].numpy()
            self.error_emas[key] = (
                self.ema_alpha * current_val + (1 - self.ema_alpha) * self.error_emas[key]
            )

    def should_train_reasoner(self, metrics: Dict[str, tf.Tensor]) -> bool:
        """
        Decision is based on a combination of a static accuracy ceiling and
        the relative convergence rates of the modules.
        """
        # First, enforce the hard static ceiling.
        if metrics['batch_accuracy'] >= self.static_ceiling:
            return False

        # Calculate the average error of non-Reasoner modules.
        # Add epsilon for numerical stability.
        epsilon = 1e-8
        explainer_ema = self.error_emas['explainer_error']
        producer_ema = self.error_emas['producer_error']
        non_reasoner_avg_error = (explainer_ema + producer_ema) / 2.0

        # Calculate the divergence ratio.
        reasoner_ema = self.error_emas['reasoner_error']
        divergence_ratio = reasoner_ema / (non_reasoner_avg_error + epsilon)

        # Check if the Reasoner is diverging (converging too fast).
        if divergence_ratio < self.divergence_threshold:
            self.divergence_counter += 1
        else:
            # If the condition is not met, reset the counter.
            self.divergence_counter = 0

        # The Reasoner should train only if the patience counter has not been met.
        return self.divergence_counter < self.patience

# ---------------------------------------------------------------------
