import keras
from typing import Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import CCNetModule
from .orchestrators import CCNetOrchestrator

# ---------------------------------------------------------------------

class EarlyStoppingCallback:
    """
    Early stopping with dual conditions: convergence of model errors and
    stagnation of gradient flow, indicating that learning has ceased.
    """

    def __init__(
            self,
            patience: int = 5,
            error_threshold: float = 1e-4,
            grad_stagnation_threshold: Optional[float] = 1e-3
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            error_threshold: Minimum change in total error to qualify as improvement.
            grad_stagnation_threshold: If the sum of gradient norms falls below this
                                       value, it's considered stagnation.
        """
        self.patience = patience
        self.error_threshold = error_threshold
        self.grad_stagnation_threshold = grad_stagnation_threshold
        self.best_error = float('inf')
        self.wait_error = 0
        self.wait_grad = 0

    def __call__(
            self,
            epoch: int,
            metrics: Dict[str, float],
            orchestrator: CCNetOrchestrator
    ):
        """Check for early stopping conditions."""
        # Condition 1: Check for convergence based on total model error
        total_error = (
                metrics['explainer_error'] +
                metrics['reasoner_error'] +
                metrics['producer_error']
        )

        if total_error < self.best_error - self.error_threshold:
            self.best_error = total_error
            self.wait_error = 0
        else:
            self.wait_error += 1

        if self.wait_error >= self.patience:
            print(f"\nEarly stopping at epoch {epoch + 1} due to convergence of model errors.")
            orchestrator.save_models(f"ccnet_checkpoint_epoch_{epoch}")
            raise StopIteration("Early stopping triggered by error convergence.")

        # Condition 2: Check for gradient stagnation if enabled
        if self.grad_stagnation_threshold is not None:
            total_grad_norm = (
                metrics.get('explainer_grad_norm', 0) +
                metrics.get('reasoner_grad_norm', 0) +
                metrics.get('producer_grad_norm', 0)
            )
            if total_grad_norm < self.grad_stagnation_threshold:
                self.wait_grad += 1
            else:
                self.wait_grad = 0

            if self.wait_grad >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} due to gradient stagnation.")
                orchestrator.save_models(f"ccnet_checkpoint_epoch_{epoch}")
                raise StopIteration("Early stopping triggered by gradient stagnation.")


# ---------------------------------------------------------------------

def wrap_keras_model(model: keras.Model) -> CCNetModule:
    """
    Wrap a Keras model to comply with CCNetModule protocol.

    Args:
        model: Keras model to wrap.

    Returns:
        Wrapped model compatible with CCNet framework.
    """

    class KerasModelWrapper:
        def __init__(self, keras_model):
            self.model = keras_model

        def __call__(self, *args, training=False, **kwargs):
            return self.model(*args, training=training, **kwargs)

        @property
        def trainable_variables(self):
            return self.model.trainable_variables

        def save(self, filepath):
            self.model.save(filepath)

    return KerasModelWrapper(model)

# ---------------------------------------------------------------------