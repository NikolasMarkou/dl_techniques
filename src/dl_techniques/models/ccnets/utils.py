import keras
from typing import Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import CCNetModule
from .orchestrators import CCNetOrchestrator

# ---------------------------------------------------------------------

class EarlyStoppingCallback:
    """
    Early stopping callback based on the convergence of model errors.
    """

    def __init__(self, patience: int = 5, threshold: float = 1e-4):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            threshold: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.threshold = threshold
        self.best_error = float('inf')
        self.wait = 0

    def __call__(self, epoch: int, metrics: Dict[str, float], orchestrator: CCNetOrchestrator):
        """Check for early stopping condition."""
        # Use the sum of the model-specific errors as the convergence metric.
        # This is a more direct measure of the system's learning objectives.
        total_error = (
                metrics['explainer_error'] +
                metrics['reasoner_error'] +
                metrics['producer_error']
        )

        if total_error < self.best_error - self.threshold:
            self.best_error = total_error
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            print(f"\nEarly stopping at epoch {epoch + 1} due to convergence of model errors.")
            orchestrator.save_models(f"ccnet_checkpoint_epoch_{epoch}")
            raise StopIteration("Early stopping triggered")


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