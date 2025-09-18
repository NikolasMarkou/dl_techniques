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
    Early stopping callback based on convergence criteria.
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
        self.best_loss = float('inf')
        self.wait = 0

    def __call__(self, epoch: int, metrics: Dict[str, float], orchestrator: CCNetOrchestrator):
        """Check for early stopping condition."""
        # Use sum of all three losses as convergence metric
        total_loss = (
                metrics['generation_loss'] +
                metrics['reconstruction_loss'] +
                metrics['inference_loss']
        )

        if total_loss < self.best_loss - self.threshold:
            self.best_loss = total_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
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
