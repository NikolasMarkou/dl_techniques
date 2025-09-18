import numpy as np
import tensorflow as tf
from typing import Dict, Optional, List, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .orchestrators import CCNetOrchestrator


# ---------------------------------------------------------------------

class CCNetTrainer:
    """
    High-level trainer for CCNet models with built-in callbacks and monitoring.
    """

    def __init__(
            self,
            orchestrator: CCNetOrchestrator,
            metrics_callback: Optional[Callable] = None
    ):
        """
        Initialize CCNet trainer.

        Args:
            orchestrator: CCNet orchestrator instance.
            metrics_callback: Optional callback for metrics logging.
        """
        self.orchestrator = orchestrator
        self.metrics_callback = metrics_callback
        self.history = {
            'generation_loss': [],
            'reconstruction_loss': [],
            'inference_loss': [],
            'explainer_error': [],
            'reasoner_error': [],
            'producer_error': []
        }

    def train(
            self,
            train_dataset: tf.data.Dataset,
            epochs: int,
            validation_dataset: Optional[tf.data.Dataset] = None,
            callbacks: Optional[List[Callable]] = None
    ):
        """
        Train the CCNet for multiple epochs.

        Args:
            train_dataset: Training dataset yielding (x, y) batches.
            epochs: Number of training epochs.
            validation_dataset: Optional validation dataset.
            callbacks: Optional list of callback functions.
        """
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Training loop
            train_losses = []
            for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
                losses = self.orchestrator.train_step(x_batch, y_batch)
                train_losses.append(losses)

                # Print progress
                if batch_idx % 10 == 0:
                    self._print_progress(batch_idx, losses)

            # Aggregate epoch metrics
            epoch_metrics = self._aggregate_metrics(train_losses)

            # Validation
            if validation_dataset is not None:
                val_losses = []
                for x_val, y_val in validation_dataset:
                    losses = self.orchestrator.evaluate(x_val, y_val)
                    val_losses.append(losses)

                val_metrics = self._aggregate_metrics(val_losses)
                logger.info(f"Validation - Gen: {val_metrics['generation_loss']:.4f}, "
                      f"Rec: {val_metrics['reconstruction_loss']:.4f}, "
                      f"Inf: {val_metrics['inference_loss']:.4f}")

            # Update history
            for key in epoch_metrics:
                self.history[key].append(epoch_metrics[key])

            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, epoch_metrics, self.orchestrator)

            if self.metrics_callback:
                self.metrics_callback(epoch, epoch_metrics)

    def _print_progress(self, batch_idx: int, losses: Dict[str, float]):
        """Print training progress."""
        logger.info(f"Batch {batch_idx} - "
              f"Gen: {losses['generation_loss']:.4f}, "
              f"Rec: {losses['reconstruction_loss']:.4f}, "
              f"Inf: {losses['inference_loss']:.4f}")

    def _aggregate_metrics(self, losses_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics over batches."""
        aggregated = {}
        for key in losses_list[0].keys():
            aggregated[key] = np.mean([losses[key] for losses in losses_list])
        return aggregated


