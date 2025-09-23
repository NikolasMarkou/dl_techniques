import numpy as np
import tensorflow as tf
from typing import Dict, Optional, List, Callable
from collections import defaultdict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .orchestrators import CCNetOrchestrator


# ---------------------------------------------------------------------

class CCNetTrainer:
    """
    High-level trainer for CCNet models with built-in callbacks, dynamic
    weighting, KL annealing, and advanced monitoring.
    """

    def __init__(
            self,
            orchestrator: CCNetOrchestrator,
            metrics_callback: Optional[Callable] = None,
            kl_annealing_epochs: Optional[int] = None
    ):
        """
        Initialize CCNet trainer.

        Args:
            orchestrator: CCNet orchestrator instance.
            metrics_callback: Optional callback for metrics logging.
            kl_annealing_epochs: Number of epochs to linearly anneal the KL
                                 weight from 0 to its configured value.
        """
        self.orchestrator = orchestrator
        self.metrics_callback = metrics_callback
        self.kl_annealing_epochs = kl_annealing_epochs
        self.initial_kl_weight = self.orchestrator.config.kl_weight
        self.history = defaultdict(list)
        self.loss_moving_averages = defaultdict(lambda: 1.0)
        self.ema_alpha = 0.1  # Smoothing factor for exponential moving average

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
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 30)

            # Apply KL annealing if enabled
            if self.kl_annealing_epochs is not None and self.kl_annealing_epochs > 0:
                annealing_ratio = min(1.0, (epoch + 1) / self.kl_annealing_epochs)
                new_kl_weight = self.initial_kl_weight * annealing_ratio
                self.orchestrator.config.kl_weight = new_kl_weight
                print(f"  KL Annealing: Current Weight = {new_kl_weight:.4f}")

            # Training loop
            train_losses = []
            for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
                # Apply dynamic weighting if enabled
                if self.orchestrator.config.dynamic_weighting:
                    self._update_dynamic_weights()

                # Execute one training step
                losses = self.orchestrator.train_step(x_batch, y_batch)

                # Update the state of the control strategy from the eager context
                self.orchestrator.control.update_state(losses)

                # Store and update metrics
                train_losses.append({k: v.numpy() for k, v in losses.items()})
                self._update_moving_averages(losses)

                if batch_idx % 10 == 0:
                    self._print_progress(batch_idx, losses)

            # Aggregate and log epoch metrics
            epoch_metrics = self._aggregate_metrics(train_losses)
            self._log_epoch_summary("Training", epoch_metrics)

            # Validation loop
            if validation_dataset is not None:
                val_metrics = self._run_evaluation(validation_dataset)
                self._log_epoch_summary("Validation", val_metrics)
                for key, value in val_metrics.items():
                    self.history[f"val_{key}"].append(value)

            # Update history with training metrics
            for key, value in epoch_metrics.items():
                self.history[key].append(value)

            # Execute callbacks
            if callbacks:
                try:
                    for callback in callbacks:
                        callback(epoch, epoch_metrics, self.orchestrator)
                except StopIteration:
                    print("StopIteration caught from a callback. Halting training.")
                    break

            if self.metrics_callback:
                self.metrics_callback(epoch, epoch_metrics)

    def _run_evaluation(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Run evaluation on a given dataset."""
        val_losses = []
        for x_val, y_val in dataset:
            losses = self.orchestrator.evaluate(x_val, y_val)
            val_losses.append({k: v.numpy() for k, v in losses.items()})
        return self._aggregate_metrics(val_losses)

    def _update_moving_averages(self, current_losses: Dict[str, tf.Tensor]):
        """Update exponential moving average of losses."""
        for key, value in current_losses.items():
            # Ensure value is a numpy float for EMA calculation
            numpy_val = value.numpy() if hasattr(value, 'numpy') else value
            self.loss_moving_averages[key] = (
                    self.ema_alpha * numpy_val +
                    (1 - self.ema_alpha) * self.loss_moving_averages[key]
            )

    def _update_dynamic_weights(self):
        """Dynamically adjust loss weights to balance module training."""
        epsilon = 1e-8
        rec_avg = self.loss_moving_averages['reconstruction_loss'] + epsilon
        gen_avg = self.loss_moving_averages['generation_loss'] + epsilon
        inf_avg = self.loss_moving_averages['inference_loss'] + epsilon

        # Balance Reasoner
        total_reasoner = rec_avg + inf_avg
        self.orchestrator.config.reasoner_weights['reconstruction'] = inf_avg / total_reasoner
        self.orchestrator.config.reasoner_weights['inference'] = rec_avg / total_reasoner

        # Balance Explainer
        total_explainer = gen_avg + inf_avg
        self.orchestrator.config.explainer_weights['generation'] = inf_avg / total_explainer
        self.orchestrator.config.explainer_weights['inference'] = gen_avg / total_explainer

        # Balance Producer
        total_producer = gen_avg + rec_avg
        self.orchestrator.config.producer_weights['generation'] = rec_avg / total_producer
        self.orchestrator.config.producer_weights['reconstruction'] = gen_avg / total_producer

    def _print_progress(self, batch_idx: int, losses: Dict[str, tf.Tensor]):
        """Print training progress for a batch."""
        gen = losses['generation_loss'].numpy()
        rec = losses['reconstruction_loss'].numpy()
        inf = losses['inference_loss'].numpy()
        acc = losses.get('batch_accuracy', np.array(0.0)).numpy()
        is_training = losses.get('reasoner_is_training', np.array(0.0)).numpy()
        print(
            f"  Batch {batch_idx:04d} -> Acc: {acc:.3f}, Gen: {gen:.4f}, Rec: {rec:.4f}, Inf: {inf:.4f}, ReaTrain: {int(is_training)}")

    def _log_epoch_summary(self, stage: str, metrics: Dict[str, float]):
        """Log a summary of metrics for an epoch."""
        gen = metrics.get('generation_loss', 0)
        rec = metrics.get('reconstruction_loss', 0)
        inf = metrics.get('inference_loss', 0)
        expl_err = metrics.get('explainer_error', 0)
        reas_err = metrics.get('reasoner_error', 0)
        prod_err = metrics.get('producer_error', 0)
        accuracy = metrics.get('batch_accuracy')

        print(f"\n{stage} Summary:")
        if accuracy is not None:
            print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Losses -> Gen: {gen:.4f}, Rec: {rec:.4f}, Inf: {inf:.4f}")
        print(f"  Errors -> Exp: {expl_err:.4f}, Rea: {reas_err:.4f}, Pro: {prod_err:.4f}")

        if 'explainer_grad_norm' in metrics:
            expl_gn = metrics.get('explainer_grad_norm', 0)
            reas_gn = metrics.get('reasoner_grad_norm', 0)
            prod_gn = metrics.get('producer_grad_norm', 0)
            print(f"  Grad Norms -> Exp: {expl_gn:.4f}, Rea: {reas_gn:.4f}, Pro: {prod_gn:.4f}")

        if 'reasoner_is_training' in metrics:
            train_ratio = metrics.get('reasoner_is_training', 0)
            print(f"  Reasoner Training Ratio: {train_ratio:.2%}")

        print("-" * 30)

    def _aggregate_metrics(self, losses_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics over all batches in an epoch."""
        if not losses_list:
            return {}
        aggregated = defaultdict(float)
        num_items = len(losses_list)
        for key in losses_list[0].keys():
            aggregated[key] = np.mean([losses[key] for losses in losses_list])
        return dict(aggregated)