"""
CapsNet Training Utilities and Trainer Class.

This module provides training utilities for the CapsNet model, including
custom loss functions, metrics, and a dedicated trainer class that handles
the training loop with margin loss and reconstruction loss.

Features:
    - Margin loss implementation for capsule networks
    - Reconstruction loss calculation
    - Custom training loop with gradient clipping
    - Comprehensive metrics tracking
    - Learning rate scheduling support
    - Model checkpointing and early stopping

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017).
      Dynamic routing between capsules. In Advances in Neural
      Information Processing Systems (pp. 3856-3866).
"""

import keras
import numpy as np
from keras import ops
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.capsule_margin_loss import capsule_margin_loss

# ---------------------------------------------------------------------

class CapsNetLoss:
    """Loss functions for CapsNet training.

    This class encapsulates the margin loss and reconstruction loss
    calculations for CapsNet training.
    """

    def __init__(
            self,
            margin_loss_weight: float = 1.0,
            reconstruction_weight: float = 0.0005,
            positive_margin: float = 0.9,
            negative_margin: float = 0.1,
            downweight: float = 0.5
    ) -> None:
        """Initialize loss functions.

        Args:
            margin_loss_weight: Weight for margin loss component.
            reconstruction_weight: Weight for reconstruction loss component.
            positive_margin: Positive margin for margin loss (m^+).
            negative_margin: Negative margin for margin loss (m^-).
            downweight: Downweight parameter for negative class loss (λ).
        """
        self.margin_loss_weight = margin_loss_weight
        self.reconstruction_weight = reconstruction_weight
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.downweight = downweight

    def margin_loss(
            self,
            y_true: keras.KerasTensor,
            y_pred_lengths: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Calculate margin loss for capsule networks.

        Args:
            y_true: True labels (one-hot encoded).
            y_pred_lengths: Predicted capsule lengths.

        Returns:
            Margin loss value.
        """
        return ops.mean(capsule_margin_loss(
            y_pred_lengths,  # y_pred comes first in the actual function
            y_true,  # y_true comes second
            self.downweight,
            self.positive_margin,
            self.negative_margin
        ))

    def reconstruction_loss(
            self,
            x_true: keras.KerasTensor,
            x_reconstructed: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Calculate reconstruction loss.

        Args:
            x_true: Original input images.
            x_reconstructed: Reconstructed images.

        Returns:
            Reconstruction loss value.
        """
        return ops.mean(ops.square(x_true - x_reconstructed))

    def total_loss(
            self,
            x_true: keras.KerasTensor,
            y_true: keras.KerasTensor,
            model_outputs: Dict[str, keras.KerasTensor]
    ) -> Tuple[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Calculate total loss combining margin and reconstruction loss.

        Args:
            x_true: Original input images.
            y_true: True labels (one-hot encoded).
            model_outputs: Dictionary of model outputs.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        # Calculate margin loss
        margin_loss_val = self.margin_loss(y_true, model_outputs["length"])
        total_loss = self.margin_loss_weight * margin_loss_val

        loss_components = {
            "margin_loss": margin_loss_val,
            "reconstruction_loss": ops.convert_to_tensor(0.0, dtype=total_loss.dtype)
        }

        # Add reconstruction loss if available
        if "reconstructed" in model_outputs and self.reconstruction_weight > 0:
            recon_loss = self.reconstruction_loss(x_true, model_outputs["reconstructed"])
            total_loss += self.reconstruction_weight * recon_loss
            loss_components["reconstruction_loss"] = recon_loss

        loss_components["total_loss"] = total_loss
        return total_loss, loss_components


class CapsNetMetrics:
    """Metrics computation for CapsNet."""

    @staticmethod
    def capsule_accuracy(
            y_true: keras.KerasTensor,
            y_pred_lengths: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Calculate accuracy based on capsule lengths.

        Args:
            y_true: True labels (one-hot encoded).
            y_pred_lengths: Predicted capsule lengths.

        Returns:
            Accuracy value.
        """
        y_true_classes = ops.argmax(y_true, axis=1)
        y_pred_classes = ops.argmax(y_pred_lengths, axis=1)
        return ops.mean(ops.cast(ops.equal(y_true_classes, y_pred_classes), "float32"))

    @staticmethod
    def top_k_accuracy(
            y_true: keras.KerasTensor,
            y_pred_lengths: keras.KerasTensor,
            k: int = 3
    ) -> keras.KerasTensor:
        """Calculate top-k accuracy.

        Args:
            y_true: True labels (one-hot encoded).
            y_pred_lengths: Predicted capsule lengths.
            k: Number of top predictions to consider.

        Returns:
            Top-k accuracy value.
        """
        y_true_classes = ops.argmax(y_true, axis=1)
        top_k_pred = ops.top_k(y_pred_lengths, k=k).indices

        # Check if true class is in top-k predictions
        correct = ops.any(ops.equal(ops.expand_dims(y_true_classes, -1), top_k_pred), axis=1)
        return ops.mean(ops.cast(correct, "float32"))


class CapsNetTrainer:
    """Trainer class for CapsNet models.

    This class handles the training process for CapsNet models, including
    custom loss functions, metrics computation, and training loop management.
    """

    def __init__(
            self,
            model: keras.Model,
            optimizer: keras.optimizers.Optimizer,
            loss_config: Optional[Dict[str, Any]] = None,
            metrics: Optional[List[str]] = None,
            gradient_clip_norm: float = 5.0
    ) -> None:
        """Initialize the trainer.

        Args:
            model: CapsNet model to train.
            optimizer: Keras optimizer for training.
            loss_config: Configuration for loss functions.
            metrics: List of metrics to track during training.
            gradient_clip_norm: Gradient clipping norm value.
        """
        self.model = model
        self.optimizer = optimizer
        self.gradient_clip_norm = gradient_clip_norm

        # Initialize loss functions
        loss_config = loss_config or {}
        self.loss_fn = CapsNetLoss(**loss_config)

        # Initialize metrics
        self.metrics = metrics or ["accuracy", "top_3_accuracy"]
        self.metric_objects = {
            "accuracy": keras.metrics.SparseCategoricalAccuracy(),
            "top_3_accuracy": keras.metrics.SparseTopKCategoricalAccuracy(k=3)
        }

        # Training state
        self.history = {"loss": [], "val_loss": []}
        for metric in self.metrics:
            self.history[metric] = []
            self.history[f"val_{metric}"] = []

    @tf.function
    def train_step(
            self,
            x_batch: keras.KerasTensor,
            y_batch: keras.KerasTensor
    ) -> Dict[str, keras.KerasTensor]:
        """Perform a single training step.

        Args:
            x_batch: Batch of input images.
            y_batch: Batch of one-hot encoded labels.

        Returns:
            Dictionary of metrics for this step.
        """
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(x_batch, training=True, mask=y_batch)

            # Calculate loss
            total_loss, loss_components = self.loss_fn.total_loss(x_batch, y_batch, outputs)

        # Calculate gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Filter out None gradients and apply clipping
        gradients_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]

        if gradients_and_vars:
            filtered_grads, filtered_vars = zip(*gradients_and_vars)

            # Apply gradient clipping
            if self.gradient_clip_norm > 0:
                filtered_grads, _ = tf.clip_by_global_norm(filtered_grads, self.gradient_clip_norm)

            # Apply gradients
            self.optimizer.apply_gradients(zip(filtered_grads, filtered_vars))

        # Calculate metrics
        metrics_dict = self._calculate_metrics(y_batch, outputs, loss_components)

        return metrics_dict

    @tf.function
    def test_step(
            self,
            x_batch: keras.KerasTensor,
            y_batch: keras.KerasTensor
    ) -> Dict[str, keras.KerasTensor]:
        """Perform a single validation/test step.

        Args:
            x_batch: Batch of input images.
            y_batch: Batch of one-hot encoded labels.

        Returns:
            Dictionary of metrics for this step.
        """
        # Forward pass
        outputs = self.model(x_batch, training=False, mask=y_batch)

        # Calculate loss
        total_loss, loss_components = self.loss_fn.total_loss(x_batch, y_batch, outputs)

        # Calculate metrics
        metrics_dict = self._calculate_metrics(y_batch, outputs, loss_components)

        return metrics_dict

    def _calculate_metrics(
            self,
            y_true: keras.KerasTensor,
            outputs: Dict[str, keras.KerasTensor],
            loss_components: Dict[str, keras.KerasTensor]
    ) -> Dict[str, keras.KerasTensor]:
        """Calculate metrics for a batch.

        Args:
            y_true: True labels (one-hot encoded).
            outputs: Model outputs.
            loss_components: Dictionary of loss components.

        Returns:
            Dictionary of calculated metrics.
        """
        metrics_dict = loss_components.copy()

        # Calculate capsule accuracy
        metrics_dict["accuracy"] = CapsNetMetrics.capsule_accuracy(
            y_true, outputs["length"]
        )

        # Calculate top-k accuracy if requested
        if "top_3_accuracy" in self.metrics:
            metrics_dict["top_3_accuracy"] = CapsNetMetrics.top_k_accuracy(
                y_true, outputs["length"], k=3
            )

        return metrics_dict

    def fit(
            self,
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 10,
            callbacks: Optional[List[keras.callbacks.Callback]] = None,
            verbose: int = 1
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            train_dataset: Training dataset.
            validation_dataset: Validation dataset (optional).
            epochs: Number of training epochs.
            callbacks: List of Keras callbacks.
            verbose: Verbosity level.

        Returns:
            Training history dictionary.
        """
        callbacks = callbacks or []

        # Initialize callback list
        callback_list = keras.callbacks.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self.model,
            verbose=verbose,
            epochs=epochs,
            steps=len(train_dataset)
        )

        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Training phase
            epoch_metrics = {"loss": [], "accuracy": []}
            if "top_3_accuracy" in self.metrics:
                epoch_metrics["top_3_accuracy"] = []
            epoch_metrics["margin_loss"] = []
            epoch_metrics["reconstruction_loss"] = []

            callback_list.on_epoch_begin(epoch)

            # Training steps
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                callback_list.on_train_batch_begin(step)

                step_metrics = self.train_step(x_batch, y_batch)

                # Accumulate metrics
                for key, value in step_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(float(value.numpy()))

                callback_list.on_train_batch_end(step, step_metrics)

            # Calculate epoch averages
            train_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}

            # Validation phase
            val_metrics = {}
            if validation_dataset is not None:
                val_epoch_metrics = {"loss": [], "accuracy": []}
                if "top_3_accuracy" in self.metrics:
                    val_epoch_metrics["top_3_accuracy"] = []
                val_epoch_metrics["margin_loss"] = []
                val_epoch_metrics["reconstruction_loss"] = []

                for x_batch, y_batch in validation_dataset:
                    step_metrics = self.test_step(x_batch, y_batch)

                    # Accumulate validation metrics
                    for key, value in step_metrics.items():
                        if key in val_epoch_metrics:
                            val_epoch_metrics[key].append(float(value.numpy()))

                # Calculate validation averages
                val_metrics = {f"val_{key}": np.mean(values)
                               for key, values in val_epoch_metrics.items()}

            # Combine metrics
            epoch_logs = {**train_metrics, **val_metrics}

            # Update history
            for key, value in epoch_logs.items():
                if key in self.history:
                    self.history[key].append(value)

            # Log epoch results
            if verbose > 0:
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in epoch_logs.items()])
                logger.info(f"Epoch {epoch + 1} - {metrics_str}")

            callback_list.on_epoch_end(epoch, epoch_logs)

        return self.history

    def evaluate(
            self,
            test_dataset: tf.data.Dataset,
            verbose: int = 1
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_dataset: Test dataset.
            verbose: Verbosity level.

        Returns:
            Dictionary of evaluation metrics.
        """
        if verbose > 0:
            logger.info("Evaluating model...")

        # Accumulate metrics
        eval_metrics = {"loss": [], "accuracy": []}
        if "top_3_accuracy" in self.metrics:
            eval_metrics["top_3_accuracy"] = []
        eval_metrics["margin_loss"] = []
        eval_metrics["reconstruction_loss"] = []

        for x_batch, y_batch in test_dataset:
            step_metrics = self.test_step(x_batch, y_batch)

            # Accumulate metrics
            for key, value in step_metrics.items():
                if key in eval_metrics:
                    eval_metrics[key].append(float(value.numpy()))

        # Calculate averages
        final_metrics = {key: np.mean(values) for key, values in eval_metrics.items()}

        if verbose > 0:
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
            logger.info(f"Test results - {metrics_str}")

        return final_metrics

    def predict(
            self,
            dataset: tf.data.Dataset,
            return_reconstructions: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate predictions on a dataset.

        Args:
            dataset: Input dataset.
            return_reconstructions: Whether to return reconstructed images.

        Returns:
            Dictionary containing predictions and optionally reconstructions.
        """
        predictions = {"lengths": [], "digit_caps": []}
        if return_reconstructions:
            predictions["reconstructions"] = []

        for x_batch in dataset:
            # Handle both (x, y) and x formats
            if isinstance(x_batch, tuple):
                x_batch = x_batch[0]

            outputs = self.model(x_batch, training=False)

            predictions["lengths"].append(outputs["length"].numpy())
            predictions["digit_caps"].append(outputs["digit_caps"].numpy())

            if return_reconstructions and "reconstructed" in outputs:
                predictions["reconstructions"].append(outputs["reconstructed"].numpy())

        # Concatenate all batches
        final_predictions = {}
        for key, values in predictions.items():
            if values:  # Only concatenate if we have values
                final_predictions[key] = np.concatenate(values, axis=0)

        return final_predictions


def create_capsnet_trainer(
        model: keras.Model,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        loss_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        gradient_clip_norm: float = 5.0
) -> CapsNetTrainer:
    """Factory function to create a CapsNetTrainer.

    Args:
        model: CapsNet model to train.
        learning_rate: Learning rate for optimizer.
        optimizer_name: Name of optimizer to use.
        loss_config: Configuration for loss functions.
        metrics: List of metrics to track.
        gradient_clip_norm: Gradient clipping norm.

    Returns:
        Configured CapsNetTrainer instance.
    """
    # Create optimizer
    optimizer_map = {
        "adam": keras.optimizers.Adam,
        "adamw": keras.optimizers.AdamW,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop
    }

    if optimizer_name.lower() not in optimizer_map:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    optimizer = optimizer_map[optimizer_name.lower()](learning_rate=learning_rate)

    return CapsNetTrainer(
        model=model,
        optimizer=optimizer,
        loss_config=loss_config,
        metrics=metrics,
        gradient_clip_norm=gradient_clip_norm
    )
