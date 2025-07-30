"""
CCNets Training Utilities

This module provides utilities for training, evaluating, and monitoring CCNets models,
including specialized callbacks, metrics, and data preparation functions.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import keras
from keras import ops
from dl_techniques.utils.logger import logger


class CCNetsMetrics:
    """
    Specialized metrics for evaluating CCNets performance.

    These metrics help assess the quality of cooperative learning across
    the three networks and the explainability of the model.
    """

    @staticmethod
    def compute_cooperation_score(
            inference_loss: float,
            generation_loss: float,
            reconstruction_loss: float
    ) -> float:
        """
        Compute cooperation score measuring balance between the three losses.

        A lower score indicates better cooperation between networks.
        Perfect cooperation would have all three losses equal.

        Args:
            inference_loss: Consistency loss between reconstruction and generation
            generation_loss: Fidelity loss for generation
            reconstruction_loss: Fidelity loss for reconstruction

        Returns:
            Cooperation score (lower is better)
        """
        losses = np.array([inference_loss, generation_loss, reconstruction_loss])
        mean_loss = np.mean(losses)
        variance = np.var(losses)

        # Cooperation score combines total loss with balance
        cooperation_score = mean_loss + variance

        return cooperation_score

    @staticmethod
    def compute_explanation_consistency(
            explanations1: np.ndarray,
            explanations2: np.ndarray
    ) -> float:
        """
        Compute consistency of explanation vectors across different samples.

        Measures how consistent the explainer network is in generating
        similar explanations for similar inputs.

        Args:
            explanations1: First set of explanation vectors [N, explanation_dim]
            explanations2: Second set of explanation vectors [N, explanation_dim]

        Returns:
            Consistency score (higher is better)
        """
        # Compute cosine similarity between explanation sets
        norms1 = np.linalg.norm(explanations1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(explanations2, axis=1, keepdims=True)

        normalized1 = explanations1 / (norms1 + 1e-8)
        normalized2 = explanations2 / (norms2 + 1e-8)

        similarity = np.sum(normalized1 * normalized2, axis=1)
        consistency = np.mean(similarity)

        return consistency

    @staticmethod
    def compute_cross_verification_accuracy(
            generated_observations: np.ndarray,
            reconstructed_observations: np.ndarray,
            threshold: float = 0.1
    ) -> float:
        """
        Compute cross-verification accuracy between generation and reconstruction paths.

        Measures how often the two paths produce similar results,
        indicating reliable cross-verification.

        Args:
            generated_observations: Outputs from Producer(y, e)
            reconstructed_observations: Outputs from Producer(y', e)
            threshold: Similarity threshold for considering outputs as matching

        Returns:
            Cross-verification accuracy (0 to 1)
        """
        differences = np.abs(generated_observations - reconstructed_observations)
        mean_differences = np.mean(differences, axis=tuple(range(1, len(differences.shape))))

        accurate_samples = np.sum(mean_differences < threshold)
        accuracy = accurate_samples / len(mean_differences)

        return accuracy


class CCNetsCallback(keras.callbacks.Callback):
    """
    Custom callback for monitoring CCNets training progress.

    Tracks cooperation metrics, explanation quality, and provides
    detailed logging of the three-network learning dynamics.
    """

    def __init__(
            self,
            validation_data: Optional[Tuple] = None,
            log_frequency: int = 10,
            save_explanations: bool = False
    ):
        super().__init__()
        self.validation_data = validation_data
        self.log_frequency = log_frequency
        self.save_explanations = save_explanations

        # Tracking metrics
        self.cooperation_scores = []
        self.explanation_history = []
        self.cross_verification_scores = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Monitor progress at the end of each epoch."""
        logs = logs or {}

        # Compute cooperation score
        if all(key in logs for key in ['inference_loss', 'generation_loss', 'reconstruction_loss']):
            cooperation_score = CCNetsMetrics.compute_cooperation_score(
                logs['inference_loss'],
                logs['generation_loss'],
                logs['reconstruction_loss']
            )
            self.cooperation_scores.append(cooperation_score)
            logs['cooperation_score'] = cooperation_score

        # Detailed logging every N epochs
        if epoch % self.log_frequency == 0:
            logger.info(f"Epoch {epoch}: Cooperation Score = {cooperation_score:.6f}")
            logger.info(f"  Inference Loss: {logs.get('inference_loss', 0):.6f}")
            logger.info(f"  Generation Loss: {logs.get('generation_loss', 0):.6f}")
            logger.info(f"  Reconstruction Loss: {logs.get('reconstruction_loss', 0):.6f}")

        # Validation analysis
        if self.validation_data is not None and epoch % self.log_frequency == 0:
            self._analyze_validation_performance()

    def _analyze_validation_performance(self):
        """Analyze validation data for cooperation quality."""
        if self.validation_data is None:
            return

        try:
            (val_x, val_y), _ = self.validation_data

            # Get model outputs
            outputs = self.model([val_x, val_y], training=False)

            # Compute cross-verification accuracy
            cross_verification_acc = CCNetsMetrics.compute_cross_verification_accuracy(
                outputs['generated_observation'].numpy(),
                outputs['reconstructed_observation'].numpy()
            )
            self.cross_verification_scores.append(cross_verification_acc)

            logger.info(f"  Cross-verification Accuracy: {cross_verification_acc:.4f}")

            # Save explanations if requested
            if self.save_explanations:
                self.explanation_history.append(outputs['explanation_vector'].numpy())

        except Exception as e:
            logger.warning(f"Could not analyze validation performance: {e}")


class CCNetsDataGenerator:
    """
    Data generator for CCNets training with proper formatting.

    Handles the specific input format required by CCNets models:
    inputs = (observations, labels)
    """

    def __init__(
            self,
            observations: np.ndarray,
            labels: np.ndarray,
            batch_size: int = 32,
            shuffle: bool = True,
            augment_fn: Optional[Callable] = None
    ):
        self.observations = observations
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_fn = augment_fn

        self.indices = np.arange(len(observations))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.observations) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Generate one batch of data."""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.observations))
        batch_indices = self.indices[start_idx:end_idx]

        # Get batch data
        batch_observations = self.observations[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Apply augmentation if provided
        if self.augment_fn is not None:
            batch_observations = self.augment_fn(batch_observations)

        # CCNets expects inputs as (observations, labels) and targets as dummy
        inputs = (batch_observations, batch_labels)
        targets = np.zeros(len(batch_indices))  # Dummy targets

        return inputs, targets

    def on_epoch_end(self):
        """Update indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def prepare_ccnets_data(
        observations: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2,
        batch_size: int = 32,
        augment_fn: Optional[Callable] = None
) -> Tuple[CCNetsDataGenerator, CCNetsDataGenerator]:
    """
    Prepare data generators for CCNets training.

    Args:
        observations: Input observations [N, input_dim]
        labels: Target labels [N, output_dim]
        validation_split: Fraction of data for validation
        batch_size: Batch size for training
        augment_fn: Optional augmentation function

    Returns:
        Tuple of (train_generator, val_generator)
    """
    # Split data
    n_samples = len(observations)
    n_val = int(n_samples * validation_split)

    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Create generators
    train_generator = CCNetsDataGenerator(
        observations[train_indices],
        labels[train_indices],
        batch_size=batch_size,
        shuffle=True,
        augment_fn=augment_fn
    )

    val_generator = CCNetsDataGenerator(
        observations[val_indices],
        labels[val_indices],
        batch_size=batch_size,
        shuffle=False,
        augment_fn=None  # No augmentation for validation
    )

    logger.info(f"Prepared CCNets data: {len(train_generator)} train batches, {len(val_generator)} val batches")

    return train_generator, val_generator


def train_ccnets_model(
        model: 'CCNetsModel',
        train_data: CCNetsDataGenerator,
        val_data: Optional[CCNetsDataGenerator] = None,
        epochs: int = 100,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        callbacks: Optional[List[keras.callbacks.Callback]] = None,
        verbose: int = 1
) -> keras.callbacks.History:
    """
    Train a CCNets model with proper setup and monitoring.

    Args:
        model: CCNets model to train
        train_data: Training data generator
        val_data: Validation data generator
        epochs: Number of training epochs
        optimizer: Optimizer for training (default: Adam)
        callbacks: Additional callbacks
        verbose: Verbosity level

    Returns:
        Training history
    """
    # Set up optimizer
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer)

    # Set up callbacks
    callback_list = callbacks or []

    # Add CCNets-specific callback
    if val_data is not None:
        # Get a sample for validation analysis
        val_sample = val_data[0]
        ccnets_callback = CCNetsCallback(
            validation_data=val_sample,
            log_frequency=max(1, epochs // 10)
        )
        callback_list.append(ccnets_callback)

    # Add early stopping for stability
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True,
        verbose=verbose
    )
    callback_list.append(early_stopping)

    # Train model
    logger.info(f"Starting CCNets training for {epochs} epochs")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callback_list,
        verbose=verbose
    )

    logger.info("CCNets training completed")

    return history


def evaluate_ccnets_model(
        model: 'CCNetsModel',
        test_data: CCNetsDataGenerator,
        return_outputs: bool = False
) -> Dict[str, float]:
    """
    Evaluate a trained CCNets model.

    Args:
        model: Trained CCNets model
        test_data: Test data generator
        return_outputs: Whether to return detailed outputs

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating CCNets model")

    # Collect predictions and metrics
    all_losses = []
    all_generated = []
    all_reconstructed = []
    all_explanations = []

    for batch_idx in range(len(test_data)):
        (observations, labels), _ = test_data[batch_idx]

        # Forward pass
        outputs = model([observations, labels], training=False)

        # Compute losses
        losses = model.compute_losses(observations, labels, outputs)
        all_losses.append({
            'inference_loss': float(losses['inference_loss']),
            'generation_loss': float(losses['generation_loss']),
            'reconstruction_loss': float(losses['reconstruction_loss']),
            'total_loss': float(losses['total_loss'])
        })

        # Store outputs for detailed analysis
        if return_outputs:
            all_generated.append(outputs['generated_observation'].numpy())
            all_reconstructed.append(outputs['reconstructed_observation'].numpy())
            all_explanations.append(outputs['explanation_vector'].numpy())

    # Aggregate results
    metrics = {}
    loss_keys = ['inference_loss', 'generation_loss', 'reconstruction_loss', 'total_loss']

    for key in loss_keys:
        values = [batch_losses[key] for batch_losses in all_losses]
        metrics[f'mean_{key}'] = np.mean(values)
        metrics[f'std_{key}'] = np.std(values)

    # Compute cooperation score
    metrics['cooperation_score'] = CCNetsMetrics.compute_cooperation_score(
        metrics['mean_inference_loss'],
        metrics['mean_generation_loss'],
        metrics['mean_reconstruction_loss']
    )

    # Compute cross-verification accuracy if outputs available
    if return_outputs and all_generated:
        all_generated = np.concatenate(all_generated, axis=0)
        all_reconstructed = np.concatenate(all_reconstructed, axis=0)

        metrics['cross_verification_accuracy'] = CCNetsMetrics.compute_cross_verification_accuracy(
            all_generated, all_reconstructed
        )

        # Return outputs for further analysis
        metrics['outputs'] = {
            'generated_observations': all_generated,
            'reconstructed_observations': all_reconstructed,
            'explanations': np.concatenate(all_explanations, axis=0) if all_explanations else None
        }

    logger.info(f"Evaluation completed. Cooperation score: {metrics['cooperation_score']:.6f}")

    return metrics


def ccnets_inference(
        model: 'CCNetsModel',
        observations: np.ndarray,
        return_explanations: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform inference with a trained CCNets model.

    Args:
        model: Trained CCNets model
        observations: Input observations for inference
        return_explanations: Whether to return explanation vectors

    Returns:
        Dictionary containing predictions and optionally explanations
    """
    # Use predict_step for inference
    predictions_dict = model.predict_step(observations)

    results = {
        'predictions': predictions_dict['predictions'].numpy()
    }

    if return_explanations:
        results['explanations'] = predictions_dict['explanations'].numpy()

    return results


def simulate_approval_scenario(
        model: 'CCNetsModel',
        rejected_application: np.ndarray,
        explanation_vector: np.ndarray,
        approval_label: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Simulate data for approval scenario (e.g., loan applications).

    This demonstrates the bidirectional inference capability of CCNets:
    generating what an approvable application would look like given
    the explanation vector from a rejected application.

    Args:
        model: Trained CCNets model
        rejected_application: Original rejected application data
        explanation_vector: Explanation vector from rejected application
        approval_label: Label representing approval

    Returns:
        Dictionary containing generated approvable application and validation
    """
    # Generate approvable application using the explanation
    approvable_application = model.producer_network([approval_label, explanation_vector], training=False)

    # Validate by reconstructing original from rejection label
    rejection_label = model.reasoner_network([rejected_application, explanation_vector], training=False)
    reconstructed_original = model.producer_network([rejection_label, explanation_vector], training=False)

    return {
        'approvable_application': approvable_application.numpy(),
        'reconstructed_original': reconstructed_original.numpy(),
        'validation_error': float(ops.mean(ops.abs(reconstructed_original - rejected_application)))
    }