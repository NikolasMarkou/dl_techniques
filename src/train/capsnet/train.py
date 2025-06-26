#!/usr/bin/env python3
"""
Simple training script for Keras-compliant CapsNet.

This script demonstrates how to train the CapsNet model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, and evaluation.

Usage:
    python train.py [--dataset mnist] [--epochs 50] [--batch-size 32]
"""

import argparse
import os
import numpy as np
import keras
import tensorflow as tf
from datetime import datetime
from typing import Tuple, Dict, Any

from dl_techniques.models.capsnet import CapsNet, create_capsnet, CapsuleAccuracy
from dl_techniques.utils.logger import logger


def setup_gpu():
    """Configure GPU settings for optimal training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")


def load_mnist_data(
        validation_split: float = 0.1,
        normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset.

    Args:
        validation_split: Fraction of training data to use for validation.
        normalize: Whether to normalize pixel values to [0, 1].

    Returns:
        Tuple of (train_data, val_data, test_data) where each is (x, y).
    """
    logger.info("Loading MNIST dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Normalize if requested
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Create validation split
    if validation_split > 0:
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]
    else:
        x_val, y_val = x_test, y_test  # Use test set as validation

    logger.info(f"Dataset shapes:")
    logger.info(f"  Train: {x_train.shape}, {y_train.shape}")
    logger.info(f"  Validation: {x_val.shape}, {y_val.shape}")
    logger.info(f"  Test: {x_test.shape}, {y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_cifar10_data(
        validation_split: float = 0.1,
        normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset.

    Args:
        validation_split: Fraction of training data to use for validation.
        normalize: Whether to normalize pixel values to [0, 1].

    Returns:
        Tuple of (train_data, val_data, test_data) where each is (x, y).
    """
    logger.info("Loading CIFAR-10 dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize if requested
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Create validation split
    if validation_split > 0:
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]
    else:
        x_val, y_val = x_test, y_test  # Use test set as validation

    logger.info(f"Dataset shapes:")
    logger.info(f"  Train: {x_train.shape}, {y_train.shape}")
    logger.info(f"  Validation: {x_val.shape}, {y_val.shape}")
    logger.info(f"  Test: {x_test.shape}, {y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_model_config(dataset: str) -> Dict[str, Any]:
    """Create model configuration based on dataset.

    Args:
        dataset: Name of dataset ('mnist' or 'cifar10').

    Returns:
        Dictionary of model configuration parameters.
    """
    if dataset.lower() == 'mnist':
        return {
            'num_classes': 10,
            'input_shape': (28, 28, 1),
            'conv_filters': [256, 256],
            'primary_capsules': 32,
            'primary_capsule_dim': 8,
            'digit_capsule_dim': 16,
            'reconstruction': True,
            'decoder_architecture': [512, 1024],
            'positive_margin': 0.9,
            'negative_margin': 0.1,
            'downweight': 0.5,
            'reconstruction_weight': 0.0005,
            'use_batch_norm': True
        }
    elif dataset.lower() == 'cifar10':
        return {
            'num_classes': 10,
            'input_shape': (32, 32, 3),
            'conv_filters': [128, 256, 256],  # More layers for complex images
            'primary_capsules': 64,  # More capsules for complex features
            'primary_capsule_dim': 16,
            'digit_capsule_dim': 32,
            'reconstruction': True,
            'decoder_architecture': [1024, 2048, 1024],  # Larger decoder
            'positive_margin': 0.9,
            'negative_margin': 0.1,
            'downweight': 0.5,
            'reconstruction_weight': 0.0005,
            'use_batch_norm': True
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def create_callbacks(
        model_name: str,
        monitor: str = 'val_capsule_accuracy',
        patience: int = 10,
        save_best_only: bool = True
) -> list:
    """Create training callbacks.

    Args:
        model_name: Name for saved model files.
        monitor: Metric to monitor for callbacks.
        patience: Patience for early stopping.
        save_best_only: Whether to save only the best model.

    Returns:
        List of Keras callbacks.
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/capsnet_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='max',  # For accuracy metrics, higher is better
            verbose=1
        ),

        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='max',  # For accuracy metrics, higher is better
            verbose=1
        ),

        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # CSV logging
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv'),
            append=False
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks


def train_model(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Command line arguments.
    """
    logger.info("Starting CapsNet training")
    logger.info(f"Arguments: {vars(args)}")

    # Setup GPU
    setup_gpu()

    # Load data
    if args.dataset.lower() == 'mnist':
        train_data, val_data, test_data = load_mnist_data(args.validation_split)
    elif args.dataset.lower() == 'cifar10':
        train_data, val_data, test_data = load_cifar10_data(args.validation_split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    # Create model
    logger.info("Creating CapsNet model...")
    model_config = create_model_config(args.dataset)

    model = create_capsnet(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        **model_config
    )

    # Build model by calling it once
    sample_input = tf.zeros((1,) + model_config['input_shape'])
    _ = model(sample_input, training=False)

    # Print model summary
    logger.info("Model architecture:")
    model.summary()

    # Create callbacks
    callbacks = create_callbacks(
        model_name=args.dataset,
        patience=args.patience,
        save_best_only=True
    )

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = model.evaluate(x_test, y_test, verbose=1)

    # Print final results
    logger.info("Training completed!")
    logger.info("Final test results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        logger.info(f"  {metric_name}: {value:.4f}")

    # Save final model
    final_model_path = f"capsnet_{args.dataset}_final.keras"
    model.save_model(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Generate some predictions for demonstration
    logger.info("Generating sample predictions...")
    sample_indices = np.random.choice(len(x_test), size=5, replace=False)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices]

    predictions = model.predict(sample_images, verbose=0)
    predicted_classes = np.argmax(predictions['length'], axis=1)
    true_classes = np.argmax(sample_labels, axis=1)

    logger.info("Sample predictions:")
    for i in range(5):
        confidence = predictions['length'][i].max()
        logger.info(
            f"  Sample {i}: True={true_classes[i]}, Predicted={predicted_classes[i]}, Confidence={confidence:.4f}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train CapsNet on image classification datasets')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use for training (default: mnist)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Fraction of training data to use for validation (default: 0.1)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')

    # Model arguments
    parser.add_argument('--no-reconstruction', action='store_true',
                        help='Disable reconstruction decoder')
    parser.add_argument('--reconstruction-weight', type=float, default=0.0005,
                        help='Weight for reconstruction loss (default: 0.0005)')

    # Parse arguments
    args = parser.parse_args()

    # Start training
    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()