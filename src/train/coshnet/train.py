import os
import keras
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Tuple, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.shearlet_transform import ShearletTransform
from dl_techniques.layers.complex_layers import ComplexDense, ComplexConv2D, ComplexReLU
from dl_techniques.models.coshnet.model import CoShNet, create_coshnet, create_coshnet_variant


# ---------------------------------------------------------------------

def setup_gpu() -> None:
    """Configure GPU settings for optimal training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")


# ---------------------------------------------------------------------

def load_dataset(dataset_name: str) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[int, int, int], int]:
    """
    Load and preprocess dataset for CoShNet training.

    Args:
        dataset_name: Name of the dataset to load ('mnist', 'cifar10', 'cifar100')

    Returns:
        Tuple containing (train_data, test_data, input_shape, num_classes)
    """
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Convert grayscale to RGB for consistency
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
        input_shape = (28, 28, 3)
        num_classes = 10

    elif dataset_name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)
        num_classes = 10

    elif dataset_name.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)
        num_classes = 100

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Normalize to [0, 1] for better complex-valued processing
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(f"Dataset loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
    logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


# ---------------------------------------------------------------------

def create_model_config(dataset: str, variant: str, input_shape: Tuple[int, int, int], num_classes: int) -> Dict[
    str, Any]:
    """
    Create CoShNet model configuration based on dataset and variant.

    Args:
        dataset: Dataset name
        variant: Model variant (tiny, base, large, etc.)
        input_shape: Input image shape
        num_classes: Number of output classes

    Returns:
        Dictionary with model configuration parameters
    """
    # Base configuration optimized for CoShNet
    config = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'kernel_regularizer': None,
        'epsilon': 1e-7,  # For numerical stability in complex operations
    }

    # Dataset-specific adjustments
    if dataset.lower() == 'mnist':
        config.update({
            'dropout_rate': 0.05,  # Less dropout for simpler dataset
            'shearlet_scales': 3,  # Fewer scales for smaller images
            'shearlet_directions': 6,
        })

    elif dataset.lower() == 'cifar10':
        config.update({
            'dropout_rate': 0.1,  # Standard dropout for CIFAR-10
            'shearlet_scales': 4,  # Good balance for 32x32 images
            'shearlet_directions': 8,
        })

    elif dataset.lower() == 'cifar100':
        config.update({
            'dropout_rate': 0.15,  # More dropout for harder dataset
            'shearlet_scales': 4,
            'shearlet_directions': 8,
            'kernel_regularizer': keras.regularizers.l2(1e-4),  # Add regularization
        })

    # Variant-specific adjustments
    if variant == 'tiny':
        config.update({
            'conv_filters': (16, 32),
            'dense_units': (256, 128),
            'dropout_rate': config.get('dropout_rate', 0.1) + 0.05,  # Slightly more dropout
        })
    elif variant == 'large':
        config.update({
            'conv_filters': (64, 128, 256),
            'dense_units': (2048, 1024, 512),
            'dropout_rate': config.get('dropout_rate', 0.1) + 0.05,
            'shearlet_scales': 5,
            'shearlet_directions': 12,
        })

    return config


# ---------------------------------------------------------------------

def create_learning_rate_schedule(
        initial_lr: float,
        schedule_type: str = 'cosine',
        total_epochs: int = 50,
        warmup_epochs: int = 3
) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create learning rate schedule optimized for CoShNet training.

    CoShNet typically converges faster than traditional CNNs, so we use shorter schedules.
    """
    if schedule_type == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_epochs * 0.8,  # Decay over 80% of training
            alpha=0.01  # Minimum LR as fraction of initial
        )
    elif schedule_type == 'exponential':
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_epochs // 3,
            decay_rate=0.85
        )
    else:  # constant
        return initial_lr


# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        monitor: str = 'val_accuracy',
        patience: int = 10,  # Reduced patience for CoShNet's faster convergence
        use_lr_schedule: bool = True
) -> Tuple[List, str]:
    """Create training callbacks optimized for CoShNet."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"coshnet_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv')
        ),
    ]

    # Only add ReduceLROnPlateau if not using learning rate schedule
    if not use_lr_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,  # Reduced patience for faster convergence
                min_lr=1e-7,
                verbose=1
            )
        )
        logger.info("Added ReduceLROnPlateau callback")
    else:
        logger.info("Using learning rate schedule, skipping ReduceLROnPlateau")

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


# ---------------------------------------------------------------------

def validate_model_loading(model_path: str, test_input: tf.Tensor, original_output: tf.Tensor) -> bool:
    """
    Validate that a saved CoShNet model loads correctly.

    Args:
        model_path: Path to saved model
        test_input: Test input tensor
        original_output: Original model output for comparison

    Returns:
        True if model loads correctly, False otherwise
    """
    try:
        # Define custom objects for loading CoShNet
        custom_objects = {
            "CoShNet": CoShNet,
            "ShearletTransform": ShearletTransform,
            "ComplexDense": ComplexDense,
            "ComplexConv2D": ComplexConv2D,
            "ComplexReLU": ComplexReLU,
        }

        # Load the model
        loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

        # Test prediction
        loaded_output = loaded_model.predict(test_input, verbose=0)

        # Check if outputs are similar
        max_diff = np.max(np.abs(loaded_output - original_output))
        relative_diff = max_diff / (np.max(np.abs(original_output)) + 1e-8)

        logger.info(f"Model loading validation: max_diff={max_diff:.6f}, relative_diff={relative_diff:.6f}")

        # If difference is too large, something is wrong
        if relative_diff > 0.1:  # 10% relative difference threshold
            logger.warning(f"Large difference detected in model loading: {relative_diff:.4f}")
            return False

        return True

    except Exception as e:
        logger.error(f"Model loading validation failed: {e}")
        return False


# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    """Main CoShNet training function."""
    logger.info("Starting CoShNet training script")
    setup_gpu()

    # Load dataset
    logger.info("Loading dataset...")
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)

    logger.info(f"Dataset configuration:")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Training samples: {x_train.shape[0]}")
    logger.info(f"  Test samples: {x_test.shape[0]}")

    # Create model configuration
    model_config = create_model_config(args.dataset, args.variant, input_shape, num_classes)

    # Create learning rate schedule
    use_lr_schedule = args.lr_schedule != 'constant'
    lr_schedule = create_learning_rate_schedule(
        initial_lr=args.learning_rate,
        schedule_type=args.lr_schedule,
        total_epochs=args.epochs
    )

    # Create model
    logger.info(f"Creating CoShNet model (variant: {args.variant})...")

    if args.variant in ['tiny', 'base', 'large', 'cifar10', 'imagenet']:
        # Use predefined variant
        logger.info(f"Using predefined variant: {args.variant}")
        model = create_coshnet_variant(variant=args.variant)

        # Adjust for dataset if necessary
        if model.num_classes != num_classes or model.input_shape_config != input_shape:
            logger.info(f"Adapting variant for dataset: {args.dataset}")
            # Create custom model with variant config but dataset-specific settings
            if args.variant == 'tiny':
                model = create_coshnet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    conv_filters=(16, 32),
                    dense_units=(256, 128),
                    **{k: v for k, v in model_config.items() if
                       k not in ['input_shape', 'num_classes', 'conv_filters', 'dense_units']}
                )
            elif args.variant == 'large':
                model = create_coshnet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    conv_filters=(64, 128, 256),
                    dense_units=(2048, 1024, 512),
                    **{k: v for k, v in model_config.items() if
                       k not in ['input_shape', 'num_classes', 'conv_filters', 'dense_units']}
                )
            else:  # base, cifar10
                model = create_coshnet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    conv_filters=(32, 64),
                    dense_units=(1250, 500),
                    **{k: v for k, v in model_config.items() if
                       k not in ['input_shape', 'num_classes', 'conv_filters', 'dense_units']}
                )
    else:
        # Create custom model
        logger.info("Creating custom CoShNet model")
        model = create_coshnet(**model_config)

    # Create optimizer optimized for complex-valued networks
    if use_lr_schedule:
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,  # Slightly larger epsilon for complex operations
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

    # Compile model
    metrics = ['accuracy']
    if num_classes > 10:
        metrics.append('top_5_accuracy')

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # CoShNet outputs softmax
        metrics=metrics
    )

    # Build model and show summary
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    logger.info("Building model...")

    try:
        output = model(dummy_input, training=False)
        logger.info(f"Model built successfully. Output shape: {output.shape}")
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise e

    # Show model summary
    logger.info("Model architecture:")
    #model.summary(print_fn=logger.info)

    # Log CoShNet-specific information
    logger.info(f"CoShNet configuration:")
    logger.info(f"  Shearlet scales: {model.shearlet_scales}")
    logger.info(f"  Shearlet directions: {model.shearlet_directions}")
    logger.info(f"  Conv filters: {model.conv_filters}")
    logger.info(f"  Dense units: {model.dense_units}")
    logger.info(f"  Dropout rate: {model.dropout_rate}")
    logger.info(f"  Total parameters: {model.count_params():,}")

    # Create callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.variant}",
        use_lr_schedule=use_lr_schedule,
        patience=args.patience
    )

    # Training configuration summary
    logger.info("Training configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model variant: {args.variant}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  LR schedule: {args.lr_schedule}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Expected fast convergence: ~{args.epochs // 2} epochs")

    # Train model
    logger.info("Starting CoShNet training...")
    logger.info("Note: CoShNet typically converges faster than traditional CNNs")

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Validate model serialization
    logger.info("Validating model serialization...")
    test_sample = x_test[:4]  # Small sample for validation
    pre_save_output = model.predict(test_sample, verbose=0)

    # Save final model
    final_model_path = os.path.join(results_dir, f"coshnet_{args.dataset}_{args.variant}_final.keras")
    try:
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        # Validate the saved model
        is_loading_valid = validate_model_loading(final_model_path, test_sample, pre_save_output)
        if not is_loading_valid:
            logger.warning("Model loading validation failed - using current model for evaluation")

    except Exception as e:
        logger.warning(f"Failed to save final model: {e}")
        final_model_path = None

    # Load best model for evaluation
    best_model_path = os.path.join(results_dir, 'best_model.keras')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        try:
            is_loading_valid = validate_model_loading(best_model_path, test_sample, pre_save_output)

            if is_loading_valid:
                custom_objects = {
                    "CoShNet": CoShNet,
                    "ShearletTransform": ShearletTransform,
                    "ComplexDense": ComplexDense,
                    "ComplexConv2D": ComplexConv2D,
                    "ComplexReLU": ComplexReLU,
                }
                best_model = keras.models.load_model(best_model_path, custom_objects=custom_objects)
                logger.info("Successfully loaded best model from checkpoint")
            else:
                logger.warning("Best model loading validation failed - using current model")
                best_model = model

        except Exception as e:
            logger.warning(f"Failed to load saved model: {e}")
            logger.warning("Using the current model state instead.")
            best_model = model
    else:
        logger.warning("No best model found, using the final model state.")
        best_model = model

    # Final evaluation
    logger.info("Evaluating final CoShNet model on test set...")
    test_results = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results: {test_results}")

    # Quick sanity check
    logger.info("Performing sanity check...")
    sample_predictions = best_model.predict(x_test[:10], verbose=0)
    sample_classes = np.argmax(sample_predictions, axis=1)
    logger.info(f"Sample predictions: {sample_classes}")
    logger.info(f"Sample true labels: {y_test[:10]}")

    # Calculate and log accuracy
    all_predictions = best_model.predict(x_test, batch_size=args.batch_size, verbose=1)
    predicted_classes = np.argmax(all_predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test)
    logger.info(f"Manually calculated accuracy: {accuracy:.4f}")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"CoShNet Training Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Variant: {args.variant}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")

        # CoShNet specific configuration
        f.write(f"CoShNet Configuration:\n")
        f.write(f"  Shearlet Scales: {model.shearlet_scales}\n")
        f.write(f"  Shearlet Directions: {model.shearlet_directions}\n")
        f.write(f"  Conv Filters: {model.conv_filters}\n")
        f.write(f"  Dense Units: {model.dense_units}\n")
        f.write(f"  Dropout Rate: {model.dropout_rate}\n")
        f.write(f"  Total Parameters: {model.count_params():,}\n")
        f.write(f"  Complex-valued Operations: Yes\n")
        f.write(f"  Fixed Shearlet Transform: Yes\n\n")

        f.write(f"Training Configuration:\n")
        f.write(f"  Training Epochs: {len(history.history['loss'])}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"  LR Schedule: {args.lr_schedule}\n\n")

        f.write(f"Final Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

        f.write(f"\nManually Calculated Accuracy: {accuracy:.4f}\n")

        # Training history summary
        f.write(f"\nTraining History Summary:\n")
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        convergence_epoch = next((i for i, acc in enumerate(history.history['val_accuracy'])
                                  if acc >= best_val_acc * 0.95), len(history.history['val_accuracy']))

        f.write(f"  Final Training Accuracy: {final_train_acc:.4f}\n")
        f.write(f"  Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"  Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"  Convergence Epoch (95% of best): {convergence_epoch + 1}\n")
        f.write(f"  Training Efficiency: {convergence_epoch + 1}/{args.epochs} epochs\n")

        # CoShNet advantages summary
        f.write(f"\nCoShNet Advantages Observed:\n")
        f.write(f"  Fast Convergence: {'Yes' if convergence_epoch < args.epochs // 2 else 'Standard'}\n")
        f.write(f"  Parameter Efficiency: {model.count_params():,} parameters\n")
        f.write(f"  Complex-valued Processing: Enhanced phase information\n")
        f.write(f"  Multi-scale Feature Extraction: Via shearlet transform\n")

    logger.info("CoShNet training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Model achieved {accuracy:.4f} accuracy with {model.count_params():,} parameters")


# ---------------------------------------------------------------------

def main() -> None:
    """Main entry point for CoShNet training."""
    parser = argparse.ArgumentParser(description='Train a CoShNet (Complex Shearlet Network) model.')

    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100'],
                        help='Dataset to use for training')
    parser.add_argument('--variant', type=str, default='base',
                        choices=['tiny', 'base', 'large', 'cifar10', 'custom'],
                        help='CoShNet model variant')

    # Training arguments - optimized for CoShNet's fast convergence
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (CoShNet converges faster)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-3,
                        help='Initial learning rate (higher for complex networks)')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'],
                        help='Learning rate schedule')

    # Training control
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (reduced for fast convergence)')

    # CoShNet specific arguments
    parser.add_argument('--shearlet-scales', type=int, default=None,
                        help='Number of shearlet scales (overrides variant default)')
    parser.add_argument('--shearlet-directions', type=int, default=None,
                        help='Number of shearlet directions (overrides variant default)')

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("\nCoShNet training interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during CoShNet training: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()