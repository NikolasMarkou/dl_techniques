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
from dl_techniques.layers.kan_linear import KANLinear
from dl_techniques.models.coshkan.model import CoshKan, create_coshkan, create_coshkan_variant


# ---------------------------------------------------------------------

def setup_gpu():
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
    """Load and preprocess dataset."""
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Convert grayscale to RGB
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

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(f"Dataset loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
    logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


# ---------------------------------------------------------------------

def create_model_config(dataset: str, variant: str, input_shape: Tuple[int, int, int], num_classes: int) -> Dict[
    str, Any]:
    """Create CoshKan model configuration based on dataset."""

    # Base configuration
    config = {
        'input_shape': input_shape,
        'num_classes': num_classes,
    }

    # Dataset-specific adjustments for CoshKan
    if dataset.lower() == 'mnist':
        config.update({
            'dropout_rate': 0.1,
            'kan_grid_size': 3,  # Smaller grid for simpler dataset
            'shearlet_scales': 3,
            'shearlet_directions': 6,
        })

    elif dataset.lower() == 'cifar10':
        config.update({
            'dropout_rate': 0.1,
            'kan_grid_size': 5,
            'shearlet_scales': 4,
            'shearlet_directions': 8,
        })

    elif dataset.lower() == 'cifar100':
        config.update({
            'dropout_rate': 0.2,  # More dropout for harder dataset
            'kan_grid_size': 7,  # Larger grid for more complex patterns
            'shearlet_scales': 5,
            'shearlet_directions': 12,
        })

    else:
        # Default configuration
        config.update({
            'dropout_rate': 0.15,
            'kan_grid_size': 5,
            'shearlet_scales': 4,
            'shearlet_directions': 8,
        })

    return config


# ---------------------------------------------------------------------

def create_learning_rate_schedule(
        initial_lr: float,
        schedule_type: str = 'cosine',
        total_epochs: int = 100,
        warmup_epochs: int = 5
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create learning rate schedule."""
    if schedule_type == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_epochs,
            alpha=0.01
        )
    elif schedule_type == 'exponential':
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_epochs // 4,
            decay_rate=0.9
        )
    else:  # constant
        return initial_lr


# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        monitor: str = 'val_accuracy',
        patience: int = 15,
        use_lr_schedule: bool = True
) -> Tuple[List, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"coshkan_{model_name}_{timestamp}")
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
                patience=5,
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
    """Validate that a saved model loads correctly."""
    try:
        # Define custom objects for loading CoshKan
        custom_objects = {
            "CoshKan": CoshKan,
            "ShearletTransform": ShearletTransform,
            "ComplexConv2D": ComplexConv2D,
            "ComplexDense": ComplexDense,
            "ComplexReLU": ComplexReLU,
            "KANLinear": KANLinear,
        }

        # Load the model
        loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

        # Test prediction
        loaded_output = loaded_model.predict(test_input, verbose=0)

        # Check if outputs are similar (they should be close but not identical due to potential differences)
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

def train_model(args: argparse.Namespace):
    """Main training function."""
    logger.info("Starting CoshKan training script")
    setup_gpu()

    # Load dataset
    logger.info("Loading dataset...")
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)

    logger.info(f"After load_dataset:")
    logger.info(f"  input_shape = {input_shape} (type: {type(input_shape)})")
    logger.info(f"  num_classes = {num_classes} (type: {type(num_classes)})")

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
    logger.info(f"Creating CoshKan model (variant: {args.variant})...")
    logger.info(f"Model will be created with input_shape: {input_shape}")

    if args.variant in ['micro', 'small', 'base', 'large', 'imagenet']:
        # Use predefined variant but update for current dataset
        model = create_coshkan_variant(args.variant)

        # For variants that don't match our dataset, we need to adjust
        if model.input_shape_config != input_shape or model.num_classes != num_classes:
            logger.info(f"Adjusting variant for dataset: {input_shape}, {num_classes} classes")

            # Get the variant config and override dataset-specific params
            variant_configs = {
                "micro": {
                    "conv_filters": (8, 16),
                    "dense_units": (64, 32),
                    "kan_units": (16,),
                    "shearlet_scales": 2,
                    "shearlet_directions": 4,
                    "kan_grid_size": 3,
                    "dropout_rate": 0.2,
                    "conv_kernel_size": 3,
                    "pool_size": 2,
                },
                "small": {
                    "conv_filters": (16, 32),
                    "dense_units": (128, 64),
                    "kan_units": (32, 16),
                    "shearlet_scales": 3,
                    "shearlet_directions": 6,
                    "kan_grid_size": 3,
                    "dropout_rate": 0.15,
                    "conv_kernel_size": 3,
                    "pool_size": 2,
                },
                "base": {
                    "conv_filters": (32, 64),
                    "dense_units": (512, 256),
                    "kan_units": (128, 64),
                    "shearlet_scales": 4,
                    "shearlet_directions": 8,
                    "kan_grid_size": 5,
                    "dropout_rate": 0.1,
                    "conv_kernel_size": 5,
                    "pool_size": 2,
                },
                "large": {
                    "conv_filters": (64, 128, 256),
                    "dense_units": (1024, 512, 256),
                    "kan_units": (256, 128, 64),
                    "shearlet_scales": 5,
                    "shearlet_directions": 12,
                    "kan_grid_size": 7,
                    "kan_spline_order": 4,
                    "dropout_rate": 0.15,
                    "conv_kernel_size": 5,
                    "pool_size": 2,
                },
                "imagenet": {
                    "conv_filters": (64, 128, 256),
                    "dense_units": (2048, 1024, 512),
                    "kan_units": (512, 256, 128),
                    "shearlet_scales": 5,
                    "shearlet_directions": 16,
                    "kan_grid_size": 8,
                    "kan_spline_order": 4,
                    "dropout_rate": 0.2,
                    "conv_kernel_size": 7,
                    "conv_strides": 2,
                    "pool_size": 3,
                },
            }

            variant_config = variant_configs[args.variant]
            # Override dataset-specific parameters
            variant_config.update(model_config)

            # Create model with corrected config
            model = create_coshkan(**variant_config)
    elif args.variant == 'cifar10':
        # Special case for CIFAR-10 variant - use model config adjustments
        model = create_coshkan_variant('cifar10')
        if num_classes != 10 or input_shape != (32, 32, 3):
            logger.info("CIFAR-10 variant requested but dataset doesn't match - using base config")
            model = create_coshkan(**model_config)
    else:
        # Create custom model with configuration
        model = create_coshkan(**model_config)

    # Create optimizer
    if use_lr_schedule:
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            clipnorm=1.0
        )

    # Compile model
    metrics = ['accuracy']
    if num_classes > 10:
        metrics.append('top_5_accuracy')

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # CoshKan outputs softmax
        metrics=metrics
    )

    # Build model and show summary
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    logger.info("Building model...")
    logger.info(f"Input shape: {input_shape}")

    # Build the model
    try:
        output = model(dummy_input, training=False)
        logger.info(f"Model built successfully. Output shape: {output.shape}")
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise e

    # Show detailed summary
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)

    # Show CoshKan specific information
    logger.info("CoshKan model details:")
    logger.info(f"  Conv filters: {model.conv_filters}")
    logger.info(f"  Dense units: {model.dense_units}")
    logger.info(f"  KAN units: {model.kan_units}")
    logger.info(f"  Shearlet scales: {model.shearlet_scales}")
    logger.info(f"  Shearlet directions: {model.shearlet_directions}")
    logger.info(f"  KAN grid size: {model.kan_grid_size}")
    logger.info(f"  Dropout rate: {model.dropout_rate}")

    # Create callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.variant}",
        use_lr_schedule=use_lr_schedule,
        patience=args.patience
    )

    # Train model
    logger.info("Starting model training...")
    logger.info(f"Training parameters:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model variant: {args.variant}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  LR schedule: {args.lr_schedule}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Weight decay: {args.weight_decay}")

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Validate model before loading for evaluation
    logger.info("Validating model serialization...")
    test_sample = x_test[:4]  # Small sample for validation
    pre_save_output = model.predict(test_sample, verbose=0)

    # Save final model
    final_model_path = os.path.join(results_dir, f"coshkan_{args.dataset}_{args.variant}_final.keras")
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
            # Validate best model loading
            is_loading_valid = validate_model_loading(best_model_path, test_sample, pre_save_output)

            if is_loading_valid:
                custom_objects = {
                    "CoshKan": CoshKan,
                    "ShearletTransform": ShearletTransform,
                    "ComplexConv2D": ComplexConv2D,
                    "ComplexDense": ComplexDense,
                    "ComplexReLU": ComplexReLU,
                    "KANLinear": KANLinear,
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
    logger.info("Evaluating final model on test set...")
    test_results = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results: {test_results}")

    # Quick sanity check - predict a few samples
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
        f.write(f"CoshKan Training Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Variant: {args.variant}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Number of Classes: {num_classes}\n")

        # Get model configuration details
        f.write(f"Conv Filters: {model.conv_filters}\n")
        f.write(f"Dense Units: {model.dense_units}\n")
        f.write(f"KAN Units: {model.kan_units}\n")
        f.write(f"Shearlet Scales: {model.shearlet_scales}\n")
        f.write(f"Shearlet Directions: {model.shearlet_directions}\n")
        f.write(f"KAN Grid Size: {model.kan_grid_size}\n")
        f.write(f"KAN Spline Order: {model.kan_spline_order}\n")
        f.write(f"KAN Activation: {model.kan_activation}\n")
        f.write(f"Dropout Rate: {model.dropout_rate}\n")

        f.write(f"Training Epochs: {len(history.history['loss'])}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"LR Schedule: {args.lr_schedule}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n\n")

        f.write(f"Final Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

        f.write(f"\nManually Calculated Accuracy: {accuracy:.4f}\n")

        # Training history summary
        f.write(f"\nTraining History Summary:\n")
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        f.write(f"  Final Training Accuracy: {final_train_acc:.4f}\n")
        f.write(f"  Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"  Best Validation Accuracy: {best_val_acc:.4f}\n")

    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")


# ---------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train a CoshKan model.')

    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--variant', type=str, default='base',
                        choices=['micro', 'small', 'base', 'large', 'cifar10', 'imagenet'],
                        help='Model variant')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'], help='Learning rate schedule')

    # Training control
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()