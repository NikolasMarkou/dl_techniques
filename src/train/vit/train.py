"""
Enhanced training script for Vision Transformer (ViT) with comprehensive visualizations.

This script demonstrates how to train the ViT model using standard Keras
workflows with model.compile() and model.fit(). The script handles data loading,
model creation, training, evaluation, and generates comprehensive visualizations
including training curves, confusion matrices, and attention visualizations.

Usage:
    python vit/train.py [--dataset cifar10] [--epochs 100] [--batch-size 128] [--scale base]
"""

import argparse
import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# Import the ViT model - adjust path as needed for your project structure
from dl_techniques.models.vit import ViT, create_vision_transformer
from dl_techniques.utils.logger import logger

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")


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


def load_mnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess MNIST dataset for ViT."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert grayscale to RGB for ViT
    x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
    x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")

    return (x_train, y_train), (x_test, y_test)


def load_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-10 dataset for ViT."""
    logger.info("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")

    return (x_train, y_train), (x_test, y_test)


def load_cifar100_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess CIFAR-100 dataset for ViT."""
    logger.info("Loading CIFAR-100 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    logger.info(f"Train data shape: {x_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")

    return (x_train, y_train), (x_test, y_test)


def create_data_augmentation(dataset: str) -> keras.Sequential:
    """Create data augmentation pipeline."""
    if dataset.lower() == 'mnist':
        # Light augmentation for MNIST
        return keras.Sequential([
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.1, 0.1),
        ], name="data_augmentation")
    else:
        # Standard augmentation for CIFAR datasets
        return keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomContrast(0.1),
            keras.layers.RandomBrightness(0.1),
        ], name="data_augmentation")


def create_model_config(dataset: str, scale: str) -> Dict[str, Any]:
    """Create ViT model configuration based on dataset."""
    if dataset.lower() == 'mnist':
        return {
            'input_shape': (28, 28, 3),
            'num_classes': 10,
            'scale': scale,
            'patch_size': 4,  # Smaller patches for smaller images
            'include_top': True,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
            'kernel_initializer': 'he_normal',
            'norm_type': 'layer',
        }
    elif dataset.lower() == 'cifar10':
        return {
            'input_shape': (32, 32, 3),
            'num_classes': 10,
            'scale': scale,
            'patch_size': 4,  # Smaller patches for smaller images
            'include_top': True,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
            'kernel_initializer': 'he_normal',
            'norm_type': 'layer',
        }
    elif dataset.lower() == 'cifar100':
        return {
            'input_shape': (32, 32, 3),
            'num_classes': 100,
            'scale': scale,
            'patch_size': 4,  # Smaller patches for smaller images
            'include_top': True,
            'dropout_rate': 0.2,  # Higher dropout for more classes
            'attention_dropout_rate': 0.2,
            'kernel_initializer': 'he_normal',
            'norm_type': 'layer',
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def plot_sample_images(
        x_data: np.ndarray,
        y_data: np.ndarray,
        class_names: List[str],
        save_path: str,
        dataset: str,
        n_samples: int = 20
) -> None:
    """Plot sample images from the dataset."""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.flatten()

    # Select random samples
    indices = np.random.choice(len(x_data), size=n_samples, replace=False)

    for i, idx in enumerate(indices):
        img = x_data[idx]
        label = y_data[idx]

        if dataset.lower() == 'mnist':
            # Convert RGB back to grayscale for display
            img_display = np.mean(img, axis=-1)
            axes[i].imshow(img_display, cmap='gray')
        else:
            axes[i].imshow(img)

        axes[i].set_title(f'{class_names[label]}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Sample Images from {dataset.upper()} Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str,
        normalize: bool = True
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: keras.callbacks.History, save_dir: str) -> None:
    """Plot training history curves."""
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Vision Transformer Training History", fontsize=16, fontweight='bold')

    # Training and Validation Loss
    axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Training and Validation Accuracy
    axes[0, 1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate
    if 'lr' in history_dict:
        axes[1, 0].plot(epochs, history_dict['lr'], 'g-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Top-5 Accuracy (if available)
    if 'top_5_accuracy' in history_dict:
        axes[1, 1].plot(epochs, history_dict['top_5_accuracy'], 'b-', label='Training Top-5')
        axes[1, 1].plot(epochs, history_dict['val_top_5_accuracy'], 'r-', label='Validation Top-5')
        axes[1, 1].set_title('Top-5 Accuracy', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-5 Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Loss zoom-in for better visualization
        axes[1, 1].plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
        axes[1, 1].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
        axes[1, 1].set_title('Loss (Zoomed)', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_ylim(0, max(min(history_dict['loss']), min(history_dict['val_loss'])) * 2)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str
) -> None:
    """Plot per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(class_names)), per_class_accuracy, color='steelblue', alpha=0.7)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class ViTVisualizationCallback(keras.callbacks.Callback):
    """Callback to generate ViT visualizations during training."""

    def __init__(
            self,
            validation_data: Tuple[np.ndarray, np.ndarray],
            class_names: List[str],
            save_dir: str,
            dataset: str,
            frequency: int = 10
    ):
        super().__init__()
        self.x_val, self.y_val = validation_data
        self.class_names = class_names
        self.save_dir = save_dir
        self.dataset = dataset
        self.frequency = frequency

        # Create directories
        self.metrics_dir = os.path.join(save_dir, 'metrics_per_epoch')
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Sample some validation data for quick evaluation
        self.sample_size = min(1000, len(self.x_val))
        indices = np.random.choice(len(self.x_val), size=self.sample_size, replace=False)
        self.x_sample = self.x_val[indices]
        self.y_sample = self.y_val[indices]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency == 0:
            logger.info(f"Generating visualizations for epoch {epoch + 1}...")

            # Get predictions
            y_pred = self.model.predict(self.x_sample, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)

            # Plot confusion matrix
            cm_path = os.path.join(self.metrics_dir, f'confusion_matrix_epoch_{epoch + 1:03d}.png')
            plot_confusion_matrix(self.y_sample, y_pred_classes, self.class_names, cm_path)

            # Plot per-class accuracy
            acc_path = os.path.join(self.metrics_dir, f'class_accuracy_epoch_{epoch + 1:03d}.png')
            plot_class_accuracy(self.y_sample, y_pred_classes, self.class_names, acc_path)


def get_class_names(dataset: str) -> List[str]:
    """Get class names for the dataset."""
    if dataset.lower() == 'mnist':
        return [str(i) for i in range(10)]
    elif dataset.lower() == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset.lower() == 'cifar100':
        return [f'class_{i}' for i in range(100)]  # CIFAR-100 has many classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def create_callbacks(
        model_name: str,
        validation_data: Tuple[np.ndarray, np.ndarray],
        class_names: List[str],
        dataset: str,
        monitor: str = 'val_accuracy',
        patience: int = 15,
        viz_frequency: int = 10
) -> Tuple[List, str]:
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"vit_{model_name}_{timestamp}")
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
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv')
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        ViTVisualizationCallback(
            validation_data,
            class_names,
            results_dir,
            dataset,
            viz_frequency
        )
    ]

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


def create_learning_rate_schedule(
        initial_lr: float,
        schedule_type: str = 'cosine',
        warmup_epochs: int = 5,
        total_epochs: int = 100
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create learning rate schedule."""
    if schedule_type == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_epochs,
            alpha=0.01  # Final learning rate will be 1% of initial
        )
    elif schedule_type == 'exponential':
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_epochs // 4,
            decay_rate=0.9
        )
    elif schedule_type == 'constant':
        return initial_lr
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")


def train_model(args: argparse.Namespace):
    """Main training function."""
    logger.info("Starting Vision Transformer training script")
    setup_gpu()

    # Load dataset
    if args.dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    elif args.dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    elif args.dataset.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = load_cifar100_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    class_names = get_class_names(args.dataset)

    # Create model configuration
    model_config = create_model_config(args.dataset, args.scale)

    # Create learning rate schedule
    lr_schedule = create_learning_rate_schedule(
        initial_lr=args.learning_rate,
        schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs
    )

    # Create and compile model
    model = create_vision_transformer(**model_config)

    # Add data augmentation if specified
    if args.augment:
        augmentation = create_data_augmentation(args.dataset)
        inputs = keras.Input(shape=model_config['input_shape'])
        x = augmentation(inputs, training=True)
        outputs = model(x)
        model = keras.Model(inputs, outputs, name=f"augmented_{model.name}")

    # Compile model
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay
    )

    metrics = ['accuracy']
    if model_config['num_classes'] > 10:
        metrics.append('top_5_accuracy')

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    # Print model summary
    model.summary(print_fn=logger.info)

    # Create callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.scale}",
        validation_data=(x_test, y_test),
        class_names=class_names,
        dataset=args.dataset,
        patience=args.patience,
        viz_frequency=args.viz_frequency
    )

    # Plot sample images
    sample_images_path = os.path.join(results_dir, 'sample_images.png')
    plot_sample_images(x_train, y_train, class_names, sample_images_path, args.dataset)

    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Load best model for evaluation
    logger.info("Training completed. Evaluating on test set...")
    best_model_path = os.path.join(results_dir, 'best_model.keras')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        best_model = keras.models.load_model(best_model_path)
    else:
        logger.warning("No best model found, using the final model state.")
        best_model = model

    # Evaluate model
    test_results = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results: {test_results}")

    # Generate predictions for detailed analysis
    logger.info("Generating predictions for detailed analysis...")
    y_pred = best_model.predict(x_test, batch_size=args.batch_size, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate final visualizations
    logger.info("Generating final visualizations...")
    plot_training_history(history, results_dir)

    # Confusion matrix
    final_cm_path = os.path.join(results_dir, 'final_confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred_classes, class_names, final_cm_path)

    # Per-class accuracy
    final_acc_path = os.path.join(results_dir, 'final_class_accuracy.png')
    plot_class_accuracy(y_test, y_pred_classes, class_names, final_acc_path)

    # Classification report
    report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)

    # Save final model
    final_model_path = os.path.join(results_dir, f"vit_{args.dataset}_{args.scale}_final.keras")
    best_model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"Vision Transformer Training Summary\n")
        f.write(f"===================================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Scale: {args.scale}\n")
        f.write(f"Input Shape: {model_config['input_shape']}\n")
        f.write(f"Number of Classes: {model_config['num_classes']}\n")
        f.write(f"Patch Size: {model_config['patch_size']}\n")
        f.write(f"Training Epochs: {len(history.history['loss'])}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Data Augmentation: {args.augment}\n\n")

        f.write(f"Final Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

        f.write(f"\nPer-Class Metrics:\n")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                f.write(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                        f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}\n")

    logger.info("Training completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train a Vision Transformer (ViT) on image classification tasks.')

    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--scale', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large', 'huge'], help='Model scale')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'], help='Learning rate schedule')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs for learning rate')

    # Regularization and augmentation
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    # Visualization arguments
    parser.add_argument('--viz-frequency', type=int, default=10,
                        help='Frequency of visualization callbacks (in epochs)')

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()