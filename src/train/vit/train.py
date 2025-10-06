"""
Vision Transformer training script using the unified training framework.

This script demonstrates how to train ViT models on various datasets
using the model-agnostic training framework with automatic visualization
and analysis.
"""

import argparse
import numpy as np
import keras
import tensorflow as tf
from typing import Tuple, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.models.vit.model import ViT, create_vision_transformer

from dl_techniques.utils.train_vision.framework import (
    TrainingConfig,
    TrainingPipeline,
    DatasetBuilder,
    DataInput,
    create_argument_parser,
    config_from_args,
)


# =============================================================================
# DATASET BUILDERS
# =============================================================================

class MNISTDatasetBuilder(DatasetBuilder):
    """
    Dataset builder for MNIST.

    Converts grayscale images to RGB by repeating channels and applies
    normalization and augmentation.
    """

    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """
        Build MNIST training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, steps_per_epoch, val_steps).
        """
        logger.info("Loading MNIST dataset...")

        # Load data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Convert grayscale to RGB
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

        # Normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        logger.info(
            f"MNIST loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples"
        )

        # Store test data for analysis
        self.x_test = x_test
        self.y_test = y_test

        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # Apply preprocessing and augmentation
        train_ds = (train_ds
            .shuffle(10000)
            .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE))

        val_ds = (val_ds
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE))

        # Calculate steps
        steps_per_epoch = len(x_train) // self.config.batch_size
        val_steps = len(x_test) // self.config.batch_size

        return train_ds, val_ds, steps_per_epoch, val_steps

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply light augmentation for MNIST."""
        # Random shift
        image = tf.image.random_crop(
            tf.pad(image, [[2, 2], [2, 2], [0, 0]], mode='CONSTANT'),
            size=tf.shape(image)
        )
        return image, label

    def get_test_data(self) -> Optional[DataInput]:
        """Get test data for analysis."""
        return DataInput(x_data=self.x_test, y_data=self.y_test)


class CIFAR10DatasetBuilder(DatasetBuilder):
    """
    Dataset builder for CIFAR-10.

    Applies normalization and standard augmentation techniques.
    """

    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """
        Build CIFAR-10 training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, steps_per_epoch, val_steps).
        """
        logger.info("Loading CIFAR-10 dataset...")

        # Load data
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # Normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        logger.info(
            f"CIFAR-10 loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples"
        )

        # Store test data for analysis
        self.x_test = x_test
        self.y_test = y_test

        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # Apply preprocessing and augmentation
        train_ds = (train_ds
            .shuffle(10000)
            .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE))

        val_ds = (val_ds
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE))

        # Calculate steps
        steps_per_epoch = len(x_train) // self.config.batch_size
        val_steps = len(x_test) // self.config.batch_size

        return train_ds, val_ds, steps_per_epoch, val_steps

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply standard augmentation for CIFAR-10."""
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random crop with padding
        image = tf.image.random_crop(
            tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT'),
            size=tf.shape(image)
        )

        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Clip to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def get_test_data(self) -> Optional[DataInput]:
        """Get test data for analysis."""
        return DataInput(x_data=self.x_test, y_data=self.y_test)


class CIFAR100DatasetBuilder(DatasetBuilder):
    """
    Dataset builder for CIFAR-100.

    Applies normalization and augmentation suitable for the more complex dataset.
    """

    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """
        Build CIFAR-100 training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, steps_per_epoch, val_steps).
        """
        logger.info("Loading CIFAR-100 dataset...")

        # Load data
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # Normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        logger.info(
            f"CIFAR-100 loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples"
        )

        # Store test data for analysis
        self.x_test = x_test
        self.y_test = y_test

        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # Apply preprocessing and augmentation
        train_ds = (train_ds
            .shuffle(10000)
            .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE))

        val_ds = (val_ds
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE))

        # Calculate steps
        steps_per_epoch = len(x_train) // self.config.batch_size
        val_steps = len(x_test) // self.config.batch_size

        return train_ds, val_ds, steps_per_epoch, val_steps

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply augmentation for CIFAR-100."""
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random crop with padding
        image = tf.image.random_crop(
            tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT'),
            size=tf.shape(image)
        )

        # Stronger brightness and contrast adjustments
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.image.random_saturation(image, lower=0.85, upper=1.15)

        # Clip to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def get_test_data(self) -> Optional[DataInput]:
        """Get test data for analysis."""
        return DataInput(x_data=self.x_test, y_data=self.y_test)


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_vit_model(config: TrainingConfig) -> keras.Model:
    """
    Build a Vision Transformer model based on configuration.

    Args:
        config: Training configuration containing model parameters.

    Returns:
        Built ViT model ready for compilation.
    """
    logger.info("Building Vision Transformer model...")

    scale = config.model_args.get('scale', 'tiny')
    patch_size = config.model_args.get('patch_size', 4)
    dropout_rate = config.model_args.get('dropout_rate', 0.1)
    attention_dropout_rate = config.model_args.get('attention_dropout_rate', 0.1)

    logger.info(f"ViT configuration: scale={scale}, patch_size={patch_size}")

    model = create_vision_transformer(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        scale=scale,
        patch_size=patch_size,
        include_top=True,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        kernel_initializer='he_normal',
        normalization_type='layer_norm',
    )

    logger.info("Building model layers...")
    dummy_input = np.zeros((1,) + config.input_shape, dtype=np.float32)
    _ = model(dummy_input, training=False)

    logger.info("ViT model created and built successfully")
    return model


# =============================================================================
# DATASET BUILDER FACTORY
# =============================================================================

def create_dataset_builder(
    dataset_name: str,
    config: TrainingConfig
) -> DatasetBuilder:
    """
    Factory function to create the appropriate dataset builder.

    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar100').
        config: Training configuration.

    Returns:
        Instantiated DatasetBuilder.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'mnist':
        return MNISTDatasetBuilder(config)
    elif dataset_name == 'cifar10':
        return CIFAR10DatasetBuilder(config)
    elif dataset_name == 'cifar100':
        return CIFAR100DatasetBuilder(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def create_vit_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with ViT-specific options and defaults.

    Overrides framework defaults with values required for stable ViT training.
    """
    parser = create_argument_parser()

    # Set ViT-appropriate defaults
    parser.set_defaults(
        optimizer='adamw',
        learning_rate=3e-4,
        weight_decay=0.05,
        lr_schedule='cosine',
    )

    vit_group = parser.add_argument_group('ViT-specific arguments')
    vit_group.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['mnist', 'cifar10', 'cifar100'],
        help='Dataset to use for training'
    )
    vit_group.add_argument(
        '--scale',
        type=str,
        default='pico',
        choices=['pico', 'tiny', 'small', 'base', 'large', 'huge'],
        help='ViT model scale'
    )
    vit_group.add_argument(
        '--patch-size',
        type=int,
        default=None,
        help='Patch size (auto-selected if not specified)'
    )
    vit_group.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Dropout rate (auto-selected if not specified)'
    )

    return parser


# =============================================================================
# CONFIGURATION HELPER
# =============================================================================

def create_vit_config(args: argparse.Namespace) -> TrainingConfig:
    """
    Create TrainingConfig from arguments with ViT-specific settings.

    CRITICAL: Sets from_logits=True because ViT model outputs raw logits
    (Dense layer without activation).

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured TrainingConfig instance.
    """
    # Dataset-specific settings
    dataset_info = {
        'mnist': {'input_shape': (28, 28, 3), 'num_classes': 10},
        'cifar10': {'input_shape': (32, 32, 3), 'num_classes': 10},
        'cifar100': {'input_shape': (32, 32, 3), 'num_classes': 100},
    }
    dataset_name = args.dataset.lower()
    info = dataset_info[dataset_name]

    # Auto-select patch size if not specified
    if args.patch_size is None:
        patch_size = 4 if dataset_name in ['mnist', 'cifar10', 'cifar100'] else 16
    else:
        patch_size = args.patch_size

    # Auto-select dropout rate if not specified
    if args.dropout is None:
        if dataset_name in ['mnist', 'cifar10']:
            dropout_rate = 0.1
        else:  # cifar100
            dropout_rate = 0.2
    else:
        dropout_rate = args.dropout

    # Create base config from arguments
    config = config_from_args(args)

    # Override with ViT-specific settings
    config.input_shape = info['input_shape']
    config.num_classes = info['num_classes']

    # CRITICAL FIX: ViT outputs logits (Dense layer without activation)
    # so we must set from_logits=True for the loss function
    config.from_logits = True

    config.model_args = {
        'scale': args.scale,
        'patch_size': patch_size,
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': dropout_rate,
        'dataset': dataset_name,
    }

    # Generate descriptive experiment name
    if config.experiment_name is None or 'model_' in config.experiment_name:
        timestamp = config.experiment_name.split('_')[-1] if config.experiment_name else ''
        config.experiment_name = f"vit_{dataset_name}_{args.scale}_{timestamp}"

    return config


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_vit_model(args: argparse.Namespace) -> None:
    """
    Main training function using the unified framework.

    Args:
        args: Parsed command-line arguments.
    """
    logger.info("Starting Vision Transformer training with unified framework")
    config = create_vit_config(args)

    logger.info("Training configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Model scale: {config.model_args['scale']}")
    logger.info(f"  Input shape: {config.input_shape}")
    logger.info(f"  Number of classes: {config.num_classes}")
    logger.info(f"  Patch size: {config.model_args['patch_size']}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Optimizer: {config.optimizer_type}")
    logger.info(f"  LR schedule: {config.lr_schedule_type}")
    logger.info(f"  Weight decay: {config.weight_decay}")
    logger.info(f"  From logits: {config.from_logits}")  # Log this critical setting

    # Create dataset builder
    dataset_builder = create_dataset_builder(args.dataset, config)

    # Create training pipeline
    pipeline = TrainingPipeline(config)

    # Register custom objects for model serialization
    keras.saving.get_custom_objects().update({
        'ViT': ViT,
        'TransformerLayer': TransformerLayer,
    })

    # Run training
    model, history = pipeline.run(
        model_builder=build_vit_model,
        dataset_builder=dataset_builder,
        custom_callbacks=None
    )

    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {pipeline.experiment_dir}")

    # Log final validation metrics
    if history:
        final_metrics = {k: v[-1] for k, v in history.history.items() if 'val_' in k}
        logger.info("Final validation metrics:")
        for metric_name, value in final_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point for the training script."""
    parser = create_vit_argument_parser()
    args = parser.parse_args()

    try:
        train_vit_model(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()