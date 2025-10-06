"""
Vision Transformer training script using the unified training framework.

This script demonstrates how to train ViT models on various datasets
using the model-agnostic training framework with comprehensive automatic
visualization and analysis.
"""

import argparse
import numpy as np
import keras
import tensorflow as tf
from typing import Tuple, Optional, List

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.models.vit.model import ViT, create_vision_transformer

from dl_techniques.optimization.train_vision import (
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

    def __init__(self, config: TrainingConfig):
        """
        Initialize MNIST dataset builder.

        Args:
            config: Training configuration.
        """
        super().__init__(config)
        self.x_test = None
        self.y_test = None
        self.class_names = [str(i) for i in range(10)]  # '0' through '9'

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
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """Get class names for visualization."""
        return self.class_names


class CIFAR10DatasetBuilder(DatasetBuilder):
    """
    Dataset builder for CIFAR-10.

    Applies normalization and standard augmentation techniques.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize CIFAR-10 dataset builder.

        Args:
            config: Training configuration.
        """
        super().__init__(config)
        self.x_test = None
        self.y_test = None
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

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
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """Get class names for visualization."""
        return self.class_names


class CIFAR100DatasetBuilder(DatasetBuilder):
    """
    Dataset builder for CIFAR-100.

    Applies normalization and augmentation suitable for the more complex dataset.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize CIFAR-100 dataset builder.

        Args:
            config: Training configuration.
        """
        super().__init__(config)
        self.x_test = None
        self.y_test = None
        # CIFAR-100 fine labels
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

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
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """Get class names for visualization."""
        return self.class_names


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

    # Set ViT-appropriate defaults for Transformer training
    parser.set_defaults(
        optimizer='adamw',
        learning_rate=3e-4,       # Conservative for transformers
        weight_decay=0.05,         # Standard transformer weight decay
        lr_schedule='cosine_decay',
        warmup_steps=2000,         # Longer warmup for transformers
        alpha=0.0001,
        gradient_clip=1.0,
        # Enable comprehensive visualization by default
        no_visualization=False,
        no_analysis=False,
        no_convergence_analysis=False,
        no_overfitting_analysis=False,
        no_classification_viz=False,
        no_final_dashboard=False,
        enable_gradient_tracking=False,
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

    Also configures transformer-appropriate hyperparameters including:
    - Higher beta_2 (0.98) for transformers
    - Longer warmup (2000 steps minimum)
    - Conservative learning rate
    - Comprehensive visualization enabled by default

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

    # Transformer-specific optimizer settings
    config.beta_2 = 0.98  # Higher beta_2 for transformers (instead of default 0.999)
    config.epsilon = 1e-9  # Smaller epsilon for transformers

    # Ensure adequate warmup for transformer stability
    if config.warmup_steps < 1000:
        logger.warning(
            f"Warmup steps ({config.warmup_steps}) is low for transformer training. "
            f"Setting to 2000 for stability."
        )
        config.warmup_steps = 2000
        config.warmup_start_lr = 1e-9  # Very low starting LR for transformers

    # Enable comprehensive visualization and analysis by default
    # (unless explicitly disabled via command line)
    if not hasattr(args, 'no_visualization') or not args.no_visualization:
        config.enable_visualization = True
        config.enable_convergence_analysis = True
        config.enable_overfitting_analysis = True
        config.enable_classification_viz = True
        config.create_final_dashboard = True
        config.visualization_frequency = 10

    if not hasattr(args, 'no_analysis') or not args.no_analysis:
        config.enable_analysis = True

    config.model_args = {
        'scale': args.scale,
        'patch_size': patch_size,
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': dropout_rate,
        'dataset': dataset_name,
    }

    # Generate descriptive experiment name
    if config.experiment_name is None or 'model_' in config.experiment_name:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    logger.info("=" * 80)
    logger.info("Starting Vision Transformer training with unified framework")
    logger.info("=" * 80)

    config = create_vit_config(args)

    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model scale: {config.model_args['scale']}")
    logger.info(f"Input shape: {config.input_shape}")
    logger.info(f"Number of classes: {config.num_classes}")
    logger.info(f"Patch size: {config.model_args['patch_size']}")
    logger.info("")
    logger.info("Hyperparameters:")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Optimizer: {config.optimizer_type}")
    logger.info(f"  Beta_2: {config.beta_2} (transformer-optimized)")
    logger.info(f"  Weight decay: {config.weight_decay}")
    logger.info(f"  Gradient clipping: {config.gradient_clipping_norm_global}")
    logger.info("")
    logger.info("Learning Rate Schedule:")
    logger.info(f"  Schedule type: {config.lr_schedule_type}")
    logger.info(f"  Warmup steps: {config.warmup_steps}")
    logger.info(f"  Warmup start LR: {config.warmup_start_lr}")
    logger.info(f"  Alpha (min LR): {config.alpha}")
    logger.info("")
    logger.info("Training Settings:")
    logger.info(f"  From logits: {config.from_logits} (CRITICAL for ViT)")
    logger.info(f"  Early stopping patience: {config.early_stopping_patience}")
    logger.info(f"  Monitor metric: {config.monitor_metric}")
    logger.info("")
    logger.info("Visualization & Analysis:")
    logger.info(f"  Visualization enabled: {config.enable_visualization}")
    logger.info(f"  Convergence analysis: {config.enable_convergence_analysis}")
    logger.info(f"  Overfitting analysis: {config.enable_overfitting_analysis}")
    logger.info(f"  Classification viz: {config.enable_classification_viz}")
    logger.info(f"  Final dashboard: {config.create_final_dashboard}")
    logger.info(f"  Analysis enabled: {config.enable_analysis}")
    logger.info(f"  Gradient tracking: {config.enable_gradient_tracking}")
    logger.info(f"  Visualization frequency: {config.visualization_frequency} epochs")
    logger.info("=" * 80 + "\n")

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
    logger.info("Starting training pipeline...")
    model, history = pipeline.run(
        model_builder=build_vit_model,
        dataset_builder=dataset_builder,
        custom_callbacks=None
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {pipeline.experiment_dir}")
    logger.info("")

    # Log final validation metrics
    if history:
        logger.info("Final validation metrics:")
        final_metrics = {k: v[-1] for k, v in history.history.items() if 'val_' in k}
        for metric_name, value in final_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        logger.info("")

    # Log output files
    logger.info("Generated outputs:")
    logger.info(f"  Configuration: {pipeline.experiment_dir / 'config.json'}")
    logger.info(f"  Best model: {pipeline.experiment_dir / 'best_model.keras'}")
    logger.info(f"  Final model: {pipeline.experiment_dir / 'final_model.keras'}")
    logger.info(f"  Training log: {pipeline.experiment_dir / 'training_log.csv'}")

    if config.enable_visualization:
        logger.info(f"  Visualizations: {pipeline.experiment_dir / 'visualizations/'}")
        logger.info("    - training_curves.png")
        logger.info("    - lr_schedule.png")
        if config.enable_convergence_analysis:
            logger.info("    - convergence_analysis.png")
        if config.enable_overfitting_analysis:
            logger.info("    - overfitting_analysis.png")
        if config.enable_classification_viz:
            logger.info("    - confusion_matrix.png")
            logger.info("    - roc_pr_curves.png")
            logger.info("    - classification_report.png")
            logger.info("    - per_class_analysis.png")
            logger.info("    - error_analysis.png")
        if config.create_final_dashboard:
            logger.info("    - final_dashboard.png")

    if config.enable_analysis:
        logger.info(f"  Analysis results: {pipeline.experiment_dir / 'analysis/'}")

    logger.info("=" * 80)


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
        logger.info("=" * 80)
        logger.info("Training interrupted by user.")
        logger.info("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error("AN ERROR OCCURRED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()