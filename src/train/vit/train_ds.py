"""
Vision Transformer training script with Deep Supervision support using the unified training framework.

This script demonstrates how to train ViT models on various datasets with optional
deep supervision for improved training dynamics. Deep supervision provides:
- Better gradient flow to earlier transformer layers
- Multi-scale feature learning and supervision
- More stable training for deeper architectures
- Curriculum learning capabilities through weight scheduling
"""
import os
import argparse
import numpy as np
import keras
import tensorflow as tf
from typing import Tuple, Optional
from keras.api.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.models.vit.model_ds import ViT, create_vision_transformer

from dl_techniques.utils.train_vision.framework import (
    TrainingConfig,
    TrainingPipeline,
    DatasetBuilder,
    DataInput,
    create_argument_parser,
    config_from_args,
)

from dl_techniques.optimization import deep_supervision_schedule_builder
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

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

        # Adapt for multi-output models (deep supervision)
        train_ds = self._adapt_for_multi_output(train_ds)
        val_ds = self._adapt_for_multi_output(val_ds)

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

    def _adapt_for_multi_output(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Adapt dataset for multi-output models by replicating labels.

        For deep supervision, each output needs the same label since they all
        predict the same classes (just at different depths).

        Args:
            dataset: Dataset producing (images, labels) tuples

        Returns:
            Dataset producing (images, tuple_of_labels) for multi-output or unchanged
        """
        num_outputs = self.config.model_args.get('num_outputs', 1)

        if num_outputs > 1:
            logger.info(f"Adapting dataset for {num_outputs} outputs (deep supervision)")

            def replicate_labels(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, Tuple]:
                """Replicate labels for each model output."""
                # Create tuple of identical labels (one for each output)
                replicated_labels = tuple([labels] * num_outputs)
                return images, replicated_labels

            dataset = dataset.map(replicate_labels, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

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

        # Adapt for multi-output models (deep supervision)
        train_ds = self._adapt_for_multi_output(train_ds)
        val_ds = self._adapt_for_multi_output(val_ds)

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

    def _adapt_for_multi_output(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Adapt dataset for multi-output models by replicating labels.

        For deep supervision, each output needs the same label since they all
        predict the same classes (just at different depths).

        Args:
            dataset: Dataset producing (images, labels) tuples

        Returns:
            Dataset producing (images, tuple_of_labels) for multi-output or unchanged
        """
        num_outputs = self.config.model_args.get('num_outputs', 1)

        if num_outputs > 1:
            logger.info(f"Adapting dataset for {num_outputs} outputs (deep supervision)")

            def replicate_labels(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, Tuple]:
                """Replicate labels for each model output."""
                # Create tuple of identical labels (one for each output)
                replicated_labels = tuple([labels] * num_outputs)
                return images, replicated_labels

            dataset = dataset.map(replicate_labels, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

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

        # Adapt for multi-output models (deep supervision)
        train_ds = self._adapt_for_multi_output(train_ds)
        val_ds = self._adapt_for_multi_output(val_ds)

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

    def _adapt_for_multi_output(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Adapt dataset for multi-output models by replicating labels.

        For deep supervision, each output needs the same label since they all
        predict the same classes (just at different depths).

        Args:
            dataset: Dataset producing (images, labels) tuples

        Returns:
            Dataset producing (images, tuple_of_labels) for multi-output or unchanged
        """
        num_outputs = self.config.model_args.get('num_outputs', 1)

        if num_outputs > 1:
            logger.info(f"Adapting dataset for {num_outputs} outputs (deep supervision)")

            def replicate_labels(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, Tuple]:
                """Replicate labels for each model output."""
                # Create tuple of identical labels (one for each output)
                replicated_labels = tuple([labels] * num_outputs)
                return images, replicated_labels

            dataset = dataset.map(replicate_labels, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

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
    enable_deep_supervision = config.model_args.get('enable_deep_supervision', False)
    supervision_layer_indices = config.model_args.get('supervision_layer_indices', None)

    logger.info(f"ViT configuration: scale={scale}, patch_size={patch_size}")
    logger.info(f"Deep supervision: {'ENABLED' if enable_deep_supervision else 'DISABLED'}")

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
        enable_deep_supervision=enable_deep_supervision,
        supervision_layer_indices=supervision_layer_indices,
    )

    logger.info("Building model layers...")
    dummy_input = np.zeros((1,) + config.input_shape, dtype=np.float32)
    _ = model(dummy_input, training=False)

    logger.info("ViT model created and built successfully")

    # Log output structure based on configuration (not model.output which may not be defined)
    if enable_deep_supervision and supervision_layer_indices:
        num_outputs = 1 + len(supervision_layer_indices)
        logger.info(f"Model has {num_outputs} outputs (deep supervision enabled)")
        logger.info(f"  Output 0: Final output (primary)")
        for i, layer_idx in enumerate(supervision_layer_indices):
            logger.info(f"  Output {i+1}: Supervision from layer {layer_idx}")

        # Store output info in config for dataset adaptation
        config.model_args['num_outputs'] = num_outputs
    else:
        logger.info(f"Model has single output")
        config.model_args['num_outputs'] = 1

    return model


# =============================================================================
# DEEP SUPERVISION WEIGHT SCHEDULER CALLBACK
# =============================================================================

class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """
    Dynamic weight scheduler for deep supervision training.

    This callback automatically updates the loss weights for multi-output models
    during training according to a configurable scheduling strategy. This allows
    the model to focus on different output scales at different training phases.

    Common strategies include:
    - Linear progression from early to late transformer layers
    - Curriculum learning with progressive output activation
    - Custom sigmoid transitions for smooth weight changes
    """

    def __init__(self, config: TrainingConfig, num_outputs: int) -> None:
        """
        Initialize the deep supervision weight scheduler.

        Args:
            config: Training configuration containing scheduling parameters
            num_outputs: Number of model outputs to schedule weights for
        """
        super().__init__()
        self.config = config
        self.num_outputs = num_outputs
        self.total_epochs = config.epochs

        # Get deep supervision schedule configuration
        ds_schedule_type = config.model_args.get('ds_schedule_type', 'step_wise')
        ds_schedule_config = config.model_args.get('ds_schedule_config', {})

        # Create the scheduling function
        ds_config = {
            'type': ds_schedule_type,
            'config': ds_schedule_config
        }
        self.scheduler = deep_supervision_schedule_builder(ds_config, self.num_outputs, invert_order=False)

        # Log expected schedule behavior
        logger.info("Deep Supervision Weight Scheduler initialized")
        logger.info(f"Schedule type: {ds_schedule_type}")
        logger.info(f"Number of outputs: {num_outputs}")
        logger.info("Expected schedule:")
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weights = self.scheduler(progress)
            weights_str = ", ".join([f"{w:.4f}" for w in weights])
            logger.info(f"  Progress {progress:.2f}: [{weights_str}]")

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """
        Update model loss weights at the beginning of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Training logs dictionary (unused)

        Note:
            The training progress is computed as epoch / (total_epochs - 1) to
            ensure the final epoch corresponds to progress = 1.0.
        """
        # Compute training progress [0, 1]
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))

        # Get new weights from scheduler
        new_weights = self.scheduler(progress)

        # Update model's loss weights directly
        self.model.loss_weights = new_weights

        # Log the weight update
        weights_str = ", ".join([f"{w:.4f}" for w in new_weights])
        logger.info(f"Epoch {epoch + 1}/{self.total_epochs} (Progress: {progress:.2f}) - DS weights: [{weights_str}]")


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
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_schedule='cosine_decay',
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

    # Deep supervision arguments
    ds_group = parser.add_argument_group('Deep Supervision arguments')
    ds_group.add_argument(
        '--enable-deep-supervision',
        action='store_true',
        default=False,
        help='Enable deep supervision training'
    )
    ds_group.add_argument(
        '--ds-schedule',
        type=str,
        default='step_wise',
        choices=[
            'constant_equal', 'constant_low_to_high', 'constant_high_to_low',
            'linear_low_to_high', 'non_linear_low_to_high', 'custom_sigmoid_low_to_high',
            'scale_by_scale_low_to_high', 'cosine_annealing', 'curriculum', 'step_wise'
        ],
        help='Deep supervision weight scheduling strategy'
    )
    ds_group.add_argument(
        '--ds-threshold',
        type=float,
        default=0.5,
        help='Threshold for step_wise schedule (when to switch to final output only)'
    )
    ds_group.add_argument(
        '--supervision-layers',
        type=str,
        default=None,
        help='Comma-separated list of transformer layer indices for supervision (e.g., "3,7,11")'
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

    # Parse supervision layer indices if provided
    supervision_layer_indices = None
    if args.supervision_layers is not None:
        try:
            supervision_layer_indices = [int(x.strip()) for x in args.supervision_layers.split(',')]
            logger.info(f"Using custom supervision layer indices: {supervision_layer_indices}")
        except ValueError:
            logger.warning(f"Invalid supervision layers format: {args.supervision_layers}. Using auto-selection.")

    # Create base config from arguments
    config = config_from_args(args)

    # Override with ViT-specific settings
    config.input_shape = info['input_shape']
    config.num_classes = info['num_classes']

    # CRITICAL FIX: ViT outputs logits (Dense layer without activation)
    # so we must set from_logits=True for the loss function
    config.from_logits = True

    # Deep supervision schedule configuration
    ds_schedule_config = {}
    if args.ds_schedule == 'step_wise':
        ds_schedule_config = {'threshold': args.ds_threshold}
    elif args.ds_schedule == 'custom_sigmoid_low_to_high':
        ds_schedule_config = {
            'k': 10.0,
            'x0': 0.5,
            'transition_point': 0.25
        }

    config.model_args = {
        'scale': args.scale,
        'patch_size': patch_size,
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': dropout_rate,
        'dataset': dataset_name,
        'enable_deep_supervision': args.enable_deep_supervision,
        'supervision_layer_indices': supervision_layer_indices,
        'ds_schedule_type': args.ds_schedule,
        'ds_schedule_config': ds_schedule_config,
    }

    # Generate descriptive experiment name
    if config.experiment_name is None or 'model_' in config.experiment_name:
        timestamp = config.experiment_name.split('_')[-1] if config.experiment_name else ''
        ds_suffix = '_ds' if args.enable_deep_supervision else ''
        config.experiment_name = f"vit_{dataset_name}_{args.scale}{ds_suffix}_{timestamp}"

    return config


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_vit_model(args: argparse.Namespace) -> None:
    """
    Main training function using the unified framework with deep supervision support.

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
    logger.info(f"  Deep supervision: {config.model_args['enable_deep_supervision']}")
    if config.model_args['enable_deep_supervision']:
        logger.info(f"    Schedule: {config.model_args['ds_schedule_type']}")
        logger.info(f"    Config: {config.model_args['ds_schedule_config']}")
        if config.model_args['supervision_layer_indices']:
            logger.info(f"    Supervision layers: {config.model_args['supervision_layer_indices']}")

    # Create dataset builder
    dataset_builder = create_dataset_builder(args.dataset, config)

    # Create training pipeline
    pipeline = TrainingPipeline(config)

    # Register custom objects for model serialization
    keras.saving.get_custom_objects().update({
        'ViT': ViT,
        'TransformerLayer': TransformerLayer,
    })

    # Build model to inspect output structure
    logger.info("Building model to inspect architecture...")
    model = build_vit_model(config)

    # Determine if model has multiple outputs by checking configuration
    # Don't access model.output before compilation as it may not be defined yet
    has_multiple_outputs = (
        model.enable_deep_supervision and
        model.include_top and
        len(model.supervision_layer_indices) > 0
    )
    num_outputs = 1 + len(model.supervision_layer_indices) if has_multiple_outputs else 1

    logger.info(f"Model output structure: {num_outputs} output(s)")
    logger.info(f"Deep supervision: {model.enable_deep_supervision}")
    if has_multiple_outputs:
        logger.info(f"Supervision layers: {model.supervision_layer_indices}")

    # Create custom callback for deep supervision weight scheduling
    custom_callbacks = []
    if config.model_args['enable_deep_supervision'] and has_multiple_outputs:
        logger.info("Adding Deep Supervision Weight Scheduler callback")
        ds_callback = DeepSupervisionWeightScheduler(config, num_outputs)
        custom_callbacks.append(ds_callback)

    # CRITICAL: For multi-output models, we need to compile differently than the pipeline expects
    # We'll compile the model ourselves with the correct multi-output structure
    if has_multiple_outputs:
        logger.info("Configuring and compiling multi-output model...")

        # Build learning rate schedule
        steps_per_epoch = 50000 // config.batch_size  # Approximate
        lr_config = {
            'type': config.lr_schedule_type,
            'learning_rate': config.learning_rate,
            'decay_steps': steps_per_epoch * config.epochs,
            'warmup_steps': steps_per_epoch * 5,
            'alpha': 0.01
        }

        # Build optimizer
        opt_config = {
            'type': config.optimizer_type,
            'gradient_clipping_by_norm': 1.0
        }

        lr_schedule = learning_rate_schedule_builder(lr_config)
        optimizer = optimizer_builder(opt_config, lr_schedule)

        # Loss configuration: same loss for all outputs
        loss_fns = ['sparse_categorical_crossentropy'] * num_outputs

        # Initial equal weights (will be updated by scheduler)
        initial_weights = [1.0 / num_outputs] * num_outputs

        # Create metrics for ALL outputs using list format
        # Each element corresponds to one output
        metrics_list = []
        for i in range(num_outputs):
            metrics_list.append(['accuracy', 'sparse_categorical_accuracy'])

        logger.info(f"Loss functions: {len(loss_fns)} outputs")
        logger.info(f"Initial loss weights: {initial_weights}")
        logger.info(f"Metrics: {len(metrics_list)} outputs, each with {len(metrics_list[0])} metrics")

        # Compile model with multi-output configuration
        model.compile(
            optimizer=optimizer,
            loss=loss_fns,
            loss_weights=initial_weights,
            metrics=metrics_list
        )

        logger.info("Multi-output model compiled successfully")

        # Mark that model is pre-compiled to avoid pipeline re-compilation
        # We'll use model.fit() directly instead of pipeline.run()
        use_pipeline = False
    else:
        logger.info("Using pipeline for single-output model compilation...")
        use_pipeline = True

    # Run training
    if use_pipeline:
        # Single-output model - use pipeline as normal
        model, history = pipeline.run(
            model_builder=lambda cfg: model,
            dataset_builder=dataset_builder,
            custom_callbacks=custom_callbacks
        )
    else:
        # Multi-output model - bypass pipeline compilation
        logger.info("Training multi-output model with direct fit()...")

        # Build datasets
        train_ds, val_ds, steps_per_epoch, val_steps = dataset_builder.build()

        experiment_dir = pipeline.experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        # Create basic callbacks
        callbacks = custom_callbacks.copy()

        # Model checkpoint
        checkpoint_path = os.path.join(experiment_dir, 'best_model.keras')
        callbacks.append(ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))

        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))

        # CSV logger
        csv_path = os.path.join(experiment_dir, 'training_log.csv')
        callbacks.append(CSVLogger(csv_path))

        # Train model
        history = model.fit(
            train_ds,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        final_model_path = os.path.join(experiment_dir, 'final_model.keras')
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {pipeline.experiment_dir}")

    # Log final validation metrics
    if history:
        final_metrics = {k: v[-1] for k, v in history.history.items() if 'val_' in k}
        logger.info("Final validation metrics:")
        for metric_name, value in final_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    # Create inference model (single output) if deep supervision was used
    if config.model_args['enable_deep_supervision'] and has_multiple_outputs:
        logger.info("Creating single-output inference model...")
        try:
            # Create new model with same architecture but deep supervision disabled
            inference_model = create_vision_transformer(
                input_shape=config.input_shape,
                num_classes=config.num_classes,
                scale=config.model_args['scale'],
                patch_size=config.model_args['patch_size'],
                include_top=True,
                dropout_rate=config.model_args['dropout_rate'],
                attention_dropout_rate=config.model_args['attention_dropout_rate'],
                kernel_initializer='he_normal',
                normalization_type='layer_norm',
                enable_deep_supervision=False,  # Single output for inference
            )

            # Build inference model
            dummy_input = np.zeros((1,) + config.input_shape, dtype=np.float32)
            _ = inference_model(dummy_input, training=False)

            # Transfer weights from trained model
            # The shared layers (patch_embed, pos_embed, transformer_layers, norm, head)
            # should have matching weights
            inference_model.set_weights(model.get_weights()[:len(inference_model.weights)])

            # Save inference model
            inference_path = pipeline.experiment_dir / "inference_model.keras"
            inference_model.save(inference_path)
            logger.info(f"Inference model saved to: {inference_path}")

        except Exception as e:
            logger.warning(f"Failed to create inference model: {e}")
            logger.info("You can still use the trained model's primary output (index 0) for inference")


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