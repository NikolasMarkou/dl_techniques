"""
Vision Transformer training script using the unified training framework.

Trains ViT models on MNIST/CIFAR-10/CIFAR-100 with comprehensive
automatic visualization and analysis via TrainingPipeline.
"""

import argparse
import numpy as np
import keras
import tensorflow as tf
from typing import Tuple, Optional, List

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers import TransformerLayer
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
    """Dataset builder for MNIST (grayscale->RGB, normalization, augmentation)."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.x_test = None
        self.y_test = None
        self.class_names = [str(i) for i in range(10)]

    def build(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int], Optional[int]]:
        logger.info("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1).astype("float32") / 255.0
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1).astype("float32") / 255.0

        logger.info(f"MNIST: {x_train.shape[0]} train, {x_test.shape[0]} test")
        self.x_test, self.y_test = x_test, y_test

        train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat().batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE))

        val_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE))

        return train_ds, val_ds, len(x_train) // self.config.batch_size, len(x_test) // self.config.batch_size

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_crop(
            tf.pad(image, [[2, 2], [2, 2], [0, 0]], mode='CONSTANT'),
            size=tf.shape(image)
        )
        return image, label

    def get_test_data(self) -> Optional[DataInput]:
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        return self.class_names


class CIFAR10DatasetBuilder(DatasetBuilder):
    """Dataset builder for CIFAR-10 with standard augmentation."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.x_test = None
        self.y_test = None
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

    def build(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int], Optional[int]]:
        logger.info("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train, y_test = y_train.flatten(), y_test.flatten()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        logger.info(f"CIFAR-10: {x_train.shape[0]} train, {x_test.shape[0]} test")
        self.x_test, self.y_test = x_test, y_test

        train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat().batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE))

        val_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE))

        return train_ds, val_ds, len(x_train) // self.config.batch_size, len(x_test) // self.config.batch_size

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(
            tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT'),
            size=tf.shape(image)
        )
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return tf.clip_by_value(image, 0.0, 1.0), label

    def get_test_data(self) -> Optional[DataInput]:
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        return self.class_names


class CIFAR100DatasetBuilder(DatasetBuilder):
    """Dataset builder for CIFAR-100 with stronger augmentation."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.x_test = None
        self.y_test = None
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

    def build(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int], Optional[int]]:
        logger.info("Loading CIFAR-100 dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        y_train, y_test = y_train.flatten(), y_test.flatten()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        logger.info(f"CIFAR-100: {x_train.shape[0]} train, {x_test.shape[0]} test")
        self.x_test, self.y_test = x_test, y_test

        train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat().batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE))

        val_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE))

        return train_ds, val_ds, len(x_train) // self.config.batch_size, len(x_test) // self.config.batch_size

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(
            tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT'),
            size=tf.shape(image)
        )
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
        return tf.clip_by_value(image, 0.0, 1.0), label

    def get_test_data(self) -> Optional[DataInput]:
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        return self.class_names


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_vit_model(config: TrainingConfig) -> keras.Model:
    """Build a Vision Transformer model based on configuration."""
    scale = config.model_args.get('scale', 'tiny')
    patch_size = config.model_args.get('patch_size', 4)
    dropout_rate = config.model_args.get('dropout_rate', 0.1)
    attention_dropout_rate = config.model_args.get('attention_dropout_rate', 0.1)

    logger.info(f"Building ViT: scale={scale}, patch_size={patch_size}")

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

    dummy_input = np.zeros((1,) + config.input_shape, dtype=np.float32)
    _ = model(dummy_input, training=False)
    logger.info("ViT model built successfully")
    return model


# =============================================================================
# DATASET BUILDER FACTORY
# =============================================================================

def create_dataset_builder(dataset_name: str, config: TrainingConfig) -> DatasetBuilder:
    """Factory to create the appropriate dataset builder."""
    builders = {
        'mnist': MNISTDatasetBuilder,
        'cifar10': CIFAR10DatasetBuilder,
        'cifar100': CIFAR100DatasetBuilder,
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in builders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return builders[dataset_name](config)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def create_vit_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with ViT-specific options and defaults."""
    parser = create_argument_parser()

    parser.set_defaults(
        optimizer='adamw', learning_rate=3e-4, weight_decay=0.05,
        lr_schedule='cosine_decay', warmup_steps=2000, alpha=0.0001,
        gradient_clip=1.0, no_visualization=False, no_analysis=False,
        no_convergence_analysis=False, no_overfitting_analysis=False,
        no_classification_viz=False, no_final_dashboard=False,
        enable_gradient_tracking=False,
    )

    vit_group = parser.add_argument_group('ViT-specific arguments')
    vit_group.add_argument('--dataset', type=str, default='cifar10',
                           choices=['mnist', 'cifar10', 'cifar100'], help='Dataset to use')
    vit_group.add_argument('--scale', type=str, default='pico',
                           choices=['pico', 'tiny', 'small', 'base', 'large', 'huge'], help='ViT model scale')
    vit_group.add_argument('--patch-size', type=int, default=None,
                           help='Patch size (auto-selected if not specified)')
    vit_group.add_argument('--dropout', type=float, default=None,
                           help='Dropout rate (auto-selected if not specified)')

    return parser


# =============================================================================
# CONFIGURATION
# =============================================================================

def create_vit_config(args: argparse.Namespace) -> TrainingConfig:
    """Create TrainingConfig with ViT-specific settings.

    Sets from_logits=True (ViT outputs raw logits) and configures
    transformer-appropriate hyperparameters (higher beta_2, longer warmup).
    """
    dataset_info = {
        'mnist': {'input_shape': (28, 28, 3), 'num_classes': 10},
        'cifar10': {'input_shape': (32, 32, 3), 'num_classes': 10},
        'cifar100': {'input_shape': (32, 32, 3), 'num_classes': 100},
    }
    dataset_name = args.dataset.lower()
    info = dataset_info[dataset_name]

    patch_size = args.patch_size or (4 if dataset_name in ['mnist', 'cifar10', 'cifar100'] else 16)

    if args.dropout is None:
        dropout_rate = 0.2 if dataset_name == 'cifar100' else 0.1
    else:
        dropout_rate = args.dropout

    config = config_from_args(args)
    config.input_shape = info['input_shape']
    config.num_classes = info['num_classes']
    config.from_logits = True

    # Transformer-specific optimizer settings
    config.beta_2 = 0.98
    config.epsilon = 1e-9

    if config.warmup_steps < 1000:
        logger.warning(
            f"Warmup steps ({config.warmup_steps}) low for transformers, setting to 2000"
        )
        config.warmup_steps = 2000
        config.warmup_start_lr = 1e-9

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

    if config.experiment_name is None or 'model_' in config.experiment_name:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"vit_{dataset_name}_{args.scale}_{timestamp}"

    return config


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_vit_model(args: argparse.Namespace) -> None:
    """Main training function using the unified framework."""
    logger.info("=" * 80)
    logger.info("Starting Vision Transformer training")
    logger.info("=" * 80)

    config = create_vit_config(args)

    logger.info(f"Dataset: {args.dataset}, Scale: {config.model_args['scale']}, "
                f"Input: {config.input_shape}, Classes: {config.num_classes}")
    logger.info(f"Batch: {config.batch_size}, Epochs: {config.epochs}, "
                f"LR: {config.learning_rate}, Optimizer: {config.optimizer_type}")
    logger.info(f"Schedule: {config.lr_schedule_type}, Warmup: {config.warmup_steps}, "
                f"Weight decay: {config.weight_decay}, from_logits: {config.from_logits}")

    dataset_builder = create_dataset_builder(args.dataset, config)
    pipeline = TrainingPipeline(config)

    keras.saving.get_custom_objects().update({
        'ViT': ViT,
        'TransformerLayer': TransformerLayer,
    })

    model, history = pipeline.run(
        model_builder=build_vit_model,
        dataset_builder=dataset_builder,
        custom_callbacks=None
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Results saved to: {pipeline.experiment_dir}")

    if history:
        final_metrics = {k: v[-1] for k, v in history.history.items() if 'val_' in k}
        for name, value in final_metrics.items():
            logger.info(f"  {name}: {value:.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = create_vit_argument_parser()
    args = parser.parse_args()

    try:
        train_vit_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
