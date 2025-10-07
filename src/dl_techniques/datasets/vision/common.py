"""
Reusable dataset builders for common vision datasets.

This module provides pre-configured dataset builders for popular datasets
that integrate seamlessly with the training framework.

MNIST Color Mode:
    MNIST can be loaded in either RGB (3 channels) or grayscale (1 channel):
    - RGB mode (default): Repeats grayscale channel to create 3-channel images.
      Use this for models expecting RGB input (e.g., pretrained models, ViT).
    - Grayscale mode: Keeps original single channel format.
      Use this for pure CNNs or when training from scratch on MNIST alone.
      More memory efficient and computationally faster.

Example:
    >>> # RGB MNIST (3 channels)
    >>> builder_rgb = create_dataset_builder('mnist', config, use_rgb=True)
    >>>
    >>> # Grayscale MNIST (1 channel)
    >>> builder_gray = create_dataset_builder('mnist', config, use_rgb=False)
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.optimization.train_vision import DatasetBuilder, DataInput


# ---------------------------------------------------------------------
# MNIST DATASET BUILDER
# ---------------------------------------------------------------------

class MNISTDatasetBuilder(DatasetBuilder):
    """
    Dataset builder for MNIST handwritten digits.

    Supports both grayscale (1 channel) and RGB (3 channels) output.
    For RGB mode, grayscale images are converted by repeating channels.

    Features:
        - Configurable RGB or grayscale output
        - Normalization to [0, 1] range
        - Light augmentation (random shifts)
        - Test data for analysis
        - Class names ('0'-'9')

    Args:
        config: Training configuration.
        use_rgb: If True, convert grayscale to RGB by repeating channels.
                 If False, keep original grayscale format. Default is True
                 for compatibility with models expecting RGB input.
    """

    def __init__(self, config, use_rgb: bool = True):
        """
        Initialize MNIST dataset builder.

        Args:
            config: Training configuration.
            use_rgb: If True, convert to RGB (3 channels). If False, keep grayscale (1 channel).
        """
        super().__init__(config)
        self.use_rgb = use_rgb
        self.x_test = None
        self.y_test = None
        self.class_names = [str(i) for i in range(10)]  # '0' through '9'

        # Validate input shape matches the RGB/grayscale setting
        expected_channels = 3 if use_rgb else 1
        if config.input_shape[2] != expected_channels:
            logger.warning(
                f"Config input_shape has {config.input_shape[2]} channels, "
                f"but MNIST builder is configured for {expected_channels} channels (use_rgb={use_rgb}). "
                f"The builder will produce {expected_channels}-channel images."
            )

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
        logger.info(f"Loading MNIST dataset (use_rgb={self.use_rgb})...")

        # Load data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Add channel dimension
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

        # Convert to RGB if requested
        if self.use_rgb:
            logger.info("Converting MNIST from grayscale to RGB")
            x_train = np.repeat(x_train, 3, axis=-1)
            x_test = np.repeat(x_test, 3, axis=-1)
        else:
            logger.info("Using MNIST in grayscale format")

        # Normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        channels = x_train.shape[-1]
        logger.info(
            f"MNIST loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples "
            f"with shape {x_train.shape[1:]} ({channels} channel{'s' if channels > 1 else ''})"
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
        """
        Apply light augmentation for MNIST.

        Args:
            image: Input image tensor.
            label: Corresponding label.

        Returns:
            Augmented image and label.
        """
        # Random shift
        image = tf.image.random_crop(
            tf.pad(image, [[2, 2], [2, 2], [0, 0]], mode='CONSTANT'),
            size=tf.shape(image)
        )
        return image, label

    def get_test_data(self) -> Optional[DataInput]:
        """
        Get test data for analysis and visualization.

        Returns:
            DataInput object containing test data, or None if not available.
        """
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """
        Get class names for visualization.

        Returns:
            List of class names for digit labels.
        """
        return self.class_names


# ---------------------------------------------------------------------
# CIFAR-10 DATASET BUILDER
# ---------------------------------------------------------------------

class CIFAR10DatasetBuilder(DatasetBuilder):
    """
    Dataset builder for CIFAR-10 image classification.

    Applies normalization and standard augmentation techniques suitable
    for natural image classification.

    Features:
        - Normalization to [0, 1] range
        - Standard augmentation (flip, crop, brightness, contrast)
        - Test data for analysis
        - Descriptive class names
    """

    def __init__(self, config):
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
        """
        Apply standard augmentation for CIFAR-10.

        Args:
            image: Input image tensor.
            label: Corresponding label.

        Returns:
            Augmented image and label.
        """
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
        """
        Get test data for analysis and visualization.

        Returns:
            DataInput object containing test data, or None if not available.
        """
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """
        Get class names for visualization.

        Returns:
            List of class names for CIFAR-10 categories.
        """
        return self.class_names


# ---------------------------------------------------------------------
# CIFAR-100 DATASET BUILDER
# ---------------------------------------------------------------------

class CIFAR100DatasetBuilder(DatasetBuilder):
    """
    Dataset builder for CIFAR-100 fine-grained image classification.

    Applies normalization and stronger augmentation suitable for the more
    complex 100-class dataset.

    Features:
        - Normalization to [0, 1] range
        - Strong augmentation (flip, crop, brightness, contrast, saturation)
        - Test data for analysis
        - All 100 fine-grained class names
    """

    def __init__(self, config):
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
        """
        Apply stronger augmentation for CIFAR-100.

        Args:
            image: Input image tensor.
            label: Corresponding label.

        Returns:
            Augmented image and label.
        """
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
        """
        Get test data for analysis and visualization.

        Returns:
            DataInput object containing test data, or None if not available.
        """
        if self.x_test is not None and self.y_test is not None:
            return DataInput(x_data=self.x_test, y_data=self.y_test)
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """
        Get class names for visualization.

        Returns:
            List of all 100 fine-grained class names.
        """
        return self.class_names


# ---------------------------------------------------------------------
# DATASET BUILDER FACTORY
# ---------------------------------------------------------------------

def create_dataset_builder(
        dataset_name: str,
        config,
        use_rgb: bool = True
) -> DatasetBuilder:
    """
    Factory function to create the appropriate dataset builder.

    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar100').
        config: Training configuration.
        use_rgb: For MNIST only - if True, convert to RGB; if False, keep grayscale.
                 Ignored for CIFAR datasets (always RGB).

    Returns:
        Instantiated DatasetBuilder.

    Raises:
        ValueError: If dataset name is not supported.

    Example:
        >>> from dl_techniques.optimization.train_vision import TrainingConfig
        >>>
        >>> # RGB MNIST (3 channels) for models expecting RGB
        >>> config_rgb = TrainingConfig(input_shape=(28, 28, 3), num_classes=10)
        >>> builder_rgb = create_dataset_builder('mnist', config_rgb, use_rgb=True)
        >>>
        >>> # Grayscale MNIST (1 channel) for pure CNNs
        >>> config_gray = TrainingConfig(input_shape=(28, 28, 1), num_classes=10)
        >>> builder_gray = create_dataset_builder('mnist', config_gray, use_rgb=False)
        >>>
        >>> # CIFAR-10 (always 3 channels, use_rgb ignored)
        >>> config_cifar = TrainingConfig(input_shape=(32, 32, 3), num_classes=10)
        >>> builder_cifar = create_dataset_builder('cifar10', config_cifar)
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'mnist':
        return MNISTDatasetBuilder(config, use_rgb=use_rgb)
    elif dataset_name == 'cifar10':
        return CIFAR10DatasetBuilder(config)
    elif dataset_name == 'cifar100':
        return CIFAR100DatasetBuilder(config)
    else:
        available = 'mnist, cifar10, cifar100'
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )


# ---------------------------------------------------------------------
# DATASET INFO HELPER
# ---------------------------------------------------------------------

def get_dataset_info(dataset_name: str, use_rgb: bool = True) -> dict:
    """
    Get metadata information for a dataset.

    Args:
        dataset_name: Name of the dataset.
        use_rgb: For MNIST only - if True, return RGB info; if False, return grayscale info.
                 Ignored for CIFAR datasets (always RGB).

    Returns:
        Dictionary containing input_shape, num_classes, and other metadata.

    Example:
        >>> # RGB MNIST info
        >>> info = get_dataset_info('mnist', use_rgb=True)
        >>> print(info['input_shape'])  # (28, 28, 3)
        >>>
        >>> # Grayscale MNIST info
        >>> info = get_dataset_info('mnist', use_rgb=False)
        >>> print(info['input_shape'])  # (28, 28, 1)
        >>>
        >>> # CIFAR-10 info (use_rgb ignored, always RGB)
        >>> info = get_dataset_info('cifar10')
        >>> print(info['num_classes'])  # 10
    """
    dataset_name = dataset_name.lower()

    # Determine input shape for MNIST based on use_rgb
    mnist_input_shape = (28, 28, 3) if use_rgb else (28, 28, 1)

    info = {
        'mnist': {
            'input_shape': mnist_input_shape,
            'num_classes': 10,
            'train_samples': 60000,
            'test_samples': 10000,
            'recommended_batch_size': 128,
            'recommended_epochs': 50,
            'color_mode': 'rgb' if use_rgb else 'grayscale',
        },
        'cifar10': {
            'input_shape': (32, 32, 3),
            'num_classes': 10,
            'train_samples': 50000,
            'test_samples': 10000,
            'recommended_batch_size': 128,
            'recommended_epochs': 200,
            'color_mode': 'rgb',
        },
        'cifar100': {
            'input_shape': (32, 32, 3),
            'num_classes': 100,
            'train_samples': 50000,
            'test_samples': 10000,
            'recommended_batch_size': 128,
            'recommended_epochs': 300,
            'color_mode': 'rgb',
        },
    }

    if dataset_name not in info:
        available = ', '.join(info.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )

    return info[dataset_name]

# ---------------------------------------------------------------------
