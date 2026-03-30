"""Common dataset loading utilities for training scripts."""

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple, List, Optional

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

def load_imagenet_dataset(
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        data_dir: Optional[str] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Tuple[int, int, int], int]:
    """
    Load ImageNet dataset using TensorFlow Datasets.

    Parameters
    ----------
    image_size : Tuple[int, int]
        Target image size (height, width).
    batch_size : int
        Batch size.
    shuffle_buffer_size : int
        Buffer size for shuffling.
    cache : bool
        Whether to cache the dataset in memory.
    data_dir : Optional[str]
        Directory to download/load data from.

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset, Tuple[int, int, int], int]
        Training dataset, validation dataset, input shape, number of classes.
    """
    logger.info("Loading ImageNet dataset from TensorFlow Datasets...")

    train_ds, train_info = tfds.load(
        "imagenet2012",
        split="train",
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
    )

    val_ds = tfds.load(
        "imagenet2012",
        split="validation",
        as_supervised=True,
        data_dir=data_dir,
    )

    num_classes = train_info.features['label'].num_classes
    input_shape = (image_size[0], image_size[1], 3)

    def preprocess_train(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess training image with data augmentation."""
        image = tf.image.resize(image, (int(image_size[0] * 1.15), int(image_size[1] * 1.15)))
        image = tf.image.random_crop(image, [image_size[0], image_size[1], 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def preprocess_val(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess validation image."""
        image = tf.image.resize(image, (int(image_size[0] * 1.15), int(image_size[1] * 1.15)))
        h, w = image_size
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=(tf.shape(image)[0] - h) // 2,
            offset_width=(tf.shape(image)[1] - w) // 2,
            target_height=h,
            target_width=w
        )
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    train_size = train_info.splits['train'].num_examples
    val_size = train_info.splits['validation'].num_examples

    logger.info(f"ImageNet dataset loaded: {train_size} train, {val_size} validation samples")
    logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

    return train_ds, val_ds, input_shape, num_classes


# ---------------------------------------------------------------------

def load_dataset(
        dataset_name: str,
        batch_size: int = 32,
        image_size: Optional[Tuple[int, int]] = None,
) -> Tuple:
    """
    Load and preprocess dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('mnist', 'cifar10', 'cifar100', 'imagenet').
    batch_size : int
        Batch size (used for ImageNet).
    image_size : Optional[Tuple[int, int]]
        Target image size for ImageNet.

    Returns
    -------
    Tuple
        Training data, test/validation data, input shape, number of classes.
    """
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name.lower() == 'imagenet':
        if image_size is None:
            image_size = (224, 224)
        return load_imagenet_dataset(
            image_size=image_size,
            batch_size=batch_size,
        )

    if dataset_name.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
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

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(f"Dataset loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
    logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


# ---------------------------------------------------------------------

def get_class_names(dataset: str, num_classes: int) -> List[str]:
    """Get class names for the dataset."""
    if dataset.lower() == 'mnist':
        return [str(i) for i in range(10)]
    elif dataset.lower() == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset.lower() == 'cifar100':
        return [f'class_{i}' for i in range(num_classes)]
    elif dataset.lower() == 'imagenet':
        try:
            info = tfds.builder('imagenet2012').info
            return info.features['label'].names
        except Exception:
            return [f'class_{i}' for i in range(num_classes)]
    else:
        return [f'class_{i}' for i in range(num_classes)]
