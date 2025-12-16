"""ImageNet dataset loader using TensorFlow Datasets."""

import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple, Optional

# ---------------------------------------------------------------------

def load_imagenet(
        split: str = "train",
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 10000,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        cache: bool = False,
        data_dir: Optional[str] = None,
) -> tf.data.Dataset:
    """
    Load ImageNet dataset using TensorFlow Datasets.

    Parameters
    ----------
    split : str, optional
        Dataset split to load ('train', 'validation', 'test'), by default "train"
    batch_size : int, optional
        Batch size, by default 32
    image_size : Tuple[int, int], optional
        Target image size (height, width), by default (224, 224)
    shuffle_buffer_size : int, optional
        Buffer size for shuffling, by default 10000
    prefetch_buffer_size : int, optional
        Buffer size for prefetching, by default tf.data.AUTOTUNE
    cache : bool, optional
        Whether to cache the dataset in memory, by default False
    data_dir : Optional[str], optional
        Directory to download/load data from, by default None

    Returns
    -------
    tf.data.Dataset
        Batched and preprocessed ImageNet dataset
    """
    # Load dataset
    dataset, info = tfds.load(
        "imagenet2012",
        split=split,
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
    )

    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess image and label.

        Parameters
        ----------
        image : tf.Tensor
            Input image
        label : tf.Tensor
            Image label

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Preprocessed image and label
        """
        # Resize image
        image = tf.image.resize(image, image_size)
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Apply preprocessing
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache if requested
    if cache:
        dataset = dataset.cache()

    # Shuffle for training
    if split == "train":
        dataset = dataset.shuffle(shuffle_buffer_size)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset

# ---------------------------------------------------------------------
