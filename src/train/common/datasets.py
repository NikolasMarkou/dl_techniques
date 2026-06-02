"""Common dataset loading utilities for training scripts."""

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from typing import Tuple, List, Optional, Set

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# CIFAR-10 per-channel normalisation constants (computed from the CIFAR-10
# training set). These are CIFAR-10 channel mean/std and are DISTINCT from
# the OpenAI CLIP IMAGE_MEAN/IMAGE_STD in common/image_text.py (those are
# [0.48145466, ...]). Do not conflate the two. Kept as plain lists so each
# call site can wrap with np.array(...) when it needs array broadcasting.
# ---------------------------------------------------------------------

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


# ---------------------------------------------------------------------
# ImageNet per-channel normalisation constants (the canonical ILSVRC /
# torchvision RGB mean/std). These are DISTINCT from both the CIFAR-10
# constants above ([0.4914, ...]) and the OpenAI CLIP IMAGE_MEAN/IMAGE_STD
# in common/image_text.py ([0.48145466, ...]). Do not conflate the three.
# Kept as plain lists so each call site can broadcast/cast as needed (the
# tf.data pipeline below subtracts/divides directly).
# ---------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------

def make_imagenet_filesystem_dataset(
        data_dir: str,
        image_size: int,
        batch_size: int,
        is_training: bool = True,
        augment: bool = True,
        augment_color: bool = False,
        shuffle_buffer: int = 10000,
        num_parallel_calls=tf.data.AUTOTUNE,
        cache_val: bool = False,
        drop_remainder: Optional[bool] = None,
        prefetch_buffer=tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Build an ImageNet-style ``tf.data`` pipeline from a class-subdir layout.

    Walks ``data_dir`` for class subdirectories (sorted, one integer label per
    subdir), collects every ``*.JPEG`` (uppercase glob — lowercase would
    silently skip files) into ``(path, label)`` lists, then builds a
    ``tf.data.Dataset`` that reads, decodes, resizes, optionally augments, scales
    to ``[0, 1]`` and normalizes with :data:`IMAGENET_MEAN` / :data:`IMAGENET_STD`.

    This is the shared extraction of the byte-near-identical ImageNet pipelines
    previously duplicated in ``train_resnet.py`` and ``train_vit.py``. The two
    differed ONLY in four colour augmentations (random brightness / contrast /
    saturation / hue), which are gated here behind ``augment_color``. The
    ``clip_by_value(0, 255)`` clamp is applied unconditionally in BOTH original
    callers and is therefore kept unconditional here (it is not a per-caller
    divergence).

    Args:
        data_dir: Root directory containing one subdirectory per class.
        image_size: Target square crop size (height == width) in pixels.
        batch_size: Number of examples per batch.
        is_training: When ``True`` the dataset is shuffled and repeated, the
            training augmentation branch (random crop + horizontal flip) runs,
            and ``drop_remainder`` defaults to ``True``.
        augment: Master switch for the training augmentation branch. Augmentation
            only runs when ``is_training and augment`` (matches the original
            ``is_training and config.augment_data`` guard). When ``False`` (or in
            validation) the deterministic resize + centre crop-or-pad branch runs.
        augment_color: When ``True`` (and augmentation is active) additionally
            applies the four colour augmentations (brightness / contrast /
            saturation / hue). ResNet opts in (``True``); ViT uses the default
            (``False``).
        shuffle_buffer: Shuffle buffer size used when ``is_training``.
        num_parallel_calls: ``num_parallel_calls`` for the per-element map.
        cache_val: When ``True`` and ``not is_training``, caches the mapped
            dataset in memory (mirrors the original ``cache_dataset and not
            is_training`` guard).
        drop_remainder: Passed to ``batch``. When ``None`` (the default) it
            resolves to ``is_training`` (drop the ragged tail during training,
            keep it for validation).
        prefetch_buffer: ``buffer_size`` for the trailing ``prefetch``.

    Returns:
        A single ``tf.data.Dataset`` yielding ``(image, label)`` batches, where
        ``image`` has shape ``(batch, image_size, image_size, 3)`` normalized with
        the ImageNet mean/std. (Both original callers returned just the dataset;
        this preserves that contract — no tuple, no ``num_classes`` / steps.)
    """
    if drop_remainder is None:
        drop_remainder = is_training

    data_dir = Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    logger.info(f"Found {len(class_names)} classes in {data_dir}")

    image_paths: List[str] = []
    labels: List[int] = []
    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        for img_file in (data_dir / class_name).glob("*.JPEG"):
            image_paths.append(str(img_file))
            labels.append(class_idx)
    logger.info(f"Found {len(image_paths)} images")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer, reshuffle_each_iteration=True
        ).repeat()

    def _preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        if is_training and augment:
            image = tf.image.resize(image, [image_size + 32, image_size + 32])
            image = tf.image.random_crop(image, [image_size, image_size, 3])
            image = tf.image.random_flip_left_right(image)
            if augment_color:
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
                image = tf.image.random_hue(image, max_delta=0.1)
        else:
            image = tf.image.resize(image, [int(image_size * 1.15), int(image_size * 1.15)])
            image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        image = tf.clip_by_value(image, 0.0, 255.0) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image, label

    def _load(path, label):
        image = tf.io.read_file(path)
        return _preprocess(image, label)

    dataset = dataset.map(_load, num_parallel_calls=num_parallel_calls)
    if cache_val and not is_training:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset


# ---------------------------------------------------------------------
# Default image file extensions for the denoiser path-collection helper
# below. This is the shared default for the byte-identical rglob +
# extension-filter preambles previously duplicated across the denoiser
# trainers (bfcnn / bfunet / cliffordnet) and their framework
# ``_create_file_list`` methods. Callers may pass their own ``extensions``
# set (e.g. ``config.image_extensions``); when they don't, this set is used.
# Matching is case-insensitive (both lower- and upper-case suffixes match).
# ---------------------------------------------------------------------

DEFAULT_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------

def collect_image_paths(
        directories: List[str],
        extensions: Optional[Set[str]] = None,
        max_files: Optional[int] = None,
        shuffle_seed: Optional[int] = None,
        sort: bool = True,
) -> List[str]:
    """Recursively collect image file paths from a list of directories.

    Shared extraction of the byte-identical ``rglob`` + extension-filter +
    cap-then-shuffle path-collection preamble previously duplicated across the
    denoiser trainers (``train_bfcnn.py`` / ``train_bfunet.py`` /
    ``train_denoiser.py``) and the ``DatasetBuilder._create_file_list`` methods.

    For each directory it recursively scans (``rglob("*")``) for files whose
    suffix matches ``extensions`` (case-insensitive). A directory that does not
    exist is skipped with a warning, and any scan error is logged as a warning
    rather than raised (one bad directory does not abort the whole scan).

    This helper does NOT raise when no files are found — it returns the
    (possibly empty) list. Callers keep their own ``if not paths: raise
    ValueError(...)`` guard so the error message stays caller-specific.

    Args:
        directories: Directories to scan recursively for image files.
        extensions: Set of file suffixes to accept (e.g. ``{".jpg", ".png"}``).
            Matching is case-insensitive — both the lower- and upper-case form
            of each suffix is accepted. When ``None`` (the default),
            :data:`DEFAULT_IMAGE_EXTENSIONS` is used.
        max_files: Optional cap on the number of returned paths. When set and
            smaller than the number of collected paths, the list is shuffled
            (see ``shuffle_seed``) and then truncated to ``max_files``. The
            shuffle happens ONLY when a cap is applied — this matches the
            original preamble semantics exactly.
        shuffle_seed: When ``max_files`` triggers a shuffle, use a seeded
            ``np.random.RandomState(shuffle_seed)`` (deterministic) if this is
            not ``None``; otherwise use the global ``np.random.shuffle``
            (non-deterministic, matching the original behaviour).
        sort: When ``True`` (the default) the collected paths are sorted before
            the optional cap-then-shuffle step. This preserves deterministic
            ordering for paired-path denoisers (target/condition alignment).

    Returns:
        A list of matching file paths as strings. Possibly empty.
    """
    if extensions is None:
        extensions = DEFAULT_IMAGE_EXTENSIONS

    # Build a case-insensitive match set (both lower- and upper-case suffixes),
    # mirroring the original ``{ext.lower()} | {ext.upper()}`` construction.
    extensions_set = {ext.lower() for ext in extensions}
    extensions_set.update({ext.upper() for ext in extensions})

    all_file_paths: List[str] = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning(f"Directory not found, skipping: {directory}")
            continue
        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in extensions_set:
                    all_file_paths.append(str(file_path))
        except Exception as e:
            logger.warning(f"Error scanning directory {directory}: {e}")

    logger.info(f"Found a total of {len(all_file_paths)} files.")

    if sort:
        all_file_paths = sorted(all_file_paths)

    # Cap-then-shuffle: shuffle ONLY when a cap is actually applied, exactly as
    # the original denoiser preambles did.
    if max_files is not None and max_files < len(all_file_paths):
        logger.info(f"Limiting to {max_files} files as per configuration.")
        if shuffle_seed is not None:
            np.random.RandomState(shuffle_seed).shuffle(all_file_paths)
        else:
            np.random.shuffle(all_file_paths)
        all_file_paths = all_file_paths[:max_files]

    return all_file_paths


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
