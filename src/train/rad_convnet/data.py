"""DTD (Describable Textures Dataset) ``tf.data`` pipeline for RADConvNet training.

DTD's on-disk layout (``/media/arxwn/data0_4tb/datasets/dtd/``, extracted from the
official ``dtd-r1.0.1.tar.gz``) is ``images/<class>/<file>.jpg`` plus ten
predefined splits under ``labels/{train,val,test}{1..10}.txt``, each line a
class-relative image path (e.g. ``banded/banded_0005.jpg``). This module uses
split ``1`` by default (``train1.txt`` / ``val1.txt`` / ``test1.txt``) --
DTD's own deterministic split, not an ad-hoc one.
"""

import keras
import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Tuple

from dl_techniques.utils.logger import logger

# Reuses the repo's own ImageNet normalization constants (dl_techniques.CLAUDE.md
# has no DTD-specific constant, and ImageNet statistics are the standard default
# for natural-image classification when a dataset-specific one is not computed).
from train.common.datasets import IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------

def _read_split_file(labels_dir: Path, split_name: str) -> List[str]:
    """Read one DTD split file into a list of class-relative image paths.

    :param labels_dir: Path to the DTD ``labels/`` directory.
    :type labels_dir: Path
    :param split_name: File name, e.g. ``"train1.txt"``.
    :type split_name: str
    :return: List of ``"<class>/<file>.jpg"`` relative paths.
    :rtype: list[str]
    :raises FileNotFoundError: If the split file does not exist.
    """
    split_path = labels_dir / split_name
    if not split_path.exists():
        raise FileNotFoundError(
            f"DTD split file not found: {split_path}. Expected the extracted "
            f"dtd-r1.0.1 layout (images/<class>/*.jpg + labels/*.txt)."
        )
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# ---------------------------------------------------------------------

def discover_dtd_classes(dtd_root: str) -> List[str]:
    """List DTD's 47 class names, sorted, from its ``images/`` subdirectory.

    :param dtd_root: Path to the extracted DTD root (contains ``images/``,
        ``labels/``).
    :type dtd_root: str
    :return: Sorted class names.
    :rtype: list[str]
    """
    images_dir = Path(dtd_root) / "images"
    return sorted(d.name for d in images_dir.iterdir() if d.is_dir())

# ---------------------------------------------------------------------

def make_dtd_dataset(
    dtd_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    split_index: int = 1,
    augment: bool = True,
    shuffle_buffer: int = 2000,
    max_samples: Optional[int] = None,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch_buffer=tf.data.AUTOTUNE,
) -> Tuple[tf.data.Dataset, List[str], int]:
    """Build a ``tf.data`` pipeline over one DTD split.

    :param dtd_root: Path to the extracted DTD root.
    :type dtd_root: str
    :param split: One of ``"train"``, ``"val"``, ``"test"``.
    :type split: str
    :param image_size: Target square resize size in pixels.
    :type image_size: int
    :param batch_size: Batch size.
    :type batch_size: int
    :param split_index: Which of DTD's 10 predefined splits to use. Defaults
        to ``1``.
    :type split_index: int
    :param augment: Whether to apply light training-time augmentation
        (random crop + horizontal flip). Only active for ``split == "train"``.
    :type augment: bool
    :param shuffle_buffer: Shuffle buffer size, used only for ``split ==
        "train"``.
    :type shuffle_buffer: int
    :param max_samples: If set, truncates the file list to this many samples
        -- for fast debug/smoke runs, not full training.
    :type max_samples: Optional[int]
    :param num_parallel_calls: ``num_parallel_calls`` for the per-element map.
    :param prefetch_buffer: ``buffer_size`` for the trailing ``prefetch``.
    :return: ``(dataset, class_names, num_samples)`` -- the batched ``(image,
        label)`` dataset, the sorted list of the 47 class names (label ``i``
        corresponds to ``class_names[i]``), and the number of (unbatched)
        samples in this split (after ``max_samples`` truncation).
    :rtype: tuple[tf.data.Dataset, list[str], int]
    :raises ValueError: If ``split`` is not one of the three valid values.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

    dtd_root_path = Path(dtd_root)
    images_dir = dtd_root_path / "images"
    labels_dir = dtd_root_path / "labels"

    class_names = discover_dtd_classes(dtd_root)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    rel_paths = _read_split_file(labels_dir, f"{split}{split_index}.txt")
    if max_samples is not None:
        rel_paths = rel_paths[:max_samples]

    image_paths = [str(images_dir / p) for p in rel_paths]
    label_list = [class_to_idx[p.split("/")[0]] for p in rel_paths]
    logger.info(f"DTD {split}{split_index}: {len(image_paths)} images, {len(class_names)} classes")

    is_training = split == "train"
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_list))
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGENET_STD, dtype=tf.float32)

    def _load(path: tf.Tensor, label: tf.Tensor):
        raw = tf.io.read_file(path)
        image = tf.io.decode_jpeg(raw, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0, 1]

        if is_training and augment:
            resize_to = int(image_size * 1.15)
            image = tf.image.resize(image, (resize_to, resize_to))
            image = tf.image.random_crop(image, (image_size, image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (image_size, image_size))

        image = (image - mean) / std
        return image, label

    dataset = dataset.map(_load, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset, class_names, len(image_paths)

# ---------------------------------------------------------------------
