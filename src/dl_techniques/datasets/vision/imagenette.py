"""Imagenette (a 10-class ImageNet subset) read from prepared TFDS records.

``imagenette/320px-v2`` is on disk under ``$TFDS_DATA_DIR`` (9,469 train and 3,925 labelled
validation images, variable size, about 320 px on the short side). Nothing is downloaded: an
absent record set is a ``FileNotFoundError`` naming the directory.

:func:`load_imagenette` serves three splits with the protocol of the other labelled-image
datasets: ``"train"`` (the records minus the selection rows), ``"select"`` (a seeded
``validation_split`` slice of train, never augmented) and ``"test"`` (the labelled
validation split, which never selects). The encoded jpeg bytes of a split are held in
memory once (about 250 MB for train), so every pipeline has a known length and no decode is
repeated off the disk. :func:`imagenette_examples` is the unbatched stage underneath it
(rows, shuffle, decode, crop policy, normalize), which the self-supervised pipeline
``train.common.image_streams`` also builds on, so there is one decode path.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD
from dl_techniques.datasets.vision.splits import (
    check_validation_split,
    seeded_split_indices,
    seeded_test_rows,
)
from dl_techniques.utils.logger import logger

#: Verified present offline at ``$TFDS_DATA_DIR``: 9469 train / 3925 validation, 10 classes.
IMAGENETTE_TFDS_NAME = "imagenette/320px-v2"
IMAGENETTE_NUM_CLASSES = 10

#: The ten wnids in TFDS label order (``label.labels.txt``) and what they are.
IMAGENETTE_WNIDS: Tuple[str, ...] = (
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
)
IMAGENETTE_CLASS_NAMES: Tuple[str, ...] = (
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
)

_SPLITS = ("train", "select", "test")
_TFDS_SPLIT = {"train": "train", "select": "train", "test": "validation"}
_NORMALIZE_MODES = ("imagenet", "unit")

#: ``crop_policy`` values. ``"resize"`` resizes the whole image to ``image_size`` square and
#: augments with a flip plus a reflect-pad-and-crop. ``"random_resized_crop"`` is the
#: ImageNet-classifier recipe (Inception-style random-resized crop plus flip for training,
#: resize-then-centre-crop for evaluation).
CROP_POLICIES: Tuple[str, ...] = ("resize", "random_resized_crop")

#: Evaluation centre-crop fraction of ``"random_resized_crop"``: the shorter side
#: is resized to ``image_size / EVAL_CROP_FRACTION`` and the centre
#: ``image_size`` square is kept (the torchvision 256 -> 224 ratio).
EVAL_CROP_FRACTION = 0.875

#: Area and aspect-ratio ranges of the training random-resized crop (the
#: Inception / torchvision ``RandomResizedCrop`` defaults).
RANDOM_RESIZED_CROP_AREA = (0.08, 1.0)
RANDOM_RESIZED_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def random_resized_crop(
        image: tf.Tensor, image_size: int, seed: Optional[int] = None,
) -> tf.Tensor:
    """Random-resized crop plus horizontal flip of one ``[0, 1]`` image.

    Samples a box covering :data:`RANDOM_RESIZED_CROP_AREA` of the image at an
    aspect ratio in :data:`RANDOM_RESIZED_CROP_RATIO` (falling back to the whole
    image when no box fits), resizes it to ``image_size`` square, then flips.
    """
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.zeros([0, 0, 4], tf.float32),
        min_object_covered=0.0,
        aspect_ratio_range=RANDOM_RESIZED_CROP_RATIO,
        area_range=RANDOM_RESIZED_CROP_AREA,
        max_attempts=10,
        use_image_if_no_bounding_boxes=True,
        seed=seed,
    )
    image = tf.slice(image, begin, size)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    return tf.image.random_flip_left_right(image, seed=seed)


def resize_and_centre_crop(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Resize the shorter side to ``image_size / EVAL_CROP_FRACTION``, keep the centre square."""
    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = (image_size / EVAL_CROP_FRACTION) / tf.reduce_min(shape)
    resized = tf.cast(tf.round(shape * scale), tf.int32)
    image = tf.image.resize(image, resized, method="bilinear")
    return tf.image.resize_with_crop_or_pad(image, image_size, image_size)


def flip_and_reflect_crop(image: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    """Random flip + reflect-pad-and-crop, on the [0,1] image, BEFORE normalization.

    Augmentation runs pre-normalization on purpose (the ``train_vit`` D-006/D-007 lesson:
    augmenting normalized data and then clipping to [0,1] saturates most pixels and silently
    creates a train/val distribution mismatch). No ``clip_by_value`` here.
    """
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_crop(
        tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"),
        size=tf.shape(image),
        seed=seed,
    )
    return image


# ---------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------

def read_imagenette_records(
        split: str, data_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """The encoded jpeg bytes and labels of one TFDS split, in file order.

    This is the only place the TFDS records are touched by :func:`load_imagenette`, and the
    seam a test replaces with a few in-memory jpegs.

    :param split: ``"train"`` or ``"validation"`` (TFDS names).
    :param data_dir: The TFDS data directory; ``None`` inherits ``$TFDS_DATA_DIR``.
    :return: ``(images, labels)``: an object array of ``bytes`` and an ``int32`` array.
    :raises FileNotFoundError: The records are not prepared under the data directory.
    """
    # Imported lazily: tfds pulls in a heavy import chain.
    import tensorflow_datasets as tfds

    builder = tfds.builder(IMAGENETTE_TFDS_NAME, data_dir=data_dir)
    if not Path(builder.data_dir).is_dir():
        raise FileNotFoundError(
            f"{IMAGENETTE_TFDS_NAME} records are not prepared at {builder.data_dir}; "
            "nothing is downloaded. Set $TFDS_DATA_DIR to the directory that holds them."
        )
    ds = builder.as_dataset(
        split=split, shuffle_files=False,
        decoders={"image": tfds.decode.SkipDecoding()},
    )
    images: List[bytes] = []
    labels: List[int] = []
    for element in tfds.as_numpy(ds):
        images.append(element["image"])
        labels.append(int(element["label"]))
    packed = np.empty(len(images), dtype=object)
    packed[:] = images
    return packed, np.asarray(labels, dtype=np.int32)


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------

@dataclass
class ImagenetteConfig:
    """Configuration for :func:`load_imagenette`.

    ``augment`` applies to the ``train`` split only; ``select`` and ``test`` are never
    augmented and keep the TFDS file order (a seeded order for ``select``).

    :param image_size: Square output side.
    :param batch_size: Batch size of the pipeline.
    :param crop_policy: One of :data:`CROP_POLICIES`.
    :param augment: Train-time augmentation of the chosen crop policy.
    :param normalize: ``"imagenet"`` ((x/255 - mean) / std) or ``"unit"`` (x/255).
    :param validation_split: Fraction of the train records held out as the selection split.
    :param seed: Seed of the selection permutation, the shuffle and the augmentation.
    :param max_images: Smoke-run cap on the train pool and on the test split.
    :param tfds_data_dir: The TFDS data directory; ``None`` inherits ``$TFDS_DATA_DIR``.
    :param num_parallel_calls: ``tf.data`` parallelism of the decode.
    """

    image_size: int = 224
    batch_size: int = 128
    crop_policy: str = "random_resized_crop"
    augment: bool = True
    normalize: str = "imagenet"
    validation_split: float = 0.1
    seed: int = 0
    max_images: Optional[int] = None
    tfds_data_dir: Optional[str] = None
    num_parallel_calls: int = tf.data.AUTOTUNE


class ImagenetteData(NamedTuple):
    """One split: the batched dataset plus what a trainer needs about it."""

    dataset: "tf.data.Dataset"
    num_samples: int
    steps_per_epoch: int
    num_classes: int
    class_names: List[str]


def _check_config(config: ImagenetteConfig) -> None:
    if config.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {config.image_size}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.crop_policy not in CROP_POLICIES:
        raise ValueError(
            f"Unknown crop_policy {config.crop_policy!r}; known: {list(CROP_POLICIES)}")
    if config.normalize not in _NORMALIZE_MODES:
        raise ValueError(
            f"normalize must be one of {_NORMALIZE_MODES}, got {config.normalize!r}")
    check_validation_split(config.validation_split)


def imagenette_examples(
        split: str, config: ImagenetteConfig,
) -> Tuple["tf.data.Dataset", int]:
    """The unbatched, normalized ``(image, label)`` examples of one Imagenette split.

    The rows come from the seeded permutation of the TRAIN records (selection first) and,
    for ``"test"``, an independent seeded order of the validation records. ``"train"`` is
    shuffled (the encoded rows, so the buffer holds bytes) and reshuffles on every pass;
    nothing is batched or repeated, so a caller can map per example before it batches.

    :param split: ``"train"``, ``"select"`` or ``"test"``.
    :param config: The pipeline settings (``batch_size`` is not used here).
    :return: ``(dataset, n)``: ``float32`` images normalized per ``config.normalize`` and
        ``int32`` labels, and the number of examples.
    :raises ValueError: A bad ``split`` or setting, or a selection slice holding out zero rows.
    :raises FileNotFoundError: The TFDS records are not prepared.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    _check_config(config)

    images, labels = read_imagenette_records(_TFDS_SPLIT[split], config.tfds_data_dir)
    # The split rows come from the seeded permutation of the TRAIN records (selection first)
    # and an independent seeded order of the validation records (the test split).
    if split == "test":
        rows = seeded_test_rows(len(images), config.seed, config.max_images)
    else:
        train_idx, select_idx, _ = seeded_split_indices(
            len(images), 1, config.seed, config.validation_split, config.max_images)
        rows = train_idx if split == "train" else select_idx

    training = split == "train"
    augment = training and config.augment
    size = config.image_size
    mean = tf.constant(IMAGENET_MEAN if config.normalize == "imagenet" else (0.0,) * 3,
                       dtype=tf.float32, shape=(1, 1, 3))
    std = tf.constant(IMAGENET_STD if config.normalize == "imagenet" else (1.0,) * 3,
                      dtype=tf.float32, shape=(1, 1, 3))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(images[rows]), tf.constant(labels[rows])))
    if training:
        # Shuffle the encoded rows: the buffer holds bytes, not decoded floats.
        dataset = dataset.shuffle(len(rows), seed=config.seed, reshuffle_each_iteration=True)

    def decode(encoded: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Records are VARIABLE-SIZE (e.g. 320x396x3): the geometry step must make them square.
        image = tf.cast(tf.io.decode_jpeg(encoded, channels=3), tf.float32) / 255.0
        if config.crop_policy == "resize":
            image = tf.image.resize(image, (size, size), method="bilinear")
            if augment:
                image = flip_and_reflect_crop(image, seed=config.seed)
        elif augment:
            image = random_resized_crop(image, size, seed=config.seed)
        else:
            image = resize_and_centre_crop(image, size)
        image = tf.ensure_shape(image, (size, size, 3))
        return (image - mean) / std, label

    dataset = dataset.map(decode, num_parallel_calls=config.num_parallel_calls)
    logger.info(
        f"imagenette [{split}]: {len(rows)} images, {size}x{size}, "
        f"crop_policy={config.crop_policy}, augment={augment}"
    )
    return dataset, int(len(rows))


def load_imagenette(split: str, config: ImagenetteConfig) -> ImagenetteData:
    """Build the pipeline of one Imagenette split.

    :param split: ``"train"``, ``"select"`` or ``"test"``.
    :param config: The pipeline settings.
    :return: The batched ``(image, label)`` dataset (``float32`` images, ``int32`` labels)
        and its counts. The train pipeline reshuffles every epoch and drops the ragged
        last batch; the others keep every row. All three are finite and do not repeat.
    :raises ValueError: A bad ``split`` or setting, or a selection slice holding out zero rows.
    :raises FileNotFoundError: The TFDS records are not prepared.
    """
    training = split == "train"
    dataset, n = imagenette_examples(split, config)
    dataset = dataset.batch(config.batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)

    steps = n // config.batch_size if training else -(-n // config.batch_size)
    return ImagenetteData(
        dataset=dataset, num_samples=n, steps_per_epoch=steps,
        num_classes=IMAGENETTE_NUM_CLASSES, class_names=list(IMAGENETTE_CLASS_NAMES),
    )
