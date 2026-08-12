"""Common dataset loading utilities for training scripts."""

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, List, Optional, Set

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
    denoiser trainers (``train/bfunet/train_bfcnn_denoiser.py`` /
    ``train_convunext_denoiser.py`` /
    ``train_unet_denoiser.py``) and the ``DatasetBuilder._create_file_list``
    methods.

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


# ---------------------------------------------------------------------
# Raw-image tf.data pipeline (imagenette via tfds, cifar10 in-memory)
# ---------------------------------------------------------------------

# DECISION plan-2026-08-12T123743-e798a9e1/D-006
# `build_raw_image_dataset`, its two private helpers and its five module
# constants live HERE, in `train/common/datasets.py`, and NOT in
# `train/energy_transformer/common.py` where they were written. Four packages
# consume them (energy_transformer, beit, dino, graph_energy_transformer), so
# an ET-package home made three of the four import across a sibling trainer
# package for generic image-dataset plumbing. `train.energy_transformer.common`
# now RE-EXPORTS these names, which preserves object identity -- do NOT
# "clean that up" into a copy: `tests/test_train/test_beit/test_common.py::
# TestBuildRawImageDataset::test_it_is_the_same_object_as_energy_transformers`
# is a bare `is` assertion that a copy fails by construction.
#
# WHAT NOT TO DO inside the function below: it carries THREE in-function
# decision anchors owned by TWO OTHER plans
# (`plan-2026-08-01T195746-12a1f2db/D-007` and `/D-008`,
# `plan-2026-08-01T105809-dc0c402e/D-040`). They pin exact branch ORDERING --
# both refusals before any tfds/keras work; `shuffle_files_seed=None` passing
# NO `ReadConfig` rather than `ReadConfig(shuffle_seed=None)`; `.enumerate()`
# AFTER `.repeat()`. They moved here character-for-character with the code and
# must NEVER be reordered, merged or "simplified"; they are not this plan's to
# renumber either. The `import tensorflow_datasets as tfds` inside the
# imagenette branch is likewise deliberate and stays LOCAL even though this
# module already imports tfds at the top: it is part of the moved body, and
# the move is behaviour-preserving by AST equality against the pre-move source
# (verification.md, step 3).
# See decisions.md D-006.

# Verified present offline at $TFDS_DATA_DIR (D-007): 9469 train / 3925 validation, 10 classes.
IMAGENETTE_TFDS_NAME = "imagenette/320px-v2"
IMAGENETTE_NUM_CLASSES = 10

DATASET_NUM_CLASSES: Dict[str, int] = {
    "imagenette": IMAGENETTE_NUM_CLASSES,
    "cifar10": 10,
}

SUPPORTED_DATASETS: Tuple[str, ...] = tuple(DATASET_NUM_CLASSES)

# Element map function: (image, label) -> whatever the trainer wants the batch element to be.
ElementMapFn = Callable[..., Any]


def _normalization_constants(dataset: str) -> Tuple[List[float], List[float]]:
    """Per-channel mean/std for the [0,1]-scaled images of ``dataset``."""
    if dataset == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    return IMAGENET_MEAN, IMAGENET_STD


def _augment(image: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
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


def build_raw_image_dataset(
        dataset: str,
        image_size: int,
        batch_size: int,
        *,
        is_training: bool,
        augment: bool = True,
        element_map_fn: Optional[ElementMapFn] = None,
        indexed_element_map_fn: Optional[ElementMapFn] = None,
        shuffle_buffer: int = 4096,
        seed: Optional[int] = None,
        shuffle_files_seed: Optional[int] = None,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_buffer: int = tf.data.AUTOTUNE,
) -> Tuple[tf.data.Dataset, int, int]:
    """Build a raw-image ``tf.data`` pipeline for imagenette or cifar10.

    The pipeline yields ``(image, label)`` — ``image`` float32, resized to
    ``(image_size, image_size, 3)``, scaled to ``[0, 1]`` and per-channel normalized; ``label``
    an ``int32`` scalar. When ``element_map_fn`` is given it is applied PER SAMPLE (before
    batching) to that pair, which is how the MIM trainer swaps in the masked-patch element
    ``((image, input_mask), target_patches, loss_weight)``.

    Args:
        dataset: ``'imagenette'`` or ``'cifar10'``.
        image_size: Side length the images are resized/cropped to.
        batch_size: Batch size. Training batches use ``drop_remainder=True``.
        is_training: Training pipeline (shuffle + repeat + augment) vs validation
            (``.cache()``, no shuffle, no repeat, no augment).
        augment: Enable train-time augmentation. Ignored when ``is_training`` is False.
        element_map_fn: Optional per-sample transform applied to ``(image, label)``.
        indexed_element_map_fn: Optional per-sample transform that ALSO receives a
            per-element counter: it is called as ``fn(index, image, label)``. When
            it is given, the training pipeline becomes
            ``.repeat() -> .enumerate() -> .map(...) -> .batch(...)``, so the SAME
            source image carries a DIFFERENT ``index`` on each epoch. That counter
            is what a stateless-RNG augmentation keys on, which is the only way to
            make the augmentation stream reproducible under
            ``num_parallel_calls=AUTOTUNE`` (see
            ``dl_techniques.datasets.vision.multi_crop
            .make_stateless_multi_crop_map_fn``). ``None`` preserves today's
            behaviour EXACTLY: no enumeration, no extra map, and the
            ``.repeat().batch()`` tail unchanged. Do NOT "simplify" the branch
            away, and do NOT fold this into ``element_map_fn`` behind a boolean:
            this function has 7 call sites across 3 trainers whose
            ``element_map_fn``s cannot accept an index (decisions.md D-003).
            Mutually exclusive with ``element_map_fn``; requires ``is_training``
            (see Raises).
        shuffle_buffer: Shuffle buffer for the training pipeline.
        seed: Seed for the element ``.shuffle()`` and for augmentation. It does NOT
            reach the TFDS FILE order -- see ``shuffle_files_seed``.
        shuffle_files_seed: Optional seed for the TFDS **file interleave** order
            (imagenette only; ``tfds.ReadConfig(shuffle_seed=...)``). ``None``
            preserves today's behaviour EXACTLY: ``builder.as_dataset(...,
            shuffle_files=is_training)`` with no read config, whose file order is
            non-deterministic across processes. Pass a seed when a consumer draws a
            SMALL fixed sample off the head of the train stream (``.take(n)``) and
            reports a number from it, because the sample -- and therefore the number
            -- otherwise changes run to run at the same ``seed``. MEASURED at
            ``image_size=96, batch_size=32``, 8 batches, ``seed=42``: four calls (two
            per process, two processes) gave four DIFFERENT label sequences, with the
            per-class count of one class ranging 16-25 out of 256 samples. See
            decisions.md D-040.
        num_parallel_calls: ``tf.data`` parallelism.
        prefetch_buffer: ``tf.data`` prefetch depth.

    Returns:
        ``(ds, num_examples, num_classes)``. ``num_examples`` is the split's cardinality, from
        which the caller derives ``steps_per_epoch``.

    Raises:
        ValueError: If ``dataset`` is not supported or ``image_size``/``batch_size`` are
            non-positive; if BOTH ``element_map_fn`` and ``indexed_element_map_fn`` are
            supplied; or if ``indexed_element_map_fn`` is supplied with
            ``is_training=False``.
    """
    # DECISION plan-2026-08-01T195746-12a1f2db/D-007
    # Both refusals are LOUD rather than a silent best-effort, and they fire
    # BEFORE any tfds/keras dataset work so the message is the first thing the
    # caller sees. `is_training=False` is refused because the eval branch has no
    # `.repeat()`: `.enumerate()` there would count a SINGLE pass, so every
    # element would key identically on every epoch -- the frozen-augmentation
    # failure that D-035 already RED-proved wrong, reintroduced silently through
    # a different door. See decisions.md D-007.
    if indexed_element_map_fn is not None:
        if element_map_fn is not None:
            raise ValueError(
                "element_map_fn and indexed_element_map_fn are mutually "
                "exclusive: they are two different calling conventions for the "
                "same map slot (`fn(image, label)` vs `fn(index, image, "
                "label)`), and applying both would feed the second one the "
                "first one's output. Pass exactly one."
            )
        if not is_training:
            raise ValueError(
                "indexed_element_map_fn requires is_training=True: the "
                "evaluation pipeline has no `.repeat()`, so the per-element "
                "counter would enumerate a SINGLE pass and hand every element "
                "the same index on every epoch. A stateless augmentation keyed "
                "on that counter would then be frozen per image, which is worse "
                "than a non-reproducible stream. Use element_map_fn for the "
                "evaluation pipeline."
            )

    dataset = dataset.lower()
    if dataset not in DATASET_NUM_CLASSES:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; supported: {sorted(DATASET_NUM_CLASSES)}"
        )
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    mean_vals, std_vals = _normalization_constants(dataset)
    mean = tf.constant(mean_vals, dtype=tf.float32, shape=(1, 1, 3))
    std = tf.constant(std_vals, dtype=tf.float32, shape=(1, 1, 3))

    if dataset == "imagenette":
        # Imported lazily: tfds pulls in a heavy import chain and cifar10 does not need it.
        import tensorflow_datasets as tfds

        split = "train" if is_training else "validation"
        # data_dir=None -> inherits $TFDS_DATA_DIR. The records are PREPARED on disk; no
        # download, no network (D-007). download=False makes an absent record set a LOUD
        # failure instead of a silent multi-GB fetch.
        builder = tfds.builder(IMAGENETTE_TFDS_NAME)
        num_examples = int(builder.info.splits[split].num_examples)

        # DECISION plan-2026-08-01T105809-dc0c402e/D-040
        # `shuffle_files_seed=None` must keep passing NO read_config, not
        # `ReadConfig(shuffle_seed=None)`. This function is shared by every
        # `src/train/` consumer, so the default has to be byte-for-byte today's
        # behaviour; an always-on ReadConfig would change the file order for every
        # existing trainer at once. Do NOT "simplify" the branch away.
        # See decisions.md D-040.
        read_config = None
        if shuffle_files_seed is not None:
            read_config = tfds.ReadConfig(shuffle_seed=int(shuffle_files_seed))
        if read_config is None:
            ds = builder.as_dataset(split=split, shuffle_files=is_training)
        else:
            ds = builder.as_dataset(
                split=split, shuffle_files=is_training, read_config=read_config)

        def _decode(element: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
            # Imagenette records are VARIABLE-SIZE (e.g. 320x396x3). The resize is MANDATORY:
            # without it the batch is ragged and tf.data raises at the first batch.
            image = tf.cast(element["image"], tf.float32) / 255.0
            image = tf.image.resize(image, (image_size, image_size), method="bilinear")
            image = tf.ensure_shape(image, (image_size, image_size, 3))
            return image, tf.cast(element["label"], tf.int32)

        ds = ds.map(_decode, num_parallel_calls=num_parallel_calls)
    else:  # cifar10 -- in-memory, mirroring train_vit.create_cifar_dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        images = (x_train if is_training else x_test).astype("float32") / 255.0
        labels = (y_train if is_training else y_test).flatten().astype("int32")
        num_examples = int(images.shape[0])

        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if image_size != 32:
            ds = ds.map(
                lambda img, lbl: (
                    tf.ensure_shape(
                        tf.image.resize(img, (image_size, image_size), method="bilinear"),
                        (image_size, image_size, 3),
                    ),
                    lbl,
                ),
                num_parallel_calls=num_parallel_calls,
            )

    logger.info(
        f"{dataset} [{'train' if is_training else 'validation'}]: {num_examples} examples, "
        f"resized to {image_size}x{image_size}"
    )

    if is_training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        if augment:
            ds = ds.map(
                lambda img, lbl: (_augment(img, seed=seed), lbl),
                num_parallel_calls=num_parallel_calls,
            )

    def _normalize(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return (image - mean) / std, label

    ds = ds.map(_normalize, num_parallel_calls=num_parallel_calls)

    if not is_training:
        # Validation is small and re-read every epoch off a spinning HDD -- cache it.
        ds = ds.cache()

    if element_map_fn is not None:
        ds = ds.map(element_map_fn, num_parallel_calls=num_parallel_calls)

    if indexed_element_map_fn is not None:
        # DECISION plan-2026-08-01T195746-12a1f2db/D-008
        # `.enumerate()` goes AFTER `.repeat()`, never before. Before it, the
        # counter would restart at 0 every epoch and the same source image would
        # draw the SAME augmentation forever -- exactly the frozen-per-image
        # failure D-035 RED-proved wrong. After it, the counter runs 0,1,2,...
        # across the infinite repeat, so epoch 2's copy of image k keys
        # differently from epoch 1's. A determinism test alone cannot see this
        # difference; the cross-epoch-variation guard in
        # tests/test_train/test_energy_transformer/test_build_raw_image_dataset.py
        # is what pins it.
        #
        # MEASURED, not assumed: `.enumerate()` emits `(index, element)` and
        # `tf.data` passes a TUPLE element as ONE nested argument, so the map
        # lambda receives 2 args -- `(index, (image, label))` -- not 3. A
        # `lambda i, *elem: fn(i, *elem)` would hand the map fn a tuple where it
        # expects an image. `_call_indexed` unpacks that nesting.
        # See decisions.md D-008.
        def _call_indexed(index: tf.Tensor, *element: Any) -> Any:
            if len(element) == 1 and isinstance(element[0], tuple):
                return indexed_element_map_fn(index, *element[0])
            return indexed_element_map_fn(index, *element)

        ds = ds.repeat().enumerate()
        ds = ds.map(_call_indexed, num_parallel_calls=num_parallel_calls)
        ds = ds.batch(batch_size, drop_remainder=True)
    elif is_training:
        ds = ds.repeat().batch(batch_size, drop_remainder=True)
    else:
        ds = ds.batch(batch_size)

    ds = ds.prefetch(prefetch_buffer)
    return ds, num_examples, DATASET_NUM_CLASSES[dataset]
