"""Dataset utilities for training scripts that the classification loader does not cover.

MNIST, CIFAR and ImageNet classification data come from ``train.common.classification_data``.
What is left here: image-file discovery (:func:`collect_image_paths`), the raw-image tf.data
pipeline of the self-supervised trainers (:func:`build_raw_image_dataset`, imagenette via TFDS
and CIFAR-10 in memory) and :func:`fill_batch`.
"""

import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, List, Optional, Sequence, Set

from dl_techniques.datasets.vision.classification import CIFAR10_MEAN, CIFAR10_STD
from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD
from dl_techniques.utils.logger import logger


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
    exist is skipped with a warning; an error while scanning one that does
    exist raises (a partial listing must not become the training set).

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

    Raises:
        OSError: Whatever listing an existing directory raises (permissions, an
            I/O error, a directory removed mid-scan).
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
        # A scan that fails raises (D-183): a warning here trained on whatever
        # part of the directory was listed before the error.
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions_set:
                all_file_paths.append(str(file_path))

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
# imagenette branch is likewise deliberate and stays LOCAL (tfds pulls in a
# heavy import chain and this module no longer imports it at the top): it is
# part of the moved body, and the move is behaviour-preserving by AST equality
# against the pre-move source (verification.md, step 3).
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

#: ``crop_policy`` values of :func:`build_raw_image_dataset`. ``"resize"`` is the
#: original behaviour every existing caller gets. ``"random_resized_crop"`` is the
#: ImageNet-classifier recipe (Inception-style random-resized crop plus flip for
#: training, resize-then-centre-crop for evaluation), imagenette only.
CROP_POLICIES: Tuple[str, ...] = ("resize", "random_resized_crop")

#: Evaluation centre-crop fraction of ``"random_resized_crop"``: the shorter side
#: is resized to ``image_size / EVAL_CROP_FRACTION`` and the centre
#: ``image_size`` square is kept (the torchvision 256 -> 224 ratio).
EVAL_CROP_FRACTION = 0.875

#: Area and aspect-ratio ranges of the training random-resized crop (the
#: Inception / torchvision ``RandomResizedCrop`` defaults).
RANDOM_RESIZED_CROP_AREA = (0.08, 1.0)
RANDOM_RESIZED_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)


def _random_resized_crop(
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


def _resize_and_centre_crop(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Resize the shorter side to ``image_size / EVAL_CROP_FRACTION``, keep the centre square."""
    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = (image_size / EVAL_CROP_FRACTION) / tf.reduce_min(shape)
    resized = tf.cast(tf.round(shape * scale), tf.int32)
    image = tf.image.resize(image, resized, method="bilinear")
    return tf.image.resize_with_crop_or_pad(image, image_size, image_size)


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
        crop_policy: str = "resize",
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
        crop_policy: Image geometry, one of :data:`CROP_POLICIES`. ``"resize"``
            (the default, what every caller before 2026-10-16 gets) resizes
            the whole image to ``image_size`` square and augments with flip
            plus reflect-pad-and-crop. ``"random_resized_crop"`` (imagenette
            only) trains on a random-resized crop plus flip of the
            full-resolution decode and evaluates on a resize to
            ``image_size / EVAL_CROP_FRACTION`` then centre crop; with
            ``augment=False`` the training pipeline uses the evaluation
            geometry. The crop runs before the shuffle buffer, so the buffer
            holds ``image_size`` images as in ``"resize"``.
        num_parallel_calls: ``tf.data`` parallelism.
        prefetch_buffer: ``tf.data`` prefetch depth.

    Returns:
        ``(ds, num_examples, num_classes)``. ``num_examples`` is the split's cardinality, from
        which the caller derives ``steps_per_epoch``.

    Raises:
        ValueError: If ``dataset`` is not supported or ``image_size``/``batch_size`` are
            non-positive; if ``crop_policy`` is unknown, or is
            ``"random_resized_crop"`` with a dataset other than imagenette; if BOTH ``element_map_fn`` and ``indexed_element_map_fn`` are
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
    if crop_policy not in CROP_POLICIES:
        raise ValueError(
            f"Unknown crop_policy {crop_policy!r}; known: {list(CROP_POLICIES)}"
        )
    if crop_policy == "random_resized_crop" and dataset != "imagenette":
        raise ValueError(
            "crop_policy='random_resized_crop' is implemented for imagenette "
            f"only (full-resolution records to crop from), not {dataset!r}"
        )

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

        # DECISION plan-2026-10-14T062616-81938e81/D-104
        # Do NOT move the random-resized crop after `.shuffle()` with the other
        # augmentation: the buffer would then hold full-resolution decodes
        # (~1.5 MB each as float32, 4096 of them by default) instead of
        # `image_size` crops. Crop-then-shuffle draws the same distribution.
        random_crop = (
            crop_policy == "random_resized_crop" and is_training and augment)

        def _decode(element: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
            # Imagenette records are VARIABLE-SIZE (e.g. 320x396x3). The resize is MANDATORY:
            # without it the batch is ragged and tf.data raises at the first batch.
            image = tf.cast(element["image"], tf.float32) / 255.0
            if crop_policy == "resize":
                image = tf.image.resize(image, (image_size, image_size), method="bilinear")
            elif random_crop:
                image = _random_resized_crop(image, image_size, seed=seed)
            else:
                image = _resize_and_centre_crop(image, image_size)
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
        # "random_resized_crop" augmented in `_decode`; only "resize" augments here.
        if augment and crop_policy == "resize":
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


# ---------------------------------------------------------------------
# Batch filling for index-addressed PyDatasets
# ---------------------------------------------------------------------

def fill_batch(
        owner: str,
        idx: int,
        batch_size: int,
        indices: Sequence[int],
        load_one: Callable[[int], Optional[Any]],
        describe: Callable[[int], str],
        failure: str,
) -> List[Any]:
    """The samples of batch ``idx``: the next window of files, a skipped file refilled.

    Interface contract (3 callers: ``train.common.megadepth.MegaDepthDataset``,
    ``train.omnipoint.data.MegaDepthPinholeDataset`` and
    ``train.omnipoint.kitti_depth.KittiDepthDataset``). Attempt ``a`` (from
    0, at most ``3 * batch_size`` and at most ``len(indices)``) reads file ``indices[(idx * batch_size + a)
    % len(indices)]`` through ``load_one(file_index)``; a ``None`` result is
    skipped, and filling stops when ``batch_size`` results are kept. When every
    file loads, batch ``i`` holds exactly files ``i*B .. i*B+B-1`` of
    ``indices``, so the batches of an epoch are disjoint and read each file
    once. A skipped file is refilled from the files after the window, which
    the next batch reads again (the D-126 trade-off). No batch reads a file
    twice: with fewer files than ``batch_size`` it holds each once (D-199).

    Args:
        owner: The dataset class name, for the error message.
        idx: The batch index.
        batch_size: The number of samples wanted.
        indices: The epoch's file order (shuffled or not).
        load_one: Loads one file by its index; ``None`` skips it.
        describe: Names a file by its index, for the error message.
        failure: Completes ``"no <failure> in N attempts"``, for example
            ``"KITTI pair loaded"``.

    Returns:
        Between 1 and ``batch_size`` results of ``load_one``, in reading order.

    Raises:
        RuntimeError: When no file of the attempts gives a result. The message
            names the batch and every file tried.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-193
    # Do NOT add the number of kept samples to this offset: the attempt count
    # already counts every read, kept or skipped, so the sum advanced by 2 per
    # sample. Consecutive batches overlapped by half and the files at odd
    # offsets were never read (D-126, then again in both OmniPoint loaders).
    kept: List[Any] = []
    tried: List[int] = []
    for attempt in range(min(batch_size * 3, len(indices))):
        if len(kept) == batch_size:
            break
        file_index = int(indices[(idx * batch_size + attempt) % len(indices)])
        tried.append(file_index)
        result = load_one(file_index)
        if result is not None:
            kept.append(result)
    if not kept:
        raise RuntimeError(
            f"{owner} batch {idx}: no {failure} in {len(tried)} attempts; tried "
            f"{[describe(i) for i in tried]}")
    return kept
