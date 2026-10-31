"""One ingestion path for the labelled-image trainers.

:func:`load_classification_data` returns ``(train, select, test)`` for MNIST, CIFAR-10,
CIFAR-100 and ImageNet-1k as one :class:`ClassificationData`: three batched
``tf.data`` pipelines, the sizes, a small in-memory test subset for figures, the
normalization statistics and the class names. The protocol is the same for every
dataset: the SELECTION split (early stopping, the best checkpoint) is a held-out slice
of train, and the TEST split is the dataset's own and never influences selection. For
MNIST and CIFAR the slice is a seeded permutation cut (``validation_split``); for
ImageNet it is ``holdout_per_class`` images per class
(:func:`train.common.imagenet.load_imagenet_selection_splits`).

Constants (mean, std, class names, shapes) come from
:mod:`dl_techniques.datasets.vision.classification`. ``_load_raw`` is the ONE place under
``src/train`` that reads ``keras.datasets``; a test replaces it.

The pipeline order is fixed: image scaled to [0, 1] (and resized or cropped), then the
``augment`` hook, then normalization, then batching. Augmenting after normalizing and
clipping saturated most pixels in ``train_vit`` once (its D-006/D-007), so the hook always
sees [0, 1], for every dataset, ImageNet and Imagenette included. A hook is plain
``(image, label)`` or seeded ``(seed, image, label)``
(:mod:`dl_techniques.datasets.vision.augment_hook`); :mod:`train.common.image_augment`
builds the shared seeded ones.

Folder and CSV data (:func:`load_folder_data`, :func:`load_csv_data`) are the same record
from a list of image files: the files are listed without reading a pixel
(:func:`scan_image_folder`, :func:`read_image_csv`, which a caller may run before it
loads anything), a seeded cut of the TRAIN list selects, the test list is optional and,
when absent, the record's ``test_ds`` is ``None`` and ``n_test`` is 0.

The listing also refuses what would leak or mislead, by RESOLVED path (a symlink is the file it
points to): a file listed twice, under two classes, or in both the train and the test split; two
csv labels of one integer value; a class with fewer than two train images. It logs a WARNING for
images below a class directory's own level and for a class the seeded cut leaves out of a split.
:func:`listing_fingerprint` hashes the listing (relative path, class, size), reading no pixel. The
loader decodes every test file it will score before any training (:func:`find_undecodable`) and
names the file behind a decode error met by the statistics pass; ``ClassificationData.image_files``
lets a trainer name the one behind an error met by ``fit``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.augment_hook import Augment, with_hook
from dl_techniques.datasets.vision.classification import DatasetSpec, get_spec
from dl_techniques.datasets.vision.crops import flip_and_pad_crop
from dl_techniques.datasets.vision.imagenet2012 import IMAGENET2012_DEFAULT_ROOT
from dl_techniques.datasets.vision.imagenette import ImagenetteConfig, load_imagenette
from dl_techniques.datasets.vision.splits import (
    check_validation_split,
    seeded_split_indices,
)
from dl_techniques.utils.logger import logger
from train.common.imagenet import (
    DEFAULT_HOLDOUT_PER_CLASS,
    add_imagenet_arguments,
    eval_head,
    load_imagenet_selection_splits,
    require_imagenet_root,
)

#: Pixels added on each side before the random crop back to size.
AUGMENT_PAD = 4
#: Default fraction of train held out for selection (in-memory datasets).
DEFAULT_VALIDATION_SPLIT = 0.1
#: Rows per pass of every evaluation pipeline.
EVAL_BATCH_SIZE = 256
#: Default number of ImageNet test images kept in memory for figures.
DEFAULT_EVAL_SAMPLES = 2000

NORMALIZE_MODES = ("standardize", "unit")
LABEL_MODES = ("sparse", "one_hot")
#: File suffixes (lower case) that folder and CSV data accept: what ``tf.io.decode_image`` reads.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")



# ---------------------------------------------------------------------
# Raw arrays: the one keras.datasets seam
# ---------------------------------------------------------------------

def _load_raw(name: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Read MNIST or CIFAR from the local Keras cache.

    :return: ``((x_train, y_train), (x_test, y_test))``: ``uint8`` images ``(N, H, W)`` for
        MNIST and ``(N, H, W, 3)`` for CIFAR, ``int`` labels ``(N,)``.
    """
    if name == "mnist":
        (xt, yt), (xe, ye) = keras.datasets.mnist.load_data()
    elif name == "cifar10":
        (xt, yt), (xe, ye) = keras.datasets.cifar10.load_data()
    elif name == "cifar100":
        (xt, yt), (xe, ye) = keras.datasets.cifar100.load_data()
    else:
        raise ValueError(f"no in-memory loader for dataset {name!r}")
    return (xt, np.asarray(yt).reshape(-1)), (xe, np.asarray(ye).reshape(-1))


# ---------------------------------------------------------------------
# Split arithmetic and pipelines
# ---------------------------------------------------------------------

def steps_per_epoch_for(
        n_train: int, batch_size: int, max_samples_flag: str = "--max-samples",
) -> int:
    """Optimizer steps per epoch of the train pipeline: ``n_train // batch_size``.

    The train pipeline drops the remainder (a smaller last batch is a second input shape
    that made XLA recompile the train step once, measured 20.7 s of an 80 s first epoch),
    so this is the exact number of batches one epoch yields and the value the LR
    schedule must be built with.

    :param max_samples_flag: What the caller's user calls the train-pool cap, named in the
        refusal. The default is the legacy flag; the API passes its own key.
    :raises ValueError: If ``n_train < batch_size`` (no batch at all would be yielded).
    """
    if n_train < batch_size:
        raise ValueError(
            f"batch_size {batch_size} exceeds the {n_train} train samples: the train "
            f"pipeline drops the incomplete last batch and would yield no step. "
            f"Use a batch size of at most {n_train}, or a larger {max_samples_flag}."
        )
    return n_train // batch_size


def capped_steps(
        steps_per_epoch: int, select_steps: int, max_steps: Optional[int],
) -> Tuple[int, int]:
    """``(steps_per_epoch, select_steps)``, each capped by ``--max-steps`` (smoke runs)."""
    if max_steps is None:
        return steps_per_epoch, select_steps
    return min(steps_per_epoch, max_steps), min(select_steps, max_steps)


def flip_pad_crop(
        pad: int = AUGMENT_PAD, mode: str = "CONSTANT", flip: bool = True,
        seed: Optional[int] = None,
) -> Augment:
    """The standard small-image augmentation: random horizontal flip, then a ``pad``-pixel
    pad and a random crop back to the image's own size.

    :param pad: Pixels added on each side.
    :param mode: ``tf.pad`` mode (``"CONSTANT"``, ``"REFLECT"``, ``"SYMMETRIC"``).
    :param flip: Whether to flip horizontally (not for digits).
    :param seed: Op-level seed (``None`` draws from the global generator).
    :return: A ``(image, label) -> (image, label)`` map for the ``augment`` argument.
    """

    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return flip_and_pad_crop(image, pad=pad, mode=mode, flip=flip, seed=seed), label

    return augment


def resizer(size: Optional[int]) -> Callable[[tf.Tensor], tf.Tensor]:
    """Bilinear resize to ``size`` x ``size`` (the identity for ``None``). Linear, so it
    commutes with the per-channel affine normalization applied after it."""
    if size is None:
        return lambda image: image
    return lambda image: tf.image.resize(image, (size, size), method="bilinear")


def array_examples(
        x: np.ndarray, y: np.ndarray, *, training: bool, seed: int,
        augment: Optional[Augment], normalize: Callable[[tf.Tensor], tf.Tensor],
        resize: Callable[[tf.Tensor], tf.Tensor] = lambda image: image,
) -> "tf.data.Dataset":
    """The unbatched ``(image, label)`` examples of an in-memory split.

    Training reshuffles every pass, then resizes, augments (on the [0, 1] image) and
    normalizes; evaluation only resizes and normalizes, in the array's order. Nothing is
    batched or repeated, so a caller can map per example before it batches: the
    classification pipelines below batch it, and ``train.common.image_streams`` adds the
    self-supervised element map first.

    ``from_tensor_slices`` embeds the arrays in the graph, so the pool must stay under
    TensorFlow's 2 GB constant limit (true for MNIST and CIFAR).
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda image, label: (resize(image), label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = with_hook(dataset, augment, seed)
    return dataset.map(lambda image, label: (normalize(image), label),
                       num_parallel_calls=tf.data.AUTOTUNE)


def _train_pipeline(
        x: np.ndarray, y: np.ndarray, batch_size: int, seed: int,
        augment: Optional[Augment], normalize: Callable[[tf.Tensor], tf.Tensor],
        resize: Callable[[tf.Tensor], tf.Tensor] = lambda image: image,
) -> "tf.data.Dataset":
    """Shuffled, augmented, normalized, batched train pipeline; drops the remainder."""
    return (
        array_examples(x, y, training=True, seed=seed, augment=augment,
                       normalize=normalize, resize=resize)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def _eval_pipeline(
        x: np.ndarray, y: np.ndarray, normalize: Callable[[tf.Tensor], tf.Tensor],
        resize: Callable[[tf.Tensor], tf.Tensor] = lambda image: image,
) -> "tf.data.Dataset":
    """Un-augmented, un-shuffled, normalized, batched pipeline (selection and test).

    Batched explicitly: ``fit`` evaluates numpy ``validation_data`` at Keras' small default
    batch when the train data is a dataset.
    """
    return (
        array_examples(x, y, training=False, seed=0, augment=None,
                       normalize=normalize, resize=resize)
        .batch(EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SplitArrays:
    """The in-memory splits of MNIST and CIFAR, scaled to [0, 1] (NOT normalized).

    For a trainer that builds its own pipeline (numpy augmentation, flattened inputs) or
    needs no pipeline at all: :func:`load_split_arrays` returns this alone. Labels follow
    the ``labels`` mode of the load. ``index_*`` are the rows of the original dataset
    (the Keras arrays' order) each split took, so a second label set read from the same
    source (CIFAR-100's coarse labels) can be aligned. ``None`` on
    :class:`ClassificationData` for ImageNet, which cannot be held in memory.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_select: np.ndarray
    y_select: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    input_shape: Tuple[int, int, int]
    num_classes: int
    class_names: list
    index_train: np.ndarray
    index_select: np.ndarray
    index_test: np.ndarray


@dataclass(frozen=True)
class ClassificationData:
    """What a classification trainer consumes, whatever the dataset.

    :param train_ds: Augmented, shuffled, batched train pipeline (drops the last batch).
    :param select_ds: The selection split: drives early stopping, the best checkpoint and
        the epoch-0 baseline. A held-out slice of train; never the test set.
    :param test_ds: The test split, to be evaluated whole; ``None`` when the data has no test
        split (folder and CSV data without one), in which case ``n_test`` is 0 and ``x_fig``
        holds no rows.
    :param steps_per_epoch: Batches in one epoch of ``train_ds``.
    :param n_train, n_select, n_test: Sample counts of the three splits.
    :param x_fig, y_fig: Test images (normalized as asked) and labels kept in memory for
        figures, ``ece`` and the analyzer: the whole test split for MNIST and CIFAR, the
        first ``eval_samples`` images for ImageNet.
    :param arrays: The in-memory splits in [0, 1], or ``None`` for ImageNet.
    :param mean, std: Per-channel statistics the inputs were standardized with
        (zeros and ones when ``normalize="unit"``), for un-standardizing figures.
    :param input_shape: ``(H, W, C)`` after the ``channels`` option.
    :param num_classes: Number of classes.
    :param class_names: One name per class.
    :param selection_split, test_split: Plain-language provenance of the two splits.
    :param image_files: The files ``train_ds`` and ``select_ds`` read (fit split, then
        selection split), for a trainer that has to name the file behind a decode error;
        ``None`` for the sources whose pixels are not files of a listing.
    :param fingerprint: :func:`listing_fingerprint` of a folder or CSV source; ``None`` for the
        built-in datasets, which carry none.
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: Optional["tf.data.Dataset"]
    steps_per_epoch: int
    n_train: int
    n_select: int
    n_test: int
    x_fig: np.ndarray
    y_fig: np.ndarray
    arrays: Optional[SplitArrays]
    mean: np.ndarray
    std: np.ndarray
    input_shape: Tuple[int, int, int]
    num_classes: int
    class_names: list
    selection_split: str
    test_split: str
    image_files: Optional[Tuple[str, ...]] = None
    fingerprint: Optional[str] = None

    @property
    def select_steps(self) -> int:
        """Batches in one pass of ``select_ds`` (it is batched at its own size, which is
        ``EVAL_BATCH_SIZE`` for the in-memory datasets and the trainer's batch size
        otherwise, so this is read from the dataset and never derived from ``batch_size``).

        :raises ValueError: The dataset's length is unknown (ImageNet).
        """
        steps = int(self.select_ds.cardinality())
        if steps < 0:
            raise ValueError("select_ds has no known length; pass validation_steps explicitly")
        return steps


# ---------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------

def _stats(spec: DatasetSpec, channels: int, normalize: str) -> Tuple[np.ndarray, np.ndarray]:
    if normalize == "unit":
        return np.zeros(channels, np.float32), np.ones(channels, np.float32)
    mean, std = np.asarray(spec.mean, np.float32), np.asarray(spec.std, np.float32)
    if len(mean) == 1 and channels > 1:
        mean, std = np.repeat(mean, channels), np.repeat(std, channels)
    return mean, std


def _resolve_channels(spec: DatasetSpec, channels: Optional[int]) -> int:
    native = spec.input_shape[-1]
    if channels is None or channels == native:
        return native
    if spec.name == "mnist" and channels == 3:
        return 3
    raise ValueError(
        f"channels={channels} is not available for {spec.name!r} (native {native}"
        + (", or 3 by repeating the channel" if spec.name == "mnist" else "") + ")"
    )


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return keras.utils.to_categorical(y, num_classes).astype(np.float32)


def load_classification_data(
        name: str,
        *,
        batch_size: int,
        seed: int,
        validation_split: Optional[float] = None,
        holdout_per_class: Optional[int] = None,
        normalize: str = "standardize",
        labels: str = "sparse",
        channels: Optional[int] = None,
        augment: Optional[Augment] = None,
        imagenet_augment: bool = True,
        imagenet_augment_color: bool = False,
        crop_policy: str = "random_resized_crop",
        resize: Optional[int] = None,
        max_samples: Optional[int] = None,
        eval_samples: int = DEFAULT_EVAL_SAMPLES,
        imagenet_root: Optional[str] = None,
        devkit_dir: Optional[str] = None,
        image_size: Optional[int] = None,
        max_samples_flag: str = "--max-samples",
) -> ClassificationData:
    """Load ``name`` as ``(train, select, test)`` with the one selection protocol.

    :param name: ``mnist``, ``cifar10``, ``cifar100``, ``imagenette`` or ``imagenet``.
    :param batch_size: Batch size of all three pipelines (the train one drops the remainder).
    :param seed: Seed of the split, the shuffle and ImageNet's augmentation.
    :param validation_split: Fraction of train held out for selection; MNIST, CIFAR and
        Imagenette (``None`` means :data:`DEFAULT_VALIDATION_SPLIT`). Strictly inside (0, 1).
    :param holdout_per_class: Selection images per class; ImageNet only, refused elsewhere (``None`` means
        :data:`train.common.imagenet.DEFAULT_HOLDOUT_PER_CLASS`).
    :param normalize: ``"standardize"`` ((x - mean) / std with the spec's constants) or
        ``"unit"`` (leave in [0, 1]).
    :param labels: ``"sparse"`` integer ids or ``"one_hot"`` float vectors.
    :param channels: MNIST only: ``3`` repeats the channel; ``None`` keeps it native.
    :param augment: A map applied to the train pipeline on the [0, 1] image, after the
        resize or crop and before normalization, for every dataset: plain
        ``(image, label)``, e.g. :func:`flip_pad_crop`, or seeded ``(seed, image, label)``,
        e.g. ``train.common.image_augment.build_augment``. ``None`` means none. ImageNet and
        Imagenette also need ``imagenet_augment=True`` (the hook runs after their crop).
    :param imagenet_augment: ImageNet and Imagenette: the loader's own train augmentation
        (the ``crop_policy`` of Imagenette; random-resized crop and flip for ImageNet).
    :param imagenet_augment_color: ImageNet only: also jitter brightness, contrast,
        saturation and hue on the train images.
    :param crop_policy: Imagenette only: ``"random_resized_crop"`` (the ImageNet recipe) or
        ``"resize"`` (resize the whole image; train flips and reflect-pad-crops).
    :param resize: In-memory datasets only: bilinear-resize every image of every pipeline to
        ``resize`` x ``resize`` before the hook (upsampling CIFAR for a model defined at
        227 px). ``input_shape`` is then the resized shape; ``arrays`` and ``x_fig`` keep
        the native resolution (a resized test set would be a very large array).
    :param max_samples: Smoke-run cap on the train pool and on the test split (ImageNet:
        ``max_images``; the selection split is never capped there; Imagenette: as for CIFAR).
    :param eval_samples: ImageNet and Imagenette: test images kept in memory for the figures; ``0``
        keeps none (``x_fig`` and ``y_fig`` are empty), for a trainer that draws no figures.
    :param imagenet_root, devkit_dir: ImageNet only, ignored otherwise (every parser carries
        ``--imagenet-root`` with a default); ``None`` means the loader's defaults.
    :param image_size: ImageNet and Imagenette: the square input side (``None`` means 224).
    :param max_samples_flag: What the caller's user calls ``max_samples``, named in the
        too-large-batch refusal of the in-memory datasets (``"--max-samples"`` for the
        legacy trainers, the API passes its own key).
    :raises ValueError: An option that does not apply to ``name``, an unknown mode, a
        selection split holding out zero samples, a batch larger than the train split.
    :raises FileNotFoundError: The ImageNet tree or its devkit, or the Imagenette records,
        are missing.
    """
    spec = get_spec(name)
    if normalize not in NORMALIZE_MODES:
        raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {normalize!r}")
    if labels not in LABEL_MODES:
        raise ValueError(f"labels must be one of {LABEL_MODES}, got {labels!r}")
    if spec.in_memory:
        if holdout_per_class is not None:
            raise ValueError(
                f"holdout_per_class={holdout_per_class} does not apply to {spec.name!r}: "
                "use validation_split"
            )
        return _load_in_memory(
            spec, batch_size, seed, validation_split, normalize, labels, channels, augment,
            resize, max_samples, max_samples_flag)
    if resize is not None:
        raise ValueError(f"resize applies to in-memory datasets; {spec.name} takes image_size")
    if channels not in (None, 3):
        raise ValueError(f"channels={channels} is not available for {spec.name!r} (3)")
    if spec.source == "records":
        if holdout_per_class is not None:
            raise ValueError(
                f"holdout_per_class={holdout_per_class} does not apply to {spec.name!r}: "
                "use validation_split"
            )
        if imagenet_augment_color:
            raise ValueError(f"imagenet_augment_color does not apply to {spec.name!r}")
        return _load_imagenette(
            spec, batch_size, seed, validation_split, normalize, labels, imagenet_augment,
            crop_policy, max_samples, eval_samples, image_size, augment)
    if validation_split is not None:
        raise ValueError(
            f"validation_split={validation_split} does not apply to ImageNet, which selects "
            "on holdout_per_class images per class taken out of train"
        )
    return _load_imagenet(
        spec, batch_size, seed, holdout_per_class, normalize, labels, imagenet_augment,
        imagenet_augment_color,
        max_samples, eval_samples, imagenet_root, devkit_dir, image_size, augment)


def load_split_arrays(
        name: str,
        *,
        seed: int,
        validation_split: Optional[float] = None,
        labels: str = "sparse",
        channels: Optional[int] = None,
        max_samples: Optional[int] = None,
) -> SplitArrays:
    """The in-memory splits of MNIST or CIFAR in [0, 1], with no pipeline built.

    The split is :func:`load_classification_data`'s (the same seeded permutation), for a
    trainer that needs arrays only, such as a numpy loop or a flattened-input MLP.

    :param name: ``mnist``, ``cifar10`` or ``cifar100``.
    :param seed: Seed of the split.
    :param validation_split: Fraction of train held out for selection (``None`` means
        :data:`DEFAULT_VALIDATION_SPLIT`), strictly inside (0, 1).
    :param labels: ``"sparse"`` or ``"one_hot"``.
    :param channels: MNIST only: ``3`` repeats the channel; ``None`` keeps it native.
    :param max_samples: Smoke-run cap on the train pool and the test split.
    :raises ValueError: ``name`` is not an in-memory dataset, a bad option, or a selection
        split holding out zero samples.
    """
    spec = get_spec(name)
    if not spec.in_memory:
        raise ValueError(f"{spec.name!r} is not an in-memory dataset; use load_classification_data")
    if labels not in LABEL_MODES:
        raise ValueError(f"labels must be one of {LABEL_MODES}, got {labels!r}")
    validation_split = DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split
    check_validation_split(validation_split)
    n_channels = _resolve_channels(spec, channels)
    (x_train, y_train), (x_test, y_test) = _load_raw(spec.name)
    if x_train.ndim == 3:
        x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
    if n_channels != x_train.shape[-1]:
        x_train, x_test = np.repeat(x_train, n_channels, -1), np.repeat(x_test, n_channels, -1)

    train_idx, select_idx, test_order = seeded_split_indices(
        len(x_train), len(x_test), seed, validation_split, max_samples)

    def unit(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return x[idx].astype(np.float32) / 255.0

    def label(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
        y = y[idx].astype(np.int32)
        return _one_hot(y, spec.num_classes) if labels == "one_hot" else y

    return SplitArrays(
        x_train=unit(x_train, train_idx), y_train=label(y_train, train_idx),
        x_select=unit(x_train, select_idx), y_select=label(y_train, select_idx),
        x_test=unit(x_test, test_order), y_test=label(y_test, test_order),
        input_shape=(*x_train.shape[1:3], n_channels), num_classes=spec.num_classes,
        class_names=spec.class_names_for(spec.num_classes),
        index_train=train_idx, index_select=select_idx, index_test=test_order,
    )


def _load_in_memory(
        spec: DatasetSpec, batch_size: int, seed: int, validation_split: Optional[float],
        normalize: str, labels: str, channels: Optional[int], augment: Optional[Augment],
        resize: Optional[int], max_samples: Optional[int],
        max_samples_flag: str = "--max-samples",
) -> ClassificationData:
    arrays = load_split_arrays(
        spec.name, seed=seed, validation_split=validation_split, labels=labels,
        channels=channels, max_samples=max_samples)
    n_channels = arrays.input_shape[-1]
    mean, std = _stats(spec, n_channels, normalize)
    mean_t, std_t = tf.constant(mean), tf.constant(std)

    def norm(image: tf.Tensor) -> tf.Tensor:
        return (image - mean_t) / std_t

    def norm_np(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    steps = steps_per_epoch_for(len(arrays.x_train), batch_size, max_samples_flag)
    logger.info(
        f"{spec.name}: train {arrays.x_train.shape}, select {arrays.x_select.shape}, "
        f"test {arrays.x_test.shape}; {normalize}, labels {labels}"
    )
    split = (f"validation_split={DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split}"
             " of the train set (seeded permutation)")
    return ClassificationData(
        train_ds=_train_pipeline(arrays.x_train, arrays.y_train, batch_size, seed, augment,
                                 norm, resizer(resize)),
        select_ds=_eval_pipeline(arrays.x_select, arrays.y_select, norm, resizer(resize)),
        test_ds=_eval_pipeline(arrays.x_test, arrays.y_test, norm, resizer(resize)),
        steps_per_epoch=steps,
        n_train=int(len(arrays.x_train)), n_select=int(len(arrays.x_select)),
        n_test=int(len(arrays.x_test)),
        x_fig=norm_np(arrays.x_test), y_fig=arrays.y_test, arrays=arrays,
        mean=mean, std=std,
        input_shape=arrays.input_shape if resize is None else (resize, resize, n_channels),
        num_classes=spec.num_classes, class_names=arrays.class_names,
        selection_split=split,
        test_split="the dataset's test set",
    )


def _finish_file_based(
        spec: DatasetSpec, batch_size: int, labels: str, normalize: str, size: int,
        eval_samples: int, train, select, test, selection_split: str, test_split: str,
        image_files: Optional[Tuple[str, ...]] = None, fingerprint: Optional[str] = None,
) -> ClassificationData:
    """The record for a file-based dataset from its three loader splits (ImageNet, Imagenette,
    folder and CSV data); ``test`` may be ``None`` (no test split)."""
    if test is None or eval_samples == 0:
        x_fig, y_fig = np.zeros((0, size, size, 3), np.float32), np.zeros((0,), np.int32)
    else:
        x_fig, y_fig = eval_head(test, eval_samples)
    mean, std = _stats(spec, 3, normalize)

    def encode(dataset: "tf.data.Dataset") -> "tf.data.Dataset":
        if labels == "one_hot":
            dataset = dataset.map(
                lambda x, y: (x, tf.one_hot(y, spec.num_classes)),
                num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(tf.data.AUTOTUNE)

    if labels == "one_hot":
        y_fig = _one_hot(y_fig, spec.num_classes)
    return ClassificationData(
        train_ds=encode(train.dataset), select_ds=encode(select.dataset),
        test_ds=None if test is None else encode(test.dataset),
        steps_per_epoch=steps_per_epoch_for(train.num_samples, batch_size),
        n_train=train.num_samples, n_select=select.num_samples,
        n_test=0 if test is None else test.num_samples,
        x_fig=x_fig, y_fig=y_fig, arrays=None, mean=mean, std=std,
        input_shape=(size, size, 3), num_classes=train.num_classes,
        class_names=list(train.class_names),
        selection_split=selection_split, test_split=test_split,
        image_files=image_files, fingerprint=fingerprint,
    )


def _load_imagenet(
        spec: DatasetSpec, batch_size: int, seed: int, holdout_per_class: Optional[int],
        normalize: str, labels: str, augment: bool, augment_color: bool,
        max_samples: Optional[int], eval_samples: int, root: Optional[str], devkit_dir: Optional[str],
        image_size: Optional[int], augment_hook: Optional[Augment],
) -> ClassificationData:
    root = IMAGENET2012_DEFAULT_ROOT if root is None else root
    holdout = DEFAULT_HOLDOUT_PER_CLASS if holdout_per_class is None else holdout_per_class
    size = spec.input_shape[0] if image_size is None else image_size
    require_imagenet_root(root)
    train, select, test = load_imagenet_selection_splits(
        root, devkit_dir, size, batch_size, seed, augment=augment,
        augment_color=augment_color, holdout_per_class=holdout,
        normalize="imagenet" if normalize == "standardize" else "unit",
        num_classes=spec.num_classes, max_images=max_samples, augment_hook=augment_hook,
    )
    return _finish_file_based(
        spec, batch_size, labels, normalize, size, eval_samples, train, select, test,
        selection_split=f"{holdout} images per class held out of train "
                        f"({select.num_samples} images)",
        test_split="the official val split",
    )


def _load_imagenette(
        spec: DatasetSpec, batch_size: int, seed: int, validation_split: Optional[float],
        normalize: str, labels: str, augment: bool, crop_policy: str,
        max_samples: Optional[int], eval_samples: int, image_size: Optional[int],
        augment_hook: Optional[Augment],
) -> ClassificationData:
    split = DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split
    size = spec.input_shape[0] if image_size is None else image_size
    config = ImagenetteConfig(
        image_size=size, batch_size=batch_size, crop_policy=crop_policy, augment=augment,
        augment_hook=augment_hook,
        normalize="imagenet" if normalize == "standardize" else "unit",
        validation_split=split, seed=seed, max_images=max_samples,
    )
    train, select, test = (load_imagenette(name, config) for name in ("train", "select", "test"))
    return _finish_file_based(
        spec, batch_size, labels, normalize, size, eval_samples, train, select, test,
        selection_split=f"validation_split={split} of the train records (seeded permutation)",
        test_split="the labelled validation split",
    )


# ---------------------------------------------------------------------
# Folder and CSV data: a list of image files
# ---------------------------------------------------------------------

class DataSourceError(ValueError):
    """A folder or CSV source that cannot be used, naming the argument that is wrong.

    A :class:`ValueError`, so a caller that wraps every loader refusal catches it; ``key``
    (``"train_dir"``, ``"test_csv"``, ``"path_column"``, ...) lets a caller with its own
    config prefix the line with the dotted key.
    """

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key


@dataclass(frozen=True)
class FileListing:
    """The image files of a folder or CSV source, listed and labelled, nothing read.

    :param source: ``"folder"`` or ``"csv"``.
    :param class_names: One name per class; the label of a file is its index here.
    :param train_paths, train_labels: Every train file and its label (``int32``).
    :param test_paths, test_labels: The same for the test split, or ``None`` and ``None``.
    :param train_base, test_base: What a file's path is relative to in
        :func:`listing_fingerprint`: the root directory of a folder source, the directory of
        the CSV file of a CSV one (``test_base`` is ``None`` without a test split).
    """

    source: str
    class_names: List[str]
    train_paths: List[str]
    train_labels: np.ndarray
    test_paths: Optional[List[str]]
    test_labels: Optional[np.ndarray]
    train_base: str = ""
    test_base: Optional[str] = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _has_image_suffix(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES)


def _nested_images(class_dir: str) -> int:
    """Image files below the sub-directories of ``class_dir`` (hidden entries skipped)."""
    count = 0
    for folder, subdirs, names in os.walk(class_dir):
        subdirs[:] = [name for name in subdirs if not name.startswith(".")]
        if folder != class_dir:
            count += sum(1 for name in names if not name.startswith(".") and _has_image_suffix(name))
    return count


def _folder_files(root: str, key: str, minimum: int = 1) -> "dict[str, List[str]]":
    """``{class name: sorted image paths}`` of the class directories of ``root``.

    :param minimum: Images a class directory must hold (2 for a train root: one for the fit
        split and one for the selection split).
    """
    if not os.path.isdir(root):
        raise DataSourceError(key, f"{root!r} is not a directory")
    classes = {}
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        files = sorted(
            item.path for item in os.scandir(entry.path)
            if item.is_file() and not item.name.startswith(".") and _has_image_suffix(item.name))
        if not files:
            raise DataSourceError(
                key, f"class directory {entry.name!r} of {root!r} holds no images "
                     f"({', '.join(IMAGE_SUFFIXES)})")
        if len(files) < minimum:
            raise DataSourceError(
                key, f"class directory {entry.name!r} of {root!r} holds {len(files)} image; "
                     f"every class needs at least {minimum} (one to fit on, one to select on)")
        nested = _nested_images(entry.path)
        if nested:
            logger.warning(
                f"class directory {entry.name!r} of {root!r}: {nested} image file(s) in its "
                "sub-directories are ignored; only the images directly inside a class "
                "directory belong to the class")
        classes[entry.name] = files
    if len(classes) < 2:
        raise DataSourceError(
            key, f"{root!r} holds {len(classes)} class directories; a classifier needs at least "
                 "two (one directory of images per class)")
    return classes


def _same_classes(train: Sequence[str], test: Sequence[str], key: str) -> None:
    """Refuse a test split whose classes are not exactly the train split's."""
    if set(train) == set(test):
        return
    raise DataSourceError(
        key, f"the test classes differ from the train classes: only in train "
             f"{sorted(set(train) - set(test))}, only in test {sorted(set(test) - set(train))}")


def _labelled(files: "dict[str, List[str]]", names: List[str]) -> Tuple[List[str], np.ndarray]:
    paths = [path for name in names for path in files[name]]
    labels = np.concatenate([np.full(len(files[name]), index, np.int32)
                             for index, name in enumerate(names)])
    return paths, labels


def _resolver() -> Callable[[str], str]:
    """A function returning the resolved path of a file (symlinks followed).

    The directory part is resolved once per directory; only a file that is itself a symlink is
    resolved on its own, so a listing of a million files makes about one ``lstat`` each.
    """
    directories: "dict[str, str]" = {}

    def resolve(path: str) -> str:
        if os.path.islink(path):
            return os.path.realpath(path)
        folder, name = os.path.split(path)
        if folder not in directories:
            directories[folder] = os.path.realpath(folder)
        return os.path.join(directories[folder], name)

    return resolve


def _examples(items: Sequence[str], total: int) -> str:
    shown = "; ".join(items[:3])
    return shown + (f"; and {total - 3} more" if total > 3 else "")


def _refuse_repeats(
        paths: Sequence[str], classes: Sequence[str], resolve: Callable[[str], str], key: str,
) -> List[str]:
    """Resolve ``paths`` and refuse a file that is listed twice or under two classes.

    :param classes: The class text of each path.
    :return: The resolved paths, in the order of ``paths``.
    :raises DataSourceError: ``key`` names the source; the message gives examples.
    """
    resolved = [resolve(path) for path in paths]
    first: "dict[str, Tuple[str, str]]" = {}
    two_classes: List[str] = []
    twice: List[str] = []
    for path, real, name in zip(paths, resolved, classes):
        if real not in first:
            first[real] = (path, name)
        elif first[real][1] != name:
            two_classes.append(f"{path!r} is under {first[real][1]!r} (as {first[real][0]!r}) and under {name!r}")
        else:
            twice.append(repr(path))
    if two_classes:
        raise DataSourceError(
            key, f"{len(two_classes)} file(s) are under two classes (files are compared by "
                 f"resolved path, so a symlink is the file it points to): "
                 f"{_examples(two_classes, len(two_classes))}")
    if twice:
        raise DataSourceError(
            key, f"{len(twice)} file(s) are listed more than once (files are compared by "
                 f"resolved path, so a symlink is the file it points to): "
                 f"{_examples(twice, len(twice))}; a file in the list twice can land in both the "
                 "fit and the selection split")
    return resolved


def _refuse_shared(
        train_resolved: Sequence[str], test_paths: Sequence[str], test_resolved: Sequence[str],
        key: str,
) -> None:
    """Refuse a test file that is also a train file (by resolved path).

    :raises DataSourceError: ``key`` names the test source.
    """
    train = set(train_resolved)
    shared = [repr(path) for path, real in zip(test_paths, test_resolved) if real in train]
    if shared:
        raise DataSourceError(
            key, f"{len(shared)} of the {len(test_paths)} test file(s) are also train files "
                 f"(compared by resolved path): {_examples(shared, len(shared))}; a model scored on "
                 "images it trained on reports an accuracy it does not have")


def scan_image_folder(train_dir: str, test_dir: Optional[str] = None) -> FileListing:
    """List the images of ``train_dir/<class>/*`` (and ``test_dir/<class>/*``), reading none.

    A class is a sub-directory; hidden entries and files of other suffixes are skipped. Images
    inside a class's own sub-directories are not listed (a WARNING counts them).

    :param train_dir: The train root.
    :param test_dir: The test root, or ``None`` for a source with no test split.
    :return: The :class:`FileListing`; the class names are the sorted directory names.
    :raises DataSourceError: A root that is not a directory, fewer than two classes, a class
        directory with no image or with one train image, test classes that are not the train
        classes, a file under two classes (through a symlink, say), or a file in both splits.
    """
    train = _folder_files(train_dir, "train_dir", minimum=2)
    names = sorted(train)
    train_paths, train_labels = _labelled(train, names)
    resolve = _resolver()
    train_real = _refuse_repeats(
        train_paths, [names[i] for i in train_labels], resolve, "train_dir")
    test_paths = test_labels = test_base = None
    if test_dir is not None:
        test = _folder_files(test_dir, "test_dir")
        _same_classes(names, sorted(test), "test_dir")
        test_paths, test_labels = _labelled(test, names)
        test_real = _refuse_repeats(
            test_paths, [names[i] for i in test_labels], resolve, "test_dir")
        _refuse_shared(train_real, test_paths, test_real, "test_dir")
        test_base = os.path.abspath(test_dir)
    return FileListing("folder", names, train_paths, train_labels, test_paths, test_labels,
                       os.path.abspath(train_dir), test_base)


def _csv_rows(
        csv_path: str, key: str, path_column: str, label_column: str,
) -> Tuple[List[str], List[str]]:
    """``(image paths, label texts)`` of one CSV file; every file must exist and be an image.

    The file is read as ``utf-8-sig``, so the byte-order mark a spreadsheet export starts with
    does not become part of the first column's name.
    """
    if not os.path.isfile(csv_path):
        raise DataSourceError(key, f"{csv_path!r} is not a file")
    base = os.path.dirname(os.path.abspath(csv_path))
    paths: List[str] = []
    labels: List[str] = []
    problems: List[str] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for column_key, column in (("path_column", path_column), ("label_column", label_column)):
            if column not in columns:
                raise DataSourceError(
                    column_key, f"column {column!r} is not in {csv_path!r} (columns: {columns})")
        for number, row in enumerate(reader, start=2):
            path, label = row.get(path_column), row.get(label_column)
            if path is None or label is None or not path.strip() or not label.strip():
                problems.append(f"line {number} has an empty {path_column!r} or {label_column!r}")
                continue
            path = path.strip()
            full = path if os.path.isabs(path) else os.path.join(base, path)
            if not _has_image_suffix(full):
                problems.append(f"line {number}: {path!r} is not one of {IMAGE_SUFFIXES}")
            elif not os.path.isfile(full):
                problems.append(f"line {number}: {full!r} does not exist")
            paths.append(full)
            labels.append(label.strip())
    if problems:
        shown = "; ".join(problems[:5]) + (f"; and {len(problems) - 5} more" if len(problems) > 5 else "")
        raise DataSourceError(key, f"{csv_path!r}: {shown}")
    if not paths:
        raise DataSourceError(key, f"{csv_path!r} holds no rows")
    return paths, labels


def _ordered_labels(labels: Sequence[str]) -> List[str]:
    """Class names: the distinct labels, numerically ordered when all are integers.

    The order is a function of the label texts alone: they are sorted before anything else,
    and two integer labels of equal value (``"01"`` and ``"1"``) are refused rather than tied,
    so the class order cannot depend on set iteration order (``PYTHONHASHSEED``).

    :raises DataSourceError: Two labels that are the same integer written differently.
    """
    distinct = sorted(set(labels))
    if all(re.fullmatch(r"-?[0-9]+", name) for name in distinct):
        by_value: "dict[int, List[str]]" = {}
        for name in distinct:
            by_value.setdefault(int(name), []).append(name)
        tied = [names for names in by_value.values() if len(names) > 1]
        if tied:
            raise DataSourceError(
                "label_column",
                f"the labels {tied[0]} are the same integer written differently, so they would be "
                "one class to you and two to the loader; write each class one way")
        return sorted(distinct, key=lambda name: (int(name), name))
    return distinct


def read_image_csv(
        train_csv: str, test_csv: Optional[str] = None, *,
        path_column: str = "path", label_column: str = "label",
) -> FileListing:
    """List the images named by a CSV file (and a test CSV), reading none.

    A relative image path is relative to the CSV file's directory. The class names are the
    distinct labels (integers in numeric order, otherwise sorted text). The file is read as
    ``utf-8-sig``.

    :raises DataSourceError: A file that is missing, a missing column, an empty cell, an image
        path that does not exist or is not an image suffix, fewer than two classes, a class
        with one train image, two labels of one integer value, test classes that are not the
        train classes, a file listed twice or under two labels (compared by resolved path), or
        a file in both CSVs.
    """
    train_paths, train_texts = _csv_rows(train_csv, "train_csv", path_column, label_column)
    names = _ordered_labels(train_texts)
    if len(names) < 2:
        raise DataSourceError(
            "train_csv", f"{train_csv!r} holds {len(names)} class(es); a classifier needs at least two")
    counts = collections.Counter(train_texts)
    single = [name for name in names if counts[name] < 2]
    if single:
        raise DataSourceError(
            "train_csv", f"{train_csv!r}: class(es) {single} have one image each; every class "
                         "needs at least 2 (one to fit on, one to select on)")
    resolve = _resolver()
    train_real = _refuse_repeats(train_paths, train_texts, resolve, "train_csv")
    index = {name: i for i, name in enumerate(names)}
    train_labels = np.asarray([index[text] for text in train_texts], np.int32)
    test_paths = test_labels = test_base = None
    if test_csv is not None:
        test_paths, test_texts = _csv_rows(test_csv, "test_csv", path_column, label_column)
        _same_classes(names, sorted(set(test_texts)), "test_csv")
        test_real = _refuse_repeats(test_paths, test_texts, resolve, "test_csv")
        _refuse_shared(train_real, test_paths, test_real, "test_csv")
        test_labels = np.asarray([index[text] for text in test_texts], np.int32)
        test_base = os.path.dirname(os.path.abspath(test_csv))
    return FileListing("csv", names, train_paths, train_labels, test_paths, test_labels,
                       os.path.dirname(os.path.abspath(train_csv)), test_base)


def listing_fingerprint(listing: FileListing) -> str:
    """A sha256 over the sorted ``(split, relative path, class, file size)`` rows of a listing.

    Computed from the listing and ``stat`` alone: no pixel is read and no tensor made, so it
    may run before ``setup_gpu``. A path is relative to :attr:`FileListing.train_base` (or
    ``test_base``), so moving a whole tree does not change it; a file added, removed, renamed,
    given another class or another size does. It does NOT see a change that keeps every name,
    class and size (two same-sized images swapped, a pixel edited).

    :return: the hex digest.
    """
    rows: List[List[object]] = []
    for split, paths, labels, base in (
            ("train", listing.train_paths, listing.train_labels, listing.train_base),
            ("test", listing.test_paths, listing.test_labels, listing.test_base)):
        if paths is None:
            continue
        for path, label in zip(paths, labels):
            rows.append([split, os.path.relpath(path, base).replace(os.sep, "/"),
                         listing.class_names[int(label)], os.path.getsize(path)])
    rows.sort()
    digest = hashlib.sha256(listing.source.encode("utf-8"))
    for row in rows:
        digest.update(b"\n" + json.dumps(row, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def decode_image_file(path: tf.Tensor, channels: int = 3) -> tf.Tensor:
    """One image file as a float32 ``(H, W, channels)`` tensor in [0, 1], at its own size.

    The one decoder of folder and CSV data; ``train.api.predictor`` calls it with the
    channel count the model was trained on, so a prediction reads a file the way the
    training pipeline did.

    :param path: a scalar string tensor (or a ``str``): the file.
    :param channels: ``1`` (grey) or ``3`` (RGB); ``tf.io.decode_image`` converts.
    :return: the decoded image, ``tf.float32`` in [0, 1].
    """
    image = tf.io.decode_image(tf.io.read_file(path), channels=channels, expand_animations=False)
    image.set_shape([None, None, channels])
    return tf.image.convert_image_dtype(image, tf.float32)


def scan_for_undecodable(paths: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Decode ``paths`` one at a time, in order, and return the first that cannot be decoded.

    The slow, exact half of :func:`find_undecodable`; a trainer calls it directly after a
    ``tf.errors.InvalidArgumentError`` has already told it that some file is bad.

    :param paths: image files.
    :return: ``(path, reason)`` of the first file :func:`decode_image_file` refuses with
        ``tf.errors.InvalidArgumentError`` (``reason`` is the first line of its message), or
        ``None`` when every file decodes. Any other error (a file that vanished) propagates.
    """
    for path in paths:
        try:
            decode_image_file(path)
        except tf.errors.InvalidArgumentError as exc:
            return path, str(exc).strip().splitlines()[0][:200]
    return None


def find_undecodable(paths: Sequence[str]) -> Optional[Tuple[str, str]]:
    """The first of ``paths`` that cannot be decoded, or ``None``; see :func:`scan_for_undecodable`.

    One parallel ``tf.data`` pass over every file; only if that pass meets a bad file are the
    files scanned one by one to name it.
    """
    if not len(paths):
        return None
    dataset = (tf.data.Dataset.from_tensor_slices(list(paths))
               .map(lambda path: tf.shape(decode_image_file(path)), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(256))
    try:
        for _ in dataset:
            pass
    except tf.errors.InvalidArgumentError:
        return scan_for_undecodable(paths)
    return None


def _name_undecodable(paths: Sequence[str], key: str, error: Exception) -> None:
    """Raise a :class:`DataSourceError` naming the undecodable file behind ``error``, if one is found.

    Returns without raising when every file decodes, so the caller re-raises the original.
    """
    found = scan_for_undecodable(paths)
    if found is not None:
        raise DataSourceError(
            key, f"{found[0]!r} cannot be decoded as an image ({found[1]}); repair or remove "
                 "the file") from error


def _source_keys(source: str) -> Tuple[str, str]:
    """``(train key, test key)`` of a source, as :class:`DataSourceError` names them."""
    return ("train_dir", "test_dir") if source == "folder" else ("train_csv", "test_csv")


def path_examples(
        paths: Sequence[str], labels: np.ndarray, *, training: bool, seed: int,
        augment: Optional[Augment], normalize: Callable[[tf.Tensor], tf.Tensor], size: int,
) -> "tf.data.Dataset":
    """The unbatched ``(image, label)`` examples of a list of image files.

    The sibling of :func:`array_examples` for files, with the same order of work: training
    reshuffles the PATHS every pass (cheap), then decodes, resizes to ``size`` x ``size`` (so
    files of mixed sizes batch), augments on the [0, 1] image and normalizes; evaluation
    keeps the list's order. ``from_tensor_slices(paths).map(decode)`` keeps the length known.
    """
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels, np.int32)))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    resize = resizer(size)
    dataset = dataset.map(lambda path, label: (resize(decode_image_file(path)), label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = with_hook(dataset, augment, seed)
    return dataset.map(lambda image, label: (normalize(image), label),
                       num_parallel_calls=tf.data.AUTOTUNE)


def _fit_split_statistics(paths: Sequence[str], size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation of the images at ``size`` x ``size``, in [0, 1].

    One pass over ``paths`` (the FIT split: the selection and test images never contribute),
    accumulated in float64.

    :raises ValueError: A channel with no variation (its standard deviation would be 0).
    """
    resize = resizer(size)
    dataset = (tf.data.Dataset.from_tensor_slices(list(paths))
               .map(lambda path: resize(decode_image_file(path)), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(64))
    total, total_square, pixels = np.zeros(3), np.zeros(3), 0
    for batch in dataset.as_numpy_iterator():
        batch = batch.astype(np.float64)
        total += batch.sum(axis=(0, 1, 2))
        total_square += np.square(batch).sum(axis=(0, 1, 2))
        pixels += batch.shape[0] * size * size
    mean = total / pixels
    std = np.sqrt(np.maximum(total_square / pixels - np.square(mean), 0.0))
    if np.any(std < 1e-6):
        raise ValueError(
            f"the train images have no variation in channel(s) {np.flatnonzero(std < 1e-6).tolist()}"
            f" (standard deviation {std.tolist()}), so they cannot be standardized; "
            "use normalize='unit'")
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class _FileSplit:
    """The four attributes ``_finish_file_based`` reads of a loader split."""

    dataset: "tf.data.Dataset"
    num_samples: int
    num_classes: int
    class_names: List[str]


def _warn_absent_classes(listing: FileListing, train_idx: np.ndarray, select_idx: np.ndarray) -> None:
    """Log a WARNING naming each class the seeded cut leaves out of the fit or the selection split.

    The cut is a plain permutation, not stratified, so a small class can miss a split. A class
    absent from selection is one the best epoch is chosen without; a class absent from fit is
    one the model never sees.
    """
    labels = np.asarray(listing.train_labels)
    for name, indices, consequence in (
            ("selection", select_idx,
             "the best epoch and early stopping are chosen without it"),
            ("fit", train_idx, "the model never trains on it")):
        present = set(labels[indices].tolist())
        absent = [listing.class_names[i] for i in range(listing.num_classes) if i not in present]
        if absent:
            logger.warning(
                f"{listing.source}: class(es) {absent} have no image in the {name} split "
                f"(the seeded cut is not stratified); {consequence}. A different seed, a "
                "smaller validation_split or more images of the class changes this")


def _load_file_listing(
        listing: FileListing, *, batch_size: int, seed: int, image_size: int,
        validation_split: Optional[float], eval_samples: int, normalize: str,
        augment: Optional[Augment],
        augment_from_mean: Optional[Callable[[Tuple[float, ...]], Optional[Augment]]],
        max_samples: Optional[int], max_samples_flag: str,
) -> ClassificationData:
    """The record of a listed folder or CSV source; see :func:`load_folder_data`."""
    if normalize not in NORMALIZE_MODES:
        raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {normalize!r}")
    if augment is not None and augment_from_mean is not None:
        raise ValueError("pass augment or augment_from_mean, not both")
    if image_size < 1:
        raise ValueError(f"image_size must be a positive int, got {image_size!r}")
    split = DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split
    train_all = np.asarray(listing.train_paths)
    n_test_files = 0 if listing.test_paths is None else len(listing.test_paths)
    train_idx, select_idx, test_order = seeded_split_indices(
        len(train_all), n_test_files, seed, split, max_samples)
    # Refuse a batch larger than the fit split now, in the caller's words, before any decode.
    steps_per_epoch_for(len(train_idx), batch_size, max_samples_flag)

    fit_paths = train_all[train_idx].tolist()
    train_key, test_key = _source_keys(listing.source)
    _warn_absent_classes(listing, train_idx, select_idx)
    if listing.test_paths is not None:
        # DECISION plan-2026-10-30T132432-a865fa08/D-070: every test file this run will score is
        # decoded NOW, before any weight is trained. Do NOT leave the test split to the
        # post-fit evaluation: one corrupt test file there costs a finished run its summary,
        # its final model and its resume. The train and selection files are NOT checked here
        # (a pass over them is a second epoch of decoding); a bad one met by the statistics
        # pass or by ``fit`` is named by ``_name_undecodable`` and the trainer's wrapper.
        used_test = np.asarray(listing.test_paths)[test_order].tolist()
        bad = find_undecodable(used_test)
        if bad is not None:
            raise DataSourceError(
                test_key, f"{bad[0]!r} cannot be decoded as an image ({bad[1]}); the test "
                          "files are checked before training so that a finished run is never "
                          "lost to one of them; repair or remove the file")
    # DECISION plan-2026-10-30T132432-a865fa08/D-019: the statistics come from the FIT split only. Do NOT pass
    # ``train_all`` (it holds the selection images, so the selection split would shape the
    # inputs it then selects on) and never the test files; and do NOT borrow ImageNet's
    # constants for an arbitrary folder, whose pixels they do not describe.
    if normalize == "standardize" or augment_from_mean is not None:
        logger.info(f"{listing.source}: statistics from one pass over {len(fit_paths)} fit images "
                    f"at {image_size} px")
        try:
            mean, std = _fit_split_statistics(fit_paths, image_size)
        except tf.errors.InvalidArgumentError as exc:
            _name_undecodable(fit_paths, train_key, exc)
            raise
    else:
        mean, std = np.zeros(3, np.float32), np.ones(3, np.float32)
    if augment_from_mean is not None:
        augment = augment_from_mean(tuple(float(m) for m in mean))
    spec = DatasetSpec(
        listing.source, listing.num_classes, (image_size, image_size, 3),
        tuple(float(m) for m in mean), tuple(float(s) for s in std),
        tuple(listing.class_names), "files", len(train_all))
    shift, scale = _stats(spec, 3, normalize)
    shift_t, scale_t = tf.constant(shift), tf.constant(scale)

    def norm(image: tf.Tensor) -> tf.Tensor:
        return (image - shift_t) / scale_t

    def evaluation(indices: np.ndarray, paths: np.ndarray, labels: np.ndarray) -> _FileSplit:
        dataset = (path_examples(paths[indices].tolist(), labels[indices], training=False,
                                 seed=0, augment=None, normalize=norm, size=image_size)
                   .batch(batch_size))
        return _FileSplit(dataset, int(len(indices)), spec.num_classes, listing.class_names)

    train_labels = np.asarray(listing.train_labels)
    train = _FileSplit(
        path_examples(fit_paths, train_labels[train_idx], training=True, seed=seed,
                      augment=augment, normalize=norm, size=image_size)
        .batch(batch_size, drop_remainder=True),
        int(len(train_idx)), spec.num_classes, listing.class_names)
    select = evaluation(select_idx, train_all, train_labels)
    test = None
    if listing.test_paths is not None:
        test = evaluation(test_order, np.asarray(listing.test_paths), np.asarray(listing.test_labels))
    logger.info(
        f"{listing.source}: train {train.num_samples}, select {select.num_samples}, "
        f"test {0 if test is None else test.num_samples}; {spec.num_classes} classes; "
        f"{normalize}, mean {spec.mean}, std {spec.std}")
    return _finish_file_based(
        spec, batch_size, "sparse", normalize, image_size, eval_samples, train, select, test,
        selection_split=f"validation_split={split} of the train files (seeded permutation)",
        test_split=("none: the data has no test split" if test is None
                    else "the test files"),
        image_files=tuple(fit_paths) + tuple(train_all[select_idx].tolist()),
        fingerprint=listing_fingerprint(listing),
    )


def load_folder_data(
        train_dir: str,
        *,
        batch_size: int,
        seed: int,
        image_size: int,
        test_dir: Optional[str] = None,
        validation_split: Optional[float] = None,
        eval_samples: int = DEFAULT_EVAL_SAMPLES,
        normalize: str = "standardize",
        augment: Optional[Augment] = None,
        augment_from_mean: Optional[Callable[[Tuple[float, ...]], Optional[Augment]]] = None,
        max_samples: Optional[int] = None,
        max_samples_flag: str = "--max-samples",
) -> ClassificationData:
    """Load ``train_dir/<class>/*.png`` (and ``test_dir``) as ``(train, select, test)``.

    The protocol is the other loaders': a seeded ``validation_split`` cut of the TRAIN files
    is the selection split, the test files are only scored. Every image is resized to
    ``image_size`` x ``image_size`` (bilinear, squashing the aspect ratio), so files of mixed
    sizes batch. ``normalize="standardize"`` uses the per-channel mean and standard deviation
    of ONE pass over the fit images (not the selection or test images), recorded as the
    record's ``mean`` and ``std``.

    :param train_dir: Root holding one directory of images per class.
    :param batch_size: Batch size of all three pipelines (the train one drops the remainder).
    :param seed: Seed of the cut, the shuffle and the augmentation.
    :param image_size: The square input side; folders have no native size.
    :param test_dir: The test root with the SAME class directories, or ``None``: the record's
        ``test_ds`` is then ``None``, ``n_test`` 0 and ``x_fig`` empty.
    :param validation_split: Fraction of the train files held out; ``None`` is
        :data:`DEFAULT_VALIDATION_SPLIT`.
    :param eval_samples: Test images kept in memory for figures; ``0`` keeps none.
    :param normalize: ``"standardize"`` or ``"unit"`` (leave in [0, 1]).
    :param augment: The train hook on the [0, 1] image (see :func:`load_classification_data`).
    :param augment_from_mean: Instead of ``augment``: a function of the fit images' per-channel
        mean returning the hook, for an op that needs it (random erasing paints the mean).
        Asking for it makes the statistics pass run even for ``normalize="unit"``.
    :param max_samples: Smoke-run cap on the train pool and the test split.
    :param max_samples_flag: What the caller's user calls ``max_samples``, named in the
        too-large-batch refusal.
    :raises DataSourceError: The listing is unusable (see :func:`scan_image_folder`).
    :raises ValueError: A bad option, a selection split holding out zero files, a batch larger
        than the fit split, or images with no variation under ``standardize``.
    """
    return _load_file_listing(
        scan_image_folder(train_dir, test_dir), batch_size=batch_size, seed=seed,
        image_size=image_size, validation_split=validation_split, eval_samples=eval_samples,
        normalize=normalize, augment=augment, augment_from_mean=augment_from_mean,
        max_samples=max_samples, max_samples_flag=max_samples_flag)


def load_csv_data(
        train_csv: str,
        *,
        batch_size: int,
        seed: int,
        image_size: int,
        test_csv: Optional[str] = None,
        path_column: str = "path",
        label_column: str = "label",
        validation_split: Optional[float] = None,
        eval_samples: int = DEFAULT_EVAL_SAMPLES,
        normalize: str = "standardize",
        augment: Optional[Augment] = None,
        augment_from_mean: Optional[Callable[[Tuple[float, ...]], Optional[Augment]]] = None,
        max_samples: Optional[int] = None,
        max_samples_flag: str = "--max-samples",
) -> ClassificationData:
    """Load the images a CSV file lists (and ``test_csv``) as ``(train, select, test)``.

    Everything but the listing is :func:`load_folder_data`'s: ``path_column`` holds the image
    path (relative to the CSV's directory, or absolute) and ``label_column`` the class.

    :raises DataSourceError: The listing is unusable (see :func:`read_image_csv`).
    """
    return _load_file_listing(
        read_image_csv(train_csv, test_csv, path_column=path_column, label_column=label_column),
        batch_size=batch_size, seed=seed, image_size=image_size,
        validation_split=validation_split, eval_samples=eval_samples, normalize=normalize,
        augment=augment, augment_from_mean=augment_from_mean, max_samples=max_samples,
        max_samples_flag=max_samples_flag)


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def add_selection_arguments(
        parser: argparse.ArgumentParser, imagenet: bool = True, in_memory: bool = True,
        figures: bool = True,
) -> None:
    """Add the flags that describe the selection protocol to ``parser``.

    ``--validation-split`` (MNIST and CIFAR), ``--max-samples``, and with ``imagenet``
    ``--holdout-per-class``, the ImageNet location flags and ``--eval-samples``. All
    default to ``None`` (or the shared default) so :func:`load_classification_data` can
    refuse an option that does not apply to the chosen dataset. A trainer of MNIST and
    CIFAR only passes ``imagenet=False``; an ImageNet-only trainer passes ``in_memory=False``
    (no ``--validation-split``); a trainer that draws no figures passes ``figures=False``
    (no ``--eval-samples``; it loads with ``eval_samples=0``).
    """
    if in_memory:
        parser.add_argument(
            "--validation-split", type=float, default=None,
            help="MNIST and CIFAR: fraction of the train set held out (seeded) for early "
                 f"stopping and checkpoint selection, strictly inside (0, 1) (default "
                 f"{DEFAULT_VALIDATION_SPLIT}); the test set never selects."
                 + (" Refused with imagenet." if imagenet else ""),
        )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the train pool and the test set at this many samples (smoke runs).",
    )
    if not imagenet:
        return
    parser.add_argument(
        "--holdout-per-class", type=int, default=None,
        help="Used only when --dataset imagenet: images per class taken out of train for "
             "early stopping and checkpoint selection; the official val split is then the "
             f"test set (default: {DEFAULT_HOLDOUT_PER_CLASS}, i.e. "
             f"{DEFAULT_HOLDOUT_PER_CLASS * 1000} images). Refused with MNIST and CIFAR.",
    )
    add_imagenet_arguments(parser, note="Used only when --dataset imagenet: ")
    if not figures:
        return
    parser.add_argument(
        "--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES,
        help="Used only when --dataset imagenet: test images kept in memory for figures "
             f"(default: {DEFAULT_EVAL_SAMPLES}); test metrics always use the whole split.",
    )
