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

The in-memory pipeline order is fixed: image scaled to [0, 1], then the ``augment``
hook, then normalization, then batching. Augmenting after normalizing and clipping
saturated most pixels in ``train_vit`` once (its D-006/D-007), so the hook always sees
[0, 1].
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.classification import DatasetSpec, get_spec
from dl_techniques.datasets.vision.imagenet2012 import IMAGENET2012_DEFAULT_ROOT
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

Augment = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


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

def split_sizes(
        n_available: int, max_samples: Optional[int], validation_split: float
) -> Tuple[int, int]:
    """Sizes of the fit and selection splits: the ONE place the split arithmetic lives.

    :func:`load_classification_data` (from the array it loaded) and a trainer's config
    validation (from the known train size, before any run directory exists) both call it,
    so an early refusal cannot disagree with the split the run will get.

    :param n_available: Samples in the train set.
    :param max_samples: The ``--max-samples`` cap, or ``None``.
    :param validation_split: Fraction of the (capped) pool held out.
    :return: ``(n_fit, n_select)``, summing to the pool.
    :raises ValueError: If the selection split would hold out zero samples.
    """
    pool = n_available if max_samples is None else min(max_samples, n_available)
    n_select = int(pool * validation_split)
    if n_select == 0:
        raise ValueError(f"validation_split={validation_split} holds out 0 of {pool} samples")
    return pool - n_select, n_select


def steps_per_epoch_for(n_train: int, batch_size: int) -> int:
    """Optimizer steps per epoch of the train pipeline: ``n_train // batch_size``.

    The train pipeline drops the remainder (a smaller last batch is a second input shape
    that made XLA recompile the train step once, measured 20.7 s of an 80 s first epoch),
    so this is the exact number of batches one epoch yields and the value the LR
    schedule must be built with.

    :raises ValueError: If ``n_train < batch_size`` (no batch at all would be yielded).
    """
    if n_train < batch_size:
        raise ValueError(
            f"batch_size {batch_size} exceeds the {n_train} train samples: the train "
            f"pipeline drops the incomplete last batch and would yield no step. "
            f"Use a batch size of at most {n_train}, or a larger --max-samples."
        )
    return n_train // batch_size


def flip_pad_crop(pad: int = AUGMENT_PAD, mode: str = "CONSTANT", flip: bool = True) -> Augment:
    """The standard small-image augmentation: random horizontal flip, then a ``pad``-pixel
    pad and a random crop back to the image's own size.

    :param pad: Pixels added on each side.
    :param mode: ``tf.pad`` mode (``"CONSTANT"``, ``"REFLECT"``, ``"SYMMETRIC"``).
    :param flip: Whether to flip horizontally (not for digits).
    :return: A ``(image, label) -> (image, label)`` map for the ``augment`` argument.
    """

    def augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if flip:
            image = tf.image.random_flip_left_right(image)
        size = tf.shape(image)
        image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode=mode)
        return tf.image.random_crop(image, size), label

    return augment


def _resizer(size: Optional[int]) -> Callable[[tf.Tensor], tf.Tensor]:
    """Bilinear resize to ``size`` x ``size`` (the identity for ``None``). Linear, so it
    commutes with the per-channel affine normalization applied after it."""
    if size is None:
        return lambda image: image
    return lambda image: tf.image.resize(image, (size, size), method="bilinear")


def _train_pipeline(
        x: np.ndarray, y: np.ndarray, batch_size: int, seed: int,
        augment: Optional[Augment], normalize: Callable[[tf.Tensor], tf.Tensor],
        resize: Callable[[tf.Tensor], tf.Tensor] = lambda image: image,
) -> "tf.data.Dataset":
    """Shuffled, augmented ([0, 1]), normalized, batched train pipeline; drops the remainder.

    ``from_tensor_slices`` embeds the arrays in the graph, so the pool must stay under
    TensorFlow's 2 GB constant limit (true for MNIST and CIFAR).
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(
        len(x), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda image, label: (resize(image), label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return (
        dataset.map(lambda image, label: (normalize(image), label),
                    num_parallel_calls=tf.data.AUTOTUNE)
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
        tf.data.Dataset.from_tensor_slices((x, y))
        .map(lambda image, label: (normalize(resize(image)), label),
             num_parallel_calls=tf.data.AUTOTUNE)
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
    :param test_ds: The test split, to be evaluated whole.
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
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: "tf.data.Dataset"
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
        resize: Optional[int] = None,
        max_samples: Optional[int] = None,
        eval_samples: int = DEFAULT_EVAL_SAMPLES,
        imagenet_root: Optional[str] = None,
        devkit_dir: Optional[str] = None,
        image_size: Optional[int] = None,
) -> ClassificationData:
    """Load ``name`` as ``(train, select, test)`` with the one selection protocol.

    :param name: ``mnist``, ``cifar10``, ``cifar100`` or ``imagenet``.
    :param batch_size: Batch size of all three pipelines (the train one drops the remainder).
    :param seed: Seed of the split, the shuffle and ImageNet's augmentation.
    :param validation_split: Fraction of train held out for selection; MNIST and CIFAR only
        (``None`` means :data:`DEFAULT_VALIDATION_SPLIT`). Strictly inside (0, 1).
    :param holdout_per_class: Selection images per class; ImageNet only (``None`` means
        :data:`train.common.imagenet.DEFAULT_HOLDOUT_PER_CLASS`).
    :param normalize: ``"standardize"`` ((x - mean) / std with the spec's constants) or
        ``"unit"`` (leave in [0, 1]).
    :param labels: ``"sparse"`` integer ids or ``"one_hot"`` float vectors.
    :param channels: MNIST only: ``3`` repeats the channel; ``None`` keeps it native.
    :param augment: In-memory datasets only: a map applied to the train pipeline on the
        [0, 1] image, e.g. :func:`flip_pad_crop`. ``None`` means no augmentation.
    :param imagenet_augment: ImageNet only: the loader's random-resized-crop and flip.
    :param imagenet_augment_color: ImageNet only: also jitter brightness, contrast,
        saturation and hue on the train images.
    :param resize: In-memory datasets only: bilinear-resize every image of every pipeline to
        ``resize`` x ``resize`` before the hook (upsampling CIFAR for a model defined at
        227 px). ``input_shape`` is then the resized shape; ``arrays`` and ``x_fig`` keep
        the native resolution (a resized test set would be a very large array).
    :param max_samples: Smoke-run cap on the train pool and on the test split (ImageNet:
        ``max_images``; the selection split is never capped there).
    :param eval_samples: ImageNet only: test images kept in memory for the figures; ``0``
        keeps none (``x_fig`` and ``y_fig`` are empty), for a trainer that draws no figures.
    :param imagenet_root, devkit_dir, image_size: ImageNet only, ignored otherwise (every
        parser carries ``--imagenet-root`` with a default); ``None`` means the loader's
        defaults: its default root, ``<root>/devkit`` and 224.
    :raises ValueError: An option that does not apply to ``name``, an unknown mode, a
        selection split holding out zero samples, a batch larger than the train split.
    :raises FileNotFoundError: The ImageNet tree or its devkit is missing.
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
            resize, max_samples)
    if validation_split is not None:
        raise ValueError(
            f"validation_split={validation_split} does not apply to ImageNet, which selects "
            "on holdout_per_class images per class taken out of train"
        )
    if augment is not None:
        raise ValueError(
            "augment applies to in-memory datasets; ImageNet augments in its loader "
            "(imagenet_augment)"
        )
    if resize is not None:
        raise ValueError("resize applies to in-memory datasets; ImageNet takes image_size")
    if channels not in (None, 3):
        raise ValueError(f"channels={channels} is not available for 'imagenet' (3)")
    return _load_imagenet(
        spec, batch_size, seed, holdout_per_class, normalize, labels, imagenet_augment,
        imagenet_augment_color,
        max_samples, eval_samples, imagenet_root, devkit_dir, image_size)


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
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            f"validation_split must be strictly inside (0, 1), got {validation_split}. "
            "0 would silently validate on the test set."
        )
    n_channels = _resolve_channels(spec, channels)
    (x_train, y_train), (x_test, y_test) = _load_raw(spec.name)
    if x_train.ndim == 3:
        x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
    if n_channels != x_train.shape[-1]:
        x_train, x_test = np.repeat(x_train, n_channels, -1), np.repeat(x_test, n_channels, -1)

    train_order = np.random.default_rng(seed).permutation(len(x_train))
    test_order = np.random.default_rng([seed, 1]).permutation(len(x_test))
    if max_samples is not None:
        train_order, test_order = train_order[:max_samples], test_order[:max_samples]
    _, n_select = split_sizes(len(x_train), max_samples, validation_split)
    select_idx, train_idx = train_order[:n_select], train_order[n_select:]

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

    steps = steps_per_epoch_for(len(arrays.x_train), batch_size)
    logger.info(
        f"{spec.name}: train {arrays.x_train.shape}, select {arrays.x_select.shape}, "
        f"test {arrays.x_test.shape}; {normalize}, labels {labels}"
    )
    split = (f"validation_split={DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split}"
             " of the train set (seeded permutation)")
    return ClassificationData(
        train_ds=_train_pipeline(arrays.x_train, arrays.y_train, batch_size, seed, augment,
                                 norm, _resizer(resize)),
        select_ds=_eval_pipeline(arrays.x_select, arrays.y_select, norm, _resizer(resize)),
        test_ds=_eval_pipeline(arrays.x_test, arrays.y_test, norm, _resizer(resize)),
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


def _load_imagenet(
        spec: DatasetSpec, batch_size: int, seed: int, holdout_per_class: Optional[int],
        normalize: str, labels: str, augment: bool, augment_color: bool,
        max_samples: Optional[int], eval_samples: int, root: Optional[str], devkit_dir: Optional[str],
        image_size: Optional[int],
) -> ClassificationData:
    root = IMAGENET2012_DEFAULT_ROOT if root is None else root
    holdout = DEFAULT_HOLDOUT_PER_CLASS if holdout_per_class is None else holdout_per_class
    size = spec.input_shape[0] if image_size is None else image_size
    require_imagenet_root(root)
    train, select, test = load_imagenet_selection_splits(
        root, devkit_dir, size, batch_size, seed, augment=augment,
        augment_color=augment_color, holdout_per_class=holdout,
        normalize="imagenet" if normalize == "standardize" else "unit",
        num_classes=spec.num_classes, max_images=max_samples,
    )
    if eval_samples == 0:
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
        test_ds=encode(test.dataset),
        steps_per_epoch=steps_per_epoch_for(train.num_samples, batch_size),
        n_train=train.num_samples, n_select=select.num_samples, n_test=test.num_samples,
        x_fig=x_fig, y_fig=y_fig, arrays=None, mean=mean, std=std,
        input_shape=(size, size, 3), num_classes=train.num_classes,
        class_names=list(train.class_names),
        selection_split=f"{holdout} images per class held out of train "
                        f"({select.num_samples} images)",
        test_split="the official val split",
    )


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
