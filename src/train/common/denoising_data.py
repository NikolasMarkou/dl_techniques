"""Image data for the denoising trainers: listed folders, a finite patch epoch and stateless noise.

The ``image_denoising`` task of ``train.api`` trains on clean ``[0, 1]`` patches cut from image files
and adds the noise itself. This module takes PLAIN VALUES (the task reads its config and passes the
numbers; no config class is imported here), lists the folders with ``os`` only, and builds
the training and validation pipelines. The legacy pipeline (``train.bfunet.common.create_dataset``) is
an endless, stateful stream whose noise draws from ``tf.random``; this one is a finite epoch of a
known length whose every random number is a function of a seed.

Interface contract
------------------
``list_images(directory, key)``
    :param directory: a directory, searched recursively (hidden entries skipped, sub-directories
        that are symbolic links not entered).
    :param key: the config key to name in a refusal (``"train_dirs"``).
    :return: the image files (``png``, ``jpg``, ``jpeg``, ``bmp``, any case), sorted by directory
        and name, so the same tree always lists the same way.
    :raises DataSourceError: for a directory that does not exist, holds no image, cannot be listed,
        or holds a file name that is not UTF-8 or a symbolic link that leads nowhere. No pixel is read.

``list_sources(train_dirs, val_dirs)``
    :return: ``(train_pools, val_pools)``: one file list per directory.
    :raises DataSourceError: as above, and for a file (compared by resolved path) in both a train
        and a validation directory (``key`` ``val_dirs``).

``list_test_sets(named_dirs, train_pools=(), val_pools=())``
    :param named_dirs: ``[(dataset name, directory), ...]``, the held-out sets of ``data.test_dirs``.
    :param train_pools, val_pools: the file lists of :func:`list_sources`; when given, a held-out file (compared by
        resolved path) that is also a train or a validation file is refused.
    :return: ``{name: image files}`` in the order given, each listed as :func:`list_images` does (``key``
        ``"test_dirs"``). The task passes the pools in ``describe_data``, before the run directory is claimed, and passes
        none from ``build_data`` (the same listing a moment later; the check has been made).
    :raises DataSourceError: as :func:`list_images`, and for the shared file (``key`` ``test_dirs``).

``select_train_files(pools, max_files, seed)``
    The files an epoch is made of. With ``max_files`` the directories weigh EQUALLY: each gets
    ``max_files // len(pools)`` files (the first ``max_files % len(pools)`` one more), drawn from its
    own seeded permutation, and a directory with fewer files than its quota WRAPS (a file repeats).
    This is what stops COCO (118,287 files) from drowning DIV2K (800). With ``max_files=None``
    every file is used once and the directories weigh by their size.
    :return: ``[(directory index, path), ...]`` in directory order.
    :raises ValueError: for ``max_files`` below the number of directories (a directory would
        contribute nothing).

``select_val_files(pools, max_files, val_seed)``
    All files of all validation directories, or ``max_files`` of them chosen by a permutation seeded
    by ``val_seed`` alone (so the validation set never depends on the run's ``seed``), kept in listing order.

``scan_images(files, *, channels, key)``
    Decodes every file once, in parallel. :return: ``{path: has_signal}``, ``has_signal`` False for a
        flat BLACK image (every value 0), which the legacy pipeline also dropped (it is degenerate: the
        flat value that maps to 0).
    :raises DataSourceError: naming the first file that cannot be decoded.

``NoiseSpec``
    The noise a run adds, plain values: ``kind`` (one of :data:`NOISE_KINDS`), ``sigma_min``,
    ``sigma_max`` (the last value of the curriculum and the upper bound of the validation sigma),
    ``sigma_max_start`` (the value at epoch 0), ``composite_ratio`` and ``clip``. No default: the
    config is the one home of the defaults.

``draw_sigma(seed, sigma_min, sigma_max)``
    :return: a scalar ``float32``, uniform in ``[sigma_min, sigma_max]``, a pure function of ``seed``
        (an ``int`` pair).

``add_noise(clean, sigma, seed, *, kind, composite_ratio, clip)``
    A PURE function of ``(clean, sigma, seed)``: the same arguments give the same bytes, on any
    device, in any order. ``clean`` is a ``float32`` image in ``[0, 1]``; ``sigma`` is on the same
    scale (a width: ``sigma_255 = 255 * sigma``). ``y | x`` is ``N(x, sigma^2)`` for ``additive``,
    ``N(x, sigma^2 x^2)`` for ``multiplicative`` (``y = x (1 + n sigma)``) and
    ``N(x, sigma^2 x^2 + (composite_ratio sigma)^2)`` for ``composite``. ``sigma == 0`` returns
    ``clean``. The result is clipped to ``[0, 1]`` only when ``clip``.

``make_sigma_max_var(noise)``
    :return: the ``tf.Variable`` the training noise reads as the upper bound of its sigma, at its epoch-0 value.
        ``dl_techniques.callbacks.noise_sigma_curriculum.NoiseSigmaCurriculumCallback`` assigns it at the
        start of every epoch.

``load_denoising_data(train_dirs, val_dirs, *, patch_size, channels, patches_per_image, batch_size, seed,
noise, val_seed, max_train_files, max_val_files, max_samples, fingerprint, augment, eval_batch_size)``
    :return: the :class:`DenoisingBatches` of the folders. Every file the run uses is decoded once, BEFORE
        any pipeline is built (:func:`scan_images`); a flat black image leaves the lists (logged, not
        replaced), so an epoch is exactly ``files * patches_per_image`` patches.
    :raises DataSourceError: for a source that cannot be used (naming the key and the file).
    :raises ValueError: for a number out of range.

``PATCH_AUGMENTS`` / ``build_patch_augment(preset)``
    ``("none", "flip", "dihedral")`` and the seeded ``(seed, patch) -> patch`` hook of one of them (``None`` for
    ``"none"``), built from ``train.common.image_augment``: a random horizontal flip, or flips and quarter turns (a
    patch is square, so a turn never changes its shape). The hook is the ``augment`` argument of
    :func:`load_denoising_data`; the validation set is never augmented. :raises ValueError: for another name.

``denoising_fingerprint(selection, mode)``
    A sha256 over ``(split, directory index, path relative to its directory, size)`` of every used file in
    listing order, or with ``mode="content"`` also every file's bytes (streamed); the scheme is
    ``classification_data.FINGERPRINT_SCHEME``.

Pipeline
--------
TRAINING. The used files are shuffled every epoch; each file is decoded ONCE (upscaled, as legacy, when
a side is below ``patch_size``) and ``patches_per_image`` random crops are cut from the decode; the
crops are shuffled across images (a buffer of patches), noised and batched with the remainder dropped.
The epoch is ``files * patches_per_image`` patches, ``// batch_size`` steps, and
``tf.data.experimental.assert_cardinality`` states it. Every random number comes from an element seed
of ``dl_techniques.datasets.vision.augment_hook.seed_stream``: fixed by ``seed``, new every epoch. Two
seeds give two noise streams; one seed gives one.

The sigma of a patch is uniform in ``[noise.sigma_min, sigma_max_var]`` where ``sigma_max_var`` is the
variable the curriculum callback assigns (:func:`make_sigma_max_var`); the iterator of an epoch is
created after the callback has assigned that epoch's value, which ``test_common_denoising_data``
proves with a real callback on a real ``fit``.

VALIDATION. The files are decoded in order, ``patches_per_image`` crops each, the crop of patch ``i`` cut with a
seed that is a function of ``(val_seed, i)``, the sigma uniform in ``[noise.sigma_min, noise.sigma_max]`` and the
noise a function of ``(val_seed, i)`` too. Nothing in it depends on ``seed``, on the epoch or on the order of
the workers, so a model's ``val_psnr`` is a function of its weights alone.

Import rule
-----------
This module imports TensorFlow and is imported inside functions by ``train.api``; nothing here runs a
TensorFlow op until a pipeline or a noise is built (the listing is ``os`` only).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.augment_hook import seed_stream
from dl_techniques.utils.logger import logger
from train.common.classification_data import (
    FINGERPRINT_MODES,
    FINGERPRINT_SCHEME,
    IMAGE_SUFFIXES,
    DataSourceError,
    _examples,
    _link_state,
    _name_undecodable,
    _resolver,
    decode_image_file,
)

#: The noise kinds, in the order of the config's ``noise_kind`` choices (``train.api.denoising_args``
#: holds the same tuple and a test holds the two equal).
NOISE_KINDS = ("additive", "multiplicative", "composite")

#: ``noise_composite_ratio`` when a composite run gives none: the legacy ratio of the additive part to the
#: multiplicative one (``sigma_a = ratio * sigma_m``).
DEFAULT_COMPOSITE_RATIO = 0.5

#: Patches held by the training shuffle (the legacy ``patch_shuffle_buffer``): a batch should mix images.
DEFAULT_PATCH_SHUFFLE_BUFFER = 2048

#: Files decoded per batch of the pre-fit scan.
SCAN_BATCH = 256

#: ``(index of the directory in its list, file path)``.
Entry = Tuple[int, str]

#: A seeded hook ``(seed, patch) -> patch`` applied to a clean training patch (a flip, a quarter turn).
PatchAugment = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

#: The training-patch augmentations a run may ask for: none, a random horizontal flip, or the dihedral group (flips and
#: quarter turns; a patch is square, so a turn never changes its shape). ``train.api.denoising_args`` holds the same
#: tuple and a test holds the two equal.
PATCH_AUGMENTS = ("none", "flip", "dihedral")


def build_patch_augment(preset: str) -> Optional[PatchAugment]:
    """The seeded ``(seed, patch) -> patch`` hook of ``preset`` (one of :data:`PATCH_AUGMENTS`), or ``None`` for ``"none"``.

    :raises ValueError: for a name that is no preset.
    """
    from train.common.image_augment import random_dihedral, random_flip

    if preset not in PATCH_AUGMENTS:
        raise ValueError(f"Unknown patch augmentation {preset!r}; known: {list(PATCH_AUGMENTS)}")
    return {"none": None, "flip": random_flip(), "dihedral": random_dihedral()}[preset]


# ---------------------------------------------------------------------
# Listing: os only
# ---------------------------------------------------------------------

def _raise(error: OSError) -> None:
    raise error


def list_images(directory: str, key: str) -> List[str]:
    """The image files below ``directory``; see the module contract."""
    if not os.path.isdir(directory):
        raise DataSourceError(key, f"{directory!r} is not a directory")
    files: List[str] = []
    problems: List[str] = []
    try:
        for folder, subdirs, names in os.walk(directory, onerror=_raise):
            subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
            for name in sorted(names):
                if name.startswith(".") or not name.lower().endswith(IMAGE_SUFFIXES):
                    continue
                path = os.path.join(folder, name)
                try:
                    name.encode("utf-8")
                except UnicodeEncodeError:
                    problems.append(f"{name!r} is not a valid UTF-8 file name")
                    continue
                if os.path.islink(path):
                    state = _link_state(path)
                    if state is not None:
                        problems.append(f"{name!r} is a symbolic link that leads nowhere ({state})")
                        continue
                files.append(path)
    except OSError as exc:
        # A partial listing must never become a training set: the walk ignores its own errors unless told to.
        raise DataSourceError(
            key, f"{directory!r} cannot be listed ({exc.strerror or exc}: {exc.filename})") from exc
    if problems:
        raise DataSourceError(
            key, f"{directory!r}: {_examples(problems, len(problems))}; rename the file to UTF-8, or "
                 "repair or remove the link (a file the loader would skip or fail on is named instead of dropped)")
    if not files:
        raise DataSourceError(key, f"{directory!r} holds no images ({', '.join(IMAGE_SUFFIXES)})")
    return files


def list_sources(train_dirs: Sequence[str], val_dirs: Sequence[str]) -> Tuple[List[List[str]], List[List[str]]]:
    """The file list of every train and validation directory, refusing a file that is in both."""
    train = [list_images(directory, "train_dirs") for directory in train_dirs]
    val = [list_images(directory, "val_dirs") for directory in val_dirs]
    resolve = _resolver()
    trained = {resolve(path) for pool in train for path in pool}
    shared = sorted({resolve(path) for pool in val for path in pool} & trained)
    if shared:
        raise DataSourceError(
            "val_dirs", f"{len(shared)} file(s) are in both data.train_dirs and data.val_dirs "
                        f"({_examples([repr(path) for path in shared], len(shared))}); a validation image "
                        "the model trained on says nothing about it")
    return train, val


def list_test_sets(
        named_dirs: Sequence[Tuple[str, str]], train_pools: Sequence[Sequence[str]] = (),
        val_pools: Sequence[Sequence[str]] = (),
) -> "dict[str, List[str]]":
    """The files of every held-out set, refusing one that is also a train or validation file."""
    sets = {name: list_images(directory, "test_dirs") for name, directory in named_dirs}
    if train_pools or val_pools:
        resolve = _resolver()
        used = {resolve(path) for pool in (*train_pools, *val_pools) for path in pool}
        for name, files in sets.items():
            shared = sorted({resolve(path) for path in files} & used)
            if shared:
                raise DataSourceError(
                    "test_dirs", f"held-out set {name!r} has {len(shared)} file(s) that are also in data.train_dirs or "
                                 f"data.val_dirs ({_examples([repr(path) for path in shared], len(shared))}); a held-out "
                                 "image the model trained or selected on says nothing about it")
    return sets


def select_train_files(pools: Sequence[Sequence[str]], max_files: Optional[int], seed: int) -> List[Entry]:
    """The files of an epoch, directories weighing equally under a cap; see the module contract."""
    if max_files is None:
        return [(index, path) for index, pool in enumerate(pools) for path in pool]
    count = len(pools)
    if max_files < count:
        raise ValueError(
            f"max_train_files {max_files} is below the {count} train directories: a directory would "
            "contribute no file")
    chosen: List[Entry] = []
    for index, pool in enumerate(pools):
        quota = max_files // count + (1 if index < max_files % count else 0)
        order = np.random.default_rng([seed, index]).permutation(len(pool))
        chosen.extend((index, pool[order[j % len(pool)]]) for j in range(quota))
    return chosen


def select_val_files(pools: Sequence[Sequence[str]], max_files: Optional[int], val_seed: int) -> List[Entry]:
    """The validation files, a seeded subset under a cap; see the module contract."""
    every = [(index, path) for index, pool in enumerate(pools) for path in pool]
    if max_files is None or max_files >= len(every):
        return every
    keep = sorted(np.random.default_rng([val_seed]).permutation(len(every))[:max_files].tolist())
    return [every[position] for position in keep]


# ---------------------------------------------------------------------
# The pre-fit scan
# ---------------------------------------------------------------------

def scan_images(files: Sequence[str], *, channels: int, key: str) -> "dict[str, bool]":
    """Decode every file once; ``{path: False}`` for a flat black image. See the module contract."""
    unique = list(dict.fromkeys(files))
    if not unique:
        return {}
    started = time.perf_counter()
    dataset = (
        tf.data.Dataset.from_tensor_slices(unique)
        .map(lambda path: tf.reduce_any(decode_image_file(path, channels) > 0.0),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(SCAN_BATCH)
    )
    try:
        flags = np.concatenate([np.asarray(batch) for batch in dataset])
    except tf.errors.InvalidArgumentError as exc:
        _name_undecodable(unique, key, exc)  # raises DataSourceError naming the file, when one is found
        raise
    logger.info(f"decoded {len(unique)} image files once before fit in {time.perf_counter() - started:.1f} s")
    return dict(zip(unique, flags.tolist()))


def _without_flat(entries: Sequence[Entry], signal: "dict[str, bool]", key: str) -> List[Entry]:
    """Drop the entries of a flat black image; refuse a directory left with nothing."""
    kept = [(index, path) for index, path in entries if signal[path]]
    dropped = sorted({path for _, path in entries if not signal[path]})
    if dropped:
        logger.warning(
            f"{len(dropped)} flat black image(s) left the {key} list (every value 0, as the legacy pipeline "
            f"dropped them; they are not replaced, so the epoch is shorter): {_examples(dropped, len(dropped))}")
    for index in sorted({index for index, _ in entries}):
        if not any(i == index for i, _ in kept):
            raise DataSourceError(key, f"directory number {index} of {key} holds no image with a value above 0")
    return kept


# ---------------------------------------------------------------------
# Noise: pure functions of a seed
# ---------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class NoiseSpec:
    """The noise of a run as plain values; see the module contract. Every field is required."""

    kind: str
    sigma_min: float
    sigma_max: float
    sigma_max_start: float
    composite_ratio: float
    clip: bool

    def __post_init__(self) -> None:
        if self.kind not in NOISE_KINDS:
            raise ValueError(f"noise kind must be one of {NOISE_KINDS}, got {self.kind!r}")
        if not 0 <= self.sigma_min < self.sigma_max:
            raise ValueError(
                f"expected 0 <= sigma_min < sigma_max, got {self.sigma_min!r} and {self.sigma_max!r}")
        if not self.sigma_min <= self.sigma_max_start:
            raise ValueError(
                f"expected sigma_min <= sigma_max_start, got {self.sigma_min!r} and {self.sigma_max_start!r}")
        if self.composite_ratio < 0:
            raise ValueError(f"composite_ratio must be at least 0, got {self.composite_ratio!r}")


def make_sigma_max_var(noise: NoiseSpec) -> tf.Variable:
    """The variable the training noise reads as its sigma upper bound; see the module contract."""
    return tf.Variable(noise.sigma_max_start, dtype=tf.float32, trainable=False, name="sigma_max")


def draw_sigma(seed: tf.Tensor, sigma_min: tf.Tensor, sigma_max: tf.Tensor) -> tf.Tensor:
    """A scalar uniform in ``[sigma_min, sigma_max]``, a pure function of ``seed``."""
    low = tf.cast(sigma_min, tf.float32)
    high = tf.cast(sigma_max, tf.float32)
    return low + (high - low) * tf.random.stateless_uniform([], seed=seed, dtype=tf.float32)


def add_noise(
        clean: tf.Tensor, sigma: tf.Tensor, seed: tf.Tensor, *, kind: str, composite_ratio: float, clip: bool,
) -> tf.Tensor:
    """The noisy copy of ``clean``: a pure function of ``(clean, sigma, seed)``; see the module contract."""
    # DECISION plan-2026-11-10T095307-d483ffab/D-008
    # All three kinds are STATELESS formulas keyed by the element seed. Do NOT call
    # `dl_techniques.utils.multiplicative_miyasawa.apply_*_gaussian` or `tf.random.normal` here: those draw
    # from the op-level global stream, so two builds of one seed disagree, a validation pass is not repeatable
    # and `val_psnr` stops being a function of the weights. The legacy draw order is NOT kept (a different
    # stream); the distribution is, and a test compares the two. The composite additive part reuses the
    # second half of the split seed so it is independent of the multiplicative draw.
    if kind not in NOISE_KINDS:
        raise ValueError(f"noise kind must be one of {NOISE_KINDS}, got {kind!r}")
    sigma = tf.cast(sigma, clean.dtype)
    first, second = tf.unstack(tf.random.experimental.stateless_split(seed, 2))
    shape = tf.shape(clean)
    gauss = tf.random.stateless_normal(shape, seed=first, dtype=clean.dtype)
    if kind == "additive":
        noisy = clean + gauss * sigma
    elif kind == "multiplicative":
        noisy = clean * (1.0 + gauss * sigma)
    else:
        floor = tf.random.stateless_normal(shape, seed=second, dtype=clean.dtype)
        noisy = clean * (1.0 + gauss * sigma) + floor * (tf.cast(composite_ratio, clean.dtype) * sigma)
    # DECISION plan-2026-11-10T095307-d483ffab/D-009
    # The clip is applied ONLY when asked (`noise.clip`, the config's `data.noise_clip`, default false). The
    # legacy trainer always clipped, which makes the noise non-Gaussian near 0 and 1 (0.09 to 8.85 percent
    # of a Kodak crop's pixels change at sigma_255 15 to 63.75) so the Tweedie identity the bias-free model is
    # used for no longer holds. Do NOT clip here by default to match legacy numbers: the parity arm sets
    # `noise_clip=true` by name.
    return tf.clip_by_value(noisy, 0.0, 1.0) if clip else noisy


def _parts(patch_seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """The crop, augmentation, sigma and noise seeds of one patch, all functions of its seed."""
    parts = tf.random.experimental.stateless_split(patch_seed, 4)
    return parts[0], parts[1], parts[2], parts[3]


# ---------------------------------------------------------------------
# The patches
# ---------------------------------------------------------------------

def _decode_for_crops(path: tf.Tensor, patch_size: int, channels: int) -> tf.Tensor:
    """One decode: float32 ``(H, W, C)`` in [0, 1], upscaled (aspect kept) so both sides reach ``patch_size``."""
    image = decode_image_file(path, channels)
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    def upscale() -> tf.Tensor:
        scale = tf.cast(patch_size, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)
        new_height = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale), tf.int32)
        new_width = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale), tf.int32)
        return tf.image.resize(image, [new_height, new_width])

    image = tf.cond(tf.logical_or(height < patch_size, width < patch_size), upscale, lambda: image)
    image.set_shape([None, None, channels])
    return image


def _crops(path: tf.Tensor, patch_seeds: tf.Tensor, patch_size: int, channels: int, count: int,
           ) -> Tuple[tf.Tensor, tf.Tensor]:
    """``count`` crops of ONE decode of ``path``: ``(crops (count, P, P, C), their patch seeds)``.

    The clean crop is clipped to [0, 1] (the upscale can overshoot, and the crop is the target).
    """
    image = _decode_for_crops(path, patch_size, channels)
    crops = []
    for k in range(count):
        crop = tf.image.stateless_random_crop(image, [patch_size, patch_size, channels], seed=_parts(patch_seeds[k])[0])
        crops.append(tf.clip_by_value(crop, 0.0, 1.0))
    return tf.stack(crops), patch_seeds


def _noisy_pair(
        clean: tf.Tensor, patch_seed: tf.Tensor, sigma_low: tf.Tensor, sigma_high: tf.Tensor, noise: NoiseSpec,
        augment: Optional[PatchAugment],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """``(noisy, clean)`` of one patch; ``clean`` has been through ``augment`` when there is one."""
    _, augment_seed, sigma_seed, noise_seed = _parts(patch_seed)
    if augment is not None:
        clean = augment(augment_seed, clean)
    sigma = draw_sigma(sigma_seed, sigma_low, sigma_high)
    noisy = add_noise(clean, sigma, noise_seed, kind=noise.kind, composite_ratio=noise.composite_ratio, clip=noise.clip)
    return noisy, clean


def _train_pipeline(
        files: Sequence[str], *, patch_size: int, channels: int, patches_per_image: int, seed: int,
        noise: NoiseSpec, sigma_max_var: tf.Variable, augment: Optional[PatchAugment], shuffle_buffer: int,
) -> "tf.data.Dataset":
    """The unbatched ``(noisy, clean)`` training examples of one epoch, of stated length."""
    paths = tf.data.Dataset.from_tensor_slices(list(files)).shuffle(
        len(files), seed=seed, reshuffle_each_iteration=True)
    seeded = tf.data.Dataset.zip((paths, seed_stream(seed)))

    def per_file(path: tf.Tensor, file_seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        seeds = tf.random.experimental.stateless_split(file_seed, patches_per_image)
        return _crops(path, seeds, patch_size, channels, patches_per_image)

    patches = seeded.map(per_file, num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    patches = patches.shuffle(min(shuffle_buffer, len(files) * patches_per_image), seed=seed + 1,
                              reshuffle_each_iteration=True)

    def noised(clean: tf.Tensor, patch_seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # The variable is read HERE, when the element is produced: the iterator of an epoch is created after
        # the curriculum callback assigned that epoch's value (measured, `test_common_denoising_data`).
        return _noisy_pair(clean, patch_seed, noise.sigma_min, sigma_max_var, noise, augment)

    return patches.map(noised, num_parallel_calls=tf.data.AUTOTUNE)


def _val_pipeline(
        files: Sequence[str], *, patch_size: int, channels: int, patches_per_image: int, val_seed: int,
        noise: NoiseSpec,
) -> "tf.data.Dataset":
    """The unbatched ``(noisy, clean)`` validation examples: fixed crops, fixed noise, listing order."""
    indices = tf.data.Dataset.range(len(files))

    def per_file(path: tf.Tensor, file_index: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        first = file_index * patches_per_image
        seeds = tf.stack([tf.stack([tf.constant(val_seed, tf.int64), first + k]) for k in range(patches_per_image)])
        return _crops(path, seeds, patch_size, channels, patches_per_image)

    patches = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(list(files)), indices)).map(
        per_file, num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    return patches.map(
        lambda clean, patch_seed: _noisy_pair(clean, patch_seed, noise.sigma_min, noise.sigma_max, noise, None),
        num_parallel_calls=tf.data.AUTOTUNE)


def _batched(dataset: "tf.data.Dataset", count: int, batch_size: int, *, drop_remainder: bool) -> "tf.data.Dataset":
    """State the length of ``dataset`` (``count`` examples), batch it and prefetch."""
    # The length is ASSERTED, not left to the maps: `unbatch` loses it, and a dataset of unknown length makes
    # Keras log a phantom zero batch (loss n/(n+1), which hid a model that learned nothing in an earlier plan,
    # `segmentation_data._batched`). Do NOT remove this because "the maps keep the count": the test that reads
    # `cardinality()` and the count a real `fit` takes is what keeps it true. The count is
    # `files * patches_per_image` because no example is filtered out (a flat black image left the list before
    # this point).
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(count))
    return dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------

_HASH_CHUNK = 1 << 20


@dataclasses.dataclass(frozen=True)
class FileSelection:
    """The directories and the used files of a run, in listing order (what the fingerprint hashes)."""

    train_dirs: Tuple[str, ...]
    val_dirs: Tuple[str, ...]
    train: Tuple[Entry, ...]
    val: Tuple[Entry, ...]


def denoising_fingerprint(selection: FileSelection, mode: str = "listing") -> str:
    """A sha256 over the used files in listing order; see the module contract."""
    if mode not in FINGERPRINT_MODES:
        raise ValueError(f"fingerprint mode must be one of {FINGERPRINT_MODES}, got {mode!r}")
    rows = []
    for split, bases, entries in (("train", selection.train_dirs, selection.train),
                                  ("val", selection.val_dirs, selection.val)):
        for index, path in entries:
            rows.append((split, index, os.path.relpath(path, bases[index]).replace(os.sep, "/"),
                         os.path.getsize(path), path))
    digest = hashlib.sha256(b"denoising-folders")
    digest.update(b"\x00" + FINGERPRINT_SCHEME.encode("utf-8"))
    if mode == "content":
        digest.update(b"\x00content")
    started = time.perf_counter()
    total = 0
    for row in rows:
        digest.update(b"\n" + json.dumps(list(row[:4]), ensure_ascii=False).encode("utf-8"))
        if mode == "content":
            digest.update(b"\n")
            with open(row[4], "rb") as handle:
                for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                    digest.update(chunk)
                    total += len(chunk)
    if mode == "content":
        logger.info(f"denoising folders: content fingerprint of {len(rows)} files, {total / 1e6:.1f} MB, "
                    f"in {time.perf_counter() - started:.2f} s")
    return digest.hexdigest()


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DenoisingBatches:
    """What a denoising trainer consumes.

    :param train_ds: batched ``(noisy, clean)`` pipeline of ONE epoch (finite, remainder dropped).
    :param select_ds: the fixed validation pipeline, batched, remainder kept; the selection split.
    :param test_ds: always ``None`` (held-out images are scored by the task from their own directories).
    :param steps_per_epoch: batches in one epoch of ``train_ds``.
    :param n_train: training patches in an epoch (``train_files * patches_per_image``).
    :param n_select: validation patches. :param n_test: 0.
    :param train_files, val_files: the used files in listing order (a wrapped file repeats), for naming the
        file behind a decode error.
    :param sigma_max_var: the variable the curriculum callback assigns (see :func:`make_sigma_max_var`).
    :param input_shape: ``(patch_size, patch_size, channels)``.
    :param num_classes: ``None``: a denoiser has no classes.
    :param selection_split, test_split: plain-language provenance of the two splits.
    :param fingerprint: :func:`denoising_fingerprint`.
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: Optional["tf.data.Dataset"]
    steps_per_epoch: int
    n_train: int
    n_select: int
    n_test: int
    train_files: Tuple[str, ...]
    val_files: Tuple[str, ...]
    sigma_max_var: tf.Variable
    input_shape: Tuple[int, int, int]
    num_classes: Optional[int]
    selection_split: str
    test_split: str
    fingerprint: str


def _cap(first: Optional[int], second: Optional[int]) -> Optional[int]:
    """The smaller of two optional caps (``None`` is no cap)."""
    values = [value for value in (first, second) if value is not None]
    return min(values) if values else None


def load_denoising_data(
        train_dirs: Sequence[str], val_dirs: Sequence[str], *, patch_size: int, channels: int,
        patches_per_image: int, batch_size: int, seed: int, noise: NoiseSpec, val_seed: int,
        max_train_files: Optional[int], max_val_files: Optional[int], max_samples: Optional[int],
        fingerprint: str = "listing", augment: Optional[PatchAugment] = None,
        eval_batch_size: Optional[int] = None, patch_shuffle_buffer: int = DEFAULT_PATCH_SHUFFLE_BUFFER,
) -> DenoisingBatches:
    """The record of the train and validation folders; see the module contract."""
    for name, value in (("patch_size", patch_size), ("channels", channels),
                        ("patches_per_image", patches_per_image), ("batch_size", batch_size)):
        if not value >= 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels!r}")
    train_pools, val_pools = list_sources(train_dirs, val_dirs)
    train = select_train_files(train_pools, _cap(max_train_files, max_samples), seed)
    val = select_val_files(val_pools, _cap(max_val_files, max_samples), val_seed)
    train = _without_flat(train, scan_images([path for _, path in train], channels=channels, key="train_dirs"),
                          "train_dirs")
    val = _without_flat(val, scan_images([path for _, path in val], channels=channels, key="val_dirs"), "val_dirs")
    n_train = len(train) * patches_per_image
    if n_train < batch_size:
        raise DataSourceError(
            "train_dirs", f"batch_size {batch_size} exceeds the {n_train} training patches an epoch holds "
                          f"({len(train)} files x {patches_per_image} patches_per_image): the train pipeline drops the "
                          "incomplete last batch and would yield no step; use a smaller batch_size or more files")
    n_select = len(val) * patches_per_image
    sigma_max_var = make_sigma_max_var(noise)
    train_files = tuple(path for _, path in train)
    val_files = tuple(path for _, path in val)
    selection = FileSelection(tuple(train_dirs), tuple(val_dirs), tuple(train), tuple(val))
    train_ds = _batched(
        _train_pipeline(train_files, patch_size=patch_size, channels=channels, patches_per_image=patches_per_image,
                        seed=seed, noise=noise, sigma_max_var=sigma_max_var, augment=augment,
                        shuffle_buffer=patch_shuffle_buffer),
        n_train, batch_size, drop_remainder=True)
    select_ds = _batched(
        _val_pipeline(val_files, patch_size=patch_size, channels=channels, patches_per_image=patches_per_image,
                      val_seed=val_seed, noise=noise),
        n_select, batch_size if eval_batch_size is None else eval_batch_size, drop_remainder=False)
    logger.info(
        f"denoising data: {len(train)} train files x {patches_per_image} patches = {n_train} patches "
        f"({n_train // batch_size} steps of {batch_size}), {len(val)} validation files x {patches_per_image} = "
        f"{n_select} fixed patches, noise {noise.kind}")
    return DenoisingBatches(
        train_ds=train_ds, select_ds=select_ds, test_ds=None, steps_per_epoch=n_train // batch_size,
        n_train=n_train, n_select=n_select, n_test=0, train_files=train_files, val_files=val_files,
        sigma_max_var=sigma_max_var, input_shape=(patch_size, patch_size, channels), num_classes=None,
        selection_split=(f"{len(val)} files of the validation directories, {patches_per_image} fixed crops each, "
                         f"noise keyed on (val_seed={val_seed}, patch index); never part of the training pool"),
        test_split="none: held-out images are scored by the task from their own directories",
        fingerprint=denoising_fingerprint(selection, fingerprint))
