"""The MegaDepth data of the ``depth_estimation`` task: a stateless train stream and fixed evaluation crops.

Every sample is a pure function of a key, so a run is reproducible from its seed and a resume never replays
its first batches:

* the scenes come from :func:`train.common.megadepth.split_megadepth_scenes` (the legacy scene-disjoint rule);
  its validation scenes, sorted by name, are dealt alternately into SELECTION scenes (positions 0, 2, ...) and
  TEST scenes (positions 1, 3, ...), so the three groups are pairwise disjoint;
* a selection or test crop is the crop of pool file ``j`` under ``default_rng([seed, salt, j])``, the pool
  being ``POOL_FACTOR * n`` files taken evenly across the group's scenes; the crops are loaded once, into memory,
  and never depend on the stream seed; the read-only arrays ``x_select``, ``y_select``, ``x_test`` and ``y_test``
  are their only copy, and ``select_ds`` and ``test_ds`` read batches from them in place (a ``from_tensor_slices``
  stream held a second copy of every crop, about 465 MiB at the defaults);
* train sample ``counter`` draws its file (uniformly from the train list), its crop and its flip from
  ``default_rng([stream_seed, TRAIN_SALT, counter])``, trying up to ``MAX_TRAIN_TRIES`` files; batch ``i`` of the
  stream is the samples of counters ``i * batch .. i * batch + batch - 1``;
* the average layout is the mask-weighted per-pixel mean of ``layout_samples`` train targets keyed on
  ``(seed, LAYOUT_SALT, i)``.

Each sample is ``(rgb, y)``: ``rgb`` ImageNet-standardized (:func:`~train.common.megadepth.standardize_imagenet`),
``y = [disparity, mask]`` on the last axis (:func:`~train.common.megadepth.load_and_process_pair`). A horizontal
flip is the only augmentation, and only on train. A crop with too few valid pixels or a constant valid depth (an
ordinal map) is never a sample.

Interface contract
------------------
``load_depth_batches(root, *, seed, batch_size, image_size, steps_per_epoch, ...) -> DepthBatches``
    :raises ValueError: naming the cause, for fewer than ``MIN_VALIDATION_SCENES`` validation scenes, a
        ``batch_size`` above ``n_select`` or ``n_test``, a group whose pool has fewer qualifying files than the
        crops asked for, or a count that is not a positive int.
``DepthStream.sample(counter) -> (rgb, y, tried)``
    :raises RuntimeError: naming the counter and the stream seed when none of ``MAX_TRAIN_TRIES`` files gives a
        qualifying crop.

This module imports Keras and TensorFlow; ``train.api`` imports it inside functions only.
"""

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger
from train.common.megadepth import (
    _take_across_scenes,
    load_and_process_pair,
    split_megadepth_scenes,
    standardize_imagenet,
)
from train.common.pretrain_data import derive_stream_seed

# ---------------------------------------------------------------------

#: Salts of the four keyed draws. Ints: a numpy ``SeedSequence`` takes int entropy only, so the string salts of
#: ``pretrain_data`` cannot key a ``default_rng``.
TRAIN_SALT, SELECT_SALT, TEST_SALT, LAYOUT_SALT = 1, 2, 3, 4
#: A fixed set of ``n`` crops is drawn from ``POOL_FACTOR * n`` files of its group (D-014).
POOL_FACTOR = 4
# DECISION plan-2026-11-17T110935-9c7b9096/D-029: 64 tries, not 8. A draw fails with probability p ** tries,
# p the share of tries that give no crop (0.18 to 0.21 measured on the MegaDepth train list at 252 px). Do NOT
# lower it toward 8: p ** 8 is 1.1e-6 at p 0.18 and 6.6e-5 at p 0.3, so at p 0.18 a run of 240,000 samples crashes with
# odds 23 percent and 1e9 samples are expected to crash about 1,100 and 65,600 times; p ** 64 is 2.2e-48 and
# 3.4e-34 (2.2e-39 and 3.4e-25 expected crashes in 1e9 samples). A draw that qualified within 8 tries is
# unchanged (the same generator, read the same way).
#: Files a train or layout draw tries before it raises.
MAX_TRAIN_TRIES = 64
#: Validation scenes needed to deal selection and test scenes (D-005).
MIN_VALIDATION_SCENES = 3

Scene = Tuple[str, List[str], List[str]]

# ---------------------------------------------------------------------


class SceneDeal(NamedTuple):
    """The three scene groups, each a list of ``discover_megadepth_scenes`` entries in scene order."""

    train: List[Scene]
    select: List[Scene]
    test: List[Scene]


def deal_megadepth_scenes(root: str, train_split: float = 0.9) -> SceneDeal:
    """Train scenes, and the validation scenes dealt alternately into selection and test scenes.

    :raises ValueError: for fewer than :data:`MIN_VALIDATION_SCENES` validation scenes, naming the count.
    """
    # DECISION plan-2026-11-17T110935-9c7b9096/D-005: the legacy validation scenes are split into two
    # groups BY SCENE, alternately over the sorted list. Do NOT split one validation pool by file (scene
    # leakage between selection and test) and do NOT carve a third group from the train scenes (that
    # changes the legacy train set the measured numbers were made on).
    train_scenes, val_scenes = split_megadepth_scenes(root, train_split)
    if len(val_scenes) < MIN_VALIDATION_SCENES:
        raise ValueError(
            f"MegaDepth at {root}: train_split={train_split} leaves {len(val_scenes)} validation scenes, and "
            f"at least {MIN_VALIDATION_SCENES} are needed to deal selection scenes and test scenes"
        )
    ordered = sorted(val_scenes, key=lambda scene: scene[0])
    return SceneDeal(train=list(train_scenes), select=ordered[0::2], test=ordered[1::2])


def _draw(
    rgb_paths: Sequence[str],
    depth_paths: Sequence[str],
    rng: np.random.Generator,
    image_size: int,
    min_valid_ratio: float,
    augment: bool,
    what: str,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """One keyed draw: up to ``MAX_TRAIN_TRIES`` files, each chosen, cropped and flipped from ``rng``.

    Returns the raw ``[-1, 1]`` rgb, ``y`` and the indices of the files tried (the last one kept).
    """
    tried: List[int] = []
    for _ in range(MAX_TRAIN_TRIES):
        k = int(rng.integers(len(rgb_paths)))
        tried.append(k)
        out = load_and_process_pair(rgb_paths[k], depth_paths[k], image_size, min_valid_ratio=min_valid_ratio,
                                    augment=augment, rng=rng)
        if out is not None:
            return out[0], out[1], tuple(tried)
    distinct = sorted(set(tried))
    raise RuntimeError(
        f"{what}: no qualifying MegaDepth crop (enough valid pixels, depth with spread) in {MAX_TRAIN_TRIES} files "
        f"drawn ({len(distinct)} distinct of the {len(rgb_paths)} listed) at image_size {image_size} with "
        f"min_valid_ratio {min_valid_ratio}; a list with a usable share of qualifying files fails this way with "
        f"negligible odds, so most likely none of its files qualifies; files tried (up to 8 shown): "
        f"{[rgb_paths[k] for k in distinct[:8]]}"
    )


class DepthStream(keras.utils.PyDataset):
    """The infinite train stream: batch ``index`` is a pure function of ``(stream_seed, index)``.

    ``fit(steps_per_epoch=)`` hands an infinite ``PyDataset`` a global batch index that keeps counting across
    epochs, also with worker processes (measured, D-014), so epoch ``e`` trains batches
    ``e * steps .. e * steps + steps - 1``. ``workers`` above 1 are processes; 0 or 1 reads in the caller.
    """

    def __init__(
        self,
        rgb_paths: Sequence[str],
        depth_paths: Sequence[str],
        *,
        stream_seed: int,
        batch_size: int,
        image_size: int,
        min_valid_ratio: float,
        augment: bool,
        workers: int,
    ) -> None:
        workers = int(workers)
        # DECISION plan-2026-11-17T110935-9c7b9096/D-014: worker PROCESSES keyed on the global index.
        # Do NOT go back to tf.data threads (GIL bound: 9.9 to 10.8 elements/s cold at W 4 to 16 and
        # 20.0 to 22.0 warm at W 8 and 16, against 14.6 and 63.8 for 8 processes) and do NOT make the stream finite
        # (Keras shuffles a finite PyDataset's indices unless told not to, and an epoch end would restart
        # the index).
        super().__init__(workers=max(workers, 1), use_multiprocessing=workers > 1,
                         max_queue_size=max(2 * workers, 10))
        if len(rgb_paths) != len(depth_paths) or not len(rgb_paths):
            raise ValueError(
                f"DepthStream needs matched, non-empty file lists, got {len(rgb_paths)} images and "
                f"{len(depth_paths)} depth maps")
        self.rgb_paths = list(rgb_paths)
        self.depth_paths = list(depth_paths)
        self.stream_seed = int(stream_seed)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.min_valid_ratio = float(min_valid_ratio)
        self.augment = bool(augment)

    @property
    def num_batches(self) -> None:
        return None

    def sample(self, counter: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
        """Train sample ``counter``: ``(standardized rgb, y, indices of the files tried)``."""
        # DECISION plan-2026-11-17T110935-9c7b9096/D-004: the file, the crop and the flip all come from
        # ONE generator keyed on (stream seed, salt, counter). Do NOT pass rng=None (the loader then
        # draws from the global np.random, different in every worker process and every run) and do NOT
        # retry with a new key (a counter would then mean different samples in different processes).
        rng = np.random.default_rng([self.stream_seed, TRAIN_SALT, int(counter)])
        rgb, y, tried = _draw(self.rgb_paths, self.depth_paths, rng, self.image_size, self.min_valid_ratio,
                              self.augment, f"train sample counter {counter} of stream {self.stream_seed}")
        return standardize_imagenet(rgb), y, tried

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start = int(index) * self.batch_size
        xs, ys, _ = zip(*(self.sample(c) for c in range(start, start + self.batch_size)))
        return np.stack(xs), np.stack(ys)


def _fixed_crops(
    scenes: Sequence[Scene],
    n: int,
    *,
    seed: int,
    salt: int,
    image_size: int,
    min_valid_ratio: float,
    group: str,
    keep: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[str]]:
    """``n`` crops of a group: slot ``i`` starts at pool file ``i * pool // n`` (or the next unused file) and
    takes the first file whose keyed crop qualifies; once that forward scan has run past the last pool file, a
    slot takes the lowest unused pool file whose crop qualifies. Each pool file is read at most once.

    Returns the first ``keep`` of them as ``(x standardized, y, files, pool indices)``, and the pool files.
    :raises ValueError: when fewer than ``n`` pool files qualify, with the count the scan made.
    """
    pool_rgb, pool_depth = _take_across_scenes(list(scenes), POOL_FACTOR * n)
    pool = len(pool_rgb)

    def crop(j: int):
        return load_and_process_pair(pool_rgb[j], pool_depth[j], image_size, min_valid_ratio=min_valid_ratio,
                                     augment=False, rng=np.random.default_rng([seed, salt, j]))

    xs, ys, keys = [], [], []
    rejected = set()  # pool files read whose crop does not qualify
    next_free = 0  # where the forward scan goes on from; the pool size once it has run out
    lowest = 0  # every pool file below it is taken or rejected
    for i in range(n):
        out = None
        for j in range(max((i * pool) // n, next_free), pool):
            out = crop(j)
            if out is not None:
                break
            rejected.add(j)
        if out is not None:
            next_free = j + 1
        else:
            # DECISION plan-2026-11-17T110935-9c7b9096/D-030: a slot whose forward scan ran out takes the LOWEST
            # unused qualifying pool file, so its crop is still the one keyed on (seed, salt, j). Do NOT refuse here
            # (unused qualifying files earlier in the pool were skipped, and the refusal said "only 12 of 16 qualify"
            # for 4 crops) and do NOT wrap the scan to the slot's own start or redraw with a new key. A set the
            # forward scan alone fills is the set it was before this rule, crop for crop.
            next_free = pool
            taken = set(keys)
            for j in range(lowest, pool):
                if j in taken or j in rejected:
                    continue
                out = crop(j)
                if out is not None:
                    break
                rejected.add(j)
            if out is None:
                # Every pool file has now been read once and each qualifying one is taken: len(keys) is the count.
                raise ValueError(
                    f"the {group} group cannot give {n} crops: only {len(keys)} of the {pool} files of its pool "
                    f"({len(scenes)} scenes) give a qualifying crop at image_size {image_size} with min_valid_ratio "
                    f"{min_valid_ratio}; ask for fewer crops"
                )
            lowest = j + 1
        xs.append(standardize_imagenet(out[0]))
        ys.append(out[1])
        keys.append(j)
    keys = keys[:keep]
    return (np.stack(xs[:keep]).astype(np.float32), np.stack(ys[:keep]).astype(np.float32),
            [pool_rgb[j] for j in keys], keys, list(pool_rgb))


def layout_map(
    rgb_paths: Sequence[str],
    depth_paths: Sequence[str],
    n: int,
    *,
    seed: int,
    image_size: int,
    min_valid_ratio: float,
) -> np.ndarray:
    """The average layout: the mask-weighted per-pixel mean disparity of ``n`` keyed train targets.

    Target ``i`` is drawn as a train sample with no flip from ``default_rng([seed, LAYOUT_SALT, i])``. A pixel
    no target holds valid is 0. Returns float32 ``(image_size, image_size, 1)``, summed in float64.
    """
    total = np.zeros((image_size, image_size), np.float64)
    weight = np.zeros((image_size, image_size), np.float64)
    for i in range(n):
        rng = np.random.default_rng([seed, LAYOUT_SALT, i])
        _, y, _ = _draw(rgb_paths, depth_paths, rng, image_size, min_valid_ratio, False, f"layout target {i}")
        y = y.astype(np.float64)
        # DECISION plan-2026-11-17T110935-9c7b9096/D-015: the layout term of the bar is MASK-WEIGHTED.
        # Do NOT take the plain mean of the disparity channel: an invalid pixel holds 0 there, so the
        # plain mean pulls every often-invalid pixel toward "far" (measured 1.0312 against 1.0194 on the
        # selection crops of MegaDepth).
        total += y[..., 0] * y[..., 1]
        weight += y[..., 1]
    layout = np.where(weight > 0, total / np.maximum(weight, 1e-300), 0.0)
    return layout.astype(np.float32)[..., None]


@dataclass(frozen=True, eq=False)
class DepthBatches:
    """What the depth task consumes: the fields the run sequence reads, plus the keys behind them.

    :param train_ds: the infinite :class:`DepthStream` (the task passes ``steps_per_epoch`` to ``fit``).
    :param select_ds, test_ds: finite ``tf.data`` streams of ``n_select // batch_size`` and
        ``n_test // batch_size`` whole batches of the in-memory crops, in order, read from the arrays below in place.
    :param steps_per_epoch, n_train, n_select, n_test: batches in an epoch, ``steps_per_epoch * batch_size``, and
        the crops each fixed set scores (whole batches).
    :param x_select, y_select, x_test, y_test: the fixed crops (standardized rgb, ``[disparity, mask]``),
        read-only, the one copy of them in the process.
    :param select_files, test_files, select_keys, test_keys: the rgb file and the pool index of each crop.
    :param select_pool, test_pool: the rgb files of each group's pool.
    :param layout: the average layout map, ``(image_size, image_size, 1)``.
    :param train_scenes, select_scenes, test_scenes: scene names of the three groups.
    :param train_rgb, train_depth: the train file list the stream draws from.
    :param fingerprint: sha256 of the listing (paths relative to the root) and of every key of the fixed sets
        and the layout; not of the stream seed, so a resume accepts it.
    """

    train_ds: DepthStream = field(repr=False)
    select_ds: "tf.data.Dataset" = field(repr=False)
    test_ds: "tf.data.Dataset" = field(repr=False)
    steps_per_epoch: int
    n_train: int
    n_select: int
    n_test: int
    input_shape: Tuple[int, int, int]
    num_classes: Optional[int]
    selection_split: str
    test_split: str
    fingerprint: str
    seed: int
    stream_seed: int
    batch_size: int
    image_size: int
    min_valid_ratio: float
    augment: bool
    workers: int
    x_select: np.ndarray = field(repr=False)
    y_select: np.ndarray = field(repr=False)
    x_test: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    layout: np.ndarray = field(repr=False)
    select_files: List[str] = field(repr=False)
    test_files: List[str] = field(repr=False)
    select_keys: List[int] = field(repr=False)
    test_keys: List[int] = field(repr=False)
    select_pool: List[str] = field(repr=False)
    test_pool: List[str] = field(repr=False)
    train_scenes: Tuple[str, ...]
    select_scenes: Tuple[str, ...]
    test_scenes: Tuple[str, ...]
    train_rgb: List[str] = field(repr=False)
    train_depth: List[str] = field(repr=False)

    def with_stream(self, completed_steps: int) -> "DepthBatches":
        """These fixed sets on the train stream of ``derive_stream_seed(seed, completed_steps)``."""
        # DECISION plan-2026-11-17T110935-9c7b9096/D-004: only the train stream moves on a resume. Do NOT
        # rebuild the fixed sets from the new seed (the same weights would score another val_loss and the
        # bar would not be comparable across a kill), and do NOT keep the old stream (the restarted index
        # would replay the first batches).
        stream_seed = derive_stream_seed(self.seed, completed_steps)
        return dataclasses.replace(self, stream_seed=stream_seed, train_ds=self._stream(stream_seed))

    def _stream(self, stream_seed: int) -> DepthStream:
        return DepthStream(self.train_rgb, self.train_depth, stream_seed=stream_seed, batch_size=self.batch_size,
                           image_size=self.image_size, min_valid_ratio=self.min_valid_ratio, augment=self.augment,
                           workers=self.workers)


def _positive_int(name: str, value) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return int(value)


def _whole_batches(name: str, n: int, batch_size: int) -> int:
    if n < batch_size:
        raise ValueError(f"batch_size {batch_size} is above {name} {n}: the {name.split('_')[1]} set would "
                         "hold no whole batch; lower the batch or raise the count")
    return (n // batch_size) * batch_size


def _batched(x: np.ndarray, y: np.ndarray, batch_size: int) -> "tf.data.Dataset":
    """One finite pass of ``len(x) // batch_size`` whole batches of ``(x, y)``, in order, read from the arrays in place.

    The arrays are made read-only and are the only copy of the crops: batch ``i`` is the slice
    ``[i * batch_size, (i + 1) * batch_size)`` of each, read when the stream asks for it.
    """
    # DECISION plan-2026-11-17T110935-9c7b9096/D-015: the fixed crops have ONE holder, these arrays. Do NOT
    # build the stream with from_tensor_slices (or a tf.constant of the arrays): TensorFlow copies them, so
    # the 465 MiB of crops are held twice (measured on 484 MiB of crops, freed memory returned to the
    # system: 973 MiB held that way, 490 MiB this way).
    x.flags.writeable = False
    y.flags.writeable = False
    size = int(batch_size)

    def take(index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = int(index) * size
        return x[lo:lo + size], y[lo:lo + size]

    def element(index: "tf.Tensor") -> Tuple["tf.Tensor", "tf.Tensor"]:
        bx, by = tf.numpy_function(take, [index], (tf.float32, tf.float32), stateful=False)
        bx.set_shape((size,) + x.shape[1:])
        by.set_shape((size,) + y.shape[1:])
        return bx, by

    return tf.data.Dataset.range(len(x) // size).map(element)


def load_depth_batches(
    root: str,
    *,
    seed: int,
    batch_size: int,
    image_size: int,
    steps_per_epoch: int,
    n_select: int = 128,
    n_test: int = 256,
    layout_samples: int = 512,
    min_valid_ratio: float = 0.1,
    max_train_files: Optional[int] = 10_000,
    augment: bool = True,
    workers: int = 8,
    train_split: float = 0.9,
    stream_seed: Optional[int] = None,
) -> DepthBatches:
    """List MegaDepth at ``root``, load the fixed selection and test crops and the layout, and open the stream.

    :param root: the MegaDepth root (``<root>/<scene>/dense<N>/{imgs,depths}``); read in place.
    :param seed: the key of the fixed sets and the layout, and the stream seed when ``stream_seed`` is ``None``.
    :param n_select, n_test: crops asked of the two fixed sets; each keeps whole batches of them.
    :param layout_samples: train targets averaged into the layout map.
    :param max_train_files: cap on the train list, taken evenly across the train scenes; ``None`` keeps all.
    :param augment: a horizontal flip with probability 0.5 on train samples (the only augmentation).
    :param workers: train stream worker processes (0 or 1: read in the caller).
    :param train_split: share of the scenes that train (the legacy rule; 0.9 gives 176 of 195).
    :param stream_seed: the train stream's seed; ``None`` means ``seed``.
    :raises ValueError: see the module contract.
    """
    batch_size = _positive_int("batch_size", batch_size)
    image_size = _positive_int("image_size", image_size)
    steps_per_epoch = _positive_int("steps_per_epoch", steps_per_epoch)
    layout_samples = _positive_int("layout_samples", layout_samples)
    asked = {"selection": _positive_int("n_select", n_select), "test": _positive_int("n_test", n_test)}
    n_select = _whole_batches("n_select", asked["selection"], batch_size)
    n_test = _whole_batches("n_test", asked["test"], batch_size)
    stream_seed = int(seed) if stream_seed is None else int(stream_seed)

    started = time.time()
    deal = deal_megadepth_scenes(root, train_split)
    train_rgb, train_depth = _take_across_scenes(deal.train, max_train_files)
    listed = time.time()
    keys = dict(seed=int(seed), image_size=image_size, min_valid_ratio=float(min_valid_ratio))
    # A set is built from the count ASKED, then cut to whole batches, so the crops a run scores do not move
    # with its batch size (a smaller batch scores a longer prefix of the same set).
    x_select, y_select, select_files, select_keys, select_pool = _fixed_crops(
        deal.select, asked["selection"], salt=SELECT_SALT, group="selection", keep=n_select, **keys)
    x_test, y_test, test_files, test_keys, test_pool = _fixed_crops(
        deal.test, asked["test"], salt=TEST_SALT, group="test", keep=n_test, **keys)
    fixed = time.time()
    layout = layout_map(train_rgb, train_depth, layout_samples, **keys)
    logger.info(
        f"MegaDepth depth data: {len(train_rgb)} train files from {len(deal.train)} scenes, {n_select} selection "
        f"crops from {len(deal.select)} scenes, {n_test} test crops from {len(deal.test)} scenes; listing "
        f"{listed - started:.1f} s, fixed sets {fixed - listed:.1f} s, layout of {layout_samples} targets "
        f"{time.time() - fixed:.1f} s")

    def relative(paths: Sequence[str]) -> List[str]:
        return [os.path.relpath(p, root) for p in paths]

    fingerprint = hashlib.sha256(json.dumps(dict(
        keys, asked=asked, layout_samples=layout_samples, train=relative(train_rgb),
        select=relative(select_files), select_keys=select_keys, test=relative(test_files), test_keys=test_keys,
    ), sort_keys=True).encode()).hexdigest()

    data = DepthBatches(
        train_ds=None, select_ds=_batched(x_select, y_select, batch_size),
        test_ds=_batched(x_test, y_test, batch_size), steps_per_epoch=steps_per_epoch,
        n_train=steps_per_epoch * batch_size, n_select=n_select, n_test=n_test,
        input_shape=(image_size, image_size, 3), num_classes=None,
        selection_split=f"{n_select} fixed crops of {len(deal.select)} selection scenes",
        test_split=f"{n_test} fixed crops of {len(deal.test)} test scenes (never selects)",
        fingerprint=fingerprint, seed=int(seed), stream_seed=stream_seed, batch_size=batch_size,
        image_size=image_size, min_valid_ratio=float(min_valid_ratio), augment=bool(augment), workers=int(workers),
        x_select=x_select, y_select=y_select, x_test=x_test, y_test=y_test, layout=layout,
        select_files=select_files, test_files=test_files, select_keys=select_keys, test_keys=test_keys,
        select_pool=select_pool, test_pool=test_pool,
        train_scenes=tuple(name for name, _, _ in deal.train), select_scenes=tuple(name for name, _, _ in deal.select),
        test_scenes=tuple(name for name, _, _ in deal.test), train_rgb=list(train_rgb), train_depth=list(train_depth),
    )
    return dataclasses.replace(data, train_ds=data._stream(stream_seed))
