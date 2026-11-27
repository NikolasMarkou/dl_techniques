"""Image streams for the self-supervised trainers: unlabelled images with a held-out slice.

BEiT MIM and tokenizer, ET masked completion and DINO feed images through a per-sample map
(a masked-patch element, a multi-crop) and only need the labels for a probe. They used to
build that from a private TFDS and CIFAR reader that validated on the labelled test split.
:func:`load_image_streams` replaces it with the protocol of ``train.common.classification_data``:
the SELECTION split (``val_*``, early stopping, the best checkpoint) is a seeded slice of the
train rows, and the test split is scored once after training.

The rows are the loader's: Imagenette comes from
``dl_techniques.datasets.vision.imagenette.imagenette_examples`` and CIFAR from
``classification_data.load_split_arrays``, so the same seed picks the same selection rows as
the classifiers do.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.classification import get_spec
from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD
from dl_techniques.datasets.vision.imagenette import (
    CROP_POLICIES,
    ImagenetteConfig,
    imagenette_examples,
)
from dl_techniques.utils.logger import logger
from train.common.classification_data import (
    DEFAULT_VALIDATION_SPLIT,
    array_examples,
    capped_steps,
    flip_pad_crop,
    load_split_arrays,
    resizer,
)

#: The datasets the self-supervised trainers take: three colour channels, small enough to
#: hold in memory (CIFAR-10) or in prepared records (Imagenette). The trainers' image-size and
#: patch-size defaults are written for these two.
SUPPORTED_DATASETS: Tuple[str, ...] = ("imagenette", "cifar10")

#: Per-sample map: ``(image, label) -> element``.
ElementMapFn = Callable[..., Any]


@dataclass(frozen=True)
class ImageStreams:
    """The three batched pipelines of one dataset plus their counts.

    ``train_ds`` is infinite (it repeats) and drops the ragged last batch; ``select_ds`` and
    ``test_ds`` are finite, never augmented and keep every row. With ``element_map_fn`` all
    three yield the mapped element; with ``indexed_element_map_fn`` only ``train_ds`` does
    (the train counter runs across epochs, an evaluation pass has none) and the other two
    yield plain ``(image, label)`` batches unless ``select_indexed_map_fn`` or
    ``test_indexed_map_fn`` is given, which map that one finite stream by row position.
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: "tf.data.Dataset"
    n_train: int
    n_select: int
    n_test: int
    batch_size: int

    @property
    def steps_per_epoch(self) -> int:
        return max(1, self.n_train // self.batch_size)

    @property
    def val_steps(self) -> int:
        """Batches in one pass of ``select_ds``."""
        return max(1, -(-self.n_select // self.batch_size))

    @property
    def test_steps(self) -> int:
        """Batches in one pass of ``test_ds``."""
        return max(1, -(-self.n_test // self.batch_size))

    def capped_steps(self, max_steps: Optional[int]) -> Tuple[int, int]:
        """``(steps_per_epoch, val_steps)``, each capped by ``--max-steps`` (smoke runs)."""
        return capped_steps(self.steps_per_epoch, self.val_steps, max_steps)


def standardizer(mean: Sequence[float], std: Sequence[float]) -> Callable[[tf.Tensor], tf.Tensor]:
    """``image -> (image - mean) / std`` per channel, with ``mean`` and ``std`` as float32.

    Interface contract: ``mean`` and ``std`` are one number per channel (``std`` positive); the
    returned function takes a float32 image or batch ``(..., channels)`` and returns it standardized,
    float32. The in-memory streams standardize with it; :func:`evaluation_transform` ends with it.
    """
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    return lambda image: (image - mean) / std


def stream_statistics(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """The per-channel ``(mean, std)`` the streams of ``dataset`` standardize with, float32 ``(3,)`` each.

    Interface contract: ``dataset`` is one of :data:`SUPPORTED_DATASETS`; raises ``ValueError``
    for another. Imagenette's records are standardized inside
    ``dl_techniques.datasets.vision.imagenette.imagenette_examples`` with the ImageNet statistics
    (its ``normalize="imagenet"``, which :func:`load_image_streams` passes); an in-memory dataset
    with its spec's, which the CIFAR branch below reads from here. A pretrain ``task.json``
    records these numbers.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; supported: {sorted(SUPPORTED_DATASETS)}")
    spec = get_spec(dataset)
    if spec.source == "records":
        return np.asarray(IMAGENET_MEAN, np.float32), np.asarray(IMAGENET_STD, np.float32)
    return np.asarray(spec.mean, np.float32), np.asarray(spec.std, np.float32)


def evaluation_transform(
        image_size: int, mean: Sequence[float], std: Sequence[float],
) -> Callable[[tf.Tensor], tf.Tensor]:
    """The transform of the selection and test streams: a ``[0, 1]`` image to the model's input.

    Interface contract: the returned function takes one float32 ``(H, W, 3)`` image in ``[0, 1]``
    (bytes divided by 255) of any size and returns the float32 ``(image_size, image_size, 3)``
    input: a bilinear resize (``classification_data.resizer``), then :func:`standardizer` with
    ``mean`` and ``std`` (:func:`stream_statistics` for the dataset). That is what the streams
    apply after their decode, with ``crop_policy="resize"``: the CIFAR branch below (it skips the
    resize at the native size, where the resize is the identity, measured bit for bit) and
    Imagenette's records (``imagenette_examples``: a bilinear resize, then the ImageNet statistics).
    """
    # DECISION plan-2026-11-19T081214-5c6cc105/D-023
    # The pretrain predictor imports THIS function; the CIFAR branch below standardizes with the
    # same `standardizer`. Do NOT copy the transform into the predictor (D-003) and do NOT add a
    # crop: the pretrain streams resize the whole image. Imagenette's own transform lives in the
    # `dl_techniques` loader and is held equal to this one by `test_pretrain_record.py`.
    resize = resizer(image_size)
    normalize = standardizer(mean, std)
    return lambda image: normalize(resize(image))


def _indexed_call(indexed_element_map_fn: ElementMapFn) -> ElementMapFn:
    """Adapt ``fn(index, image, label)`` to what ``.enumerate().map`` hands over.

    MEASURED: ``.enumerate()`` emits ``(index, element)`` and ``tf.data`` passes a TUPLE
    element as ONE nested argument, so the map receives 2 arguments, ``(index, (image,
    label))``, not 3. This unpacks that nesting.
    """

    def call(index: tf.Tensor, *element: Any) -> Any:
        if len(element) == 1 and isinstance(element[0], tuple):
            return indexed_element_map_fn(index, *element[0])
        return indexed_element_map_fn(index, *element)

    return call


def load_image_streams(
        dataset: str,
        *,
        image_size: int,
        batch_size: int,
        seed: int,
        stream_seed: Optional[int] = None,
        validation_split: Optional[float] = None,
        max_samples: Optional[int] = None,
        augment: bool = True,
        crop_policy: str = "resize",
        element_map_fn: Optional[ElementMapFn] = None,
        indexed_element_map_fn: Optional[ElementMapFn] = None,
        select_indexed_map_fn: Optional[ElementMapFn] = None,
        test_indexed_map_fn: Optional[ElementMapFn] = None,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        shuffle_train: bool = True,
) -> ImageStreams:
    """Build the train / selection / test pipelines of ``dataset`` for a self-supervised trainer.

    Every pipeline yields ``(image, label)`` (``float32`` images of
    ``image_size x image_size x 3`` normalized with the dataset's statistics, ``int32``
    labels), then the map, then batches. The train stream reshuffles every pass, repeats and
    drops the ragged last batch.

    :param dataset: One of :data:`SUPPORTED_DATASETS`.
    :param image_size: Side length the images are resized or cropped to.
    :param batch_size: Batch size of all three pipelines.
    :param seed: Seed of the split (always), and of the shuffle and the augmentation unless
        ``stream_seed`` is given.
    :param stream_seed: Seed of the train shuffle and the train augmentation ONLY; ``None``
        (the default) means ``seed``, so a caller that passes nothing is unchanged. ``0`` is a
        seed, not "unset". It never reaches the split: the selection and test rows, and the
        train/selection partition, come from ``seed`` whatever ``stream_seed`` is, which is
        what lets a resumed run train on a fresh stream without re-splitting.
    :param validation_split: Fraction of the train rows held out as the selection split
        (``None`` means :data:`DEFAULT_VALIDATION_SPLIT`), strictly inside (0, 1).
    :param max_samples: Smoke-run cap on the train pool (the test split is never capped).
    :param augment: Train-time augmentation of the ``crop_policy``; ignored by the other two.
    :param crop_policy: One of :data:`CROP_POLICIES`. ``"random_resized_crop"`` is Imagenette only.
    :param element_map_fn: Per-sample ``fn(image, label)`` applied to all three streams.
        The masked-patch trainers put their element here.
    :param indexed_element_map_fn: Per-sample ``fn(index, image, label)`` for the TRAIN stream
        only. The pipeline becomes ``.repeat() -> .enumerate() -> .map -> .batch``, so the SAME
        source image carries a DIFFERENT ``index`` on each epoch; that counter is what a
        stateless-RNG augmentation keys on, the only way to make the stream reproducible under
        ``num_parallel_calls=AUTOTUNE`` (``dl_techniques.datasets.vision.multi_crop``). The
        selection and test streams are then plain ``(image, label)`` batches. Mutually
        exclusive with ``element_map_fn``.
    :param select_indexed_map_fn: Per-sample ``fn(index, image, label)`` for the SELECTION
        stream only; ``index`` is the 0-based row position of the finite pass (int64), the
        same on every pass, and does not depend on ``seed`` or ``stream_seed``. A mask keyed
        on it scores the same weights the same way on every pass. Mutually exclusive with
        ``element_map_fn``; it may be combined with the other indexed maps.
    :param test_indexed_map_fn: Same as ``select_indexed_map_fn``, for the TEST stream.
    :param num_parallel_calls: ``tf.data`` parallelism of the maps.
    :param shuffle_train: ``False`` leaves the train rows in the seeded-permutation order of the split (the fit rows of
        ``dl_techniques.datasets.vision.splits.seeded_split_indices``, uniform and fixed by ``seed``), so the stream is a function of ``seed``
        alone; ``True`` (the default) is the stream that reshuffles every pass, whose order follows TensorFlow's global seed as well. The
        stream still repeats, so it wraps to its first row after one pass. Refused with ``augment`` (an augmented stream is always shuffled).
    :return: The :class:`ImageStreams`.
    :raises ValueError: An unsupported ``dataset``, a non-positive ``image_size`` or
        ``batch_size``, an unknown ``crop_policy`` (or ``"random_resized_crop"`` with an
        in-memory dataset), ``element_map_fn`` beside any indexed map, ``shuffle_train=False`` with ``augment``, or a selection split holding
        out zero rows.
    :raises FileNotFoundError: The Imagenette records are not prepared.
    """
    # DECISION plan-2026-08-01T195746-12a1f2db/D-007
    # Both refusals are LOUD rather than a silent best-effort and fire BEFORE any data work.
    # The TRAIN indexed map has no evaluation form: an eval pass has no `.repeat()`, so
    # `.enumerate()` would count a SINGLE pass and every element would key identically on
    # every epoch -- the frozen-augmentation failure D-035 RED-proved wrong, reintroduced
    # silently through a different door. (The row-position evaluation maps are the separate
    # parameters of the plan-2026-11-12T015747-46e5318f/D-005 anchor below.) Pinned by tests/test_train/test_image_streams.py::
    # TestTheRefusals and ::TestIndexedElementMapFn::test_selection_and_test_are_plain_image_label_batches.
    # DECISION plan-2026-11-12T015747-46e5318f/D-005
    # The select and test maps are SEPARATE parameters, not a lifted restriction on
    # `indexed_element_map_fn`: their index is the row position of ONE finite pass (the same
    # on every pass, so a position-keyed mask scores the same weights the same way each
    # time), while the train counter keeps climbing across epochs. Do NOT make the train
    # counter serve the eval streams (the frozen-augmentation failure above) and do NOT key
    # an eval map on the stream seed: it changes on every resume, so the same weights would
    # score a different val_loss after a kill and the best-checkpoint reload check would
    # warn on every resumed run.
    for name, fn in (("indexed_element_map_fn", indexed_element_map_fn),
                     ("select_indexed_map_fn", select_indexed_map_fn),
                     ("test_indexed_map_fn", test_indexed_map_fn)):
        if fn is not None and element_map_fn is not None:
            raise ValueError(
                f"element_map_fn and {name} are mutually exclusive: they are two "
                "different calling conventions for the same map slot (`fn(image, label)` vs "
                "`fn(index, image, label)`), and applying both would feed the second one the "
                "first one's output. Pass exactly one."
            )
    # DECISION plan-2026-11-26T224507-e3381378/D-095
    # `shuffle_train=False` is the fit rows in the split's seeded order, a function of `seed` alone (a tf.data shuffle also follows TensorFlow's global
    # seed, so the dino probe's bank could not be read twice the same by a caller that had not seeded the process). It defaults to the shuffle every other
    # caller reads. Do NOT default it to False, and do NOT allow it with `augment`: the augmented stream draws another crop of a row on every pass.
    if not shuffle_train and augment:
        raise ValueError(
            "shuffle_train=False needs augment=False: an unshuffled train stream is the fixed-order fit rows of a probe, and an augmented one "
            "would draw a different crop of each on every pass")
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; supported: {sorted(SUPPORTED_DATASETS)}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if crop_policy not in CROP_POLICIES:
        raise ValueError(f"Unknown crop_policy {crop_policy!r}; known: {list(CROP_POLICIES)}")
    spec = get_spec(dataset)
    if crop_policy == "random_resized_crop" and spec.source != "records":
        raise ValueError(
            "crop_policy='random_resized_crop' is implemented for imagenette only "
            f"(full-resolution records to crop from), not {dataset!r}"
        )
    split = DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split
    # DECISION plan-2026-11-12T015747-46e5318f/D-004
    # `stream_seed` feeds the shuffle and the augmentation and NEVER the split. A resumed
    # fit starts a fresh iterator that would replay the stream from its first element, so the
    # resume wants a different stream; moving `seed` itself (the legacy language idea of
    # `seed + step`) would re-split and leak selection rows into train. `is None`, not
    # truthiness: `stream_seed=0` is a seed. Do not derive the split from this value.
    stream = seed if stream_seed is None else stream_seed

    if spec.source == "records":
        config = ImagenetteConfig(
            image_size=image_size, batch_size=batch_size, crop_policy=crop_policy,
            augment=augment, normalize="imagenet", validation_split=split, seed=seed,
            stream_seed=stream_seed, max_images=max_samples, shuffle=shuffle_train,
        )
        train, n_train = imagenette_examples("train", config)
        select, n_select = imagenette_examples("select", config)
        test, n_test = imagenette_examples("test", config)
    else:
        arrays = load_split_arrays(
            dataset, seed=seed, validation_split=split, max_samples=max_samples)
        common = dict(
            seed=stream, normalize=standardizer(*stream_statistics(dataset)),
            resize=resizer(None if image_size == spec.input_shape[0] else image_size))
        hook = flip_pad_crop(mode="REFLECT", seed=stream) if augment else None
        train = array_examples(arrays.x_train, arrays.y_train, training=shuffle_train,
                               augment=hook, **common)
        select = array_examples(arrays.x_select, arrays.y_select, training=False,
                                augment=None, **common)
        test = array_examples(arrays.x_test, arrays.y_test, training=False,
                              augment=None, **common)
        n_train, n_select, n_test = len(arrays.x_train), len(arrays.x_select), len(arrays.x_test)

    if element_map_fn is not None:
        train, select, test = (
            part.map(element_map_fn, num_parallel_calls=num_parallel_calls)
            for part in (train, select, test)
        )

    if indexed_element_map_fn is not None:
        # DECISION plan-2026-08-01T195746-12a1f2db/D-008
        # `.enumerate()` goes AFTER `.repeat()`, never before. Before it the counter would
        # restart at 0 every epoch and the same source image would draw the SAME augmentation
        # forever. After it the counter runs 0, 1, 2, ... across the infinite repeat, so epoch
        # 2's copy of image k keys differently from epoch 1's. A determinism test alone cannot
        # see the difference; `test_the_counter_keeps_climbing_across_epochs` pins it.
        train = train.repeat().enumerate().map(
            _indexed_call(indexed_element_map_fn), num_parallel_calls=num_parallel_calls)
    else:
        train = train.repeat()
    if select_indexed_map_fn is not None:
        select = select.enumerate().map(
            _indexed_call(select_indexed_map_fn), num_parallel_calls=num_parallel_calls)
    if test_indexed_map_fn is not None:
        test = test.enumerate().map(
            _indexed_call(test_indexed_map_fn), num_parallel_calls=num_parallel_calls)
    train = train.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    select = select.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test = test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info(
        f"{dataset}: train {n_train}, select {n_select}, test {n_test}, "
        f"{image_size}x{image_size}, crop_policy={crop_policy}, augment={augment}"
    )
    return ImageStreams(
        train_ds=train, select_ds=select, test_ds=test,
        n_train=n_train, n_select=n_select, n_test=n_test, batch_size=batch_size,
    )
