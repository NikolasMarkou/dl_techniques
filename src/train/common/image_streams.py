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
from typing import Any, Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.classification import get_spec
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
    (an evaluation pass has no counter to key on) and the other two yield plain
    ``(image, label)`` batches.
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
        validation_split: Optional[float] = None,
        max_samples: Optional[int] = None,
        augment: bool = True,
        crop_policy: str = "resize",
        element_map_fn: Optional[ElementMapFn] = None,
        indexed_element_map_fn: Optional[ElementMapFn] = None,
        num_parallel_calls: int = tf.data.AUTOTUNE,
) -> ImageStreams:
    """Build the train / selection / test pipelines of ``dataset`` for a self-supervised trainer.

    Every pipeline yields ``(image, label)`` (``float32`` images of
    ``image_size x image_size x 3`` normalized with the dataset's statistics, ``int32``
    labels), then the map, then batches. The train stream reshuffles every pass, repeats and
    drops the ragged last batch.

    :param dataset: One of :data:`SUPPORTED_DATASETS`.
    :param image_size: Side length the images are resized or cropped to.
    :param batch_size: Batch size of all three pipelines.
    :param seed: Seed of the split, the shuffle and the augmentation.
    :param validation_split: Fraction of the train rows held out as the selection split
        (``None`` means :data:`DEFAULT_VALIDATION_SPLIT`), strictly inside (0, 1).
    :param max_samples: Smoke-run cap on the train pool and on the test split.
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
    :param num_parallel_calls: ``tf.data`` parallelism of the maps.
    :return: The :class:`ImageStreams`.
    :raises ValueError: An unsupported ``dataset``, a non-positive ``image_size`` or
        ``batch_size``, an unknown ``crop_policy`` (or ``"random_resized_crop"`` with an
        in-memory dataset), both map slots at once, or a selection split holding out zero rows.
    :raises FileNotFoundError: The Imagenette records are not prepared.
    """
    # DECISION plan-2026-08-01T195746-12a1f2db/D-007
    # Both refusals are LOUD rather than a silent best-effort and fire BEFORE any data work.
    # The indexed map has no evaluation form: an eval pass has no `.repeat()`, so
    # `.enumerate()` would count a SINGLE pass and every element would key identically on
    # every epoch -- the frozen-augmentation failure D-035 RED-proved wrong, reintroduced
    # silently through a different door. Pinned by tests/test_train/test_image_streams.py::
    # TestTheRefusals and ::TestIndexedElementMapFn::test_selection_and_test_are_plain_image_label_batches.
    if indexed_element_map_fn is not None and element_map_fn is not None:
        raise ValueError(
            "element_map_fn and indexed_element_map_fn are mutually exclusive: they are two "
            "different calling conventions for the same map slot (`fn(image, label)` vs "
            "`fn(index, image, label)`), and applying both would feed the second one the "
            "first one's output. Pass exactly one."
        )
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

    if spec.source == "records":
        config = ImagenetteConfig(
            image_size=image_size, batch_size=batch_size, crop_policy=crop_policy,
            augment=augment, normalize="imagenet", validation_split=split, seed=seed,
            max_images=max_samples,
        )
        train, n_train = imagenette_examples("train", config)
        select, n_select = imagenette_examples("select", config)
        test, n_test = imagenette_examples("test", config)
    else:
        arrays = load_split_arrays(
            dataset, seed=seed, validation_split=split, max_samples=max_samples)
        mean, std = np.asarray(spec.mean, np.float32), np.asarray(spec.std, np.float32)
        common = dict(
            seed=seed, normalize=lambda image: (image - mean) / std,
            resize=resizer(None if image_size == spec.input_shape[0] else image_size))
        hook = flip_pad_crop(mode="REFLECT", seed=seed) if augment else None
        train = array_examples(arrays.x_train, arrays.y_train, training=True,
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
