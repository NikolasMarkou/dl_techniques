"""Direct tests for `build_raw_image_dataset` — the shared pipeline builder.

Written for plan-2026-08-01T195746-12a1f2db step 5. Until then this function had
NO dedicated test module, despite being shared by **7 call sites across 3
trainers** (`train_classification.py` x2, `train_masked_completion.py` x2,
`train_dino.py` x3). The change that motivated the module — an
`indexed_element_map_fn` that moves `.repeat()` before the element map and
inserts an `.enumerate()` — alters the pipeline SHAPE, not just a parameter, so
the DEFAULT-OFF path needs a guard that can see a shape change.

Three things here are load bearing:

1. **`TestDefaultOffIsUnchanged` compares against a REFERENCE pipeline built by
   hand in this file**, in the exact pre-change order (`shuffle` -> `normalize`
   -> `element_map_fn` -> `repeat` -> `batch`). That reference is what makes the
   guard durable: the decisive one-off comparison was run against a pristine
   `git worktree` at the pre-step commit `5f23df61` and is recorded in the
   plan's `verification.md`, but a worktree is not something a committed test
   can rely on existing.
2. **`test_an_index_ignoring_map_fn_reproduces_the_default_element_sequence`**
   is the plan's falsifiable prediction (A-4): every intervening map is
   order-preserving and elementwise, so enumerating cannot reorder anything.
   If it ever fails, the pipeline shape changed something the plan did not
   model.
3. **The counter must come from `.enumerate()` AFTER `.repeat()`.** A counter
   that restarts each epoch freezes the augmentation per image, which is the
   failure D-035 RED-proved wrong; `test_the_counter_keeps_climbing_across_
   epochs` is the only thing here that can tell the two apart.

Every dataset below is `cifar10`, which `build_raw_image_dataset` builds
IN-MEMORY from `keras.datasets` — no TFDS, no network, no spinning disk.
"""

from typing import Any, List, Tuple

import numpy as np
import pytest
import tensorflow as tf

from train.energy_transformer.common import (
    DATASET_NUM_CLASSES,
    build_raw_image_dataset,
)

IMAGE_SIZE = 32
BATCH_SIZE = 4
SEED = 7


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _pair_map_fn(image: tf.Tensor, label: tf.Tensor) -> Tuple[Any, Any]:
    """A stand-in for the trainers' real `element_map_fn`s.

    Deliberately shape-changing (it stacks two copies on a new leading axis, as
    the multi-crop transform does), so a test using it would notice the map
    being dropped, applied twice, or applied on the wrong side of `.batch()`.
    """
    return tf.stack([image, image + 1.0], axis=0), label


def _indexed_pair_map_fn(
        index: tf.Tensor, image: tf.Tensor, label: tf.Tensor
) -> Tuple[Any, Any]:
    """`_pair_map_fn`'s indexed twin, IGNORING the index (A-4's control)."""
    del index
    return _pair_map_fn(image, label)


def _index_reporting_map_fn(
        index: tf.Tensor, image: tf.Tensor, label: tf.Tensor
) -> Tuple[Any, Any]:
    """Emit the counter itself, so the enumeration is directly observable."""
    del image
    return tf.cast(index, tf.int64), label


def _take(ds: tf.data.Dataset, n_batches: int) -> List[np.ndarray]:
    return [np.asarray(x) for x, _ in ds.take(n_batches)]


def _reference_pipeline(*, element_map_fn=None, is_training: bool):
    """The pre-change pipeline, rebuilt by hand from cifar10's numpy arrays.

    This is the CONTROL for `indexed_element_map_fn=None`. It repeats the order
    `build_raw_image_dataset` documents rather than calling it, so a reordering
    inside that function is visible here instead of cancelling out.
    """
    import keras

    from train.common import CIFAR10_MEAN, CIFAR10_STD

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    images = (x_train if is_training else x_test).astype("float32") / 255.0
    labels = (y_train if is_training else y_test).flatten().astype("int32")

    mean = tf.constant(CIFAR10_MEAN, dtype=tf.float32, shape=(1, 1, 3))
    std = tf.constant(CIFAR10_STD, dtype=tf.float32, shape=(1, 1, 3))

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda img, lbl: ((img - mean) / std, lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    if not is_training:
        ds = ds.cache()
    if element_map_fn is not None:
        ds = ds.map(element_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.repeat().batch(BATCH_SIZE, drop_remainder=True)
    else:
        ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# 1. the refusals
# ---------------------------------------------------------------------


class TestTheRefusals:
    """Both illegal combinations must name their reason, not fail obscurely."""

    def test_supplying_both_map_fn_slots_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_raw_image_dataset(
                "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=True,
                augment=False, element_map_fn=_pair_map_fn,
                indexed_element_map_fn=_indexed_pair_map_fn)

    def test_an_indexed_map_fn_on_the_eval_pipeline_is_refused(self) -> None:
        """No `.repeat()` there, so the counter would enumerate ONE pass.

        Refusing is the point: silently enumerating a single pass hands every
        element the same index on every epoch, i.e. the frozen-per-image
        augmentation D-035 already RED-proved wrong, reintroduced through a
        different door.
        """
        with pytest.raises(ValueError, match=r"requires is_training=True"):
            build_raw_image_dataset(
                "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=False,
                indexed_element_map_fn=_indexed_pair_map_fn)


# ---------------------------------------------------------------------
# 2. the default-off byte-identity guard
# ---------------------------------------------------------------------


class TestDefaultOffIsUnchanged:
    """The other 6 call sites must not be able to tell this change happened."""

    @pytest.mark.parametrize("is_training", [True, False])
    @pytest.mark.parametrize("with_element_map_fn", [False, True])
    def test_the_default_pipeline_matches_the_pre_change_shape(
            self, is_training: bool, with_element_map_fn: bool) -> None:
        element_map_fn = _pair_map_fn if with_element_map_fn else None

        ds, num_examples, num_classes = build_raw_image_dataset(
            "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=is_training,
            augment=False, element_map_fn=element_map_fn, seed=SEED)
        reference = _reference_pipeline(
            element_map_fn=element_map_fn, is_training=is_training)

        assert num_classes == DATASET_NUM_CLASSES["cifar10"]
        assert num_examples == (50000 if is_training else 10000)
        assert ds.element_spec[0].shape.as_list() == \
            reference.element_spec[0].shape.as_list(), (
            f"the batched element SPEC moved: {ds.element_spec[0].shape} vs "
            f"{reference.element_spec[0].shape}. A spec change reaches every "
            f"one of the 7 call sites before a single value is compared."
        )

        got, want = _take(ds, 3), _take(reference, 3)
        assert len(got) == len(want) == 3
        for index, (a, b) in enumerate(zip(got, want)):
            np.testing.assert_array_equal(a, b, err_msg=(
                f"batch {index} differs from the pre-change pipeline at "
                f"is_training={is_training}, element_map_fn="
                f"{with_element_map_fn}. `indexed_element_map_fn=None` must "
                f"preserve today's behaviour EXACTLY -- this function is "
                f"shared by 3 trainers."
            ))


# ---------------------------------------------------------------------
# 3. the enumerated pipeline
# ---------------------------------------------------------------------


class TestIndexedElementMapFn:
    """The opt-in branch: same elements, plus a counter that climbs."""

    def test_an_index_ignoring_map_fn_reproduces_the_default_element_sequence(
            self) -> None:
        """A-4, executed rather than assumed.

        Every map between the source and the element map is order-preserving
        and elementwise, so moving `.repeat()` earlier and inserting
        `.enumerate()` must not change WHICH elements come out or in what
        order. If this fails, the pipeline shape changed something the plan did
        not model.
        """
        default_ds, _, _ = build_raw_image_dataset(
            "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=True, augment=False,
            element_map_fn=_pair_map_fn, seed=SEED)
        indexed_ds, _, _ = build_raw_image_dataset(
            "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=True, augment=False,
            indexed_element_map_fn=_indexed_pair_map_fn, seed=SEED)

        assert indexed_ds.element_spec[0].shape.as_list() == \
            default_ds.element_spec[0].shape.as_list()

        for index, (a, b) in enumerate(
                zip(_take(indexed_ds, 3), _take(default_ds, 3))):
            np.testing.assert_array_equal(a, b, err_msg=(
                f"batch {index}: enumerating changed the emitted element "
                f"sequence. The index-ignoring map fn makes this a pure test "
                f"of pipeline ORDER."
            ))

    def test_the_counter_keeps_climbing_across_epochs(self) -> None:
        """`.enumerate()` AFTER `.repeat()`, which is the whole point.

        `steps_per_epoch` here is `50000 // 4`, so reading past it lands in the
        second epoch. A counter placed BEFORE `.repeat()` would restart at 0
        there, every source image would key identically on every epoch, and the
        stateless augmentation built on this counter would be frozen per image.
        """
        ds, num_examples, _ = build_raw_image_dataset(
            "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=True, augment=False,
            indexed_element_map_fn=_index_reporting_map_fn, seed=SEED)

        steps_per_epoch = num_examples // BATCH_SIZE
        counters = np.concatenate(_take(ds.skip(steps_per_epoch - 2), 4))

        expected_first = (steps_per_epoch - 2) * BATCH_SIZE
        np.testing.assert_array_equal(
            counters,
            np.arange(expected_first, expected_first + 4 * BATCH_SIZE),
            err_msg=(
                "the per-element counter does not run monotonically across the "
                "epoch boundary. `.enumerate()` must sit AFTER `.repeat()`; "
                "before it, the counter restarts each epoch and hands the same "
                "source image the same key forever."
            ),
        )
        # Non-vacuity: the window really does straddle the epoch boundary, so
        # the monotonicity above is a claim about epoch 2 and not about a
        # comfortable stretch in the middle of epoch 1.
        assert counters[-1] >= num_examples > counters[0]
