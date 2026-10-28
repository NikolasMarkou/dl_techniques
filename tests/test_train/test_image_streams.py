"""``train.common.image_streams.load_image_streams``: the self-supervised trainers' pipelines.

If this module is ever reported as carrying "pre-existing failures", read
``research/2026_test_suite_contention_and_false_failures.md`` FIRST: a prior report of
exactly that was refuted, and the faults were GPU contention from concurrent test runs.

Three things are pinned here:

1. **The protocol.** The fit, selection and test rows are pairwise disjoint, the first two
   cover the train pool, selection is the seeded slice the classifiers get for the same seed,
   and the test rows are the seeded test order. Both sources are audited image by image: every
   image has the mean colour ``4 * i`` (+ an offset for the test split), so its identity is
   read back from the pixels.
2. **The map slots.** ``element_map_fn`` reaches all three streams; ``indexed_element_map_fn``
   reaches the train stream only, and its counter comes from ``.enumerate()`` AFTER
   ``.repeat()`` (a counter that restarts each epoch freezes the augmentation per image, the
   failure D-035 RED-proved wrong). The two slots are mutually exclusive.
3. **Determinism.** The rows and their order are a function of the seed alone; the
   real-records test at the bottom measures that across builds.

The source seams are fakes (``tests/classification_fakes.py``): Imagenette is 60 tiny jpegs
plus 20 validation jpegs, CIFAR-10 is 60 + 20 ramp arrays. Nothing touches TFDS or
the Keras cache except the one skip-guarded real-records class.
"""

import os
from glob import glob
from typing import Any, List, Tuple

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.classification import get_spec
from dl_techniques.datasets.vision.imagenette import (
    IMAGENETTE_TFDS_NAME,
    IMAGENETTE_TRAIN_SIZE,
    IMAGENETTE_VALIDATION_SIZE,
)
from dl_techniques.datasets.vision.splits import seeded_split_indices, seeded_test_rows
from tests.classification_fakes import (
    IMAGENETTE_FAKE_TEST,
    IMAGENETTE_FAKE_TRAIN,
    identities_of,
    use_fake_imagenette,
    use_identity_cifar10,
)
from train.common.image_streams import SUPPORTED_DATASETS, load_image_streams

IMAGE_SIZE = 32
BATCH_SIZE = 4
SEED = 7
N_TRAIN, N_TEST = IMAGENETTE_FAKE_TRAIN, IMAGENETTE_FAKE_TEST
TEST_OFFSET = 2  # the fake validation jpegs are `4 * i + 2`, the fake CIFAR test arrays too


@pytest.fixture(params=["imagenette", "cifar10"])
def dataset(request, monkeypatch) -> str:
    """Each protocol test runs on both sources."""
    if request.param == "imagenette":
        use_fake_imagenette(monkeypatch)
    else:
        use_identity_cifar10(monkeypatch)
    return request.param


def _streams(dataset: str, **kwargs):
    settings = dict(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED, augment=False)
    settings.update(kwargs)
    return load_image_streams(dataset, **settings)


def _identities(dataset: str, ds: tf.data.Dataset, n_batches: int, offset: int) -> np.ndarray:
    return identities_of(dataset, ds, offset, n_batches)


def _all(dataset: str, ds: tf.data.Dataset, offset: int) -> np.ndarray:
    return identities_of(dataset, ds, offset)


def _pair_map_fn(image: tf.Tensor, label: tf.Tensor) -> Tuple[Any, Any]:
    """Shape-changing (two stacked copies), so a dropped, doubled or misplaced map is visible."""
    return tf.stack([image, image + 1.0], axis=0), label


def _indexed_pair_map_fn(index: tf.Tensor, image: tf.Tensor, label: tf.Tensor) -> Tuple[Any, Any]:
    del index
    return _pair_map_fn(image, label)


def _index_reporting_map_fn(index: tf.Tensor, image: tf.Tensor, label: tf.Tensor) -> Tuple[Any, Any]:
    del image
    return tf.cast(index, tf.int64), label


def _take(ds: tf.data.Dataset, n_batches: int) -> List[np.ndarray]:
    return [np.asarray(x) for x, _ in ds.take(n_batches)]


# ---------------------------------------------------------------------
# 1. the refusals
# ---------------------------------------------------------------------


class TestTheRefusals:
    """Each illegal request names its reason, before any data is read."""

    def test_supplying_both_map_slots_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_image_streams("cifar10", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED,
                               element_map_fn=_pair_map_fn,
                               indexed_element_map_fn=_indexed_pair_map_fn)

    @pytest.mark.parametrize("kwargs, message", [
        (dict(dataset="mnist"), "Unsupported dataset 'mnist'"),
        (dict(image_size=0), "image_size must be positive"),
        (dict(batch_size=0), "batch_size must be positive"),
        (dict(crop_policy="centre"), "Unknown crop_policy"),
        (dict(crop_policy="random_resized_crop"), "implemented for imagenette only"),
    ])
    def test_a_bad_setting_is_refused_naming_it(self, kwargs, message) -> None:
        settings = dict(dataset="cifar10", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED)
        settings.update(kwargs)
        with pytest.raises(ValueError, match=message):
            load_image_streams(settings.pop("dataset"), **settings)

    @pytest.mark.parametrize("split", [0.0, 1.0, -0.1])
    def test_a_selection_split_outside_the_open_interval_is_refused(self, split, monkeypatch) -> None:
        use_identity_cifar10(monkeypatch)
        with pytest.raises(ValueError, match="strictly inside"):
            _streams("cifar10", validation_split=split)

    def test_the_supported_datasets_are_the_two_the_trainers_size_for(self) -> None:
        assert set(SUPPORTED_DATASETS) == {"imagenette", "cifar10"}


# ---------------------------------------------------------------------
# 2. the protocol
# ---------------------------------------------------------------------


class TestTheSelectionProtocol:

    def test_sizes_and_steps(self, dataset) -> None:
        streams = _streams(dataset)
        assert (streams.n_train, streams.n_select, streams.n_test) == (54, 6, 20)
        assert streams.steps_per_epoch == 54 // BATCH_SIZE
        assert streams.val_steps == 2  # ceil(6 / 4): every selection row is scored
        assert streams.test_steps == 5  # 20 / 4
        assert int(streams.select_ds.cardinality()) == streams.val_steps
        assert int(streams.test_ds.cardinality()) == streams.test_steps

    def test_fit_selection_and_test_rows_are_disjoint_and_cover_the_pool(self, dataset) -> None:
        streams = _streams(dataset)
        fit = _identities(dataset, streams.train_ds, streams.n_train // BATCH_SIZE, 0)
        select = _all(dataset, streams.select_ds, 0)
        test = _all(dataset, streams.test_ds, TEST_OFFSET)
        assert len(select) == 6 and len(test) == 20
        assert not set(fit) & set(select), "a selection image is also a fit image"
        assert set(fit) | set(select) <= set(range(N_TRAIN))
        assert len(set(fit) | set(select)) >= 52  # 6 + the 52 rows 13 full batches use
        assert sorted(test) == list(range(N_TEST))

    def test_the_selection_rows_are_the_seeded_slice_the_classifiers_get(self, dataset) -> None:
        streams = _streams(dataset)
        _, select_rows, _ = seeded_split_indices(N_TRAIN, N_TEST, SEED, 0.1)
        assert sorted(_all(dataset, streams.select_ds, 0)) == sorted(select_rows.tolist())
        assert list(_all(dataset, streams.select_ds, 0)) == select_rows.tolist()

    def test_the_test_rows_are_the_seeded_test_order(self, dataset) -> None:
        streams = _streams(dataset)
        assert list(_all(dataset, streams.test_ds, TEST_OFFSET)) == seeded_test_rows(
            N_TEST, SEED).tolist()

    def test_the_split_follows_validation_split_and_max_samples(self, dataset) -> None:
        assert _streams(dataset, validation_split=0.25).n_select == 15
        capped = _streams(dataset, max_samples=15, batch_size=3)
        assert (capped.n_train, capped.n_select, capped.n_test) == (14, 1, 15)

    def test_the_rows_depend_on_the_seed_and_only_on_it(self, dataset) -> None:
        a, b, c = (_all(dataset, _streams(dataset, seed=s).select_ds, 0) for s in (1, 1, 2))
        np.testing.assert_array_equal(a, b)
        assert sorted(a) != sorted(c)

    def test_the_train_stream_is_reproducible_across_builds_at_one_seed(self, dataset) -> None:
        first = _take(_streams(dataset).train_ds, 6)
        again = _take(_streams(dataset).train_ds, 6)
        for a, b in zip(first, again):
            np.testing.assert_array_equal(a, b)

    def test_the_train_stream_repeats_and_reshuffles_every_pass(self, dataset) -> None:
        streams = _streams(dataset)
        steps = streams.n_train // BATCH_SIZE
        ids = _identities(dataset, streams.train_ds, 2 * steps, 0)
        first, second = ids[:steps * BATCH_SIZE], ids[steps * BATCH_SIZE:]
        assert len(second) == len(first) > 0
        assert list(first) != list(second), "the second pass replayed the first pass's order"

    def test_train_drops_the_ragged_batch_and_the_others_keep_every_row(self, dataset) -> None:
        streams = _streams(dataset, batch_size=5)
        assert {x.shape[0] for x, _ in streams.train_ds.take(8)} == {5}
        assert sum(x.shape[0] for x, _ in streams.select_ds) == 6
        assert sum(x.shape[0] for x, _ in streams.test_ds) == N_TEST

    def test_selection_and_test_are_never_augmented(self, dataset) -> None:
        for stream in ("select_ds", "test_ds"):
            first = _take(getattr(_streams(dataset, augment=True), stream), 2)
            again = _take(getattr(_streams(dataset, augment=False), stream), 2)
            for a, b in zip(first, again):
                np.testing.assert_array_equal(a, b)

    def test_training_augmentation_changes_the_pixels(self, dataset) -> None:
        plain = _take(_streams(dataset, augment=False).train_ds, 1)[0]
        augmented = _take(_streams(dataset, augment=True).train_ds, 1)[0]
        assert not np.array_equal(plain, augmented)

    def test_the_elements_are_normalized_float_images_and_int_labels(self, dataset) -> None:
        x, y = next(iter(_streams(dataset).train_ds))
        assert x.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3) and x.dtype == tf.float32
        assert y.dtype == tf.int32
        spec = get_spec(dataset)
        raw = np.asarray(x) * np.asarray(spec.std) + np.asarray(spec.mean)
        assert raw.min() >= -1e-3 and raw.max() <= 1.0 + 1e-3, "not a normalized [0, 1] image"

    def test_capped_steps_clamp_both_counts(self, dataset) -> None:
        streams = _streams(dataset)
        assert streams.capped_steps(None) == (13, 2)
        assert streams.capped_steps(1) == (1, 1)
        assert streams.capped_steps(100) == (13, 2)


# ---------------------------------------------------------------------
# 3. the map slots
# ---------------------------------------------------------------------


class TestElementMapFn:

    def test_the_map_reaches_all_three_streams(self, dataset) -> None:
        streams = _streams(dataset, element_map_fn=_pair_map_fn)
        for ds in (streams.train_ds, streams.select_ds, streams.test_ds):
            x, _ = next(iter(ds))
            assert x.shape[1:] == (2, IMAGE_SIZE, IMAGE_SIZE, 3)

    def test_the_map_runs_on_each_example_before_batching(self, dataset) -> None:
        streams = _streams(dataset, element_map_fn=_pair_map_fn)
        x, _ = next(iter(streams.train_ds))
        assert x.shape == (BATCH_SIZE, 2, IMAGE_SIZE, IMAGE_SIZE, 3)
        np.testing.assert_allclose(np.asarray(x[:, 1] - x[:, 0]), 1.0, atol=1e-6)


class TestIndexedElementMapFn:

    def test_an_index_ignoring_map_reproduces_the_element_sequence_of_the_plain_slot(
            self, dataset) -> None:
        """Every map between the source and the element map is order-preserving and
        elementwise, so moving `.repeat()` earlier and inserting `.enumerate()` must not
        change WHICH elements come out or in what order."""
        plain = _streams(dataset, element_map_fn=_pair_map_fn)
        indexed = _streams(dataset, indexed_element_map_fn=_indexed_pair_map_fn)
        assert indexed.train_ds.element_spec[0].shape.as_list() == \
            plain.train_ds.element_spec[0].shape.as_list()
        for a, b in zip(_take(indexed.train_ds, 4), _take(plain.train_ds, 4)):
            np.testing.assert_array_equal(a, b)

    def test_the_counter_keeps_climbing_across_epochs(self, dataset) -> None:
        streams = _streams(dataset, indexed_element_map_fn=_index_reporting_map_fn)
        steps = streams.n_train // BATCH_SIZE
        counters = np.concatenate(_take(streams.train_ds.skip(steps - 2), 4))
        first = (steps - 2) * BATCH_SIZE
        np.testing.assert_array_equal(counters, np.arange(first, first + 4 * BATCH_SIZE), err_msg=(
            "the per-element counter does not run monotonically across the epoch boundary: "
            "`.enumerate()` must sit AFTER `.repeat()`"))
        assert counters[-1] >= streams.n_train > counters[0], "the window must straddle the boundary"

    def test_selection_and_test_are_plain_image_label_batches(self, dataset) -> None:
        streams = _streams(dataset, indexed_element_map_fn=_index_reporting_map_fn)
        for ds in (streams.select_ds, streams.test_ds):
            x, y = next(iter(ds))
            assert x.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, 3) and x.dtype == tf.float32
            assert y.dtype == tf.int32


# ---------------------------------------------------------------------
# 4. the real Imagenette records
# ---------------------------------------------------------------------

_TFDS_DATA_DIR = os.environ.get("TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets"))
_RECORD_DIR = os.path.join(_TFDS_DATA_DIR, *IMAGENETTE_TFDS_NAME.split("/"), "1.0.0")
_RECORD_SHARDS = sorted(glob(os.path.join(_RECORD_DIR, "imagenette-train.tfrecord-*")))
_SKIP_REASON = (
    f"imagenette TFDS records are not prepared: no `imagenette-train.tfrecord-*` under "
    f"{_RECORD_DIR!r} (TFDS_DATA_DIR={_TFDS_DATA_DIR!r}). This test reads REAL records and "
    f"never downloads anything. A SKIP here is NOT a pass: the real-record guard did not run."
)


def _index_and_image_digest_map_fn(index, image, label):
    """The counter and a per-record fingerprint of the image: `(sum, max, min)`."""
    del label
    digest = tf.stack([tf.reduce_sum(image), tf.reduce_max(image), tf.reduce_min(image)])
    return tf.cast(index, tf.int64), tf.cast(digest, tf.float64)


@pytest.mark.skipif(not _RECORD_SHARDS, reason=_SKIP_REASON)
class TestTheRealImagenetteRecords:
    """A consumer that draws a SMALL fixed sample off the head of the train stream and reports
    a number from it (the DINO k-NN bank) needs the same sample in every process at one seed.
    The records are a seeded permutation, so the same ``--seed`` must give the same window
    across builds and a different ``--seed`` a different one. Bounded to ``.take(2)`` batches
    at 32 px (a few seconds, CPU).
    """

    N_BATCHES = 2

    def _window(self, seed: int):
        streams = load_image_streams(
            "imagenette", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=seed, augment=False,
            indexed_element_map_fn=_index_and_image_digest_map_fn)
        counters, digests = [], []
        for counter, digest in streams.train_ds.take(self.N_BATCHES):
            counters.append(np.asarray(counter))
            digests.append(np.asarray(digest))
        return streams, np.concatenate(counters), np.concatenate(digests)

    def test_the_sizes_are_the_documented_split(self) -> None:
        streams, _, _ = self._window(SEED)
        assert streams.n_train + streams.n_select == IMAGENETTE_TRAIN_SIZE
        assert streams.n_select == IMAGENETTE_TRAIN_SIZE // 10
        assert streams.n_test == IMAGENETTE_VALIDATION_SIZE

    def test_one_seed_reproduces_its_window_and_another_seed_changes_it(self) -> None:
        window = self.N_BATCHES * BATCH_SIZE
        _, counters, digests = self._window(SEED)
        _, counters_again, digests_again = self._window(SEED)
        _, _, other = self._window(SEED + 1)
        assert digests.shape == (window, 3), "an empty window passes every equality below"
        assert np.unique(digests, axis=0).shape[0] == window, "the window replays one record"
        np.testing.assert_array_equal(counters, np.arange(window))
        np.testing.assert_array_equal(counters, counters_again)
        np.testing.assert_array_equal(digests, digests_again)
        assert not np.array_equal(digests, other), "the seed changed nothing"
