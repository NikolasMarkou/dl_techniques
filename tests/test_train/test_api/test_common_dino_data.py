"""The ``dino`` task's data layer: ``train.common.dino_data`` (plan-2026-11-26T224507-e3381378, Phase 4k, Step 4).

A DINO run reads FOUR kinds of stream and the failure of each is silent (a number moves, nothing raises):

1. **The train stream** is the multi-crop element ``(views, label)`` of ``make_stateless_multi_crop_map_fn``, infinite, whole batches only.
   Its shuffle seed AND the seed of its stateless crops follow the stream seed, so a resumed run (``with_stream(completed_steps)``)
   trains on crops it has not seen; a map seed that stayed put would replay the augmentation of epoch 0 on a reshuffled order.
2. **The selection and test streams** are plain, finite ``(image, label)`` batches and a function of the SPLIT seed alone: the same weights
   score the same rows after a kill (a split keyed on the stream seed would move the rows under the weights).
3. **The probe's bank and queries** are cached ``.take(n)`` of the plain fit rows and of the selection rows, built once. The bank is fit
   rows and the queries selection rows, disjoint numerically (a bank drawn from the selection rows lets a query find itself at cosine 1.0
   and score a free hit), and ``with_stream`` leaves them on the original seed so a resumed run's probe stays comparable. The bank is the first
   fit rows in the split's seeded order and so a function of ``seed`` alone, whatever TensorFlow's global seed (D-095).
4. **The stated length.** Keras logs ``n / (n + 1)`` of a loss over a stream of unknown length, so every finite stream states its length
   through the one shared rule (``train.common.pretrain_data.states_its_length``).

The world is the fake CIFAR-10 of ``_dino_world`` (noise rows: every row is its own identity, so "the same rows" is a numerical statement,
read through an oracle that never goes through the loader under test). The tests write nothing.
"""

import dataclasses
import math
import weakref
from typing import Any, Dict, List

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.splits import seeded_split_indices
from train.common import dino_data, pretrain_data
from train.common.classification_data import DEFAULT_VALIDATION_SPLIT
from train.common.image_streams import load_image_streams

from ._dino_world import (  # noqa: F401  (``fake_rows`` is a fixture)
    BANK_BATCHES, BATCH, IMAGE, N_LOCAL, N_TEST, N_TRAIN, N_VIEWS, QUERY_BATCHES, SEED, build, fake_rows, labels_of, nearest_rows, raw_rows)
from .test_beit_tokenizer_task import fresh_process  # the 4j helpers, imported and not copied (D-116, as D-082 did)
from .test_common_tokenizer_data import images_of, standardized

STREAM_SEEDS = (0, 1, 7, 4_000_000_000)

#: The images of the first FIRST_BATCHES train batches of each object, read the moment it was made. The train stream reshuffles on EVERY
#: pass, so only a first pass compares with another object's first pass.
FIRST_PASS: "weakref.WeakKeyDictionary[dino_data.DinoBatches, np.ndarray]" = weakref.WeakKeyDictionary()
FIRST_BATCHES = 2


def built(**changes: Any) -> dino_data.DinoBatches:
    batches = build(**changes)
    FIRST_PASS[batches] = images_of(batches.train_ds, FIRST_BATCHES)
    return batches


def first_pass(batches: dino_data.DinoBatches) -> np.ndarray:
    return FIRST_PASS[batches]


def with_stream_of(batches: dino_data.DinoBatches, completed_steps: int) -> dino_data.DinoBatches:
    resumed = batches.with_stream(completed_steps)
    FIRST_PASS[resumed] = images_of(resumed.train_ds, FIRST_BATCHES)
    return resumed


@pytest.fixture(scope="module")
def fresh(fake_rows) -> dino_data.DinoBatches:
    return built()


@pytest.fixture(scope="module")
def by_stream_seed(fresh) -> Dict[int, dino_data.DinoBatches]:
    return {stream_seed: built(stream_seed=stream_seed) for stream_seed in STREAM_SEEDS}


@pytest.fixture(scope="module")
def split(fake_rows):
    """The oracle: the seeded cut of the fake rows, standardized, straight from the arrays (never through the loader under test)."""
    x_train, y_train, x_test, _ = raw_rows()
    train_idx, select_idx, test_idx = seeded_split_indices(len(x_train), len(x_test), SEED, DEFAULT_VALIDATION_SPLIT, None)
    return {
        "fit": standardized(x_train[train_idx]), "fit_labels": y_train[train_idx],
        "select": standardized(x_train[select_idx]), "select_labels": y_train[select_idx],
        "test": standardized(x_test[test_idx]),
    }


# ---------------------------------------------------------------------
# 1. what the object states
# ---------------------------------------------------------------------

#: What the run sequence, the trainer and the probe read of the data object.
READ_BY_THE_CODE = (
    "train_ds", "select_ds", "test_ds", "probe_bank_ds", "probe_query_ds", "loss_batch_ds", "steps_per_epoch", "val_steps", "test_steps",
    "n_train", "n_select", "n_test", "bank_batches", "query_batches", "input_shape", "n_views", "num_classes",
    "selection_split", "test_split", "seed", "stream_seed", "request",
)


def test_the_batches_state_what_the_code_reads_and_count_the_fake_rows(fresh, split):
    missing = set(READ_BY_THE_CODE) - {field.name for field in dataclasses.fields(fresh)}
    assert not missing, f"DinoBatches has no {sorted(missing)}"
    assert (fresh.n_train, fresh.n_select, fresh.n_test) == (len(split["fit"]), len(split["select"]), N_TEST)
    assert fresh.n_train + fresh.n_select == N_TRAIN
    assert fresh.steps_per_epoch == fresh.n_train // BATCH and fresh.val_steps == math.ceil(fresh.n_select / BATCH)
    assert fresh.test_steps == math.ceil(N_TEST / BATCH)
    assert fresh.input_shape == (IMAGE, IMAGE, 3) and fresh.n_views == N_VIEWS == 2 + N_LOCAL and fresh.num_classes is None
    assert (fresh.seed, fresh.stream_seed) == (SEED, SEED), "no stream seed given: the stream follows the split seed"
    assert f"validation_split={DEFAULT_VALIDATION_SPLIT}" in fresh.selection_split and "test" in fresh.test_split
    assert fresh.request["batch_size"] == BATCH and fresh.request["knn_bank_batches"] == BANK_BATCHES
    with pytest.raises(dataclasses.FrozenInstanceError):
        fresh.n_train = 1  # type: ignore[misc]


def test_the_train_stream_is_the_infinite_multi_crop_element_and_the_other_streams_are_plain_and_state_their_length(fresh):
    views, label = fresh.train_ds.element_spec
    assert views.shape.as_list() == [BATCH, N_VIEWS, IMAGE, IMAGE, 3] and views.dtype == tf.float32 and label.dtype == tf.int32
    assert int(fresh.train_ds.cardinality()) == int(tf.data.INFINITE_CARDINALITY)
    for stream in (fresh.select_ds, fresh.test_ds, fresh.probe_bank_ds, fresh.probe_query_ds):
        image, label = stream.element_spec  # no element map: exactly the loader's (image, label) batches, one crop at the global size
        assert image.shape[1:].as_list() == [IMAGE, IMAGE, 3] and image.dtype == tf.float32 and label.dtype == tf.int32
    # a dataset of unknown length makes Keras log n / (n + 1) of a loss, so every finite stream states its own
    assert int(fresh.select_ds.cardinality()) == fresh.val_steps and int(fresh.test_ds.cardinality()) == fresh.test_steps
    assert int(fresh.probe_bank_ds.cardinality()) == fresh.bank_batches == BANK_BATCHES
    assert int(fresh.probe_query_ds.cardinality()) == fresh.query_batches == min(QUERY_BATCHES, fresh.val_steps)
    sizes = [len(batch[0]) for batch in fresh.select_ds]
    assert sizes == [BATCH] * (fresh.n_select // BATCH) + [fresh.n_select % BATCH], "the ragged last batch of the selection split is kept"
    assert len(next(iter(fresh.train_ds))[0]) == BATCH, "the train stream drops the ragged batch: every step is a whole batch"


def test_the_multi_crop_views_are_global_first_and_every_view_is_the_global_size(fresh):
    views = np.asarray(next(iter(fresh.train_ds))[0])
    assert views.shape == (BATCH, N_VIEWS, IMAGE, IMAGE, 3)
    assert np.isfinite(views).all()
    assert not np.array_equal(views[:, 0], views[:, 1]), "the two global crops of an image are two draws"


def test_the_loss_batch_is_one_multi_crop_batch_of_the_first_selection_rows_and_spends_no_pass_of_any_shuffled_stream(fresh, split):
    views, label = fresh.loss_batch_ds.element_spec
    assert views.shape[1:].as_list() == [N_VIEWS, IMAGE, IMAGE, 3] and views.dtype == tf.float32 and label.dtype == tf.int32
    batches = list(fresh.loss_batch_ds)
    assert len(batches) == 1 and np.asarray(batches[0][0]).shape == (BATCH, N_VIEWS, IMAGE, IMAGE, 3)
    np.testing.assert_array_equal(np.asarray(batches[0][0]), np.asarray(next(iter(fresh.loss_batch_ds))[0]))  # no shuffle: the same every time
    np.testing.assert_array_equal(np.asarray(batches[0][1]), split["select_labels"][:BATCH], err_msg="the labels of the first selection batch")
    assert np.isfinite(np.asarray(batches[0][0])).all()
    # reading it spends no pass of the train stream, and does not touch the bank: a used object's first train pass and its bank are those of a fresh one
    used, other, spent = build(), build(), build()
    for _ in range(3):
        list(used.loss_batch_ds)
    next(iter(spent.train_ds))  # the control: one pass of the train stream itself DOES change the next pass (every pass is a new shuffle)
    first_used, first_other, after_one_pass = (images_of(batches.train_ds, FIRST_BATCHES) for batches in (used, other, spent))
    np.testing.assert_array_equal(first_used, first_other)
    assert not np.array_equal(first_used, after_one_pass)
    np.testing.assert_array_equal(images_of(used.probe_bank_ds), images_of(other.probe_bank_ds))


def test_a_half_read_of_the_bank_cache_refills_the_same_rows_because_the_bank_is_not_shuffled(fake_rows):
    """``.take(1)`` over a cache is a partial pass and tf.data drops the half-filled cache: a SHUFFLED source refilled it with other rows (D-038)."""
    spoiled, whole = build(), build()
    list(spoiled.probe_bank_ds.take(1))
    np.testing.assert_array_equal(images_of(spoiled.probe_bank_ds), images_of(whole.probe_bank_ds), err_msg="a half read changed the bank")
    np.testing.assert_array_equal(images_of(whole.probe_bank_ds), images_of(whole.probe_bank_ds))


def test_the_probe_query_count_is_the_asked_count_up_to_the_selection_batches(fake_rows):
    batches = build(knn_query_batches=32)
    assert batches.query_batches == batches.val_steps == 3
    assert int(batches.probe_query_ds.cardinality()) == 3
    assert batches.bank_batches == BANK_BATCHES


# ---------------------------------------------------------------------
# 2. the split follows the split seed, the train stream follows the stream seed
# ---------------------------------------------------------------------

def test_the_selection_and_test_rows_are_the_seeded_cut_of_the_real_rows(fresh, split):
    np.testing.assert_allclose(images_of(fresh.select_ds), split["select"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(images_of(fresh.test_ds), split["test"], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(labels_of(fresh.select_ds), split["select_labels"])


@pytest.mark.parametrize("stream_seed", STREAM_SEEDS)
def test_the_selection_and_test_rows_do_not_follow_the_stream_seed_but_the_train_stream_does(fresh, by_stream_seed, stream_seed):
    other = by_stream_seed[stream_seed]
    assert (other.seed, other.stream_seed) == (SEED, stream_seed), "0 is a stream seed, not 'unset'"
    assert (other.n_train, other.n_select, other.n_test) == (fresh.n_train, fresh.n_select, fresh.n_test)
    np.testing.assert_array_equal(images_of(other.select_ds), images_of(fresh.select_ds))
    np.testing.assert_array_equal(images_of(other.test_ds), images_of(fresh.test_ds))
    np.testing.assert_array_equal(labels_of(other.select_ds), labels_of(fresh.select_ds))
    assert not np.array_equal(first_pass(other), first_pass(fresh)), "the train stream is another one"


def test_two_builds_of_one_stream_seed_agree_so_the_difference_above_is_the_seed(fresh, by_stream_seed):
    np.testing.assert_array_equal(first_pass(built(stream_seed=STREAM_SEEDS[2])), first_pass(by_stream_seed[STREAM_SEEDS[2]]))
    np.testing.assert_array_equal(first_pass(built()), first_pass(fresh))
    assert len({first_pass(by_stream_seed[seed]).tobytes() for seed in STREAM_SEEDS}) == len(STREAM_SEEDS), "four seeds, four streams"


def test_with_stream_is_a_fresh_train_stream_that_follows_the_completed_steps_on_the_same_split(fresh):
    k = 120
    resumed, again, later, at_zero = (with_stream_of(fresh, n) for n in (k, k, k + 1, 0))
    assert resumed.stream_seed == pretrain_data.derive_stream_seed(SEED, k) != fresh.stream_seed
    assert resumed.seed == SEED and resumed.request == {**fresh.request, "stream_seed": resumed.stream_seed}
    np.testing.assert_array_equal(first_pass(resumed), first_pass(again))  # the control: one count, one stream
    assert not np.array_equal(first_pass(resumed), first_pass(fresh)), "a resume that replays the fresh stream trains epoch 0 again"
    assert not np.array_equal(first_pass(resumed), first_pass(later)), "another count, another stream"
    assert not np.array_equal(first_pass(at_zero), first_pass(fresh)), "a resume at step 0 is still not the fresh run"
    assert (resumed.n_train, resumed.n_select, resumed.n_test) == (fresh.n_train, fresh.n_select, fresh.n_test)
    assert (resumed.steps_per_epoch, resumed.val_steps) == (fresh.steps_per_epoch, fresh.val_steps)
    np.testing.assert_array_equal(images_of(resumed.select_ds), images_of(fresh.select_ds))
    np.testing.assert_array_equal(images_of(resumed.test_ds), images_of(fresh.test_ds))
    # a resume of a resume is a function of the split seed and the count, never of the stream it came from
    assert resumed.with_stream(5).stream_seed == fresh.with_stream(5).stream_seed


@pytest.fixture
def spied(fake_rows, monkeypatch):
    """Every ``load_image_streams`` call and every multi-crop map seed of a build, in order."""
    calls: List[Dict[str, Any]] = []
    map_seeds: List[int] = []
    real_load, real_map = dino_data.load_image_streams, dino_data.make_stateless_multi_crop_map_fn

    def load(*args: Any, **kwargs: Any):
        calls.append({"args": args, **kwargs})
        return real_load(*args, **kwargs)

    def make_map(*args: Any, **kwargs: Any):
        map_seeds.append(kwargs["seed"])
        return real_map(*args, **kwargs)

    monkeypatch.setattr(dino_data, "load_image_streams", load)
    monkeypatch.setattr(dino_data, "make_stateless_multi_crop_map_fn", make_map)
    return calls, map_seeds


def test_a_resume_moves_the_shuffle_seed_and_the_stateless_crop_seed_and_never_the_split_seed(spied):
    calls, map_seeds = spied
    fresh = build()
    k = 120
    derived = pretrain_data.derive_stream_seed(SEED, k)
    fresh.with_stream(k)
    assert len(calls) == 4, "two loads per build: the multi-crop train stream and the plain streams"
    assert map_seeds == [SEED, derived], "the crops of a fresh run are keyed on the seed, those of a resume on the derived stream seed"
    assert all(call["seed"] == SEED for call in calls), "every load is split on the split seed alone"
    train_calls = [call for call in calls if call.get("indexed_element_map_fn") is not None]
    plain_calls = [call for call in calls if call.get("indexed_element_map_fn") is None]
    assert [call["stream_seed"] for call in train_calls] == [SEED, derived], "the shuffle follows the stream seed"
    assert all(call["stream_seed"] in (None, SEED) for call in plain_calls), "the plain streams, the probe's among them, never move"
    assert all(call["augment"] is False for call in calls), "the DINO recipe's augmentation is the multi-crop map alone"


# ---------------------------------------------------------------------
# 3. the probe streams
# ---------------------------------------------------------------------

def test_the_bank_is_fit_rows_and_the_queries_are_selection_rows_and_the_two_are_disjoint_numerically(fresh, split):
    bank = images_of(fresh.probe_bank_ds)
    assert len(bank) == BANK_BATCHES * BATCH
    in_fit, to_fit = nearest_rows(bank, split["fit"])
    in_select, to_select = nearest_rows(bank, split["select"])
    assert to_fit.max() < 1e-3, "every bank row is a fit row"
    assert in_select.size and to_select.min() > 1.0, "no bank row is a selection row (the nearest selection row is far away)"
    assert len(set(in_fit.tolist())) == len(bank), "forty-eight batches of the seeded fit order (198 rows) hold no row twice"
    np.testing.assert_array_equal(labels_of(fresh.probe_bank_ds), split["fit_labels"][in_fit], err_msg="the labels travel with the rows")

    queries = images_of(fresh.probe_query_ds)
    np.testing.assert_allclose(queries, split["select"][: len(queries)], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(labels_of(fresh.probe_query_ds), split["select_labels"][: len(queries)])
    _, query_to_bank = nearest_rows(queries, bank)
    _, query_to_fit = nearest_rows(queries, split["fit"])
    assert query_to_bank.min() > 1.0 and query_to_fit.min() > 1.0, "disjoint numerically: no query is a bank row or a fit row"
    assert in_fit.max() < len(split["fit"]), "the control: the oracle's own rows are what the bank matched"


def test_the_probe_rows_are_the_same_on_every_pass_because_they_are_cached(fresh):
    np.testing.assert_array_equal(images_of(fresh.probe_bank_ds), images_of(fresh.probe_bank_ds))
    np.testing.assert_array_equal(images_of(fresh.probe_query_ds), images_of(fresh.probe_query_ds))


@pytest.mark.parametrize("completed", [0, 120, 121])
def test_with_stream_leaves_the_probe_rows_untouched(fresh, completed):
    resumed = fresh.with_stream(completed)
    np.testing.assert_array_equal(images_of(resumed.probe_bank_ds), images_of(fresh.probe_bank_ds))
    np.testing.assert_array_equal(labels_of(resumed.probe_bank_ds), labels_of(fresh.probe_bank_ds))
    np.testing.assert_array_equal(images_of(resumed.probe_query_ds), images_of(fresh.probe_query_ds))
    assert (resumed.bank_batches, resumed.query_batches) == (fresh.bank_batches, fresh.query_batches)


def test_the_bank_follows_the_split_seed_so_another_seed_has_another_bank(fresh, fake_rows):
    other = build(seed=SEED + 1)
    assert not np.array_equal(images_of(other.probe_bank_ds), images_of(fresh.probe_bank_ds)), "the control: the bank is a function of the seed"


def test_the_bank_is_a_function_of_the_seed_alone_the_first_fit_rows_in_their_seeded_order_whatever_the_global_seed(fake_rows, split):
    """D-095: the first pass of a seeded ``shuffle`` follows TensorFlow's global seed too (review-iter-1 concern 2); the seeded order does not."""
    n_bank = BANK_BATCHES * BATCH
    try:
        banks = {}
        for global_seed in (None, 42, 7):
            tf.random.set_seed(global_seed)
            banks[global_seed] = built_bank()
        # The control: the loader's own shuffled fit stream IS moved by the global seed, so the instrument can see the dependence.
        shuffled = {}
        for global_seed in (42, 7):
            tf.random.set_seed(global_seed)
            streams = load_image_streams(
                "cifar10", image_size=IMAGE, batch_size=BATCH, seed=SEED, augment=False, shuffle_train=True)
            shuffled[global_seed] = images_of(streams.train_ds, BANK_BATCHES)
    finally:
        tf.random.set_seed(None)
    assert not np.array_equal(shuffled[42], shuffled[7]), "the control: a shuffled fit stream follows the global seed"
    for global_seed, (rows, labels) in banks.items():
        np.testing.assert_array_equal(rows, banks[None][0], err_msg=f"the bank moved under global seed {global_seed}")
        np.testing.assert_array_equal(labels, banks[None][1])
    rows, labels = banks[None]
    np.testing.assert_array_equal(rows, split["fit"][:n_bank], err_msg="the bank is the first fit rows in the seeded permutation's order")
    np.testing.assert_array_equal(labels, split["fit_labels"][:n_bank])
    tf.random.set_seed(7)
    try:
        other = built_bank(seed=SEED + 1)
    finally:
        tf.random.set_seed(None)
    assert not np.array_equal(other[0], rows), "another seed is another bank, under any global seed"


def built_bank(**changes):
    """``(images, labels)`` of a fresh object's bank, read the moment it is made."""
    batches = build(**changes)
    return images_of(batches.probe_bank_ds), labels_of(batches.probe_bank_ds)


# ---------------------------------------------------------------------
# 4. refusals
# ---------------------------------------------------------------------

def test_a_seed_that_is_not_an_int_is_refused_before_any_data_is_read(monkeypatch):
    def boom(*args: Any, **kwargs: Any):
        raise AssertionError("the data was read before the request was checked")

    monkeypatch.setattr(dino_data, "load_image_streams", boom)
    for changes in ({"seed": True}, {"seed": 1.5}, {"stream_seed": True}, {"stream_seed": 1.5}, {"stream_seed": "7"}):
        with pytest.raises(TypeError, match="int"):
            build(**changes)
    for changes in ({"n_local_crops": -1}, {"knn_bank_batches": 0}, {"knn_query_batches": -2}):
        with pytest.raises(ValueError):
            build(**changes)
    for changes in ({"n_local_crops": 2.0}, {"n_local_crops": True}, {"knn_bank_batches": "6"}):
        with pytest.raises(TypeError, match="int"):
            build(**changes)


@pytest.mark.parametrize("steps, error", [(None, TypeError), (True, TypeError), (2.0, TypeError), (-1, ValueError)])
def test_with_stream_refuses_what_is_not_a_count_of_steps(fresh, steps, error):
    with pytest.raises(error):  # None is not "a fresh run": a resume asked for a stream by a count it does not know
        fresh.with_stream(steps)


@pytest.mark.parametrize("stream", ["select_ds", "test_ds"])
def test_a_stream_of_unknown_length_is_refused_by_name_stating_both_lengths(fresh, stream, monkeypatch):
    real = dino_data.load_image_streams

    def lying(*args: Any, **kwargs: Any):
        streams = real(*args, **kwargs)
        return dataclasses.replace(streams, **{stream: getattr(streams, stream).filter(lambda *_: True)})

    monkeypatch.setattr(dino_data, "load_image_streams", lying)
    with pytest.raises(ValueError) as caught:
        build()
    assert f"{stream} has cardinality unknown" in str(caught.value) and str(int(getattr(fresh, stream).cardinality())) in str(caught.value)


def test_the_two_shared_definitions_are_imported_from_pretrain_data_and_not_copied():
    assert dino_data.derive_stream_seed is pretrain_data.derive_stream_seed
    assert dino_data.states_its_length is pretrain_data.states_its_length


# ---------------------------------------------------------------------
# 5. the import rule
# ---------------------------------------------------------------------

def test_the_front_door_does_not_load_the_dino_modules():
    """``train.api`` may import the data and run modules inside functions only: loading the package, the CLI and the task table loads neither."""
    done = fresh_process(
        "import sys\n"
        "import train.api, train.api.cli, train.api.tasks\n"
        "print('LOADED', sorted(m for m in ('train.common.dino_data', 'train.common.dino_run') if m in sys.modules))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "LOADED []"


def test_importing_the_two_modules_initialises_no_tensorflow_device():
    """``setup_gpu`` must still be able to enable memory growth after the modules are imported: no op may run at import."""
    done = fresh_process(
        "import keras\n"
        "from tensorflow.python.eager import context as eager\n"
        "import train.common.dino_data, train.common.dino_run\n"
        "before = eager.context()._context_handle is not None\n"
        "keras.ops.zeros((1,))\n"
        "print('INITIALISED', before, 'CONTROL', eager.context()._context_handle is not None)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "INITIALISED False CONTROL True", "the control: an eager op does initialise the context"
