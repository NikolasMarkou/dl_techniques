"""The tokenizer's data layer and its floor function: ``train.common.tokenizer_data`` and two functions of ``train.common.pretrain_run``.

The ``beit_tokenizer`` task of ``train.api`` (plan-2026-11-25T124419-fa94f59c, Phase 4j) trains the library ``VQVAERotationTrick`` on plain
``(image, label)`` batches and judges it twice: its reconstruction against a constant image, and its codebook against a floor of effective
codes. What can go wrong here is silent (a selection split that follows the stream seed, a baseline in the wrong reduction, a floor that
refuses exactly 64 codes), so each rule is pinned by a test that sees its failure:

1. **The rows.** The batches are the REAL CIFAR-10 rows, capped by ``max_samples`` (the cap bounds the train pool and never the test split). The
   selection and test streams are a function of the split seed alone: they are equal across stream seeds, and the train stream is not.
   ``with_stream(k)`` is a fresh train stream, another one for another ``k``, on the same split.
2. **The baseline.** ``constant_image_baseline`` is the lower of the best per-channel constant and the in-sample mean image, scored in the
   metric's own reduction (the mean over batches of the per-batch pixel mean, a ragged last batch counting as much as a full one). It equals the
   ``evaluate`` of a ``VQVAERotationTrick`` that has a bring-your-own constant decoder within the repo's float32 tie band, which the band
   test prints; a pooled pixel mean is off by orders of magnitude more than that band, so the band can tell the two apart. Hand-computed and planted
   datasets pin the arithmetic and the kind.
3. **The code ids.** ``code_ids_of`` is one eager pass over EVERY selection position, in row order.
4. **The floor.** ``usage_meets_floor`` passes exactly 64 equally used codes (``exp(log(64))`` is 63.99999999999998, so a bare compare of the
   perplexity refuses them), refuses 63, refuses the 1-epoch tokenizer the floor was measured on and passes the 5-epoch one, and reads the
   module's floor when it is called. ``checked_tokenizer_usage`` raises the message it always raised, whichever command it names.
5. **The epoch-zero key.** ``epoch_zero_val_loss`` reads ``initial.metrics.<key>`` of the fit sidecar; with no key it reads ``loss``, as it did.

Sizes: 700 rows of the real CIFAR-10 (630 fit rows, 70 selection rows in batches of 32, 32 and 6, the whole test split), 32 px, batch 32, no
augmentation (so two builds of one seed compare exactly). The tests write nothing: no model is saved and no run directory is made.
"""

import dataclasses
import math
import re
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.splits import seeded_split_indices
from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick
from train.common import classification_run, pretrain_data, pretrain_run
from train.common import tokenizer_data as td
from train.common.classification_data import DEFAULT_VALIDATION_SPLIT
from train.common.forecast_run import tie_band
from train.common.image_streams import stream_statistics
from train.common.pretrain_data import TokenizerUsage, derive_stream_seed, tokenizer_usage
from train.common.pretrain_run import MIN_TOKENIZER_PERPLEXITY, checked_tokenizer_usage, epoch_zero_val_loss, usage_meets_floor
from train.common.resume_state import FIT_STATE_NAME, update_fit_state

from ._tokenizer_world import cell_index_codes, cifar, constant_model, steps_between, with_image_wide_bits  # noqa: F401  (``cifar`` is a fixture)

IMAGE = 32
BATCH = 32
SEED = 42
CAP = 700                              # rows of the train pool: 630 fit rows and 70 selection rows
STREAM_SEEDS = (0, 1, 7, 4_000_000_000)
N_TEST = 10_000                        # the whole CIFAR-10 test split: the cap never touches it
GRID = IMAGE // 4                      # the tiny models below down-sample by 4


#: The images of the first FIRST_BATCHES train batches of each object ``build`` made, read the moment it was made. The train stream
#: reshuffles on EVERY pass, so only a first pass compares with another object's first pass; a later pass of the same object is a
#: different order even for the same seed (which would make "the stream seed changes the stream" true of a seed that did not change).
FIRST_PASS: "weakref.WeakKeyDictionary[td.TokenizerBatches, np.ndarray]" = weakref.WeakKeyDictionary()
FIRST_BATCHES = 2


def build(**changes: Any) -> td.TokenizerBatches:
    """``load_tokenizer_batches`` at the test sizes (un-augmented, so two builds of one seed compare exactly), its first train pass recorded."""
    settings: Dict[str, Any] = dict(
        dataset="cifar10", image_size=IMAGE, batch_size=BATCH, seed=SEED, stream_seed=None,
        validation_split=None, max_samples=CAP, augment=False)
    settings.update(changes)
    batches = td.load_tokenizer_batches(**settings)
    FIRST_PASS[batches] = images_of(batches.train_ds, FIRST_BATCHES)
    return batches


def first_pass(batches: td.TokenizerBatches) -> np.ndarray:
    """The first train pass of ``batches`` (recorded when it was built through ``build`` or ``with_stream_of``)."""
    return FIRST_PASS[batches]


def with_stream_of(batches: td.TokenizerBatches, completed_steps: int) -> td.TokenizerBatches:
    """``batches.with_stream(completed_steps)``, its first train pass recorded."""
    resumed = batches.with_stream(completed_steps)
    FIRST_PASS[resumed] = images_of(resumed.train_ds, FIRST_BATCHES)
    return resumed


def images_of(ds: tf.data.Dataset, batches: Optional[int] = None) -> np.ndarray:
    """The images of ``batches`` batches of ``ds`` (the whole finite pass when ``None``), as one array."""
    taken = ds if batches is None else ds.take(batches)
    return np.concatenate([np.asarray(batch[0]) for batch in taken])


@pytest.fixture(scope="module")
def fresh(cifar) -> td.TokenizerBatches:
    return build()


@pytest.fixture(scope="module")
def by_stream_seed(fresh) -> Dict[int, td.TokenizerBatches]:
    return {stream_seed: build(stream_seed=stream_seed) for stream_seed in STREAM_SEEDS}


@pytest.fixture(scope="module")
def raw_rows(cifar):
    """The raw CIFAR-10 arrays ``(x_train, x_test)``, uint8, read straight from the Keras cache (never through the loader under test)."""
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    return x_train, x_test


def standardized(rows: np.ndarray) -> np.ndarray:
    """uint8 rows to the streams' input: ``(rows / 255 - mean) / std`` per channel, in float32."""
    mean, std = stream_statistics("cifar10")
    return (rows.astype(np.float32) / 255.0 - mean) / std


# ---------------------------------------------------------------------
# 1. the rows
# ---------------------------------------------------------------------

#: The fields of the data object the shared sequence and the trainer read (``train.api.trainer``, ``run_sequence``, ``fit_callbacks``,
#: ``resume_decisions``) beside the three streams; a task with no classes holds ``num_classes = None`` (``check_holds_no_classes``).
READ_BY_THE_SEQUENCE = (
    "train_ds", "select_ds", "test_ds", "steps_per_epoch", "val_steps", "n_train", "n_select", "n_test", "input_shape",
    "num_classes", "selection_split", "test_split", "seed", "stream_seed", "request",
)


def test_the_batches_state_what_the_sequence_reads_and_the_real_rows_are_capped(fresh, raw_rows):
    missing = set(READ_BY_THE_SEQUENCE) - {field.name for field in dataclasses.fields(fresh)}
    assert not missing, f"TokenizerBatches has no {sorted(missing)}: the run sequence reads them"
    x_train, x_test = raw_rows
    n_select = int(CAP * DEFAULT_VALIDATION_SPLIT)
    # the cap bounds the train pool (fit and selection together) and never the test split
    assert (fresh.n_train, fresh.n_select, fresh.n_test) == (CAP - n_select, n_select, len(x_test))
    assert fresh.n_test == N_TEST and fresh.n_train + fresh.n_select == CAP < len(x_train)
    assert fresh.steps_per_epoch == fresh.n_train // BATCH and fresh.val_steps == math.ceil(fresh.n_select / BATCH)
    assert fresh.input_shape == (IMAGE, IMAGE, 3) and fresh.num_classes is None
    assert (fresh.seed, fresh.stream_seed) == (SEED, SEED), "no stream seed given: the stream follows the split seed"
    assert f"validation_split={DEFAULT_VALIDATION_SPLIT}" in fresh.selection_split and "test" in fresh.test_split
    assert fresh.request["max_samples"] == CAP and fresh.request["batch_size"] == BATCH
    with pytest.raises(dataclasses.FrozenInstanceError):
        fresh.n_train = 1  # type: ignore[misc]


def test_the_streams_are_plain_image_label_batches_that_state_their_length(fresh):
    for ds, batch in ((fresh.train_ds, BATCH), (fresh.select_ds, None), (fresh.test_ds, None)):
        image, label = ds.element_spec  # no element map: exactly the loader's (image, label); only the train stream drops the ragged batch
        assert image.shape.as_list() == [batch, IMAGE, IMAGE, 3] and image.dtype == tf.float32 and label.dtype == tf.int32
    assert int(fresh.train_ds.cardinality()) == int(tf.data.INFINITE_CARDINALITY)
    # Keras logs n / (n + 1) of a loss over a dataset of unknown length, so the finite streams state theirs
    assert int(fresh.select_ds.cardinality()) == fresh.val_steps
    assert int(fresh.test_ds.cardinality()) == math.ceil(fresh.n_test / BATCH)
    sizes = [len(batch[0]) for batch in fresh.select_ds]
    assert sizes == [BATCH] * (fresh.n_select // BATCH) + [fresh.n_select % BATCH], "the ragged last batch is kept"
    assert sum(sizes) == fresh.n_select
    train_batch = next(iter(fresh.train_ds))
    assert len(train_batch[0]) == BATCH, "the train stream drops the ragged batch: every step is a whole batch"


def test_the_selection_and_test_rows_are_the_seeded_cut_of_the_real_rows(fresh, raw_rows):
    x_train, x_test = raw_rows
    _, select_idx, test_idx = seeded_split_indices(len(x_train), len(x_test), SEED, DEFAULT_VALIDATION_SPLIT, CAP)
    np.testing.assert_allclose(images_of(fresh.select_ds), standardized(x_train[select_idx]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(images_of(fresh.test_ds, 4), standardized(x_test[test_idx[:4 * BATCH]]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("stream_seed", STREAM_SEEDS)
def test_the_selection_and_test_rows_do_not_follow_the_stream_seed_but_the_train_stream_does(fresh, by_stream_seed, stream_seed):
    other = by_stream_seed[stream_seed]
    assert (other.seed, other.stream_seed) == (SEED, stream_seed), "0 is a stream seed, not 'unset'"
    assert (other.n_train, other.n_select, other.n_test) == (fresh.n_train, fresh.n_select, fresh.n_test)
    # a function of the split seed only: the same rows in the same order
    np.testing.assert_array_equal(images_of(other.select_ds), images_of(fresh.select_ds))
    np.testing.assert_array_equal(images_of(other.test_ds, 4), images_of(fresh.test_ds, 4))
    # ... and the train stream is another one (the control below: one seed, one stream)
    assert not np.array_equal(first_pass(other), first_pass(fresh))


def test_two_builds_of_one_stream_seed_agree_so_the_difference_above_is_the_seed(fresh, by_stream_seed):
    np.testing.assert_array_equal(first_pass(build(stream_seed=STREAM_SEEDS[2])), first_pass(by_stream_seed[STREAM_SEEDS[2]]))
    np.testing.assert_array_equal(first_pass(build()), first_pass(fresh))
    assert len({first_pass(by_stream_seed[seed]).tobytes() for seed in STREAM_SEEDS}) == len(STREAM_SEEDS), "four seeds, four streams"


def test_with_stream_is_a_fresh_train_stream_that_follows_the_completed_steps_on_the_same_split(fresh):
    k = 120
    resumed, again, later, at_zero = (with_stream_of(fresh, n) for n in (k, k, k + 1, 0))
    assert resumed.stream_seed == derive_stream_seed(SEED, k) and resumed.stream_seed != fresh.stream_seed
    assert resumed.seed == SEED and resumed.request == {**fresh.request, "stream_seed": resumed.stream_seed}
    np.testing.assert_array_equal(first_pass(resumed), first_pass(again))  # the control: one count, one stream
    assert not np.array_equal(first_pass(resumed), first_pass(fresh)), "a resume that replays the fresh stream trains epoch 0 again"
    assert not np.array_equal(first_pass(resumed), first_pass(later)), "another count, another stream"
    assert not np.array_equal(first_pass(at_zero), first_pass(fresh)), "a resume at step 0 is still not the fresh run"
    # the split, the counts and the finite streams are the fresh run's
    assert (resumed.n_train, resumed.n_select, resumed.n_test) == (fresh.n_train, fresh.n_select, fresh.n_test)
    assert (resumed.steps_per_epoch, resumed.val_steps) == (fresh.steps_per_epoch, fresh.val_steps)
    np.testing.assert_array_equal(images_of(resumed.select_ds), images_of(fresh.select_ds))
    # the stream of a resume of a resume is a function of the split seed and the count, never of the stream it came from
    assert resumed.with_stream(5).stream_seed == fresh.with_stream(5).stream_seed


def test_a_seed_that_is_not_an_int_is_refused_before_any_data_is_read(monkeypatch):
    def boom(*args: Any, **kwargs: Any):
        raise AssertionError("the data was read before the seed was checked")

    monkeypatch.setattr(td, "load_image_streams", boom)
    for changes in ({"seed": True}, {"seed": 1.5}, {"stream_seed": True}, {"stream_seed": 1.5}, {"stream_seed": "7"}):
        with pytest.raises(TypeError, match="int"):
            build(**changes)


@pytest.mark.parametrize("steps, error", [(None, TypeError), (True, TypeError), (2.0, TypeError), (-1, ValueError)])
def test_with_stream_refuses_what_is_not_a_count_of_steps(fresh, steps, error):
    with pytest.raises(error):  # None is not "a fresh run": a resume asked for a stream by a count it does not know
        fresh.with_stream(steps)


@pytest.mark.parametrize("stream", ["select_ds", "test_ds"])
def test_a_stream_of_unknown_length_is_refused_by_name_stating_both_lengths(fresh, stream, monkeypatch):
    # Both finite streams are scored by Keras, which logs n / (n + 1) of the loss over a dataset of unknown length: the select arm and the
    # test arm are two lines of the loader, so each is held by its own row.
    real = td.load_image_streams

    def lying(*args: Any, **kwargs: Any):
        streams = real(*args, **kwargs)
        return dataclasses.replace(streams, **{stream: getattr(streams, stream).filter(lambda *_: True)})

    monkeypatch.setattr(td, "load_image_streams", lying)
    with pytest.raises(ValueError) as caught:
        build()
    assert f"{stream} has cardinality unknown" in str(caught.value) and str(int(getattr(fresh, stream).cardinality())) in str(caught.value)


def test_the_length_check_is_one_public_function_of_pretrain_data_that_this_module_calls_and_does_not_copy():
    assert not hasattr(pretrain_data, "_states_its_length"), "renamed, with no alias of the private name"
    assert td.states_its_length is pretrain_data.states_its_length
    assert td.derive_stream_seed is pretrain_data.derive_stream_seed
    ds = tf.data.Dataset.range(6).batch(2)
    pretrain_data.states_its_length("sample", ds, 3)
    with pytest.raises(ValueError, match=r"sample has cardinality 3, but one pass is 4 batches"):
        pretrain_data.states_its_length("sample", ds, 4)
    with pytest.raises(ValueError, match=r"sample has cardinality unknown, but one pass is 3 batches"):
        pretrain_data.states_its_length("sample", ds.filter(lambda _: True), 3)


# ---------------------------------------------------------------------
# 2. the baseline
# ---------------------------------------------------------------------


def one_pass_arrays(ds: tf.data.Dataset) -> List[np.ndarray]:
    return [np.asarray(batch[0], np.float64) for batch in ds]


def test_the_baseline_equals_the_evaluate_of_a_constant_model_within_the_tie_band(fresh, raw_rows):
    x_train, _ = raw_rows
    _, select_idx, _ = seeded_split_indices(len(x_train), N_TEST, SEED, DEFAULT_VALIDATION_SPLIT, CAP)
    rows = standardized(x_train[select_idx]).astype(np.float64)  # the independent oracle: numpy, straight from the raw rows
    constants = {td.CHANNEL_MEAN: rows.mean(axis=(0, 1, 2)), td.MEAN_IMAGE: rows.mean(axis=0)}
    baseline = td.constant_image_baseline(fresh.select_ds)
    reported = {td.CHANNEL_MEAN: baseline.channel_mean, td.MEAN_IMAGE: baseline.mean_image}
    band = tie_band(fresh.val_steps)  # float32 steps of the baseline: the band `beats_bar` clears a bar by
    for kind, constant in constants.items():
        measured = classification_run.evaluate_dict(constant_model(constant), fresh.select_ds)["reconstruction_loss"]
        steps = steps_between(measured, reported[kind])
        print(f"BAND {kind}: baseline {reported[kind]!r}  evaluate {measured!r}  difference {steps:.2f} float32 steps  "
              f"band {band} steps ({band * float(np.spacing(np.float32(reported[kind]))):.3e})  batches {fresh.val_steps}")
        assert steps <= band, f"the {kind} baseline is {steps:.2f} float32 steps from the model's own evaluate, the band is {band}"
    smaller = min(reported, key=reported.get)
    assert baseline.kind == smaller == td.MEAN_IMAGE, "in-sample, the mean image is the smaller term on these rows"
    assert baseline.reconstruction_loss == reported[smaller] == float(np.float32(reported[smaller])), "float32, the smaller of the two"
    assert baseline.mean_image < baseline.channel_mean


def test_a_pooled_pixel_mean_is_not_the_metrics_reduction_and_the_band_can_tell(fresh):
    # The metric adds one per-batch pixel mean per batch (Keras `Mean` of a scalar), so the ragged last batch of 6 rows counts as much as a
    # whole one of 32. A pooled mean over all 70 rows weights it by its rows. The two differ by far more than the band the test above holds.
    batches = one_pass_arrays(fresh.select_ds)
    assert len({len(batch) for batch in batches}) > 1, "the instrument has a ragged last batch to see"
    image = np.concatenate(batches).mean(axis=0)
    pooled = float(np.mean(np.concatenate([(batch - image) ** 2 for batch in batches])))
    by_batches = float(np.mean([np.mean((batch - image) ** 2) for batch in batches]))
    baseline = td.constant_image_baseline(fresh.select_ds)
    band = tie_band(fresh.val_steps) * float(np.spacing(np.float32(by_batches)))
    print(f"BAND control: pooled {pooled!r}  mean over batches {by_batches!r}  baseline {baseline.mean_image!r}  "
          f"pooled is {abs(pooled - by_batches) / band:.0f} bands away")
    assert abs(pooled - by_batches) > 100 * band
    assert abs(baseline.mean_image - by_batches) <= band and abs(baseline.mean_image - pooled) > 100 * band


def dataset_of(rows: np.ndarray, batch: int) -> tf.data.Dataset:
    """A finite ``(image, label)`` stream of ``rows``, the way the loader batches them (a ragged last batch kept)."""
    return tf.data.Dataset.from_tensor_slices((np.asarray(rows, np.float32), np.zeros(len(rows), np.int32))).batch(batch)


def test_the_baseline_is_the_mean_over_batches_by_hand():
    # One pixel, one channel, rows 0, 2 and 10 in batches of 2 and 1: the constant is 4; the batch means are (16 + 4) / 2 = 10 and 36, so the
    # metric reads (10 + 36) / 2 = 23. A pooled mean reads (16 + 4 + 36) / 3 = 18.67.
    baseline = td.constant_image_baseline(dataset_of(np.array([0.0, 2.0, 10.0]).reshape(3, 1, 1, 1), batch=2))
    assert baseline.reconstruction_loss == 23.0
    assert (baseline.channel_mean, baseline.mean_image) == (23.0, 23.0)
    assert baseline.kind == td.CHANNEL_MEAN, "a tie names the simpler constant"


def test_the_mean_image_is_the_smaller_term_on_a_planted_dataset_where_it_should_be():
    # Ten copies of one image that is not flat: the mean image IS the image, so it scores exactly 0; no per-channel constant can.
    image = np.random.RandomState(0).normal(size=(4, 4, 3)).astype(np.float32)
    baseline = td.constant_image_baseline(dataset_of(np.stack([image] * 10), batch=4))
    assert baseline.kind == td.MEAN_IMAGE
    assert baseline.mean_image == 0.0 == baseline.reconstruction_loss
    assert baseline.channel_mean > 0.1, "the control: the same rows against their channel means are far from 0"
    assert baseline.mean_image < baseline.channel_mean


def test_a_dataset_the_two_constants_fit_equally_names_the_channel_mean():
    flat = np.broadcast_to(np.array([0.25, 0.5, 0.75], np.float32), (4, 4, 3))
    baseline = td.constant_image_baseline(dataset_of(np.stack([flat] * 6), batch=4))
    assert (baseline.channel_mean, baseline.mean_image, baseline.reconstruction_loss) == (0.0, 0.0, 0.0)
    assert baseline.kind == td.CHANNEL_MEAN


def test_the_baseline_is_a_plain_record_of_numbers():
    baseline = td.constant_image_baseline(dataset_of(np.arange(8.0).reshape(8, 1, 1, 1), batch=3))
    assert baseline.as_dict() == {"kind": baseline.kind, "reconstruction_loss": baseline.reconstruction_loss,
                                  "channel_mean": baseline.channel_mean, "mean_image": baseline.mean_image}
    assert all(isinstance(baseline.as_dict()[key], float) for key in ("reconstruction_loss", "channel_mean", "mean_image"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        baseline.kind = "other"  # type: ignore[misc]


@pytest.mark.parametrize("function", ["constant_image_baseline", "code_ids_of"])
def test_a_pass_over_an_empty_or_an_endless_stream_is_refused_by_name(function, tiny_model):
    calls = {"constant_image_baseline": td.constant_image_baseline, "code_ids_of": lambda ds: td.code_ids_of(tiny_model, ds)}
    empty = dataset_of(np.zeros((0, IMAGE, IMAGE, 3), np.float32), batch=4)
    endless = dataset_of(np.ones((4, IMAGE, IMAGE, 3), np.float32), batch=4).repeat()
    with pytest.raises(ValueError, match="empty"):
        calls[function](empty)
    with pytest.raises(ValueError, match="endless"):
        calls[function](endless)


# ---------------------------------------------------------------------
# 3. the code ids
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model() -> VQVAERotationTrick:
    """A tiny auto-built tokenizer at its initial weights (16 codes, an 8 x 8 grid at 32 px)."""
    keras.utils.set_random_seed(0)
    model = VQVAERotationTrick(num_embeddings=16, embedding_dim=8, input_shape=(IMAGE, IMAGE, 3), hidden_channels=8,
                               downsample_factor=4, num_res_blocks=1)
    model.build((None, IMAGE, IMAGE, 3))
    return model


def test_code_ids_cover_every_selection_position_in_row_order(fresh, tiny_model):
    ids = td.code_ids_of(tiny_model, fresh.select_ds)
    assert ids.shape == (fresh.n_select, GRID, GRID) and ids.dtype.kind in "iu"
    assert 0 <= ids.min() and ids.max() < tiny_model.num_embeddings
    # one row at a time, in the stream's order: the pass is not a function of how the rows are batched
    rows = images_of(fresh.select_ds)
    one_by_one = np.concatenate([np.asarray(tiny_model.encode_to_indices(rows[i:i + 1])) for i in range(len(rows))])
    np.testing.assert_array_equal(ids, one_by_one)
    assert len(np.unique(ids)) > 2, "vacuity guard: the toy tokenizer uses several codes, so a wrong order would show"
    np.testing.assert_array_equal(td.code_ids_of(tiny_model, fresh.select_ds), ids)  # a pass is deterministic
    usage = tokenizer_usage(ids)
    assert usage.n_codes == fresh.n_select * GRID * GRID, "every position of every selection image is counted, not the first batch's"


def test_code_ids_refuse_a_model_that_has_no_code_grid(fresh):
    with pytest.raises(TypeError, match="encode_to_indices"):
        td.code_ids_of(keras.Sequential([keras.layers.Input((IMAGE, IMAGE, 3)), keras.layers.Flatten()]), fresh.select_ds)


# ---------------------------------------------------------------------
# 4. the floor
# ---------------------------------------------------------------------


def usage_of(entropy: float, **recorded: Any) -> TokenizerUsage:
    """A usage record holding what a README records (the floor reads the entropy); what it does not record is NaN or 0."""
    return TokenizerUsage(
        n_codes=recorded.get("n_codes", 0), distinct=recorded.get("distinct", 0), perplexity=math.exp(entropy),
        top1_share=recorded.get("top1_share", float("nan")), entropy=entropy)


@pytest.mark.parametrize("repeats", [1, 3, 200])
def test_exactly_64_equally_used_codes_meet_the_floor_and_63_do_not(repeats):
    at_the_floor = tokenizer_usage(np.repeat(np.arange(64), repeats))
    assert at_the_floor.distinct == 64 and at_the_floor.n_codes == 64 * repeats
    assert at_the_floor.perplexity < 64.0, "the control: the real calculation does come out below 64, so a bare compare of the perplexity would refuse it"
    assert usage_meets_floor(at_the_floor) is True, "exp(log(64)) is 63.99999999999998: the floor is judged on the entropy"
    assert usage_meets_floor(tokenizer_usage(np.repeat(np.arange(65), repeats))) is True
    assert usage_meets_floor(tokenizer_usage(np.repeat(np.arange(63), repeats))) is False


def test_64_codes_used_unequally_do_not_meet_the_floor_and_a_wide_book_that_is_used_widely_does():
    skewed = np.concatenate([np.zeros(500, np.int64), np.arange(64)])  # 64 distinct codes, one of them 8 times in 9
    usage = tokenizer_usage(skewed)
    assert usage.distinct == 64 and usage.perplexity < MIN_TOKENIZER_PERPLEXITY
    assert usage_meets_floor(usage) is False
    wide = tokenizer_usage(np.repeat(np.arange(1000), 2))
    assert wide.perplexity > MIN_TOKENIZER_PERPLEXITY and usage_meets_floor(wide) is True


def test_the_floor_refuses_the_one_epoch_tokenizer_it_was_measured_on_and_passes_the_five_epoch_one():
    # src/train/README.md, "The usage floor" (plan 46e5318f D-007 and D-019): two stage-0 tokenizers on CIFAR-10 at 32 px, down-sample 4,
    # 512 codes, over the 5,000-image selection split. Over all 320,000 positions the 5-epoch one has entropy 5.8628 nats (perplexity 351.7)
    # and over the masked positions perplexity 354.8; the 1-epoch one entropy 3.1495 nats (perplexity 23.3) and 24.7 over the masked ones.
    assert math.isclose(math.exp(3.1495), 23.3, abs_tol=0.05) and math.isclose(math.exp(5.8628), 351.7, abs_tol=0.1)
    recorded = {
        "1-epoch, all positions": (usage_of(3.1495, distinct=350, n_codes=320_000, top1_share=0.1623), False),
        "1-epoch, masked positions": (usage_of(math.log(24.7), distinct=285), False),
        "5-epoch, all positions": (usage_of(5.8628, distinct=508, n_codes=320_000, top1_share=0.0337), True),
        "5-epoch, masked positions": (usage_of(math.log(354.8), distinct=507), True),
    }
    for name, (usage, meets) in recorded.items():
        assert usage_meets_floor(usage) is meets, name
    assert math.log(24.7) < math.log(MIN_TOKENIZER_PERPLEXITY) < math.log(354.8)
    assert MIN_TOKENIZER_PERPLEXITY == 64.0


def test_usage_meets_floor_reads_the_modules_floor_when_it_is_called(monkeypatch):
    ten = tokenizer_usage(np.repeat(np.arange(10), 5))
    seven = tokenizer_usage(np.repeat(np.arange(7), 5))
    assert usage_meets_floor(ten) is False  # the real floor: a toy tokenizer cannot reach it
    monkeypatch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", 8.0)  # what the toy-run tests of the tasks do
    assert usage_meets_floor(ten) is True and usage_meets_floor(seven) is False


def test_both_floors_judge_through_the_one_comparison_so_one_tolerance_moves_both(monkeypatch):
    # D-028 and D-079: "an entropy reaches a floor" (at least the log of it, less a rounding slack) is ONE function, `pretrain_run.entropy_reaches`, that the pooled
    # floor and the per-cell floor both call: two copies would change for one reason (the rounding of exp(log(k))) and could drift apart.
    pooled, per_cell = math.log(MIN_TOKENIZER_PERPLEXITY), math.log(td.MIN_PER_CELL_PERPLEXITY)

    def judged(shortfall: float) -> tuple:
        """The verdicts (pooled, per cell) of an entropy ``shortfall`` nats under each floor's log."""
        cells = td.CellUsage(cells=64, images=100, entropy=per_cell - shortfall, perplexity=math.exp(per_cell - shortfall))
        return usage_meets_floor(usage_of(pooled - shortfall)), td.per_cell_meets_floor(cells)

    assert judged(0.0) == (True, True) and judged(0.5) == (False, False), "the control: at its floor each meets it, half a nat under neither does"
    monkeypatch.setattr(pretrain_run, "ENTROPY_ROUNDING_TOLERANCE", 1.0)  # the one slack, read by both floors when they are called
    assert judged(0.5) == (True, True), "the per-cell floor reads the slack of the one comparison, not a copy of it"
    assert td.entropy_reaches is pretrain_run.entropy_reaches
    monkeypatch.undo()
    # and each floor CALLS it, with the floor it reads when it is called (a comparison written inline beside it would not show here)
    asked = []
    monkeypatch.setattr(pretrain_run, "entropy_reaches", lambda entropy, floor: asked.append(floor) or True)
    monkeypatch.setattr(td, "entropy_reaches", pretrain_run.entropy_reaches)
    monkeypatch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", 11.0)
    monkeypatch.setattr(td, "MIN_PER_CELL_PERPLEXITY", 3.0)
    assert judged(0.5) == (True, True) and asked == [11.0, 3.0], f"both floors call the one comparison with the floor they read at the call: {asked}"


@pytest.mark.parametrize("ids", [np.repeat(np.arange(64), 3), np.repeat(np.arange(65), 3), np.arange(96) % 60, np.zeros(5, np.int64),
                                 np.repeat(np.arange(63), 3)], ids=["64", "65", "60 of 96", "one code", "63"])
def test_the_refusal_and_the_verdict_agree(ids):
    meets = usage_meets_floor(tokenizer_usage(ids))
    if meets:
        assert checked_tokenizer_usage(ids, path="t.keras").n_codes == len(ids)
    else:
        with pytest.raises(ValueError, match="below the floor of 64"):
            checked_tokenizer_usage(ids, path="t.keras")


def test_checked_tokenizer_usage_raises_the_message_it_always_raised_apart_from_the_command_it_names():
    ids = np.arange(96) % 60                       # 96 positions, 60 distinct codes
    usage = tokenizer_usage(ids)
    with pytest.raises(ValueError) as caught:
        checked_tokenizer_usage(ids, path="t.keras")
    head = (f"the tokenizer 't.keras' uses its codebook too narrowly: {usage.distinct} distinct codes over the {usage.n_codes} masked "
            f"positions of the selection split, perplexity {usage.perplexity:.2f} (top-1 share {usage.top1_share:.1%}), below the floor of "
            f"{MIN_TOKENIZER_PERPLEXITY:g} effective codes. A bar cut from targets this concentrated is vacuous. Either the tokenizer is "
            "too narrow (train the tokenizer longer: ")
    tail = (f") or the split is too small, since the estimate is limited by the split size: {usage.n_codes} positions cannot show more "
            f"than {usage.n_codes} codes (raise data.max_samples)")
    assert re.fullmatch(re.escape(head) + r"python -m [^)]+" + re.escape(tail), str(caught.value)), str(caught.value)


# ---------------------------------------------------------------------
# 4b. the per-cell floor: what a pooled perplexity cannot tell (review pass 1, W3)
# ---------------------------------------------------------------------
#
# CIFAR-10 at 32 px and patch 4 has 8 x 8 = 64 cells, exactly the pooled floor: a code that is the CELL INDEX alone, ignoring the image,
# has a pooled perplexity of exactly 64 and meets it. The per-cell perplexity, exp of the mean over the cells of H(code | cell), is 1 for it.


def loop_per_cell_perplexity(codes: np.ndarray) -> float:
    """exp of the mean over cells of H(code | cell), written with a loop and a dict per cell (no code shared with the module)."""
    images, rows, columns = codes.shape
    entropies = []
    for row in range(rows):
        for column in range(columns):
            counts: Dict[int, int] = {}
            for code in codes[:, row, column]:
                counts[int(code)] = counts.get(int(code), 0) + 1
            entropies.append(-sum(k / images * math.log(k / images) for k in counts.values()))
    return math.exp(sum(entropies) / len(entropies))


def test_a_code_that_is_the_cell_index_alone_meets_the_pooled_floor_and_is_worth_one_code_per_cell():
    codes = cell_index_codes(40)
    pooled = tokenizer_usage(codes)
    assert pooled.distinct == 64 and usage_meets_floor(pooled) is True, "the control: the pooled floor calls this code a 64-code tokenizer"
    usage = td.per_cell_usage(codes)
    assert (usage.cells, usage.images) == (64, 40) and usage.entropy == 0.0 and usage.perplexity == 1.0
    assert td.per_cell_meets_floor(usage) is False


@pytest.mark.parametrize("bits, meets", [(0, False), (1, False), (2, False), (3, True), (4, True)])
def test_position_plus_image_wide_bits_is_worth_two_to_the_bits_per_cell_and_the_floor_wants_three(bits, meets):
    codes = with_image_wide_bits(bits)
    pooled, usage = tokenizer_usage(codes), td.per_cell_usage(codes)
    assert usage_meets_floor(pooled) is True, f"{bits} bits: a pooled perplexity of {pooled.perplexity:.1f} meets the pooled floor whatever the bits are"
    assert usage.perplexity == pytest.approx(2.0 ** bits, rel=1e-12) and pooled.perplexity == pytest.approx(64 * 2.0 ** bits, rel=1e-12)
    assert td.per_cell_meets_floor(usage) is meets


def test_codes_that_describe_content_are_worth_the_same_pooled_and_per_cell():
    # Every cell sees each of 16 codes four times, whatever the cell: nothing about the code is positional.
    images, rows, columns = 64, 3, 5
    codes = (np.arange(images).reshape(images, 1, 1) + 3 * np.arange(rows).reshape(1, rows, 1) + 5 * np.arange(columns).reshape(1, 1, columns)) % 16
    assert td.per_cell_usage(codes).perplexity == pytest.approx(16.0, rel=1e-12) == pytest.approx(tokenizer_usage(codes).perplexity, rel=1e-12)


def test_an_image_wide_code_repeated_at_every_cell_meets_both_floors_so_the_floors_do_not_test_locality():
    # Review pass 2, N1: both floors count how the codes vary ACROSS images and cells; nothing in them asks that a code describe ITS OWN patch. One id of the whole
    # image (its cluster among 128, say), the same at all 64 cells, is worth 128 pooled and 128 per cell: the control that pins the limit the README states.
    images, ids = 256, 128
    codes = np.broadcast_to((np.arange(images) % ids).reshape(images, 1, 1), (images, 8, 8)).copy()
    pooled, usage = tokenizer_usage(codes), td.per_cell_usage(codes)
    assert pooled.perplexity == pytest.approx(ids, rel=1e-12) and usage.perplexity == pytest.approx(ids, rel=1e-12)
    assert usage_meets_floor(pooled) is True and td.per_cell_meets_floor(usage) is True


def test_the_per_cell_usage_equals_a_loop_over_the_cells_on_random_codes():
    codes = np.random.default_rng(0).integers(0, 7, size=(50, 3, 4))
    usage = td.per_cell_usage(codes)
    assert usage.perplexity == pytest.approx(loop_per_cell_perplexity(codes), rel=1e-12) and math.exp(usage.entropy) == usage.perplexity
    assert usage.perplexity < tokenizer_usage(codes).perplexity < 7.0 + 1e-9, "per cell is below the pooled perplexity, which is below the codebook"
    assert usage.as_dict() == {"cells": 12, "images": 50, "entropy": usage.entropy, "perplexity": usage.perplexity}


@pytest.mark.parametrize("bad", [np.zeros((4, 8)), np.zeros((0, 8, 8), np.int64), np.zeros((4, 8, 8)), -np.ones((4, 2, 2), np.int64)],
                         ids=["not a grid", "no image", "floats", "a negative id"])
def test_a_per_cell_usage_of_something_that_is_not_a_grid_of_code_ids_is_refused(bad):
    with pytest.raises(ValueError):
        td.per_cell_usage(bad)


def test_the_per_cell_floor_is_eight_and_is_read_when_it_is_called(monkeypatch):
    assert td.MIN_PER_CELL_PERPLEXITY == 8.0, "D-067: derived from the measured arms and the positional controls, not typed twice"

    def label_only(classes: int) -> "td.CellUsage":  # an image-wide label of `classes` equally used values at every cell, nothing positional
        label = (np.arange(8 * classes) * classes) // (8 * classes)
        return td.per_cell_usage(np.broadcast_to(label.reshape(-1, 1, 1), (8 * classes, 8, 8)).copy())

    seven, eight = label_only(7), label_only(8)
    assert (seven.perplexity, eight.perplexity) == (pytest.approx(7.0, rel=1e-12), pytest.approx(8.0, rel=1e-12))
    assert td.per_cell_meets_floor(eight) is True, "exactly the floor meets it (judged on the entropy, as the pooled floor is)"
    assert td.per_cell_meets_floor(seven) is False
    one_bit = td.per_cell_usage(with_image_wide_bits(1))
    monkeypatch.setattr(td, "MIN_PER_CELL_PERPLEXITY", 2.0)  # what a toy-run test may do
    assert td.per_cell_meets_floor(one_bit) is True and td.per_cell_meets_floor(seven) is True
    monkeypatch.setattr(td, "MIN_PER_CELL_PERPLEXITY", 16.0)
    assert td.per_cell_meets_floor(eight) is False
    assert usage_meets_floor(tokenizer_usage(with_image_wide_bits(0))) is True, "the pooled floor is untouched by it"


# ---------------------------------------------------------------------
# 5. the epoch-zero key
# ---------------------------------------------------------------------


def sidecar_with(tmp_path: Path, metrics: Dict[str, Any]) -> Path:
    path = tmp_path / FIT_STATE_NAME
    update_fit_state(path, initial={"metrics": metrics, "ratio": 1.0, "warned": None})
    return path


def test_epoch_zero_val_loss_with_no_key_reads_the_loss_exactly_as_it_did(tmp_path):
    path = sidecar_with(tmp_path, {"loss": 0.7, "reconstruction_loss": 0.4, "vq_loss": 0.3})
    assert epoch_zero_val_loss(path) == 0.7
    missing = tmp_path / "no_such_sidecar.json"
    with pytest.raises(ValueError) as caught:
        epoch_zero_val_loss(missing)
    assert str(caught.value) == (
        f"the epoch-zero val_loss is not in {missing}: the fit sidecar holds no initial.metrics.loss, so the success bar cannot be cut "
        "(a run writes it before its first epoch)")
    (tmp_path / "flag").mkdir()
    with pytest.raises(ValueError, match="epoch-zero"):  # a bool is not a number
        epoch_zero_val_loss(sidecar_with(tmp_path / "flag", {"loss": True}))


def test_epoch_zero_val_loss_reads_the_key_it_is_given(tmp_path):
    path = sidecar_with(tmp_path, {"loss": 0.7, "reconstruction_loss": 0.4, "vq_loss": 0.3})
    assert epoch_zero_val_loss(path, key="loss") == epoch_zero_val_loss(path) == 0.7
    assert epoch_zero_val_loss(path, key="reconstruction_loss") == 0.4
    assert epoch_zero_val_loss(path, key="vq_loss") == 0.3


def test_epoch_zero_val_loss_names_the_key_it_did_not_find(tmp_path):
    path = sidecar_with(tmp_path, {"loss": 0.7})
    with pytest.raises(ValueError) as caught:
        epoch_zero_val_loss(path, key="reconstruction_loss")
    message = str(caught.value)
    assert "epoch-zero val_reconstruction_loss" in message and "initial.metrics.reconstruction_loss" in message and str(path) in message
