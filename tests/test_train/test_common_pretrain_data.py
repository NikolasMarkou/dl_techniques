"""``train.common.pretrain_data`` and the baselines of ``train.common.pretrain_run``.

The pretrain task of ``train.api`` trains on a masked element (ET masked completion, BEiT
masked image modelling) and scores a ``val_loss`` that its success bar is cut from. What can
go wrong here is silent, so each rule below is pinned by a test that sees its failure:

1. **The record.** ``PretrainBatches`` carries what the run sequence reads; the train stream
   is infinite, the select and test streams state the length of one pass.
2. **Determinism.** The select and test masks are keyed on ``(split seed, split, row
   position)``, so two ``model.evaluate`` passes of the same weights agree EXACTLY, and they
   do not move with the stream seed. The legacy stateful maps are run over the same stream
   as the instrument check: they DISAGREE with themselves, so the equality above can fail.
3. **The keys.** The train mask follows ``(stream seed, counter)``; train, select and test
   have their own salt.
4. **The stream seed.** A resume derives a new stream seed from the completed-step count;
   the split never follows it, so no selection row reaches the train stream under any seed.
5. **The baselines.** The constant model's ``val_loss`` equals a float64 numpy computation of
   the same weighted quantity; a BEiT model's ``val_loss`` is the mean over examples of the
   per-example masked-token cross-entropy.
6. **TokenizerUsage**, a pure function over code ids.

Sizes are tiny: the fake cifar10 of ``tests/classification_fakes.py`` (60 train rows, 54 fit
and 6 selected, 20 test rows), 32 px, patch 4 (a grid of 8 x 8 = 64 patches), batch 4. Real
imagenette at 224 px / patch 16 has 196 patches; with a 8192-code BEiT head and batch 32 the
constant model's ``(B, N, V)`` logits are ``32 * 196 * 8192 * 4`` bytes, about 205 MB per
evaluation batch (the arithmetic is pinned below at a small size).
"""

import hashlib
import math
from typing import Any, Dict, List

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.beit_masking import make_beit_mim_map_fn
from dl_techniques.datasets.vision.masked_patches import make_masked_patch_map_fn
from dl_techniques.models.vision.beit import create_beit_mim
from dl_techniques.models.vision.energy_transformer import create_energy_transformer_mim
from tests.classification_fakes import (
    IMAGENETTE_FAKE_TEST,
    IMAGENETTE_FAKE_TRAIN,
    identities_of,
    use_fake_imagenette,
    use_identity_cifar10,
)
from train.common import pretrain_data
from train.common.image_streams import ImageStreams, load_image_streams
from train.common.pretrain_data import (
    METHODS,
    PretrainBatches,
    TokenizerUsage,
    derive_stream_seed,
    load_pretrain_batches,
    masked_code_ids,
    tokenizer_usage,
)
from train.common.pretrain_run import (
    Baseline,
    ConstantPredictor,
    constant_baseline,
    pretrain_loss,
)

IMAGE = 32
PATCH = 4
N = (IMAGE // PATCH) ** 2          # 64 patches
PATCH_DIM = PATCH * PATCH * 3
BATCH = 4
SEED = 42
VOCAB = 16
N_SELECT = 6                       # 10 percent of the 60 fake train rows, rounded up to 6
N_FIT = IMAGENETTE_FAKE_TRAIN - N_SELECT
STREAM_SEEDS = (0, 1, 7, 123, 4_000_000_000)

ET, BEIT = "et_masked", "beit_mim"


def code_fn(images: tf.Tensor) -> tf.Tensor:
    """A cheap batched stand-in tokenizer: ``(B, 32, 32, 3) -> (B, 8, 8)`` ids in ``[0, VOCAB)``.

    It has the contract of ``train.beit.common.load_frozen_tokenizer``'s callable and its
    ids vary over the images and the columns of the grid (the fake rows differ by a mean
    colour and a horizontal ramp), so a unigram over them is not one code.
    """
    pooled = tf.nn.avg_pool2d(tf.reduce_mean(images, axis=-1, keepdims=True), PATCH, PATCH, "VALID")
    return tf.cast(tf.math.floormod(tf.floor(pooled[..., 0] * 6.0), VOCAB), tf.int32)


def build(method: str, dataset: str = "cifar10", **overrides: Any) -> PretrainBatches:
    """``load_pretrain_batches`` at the test sizes (un-augmented, so streams compare exactly)."""
    settings: Dict[str, Any] = dict(
        method=method, dataset=dataset, image_size=IMAGE, patch_size=PATCH, batch_size=BATCH,
        seed=SEED, stream_seed=None, validation_split=None, max_samples=None, augment=False,
    )
    if method == ET:
        settings.update(mask_ratio=0.5, mask_token_frac=0.9)
    else:
        settings.update(num_mask_patches=24, min_mask_patches_per_block=5, tokenizer_fn=code_fn)
    settings.update(overrides)
    return load_pretrain_batches(**settings)


def use_fake(dataset: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if dataset == "cifar10":
        use_identity_cifar10(monkeypatch)
    else:
        use_fake_imagenette(monkeypatch)


@pytest.fixture(params=["cifar10", "imagenette"])
def dataset(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    use_fake(request.param, monkeypatch)
    return request.param


@pytest.fixture
def cifar(monkeypatch: pytest.MonkeyPatch) -> str:
    use_fake("cifar10", monkeypatch)
    return "cifar10"


@pytest.fixture(scope="module")
def models() -> Dict[str, keras.Model]:
    """One tiny compiled model per method (weights are never trained here)."""
    keras.utils.set_random_seed(0)
    et = create_energy_transformer_mim("tiny", (IMAGE, IMAGE, 3), PATCH)
    et.build([(None, IMAGE, IMAGE, 3), (None, N)])
    et.compile(optimizer="sgd", loss=pretrain_loss(ET))
    beit = create_beit_mim("tiny", (IMAGE, IMAGE, 3), PATCH, vocab_size=VOCAB)
    beit.build([(None, IMAGE, IMAGE, 3), (None, N)])
    beit.compile(optimizer="sgd", loss=pretrain_loss(BEIT))
    return {ET: et, BEIT: beit}


def evaluate(model: keras.Model, ds: tf.data.Dataset) -> float:
    return float(model.evaluate(ds, verbose=0))


def collect(ds: tf.data.Dataset, n_batches: int = None) -> List[Any]:
    """The elements of ``ds`` (or its first ``n_batches``) as numpy nests."""
    batches = ds if n_batches is None else ds.take(n_batches)
    return [tf.nest.map_structure(lambda t: t.numpy(), element) for element in batches]


def same(first: List[Any], second: List[Any]) -> bool:
    """Exact equality of two collected streams, every array of every element."""
    if len(first) != len(second):
        return False
    return all(
        all(np.array_equal(a, b) for a, b in zip(tf.nest.flatten(x), tf.nest.flatten(y)))
        for x, y in zip(first, second)
    )


def input_masks(ds: tf.data.Dataset, n_batches: int = None) -> np.ndarray:
    """The ``(rows, N)`` boolean input mask of every row of ``ds``."""
    return np.concatenate([element[0][1] for element in collect(ds, n_batches)]).astype(bool)


# ---------------------------------------------------------------------
# 1. the record
# ---------------------------------------------------------------------


class TestThePretrainBatchesRecord:

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_it_carries_what_the_run_sequence_reads(self, method, dataset) -> None:
        batches = build(method, dataset)
        n_fit, n_select, n_test = N_FIT, N_SELECT, IMAGENETTE_FAKE_TEST
        assert isinstance(batches, PretrainBatches)
        assert (batches.n_train, batches.n_select, batches.n_test) == (n_fit, n_select, n_test)
        assert batches.steps_per_epoch == n_fit // BATCH
        assert batches.val_steps == math.ceil(n_select / BATCH)
        assert int(batches.test_ds.cardinality()) == math.ceil(n_test / BATCH)
        assert batches.request["batch_size"] == BATCH
        assert tuple(batches.input_shape) == (IMAGE, IMAGE, 3)
        assert batches.num_classes is None
        assert isinstance(batches.selection_split, str) and batches.selection_split
        assert isinstance(batches.test_split, str) and batches.test_split
        assert batches.request["max_samples"] is None
        assert batches.seed == SEED and batches.stream_seed == SEED  # None means the split seed
        for name in ("train_ds", "select_ds", "test_ds"):
            assert isinstance(getattr(batches, name), tf.data.Dataset)

    def test_the_counts_are_what_the_resume_check_compares(self, cifar) -> None:
        from train.api import resume_decisions  # noqa: PLC0415 - the real reader of the record
        counts = resume_decisions.data_counts(build(ET))
        assert counts == {"n_train": N_FIT, "n_select": N_SELECT, "n_test": IMAGENETTE_FAKE_TEST}

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_the_train_stream_is_infinite_and_the_others_state_one_pass(self, method, dataset) -> None:
        batches = build(method, dataset)
        assert int(batches.train_ds.cardinality()) == int(tf.data.INFINITE_CARDINALITY)
        assert int(batches.select_ds.cardinality()) == math.ceil(batches.n_select / BATCH)
        assert int(batches.test_ds.cardinality()) == math.ceil(batches.n_test / BATCH)
        # and one pass really yields that many batches
        assert len(collect(batches.select_ds)) == batches.val_steps
        assert len(collect(batches.test_ds)) == math.ceil(batches.n_test / BATCH)

    def test_an_et_element_is_the_masked_patch_element(self, cifar) -> None:
        (image, mask), target, weight = collect(build(ET).select_ds, 1)[0]
        assert image.shape == (BATCH, IMAGE, IMAGE, 3) and image.dtype == np.float32
        assert mask.shape == (BATCH, N) and mask.dtype == np.bool_
        assert target.shape == (BATCH, N, PATCH_DIM) and weight.shape == (BATCH, N)
        n_loss = round(0.5 * N)
        assert np.all((weight > 0).sum(axis=1) == n_loss)
        assert np.allclose(weight.sum(axis=1), N)  # weight N / n_loss on the loss set

    def test_a_beit_element_is_the_masked_code_element(self, cifar) -> None:
        (image, mask), ids, weight = collect(build(BEIT).select_ds, 1)[0]
        assert image.shape == (BATCH, IMAGE, IMAGE, 3)
        assert mask.shape == (BATCH, N) and mask.dtype == np.bool_
        assert ids.shape == (BATCH, N) and ids.dtype == np.int32
        assert ids.min() >= 0 and ids.max() < VOCAB
        assert np.allclose(weight.sum(axis=1), N)           # sums to N when anything is masked
        assert np.all((weight > 0) == mask)

    def test_the_record_is_frozen(self, cifar) -> None:
        batches = build(ET)
        with pytest.raises(AttributeError):
            batches.n_train = 1  # type: ignore[misc]

    def test_a_cap_below_one_batch_still_makes_an_epoch(self, cifar) -> None:
        # a 10 percent split of 3 rows holds out 0 (the loader refuses that), so split in half
        batches = build(ET, max_samples=3, validation_split=0.5)
        assert batches.request["max_samples"] == 3
        assert (batches.n_train, batches.n_select) == (2, 1) and batches.n_train < BATCH
        assert batches.steps_per_epoch == 1
        (image, _), _, _ = collect(batches.train_ds, 1)[0]
        assert image.shape[0] == BATCH  # repeat comes before batch: the stream is never empty
        assert batches.val_steps == 1 and batches.n_test == IMAGENETTE_FAKE_TEST

    @pytest.mark.parametrize("stream", ["select_ds", "test_ds"])
    def test_a_stream_of_unknown_length_is_refused_by_name_stating_both_lengths(
            self, stream, cifar, monkeypatch) -> None:
        # Both finite streams are scored by Keras, which logs n / (n + 1) of the loss over a dataset of unknown length: the
        # select arm and the test arm are two lines of the wrapper, so each is held by its own row.
        real = pretrain_data.load_image_streams
        honest = build(ET)

        def lying(*args: Any, **kwargs: Any) -> ImageStreams:
            streams = real(*args, **kwargs)
            from dataclasses import replace
            return replace(streams, **{stream: getattr(streams, stream).filter(lambda *_: True)})

        monkeypatch.setattr(pretrain_data, "load_image_streams", lying)
        with pytest.raises(ValueError) as caught:
            build(ET)
        text = str(caught.value)
        assert f"{stream} has cardinality unknown" in text and str(int(getattr(honest, stream).cardinality())) in text

    @pytest.mark.parametrize("kwargs, message", [
        (dict(method="nope"), "method"),
        (dict(seed=True), "seed"),
        (dict(stream_seed=1.5), "stream_seed"),
        (dict(patch_size=5), "divisible"),
        (dict(mask_ratio=None), "mask_ratio"),
        (dict(mask_ratio=0.5, num_mask_patches=24), "num_mask_patches"),
        (dict(tokenizer_fn=code_fn), "tokenizer_fn"),
        (dict(dataset="mnist"), "mnist"),
    ])
    def test_a_bad_request_is_refused_naming_it(self, kwargs, message, cifar) -> None:
        with pytest.raises((ValueError, TypeError), match=message):
            build(kwargs.pop("method", ET), **kwargs)

    @pytest.mark.parametrize("kwargs, message", [
        (dict(tokenizer_fn=None), "tokenizer_fn"),
        (dict(num_mask_patches=None), "num_mask_patches"),
        (dict(mask_ratio=0.5), "mask_ratio"),
    ])
    def test_a_bad_beit_request_is_refused_naming_it(self, kwargs, message, cifar) -> None:
        with pytest.raises((ValueError, TypeError), match=message):
            build(BEIT, **kwargs)

    def test_the_methods_are_the_two_the_task_has(self) -> None:
        assert tuple(METHODS) == (ET, BEIT)


# ---------------------------------------------------------------------
# 2. determinism: the same weights score the same
# ---------------------------------------------------------------------


class TestAValidationPassIsAFunctionOfTheWeights:

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_two_evaluate_passes_agree_exactly(self, method, models, cifar) -> None:
        batches = build(method)
        for ds in (batches.select_ds, batches.test_ds):
            first, second = evaluate(models[method], ds), evaluate(models[method], ds)
            assert first == second, f"{first!r} != {second!r}"
            assert math.isfinite(first) and first > 0

    def test_the_legacy_stateful_et_map_disagrees_with_itself(self, models, cifar) -> None:
        """The instrument can see the defect: the legacy map draws from one shared generator."""
        streams = load_image_streams(
            "cifar10", image_size=IMAGE, batch_size=BATCH, seed=SEED, augment=False,
            element_map_fn=make_masked_patch_map_fn(PATCH, IMAGE, 0.5, 0.9, seed=SEED))
        first, second = evaluate(models[ET], streams.select_ds), evaluate(models[ET], streams.select_ds)
        assert first != second

    def test_the_legacy_stateful_beit_map_disagrees_with_itself(self, models, cifar) -> None:
        streams = load_image_streams(
            "cifar10", image_size=IMAGE, batch_size=BATCH, seed=SEED, augment=False,
            element_map_fn=make_beit_mim_map_fn(
                lambda image: code_fn(image[None])[0], (8, 8), 24, 5))
        first, second = evaluate(models[BEIT], streams.select_ds), evaluate(models[BEIT], streams.select_ds)
        assert first != second

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_select_and_test_do_not_follow_the_stream_seed(self, method, models, cifar) -> None:
        reference = build(method)
        for stream_seed in STREAM_SEEDS:
            other = build(method, stream_seed=stream_seed)
            assert same(collect(other.select_ds), collect(reference.select_ds)), stream_seed
            assert same(collect(other.test_ds), collect(reference.test_ds)), stream_seed
            assert evaluate(models[method], other.select_ds) == evaluate(models[method], reference.select_ds)
            assert evaluate(models[method], other.test_ds) == evaluate(models[method], reference.test_ds)

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_the_split_seed_changes_the_select_masks(self, method, cifar) -> None:
        """Control for the test above: the masks do depend on the split seed."""
        first = input_masks(build(method).select_ds)
        second = input_masks(build(method, seed=SEED + 1).select_ds)
        assert not np.array_equal(first, second)


# ---------------------------------------------------------------------
# 3. the keys
# ---------------------------------------------------------------------


class TestTheMaskKeys:

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_the_train_mask_follows_the_stream_seed(self, method, cifar) -> None:
        masks = {s: input_masks(build(method, stream_seed=s).train_ds, 3) for s in STREAM_SEEDS}
        for a in STREAM_SEEDS:
            for b in STREAM_SEEDS:
                if a < b:
                    # the mask of every counter 0..11 differs between two stream seeds
                    assert not np.any(np.all(masks[a] == masks[b], axis=1)), (a, b)

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_the_train_mask_follows_the_counter(self, method, cifar) -> None:
        masks = input_masks(build(method).train_ds, 5)          # counters 0..19
        assert len({row.tobytes() for row in masks}) == len(masks)

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_the_three_streams_have_their_own_mask_at_one_index(self, method, cifar) -> None:
        batches = build(method)                                  # stream seed == split seed
        train, select, test = (
            input_masks(ds, 1)[0] for ds in (batches.train_ds, batches.select_ds, batches.test_ds))
        assert not np.array_equal(train, select)
        assert not np.array_equal(train, test)
        assert not np.array_equal(select, test)


# ---------------------------------------------------------------------
# 4. the stream seed: leak and resume
# ---------------------------------------------------------------------


def _train_identities(dataset: str, batches: PretrainBatches, n_batches: int) -> np.ndarray:
    return identities_of(dataset, batches.train_ds, 0, n_batches)


class TestNoSelectionRowReachesTheTrainStream:

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_under_every_stream_seed(self, method, dataset) -> None:
        reference = build(method, dataset)
        select = set(identities_of(dataset, reference.select_ds, 0).tolist())
        assert len(select) == N_SELECT
        for stream_seed in STREAM_SEEDS:
            # augmented too: the row identity survives a flip and a crop
            batches = build(method, dataset, stream_seed=stream_seed, augment=True)
            train = set(_train_identities(dataset, batches, 4 * batches.steps_per_epoch).tolist())
            assert train, stream_seed
            assert not train & select, f"stream_seed={stream_seed}: {sorted(train & select)}"
            assert train == set(range(IMAGENETTE_FAKE_TRAIN)) - select  # the whole fit pool is reached

    def test_the_split_does_not_move_with_the_stream_seed(self, dataset) -> None:
        reference = identities_of(dataset, build(ET, dataset).select_ds, 0)
        for stream_seed in STREAM_SEEDS:
            other = identities_of(dataset, build(ET, dataset, stream_seed=stream_seed).select_ds, 0)
            assert np.array_equal(reference, other), stream_seed


class TestDerivedStreamSeeds:

    def test_it_is_the_first_four_bytes_of_the_hash(self) -> None:
        for seed, steps in ((42, 0), (42, 1), (42, 7), (0, 7), (123, 99999)):
            expected = int(hashlib.sha256(f"{seed}:{steps}".encode()).hexdigest()[:8], 16)
            assert derive_stream_seed(seed, steps) == expected

    def test_a_fresh_run_uses_the_seed_itself(self) -> None:
        assert derive_stream_seed(42, None) == 42
        assert derive_stream_seed(0, None) == 0
        assert derive_stream_seed(42) == 42

    def test_it_is_distinct_per_completed_step_count(self) -> None:
        seen = {derive_stream_seed(SEED, steps) for steps in range(5000)}
        assert len(seen) == 5000
        assert derive_stream_seed(SEED, 0) != SEED          # a resume at step 0 is not the fresh run
        assert derive_stream_seed(1, 7) != derive_stream_seed(2, 7)

    @pytest.mark.parametrize("seed, steps", [(True, 1), (1, True), (1, -1), (1, 1.5), ("1", 1)])
    def test_a_bad_value_is_refused(self, seed, steps) -> None:
        with pytest.raises((TypeError, ValueError)):
            derive_stream_seed(seed, steps)


class TestTheResumeRebuild:

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_the_split_and_the_evaluation_streams_are_the_same(self, method, dataset) -> None:
        fresh = build(method, dataset)
        resumed = fresh.with_stream(35)
        assert (resumed.n_train, resumed.n_select, resumed.n_test) == (
            fresh.n_train, fresh.n_select, fresh.n_test)
        assert resumed.steps_per_epoch == fresh.steps_per_epoch
        assert resumed.val_steps == fresh.val_steps
        assert int(resumed.test_ds.cardinality()) == int(fresh.test_ds.cardinality())
        assert resumed.seed == fresh.seed == SEED
        assert resumed.stream_seed == derive_stream_seed(SEED, 35) != fresh.stream_seed
        assert same(collect(resumed.select_ds), collect(fresh.select_ds))
        assert same(collect(resumed.test_ds), collect(fresh.test_ds))

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_the_train_stream_is_neither_a_replay_nor_another_resumes(self, method, dataset) -> None:
        fresh = build(method, dataset)
        first = _train_identities(dataset, fresh, 3)
        a = _train_identities(dataset, fresh.with_stream(13), 3)
        b = _train_identities(dataset, fresh.with_stream(26), 3)
        assert not np.array_equal(first, a)
        assert not np.array_equal(first, b)
        assert not np.array_equal(a, b)
        # the masks differ too, not just the order
        assert not np.any(np.all(input_masks(fresh.train_ds, 3) == input_masks(fresh.with_stream(13).train_ds, 3), axis=1))

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_one_completed_step_count_is_one_deterministic_stream(self, method, cifar) -> None:
        fresh = build(method)
        assert same(collect(fresh.with_stream(13).train_ds, 4), collect(fresh.with_stream(13).train_ds, 4))

    def test_it_keeps_every_request_value(self, cifar) -> None:
        fresh = build(BEIT, max_samples=30, validation_split=0.2)
        resumed = fresh.with_stream(5)
        assert resumed.request["max_samples"] == 30
        assert (resumed.n_train, resumed.n_select) == (fresh.n_train, fresh.n_select)

    @pytest.mark.parametrize("steps", [-1, True, 2.5])
    def test_a_bad_count_is_refused(self, steps, cifar) -> None:
        with pytest.raises((TypeError, ValueError)):
            build(ET).with_stream(steps)

    @pytest.mark.parametrize("method", [ET, BEIT])
    def test_a_derived_seed_above_the_int32_range_builds_augmented(self, method, dataset) -> None:
        batches = build(method, dataset, stream_seed=4_000_000_000, augment=True)
        assert len(collect(batches.train_ds, 2)) == 2


# ---------------------------------------------------------------------
# 5. the baselines and the BEiT weight, against float64 numpy
# ---------------------------------------------------------------------

#: The constant model's loss and the float64 formula differ only by float32 accumulation
#: (MEASURED 4.8e-7 absolute at val_loss ~ 5.9); one part in 1e5 is the stated tolerance.
RTOL = 1e-5


def et_zero_oracle(ds: tf.data.Dataset) -> float:
    """Mean over examples of the mean over the loss set of the zero predictor's patch MSE."""
    total, count = 0.0, 0
    for (_, _), target, weight in collect(ds):
        target, loss_set = target.astype(np.float64), weight > 0
        per_patch = (target ** 2).mean(axis=-1)
        total += ((per_patch * loss_set).sum(axis=1) / loss_set.sum(axis=1)).sum()
        count += len(target)
    return total / count


def beit_unigram_oracle(ds: tf.data.Dataset) -> float:
    """Mean over examples of the masked tokens' mean ``-log p`` under the pooled masked unigram."""
    elements = collect(ds)
    ids = np.concatenate([e[1] for e in elements])
    masks = np.concatenate([e[0][1] for e in elements]).astype(bool)
    counts = np.bincount(ids[masks], minlength=VOCAB).astype(np.float64)
    log_p = np.log(np.maximum(counts / counts.sum(), 1e-30))
    per_example = [(-log_p[row[mask]]).mean() if mask.any() else 0.0 for row, mask in zip(ids, masks)]
    return float(np.mean(per_example))


def log_softmax64(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class TestTheConstantBaselines:

    def test_the_et_baseline_is_the_zero_patch_predictor(self, cifar) -> None:
        batches = build(ET)
        baseline = constant_baseline(ET, batches.select_ds)
        assert isinstance(baseline, Baseline) and baseline.kind == "zero_patches"
        oracle = et_zero_oracle(batches.select_ds)
        assert baseline.val_loss == pytest.approx(oracle, rel=RTOL)
        assert 0.1 < baseline.val_loss < 5.0  # normalized pixels: about 1 (MEASURED 0.9995 on cifar10)

    def test_the_et_baseline_equals_evaluate_of_a_model_that_outputs_zeros(self, cifar) -> None:
        batches = build(ET)
        image, mask = keras.Input((IMAGE, IMAGE, 3)), keras.Input((N,), dtype="bool")
        zeros = keras.layers.Lambda(
            lambda t: keras.ops.zeros((keras.ops.shape(t[0])[0], N, PATCH_DIM)))([image, mask])
        model = keras.Model([image, mask], zeros)
        model.compile(optimizer="sgd", loss=pretrain_loss(ET))
        assert constant_baseline(ET, batches.select_ds).val_loss == evaluate(model, batches.select_ds)

    def test_the_beit_baseline_is_the_in_sample_unigram_of_the_masked_positions(self, cifar) -> None:
        batches = build(BEIT)
        baseline = constant_baseline(BEIT, batches.select_ds, vocab_size=VOCAB)
        assert baseline.kind == "unigram_codes"
        oracle = beit_unigram_oracle(batches.select_ds)
        assert baseline.val_loss == pytest.approx(oracle, rel=RTOL)
        assert baseline.val_loss < math.log(VOCAB)         # a skewed unigram beats the uniform bar

    def test_the_beit_baseline_needs_a_vocabulary_and_codes_inside_it(self, cifar) -> None:
        batches = build(BEIT)
        with pytest.raises(ValueError, match="vocab_size"):
            constant_baseline(BEIT, batches.select_ds)
        with pytest.raises(ValueError, match="outside"):
            constant_baseline(BEIT, batches.select_ds, vocab_size=3)

    def test_an_example_that_masked_nothing_weighs_zero_and_makes_no_nan(self) -> None:
        rng = np.random.default_rng(0)
        images = rng.normal(size=(5, IMAGE, IMAGE, 3)).astype("float32")
        ids = rng.integers(0, VOCAB, size=(5, N)).astype("int32")
        masks = rng.random((5, N)) < 0.4
        masks[2] = False                                      # one example masked nothing
        weight = masks * (N / np.maximum(masks.sum(axis=1, keepdims=True), 1))
        ds = tf.data.Dataset.from_tensor_slices(
            ((images, masks), ids, weight.astype("float32"))).batch(BATCH)
        baseline = constant_baseline(BEIT, ds, vocab_size=VOCAB)
        assert math.isfinite(baseline.val_loss)
        assert baseline.val_loss == pytest.approx(beit_unigram_oracle(ds), rel=RTOL)

    def test_the_baseline_reads_the_stream_and_no_weights(self, cifar) -> None:
        """A baseline reads the select stream and no model: it is the same on every call."""
        batches = build(ET)
        assert constant_baseline(ET, batches.select_ds) == constant_baseline(ET, batches.select_ds)

    def test_a_constant_predictor_broadcasts_to_the_batch(self) -> None:
        value = np.arange(VOCAB, dtype="float32")
        model = ConstantPredictor(value, (N, VOCAB))
        out = model([np.zeros((3, IMAGE, IMAGE, 3), "float32"), np.zeros((3, N), bool)])
        assert tuple(out.shape) == (3, N, VOCAB)
        assert np.array_equal(np.asarray(out)[2, 5], value)
        assert np.asarray(out).nbytes == 3 * N * VOCAB * 4     # 32 * 196 * 8192 * 4 = 205 MB at imagenette
        assert 32 * 196 * 8192 * 4 == 205_520_896


class TestTheLossIsTheMeanOverExamples:

    def test_the_et_val_loss_is_the_mean_loss_set_patch_mse(self, models, cifar) -> None:
        batches = build(ET)
        elements = collect(batches.select_ds)
        predictions = models[ET].predict(batches.select_ds, verbose=0).astype(np.float64)
        per_example, row = [], 0
        for (_, _), target, weight in elements:
            for t, w in zip(target.astype(np.float64), weight):
                mse = ((t - predictions[row]) ** 2).mean(axis=-1)
                per_example.append(mse[w > 0].mean())
                row += 1
        assert evaluate(models[ET], batches.select_ds) == pytest.approx(np.mean(per_example), rel=RTOL)

    def test_the_beit_val_loss_is_the_mean_over_examples_of_the_masked_token_loss(
            self, models, cifar) -> None:
        batches = build(BEIT)
        elements = collect(batches.select_ds)
        logits = models[BEIT].predict(batches.select_ds, verbose=0)
        ids = np.concatenate([e[1] for e in elements])
        masks = np.concatenate([e[0][1] for e in elements]).astype(bool)
        log_p = log_softmax64(logits)
        per_example = [
            -log_p[i][np.arange(N), ids[i]][masks[i]].mean() for i in range(len(ids)) if masks[i].any()
        ]
        # an example that masked nothing contributes 0 to the sum and 1 to the count
        expected = np.sum(per_example) / len(ids)
        assert evaluate(models[BEIT], batches.select_ds) == pytest.approx(expected, rel=RTOL)
        assert expected > math.log(VOCAB) * 0.5            # on the scale of ln V, not mask_fraction * ln V


# ---------------------------------------------------------------------
# 6. TokenizerUsage
# ---------------------------------------------------------------------


class TestTokenizerUsage:

    def test_a_known_histogram(self) -> None:
        usage = tokenizer_usage(np.array([0] * 6 + [1] * 2 + [2] * 2))
        p = np.array([0.6, 0.2, 0.2])
        entropy = float(-(p * np.log(p)).sum())
        assert isinstance(usage, TokenizerUsage)
        assert usage.n_codes == 10 and usage.distinct == 3
        assert usage.top1_share == pytest.approx(0.6)
        assert usage.entropy == pytest.approx(entropy)
        assert usage.perplexity == pytest.approx(math.exp(entropy))

    def test_a_uniform_use_of_k_codes_has_perplexity_k(self) -> None:
        for k in (1, 2, 8, 64):
            usage = tokenizer_usage(np.tile(np.arange(k), 5))
            assert usage.perplexity == pytest.approx(k)
            assert usage.entropy == pytest.approx(math.log(k), abs=1e-12)
            assert usage.distinct == k and usage.top1_share == pytest.approx(1.0 / k)

    def test_one_code_is_a_collapse(self) -> None:
        usage = tokenizer_usage(np.full((4, 64), 7))
        assert (usage.distinct, usage.perplexity, usage.top1_share, usage.entropy) == (1, 1.0, 1.0, 0.0)

    def test_the_shape_does_not_matter_and_the_count_does_not_scale_the_answer(self) -> None:
        flat = np.array([3, 3, 1, 9, 9, 9, 0, 3])
        assert tokenizer_usage(flat.reshape(2, 4)) == tokenizer_usage(flat)
        assert tokenizer_usage(np.tile(flat, 10)).perplexity == pytest.approx(tokenizer_usage(flat).perplexity)

    def test_it_reads_the_masked_positions_of_a_stream(self, cifar) -> None:
        batches = build(BEIT)
        ids = masked_code_ids(batches.select_ds)
        elements = collect(batches.select_ds)
        expected = np.concatenate([e[1][e[0][1].astype(bool)] for e in elements])
        assert ids.ndim == 1 and np.array_equal(ids, expected)
        assert tokenizer_usage(ids).distinct > 1

    @pytest.mark.parametrize("bad", [np.array([]), np.array([0.5, 1.5]), np.array([-1, 2]), np.array([True])])
    def test_a_bad_input_is_refused(self, bad) -> None:
        with pytest.raises(ValueError):
            tokenizer_usage(bad)

    def test_it_serializes_to_plain_numbers(self) -> None:
        record = tokenizer_usage(np.array([0, 0, 1])).as_dict()
        assert set(record) == {"n_codes", "distinct", "perplexity", "top1_share", "entropy"}
        assert all(isinstance(value, (int, float)) for value in record.values())
