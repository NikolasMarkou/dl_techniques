"""Direct tests for `build_raw_image_dataset` — the shared pipeline builder.

Written for plan-2026-08-01T195746-12a1f2db step 5. Until then this function had
NO dedicated test module, despite being shared by **7 call sites across 3
trainers** (`train_classification.py` x2, `train_masked_completion.py` x2,
`train_dino.py` x3). The change that motivated the module — an
`indexed_element_map_fn` that moves `.repeat()` before the element map and
inserts an `.enumerate()` — alters the pipeline SHAPE, not just a parameter, so
the then-DEFAULT-OFF path needs a guard that can see a shape change.

Four things here are load bearing:

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
4. **`test_a_file_order_seed_composes_with_an_indexed_map_fn`** covers the
   combination the DINO trainer now runs BY DEFAULT (see the inventory below).
   Read its docstring for what it does and does not reach.
5. **`TestTheImagenetteBranchWhereTheSeedIsLive`** covers that same pairing on
   the branch where `shuffle_files_seed` is actually READ, which is the one
   thing item 4 structurally cannot do (plan-2026-08-03T043010-cecf4357
   step 9).

Every dataset here is `cifar10` — which `build_raw_image_dataset` builds
IN-MEMORY from `keras.datasets`, no TFDS, no network, no spinning disk — WITH
ONE DELIBERATE EXCEPTION: `TestTheImagenetteBranchWhereTheSeedIsLive` reads
real prepared `imagenette/320px-v2` TFRecords off disk, because
`shuffle_files_seed` reaches `tfds.ReadConfig(shuffle_seed=...)` on that branch
and NOWHERE else, so no cifar10 test can observe it. That class is
`skipif`-guarded on the records being present and never downloads anything;
it is bounded to `.take(2)` batches of 4 at `image_size=32` and MEASURED at
~2 s. Do not generalize it into a second TFDS-dependent test without measuring
the cost — the "no spinning disk" property of the rest of this module is why it
is cheap.

---

## The non-DINO impact audit (plan-2026-08-02T132301-93deeae2 step 9)

Recorded HERE rather than in a plan directory because `plans/` is gitignored:
after a fresh checkout this docstring is the only surviving copy.

**The `indexed_element_map_fn=None` path is unchanged STRUCTURALLY, not merely
measured-equal.** Commit `5f31ad3a` added a leading `if indexed_element_map_fn
is not None:` branch and turned the pre-existing `if is_training:` tail into
`elif is_training:`, touching nothing inside either pre-existing branch body.
Python's `elif` can only divert control flow when the new leading condition is
true, so at `indexed_element_map_fn=None` the `elif is_training / else` pair
evaluates exactly as the old `if / else` did. The byte-identity measurements
(this module's `TestDefaultOffIsUnchanged`, and the 56-array / 414,832-value
worktree comparison in that commit's message) AGREE with that reading; they are
not what establishes it.

**No call site can route into the new branch by accident.**
`indexed_element_map_fn` sits after the `*` in the signature
(`common.py:88-102`), i.e. it is keyword-only, so no positional call — however
many arguments it passes — can ever land in that slot.

**Call-site inventory, re-derived at plan-2026-08-02T132301-93deeae2 step 9.**
Still 7 invocations across 3 trainers. Line numbers drift; the enclosing symbol
is the durable citation:

| # | File / enclosing symbol | Split | Relevant kwargs |
|---|---|---|---|
| 1 | `train_masked_completion.py` `build_datasets` | train | `element_map_fn=map_fn`, `augment`, `seed` |
| 2 | `train_masked_completion.py` `build_datasets` | val | `element_map_fn=map_fn`, `seed`, `is_training=False` |
| 3 | `train_classification.py` `build_datasets` | train | `augment`, `seed` (no map fn) |
| 4 | `train_classification.py` `build_datasets` | val | `seed`, `is_training=False` (no map fn) |
| 5 | `train_dino.py` `build_dataset` | train | `augment=False`, `seed`, `**map_fn_kwarg`, `**stream_seed_kwarg` |
| 6 | `train_dino.py` `build_knn_datasets` | bank | `augment=False`, `seed`, `shuffle_files_seed=config.seed` |
| 7 | `train_dino.py` `build_knn_datasets` | query | `seed`, `is_training=False` (no map fn) |

Sites 1-4, 6 and 7 pass every argument EXPLICITLY — no `**` spread — and the
string `indexed_element_map_fn` does not occur anywhere in
`train_masked_completion.py` or `train_classification.py`. Those six cannot
reach the new branch at any flag setting; only site 5's `map_fn_kwarg` dict can
name that slot.

**The old "all 7 sites are non-indexed at defaults" sentence is now FALSE and
must not be reinstated.** `plan-2026-08-02T132301-93deeae2` step 4 flipped
`TrainingConfig.stateless_augmentation` to `True`, so site 5 takes the INDEXED
branch BY DEFAULT; `--no-stateless-augmentation` is the off-switch back to
`element_map_fn`. `seed_training_stream` flipped to `True` in the same step, so
site 5 also passes `shuffle_files_seed=config.seed` by default — which is why
that pairing gets a test below.

**Three latent hazards for a FUTURE caller**, none of them a defect today:

1. The `is_training=False` refusal (`common.py:181-190`) is a hard stop with no
   fallback. A caller wanting a reproducible EVAL-time augmentation cannot get
   one through this seam and needs a different mechanism — relaxing the guard
   would reintroduce the frozen-per-image failure D-035 RED-proved wrong.
2. `_call_indexed`'s nested-tuple unpacking heuristic
   (`len(element) == 1 and isinstance(element[0], tuple)`, `common.py:304-307`)
   is untested against anything but a 2-tuple `(image, label)`, which is all any
   of the 7 sites produces. A `dict` element, or a 3+-element flat tuple, could
   misroute the unpacking silently rather than raise.
3. `shuffle_files_seed` together with `indexed_element_map_fn` — now covered on
   BOTH branches, by two tests with deliberately different reach.
   `test_a_file_order_seed_composes_with_an_indexed_map_fn` runs on `cifar10`,
   whose in-memory branch (`common.py`'s `else`) accepts `shuffle_files_seed`
   and never reads it, so it closes REFUSAL-FREEDOM (the two kwargs compose
   without raising) and COUNTER-MONOTONICITY (the per-element index still
   advances) ONLY — it asserts that inertness positively and cannot see a file
   interleave. `TestTheImagenetteBranchWhereTheSeedIsLive` closes the rest on
   the branch that does reach `tfds.ReadConfig(shuffle_seed=...)`, which is
   where site 5 runs in production: the seed is READ (different seeds emit
   different records), it REPRODUCES across builds, and the counter still
   climbs. **The old sentence here — "Nothing in this repository exercises
   seeded FILE ORDER together with an indexed map fn on the branch where the
   seed is live" — is FALSE as of
   plan-2026-08-03T043010-cecf4357 step 9 and must not be reinstated.** What
   remains uncovered is narrower and worth stating: only 8 records off the HEAD
   of the train stream are compared, at 2 of the split's possible file orders
   (`imagenette/320px-v2` ships 2 train shards here), and nothing asserts WHICH
   order a given seed selects — that is a tfds internal a version bump may
   permute.
"""

import os
from glob import glob
from typing import Any, List, Tuple

import numpy as np
import pytest
import tensorflow as tf

from train.energy_transformer.common import (
    DATASET_NUM_CLASSES,
    IMAGENETTE_TFDS_NAME,
    build_raw_image_dataset,
)

IMAGE_SIZE = 32
BATCH_SIZE = 4
SEED = 7

# The ONE imagenette-dependent test below (class
# `TestTheImagenetteBranchWhereTheSeedIsLive`) reads these prepared TFDS
# records off disk. Everything else in this module is cifar10 and needs none
# of it. `data_dir=None` inside `build_raw_image_dataset` inherits
# `$TFDS_DATA_DIR`, so the skip predicate must resolve the same way TFDS does.
_TFDS_DATA_DIR = os.environ.get(
    "TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets"))
_IMAGENETTE_RECORD_DIR = os.path.join(
    _TFDS_DATA_DIR, *IMAGENETTE_TFDS_NAME.split("/"), "1.0.0")
_IMAGENETTE_TRAIN_SHARDS = sorted(
    glob(os.path.join(_IMAGENETTE_RECORD_DIR, "imagenette-train.tfrecord-*")))
_IMAGENETTE_SKIP_REASON = (
    f"imagenette TFDS records are not prepared: no "
    f"`imagenette-train.tfrecord-*` under {_IMAGENETTE_RECORD_DIR!r} "
    f"(TFDS_DATA_DIR={_TFDS_DATA_DIR!r}). This test reads REAL records; it "
    f"must never download anything. Prepare `{IMAGENETTE_TFDS_NAME}` offline "
    f"to run it. A SKIP here is NOT a pass -- the seeded-file-order guard "
    f"simply did not run on this machine."
)


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


def _index_and_image_digest_map_fn(
        index: tf.Tensor, image: tf.Tensor, label: tf.Tensor
) -> Tuple[Any, Any]:
    """Emit the counter AND a per-record fingerprint of the image.

    The label alone cannot identify a record (10 classes, 8 samples), so it
    cannot answer "were 8 DISTINCT records actually read" nor "did the file
    order change". `(sum, max, min)` over the normalized image does: it is a
    float64 triple that differs between any two natural images in practice,
    and it is cheap enough to compute inside the pipeline.
    """
    del label
    digest = tf.stack(
        [tf.reduce_sum(image), tf.reduce_max(image), tf.reduce_min(image)])
    return tf.cast(index, tf.int64), tf.cast(digest, tf.float64)


def _take(ds: tf.data.Dataset, n_batches: int) -> List[np.ndarray]:
    return [np.asarray(x) for x, _ in ds.take(n_batches)]


def _take_indices_and_digests(
        ds: tf.data.Dataset, n_batches: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Drain `n_batches` of an `_index_and_image_digest_map_fn` pipeline.

    Returns `(counters[n_batches * BATCH_SIZE], digests[same, 3])`, both
    concatenated across batches so the caller sees one flat window.
    """
    counters: List[np.ndarray] = []
    digests: List[np.ndarray] = []
    for counter, digest in ds.take(n_batches):
        counters.append(np.asarray(counter))
        digests.append(np.asarray(digest))
    return np.concatenate(counters), np.concatenate(digests)


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

    def test_a_file_order_seed_composes_with_an_indexed_map_fn(self) -> None:
        """`shuffle_files_seed` + `indexed_element_map_fn`, the DINO default.

        `train_dino.py`'s `build_dataset` passes BOTH whenever
        `stateless_augmentation` and `seed_training_stream` are on, which since
        plan-2026-08-02T132301-93deeae2 step 4 is the shipped default. No test
        had ever driven the two kwargs together.

        WHAT THIS COVERS. That the two compose at all — neither refusal in
        `common.py:172-190` fires, and the `.enumerate()`-after-`.repeat()`
        counter still runs monotonically across the epoch boundary with a file
        seed also set. A future refusal or reordering coupling the two would
        land here.

        WHAT THIS DOES NOT COVER, AND WHAT NOW DOES. `shuffle_files_seed` only
        reaches `tfds.ReadConfig(shuffle_seed=...)` on the IMAGENETTE branch;
        on the in-memory cifar10 branch this test uses it is accepted and then
        never read. So THIS test cannot observe the file interleave, and does
        not pretend to: it asserts that inertness POSITIVELY (the emitted
        elements match the no-file-seed build). The sentence that used to
        follow — that an imagenette version "would need TFDS records on disk,
        which no test in this module depends on" — is now OUT OF DATE:
        `TestTheImagenetteBranchWhereTheSeedIsLive` at the bottom of this
        module does exactly that, `skipif`-guarded on the prepared records,
        and is where the seed being READ and REPRODUCIBLE is asserted. Keep
        this cifar10 test anyway: it is the only one that runs when those
        records are absent, and it is the only one that pins the seed's
        INERTNESS on the in-memory branch — if `shuffle_files_seed` ever starts
        perturbing cifar10, this fires and the imagenette test does not.
        The file-order effect is also measured in `train_dino.py`'s
        `build_knn_datasets` docstring (D-040).
        """
        with_seed, _, _ = build_raw_image_dataset(
            "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=True, augment=False,
            indexed_element_map_fn=_indexed_pair_map_fn, seed=SEED,
            shuffle_files_seed=SEED)
        without_seed, _, _ = build_raw_image_dataset(
            "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=True, augment=False,
            indexed_element_map_fn=_indexed_pair_map_fn, seed=SEED)

        for index, (a, b) in enumerate(
                zip(_take(with_seed, 3), _take(without_seed, 3))):
            np.testing.assert_array_equal(a, b, err_msg=(
                f"batch {index}: `shuffle_files_seed` changed the cifar10 "
                f"stream. It must reach ONLY the TFDS file interleave; if it "
                f"starts affecting the in-memory branch, this test's stated "
                f"coverage boundary is wrong, not just its assertion."
            ))

        counting_ds, num_examples, _ = build_raw_image_dataset(
            "cifar10", IMAGE_SIZE, BATCH_SIZE, is_training=True, augment=False,
            indexed_element_map_fn=_index_reporting_map_fn, seed=SEED,
            shuffle_files_seed=SEED)
        steps_per_epoch = num_examples // BATCH_SIZE
        counters = np.concatenate(
            _take(counting_ds.skip(steps_per_epoch - 2), 4))
        expected_first = (steps_per_epoch - 2) * BATCH_SIZE
        np.testing.assert_array_equal(
            counters,
            np.arange(expected_first, expected_first + 4 * BATCH_SIZE),
            err_msg=(
                "the per-element counter stops running monotonically across "
                "the epoch boundary once `shuffle_files_seed` is also passed. "
                "The two knobs are supposed to be independent -- this is the "
                "combination the DINO trainer runs by default."
            ),
        )
        assert counters[-1] >= num_examples > counters[0]


# ---------------------------------------------------------------------
# 4. the imagenette branch -- the ONE place `shuffle_files_seed` is live
# ---------------------------------------------------------------------


@pytest.mark.skipif(not _IMAGENETTE_TRAIN_SHARDS, reason=_IMAGENETTE_SKIP_REASON)
class TestTheImagenetteBranchWhereTheSeedIsLive:
    """Real TFDS records, because cifar10 structurally cannot carry this claim.

    `test_a_file_order_seed_composes_with_an_indexed_map_fn` above closes F-15c
    for REFUSAL-FREEDOM and COUNTER-MONOTONICITY only: it runs on cifar10,
    whose in-memory branch accepts `shuffle_files_seed` and never reads it.
    This class runs the SAME pairing on `imagenette`, the only branch that
    builds `tfds.ReadConfig(shuffle_seed=...)` and hands it to
    `builder.as_dataset` -- i.e. the branch DINO site 5 actually runs in
    production.

    Cost control, so this stays a unit test: `image_size=32`, `batch_size=4`,
    `.take(2)`, and an explicit `shuffle_buffer` of 32 (the 4096 default would
    decode 4096 full-size JPEGs per build). CPU only, no model, no `fit()`.
    MEASURED at 16 builds: ~2 s total.
    """

    # Two `.take()` batches of `BATCH_SIZE`; small on purpose (see class doc).
    N_BATCHES = 2

    # DECISION plan-2026-08-03T043010-cecf4357/D-014
    # Do NOT reduce this to a single same-seed determinism check, and do NOT
    # reduce it to one hardcoded "seed A differs from seed B" pair.
    # `imagenette/320px-v2` ships exactly TWO train shards, so the file order
    # the seed selects has only TWO possible values. Consequences, both
    # measured: (i) a same-seed equality assertion ALONE is near-vacuous,
    # because a seed-IGNORED implementation still coincides ~50% of the time;
    # (ii) a single different-seed pair is a coin flip in the other direction,
    # so it would RED-prove nothing reliably. Asserting the PARTITION over
    # several seeds is what makes both halves sharp: a seed-ignored build
    # draws its order at random per call, so it must fail per-seed
    # reproducibility with probability 1 - 2^-len(FILE_ORDER_SEEDS).
    # The seed->order map is NOT hardcoded (it is a tfds-internal detail that a
    # tfds upgrade may permute); the test only asserts that the seeds span more
    # than one order. See decisions.md D-014.
    FILE_ORDER_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)

    def _build(self, shuffle_files_seed: int) -> Tuple[np.ndarray, np.ndarray]:
        ds, num_examples, num_classes = build_raw_image_dataset(
            "imagenette", IMAGE_SIZE, BATCH_SIZE, is_training=True,
            augment=False,
            indexed_element_map_fn=_index_and_image_digest_map_fn,
            seed=SEED, shuffle_files_seed=shuffle_files_seed,
            shuffle_buffer=32)
        assert num_examples == 9469 and num_classes == 10, (
            f"the imagenette train split reports {num_examples} examples / "
            f"{num_classes} classes, not 9469/10. The records on disk are not "
            f"the split this test was measured against; every number below is "
            f"about a different dataset."
        )
        return _take_indices_and_digests(ds, self.N_BATCHES)

    def test_a_live_file_order_seed_composes_with_an_indexed_map_fn(
            self) -> None:
        """The F-15c residual, closed on the branch where the seed is live.

        Three claims, none of which cifar10 can support:

        1. **The seed is READ.** Different `shuffle_files_seed` values produce
           different emitted records, so the kwarg is not inert here.
        2. **The seed REPRODUCES.** The same value, built twice, emits the
           identical window -- which is the entire point of D-040 (a consumer
           taking a small `.take(n)` sample and reporting a number off it).
        3. **The two knobs COMPOSE.** The `indexed_element_map_fn` counter
           still runs `0,1,2,...` across that window; neither knob clobbers
           the other.

        Plus the anti-vacuity floor: a `.take()` that yields nothing satisfies
        every equality assertion trivially, so the number of DISTINCT records
        actually read is asserted explicitly.
        """
        window = self.N_BATCHES * BATCH_SIZE
        fingerprints = {}

        for file_seed in self.FILE_ORDER_SEEDS:
            first_counters, first_digests = self._build(file_seed)
            second_counters, second_digests = self._build(file_seed)

            # ANTI-VACUITY. Everything else here is an equality assertion, and
            # equality over an EMPTY window is free.
            assert first_digests.shape == (window, 3), (
                f"shuffle_files_seed={file_seed}: read "
                f"{first_digests.shape[0]} records, expected {window}. An "
                f"empty or short `.take()` would pass every equality "
                f"assertion below trivially."
            )
            assert np.unique(first_digests, axis=0).shape[0] == window, (
                f"shuffle_files_seed={file_seed}: the {window} records read "
                f"are not {window} DISTINCT records "
                f"({np.unique(first_digests, axis=0).shape[0]} unique image "
                f"fingerprints). A pipeline replaying one record would make "
                f"the determinism assertion meaningless."
            )

            # (3) the counter still climbs -- the two knobs compose.
            np.testing.assert_array_equal(
                first_counters, np.arange(window), err_msg=(
                    f"shuffle_files_seed={file_seed}: the per-element counter "
                    f"is not 0..{window - 1} on the imagenette branch. "
                    f"`shuffle_files_seed` must reach the TFDS file "
                    f"interleave ONLY -- if it perturbs `.enumerate()`, the "
                    f"stateless augmentation keyed on that counter is keyed "
                    f"on something else."
                ))

            # (2) same seed -> same window, ACROSS BUILDS.
            np.testing.assert_array_equal(
                first_digests, second_digests, err_msg=(
                    f"shuffle_files_seed={file_seed}: two builds at the SAME "
                    f"file-order seed emitted DIFFERENT records. The seed "
                    f"exists so a small `.take(n)` sample is stable run to "
                    f"run (D-040); if it is dropped from "
                    f"`tfds.ReadConfig(shuffle_seed=...)`, the file order is "
                    f"redrawn per call and this is the assertion that fires."
                ))
            np.testing.assert_array_equal(
                first_counters, second_counters,
                err_msg="the counter itself is not reproducible across builds")

            fingerprints[file_seed] = first_digests.tobytes()

        # (1) the seed is READ: it selects among the available file orders.
        assert len(set(fingerprints.values())) >= 2, (
            f"all {len(self.FILE_ORDER_SEEDS)} file-order seeds emitted the "
            f"IDENTICAL window, so `shuffle_files_seed` changed nothing and "
            f"the reproducibility assertion above is vacuous for the seed. "
            f"`imagenette/320px-v2` has 2 train shards here, i.e. 2 possible "
            f"orders; MEASURED on this machine, seeds {{4, 5}} take one order "
            f"and {{0, 1, 2, 3, 6, 7}} the other. If a tfds upgrade collapsed "
            f"these 8 seeds onto one order this fires WITHOUT a defect -- "
            f"widen FILE_ORDER_SEEDS rather than deleting the assertion."
        )
