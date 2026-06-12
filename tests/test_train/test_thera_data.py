"""Scoped smoke test for the THERA arbitrary-scale ``tf.data`` pipeline.

Builds a tiny random-PNG corpus, pulls two batches, and asserts EXACT output
shapes plus distributional invariants (value ranges, no NaNs, randomness). This
is a smoke test, not a correctness oracle for the SR math — but per LESSONS it
asserts ranges, not merely "it ran".
"""

import numpy as np
import pytest
import tensorflow as tf

from train.thera.data import build_arbitrary_scale_dataset, _make_grid_tf
from dl_techniques.layers.grid_sample import make_grid


SOURCE_SIZE = 48
TARGET_SAMPLES = 48
BATCH = 4
SCALE_RANGE = (1.2, 4.0)
AUGMENT_SCALE_RANGE = (1.0, 2.0)


@pytest.fixture()
def corpus(tmp_path):
    """Write ~6 random 128x128 RGB PNGs to a temp dir; return the dir path."""
    rng = np.random.default_rng(1234)
    for i in range(6):
        arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        png = tf.io.encode_png(tf.convert_to_tensor(arr))
        path = tmp_path / f"img_{i:02d}.png"
        tf.io.write_file(str(path), png)
    return str(tmp_path)


def test_import_smoke():
    """The public factory imports under the canonical train.thera path."""
    from train.thera.data import build_arbitrary_scale_dataset as fn

    assert callable(fn)


def test_shapes_exact(corpus):
    """One batch has EXACTLY the documented shapes."""
    ds = build_arbitrary_scale_dataset(
        corpus,
        source_size=SOURCE_SIZE,
        target_samples=TARGET_SAMPLES,
        scale_range=SCALE_RANGE,
        augment_scale_range=AUGMENT_SCALE_RANGE,
        batch_size=BATCH,
        repeat=False,
        seed=0,
    )
    batch = next(iter(ds))

    assert tuple(batch["source"].shape) == (BATCH, 48, 48, 3)
    assert tuple(batch["target_coords"].shape) == (BATCH, 48, 48, 2)
    assert tuple(batch["target"].shape) == (BATCH, 48, 48, 3)
    assert tuple(batch["source_nearest"].shape) == (BATCH, 48, 48, 3)
    assert tuple(batch["scale"].shape) == (BATCH,)


def test_distributional_invariants(corpus):
    """Value ranges, no-NaN, and scale bounds all hold."""
    ds = build_arbitrary_scale_dataset(
        corpus,
        source_size=SOURCE_SIZE,
        target_samples=TARGET_SAMPLES,
        scale_range=SCALE_RANGE,
        augment_scale_range=AUGMENT_SCALE_RANGE,
        batch_size=BATCH,
        repeat=False,
        seed=0,
    )
    batch = next(iter(ds))

    # No NaNs anywhere.
    for key, val in batch.items():
        arr = val.numpy()
        assert not np.isnan(arr).any(), f"NaN in '{key}'"

    # Image tensors in [0, 1] (tiny epsilon for bicubic overshoot tolerance).
    for key in ("source", "target", "source_nearest"):
        arr = batch[key].numpy()
        assert arr.min() >= -0.01, f"{key} below 0: {arr.min()}"
        assert arr.max() <= 1.01, f"{key} above 1: {arr.max()}"

    # target_coords pixel-center grid in [-0.5, 0.5].
    coords = batch["target_coords"].numpy()
    assert coords.min() >= -0.5 - 1e-5, coords.min()
    assert coords.max() <= 0.5 + 1e-5, coords.max()

    # scale in [scale_min*~0.9, scale_max*augment_max*1.1] and > 1.0.
    scale = batch["scale"].numpy()
    lo = SCALE_RANGE[0] * 0.9
    hi = SCALE_RANGE[1] * AUGMENT_SCALE_RANGE[1] * 1.1
    assert (scale > 1.0).all(), f"scale not > 1: {scale}"
    assert (scale >= lo).all(), f"scale below {lo}: {scale.min()}"
    assert (scale <= hi).all(), f"scale above {hi}: {scale.max()}"


def test_second_batch_differs(corpus):
    """Randomness/shuffle: a second batch differs from the first."""
    # repeat=True so two full batches exist even though the tiny corpus (6
    # images, drop_remainder) yields only one batch per epoch.
    ds = build_arbitrary_scale_dataset(
        corpus,
        source_size=SOURCE_SIZE,
        target_samples=TARGET_SAMPLES,
        scale_range=SCALE_RANGE,
        augment_scale_range=AUGMENT_SCALE_RANGE,
        batch_size=BATCH,
        repeat=True,
        seed=0,
    )
    it = iter(ds.take(2))
    b0 = next(it)
    b1 = next(it)
    # The source LR images of the two batches should not be identical.
    assert not np.allclose(
        b0["source"].numpy(), b1["source"].numpy()
    ), "two consecutive batches are identical (randomness not working)"


# ---------------------------------------------------------------------
# A9(d) (review): the dynamic-side pipeline grid `_make_grid_tf` must match the
# static inference grid `make_grid` elementwise, so the training-time query
# coords and the inference-time coords share ONE pixel-center convention. A
# divergence would silently shift every training query off the inference grid.
# ---------------------------------------------------------------------


def test_make_grid_matches_grid_sample():
    n = 5
    pipeline = _make_grid_tf(tf.constant(n, dtype=tf.int32)).numpy()  # (n, n, 2)
    inference = make_grid(n)  # (n, n, 2) numpy, int side
    assert pipeline.shape == inference.shape == (n, n, 2)
    np.testing.assert_allclose(pipeline, inference, atol=1e-6)


# ---------------------------------------------------------------------
# 9. THERA review caveats (plan_2026-06-12_f8843c4f):
#    OBS-1 deterministic validation, INV-1 train-still-random, REV-W1
#    small-corpus >=1 batch, deterministic val scale.
# ---------------------------------------------------------------------


_VAL_KW = dict(
    source_size=48,
    target_samples=16,
    scale_range=(1.2, 2.0),
    augment_scale_range=(1.0, 2.0),
    augment_scale_prob=0.5,
)


def _build_val(corpus, **overrides):
    kw = dict(_VAL_KW)
    kw.update(overrides)
    return build_arbitrary_scale_dataset(
        corpus,
        training=False,
        drop_remainder=False,
        repeat=False,
        shuffle=False,
        batch_size=kw.pop("batch_size", 2),
        **kw,
    )


_VAL_KEYS = ("source", "target", "target_coords", "source_nearest", "scale")


def test_val_pipeline_deterministic(corpus):
    """OBS-1: two independent val builds AND two iterations of one build are
    elementwise identical for every output field."""
    ds1 = _build_val(corpus)
    ds2 = _build_val(corpus)
    b1 = next(iter(ds1))
    b2 = next(iter(ds2))
    for k in _VAL_KEYS:
        assert np.array_equal(
            np.array(b1[k]), np.array(b2[k])
        ), f"val field '{k}' differs across two independent builds (non-deterministic)"

    # Re-iterate the SAME dataset twice: first batch must be identical (stable
    # across epochs).
    a = next(iter(ds1))
    b = next(iter(ds1))
    for k in _VAL_KEYS:
        assert np.array_equal(
            np.array(a[k]), np.array(b[k])
        ), f"val field '{k}' differs across two iterations of one dataset"


def test_train_pipeline_still_random(corpus):
    """INV-1 guard: the training path stays stochastic. Two consecutive batches
    from a training dataset must differ (determinism must NOT leak into train)."""
    ds = build_arbitrary_scale_dataset(
        corpus,
        training=True,
        drop_remainder=False,
        repeat=True,
        shuffle=True,
        batch_size=2,
        **_VAL_KW,
    )
    it = iter(ds)
    b1 = next(it)
    b2 = next(it)
    target_differs = not np.array_equal(
        np.array(b1["target"]), np.array(b2["target"])
    )
    scale_differs = not np.array_equal(
        np.array(b1["scale"]), np.array(b2["scale"])
    )
    assert target_differs or scale_differs, (
        "two consecutive training batches are identical — determinism leaked "
        "into the training path (INV-1 violated)"
    )


def test_val_small_corpus_yields_batch(tmp_path):
    """REV-W1: a non-empty corpus smaller than batch_size still yields >=1 val
    batch when drop_remainder=False."""
    rng = np.random.default_rng(7)
    for i in range(2):  # 2 images < batch_size=8
        arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        png = tf.io.encode_png(tf.convert_to_tensor(arr))
        tf.io.write_file(str(tmp_path / f"img_{i:02d}.png"), png)
    corpus = str(tmp_path)

    ds_keep = build_arbitrary_scale_dataset(
        corpus,
        training=False,
        drop_remainder=False,
        repeat=False,
        shuffle=False,
        batch_size=8,
        **_VAL_KW,
    )
    assert len(list(ds_keep)) >= 1, "small val corpus yielded zero batches (REV-W1)"

    # Contrast: drop_remainder=True drops the only (partial) batch -> 0 batches.
    ds_drop = build_arbitrary_scale_dataset(
        corpus,
        training=False,
        drop_remainder=True,
        repeat=False,
        shuffle=False,
        batch_size=8,
        **_VAL_KW,
    )
    assert len(list(ds_drop)) == 0, (
        "drop_remainder=True should drop the only partial batch (documents the "
        "exact failure REV-W1 fixes)"
    )


def test_val_scale_is_midpoint(corpus):
    """The deterministic val ``scale`` (effective_scale = target_size/source_size)
    is constant across the batch and identical across independent builds. We do
    NOT couple to the raw midpoint value (target_size rounding)."""
    ds1 = _build_val(corpus, scale_range=(1.2, 2.0))
    ds2 = _build_val(corpus, scale_range=(1.2, 2.0))
    s1 = np.array(next(iter(ds1))["scale"])
    s2 = np.array(next(iter(ds2))["scale"])
    assert np.allclose(s1, s1.flat[0]), f"val scale not constant across batch: {s1}"
    assert np.array_equal(s1, s2), "val scale differs across two builds (non-deterministic)"
