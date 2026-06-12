"""Scoped smoke test for the THERA arbitrary-scale ``tf.data`` pipeline.

Builds a tiny random-PNG corpus, pulls two batches, and asserts EXACT output
shapes plus distributional invariants (value ranges, no NaNs, randomness). This
is a smoke test, not a correctness oracle for the SR math — but per LESSONS it
asserts ranges, not merely "it ran".
"""

import numpy as np
import pytest
import tensorflow as tf

from train.thera.data import build_arbitrary_scale_dataset


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
