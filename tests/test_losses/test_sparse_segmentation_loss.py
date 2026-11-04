"""Dice, CE + Dice and focal on logits and sparse targets with a void label (D-014).

For each of ``SparseDiceLoss``, ``SparseCeDiceLoss`` and ``SparseFocalLoss``:

* a value worked out by hand with numpy on a 2x2 image of 2 classes (one void pixel);
* the value AND the gradient do not change when the logits at void pixels change;
* a batch with no valid pixel is exactly 0, finite, with a zero gradient (and goes NaN without
  the smoothing, which the RED mutants of the step prove);
* ``loss(w=[1,..,1,0]) * B/(B-1) == loss(batch[:-1])`` (``losses/AGENTS.md``: one value per image);
* a ``.keras`` file compiled with the loss reloads in an interpreter that cannot import ``train``;
* finite under ``mixed_float16``; one ``fit`` step lowers the loss.

Plus the relations to the references the definitions were fixed from: focal at ``gamma = 0`` is
the cross-entropy, ``ce_dice`` is twice the legacy ``combo`` at its smoothing, the dense
``dice_per_sample`` is unchanged when no mask is given, and the old two-sided focal defect (a
confident correct pixel charged) is gone.
"""

import json
import subprocess
import sys
import textwrap

import keras
import numpy as np
import pytest

from dl_techniques.losses import segmentation_metrics as backend
from dl_techniques.losses.segmentation_loss import LossConfig
from dl_techniques.losses.segmentation_wrapper_loss import SegmentationWrapperLoss
from dl_techniques.losses.sparse_segmentation_loss import (
    DEFAULT_DICE_SMOOTH,
    SparseCeDiceLoss,
    SparseDiceLoss,
    SparseFocalLoss,
)

VOID = 255

#: A 2x2 image, 2 classes, one void pixel. Hand-worked below.
LOGITS = np.array([[[[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]]], dtype="float32")
TARGET = np.array([[[0, 1], [VOID, 1]]], dtype="int32")

LOSSES = {
    "dice": lambda ignore=VOID, **kw: SparseDiceLoss(ignore_index=ignore, **kw),
    "ce_dice": lambda ignore=VOID, **kw: SparseCeDiceLoss(ignore_index=ignore, **kw),
    "focal": lambda ignore=VOID, **kw: SparseFocalLoss(ignore_index=ignore, **kw),
}


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def hand_dice(logits, target, ignore, smooth=DEFAULT_DICE_SMOOTH):
    """Per image: mean over classes of 1 - (2 I + s) / (G + P + s), void pixels left out."""
    p = softmax(logits.astype("float64"))
    out = []
    for b in range(logits.shape[0]):
        valid = (target[b] != ignore) if ignore is not None else np.ones(target[b].shape, bool)
        g = np.eye(logits.shape[-1])[np.where(valid, target[b], 0)] * valid[..., None]
        scores = []
        for c in range(logits.shape[-1]):
            inter = (g[..., c] * p[b][..., c] * valid).sum()
            scores.append((2 * inter + smooth) / (g[..., c].sum() + (p[b][..., c] * valid).sum() + smooth))
        out.append(1.0 - np.mean(scores))
    return np.array(out)


def hand_ce_and_focal(logits, target, ignore, gamma):
    p = softmax(logits.astype("float64"))
    out = []
    for b in range(logits.shape[0]):
        valid = target[b] != ignore
        pt = np.take_along_axis(p[b], np.where(valid, target[b], 0)[..., None], -1)[..., 0]
        fl = -((1 - pt) ** gamma) * np.log(pt)
        out.append((fl * valid).sum() / max(valid.sum(), 1))
    return np.array(out)


def expected(name, logits, target, ignore=VOID):
    if name == "dice":
        return hand_dice(logits, target, ignore)
    if name == "focal":
        return hand_ce_and_focal(logits, target, ignore, 2.0)
    return hand_ce_and_focal(logits, target, ignore, 0.0) + hand_dice(logits, target, ignore)


def value_and_grad(loss, y, z):
    z = keras.ops.convert_to_tensor(z)
    import tensorflow as tf

    with tf.GradientTape() as tape:
        tape.watch(z)
        value = loss(y, z)
    return float(value), tape.gradient(value, z).numpy()


def rng_batch(batch=4, size=6, classes=3, void_share=0.3, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, (batch, size, size, classes)).astype("float32")
    target = rng.integers(0, classes, (batch, size, size)).astype("int32")
    target[rng.random(target.shape) < void_share] = VOID
    return logits, target


# ---------------------------------------------------------------------
# hand-computed values
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LOSSES))
def test_the_value_equals_the_numpy_hand_computation(name):
    got = float(LOSSES[name]()(TARGET, LOGITS))
    assert got == pytest.approx(float(expected(name, LOGITS, TARGET)[0]), abs=1e-6)


def test_the_hand_values_are_these_numbers():
    # Worked out on paper for the fixture: p(class 0) = [[.8808, .2689], [.5, .1192]] and the labels
    # [[0, 1], [void, 1]] keep three pixels with p_t = .8808, .7311, .8808.
    assert float(SparseDiceLoss(VOID)(TARGET, LOGITS)) == pytest.approx(0.179792, abs=1e-5)
    assert float(SparseFocalLoss(VOID)(TARGET, LOGITS)) == pytest.approx(0.0087551, abs=1e-6)
    assert float(SparseCeDiceLoss(VOID)(TARGET, LOGITS)) == pytest.approx(0.368831, abs=1e-5)


def test_the_gamma_zero_focal_is_the_stock_cross_entropy_of_each_image():
    logits, target = rng_batch(batch=3, void_share=0.0, seed=1)
    stock = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for b in range(3):
        one = float(SparseFocalLoss(None, gamma=0.0)(target[b:b + 1], logits[b:b + 1]))
        assert one == pytest.approx(float(stock(target[b:b + 1], logits[b:b + 1])), abs=1e-6)


def test_ce_dice_is_twice_the_legacy_combo_at_the_legacy_smoothing():
    logits, target = rng_batch(batch=3, void_share=0.0, seed=2)
    one_hot = np.eye(3, dtype="float32")[target]
    combo = SegmentationWrapperLoss("combo", LossConfig(num_classes=3, from_logits=True))
    ours = SparseCeDiceLoss(ignore_index=None, smooth=1e-6)
    assert float(ours(target, logits)) == pytest.approx(2.0 * float(combo(one_hot, logits)), rel=1e-5)


def test_a_confident_correct_pixel_costs_nothing_to_the_focal_loss():
    # The two-sided function this replaced charged p_t^gamma log(1 - p_t) on the TRUE class.
    logits = np.zeros((1, 2, 2, 3), "float32")
    logits[..., 1] = 30.0
    target = np.ones((1, 2, 2), "int32")
    assert float(SparseFocalLoss(VOID)(target, logits)) == pytest.approx(0.0, abs=1e-12)
    assert float(backend.focal_scalar(np.eye(3, dtype="float32")[target], logits, from_logits=True)) < 1e-12


def test_the_dense_dice_without_a_mask_is_the_number_it_always_was():
    logits, target = rng_batch(batch=2, void_share=0.0, seed=3)
    one_hot = np.eye(3, dtype="float32")[target]
    probs = softmax(logits)
    unmasked = backend.dice_per_sample(one_hot, probs, smooth=1e-6)
    ones = backend.dice_per_sample(one_hot, probs, smooth=1e-6, valid=np.ones(target.shape, "float32"))
    assert np.array_equal(keras.ops.convert_to_numpy(unmasked), keras.ops.convert_to_numpy(ones))
    assert np.allclose(unmasked, hand_dice(logits, target, None, smooth=1e-6), atol=1e-6)


def test_the_focal_function_refuses_what_it_cannot_score():
    ok = np.zeros((1, 2, 2, 2), "float32")
    with pytest.raises(ValueError, match="gamma"):
        backend.focal_per_sample(ok, ok, gamma=-1.0)
    one_channel = np.zeros((1, 2, 2, 1), "float32")
    with pytest.raises(ValueError, match="at least 2"):
        backend.focal_per_sample(one_channel, one_channel)


# ---------------------------------------------------------------------
# void: no value, no gradient
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LOSSES))
def test_void_logits_change_neither_the_value_nor_the_gradient(name):
    logits, target = rng_batch(seed=4)
    void = target == VOID
    assert void.any() and not void.all()
    loss = LOSSES[name]()
    value, grad = value_and_grad(loss, target, logits)
    changed = logits.copy()
    rng = np.random.default_rng(5)
    changed[void] = rng.normal(0, 1e4, changed[void].shape).astype("float32")
    other_value, other_grad = value_and_grad(loss, target, changed)
    assert other_value == value
    assert np.array_equal(other_grad, grad)
    assert np.all(grad[void] == 0.0) and np.any(grad[~void] != 0.0)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_a_batch_with_no_valid_pixel_is_zero_finite_and_without_gradient(name):
    logits, _ = rng_batch(seed=6)
    target = np.full(logits.shape[:3], VOID, "int32")
    value, grad = value_and_grad(LOSSES[name](), target, logits)
    assert value == 0.0
    assert np.all(grad == 0.0)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_one_void_image_in_a_batch_adds_nothing_but_the_divisor(name):
    logits, target = rng_batch(batch=3, void_share=0.0, seed=7)
    with_void = target.copy()
    with_void[2] = VOID
    full = LOSSES[name]()
    rows = full.call(keras.ops.convert_to_tensor(with_void), keras.ops.convert_to_tensor(logits))
    assert float(rows[2]) == 0.0
    assert float(full(with_void, logits)) == pytest.approx(float(np.mean(keras.ops.convert_to_numpy(rows))), abs=1e-7)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_the_value_is_one_number_per_image(name):
    logits, target = rng_batch(batch=4, seed=8)
    rows = keras.ops.convert_to_numpy(
        LOSSES[name]().call(keras.ops.convert_to_tensor(target), keras.ops.convert_to_tensor(logits))
    )
    assert rows.shape == (4,)
    assert np.allclose(rows, expected(name, logits, target), atol=1e-5)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_the_weight_of_an_image_selects_it(name):
    # loss(w=[1,..,1,0]) * B/(B-1) == loss(batch[:-1]); a batch-level term would break it.
    logits, target = rng_batch(batch=4, seed=9)
    loss = LOSSES[name]()
    weighted = float(loss(target, logits, sample_weight=np.array([1, 1, 1, 0], "float32")))
    assert weighted * 4 / 3 == pytest.approx(float(loss(target[:3], logits[:3])), rel=1e-6)


# ---------------------------------------------------------------------
# the shapes and the dtypes a run hands over
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LOSSES))
def test_targets_with_a_trailing_axis_or_as_floats_or_uint8_score_the_same(name):
    logits, target = rng_batch(seed=10)
    loss = LOSSES[name]()
    base = float(loss(target, logits))
    assert float(loss(target[..., None], logits)) == base
    assert float(loss(target.astype("float32"), logits)) == base
    uint8 = np.where(target == VOID, 255, target).astype("uint8")
    assert float(loss(uint8, logits)) == base


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_without_an_ignore_index_every_pixel_counts(name):
    logits, target = rng_batch(void_share=0.0, seed=11)
    got = float(LOSSES[name](ignore=None)(target, logits))
    assert got == pytest.approx(float(np.mean(expected(name, logits, np.where(target == VOID, 0, target), ignore=-1))), abs=1e-6)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_saturated_logits_give_a_finite_gradient(name):
    logits, target = rng_batch(seed=12)
    logits = logits * 1e4
    value, grad = value_and_grad(LOSSES[name](), target, logits)
    assert np.isfinite(value) and np.all(np.isfinite(grad))


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0, 5.0])
def test_the_focal_gradient_is_finite_at_a_saturated_pixel_for_any_gamma(gamma):
    logits = np.zeros((1, 2, 2, 3), "float32")
    logits[..., 0] = 200.0
    target = np.zeros((1, 2, 2), "int32")
    value, grad = value_and_grad(SparseFocalLoss(VOID, gamma=gamma), target, logits)
    assert np.isfinite(value) and np.all(np.isfinite(grad))


def test_mixed_float16_logits_are_scored_in_float32_and_stay_finite():
    logits, target = rng_batch(seed=13)
    for name in LOSSES:
        value16 = float(LOSSES[name]()(target, logits.astype("float16")))
        assert np.isfinite(value16)
        assert value16 == pytest.approx(float(LOSSES[name]()(target, logits.astype("float16").astype("float32"))), abs=1e-6)
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_float16")
    try:
        model = keras.Sequential([keras.layers.Input((6, 6, 3)), keras.layers.Conv2D(3, 3, padding="same")])
        assert model.compute_dtype == "float16"
        for name in LOSSES:
            model.compile(optimizer=keras.optimizers.SGD(0.01), loss=LOSSES[name]())
            images = np.random.default_rng(0).normal(size=(4, 6, 6, 3)).astype("float32")
            history = model.fit(images, target, batch_size=2, epochs=1, verbose=0)
            assert np.isfinite(history.history["loss"][0])
    finally:
        keras.config.set_dtype_policy(previous)


# ---------------------------------------------------------------------
# training, serialization, validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LOSSES))
def test_a_fit_lowers_the_loss_on_a_fixture(name):
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    images = rng.normal(size=(16, 8, 8, 3)).astype("float32")
    target = (images[..., 0] > 0).astype("int32") + (images[..., 1] > 0.5).astype("int32")
    target[:, :2, :] = VOID
    model = keras.Sequential([keras.layers.Input((8, 8, 3)), keras.layers.Conv2D(3, 3, padding="same")])
    model.compile(optimizer=keras.optimizers.Adam(0.05), loss=LOSSES[name]())
    before = model.evaluate(images, target, batch_size=8, verbose=0)
    model.fit(images, target, batch_size=8, epochs=30, verbose=0)
    after = model.evaluate(images, target, batch_size=8, verbose=0)
    assert after < 0.7 * before


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_the_config_round_trips(name):
    loss = LOSSES[name]()
    clone = type(loss).from_config(loss.get_config())
    assert clone.get_config() == loss.get_config()
    assert float(clone(TARGET, LOGITS)) == float(loss(TARGET, LOGITS))
    json.dumps(loss.get_config())


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_a_compiled_model_reloads_where_train_cannot_be_imported(name, tmp_path):
    model = keras.Sequential([keras.layers.Input((6, 6, 3)), keras.layers.Conv2D(3, 3, padding="same")])
    model.compile(optimizer="sgd", loss=LOSSES[name]())
    path = tmp_path / "model.keras"
    model.save(path)
    child = textwrap.dedent(
        """
        import json, sys
        sys.modules["train"] = None
        import keras
        import dl_techniques
        loaded = keras.models.load_model(sys.argv[1])
        print(json.dumps({"loss": type(loaded.loss).__name__, "ignore": loaded.loss.ignore_index,
                          "train_blocked": sys.modules["train"] is None}))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", child, str(path)], capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen == {"loss": type(LOSSES[name]()).__name__, "ignore": VOID, "train_blocked": True}


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: SparseDiceLoss(smooth=0.0), "smooth"),
        (lambda: SparseCeDiceLoss(smooth=-1.0), "smooth"),
        (lambda: SparseFocalLoss(gamma=-0.5), "gamma"),
        (lambda: SparseDiceLoss(ignore_index="255"), "ignore_index"),
        (lambda: SparseFocalLoss(ignore_index=True), "ignore_index"),
    ],
)
def test_a_bad_construction_is_refused_by_name(build, message):
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_a_single_channel_output_is_refused(name):
    with pytest.raises(ValueError, match="at least 2"):
        LOSSES[name]()(np.zeros((1, 2, 2), "int32"), np.zeros((1, 2, 2, 1), "float32"))


def test_the_per_image_mean_differs_from_the_stock_pooled_mean_exactly_when_the_void_share_does():
    # Documented in the module contract: stock CE divides by the batch's valid-pixel count.
    logits, target = rng_batch(batch=2, void_share=0.0, seed=14)
    stock = keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=VOID)
    ours = SparseFocalLoss(VOID, gamma=0.0)
    assert float(ours(target, logits)) == pytest.approx(float(stock(target, logits)), abs=1e-6)
    target[1, :, :] = VOID
    target[1, 0, 0] = 0
    assert abs(float(ours(target, logits)) - float(stock(target, logits))) > 1e-3
