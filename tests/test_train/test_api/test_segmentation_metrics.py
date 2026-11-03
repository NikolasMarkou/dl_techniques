"""The loss and metrics a segmentation run compiles: stock CE and mIoU with a void label, one custom accuracy.

The fixture is two 2x2 images of two classes and two void pixels (label 255), with the probabilities
written down (``test_void_aware_accuracy.PROBS``), so the three numbers are hand counts:

* cross-entropy over the 6 valid pixels: ``-(ln .8 + ln .6 + ln .3 + ln .9 + ln .4 + ln .9) / 6``
  = 0.510826 (an independent loop below shares no library code);
* accuracy 4/6 (the stock accuracy would say 4/8);
* mean IoU 0.5: the confusion matrix is ``[[2, 1], [1, 2]]``, both IoUs are 2/4.

What the model says at a void pixel, however extreme, changes none of the three nor the gradient; a
fully void batch gives finite numbers; the compiled model reloads in a process where the ``train``
package cannot be imported; and the same compile runs under XLA (CPU here; the GPU measurement is in
the plan's ``scratch_step27/logs/jit_gpu.txt``).

``SegMetricArgs`` is the group that selects the best epoch: ``val_miou`` is a maximum, the bar is
null (the task's majority-class baseline) unless given, and a value outside [0, 1] is refused.
"""

import json
import math
import os
import subprocess
import sys
import textwrap
import zipfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.metrics.void_aware_accuracy import VoidAwareAccuracy
from train.api import segmentation_args as sa
from train.api.args import AutoContext
from train.api.config import assert_resolved, validate
from train.api.errors import ConfigValidationError, UnresolvedAutoError
from train.common import segmentation_run as run

from tests.test_metrics.test_void_aware_accuracy import LABELS, PROBS, VOID, VOID_AT

LOGITS = np.log(PROBS)
CE = -(math.log(0.8) + math.log(0.6) + math.log(0.3) + math.log(0.9) + math.log(0.4) + math.log(0.9)) / 6


def hand_values(logits, labels):
    """Cross-entropy, accuracy and mIoU over the valid pixels, in plain Python (no library code)."""
    pixels = [(row, int(label)) for image_logits, image_labels in zip(logits, labels)
              for r_logits, r_labels in zip(image_logits, image_labels)
              for row, label in zip(r_logits, r_labels) if label != VOID]
    total, right, confusion = 0.0, 0, [[0, 0], [0, 0]]
    for row, label in pixels:
        shifted = [float(v) - max(float(w) for w in row) for v in row]
        total -= shifted[label] - math.log(sum(math.exp(v) for v in shifted))
        predicted = max(range(2), key=lambda k: float(row[k]))
        right += predicted == label
        confusion[label][predicted] += 1
    ious = [confusion[k][k] / (sum(confusion[k]) + confusion[0][k] + confusion[1][k] - confusion[k][k]) for k in (0, 1)]
    return total / len(pixels), right / len(pixels), sum(ious) / 2


def identity_model(ignore_index=VOID, jit_compile=False, classes=2):
    model = keras.Sequential([keras.layers.Input((2, 2, classes)), keras.layers.Identity()])
    model.compile(loss=run.segmentation_loss(ignore_index), metrics=run.segmentation_metrics(classes, ignore_index),
                  jit_compile=jit_compile)
    return model


def evaluate(model, logits=LOGITS, labels=LABELS, batch_size=2):
    return model.evaluate(logits, labels, batch_size=batch_size, verbose=0, return_dict=True)


# ---------------------------------------------------------------------
# What is compiled
# ---------------------------------------------------------------------

def test_the_builders_return_stock_keras_and_one_custom_accuracy_only_with_a_void_label():
    loss = run.segmentation_loss(VOID)
    assert isinstance(loss, keras.losses.SparseCategoricalCrossentropy)
    assert loss.get_config()["from_logits"] is True and loss.get_config()["ignore_class"] == VOID
    accuracy, miou = run.segmentation_metrics(3, VOID)
    assert type(accuracy) is VoidAwareAccuracy and accuracy.name == "accuracy" and accuracy.ignore_index == VOID
    assert type(miou) is keras.metrics.MeanIoU and miou.name == "miou"
    assert miou.get_config()["ignore_class"] == VOID and miou.get_config()["sparse_y_true"] is True
    assert miou.get_config()["sparse_y_pred"] is False and miou.get_config()["num_classes"] == 3
    plain_loss = run.segmentation_loss(None)
    plain_accuracy, plain_miou = run.segmentation_metrics(3, None)
    assert plain_loss.get_config()["ignore_class"] is None
    assert type(plain_accuracy) is keras.metrics.SparseCategoricalAccuracy and plain_accuracy.name == "accuracy"
    assert plain_miou.get_config()["ignore_class"] is None


def test_the_names_are_known_without_building_a_metric_and_equal_the_built_ones():
    assert run.SEGMENTATION_METRIC_NAMES == ("accuracy", "miou")
    for ignore in (None, VOID):
        assert tuple(metric.name for metric in run.segmentation_metrics(2, ignore)) == run.SEGMENTATION_METRIC_NAMES


# ---------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------

def test_the_fixture_values_are_the_hand_counts():
    assert hand_values(LOGITS, LABELS) == pytest.approx((CE, 4 / 6, 0.5), abs=1e-9)
    assert CE == pytest.approx(0.510826, abs=1e-6)


def test_the_compiled_loss_and_metrics_equal_the_hand_counts_through_evaluate():
    scores = evaluate(identity_model())
    assert scores["loss"] == pytest.approx(CE, abs=1e-5)
    assert scores["accuracy"] == pytest.approx(4 / 6, abs=1e-6)
    assert scores["miou"] == pytest.approx(0.5, abs=1e-6)


def test_the_loss_object_alone_equals_the_hand_count():
    assert float(run.segmentation_loss(VOID)(LABELS, LOGITS)) == pytest.approx(CE, abs=1e-5)


def test_without_a_void_label_the_stock_pieces_are_the_same_numbers():
    labels = np.where(VOID_AT, 1, LABELS)
    scores = evaluate(identity_model(None), labels=labels)
    expected = hand_values(LOGITS, labels)  # the void pixels became class 1: every pixel is valid
    assert scores["loss"] == pytest.approx(expected[0], abs=1e-5) and scores["accuracy"] == pytest.approx(expected[1], abs=1e-6)


def test_the_stock_accuracy_with_a_void_label_would_be_wrong_and_ours_is_not():
    stock = keras.metrics.SparseCategoricalAccuracy()
    stock.update_state(LABELS, LOGITS)
    assert float(stock.result()) == pytest.approx(0.5) != pytest.approx(4 / 6)
    assert evaluate(identity_model())["accuracy"] == pytest.approx(4 / 6, abs=1e-6)


def test_a_mean_iou_without_ignore_class_cannot_take_a_void_label():
    miou = keras.metrics.MeanIoU(num_classes=2, sparse_y_true=True, sparse_y_pred=False)
    with pytest.raises(tf.errors.InvalidArgumentError, match="does not index"):
        miou.update_state(LABELS, LOGITS)


# ---------------------------------------------------------------------
# Void is out of the value and the gradient
# ---------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 1e3, 1e9])
def test_what_the_model_says_at_a_void_pixel_changes_none_of_the_three(scale):
    reference = evaluate(identity_model())
    rng = np.random.default_rng(1)
    for _ in range(10):
        logits = LOGITS.copy()
        logits[VOID_AT] = rng.normal(size=(int(VOID_AT.sum()), 2)).astype("float32") * scale
        scores = evaluate(identity_model(), logits=logits)
        for key in ("loss", "accuracy", "miou"):
            assert scores[key] == pytest.approx(reference[key], abs=1e-6), key


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_even_an_infinite_or_nan_logit_at_a_void_pixel_changes_no_value(value):
    """Measured (``scratch_step27/logs/void_nan_inf.txt``): the VALUES hold. The loss gradient at that
    void pixel is then NaN (``0 * inf``), which is why only a finite logit is promised a zero gradient
    (next test): a model that emits inf or NaN anywhere has diverged whatever the mask says."""
    reference = evaluate(identity_model())
    logits = LOGITS.copy()
    logits[VOID_AT] = value
    scores = evaluate(identity_model(), logits=logits)
    for key in ("loss", "accuracy", "miou"):
        assert scores[key] == pytest.approx(reference[key], abs=1e-6), key


def test_the_gradient_at_a_void_pixel_is_zero_and_at_a_valid_pixel_is_not():
    for scale in (1.0, 1e9):  # a finite logit of any size
        logits_np = LOGITS.copy()
        logits_np[VOID_AT] *= scale
        logits = tf.Variable(logits_np)
        with tf.GradientTape() as tape:
            loss = run.segmentation_loss(VOID)(LABELS, logits)
        gradient = tape.gradient(loss, logits).numpy()
        assert np.all(gradient[VOID_AT] == 0.0)
        assert np.all(np.abs(gradient[~VOID_AT]).sum(-1) > 0.0)


def test_a_fully_void_batch_is_finite_and_does_not_poison_the_next_batch():
    everything = np.full(LABELS.shape, VOID, "int32")
    assert float(run.segmentation_loss(VOID)(everything, LOGITS)) == 0.0
    accuracy, miou = run.segmentation_metrics(2, VOID)
    for metric in (accuracy, miou):
        metric.update_state(everything, LOGITS)
        assert float(metric.result()) == 0.0, metric.name
        metric.update_state(LABELS, LOGITS)
    assert float(accuracy.result()) == pytest.approx(4 / 6)
    assert float(miou.result()) == pytest.approx(0.5)
    scores = evaluate(identity_model(), labels=everything)
    assert all(np.isfinite(value) for value in scores.values())


def test_the_loss_of_a_run_is_a_mean_of_batch_means_while_accuracy_and_miou_pool_pixels():
    """Measured: Keras averages the per-batch valid-pixel loss by IMAGES, so a batch with few valid pixels
    weighs as much as one with many; the metrics are pooled over pixels. The reported ``val_loss`` is
    therefore not the pooled cross-entropy once the void share differs between batches."""
    labels = LABELS.copy()
    labels[1] = [[VOID, VOID], [1, VOID]]  # image 1 keeps ONE valid pixel (label 1, predicted 0: wrong)
    whole = evaluate(identity_model(), labels=labels, batch_size=2)
    singly = evaluate(identity_model(), labels=labels, batch_size=1)
    image_losses = [hand_values(LOGITS[i:i + 1], labels[i:i + 1])[0] for i in (0, 1)]
    pooled = hand_values(LOGITS, labels)[0]
    assert whole["loss"] == pytest.approx(pooled, abs=1e-5)
    assert singly["loss"] == pytest.approx(sum(image_losses) / 2, abs=1e-5)
    assert abs(singly["loss"] - pooled) > 0.05, "the two readings of 'the loss' differ on this fixture"
    for key in ("accuracy", "miou"):
        assert singly[key] == pytest.approx(whole[key], abs=1e-6), key


# ---------------------------------------------------------------------
# fit: XLA, mixed precision, reload
# ---------------------------------------------------------------------

def tiny_segmenter(classes=3, policy=None):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8, 8, 3))
    outputs = keras.layers.Conv2D(classes, 3, padding="same", dtype=policy)(keras.layers.Conv2D(4, 3, padding="same", activation="relu", dtype=policy)(inputs))
    return keras.Model(inputs, outputs)


def void_data(n=16, classes=3):
    rng = np.random.default_rng(0)
    x = rng.random((n, 8, 8, 3), dtype=np.float32)
    y = rng.integers(0, classes, size=(n, 8, 8)).astype("int32")
    y[np.kron(rng.random((n, 2, 2)) < 0.3, np.ones((4, 4), bool))] = VOID
    return x, y


def compiled(model, jit_compile=False):
    model.compile(optimizer=keras.optimizers.AdamW(1e-2), loss=run.segmentation_loss(VOID),
                  metrics=run.segmentation_metrics(3, VOID), jit_compile=jit_compile)
    return model


def test_fit_under_xla_matches_fit_without_it_and_the_numbers_are_finite():
    x, y = void_data()
    histories = []
    for jit in (False, True):
        model = compiled(tiny_segmenter(), jit_compile=jit)
        histories.append(model.fit(x, y, batch_size=8, epochs=2, shuffle=False, verbose=0).history)
    for key in ("loss", "accuracy", "miou"):
        assert np.all(np.isfinite(histories[1][key])), key
        assert histories[1][key] == pytest.approx(histories[0][key], abs=2e-3), key
    assert histories[1]["loss"][1] < histories[1]["loss"][0], "the loss goes down"


def test_fit_under_mixed_float16_is_finite_and_the_logits_are_half_precision():
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        x, y = void_data()
        model = tiny_segmenter()
        assert model.output.dtype == "float16"
        compiled(model)
        history = model.fit(x, y, batch_size=8, epochs=2, shuffle=False, verbose=0).history
    finally:
        keras.mixed_precision.set_global_policy("float32")
    for key in ("loss", "accuracy", "miou"):
        assert np.all(np.isfinite(history[key])), key


CHILD = textwrap.dedent(
    """
    import json, sys
    sys.modules["train"] = None
    import numpy as np
    import keras
    import dl_techniques  # the library must be importable; the trainer package is not

    model = keras.models.load_model(sys.argv[1])
    data = np.load(sys.argv[2])
    scores = model.evaluate(data["x"], data["y"], batch_size=8, verbose=0, return_dict=True)
    print(json.dumps({"scores": scores, "train_blocked": sys.modules["train"] is None}))
    """
)


def test_the_compiled_model_reloads_without_the_train_package_and_scores_the_same(tmp_path):
    x, y = void_data()
    model = compiled(tiny_segmenter())
    model.fit(x, y, batch_size=8, epochs=1, shuffle=False, verbose=0)
    expected = model.evaluate(x, y, batch_size=8, verbose=0, return_dict=True)
    archive = tmp_path / "m.keras"
    model.save(archive)
    np.savez(tmp_path / "d.npz", x=x, y=y)
    with zipfile.ZipFile(archive) as zf:
        compile_config = json.dumps(json.loads(zf.read("config.json"))["compile_config"])
    assert "dl_techniques.metrics.void_aware_accuracy" in compile_config
    assert '"train.' not in compile_config and "train.common" not in compile_config
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "src")
    env = dict(os.environ, PYTHONPATH=src, CUDA_VISIBLE_DEVICES="-1")
    done = subprocess.run([sys.executable, "-c", CHILD, str(archive), str(tmp_path / "d.npz")],
                          capture_output=True, text=True, env=env, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen["train_blocked"] is True
    for key in ("loss", "accuracy", "miou"):
        assert seen["scores"][key] == pytest.approx(expected[key], abs=1e-5), key


# ---------------------------------------------------------------------
# The metric group
# ---------------------------------------------------------------------

def context(recipe=None):
    return AutoContext(family="convunext", variant="tiny", dataset="oxford_pet", num_classes=3,
                       image_size=0, defaults={"metric_args": recipe} if recipe is not None else {})


def test_val_miou_is_a_maximum_and_val_loss_a_minimum_and_the_bar_is_null_by_default():
    miou = sa.SegMetricArgs().resolve_auto(context({"watch_metric": "val_miou"}))
    assert (miou.watch_metric, miou.watch_mode, miou.success_threshold) == ("val_miou", "max", None)
    loss = sa.SegMetricArgs(watch_metric="val_loss").resolve_auto(context({"watch_metric": "val_miou"}))
    assert (loss.watch_metric, loss.watch_mode) == ("val_loss", "min")
    accuracy = sa.SegMetricArgs(watch_metric="val_accuracy").resolve_auto(context())
    assert accuracy.watch_mode == "max"


def test_an_explicit_mode_and_threshold_are_untouched():
    resolved = sa.SegMetricArgs(watch_metric="val_miou", watch_mode="min", success_threshold=0.4).resolve_auto(context({"watch_metric": "val_loss"}))
    assert (resolved.watch_metric, resolved.watch_mode, resolved.success_threshold) == ("val_miou", "min", 0.4)


def test_a_metric_the_recipe_lacks_stays_auto_for_assert_resolved():
    resolved = sa.SegMetricArgs().resolve_auto(context())
    assert resolved.watch_metric == "auto" and resolved.watch_mode == "auto"
    with pytest.raises(UnresolvedAutoError, match="watch_metric"):
        assert_resolved(resolved)


@pytest.mark.parametrize("value", [-0.1, 1.5, 2.0])
def test_a_threshold_outside_zero_one_is_refused(value):
    with pytest.raises(ConfigValidationError, match=r"success_threshold.*fraction in \[0, 1\]"):
        validate(sa.SegMetricArgs, {"success_threshold": value})


@pytest.mark.parametrize("value", [0, 0.25, 1])
def test_a_threshold_in_range_is_taken(value):
    assert validate(sa.SegMetricArgs, {"success_threshold": value}).success_threshold == float(value)


def test_the_group_has_no_top_k_and_a_bad_recipe_key_is_refused():
    assert list(sa.SegMetricArgs.__dataclass_fields__) == ["watch_metric", "watch_mode", "success_threshold"]
    with pytest.raises(ConfigValidationError, match="top_k"):
        validate(sa.SegMetricArgs, {"top_k": 5})
    with pytest.raises(ConfigValidationError, match="Family recipe for 'metric_args'"):
        sa.SegMetricArgs().resolve_auto(context({"top_k": 5}))
