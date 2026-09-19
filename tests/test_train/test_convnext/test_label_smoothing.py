"""``--label-smoothing`` of the ConvNeXt trainers (iteration 4, decision D-040).

An opt-in float in ``[0, 1)``, default 0.0. At 0 the loss is the stock
``SparseCategoricalCrossentropy(from_logits=True)`` object, so every earlier number is
unchanged; above 0 it is :class:`common.SmoothedSparseCategoricalCrossentropy`, which must
EQUAL ``keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=a)`` on
one-hot labels (Keras convention: the smoothing mass ``a / C`` goes to every class, the
true one included).

Each guard was proven RED by injecting the defect it names
(``findings/red-proofs-iter4.md``).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.convnext.common as common  # noqa: E402

from .test_train_convnext import _config, _fake_loader, _strict  # noqa: E402

SMOOTHED = common.SmoothedSparseCategoricalCrossentropy


def _batch(num_classes: int, n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = (rng.normal(size=(n, num_classes)) * 3.0).astype("float32")
    labels = rng.integers(0, num_classes, size=(n,))
    return logits, labels


def _manual(logits: np.ndarray, labels: np.ndarray, a: float) -> float:
    """float64 numpy: mean over the batch of -sum_c target_c * log_softmax_c."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    log_softmax = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    num_classes = logits.shape[-1]
    target = np.eye(num_classes)[labels] * (1.0 - a) + a / num_classes
    return float(np.mean(-(target * log_softmax).sum(axis=-1)))


# ---------------------------------------------------------------------
# The loss value
# ---------------------------------------------------------------------


@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("a", [0.05, 0.1, 0.5, 0.9])
def test_the_smoothed_loss_equals_the_manual_computation_and_keras_categorical(num_classes, a) -> None:
    logits, labels = _batch(num_classes)
    ours = float(SMOOTHED(label_smoothing=a)(labels, logits))
    assert ours == pytest.approx(_manual(logits, labels, a), rel=1e-5)
    keras_reference = float(keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=a)(np.eye(num_classes)[labels].astype("float32"), logits))
    assert ours == pytest.approx(keras_reference, rel=1e-5)
    # The mean-over-classes form the plan states: (1 - a) * CE + a * mean_c(-log_softmax).
    z = logits.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    log_softmax = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    ce = -log_softmax[np.arange(len(labels)), labels].mean()
    assert ours == pytest.approx((1 - a) * ce + a * (-log_softmax).mean(), rel=1e-5)


def test_a_column_of_labels_gives_the_same_loss_as_a_flat_vector() -> None:
    """The trainer's labels are ``(N,)`` after the pipeline but Keras callers pass ``(N, 1)``."""
    logits, labels = _batch(10)
    flat = float(SMOOTHED(0.2)(labels, logits))
    column = float(SMOOTHED(0.2)(labels[:, None], logits))
    assert column == pytest.approx(flat, rel=1e-6)
    assert flat == pytest.approx(_manual(logits, labels, 0.2), rel=1e-5)


def test_smoothing_moves_the_loss_and_its_gradient_the_way_the_keras_loss_does() -> None:
    """The gradient w.r.t. the logits (what training sees) equals the Keras one."""
    import tensorflow as tf

    logits, labels = _batch(10)
    a = 0.3
    x = tf.Variable(logits)
    with tf.GradientTape() as tape:
        loss = SMOOTHED(a)(labels, x)
    ours = tape.gradient(loss, x).numpy()
    y = tf.Variable(logits)
    with tf.GradientTape() as tape:
        reference = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=a)(
            np.eye(10)[labels].astype("float32"), y)
    np.testing.assert_allclose(ours, tape.gradient(reference, y).numpy(), atol=1e-6)


@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("a", [0.0, 0.1, 0.5])
def test_uniform_logits_give_ln_c_for_any_smoothing(num_classes, a) -> None:
    """Why the initial-loss guard (ratio to ln C) stays valid under smoothing."""
    logits = np.zeros((16, num_classes), dtype="float32")
    labels = np.arange(16) % num_classes
    value = float(common.build_loss(a)(labels, logits))
    assert value == pytest.approx(np.log(num_classes), rel=1e-5)


# ---------------------------------------------------------------------
# Which object is built
# ---------------------------------------------------------------------


def test_zero_smoothing_is_the_stock_loss_object_and_nothing_wraps_it() -> None:
    loss = common.build_loss(0.0)
    assert type(loss) is keras.losses.SparseCategoricalCrossentropy
    assert loss.get_config()["from_logits"] is True
    assert not isinstance(loss, SMOOTHED)


def test_positive_smoothing_builds_the_registered_smoothed_loss() -> None:
    loss = common.build_loss(0.1)
    assert type(loss) is SMOOTHED and loss.label_smoothing == 0.1
    assert loss.get_config()["label_smoothing"] == 0.1


def test_the_loss_config_round_trips_the_smoothing_value() -> None:
    rebuilt = SMOOTHED.from_config(SMOOTHED(label_smoothing=0.3).get_config())
    assert rebuilt.label_smoothing == 0.3
    assert keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(SMOOTHED(0.3))).label_smoothing == 0.3


# ---------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
def test_the_config_refuses_a_smoothing_outside_zero_to_one(bad) -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        common.TrainingConfig(label_smoothing=bad)
    with pytest.raises(ValueError, match="label_smoothing"):
        SMOOTHED(label_smoothing=bad)


@pytest.mark.parametrize("ok", [0.0, 0.1, 0.999])
def test_the_config_accepts_a_smoothing_inside_zero_to_one(ok) -> None:
    assert common.TrainingConfig(label_smoothing=ok).label_smoothing == ok


def test_the_default_is_off_and_the_flag_default_is_read_off_the_config() -> None:
    assert common.TrainingConfig().label_smoothing == 0.0
    for family in ("v1", "v2"):
        args = common.parse_arguments([], family)
        assert args.label_smoothing == common.TrainingConfig(model_family=family).label_smoothing
        args = common.parse_arguments(["--label-smoothing", "0.15"], family)
        assert common.config_from_args(args, family).label_smoothing == 0.15


# ---------------------------------------------------------------------
# Save and load of a compiled model with the custom loss
# ---------------------------------------------------------------------


def test_a_compiled_model_with_the_smoothed_loss_saves_and_reloads(tmp_path) -> None:
    """``best_model.keras`` is reloaded with ``load_model``'s compile default in
    ``run_summary.load_best_metrics`` and ``validate_model_loading``, so the registered loss
    must deserialize with the value it was saved with (0.3, not the class default 0.1)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 6)).astype("float32")
    y = rng.integers(0, 4, size=(64,))
    model = keras.Sequential([keras.layers.Input((6,)), keras.layers.Dense(4)])
    model.compile(optimizer="adam", loss=common.build_loss(0.3),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    model.fit(x, y, epochs=1, batch_size=16, verbose=0)
    path = tmp_path / "compiled.keras"
    model.save(path)

    reloaded = keras.models.load_model(path)

    assert type(reloaded.loss) is SMOOTHED and reloaded.loss.label_smoothing == 0.3
    before = model.evaluate(x, y, verbose=0, return_dict=True)
    after = reloaded.evaluate(x, y, verbose=0, return_dict=True)
    assert after["loss"] == pytest.approx(before["loss"], rel=1e-5)
    assert after["loss"] == pytest.approx(_manual(model.predict(x, verbose=0), y, 0.3), rel=1e-4)


# ---------------------------------------------------------------------
# Through the real training path
# ---------------------------------------------------------------------


def _train(monkeypatch, tmp_path, name: str, **overrides):
    monkeypatch.setattr(common, "load_dataset", _fake_loader(10))
    monkeypatch.setattr(common, "run_model_analysis", lambda *a, **k: None)
    config = _config(tmp_path, name, epochs=1, max_samples=64, model_analysis=False, **overrides)
    return common.train(config), tmp_path / name


def test_a_run_with_smoothing_records_it_notes_the_loss_scale_and_reloads(monkeypatch, tmp_path) -> None:
    summary, run_dir = _train(monkeypatch, tmp_path, "smooth", label_smoothing=0.5)

    assert summary["label_smoothing"] == 0.5
    saved = _strict((run_dir / "results_summary.json").read_text())
    assert saved["label_smoothing"] == 0.5
    assert json.loads((run_dir / "config.json").read_text())["label_smoothing"] == 0.5
    notes = [n for n in summary["notes"] if n.startswith("label_smoothing=0.5")]
    assert len(notes) == 1, summary["notes"]
    assert "NOT comparable with unsmoothed runs" in notes[0]
    assert "accuracy, top-5 accuracy and ECE keep their definitions" in notes[0]
    # The compiled checkpoint reloads through the trainer's own checks.
    assert summary["best_checkpoint_load_error"] is None
    assert summary["model_loading_validated"] is True
    best = keras.models.load_model(run_dir / "best_model.keras")
    assert type(best.loss) is SMOOTHED and best.loss.label_smoothing == 0.5

    # The reported test loss is the SMOOTHED cross-entropy of the best model's own logits:
    # the loss reaches compile, evaluate and the summary, not just the config. ``evaluate``
    # (XLA) and ``predict`` differ by about 0.4% on this untrained-scale model (the plain run
    # below shows the same gap), so the value is checked at 2% AND must sit closer to the
    # smoothed manual value than to the unsmoothed one. The run uses a = 0.5 because at 0.1 the
    # two manual values differ by only 0.03, about twice that noise.
    data = common.prepare_data(_config(tmp_path, "unused", epochs=1, max_samples=64))
    logits = best.predict(data.x_test, batch_size=common.EVAL_BATCH_SIZE, verbose=0)
    labels = np.asarray(data.y_test).reshape(-1)
    reported = summary["test_metrics_best"]["loss"]
    smoothed, plain = _manual(logits, labels, 0.5), _manual(logits, labels, 0.0)
    assert reported == pytest.approx(smoothed, rel=2e-2)
    assert abs(reported - smoothed) < abs(reported - plain), (reported, smoothed, plain)
    # The initial-loss guard: a near-uniform model reads a ratio near 1 either way.
    assert 0.5 < summary["initial_loss_ratio"] < 3.0


def test_a_run_without_smoothing_uses_the_stock_loss_and_has_no_note(monkeypatch, tmp_path) -> None:
    summary, run_dir = _train(monkeypatch, tmp_path, "plain")

    assert summary["label_smoothing"] == 0.0
    assert not any("label_smoothing" in n for n in summary["notes"]), summary["notes"]
    best = keras.models.load_model(run_dir / "best_model.keras")
    assert type(best.loss) is keras.losses.SparseCategoricalCrossentropy
    data = common.prepare_data(_config(tmp_path, "unused", epochs=1, max_samples=64))
    logits = best.predict(data.x_test, batch_size=common.EVAL_BATCH_SIZE, verbose=0)
    labels = np.asarray(data.y_test).reshape(-1)
    reported = summary["test_metrics_best"]["loss"]
    assert reported == pytest.approx(_manual(logits, labels, 0.0), rel=2e-2)
    assert abs(reported - _manual(logits, labels, 0.0)) < abs(reported - _manual(logits, labels, 0.5))
