"""Behavioural guards for ``src/train/convnext/common.py`` (the V1 / V2 trainer).

Each guard exists because the pre-normalization trainers got the thing wrong (or
because the new orchestrator could silently regress it), and each was proven RED by
injecting the defect (``findings/red-proofs-iter1-step4.md`` in the plan directory):

- the cosine schedule must span the WHOLE run: without ``steps_per_epoch`` it fell to
  its 1e-5 floor within ``epochs`` optimizer steps (``test_the_cosine_spans_the_whole_run``,
  and the CSV-level ``test_the_csv_lr_column_is_a_live_cosine``);
- the 100-class top-5 metric must be a metric OBJECT: the string ``"top_5_accuracy"``
  crashed ``compile`` (``test_a_100_class_run_reports_top_5_accuracy``);
- a reused experiment name is refused and the first run stays byte-identical;
- the figures are drawn from PROBABILITIES (softmax of the logits), never logits and
  never a softmax of probabilities;
- ``final_model.keras`` holds the LAST epoch's weights, not the restored best ones;
- the validation split is a seeded cut of the TRAIN set and never touches the test set;
- the analytic feature-map geometry equals what the real ``include_top=False`` model
  produces.

Everything is written under ``tmp_path`` (a conftest guard fails on any write to the
repo-root ``results/``). The datasets are synthetic and injected through
``common.load_dataset``; nothing here reads the Keras dataset cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.convnext.common as common  # noqa: E402

N_TRAIN, N_TEST, MAX_SAMPLES = 400, 400, 240
EPOCHS = 3
E2E_LEARNING_RATE = 5e-3
# Added to the LAST epoch's ``val_loss`` by a callback of the fixture, so that the last
# epoch is deterministically NOT the best one. On random labels the natural val-loss
# curve is flat noise around ln(10) and its argmin depends on the device; the
# best-versus-final guards need a run whose best and final weights really differ.
LAST_EPOCH_PENALTY = 100.0


def _strict(text: str) -> Any:
    def refuse(token: str):
        raise ValueError(f"non-strict JSON constant {token!r}")

    return json.loads(text, parse_constant=refuse)


def _fake_loader(num_classes: int, n_train: int = N_TRAIN, n_test: int = N_TEST):
    """A drop-in for ``train.common.load_dataset``: random 32x32x3 images and labels.

    Labels are ``(N, 1)`` uint8 like the real CIFAR loader; the images carry no class
    signal, so an epoch-1 model is at chance and later epochs only overfit.
    """

    def load(name, *args, **kwargs):
        rng = np.random.default_rng(3)

        def split(n):
            x = rng.random((n, 32, 32, 3), dtype=np.float32)
            y = rng.integers(0, num_classes, size=(n, 1)).astype(np.uint8)
            return x, y

        return split(n_train), split(n_test), (32, 32, 3), num_classes

    return load


def _identity_loader(n_train: int, n_test: int):
    """Every image is a constant equal to its identity, so a split can be audited.

    Train images are ``i / T`` for ``i in [0, n_train)``, test images continue the count
    (``(n_train + j) / T``), ``T = n_train + n_test``. Standardization is affine per
    channel, so :func:`_ids` recovers the identity from a standardized array.
    """
    total = n_train + n_test

    def load(name, *args, **kwargs):
        def split(start, n):
            ids = np.arange(start, start + n, dtype=np.float32)
            x = np.broadcast_to((ids / total)[:, None, None, None], (n, 32, 32, 3)).copy()
            return x, (np.arange(n) % 10).astype(np.uint8).reshape(-1, 1)

        return split(0, n_train), split(n_train, n_test), (32, 32, 3), 10

    return load, total


def _ids(data: "common.SplitData", x: np.ndarray, total: int) -> np.ndarray:
    raw = x[:, 0, 0, 0] * data.std[0] + data.mean[0]
    return np.rint(raw * total).astype(int)


def _config(tmp_path: Path, name: str, **overrides) -> "common.TrainingConfig":
    settings = dict(
        variant="cifar10", epochs=EPOCHS, batch_size=32, learning_rate=E2E_LEARNING_RATE,
        max_samples=MAX_SAMPLES, seed=5, output_dir=str(tmp_path), experiment_name=name,
    )
    settings.update(overrides)
    return common.TrainingConfig(**settings)


def _snapshot(directory: Path) -> Dict[str, str]:
    """``{relative path: sha256}`` of every file under ``directory``."""
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*")) if p.is_file()
    }


def _file_handlers():
    return [h for h in logging.getLogger("dl").handlers if isinstance(h, logging.FileHandler)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------
# One real tiny end-to-end run (CIFAR-10-shaped, 3 epochs), shared by many tests
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Run ``common.train`` once and keep everything the assertions need.

    Three hooks are installed around the run and none is part of the trainer: a callback
    appended to ``fit`` that copies the weights after every epoch (the independent record
    of "what the last epoch held"), a callback put first that makes the last epoch's
    ``val_loss`` the worst (:data:`LAST_EPOCH_PENALTY`), and a wrapper of
    ``plot_calibration`` that keeps the array the trainer hands it.
    """
    out = tmp_path_factory.mktemp("convnext_e2e")
    epoch_weights = []
    seen: Dict[str, Any] = {}

    class Recorder(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epoch_weights.append(self.model.get_weights())

    class LastEpochPenalty(keras.callbacks.Callback):
        """First in the list: EarlyStopping, ModelCheckpoint and History read ``logs`` later."""

        def on_epoch_end(self, epoch, logs=None):
            if epoch == EPOCHS - 1:
                logs["val_loss"] = float(logs["val_loss"]) + LAST_EPOCH_PENALTY

    real_fit = keras.Model.fit
    real_calibration = common.plot_calibration

    def fit(self, *args, **kwargs):
        kwargs["callbacks"] = [LastEpochPenalty(), *kwargs.get("callbacks", []), Recorder()]
        return real_fit(self, *args, **kwargs)

    def calibration(y_true, probs, out_path, *args, **kwargs):
        seen["calibration_probs"] = np.array(probs)
        return real_calibration(y_true, probs, out_path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(common, "load_dataset", _fake_loader(10))
        mp.setattr(keras.Model, "fit", fit)
        mp.setattr(common, "plot_calibration", calibration)
        config = _config(out, "e2e")
        summary = common.train(config)
        data = common.prepare_data(config)

    run_dir = out / "e2e"
    best = keras.models.load_model(run_dir / "best_model.keras")
    final = keras.models.load_model(run_dir / "final_model.keras")
    return SimpleNamespace(
        out=out, run_dir=run_dir, config=config, summary=summary, data=data,
        epoch_weights=epoch_weights, best=best, final=final, **seen,
    )


ARTIFACTS = (
    "config.json", "training_log.csv", "training_history.json", "best_model.keras",
    "final_model.keras", "results_summary.json", "run.log",
)
FIGURES = (
    "training_dashboard.png", "confusion_matrix.png", "per_class_metrics.png",
    "confidence_calibration.png", "misclassifications.png", "classification_report.json",
)


def test_the_run_writes_the_full_artifact_set(e2e) -> None:
    for name in ARTIFACTS:
        assert (e2e.run_dir / name).stat().st_size > 0, name
    for name in FIGURES:
        assert (e2e.run_dir / "visualizations" / name).stat().st_size > 0, name
    for name in FIGURES:
        if name.endswith(".png"):
            assert (e2e.run_dir / "visualizations" / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (e2e.run_dir / "model_analysis" / "analysis_results.json").is_file()
    # The trainer must not have touched anything outside its run directory.
    assert sorted(p.name for p in e2e.out.iterdir()) == ["e2e"]
    assert _file_handlers() == [], "run.log handler must be removed when train returns"


def test_the_summary_is_strict_json_with_every_promised_key(e2e) -> None:
    text = (e2e.run_dir / "results_summary.json").read_text()
    summary = _strict(text)
    assert summary["status"] == "ok"
    assert summary == _strict(json.dumps(e2e.summary, default=common.json_numpy_default))
    for key in (
        "initial_loss_ratio", "initial_loss_sanity_eval", "stage_feature_map_sizes",
        "test_metrics_best", "test_metrics_final", "epoch_times", "gpu_name",
        "tf_visible_devices", "cuda_visible_devices", "best_epoch", "final_is_best",
        "best_checkpoint_max_abs_diff", "ece", "analyzer", "steps_per_epoch", "params",
    ):
        assert key in summary, key
    assert summary["gpu_name"] is None or isinstance(summary["gpu_name"], str)
    assert summary["epochs_run"] == EPOCHS and len(summary["epoch_times"]) == EPOCHS
    assert all(t > 0.0 for t in summary["epoch_times"])
    # 240-sample cap, 10% validation: 216 train / 24 val, 240 test; ceil(216/32) = 7.
    assert (summary["n_train"], summary["n_val"], summary["n_test"]) == (216, 24, 240)
    assert summary["steps_per_epoch"] == 7
    assert summary["stage_feature_map_sizes"] == [[8, 8], [2, 2]]
    assert summary["initial_loss_ratio"] == pytest.approx(
        summary["initial_loss_sanity_eval"]["loss"] / np.log(10))
    assert summary["analyzer"]["status"] == "success", summary["analyzer"]
    assert summary["visualizations"]["failed"] == []
    assert summary["best_checkpoint_load_error"] is None
    assert summary["best_checkpoint_max_abs_diff"] <= common.WEIGHT_MISMATCH_TOLERANCE
    assert summary["model_loading_validated"] is True
    assert 0.0 <= summary["ece"] <= 1.0
    saved = _strict((e2e.run_dir / "config.json").read_text())
    assert saved["experiment_name"] == "e2e" and saved["drop_path_rate"] == 0.1
    log = (e2e.run_dir / "run.log").read_text()
    assert "Run directory:" in log and "Stage feature maps (H, W)" in log
    assert "Analyzer status (read back from disk): success" in log


# ---------------------------------------------------------------------
# Learning-rate schedule (D-013)
# ---------------------------------------------------------------------


def _schedule_values(config, steps_per_epoch):
    schedule = common.build_lr_schedule(config, steps_per_epoch)
    total = config.epochs * steps_per_epoch
    return steps_per_epoch, [float(schedule(step)) for step in range(total + 1)]


def test_the_cosine_spans_the_whole_run() -> None:
    """Without ``steps_per_epoch`` the cosine reaches its floor after ``epochs`` steps."""
    config = common.TrainingConfig(epochs=5, learning_rate=1e-3)
    steps, lrs = _schedule_values(config, steps_per_epoch=20)
    assert lrs[0] == pytest.approx(1e-3)
    epoch_2 = lrs[steps:2 * steps]
    assert min(epoch_2) > 5e-4, f"epoch 2 already collapsed: {min(epoch_2):.3g}"
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "cosine must never increase"
    assert lrs[-1] == pytest.approx(1e-3 * 0.01, rel=1e-3), "the floor is reached only at the end"
    assert lrs[len(lrs) // 2] == pytest.approx(0.5e-3, rel=0.05), "midpoint of the run is half the peak"


def test_warmup_ramps_up_then_the_cosine_never_increases() -> None:
    config = common.TrainingConfig(epochs=5, learning_rate=1e-3, warmup_epochs=1)
    steps, lrs = _schedule_values(config, steps_per_epoch=20)
    assert lrs[0] < 1e-5 and lrs[steps] == pytest.approx(1e-3, rel=1e-3)
    assert all(a <= b for a, b in zip(lrs[:steps], lrs[1:steps + 1])), "warmup must rise"
    after = lrs[steps:]
    assert all(a >= b for a, b in zip(after, after[1:])), "non-increasing after warmup"
    assert min(lrs[steps:2 * steps]) > 5e-4
    assert lrs[-1] == pytest.approx(1e-5, rel=0.2)


def test_the_constant_schedule_is_a_plain_float() -> None:
    config = common.TrainingConfig(lr_schedule="constant", learning_rate=2e-4)
    assert common.build_lr_schedule(config, 10) == 2e-4


def test_the_csv_lr_column_is_a_live_cosine(e2e) -> None:
    """The rate the optimizer really used, read from ``training_log.csv``.

    The column is read after each epoch's last step, i.e. it is the rate of the NEXT
    epoch's first step: at 2 of 3 epochs a live cosine is ~26% of the peak, a collapsed
    one is ~1% of it.
    """
    with open(e2e.run_dir / "training_log.csv") as f:
        rows = list(csv.DictReader(f))
    lrs = [float(r["lr"]) for r in rows]
    assert len(lrs) == EPOCHS
    assert lrs[1] > 0.2 * E2E_LEARNING_RATE and lrs[1] != pytest.approx(1e-5, abs=5e-5)
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert e2e.summary["lr_first_epoch"] == pytest.approx(lrs[0])
    assert e2e.summary["lr_last_epoch"] == pytest.approx(lrs[-1])


# ---------------------------------------------------------------------
# Probabilities into the figures, best versus final weights
# ---------------------------------------------------------------------


def test_the_calibration_figure_is_fed_softmax_probabilities(e2e) -> None:
    """Logits fed as probabilities, or probabilities softmaxed again, both mislead ECE."""
    probs = e2e.calibration_probs
    assert probs.shape == (240, 10)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-4)
    expected = _softmax(e2e.best.predict(e2e.data.x_test, batch_size=64, verbose=0))
    np.testing.assert_allclose(probs, expected, atol=1e-4)


def test_the_precondition_of_the_final_versus_best_guards_holds(e2e) -> None:
    """If the last epoch is the best one, the guards below cannot tell final from best."""
    assert e2e.summary["final_is_best"] is False, (
        f"best epoch {e2e.summary['best_epoch']} of {EPOCHS}: the penalty callback did not act"
    )
    assert e2e.summary["best_epoch"] < EPOCHS


def _max_abs_diff(a, b) -> float:
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def test_final_model_holds_the_last_epoch_weights_and_best_holds_the_best_epoch(e2e) -> None:
    """Keras 3.8 EarlyStopping restores the best weights at train end, so the last
    epoch's weights exist only if the trainer captured them."""
    last, best_epoch = e2e.epoch_weights[-1], e2e.summary["best_epoch"]
    assert len(e2e.epoch_weights) == EPOCHS
    assert _max_abs_diff(e2e.final.get_weights(), last) < 1e-6
    assert _max_abs_diff(e2e.best.get_weights(), e2e.epoch_weights[best_epoch - 1]) < 1e-6
    assert _max_abs_diff(e2e.final.get_weights(), e2e.best.get_weights()) > 1e-4


def test_test_metrics_final_is_the_last_epoch_evaluated_on_the_test_set(e2e) -> None:
    data = e2e.data
    final = e2e.final.evaluate(data.x_test, data.y_test, batch_size=64, verbose=0, return_dict=True)
    best = e2e.best.evaluate(data.x_test, data.y_test, batch_size=64, verbose=0, return_dict=True)
    summary = e2e.summary
    assert summary["test_metrics_final"]["loss"] == pytest.approx(final["loss"], rel=1e-3)
    assert summary["test_metrics_best"]["loss"] == pytest.approx(best["loss"], rel=1e-3)
    assert summary["test_metrics_final"]["loss"] != pytest.approx(
        summary["test_metrics_best"]["loss"], rel=1e-4), "final and best are different models here"


def test_last_epoch_weights_callback_keeps_a_copy_of_the_latest_epoch() -> None:
    """Direct unit test of the capture: the latest completed epoch, as a copy."""
    model = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2)])
    callback = common._LastEpochWeights()
    callback.set_model(model)
    assert callback.weights is None
    first = [np.full_like(w, 1.0) for w in model.get_weights()]
    second = [np.full_like(w, 2.0) for w in model.get_weights()]
    model.set_weights(first)
    callback.on_epoch_end(0)
    model.set_weights(second)
    callback.on_epoch_end(1)
    model.set_weights(first)  # the "restore best weights" step after training
    assert _max_abs_diff(callback.weights, second) == 0.0, "must hold the LAST epoch, not a later state"
    assert _max_abs_diff(callback.weights, first) == 1.0


# ---------------------------------------------------------------------
# Reused experiment names are refused (D-002)
# ---------------------------------------------------------------------


def test_a_reused_experiment_name_is_refused_and_the_first_run_is_byte_identical(
        monkeypatch, e2e) -> None:
    """A second run under the same name used to merge into the first directory."""
    monkeypatch.setattr(common, "load_dataset", _fake_loader(10))
    monkeypatch.setattr(common, "run_model_analysis", lambda *a, **k: None)
    before = _snapshot(e2e.run_dir)
    assert "results_summary.json" in before and "run.log" in before

    retry = _config(e2e.out, "e2e", epochs=1, max_samples=40, seed=6)
    with pytest.raises(FileExistsError) as raised:
        common.train(retry)

    text = str(raised.value)
    assert str(e2e.run_dir) in text and "--experiment-name" in text, text
    assert _snapshot(e2e.run_dir) == before, "the refusal wrote or deleted something"
    assert _file_handlers() == [], "no handler may be attached before the refusal"


# ---------------------------------------------------------------------
# Top-5 metric on a 100-class task
# ---------------------------------------------------------------------


def test_the_top_5_metric_exists_only_above_ten_classes_and_is_an_object() -> None:
    small = common.build_metrics(10)
    assert [m.name for m in small] == ["accuracy"]
    wide = common.build_metrics(100)
    assert [m.name for m in wide] == ["accuracy", "top_5_accuracy"]
    assert all(isinstance(m, keras.metrics.Metric) for m in wide), (
        "the string alias 'top_5_accuracy' is not resolvable by Keras compile"
    )


def test_a_100_class_run_reports_top_5_accuracy(monkeypatch, tmp_path) -> None:
    """The shipped V2 CIFAR-100 script crashed at ``compile`` on the string alias."""
    monkeypatch.setattr(common, "load_dataset", _fake_loader(100))
    monkeypatch.setattr(common, "run_model_analysis", lambda *a, **k: None)
    # A 100x100 confusion matrix costs ~30 s on CPU and is not what this guard is about.
    monkeypatch.setattr(
        common, "_write_visualizations",
        lambda *a, **k: {"files": [], "ece": None, "failed": []})
    config = _config(tmp_path, "wide", dataset="cifar100", epochs=1, max_samples=200,
                     learning_rate=1e-3)

    summary = common.train(config)

    history = _strict((tmp_path / "wide" / "training_history.json").read_text())
    assert "top_5_accuracy" in history and "val_top_5_accuracy" in history, sorted(history)
    assert "top_5_accuracy" in summary["test_metrics_best"]
    assert "top_5_accuracy" in summary["test_metrics_final"]
    assert summary["num_classes"] == 100 and summary["drop_path_rate"] == 0.2
    csv_header = (tmp_path / "wide" / "training_log.csv").read_text().splitlines()[0]
    assert "top_5_accuracy" in csv_header


# ---------------------------------------------------------------------
# Validation split (D-004)
# ---------------------------------------------------------------------


def _split_ids(monkeypatch, seed: int, validation_split: float = 0.1, max_samples=None):
    loader, total = _identity_loader(n_train=200, n_test=100)
    monkeypatch.setattr(common, "load_dataset", loader)
    config = common.TrainingConfig(
        seed=seed, validation_split=validation_split, max_samples=max_samples)
    data = common.prepare_data(config)
    return data, {
        "train": _ids(data, data.x_train, total),
        "val": _ids(data, data.x_val, total),
        "test": _ids(data, data.x_test, total),
    }


def test_the_validation_split_is_a_disjoint_cut_of_the_train_set(monkeypatch) -> None:
    data, ids = _split_ids(monkeypatch, seed=7)
    assert len(ids["val"]) == 20 and len(ids["train"]) == 180 and len(ids["test"]) == 100
    train, val, test = (set(ids[k].tolist()) for k in ("train", "val", "test"))
    assert len(train) == 180 and len(val) == 20, "no repeated sample inside a split"
    assert train.isdisjoint(val), "validation leaks into the fit set"
    assert val.isdisjoint(test) and train.isdisjoint(test), "the test set is not validation data"
    assert train | val == set(range(200)), "train + val is exactly the train pool"
    assert test == set(range(200, 300)), "the test set is the whole, untouched test pool"
    assert data.num_classes == 10 and data.input_shape == (32, 32, 3)


def test_the_split_is_seeded(monkeypatch) -> None:
    _, first = _split_ids(monkeypatch, seed=7)
    _, again = _split_ids(monkeypatch, seed=7)
    _, other = _split_ids(monkeypatch, seed=8)
    for part in ("train", "val", "test"):
        np.testing.assert_array_equal(first[part], again[part])
    assert not np.array_equal(first["val"], other["val"]), "a different seed must cut differently"
    assert set(first["val"].tolist()) != set(other["val"].tolist())


def test_max_samples_caps_the_train_pool_and_the_test_set(monkeypatch) -> None:
    _, ids = _split_ids(monkeypatch, seed=7, max_samples=50)
    assert (len(ids["train"]), len(ids["val"]), len(ids["test"])) == (45, 5, 50)
    assert set(ids["val"].tolist()) | set(ids["train"].tolist()) <= set(range(200))
    assert set(ids["test"].tolist()) <= set(range(200, 300))


def test_a_split_that_holds_out_nothing_is_refused(monkeypatch) -> None:
    loader, _ = _identity_loader(n_train=200, n_test=100)
    monkeypatch.setattr(common, "load_dataset", loader)
    config = common.TrainingConfig(max_samples=5, validation_split=0.1)
    with pytest.raises(ValueError, match="holds out 0"):
        common.prepare_data(config)


# ---------------------------------------------------------------------
# Geometry helper equals the real model
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family,variant", [("v1", "cifar10"), ("v2", "atto")])
@pytest.mark.parametrize("strides", [2, 4])
@pytest.mark.parametrize("size", [32, 28])
def test_stage_feature_map_sizes_equals_the_real_last_stage(family, variant, strides, size) -> None:
    """``include_top=False`` returns the last stage's feature map; the helper predicts it."""
    factory = common.MODEL_FAMILIES[family].factory
    model = factory(
        variant=variant, num_classes=10, input_shape=(size, size, 3), strides=strides,
        include_top=False,
    )
    out = model(np.zeros((1, size, size, 3), dtype=np.float32))
    sizes = common.stage_feature_map_sizes((size, size), model.depths, strides)
    assert len(sizes) == len(model.depths)
    assert tuple(out.shape[1:3]) == sizes[-1], (family, variant, strides, size, sizes)


@pytest.mark.parametrize(
    "hw,depths,strides,expected",
    [
        ((32, 32), [5, 5], 4, [(8, 8), (2, 2)]),
        ((32, 32), [2, 2, 6, 2], 4, [(8, 8), (2, 2), (1, 1), (1, 1)]),
        ((28, 28), [2, 2, 6, 2], 4, [(7, 7), (2, 2), (1, 1), (1, 1)]),
        ((32, 32), [2, 2, 6, 2], 2, [(16, 16), (8, 8), (4, 4), (2, 2)]),
        ((32, 28), [5, 5], 4, [(8, 7), (2, 2)]),
    ],
)
def test_stage_feature_map_sizes_worked_examples(hw, depths, strides, expected) -> None:
    assert common.stage_feature_map_sizes(hw, depths, strides) == expected


# ---------------------------------------------------------------------
# imagenet
# ---------------------------------------------------------------------


def test_imagenet_raises_the_explicit_error_and_no_run_directory(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="imagenet2012"):
        common.TrainingConfig(dataset="imagenet", output_dir=str(tmp_path), experiment_name="x")
    assert list(tmp_path.iterdir()) == []
