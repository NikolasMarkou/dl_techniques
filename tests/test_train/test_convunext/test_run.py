"""Behavioural guards for ``train()`` of the ConvUNext segmentation trainer.

One real tiny end-to-end run (32 px, 3 epochs, batch 4) on SYNTHETIC arrays, with the ONE
function that touches TFDS (``common.load_oxford_pet``) stubbed, is shared by most tests.
The synthetic mask is a learnable function of the image (three horizontal colour bands), so
the run really learns; every image also carries its identity in its corner pixel, so the
train / validation / test disjointness can be audited from the arrays the trainer holds.

Each guard exists because the defect it names is silent in a green run:

- the best checkpoint would be the LAST epoch (Keras 3.8 ``EarlyStopping`` restores the
  best weights at every train end unless switched off), so ``final_model.keras`` would
  equal ``best_model.keras``;
- the CSV ``lr`` would be the NEXT epoch's first-step rate instead of the rate the epoch
  trained at;
- the test split would leak into selection (validation from the test set, or an evaluate
  on it before ``fit`` ended);
- the head or the loss would be a softmax/``from_logits=False`` mismatch that trains but
  reports a wrong loss;
- the mask would be flipped independently of the image;
- the mIoU would be trusted from one instrument only (a Keras metric and a numpy confusion
  matrix are cross-checked here);
- a reused name would be merged, or a missing dataset cache would fail after a run
  directory was created;
- a diverged run would look finished.

Everything is written under ``tmp_path`` (a conftest guard fails any write to the repo-root
``results/``). Nothing reads the TFDS cache; the loader test drives ``load_oxford_pet``
against a stubbed ``tfds.load``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

import train.convunext.common as common  # noqa: E402
from train.common import json_numpy_default  # noqa: E402

SIZE = 32
N_TRAIN, N_TEST = 120, 40
MAX_SAMPLES = 100
EPOCHS = 3
BATCH = 4
PEAK_LR = 3e-3
TEST_ID_OFFSET = 1000
# Added to the LAST epoch's ``val_loss`` so that the last epoch is deterministically NOT the
# best one; the natural curve of a 3-epoch run has a device-dependent argmin.
LAST_EPOCH_PENALTY = 100.0
BAND_COLOURS = np.array([[220, 40, 40], [40, 200, 60], [50, 60, 230]], dtype=np.int16)


def _strict(text: str) -> Any:
    def refuse(token: str):
        raise ValueError(f"non-strict JSON constant {token!r}")

    return json.loads(text, parse_constant=refuse)


def _split(n: int, size: int, first_id: int, seed: int):
    """``n`` band images and masks. Class = horizontal third; colour per band + jitter.

    Pixel ``[0, 0]`` holds the image identity (``id % 256``, ``id // 256`` in channels 0, 1).
    """
    rng = np.random.default_rng(seed)
    band = np.minimum(np.arange(size) * 3 // size, 2)
    colour = BAND_COLOURS[band][None, :, None, :] + rng.integers(-20, 21, size=(n, 1, 1, 3))
    x = np.broadcast_to(colour, (n, size, size, 3)).clip(0, 255).astype(np.uint8).copy()
    y = np.broadcast_to(band[None, :, None], (n, size, size)).astype(np.uint8).copy()
    ids = np.arange(first_id, first_id + n)
    x[:, 0, 0, 0], x[:, 0, 0, 1] = ids % 256, ids // 256
    return x, y


def _band_loader(events: List[Any]):
    """Stub of ``load_oxford_pet``: ``((x_train, y_train), (x_test, y_test))`` at ``image_size``."""

    def load(image_size: int):
        events.append(("load",))
        return _split(N_TRAIN, image_size, 0, 1), _split(N_TEST, image_size, TEST_ID_OFFSET, 2)

    return load


def _ids(x: np.ndarray) -> np.ndarray:
    return x[:, 0, 0, 0].astype(int) + 256 * x[:, 0, 0, 1].astype(int)


def _config(tmp_path: Path, name: str, **overrides) -> "common.SegTrainingConfig":
    settings = dict(
        variant="tiny", image_size=SIZE, epochs=EPOCHS, batch_size=BATCH, learning_rate=PEAK_LR,
        warmup_epochs=1, max_samples=MAX_SAMPLES, seed=5, output_dir=str(tmp_path),
        experiment_name=name,
    )
    settings.update(overrides)
    return common.SegTrainingConfig(**settings)


def _snapshot(directory: Path) -> Dict[str, str]:
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*")) if p.is_file()
    }


def _file_handlers():
    return [h for h in logging.getLogger("dl").handlers if isinstance(h, logging.FileHandler)]


def _csv_rows(run_dir: Path) -> List[Dict[str, str]]:
    with open(run_dir / "training_log.csv") as f:
        return list(csv.DictReader(f))


def _max_abs_diff(a, b) -> float:
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


class _Recorder(keras.callbacks.Callback):
    """Copies every weight tensor at the end of each epoch that runs (behind the stock callbacks)."""

    def __init__(self) -> None:
        super().__init__()
        self.weights: List[List[np.ndarray]] = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([np.array(w) for w in self.model.get_weights()])


class _LastEpochPenalty(keras.callbacks.Callback):
    """In FRONT of the list: EarlyStopping, ModelCheckpoint and CSVLogger read ``logs`` later."""

    def on_epoch_end(self, epoch, logs=None):
        if epoch == EPOCHS - 1:
            logs["val_loss"] = float(logs["val_loss"]) + LAST_EPOCH_PENALTY


class _NanLossAtEpoch(keras.callbacks.Callback):
    """A NaN epoch loss plus the stop ``TerminateOnNaN`` issues after a NaN batch.

    A real divergence is not reproducible on demand, so this injects its two observable
    effects: the NaN in the epoch ``logs`` (which History and CSVLogger record) and
    ``stop_training``.
    """

    def __init__(self, epoch_index: int) -> None:
        super().__init__()
        self.epoch_index = epoch_index

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epoch_index:
            logs["loss"] = float("nan")
            self.model.stop_training = True


def _patched_callbacks(monkeypatch, front=(), back=()) -> None:
    original = common.create_callbacks

    def patched(*args, **kwargs):
        callbacks, results_dir = original(*args, **kwargs)
        return [*front, *callbacks, *back], results_dir

    monkeypatch.setattr(common, "create_callbacks", patched)


def _spy_fit_and_eval_datasets(monkeypatch, events: List[Any], seen: Dict[str, Any]) -> None:
    """Record fit start/end, the callbacks and validation size of ``fit``, and every eval dataset built."""
    real_fit = keras.Model.fit
    real_eval = common.make_eval_dataset

    def fit(self, *args, **kwargs):
        events.append(("fit_start",))
        seen["callbacks"] = list(kwargs.get("callbacks", []))
        seen["val_n"] = sum(int(x.shape[0]) for x, _ in kwargs["validation_data"])
        try:
            return real_fit(self, *args, **kwargs)
        finally:
            events.append(("fit_end",))

    def make_eval(x, y, batch_size):
        events.append(("eval_ds", len(x)))
        return real_eval(x, y, batch_size)

    monkeypatch.setattr(keras.Model, "fit", fit)
    monkeypatch.setattr(common, "make_eval_dataset", make_eval)


# ---------------------------------------------------------------------
# One real tiny end-to-end run, shared by many tests
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    out = tmp_path_factory.mktemp("convunext_e2e")
    events: List[Any] = []
    seen: Dict[str, Any] = {}
    recorder = _Recorder()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(common, "load_oxford_pet", _band_loader(events))
        _patched_callbacks(mp, front=[_LastEpochPenalty()], back=[recorder])
        _spy_fit_and_eval_datasets(mp, events, seen)
        config = _config(out, "e2e")
        summary = common.train(config)
        data = common.prepare_data(config)
    run_dir = out / "e2e"
    return SimpleNamespace(
        out=out, run_dir=run_dir, config=config, summary=summary, data=data, events=events,
        epoch_weights=recorder.weights, seen=seen,
        best=keras.models.load_model(run_dir / "best_model.keras"),
        final=keras.models.load_model(run_dir / "final_model.keras"),
    )


ARTIFACTS = (
    "config.json", "run.log", "training_log.csv", "training_history.json", "best_model.keras",
    "final_model.keras", "results_summary.json",
)


def test_the_run_writes_the_full_artifact_set(e2e) -> None:
    for name in ARTIFACTS:
        assert (e2e.run_dir / name).stat().st_size > 0, name
    assert (e2e.run_dir / "visualizations" / "training_dashboard.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(p.name for p in e2e.out.iterdir()) == ["e2e"], "nothing outside the run directory"
    assert _file_handlers() == [], "the run.log handler must be removed when train returns"
    log = (e2e.run_dir / "run.log").read_text()
    assert "Run directory:" in log and "Untrained evaluate BEFORE fit" in log and "Data (oxford_iiit_pet" in log


def test_the_summary_is_strict_json_with_every_promised_key(e2e) -> None:
    summary = _strict((e2e.run_dir / "results_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary == _strict(json.dumps(e2e.summary, default=json_numpy_default))
    for key in (
        "run_dir", "experiment_name", "model_family", "dataset", "variant", "params", "input_shape",
        "num_classes", "class_names", "optimizer", "learning_rate", "lr_schedule", "warmup_epochs",
        "weight_decay", "batch_size", "seed", "monitor", "n_train", "n_val", "n_test",
        "steps_per_epoch", "initial_loss_sanity_eval", "initial_loss_ratio", "init_scale_warning",
        "gpu_name", "tf_visible_devices", "cuda_visible_devices", "epochs_run", "stopped_early",
        "best_epoch", "best_epoch_csv_index", "final_is_best", "lr_first_epoch", "lr_last_epoch",
        "lr_last_step", "best_val_metrics", "final_val_metrics", "test_metrics_best",
        "test_metrics_final", "trivial_baseline", "best_checkpoint_load_error",
        "best_checkpoint_max_abs_diff", "epoch_times", "fit_wall_seconds", "data_load_seconds",
        "model_loading_validated", "notes",
    ):
        assert key in summary, key
    assert summary["model_family"] == "convunext-seg" and summary["dataset"] == "oxford_iiit_pet"
    assert summary["input_shape"] == [SIZE, SIZE, 3] and summary["class_names"] == ["pet", "background", "border"]
    assert summary["optimizer"] == "AdamW" and summary["lr_schedule"] == "cosine"
    assert summary["monitor"] == "val_loss"
    assert summary["epochs_run"] == EPOCHS and len(summary["epoch_times"]) == EPOCHS
    assert all(t > 0.0 for t in summary["epoch_times"])
    for which in ("test_metrics_best", "test_metrics_final"):
        block = summary[which]
        for key in ("loss", "accuracy", "miou", "miou_from_confusion", "pixel_accuracy",
                    "per_class_iou", "confusion"):
            assert key in block, (which, key)
        assert len(block["per_class_iou"]) == 3 and np.array(block["confusion"]).shape == (3, 3)
        assert int(np.sum(block["confusion"])) == N_TEST * SIZE * SIZE
    assert summary["initial_loss_ratio"] == pytest.approx(
        summary["initial_loss_sanity_eval"]["loss"] / np.log(3))
    assert summary["initial_loss_sanity_eval"]["before_fit"] is True
    assert summary["best_checkpoint_load_error"] is None
    assert summary["best_checkpoint_max_abs_diff"] <= common.BEST_CHECKPOINT_TOLERANCE
    assert summary["model_loading_validated"] is True
    assert (e2e.run_dir / "results_summary.json").read_text().count("NaN") == 0


def test_the_split_sizes_follow_the_cap_and_the_test_split_is_never_capped(e2e) -> None:
    """max_samples 100 of 120 train images: 10 validation, 90 fit; all 40 test images."""
    summary = e2e.summary
    assert (summary["n_train"], summary["n_val"], summary["n_test"]) == (90, 10, N_TEST)
    assert summary["steps_per_epoch"] == 90 // BATCH == 22
    assert summary["max_samples"] == MAX_SAMPLES


def test_the_train_validation_and_test_splits_are_disjoint_and_the_pool_is_train_only(e2e) -> None:
    data = e2e.data
    train, val, test = _ids(data["x_train"]), _ids(data["x_val"]), _ids(data["x_test"])
    assert not set(train) & set(val), "validation must be disjoint from the fit split"
    assert len(set(train)) == len(train) and len(set(val)) == len(val)
    assert set(test) == set(range(TEST_ID_OFFSET, TEST_ID_OFFSET + N_TEST)), "the full test split"
    assert not (set(train) | set(val)) & set(test), "no train or validation image is a test image"
    assert max(train.max(), val.max()) < N_TRAIN, "the pool is drawn from the TRAIN split only"
    assert len(set(train) | set(val)) == MAX_SAMPLES
    # Masks travel with their images.
    np.testing.assert_array_equal(data["y_train"][:, 5, 0], np.zeros(len(train)))
    np.testing.assert_array_equal(data["y_train"][:, -1, 0], np.full(len(train), 2))


def test_the_split_is_seeded(e2e, monkeypatch) -> None:
    monkeypatch.setattr(common, "load_oxford_pet", _band_loader([]))
    same = common.prepare_data(e2e.config)
    np.testing.assert_array_equal(_ids(same["x_val"]), _ids(e2e.data["x_val"]))
    other = common.prepare_data(_config(e2e.out, "other", seed=6))
    assert not np.array_equal(_ids(other["x_val"]), _ids(e2e.data["x_val"])), "the seed picks the slice"


def test_the_test_split_is_read_only_after_fit_and_selection_monitors_validation_loss(e2e) -> None:
    events = [event for event in e2e.events if event[0] != "load"]
    fit_end = events.index(("fit_end",))
    test_reads = [i for i, event in enumerate(events) if event == ("eval_ds", N_TEST)]
    assert test_reads, "the test split is evaluated"
    assert min(test_reads) > fit_end, f"the test split was read before fit ended: {events}"
    assert e2e.seen["val_n"] == 10, "fit validates on the train-split slice, not on the test split"
    monitors = {type(cb).__name__: cb.monitor for cb in e2e.seen["callbacks"] if hasattr(cb, "monitor")}
    assert monitors == {"EarlyStopping": "val_loss", "ModelCheckpoint": "val_loss"}, monitors
    assert e2e.summary["monitor"] == "val_loss"


def test_the_csv_epoch_is_zero_based_and_lr_is_the_rate_at_the_start_of_each_epoch(e2e) -> None:
    rows = _csv_rows(e2e.run_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1, 2]
    steps = e2e.summary["steps_per_epoch"]
    schedule = common.build_lr_schedule(e2e.config, steps)
    expected = [float(keras.ops.convert_to_numpy(schedule(epoch * steps))) for epoch in range(EPOCHS)]
    got = [float(r["lr"]) for r in rows]
    np.testing.assert_allclose(got, expected, rtol=1e-4)
    assert got[0] < 1e-6 < PEAK_LR * 0.9, "epoch 0 starts at the warmup start value, far below the peak"
    assert got[1] == pytest.approx(PEAK_LR, rel=1e-3), "epoch 1 starts where the warmup ends"
    assert e2e.summary["lr_first_epoch"] == pytest.approx(got[0], rel=1e-4)
    assert e2e.summary["lr_last_epoch"] == pytest.approx(got[-1], rel=1e-4)
    last_step_lr = float(keras.ops.convert_to_numpy(schedule(EPOCHS * steps - 1)))
    assert e2e.summary["lr_last_step"] == pytest.approx(last_step_lr, rel=1e-4)


def test_the_run_learns_the_synthetic_bands_better_than_the_majority_class(e2e) -> None:
    """The stub's mask is a function of the image: a run that does not beat the constant
    predictor has a wiring defect (labels, metric, loss) rather than too few steps."""
    best = e2e.summary["test_metrics_best"]
    trivial = e2e.summary["trivial_baseline"]
    assert best["miou"] > trivial["miou"] + 0.2, (best["miou"], trivial["miou"])
    assert min(v for v in best["per_class_iou"] if v is not None) > 0.0, "every class is predicted"


def test_the_precondition_of_the_final_versus_best_guards_holds(e2e) -> None:
    assert e2e.summary["final_is_best"] is False, (
        f"best epoch {e2e.summary['best_epoch']} of {EPOCHS}: the penalty callback did not act")
    assert e2e.summary["best_epoch"] < EPOCHS and e2e.summary["best_epoch_csv_index"] == e2e.summary["best_epoch"] - 1


def test_final_model_holds_the_last_epoch_weights_and_best_holds_the_best_epoch(e2e) -> None:
    """Keras 3.8 EarlyStopping restores the best weights at train end unless switched off."""
    assert len(e2e.epoch_weights) == EPOCHS
    assert _max_abs_diff(e2e.final.get_weights(), e2e.epoch_weights[-1]) < 1e-6
    assert _max_abs_diff(e2e.best.get_weights(), e2e.epoch_weights[e2e.summary["best_epoch"] - 1]) < 1e-6
    assert _max_abs_diff(e2e.final.get_weights(), e2e.best.get_weights()) > 1e-4


def test_early_stopping_is_built_with_restore_best_weights_off(e2e) -> None:
    stops = [cb for cb in e2e.seen["callbacks"] if isinstance(cb, keras.callbacks.EarlyStopping)]
    assert len(stops) == 1 and stops[0].restore_best_weights is False


def test_the_best_checkpoint_reproduces_the_best_epoch_and_final_differs_from_it(e2e) -> None:
    summary = e2e.summary
    best_i = summary["best_epoch_csv_index"]
    rows = _csv_rows(e2e.run_dir)
    assert float(rows[best_i]["val_loss"]) == pytest.approx(min(float(r["val_loss"]) for r in rows))
    assert summary["test_metrics_final"]["loss"] != pytest.approx(summary["test_metrics_best"]["loss"], rel=1e-4)


def _numpy_miou(model, x: np.ndarray, y: np.ndarray):
    """mIoU, pixel accuracy and confusion of ``model`` on uint8 ``x`` / ``y``, written out
    here (no import from the trainer) so the trainer's own confusion code is cross-checked."""
    logits = model.predict(x.astype("float32") / 255.0, batch_size=8, verbose=0)
    pred = logits.argmax(-1).reshape(-1)
    true = y.reshape(-1).astype(int)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(true, pred):
        confusion[t, p] += 1
    ious = []
    for c in range(3):
        union = confusion[c].sum() + confusion[:, c].sum() - confusion[c, c]
        if union:
            ious.append(confusion[c, c] / union)
    return float(np.mean(ious)), float(np.trace(confusion) / confusion.sum()), confusion


@pytest.mark.parametrize("which,model_attr", [("test_metrics_best", "best"), ("test_metrics_final", "final")])
def test_the_confusion_matrix_miou_equals_the_keras_metric_and_an_independent_recount(
        e2e, which, model_attr) -> None:
    block = e2e.summary[which]
    assert block["miou_from_confusion"] == pytest.approx(block["miou"], abs=2e-3)
    assert block["pixel_accuracy"] == pytest.approx(block["accuracy"], abs=2e-3)
    miou, accuracy, confusion = _numpy_miou(getattr(e2e, model_attr), e2e.data["x_test"], e2e.data["y_test"])
    assert miou == pytest.approx(block["miou_from_confusion"], abs=2e-3)
    assert accuracy == pytest.approx(block["pixel_accuracy"], abs=2e-3)
    np.testing.assert_allclose(np.array(block["confusion"]), confusion, atol=max(2, 0.001 * confusion.sum()))


def test_the_trivial_baseline_is_the_majority_class_of_the_test_masks(e2e) -> None:
    baseline = e2e.summary["trivial_baseline"]
    counts = np.bincount(e2e.data["y_test"].reshape(-1), minlength=3)
    assert baseline["predicted_class"] == int(np.argmax(counts))
    assert baseline["pixel_accuracy"] == pytest.approx(counts.max() / counts.sum())
    assert baseline["miou"] == pytest.approx((counts.max() / counts.sum()) / 3)


def test_trivial_baseline_on_a_hand_built_mask() -> None:
    """10 pixels: 6 of class 0, 3 of class 1, 1 of class 2. Predicting 0 everywhere gives
    IoU_0 = 6/10 (union is every pixel), 0 for the others, mIoU 0.2, accuracy 0.6."""
    mask = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 2]])
    baseline = common.trivial_baseline(mask)
    assert baseline["predicted_class"] == 0
    assert baseline["confusion"] == [[6, 0, 0], [3, 0, 0], [1, 0, 0]]
    assert baseline["per_class_iou"] == pytest.approx([0.6, 0.0, 0.0])
    assert baseline["miou"] == pytest.approx(0.2) and baseline["pixel_accuracy"] == pytest.approx(0.6)


def test_trivial_baseline_ties_go_to_the_lowest_class_and_an_absent_class_is_left_out() -> None:
    tie = common.trivial_baseline(np.array([1, 1, 2, 2]))
    assert tie["predicted_class"] == 1
    absent = common.trivial_baseline(np.array([0, 0, 0, 1]))  # class 2 never occurs
    assert absent["per_class_iou"][2] is None
    assert absent["miou"] == pytest.approx((0.75 + 0.0) / 2), "the empty-union class is not in the mean"


def test_segmentation_scores_worked_example() -> None:
    confusion = np.array([[5, 1, 0], [2, 3, 0], [0, 0, 0]])
    scores = common.segmentation_scores(confusion)
    assert scores["per_class_iou"][0] == pytest.approx(5 / (6 + 7 - 5))
    assert scores["per_class_iou"][1] == pytest.approx(3 / (5 + 4 - 3))
    assert scores["per_class_iou"][2] is None
    assert scores["miou"] == pytest.approx((5 / 8 + 3 / 6) / 2)
    assert scores["pixel_accuracy"] == pytest.approx(8 / 11)


def test_confusion_matrix_refuses_an_out_of_range_class() -> None:
    with pytest.raises(ValueError, match="class ids"):
        common.confusion_matrix(np.array([0, 3]), np.array([0, 1]), 3)
    with pytest.raises(ValueError, match="elements"):
        common.confusion_matrix(np.array([0, 1]), np.array([0]), 3)
    np.testing.assert_array_equal(
        common.confusion_matrix(np.array([0, 1, 1]), np.array([0, 2, 1]), 3),
        [[1, 0, 0], [0, 1, 1], [0, 0, 0]])


# ---------------------------------------------------------------------
# Reused name, missing cache, divergence
# ---------------------------------------------------------------------

def test_a_reused_experiment_name_is_refused_before_any_load_and_the_first_run_is_byte_identical(
        monkeypatch, e2e) -> None:
    events: List[Any] = []
    monkeypatch.setattr(common, "load_oxford_pet", _band_loader(events))
    before = _snapshot(e2e.run_dir)
    assert "results_summary.json" in before and "run.log" in before
    with pytest.raises(FileExistsError) as raised:
        common.train(_config(e2e.out, "e2e", epochs=2, seed=6))
    assert str(e2e.run_dir) in str(raised.value)
    assert events == [], "the refusal must come before the dataset is touched"
    assert _snapshot(e2e.run_dir) == before, "the refusal wrote or deleted something"
    assert _file_handlers() == [], "no handler may be attached before the refusal"


def test_a_missing_dataset_cache_raises_before_any_run_directory_exists(monkeypatch, tmp_path) -> None:
    def broken(image_size):
        raise RuntimeError("oxford_iiit_pet is not in the TFDS cache")

    monkeypatch.setattr(common, "load_oxford_pet", broken)
    with pytest.raises(RuntimeError, match="not in the TFDS cache"):
        common.train(_config(tmp_path, "no_cache"))
    assert list(tmp_path.iterdir()) == [], "no run directory, config.json or run.log may exist"
    assert _file_handlers() == []


@pytest.fixture(scope="module")
def diverged(tmp_path_factory):
    out = tmp_path_factory.mktemp("convunext_diverged")
    raised = None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(common, "load_oxford_pet", _band_loader([]))
        _patched_callbacks(mp, front=[_NanLossAtEpoch(1)])
        try:
            common.train(_config(out, "diverged", epochs=3, warmup_epochs=0, max_samples=40))
        except RuntimeError as error:
            raised = error
    return SimpleNamespace(run_dir=out / "diverged", raised=raised)


def test_a_diverged_run_writes_a_strict_diverged_summary_and_no_final_model(diverged) -> None:
    assert diverged.raised is not None and "diverged" in str(diverged.raised)
    summary = _strict((diverged.run_dir / "results_summary.json").read_text())
    assert summary["status"] == "diverged"
    assert summary["epochs_run"] == 2 and summary["best_epoch"] is None and summary["stopped_early"] is None
    assert "loss" in summary["non_finite_metrics"]
    assert summary["history"]["loss"][1] is None, "the NaN is written as null (strict JSON)"
    for key in ("n_train", "n_val", "n_test", "initial_loss_ratio", "epoch_times", "fit_wall_seconds"):
        assert key in summary, key
    assert "test_metrics_best" not in summary and "trivial_baseline" not in summary
    assert not (diverged.run_dir / "final_model.keras").exists()
    assert (diverged.run_dir / "best_model.keras").exists(), "the checkpoint of epoch 1 stays"
    assert _file_handlers() == []


# ---------------------------------------------------------------------
# Model (D-004): linear head, sparse cross-entropy on logits, named metrics
# ---------------------------------------------------------------------

def test_the_head_is_linear_logits_and_the_loss_reads_logits(tmp_path) -> None:
    config = _config(tmp_path, "model")
    schedule = common.build_lr_schedule(config, 10)
    model = common.build_model(config, schedule)
    assert model.output_shape == (None, SIZE, SIZE, 3)
    assert model.layers[-1].get_config()["activation"] == "linear"
    assert model.layers[-1].get_config()["use_bias"] is True
    loss = model.loss
    assert isinstance(loss, keras.losses.SparseCategoricalCrossentropy) and loss.from_logits is True

    x = np.random.default_rng(0).random((2, SIZE, SIZE, 3), dtype=np.float32)
    logits = model.predict(x, verbose=0)
    assert not np.allclose(logits.sum(-1), 1.0), "a softmax head would sum to 1: this must be logits"
    y = np.random.default_rng(1).integers(0, 3, size=(2, SIZE, SIZE))
    shifted = logits - logits.max(-1, keepdims=True)
    log_softmax = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    reference = -np.mean(np.take_along_axis(log_softmax, y[..., None], axis=-1))
    assert float(loss(y, logits)) == pytest.approx(reference, rel=1e-4)


def test_the_metrics_are_the_stock_keras_ones_named_accuracy_and_miou(tmp_path) -> None:
    config = _config(tmp_path, "metrics")
    model = common.build_model(config, common.build_lr_schedule(config, 10))
    x = np.zeros((2, SIZE, SIZE, 3), dtype=np.float32)
    y = np.zeros((2, SIZE, SIZE), dtype=np.int32)
    metrics = model.evaluate(x, y, verbose=0, return_dict=True)
    assert set(metrics) == {"loss", "accuracy", "miou"}


# ---------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------

def _pattern_arrays(n: int = 48, width: int = 12):
    """Column index in channel 0, and mask = column index % 3: a flip that moves the image
    but not the mask (or the reverse) breaks ``mask == column % 3`` of the flipped image."""
    column = np.arange(width, dtype=np.uint8)
    x = np.zeros((n, 4, width, 3), dtype=np.uint8)
    x[..., 0] = column[None, None, :]
    y = np.broadcast_to((column % 3)[None, None, :], (n, 4, width)).astype(np.uint8).copy()
    return x, y


def test_the_train_pipeline_flips_image_and_mask_together_and_scales_to_unit_range() -> None:
    x, y = _pattern_arrays()
    tf.random.set_seed(0)
    flipped = 0
    for images, masks in common.make_train_dataset(x, y, batch_size=8, seed=3):
        assert images.dtype == tf.float32 and masks.dtype == tf.int32
        assert float(images.numpy().max()) <= 1.0 and float(images.numpy().min()) >= 0.0
        column = np.rint(images.numpy()[..., 0] * 255).astype(int)  # column index of each pixel
        np.testing.assert_array_equal(masks.numpy(), column % 3)
        flipped += int((column[:, 0, 0] != 0).sum())
    assert 0 < flipped < 48, f"{flipped} of 48 flipped: the flip must be random per image"


def test_the_train_pipeline_yields_exactly_steps_per_epoch_full_batches_and_reshuffles() -> None:
    x, y = _pattern_arrays(n=45)
    dataset = common.make_train_dataset(x, y, batch_size=8, seed=3)
    epoch_1 = [b[0].shape[0] for b in dataset]
    assert epoch_1 == [8] * common.steps_per_epoch_for(45, 8) == [8] * 5

    ids = np.arange(40)
    xi = np.zeros((40, 2, 2, 3), dtype=np.uint8)
    xi[:, :, :, 0] = ids[:, None, None]  # every pixel: a flip cannot move the identity away
    yi = np.zeros((40, 2, 2), dtype=np.uint8)
    dataset = common.make_train_dataset(xi, yi, batch_size=40, seed=1)
    first = [int(round(v * 255)) for v in next(iter(dataset))[0].numpy()[:, 0, 0, 0]]
    second = [int(round(v * 255)) for v in next(iter(dataset))[0].numpy()[:, 0, 0, 0]]
    assert sorted(first) == sorted(second) == list(range(40)) and first != second, \
        "every pass reshuffles the pool"


def test_the_eval_pipeline_is_ordered_unaugmented_and_keeps_the_partial_last_batch() -> None:
    x, y = _pattern_arrays(n=10)
    batches = list(common.make_eval_dataset(x, y, batch_size=4))
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]
    images = np.concatenate([b[0].numpy() for b in batches])
    np.testing.assert_allclose(images, x.astype(np.float32) / 255.0)
    np.testing.assert_array_equal(np.concatenate([b[1].numpy() for b in batches]), y)


# ---------------------------------------------------------------------
# The TFDS loader (the one function that touches TFDS), against a stubbed tfds.load
# ---------------------------------------------------------------------

def _fake_tfds_split(n: int, height: int, width: int, mask_value: int = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(n, height, width, 3)).astype(np.uint8)
    if mask_value is None:
        mask = np.empty((n, height, width, 1), dtype=np.uint8)
        mask[:, :, :width // 2] = 1
        mask[:, :, width // 2:] = 2
        mask[:, :4] = 3
    else:
        mask = np.full((n, height, width, 1), mask_value, dtype=np.uint8)
    return tf.data.Dataset.from_tensor_slices({"image": image, "segmentation_mask": mask})


def test_the_loader_maps_tfds_masks_to_zero_based_classes_and_resizes(monkeypatch) -> None:
    import tensorflow_datasets as tfds

    calls: List[Any] = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return _fake_tfds_split(6 if kwargs["split"] == "train" else 3, 20, 30)

    monkeypatch.setattr(tfds, "load", load)
    (x_train, y_train), (x_test, y_test) = common.load_oxford_pet(16)
    assert [(c[0], c[1]["split"], c[1]["download"]) for c in calls] == [
        ("oxford_iiit_pet:4.*.*", "train", False), ("oxford_iiit_pet:4.*.*", "test", False)]
    assert x_train.shape == (6, 16, 16, 3) and x_train.dtype == np.uint8
    assert y_train.shape == (6, 16, 16) and y_train.dtype == np.uint8
    assert x_test.shape[0] == y_test.shape[0] == 3
    assert set(np.unique(y_train)) == {0, 1, 2}, "TFDS 1, 2, 3 become 0, 1, 2 (nearest, no new value)"
    # Left half is TFDS 1 (pet, class 0), right half TFDS 2 (background, class 1), top rows TFDS 3
    # (border, class 2): read a pixel of each region away from the borders.
    assert y_train[0, 8, 2] == 0 and y_train[0, 8, 13] == 1 and y_train[0, 0, 8] == 2


@pytest.mark.parametrize("value,expected", [(1, 0), (2, 1), (3, 2)])
def test_the_loader_label_is_the_tfds_mask_minus_one(monkeypatch, value, expected) -> None:
    import tensorflow_datasets as tfds

    monkeypatch.setattr(tfds, "load", lambda name, **kw: _fake_tfds_split(2, 10, 10, mask_value=value))
    (_, y_train), _ = common.load_oxford_pet(8)
    assert set(np.unique(y_train)) == {expected}


def test_the_loader_refuses_a_mask_outside_the_three_classes(monkeypatch) -> None:
    import tensorflow_datasets as tfds

    monkeypatch.setattr(tfds, "load", lambda name, **kw: _fake_tfds_split(2, 10, 10, mask_value=4))
    with pytest.raises(ValueError, match="mask - 1 must lie in"):
        common.load_oxford_pet(8)


def test_the_loader_turns_a_missing_cache_into_one_clear_error(monkeypatch) -> None:
    import tensorflow_datasets as tfds

    def load(name, **kwargs):
        raise AssertionError("Dataset oxford_iiit_pet cannot be loaded at version 4.*.*, only: 3.0.0")

    monkeypatch.setattr(tfds, "load", load)
    with pytest.raises(RuntimeError) as raised:
        common.load_oxford_pet(16)
    text = str(raised.value)
    assert "TFDS_DATA_DIR" in text and "download=False" in text and "only: 3.0.0" in text
