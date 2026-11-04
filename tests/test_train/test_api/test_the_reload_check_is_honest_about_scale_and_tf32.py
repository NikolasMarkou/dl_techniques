"""The reload check of a segmentation run is honest on a correct file and blind to no wrong one (D-088, D-029).

Review-iter-3 concern 4. A correct AccUNet ``final_model.keras`` was reported ``model_loading_validated:
false`` on the RTX 4090 (``max_diff=0.308`` on scores of scale 220) while ``status`` was ``ok`` and
``success`` true, with no note: the absolute 1e-5 of ``validate_model_loading`` cannot hold on scores that
large. D-088 answered with an output tolerance of 7% of the largest score under TF32, blind to a wrong file
below it. The iteration 1 gate (plan 98b866fb) found two correct AccUNet files still 9.4% and 15.3% apart
with EQUAL weights, and fix 9.3 (D-029) found the cause, which no tolerance should hide: the in-memory model
and the reloaded one did not predict through the same call path. AccUNet's ``compile`` forces
``jit_compile=False`` (XLA cannot compile its backward pass), so ``model.predict`` ran without XLA, while
``load_model(compile=False)`` runs no ``compile`` and Keras resolves ``jit_compile="auto"`` to XLA on a GPU.
Copying the in-memory model's ``jit_compile`` onto the reloaded one made the difference EXACTLY 0.0 on the GPU
with TF32 on, for every correct file, and a planted wrong file still fails. So the check has one tolerance,
``LOAD_RTOL``, no TF32 branch, and compares the weights exactly as well.

What is claimed, on small models on the CPU (``reload_call_path_evidence.json`` holds the GPU numbers):

* the constant sits where the evidence puts it: ``LOAD_RTOL`` at least 10 times the worst correct file through
  one call path (GPU, TF32 on and off, the families of the gate) and under every planted wrong file, at least five
  times under the smallest of the files with ONE kernel 1% off; the old two-path comparison of the same files
  is above it (the defect, kept as a control);
* scores of scale 200 that differ by 0.02% of it are a correct file for the relative check and a wrong one
  for the absolute check (the review's defect), and a wrong file (other weights) fails the relative one;
* a file whose one kernel is 1% off fails on its outputs AND on its weights, each alone; a file whose one
  weight differs by a hair passes the outputs and fails the weights (the weights check is the one with no blind spot);
* the reloaded model predicts through the in-memory model's call path (``jit_compile``, ``run_eagerly``),
  set BEFORE it predicts, for both values of the flag;
* ``tf32_active`` is a GPU with TF32 on, nothing else, and TF32 is recorded, not a tolerance;
* ``finalize_last_epoch(exact_reload=True)`` fills the detail and a wrong file is reported false; the
  default (classification) path is untouched;
* a false check is never silent: status ``ok_reload_unverified``, a note with the numbers, the detail in
  ``model_loading_check``; a true one keeps ``ok`` and adds no note.

Everything is written under ``tmp_path``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.common import classification_run, evaluation
from train.common.evaluation import (
    LOAD_RTOL,
    check_model_loading,
    tf32_active,
    validate_model_loading,
)

HERE = Path(__file__).resolve().parent
#: D-088's measurement: the two paths of a correct file, and the wrong files, on trained weights.
EVIDENCE = json.loads((HERE / "load_check_evidence.json").read_text())["rows"]
#: D-029's measurement: the same files through ONE call path (the repo's own check, GPU, TF32 on).
CALL_PATH = json.loads((HERE / "reload_call_path_evidence.json").read_text())


# ---------------------------------------------------------------------
# The constant against the measurement
# ---------------------------------------------------------------------

def one_path_rows():
    return CALL_PATH["rows"]


def test_the_evidence_is_the_gpu_with_tf32_on_and_covers_the_family_that_showed_the_defect():
    assert CALL_PATH["rows"], "no rows"
    for row in one_path_rows():
        assert row["gpu"] is True, row["run"]
        assert row["correct_new_path_ok"] is True, f"{row['run']}: a correct file was reported false through one call path"
    assert any(row["tf32_active"] for row in one_path_rows()), "no row with TF32 on, the setting the defect appeared under"
    assert {row["family"] for row in one_path_rows()} >= {"accunet", "convunext"}
    assert any(row["model_jit"] is False for row in one_path_rows()), "no row of a model that predicts without XLA"
    assert any(row["model_jit"] is True for row in one_path_rows()), "no row of a model that predicts with XLA"


def test_the_tolerance_is_ten_times_the_worst_correct_file_of_one_call_path():
    worst = max(row["correct_new_path_worst_rel"] for row in one_path_rows())
    assert LOAD_RTOL >= 10 * worst, f"{LOAD_RTOL} against a worst correct file of {worst}"


def test_the_tolerance_is_under_every_planted_wrong_file_of_one_call_path():
    seen = [
        (row["run"], name, rel)
        for row in one_path_rows() for name, rel in row["wrong_file_smallest_rel"].items()
    ]
    kernels = [rel for _, name, rel in seen if name.startswith("kernel_x1.01")]
    assert kernels, "the 1% kernel is the planted file the claim is about"
    for run, name, rel in seen:
        assert rel > LOAD_RTOL, f"{run} {name}: {rel} is not above the tolerance {LOAD_RTOL}"
    # D-029: never widened past a planted 1% change of one kernel: at least five times under the smallest of them.
    assert LOAD_RTOL <= min(kernels) / 5, f"{LOAD_RTOL} is not five times under the smallest 1% kernel file ({min(kernels)})"
    # And the weights say it anyway: every planted file was reported false, and none of them has equal weights.
    assert all(not equal for row in one_path_rows() for equal in row["wrong_file_weights_equal"].values())
    assert all(flag for row in one_path_rows() for flag in row["wrong_file_reported_false"].values())


def test_the_one_percent_kernel_of_the_older_trained_weights_is_also_five_times_above():
    """D-088's own measurement (other trained weights, 64-image sets, GPU), an independent check of the smallest wrong file."""
    rels = [row["wrong_files"]["largest_tensor_x1.01"]["min_rel"] for row in EVIDENCE]
    assert len(rels) == 4 and all(rel >= 5 * LOAD_RTOL for rel in rels), rels


def test_the_two_paths_of_the_same_correct_file_are_the_defect_the_one_path_removes():
    """The control: through two paths the AccUNet files are above the tolerance (and above D-088's 0.07 in
    the gate); through one they are under a hundredth of it."""
    accunet = [row for row in one_path_rows() if row["model_jit"] is False]
    assert accunet and all(row["correct_old_two_path_worst_rel"] > LOAD_RTOL for row in accunet if row["tf32_active"])
    assert all(row["correct_new_path_worst_rel"] <= LOAD_RTOL / 10 for row in one_path_rows())
    gate = CALL_PATH["gate_cells"]
    accunet_cells = [cell for cell in gate if cell["family"] == "accunet"]
    assert len(accunet_cells) == 4 and sum(not cell["ml_ok"] for cell in accunet_cells) == 2
    assert all(cell["rel"] > LOAD_RTOL for cell in accunet_cells)
    for cell in (c for c in gate if c["family"] != "accunet"):
        assert cell["rel"] <= LOAD_RTOL / 10, cell["run"]


# ---------------------------------------------------------------------
# The check, on small models
# ---------------------------------------------------------------------

def scores_model(seed=0, kernel_scale=1.0):
    """A model whose scores have a scale of a few hundred, the scale of AccUNet's trained logits (220)."""
    rng = np.random.default_rng(seed)
    inputs = keras.Input((6,))
    layer = keras.layers.Dense(5, name="head")
    model = keras.Model(inputs, layer(inputs))
    layer.set_weights([
        (rng.standard_normal((6, 5)) * 120.0 * kernel_scale).astype("float32"), rng.standard_normal(5).astype("float32")])
    return model


@pytest.fixture(scope="module")
def sample():
    return np.random.default_rng(1).standard_normal((16, 6)).astype("float32")


def saved(model, tmp_path, name="m.keras"):
    path = tmp_path / name
    model.save(path)
    return str(path)


def test_scores_of_scale_200_that_differ_by_a_hair_are_a_correct_file_for_the_relative_check(sample, tmp_path):
    model = scores_model()
    expected = model.predict(sample, verbose=0)
    scale = float(np.abs(expected).max())
    assert scale > 100
    nudged = expected + 2e-4 * scale  # 0.02% of the scale: what a different convolution algorithm does
    path = saved(model, tmp_path)
    assert validate_model_loading(path, sample, nudged) is False, "the absolute 1e-5 of the review's defect"
    check = check_model_loading(path, sample, nudged, relative_tolerance=LOAD_RTOL, reference_model=model)
    assert check.ok and check.weights_equal is True
    seen = float(np.abs(nudged).max())  # the scale is the EXPECTED output's
    assert 0 < check.max_abs_diff <= check.tolerance and check.scale == pytest.approx(seen, rel=1e-6)
    assert check.tolerance == pytest.approx(1e-5 + LOAD_RTOL * seen, rel=1e-6)


def test_a_file_with_other_weights_fails_the_relative_check_by_its_outputs(sample, tmp_path):
    model, other = scores_model(0), scores_model(1)
    expected = model.predict(sample, verbose=0)
    check = check_model_loading(
        saved(other, tmp_path), sample, expected, relative_tolerance=LOAD_RTOL, reference_model=model)
    assert check.ok is False and check.max_abs_diff > check.tolerance and check.weights_equal is False


def test_a_one_percent_error_in_one_kernel_fails_on_its_outputs_and_on_its_weights_each_alone(sample, tmp_path):
    model = scores_model()
    expected = model.predict(sample, verbose=0)
    off = scores_model()
    kernel, bias = off.get_weights()
    off.set_weights([kernel * 1.01, bias])
    path = saved(off, tmp_path)
    outputs_only = check_model_loading(path, sample, expected, relative_tolerance=LOAD_RTOL)
    assert outputs_only.ok is False and outputs_only.weights_equal is None, "no reference model: the outputs alone see 1%"
    assert outputs_only.max_abs_diff > outputs_only.tolerance
    exact = check_model_loading(path, sample, expected, relative_tolerance=LOAD_RTOL, reference_model=model)
    assert exact.ok is False and exact.weights_equal is False and exact.max_abs_diff > exact.tolerance


def test_one_weight_off_by_a_hair_passes_the_outputs_and_fails_the_weights(sample, tmp_path):
    """The blind spot of any output tolerance, and the reason the weights are compared exactly."""
    model = scores_model()
    expected = model.predict(sample, verbose=0)
    off = scores_model()
    kernel, bias = off.get_weights()
    kernel[0, 0] = kernel[0, 0] * (1 + 1e-6)
    off.set_weights([kernel, bias])
    check = check_model_loading(saved(off, tmp_path), sample, expected, relative_tolerance=LOAD_RTOL, reference_model=model)
    assert check.max_abs_diff <= check.tolerance, "the outputs cannot see it"
    assert check.weights_equal is False and check.ok is False


def test_the_same_model_is_a_true_check_with_equal_weights(sample, tmp_path):
    model = scores_model()
    check = check_model_loading(
        saved(model, tmp_path), sample, model.predict(sample, verbose=0), relative_tolerance=LOAD_RTOL,
        reference_model=model)
    assert check.ok and check.weights_equal is True and check.max_abs_diff == 0.0
    assert check.argmax_agreement == 1.0 and check.relative_tolerance == LOAD_RTOL


def test_the_classification_check_is_the_absolute_one_it_always_was(sample, tmp_path):
    model = scores_model()
    path = saved(model, tmp_path)
    expected = model.predict(sample, verbose=0)
    assert validate_model_loading(path, sample, expected) is True
    assert check_model_loading(path, sample, expected + 1e-3).ok is False
    # numpy's own allclose: its default rtol of 1e-5 of the reloaded value is part of the old verdict.
    assert check_model_loading(path, sample, expected * (1 + 5e-6)).ok is True


def test_a_dict_of_outputs_and_a_shape_mismatch_are_verdicts_not_crashes(sample, tmp_path):
    model = scores_model()
    path = saved(model, tmp_path)
    expected = model.predict(sample, verbose=0)
    assert check_model_loading(path, sample, {"a": expected}).ok is False
    assert check_model_loading(path, sample, expected[:, :3]).ok is False
    assert check_model_loading(path, sample, expected[:, :3]).max_abs_diff is None


# ---------------------------------------------------------------------
# One call path (D-029)
# ---------------------------------------------------------------------

class SpyOnPredict:
    """Wraps ``load_model`` so the reloaded model reports the flags it has WHEN it predicts."""

    def __init__(self, monkeypatch, reloaded_jit):
        self.seen = []
        real = keras.models.load_model

        def load(path, **kwargs):
            loaded = real(path, **kwargs)
            loaded.jit_compile = reloaded_jit  # what Keras resolves for a model that was never compiled, forced
            real_predict = loaded.predict

            def predict(*args, **kw):
                self.seen.append((loaded.jit_compile, loaded.run_eagerly))
                return real_predict(*args, **kw)

            loaded.predict = predict
            return loaded

        monkeypatch.setattr(keras.models, "load_model", load)


@pytest.mark.parametrize("model_jit", [False, True])
def test_the_reloaded_model_predicts_with_the_in_memory_models_jit_compile(model_jit, sample, tmp_path, monkeypatch):
    """AccUNet compiles with ``jit_compile=False`` and ``load_model(compile=False)`` leaves ``auto`` (XLA on a GPU)."""
    model = scores_model()
    model.compile(loss="mse", jit_compile=model_jit)
    assert model.jit_compile is model_jit
    spy = SpyOnPredict(monkeypatch, reloaded_jit=not model_jit)  # the reloaded one starts on the OTHER setting
    check = check_model_loading(
        saved(model, tmp_path), sample, model.predict(sample, verbose=0), relative_tolerance=LOAD_RTOL, reference_model=model)
    assert spy.seen == [(model_jit, False)], "the flag must be the model's at the moment of the predict"
    assert check.jit_compile is model_jit and check.as_record()["jit_compile"] is model_jit and check.ok


def test_the_reloaded_model_runs_eagerly_when_the_in_memory_model_does(sample, tmp_path, monkeypatch):
    model = scores_model()
    model.compile(loss="mse", run_eagerly=True)
    spy = SpyOnPredict(monkeypatch, reloaded_jit=False)
    check_model_loading(
        saved(model, tmp_path), sample, model.predict(sample, verbose=0), relative_tolerance=LOAD_RTOL, reference_model=model)
    assert spy.seen == [(False, True)]


def test_without_a_reference_model_nothing_of_the_reloaded_model_is_touched(sample, tmp_path, monkeypatch):
    """The classification check (and a caller with no model in memory) keeps Keras' own resolution."""
    model = scores_model()
    spy = SpyOnPredict(monkeypatch, reloaded_jit=True)
    check = check_model_loading(saved(model, tmp_path), sample, model.predict(sample, verbose=0))
    assert spy.seen == [(True, False)] and check.jit_compile is None


def test_the_run_passes_its_own_model_as_the_reference(tmp_path, monkeypatch):
    """``finalize_last_epoch(exact_reload=True)`` hands the model to the check, so the path is the model's."""
    model = scores_model()
    model.compile(loss="mse", jit_compile=False)
    spy = SpyOnPredict(monkeypatch, reloaded_jit=True)
    detail = {}
    _, ok = finalize(model, tmp_path, exact_reload=True, load_detail=detail)
    assert ok is True and spy.seen == [(False, False)] and detail["jit_compile"] is False


# ---------------------------------------------------------------------
# TF32
# ---------------------------------------------------------------------

def test_tf32_is_a_gpu_with_tf32_execution_on_and_nothing_else(monkeypatch):
    for enabled, gpus, wanted in ((True, ["gpu"], True), (False, ["gpu"], False), (True, [], False), (False, [], False)):
        monkeypatch.setattr(tf.config.experimental, "tensor_float_32_execution_enabled", lambda e=enabled: e)
        monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind, g=gpus: g)
        assert tf32_active() is wanted


def test_the_tf32_state_is_recorded_in_the_check(sample, tmp_path, monkeypatch):
    model = scores_model()
    path, expected = saved(model, tmp_path), model.predict(sample, verbose=0)
    monkeypatch.setattr(evaluation, "tf32_active", lambda: True)
    assert check_model_loading(path, sample, expected).tf32 is True
    monkeypatch.setattr(evaluation, "tf32_active", lambda: False)
    assert check_model_loading(path, sample, expected).tf32 is False


# ---------------------------------------------------------------------
# finalize_last_epoch and the summary
# ---------------------------------------------------------------------

def finalize(model, run_dir, **kwargs):
    run_dir.mkdir(parents=True, exist_ok=True)
    data = SimpleNamespace(test_ds=object(), select_ds=None, x_fig=np.random.default_rng(2).standard_normal((20, 6)).astype("float32"))
    last = SimpleNamespace(weights=model.get_weights())
    return classification_run.finalize_last_epoch(
        model, last, data, run_dir, SimpleNamespace(history={}), 1, lambda m, d: {"loss": 0.0}, **kwargs)


def test_finalize_last_epoch_fills_the_detail_for_the_exact_check_only(tmp_path):
    model = scores_model()
    detail = {}
    _, ok = finalize(model, tmp_path / "a", exact_reload=True, load_detail=detail)
    assert ok is True
    assert detail["weights_equal"] is True and detail["max_abs_diff"] == 0.0 and detail["tf32"] is False
    assert detail["relative_tolerance"] == LOAD_RTOL and detail["tolerance"] > 1e-5
    legacy = {}
    _, ok = finalize(scores_model(), tmp_path / "b", load_detail=legacy)
    assert ok is True and legacy == {}, "the classification path asks for nothing and gets nothing"


def test_a_run_under_tf32_gets_the_same_tolerance_and_says_it_was_under_tf32(tmp_path, monkeypatch):
    """D-029: TF32 is recorded, not a tolerance (D-088's 0.07 hid a difference of call paths)."""
    monkeypatch.setattr(evaluation, "tf32_active", lambda: True)
    detail = {}
    finalize(scores_model(), tmp_path, exact_reload=True, load_detail=detail)
    assert detail["relative_tolerance"] == LOAD_RTOL and detail["tf32"] is True
    assert not hasattr(evaluation, "LOAD_RTOL_TF32")


def test_a_wrong_final_file_is_reported_false_by_the_exact_check(tmp_path, monkeypatch):
    model, other = scores_model(0), scores_model(1)
    real = keras.models.load_model
    monkeypatch.setattr(keras.models, "load_model", lambda path, **kw: other)  # the file holds other weights
    detail = {}
    _, ok = finalize(model, tmp_path, exact_reload=True, load_detail=detail)
    assert ok is False and detail["weights_equal"] is False and detail["max_abs_diff"] > detail["tolerance"]
    monkeypatch.setattr(keras.models, "load_model", real)


def summary_of(load_check, load_detail):
    hist = {"loss": [1.0, 0.5], "val_loss": [1.1, 0.6], "lr": [1e-3, 1e-4]}
    return classification_run.base_summary(
        {"model": "m"}, hist, 2, 2, 2, [], None, None, None, 0.0, [1.0, 1.0], 2.0, {}, load_check, None, {"status": "success"},
        notes=["first"], load_detail=load_detail)


DETAIL = {"ok": False, "max_abs_diff": 0.308, "scale": 220.0, "tolerance": 0.22, "relative_tolerance": 1e-3,
          "weights_equal": True, "argmax_agreement": 0.9985, "tf32": True, "jit_compile": False}


def test_a_false_check_is_never_silent():
    summary = summary_of(False, DETAIL)
    assert summary["status"] == classification_run.STATUS_RELOAD_UNVERIFIED == "ok_reload_unverified"
    assert summary["model_loading_validated"] is False and summary["model_loading_check"] == DETAIL
    note = summary["notes"][-1]
    assert summary["notes"][0] == "first"
    for part in ("model_loading_validated", "ok_reload_unverified", "0.308", "0.22", "TF32 was on",
                 "reloaded weights are equal to", "0.9985"):
        assert part in note, part


def test_a_false_check_without_detail_still_gets_a_note_and_the_status():
    summary = summary_of(False, None)
    assert summary["status"] == "ok_reload_unverified" and "model_loading_check" not in summary
    assert "is false" in summary["notes"][-1]


def test_a_true_check_is_plain_ok_with_no_added_note_and_a_null_one_is_not_a_failure():
    true = summary_of(True, {**DETAIL, "ok": True})
    assert true["status"] == "ok" and true["notes"] == ["first"] and true["model_loading_check"]["ok"] is True
    null = summary_of(None, None)
    assert null["status"] == "ok" and null["notes"] == ["first"] and "model_loading_check" not in null
