"""The reload check of a segmentation run is honest on a correct file and blind to no wrong one (D-088).

Review-iter-3 concern 4. A correct AccUNet ``final_model.keras`` was reported ``model_loading_validated:
false`` on the RTX 4090 (``max_diff=0.308`` on scores of scale 220) while ``status`` was ``ok`` and
``success`` true, with no note: the absolute 1e-5 of ``validate_model_loading`` cannot hold on scores that
large, and under TF32 (TensorFlow's default on Ampere and later) two correct executions of one set of
weights differ by 3% of the largest score. Measured first on the GPU with trained weights
(``load_check_evidence.json``, a copy of ``scratch_step33/logs/tf32_*.json``):

========= ===== ================ ===================== ===========================
family    TF32  correct file     wrong file, smallest   wrong files below LOAD_RTOL*
========= ===== ================ ===================== ===========================
AccUNet   off   9.6e-5           0.031 (x1.01 tensor)   none
AccUNet   on    3.2e-2           0.049                  none below 0.07 but one
ConvUNext off   0.0              3.4e-3 (noise 1e-3)    none
ConvUNext on    0.0              3.4e-3                 noise 1e-3, noise 1e-2, x1.01
========= ===== ================ ===================== ===========================

(relative differences of the scores, to the largest score; ``LOAD_RTOL*`` is ``LOAD_RTOL`` without TF32
and ``LOAD_RTOL_TF32`` with it.) So the output tolerance is relative to the scale, and under TF32 it
cannot see a wrong file below 7% of the scale: the weights of the reloaded model are compared EXACTLY as
well, which no execution noise touches.

What is claimed, on small models on the CPU (the evidence holds the GPU numbers):

* the constants sit where the evidence puts them: ``LOAD_RTOL`` at least 10 times the worst correct
  file without TF32 and below every wrong file; ``LOAD_RTOL_TF32`` at least twice the worst correct
  file with TF32;
* scores of scale 200 that differ by 0.02% of it are a correct file for the relative check and a wrong one
  for the absolute check (the review's defect), and a wrong file (other weights) fails the relative one;
* a file whose one kernel is 1% off passes the TF32 output tolerance and fails on its weights, and the
  weights check is the only thing that sees it;
* ``tf32_active`` is a GPU with TF32 on, nothing else;
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
    LOAD_RTOL_TF32,
    check_model_loading,
    tf32_active,
    validate_model_loading,
)

EVIDENCE = json.loads((Path(__file__).resolve().parent / "load_check_evidence.json").read_text())["rows"]


# ---------------------------------------------------------------------
# The constants against the measurement
# ---------------------------------------------------------------------

def rows(tf32):
    return [row for row in EVIDENCE if row["tf32"] is tf32]


def wrong_rels(row):
    return {name: entry["min_rel"] for name, entry in row["wrong_files"].items() if entry["min_rel"] is not None}


def test_the_relative_tolerance_without_tf32_is_ten_times_the_worst_correct_file_and_under_every_wrong_one():
    worst = max(row["correct_file_worst_rel"] for row in rows(False))
    assert worst > 0 and LOAD_RTOL >= 10 * worst
    for row in rows(False):
        for name, rel in wrong_rels(row).items():
            assert rel > LOAD_RTOL, f"{row['family']} {name}: {rel} is under the tolerance"


def test_the_relative_tolerance_with_tf32_is_twice_the_worst_correct_file():
    worst = max(row["correct_file_worst_rel"] for row in rows(True))
    assert LOAD_RTOL_TF32 >= 2 * worst
    # The TF32 noise is what the evidence says it is: hundreds of times the noise without it.
    without = max(row["correct_file_worst_rel"] for row in rows(False))
    assert worst > 100 * without
    # And the tolerance is blind to a wrong file below it, which the weights comparison is for.
    blind = [(row["family"], name) for row in rows(True) for name, rel in wrong_rels(row).items() if rel < LOAD_RTOL_TF32]
    assert blind, "if no wrong file is under the TF32 tolerance the weights comparison would be redundant"


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
    check = check_model_loading(path, sample, nudged, relative_tolerance=LOAD_RTOL, reference_weights=model.get_weights())
    assert check.ok and check.weights_equal is True
    seen = float(np.abs(nudged).max())  # the scale is the EXPECTED output's
    assert 0 < check.max_abs_diff <= check.tolerance and check.scale == pytest.approx(seen, rel=1e-6)
    assert check.tolerance == pytest.approx(1e-5 + LOAD_RTOL * seen, rel=1e-6)


def test_a_file_with_other_weights_fails_the_relative_check_by_its_outputs(sample, tmp_path):
    model, other = scores_model(0), scores_model(1)
    expected = model.predict(sample, verbose=0)
    check = check_model_loading(
        saved(other, tmp_path), sample, expected, relative_tolerance=LOAD_RTOL, reference_weights=model.get_weights())
    assert check.ok is False and check.max_abs_diff > check.tolerance and check.weights_equal is False


def test_a_one_percent_error_in_one_kernel_passes_the_tf32_outputs_and_fails_the_weights(sample, tmp_path):
    model = scores_model()
    expected = model.predict(sample, verbose=0)
    off = scores_model()
    kernel, bias = off.get_weights()
    off.set_weights([kernel * 1.01, bias])
    path = saved(off, tmp_path)
    outputs_only = check_model_loading(path, sample, expected, relative_tolerance=LOAD_RTOL_TF32)
    assert outputs_only.ok is True and outputs_only.weights_equal is None, "blind: 1% is under the 7% tolerance"
    exact = check_model_loading(
        path, sample, expected, relative_tolerance=LOAD_RTOL_TF32, reference_weights=model.get_weights())
    assert exact.ok is False and exact.weights_equal is False
    assert check_model_loading(
        path, sample, expected, relative_tolerance=LOAD_RTOL, reference_weights=model.get_weights()).ok is False


def test_the_same_model_is_a_true_check_with_equal_weights(sample, tmp_path):
    model = scores_model()
    check = check_model_loading(
        saved(model, tmp_path), sample, model.predict(sample, verbose=0), relative_tolerance=LOAD_RTOL,
        reference_weights=model.get_weights())
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


def test_a_run_under_tf32_gets_the_tf32_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(classification_run, "tf32_active", lambda: True)
    detail = {}
    finalize(scores_model(), tmp_path, exact_reload=True, load_detail=detail)
    assert detail["relative_tolerance"] == LOAD_RTOL_TF32 > LOAD_RTOL


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
          "weights_equal": True, "argmax_agreement": 0.9985, "tf32": True}


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
