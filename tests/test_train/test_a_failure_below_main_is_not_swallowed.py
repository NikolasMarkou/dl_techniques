"""Failures one call below ``main`` that a broad ``except`` used to swallow now fail the run.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.2 (D-183). The census
in ``test_a_broad_handler_outside_main_is_classified.py`` keeps the handlers from
coming back; these tests pin what each fixed site now does, written before the fix.
Every write goes under ``tmp_path``.
"""

from pathlib import Path

import keras
import numpy as np
import pytest


class _DiskFull(OSError):
    """A stand-in write failure."""


def _tiny_model():
    return keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2)])


def _failing_save(fail_paths):
    def save(path, *args, **kwargs):
        if any(str(path).endswith(name) for name in fail_paths):
            raise _DiskFull(f"no space left for {path}")
        Path(path).write_bytes(b"model")
    return save


# ---------------------------------------------------------------------
# train.common.step_checkpoint.StepCheckpointCallback
# ---------------------------------------------------------------------

def _step_callback(tmp_path, model):
    from train.common.step_checkpoint import StepCheckpointCallback

    callback = StepCheckpointCallback(
        save_dir=str(tmp_path), save_every_steps=1, analyze_every_steps=0,
        log_every_steps=1000, plot_every_steps=0, max_checkpoints=10)
    callback.set_model(model)
    return callback


def test_a_failed_final_checkpoint_raises(tmp_path):
    model = _tiny_model()
    model.save = _failing_save(["final.keras"])
    callback = _step_callback(tmp_path, model)
    with pytest.raises(_DiskFull, match="final.keras"):
        callback.on_train_end()


def test_a_failed_step_checkpoint_fails_the_run_after_the_final_save(tmp_path):
    model = _tiny_model()
    model.save = _failing_save(["step_0000002.keras"])
    callback = _step_callback(tmp_path, model)
    for batch in range(3):
        callback.on_train_batch_end(batch, {"loss": 1.0})
    checkpoints = tmp_path / "checkpoints"
    assert sorted(p.name for p in checkpoints.glob("step_*.keras")) == [
        "step_0000001.keras", "step_0000003.keras"]
    with pytest.raises(RuntimeError, match="step_0000002.keras"):
        callback.on_train_end()
    assert (checkpoints / "final.keras").is_file()


def test_step_checkpoints_that_all_save_end_the_run_cleanly(tmp_path):
    model = _tiny_model()
    model.save = _failing_save([])
    callback = _step_callback(tmp_path, model)
    for batch in range(2):
        callback.on_train_batch_end(batch, {"loss": 1.0})
    callback.on_train_end()
    assert (tmp_path / "checkpoints" / "final.keras").is_file()


# ---------------------------------------------------------------------
# train.common.datasets.collect_image_paths
# ---------------------------------------------------------------------

def test_a_directory_scan_that_fails_raises(monkeypatch, tmp_path):
    from train.common.datasets import collect_image_paths

    (tmp_path / "a.png").write_bytes(b"x")

    def unreadable(self, pattern):
        raise PermissionError(f"cannot list {self}")

    monkeypatch.setattr(Path, "rglob", unreadable)
    with pytest.raises(PermissionError, match="cannot list"):
        collect_image_paths([str(tmp_path)])


def test_a_directory_scan_finds_its_images(tmp_path):
    from train.common.datasets import collect_image_paths

    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert collect_image_paths([str(tmp_path)]) == [str(tmp_path / "a.png")]


# ---------------------------------------------------------------------
# train.common.timeseries: the window generators, post-hoc metrics, ONNX
# ---------------------------------------------------------------------

class _OnePatternFails:
    """``generate_task_data`` for two patterns; ``bad`` raises (a generator bug)."""

    def generate_task_data(self, name):
        if name == "bad":
            raise ValueError(f"cannot generate {name}")
        return np.sin(np.arange(400, dtype=np.float32) / 7.0)[:, None]


def _processor():
    from train.common.timeseries import (
        BaseTimeSeriesTrainingConfig,
        WindowedTimeSeriesProcessor,
    )

    return WindowedTimeSeriesProcessor(
        BaseTimeSeriesTrainingConfig(), _OnePatternFails(), ["bad", "good"],
        context_len=16, horizon_len=4, patterns_to_mix=50, windows_per_pattern=1)


def test_a_pattern_that_fails_to_generate_fails_the_fixed_split():
    with pytest.raises(ValueError, match="cannot generate bad"):
        _processor()._generate_fixed_dataset("val", 4)


def test_a_pattern_that_fails_to_generate_fails_the_training_stream():
    with pytest.raises(ValueError, match="cannot generate bad"):
        next(_processor()._training_generator())


def test_a_pattern_that_fails_to_generate_fails_the_visualization_stream():
    with pytest.raises(ValueError, match="cannot generate bad"):
        list(_processor()._test_generator_raw())


def test_post_hoc_metrics_that_fail_raise():
    from types import SimpleNamespace

    from dl_techniques.models.time_series.forecast import ForecastMixin
    from train.common.timeseries import BaseTimeSeriesTrainer

    class Broken(ForecastMixin):
        def _forecast(self, x, **kwargs):
            raise RuntimeError("forecast head failed")

    trainer = SimpleNamespace(model=Broken(), config=SimpleNamespace(quantile_levels=None))
    pipeline = {"test_data_raw": (np.zeros((2, 16, 1)), np.zeros((2, 4, 1)))}
    with pytest.raises(RuntimeError, match="forecast head failed"):
        BaseTimeSeriesTrainer._compute_post_hoc_metrics(trainer, pipeline)
    # The documented skips stay skips: no test data, or a model with no forecast.
    assert BaseTimeSeriesTrainer._compute_post_hoc_metrics(trainer, {}) == {}
    trainer.model = object()
    assert BaseTimeSeriesTrainer._compute_post_hoc_metrics(trainer, pipeline) == {}


def test_a_requested_onnx_export_that_fails_writes_results_then_raises(monkeypatch, tmp_path):
    import dataclasses
    import json

    from .test_timeseries_base import _StubConfig, _StubTrainer
    from dl_techniques.datasets.time_series import TimeSeriesGeneratorConfig

    def export(self, *args, **kwargs):
        raise RuntimeError("no ONNX converter (test)")

    @dataclasses.dataclass
    class OnnxConfig(_StubConfig):
        input_length: int = 16  # sizes the export's input signature

    def run(config):
        trainer = _StubTrainer(config, TimeSeriesGeneratorConfig(
            n_samples=2000, random_seed=42, default_noise_level=0.1))
        trainer.run_experiment()

    monkeypatch.setattr(keras.Model, "export", export)
    with pytest.raises(RuntimeError, match="ONNX export failed: RuntimeError: no ONNX converter"):
        run(OnnxConfig(result_dir=str(tmp_path / "a"), export_onnx=True))
    (results,) = (tmp_path / "a").glob("*/results.json")
    loaded = json.loads(results.read_text())
    assert loaded["onnx_path"] is None and "test_metrics" in loaded
    # A requested export the config cannot size is a failure too, not a log line.
    with pytest.raises(RuntimeError, match="ONNX export failed: ValueError: .*input_length"):
        run(dataclasses.replace(_StubConfig(result_dir=str(tmp_path / "b")), export_onnx=True))
    assert list((tmp_path / "b").glob("*/results.json"))


# ---------------------------------------------------------------------
# Data and inference helpers: the bert streams, THERA, DocRes
# ---------------------------------------------------------------------

_LONG_TEXT = "an english sentence long enough to pass the length filter. " * 4


def _stream_that_breaks():
    """A stream that yields one record and then fails, as a dropped connection does."""
    yield {"text": _LONG_TEXT}
    raise ConnectionError("stream dropped (test)")


@pytest.mark.parametrize("module_name, function_name", [
    ("train.bert.wikipedia.pretrain", "get_dataset_generator"),
    ("train.bert.wikipedia.pretrain_english", "get_english_dataset_generator"),
])
def test_a_bert_corpus_stream_that_fails_raises(monkeypatch, module_name, function_name):
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.datasets, "load_dataset", lambda *a, **k: object())
    monkeypatch.setattr(module.datasets, "interleave_datasets",
                        lambda *a, **k: _stream_that_breaks())
    stream = getattr(module, function_name)(module.PretrainConfig())
    assert next(stream) == _LONG_TEXT
    with pytest.raises(ConnectionError, match="stream dropped"):
        next(stream)


def test_the_thera_eval_takes_its_statistics_from_the_trainer_only(monkeypatch):
    import importlib
    import sys

    from train.thera import train_thera

    monkeypatch.delitem(sys.modules, "train.thera.eval_thera", raising=False)
    monkeypatch.setitem(sys.modules, "train.thera.train_thera", None)
    with pytest.raises(ImportError):
        importlib.import_module("train.thera.eval_thera")
    monkeypatch.setitem(sys.modules, "train.thera.train_thera", train_thera)
    monkeypatch.delitem(sys.modules, "train.thera.eval_thera", raising=False)
    eval_thera = importlib.import_module("train.thera.eval_thera")
    assert eval_thera.THERA_MEAN is train_thera.THERA_MEAN


def _doc_res_main(monkeypatch, tmp_path, reports):
    from train.doc_res import prepare_doc_res_data as module

    monkeypatch.setattr(module, "stage_all", lambda *a, **k: reports)
    return module.main(["--root", str(tmp_path), "--datasets", reports[0].dataset])


def test_a_doc_res_staging_that_failed_returns_1(monkeypatch, tmp_path):
    from train.doc_res.prepare_doc_res_data import (
        ArchiveResult, DatasetReport, SidecarResult, dataset_names)

    name = dataset_names()[0]
    archive = ArchiveResult(filename="a.zip", url="http://x", status="failed",
                            detail="OSError: test")
    assert _doc_res_main(monkeypatch, tmp_path, [DatasetReport(
        dataset=name, task="t", status="staged", archives=(archive,))]) == 1
    assert _doc_res_main(monkeypatch, tmp_path, [DatasetReport(
        dataset=name, task="t", status="refused", detail="disk")]) == 1
    sidecars = SidecarResult(dataset=name, task="t", status="ok", failed=2)
    assert _doc_res_main(monkeypatch, tmp_path, [DatasetReport(
        dataset=name, task="t", status="staged", sidecars=sidecars)]) == 1
    staged = ArchiveResult(filename="a.zip", url="http://x", status="staged")
    assert _doc_res_main(monkeypatch, tmp_path, [DatasetReport(
        dataset=name, task="t", status="staged", archives=(staged,))]) == 0


# ---------------------------------------------------------------------
# The multi-seed sweep launchers
# ---------------------------------------------------------------------

def _run(status):
    return lambda spec, timeout_s, env: {"status": status, "wall_s": 0.0, "stderr_tail": ""}


def test_a_logic_multiseed_sweep_with_a_failed_seed_raises_after_its_summary(
        monkeypatch, tmp_path):
    from train.logic import multiseed_sweep as module

    monkeypatch.setattr(module, "run_one", _run("exit_1"))
    monkeypatch.setattr("sys.argv", ["multiseed_sweep", "--seeds", "0", "--experiments",
                                     "e1", "--out-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="seed_0: exit_1"):
        module.main()
    assert list(tmp_path.glob("*.md")), "the summary was not written"


def test_a_logic_multiseed_sweep_whose_seeds_all_pass_returns_0(monkeypatch, tmp_path):
    import pandas as pd

    from train.logic import multiseed_sweep as module

    monkeypatch.setattr(module, "run_one", _run("ok"))
    monkeypatch.setattr(module, "collect_csvs", lambda specs, experiment, unreadable: pd.DataFrame())
    monkeypatch.setattr("sys.argv", ["multiseed_sweep", "--seeds", "0", "--experiments",
                                     "e1", "--out-dir", str(tmp_path)])
    assert module.main() == 0


def test_a_mothnet_multiseed_sweep_with_a_failed_seed_raises_after_its_summary(
        monkeypatch, tmp_path):
    from train.mothnet import multiseed_sweep as module

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # its --gpu 1 default
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)  # it writes under REPO_ROOT/results
    monkeypatch.setattr(module, "run_one", _run("exit_1"))
    with pytest.raises(RuntimeError, match="seed_1: run exit_1"):
        module.main(["--seeds", "1"])
    (summary_dir,) = (tmp_path / "results").glob("mothnet_mb_units_sweep_*")
    assert (summary_dir / "raw_results.csv").is_file()


def test_a_mothnet_multiseed_sweep_whose_seeds_all_pass_returns_0(monkeypatch, tmp_path):
    from train.mothnet import multiseed_sweep as module

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "run_one", _run("ok"))
    for spec in module.build_run_specs(seeds=[1, 2], mb_units_baseline=4000,
                                       mb_units_candidate=8000):
        log = tmp_path / "results" / spec.experiment_name / "training_log.csv"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("epoch,val_accuracy\n0,0.5\n")
    assert module.main(["--seeds", "1,2", "--mb-units-baseline", "4000",
                        "--mb-units-candidate", "8000"]) == 0
