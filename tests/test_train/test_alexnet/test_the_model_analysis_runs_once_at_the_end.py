"""AlexNet's weights + spectral analysis runs once, at the end, by default.

Guard for plan-2026-10-14T062616-81938e81 step 13.6 (d) (D-114). Measured at
e259d9977: ``create_callbacks`` took the shared ``create_callbacks`` default
``include_analyzer=True`` with ``analyzer_epoch_frequency=1``, so an
``EpochAnalyzerCallback`` ran after every epoch: 296 to 305 s of analysis
against 7.5 s of training and validation per Imagenette epoch at 227 px (RTX
4070; about 230 s against 7 s on the RTX 4090 in Step 13.5). The per-epoch
analysis is now opt-in (``--epoch-analysis-every N``, default 0), and one
end-of-run analysis of the trained model runs through the shared
``train.common.run_summary.run_data_free_analysis``; a failed analysis fails
the run (the D-102 / D-105 rule).
"""

import numpy as np
import pytest

from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

import train.common.classification_data as classification_data
from train.alexnet import train_alexnet as module


def _analyzers(callbacks):
    return [cb for cb in callbacks if isinstance(cb, EpochAnalyzerCallback)]


def test_no_per_epoch_analysis_by_default(tmp_path):
    config = module.TrainingConfig()
    assert config.epoch_analysis_every == 0
    callbacks, _ = module.create_callbacks(config, tmp_path)
    assert _analyzers(callbacks) == []


def test_the_per_epoch_analysis_is_opt_in(tmp_path):
    config = module.TrainingConfig(epoch_analysis_every=3)
    (analyzer,) = _analyzers(module.create_callbacks(config, tmp_path)[0])
    assert analyzer.epoch_frequency == 3


def test_the_flag_reaches_the_config(monkeypatch):
    seen = {}

    def capture(config, gpu_id=None):
        seen["config"] = config
        raise SystemExit(0)

    monkeypatch.setattr(module, "train_alexnet", capture)
    monkeypatch.setattr(module.sys, "argv", ["train_alexnet", "--epoch-analysis-every", "4"])
    with pytest.raises(SystemExit):
        module.main()
    assert seen["config"].epoch_analysis_every == 4
    assert module.parse_arguments([]).epoch_analysis_every == 0


def _tiny_cifar(name="cifar10"):
    """40 train images (the 10% selection cut leaves 4 out) and 4 test images."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8)
    y = (np.arange(40) % 10).astype(np.int64)
    return (x, y), (x[:4], y[:4])


def _tiny_config(tmp_path):
    return module.TrainingConfig(
        dataset="cifar10", image_size=64, batch_size=4, epochs=1,
        augment_data=False, num_classes=10, learning_rate=0.01,
        output_dir=str(tmp_path), experiment_name="run",
    )


def _spy_analysis(monkeypatch, status="ok"):
    """Record what ``train_alexnet`` hands the shared analysis; return ``status``.

    The analysis itself (about 3 minutes on this model even at 64 px) is the
    shared helper's, covered by its own tests and exercised by the smoke run.
    """
    calls = []

    def spy(model, sample, labels, history, model_name, run_dir, enabled=True):
        calls.append({"model": model, "sample": sample, "labels": labels,
                      "history": history, "run_dir": run_dir})
        return {"status": status, "analyzers": [], "path": None,
                "error": None if status == "ok" else "analysis broke"}

    monkeypatch.setattr(module, "run_data_free_analysis", spy)
    return calls


def test_one_analysis_of_the_trained_model_runs_after_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(classification_data, "_load_raw", _tiny_cifar)
    calls = _spy_analysis(monkeypatch)
    result = module.train_alexnet(_tiny_config(tmp_path))
    (call,) = calls
    assert call["model"] is result["model"]
    assert call["history"] is result["history"]
    assert call["run_dir"] == tmp_path / "run"
    assert call["sample"].shape == (4, 64, 64, 3) and len(call["labels"]) == 4
    assert not (tmp_path / "run" / "epoch_analysis").exists()


def test_a_failed_analysis_fails_the_run(monkeypatch, tmp_path):
    monkeypatch.setattr(classification_data, "_load_raw", _tiny_cifar)
    _spy_analysis(monkeypatch, status="error")
    with pytest.raises(RuntimeError, match="analysis broke"):
        module.train_alexnet(_tiny_config(tmp_path))
