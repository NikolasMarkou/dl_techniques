"""Run-health contract of ``train_model``: divergence, ``run.log``, derived LR/early-stop lines.

Review iteration 2 (C5): a real ``deep`` + ``--k 3`` run warned at the initial-loss
guard and then hit ``Invalid loss, terminating training`` at batch 2, yet the trainer
went on to evaluate NaN weights and wrote a summary with ``NaN`` tokens (not JSON)
whose ``best_epoch`` pointed at the NaN epoch (``np.argmin([nan]) == 0``). The
contract pinned here:

- any non-finite epoch ``loss`` / ``val_loss`` -> ``results_summary.json`` with
  ``status: "diverged"``, ``best_epoch: null``, non-finite values as ``null`` (strict
  JSON), NO ``final_model.keras``, and only THEN a ``RuntimeError``;
- ``<run_dir>/run.log`` exists for every run, holds the ``dl`` logger lines plus the
  LR-reduction and early-stop lines derived from the history, and its file handler is
  removed when ``train_model`` returns OR raises.

Two arms per contract: a stubbed ``fit`` (deterministic, always runs) and one real
tiny ``deep`` + ``--k 3`` run on the cached real MNIST (skipped when it is absent).
Everything is written under ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.power_mlp.train_power_mlp as tpm  # noqa: E402

REAL_MNIST_CACHED = (Path.home() / ".keras" / "datasets" / "mnist.npz").exists()


def _strict(text: str):
    def refuse(token: str):
        raise ValueError(f"non-strict JSON constant {token!r}")

    return json.loads(text, parse_constant=refuse)


def _tiny_load_dataset(name, *args, **kwargs):
    """A 300 / 100 synthetic MNIST-shaped split (28x28x3 in [0, 1])."""
    assert name == "mnist", name
    rng = np.random.default_rng(11)

    def split(n):
        y = (np.arange(n) % 10).astype(np.uint8)
        x = rng.random((n, 28, 28, 1), dtype=np.float32) * 0.5
        return np.repeat(x, 3, axis=-1), y

    return split(300), split(100), (28, 28, 3), 10


def _history(**series):
    """A ``keras.callbacks.History`` carrying exactly ``series``."""
    history = keras.callbacks.History()
    history.history = {k: list(v) for k, v in series.items()}
    return history


def _stub_fit(monkeypatch, history):
    """Replace ``keras.Model.fit`` (call-through is pointless: we want a chosen history)."""
    monkeypatch.setattr(keras.Model, "fit", lambda self, *a, **k: history)


def _config(tmp_path, name, **overrides) -> tpm.TrainingConfig:
    settings = dict(epochs=5, batch_size=64, seed=1, output_dir=str(tmp_path), experiment_name=name)
    settings.update(overrides)
    return tpm.TrainingConfig(**settings)


def _file_handlers():
    return [h for h in logging.getLogger("dl").handlers if isinstance(h, logging.FileHandler)]


# ---------------------------------------------------------------------
# Divergence (stub arm)
# ---------------------------------------------------------------------


def test_a_non_finite_history_writes_a_diverged_summary_then_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tpm, "load_dataset", _tiny_load_dataset)
    nan = float("nan")
    _stub_fit(monkeypatch, _history(
        loss=[nan], val_loss=[nan], accuracy=[0.1], val_accuracy=[0.1], lr=[3e-4]))
    run_dir = tmp_path / "diverged_stub"

    with pytest.raises(RuntimeError, match="diverged"):
        tpm.train_model(_config(tmp_path, "diverged_stub", batch_normalization=False))

    summary = _strict((run_dir / "results_summary.json").read_text())
    assert summary["status"] == "diverged" == tpm.STATUS_DIVERGED
    assert summary["best_epoch"] is None
    assert summary["non_finite_metrics"] == ["loss", "val_loss"]
    assert summary["history"]["loss"] == [None] and summary["history"]["val_loss"] == [None]
    assert summary["epochs_run"] == 1 and summary["epochs_requested"] == 5
    assert summary["initial_loss_mode"] == "inference"
    assert "test_metrics_final" not in summary, "no evaluation of NaN weights"
    assert not (run_dir / "final_model.keras").exists()
    assert not (run_dir / "visualizations" / "confusion_matrix.png").exists()
    assert not (run_dir / "model_analysis").exists()


def test_the_summary_records_the_batch_norm_sanity_mode_end_to_end(monkeypatch, tmp_path) -> None:
    """The mode ``_check_initial_loss`` returns reaches ``results_summary.json``
    (a batch-normalized run measures in training mode)."""
    monkeypatch.setattr(tpm, "load_dataset", _tiny_load_dataset)
    _stub_fit(monkeypatch, _history(loss=[float("nan")], val_loss=[float("nan")], lr=[3e-4]))

    with pytest.raises(RuntimeError):
        tpm.train_model(_config(tmp_path, "bn_mode", batch_normalization=True))

    summary = _strict((tmp_path / "bn_mode" / "results_summary.json").read_text())
    assert summary["initial_loss_mode"] == "training"
    assert summary["batch_normalization"] is True
    assert "Sanity evaluate BEFORE fit (training mode)" in (tmp_path / "bn_mode" / "run.log").read_text()


def test_a_diverged_run_still_leaves_its_run_log_and_no_handler(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tpm, "load_dataset", _tiny_load_dataset)
    _stub_fit(monkeypatch, _history(loss=[1.0, float("inf")], val_loss=[1.0, 1.0], lr=[3e-4, 3e-4]))

    with pytest.raises(RuntimeError):
        tpm.train_model(_config(tmp_path, "diverged_log"))

    text = (tmp_path / "diverged_log" / "run.log").read_text()
    assert "Run directory" in text and "Training diverged" in text
    assert _file_handlers() == []


# ---------------------------------------------------------------------
# run.log lifecycle, derived LR-reduction / early-stop lines (stub arm)
# ---------------------------------------------------------------------


def test_run_log_holds_the_derived_lr_and_early_stop_lines_and_the_summary_keys(
        monkeypatch, tmp_path) -> None:
    """Keras prints the reduction / early-stop messages to stdout, never to the logger,
    so the trainer derives them from the history: 3 of 5 epochs run, the rate halves
    at epoch 3."""
    monkeypatch.setattr(tpm, "load_dataset", _tiny_load_dataset)
    _stub_fit(monkeypatch, _history(
        loss=[2.0, 1.0, 0.8], val_loss=[2.0, 1.0, 0.9],
        accuracy=[0.3, 0.6, 0.7], val_accuracy=[0.3, 0.6, 0.7],
        lr=[3e-4, 3e-4, 1.5e-4]))

    summary = tpm.train_model(_config(tmp_path, "derived_lines", epochs=5))

    assert summary["status"] == "ok"
    assert summary["lr_reduction_epochs"] == [3]
    assert summary["epochs_run"] == 3 and summary["stopped_early"] is True
    assert summary["best_epoch"] == 3
    on_disk = _strict((tmp_path / "derived_lines" / "results_summary.json").read_text())
    assert on_disk["lr_reduction_epochs"] == [3] and on_disk["status"] == "ok"
    text = (tmp_path / "derived_lines" / "run.log").read_text()
    assert "ReduceLROnPlateau: learning rate 0.0003 -> 0.00015 from epoch 3" in text, text
    assert "EarlyStopping: stopped after epoch 3 of 5" in text, text
    assert _file_handlers() == [], "the handler must be gone after train_model RETURNS"


def test_the_run_log_handler_is_removed_when_train_model_raises_midway(
        monkeypatch, tmp_path) -> None:
    """The ``finally``: a crash after the handler was attached must not leave it on the
    ``dl`` logger (a later run in this process would write into this run's file)."""
    monkeypatch.setattr(tpm, "load_dataset", _tiny_load_dataset)

    def boom(*args, **kwargs):
        raise KeyError("prepare_data exploded")

    monkeypatch.setattr(tpm, "prepare_data", boom)
    with pytest.raises(KeyError):
        tpm.train_model(_config(tmp_path, "raises_midway"))

    assert "Run directory" in (tmp_path / "raises_midway" / "run.log").read_text()
    assert _file_handlers() == []


# ---------------------------------------------------------------------
# Real tiny deep + k=3 run (real MNIST)
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_MNIST_CACHED, reason="real MNIST is not cached; the suite never downloads")
def test_a_real_tiny_deep_k3_run_diverges_and_is_reported_as_diverged(tmp_path) -> None:
    """The exact failing configuration of the review: ``deep`` + ``--k 3`` at the shipped
    defaults warns at the guard (ratio ~1e6) and hits ``Invalid loss`` at batch 2."""
    # BN off: the review measured this failure without BN (the BN guard reads ~1.2 there).
    config = _config(
        tmp_path, "real_deep_k3", architecture="deep", k=3, epochs=2, batch_size=128,
        batch_normalization=False,
    )

    with pytest.raises(RuntimeError, match="diverged"):
        tpm.train_model(config)

    run_dir = tmp_path / "real_deep_k3"
    summary = _strict((run_dir / "results_summary.json").read_text())
    assert summary["status"] == "diverged"
    assert summary["best_epoch"] is None
    assert summary["init_scale_warning"] is True and summary["initial_loss_ratio"] > 10.0
    assert not (run_dir / "final_model.keras").exists()
    assert (run_dir / "run.log").is_file()
    assert _file_handlers() == []
