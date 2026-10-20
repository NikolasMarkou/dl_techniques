"""Failures one call below ``main`` that a broad ``except`` used to swallow now fail the run.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.2 (D-183). The census
in ``test_a_broad_handler_outside_main_is_classified.py`` keeps the handlers from
coming back; these tests pin what each fixed site now does, written before the fix.
Every write goes under ``tmp_path``.
"""

from pathlib import Path

import keras
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

