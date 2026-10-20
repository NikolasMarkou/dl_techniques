"""The step log has one schema: the step plus every numeric value Keras logs.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.4 (D-195). Until
then ``StepCheckpointCallback`` also took ``csv_fields``, a fixed schema with
``epoch`` and ``lr`` columns and an epoch-end row. Nothing in ``src`` passed
it, and the step log's one reader, ``step_plots.plot_step_metrics``, needs only
``step``, so the branch was deleted rather than kept for no caller. Its
``_global_step`` back-compat alias had no caller either and went with it.
"""

import csv

import keras
import pytest

from train.common.step_checkpoint import StepCheckpointCallback


def _callback(tmp_path, **kwargs):
    return StepCheckpointCallback(
        save_dir=str(tmp_path), save_every_steps=1000, analyze_every_steps=0,
        log_every_steps=1, plot_every_steps=0, **kwargs)


def test_the_step_callback_takes_no_fixed_schema(tmp_path):
    with pytest.raises(TypeError, match="csv_fields"):
        _callback(tmp_path, csv_fields=("step", "epoch", "lr", "loss"))


def test_the_step_log_holds_the_step_and_the_logged_values(tmp_path):
    """Control: the one schema is written as before."""
    callback = _callback(tmp_path)
    callback.set_model(keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)]))
    for batch in range(2):
        callback.on_train_batch_end(batch, {"loss": 1.5 - batch, "note": "skipped"})
    callback.close()
    with open(tmp_path / "training_log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"step": "1", "loss": "1.5"}, {"step": "2", "loss": "0.5"}]


def test_the_step_callback_keeps_no_back_compat_step_alias():
    """``_global_step`` was an alias of ``global_step`` for call sites that no longer exist."""
    assert not hasattr(StepCheckpointCallback, "_global_step")
    assert isinstance(StepCheckpointCallback.global_step, property)
