"""Unit tests for ``train.common.run_io``.

This module replaced three blocks that were copy-pasted across ~20 trainers, so
the contracts worth pinning are the ones a call site would silently depend on:

- the JSON dump is byte-identical to the inline block it replaced
- a failed write RAISES (plan-2026-10-14T062616-81938e81 D-174; until then it
  warned and returned None, so a run with its record missing exited 0). Every
  call site writes after the model save, so the raise never costs the weights
  (``test_a_history_write_follows_the_model_save.py``)
- ``prepare_run_dir`` both creates the directory and writes the config
- ``default_experiment_name`` drops empty fragments so a disabled optional
  suffix does not leave a doubled underscore
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from train.common.run_io import (
    TIMESTAMP_FORMAT,
    default_experiment_name,
    prepare_run_dir,
    run_timestamp,
    save_training_history_json,
)


class _FakeHistory:
    """Stand-in for ``keras.callbacks.History`` (only ``.history`` is used)."""

    def __init__(self, history):
        self.history = history


@dataclass
class _Config:
    output_dir: str
    experiment_name: str
    learning_rate: float = 1e-3
    seed: Optional[int] = None


# ---------------------------------------------------------------------
# run_timestamp / default_experiment_name
# ---------------------------------------------------------------------

class TestExperimentName:

    def test_run_timestamp_format(self):
        stamp = run_timestamp()
        assert re.fullmatch(r"\d{8}_\d{6}", stamp)
        assert TIMESTAMP_FORMAT == "%Y%m%d_%H%M%S"

    def test_parts_are_joined_and_timestamped(self):
        name = default_experiment_name("vit", "cifar10", "small")
        assert re.fullmatch(r"vit_cifar10_small_\d{8}_\d{6}", name)

    def test_empty_and_none_parts_are_dropped(self):
        """A disabled optional fragment must not leave a doubled underscore."""
        name = default_experiment_name("resnet", "", None, "b0")
        assert "__" not in name
        assert re.fullmatch(r"resnet_b0_\d{8}_\d{6}", name)

    def test_non_string_parts_are_coerced(self):
        name = default_experiment_name("run", 7, 3.5)
        assert name.startswith("run_7_3.5_")

    def test_no_parts_yields_bare_timestamp(self):
        assert re.fullmatch(r"\d{8}_\d{6}", default_experiment_name())


# ---------------------------------------------------------------------
# prepare_run_dir
# ---------------------------------------------------------------------

class TestPrepareRunDir:

    def test_creates_directory_and_writes_config(self, tmp_path):
        config = _Config(output_dir=str(tmp_path), experiment_name="exp_a")

        run_dir = prepare_run_dir(config)

        assert run_dir == tmp_path / "exp_a"
        assert run_dir.is_dir()
        written = json.loads((run_dir / "config.json").read_text())
        assert written["experiment_name"] == "exp_a"
        assert written["learning_rate"] == 1e-3

    def test_nested_parents_are_created(self, tmp_path):
        config = _Config(output_dir=str(tmp_path / "a" / "b"), experiment_name="c")
        assert prepare_run_dir(config).is_dir()

    def test_is_idempotent(self, tmp_path):
        """Re-running must not raise on an existing directory."""
        config = _Config(output_dir=str(tmp_path), experiment_name="exp_a")
        first = prepare_run_dir(config)
        second = prepare_run_dir(config)
        assert first == second

    def test_explicit_output_dir_overrides_derivation(self, tmp_path):
        """Trainers that resolve the path themselves pass it in directly."""
        config = _Config(output_dir=str(tmp_path / "ignored"), experiment_name="ignored")
        explicit = tmp_path / "resolved" / "elsewhere"

        run_dir = prepare_run_dir(config, output_dir=explicit)

        assert run_dir == explicit
        assert run_dir.is_dir()
        assert not (tmp_path / "ignored").exists()

    def test_config_filename_is_configurable(self, tmp_path):
        config = _Config(output_dir=str(tmp_path), experiment_name="exp_a")
        run_dir = prepare_run_dir(config, config_filename="run_config.json")
        assert (run_dir / "run_config.json").exists()


# ---------------------------------------------------------------------
# save_training_history_json
# ---------------------------------------------------------------------

class TestSaveTrainingHistoryJson:

    @staticmethod
    def _history():
        return {
            "loss": [np.float32(0.5), np.float64(0.25), 0.125],
            "val_accuracy": [np.float32(0.1), 0.2, np.float64(0.3)],
        }

    def test_byte_identical_to_the_inline_block_it_replaced(self, tmp_path):
        """Pins the exact serialization the ~20 call sites produced before."""
        history = self._history()

        # The inline block, verbatim.
        expected_path = tmp_path / "expected.json"
        history_dict = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        with open(expected_path, "w") as handle:
            json.dump(history_dict, handle, indent=2)

        save_training_history_json(
            _FakeHistory(history), tmp_path, filename="actual.json"
        )

        assert (tmp_path / "actual.json").read_bytes() == expected_path.read_bytes()

    def test_accepts_a_raw_dict(self, tmp_path):
        save_training_history_json(self._history(), tmp_path, filename="d.json")
        assert json.loads((tmp_path / "d.json").read_text())["loss"] == [0.5, 0.25, 0.125]

    def test_numpy_scalars_become_plain_json_numbers(self, tmp_path):
        save_training_history_json(self._history(), tmp_path)
        raw = (tmp_path / "training_history.json").read_text()
        # Would appear as a quoted repr if float() coercion were dropped.
        assert "np.float" not in raw and '"0.5"' not in raw

    def test_returns_the_written_path(self, tmp_path):
        path = save_training_history_json(self._history(), tmp_path)
        assert path == tmp_path / "training_history.json"

    def test_an_unserializable_value_raises(self, tmp_path):
        with pytest.raises(TypeError):
            save_training_history_json(_FakeHistory({"loss": [object()]}), tmp_path)
        assert not (tmp_path / "training_history.json").exists()

    def test_a_missing_directory_raises(self, tmp_path):
        missing = tmp_path / "does" / "not" / "exist"
        with pytest.raises(FileNotFoundError):
            save_training_history_json(self._history(), missing)

    def test_a_failed_open_raises_the_os_error(self, tmp_path, monkeypatch):
        """A full disk or a read-only run dir: the error reaches the caller unchanged."""
        def full_disk(*_args, **_kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("builtins.open", full_disk)
        with pytest.raises(OSError, match="No space left"):
            save_training_history_json(self._history(), tmp_path)

    def test_control_a_real_write_is_detectable(self, tmp_path):
        """Guard against the failure tests passing for the wrong reason.

        The success path writes the file and returns its path.
        """
        assert save_training_history_json(self._history(), tmp_path) == (
            tmp_path / "training_history.json")
        assert (tmp_path / "training_history.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
