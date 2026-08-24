"""The wave_field trainer must PERSIST its run config and training history.

Companion to `tests/test_train/test_gpt2/test_run_artifacts.py`; same F-06
adoption, same reason the guard executes the trainer body instead of grepping
its source. `src/train/wave_field/` had zero test coverage of any kind before
this file (`src/train/CLAUDE.md` § (f)).

This module covered a SECOND trainer, `train.wave_field.train_memory`, in a
`TestTrainMemoryLlmWritesRunArtifacts` class of 5 tests. That trainer was
DELETED on user instruction (2026-08-13) and the class went with it; nothing
else changed here. `src/train/wave_field/` is now a one-trainer package.
"""

import dataclasses
import json

import pytest

from train.wave_field import pretrain as wf

from .._trainer_run_harness import _StubHistory, _StubModel, run_trainer


# ---------------------------------------------------------------------
# train.wave_field.pretrain.train_wave_field_llm
# ---------------------------------------------------------------------

class TestTrainWaveFieldLlmWritesRunArtifacts:

    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "run"
        results_dir.mkdir()
        config = wf.TrainingConfig(save_dir=str(tmp_path / "save"), field_size=4)
        model = _StubModel(_StubHistory({"loss": [5.0, 4.5], "val_loss": [5.1, 4.4]}))
        run_trainer(
            monkeypatch, wf, wf.train_wave_field_llm, config, results_dir,
            model=model,
        )
        return results_dir, config, model

    def test_fit_actually_ran(self, run):
        _, _, model = run
        assert model.fit_calls == 1

    def test_config_json_is_written(self, run):
        results_dir, _, _ = run
        assert (results_dir / "config.json").is_file(), (
            "train_wave_field_llm did not write config.json into results_dir"
        )

    def test_config_json_records_every_config_field(self, run):
        results_dir, config, _ = run
        payload = json.loads((results_dir / "config.json").read_text())
        assert set(payload) == {f.name for f in dataclasses.fields(config)}
        assert payload

    def test_config_json_records_the_wave_field_specific_value(self, run):
        results_dir, config, _ = run
        payload = json.loads((results_dir / "config.json").read_text())
        assert payload["field_size"] == 4 == config.field_size
        assert payload["save_dir"] == config.save_dir

    def test_training_history_json_is_written_with_the_fit_values(self, run):
        results_dir, _, model = run
        path = results_dir / "training_history.json"
        assert path.is_file()
        assert json.loads(path.read_text()) == model.history.history

