"""The gpt2 trainers must PERSIST their run config and training history.

`train.gpt2.pretrain.train_gpt2` and `train.gpt2.finetune.finetune_gpt2` adopted
`save_config_json` / `save_training_history_json` (F-06). Both adoptions are a
single line each, which is precisely why they need a guard that EXECUTES the
trainer body and looks at the filesystem: a mis-placed or deleted one-liner is
invisible to every other test in this tree.

The assertions are on CONTENT, not on existence. A `config.json` holding `{}` is
the measured failure mode of this adoption -- `train.gpt2.finetune`'s
`FinetuneConfig` was a plain class with class-level defaults, on which
`save_config_json` falls back to `vars(config)` and writes 0 keys. See
`# DECISION plan-2026-08-13T091555-230c101d/D-012` in `src/train/gpt2/finetune.py`.
"""

import json
from pathlib import Path

import pytest

from train.gpt2 import finetune as ft
from train.gpt2 import pretrain as pt

from .._trainer_run_harness import _StubHistory, _StubModel, run_trainer


# ---------------------------------------------------------------------
# train.gpt2.pretrain.train_gpt2
# ---------------------------------------------------------------------

class TestTrainGpt2WritesRunArtifacts:

    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "run"
        results_dir.mkdir()
        config = pt.TrainingConfig(save_dir=str(tmp_path / "save"))
        model = _StubModel(_StubHistory({"loss": [4.0, 3.25], "val_loss": [3.9, 3.1]}))
        run_trainer(
            monkeypatch, pt, pt.train_gpt2, config, results_dir, model=model,
        )
        return results_dir, config, model

    def test_fit_actually_ran(self, run):
        _, _, model = run
        assert model.fit_calls == 1

    def test_config_json_is_written(self, run):
        results_dir, _, _ = run
        assert (results_dir / "config.json").is_file()

    def test_config_json_records_every_config_field(self, run):
        results_dir, config, _ = run
        payload = json.loads((results_dir / "config.json").read_text())
        import dataclasses
        expected = {f.name for f in dataclasses.fields(config)}
        assert set(payload) == expected
        assert payload, "config.json must not be an empty object"

    def test_config_json_records_the_actual_values(self, run):
        results_dir, config, _ = run
        payload = json.loads((results_dir / "config.json").read_text())
        assert payload["model_variant"] == config.model_variant
        assert payload["learning_rate"] == config.learning_rate
        assert payload["save_dir"] == config.save_dir

    def test_training_history_json_is_written_with_the_fit_values(self, run):
        results_dir, _, model = run
        path = results_dir / "training_history.json"
        assert path.is_file()
        assert json.loads(path.read_text()) == model.history.history


# ---------------------------------------------------------------------
# train.gpt2.finetune.finetune_gpt2
# ---------------------------------------------------------------------

class TestFinetuneGpt2WritesRunArtifacts:

    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        results_dir = tmp_path / "run"
        results_dir.mkdir()
        config = ft.FinetuneConfig()
        config.save_dir = str(tmp_path / "save")
        model = _StubModel(_StubHistory({"loss": [2.0, 1.5], "val_loss": [2.1, 1.6]}))
        run_trainer(
            monkeypatch, ft, ft.finetune_gpt2, config, results_dir,
            dataset_loader="load_finetune_datasets",
            steps_attr=None,
            plot_callback=True,
            model_loader="load_pretrained_model",
            model=model,
        )
        return results_dir, config, model

    def test_fit_actually_ran(self, run):
        _, _, model = run
        assert model.fit_calls == 1

    def test_config_json_is_written(self, run):
        results_dir, _, _ = run
        assert (results_dir / "config.json").is_file(), (
            "finetune_gpt2 did not write config.json into results_dir"
        )

    def test_config_json_records_every_config_field(self, run):
        results_dir, config, _ = run
        payload = json.loads((results_dir / "config.json").read_text())
        import dataclasses
        expected = {f.name for f in dataclasses.fields(config)}
        assert set(payload) == expected

    def test_config_json_records_fields_never_assigned_on_the_instance(self, run):
        """The D-012 failure mode, stated as an assertion.

        `main()` assigns only a SUBSET of the config's attributes. On a plain
        class the rest live on the CLASS, `vars(instance)` never sees them, and
        `save_config_json` writes a config.json that silently omits them. This
        is deliberately NOT written as `payload != {}` -- that weaker form is
        vacuous, because a single instance assignment (`config.save_dir = ...`)
        already makes `vars()` non-empty, and it was measured NOT to fire on
        the `@dataclass`-removal injection this test exists to catch.
        """
        results_dir, _, _ = run
        payload = json.loads((results_dir / "config.json").read_text())
        # None of these three is ever assigned in `main()`.
        assert payload["warmup_ratio"] == 0.05
        assert payload["weight_decay"] == 0.01
        assert payload["encoding_name"] == "gpt2"

    def test_training_history_json_is_written_with_the_fit_values(self, run):
        results_dir, _, model = run
        path = results_dir / "training_history.json"
        assert path.is_file()
        assert json.loads(path.read_text()) == model.history.history

    def test_artifacts_land_in_results_dir_not_save_dir(self, run):
        """Both artifacts belong to the timestamped run dir, not `config.save_dir`."""
        results_dir, config, _ = run
        assert not (Path(config.save_dir) / "config.json").exists()
        assert not (Path(config.save_dir) / "training_history.json").exists()
