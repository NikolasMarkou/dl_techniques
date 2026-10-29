"""
Tests for FisherAnalyzer and its ModelAnalyzer integration.

Covers: the analyzer contract, the fields it publishes, per-model failure isolation (a
numeric failure skips the model, a programming error re-raises), the shared disjoint sample
plan, the persistence arm, strict-JSON serialization including the raw-table gate and the
read-only guarantee of ``save_results`` (F-040), the version / schema bumps, and a meta-test
that no ``fisher_*`` config field goes unread.
"""

import ast
import dataclasses
import json
import pathlib

import keras
import numpy as np
import pandas as pd
import pytest

import dl_techniques.analyzer as analyzer_package
from dl_techniques.analyzer import AnalysisConfig, DataInput, ModelAnalyzer
from dl_techniques.analyzer import fisher_metrics as fm
from dl_techniques.analyzer.analyzers.fisher_analyzer import FisherAnalyzer
from dl_techniques.analyzer.constants import FISHER_TABLE_COLUMNS, FisherMetricNames as FM
from dl_techniques.analyzer.data_types import AnalysisResults
from dl_techniques.analyzer.model_analyzer import RESULTS_SCHEMA_VERSION
from tests.numerics import true_float32_matmul

N_FEATURES = 4
N_CLASSES = 3
SRC = pathlib.Path(analyzer_package.__file__).parent


@pytest.fixture(autouse=True)
def cpu_float32():
    with keras.device("cpu"), true_float32_matmul():
        yield


def classifier(seed: int, units: int = 5) -> keras.Model:
    keras.utils.set_random_seed(seed)
    return keras.Sequential([
        keras.layers.Input((N_FEATURES,)),
        keras.layers.Dense(units, activation="tanh"),
        keras.layers.Dense(N_CLASSES),
    ])


def make_data(n: int = 80, seed: int = 0) -> DataInput:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, N_FEATURES)).astype("float32")
    return DataInput(x_data=x, y_data=rng.integers(0, N_CLASSES, size=n))


def config(**kw) -> AnalysisConfig:
    base = dict(
        analyze_weights=False, analyze_calibration=False, analyze_information_flow=False,
        analyze_training_dynamics=False, analyze_spectral=False, analyze_fisher=True,
        fisher_n_samples=30, fisher_batch_size=4, fisher_token_level=False,
        random_state=3, fisher_selection_percentages=(5, 10))
    base.update(kw)
    return AnalysisConfig(**base)


def run(models, cfg=None, data=None, signal=None) -> AnalysisResults:
    results = AnalysisResults()
    FisherAnalyzer(models, cfg or config(), signal).analyze(results, data or make_data(), None)
    return results


def reward_signal() -> fm.FisherSignal:
    return fm.FisherSignal(reward_fn=lambda x_i, y, target: float(int(np.asarray(y)) == int(target)))


class TestContract:

    def test_requires_data_is_true(self):
        assert FisherAnalyzer({"a": classifier(0)}, config()).requires_data() is True

    def test_it_is_registered_in_the_orchestrator(self):
        from dl_techniques.analyzer import model_analyzer as ma
        assert ma.ANALYSIS_TYPE_FISHER == "fisher" and "fisher" in ma.DATA_REQUIRED_ANALYSES

    def test_version_and_schema_were_bumped(self):
        major, minor, *_ = (int(p) for p in analyzer_package.__version__.split("."))
        assert (major, minor) >= (1, 2)
        assert RESULTS_SCHEMA_VERSION >= 5


class TestPublishedFields:

    @pytest.fixture(scope="class")
    def results(self):
        with keras.device("cpu"), true_float32_matmul():
            return run({"a": classifier(0), "b": classifier(1)})

    def test_fisher_populates_the_results_fields(self, results):
        table = results.fisher_analysis
        assert list(table.columns) == list(FISHER_TABLE_COLUMNS)
        assert set(table["model_name"]) == {"a", "b"}
        assert len(table) == 2 * results.fisher_outputs["a"].n_parameters
        assert (table["fisher_value"] >= 0).all()
        # no gradient arm ran: the columns say NaN, never 0.0 (D-036)
        assert table["gradient_value"].isna().all() and table["gradient_rank"].isna().all()
        for name in ("a", "b"):
            summary = results.fisher_summary_per_model[name]
            assert summary[FM.STATUS] == "ok"
            for key in (FM.TOTAL_FISHER, FM.MEAN_FISHER, FM.STD_FISHER, FM.N_PARAMETERS, FM.N_SAMPLES,
                        FM.SKIPPED_SAMPLES, FM.COMPUTATION_TIME):
                assert key in summary, key
            assert FM.SIGNAL_SCALE not in summary and FM.VIOLATION_RATE not in summary
        assert results.fisher_gradient_recall == {} and results.fisher_persistence is None

    def test_one_plan_serves_every_model_and_is_fixed_by_the_seed(self, results):
        indices = [[s["index"] for s in results.fisher_outputs[m].samples_used] for m in ("a", "b")]
        assert indices[0] == indices[1] == list(results.fisher_summary["sample_plan"]["fisher_indices"])
        again = run({"a": classifier(0)})
        assert again.fisher_summary["sample_plan_fingerprint"] == results.fisher_summary["sample_plan_fingerprint"]
        other = run({"a": classifier(0)}, config(random_state=4))
        assert other.fisher_summary["sample_plan_fingerprint"] != results.fisher_summary["sample_plan_fingerprint"]

    def test_the_table_is_in_the_canonical_order(self, results):
        out = results.fisher_outputs["a"]
        sub = results.fisher_analysis[results.fisher_analysis["model_name"] == "a"]
        assert (sub["flat_index"].to_numpy() == np.arange(out.n_parameters)).all()
        np.testing.assert_allclose(sub["fisher_value"].to_numpy(), out.fisher_flat)
        assert list(dict.fromkeys(sub["variable_path"].astype(str))) == [p for p, _, _ in out.param_slices]

    def test_an_identical_seed_reproduces_byte_equal_fisher(self, results):
        again = run({"a": classifier(0), "b": classifier(1)})
        for name in ("a", "b"):
            assert (again.fisher_outputs[name].fisher_flat.tobytes()
                    == results.fisher_outputs[name].fisher_flat.tobytes())
            assert (again.fisher_outputs[name].skipped_samples
                    == results.fisher_outputs[name].skipped_samples)

    def test_the_sampling_batch_is_one_number_for_the_whole_run(self):
        tight = run({"a": classifier(0), "b": classifier(1, units=9)}, config(memory_limit_mb=0))
        assert tight.fisher_summary["batch_size"] == 1


class TestFailureIsolation:

    def test_a_model_the_analyzer_cannot_score_is_reported_and_the_others_still_run(self):
        flat_output = keras.Sequential([keras.layers.Input((N_FEATURES,)), keras.layers.Dense(2),
                                        keras.layers.Reshape((1, 2)), keras.layers.Reshape((1, 1, 2))])
        results = run({"good": classifier(0), "bad": flat_output})
        assert "good" in results.fisher_outputs and "bad" not in results.fisher_outputs
        bad = results.fisher_summary_per_model["bad"]
        assert bad[FM.STATUS] == FM.ERROR and FM.TOTAL_FISHER not in bad  # never a 0.0 row
        assert results.fisher_summary["failed_models"].keys() == {"bad"}
        assert results.fisher_summary_per_model["good"][FM.STATUS] == "ok"

    def test_programming_errors_are_not_swallowed(self):
        with pytest.raises((AttributeError, TypeError)):
            run({"good": classifier(0), "bug": object()})


class TestConfigurationRefusals:

    def test_gradients_without_a_signal_raise_before_any_model_runs(self):
        with pytest.raises(ValueError, match="FisherSignal"):
            run({"a": classifier(0)}, config(fisher_compute_gradients=True, fisher_n_responses=2))

    def test_one_response_rl_raises(self):
        with pytest.raises(ValueError, match="n_responses must be >= 2"):
            run({"a": classifier(0)}, config(fisher_compute_gradients=True, fisher_n_responses=1),
                signal=reward_signal())

    def test_persistence_names_are_validated(self):
        with pytest.raises(ValueError, match="unknown models"):
            run({"a": classifier(0)}, config(fisher_persistence_indices=["a", "zzz"]))
        with pytest.raises(ValueError, match="at least one later checkpoint"):
            run({"a": classifier(0)}, config(fisher_persistence_indices=["a"]))


class TestGradientArmThroughTheAnalyzer:

    @pytest.fixture(scope="class")
    def results(self):
        with keras.device("cpu"), true_float32_matmul():
            cfg = config(fisher_compute_gradients=True, fisher_n_responses=3, fisher_n_samples=40,
                         fisher_n_gradient_prompts=25)
            return run({"a": classifier(0)}, cfg, make_data(120), reward_signal())

    def test_the_arms_use_disjoint_prompts(self, results):
        plan = results.fisher_summary["sample_plan"]
        assert not set(plan["fisher_indices"]) & set(plan["gradient_indices"])
        assert len(plan["gradient_indices"]) == 25 and results.fisher_summary["n_gradient_prompts"] == 25

    def test_signal_scale_recall_and_violation_are_published(self, results):
        summary = results.fisher_summary_per_model["a"]
        assert summary[FM.SIGNAL_SCALE] > 0
        assert 0.0 <= summary[FM.VIOLATION_RATE] <= 1.0
        assert set(results.fisher_gradient_recall["a"]) == {5.0, 10.0}
        table = results.fisher_analysis
        assert table["gradient_value"].notna().all()
        assert sorted(table["gradient_rank"])[0] == 1.0

    def test_a_sample_too_small_to_hold_disjoint_arms_is_refused(self):
        with pytest.raises(ValueError, match="disjointly"):
            run({"a": classifier(0)},
                config(fisher_compute_gradients=True, fisher_n_responses=2, fisher_n_samples=60,
                       fisher_n_gradient_prompts=60), make_data(50), reward_signal())


class TestPersistenceArm:

    def test_the_persistence_table_is_built_from_the_named_models(self):
        """Same weights, same plan, independent on-policy draws: Fisher persists far above chance.

        The sets must be big enough for the comparison to mean something: with 43
        parameters, q = 5% selects 2 coordinates and the draw noise alone decides them.
        """
        results = run({"init": classifier(0), "later": classifier(0)},
                      config(fisher_persistence_indices=["init", "later"], fisher_n_samples=300,
                             fisher_selection_percentages=(25, 50)), make_data(300))
        table = results.fisher_persistence
        assert isinstance(table, pd.DataFrame) and set(table["checkpoint"]) == {"later"}
        assert (table["retention_high"] > table["random_high"] + 20).all()
        assert (table["random_high"] == table["q"]).all() and (table["random_low"] == 100 - table["q"]).all()
        assert table["mass_ratio"].between(0.8, 1.25).all()

    def test_models_that_are_not_the_same_architecture_do_not_silently_align(self):
        with pytest.raises(ValueError, match="do not align"):
            run({"init": classifier(0), "later": classifier(0, units=9)},
                config(fisher_persistence_indices=["init", "later"]))


class TestSequenceModelThroughTheAnalyzer:

    def test_a_token_model_gets_the_token_level_arm(self):
        keras.utils.set_random_seed(0)
        ids = keras.layers.Input((None,), dtype="int32")
        emb = keras.layers.Embedding(5, 4)(ids)
        model = keras.Model(ids, keras.layers.Dense(5)(emb))
        rng = np.random.default_rng(0)
        data = DataInput(x_data=rng.integers(0, 5, size=(20, 3)).astype("int32"),
                         y_data=np.zeros(20))
        cfg = config(fisher_n_samples=8, fisher_max_tokens=3, fisher_token_level=True,
                     fisher_n_positions_per_response=2, fisher_n_token_pairs_per_position=2)
        results = run({"lm": model}, cfg, data)
        assert results.fisher_outputs["lm"].output_kind == fm.KIND_SEQUENCE
        token = results.fisher_token_level["lm"]
        assert token["token_fisher_by_position"].shape[0] == 2
        assert FM.REAL_UNIFORM_RATIO in results.fisher_summary_per_model["lm"]


class TestSerialization:

    @staticmethod
    def analyze(tmp_path, **kw):
        cfg = config(**kw)
        analyzer = ModelAnalyzer({"a": classifier(0)}, cfg, output_dir=tmp_path)
        analyzer.analyze(make_data(), {"fisher"})
        return analyzer, json.loads((tmp_path / "analysis_results.json").read_text())

    def test_the_round_trip_is_strict_json_with_the_summary_and_without_the_raw_table(self, tmp_path):
        analyzer, saved = self.analyze(tmp_path)
        assert saved["schema_version"] == RESULTS_SCHEMA_VERSION
        assert saved["analyzer_version"] == analyzer_package.__version__
        assert saved["fisher_summary_per_model"]["a"][FM.STATUS] == "ok"
        assert saved["fisher_summary"]["sample_plan"]["fisher_indices"]
        assert "fisher_analysis" not in saved  # the raw gate defaults to closed
        assert saved["config"]["analyze_fisher"] is True
        assert saved["config"]["json_include_fisher_raw"] is False

    def test_the_raw_gate_emits_one_row_per_parameter(self, tmp_path):
        analyzer, saved = self.analyze(tmp_path, json_include_fisher_raw=True)
        rows = saved["fisher_analysis"]
        assert len(rows) == analyzer.results.fisher_outputs["a"].n_parameters
        assert set(rows[0]) == set(FISHER_TABLE_COLUMNS)
        assert rows[0]["gradient_value"] is None  # NaN became null, not 0.0

    def test_saving_twice_leaves_the_results_intact(self, tmp_path):
        analyzer, first = self.analyze(tmp_path, json_include_fisher_raw=True)
        before = analyzer.results.fisher_analysis.copy()
        flat = analyzer.results.fisher_outputs["a"].fisher_flat.copy()
        analyzer.save_results()
        pd.testing.assert_frame_equal(analyzer.results.fisher_analysis, before)
        np.testing.assert_array_equal(analyzer.results.fisher_outputs["a"].fisher_flat, flat)
        assert json.loads((tmp_path / "analysis_results.json").read_text())["fisher_analysis"] == first["fisher_analysis"]

    def test_the_default_run_renders_plots(self, tmp_path):
        self.analyze(tmp_path)
        assert (tmp_path / "fisher_distribution.png").exists()
        assert (tmp_path / "fisher_top_coordinates.png").exists()
        assert not (tmp_path / "fisher_persistence.png").exists()  # no persistence arm, no frame

    def test_the_orchestrator_runs_fisher_end_to_end_with_a_signal(self, tmp_path):
        cfg = config(fisher_compute_gradients=True, fisher_n_responses=2, fisher_n_samples=20,
                     fisher_n_gradient_prompts=15)
        analyzer = ModelAnalyzer({"a": classifier(0)}, cfg, output_dir=tmp_path,
                                 fisher_signal=reward_signal())
        analyzer.analyze(make_data(80), {"fisher"})
        saved = json.loads((tmp_path / "analysis_results.json").read_text())
        assert saved["fisher_summary_per_model"]["a"][FM.SIGNAL_SCALE] > 0
        assert "5.0" in saved["fisher_gradient_recall"]["a"]
        assert (tmp_path / "fisher_gradient_recall.png").exists()
        assert (tmp_path / "fisher_vs_gradient.png").exists()


class TestEveryKnobHasOneHome:

    def test_every_fisher_config_field_flows_into_fisher_config(self):
        """Set each source field to a non-default value and see it arrive."""
        overrides = dict(
            fisher_force_sequence=True, fisher_n_samples=7, fisher_n_responses=3,
            fisher_n_gradient_prompts=5, fisher_compute_gradients=True, fisher_signal_type="opd_full",
            fisher_token_level=False, fisher_max_tokens=9, fisher_n_positions_per_response=2,
            fisher_n_token_pairs_per_position=3, fisher_batch_size=2, fisher_accumulate_dtype="float64",
            fisher_sampling_temperature=0.5, fisher_regression_variance=0.3,
            fisher_score_scope="response_plus_prompt", fisher_selection_percentages=(2, 3),
            fisher_zero_threshold=1e-6, fisher_sft_chi2_cap=5.0, fisher_assert_no_weight_drift=False,
            random_state=11, memory_limit_mb=77, output_activation="sigmoid")
        cfg = fm.FisherConfig.from_analysis_config(AnalysisConfig(**overrides))
        for f in dataclasses.fields(fm.FisherConfig):
            source = fm.UNPREFIXED_CONFIG_SOURCES.get(f.name, f"fisher_{f.name}")
            assert getattr(cfg, f.name) == overrides[source], f.name

    def test_every_fisher_field_of_analysis_config_is_read_somewhere(self):
        consumers = {
            "fisher_metrics.py": (SRC / "fisher_metrics.py").read_text(),
            "analyzers/fisher_analyzer.py": (SRC / "analyzers" / "fisher_analyzer.py").read_text(),
            "model_analyzer.py": (SRC / "model_analyzer.py").read_text(),
            "visualizers/fisher_visualizer.py": (SRC / "visualizers" / "fisher_visualizer.py").read_text(),
        }
        mapped = {fm.UNPREFIXED_CONFIG_SOURCES.get(f.name, f"fisher_{f.name}")
                  for f in dataclasses.fields(fm.FisherConfig)}
        for f in dataclasses.fields(AnalysisConfig):
            if not (f.name.startswith("fisher_") or f.name == "json_include_fisher_raw"):
                continue
            read_directly = any(f.name in text for text in consumers.values())
            assert f.name in mapped or read_directly, f"{f.name} is declared and never read"

    def test_there_is_no_dead_or_unsupported_fisher_knob(self):
        names = {f.name for f in dataclasses.fields(AnalysisConfig)}
        assert "fisher_decay" not in names and "fisher_compute_dtype" not in names
