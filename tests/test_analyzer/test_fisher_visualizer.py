"""
Tests for FisherVisualizer.

Covers: every plot renders and saves, each title names its ESTIMATOR and the N_F / N_g / G
of the pass (the B.3 ambiguity is pinned, not left implicit), the recall plot always carries
the random-baseline diagonal, a missing or failed bound is stamped on the figure, and an arm
that did not run produces no figure and no placeholder file.
"""

import dataclasses

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dl_techniques.analyzer import AnalysisConfig, DataInput
from dl_techniques.analyzer import fisher_metrics as fm
from dl_techniques.analyzer.analyzers.fisher_analyzer import FisherAnalyzer
from dl_techniques.analyzer.data_types import AnalysisResults
from dl_techniques.analyzer.visualizers import fisher_visualizer as fv
from dl_techniques.analyzer.visualizers.fisher_visualizer import FisherVisualizer
from tests.numerics import true_float32_matmul

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def cpu_float32():
    with keras.device("cpu"), true_float32_matmul():
        yield
    plt.close("all")


def classifier(seed: int) -> keras.Model:
    keras.utils.set_random_seed(seed)
    return keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(5, activation="tanh"),
                             keras.layers.Dense(3)])


def lm() -> keras.Model:
    keras.utils.set_random_seed(0)
    ids = keras.layers.Input((None,), dtype="int32")
    return keras.Model(ids, keras.layers.Dense(5)(keras.layers.Embedding(5, 4)(ids)))


def config(**kw) -> AnalysisConfig:
    base = dict(analyze_weights=False, analyze_calibration=False, analyze_information_flow=False,
                analyze_training_dynamics=False, analyze_spectral=False, analyze_fisher=True,
                fisher_n_samples=25, fisher_batch_size=4, fisher_token_level=False, random_state=1,
                fisher_selection_percentages=(5, 10))
    base.update(kw)
    return AnalysisConfig(**base)


def build(models, cfg, data, signal=None) -> AnalysisResults:
    results = AnalysisResults(config=cfg)
    FisherAnalyzer(models, cfg, signal).analyze(results, data, None)
    return results


def visualizer(results, cfg, tmp_path) -> FisherVisualizer:
    colors = {m: f"C{i}" for i, m in enumerate(results.fisher_summary_per_model)}
    return FisherVisualizer(results, cfg, tmp_path, colors)


@pytest.fixture(scope="module")
def full():
    """Gradient arm + persistence arm on two classifiers, and a token-level language model."""
    with keras.device("cpu"), true_float32_matmul():
        rng = np.random.default_rng(0)
        data = DataInput(rng.normal(size=(120, 4)).astype("float32"), rng.integers(0, 3, 120))
        signal = fm.FisherSignal(reward_fn=lambda x, y, t: float(int(np.asarray(y)) == int(t)))
        cfg = config(fisher_compute_gradients=True, fisher_n_responses=3, fisher_n_gradient_prompts=30,
                     fisher_persistence_indices=["a", "b"])
        results = build({"a": classifier(0), "b": classifier(0)}, cfg, data, signal)
        tok_cfg = config(fisher_n_samples=6, fisher_max_tokens=3, fisher_token_level=True,
                         fisher_n_positions_per_response=2, fisher_n_token_pairs_per_position=2)
        tok_data = DataInput(rng.integers(0, 5, size=(12, 3)).astype("int32"), np.zeros(12))
        tokens = build({"lm": lm()}, tok_cfg, tok_data)
        results.fisher_token_level = tokens.fisher_token_level
        results.fisher_outputs["lm"] = tokens.fisher_outputs["lm"]
        results.fisher_summary_per_model["lm"] = tokens.fisher_summary_per_model["lm"]
        return results, cfg


class TestEveryPlotRenders:

    @pytest.mark.parametrize("method", ["plot_fisher_distribution", "plot_top_coordinates",
                                        "plot_fisher_vs_gradient", "plot_gradient_recall",
                                        "plot_token_level", "plot_persistence", "plot_variable_summary"])
    def test_each_plot_returns_a_figure_and_saves_a_png(self, full, tmp_path, method):
        results, cfg = full
        viz = visualizer(results, cfg, tmp_path)
        fig = getattr(viz, method)()
        assert isinstance(fig, plt.Figure)
        viz._save_figure(fig, method)
        assert (tmp_path / f"{method}.png").stat().st_size > 1000

    def test_create_visualizations_saves_all_seven(self, full, tmp_path):
        results, cfg = full
        visualizer(results, cfg, tmp_path).create_visualizations()
        for name in ("fisher_distribution", "fisher_top_coordinates", "fisher_vs_gradient",
                     "fisher_gradient_recall", "fisher_token_level", "fisher_persistence",
                     "fisher_variable_summary"):
            assert (tmp_path / f"{name}.png").exists(), name


class TestLabelsAreExplicit:

    @staticmethod
    def all_text(fig):
        texts = [fig._suptitle.get_text()] if fig._suptitle else []
        texts += [ax.get_title() for ax in fig.axes]
        return "\n".join(texts)

    def test_sequence_plots_name_the_sequence_estimator_and_the_sample_sizes(self, full, tmp_path):
        results, cfg = full
        viz = visualizer(results, cfg, tmp_path)
        for method in ("plot_fisher_distribution", "plot_top_coordinates", "plot_gradient_recall",
                       "plot_variable_summary", "plot_persistence", "plot_fisher_vs_gradient"):
            text = self.all_text(getattr(viz, method)())
            assert fv.LABEL_SEQUENCE in text, method
            assert "N_F=" in text and "N_g=30" in text and "G=3" in text, method

    def test_the_token_plot_names_the_other_estimator(self, full, tmp_path):
        results, cfg = full
        text = self.all_text(visualizer(results, cfg, tmp_path).plot_token_level())
        assert fv.LABEL_TOKEN in text and fv.LABEL_SEQUENCE not in text

    def test_the_two_labels_differ_so_the_ambiguity_cannot_collapse(self):
        assert fv.LABEL_SEQUENCE != fv.LABEL_TOKEN and "not the sequence Fisher" in fv.LABEL_TOKEN


class TestBaselinesAndFailedBounds:

    def test_the_recall_plot_always_draws_the_random_diagonal(self, full, tmp_path):
        results, cfg = full
        ax = visualizer(results, cfg, tmp_path).plot_gradient_recall().axes[0]
        diagonal = [line for line in ax.get_lines() if line.get_label() == "random subset (y = q)"]
        assert len(diagonal) == 1
        x, y = diagonal[0].get_data()
        np.testing.assert_allclose(x, y)

    def test_persistence_curves_carry_both_random_baselines(self, full, tmp_path):
        results, cfg = full
        axes = visualizer(results, cfg, tmp_path).plot_persistence().axes
        assert any(l.get_label() == "random (q)" for l in axes[0].get_lines())
        assert any(l.get_label() == "random (100 - q)" for l in axes[1].get_lines())

    @staticmethod
    def with_violation(results, rate):
        out = results.fisher_outputs["a"]
        patched = dataclasses.replace(out, violation_rate=rate)
        return dataclasses.replace(results, fisher_outputs={**results.fisher_outputs, "a": patched})

    @pytest.mark.parametrize("rate,expected", [(float("nan"), "NOT COMPUTED"), (0.2, "BOUND VIOLATED"),
                                               (0.0, "violations: 0.00%")])
    def test_the_bound_status_is_stamped_on_the_figure(self, full, tmp_path, rate, expected):
        results, cfg = full
        viz = visualizer(self.with_violation(results, rate), cfg, tmp_path)
        titles = " ".join(ax.get_title() for ax in viz.plot_fisher_vs_gradient().axes)
        assert expected in titles


class TestAbsentArmsAreSkipped:

    @pytest.fixture(scope="class")
    def plain(self):
        with keras.device("cpu"), true_float32_matmul():
            rng = np.random.default_rng(0)
            data = DataInput(rng.normal(size=(40, 4)).astype("float32"), rng.integers(0, 3, 40))
            cfg = config()
            return build({"a": classifier(0)}, cfg, data), cfg

    def test_arms_that_did_not_run_return_none(self, plain, tmp_path):
        results, cfg = plain
        viz = visualizer(results, cfg, tmp_path)
        assert viz.plot_fisher_vs_gradient() is None and viz.plot_gradient_recall() is None
        assert viz.plot_token_level() is None and viz.plot_persistence() is None

    def test_no_placeholder_files_are_written(self, plain, tmp_path):
        results, cfg = plain
        visualizer(results, cfg, tmp_path).create_visualizations()
        written = {p.stem for p in tmp_path.glob("*.png")}
        assert written == {"fisher_distribution", "fisher_top_coordinates", "fisher_variable_summary"}

    def test_empty_results_do_nothing(self, tmp_path):
        viz = FisherVisualizer(AnalysisResults(), AnalysisConfig(), tmp_path, {})
        viz.create_visualizations()
        assert list(tmp_path.iterdir()) == []

    def test_a_nan_filled_output_does_not_crash_the_plots(self, plain, tmp_path):
        results, cfg = plain
        out = results.fisher_outputs["a"]
        nan_out = dataclasses.replace(out, standard_errors=np.full_like(out.fisher_flat, np.nan),
                                      violation_rate=float("nan"))
        patched = dataclasses.replace(results, fisher_outputs={"a": nan_out})
        viz = visualizer(patched, cfg, tmp_path)
        viz.create_visualizations()
        assert (tmp_path / "fisher_distribution.png").exists()
