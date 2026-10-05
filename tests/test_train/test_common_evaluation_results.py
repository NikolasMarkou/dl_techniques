"""The result-container builders, and the reachability they were added to restore.

`dl_techniques/visualization` ships 25 plugin templates. A plugin is only reachable if
something BOTH registers it AND constructs the container its `can_handle` type-checks
against. Measured across `src/` and `tests/`, 19 of the 25 had no user at all, and the
cause was almost never a missing `register_template` call — it was that nobody built the
input:

| container                    | constructed by                        | plugins stranded without it |
|------------------------------|---------------------------------------|-----------------------------|
| `RegressionResults`          | **nothing, anywhere**                 | 6 (regression family)       |
| `TimeSeriesEvaluationResults`| **nothing, anywhere**                 | `ForecastVisualization`     |
| `ClassificationResults`      | one dead helper + one live trainer    | 5 of 7                      |

`create_regression_results` / `create_multi_model_regression` / `create_timeseries_results`
are those missing constructors. Each guard below was proven RED before the builder existed
— the container classes imported and asserted against, the plugin `can_handle` calls
returning `False`.

The trap this file exists to pin is `ROCPRCurves`. Every other classification plugin
degrades honestly without probabilities, but that one does not: its per-model loop is
`if results.y_prob is None: continue`, so it draws no curves, its `legend()` then raises
'No artists with labels found', and `VisualizationManager.visualize` swallows that and
returns None (core.py:508-512). The caller gets a MISSING file and no exception — only a
log line — so a trainer registering `roc_pr_curves` on a probability-free model reads as
a successful run that happened to produce no ROC figure. (It does NOT write a blank PNG;
`visualize` never reaches `save_figure`, because `create_visualization` raises first.
That correction is measured, not assumed.) The xfail test at the bottom is the RED proof
of exactly that, so nobody adds the registration later thinking it is free.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dl_techniques.visualization import (  # noqa: E402
    ForecastVisualization,
    MultiModelRegression,
    PredictionErrorVisualization,
    QQPlotVisualization,
    RegressionResults,
    ResidualDistributionVisualization,
    ResidualsPlotVisualization,
    TimeSeriesEvaluationResults,
)
from train.common.evaluation import (  # noqa: E402
    create_multi_model_regression,
    create_regression_results,
    create_timeseries_results,
)


@pytest.fixture
def viz_manager(tmp_path):
    """A manager whose plugin name matches its template key — the two MUST agree,
    because `visualize` registers the instance under `plugin.name` and then re-reads
    it with `self.plugins[plugin_name]` (`visualization/core.py:479-483`)."""
    from dl_techniques.visualization import VisualizationManager

    return VisualizationManager(
        experiment_name="", output_dir=str(tmp_path), timestamp=""
    )


@pytest.fixture
def regression_pair():
    """A seeded (y_true, y_pred) pair long enough for a QQ plot to have a tail."""
    rng = np.random.default_rng(20261006)
    y_true = rng.normal(loc=10.0, scale=3.0, size=512)
    return y_true, y_true + rng.normal(loc=0.0, scale=0.5, size=512)


@pytest.fixture
def forecast_triple():
    """(inputs, true, predicted, quantiles) for four windows of horizon 3."""
    rng = np.random.default_rng(20261006)
    n, hist, horizon, levels = 4, 12, 3, [0.1, 0.5, 0.9]
    inputs = rng.normal(size=(n, hist))
    true_forecasts = rng.normal(size=(n, horizon))
    predicted = true_forecasts + rng.normal(scale=0.3, size=(n, horizon))
    quantiles = np.stack(
        [predicted - 1.0, predicted, predicted + 1.0], axis=-1
    )
    return inputs, true_forecasts, predicted, quantiles, levels


class TestTheRegressionContainerIsNowReachable:
    """The six regression plugins type-check against `RegressionResults`."""

    def test_the_builder_produces_what_every_regression_plugin_demands(
        self, regression_pair, viz_manager
    ):
        y_true, y_pred = regression_pair
        results = create_regression_results(y_true, y_pred, model_name="probe")

        assert isinstance(results, RegressionResults)
        for registry_name, plugin_cls in (
            ("prediction_error", PredictionErrorVisualization),
            ("residuals_plot", ResidualsPlotVisualization),
            ("residual_distribution", ResidualDistributionVisualization),
            ("qq_plot", QQPlotVisualization),
        ):
            viz_manager.register_template(registry_name, plugin_cls)
            assert viz_manager.create_plugin_from_template(registry_name).can_handle(
                results
            ), (
                f"{plugin_cls.__name__} refused the container "
                f"create_regression_results builds — the builder and the plugins "
                f"have drifted apart."
            )

    def test_every_registry_key_used_here_equals_its_plugins_own_name(
        self, viz_manager
    ):
        """`ErrorAnalysisDashboard` is named `"error_analysis"`, not
        `"error_analysis_dashboard"` — a template key that disagrees with
        `plugin.name` misses `visualize`'s second lookup with a bare KeyError.
        Pinned so the mothnet registrations cannot drift back."""
        for registry_name, plugin_cls in (
            ("prediction_error", PredictionErrorVisualization),
            ("residuals_plot", ResidualsPlotVisualization),
            ("residual_distribution", ResidualDistributionVisualization),
            ("qq_plot", QQPlotVisualization),
        ):
            viz_manager.register_template(registry_name, plugin_cls)
            assert viz_manager.create_plugin_from_template(registry_name).name == (
                registry_name
            )

    def test_both_arrays_are_flattened_so_a_column_vector_still_pairs(self, regression_pair):
        """A `(N, 1)` prediction is the common shape; unflattened it breaks every
        per-sample mask in the plugins with an axis error."""
        y_true, y_pred = regression_pair
        results = create_regression_results(y_true[:, None], y_pred[:, None])

        assert results.y_true.shape == (y_true.size,)
        assert results.y_pred.shape == (y_pred.size,)

    def test_model_name_and_feature_names_reach_the_container(self, regression_pair):
        y_true, y_pred = regression_pair
        results = create_regression_results(
            y_true, y_pred, model_name="nbeats", feature_names=["f0", "f1"]
        )

        assert results.model_name == "nbeats"
        assert results.feature_names == ["f0", "f1"]


class TestTheMultiModelContainerIsNowReachable:
    def test_raw_pairs_are_built_into_per_model_containers(self, regression_pair):
        y_true, y_pred = regression_pair
        bundle = create_multi_model_regression({"a": (y_true, y_pred)})

        assert isinstance(bundle, MultiModelRegression)
        assert set(bundle.results) == {"a"}
        assert isinstance(bundle.results["a"], RegressionResults)
        assert bundle.results["a"].model_name == "a", (
            "Each member must be labelled with its own key, or the comparison "
            "dashboard titles every panel with the same name."
        )

    def test_already_built_containers_pass_through_untouched(self, regression_pair):
        y_true, y_pred = regression_pair
        already = create_regression_results(y_true, y_pred, model_name="custom")

        bundle = create_multi_model_regression({"a": already})

        assert bundle.results["a"] is already, (
            "A pre-built RegressionResults must not be re-wrapped — that would "
            "discard the caller's model_name."
        )

    def test_an_empty_bundle_is_refused(self):
        """An empty dict would build a zero-panel dashboard and save a blank PNG."""
        with pytest.raises(ValueError, match="at least one model"):
            create_multi_model_regression({})


class TestTheForecastContainerIsNowReachable:
    def test_the_builder_produces_what_the_forecast_plugin_demands(
        self, forecast_triple, viz_manager
    ):
        inputs, true_forecasts, predicted, quantiles, levels = forecast_triple
        results = create_timeseries_results(
            inputs, true_forecasts, predicted, quantiles,
            model_name="probe", quantile_levels=levels,
        )

        assert isinstance(results, TimeSeriesEvaluationResults)
        viz_manager.register_template("forecast_visualization", ForecastVisualization)
        assert viz_manager.create_plugin_from_template(
            "forecast_visualization"
        ).can_handle(results)

    def test_a_horizon_of_one_keeps_its_second_axis(self):
        """A `(N, 1)` forecast is N single-step samples. Squeezing it to 1-D makes
        the plugin's per-sample loop index characters, so the axis is preserved."""
        results = create_timeseries_results(np.zeros((6, 10)), np.zeros((6, 1)))

        assert results.all_true_forecasts.shape == (6, 1)

    def test_a_single_window_is_promoted_to_a_batch_of_one(self):
        """Bare 1-D arrays describe ONE window on BOTH axes: 10 steps of history
        predicting 3 steps ahead. Promoting the forecast axis the other way —
        `(3,) -> (3, 1)`, three single-step samples — would make the same call mean
        different things depending on which array it touched."""
        results = create_timeseries_results(np.zeros(10), np.zeros(3))

        assert results.all_inputs.shape == (1, 10)
        assert results.all_true_forecasts.shape == (1, 3)

    def test_quantiles_without_levels_are_refused(self, forecast_triple):
        """The plugin needs the levels to label the bands; without them it draws an
        unlabelled envelope that reads as an interval of unknown width."""
        inputs, true_forecasts, _predicted, quantiles, _levels = forecast_triple
        with pytest.raises(ValueError, match="quantile_levels is required"):
            create_timeseries_results(
                inputs, true_forecasts, None, quantiles, quantile_levels=None
            )

    def test_a_mismatched_quantile_count_is_refused(self, forecast_triple):
        inputs, true_forecasts, predicted, quantiles, _levels = forecast_triple
        with pytest.raises(ValueError, match="quantile levels"):
            create_timeseries_results(
                inputs, true_forecasts, predicted, quantiles,
                quantile_levels=[0.1, 0.5],
            )

    def test_point_forecasts_that_disagree_with_the_truth_are_refused(self, forecast_triple):
        inputs, true_forecasts, _predicted, _quantiles, _levels = forecast_triple
        with pytest.raises(ValueError, match="must match all_true_forecasts"):
            create_timeseries_results(inputs, true_forecasts, np.zeros((4, 9)))


class TestThePairGuardsThatMakeTheseContainersSafeToIndex:
    def test_a_length_mismatch_is_refused_with_both_shapes_named(self):
        with pytest.raises(ValueError, match="2 and 3"):
            create_regression_results([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_predictions_are_refused(self):
        """A zero-length container renders an empty figure and saves a blank PNG —
        the same silent failure as the ROC trap below, reached by a different route."""
        with pytest.raises(ValueError, match="empty"):
            create_regression_results(np.array([]), np.array([]))

    def test_disagreeing_sample_counts_are_refused(self):
        with pytest.raises(ValueError, match="agree on num_samples"):
            create_timeseries_results(np.zeros((4, 10)), np.zeros((3, 3)))


class TestThePluginsActuallyRenderWhatTheBuildersHandThem:
    """The reachability claim is only worth anything if the plugins draw. Each
    plugin is driven through a real `VisualizationManager` and asserted to have
    written a file — an exception-swallowing `visualize()` returns None silently."""

    @pytest.mark.parametrize(
        "plugin_name",
        ["prediction_error", "residuals_plot", "residual_distribution", "qq_plot"],
    )
    def test_each_regression_plugin_writes_a_figure(
        self, regression_pair, plugin_name, tmp_path
    ):
        from dl_techniques.visualization import VisualizationManager

        y_true, y_pred = regression_pair
        manager = VisualizationManager(
            experiment_name="", output_dir=str(tmp_path), timestamp=""
        )
        manager.register_template(plugin_name, _plugin_class(plugin_name))

        figure = manager.visualize(
            data=create_regression_results(y_true, y_pred, model_name="probe"),
            plugin_name=plugin_name,
            show=False,
        )

        assert figure is not None, f"{plugin_name} rendered nothing."
        assert list(tmp_path.glob("*.png")), f"{plugin_name} saved no PNG."

    def test_the_forecast_plugin_writes_a_figure(self, forecast_triple, tmp_path):
        from dl_techniques.visualization import VisualizationManager

        inputs, true_forecasts, predicted, quantiles, levels = forecast_triple
        manager = VisualizationManager(
            experiment_name="", output_dir=str(tmp_path), timestamp=""
        )
        manager.register_template("forecast_visualization", ForecastVisualization)

        figure = manager.visualize(
            data=create_timeseries_results(
                inputs, true_forecasts, predicted, quantiles,
                model_name="probe", quantile_levels=levels,
            ),
            plugin_name="forecast_visualization",
            show=False,
        )

        assert figure is not None
        assert list(tmp_path.glob("*.png"))


def _plugin_class(plugin_name: str):
    """The template class behind each registry name the parametrization uses."""
    return {
        "prediction_error": PredictionErrorVisualization,
        "residuals_plot": ResidualsPlotVisualization,
        "residual_distribution": ResidualDistributionVisualization,
        "qq_plot": QQPlotVisualization,
    }[plugin_name]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ROCPRCurves SKIPS a model whose y_prob is None (classification.py:269 "
        "`if results.y_prob is None: continue`), so it draws no curves, then its "
        "legend() call raises 'No artists with labels found'. "
        "VisualizationManager.visualize swallows that and returns None "
        "(core.py:508-512), so the caller sees a MISSING file and no exception — "
        "only a log line. mothnet builds exactly such a probability-free object, "
        "which is why roc_pr_curves is deliberately NOT registered there. This RED "
        "proof fails while that silent no-output stands; it should XPASS once the "
        "plugin refuses the missing probabilities or visualize propagates the "
        "error, at which point the mothnet exclusion can be revisited."
    ),
)
def test_the_roc_pr_trap_yields_no_figure_and_raises_nothing(viz_manager, tmp_path):
    """MEASURED correction to an earlier assumption: the failure mode is a MISSING
    file plus a swallowed error, NOT a blank PNG. `visualize` never reaches
    `save_figure` because `create_visualization` raises first, so a blank image is
    not what a caller gets."""
    from dl_techniques.visualization import (
        ClassificationResults,
        ClassificationReportVisualization,
        ROCPRCurves,
    )

    rng = np.random.default_rng(20261006)
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 4, size=200)
    class_names = [str(i) for i in range(4)]

    viz_manager.register_template("roc_pr_curves", ROCPRCurves)
    viz_manager.register_template(
        "classification_report", ClassificationReportVisualization
    )

    probability_free = ClassificationResults(
        y_true=y_true, y_pred=y_pred, class_names=class_names, model_name="mothnet"
    )

    # The control: a plugin needing only y_true/y_pred draws and saves fine, which is
    # what separates "the container is unusable" from "this one plugin yields nothing".
    assert viz_manager.visualize(
        data=probability_free, plugin_name="classification_report", show=False
    ) is not None
    assert (tmp_path / "classification_report.png").exists()

    figure = viz_manager.visualize(
        data=probability_free, plugin_name="roc_pr_curves", show=False
    )

    assert figure is not None, (
        "ROC/PR is expected to be impossible without probabilities. If this XPASSes "
        "the plugin now handles the missing y_prob — revisit mothnet's deliberate "
        "exclusion of roc_pr_curves."
    )
    assert (tmp_path / "roc_pr_curves.png").exists(), (
        "ROC/PR produced no file without raising. visualize() swallows the error, "
        "so a registration that cannot work shows up as a silently absent figure."
    )