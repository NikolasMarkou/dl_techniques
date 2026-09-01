# `dl_techniques.visualization`

A plugin-based plotting framework. You put your results into one of the standardized dataclasses,
register the plugin you want with a `VisualizationManager`, and call `visualize()` — the manager
renders it with a consistent style and saves it to disk.

There is no test directory for this package; this file and the source are the documentation.

## Quick start

```python
from dl_techniques.visualization import (
    VisualizationManager, TrainingHistory, TrainingCurvesVisualization,
)

viz = VisualizationManager(experiment_name="my_experiment", output_dir="results/plots")
viz.register_template("training_curves", TrainingCurvesVisualization)

history = TrainingHistory(
    epochs=list(range(len(keras_history.history["loss"]))),
    train_loss=keras_history.history["loss"],
    val_loss=keras_history.history.get("val_loss"),
    train_metrics={"accuracy": keras_history.history["accuracy"]},
    val_metrics={"accuracy": keras_history.history["val_accuracy"]},
)

viz.visualize(data=history, plugin_name="training_curves")
# -> results/plots/my_experiment/<timestamp>/training_curves.png
```

**You must register templates yourself.** `auto_discover=True` is the constructor default but
`_discover_plugins()` is an empty stub — a fresh manager knows about zero plugins.

## `VisualizationManager`

```python
VisualizationManager(
    experiment_name: str,
    output_dir: str | Path = "visualizations",
    config: PlotConfig | None = None,
    auto_discover: bool = True,     # no-op
    timestamp: str | None = None,   # pass "" to skip the timestamp subdirectory
)
```

| Method | Notes |
|---|---|
| `register_template(name, plugin_class)` | lazily instantiated on first use |
| `register_plugin(plugin_instance)` | keyed by `plugin.name`, not by the string you pass |
| `visualize(data, plugin_name=None, save=True, show=False, filename=None, **kwargs)` | returns the `Figure`, or `None` on failure |
| `create_dashboard(data: dict, layout=None, save=True, show=False)` | `data` maps plugin name → that plugin's data; saves as `dashboard.png` |
| `get_available_plugins()` / `get_available_templates()` | |
| `save_metadata(dict)` | writes `metadata.json` next to the figures |

Files land in `output_dir / experiment_name / timestamp / <name>.<save_format>`, where `<name>` is
`filename` if given, else the plugin's own `name`.

With `plugin_name=None` the manager searches registered plugins for one whose `can_handle(data)`
is true; if several match it logs an error and returns `None`, so pass `plugin_name` explicitly
whenever the data type is ambiguous (a bare `dict` matches four plugins).

## Plugins

Register with `register_template(<name>, <Class>)`, then pass `<name>` as `plugin_name`.

| Name | Class | Accepts |
|---|---|---|
| `training_curves` | `TrainingCurvesVisualization` | `TrainingHistory`, `dict` |
| `lr_schedule` | `LearningRateScheduleVisualization` | list of numbers, or dict of such lists |
| `model_comparison_bars` | `ModelComparisonBarChart` | `ModelComparison`, `dict` |
| `performance_radar` | `PerformanceRadarChart` | `ModelComparison`, `dict` |
| `convergence_analysis` | `ConvergenceAnalysis` | `TrainingHistory`, `dict` |
| `overfitting_analysis` | `OverfittingAnalysis` | `TrainingHistory`, `dict` |
| `performance_dashboard` | `PerformanceDashboard` | `ModelComparison` |
| `confusion_matrix` | `ConfusionMatrixVisualization` | `ClassificationResults`, `MultiModelClassification` |
| `roc_pr_curves` | `ROCPRCurves` | `ClassificationResults`, `MultiModelClassification` |
| `classification_report` | `ClassificationReportVisualization` | `ClassificationResults`, `MultiModelClassification` |
| `per_class_analysis` | `PerClassAnalysis` | `ClassificationResults`, `MultiModelClassification` |
| `error_analysis` | `ErrorAnalysisDashboard` | `ClassificationResults` |
| `data_distribution` | `DataDistributionAnalysis` | `DatasetInfo`, `np.ndarray`, `pd.DataFrame` |
| `class_balance` | `ClassBalanceVisualization` | `DatasetInfo`, tuple |
| `network_architecture` | `NetworkArchitectureVisualization` | `keras.Model` (anything with `.layers`) |
| `activations` | `ActivationVisualization` | `ActivationData` |
| `feature_maps` | `FeatureMapVisualization` | `ActivationData` |
| `weights` | `WeightVisualization` | `WeightData`, `keras.Model` |
| `gradients` | `GradientVisualization` | `GradientData` |
| `gradient_topology` | `GradientTopologyVisualization` | `GradientTopologyData` |
| `generic_matrix` | `GenericMatrixVisualization` | `MatrixData`, 2-D `np.ndarray` |
| `image_comparison` | `ImageComparisonVisualization` | `ImageData`, list of `np.ndarray` |
| `forecast_visualization` | `ForecastVisualization` | `TimeSeriesEvaluationResults` |
| `prediction_error` | `PredictionErrorVisualization` | `RegressionResults` |
| `residuals_plot` | `ResidualsPlotVisualization` | `RegressionResults` |
| `residual_distribution` | `ResidualDistributionVisualization` | `RegressionResults` |
| `qq_plot` | `QQPlotVisualization` | `RegressionResults` |
| `regression_dashboard` | `RegressionEvaluationDashboard` | `RegressionResults` |

## Data containers

Required fields first, optional after the `|`.

| Container | Fields |
|---|---|
| `TrainingHistory` | `epochs`, `train_loss` \| `val_loss`, `train_metrics`, `val_metrics`, `grad_norms` |
| `ModelComparison` | `model_names`, `metrics` \| `histories`, `predictions` |
| `ClassificationResults` | `y_true`, `y_pred` \| `y_prob`, `class_names`, `model_name` |
| `MultiModelClassification` | `results` \| `dataset_name` |
| `RegressionResults` | `y_true`, `y_pred` \| `model_name`, `feature_names` |
| `MultiModelRegression` | `results` \| `dataset_name` |
| `DatasetInfo` | `x_train`, `y_train` \| `x_test`, `y_test`, `feature_names`, `class_names`, `metadata` |
| `ActivationData` | `layer_names`, `activations` \| `model_name` |
| `WeightData` | `layer_names`, `weights` \| `model_name` |
| `GradientData` | `layer_names`, `gradients` \| `model_name` |
| `GradientTopologyData` | `model`, `gradients` \| `model_name` |
| `MatrixData` | `matrix` \| `title`, `xlabel`, `ylabel`, `xticklabels`, `yticklabels` |
| `ImageData` | `images` \| `titles`, `super_title` |
| `TimeSeriesEvaluationResults` | `all_inputs`, `all_true_forecasts` \| `all_predicted_forecasts`, `all_predicted_quantiles`, `model_name`, `quantile_levels` |

## Styling

`PlotConfig` is a dataclass with everything defaulted. The knobs you will actually touch:

| Field | Default |
|---|---|
| `fig_size` | `(12, 8)` |
| `dpi` / `save_dpi` | `100` / `300` — screen vs. file resolution |
| `style` | `PlotStyle.SCIENTIFIC`; also `MINIMAL`, `DARK`, `PRESENTATION`, `PUBLICATION` |
| `color_scheme` | a `ColorScheme` (`primary`, `secondary`, `success`, `warning`, `info`, `background`, `grid`, `text`, `palette`, `model_colors`) |
| `save_format` | `'png'`; any matplotlib format — it also becomes the file extension |
| `transparent_background`, `bbox_inches`, `tight_layout`, `constrained_layout` | `False`, `'tight'`, `True`, `False` |
| `title_fontsize`, `label_fontsize`, `tick_fontsize`, `legend_fontsize`, `annotation_fontsize` | `16`, `12`, `10`, `10`, `9` |
| `show_grid`, `grid_alpha`, `grid_style`, `legend_location`, `legend_frameon` | `True`, `0.3`, `'--'`, `'best'`, `True` |

```python
from dl_techniques.visualization import VisualizationManager, PlotConfig, PlotStyle

config = PlotConfig(style=PlotStyle.PUBLICATION, save_format="pdf", save_dpi=600)
viz = VisualizationManager("paper_figures", output_dir="figures", config=config, timestamp="")
```

`ColorScheme.get_model_color(model_name, index)` gives a stable per-model colour so the same
model keeps the same colour across every figure.

## Writing a plugin

Subclass `VisualizationPlugin` and implement four members:

```python
import matplotlib.pyplot as plt
from dl_techniques.visualization import VisualizationPlugin

class MyPlot(VisualizationPlugin):
    @property
    def name(self) -> str:
        return "my_plot"

    @property
    def description(self) -> str:
        return "What this shows"

    def can_handle(self, data) -> bool:
        return isinstance(data, MyDataContainer)

    def create_visualization(self, data, ax=None, **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.config.fig_size)
        ...
        return fig

viz.register_template("my_plot", MyPlot)
viz.visualize(my_data, plugin_name="my_plot")
```

`self.config` is the `PlotConfig`, `self.context` is the `VisualizationContext`, and
`self.save_figure(fig, name, subdir=None)` writes through the same path rules as the manager.
Accept an `ax=None` kwarg if you want your plugin to be usable inside `create_dashboard`.

## Gotchas

- **`auto_discover` does nothing.** Register every template you intend to use, or `visualize`
  logs "Plugin or template not found" and returns `None`.
- **`visualize` and `create_dashboard` swallow exceptions.** A plugin that raises produces a log
  line and a `None` return, not a traceback. A missing figure means reading the log.
- **The output path is nested three deep**: `output_dir/experiment_name/timestamp/`. Pass
  `timestamp=""` to write straight into `output_dir/experiment_name/`.
- `register_plugin(instance)` keys the plugin by `instance.name`, ignoring any name you had in
  mind; `register_template(name, cls)` keys by the string you pass. The two can disagree.
- Auto-detection (`plugin_name=None`) is ambiguous for `dict` and for `TrainingHistory`, which
  several plugins accept. Name the plugin.
- `create_dashboard` always saves to `dashboard.png` (or your `save_format`), overwriting any
  previous dashboard in the same directory.
- These plugins render with matplotlib; on a headless machine set `MPLBACKEND=Agg`.

## See also

- `CLAUDE.md` — module map.
- `dl_techniques/optimization/train_vision/framework.py` — the largest in-repo consumer; a
  working example of registering ~12 templates and driving them from a Keras callback.
