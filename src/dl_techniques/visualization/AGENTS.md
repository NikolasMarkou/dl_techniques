# Visualization Package

Plugin-based visualization framework for training analysis, classification evaluation, data inspection, and time series forecasting.

## Public API

```python
from dl_techniques.visualization import (
    # Core framework
    VisualizationManager, PlotConfig, PlotStyle, ColorScheme,
    VisualizationContext, VisualizationPlugin, CompositeVisualization,
    # Data containers
    TrainingHistory, ModelComparison,
    ClassificationResults, MultiModelClassification,
    DatasetInfo, ActivationData, WeightData, GradientData,
    TimeSeriesEvaluationResults, ForecastVisualization,
    # Plugin implementations (register with VisualizationManager)
    TrainingCurvesVisualization, LearningRateScheduleVisualization,
    ConfusionMatrixVisualization, ROCPRCurves, NetworkArchitectureVisualization,
    # ... and many more
)
```

## Modules

- `core.py` — `VisualizationManager` (central registry), `PlotConfig`, `PlotStyle`, `ColorScheme`, `VisualizationPlugin` base class, `CompositeVisualization`
- `training_performance.py` — Training curves, LR schedules, model comparison, convergence/overfitting analysis, performance dashboard
- `classification.py` — Confusion matrix, ROC/PR curves, classification report, per-class analysis, error dashboard
- `data_nn.py` — Data distribution, class balance, network architecture, activation/weight/feature map/gradient visualization
- `regression.py` — Regression evaluation visualizations
- `time_series.py` — Time series forecast visualization, evaluation results

## Conventions

- Plugin architecture: visualizations register with `VisualizationManager`
- Data containers are standardized dataclasses for each domain
- Uses matplotlib and seaborn for rendering
- `PlotStyle` supports `'publication'` mode for paper-quality figures

## Testing

No dedicated test directory. This line previously claimed visualization is "typically tested
through integration with the analyzer package" — that was **false**: exactly two test files
imported `dl_techniques.visualization` at all (`test_train/test_power_mlp`, and
`test_train/test_mothnet`), and neither exercises the analyzer path.

**A plugin is unreachable unless BOTH halves exist**, and the missing half is almost always
the second one:

| container | constructed by | stranded plugins |
|---|---|---|
| `TrainingHistory` | live, every trainer | none |
| `ClassificationResults` | `create_classification_results`; `train_mothnet.py` | 5 of 7 |
| `RegressionResults` | `create_regression_results` / `create_multi_model_regression` (`train/common/evaluation.py`) | 6 |
| `TimeSeriesEvaluationResults` | `create_timeseries_results` (`train/common/evaluation.py`) | `ForecastVisualization` |

So before registering a plugin, check that something builds its input container. Two further
traps, both pinned in `tests/test_train/test_common_evaluation_results.py`:

- **A template key must equal the class's own `name`.** `visualize` instantiates the
  template, registers it under `plugin.name`, then re-reads `self.plugins[plugin_name]`
  (`core.py:479-483`) — a disagreeing key misses that lookup with a bare `KeyError`.
  `ErrorAnalysisDashboard` is named `"error_analysis"`, *not* `"error_analysis_dashboard"`.
- **`ROCPRCurves` yields nothing without `y_prob`, and says so only in the log.** Its
  per-model loop skips such a model (`classification.py:269`), `legend()` then raises
  'No artists with labels found', and `visualize` swallows it and returns `None` — no file,
  no exception at the call site. Every other classification plugin degrades visibly.
