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

`tests/test_visualization/` — currently ONE guard,
`test_per_class_axis_alignment.py`. That is a thin suite: treat this package as
**under-tested, not untested**. Three further modules import it —
`test_train/test_mothnet/`, `test_train/test_power_mlp/`, and
`test_train/test_common_evaluation_results.py`.

This section previously claimed visualization is "typically tested through integration
with the analyzer package", and a `REPO_MAP.md` exception claimed the package had
"no test directory at all" and that "no test module anywhere imports it". Both were
**false** — exactly two test modules imported the package, and neither exercised the
analyzer path. Both claims are corrected; do not restore them.

**A plugin is unreachable unless BOTH halves exist**, and the missing half is almost always
the second one:

Counts below are **measured**, not estimated — `can_handle` is an `isinstance` check, so
each container's consumer set is countable:

| container | plugins that accept it | constructed by | state |
|---|---|---|---|
| `TrainingHistory` | 4 | live, every trainer | used |
| `ClassificationResults` | **5** | `create_classification_results` (dead); `train_mothnet.py` (live) | 4 of 5 now registered by mothnet; `ROCPRCurves` deliberately excluded |
| `RegressionResults` / `MultiModelRegression` | **5** | **nothing, before this change** | `create_regression_results` / `create_multi_model_regression` added; **no caller yet** |
| `TimeSeriesEvaluationResults` | **1** (`ForecastVisualization`) | **nothing, before this change** | `create_timeseries_results` added; **no caller yet** |

The five `RegressionResults` consumers are `prediction_error`, `residuals_plot`,
`residual_distribution`, `qq_plot` and `regression_evaluation_dashboard`; the five
`ClassificationResults` consumers are `confusion_matrix`, `roc_pr_curves`,
`classification_report`, `per_class_analysis` and `error_analysis`.

**The regression and forecast rows were unreachable by CONSTRUCTION before those three
builders existed** — nothing in the tree constructed `RegressionResults` or
`TimeSeriesEvaluationResults`, so adding a `register_template` call could not have
reached them. They are now *reachable*, which is not the same as *used*: as of this
commit no trainer calls the three builders, so those six plugins still have no runtime
consumer. That gap is deliberate and tracked, not an oversight.

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
  It does **not** write a blank PNG: `create_visualization` raises before `save_figure`
  is reached, so the artifact is *absent*, which is the harder failure to notice.

## Two live defects this package shipped, and what they teach

Both were latent in `PerClassAnalysis` and only surfaced once a real trainer registered the
plugin and a real run drew a real figure. Both are now guarded by
`tests/test_visualization/test_per_class_axis_alignment.py`.

- **`np.bincount` returns `max(label) + 1`, so two label arrays give two DIFFERENT
  lengths.** `y_true` reaching class 9 while `y_pred` stops at 8 yields counts of length
  10 and 9; `ax.bar` against a 10-wide axis raises `shape mismatch: ... (10,) ... (9,)`.
  The original code only ever *sliced* the counts down to `len(classes)`, which cannot
  lengthen the short one. Count arrays must be `np.pad`-ed, not sliced. This is the same
  shape as the `train/mothnet` D-007 decision, which fixed the confusion-matrix panel and
  never reached this one.
- **Tick labels and data marks must be reconciled against the DATA's class count, not
  against `class_names`.** `PerClassAnalysis._axis_classes` does this. A caller that
  derives `class_names` from the labels actually present — which the confusion matrix
  *requires* — can hold fewer names than there are classes, and `set_xticklabels` raises the
  same `shape mismatch`.

Also note this repo's `filterwarnings = ["error::UserWarning", ...]`: a bare
`ax.legend()` on unlabelled artists is a test failure here, not a cosmetic warning.
Label the series.
