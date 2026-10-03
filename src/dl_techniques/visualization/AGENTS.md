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

No dedicated test directory — visualization is typically tested through integration with the analyzer package.
