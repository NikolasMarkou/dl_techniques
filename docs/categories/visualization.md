# Visualization

Visualization utilities

**7 modules in this category**

## Classification

### visualization.classification
Classification and Evaluation Visualization Templates

**Classes:**
- `ClassificationResults`
- `MultiModelClassification`
- `ConfusionMatrixVisualization`
- `ROCPRCurves`
- `ClassificationReportVisualization`
- `PerClassAnalysis`
- `ErrorAnalysisDashboard`

**Functions:** `name`, `description`, `can_handle`, `create_visualization`, `name` (and 14 more)

*📁 File: `src/dl_techniques/visualization/classification.py`*

## Core

### visualization

*📁 File: `src/dl_techniques/visualization/__init__.py`*

### visualization.core
Visualization Framework Core Module

**Classes:**
- `PlotStyle`
- `ColorScheme`
- `PlotConfig`
- `VisualizationContext`
- `VisualizationPlugin`
- `CompositeVisualization`
- `VisualizationManager`

**Functions:** `setup_logging`, `create_color_palette`, `adjust_lightness`, `get_model_color`, `get_style_params` (and 18 more)

*📁 File: `src/dl_techniques/visualization/core.py`*

## Data_Nn

### visualization.data_nn
Data Distribution and Neural Network Visualization Templates

**Classes:**
- `DatasetInfo`
- `ActivationData`
- `WeightData`
- `GradientData`
- `GradientTopologyData`
- `MatrixData`
- `ImageData`
- `GradientTopologyVisualizer`
- `DataDistributionAnalysis`
- `ClassBalanceVisualization`
- `NetworkArchitectureVisualization`
- `ActivationVisualization`
- `WeightVisualization`
- `FeatureMapVisualization`
- `GradientVisualization`
- `GradientTopologyVisualization`
- `GenericMatrixVisualization`
- `ImageComparisonVisualization`

**Functions:** `create_gradient_heatmap`, `get_statistics`, `name`, `description`, `can_handle` (and 37 more)

*📁 File: `src/dl_techniques/visualization/data_nn.py`*

## Regression

### visualization.regression
Regression Visualization Templates

**Classes:**
- `RegressionResults`
- `MultiModelRegression`
- `PredictionErrorVisualization`
- `ResidualsPlotVisualization`
- `ResidualDistributionVisualization`
- `QQPlotVisualization`
- `RegressionEvaluationDashboard`

**Functions:** `name`, `description`, `can_handle`, `create_visualization`, `name` (and 15 more)

*📁 File: `src/dl_techniques/visualization/regression.py`*

## Time_Series

### visualization.time_series
Time Series Forecasting Visualization Templates

**Classes:**
- `TimeSeriesEvaluationResults`
- `ForecastVisualization`

**Functions:** `name`, `description`, `can_handle`, `create_visualization`

*📁 File: `src/dl_techniques/visualization/time_series.py`*

## Training_Performance

### visualization.training_performance
Training and Performance Visualization Templates

**Classes:**
- `TrainingHistory`
- `ModelComparison`
- `TrainingCurvesVisualization`
- `LearningRateScheduleVisualization`
- `ModelComparisonBarChart`
- `PerformanceRadarChart`
- `ConvergenceAnalysis`
- `OverfittingAnalysis`
- `PerformanceDashboard`

**Functions:** `name`, `description`, `can_handle`, `create_visualization`, `name` (and 22 more)

*📁 File: `src/dl_techniques/visualization/training_performance.py`*