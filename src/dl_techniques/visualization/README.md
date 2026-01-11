# Visualization Framework

This document provides a comprehensive guide to using the `visualization` framework, a modular library for creating publication-quality plots for machine learning projects. This guide covers everything from basic usage to advanced dashboards and custom plugin development.

## Table of Contents

1.  [Core Concepts](#core-concepts)
2.  [Quick Start](#quick-start)
3.  [Available Visualizations](#available-visualizations)
    -   [Training and Performance](#training-and-performance)
    -   [Classification Analysis](#classification-analysis)
    -   [Data and Neural Network Inspection](#data-and-neural-network-inspection)
    -   [Time Series Analysis](#time-series-analysis)
4.  [Visualization Cookbook](#visualization-cookbook)
    -   [Visualizing Training and Performance](#visualizing-training-and-performance-1)
    -   [Comparing Multiple Models](#comparing-multiple-models)
    -   [Analyzing Classification Results](#analyzing-classification-results)
    -   [Analyzing Time Series Forecasts](#analyzing-time-series-forecasts)
    -   [Inspecting Neural Networks](#inspecting-neural-networks)
5.  [Advanced Usage](#advanced-usage)
    -   [Customizing Plot Appearance](#customizing-plot-appearance)
    -   [Creating Multi-Plot Dashboards](#creating-multi-plot-dashboards)
6.  [Extending the Framework](#extending-the-framework)

---

## Core Concepts

The framework is built on three fundamental components:

1.  **`VisualizationManager`**: The central orchestrator. It manages configuration, discovers plugins, and routes your data to the appropriate visualization template. An instance is typically created for each experiment.

2.  **`VisualizationPlugin`**: The abstract base class for all visualizations. Each template (e.g., `ConfusionMatrixVisualization`) is a plugin that defines what kind of data it accepts (`can_handle`) and how to render it (`create_visualization`).

3.  **Data Structures**: A set of `dataclasses` (e.g., `TrainingHistory`, `ClassificationResults`) that serve as standardized containers for your data, ensuring compatibility between your data and the visualization plugins.

---

## Quick Start

This example demonstrates the core workflow: defining data, initializing a manager, registering a template, and generating a plot.

```python
import numpy as np
from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    TrainingCurvesVisualization
)

# 1. Define your data using the provided data structures.
history = TrainingHistory(
    epochs=list(range(50)),
    train_loss=np.exp(-np.linspace(0, 2, 50)),
    val_loss=np.exp(-np.linspace(0, 1.8, 50)) + 0.1,
    train_metrics={'accuracy': 1 - np.exp(-np.linspace(0, 3, 50))},
    val_metrics={'accuracy': 1 - np.exp(-np.linspace(0, 2.5, 50))}
)

# 2. Initialize the Visualization Manager for your experiment.
viz_manager = VisualizationManager(
    experiment_name="quick_start_experiment",
    output_dir="visualizations_output"  # Plots will be saved here
)

# 3. Register the visualization template you intend to use.
viz_manager.register_template("training_curves", TrainingCurvesVisualization)

# 4. Generate and save the visualization.
# The plot is saved automatically to the output directory.
viz_manager.visualize(
    data=history,
    plugin_name="training_curves",
    show=True  # Set to True to display the plot interactively.
)

print("Visualization created successfully!")
```

---

## Available Visualizations

This section provides a detailed reference for all built-in visualization plugins.

### Training and Performance

*Visualizations from `training_performance.py` for analyzing model training dynamics and comparing performance.*

| Plugin (`plugin_name`) | Description | Required Data | Key Options |
| :--- | :--- | :--- | :--- |
| **`TrainingCurvesVisualization`**<br/>`"training_curves"` | Plots training and validation loss/metrics over epochs to monitor model learning. | `TrainingHistory` or `Dict[str, TrainingHistory]` | `metrics_to_plot: List[str]` (default: all)<br/>`smooth_factor: float` (0-1, default: 0)<br/>`show_best_epoch: bool` (default: `True`) |
| **`LearningRateScheduleVisualization`**<br/>`"lr_schedule"` | Shows how the learning rate changes over epochs, helping to debug schedulers. | `List[float]` or `Dict[str, List[float]]` | `show_phases: bool` (default: `True`)<br/>`phase_boundaries: List[int]` |
| **`ModelComparisonBarChart`**<br/>`"model_comparison_bars"` | Creates grouped bar charts to compare final performance metrics across multiple models. | `ModelComparison` or `Dict[str, Dict[str, float]]` | `metrics_to_show: List[str]` (default: all)<br/>`sort_by: str` (sorts by a metric)<br/>`show_values: bool` (default: `True`) |
| **`PerformanceRadarChart`**<br/>`"performance_radar"` | Generates a radar (or spider) chart to visualize trade-offs in multiple performance metrics between models. | `ModelComparison` or `Dict[str, Dict[str, float]]` | `metrics_to_show: List[str]` (default: all)<br/>`normalize: bool` (scales axes to best model, default: `True`) |
| **`ConvergenceAnalysis`**<br/>`"convergence_analysis"` | A dashboard analyzing loss convergence, gradient flow, and validation gap to diagnose training stability. | `TrainingHistory` or `Dict[str, TrainingHistory]` | A composite dashboard with subplots for:<br/>- Loss Convergence<br/>- Gradient Flow<br/>- Validation Gap<br/>- Convergence Rate<br/>*Note: Gradient Flow requires `grad_norms` in the data.* |
| **`OverfittingAnalysis`**<br/>`"overfitting_analysis"` | A dashboard that identifies when and how a model starts overfitting by analyzing the generalization gap. | `TrainingHistory` or `Dict[str, TrainingHistory]` | `patience: int` (epochs to wait before marking overfit point, default: 10) |
| **`PerformanceDashboard`**<br/>`"performance_dashboard"` | A comprehensive dashboard summarizing model performance with curves, heatmaps, and rankings. | `ModelComparison` | A comprehensive dashboard with subplots for:<br/>- Training Curves<br/>- Metric Comparison<br/>- Performance Heatmap<br/>- Ranking Table<br/>- Statistical Comparison<br/>`metric_to_display: str` (selects a metric for the bar chart) |

### Classification Analysis

*Visualizations from `classification.py` for evaluating the performance of classification models.*

| Plugin (`plugin_name`) | Description | Required Data | Key Options |
| :--- | :--- | :--- | :--- |
| **`ConfusionMatrixVisualization`**<br/>`"confusion_matrix"` | Displays a heatmap of true vs. predicted labels to show classification accuracy and error patterns per class. | `ClassificationResults` or `MultiModelClassification` | `normalize: str` (`'true'`, `'pred'`, `'all'`, default: `'true'`)<br/>`show_percentages: bool` (default: `True`)<br/>`cmap: str` (default: `'Blues'`) |
| **`ROCPRCurves`**<br/>`"roc_pr_curves"` | Plots ROC and/or Precision-Recall curves to evaluate a classifier's performance across different thresholds. | `ClassificationResults` or `MultiModelClassification` | `plot_type: str` (`'roc'`, `'pr'`, `'both'`, default: `'both'`)<br/>`show_thresholds: bool` (annotates curves with thresholds, default: `False`)<br/>*Note: Requires `y_prob` probability data.* |
| **`ClassificationReportVisualization`**<br/>`"classification_report"` | Presents a heatmap of key classification metrics (precision, recall, F1-score) for each class. | `ClassificationResults` or `MultiModelClassification` | `metrics: List[str]` (default: `['precision', 'recall', 'f1-score']`) |
| **`PerClassAnalysis`**<br/>`"per_class_analysis"` | A dashboard offering a deep dive into per-class performance, highlighting class distribution, accuracy, and common confusions. | `ClassificationResults` or `MultiModelClassification` | A composite dashboard with subplots for:<br/>- Class Distribution<br/>- Per-Class Accuracy<br/>- Class Confusion<br/>- Hardest Examples |
| **`ErrorAnalysisDashboard`**<br/>`"error_analysis"` | A detailed dashboard that analyzes misclassifications, showing error rates, confidence distributions, and error hotspots. | `ClassificationResults` | `show_examples: bool` (plots misclassified data)<br/>`x_data: np.ndarray` (raw input data, needed for `show_examples`)<br/>A dashboard with error rates, confidence analysis, and confusion hotspots. |

### Data and Neural Network Inspection

*Visualizations from `data_nn.py` for inspecting datasets and neural network internals.*

| Plugin (`plugin_name`) | Description | Required Data | Key Options |
| :--- | :--- | :--- | :--- |
| **`DataDistributionAnalysis`**<br/>`"data_distribution"` | Generates plots (histograms, KDE, etc.) to show the statistical distribution of features in a dataset. | `DatasetInfo`, `np.ndarray`, `pd.DataFrame` | `features_to_plot: List[int]` (default: first 12)<br/>`plot_type: str` (`'hist'`, `'kde'`, `'box'`, `'violin'`, `'auto'`) |
| **`ClassBalanceVisualization`**<br/>`"class_balance"` | Creates pie and bar charts to show the proportion of each class in the dataset, highlighting potential imbalances. | `DatasetInfo` or `Tuple[np.ndarray, np.ndarray]` | `show_percentages: bool` (default: `True`) |
| **`NetworkArchitectureVisualization`**<br/>`"network_architecture"` | Generates a diagram of the model's architecture, showing layers, shapes, and parameter counts. | `keras.Model` | `show_params: bool` (default: `True`)<br/>`show_shapes: bool` (default: `True`)<br/>`orientation: str` (`'vertical'` or `'horizontal'`) |
| **`ActivationVisualization`**<br/>`"activations"` | Plots the distribution or heatmap of activations from a model's layers to diagnose issues like dead neurons. | `ActivationData` | `layers_to_show: List[str]` (default: first 6)<br/>`plot_type: str` (`'distribution'`, `'heatmap'`, `'stats'`) |
| **`WeightVisualization`**<br/>`"weights"` | Shows the distribution, matrix, or filters of a model's weights to inspect what the model has learned. | `WeightData` or `keras.Model` | `layers_to_show: List[str]` (default: first 6)<br/>`plot_type: str` (`'distribution'`, `'matrix'`, `'filters'`) |
| **`FeatureMapVisualization`**<br/>`"feature_maps"` | Displays the output feature maps from convolutional layers to see what features the network detects in an input sample. | `ActivationData` | `sample_idx: int` (default: 0)<br/>`layers_to_show: List[str]` (default: first 4 conv layers)<br/>`max_features: int` (default: 16) |
| **`GradientVisualization`**<br/>`"gradients"` | Plots gradient norms or distributions across layers to diagnose vanishing or exploding gradient problems. | `GradientData` | `plot_type: str` (`'flow'`, `'distribution'`, `'vanishing'`) |
| **`GradientTopologyVisualization`**<br/>`"gradient_topology"` | Visualizes the entire model's gradient flow as a topological heatmap, showing how gradients propagate between connected layers. | `GradientTopologyData` | `target_size: int` (default: 64)<br/>`aggregation: str` (`'norm'`, `'mean'`, `'max'`)<br/>`log_scale: bool` (default: `True`)<br/>`cmap: str` (default: `'viridis'`) |
| **`GenericMatrixVisualization`**<br/>`"generic_matrix"` | Renders any 2D NumPy array as a heatmap, useful for correlation matrices or custom data. | `MatrixData` or `np.ndarray` | `title: str`<br/>`annot: bool` (default: `True`)<br/>`fmt: str` (default: `'.2f'`)<br/>`xticklabels: List[str]` |
| **`ImageComparisonVisualization`**<br/>`"image_comparison"` | Displays a list of images side-by-side, ideal for comparing original vs. reconstructed images or data augmentations. | `ImageData` or `List[np.ndarray]` | `titles: List[str]`<br/>`super_title: str`<br/>`cmap: str` (default: `'gray'`) |

### Time Series Analysis

*Visualizations from `time_series.py` for evaluating the performance of forecasting models.*

| Plugin (`plugin_name`) | Description | Required Data | Key Options |
| :--- | :--- | :--- | :--- |
| **`ForecastVisualization`**<br/>`"forecast_visualization"` | Plots sample forecasts against true values, with optional support for visualizing uncertainty via prediction quantiles. | `TimeSeriesEvaluationResults` | `num_samples: int` (default: 6)<br/>`plot_type: str` (`'auto'`, `'point'`, `'quantile'`) |

---

## Visualization Cookbook

This section provides practical, self-contained examples for common visualization tasks.

### Visualizing Training and Performance

#### Training and Validation Curves (`training_curves`)

Visualize how a model's loss and metrics evolve over epochs.

```python
# Assumes `history` and `viz_manager` from the Quick Start example.
viz_manager.visualize(
    data=history,
    plugin_name="training_curves",
    smooth_factor=0.1,  # Apply light exponential smoothing to the curves.
    show=True
)
```

#### Learning Rate Schedule (`lr_schedule`)

Visualize a learning rate schedule to debug schedulers like cosine annealing or step decay.

```python
from dl_techniques.visualization import LearningRateScheduleVisualization

# Data: A list of LR values.
lr_data = np.concatenate([
    np.linspace(1e-3, 1e-4, 50),
    np.linspace(1e-4, 1e-5, 50)
])

viz_manager.register_template("lr_schedule", LearningRateScheduleVisualization)
viz_manager.visualize(
    data=lr_data.tolist(),
    plugin_name="lr_schedule",
    show=True
)
```

### Comparing Multiple Models

#### Bar and Radar Charts (`model_comparison_bars`, `performance_radar`)

Compare final metrics across multiple models using bar charts for direct comparison or radar charts for visualizing trade-offs.

```python
from dl_techniques.visualization import (
    ModelComparison,
    ModelComparisonBarChart,
    PerformanceRadarChart
)

# Data: A ModelComparison object.
comparison_data = ModelComparison(
    model_names=["ResNet50", "VGG16", "EfficientNet"],
    metrics={
        "ResNet50": {"accuracy": 0.94, "f1_score": 0.93, "precision": 0.95},
        "VGG16": {"accuracy": 0.91, "f1_score": 0.90, "precision": 0.92},
        "EfficientNet": {"accuracy": 0.95, "f1_score": 0.94, "precision": 0.96},
    }
)

viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
viz_manager.register_template("performance_radar", PerformanceRadarChart)

# Create a bar chart sorted by accuracy.
viz_manager.visualize(
    data=comparison_data,
    plugin_name="model_comparison_bars",
    sort_by="accuracy",
    show=True
)

# Create a normalized radar chart.
viz_manager.visualize(
    data=comparison_data,
    plugin_name="performance_radar",
    normalize=True,
    show=True
)
```

### Analyzing Classification Results

The following examples use a `ClassificationResults` object, demonstrated below with sample data.

```python
import numpy as np
from dl_techniques.visualization import ClassificationResults

# Create sample prediction data.
y_true = np.random.randint(0, 3, 100)
y_pred = y_true.copy()
y_pred[np.random.choice(100, 15, replace=False)] = np.random.randint(0, 3, 15)
y_prob = np.random.rand(100, 3)
y_prob /= y_prob.sum(axis=1)[:, np.newaxis]

eval_data = ClassificationResults(
    y_true=y_true, y_pred=y_pred, y_prob=y_prob,
    class_names=["Cat", "Dog", "Bird"], model_name="MyClassifier"
)
```

#### Confusion Matrix (`confusion_matrix`)

```python
from dl_techniques.visualization import ConfusionMatrixVisualization

viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
viz_manager.visualize(
    data=eval_data,
    plugin_name="confusion_matrix",
    normalize='true',  # Normalize by the number of true instances per class.
    show=True
)
```

#### Comparing Multiple Confusion Matrices

To visualize multiple models side-by-side (e.g., comparing a Baseline to a New Model), use the `MultiModelClassification` container.

```python
from dl_techniques.visualization import (
    ClassificationResults, 
    MultiModelClassification,
    ConfusionMatrixVisualization
)

# Assume results_A and results_B are existing ClassificationResults objects
# for two different models.
multi_model_data = MultiModelClassification(
    results={
        "Baseline Model": results_A, 
        "New Model": results_B
    },
    dataset_name="Test Set"
)

viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
viz_manager.visualize(
    data=multi_model_data,
    plugin_name="confusion_matrix",
    normalize='true', 
    show=True
)
```

#### ROC and PR Curves (`roc_pr_curves`)

```python
from dl_techniques.visualization import ROCPRCurves

viz_manager.register_template("roc_pr_curves", ROCPRCurves)
viz_manager.visualize(
    data=eval_data,
    plugin_name="roc_pr_curves",
    plot_type='both',  # Generate both ROC and Precision-Recall curves.
    show=True
)
```

### Analyzing Time Series Forecasts

This example demonstrates how to visualize both a simple point forecast and a more complex probabilistic (quantile) forecast.

```python
import numpy as np
from dl_techniques.visualization import (
    TimeSeriesEvaluationResults,
    ForecastVisualization,
    VisualizationManager
)

# --- 1. Generate Sample Data ---
# Let's create dummy data for 100 samples
# Input length = 50, Forecast length = 12
num_samples = 100
input_len = 50
forecast_len = 12
quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

all_inputs = np.random.randn(num_samples, input_len)
all_true_forecasts = np.random.randn(num_samples, forecast_len)

# Point forecasts (e.g., from a model like N-BEATS)
all_point_forecasts = all_true_forecasts + np.random.randn(num_samples, forecast_len) * 0.5

# Quantile forecasts (e.g., from a model like TiRex)
# Shape: (num_samples, num_quantiles, forecast_len)
all_quantile_forecasts = np.random.randn(num_samples, len(quantile_levels), forecast_len)
all_quantile_forecasts.sort(axis=1) # Ensure quantiles are ordered

# --- 2. Create Data Containers ---
# a) For a point forecasting model
point_forecast_data = TimeSeriesEvaluationResults(
    all_inputs=all_inputs,
    all_true_forecasts=all_true_forecasts,
    all_predicted_forecasts=all_point_forecasts,
    model_name="N-BEATS_Example"
)

# b) For a probabilistic forecasting model
quantile_forecast_data = TimeSeriesEvaluationResults(
    all_inputs=all_inputs,
    all_true_forecasts=all_true_forecasts,
    all_predicted_quantiles=all_quantile_forecasts,
    quantile_levels=quantile_levels,
    model_name="TiRex_Example"
)

# --- 3. Visualize ---
viz_manager = VisualizationManager(experiment_name="time_series_cookbook")
viz_manager.register_template("forecast_visualization", ForecastVisualization)

# a) Visualize the point forecast
print("Visualizing point forecast...")
viz_manager.visualize(
    data=point_forecast_data,
    plugin_name="forecast_visualization",
    show=True
)

# b) Visualize the quantile forecast
print("Visualizing quantile forecast...")
viz_manager.visualize(
    data=quantile_forecast_data,
    plugin_name="forecast_visualization",
    show=True
)
```

### Inspecting Neural Networks

#### Network Architecture (`network_architecture`)

Generate a high-level visual summary of a model's layers and parameters.

```python
import keras
from dl_techniques.visualization import NetworkArchitectureVisualization

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax"),
])

viz_manager.register_template("network_architecture", NetworkArchitectureVisualization)
viz_manager.visualize(model, plugin_name="network_architecture", show=True)
```

---

## Advanced Usage

### Customizing Plot Appearance

Global visual styles can be configured by passing a `PlotConfig` object during the manager's initialization.

```python
from dl_techniques.visualization import PlotConfig, PlotStyle, ColorScheme

# Define a custom configuration for publication-quality PDF outputs.
config = PlotConfig(
    style=PlotStyle.PUBLICATION,
    color_scheme=ColorScheme(primary="#2E86AB", secondary="#A23B72"),
    title_fontsize=18,
    save_format="pdf"
)

custom_viz_manager = VisualizationManager(
    experiment_name="custom_style_experiment",
    config=config
)

# All plots created with `custom_viz_manager` will now use this new style.
```

### Creating Multi-Plot Dashboards

Combine multiple visualizations into a single figure using the `create_dashboard` method. You can let the manager handle the layout automatically or specify a custom grid layout.

```python
from dl_techniques.visualization import ClassificationReportVisualization

# Assumes `history` and `eval_data` from previous examples exist.
# Register any needed templates.
viz_manager.register_template("classification_report", ClassificationReportVisualization)

# Map plugin names to the data they should visualize.
dashboard_data = {
    "training_curves": history,
    "confusion_matrix": eval_data,
    "classification_report": eval_data,
}

# Example 1: Automatic Layout
viz_manager.create_dashboard(data=dashboard_data, show=True)

# Example 2: Custom Layout
# Define a layout with (row, column) coordinates for each plot.
custom_layout = {
    "training_curves": (0, 0),
    "confusion_matrix": (0, 1),
    "classification_report": (1, 0), # Place this on the next row
}

viz_manager.create_dashboard(data=dashboard_data, layout=custom_layout, show=True)
```

---

## Extending the Framework

New visualizations can be added by creating a custom plugin. This involves inheriting from `VisualizationPlugin` and implementing three required properties/methods.

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional
from dl_techniques.visualization import VisualizationPlugin, VisualizationManager

class SimpleScatterPlugin(VisualizationPlugin):
    """A custom plugin to create a 2D scatter plot."""

    @property
    def name(self) -> str:
        return "simple_scatter"

    @property
    def description(self) -> str:
        return "Creates a 2D scatter plot from a tuple of two numpy arrays."

    def can_handle(self, data: Any) -> bool:
        # This plugin handles a tuple of two 1D numpy arrays of the same length.
        return (isinstance(data, tuple) and len(data) == 2 and
                isinstance(data[0], np.ndarray) and data[0].ndim == 1 and
                isinstance(data[1], np.ndarray) and data[1].ndim == 1 and
                len(data[0]) == len(data[1]))

    def create_visualization(
        self,
        data: Any,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.fig_size, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()

        x_data, y_data = data
        ax.scatter(x_data, y_data, alpha=0.6, color=self.config.color_scheme.primary)
        ax.set_title(kwargs.get("title", "Scatter Plot"))
        ax.set_xlabel(kwargs.get("xlabel", "X-axis"))
        ax.set_ylabel(kwargs.get("ylabel", "Y-axis"))
        ax.grid(True, alpha=0.3)
        return fig

# --- How to use the custom plugin ---

# 1. Initialize a manager.
viz_manager = VisualizationManager(experiment_name="custom_plugin_test")

# 2. Register the new plugin template with the manager.
viz_manager.register_template("scatter", SimpleScatterPlugin)

# 3. Create data and generate the visualization.
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)

viz_manager.visualize(
    data=(x, y),
    plugin_name="scatter", # Use the name you defined in the plugin
    title="Custom Scatter Plot",
    xlabel="Independent Var",
    ylabel="Dependent Var",
    show=True
)
```
