# Visualization Framework

This document provides a comprehensive guide to using the `visualization` framework, a modular library for creating publication-quality plots for machine learning projects. This guide covers everything from basic usage to advanced dashboards and custom plugin development.

## Table of Contents

1.  [Core Concepts](#core-concepts)
2.  [Quick Start](#quick-start)
3.  [Available Visualizations and Options](#available-visualizations-and-options)
    *   [Training and Performance](#training-and-performance)
    *   [Classification Analysis](#classification-analysis)
    *   [Data and Neural Network Inspection](#data-and-neural-network-inspection)
4.  [Visualization Cookbook](#visualization-cookbook)
    *   [Visualizing Training and Performance](#visualizing-training-and-performance-1)
    *   [Comparing Multiple Models](#comparing-multiple-models)
    *   [Analyzing Classification Results](#analyzing-classification-results)
    *   [Inspecting Neural Networks](#inspecting-neural-networks)
5.  [Advanced Usage](#advanced-usage)
    *   [Customizing Plot Appearance](#customizing-plot-appearance)
    *   [Creating Multi-Plot Dashboards](#creating-multi-plot-dashboards)
6.  [Extending the Framework: Creating a Custom Plugin](#extending-the-framework-creating-a-custom-plugin)

## Core Concepts

The framework is built on three fundamental components:

1.  **`VisualizationManager`**: The central orchestrator. It manages configuration, discovers plugins, and routes your data to the appropriate visualization template. An instance is typically created for each experiment.

2.  **`VisualizationPlugin`**: The abstract base class for all visualizations. Each template (e.g., `ConfusionMatrixVisualization`) is a plugin that defines what kind of data it accepts (`can_handle`) and how to render it (`create_visualization`).

3.  **Data Structures**: A set of `dataclasses` (e.g., `TrainingHistory`, `ClassificationResults`) that serve as standardized containers for your data, ensuring compatibility between your data and the visualization plugins.

## Quick Start

This example demonstrates the core workflow: defining data, initializing a manager, registering a template, and generating a plot.

```python
import numpy as np
from dl_techniques.visualization import VisualizationManager, TrainingHistory
from dl_techniques.visualization import TrainingCurvesVisualization

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
    output_dir="visualizations_output" # Plots will be saved here
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

## Available Visualizations and Options

This section provides a detailed reference for all built-in visualization plugins available in the framework.

### Training and Performance
*From `training_performance.py`*

---
**`TrainingCurvesVisualization`** (`plugin_name="training_curves"`)
- **Description**: Visualize training and validation curves for loss and multiple metrics. Can also compare curves for multiple models.
- **Required Data**: `TrainingHistory` or `Dict[str, TrainingHistory]`.
- **Key Options**:
  - `metrics_to_plot: Optional[List[str]]`: List of metric names to plot (e.g., `['accuracy', 'f1_score']`). Defaults to all available metrics.
  - `smooth_factor: float`: Exponential smoothing factor (0 to 1). `0` means no smoothing.
  - `show_best_epoch: bool`: If `True`, marks the epoch with the best validation loss with a star.

---
**`LearningRateScheduleVisualization`** (`plugin_name="lr_schedule"`)
- **Description**: Visualize a learning rate schedule over training epochs.
- **Required Data**: `List[float]` or `Dict[str, List[float]]`.
- **Key Options**:
  - `show_phases: bool`: If `True`, shows vertical lines for phase changes.
  - `phase_boundaries: Optional[List[int]]`: A list of epoch indices where phase boundaries should be drawn.

---
**`ModelComparisonBarChart`** (`plugin_name="model_comparison_bars"`)
- **Description**: Compare final performance metrics of multiple models using grouped bar charts.
- **Required Data**: `ModelComparison`.
- **Key Options**:
  - `metrics_to_show: Optional[List[str]]`: List of metrics to include in the chart. Defaults to all.
  - `sort_by: Optional[str]`: Metric name to sort the models by (descending).
  - `show_values: bool`: If `True`, displays the numeric value on top of each bar.

---
**`PerformanceRadarChart`** (`plugin_name="performance_radar"`)
- **Description**: Compare models across multiple metrics using a radar chart to visualize trade-offs.
- **Required Data**: `ModelComparison`.
- **Key Options**:
  - `metrics_to_show: Optional[List[str]]`: List of metrics to use as axes on the chart.
  - `normalize: bool`: If `True`, normalizes each metric axis from 0 to 1 based on the best-performing model for that metric.

---
**`ConvergenceAnalysis`** (`plugin_name="convergence_analysis"`)
- **Description**: A composite dashboard for analyzing training convergence patterns.
- **Required Data**: `TrainingHistory` or `Dict[str, TrainingHistory]`.
- **Subplots**: Loss Convergence, Gradient Flow, Validation Gap, Convergence Rate.
- **Note**: For "Gradient Flow", the `TrainingHistory` object must contain a `grad_norms` dictionary with the key `"global_grad_norm"`.

---
**`OverfittingAnalysis`** (`plugin_name="overfitting_analysis"`)
- **Description**: A dashboard for detecting and visualizing overfitting through loss curves and generalization gaps.
- **Required Data**: `TrainingHistory` or `Dict[str, TrainingHistory]`.
- **Key Options**:
  - `patience: int`: Number of epochs with no improvement in validation loss before marking an epoch as the start of overfitting.

---
**`PerformanceDashboard`** (`plugin_name="performance_dashboard"`)
- **Description**: A comprehensive dashboard summarizing the performance of multiple models.
- **Required Data**: `ModelComparison`.
- **Subplots**: Training Curves, Metric Comparison, Performance Heatmap, Ranking Table, Statistical Comparison.
- **Key Options**:
  - `metric_to_display: Optional[str]`: The specific metric to show in the "Metric Comparison" bar chart. If `None`, shows a grouped bar chart of all metrics.

### Classification Analysis
*From `classification.py`*

---
**`ConfusionMatrixVisualization`** (`plugin_name="confusion_matrix"`)
- **Description**: Visualize a confusion matrix with detailed annotations and normalization options.
- **Required Data**: `ClassificationResults` or `MultiModelClassification`.
- **Key Options**:
  - `normalize: str`: Normalization mode. Can be `'true'` (rows), `'pred'` (columns), `'all'`, or `None`.
  - `show_percentages: bool`: If `True`, shows both the count and percentage in each cell.
  - `cmap: str`: The matplotlib colormap to use (e.g., `'Blues'`, `'Greens'`).

---
**`ROCPRCurves`** (`plugin_name="roc_pr_curves"`)
- **Description**: Visualize Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
- **Required Data**: `ClassificationResults` or `MultiModelClassification` (data must include `y_prob`).
- **Key Options**:
  - `plot_type: str`: Which curves to plot. Can be `'roc'`, `'pr'`, or `'both'`.
  - `show_thresholds: bool`: If `True`, annotates a few points on the curves with their decision thresholds.

---
**`ClassificationReportVisualization`** (`plugin_name="classification_report"`)
- **Description**: Display a `sklearn.metrics.classification_report` as a color-coded heatmap.
- **Required Data**: `ClassificationResults` or `MultiModelClassification`.
- **Key Options**:
  - `metrics: List[str]`: A list of metrics from the report to display. Defaults to `['precision', 'recall', 'f1-score']`.

---
**`PerClassAnalysis`** (`plugin_name="per_class_analysis"`)
- **Description**: A composite dashboard for detailed per-class performance analysis.
- **Required Data**: `ClassificationResults` or `MultiModelClassification`.
- **Subplots**: Class Distribution, Per-Class Accuracy, Class Confusion, Hardest Examples (confidence of errors).

---
**`ErrorAnalysisDashboard`** (`plugin_name="error_analysis"`)
- **Description**: A comprehensive dashboard for analyzing model prediction errors.
- **Required Data**: `ClassificationResults`.
- **Key Options**:
  - `show_examples: bool`: If `True`, attempts to plot examples of misclassified data.
  - `x_data: Optional[np.ndarray]`: The raw input data (e.g., images) needed for `show_examples`.

### Data and Neural Network Inspection
*From `data_nn.py`*

---
**`DataDistributionAnalysis`** (`plugin_name="data_distribution"`)
- **Description**: Analyze and visualize the distribution of features in a dataset.
- **Required Data**: `DatasetInfo`, `np.ndarray`, or `pd.DataFrame`.
- **Key Options**:
  - `features_to_plot: Optional[List[int]]`: List of feature indices to plot. Defaults to the first 12.
  - `plot_type: str`: Type of plot. Can be `'hist'`, `'kde'`, `'box'`, `'violin'`, or `'auto'`.

---
**`ClassBalanceVisualization`** (`plugin_name="class_balance"`)
- **Description**: Visualize class distribution and imbalance in a dataset.
- **Required Data**: `DatasetInfo` or `Tuple[np.ndarray, np.ndarray]` (for train/test labels).
- **Key Options**:
  - `show_percentages: bool`: If `True`, shows percentages on the pie chart.

---
**`NetworkArchitectureVisualization`** (`plugin_name="network_architecture"`)
- **Description**: Generate a high-level visual summary of a Keras model's layers, shapes, and parameters.
- **Required Data**: `keras.Model`.
- **Key Options**:
  - `show_params: bool`: If `True`, displays the parameter count for each layer.
  - `show_shapes: bool`: If `True`, displays the output shape for each layer.
  - `orientation: str`: Layout of the diagram. Can be `'vertical'` or `'horizontal'`.

---
**`ActivationVisualization`** (`plugin_name="activations"`)
- **Description**: Visualize the distribution or heatmap of activations from model layers.
- **Required Data**: `ActivationData`.
- **Key Options**:
  - `layers_to_show: Optional[List[str]]`: List of layer names to visualize. Defaults to the first 6.
  - `plot_type: str`: Type of plot. Can be `'distribution'`, `'heatmap'`, or `'stats'`.

---
**`WeightVisualization`** (`plugin_name="weights"`)
- **Description**: Visualize the distributions, matrices, or filters of model weights.
- **Required Data**: `WeightData` or `keras.Model`.
- **Key Options**:
  - `layers_to_show: Optional[List[str]]`: List of layer names to visualize.
  - `plot_type: str`: Type of plot. Can be `'distribution'`, `'matrix'`, or `'filters'` (for Conv2D layers).

---
**`FeatureMapVisualization`** (`plugin_name="feature_maps"`)
- **Description**: Visualize the feature maps (activations) produced by convolutional layers for a specific input sample.
- **Required Data**: `ActivationData`.
- **Key Options**:
  - `sample_idx: int`: The index of the sample in the batch to visualize.
  - `layers_to_show: Optional[List[str]]`: List of convolutional layers to show.
  - `max_features: int`: The maximum number of feature maps to display per layer.

---
**`GradientVisualization`** (`plugin_name="gradients"`)
- **Description**: Visualize gradient flow, distribution, and potential vanishing/exploding issues.
- **Required Data**: `GradientData`.
- **Key Options**:
  - `plot_type: str`: Type of analysis. Can be `'flow'` (norm vs. layer), `'distribution'`, or `'vanishing'` (% of near-zero gradients).

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

# Data: A dictionary mapping schedule names to lists of LR values.
lr_data = np.concatenate([
    np.linspace(1e-3, 1e-4, 50),
    np.linspace(1e-4, 1e-5, 50)
])

viz_manager.register_template("lr_schedule", LearningRateScheduleVisualization)
viz_manager.visualize(
    data={"Cosine Annealing": lr_data},
    plugin_name="lr_schedule",
    show=True
)
```

### Comparing Multiple Models

#### Bar Chart and Radar Comparison (`model_comparison_bars`, `performance_radar`)
Compare final metrics across multiple models using bar charts for direct comparison or radar charts for visualizing trade-offs.

```python
from dl_techniques.visualization import ModelComparison, ModelComparisonBarChart, PerformanceRadarChart

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
from dl_techniques.visualization import ClassificationResults

# Create sample prediction data.
y_true = np.random.randint(0, 3, 100)
y_pred = y_true.copy()
y_pred[np.random.choice(100, 15, replace=False)] = np.random.randint(0, 3, 15)
y_prob = np.random.rand(100, 3); y_prob /= y_prob.sum(axis=1)[:, np.newaxis]

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
# All plots created with `custom_viz_manager` will now use the new style.
```

### Creating Multi-Plot Dashboards

Combine multiple visualizations into a single figure using the `create_dashboard` method. The manager automatically handles the subplot layout.

```python
from dl_techniques.visualization import ClassificationReportVisualization

# Assumes `history` and `eval_data` from previous examples exist.
# Register any needed templates
viz_manager.register_template("classification_report", ClassificationReportVisualization)

dashboard_data = {
    # Plugin Name -> Data for that plugin
    "training_curves": history,
    "confusion_matrix": eval_data,
    "classification_report": eval_data,
}

viz_manager.create_dashboard(data=dashboard_data, show=True)
```

## Extending the Framework: Creating a Custom Plugin

New visualizations can be added by creating a plugin. This involves inheriting from `VisualizationPlugin` and implementing three required properties/methods.

```python
import matplotlib.pyplot as plt
from typing import Any, Optional
from dl_techniques.visualization import VisualizationPlugin

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

    def create_visualization(self, data: Any, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Figure:
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
# 1. Register the new plugin template with the manager.
viz_manager.register_template("scatter", SimpleScatterPlugin)

# 2. Create data and generate the visualization.
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)
viz_manager.visualize(
    data=(x, y),
    plugin_name="scatter",
    title="Custom Scatter Plot",
    show=True
)
```