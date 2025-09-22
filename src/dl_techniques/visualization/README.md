# Visualization Framework

This document provides a comprehensive guide to using the `visualization` framework, a modular library for creating publication-quality plots for machine learning projects. This guide covers everything from basic usage to advanced dashboards and custom plugin development.

## Table of Contents

1.  [Core Concepts](#core-concepts)
2.  [Quick Start](#quick-start)
3.  [Visualization Cookbook](#visualization-cookbook)
    *   [Visualizing Training and Performance](#visualizing-training-and-performance)
    *   [Comparing Multiple Models](#comparing-multiple-models)
    *   [Analyzing Classification Results](#analyzing-classification-results)
    *   [Inspecting Neural Networks](#inspecting-neural-networks)
4.  [Specialized Visualizations](#specialized-visualizations)
    *   [Experiment Comparison Dashboard](#experiment-comparison-dashboard)
    *   [Loss Landscape Visualization](#loss-landscape-visualization)
    *   [Attention Head Visualization](#attention-head-visualization)
    *   [Embedding Space Visualization](#embedding-space-visualization)
5.  [Advanced Usage](#advanced-usage)
    *   [Customizing Plot Appearance](#customizing-plot-appearance)
    *   [Creating Multi-Plot Dashboards](#creating-multi-plot-dashboards)
6.  [End-to-End Workflow Example](#end-to-end-workflow-example)
7.  [Extending the Framework: Creating a Custom Plugin](#extending-the-framework-creating-a-custom-plugin)

## Core Concepts

The framework is built on three fundamental components:

1.  **`VisualizationManager`**: The central orchestrator. It manages configuration, discovers plugins, and routes your data to the appropriate visualization template. An instance is typically created for each experiment.

2.  **`VisualizationPlugin`**: The abstract base class for all visualizations. Each template (e.g., `ConfusionMatrixVisualization`) is a plugin that defines what kind of data it accepts (`can_handle`) and how to render it (`create_visualization`).

3.  **Data Structures**: A set of `dataclasses` (e.g., `TrainingHistory`, `ClassificationResults`) that serve as standardized containers for your data, ensuring compatibility between your data and the visualization plugins.

## Quick Start

This example demonstrates the core workflow: defining data, initializing a manager, registering a template, and generating a plot.

```python
import numpy as np
from visualization import VisualizationManager, TrainingHistory
from visualization import TrainingCurvesVisualization

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

This script produces a detailed plot showing training/validation loss and accuracy curves, saved within a versioned experiment directory.

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
from visualization import LearningRateScheduleVisualization

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
from visualization import ModelComparison, ModelComparisonBarChart, PerformanceRadarChart

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
from visualization import ClassificationResults

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
from visualization import ConfusionMatrixVisualization
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
from visualization import ROCPRCurves
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
from visualization import NetworkArchitectureVisualization

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

## Specialized Visualizations

The framework includes several advanced, specialized templates for deeper analysis.

### Experiment Comparison Dashboard

Compare results from multiple experimental runs, including training curves, final metrics, and hyperparameter impact.

**Data Format**: A dictionary containing an `experiments` key. This key holds another dictionary where each key is an experiment name and the value contains its `history`, `final_metrics`, and `hyperparameters`.

```python
from visualization import ExperimentComparisonDashboard
# Assume `history1`, `history2` are TrainingHistory objects
exp_data = {
    'experiments': {
        'exp_lr_0.01': {
            'history': {'loss': history1.val_loss, 'val_accuracy': history1.val_metrics['accuracy']},
            'final_metrics': {'accuracy': 0.92, 'loss': 0.15},
            'hyperparameters': {'learning_rate': 0.01, 'batch_size': 32}
        },
        'exp_lr_0.001': {
            'history': {'loss': history2.val_loss, 'val_accuracy': history2.val_metrics['accuracy']},
            'final_metrics': {'accuracy': 0.95, 'loss': 0.11},
            'hyperparameters': {'learning_rate': 0.001, 'batch_size': 32}
        }
    }
}
viz_manager.register_template("experiment_comparison", ExperimentComparisonDashboard)
viz_manager.visualize(exp_data, plugin_name="experiment_comparison", show=True)
```

### Loss Landscape Visualization

Visualize the loss surface around a trained model to understand properties like sharpness and flatness.

**Data Format**: A dictionary containing a `landscape` key, which holds grid coordinates (`X`, `Y`) and loss values (`Z`).

```python
from visualization import LossLandscapeVisualization
# Data is typically generated by sampling the loss function around model weights.
landscape_data = {
    'landscape': True, # Required key for can_handle
    'X': np.linspace(-1, 1, 50),
    'Y': np.linspace(-1, 1, 50),
    'Z': np.random.rand(50, 50), # Dummy loss values
    'model_pos': (0, 0)
}
viz_manager.register_template("loss_landscape", LossLandscapeVisualization)
viz_manager.visualize(landscape_data, plugin_name="loss_landscape", plot_type='2d', show=True)
```

### Attention Head Visualization

Inspect attention patterns in Transformer models by visualizing the attention weight matrix.

**Data Format**: A dictionary with an `attention_weights` key holding a NumPy array of shape `(batch, heads, seq_len, seq_len)`.

```python
from visualization import AttentionVisualization
attention_data = {
    'attention_weights': np.random.rand(1, 8, 10, 10), # B, H, L, L
    'tokens': [f'token_{i}' for i in range(10)]
}
viz_manager.register_template("attention", AttentionVisualization)
viz_manager.visualize(attention_data, plugin_name="attention", layer_idx=0, head_idx=1, show=True)
```

### Embedding Space Visualization

Visualize high-dimensional word or item embeddings in 2D using dimensionality reduction techniques.

**Data Format**: A dictionary with an `embeddings` key holding a NumPy array of shape `(num_items, embedding_dim)`.

```python
from visualization import EmbeddingVisualization
embedding_data = {
    'embeddings': np.random.rand(100, 64),
    'labels': np.random.randint(0, 5, 100),
    'words': [f'word_{i}' for i in range(100)]
}
viz_manager.register_template("embeddings", EmbeddingVisualization)
viz_manager.visualize(embedding_data, plugin_name="embeddings", method='tsne', show=True)
```

## Advanced Usage

### Customizing Plot Appearance

Global visual styles can be configured by passing a `PlotConfig` object during the manager's initialization.

```python
from visualization import PlotConfig, PlotStyle, ColorScheme

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
# Assumes `history` and `eval_data` from previous examples exist.
dashboard_data = {
    # Plugin Name -> Data for that plugin
    "training_curves": history,
    "confusion_matrix": eval_data,
    "classification_report": eval_data,
}

viz_manager.create_dashboard(data=dashboard_data, show=True)
```

## End-to-End Workflow Example

For complex projects, it is useful to encapsulate the visualization logic in a workflow class. This promotes reusability and standardization across experiments.

```python
from visualization import MLExperimentWorkflow # Assuming examples.py is accessible

# Initialize the workflow for a new experiment.
workflow = MLExperimentWorkflow("complete_example_run")

# --- Generate example data (replace with your actual data) ---
np.random.seed(42)
n_epochs = 100
history = TrainingHistory(
    epochs=list(range(n_epochs)),
    train_loss=np.exp(-np.linspace(0, 2, n_epochs)),
    val_loss=np.exp(-np.linspace(0, 1.8, n_epochs)),
    train_metrics={'accuracy': 1 - np.exp(-np.linspace(0, 3, n_epochs))},
    val_metrics={'accuracy': 1 - np.exp(-np.linspace(0, 2.5, n_epochs))}
)
y_true = np.random.randint(0, 10, 1000)
y_pred = y_true.copy()
error_mask = np.random.random(1000) < 0.15
y_pred[error_mask] = np.random.randint(0, 10, error_mask.sum())
# --- End of data generation ---

# Package all data into a dictionary.
all_data = {
    'training_history': history,
    'predictions': {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': None, # No probability data in this example
        'class_names': [f'Class_{i}' for i in range(10)]
    }
}

# Generate a complete report with multiple visualizations.
workflow.create_full_report(all_data)
```

## Extending the Framework: Creating a Custom Plugin

New visualizations can be added by creating a plugin. This involves inheriting from `VisualizationPlugin` and implementing three required properties/methods.

```python
import matplotlib.pyplot as plt
from typing import Any, Optional

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