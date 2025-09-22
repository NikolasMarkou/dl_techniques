# ML Visualization Framework: Documentation and Usage Guide

This document provides a comprehensive guide to using the `visualization` framework, a modular library for creating publication-quality plots for machine learning projects.

## Getting Started: A Basic Example

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

This script produces a detailed plot showing the training/validation loss and accuracy curves, saved within a versioned experiment directory.

## Core Concepts

The framework is built on three fundamental components:

1.  **`VisualizationManager`**: The central orchestrator. It manages configuration, discovers plugins, and routes your data to the appropriate visualization template. An instance is typically created for each experiment.

2.  **`VisualizationPlugin`**: The abstract base class for all visualizations. Each template (e.g., `ConfusionMatrixVisualization`) is a plugin that defines what kind of data it accepts (`can_handle`) and how to render it (`create_visualization`).

3.  **Data Structures**: A set of `dataclasses` (e.g., `TrainingHistory`, `ClassificationResults`) that serve as standardized containers for your data, ensuring compatibility between your data and the visualization plugins.

---

## Usage Examples and Cookbook

This section provides practical, self-contained examples for common visualization tasks.

### Visualizing the Training Process

#### Training and Validation Curves (`training_curves`)
Visualize how a model's loss and metrics evolve over epochs.

```python
# Assumes `history` and `viz_manager` from the Getting Started example.
viz_manager.visualize(
    data=history,
    plugin_name="training_curves",
    smooth_factor=0.1,  # Apply light exponential smoothing to the curves.
    show=True
)
```

#### Learning Rate Schedule (`lr_schedule`)
Visualize a learning rate schedule to debug schedulers like cosine annealing or step decay. The data can be a list or a dictionary for comparing multiple schedules.

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

### Comparing Model Performance

#### Bar Chart Comparison (`model_comparison_bars`)
A bar chart provides a clear comparison of final metrics across multiple models.

```python
from visualization import ModelComparison, ModelComparisonBarChart

# Data: A ModelComparison object.
comparison_data = ModelComparison(
    model_names=["ResNet50", "VGG16", "EfficientNet"],
    metrics={
        "ResNet50": {"accuracy": 0.94, "f1_score": 0.93},
        "VGG16": {"accuracy": 0.91, "f1_score": 0.90},
        "EfficientNet": {"accuracy": 0.95, "f1_score": 0.94},
    }
)

viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
viz_manager.visualize(
    data=comparison_data,
    plugin_name="model_comparison_bars",
    sort_by="accuracy",  # Sort models by their accuracy score.
    show=True
)
```

### Analyzing Classification Results

The following examples use a `ClassificationResults` object, demonstrated below with sample data.

```python
from visualization import ClassificationResults

# Create sample prediction data for demonstration.
y_true = np.random.randint(0, 3, 100)
y_pred = y_true.copy()
y_pred[np.random.choice(100, 15, replace=False)] = np.random.randint(0, 3, 15) # Add errors
y_prob = np.random.rand(100, 3); y_prob /= y_prob.sum(axis=1)[:, np.newaxis] # Dummy probabilities

eval_data = ClassificationResults(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    class_names=["Cat", "Dog", "Bird"],
    model_name="MyClassifier"
)
```

#### Confusion Matrix (`confusion_matrix`)
Identify which classes a model confuses with each other.

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
Evaluate the trade-off between true positive rate and false positive rate.

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

#### Classification Report Heatmap (`classification_report`)
Render the `sklearn.metrics.classification_report` as an intuitive, color-coded heatmap.

```python
from visualization import ClassificationReportVisualization
viz_manager.register_template("classification_report", ClassificationReportVisualization)
viz_manager.visualize(
    data=eval_data,
    plugin_name="classification_report",
    show=True
)
```

### Inspecting Neural Networks

#### Network Architecture (`network_architecture`)
Generate a high-level visual summary of a model's layers, parameters, and output shapes.

```python
import keras
from visualization import NetworkArchitectureVisualization

# Data: A Keras model object.
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax"),
])

viz_manager.register_template("network_architecture", NetworkArchitectureVisualization)
viz_manager.visualize(model, plugin_name="network_architecture", show=True)
```

---

## Advanced Techniques

#### Customizing Plot Appearance
Global visual styles can be configured by passing a `PlotConfig` object during the manager's initialization.

```python
from visualization import PlotConfig, PlotStyle, ColorScheme

# Define a custom configuration for publication-quality PDF outputs.
config = PlotConfig(
    style=PlotStyle.PUBLICATION,
    color_scheme=ColorScheme(
        primary="#2E86AB",
        secondary="#A23B72",
        background="#F0F0F0"
    ),
    title_fontsize=18,
    save_format="pdf"  # Save all plots as PDFs.
)

# Initialize a new manager with this custom configuration.
custom_viz_manager = VisualizationManager(
    experiment_name="custom_style_experiment",
    config=config
)

# Any plot created with `custom_viz_manager` will now use the new style.
```

#### Comparing Multiple Models in a Single Plot
Most plugins automatically generate comparative plots when supplied with multi-model data structures like `MultiModelClassification`.

```python
from visualization import MultiModelClassification

# Assume `eval_data_model_A` and `eval_data_model_B` are ClassificationResults objects.
multi_model_data = MultiModelClassification(
    results={
        "Model A": eval_data_model_A,
        "Model B": eval_data_model_B,
    },
    dataset_name="CIFAR-10"
)

# The `roc_pr_curves` plugin will plot ROC curves for both models on the same axes.
viz_manager.visualize(
    data=multi_model_data,
    plugin_name="roc_pr_curves",
    plot_type="roc"
)
```

#### Creating Dashboards
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

---

## Extending the Framework: Creating a Custom Plugin

New visualizations can be added by creating a plugin. This involves inheriting from `VisualizationPlugin` and implementing three required properties/methods.

The following example demonstrates a simple scatter plot plugin.

```python
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Optional
from visualization import VisualizationPlugin, PlotConfig, VisualizationContext

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
        # Create a new figure if no axes are provided.
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
    xlabel="Feature A",
    ylabel="Feature B",
    show=True
)
```