# A Practical Guide to the ML Visualization Framework

Welcome! This guide will show you how to use the `visualization` framework to create publication-quality plots for your machine learning projects with minimal effort. We'll go from a simple "hello world" plot to building complex, automated analysis dashboards.

## 🚀 Quick Start: Your First Visualization

Let's generate a plot of training and validation curves. This entire process takes less than 10 lines of Python.

```python
import numpy as np
from visualization import VisualizationManager, TrainingHistory

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

# 3. Register the visualization template you want to use.
from visualization import TrainingCurvesVisualization
viz_manager.register_template("training_curves", TrainingCurvesVisualization)

# 4. Generate the visualization!
# The plot is saved automatically and can be displayed interactively.
viz_manager.visualize(
    data=history,
    plugin_name="training_curves",
    show=True  # Set to True to display the plot
)

print("Visualization created successfully!")
```

This simple script produces a detailed plot showing the training/validation loss and accuracy curves, saved neatly in its own versioned experiment directory.

## 🏛️ Core Concepts

The framework is built on three simple ideas:

1.  **`VisualizationManager`**: The central orchestrator. It manages configuration, discovers plugins, and routes your data to the right visualization template. You create one of these for each experiment.

2.  **`VisualizationPlugin`**: The blueprint for all visualizations. Each template (e.g., `ConfusionMatrixVisualization`) is a plugin that knows what kind of data it can handle (`can_handle`) and how to plot it (`create_visualization`).

3.  **Data Structures**: Simple Python `dataclasses` (e.g., `TrainingHistory`, `ClassificationResults`) that act as standardized containers for your data. This ensures your data is always in the format a plugin expects.

---

## 📚 Visualization Cookbook: Recipes for Common Tasks

This section provides practical, copy-paste-ready examples for the most common visualizations.

### 📈 Visualizing the Training Process

#### Training & Validation Curves (`training_curves`)
This is the most fundamental plot, showing how your model's loss and metrics evolve over epochs.

```python
# Data: A TrainingHistory object (created in Quick Start)
viz_manager.visualize(
    data=history,
    plugin_name="training_curves",
    smooth_factor=0.1,  # Apply light smoothing to the curves
    show=True
)
```

#### Learning Rate Schedule (`lr_schedule`)
Visualize how your learning rate changes over time, which is essential for debugging schedulers like cosine annealing or step decay.

```python
# Data: A list of learning rate values per epoch
from visualization import LearningRateScheduleVisualization

lr_data = np.concatenate([
    np.linspace(1e-3, 1e-4, 50),
    np.linspace(1e-4, 1e-5, 50)
])

viz_manager.register_template("lr_schedule", LearningRateScheduleVisualization)
viz_manager.visualize(
    data={"My LR Schedule": lr_data},
    plugin_name="lr_schedule",
    show=True
)
```

### ⚖️ Comparing Model Performance

#### Bar Chart Comparison (`model_comparison_bars`)
When you have final metrics for several models, a bar chart is the clearest way to compare them.

```python
from visualization import ModelComparison, ModelComparisonBarChart

# Data: A ModelComparison object
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
    sort_by="accuracy",  # Sort models by their accuracy score
    show=True
)
```

### 🎯 Analyzing Classification Results

For the following examples, let's assume you have a `ClassificationResults` object containing your model's predictions.

```python
from visualization import ClassificationResults

# Create some dummy prediction data
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
Instantly see which classes your model is confusing with each other.

```python
from visualization import ConfusionMatrixVisualization
viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
viz_manager.visualize(
    data=eval_data,
    plugin_name="confusion_matrix",
    normalize='true',  # Show percentages relative to the true class
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
    plot_type='both',  # Show both ROC and Precision-Recall curves
    show=True
)
```

#### Classification Report Heatmap (`classification_report`)
Turn a text-based `sklearn.metrics.classification_report` into an intuitive, color-coded heatmap.

```python
from visualization import ClassificationReportVisualization
viz_manager.register_template("classification_report", ClassificationReportVisualization)
viz_manager.visualize(
    data=eval_data,
    plugin_name="classification_report",
    show=True
)
```

### 🧠 Inspecting Neural Networks

#### Network Architecture (`network_architecture`)
Get a high-level, visual summary of your model's layers, parameters, and output shapes.

```python
import keras
from visualization import NetworkArchitectureVisualization

# Data: A Keras model object
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

## 🛠️ Advanced Techniques

#### Customizing Plot Appearance
You can override every visual aspect by passing a `PlotConfig` object to the manager. All subsequent plots will use your new style.

```python
from visualization import PlotConfig, PlotStyle, ColorScheme

# Define a custom color scheme and publication-quality style
config = PlotConfig(
    style=PlotStyle.PUBLICATION,
    color_scheme=ColorScheme(
        primary="#2E86AB",
        secondary="#A23B72",
        background="#F0F0F0"
    ),
    title_fontsize=18,
    save_format="pdf"  # Save all plots as PDFs
)

# Initialize a new manager with this configuration
custom_viz_manager = VisualizationManager(
    experiment_name="custom_style_experiment",
    config=config
)

# Any plot created with `custom_viz_manager` will now have the new look.
```

#### Comparing Multiple Models in One Plot
Most plugins automatically detect multi-model data structures and generate comparative plots. For example, using `ROCPRCurves` with `MultiModelClassification` data will plot ROC curves for all models on the same axes.

```python
from visualization import MultiModelClassification

# Assume you have 'eval_data_model_A' and 'eval_data_model_B'
multi_model_data = MultiModelClassification(
    results={
        "Model A": eval_data_model_A,
        "Model B": eval_data_model_B,
    },
    dataset_name="CIFAR-10"
)

# The plugin automatically detects the multi-model data and plots both
viz_manager.visualize(
    data=multi_model_data,
    plugin_name="roc_pr_curves",
    plot_type="roc"
)
```

#### Creating Dashboards
Combine several plots into a single figure for a comprehensive report. The manager handles the layout for you.

```python
# Assume you have 'history' and 'eval_data' from previous examples
dashboard_data = {
    # Plugin Name -> Data for that plugin
    "training_curves": history,
    "confusion_matrix": eval_data,
    "classification_report": eval_data,
}

viz_manager.create_dashboard(data=dashboard_data, show=True)
```

---

## 🧩 Extending the Framework: Creating Your Own Plugin

If you need a unique visualization, creating a new plugin is straightforward. Just inherit from `VisualizationPlugin` and implement three methods.

Here’s an example of a simple scatter plot plugin:

```python
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Optional
from visualization import VisualizationPlugin

class SimpleScatterPlugin(VisualizationPlugin):
    """A custom plugin to create a scatter plot."""

    @property
    def name(self) -> str:
        return "simple_scatter"

    @property
    def description(self) -> str:
        return "Creates a 2D scatter plot from two numpy arrays."

    def can_handle(self, data: Any) -> bool:
        # This plugin handles a tuple of two 1D numpy arrays of the same length
        return (isinstance(data, tuple) and len(data) == 2 and
                isinstance(data[0], np.ndarray) and data[0].ndim == 1 and
                isinstance(data[1], np.ndarray) and data[1].ndim == 1 and
                len(data[0]) == len(data[1]))

    def create_visualization(self, data: Any, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Figure:
        # Create a new figure if no axes are provided
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

# --- How to use your new plugin ---
# 1. Register the new plugin class with the manager
viz_manager.register_template("scatter", SimpleScatterPlugin)

# 2. Create data and visualize
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)
viz_manager.visualize(
    data=(x, y),
    plugin_name="scatter",
    title="My Custom Scatter Plot",
    show=True
)
```