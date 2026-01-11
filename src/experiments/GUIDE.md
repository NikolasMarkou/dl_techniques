# Experiment Structure Guide

This guide outlines the standard architectural pattern for creating machine learning experiments that integrate seamlessly with the **Visualization Framework**. Following this structure ensures you can generate publication-quality dashboards, avoid common pitfalls (like file overwriting), and utilize the full analytical power of the library.

---

## 1. The Architectural Pattern

The framework relies on an **Adapter Pattern**. Your experiment code should follow this flow:

1.  **Configuration**: Define hyperparameters using Dataclasses.
2.  **Training**: Run standard Keras/PyTorch loops.
3.  **Adaptation**: Convert raw outputs (arrays, dictionaries) into Framework Data Structures (`TrainingHistory`, `ClassificationResults`, etc.).
4.  **Visualization**: Pass these structures to the `VisualizationManager`.

---

## 2. Step-by-Step Implementation

### Step 1: Configuration & Setup
Always use `dataclasses` for configuration. It makes the code readable and type-safe.

```python
from dataclasses import dataclass
from pathlib import Path
from dl_techniques.visualization import PlotConfig, PlotStyle, VisualizationManager

@dataclass
class ExperimentConfig:
    experiment_name: str = "my_experiment"
    epochs: int = 50
    # ... other params ...

def main():
    config = ExperimentConfig()
    
    # 1. Initialize Manager
    viz_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        config=PlotConfig(style=PlotStyle.SCIENTIFIC)
    )
    
    # 2. Register Templates (Do this once at the start)
    from dl_techniques.visualization import TrainingCurvesVisualization, ConfusionMatrixVisualization
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
```

### Step 2: The Training Loop (Collection Phase)
As you train multiple models (e.g., in a loop), collect the raw objects in native Python dictionaries.

```python
trained_models = {}
raw_histories = {} # Store raw Keras history.history dicts

for model_name in ["Model_A", "Model_B"]:
    model = build_model(...)
    history = model.fit(...)
    
    trained_models[model_name] = model
    raw_histories[model_name] = history.history
```

### Step 3: Data Adaptation (The Critical Step)
Before visualizing, mapping your raw data to the framework's strict Data Structures is required.

#### A. Training History
```python
from dl_techniques.visualization import TrainingHistory

# Convert dicts to objects
history_objects = {}
for name, raw in raw_histories.items():
    history_objects[name] = TrainingHistory(
        epochs=list(range(len(raw['loss']))),
        train_loss=raw['loss'],
        val_loss=raw.get('val_loss'),
        train_metrics={'accuracy': raw.get('accuracy')},
        val_metrics={'accuracy': raw.get('val_accuracy')}
    )

# Visualize combined curves
viz_manager.visualize(
    data=history_objects, 
    plugin_name="training_curves", 
    filename="combined_training_dynamics"
)
```

#### B. Classification Results
To enable advanced features like ROC curves and Error Analysis, you **must** capture probabilities (`y_prob`).

```python
from dl_techniques.visualization import ClassificationResults

all_results = {}

for name, model in trained_models.items():
    # 1. Get raw predictions
    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)
    
    # 2. Wrap in Data Structure
    results = ClassificationResults(
        y_true=y_test_indices,
        y_pred=y_pred,
        y_prob=y_prob,          # <--- CRITICAL for ROC/PR curves
        class_names=class_names,
        model_name=name
    )
    all_results[name] = results

    # 3. Individual Plot (Use filename to prevent overwriting!)
    viz_manager.visualize(
        data=results,
        plugin_name="error_analysis",
        filename=f"error_analysis_{name}", # <--- UNIQUE FILENAME
        x_data=x_test # Optional: for showing image examples
    )
```

### Step 4: Multi-Model Comparison
Use container classes to group data for comparative visualizations.

```python
from dl_techniques.visualization import MultiModelClassification, ModelComparison

# 1. Classification Comparison (Confusion Matrices, ROC)
multi_data = MultiModelClassification(
    results=all_results,
    dataset_name="Test Set"
)

viz_manager.visualize(data=multi_data, plugin_name="confusion_matrix", normalize='true')
viz_manager.visualize(data=multi_data, plugin_name="roc_pr_curves")

# 2. Metrics Bar Charts / Radar Charts
comp_data = ModelComparison(
    model_names=list(performance_metrics.keys()),
    metrics=performance_metrics # Dict[model_name, Dict[metric_name, float]]
)

viz_manager.visualize(data=comp_data, plugin_name="performance_radar")
```

---

## 3. Best Practices & Pitfalls

### 1. File Overwriting (The Loop Trap)
**Problem:** Calling `viz_manager.visualize(..., plugin_name="error_analysis")` inside a loop will overwrite `error_analysis.png` repeatedly.
**Solution:** Always use the `filename` argument inside loops.
```python
viz_manager.visualize(..., filename=f"error_analysis_{model_name}")
```

### 2. Missing Probability Data
**Problem:** `ROCPRCurves` and `ErrorAnalysisDashboard` crash or show empty plots.
**Cause:** `ClassificationResults` was initialized without `y_prob`.
**Solution:** Always store the raw softmax outputs (`model.predict()`) and pass them to the data structure.

### 3. Folder Structure
Organize your experiment outputs using `pathlib` to keep runs distinct:
```text
results/
  └── experiment_name_TIMESTAMP/
      ├── visualizations/      # Created by VisualizationManager
      ├── checkpoints/         # Model weights
      ├── training_plots/      # Raw matplotlib plots (optional)
      └── model_analysis/      # Deep analyzer stats
```

### 4. Integration with ModelAnalyzer
The `ModelAnalyzer` is separate from the `VisualizationManager`.
*   **ModelAnalyzer**: Calculates heavy statistics (Spectral analysis, information flow, deep calibration metrics).
*   **VisualizationManager**: Standardizes plotting for presentation.
*   **Workflow**: Run `ModelAnalyzer` first to get JSON metrics, then use `VisualizationManager` to plot the performance summary.

---

## 4. Quick Reference: Data Structures

| Task | Framework Data Structure | Required Inputs |
| :--- | :--- | :--- |
| **Training** | `TrainingHistory` | `epochs`, `train_loss`, `val_loss`, `metrics` |
| **Classification** | `ClassificationResults` | `y_true`, `y_pred`, `y_prob`, `class_names` |
| **Multi-Class** | `MultiModelClassification` | Dict of `ClassificationResults` |
| **Comparison** | `ModelComparison` | `model_names`, `metrics` (nested dict) |
| **Weights** | `WeightData` | `layer_names`, `weights` (list of arrays) |
| **Time Series** | `TimeSeriesEvaluationResults` | `all_inputs`, `all_true`, `all_predicted` |
| **Regression** | `RegressionResults` | `y_true`, `y_pred` |

---

## 5. Experiment Template

Copy this skeleton for new experiments:

```python
import numpy as np
from dl_techniques.visualization import (
    VisualizationManager, TrainingHistory, ClassificationResults, 
    MultiModelClassification, TrainingCurvesVisualization, ConfusionMatrixVisualization
)

def run_experiment():
    # 1. Setup
    viz_manager = VisualizationManager(experiment_name="new_experiment")
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    # 2. Train Loop
    histories = {}
    results = {}
    
    for model_name in ["A", "B"]:
        # Train...
        # history = model.fit(...)
        # probs = model.predict(...)
        
        # Adapt History
        histories[model_name] = TrainingHistory(
            epochs=history.epoch, 
            train_loss=history.history['loss'],
            # ... map other metrics
        )
        
        # Adapt Results
        results[model_name] = ClassificationResults(
            y_true=..., y_pred=..., y_prob=probs, model_name=model_name
        )

    # 3. Visualize
    # Combined Training
    viz_manager.visualize(data=histories, plugin_name="training_curves")
    
    # Combined Confusion Matrix
    multi_data = MultiModelClassification(results=results)
    viz_manager.visualize(data=multi_data, plugin_name="confusion_matrix")
```