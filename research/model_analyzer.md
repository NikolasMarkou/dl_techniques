# Guide to the Model Analyzer

A comprehensive, modular analysis toolkit for deep learning models built on Keras. This module provides multi-dimensional model analysis including weight distributions, calibration metrics, information flow patterns, and training dynamics, complete with publication-ready visualizations.

This guide will walk you through the setup, basic and advanced usage, and how to interpret the results and extend the analyzer for your own needs.

## 1. Core Concepts

Before diving in, it's important to understand the four key components you'll interact with:

-   **`ModelAnalyzer`**: The main orchestrator class. You initialize it with your models and a configuration, and it runs the entire analysis pipeline.
-   **`AnalysisConfig`**: A dataclass that holds all the settings for your analysis. You can use it to enable/disable specific analyses, set parameters (like the number of calibration bins), and configure plot styles.
-   **`DataInput`**: A simple, structured container for your input data (`x_data`) and labels (`y_data`). This ensures data is handled consistently.
-   **`AnalysisResults`**: A dataclass that stores all the outputs from the analysis. After running the analyzer, you will find all calculated metrics, statistics, and other data neatly organized in this object.

## 2. Installation & Prerequisites

The analyzer is built on a standard scientific Python stack. Ensure you have the following libraries installed:

```bash
pip install tensorflow keras matplotlib seaborn scikit-learn numpy scipy pandas
```

## 3. Basic Usage: A Quick Start

Here’s the simplest way to get started. This example analyzes two dummy models on random data.

```python
import keras
import numpy as np
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# 1. Create or load your Keras models
def create_dummy_model(name):
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model_a = create_dummy_model('Model_A')
model_b = create_dummy_model('Model_B')

models_to_analyze = {
    'Model_A': model_a,
    'Model_B': model_b
}

# 2. Prepare your data
# For a real use case, load your test or validation data here.
x_test = np.random.rand(100, 32, 32, 3)
y_test = np.random.randint(0, 10, 100)
test_data = DataInput(x_data=x_test, y_data=y_test)

# 3. Configure the analysis
# Use default settings for a quick run.
config = AnalysisConfig(
    save_plots=True,          # Save all generated plots
    compute_weight_pca=True   # Enable PCA for model similarity plot
)

# 4. Initialize and run the analyzer
analyzer = ModelAnalyzer(
    models=models_to_analyze,
    config=config,
    output_dir='my_first_analysis'
)
results = analyzer.analyze(data=test_data)

# 5. Access the results
print("Analysis complete! Results are saved in 'my_first_analysis'.")
print("\n--- Summary Statistics ---")
summary = analyzer.get_summary_statistics()

# Example: Print ECE score for Model A
ece_a = summary['calibration_summary'].get('Model_A', {}).get('ece', 'N/A')
print(f"Model A - Expected Calibration Error (ECE): {ece_a:.4f}")

# Example: Print total parameters for Model B
params_b = summary['weight_summary'].get('Model_B', {}).get('total_parameters', 'N/A')
print(f"Model B - Total Parameters: {params_b}")
```

## 4. Configuring the Analysis (`AnalysisConfig`)

The `AnalysisConfig` class is the control panel for the analyzer. You can customize nearly every aspect of the analysis.

```python
config = AnalysisConfig(
    # --- Main Toggles ---
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    analyze_training_dynamics=True,

    # --- Calibration Parameters ---
    calibration_bins=15,          # Number of bins for ECE and reliability diagrams

    # --- Training Dynamics Parameters ---
    smooth_training_curves=True,  # Apply a moving average to smooth loss/accuracy curves
    smoothing_window=5,           # The window size for smoothing

    # --- Weight Analysis Parameters ---
    compute_weight_pca=True,      # Perform PCA on weight stats for similarity plots
    analyze_biases=False,         # Whether to include bias terms in weight analysis
    # Only analyze weights of these layer types. If None, analyze all applicable.
    weight_layer_types=['Dense', 'Conv2D'], 

    # --- Visualization Settings ---
    save_plots=True,
    save_format='png',            # 'png', 'pdf', 'svg'
    dpi=300,                      # Resolution for saved plots
    plot_style='publication',     # 'publication', 'presentation', or 'draft'
)
```

## 5. Advanced Usage

### Analyzing Training Dynamics

To get deep insights into how your models learned, you must provide the training history. The history should be a dictionary mapping model names to their Keras history objects (`history.history`).

```python
# Assume 'histories' is a dictionary like:
# histories = {
#     'Model_A': history_a.history,
#     'Model_B': history_b.history
# }

analyzer = ModelAnalyzer(
    models=models_to_analyze,
    training_history=histories,  # Pass the training histories here
    config=config,
    output_dir='training_dynamics_analysis'
)
results = analyzer.analyze(data=test_data)

# You can now access detailed training metrics
peak_acc_a = results.training_metrics.peak_performance.get('Model_A', {}).get('val_accuracy')
overfit_idx_a = results.training_metrics.overfitting_index.get('Model_A')

print(f"Model A - Peak Validation Accuracy: {peak_acc_a:.4f}")
print(f"Model A - Overfitting Index: {overfit_idx_a:.4f}")
```

### Creating a Pareto Front Analysis

If you've run an experiment with multiple models (e.g., a hyperparameter sweep), you can generate a Pareto front plot to find the best trade-offs between competing objectives, like accuracy vs. overfitting.

This requires `training_history` to be provided.

```python
# After running the analyzer with training history...
pareto_figure = analyzer.create_pareto_analysis()

# The plot is automatically saved to the output directory if config.save_plots is True.
# The figure object is returned if you want to display or modify it further.
if pareto_figure:
    pareto_figure.show()
```

## 6. Understanding the Output (`AnalysisResults`)

The `analyze()` method returns an `AnalysisResults` object. This object contains all the raw data, computed statistics, and metrics.

Here is a map of its key attributes:

| Attribute | Content | Description |
| :--- | :--- | :--- |
| `results.model_metrics` | `dict` | Basic performance metrics (loss, accuracy) from `model.evaluate()`. |
| `results.weight_stats` | `dict` | Detailed statistics (mean, std, norms, sparsity) for each weight tensor in each model. |
| `results.weight_pca` | `dict` | The results of PCA on weight statistics, including components and explained variance. Used for the model similarity plot. |
| `results.calibration_metrics`| `dict` | Key calibration scores like **ECE** (Expected Calibration Error) and **Brier Score**. |
| `results.confidence_metrics`| `dict` | Per-prediction confidence scores, including **max probability**, **margin**, and **entropy**. |
| `results.reliability_data`| `dict` | Data needed to plot reliability diagrams (bin accuracies vs. bin confidences). |
| `results.information_flow`| `dict` | Layer-by-layer statistics like activation mean/std, sparsity, and **effective rank**. |
| `results.activation_stats`| `dict` | Detailed activation distributions for a few key layers (e.g., last conv, middle dense). |
| `results.training_history`| `dict` | The raw training history you provided. |
| `results.training_metrics`| `TrainingMetrics` | A dataclass containing computed training metrics like **epochs to convergence**, **overfitting index**, and **peak performance**. |

## 7. Interpreting Visualizations

The analyzer automatically generates a suite of plots, saved in your output directory.

-   `summary_dashboard.png`: An at-a-glance overview comparing models across key metrics from all analyses. **This is the best place to start.**
-   `training_dynamics.png`: Shows loss/accuracy curves, overfitting gap evolution, and a summary table of training efficiency.
-   `weight_learning_journey.png`: Visualizes how weight magnitudes (L2 norm) evolve through the network and provides a "health heatmap" for each layer.
-   `confidence_calibration_analysis.png`: Contains reliability diagrams, confidence distributions, and other plots to assess if your model's confidence is trustworthy.
-   `information_flow_analysis.png`: Shows how information dimensionality (effective rank) and activation health change from layer to layer.
-   `pareto_analysis.png`: (If generated) Helps you pick the "best" model based on trade-offs between accuracy, overfitting, and convergence speed.

## 8. Extending the Analyzer

The analyzer is designed to be extensible. You can add your own custom analysis and visualization modules.

#### Creating a Custom Analyzer

1.  Inherit from `BaseAnalyzer`.
2.  Implement the `requires_data()` and `analyze()` methods.
3.  Store your results in a new attribute on the `results` object.

```python
from dl_techniques.analyzer.analyzers.base import BaseAnalyzer
from dl_techniques.analyzer.data_types import AnalysisResults, DataInput

class MyCustomAnalyzer(BaseAnalyzer):
    def requires_data(self) -> bool:
        # Does this analysis need test data?
        return True

    def analyze(self, results: AnalysisResults, data: DataInput, cache: dict) -> None:
        print("Running my custom analysis!")
        # Your logic here...
        my_custom_results = {}
        for model_name, model in self.models.items():
            # ... compute custom metrics ...
            my_custom_results[model_name] = {"my_score": 0.99}
        
        # Add your results to the main results object
        results.my_custom_analysis = my_custom_results
```

#### Integrating a Custom Analyzer

You can manually add your custom analyzer to the `ModelAnalyzer` instance.

```python
analyzer = ModelAnalyzer(models, config)
analyzer.analyzers['custom'] = MyCustomAnalyzer(analyzer.models, analyzer.config)

# Now, when you run analyze, 'custom' will be run if specified or if no types are given.
analyzer.analyze(data=test_data, analysis_types={'weights', 'custom'}) 
```

## 9. Troubleshooting and FAQ

-   **Multi-Input Models**: The analyzer has *limited support* for models with multiple inputs. Standard analyses like weight and training dynamics will work, but data-dependent analyses like calibration and information flow may be skipped or produce errors. This is because automatically handling arbitrary multi-input data structures is complex.
-   **"Metric not found" in Training History**: The analyzer uses pattern matching to find metrics like `loss` and `val_accuracy`. If your Keras model uses non-standard metric names (e.g., `my_custom_loss`), the analyzer might not find them. It's best practice to use standard Keras metric names.
-   **Slow Analysis**: Information flow analysis can be slow as it involves getting activations from many layers. For quick runs, you can disable it: `AnalysisConfig(analyze_information_flow=False)`.
-   **Warnings about `tf.function` retracing**: These are TensorFlow warnings and generally safe to ignore for this tool's usage, as the analysis involves running predictions on different model sub-graphs.