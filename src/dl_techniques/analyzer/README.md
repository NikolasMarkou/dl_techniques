# Model Analyzer: Complete Usage Guide

A comprehensive, modular analysis toolkit for deep learning models built on Keras and TensorFlow. This module provides multi-dimensional model analysis including weight distributions, calibration metrics, information flow patterns, and training dynamics with publication-ready visualizations.

## 1. Overview

The Model Analyzer is designed to provide deep insights into your neural network models beyond simple accuracy metrics. It helps answer critical questions about model behavior, training efficiency, and production readiness. By automating complex analyses and generating intuitive visualizations, it streamlines the process of model selection, debugging, and hyperparameter tuning.

### Key Features

-   🔍 **Comprehensive Analysis**: Four specialized analysis modules covering weights, calibration, information flow, and training dynamics.
-   📊 **Rich Visualizations**: Publication-ready plots and summary dashboards with consistent styling and color schemes.
-   🧩 **Modular & Extensible**: Each analysis is independent. The architecture is designed for adding custom analyzers and visualizers.
-   🚀 **Training & Hyperparameter Insights**: Deep analysis of training history, convergence patterns, and a powerful Pareto-front analysis for optimal model selection.
-   💾 **Serializable Results**: Export all raw metrics to a single JSON file for reproducible analysis, reporting, or further programmatic use.
-   💪 **Robust & Efficient**: Handles large datasets through smart sampling, caches intermediate results to avoid re-computation, and includes robust error handling.

### Module Structure

The toolkit is organized into distinct components for analysis, visualization, and configuration.

```
analyzer/
├── analyzers/                          # Core analysis logic components
│   ├── base.py                         # Abstract base analyzer interface
│   ├── weight_analyzer.py              # Weight distribution and health analysis
│   ├── calibration_analyzer.py         # Model confidence and calibration metrics
│   ├── information_flow_analyzer.py    # Activation patterns and information flow
│   └── training_dynamics_analyzer.py   # Training history and convergence analysis
├── visualizers/                        # Visualization generation components
│   ├── base.py                         # Abstract base visualizer interface
│   ├── weight_visualizer.py            # Weight analysis visualizations
│   ├── calibration_visualizer.py       # Calibration and confidence plots
│   ├── information_flow_visualizer.py  # Information flow visualizations
│   ├── training_dynamics_visualizer.py # Training dynamics plots
│   └── summary_visualizer.py           # Unified summary dashboard
├── config.py                           # Configuration classes and plotting setup
├── data_types.py                       # Structured data types (DataInput, AnalysisResults)
├── constants.py                        # Analysis constants and thresholds
├── utils.py                            # Utility functions and helpers
├── model_analyzer.py                   # Main coordinator class
└── README.md                           # This file
```

## 2. Installation & Quick Start

### Prerequisites

Ensure you have the required libraries installed.

```bash
pip install keras tensorflow matplotlib seaborn scikit-learn numpy scipy pandas tqdm
```

### 5-Minute Setup

Get comprehensive analysis results for multiple models in just a few lines of code.

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
import keras
import numpy as np

# 1. Prepare your models (dictionary format with descriptive names)
models = {
    'ResNet_v1': your_resnet_model,
    'ConvNext_v2': your_convnext_model
}

# 2. Prepare your test data and training histories (if available)
x_test, y_test = np.random.rand(100, 32, 32, 3), np.random.randint(0, 10, 100)
test_data = DataInput(x_data=x_test, y_data=y_test)
# training_histories = {'ResNet_v1': history1, 'ConvNext_v2': history2} # Optional

# 3. Configure and run the analysis
config = AnalysisConfig(analyze_training_dynamics=True) # Enable training analysis
analyzer = ModelAnalyzer(
    models=models,
    # training_history=training_histories, # Uncomment if you have histories
    config=config,
    output_dir='analysis_results'
)
results = analyzer.analyze(test_data)

print("Analysis complete! Check the 'analysis_results' folder for plots and data.")
```

## 3. Analysis Capabilities

The analyzer computes a wide range of metrics across four key areas of model behavior.

### Weight Analysis Metrics

| Metric                | Description                                         | Interpretation                                            |
| --------------------- | --------------------------------------------------- | --------------------------------------------------------- |
| **L1/L2/Spectral Norms** | Measures of weight magnitude and complexity.        | Higher values indicate larger weights; controls stability. |
| **Weight Distribution** | Statistical properties (mean, std, skew, kurtosis). | Indicates weight health and potential for vanishing/exploding gradients. |
| **Sparsity**          | Fraction of near-zero weights in a layer.           | High sparsity can indicate under-utilized or dead neurons. |
| **Health Score**      | A combined metric (0-1) summarizing weight health.  | Higher score = healthier weight distribution.             |

### Calibration & Confidence Metrics

| Metric             | Description                                          | Ideal Value     |
| ------------------ | ---------------------------------------------------- | --------------- |
| **ECE**            | Expected Calibration Error. Measures gap between confidence and accuracy. | 0 (perfect calibration) |
| **Brier Score**    | Mean squared error for probabilistic predictions.    | 0 (perfect predictions) |
| **Mean Confidence**| Average of the max probability for each prediction.  | Context-dependent |
| **Mean Entropy**   | Average uncertainty across all predictions.          | Context-dependent |

### Information Flow Metrics

| Metric                  | Description                                            | Interpretation                                                              |
| ----------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------- |
| **Activation Statistics** | Mean, std, and sparsity of layer activations.          | Indicates layer health, utilization, and potential for dead neurons.        |
| **Effective Rank**      | Dimensionality of the information represented by a layer. | Higher rank suggests more diverse and expressive feature representations.  |
| **Positive Ratio**      | Fraction of positive activations (e.g., after ReLU).   | Indicates activation patterns; values near 0 or 1 suggest saturation.       |
| **Specialization Score**| A combined metric measuring a layer's feature learning quality. | Higher score suggests a good balance of activation, diversity, and utilization. |

### Training Dynamics Metrics

| Metric                  | Description                                                     | Interpretation                                            |
| ----------------------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| **Epochs to Convergence** | Epochs to reach 95% of peak validation performance.             | Lower is faster and more efficient learning.              |
| **Overfitting Index**   | Average (Val Loss - Train Loss) in the final third of training. | Positive values indicate overfitting.                     |
| **Training Stability**  | Standard deviation of recent validation losses.                 | Lower values indicate more stable training convergence.   |
| **Peak Performance**    | Best validation accuracy/loss achieved and at which epoch.      | The model's best potential performance during training.   |
| **Final Gap**           | Difference between validation and training loss at the last epoch. | Indicates the final overfitting or underfitting state.    |

## 4. Usage Patterns & Use Cases

### Pattern 1: Single Model Deep Dive

Use the analyzer to thoroughly debug a single model's performance and behavior.

```python
# Scenario: A model's performance is unexpectedly low.
model = keras.models.load_model('path/to/problem_model.keras')
# history = training_history_for_the_model

config = AnalysisConfig(
    analyze_weights=True,
    analyze_information_flow=True,  # Check for dead neurons/layers
    analyze_calibration=True,
    analyze_training_dynamics=True
)

analyzer = ModelAnalyzer(
    models={'ProblemModel': model},
    training_history={'ProblemModel': history},
    config=config
)
results = analyzer.analyze(test_data)

# Next steps:
# 1. Check `weight_learning_journey.png` for exploding/vanishing weights.
# 2. Check `information_flow_analysis.png` for dead layers (low activation/rank).
# 3. Check `training_dynamics.png` for severe overfitting or unstable training.
```

### Pattern 2: Multi-Model Comparison for Selection

Compare different architectures or training runs to select the best candidate.

```python
# Scenario: Choose the best model from multiple architectures for production.
models = {
    'ResNet50': resnet_model,
    'EfficientNet': efficientnet_model,
    'ConvNext': convnext_model
}

analyzer = ModelAnalyzer(models=models)
results = analyzer.analyze(test_data)

# Next steps:
# 1. Start with `summary_dashboard.png`. The performance table gives a quick overview.
# 2. Check the "Calibration Landscape" plot. Models in the bottom-left (low ECE, low Brier) are best.
# 3. Use the "Model Similarity" plot to see which models learned similar representations.
```

### Pattern 3: Hyperparameter Sweep Analysis

Efficiently analyze the results of a hyperparameter sweep to find the optimal configuration.

```python
# Scenario: Find the best learning rate and batch size from a sweep.
models = {
    'lr_0.001_batch_32': model_1,
    'lr_0.01_batch_32': model_2,
    'lr_0.001_batch_64': model_3,
    'lr_0.01_batch_64': model_4
}
histories = {name: h for name, h in sweep_histories.items()}

config = AnalysisConfig(
    analyze_training_dynamics=True,
    pareto_analysis_threshold=2 # Generate Pareto plot if 2+ models are provided
)

analyzer = ModelAnalyzer(models=models, training_history=histories, config=config)
results = analyzer.analyze(validation_data)

# Generate and save the Pareto analysis plot
pareto_fig = analyzer.create_pareto_analysis()

# Next steps:
# 1. Open `pareto_analysis.png`. The scatter plot shows models on the Pareto front (optimal trade-offs between accuracy and overfitting).
# 2. Use the heatmap to compare normalized performance across all models and metrics.
```

## 5. Advanced Configuration

The `AnalysisConfig` class provides fine-grained control over the analysis process.

```python
config = AnalysisConfig(
    # === Main Toggles ===
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    analyze_training_dynamics=True,

    # === Data Sampling ===
    n_samples=1000,  # Max samples to use for data-dependent analyses (e.g., calibration)

    # === Weight Analysis ===
    weight_layer_types=['Dense', 'Conv2D'],  # Only analyze weights from these layer types
    analyze_biases=False,                  # Exclude bias vectors from analysis
    compute_weight_pca=True,               # Enable model similarity analysis

    # === Calibration Analysis ===
    calibration_bins=15,                   # Number of bins for ECE calculation

    # === Training Dynamics ===
    smooth_training_curves=True,           # Apply a moving average filter to training curves
    smoothing_window=5,                    # The window size for the smoothing filter

    # === Visualization & Output ===
    plot_style='publication',              # 'publication', 'presentation', or 'draft'
    save_plots=True,
    save_format='png',                     # 'png', 'pdf', 'svg'
    dpi=300,

    # === Performance & Limits ===
    max_layers_heatmap=12,                 # Max layers in the weight health heatmap
    max_layers_info_flow=8,                # Max layers in information flow plots
    verbose=True,                          # Enable detailed logging
)
```

## 6. Understanding the Output

After running, the analyzer saves plots and a JSON data file to the output directory.

```
analysis_results/
├── summary_dashboard.png              # START HERE: High-level overview of all models.
├── training_dynamics.png              # Training curves, overfitting, and convergence analysis.
├── weight_learning_journey.png        # Weight magnitude evolution and health heatmap.
├── confidence_calibration_analysis.png  # Deep dive into model confidence and calibration.
├── information_flow_analysis.png      # Layer-wise analysis of activations and information.
├── pareto_analysis.png               # (Optional) Hyperparameter optimization insights.
└── analysis_results.json             # Raw data for all computed metrics.
```

### Key Visualizations Explained

#### 1. Summary Dashboard (`summary_dashboard.png`)

A 2x2 grid providing a holistic view of all models.

-   **Performance Table**: Key metrics for each model, including training insights if history is provided.
-   **Model Similarity**: A 2D PCA plot of weight statistics. Models that are close together have learned similar weight distributions.
-   **Confidence Profiles**: Violin plots showing the distribution of prediction confidence (max probability) for each model.
-   **Calibration Landscape**: A scatter plot of ECE vs. Brier Score. Models in the bottom-left quadrant are well-calibrated and have high probabilistic accuracy.

#### 2. Training Dynamics (`training_dynamics.png`)

A deep dive into the learning process.

-   **Loss/Accuracy Curves**: Smoothed training and validation curves.
-   **Overfitting Analysis**: Plots the gap (validation loss - training loss) over epochs. A positive gap indicates overfitting.
-   **Best Epoch Performance**: A scatter plot showing each model's peak validation accuracy versus the epoch it was achieved. Helps identify models that learn faster.
-   **Summary Table**: Quantitative training metrics like convergence speed and stability.

#### 3. Weight Learning Journey (`weight_learning_journey.png`)

Assesses the health and evolution of model weights.

-   **Weight Evolution**: Shows how the L2 norm of weights changes across the layers of the network. Look for smooth progressions, not sudden explosions or collapses.
-   **Health Heatmap**: A layer-by-layer health score for each model. Green indicates healthy weights; red indicates potential issues like high sparsity or unstable distributions.

#### 4. Confidence & Calibration Analysis (`confidence_calibration_analysis.png`)

Evaluates the reliability of model predictions.

-   **Reliability Diagram**: Compares predicted probability to the actual fraction of positives. A perfectly calibrated model follows the diagonal.
-   **Confidence Distributions**: Violin plots showing how confident each model is.
-   **Per-Class ECE**: Bar chart showing calibration error for each class, helping to identify problematic classes.
-   **Uncertainty Landscape**: A 2D density plot of prediction confidence vs. entropy. Shows the model's uncertainty profile.

#### 5. Information Flow Analysis (`information_flow_analysis.png`)

Diagnoses how information propagates through the network.

-   **Activation Flow Overview**: Tracks the mean and standard deviation of activations through the layers. Helps spot vanishing or exploding activations.
-   **Effective Rank Evolution**: Plots the dimensionality of information at each layer. A collapsing rank may indicate a bottleneck.
-   **Activation Health Dashboard**: A heatmap showing potential issues like dead neurons (high sparsity) or saturated activations.
-   **Layer Specialization Analysis**: Plots a "specialization score" for each layer, indicating how well it's learning diverse features.

#### 6. Pareto Analysis (`pareto_analysis.png`)

(Generated with `create_pareto_analysis()`) A powerful tool for hyperparameter tuning.

-   **Pareto Front Plot**: A scatter plot of Peak Accuracy vs. Overfitting Index. Models on the red-dashed "Pareto Front" represent the best possible trade-offs.
-   **Normalized Performance Heatmap**: A heatmap comparing all models across normalized metrics (accuracy, low overfitting, fast convergence), making it easy to see which configuration excels in which area.

## 7. Troubleshooting

-   **Multi-Input Models**: The analyzer has limited support for models with multiple inputs. It will log warnings and automatically skip incompatible analyses (like calibration and information flow) for these models to prevent errors.
-   **"No training metrics found"**: The analyzer robustly searches for common metric names (`accuracy`, `val_loss`, etc.). If you use non-standard names in your `history` object, analysis will be limited. Ensure your Keras history keys are standard.
-   **Memory Issues**: For very large models or datasets, analysis can be memory-intensive. Reduce the sample size and disable the most expensive analyses in `AnalysisConfig`:
    ```python
    config = AnalysisConfig(
        n_samples=500,                  # Reduce from default 1000
        analyze_information_flow=False, # This is often the most memory-intensive
        max_layers_heatmap=8            # Limit heatmap size
    )
    ```
-   **Plots look wrong/empty**: Enable verbose logging (`config = AnalysisConfig(verbose=True)`) and check the console output. You can also inspect the `analysis_results.json` file to see what data was successfully computed.

## 8. Extensions

The toolkit is designed to be extensible. You can add your own custom analysis and visualization modules by inheriting from the base classes.

### Creating a Custom Analyzer

Extend the `BaseAnalyzer` class and implement the `analyze` method.

```python
from analyzer.analyzers.base import BaseAnalyzer
from analyzer.data_types import AnalysisResults, DataInput

class MyCustomAnalyzer(BaseAnalyzer):
    def requires_data(self) -> bool:
        return True # This analyzer needs data

    def analyze(self, results: AnalysisResults, data: DataInput, cache: dict) -> None:
        # Your custom logic here
        custom_metrics = {}
        for model_name, model in self.models.items():
            # ... compute your metrics ...
            custom_metrics[model_name] = {'my_score': 0.99}

        # Store results in the main results object
        results.custom_metrics = custom_metrics
```

### Creating a Custom Visualizer

Extend the `BaseVisualizer` class and implement the plotting logic.

```python
from analyzer.visualizers.base import BaseVisualizer
import matplotlib.pyplot as plt

class MyCustomVisualizer(BaseVisualizer):
    def create_visualizations(self) -> None:
        if not hasattr(self.results, 'custom_metrics'):
            return

        fig, ax = plt.subplots()
        # ... your custom plotting logic ...
        # You can access self.results, self.config, and self.model_colors

        if self.config.save_plots:
            self._save_figure(fig, 'my_custom_plot')
        plt.close(fig)
```