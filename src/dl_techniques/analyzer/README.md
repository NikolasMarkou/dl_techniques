# Model Analyzer: Complete Usage Guide

A comprehensive, modular analysis toolkit for deep learning models built on Keras and TensorFlow. This module provides multi-dimensional model analysis including weight distributions, calibration metrics, information flow patterns, training dynamics, and advanced spectral analysis with publication-ready visualizations.

## 1. Overview

The Model Analyzer is designed to provide deep insights into your neural network models beyond simple accuracy metrics. It helps answer critical questions about model behavior, training efficiency, and production readiness. By automating complex analyses and generating intuitive visualizations, it streamlines the process of model selection, debugging, and hyperparameter tuning.

### Key Features

-   **Comprehensive Analysis**: Five specialized analysis modules covering weights, calibration, information flow, training dynamics, and spectral properties.
-   **Advanced Spectral Analysis (WeightWatcher)**: Assess training quality, complexity, and generalization potential using power-law and concentration analysis, often without needing test data.
-   **Rich Visualizations**: Publication-ready plots and summary dashboards with consistent styling and color schemes.
-   **Modular & Extensible**: Each analysis is independent. The architecture is designed for adding custom analyzers and visualizers.
-   **Training & Hyperparameter Insights**: Deep analysis of training history, convergence patterns, and a powerful Pareto-front analysis for optimal model selection.
-   **Serializable Results**: Export all raw metrics to a single JSON file for reproducible analysis, reporting, or further programmatic use.
-   **Robust & Efficient**: Handles large datasets through smart sampling, caches intermediate results to avoid re-computation, and includes robust error handling.

### Module Structure

The toolkit is organized into distinct components for analysis, visualization, and configuration.

```
analyzer/
├── analyzers/                          # Core analysis logic components
│   ├── base.py                         # Abstract base analyzer interface
│   ├── weight_analyzer.py              # Weight distribution and basic health analysis
│   ├── calibration_analyzer.py         # Model confidence and calibration metrics
│   ├── information_flow_analyzer.py    # Activation patterns and information flow
│   ├── training_dynamics_analyzer.py   # Training history and convergence analysis
│   └── spectral_analyzer.py            # Spectral analysis of weights (WeightWatcher)
├── visualizers/                        # Visualization generation components
│   ├── base.py                         # Abstract base visualizer interface
│   ├── weight_visualizer.py            # Weight analysis visualizations
│   ├── calibration_visualizer.py       # Calibration and confidence plots
│   ├── information_flow_visualizer.py  # Information flow visualizations
│   ├── training_dynamics_visualizer.py # Training dynamics plots
│   ├── spectral_visualizer.py          # Spectral analysis visualizations
│   └── summary_visualizer.py           # Unified summary dashboard
├── config.py                           # Configuration classes and plotting setup
├── data_types.py                       # Structured data types (DataInput, AnalysisResults)
├── constants.py                        # Analysis constants and thresholds
├── spectral_metrics.py                 # Core spectral metric calculations
├── spectral_utils.py                   # Utilities for spectral analysis
├── utils.py                            # General utility functions and helpers
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
import keras
import numpy as np
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# 1. Prepare your models (dictionary format with descriptive names)
# The keys ('ResNet_v1', 'ConvNext_v2') will be used as labels in all plots.
models = {
    'ResNet_v1': your_resnet_model,
    'ConvNext_v2': your_convnext_model
}

# 2. Prepare your test data
# This is required for calibration and information flow analyses.
x_test, y_test = np.random.rand(100, 32, 32, 3), np.random.randint(0, 10, 100)
test_data = DataInput(x_data=x_test, y_data=y_test)

# 3. Prepare your training histories (optional, but required for training dynamics)
# This should be a dictionary where keys match the model names.
# The value for each key is the `history` object from a Keras `model.fit()` call.
# See the "Preparing Your Inputs" section for a detailed example.
training_histories = {
    'ResNet_v1': history1, # e.g., result from your_resnet_model.fit(...)
    'ConvNext_v2': history2  # e.g., result from your_convnext_model.fit(...)
}

# 4. Configure and run the analysis
config = AnalysisConfig(
    analyze_training_dynamics=True, # Enable training analysis
    analyze_spectral=True           # Enable spectral analysis
)
analyzer = ModelAnalyzer(
    models=models,
    training_history=training_histories, # Pass the histories to the analyzer
    config=config,
    output_dir='analysis_results'
)
results = analyzer.analyze(test_data)

print("Analysis complete! Check the 'analysis_results' folder for plots and data.")
```

## 3. Preparing Your Inputs: A Detailed Guide

To get the most out of the Model Analyzer, it's important to provide the input data in the correct format. The analyzer is initialized with three main components: `models`, `training_history`, and `config`. The analysis itself is run with a `DataInput` object.

### 3.1 The `models` Dictionary

This is the primary input and is required. It's a Python dictionary that maps a human-readable string name to a compiled Keras model instance.

-   **Structure**: `Dict[str, keras.Model]`
-   **Keys**: The string keys (e.g., `'ResNet_v1'`) are crucial as they are used to label your models in all generated plots, tables, and the final JSON output. Choose descriptive names.
-   **Values**: The values must be instances of `keras.Model`.

```python
# Example `models` dictionary
models = {
    'MyCNN_v1': cnn_model_v1,
    'MyCNN_v2_with_dropout': cnn_model_v2
}
```

### 3.2 The `DataInput` Object

This object wraps your dataset and is passed to the `analyzer.analyze()` method. It is **required** for any analysis that depends on data, such as **calibration** and **information flow**.

-   **Structure**: A `DataInput` named tuple with two fields: `x_data` and `y_data`.
-   **`x_data`**: A NumPy array containing your input features. The shape should be `(n_samples, ...)`, for example, `(10000, 28, 28, 1)` for MNIST images.
-   **`y_data`**: A NumPy array containing the corresponding true labels. The analyzer can handle both integer labels (e.g., shape `(10000,)`) and one-hot encoded labels (e.g., shape `(10000, 10)`).

```python
import numpy as np
from dl_techniques.analyzer import DataInput

# Example test data
x_test = np.random.rand(500, 64, 64, 3) # 500 samples of 64x64 color images
y_test = np.random.randint(0, 5, 500) # 500 integer labels for 5 classes

# Create the DataInput object
test_data = DataInput(x_data=x_test, y_data=y_test)
```

### 3.3 The `training_history` Dictionary

This input is **optional** but is **required** to enable the `TrainingDynamicsAnalyzer`. It provides the epoch-by-epoch learning history for each model.

-   **Structure**: `Dict[str, Dict[str, List[float]]]`
-   **Keys**: The keys of this dictionary **must match the keys of your `models` dictionary exactly**. This is how the analyzer associates a history with a model.
-   **Values**: The value for each model is the `history` attribute of the History object returned by a Keras `model.fit()` call. This is a dictionary where keys are metric names (e.g., `'loss'`, `'val_accuracy'`) and values are lists of floats, with one entry per epoch.

**Crucial Note on Metric Names**: The analyzer robustly searches for common metric names (e.g., it will find `accuracy`, `acc`, or `categorical_accuracy`). However, for best results, use standard Keras metric names. The required metrics for full analysis are a training loss, a validation loss, a training accuracy, and a validation accuracy.

Here is a complete example of how to generate and structure the `training_history` object:

```python
# Assume you have two models defined: cnn_model_v1, cnn_model_v2
# And training/validation data: (x_train, y_train), (x_val, y_val)

# 1. Train your models and capture the History object
history_v1 = cnn_model_v1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)
history_v2 = cnn_model_v2.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

# 2. The object we need is the `.history` attribute, which is a dictionary
#    For example, history_v1.history looks like this:
#    {
#        'loss': [1.2, 0.8, 0.6, ...],
#        'accuracy': [0.65, 0.72, 0.78, ...],
#        'val_loss': [1.0, 0.7, 0.5, ...],
#        'val_accuracy': [0.68, 0.75, 0.80, ...]
#    }

# 3. Construct the training_history dictionary for the analyzer
#    The keys here MUST match the keys you will use in your `models` dictionary.
training_histories = {
    'MyCNN_v1': history_v1.history,
    'MyCNN_v2_with_dropout': history_v2.history
}

# 4. Now you can initialize the analyzer
analyzer = ModelAnalyzer(
    models={'MyCNN_v1': cnn_model_v1, 'MyCNN_v2_with_dropout': cnn_model_v2},
    training_history=training_histories
)
```

## 4. Analysis Capabilities

The analyzer computes a wide range of metrics across five key areas of model behavior.

### Weight Analysis Metrics

This analysis inspects the internal parameters of the model to diagnose its structural health and complexity.

| Metric                | Description                                                                                             | Interpretation                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **L1/L2/Spectral Norms** | Mathematical norms that measure the aggregate magnitude of weights in a layer.                          | Higher values indicate larger, more complex weights which can lead to instability or overfitting. Consistently low norms might suggest underfitting. |
| **Weight Distribution** | Statistical properties like mean, standard deviation, skewness, and kurtosis of the weight values.    | A distribution centered near zero with moderate variance is often ideal. High skew or kurtosis can signal issues like dying ReLUS or unstable gradients. |
| **Sparsity**          | The fraction of weights in a layer that are very close to zero.                                         | High sparsity can indicate that many neurons are not contributing to the model's predictions (i.e., they are "dead" or under-utilized).            |
| **Health Score**      | A composite score (0-1) derived from norm, sparsity, and distribution health.                           | A single-glance metric for layer health. Higher scores (closer to 1) indicate a healthier, more balanced weight distribution.                 |

### Spectral Analysis Metrics (WeightWatcher)

This analysis examines the eigenvalue spectrum of weight matrices to assess training quality and complexity without requiring test data. It is based on the theory of **Heavy-Tailed Self-Regularization (HTSR)**, which posits that SGD implicitly regularizes deep neural networks, causing the distribution of eigenvalues of their weight matrices—the Empirical Spectral Density (ESD)—to develop a characteristic heavy-tailed shape. This shape, quantifiable with a power-law, correlates strongly with the model's ability to generalize.

| Metric                     | Description                                                                                                       | Interpretation                                                                                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Power-Law Exponent (α)** | The exponent of the power-law fit `P(λ) ~ λ^(-α)` to the tail of the eigenvalue distribution (ESD) of weight matrices.                              | This is the key indicator of training quality. `α < 2.0` suggests an extremely heavy-tailed spectrum, which can indicate over-training or memorization. `2.0 < α < 6.0` typically corresponds to a well-trained model that generalizes well. `α > 6.0` may indicate under-training. The value `α ≈ 2.0` represents a critical phase boundary for "Ideal Learning."             |
| **Concentration Score**    | A composite metric measuring the inequality of the eigenvalue distribution. It combines the Gini Coefficient, Dominance Ratio, and Participation Ratio.               | High values suggest information is concentrated in a few dominant patterns, which can indicate model brittleness, highlight critical layers, or suggest that pruning/quantization may be risky.         |
| **Matrix Entropy**         | A measure of the information distribution across eigenvalues, calculated from the Shannon entropy of the normalized singular values.     | Higher entropy indicates a more uniform spread of information across all learned features (eigenvectors), which is often a sign of better generalization and robustness.        |
| **Stable Rank**            | The effective rank of a weight matrix, `||W||_F^2 / ||W||_2^2`, which simplifies to `(Σ λ_i) / max(λ_i)`. It indicates the dimensionality of the space spanned by its singular vectors. | A higher stable rank suggests the layer is utilizing its full capacity to learn diverse features. A low stable rank can indicate over-parameterization, redundant features, or an information bottleneck. |
| **Gini Coefficient** | A statistical measure of inequality in the eigenvalue distribution, ranging from 0 (perfect equality) to 1 (perfect inequality). | A sub-metric of the Concentration Score. High Gini values mean a few eigenvalues are much larger than the rest, indicating information concentration. |
| **Dominance Ratio** | The ratio of the largest eigenvalue to the sum of all other eigenvalues. | A sub-metric of the Concentration Score. This directly quantifies how much a single feature or pattern dominates the layer's learned representation. |

### Calibration & Confidence Metrics

This analysis evaluates how well the model's predicted probabilities reflect the true likelihood of outcomes.

| Metric             | Description                                                                                             | Ideal Value     |
| ------------------ | ------------------------------------------------------------------------------------------------------- | --------------- |
| **ECE**            | Expected Calibration Error. Measures the average gap between a model's prediction confidence and its actual accuracy. | 0 (perfect calibration) |
| **Brier Score**    | The mean squared error between predicted probabilities and the one-hot encoded true labels. A measure of both calibration and resolution. | 0 (perfect predictions) |
| **Mean Confidence**| The average of the highest probability assigned by the model for each prediction in the dataset.      | Context-dependent; very high values might indicate overconfidence. |
| **Mean Entropy**   | The average Shannon entropy across all prediction distributions, quantifying the model's overall uncertainty. | Context-dependent; lower values indicate more confident (peaked) predictions. |

### Information Flow Metrics

This analysis tracks how information (activations) propagates through the network, helping to identify bottlenecks and pathologies.

| Metric                  | Description                                                                                                   | Interpretation                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Activation Statistics** | The mean, standard deviation, and sparsity of activations for each layer's output.                            | Helps diagnose vanishing gradients (mean/std near zero), exploding gradients (large values), or dead neurons (high sparsity after ReLU).       |
| **Effective Rank**      | A measure of the dimensionality of the feature space represented by a layer's activations.                    | A higher rank suggests that the layer is learning a diverse and rich set of features. A sudden drop in rank can indicate an information bottleneck. |
| **Positive Ratio**      | The fraction of activations that are positive (typically after a ReLU activation).                            | Values near 0 or 1 indicate that the layer is saturated (either always off or always on), which hinders learning. A balanced ratio is healthier. |
| **Specialization Score**| A composite score (0-1) that combines activation health, balance, and effective rank to measure feature learning quality. | Higher scores suggest a layer is effectively transforming information without losing diversity or becoming saturated.                           |

### Training Dynamics Metrics

This analysis examines the model's learning history to understand its training efficiency, stability, and tendency to overfit.

| Metric                  | Description                                                                                             | Interpretation                                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Epochs to Convergence** | The number of epochs required for the model to reach 95% of its peak validation performance.              | A measure of training speed. Lower is faster and indicates more efficient learning.                                                       |
| **Overfitting Index**   | The average difference between validation loss and training loss during the final third of training.    | A positive value indicates overfitting (model performs better on training data). A negative value indicates underfitting.                  |
| **Training Stability**  | The standard deviation of validation loss over the last several epochs.                                 | A lower value indicates a smooth and stable convergence. High values suggest an unstable training process (e.g., learning rate is too high). |
| **Peak Performance**    | The best validation accuracy or loss achieved during the entire training process, and the epoch it occurred. | Represents the model's maximum potential performance. If it occurs early, it may be a sign of early overfitting.                          |
| **Final Gap**           | The difference between validation and training loss at the very last epoch of training.                 | A snapshot of the model's generalization state at the end of training.                                                                    |

## 5. Usage Patterns & Use Cases

### Pattern 1: Single Model Deep Dive

Use the analyzer to thoroughly debug a single model's performance and behavior.

```python
# Scenario: A model's performance is unexpectedly low.
model = keras.models.load_model('path/to/problem_model.keras')
# history = training_history_for_the_model (see section 3.3 for structure)

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
# 3. Check `spectral_summary.png` to compare training quality (alpha values) and complexity.
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
# sweep_histories is a dict where keys match model names, and values are Keras history dicts
# See section 3.3 for the required structure
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
# 1. Open `pareto_analysis.png`. Models on the Pareto front offer optimal trade-offs.
# 2. Use the heatmap to compare normalized performance across all models and metrics.
```

### Pattern 4: Data-Free Training Quality Check

Use spectral analysis to quickly assess if a model is over-trained or under-trained without a test set.

```python
# Scenario: Quickly validate a newly trained model's quality.
model = keras.models.load_model('path/to/new_model.keras')

config = AnalysisConfig(analyze_spectral=True) # Only run spectral analysis
analyzer = ModelAnalyzer(models={'NewModel': model}, config=config)
results = analyzer.analyze() # No data needed for this analysis

# Next steps:
# 1. Open `spectral_summary.png`. Check if the Mean Alpha (α) is in the 2.0-6.0 range.
# 2. Read the recommendations in `analysis_results.json` under `spectral_recommendations`.
```

### Pattern 5: Improving Generalization with SVD Smoothing

Create a smoothed version of a model to potentially improve its generalization performance.

```python
# Scenario: A model shows signs of being slightly over-trained.
analyzer = ModelAnalyzer(models={'OverTrainedModel': model})
analyzer.analyze() # Run analysis to get spectral data

# Create a smoothed version of the model
smoothed_model = analyzer.create_smoothed_model(
    model_name='OverTrainedModel',
    method='detX' # 'svd', 'detX', or 'lambda_min'
)

# You can now save and evaluate the smoothed_model
```

## 6. Advanced Configuration

The `AnalysisConfig` class provides fine-grained control over the analysis process.

```python
config = AnalysisConfig(
    # === Main Toggles ===
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    analyze_training_dynamics=True,
    analyze_spectral=True,

    # === Data Sampling ===
    n_samples=1000,  # Max samples for data-dependent analyses

    # === Weight Analysis ===
    weight_layer_types=['Dense', 'Conv2D'],
    analyze_biases=False,
    compute_weight_pca=True,

    # === Spectral Analysis (WeightWatcher) ===
    spectral_min_evals=10,                # Min eigenvalues for a layer to be analyzed
    spectral_concentration_analysis=True, # Enable concentration metrics
    spectral_randomize=False,             # Compare with randomized weights

    # === Calibration Analysis ===
    calibration_bins=15,

    # === Training Dynamics ===
    smooth_training_curves=True,
    smoothing_window=5,

    # === Visualization & Output ===
    plot_style='publication',
    save_plots=True,
    save_format='png',
    dpi=300,

    # === JSON Serialization Options to manage file size
    json_include_per_sample_data=False, # Set to True to include bulky arrays like confidence/entropy for every sample.
    json_include_raw_esds=False,        # Set to True to include raw eigenvalue arrays for every layer.
    
    # === Performance & Limits ===
    max_layers_heatmap=12,
    max_layers_info_flow=8,
    verbose=True,
)
```

## 7. Understanding the Output

After running, the analyzer saves plots and a JSON data file to the output directory.

```
analysis_results/
├── summary_dashboard.png              # START HERE: High-level overview of all models.
├── spectral_summary.png               # Compares models on spectral metrics (training quality).
├── training_dynamics.png              # Training curves, overfitting, and convergence analysis.
├── weight_learning_journey.png        # Weight magnitude evolution and health heatmap.
├── confidence_calibration_analysis.png  # Deep dive into model confidence and calibration.
├── information_flow_analysis.png      # Layer-wise analysis of activations and information.
├── pareto_analysis.png               # (Optional) Hyperparameter optimization insights.
├── spectral_plots/                   # Directory with detailed per-layer power-law plots.
│   └── ModelA_layer_5_dense_powerlaw.png
└── analysis_results.json             # Raw data for all computed metrics.
```

### Key Visualizations Explained

#### 1. Summary Dashboard (`summary_dashboard.png`)

A 2x2 grid providing a holistic view of all models.

-   **Performance Table**: A comprehensive summary of key performance indicators, including training efficiency metrics if history is provided.
-   **Model Similarity**: A 2D PCA plot of weight statistics. Models that are close together have learned similar weight distributions.
-   **Confidence Profiles**: Violin plots showing the distribution of prediction confidence for each model.
-   **Calibration Landscape**: A scatter plot of ECE vs. Brier Score. The goal is to be in the bottom-left quadrant (low error, good calibration).

#### 2. Spectral Analysis Summary (`spectral_summary.png`)

A dashboard for comparing models based on their weight matrix spectral properties.

-   **Mean Power-Law Exponent (α)**: A bar chart comparing the average α value for each model. This is the single most important plot for data-free generalization estimates. A green bar indicates the ideal range (2.0-6.0), red indicates potential over- or under-training, and yellow is borderline.
-   **Mean Concentration Score**: A bar chart comparing the average information concentration. Higher scores suggest some layers are more "brittle" or specialized. Use this to gauge model robustness, especially if you plan to prune or quantize the model.
-   **Recommendations**: The `analysis_results.json` file contains specific, actionable recommendations based on the spectral analysis for each model (e.g., "Model may be over-trained. Consider early stopping...").

#### 3. Training Dynamics (`training_dynamics.png`)

A deep dive into the learning process.

-   **Loss/Accuracy Curves**: Smoothed training and validation curves for a clear view of the learning trajectory.
-   **Overfitting Analysis**: Plots the gap (validation loss - training loss) over epochs to diagnose overfitting.
-   **Best Epoch Performance**: A scatter plot showing each model's peak validation accuracy versus the epoch it was achieved.
-   **Summary Table**: A detailed table of quantitative training metrics like convergence speed and stability.

#### 4. Weight Learning Journey (`weight_learning_journey.png`)

Assesses the health and evolution of model weights.

-   **Weight Evolution**: Shows how the L2 norm of weights changes across layers, helping detect exploding or vanishing gradients.
-   **Health Heatmap**: A layer-by-layer health score for each model, allowing quick identification of problematic layers.

#### 5. Confidence & Calibration Analysis (`confidence_calibration_analysis.png`)

Evaluates the reliability of model predictions.

-   **Reliability Diagram**: Compares predicted probability to actual accuracy. A perfect model lies on the y=x diagonal.
-   **Confidence Distributions**: Violin plots showing the shape of each model's confidence distribution.
-   **Per-Class ECE**: A bar chart showing calibration error for each class, identifying unreliable classes.
-   **Uncertainty Landscape**: A 2D density plot of confidence vs. entropy, showing the model's uncertainty profile.

#### 6. Information Flow Analysis (`information_flow_analysis.png`)

Diagnoses how information propagates through the network.

-   **Activation Flow Overview**: Tracks activation mean and standard deviation to spot vanishing/exploding signals.
-   **Effective Rank Evolution**: Plots the dimensionality of information at each layer to find bottlenecks.
-   **Activation Health Dashboard**: A heatmap showing issues like dead or saturated neurons.
-   **Layer Specialization Analysis**: Plots a score measuring how well each layer learns diverse features.

#### 7. Pareto Analysis (`pareto_analysis.png`)

(Generated with `create_pareto_analysis()`) A powerful tool for hyperparameter tuning.

-   **Pareto Front Plot**: A scatter plot of Peak Accuracy vs. Overfitting Index. Models on the "Pareto Front" represent the best possible trade-offs.
-   **Normalized Performance Heatmap**: Compares all models across key metrics, making it easy to identify the best configuration based on priorities.

#### 8. Detailed Spectral Plots (`spectral_plots/*.png`)

These plots provide a layer-by-layer deep dive into the power-law fit that is summarized in the main spectral dashboard. Each plot visualizes the Empirical Spectral Density (ESD) of a single layer's weight matrix.

-   **What it shows**: A log-log plot of the eigenvalue (λ) distribution. A straight line in the tail of this plot is the signature of a power-law.
-   **How to read it**:
    -   The **blue dots** represent the actual binned histogram of the layer's eigenvalues.
    -   The **red line** is the best-fit power-law model, `P(λ) ~ λ^(-α)`. The steepness of this line's slope is the exponent `α`.
    -   The **vertical dashed line (`xmin`)** marks the beginning of the power-law tail, where the fit is applied.
    -   **Interpretation**: A well-trained layer will show the blue dots in the tail (right side of the plot) aligning closely with the red line. A poor fit may indicate that the layer has not developed a clear heavy-tailed structure, which could be a sign of training issues.

## 8. Troubleshooting
-   **`analysis_results.json` is too large**: By default, the analyzer saves summary statistics to keep the file size small. If you need the raw, per-sample data (e.g., the confidence score for every single prediction), you can enable it in the configuration. Be aware this can increase the JSON file size from kilobytes to many megabytes.
    ```python
    config = AnalysisConfig(
        json_include_per_sample_data=True, # Saves raw confidence/entropy arrays
        json_include_raw_esds=True         # Saves raw eigenvalue arrays
    )
    ```
-   **Multi-Input Models**: The analyzer has limited support for models with multiple inputs. It will log warnings and automatically skip incompatible analyses (like calibration and information flow) for these models to prevent errors.
-   **"No training metrics found"**: The analyzer robustly searches for common metric names (`accuracy`, `val_loss`, etc.). If you use non-standard names in your `history` object, analysis will be limited. Ensure your Keras history keys are standard. **See Section 3.3 for the exact required structure of the `training_history` dictionary.**
-   **Memory Issues**: For very large models or datasets, analysis can be memory-intensive. Reduce the sample size and disable the most expensive analyses in `AnalysisConfig`:
    ```python
    config = AnalysisConfig(
        n_samples=500,                  # Reduce from default 1000
        analyze_information_flow=False, # This is often the most memory-intensive
        max_layers_heatmap=8            # Limit heatmap size
    )
    ```
-   **Plots look wrong/empty**: Enable verbose logging (`config = AnalysisConfig(verbose=True)`) and check the console output. You can also inspect the `analysis_results.json` file to see what data was successfully computed.
-   **Spectral Analysis Fails**: Spectral analysis requires layers to have a minimum number of weights (e.g., a 10x10 matrix). Very small layers will be skipped automatically. Check the console log for warnings about skipped layers.

## 9. Extensions

The toolkit is designed to be extensible. You can add your own custom analysis and visualization modules by inheriting from the base classes.

### Creating a Custom Analyzer

Extend the `BaseAnalyzer` class and implement the `analyze` method.

```python
from dl_techniques.analyzer.analyzers.base import BaseAnalyzer
from dl_techniques.analyzer.data_types import AnalysisResults, DataInput

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
import matplotlib.pyplot as plt
from dl_techniques.analyzer.visualizers.base import BaseVisualizer

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

## 10. Theoretical Background & References

The analyses performed by this toolkit are grounded in established research from machine learning, statistics, and statistical physics. Below are key references for the theoretical underpinnings of the main analysis modules.

### Key References

1.  **Martin, C., & Mahoney, M. W. (2021). "Heavy-Tailed Universals in Deep Neural Networks." arXiv preprint arXiv:2106.07590.** (Foundational work on Heavy-Tailed Self-Regularization and its connection to generalization).
2.  **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On calibration of modern neural networks." ICML.** (A seminal paper on model calibration, introducing Expected Calibration Error (ECE) as a standard metric).
3.  **Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). "Power-law distributions in empirical data." SIAM review, 51(4), 661-703.** (Provides the statistical methodology for fitting power-law distributions, which is central to the spectral analysis).
4.  **Roy, O., & Vetterli, M. (2007). "The effective rank: A measure of effective dimensionality." LATS.** (Introduces the concept of "effective rank" used in the information flow analysis to measure feature dimensionality).
5.  **Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.** (A comprehensive textbook covering many of the foundational concepts used in the analyzer, such as overfitting, norms, and training dynamics).