# TensorFlow WeightWatcher

A diagnostic tool for analyzing TensorFlow/Keras neural network weight matrices.

## Overview

TensorFlow WeightWatcher is a tool that analyzes the weight matrices of neural networks to assess:

- Training quality (over/under-training)
- Effective complexity of the model
- Generalization potential
- Opportunities for model compression

By analyzing the statistical properties of the weight matrices, TensorFlow WeightWatcher can provide insights without requiring test data.

## Installation

```bash
pip install tf-weightwatcher
```

## Quick Start

```python
import tensorflow as tf
from tensorflow import keras
from tf_weightwatcher import TFWeightWatcher

# Load a model
model = keras.models.load_model('my_model.keras')

# Initialize the watcher
watcher = TFWeightWatcher(model)

# Analyze the model
analysis = watcher.analyze(plot=True, savefig='output_dir')

# Get summary metrics
summary = watcher.get_summary()
print(summary)

# Generate HTML report
from tf_weightwatcher import generate_report
generate_report(analysis, save_path="report.html")
```

## Key Features

### 1. Model Analysis

Analyze the weight matrices of TensorFlow/Keras models to extract key metrics:

- Power-law exponents (α) to assess training quality
- Stable rank to evaluate effective parameter usage
- Matrix entropy to measure information distribution
- Rank loss to identify redundancy

### 2. SVD Smoothing

Apply Singular Value Decomposition (SVD) smoothing to improve model generalization:

```python
from tf_weightwatcher import smooth_model

# Apply SVD smoothing with DetX method
smoothed_model = smooth_model(model, method='detX')

# Save the smoothed model
smoothed_model.save('smoothed_model.keras')
```

### 3. Visualization

Create visualizations to interpret analysis results:

```python
from tf_weightwatcher import WeightWatcherVisualizer

# Plot distribution of power-law exponents
WeightWatcherVisualizer.plot_alpha_distribution(analysis, save_path='alpha_dist.png')

# Plot metrics by layer
WeightWatcherVisualizer.plot_layer_metrics(analysis, save_path='layer_metrics.png')

# Compare original and smoothed models
smoothed_analysis = watcher.analyze(model=smoothed_model)
WeightWatcherVisualizer.compare_models(analysis, smoothed_analysis, save_path='comparison.png')
```

## Advanced Usage: Direct Access to Metrics Functions

For advanced applications, you can directly use the underlying metric calculation functions:

```python
from tf_weightwatcher import (
    compute_eigenvalues,
    fit_powerlaw,
    calculate_matrix_entropy,
    calculate_spectral_metrics,
    jensen_shannon_distance
)

# Get eigenvalues from a weight matrix
import numpy as np
W = np.random.randn(100, 50)
N, M = W.shape
evals, sv_max, sv_min, rank_loss = compute_eigenvalues([W], N, M, M)

# Fit power law to eigenvalues
alpha, xmin, D, sigma, num_pl_spikes, status, warning = fit_powerlaw(evals)

# Calculate other metrics
entropy = calculate_matrix_entropy(np.sqrt(evals), N)
spectral_metrics = calculate_spectral_metrics(evals, alpha)

print(f"Alpha: {alpha}, Entropy: {entropy}")
print(f"Stable rank: {spectral_metrics['stable_rank']}")
```

## Understanding the Results

### Power-Law Exponent (α)

- α < 2.0: May indicate over-training or memorization
- 2.0 < α < 6.0: Well-trained range
- α > 6.0: May indicate under-training

### Stable Rank

The stable rank is a robust measure of the effective rank of a matrix:
- Higher values indicate more effective parameters
- Low values may indicate redundancy or over-parameterization

### Matrix Entropy

Matrix entropy measures the "flatness" of the eigenvalue distribution:
- Higher values indicate more uniformly distributed eigenvalues
- Lower values indicate concentration of information in fewer dimensions

## Module Structure

- `tf_weightwatcher.py`: Main class for analyzing Keras models
- `metrics.py`: Standalone functions for computing matrix metrics
- `visualization.py`: Tools for visualizing analysis results
- `constants.py`: Centralized constants and configuration

## Citation

If you use TensorFlow WeightWatcher in your research, please cite:

```
@misc{tensorflow-weightwatcher,
  author = {TensorFlow WeightWatcher Contributors},
  title = {TensorFlow WeightWatcher: A Diagnostic Tool for TensorFlow/Keras Models},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/tf-weightwatcher}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.