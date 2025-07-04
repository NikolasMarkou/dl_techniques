# ModelAnalyzer User Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Analysis Types](#analysis-types)
5. [Advanced Features](#advanced-features)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Overview

The ModelAnalyzer is a comprehensive tool for analyzing and comparing neural network models in Keras. It provides deep insights into model behavior, performance, calibration, and training dynamics through automated analysis and visualization.

### Key Features

- **Multi-Model Comparison**: Analyze and compare multiple models simultaneously
- **Weight Analysis**: Examine weight distributions, norms, and health metrics
- **Calibration Analysis**: Evaluate prediction confidence and reliability
- **Information Flow**: Analyze activation patterns and feature representations
- **Training Dynamics**: Understand convergence behavior and overfitting patterns
- **Rich Visualizations**: Publication-ready plots and dashboards
- **Automated Reporting**: Generate comprehensive analysis reports

### What You Can Discover

- Which models are best calibrated for reliable predictions
- How different architectures or loss functions affect learning
- Whether models are overfitting or converging properly
- Which layers are most specialized or suffering from dead neurons
- How weight distributions evolve across different approaches
- Pareto-optimal models for balancing accuracy and robustness

## Quick Start

### Basic Usage

```python
from dl_techniques.utils.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# 1. Prepare your models (dictionary mapping names to Keras models)
models = {
    'Model_A': trained_model_a,
    'Model_B': trained_model_b,
    'Model_C': trained_model_c
}

# 2. Prepare test data
test_data = DataInput(x_data=x_test, y_data=y_test)

# 3. Create analyzer with default settings
analyzer = ModelAnalyzer(models=models)

# 4. Run comprehensive analysis
results = analyzer.analyze(data=test_data)

# 5. Access results
print(f"Models analyzed: {len(results.model_metrics)}")
```

### With Training History

```python
# Include training history for training dynamics analysis
training_histories = {
    'Model_A': history_a.history,
    'Model_B': history_b.history,
    'Model_C': history_c.history
}

analyzer = ModelAnalyzer(
    models=models,
    training_history=training_histories
)

results = analyzer.analyze(data=test_data)
```

## Configuration

### AnalysisConfig Options

```python
from dl_techniques.utils.analyzer import AnalysisConfig

config = AnalysisConfig(
    # Analysis toggles
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    analyze_training_dynamics=True,
    
    # Sampling parameters
    n_samples=1000,                    # Number of samples for analysis
    n_samples_per_digit=3,             # Samples per class for visualization
    
    # Calibration settings
    calibration_bins=15,               # Number of bins for ECE calculation
    
    # Training analysis
    smooth_training_curves=True,       # Apply smoothing to training curves
    smoothing_window=5,                # Window size for smoothing
    
    # Visualization settings
    plot_style='publication',          # 'publication', 'presentation', 'draft'
    color_palette='deep',              # Seaborn color palette
    fig_width=12,                      # Figure width
    fig_height=8,                      # Figure height
    dpi=300,                          # Resolution for saved plots
    save_plots=True,                  # Whether to save plots
    save_format='png',                # Format for saved plots
    
    # Advanced options
    show_statistical_tests=True,       # Include statistical significance tests
    show_confidence_intervals=True,    # Show confidence intervals
    verbose=True                       # Verbose logging
)

analyzer = ModelAnalyzer(models=models, config=config)
```

### Output Directory Structure

```python
from pathlib import Path
from datetime import datetime

# Custom output directory
output_dir = Path("results") / datetime.now().strftime('%Y%m%d_%H%M%S')

analyzer = ModelAnalyzer(
    models=models,
    config=config,
    output_dir=output_dir
)
```

The analyzer creates the following directory structure:
```
output_dir/
├── weight_learning_journey.png
├── confidence_calibration_analysis.png
├── information_flow_analysis.png
├── training_dynamics.png
├── enhanced_summary_dashboard.png
├── pareto_analysis.png (if applicable)
└── analysis_results.json
```

## Analysis Types

### 1. Weight Analysis

Examines weight distributions, norms, and health across layers.

**What it reveals:**
- Weight magnitude evolution through network depth
- Layer health indicators (dead neurons, saturation)
- Weight distribution statistics (mean, std, skewness, kurtosis)
- L1, L2, and spectral norms

**Key visualizations:**
- Weight Learning Journey: Evolution of weight magnitudes
- Weight Health Heatmap: Layer-by-layer health metrics

```python
# Enable only weight analysis
config = AnalysisConfig(
    analyze_weights=True,
    analyze_calibration=False,
    analyze_information_flow=False,
    analyze_training_dynamics=False
)
```

### 2. Calibration Analysis

Evaluates how well predicted probabilities reflect true confidence.

**What it reveals:**
- Expected Calibration Error (ECE)
- Brier Score for prediction quality
- Reliability diagrams
- Confidence distribution patterns
- Per-class calibration performance

**Key visualizations:**
- Reliability Diagrams with confidence intervals
- Confidence Distribution (Raincloud plots)
- Per-Class ECE comparison
- Uncertainty Landscape (density contours)

```python
# Focus on calibration with custom binning
config = AnalysisConfig(
    analyze_calibration=True,
    calibration_bins=20,  # More fine-grained calibration analysis
    analyze_weights=False,
    analyze_information_flow=False,
    analyze_training_dynamics=False
)
```

### 3. Information Flow Analysis

Analyzes how information flows through the network and activation patterns.

**What it reveals:**
- Layer-wise activation statistics
- Effective rank (information dimensionality)
- Activation health (dead neurons, saturation)
- Layer specialization metrics
- Information bottleneck characteristics

**Key visualizations:**
- Activation Flow Overview
- Effective Rank Evolution
- Activation Health Dashboard
- Layer Specialization Analysis

```python
# Deep dive into information flow
config = AnalysisConfig(
    analyze_information_flow=True,
    n_samples=2000,  # More samples for better statistics
    analyze_weights=False,
    analyze_calibration=False,
    analyze_training_dynamics=False
)
```

### 4. Training Dynamics Analysis

Analyzes training behavior and convergence patterns (requires training history).

**What it reveals:**
- Epochs to convergence
- Training stability scores
- Overfitting indices
- Peak performance timing
- Training curve smoothing

**Key visualizations:**
- Loss and Accuracy Evolution
- Overfitting Analysis
- Best Epoch Performance
- Training Summary Table

```python
# Training dynamics with custom smoothing
config = AnalysisConfig(
    analyze_training_dynamics=True,
    smooth_training_curves=True,
    smoothing_window=10,  # Stronger smoothing
    analyze_weights=False,
    analyze_calibration=False,
    analyze_information_flow=False
)
```

## Advanced Features

### Selective Analysis

Run only specific analysis types:

```python
# Run only calibration and training analysis
results = analyzer.analyze(
    data=test_data,
    analysis_types={'calibration', 'training_dynamics'}
)
```

### Custom Data Input

```python
# From tuple
data = DataInput.from_tuple((x_test, y_test))

# From object with attributes
class TestData:
    def __init__(self, x, y):
        self.x_test = x
        self.y_test = y

test_obj = TestData(x_test, y_test)
data = DataInput.from_object(test_obj)
```

### Pareto Analysis

For comparing many models (>10) to find Pareto-optimal solutions:

```python
# Generate Pareto analysis (automatically triggered for 10+ models)
if len(models) >= 10:
    pareto_fig = analyzer.create_pareto_analysis(save_plot=True)
```

### Summary Statistics

```python
# Get quantitative summary
summary = analyzer.get_summary_statistics()

print(f"Number of models: {summary['n_models']}")
print(f"Analyses performed: {summary['analyses_performed']}")

# Access specific metrics
for model_name, perf in summary['model_performance'].items():
    print(f"{model_name}: {perf['accuracy']:.3f} accuracy")
```

## Interpreting Results

### Weight Analysis Results

```python
# Access weight statistics
weight_stats = results.weight_stats['Model_A']['conv2d_w0']

print(f"L2 norm: {weight_stats['norms']['l2']}")
print(f"Sparsity: {weight_stats['distribution']['zero_fraction']}")
print(f"Mean: {weight_stats['basic']['mean']}")
```

**Health Indicators:**
- **Green regions**: Healthy weights with good distribution
- **Yellow regions**: Moderate concerns (high sparsity or extreme values)
- **Red regions**: Problematic weights (dead neurons, saturation)

### Calibration Results

```python
# Access calibration metrics
cal_metrics = results.calibration_metrics['Model_A']

print(f"ECE: {cal_metrics['ece']:.4f}")  # Lower is better
print(f"Brier Score: {cal_metrics['brier_score']:.4f}")  # Lower is better
```

**Interpretation:**
- **ECE < 0.1**: Well-calibrated model
- **ECE 0.1-0.2**: Moderately calibrated
- **ECE > 0.2**: Poorly calibrated (overconfident or underconfident)

### Training Dynamics Results

```python
# Access training metrics
if results.training_metrics:
    conv_epochs = results.training_metrics.epochs_to_convergence
    stability = results.training_metrics.training_stability_score
    overfit_index = results.training_metrics.overfitting_index
    
    for model in conv_epochs:
        print(f"{model}: {conv_epochs[model]} epochs to converge")
        print(f"  Stability: {stability[model]:.4f}")
        print(f"  Overfitting: {overfit_index[model]:+.4f}")
```

**Interpretation:**
- **Positive overfitting index**: Model is overfitting (val_loss > train_loss)
- **Negative overfitting index**: Model is underfitting
- **Lower stability score**: More stable training

## Best Practices

### 1. Data Preparation

```python
# Ensure consistent data format
assert len(x_test.shape) == 4  # For image data
assert len(y_test.shape) == 2  # For one-hot encoded labels

# Use appropriate sample size
config.n_samples = min(1000, len(x_test))  # Don't exceed available data
```

### 2. Model Naming

```python
# Use descriptive model names
models = {
    'ResNet50_CE': model_resnet_crossentropy,
    'ResNet50_FL': model_resnet_focal,
    'MobileNet_GAL': model_mobilenet_goodhart
}
```

### 3. Training History Format

```python
# Ensure history is in the correct format
training_histories = {
    'Model_A': {
        'loss': [0.8, 0.6, 0.4, ...],
        'val_loss': [0.9, 0.7, 0.5, ...],
        'accuracy': [0.6, 0.7, 0.8, ...],
        'val_accuracy': [0.5, 0.6, 0.7, ...]
    }
}
```

### 4. Memory Management

```python
# For large models or datasets
config.n_samples = 500  # Reduce sample size
config.analyze_information_flow = False  # Skip memory-intensive analysis

# Clear models after analysis if needed
del models
gc.collect()
```

### 5. Visualization Settings

```python
# For publications
config.plot_style = 'publication'
config.dpi = 300
config.save_format = 'pdf'

# For presentations
config.plot_style = 'presentation'
config.fig_width = 16
config.fig_height = 10
```

## Examples

### Example 1: Activation Function Comparison

```python
# From the MNIST experiment
activations = ['ReLU', 'GELU', 'Mish', 'Tanh']
models = {}
histories = {}

for activation in activations:
    # Train model with specific activation
    model = train_mnist_model(activation)
    models[activation] = model
    histories[activation] = model.history.history

# Analyze
config = AnalysisConfig(
    analyze_weights=True,
    analyze_calibration=True,
    analyze_training_dynamics=True,
    plot_style='publication'
)

analyzer = ModelAnalyzer(
    models=models,
    config=config,
    training_history=histories
)

results = analyzer.analyze(data=DataInput.from_object(mnist_data))
```

### Example 2: Loss Function Comparison

```python
# From the CIFAR-10 experiment
loss_functions = ['CrossEntropy', 'FocalLoss', 'GoodhartAware']
models = {}
histories = {}

for loss_name in loss_functions:
    # Train model with specific loss
    model = train_cifar10_model(loss_name)
    models[loss_name] = model
    histories[loss_name] = model.history.history

# Focus on calibration analysis
config = AnalysisConfig(
    analyze_calibration=True,
    analyze_training_dynamics=True,
    calibration_bins=20,
    smooth_training_curves=True
)

analyzer = ModelAnalyzer(
    models=models,
    config=config,
    training_history=histories
)

results = analyzer.analyze(data=DataInput.from_object(cifar10_data))

# Print calibration summary
for model_name, metrics in results.calibration_metrics.items():
    print(f"{model_name}: ECE={metrics['ece']:.4f}, "
          f"Brier={metrics['brier_score']:.4f}")
```

### Example 3: Architecture Comparison

```python
# Compare different architectures
architectures = {
    'ResNet18': build_resnet18(),
    'MobileNetV2': build_mobilenetv2(),
    'EfficientNetB0': build_efficientnet()
}

# Train all models
trained_models = {}
for name, model in architectures.items():
    history = train_model(model, train_data, val_data)
    trained_models[name] = model

# Comprehensive analysis
config = AnalysisConfig(
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    plot_style='publication',
    save_plots=True
)

analyzer = ModelAnalyzer(
    models=trained_models,
    config=config,
    output_dir=Path("architecture_comparison")
)

results = analyzer.analyze(data=test_data)

# Generate Pareto analysis for model selection
if len(trained_models) >= 3:
    pareto_fig = analyzer.create_pareto_analysis()
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors

```python
# Reduce sample size
config.n_samples = 200

# Disable memory-intensive analyses
config.analyze_information_flow = False

# Use smaller batch sizes for prediction
# The analyzer automatically handles this
```

#### 2. No Training History

```python
# If you don't have training history
config.analyze_training_dynamics = False

# Or provide empty histories
training_histories = {model_name: {} for model_name in models.keys()}
```

#### 3. Shape Mismatches

```python
# Ensure correct data shapes
print(f"Data shapes: {x_test.shape}, {y_test.shape}")

# Convert integer labels to one-hot if needed
if len(y_test.shape) == 1:
    y_test = keras.utils.to_categorical(y_test, num_classes)
```

#### 4. Missing Dependencies

```python
# Required imports
from dl_techniques.utils.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.calibration_metrics import compute_ece, compute_brier_score
```

#### 5. Custom Layer Issues

```python
# For models with custom layers, ensure they're registered
@keras.saving.register_keras_serializable()
class CustomLayer(keras.layers.Layer):
    # Your custom layer implementation
    pass

# Or disable weight analysis for problematic layers
config.weight_layer_types = ['Dense', 'Conv2D']  # Only analyze these types
```

### Performance Tips

1. **Start Small**: Begin with `n_samples=100` to ensure everything works
2. **Selective Analysis**: Only enable analyses you need
3. **Batch Processing**: For many models, analyze in batches
4. **Memory Monitoring**: Use `gc.collect()` between analyses
5. **Parallel Processing**: The analyzer is not thread-safe, analyze models sequentially

### Getting Help

The analyzer provides extensive logging. Enable verbose mode to see detailed progress:

```python
config = AnalysisConfig(verbose=True)

# Check the logs for detailed information about what's happening
from dl_techniques.utils.logger import logger
logger.info("Starting analysis...")
```

For debugging, access intermediate results:

```python
# Check if models were properly cached
print(f"Cached predictions: {list(analyzer._prediction_cache.keys())}")

# Examine configuration
print(f"Analysis config: {config.__dict__}")

# Check results structure
print(f"Available results: {list(results.__dict__.keys())}")
```

This comprehensive guide should help you effectively use the ModelAnalyzer for deep insights into your neural network models. The tool is designed to provide actionable insights for model selection, debugging, and optimization.