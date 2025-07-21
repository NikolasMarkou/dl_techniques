# Model Analyzer: Complete Usage Guide

A comprehensive, modular analysis toolkit for deep learning models built on Keras 3.8+ and TensorFlow 2.18+. This module provides multi-dimensional model analysis including weight distributions, calibration metrics, information flow patterns, and training dynamics with publication-ready visualizations.

## 1. Introduction

The Model Analyzer is designed to provide deep insights into your neural network models beyond simple accuracy metrics. It helps answer critical questions about model behavior:

- **Weight Health**: Are my model weights well-distributed and healthy?
- **Calibration**: How confident should I be in my model's predictions?
- **Information Flow**: How does information propagate through my network layers?
- **Training Dynamics**: Did my model train efficiently and converge properly?

### Key Features

- 🔍 **Comprehensive Analysis**: Four specialized analysis modules covering different aspects of model behavior
- 📊 **Rich Visualizations**: Publication-ready plots with consistent styling and color schemes
- 🧩 **Modular Design**: Extensible architecture for adding custom analysis components
- 🎯 **Training Insights**: Deep analysis of training history and convergence patterns
- 🎨 **Dashboard**: Unified summary dashboard for model comparison
- 💾 **Serializable Results**: JSON export for reproducible analysis and reporting

### Module Structure

```
analyzer/
├── analyzers/                     # Analysis components
│   ├── base.py                   # Abstract base analyzer interface
│   ├── weight_analyzer.py        # Weight distribution and health analysis
│   ├── calibration_analyzer.py   # Model confidence and calibration metrics
│   ├── information_flow_analyzer.py  # Activation patterns and information flow
│   ├── training_dynamics_analyzer.py # Training history and convergence analysis
│   └── __init__.py
├── visualizers/                   # Visualization components
│   ├── base.py                   # Abstract base visualizer interface
│   ├── weight_visualizer.py      # Weight analysis visualizations
│   ├── calibration_visualizer.py # Calibration and confidence plots
│   ├── information_flow_visualizer.py # Information flow visualizations
│   ├── training_dynamics_visualizer.py # Training dynamics plots
│   ├── summary_visualizer.py     # Unified summary dashboard
│   └── __init__.py
├── config.py                     # Configuration classes and plotting setup
├── data_types.py                 # Structured data types and containers
├── constants.py                  # Analysis constants and thresholds
├── utils.py                      # Utility functions and helpers
├── model_analyzer.py             # Main coordinator class
├── __init__.py                   # Public API exports
└── README.md                     # This file
```

### Core Components

- **ModelAnalyzer**: Main coordinator class that orchestrates all analysis
- **AnalysisConfig**: Configuration class for customizing analysis behavior
- **DataInput**: Structured input data container
- **AnalysisResults**: Comprehensive results container with all analysis outputs

### Design Principles

- **Modularity**: Each analysis type is independent and can be run separately
- **Extensibility**: Easy to add new analyzers and visualizers
- **Consistency**: Unified color schemes and styling across all visualizations
- **Robustness**: Comprehensive error handling and graceful degradation
- **Performance**: Efficient caching and sampling for large datasets

### Technical Features

- **Backend Agnostic**: Built on Keras 3.x for compatibility across backends
- **Memory Efficient**: Smart sampling and caching strategies
- **Publication Ready**: High-quality plots with configurable DPI and formats
- **Serializable**: Complete analysis results can be saved and reloaded
- **Multi-Input Support**: Limited support for complex model architectures

## 2. Analysis Capabilities

### Weight Analysis Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **L1/L2 Norms** | Weight magnitude measures | Higher values indicate larger weights |
| **Spectral Norm** | Largest singular value | Controls Lipschitz constant |
| **Weight Distribution** | Statistical properties (mean, std, skew, kurtosis) | Indicates weight health |
| **Sparsity** | Fraction of near-zero weights | High sparsity may indicate dead neurons |
| **Health Score** | Combined metric (0-1) | Higher = healthier weight distribution |

### Calibration Metrics

| Metric | Description | Range | Ideal Value |
|--------|-------------|-------|-------------|
| **ECE** | Expected Calibration Error | [0, 1] | 0 (perfect calibration) |
| **Brier Score** | Probabilistic accuracy measure | [0, 1] | 0 (perfect predictions) |
| **Reliability** | Bin-wise calibration accuracy | [0, 1] | Close to diagonal |
| **Confidence** | Max probability statistics | [0, 1] | Context dependent |

### Information Flow Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Activation Statistics** | Mean, std, sparsity of activations | Layer health and utilization |
| **Effective Rank** | Information dimensionality | Higher = more diverse representations |
| **Positive Ratio** | Fraction of positive activations | Indicates activation patterns |
| **Specialization Score** | Layer specialization measure | Higher = better feature learning |

### Training Dynamics Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Epochs to Convergence** | Time to reach 95% of peak performance | Lower = faster learning |
| **Overfitting Index** | Val loss - train loss in final third | Positive = overfitting |
| **Training Stability** | Std of recent validation losses | Lower = more stable |
| **Peak Performance** | Best validation metrics achieved | Higher = better model |
| **Final Gap** | Final validation - training loss | Indicates final overfitting state |

## 3. Quick Start

### 5-Minute Setup
```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
import numpy as np

# 1. Prepare your models (dictionary format)
models = {
    'ResNet_v1': your_resnet_model,
    'ConvNext_v2': your_convnext_model
}

# 2. Prepare your test data
test_data = DataInput(x_data=x_test, y_data=y_test)

# 3. Run analysis with defaults
config = AnalysisConfig()
analyzer = ModelAnalyzer(models, config=config, output_dir='analysis_results')
results = analyzer.analyze(test_data)

print("Analysis complete! Check the 'analysis_results' folder for plots.")
```

That's it! The analyzer will generate comprehensive visualizations and save them to your output directory.

## 4. Installation & Setup

### Prerequisites
```bash
pip install keras>=3.8.0 tensorflow>=2.18.0 matplotlib seaborn scikit-learn numpy scipy pandas
```

### Project Structure
```
your_project/
├── models/
│   ├── model_v1.keras
│   └── model_v2.keras
├── data/
│   ├── x_test.npy
│   └── y_test.npy
└── analysis/
    └── run_analysis.py  # Your analysis script
```

## 5. Basic Usage Patterns

### Pattern 1: Single Model Analysis
```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
import keras

# Load your model
model = keras.models.load_model('path/to/your/model.keras')

# Single model analysis
models = {'MyModel': model}
config = AnalysisConfig(
    analyze_weights=True,
    analyze_calibration=True,
    save_plots=True
)

analyzer = ModelAnalyzer(models, config=config)
results = analyzer.analyze(test_data)
```

### Pattern 2: Multi-Model Comparison
```python
# Compare different architectures
models = {
    'ResNet50': resnet_model,
    'EfficientNet': efficientnet_model, 
    'ConvNext': convnext_model,
    'Custom_CNN': custom_model
}

# The analyzer automatically uses consistent colors and creates comparison plots
analyzer = ModelAnalyzer(models, config=config)
results = analyzer.analyze(test_data)
```

### Pattern 3: Hyperparameter Sweep Analysis
```python
# Analyze different hyperparameter configurations
models = {
    'lr_0.001_batch_32': model_1,
    'lr_0.01_batch_32': model_2,
    'lr_0.001_batch_64': model_3,
    'lr_0.01_batch_64': model_4
}

# Enable training dynamics for hyperparameter insights
config = AnalysisConfig(
    analyze_training_dynamics=True,
    pareto_analysis_threshold=2  # Enable Pareto analysis with 2+ models
)

# Include training histories
training_histories = {
    'lr_0.001_batch_32': history_1.history,
    'lr_0.01_batch_32': history_2.history,
    # ... etc
}

analyzer = ModelAnalyzer(
    models=models,
    training_history=training_histories,
    config=config
)
results = analyzer.analyze(test_data)

# Generate Pareto analysis for hyperparameter selection
pareto_fig = analyzer.create_pareto_analysis()
```

## 6. Advanced Configuration

### Complete Configuration Example
```python
config = AnalysisConfig(
    # === Analysis Toggles ===
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    analyze_training_dynamics=True,
    
    # === Sampling Parameters ===
    n_samples=1000,  # Reduce for faster analysis on large datasets
    
    # === Weight Analysis ===
    weight_layer_types=['Dense', 'Conv2D'],  # Only analyze these layer types
    analyze_biases=False,  # Skip bias analysis for speed
    compute_weight_pca=True,  # Enable model similarity analysis
    
    # === Calibration Settings ===
    calibration_bins=15,  # More bins = finer calibration analysis
    
    # === Training Dynamics ===
    smooth_training_curves=True,
    smoothing_window=5,
    
    # === Visualization ===
    plot_style='publication',  # 'publication', 'presentation', or 'draft'
    save_plots=True,
    save_format='png',  # 'png', 'pdf', 'svg'
    dpi=300,
    
    # === Performance Tuning ===
    max_layers_heatmap=12,  # Limit layers shown in weight heatmap
    max_layers_info_flow=8,  # Limit layers in information flow analysis
)
```

### Training History Integration
```python
# Proper training history format
training_histories = {}

for model_name, model in models.items():
    # Assume you have Keras History objects
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), ...)
    training_histories[model_name] = history.history

# Initialize analyzer with training history
analyzer = ModelAnalyzer(
    models=models,
    training_history=training_histories,
    config=config,
    output_dir='training_analysis'
)

results = analyzer.analyze(test_data)
```

## 7. Understanding the Output

### File Structure
After running the analyzer, you'll get:
```
analysis_results/
├── summary_dashboard.png          # Start here - overview of all models
├── training_dynamics.png          # Training curves and convergence analysis
├── weight_learning_journey.png    # Weight magnitude evolution
├── confidence_calibration_analysis.png  # Model confidence assessment
├── information_flow_analysis.png  # Layer-wise information flow
├── pareto_analysis.png           # Hyperparameter optimization insights
└── analysis_results.json         # Raw data for further analysis
```

### Key Visualizations Explained

#### 1. Summary Dashboard (`summary_dashboard.png`)
**What it shows:** 2x2 grid with performance table, model similarity, confidence distributions, and calibration landscape.

**How to read it:**
- **Performance Table**: Start here - shows key metrics for each model
- **Model Similarity**: Models close together have similar weight patterns
- **Confidence Profiles**: Shows how confident each model is in its predictions
- **Calibration Landscape**: Models in bottom-left are well-calibrated

#### 2. Training Dynamics (`training_dynamics.png`)
**What it shows:** Training curves, overfitting analysis, and convergence metrics.

**How to read it:**
- **Loss/Accuracy Curves**: Look for smooth convergence
- **Overfitting Analysis**: Positive values indicate overfitting
- **Best Epoch Performance**: Earlier epochs = faster convergence
- **Summary Table**: Quantitative training metrics

#### 3. Weight Learning Journey (`weight_learning_journey.png`)
**What it shows:** How weight magnitudes evolve through network layers.

**How to read it:**
- **Weight Evolution**: Smooth progression indicates healthy learning
- **Health Heatmap**: Green = healthy weights, Red = potential issues

#### 4. Confidence Calibration Analysis (`confidence_calibration_analysis.png`)
**What it shows:** How well model confidence matches actual accuracy.

**How to read it:**
- **Reliability Diagram**: Points on diagonal = perfect calibration
- **Confidence Distributions**: Wider = more varied confidence
- **Per-Class ECE**: Shows which classes are poorly calibrated

### Accessing Raw Results
```python
# Get structured summary
summary = analyzer.get_summary_statistics()
print(f"Best model: {summary['model_performance']}")

# Access specific metrics
for model_name, metrics in results.calibration_metrics.items():
    print(f"{model_name} ECE: {metrics['ece']:.3f}")
    print(f"{model_name} Brier Score: {metrics['brier_score']:.3f}")

# Training insights (if available)
if results.training_metrics:
    for model_name, epochs in results.training_metrics.epochs_to_convergence.items():
        print(f"{model_name} converged in {epochs} epochs")
```

## 8. Common Use Cases

### Use Case 1: Model Selection
```python
# Scenario: Choose the best model from multiple architectures

models = {
    'ResNet50': resnet50_model,
    'EfficientNetB3': efficientnet_model,
    'ConvNeXtTiny': convnext_model
}

config = AnalysisConfig(
    analyze_calibration=True,  # Important for deployment
    compute_weight_pca=True    # See which models are similar
)

analyzer = ModelAnalyzer(models, config=config)
results = analyzer.analyze(test_data)

# Decision criteria:
# 1. Check summary_dashboard.png - look at performance table
# 2. Check calibration_landscape - want bottom-left quadrant
# 3. Consider model similarity - diverse models may ensemble better
```

### Use Case 2: Debugging Training Issues
```python
# Scenario: Model isn't training well, need to diagnose

config = AnalysisConfig(
    analyze_training_dynamics=True,
    analyze_weights=True,
    smooth_training_curves=True,
    analyze_information_flow=True  # Check for dead neurons
)

analyzer = ModelAnalyzer(
    models={'problematic_model': model},
    training_history={'problematic_model': training_history},
    config=config
)

results = analyzer.analyze(test_data)

# Look for:
# 1. training_dynamics.png - overfitting? slow convergence?
# 2. weight_learning_journey.png - dead layers? exploding weights?
# 3. information_flow_analysis.png - dead neurons? poor specialization?
```

### Use Case 3: Production Readiness Assessment
```python
# Scenario: Is this model ready for production deployment?

config = AnalysisConfig(
    analyze_calibration=True,    # Critical for production
    calibration_bins=20,         # Detailed calibration analysis
    analyze_information_flow=True # Check layer health
)

analyzer = ModelAnalyzer(
    models={'production_candidate': model},
    config=config
)

results = analyzer.analyze(production_test_data)

# Production readiness checklist:
# 1. ECE < 0.05 (good calibration)
# 2. Brier score reasonable for your domain
# 3. No dead layers in information flow analysis
# 4. Confidence distribution makes sense for your use case
```

### Use Case 4: Hyperparameter Optimization Results
```python
# Scenario: Ran a hyperparameter sweep, need to pick best config

models = {f'config_{i}': model for i, model in enumerate(sweep_models)}
histories = {f'config_{i}': hist for i, hist in enumerate(sweep_histories)}

config = AnalysisConfig(
    analyze_training_dynamics=True,
    pareto_analysis_threshold=2
)

analyzer = ModelAnalyzer(models, training_history=histories, config=config)
results = analyzer.analyze(validation_data)

# Generate Pareto front
pareto_fig = analyzer.create_pareto_analysis()

# Decision process:
# 1. Look at pareto_analysis.png for optimal trade-offs
# 2. Check training_dynamics.png for stability
# 3. Consider deployment constraints (inference speed, memory)
```

### Use Case 5: Production Model Monitoring
```python
# Monitor deployed model health
config = AnalysisConfig(
    analyze_weights=True,
    analyze_calibration=True,
    compute_weight_pca=True
)

analyzer = ModelAnalyzer({'production_model': model}, config=config)
results = analyzer.analyze(recent_data)

# Check for weight drift, calibration degradation
summary = analyzer.get_summary_statistics()
```

## 9. Troubleshooting

### Common Issues and Solutions

#### Issue 1: "RuntimeError: dictionary changed size during iteration"
**Cause:** Bug in model evaluation loop (fixed in latest version)
**Solution:** Update to latest version or check model evaluation code

#### Issue 2: Multi-input models give warnings/errors
**Cause:** Analyzer has limited support for complex multi-input architectures
```python
# Solution: Handle multi-input models explicitly
if model_name in analyzer._multi_input_models:
    logger.warning(f"Limited analysis for multi-input model {model_name}")
    # Some analyses will be skipped automatically
```

#### Issue 3: "No training metrics found"
**Cause:** Training history uses non-standard metric names
```python
# Solution: Check your history keys
print("Available metrics:", list(training_history.keys()))

# Make sure you have standard names like:
# 'loss', 'val_loss', 'accuracy', 'val_accuracy'
```

#### Issue 4: Memory issues with large models/datasets
```python
# Solution: Reduce sampling
config = AnalysisConfig(
    n_samples=500,  # Reduce from default 1000
    analyze_information_flow=False,  # Skip memory-intensive analysis
    max_layers_heatmap=8  # Limit layer analysis
)
```

#### Issue 5: Plots look wrong/empty
**Cause:** Missing data or configuration issues
```python
# Debug: Check what data is available
summary = analyzer.get_summary_statistics()
print("Available analyses:", summary['analyses_performed'])

# Check individual results
print("Calibration metrics:", bool(results.calibration_metrics))
print("Training metrics:", bool(results.training_metrics))
```

### Debug Mode
```python
# Enable verbose logging for debugging
config = AnalysisConfig(verbose=True)

# Check results step by step
results = analyzer.analyze(test_data)
print("Analysis completed. Available data:")
print(f"- Model metrics: {len(results.model_metrics)}")
print(f"- Weight stats: {len(results.weight_stats)}")
print(f"- Calibration: {len(results.calibration_metrics)}")
```

## 10. Best Practices

### 1. Data Preparation
```python
# Use representative test data
test_data = DataInput(x_data=x_test, y_data=y_test)

# For large datasets, sampling is automatic but you can control it
config = AnalysisConfig(n_samples=1000)  # Adjust based on memory/speed needs

# Ensure your test data matches training distribution
```

### 2. Model Naming
```python
# Use descriptive names for easy interpretation
models = {
    'ResNet50_Adam_lr001': model1,  # Architecture + optimizer + hyperparams
    'ResNet50_SGD_lr01': model2,
    'EfficientNet_Adam_lr001': model3
}

# Avoid generic names like 'model1', 'model2'
```

### 3. Configuration Strategy
```python
# Start with defaults for quick overview
config = AnalysisConfig()

# Then enable specific analyses based on your needs
if need_training_insights:
    config.analyze_training_dynamics = True
    
if preparing_for_production:
    config.analyze_calibration = True
    config.calibration_bins = 20  # More detailed calibration

if debugging_model:
    config.analyze_information_flow = True
    config.analyze_weights = True
```

### 4. Workflow Integration
```python
# Example training + analysis workflow
def train_and_analyze(model_configs, x_train, y_train, x_val, y_val, x_test, y_test):
    models = {}
    histories = {}
    
    # Training phase
    for name, config in model_configs.items():
        model = create_model(config)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val))
        
        models[name] = model
        histories[name] = history.history
    
    # Analysis phase
    analyzer = ModelAnalyzer(
        models=models,
        training_history=histories,
        config=AnalysisConfig(analyze_training_dynamics=True),
        output_dir=f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    test_data = DataInput(x_data=x_test, y_data=y_test)
    results = analyzer.analyze(test_data)
    
    return models, results
```

### 5. Results Interpretation
```python
# Always start with the summary dashboard
summary = analyzer.get_summary_statistics()
print("Models analyzed:", summary['n_models'])
print("Analyses completed:", summary['analyses_performed'])

# Look for red flags
for model_name, metrics in results.calibration_metrics.items():
    if metrics['ece'] > 0.1:  # Poor calibration
        print(f"WARNING: {model_name} has poor calibration (ECE={metrics['ece']:.3f})")
    
    if metrics['brier_score'] > 0.3:  # Poor probabilistic performance
        print(f"WARNING: {model_name} has poor Brier score ({metrics['brier_score']:.3f})")
```

### 6. Saving and Sharing Results
```python
# Results are automatically saved to output_dir
# Share the entire directory for complete analysis

# For presentations, use publication style
config = AnalysisConfig(
    plot_style='presentation',  # Larger fonts, clearer plots
    dpi=300,                   # High resolution
    save_format='png'          # Universal format
)

# For papers, use publication style with PDF
config = AnalysisConfig(
    plot_style='publication',
    save_format='pdf'
)
```

### 7. Performance Optimization
```python
# For large-scale analysis
config = AnalysisConfig(
    n_samples=500,                    # Reduce sampling
    analyze_information_flow=False,   # Skip if not needed (memory intensive)
    max_layers_heatmap=8,            # Limit layer analysis
    max_layers_info_flow=6,          # Limit information flow layers
)

# For quick analysis during development
config = AnalysisConfig(
    analyze_weights=False,           # Skip if not needed
    analyze_information_flow=False,  # Skip expensive analysis
    save_plots=False                 # Skip saving for speed
)
```

## 11. Extensions

### Creating Custom Analyzers

Extend the `BaseAnalyzer` class to create custom analysis components:

```python
from analyzer.analyzers.base import BaseAnalyzer
from analyzer.data_types import AnalysisResults, DataInput

class CustomAnalyzer(BaseAnalyzer):
    """Custom analyzer for specific domain metrics."""
    
    def requires_data(self) -> bool:
        """Return True if this analyzer needs input data."""
        return True
    
    def analyze(self, results: AnalysisResults, 
                data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Perform custom analysis."""
        # Your custom analysis logic here
        for model_name, model in self.models.items():
            # Compute custom metrics
            custom_metrics = self._compute_custom_metrics(model, data)
            
            # Store in results
            if not hasattr(results, 'custom_metrics'):
                results.custom_metrics = {}
            results.custom_metrics[model_name] = custom_metrics
    
    def _compute_custom_metrics(self, model, data):
        """Implement your custom metric computation."""
        return {'custom_score': 0.95}
```

### Creating Custom Visualizers

Extend the `BaseVisualizer` class for custom visualizations:

```python
from analyzer.visualizers.base import BaseVisualizer
import matplotlib.pyplot as plt

class CustomVisualizer(BaseVisualizer):
    """Custom visualizer for domain-specific plots."""
    
    def create_visualizations(self) -> None:
        """Create custom visualizations."""
        if not hasattr(self.results, 'custom_metrics'):
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_custom_analysis(ax)
        
        if self.config.save_plots:
            self._save_figure(fig, 'custom_analysis')
        plt.close(fig)
    
    def _plot_custom_analysis(self, ax):
        """Implement your custom plotting logic."""
        # Your plotting code here
        for model_name, metrics in self.results.custom_metrics.items():
            color = self.model_colors.get(model_name, '#333333')
            # Plot your custom metrics
            pass
```

### Integration with Analysis Pipeline

Register your custom components:

```python
# Add to analyzer initialization
analyzer = ModelAnalyzer(models, config)
analyzer.analyzers['custom'] = CustomAnalyzer(models, config)

# Add custom visualizer
custom_viz = CustomVisualizer(results, config, output_dir, model_colors)
custom_viz.create_visualizations()
```

---

This comprehensive guide should get you started with practical usage of the Model Analyzer. The key is to start simple with default settings, then gradually enable more specific analyses based on your needs. The modular design makes it easy to focus on the aspects most relevant to your particular use case, whether that's model selection, training debugging, or production readiness assessment.