# WeightWatcher

A comprehensive diagnostic tool for analyzing neural network weight matrices using spectral methods, power-law analysis, and concentration metrics. WeightWatcher helps assess training quality, model complexity, and generalization potential without requiring test data.

## Features

- **Power-Law Analysis**: Fit eigenvalue distributions to power-law models to assess training quality
- **Spectral Metrics**: Calculate matrix entropy, stable rank, and other spectral properties
- **Concentration Analysis**: Identify critical weights and information concentration patterns
- **Model Comparison**: Compare models before/after optimization, pruning, or fine-tuning
- **SVD Smoothing**: Apply singular value decomposition to improve model generalization
- **Rich Visualizations**: Generate comprehensive plots and HTML reports
- **Keras Integration**: Native support for Keras models with minimal dependencies

## Installation

```bash
pip install keras tensorflow scipy matplotlib pandas
```

Then add the WeightWatcher module to your project:

```python
from dl_techniques.analysis.weightwatcher import WeightWatcher, analyze_model
```

## Quick Start

### Basic Analysis

```python
import keras
from dl_techniques.analysis.weightwatcher import analyze_model

# Load your model
model = keras.models.load_model('my_model.keras')

# Perform comprehensive analysis
results = analyze_model(
    model, 
    plot=True, 
    concentration_analysis=True,
    savedir='analysis_results'
)

# View summary metrics
print("Summary:", results['summary'])
print("Recommendations:", results['recommendations'])
```

### Detailed Analysis with WeightWatcher Class

```python
from dl_techniques.analysis.weightwatcher import WeightWatcher

# Initialize WeightWatcher
watcher = WeightWatcher(model)

# Analyze with full options
analysis_df = watcher.analyze(
    concentration_analysis=True,  # Enable concentration metrics
    plot=True,                   # Generate visualizations
    randomize=True,              # Compare with randomized weights
    savefig='plots'              # Save plots directory
)

# Get summary metrics
summary = watcher.get_summary()
print(f"Mean α (alpha): {summary['alpha']:.3f}")
print(f"Mean concentration score: {summary['concentration_score']:.3f}")

# Get eigenvalue spectrum for specific layer
evals = watcher.get_ESD(layer_id=5)
print(f"Layer 5 has {len(evals)} eigenvalues")
```

## Key Metrics Explained

### Power-Law Exponent (α)
- **α < 2.0**: May indicate over-training or memorization
- **2.0 < α < 6.0**: Well-trained range, good generalization expected
- **α > 6.0**: May indicate under-training

### Concentration Metrics
- **Gini Coefficient**: Measures inequality in eigenvalue distribution (0=equal, 1=concentrated)
- **Dominance Ratio**: Ratio of largest eigenvalue to sum of others
- **Participation Ratio**: Measures localization in eigenvectors (lower = more concentrated)
- **Concentration Score**: Combined metric indicating information concentration

### Spectral Properties
- **Stable Rank**: Effective rank of weight matrices
- **Matrix Entropy**: Information distribution across eigenvalues
- **Spectral Norm**: Largest eigenvalue (related to gradient flow)

## Advanced Usage

### Model Comparison

```python
from dl_techniques.analysis.weightwatcher import compare_models

# Compare original vs fine-tuned model
comparison = compare_models(
    original_model=base_model,
    modified_model=finetuned_model,
    test_data=(x_test, y_test),
    savedir='comparison_results'
)

# Check how metrics changed
for metric, values in comparison['metric_comparison'].items():
    change = values['percent_change']
    print(f"{metric}: {change:+.1f}% change")
```

### SVD Smoothing

```python
from dl_techniques.analysis.weightwatcher import create_smoothed_model

# Create smoothed version using DetX method
smoothed_model, comparison = create_smoothed_model(
    model,
    method='detX',        # 'svd', 'detX', or 'lambda_min'
    analyze_smoothed=True # Return comparison results
)

# Save smoothed model
smoothed_model.save('smoothed_model.keras')
```

### Critical Layer Analysis

```python
from dl_techniques.analysis.weightwatcher import get_critical_layers

# Find layers with highest concentration
critical_layers = get_critical_layers(
    model, 
    criterion='concentration',  # 'alpha', 'entropy', 'parameters'
    top_k=5
)

for layer_info in critical_layers:
    print(f"Layer {layer_info['layer_id']}: {layer_info['name']}")
    print(f"  Concentration Score: {layer_info['concentration_score']:.3f}")
```

### Using Individual Metric Functions

```python
from dl_techniques.analysis.weightwatcher.metrics import (
    calculate_gini_coefficient,
    calculate_concentration_metrics,
    fit_powerlaw
)

# Calculate specific metrics on your own matrices
import numpy as np

# Example weight matrix
W = np.random.randn(100, 50)

# Calculate concentration metrics
concentration = calculate_concentration_metrics(W)
print(f"Gini coefficient: {concentration['gini_coefficient']:.3f}")
print(f"Critical weights found: {concentration['critical_weight_count']}")

# Fit power law to eigenvalues
eigenvalues = np.linalg.eigvals(W @ W.T)
alpha, xmin, D, sigma, num_spikes, status, warning = fit_powerlaw(eigenvalues)
print(f"Power-law exponent α: {alpha:.3f} (status: {status})")
```

## Understanding the Results

### Training Quality Assessment

```python
summary = watcher.get_summary()

if summary['alpha'] < 2.0:
    print("⚠️  Model may be over-trained")
elif summary['alpha'] > 6.0:
    print("⚠️  Model may be under-trained") 
else:
    print("✅ Model training quality looks good")

if summary['concentration_score'] > 5.0:
    print("⚠️  High information concentration detected")
    print("   Be careful with pruning/quantization")
```

### Layer-by-Layer Analysis

```python
# Analyze specific layers
analysis_df = watcher.analyze()

# Find problematic layers
over_trained = analysis_df[analysis_df['alpha'] < 2.0]
high_concentration = analysis_df[analysis_df['concentration_score'] > 
                                analysis_df['concentration_score'].quantile(0.9)]

print("Over-trained layers:", over_trained['name'].tolist())
print("High concentration layers:", high_concentration['name'].tolist())
```

### Visualization and Reporting

The analysis automatically generates:

- **Power-law fit plots**: Eigenvalue distributions with fitted curves
- **Layer comparison charts**: Metrics across different layers
- **Concentration analysis plots**: Critical weight visualizations
- **HTML reports**: Comprehensive analysis summaries
- **CSV exports**: Detailed numerical results

## File Outputs

When you run an analysis, WeightWatcher creates:

```
analysis_results/
├── analysis_report.html      # Comprehensive HTML report
├── analysis_summary.json     # Summary metrics and recommendations
├── layer_analysis.csv        # Detailed per-layer metrics
├── plots/                    # Visualization plots
│   ├── layer_0_powerlaw.png
│   ├── layer_1_powerlaw.png
│   └── ...
└── detailed_plots/          # Layer weight visualizations (if requested)
    ├── layer_0_weights.png
    └── ...
```

## Best Practices

### For Model Development
1. **Monitor α during training**: Track power-law exponents to detect over/under-training
2. **Check concentration metrics**: High concentration may indicate brittleness
3. **Compare before/after optimization**: Use model comparison for validation

### For Model Optimization
1. **Preserve critical layers**: Avoid aggressive pruning on high-concentration layers
2. **Use SVD smoothing**: Can improve generalization on over-trained models
3. **Validate with test data**: Always check performance impact of optimizations

### For Production Deployment
1. **Document baseline metrics**: Keep analysis results for model versioning
2. **Monitor drift**: Re-analyze models periodically to detect changes
3. **Use for debugging**: Spectral analysis can help diagnose training issues

## API Reference

### Main Functions

- `analyze_model(model, **kwargs)`: Comprehensive model analysis
- `compare_models(model1, model2, **kwargs)`: Compare two models
- `create_smoothed_model(model, method='detX')`: Apply SVD smoothing
- `get_critical_layers(model, criterion='concentration')`: Find important layers

### WeightWatcher Class

- `analyze(**kwargs)`: Perform spectral analysis
- `get_summary()`: Get aggregated metrics
- `get_ESD(layer_id)`: Get eigenvalue spectrum for layer
- `get_layer_concentration_metrics(layer_id)`: Detailed layer analysis

### Utility Functions

- `infer_layer_type(layer)`: Determine layer type
- `compute_weight_statistics(model)`: Basic weight statistics
- `create_weight_visualization(model, layer_id)`: Visualize layer weights

## Requirements

- **Python**: 3.8+
- **Keras**: 3.0+
- **Required packages**: numpy, scipy, matplotlib, pandas
- **Optional**: seaborn (for enhanced plots)

## Contributing

This is part of the `dl_techniques` package. For issues or contributions, please follow the project's contributing guidelines.

## License

This project follows the same license as the parent `dl_techniques` package.

## Citation

If you use WeightWatcher in your research, please consider citing the theoretical foundations:

- Martin, C. H., & Mahoney, M. W. (2019). "Traditional and Heavy-Tail Self Regularization in Neural Network Models"
- Martin, C. H., & Mahoney, M. W. (2021). "Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data"

## Examples and Tutorials

For more examples and detailed tutorials, check the `experiments/` directory in the parent repository, which contains practical applications of WeightWatcher to various model types and optimization scenarios.