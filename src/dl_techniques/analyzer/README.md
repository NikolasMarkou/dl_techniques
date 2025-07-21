# Model Analyzer Module

A comprehensive, modular analyzer for neural networks with training dynamics and refined visualizations.

## Features

- **Comprehensive Weight Analysis**: Distribution statistics, health metrics, and PCA visualization
- **Confidence & Calibration Analysis**: ECE, Brier scores, reliability diagrams, and uncertainty landscapes
- **Information Flow Analysis**: Activation patterns, effective rank evolution, and layer specialization
- **Training Dynamics**: Convergence analysis, overfitting metrics, and training stability
- **Summary Dashboard**: Unified view of all analyses with actionable insights

## Module Structure

```
analyzer/
├── __init__.py                    # Public API exports
├── config.py                      # Configuration classes
├── constants.py                   # All constants
├── data_types.py                  # Data type definitions
├── utils.py                       # Utility functions
├── model_analyzer.py              # Main coordinator class
├── analyzers/                     # Analysis logic
│   ├── __init__.py
│   ├── base.py                    # Base analyzer interface
│   ├── weight_analyzer.py         # Weight analysis
│   ├── calibration_analyzer.py    # Calibration analysis
│   ├── information_flow_analyzer.py # Information flow
│   └── training_dynamics_analyzer.py # Training dynamics
└── visualizers/                   # Visualization logic
    ├── __init__.py
    ├── base.py                    # Base visualizer interface
    ├── weight_visualizer.py       # Weight visualizations
    ├── calibration_visualizer.py  # Calibration plots
    ├── information_flow_visualizer.py # Info flow plots
    ├── training_dynamics_visualizer.py # Training plots
    └── summary_visualizer.py      # Summary dashboard
```

## Quick Start

```python
from dl_techniques.utils.analyzer import ModelAnalyzer, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    analyze_training_dynamics=True,
    plot_style='publication'
)

# Create analyzer
analyzer = ModelAnalyzer(
    models={'model1': model1, 'model2': model2},
    config=config,
    training_history=training_histories  # Optional
)

# Run analysis
results = analyzer.analyze(test_data)
```

## Extending the Analyzer

### Adding a New Analyzer

1. Create a new analyzer class inheriting from `BaseAnalyzer`:

```python
from .base import BaseAnalyzer

class MyNewAnalyzer(BaseAnalyzer):
    def requires_data(self) -> bool:
        return True  # or False
    
    def analyze(self, results, data=None, cache=None):
        # Implement analysis logic
        pass
```

2. Add it to the main `ModelAnalyzer`:

```python
# In model_analyzer.py, _init_analyzers method
self.analyzers['my_new_analysis'] = MyNewAnalyzer(self.models, self.config)
```

### Adding a New Visualizer

1. Create a new visualizer class inheriting from `BaseVisualizer`:

```python
from .base import BaseVisualizer

class MyNewVisualizer(BaseVisualizer):
    def create_visualizations(self):
        # Implement visualization logic
        pass
```

2. Add it to the visualization mapping in `ModelAnalyzer._create_visualizations`.

## Configuration Options

### Analysis Toggles
- `analyze_weights`: Enable weight distribution analysis
- `analyze_calibration`: Enable confidence and calibration analysis
- `analyze_information_flow`: Enable information flow analysis
- `analyze_training_dynamics`: Enable training history analysis

### Sampling Parameters
- `n_samples`: Number of samples to use for analysis
- `calibration_bins`: Number of bins for calibration analysis

### Visualization Settings
- `plot_style`: 'publication', 'presentation', or 'draft'
- `save_plots`: Whether to save plots to disk
- `save_format`: Image format ('png', 'pdf', 'svg')
- `dpi`: Resolution for saved images

### Analysis Options
- `compute_weight_pca`: Perform PCA on final layer weights
- `smooth_training_curves`: Apply smoothing to training curves
- `smoothing_window`: Window size for curve smoothing

## Output

The analyzer creates:
- Individual analysis plots in the output directory
- A summary dashboard combining key insights
- JSON file with all analysis results
- Optional Pareto analysis for model selection

## Requirements

- keras >= 3.8.0
- tensorflow >= 2.18.0
- numpy
- scipy
- matplotlib
- seaborn
- pandas
- scikit-learn
- tqdm

## Tips for Production Use

1. **Memory Management**: For large models, reduce `n_samples` to avoid memory issues
2. **Parallel Analysis**: Analyzers are independent and can be parallelized
3. **Custom Metrics**: Extend analyzers to include domain-specific metrics
4. **Batch Processing**: Process multiple model checkpoints efficiently
5. **CI/CD Integration**: Use the JSON output for automated model quality checks

## Known Limitations

- Information flow analysis requires models with standard Keras layers
- Training dynamics require compatible history format
- **Multi-input models**: Currently have limited support. Calibration analysis is skipped for multi-input models. Information flow analysis attempts simple input splitting but may not work for complex multi-input architectures. Users should extend the analyzer for proper multi-input handling.

## Contributing

When adding new features:
1. Follow the existing module structure
2. Implement proper base class interfaces
3. Add comprehensive docstrings
4. Include type hints
5. Add tests for new functionality