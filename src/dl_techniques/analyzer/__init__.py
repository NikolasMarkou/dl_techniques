"""
Model Analyzer Module
============================================================================

A comprehensive, modular analyzer with training dynamics and refined visualizations

Key Features:
- Comprehensive training dynamics analysis
- Weight distribution and health analysis
- Confidence and calibration metrics
- Information flow through network layers
- Spectral weight analysis (WeightWatcher integration)
- Quantitative training metrics
- Summary dashboard with training insights

Example Usage:
    ```python
    from dl_techniques.utils.analyzer import ModelAnalyzer, AnalysisConfig

    # Configure analysis
    config = AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        analyze_training_dynamics=True,
        analyze_spectral=True, # Enable spectral analysis
        plot_style='publication'
    )

    # Create analyzer with training history
    analyzer = ModelAnalyzer(
        models=models,
        config=config,
        training_history=training_histories
    )

    # Run comprehensive analysis
    results = analyzer.analyze(test_data)
    ```

Multi-Input Model Support:
    The analyzer has limited support for multi-input models. For simple cases:
    ```python
    from dl_techniques.utils.analyzer import DataInput

    # Concatenate multiple inputs
    data = DataInput.from_multi_input([input1, input2], targets)
    ```

    For complex multi-input architectures, extend the analyzer classes.
"""

# Public API exports
from .model_analyzer import ModelAnalyzer
from .config import AnalysisConfig
from .data_types import DataInput, AnalysisResults, TrainingMetrics
from .constants import LayerType, SmoothingMethod, StatusCode, MetricNames

__all__ = [
    'ModelAnalyzer',
    'AnalysisConfig',
    'DataInput',
    'AnalysisResults',
    'TrainingMetrics',
    'LayerType',
    'SmoothingMethod',
    'StatusCode',
    'MetricNames'
]

__version__ = '1.1.0' # Updated version