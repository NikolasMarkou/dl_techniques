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
    from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig

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

Building a DataInput:
    ```python
    from dl_techniques.analyzer import DataInput

    # Directly, from arrays
    data = DataInput(x_data=x_test, y_data=y_test)

    # From a plain (x, y) tuple
    data = DataInput.from_tuple((x_test, y_test))

    # From any object exposing x_data/y_data (or the legacy x_test/y_test)
    data = DataInput.from_object(my_dataset)
    ```

Multi-Input Model Support:
    The analyzer has limited support for multi-input models. Pass the inputs as the
    dict the model itself expects — `x_data` accepts a `Dict[str, Any]`, and the
    information-flow analyzer subsamples that dict key by key:
    ```python
    data = DataInput(x_data={"left": left_inputs, "right": right_inputs}, y_data=targets)
    ```
    Multi-input models are detected at construction and reported in a WARNING; their
    calibration and information-flow analyses are limited. For complex multi-input
    architectures, extend the analyzer classes.

    There is no `DataInput.from_multi_input`, and there is no
    `dl_techniques.utils.analyzer` module: this docstring advertised both for months.
    Every import path and attribute named above is now executed by
    `tests/test_analyzer/test_analyzer_docs.py -k init_docstring`.
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