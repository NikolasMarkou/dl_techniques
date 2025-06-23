"""
Enhanced TensorFlow WeightWatcher - Comprehensive neural network weight analysis

A diagnostic tool for analyzing TensorFlow/Keras neural network weight matrices using
spectral methods, power-law analysis, and concentration metrics.

Main Features:
- Power-law analysis of eigenvalue distributions
- Spectral metrics (entropy, stable rank, etc.)
- Concentration analysis (Gini coefficient, dominance ratio, participation ratio)
- Model comparison and smoothing capabilities
- Comprehensive visualization and reporting

Example Usage:
    ```python
    import keras
    from dl_techniques.analysis.tf_weightwatcher import analyze_model, WeightWatcher

    # Load your model
    model = keras.models.load_model('my_model.keras')

    # Quick analysis
    results = analyze_model(model, plot=True, savedir='analysis_results')

    # Detailed analysis
    watcher = WeightWatcher(model)
    analysis_df = watcher.analyze(concentration_analysis=True, plot=True)
    summary = watcher.get_summary()
    ```
"""

# Main classes and functions
from .weightwatcher import WeightWatcher
from .analyzer import (
    analyze_model,
    compare_models,
    create_smoothed_model,
    get_critical_layers
)

# Utility functions
from .weights_utils import (
    infer_layer_type,
    get_layer_weights_and_bias,
    get_weight_matrices,
    create_weight_visualization,
    compute_weight_statistics,
    extract_conv_filters
)

# Standalone metric functions
from .metrics import (
    compute_eigenvalues,
    fit_powerlaw,
    calculate_matrix_entropy,
    calculate_spectral_metrics,
    calculate_gini_coefficient,
    calculate_dominance_ratio,
    calculate_participation_ratio,
    calculate_concentration_metrics,
    get_top_eigenvectors,
    find_critical_weights,
    jensen_shannon_distance,
    smooth_matrix
)

# Constants
from .constants import (
    LayerType,
    SmoothingMethod,
    StatusCode,
    MetricNames,
    DEFAULT_SUMMARY_METRICS,
    HIGH_CONCENTRATION_PERCENTILE
)

# Version info
__version__ = "2.0.0"
__author__ = "Enhanced WeightWatcher Team"

# Main exports
__all__ = [
    # Main interface
    'WeightWatcher',
    'analyze_model',
    'compare_models',
    'create_smoothed_model',
    'get_critical_layers',

    # Utility functions
    'infer_layer_type',
    'get_layer_weights_and_bias',
    'get_weight_matrices',
    'create_weight_visualization',
    'compute_weight_statistics',
    'extract_conv_filters',

    # Metric functions
    'compute_eigenvalues',
    'fit_powerlaw',
    'calculate_matrix_entropy',
    'calculate_spectral_metrics',
    'calculate_gini_coefficient',
    'calculate_dominance_ratio',
    'calculate_participation_ratio',
    'calculate_concentration_metrics',
    'get_top_eigenvectors',
    'find_critical_weights',
    'jensen_shannon_distance',
    'smooth_matrix',

    # Constants and enums
    'LayerType',
    'SmoothingMethod',
    'StatusCode',
    'MetricNames',
    'DEFAULT_SUMMARY_METRICS',
    'HIGH_CONCENTRATION_PERCENTILE',

    # Version
    '__version__'
]