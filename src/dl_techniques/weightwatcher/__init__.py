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