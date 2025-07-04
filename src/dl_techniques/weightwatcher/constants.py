"""
Constants module for TensorFlow WeightWatcher.

This module contains all constant values used throughout the TensorFlow WeightWatcher
package, ensuring consistency and making the code more maintainable.
"""

from enum import Enum

# Numeric constants for numerical stability
EPSILON = 1e-10  # Small value to avoid division by zero
EVALS_THRESH = 1e-5  # Threshold for considering eigenvalues significant

# Training quality thresholds
OVER_TRAINED_THRESH = 2.0  # Alpha below this value may indicate over-training
UNDER_TRAINED_THRESH = 6.0  # Alpha above this value may indicate under-training

# Default values for analysis
DEFAULT_MIN_EVALS = 10  # Minimum number of eigenvalues required for analysis
DEFAULT_MAX_EVALS = 15000  # Maximum number of eigenvalues to analyze
DEFAULT_MAX_N = 50000  # Maximum matrix dimension to analyze
DEFAULT_SAVEDIR = 'ww-img'  # Default directory for saving figures
WEAK_RANK_LOSS_TOLERANCE = 0.000001  # Tolerance for rank loss calculation

# Histogram and plotting parameters
DEFAULT_BINS = 100  # Default number of bins for histograms
DEFAULT_FIG_SIZE = (10, 6)  # Default figure size (width, height) in inches
DEFAULT_DPI = 300  # Default DPI for saved figures

# SVD parameters
DEFAULT_SVD_SMOOTH_PERCENT = 0.8  # Default percentage of eigenvalues to keep in SVD smoothing


class LayerType(str, Enum):
    """Enum for supported layer types"""
    UNKNOWN = 'unknown'
    DENSE = 'dense'
    CONV1D = 'conv1d'
    CONV2D = 'conv2d'
    CONV3D = 'conv3d'
    EMBEDDING = 'embedding'
    LSTM = 'lstm'
    GRU = 'gru'
    NORM = 'norm'


class SmoothingMethod(str, Enum):
    """Enum for SVD smoothing methods"""
    SVD = 'svd'  # Standard SVD truncation based on percentage
    DETX = 'detX'  # Truncation based on Det(X) = 1 constraint
    LAMBDA_MIN = 'lambda_min'  # Truncation based on power-law xmin value


class StatusCode(str, Enum):
    """Enum for analysis status codes"""
    SUCCESS = 'success'
    FAILED = 'failed'
    WARN_OVER_TRAINED = 'over-trained'
    WARN_UNDER_TRAINED = 'under-trained'


class MetricNames:
    """Class holding the standard names of metrics used in analysis"""
    # Core spectral metrics
    ALPHA = 'alpha'
    STABLE_RANK = 'stable_rank'
    ENTROPY = 'entropy'
    LOG_NORM = 'log_norm'
    LOG_SPECTRAL_NORM = 'log_spectral_norm'
    ALPHA_WEIGHTED = 'alpha_weighted'
    LOG_ALPHA_NORM = 'log_alpha_norm'

    # Eigenvalue metrics
    NUM_EVALS = 'num_evals'
    LAMBDA_MAX = 'lambda_max'

    # Singular value metrics
    SV_MAX = 'sv_max'
    SV_MIN = 'sv_min'

    # Rank metrics
    RANK_LOSS = 'rank_loss'
    WEAK_RANK_LOSS = 'weak_rank_loss'

    # Power-law fitting metrics
    XMIN = 'xmin'
    D = 'D'
    SIGMA = 'sigma'
    NUM_PL_SPIKES = 'num_pl_spikes'

    # Status and metadata
    STATUS = 'status'
    WARNING = 'warning'
    HAS_ESD = 'has_esd'

    # Concentration metrics (previously "black hole" metrics)
    GINI_COEFFICIENT = 'gini_coefficient'
    DOMINANCE_RATIO = 'dominance_ratio'
    PARTICIPATION_RATIO = 'participation_ratio'
    CONCENTRATION_SCORE = 'concentration_score'
    CRITICAL_WEIGHT_COUNT = 'critical_weight_count'


# Dictionary of default metrics to include in summary calculations
DEFAULT_SUMMARY_METRICS = [
    MetricNames.ALPHA,
    MetricNames.STABLE_RANK,
    MetricNames.ENTROPY,
    MetricNames.LOG_SPECTRAL_NORM,
    MetricNames.LOG_NORM,
    MetricNames.GINI_COEFFICIENT,
    MetricNames.DOMINANCE_RATIO,
    MetricNames.PARTICIPATION_RATIO,
    MetricNames.CONCENTRATION_SCORE
]

# Concentration analysis thresholds
HIGH_CONCENTRATION_PERCENTILE = 0.8  # Percentile threshold for high concentration layers
CRITICAL_WEIGHT_THRESHOLD = 0.1  # Threshold for identifying critical weights
MAX_CRITICAL_WEIGHTS_REPORTED = 10  # Maximum number of critical weights to report per layer