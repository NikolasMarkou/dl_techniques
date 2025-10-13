"""
Constants for Model Analyzer

Central location for all constants used throughout the analyzer module.
"""
from enum import Enum

# Health score thresholds and weights
WEIGHT_HEALTH_L2_NORMALIZER = 10.0  # L2 norm normalization factor for weight health calculation
WEIGHT_HEALTH_SPARSITY_THRESHOLD = 0.8  # Maximum acceptable sparsity before considering weights unhealthy
LAYER_SPECIALIZATION_MAX_RANK = 10.0  # Maximum effective rank for normalization in specialization analysis
ACTIVATION_MAGNITUDE_NORMALIZER = 5.0  # Activation magnitude normalization factor for health scoring

# Training analysis constants
CONVERGENCE_THRESHOLD = 0.95  # Fraction of peak performance to consider model "converged"
TRAINING_STABILITY_WINDOW = 10  # Number of recent epochs to analyze for stability (higher = smoother estimate)
OVERFITTING_ANALYSIS_FRACTION = 0.33  # Final fraction of training to analyze for overfitting metrics

# Metric name patterns for flexible history parsing
LOSS_PATTERNS = ['loss', 'total_loss', 'train_loss']
VAL_LOSS_PATTERNS = ['val_loss', 'validation_loss', 'valid_loss']
ACC_PATTERNS = ['accuracy', 'acc', 'categorical_accuracy', 'sparse_categorical_accuracy',
                'binary_accuracy', 'top_k_categorical_accuracy']
VAL_ACC_PATTERNS = ['val_accuracy', 'val_acc', 'validation_accuracy', 'val_categorical_accuracy',
                    'val_sparse_categorical_accuracy', 'val_binary_accuracy']

# ==============================================================================
# Spectral Analysis (WeightWatcher) Constants
# ==============================================================================

SPECTRAL_EPSILON = 1e-10
SPECTRAL_EVALS_THRESH = 1e-5
SPECTRAL_OVER_TRAINED_THRESH = 2.0
SPECTRAL_UNDER_TRAINED_THRESH = 6.0
SPECTRAL_DEFAULT_MIN_EVALS = 10
SPECTRAL_DEFAULT_MAX_EVALS = 15000
SPECTRAL_WEAK_RANK_LOSS_TOLERANCE = 1e-6
SPECTRAL_DEFAULT_BINS = 100
SPECTRAL_DEFAULT_FIG_SIZE = (10, 6)
SPECTRAL_DEFAULT_DPI = 300
SPECTRAL_HIGH_CONCENTRATION_PERCENTILE = 0.8
SPECTRAL_CRITICAL_WEIGHT_THRESHOLD = 0.1
SPECTRAL_MAX_CRITICAL_WEIGHTS_REPORTED = 10

class LayerType(str, Enum):
    """Enum for supported layer types for spectral analysis"""
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
    SVD = 'svd'
    DETX = 'detX'
    LAMBDA_MIN = 'lambda_min'

class StatusCode(str, Enum):
    """Enum for spectral analysis status codes"""
    SUCCESS = 'success'
    FAILED = 'failed'
    WARN_OVER_TRAINED = 'over-trained'
    WARN_UNDER_TRAINED = 'under-trained'

class MetricNames:
    """Class holding the standard names of metrics used in spectral analysis"""
    ALPHA = 'alpha'
    STABLE_RANK = 'stable_rank'
    ENTROPY = 'entropy'
    LOG_NORM = 'log_norm'
    LOG_SPECTRAL_NORM = 'log_spectral_norm'
    ALPHA_WEIGHTED = 'alpha_weighted'
    LOG_ALPHA_NORM = 'log_alpha_norm'
    NUM_EVALS = 'num_evals'
    LAMBDA_MAX = 'lambda_max'
    SV_MAX = 'sv_max'
    SV_MIN = 'sv_min'
    RANK_LOSS = 'rank_loss'
    WEAK_RANK_LOSS = 'weak_rank_loss'
    XMIN = 'xmin'
    D = 'D'
    SIGMA = 'sigma'
    NUM_PL_SPIKES = 'num_pl_spikes'
    STATUS = 'status'
    WARNING = 'warning'
    HAS_ESD = 'has_esd'
    GINI_COEFFICIENT = 'gini_coefficient'
    DOMINANCE_RATIO = 'dominance_ratio'
    PARTICIPATION_RATIO = 'participation_ratio'
    CONCENTRATION_SCORE = 'concentration_score'
    CRITICAL_WEIGHT_COUNT = 'critical_weight_count'

SPECTRAL_DEFAULT_SUMMARY_METRICS = [
    MetricNames.ALPHA, MetricNames.STABLE_RANK, MetricNames.ENTROPY,
    MetricNames.LOG_SPECTRAL_NORM, MetricNames.LOG_NORM,
    MetricNames.GINI_COEFFICIENT, MetricNames.DOMINANCE_RATIO,
    MetricNames.PARTICIPATION_RATIO, MetricNames.CONCENTRATION_SCORE
]