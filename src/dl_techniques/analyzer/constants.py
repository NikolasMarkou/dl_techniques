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
# Sentinel for `pl_pvalue` meaning "the goodness-of-fit test did not run", as
# distinct from a computed p-value of 0.0 ("certainly not a power law").
# Documented in README.md's spectral column table.
SPECTRAL_PVALUE_NOT_COMPUTED = -1.0
SPECTRAL_DEFAULT_MAX_EVALS = 15000
# Sanity bound on a fitted power-law exponent. WeightWatcher treats alpha > 8 as an
# unreliable fit; `fit_powerlaw`'s `1 + n_tail/denominator` is guarded only by
# `denominator > 1e-10`, so a near-degenerate tail can return alpha ~ 3.6e7 at
# status "success". Metrics derived from such an alpha are FLAGGED, never clamped.
SPECTRAL_ALPHA_SANITY_MAX = 8.0
SPECTRAL_WEAK_RANK_LOSS_TOLERANCE = 1e-6
SPECTRAL_DEFAULT_BINS = 100
SPECTRAL_DEFAULT_FIG_SIZE = (10, 6)
SPECTRAL_DEFAULT_DPI = 300
SPECTRAL_HIGH_CONCENTRATION_PERCENTILE = 0.8
SPECTRAL_CRITICAL_WEIGHT_THRESHOLD = 0.1
SPECTRAL_MAX_CRITICAL_WEIGHTS_REPORTED = 10
SPECTRAL_SMALL_N_CUTOFF = 20             # WeightWatcher SMALL_N_CUTOFF: tails with N < 20 use bias-corrected alpha
SPECTRAL_SMALL_N_KMIN = 8                # WeightWatcher k_min: minimum tail size considered in small-N xmin search

# Correlation Trap Detection (Marchenko-Pastur + Tracy-Widom)
# DECISION plan-2026-09-02T041737-e85f2027/D-006
# How many Tracy-Widom units of headroom the trap threshold allows above the MP
# edge. Do NOT put this back to 1.0: at one TW unit the margin is the width of
# the fluctuation itself, and MEASURED on 300 clean Gaussian Wisharts per shape
# the false-positive rate was 0.0900 (200x50), 0.1300 (100x100) and 0.0967
# (500x100) - which tracks the Tracy-Widom law's own P(W1 > 1) of about 8%, so
# the constant was right for what it multiplies and simply too small. At 3.0 the
# same draws give 0.0067 / 0.0100 / 0.0133 while detection power against the
# SETOL 7.1 element-trap geometry stays at 1.000 for amplitude >= 20. The value
# was NOT inherited: under the pre-fix square-root offset this knob was inert,
# so its old setting carried no information. See decisions.md D-006.
SPECTRAL_TW_SAFETY_FACTOR = 3.0          # Tracy-Widom units of headroom above lambda_plus
SPECTRAL_TRAP_SEVERITY_MILD = 0.1        # severity < 0.1 = no trap
SPECTRAL_TRAP_SEVERITY_MODERATE = 0.3    # 0.1-0.3 = mild, 0.3-0.5 = moderate
SPECTRAL_TRAP_SEVERITY_SEVERE = 0.5      # 0.5-1.0 = severe
SPECTRAL_TRAP_SEVERITY_CRITICAL = 1.0    # > 1.0 = critical

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
    """Per-metric status codes shared by the spectral and weight analyzers.

    The two ``WEIGHT_*`` members classify a weight tensor whose statistics could
    not all be computed as finite numbers. They are deliberately DISTINCT: a
    zeros-initialised table is legal input whose standardized moments are merely
    undefined, whereas a tensor that already contains ``NaN``/``inf`` is a
    corrupt model. Collapsing them would launder a corruption into a plausible
    number.
    """
    SUCCESS = 'success'
    FAILED = 'failed'
    WARN_OVER_TRAINED = 'over-trained'
    WARN_UNDER_TRAINED = 'under-trained'
    #: The tensor is finite and legal, but at least one statistic was undefined
    #: or unrepresentable and was recomputed at higher precision or substituted.
    #: ``degenerate_fields`` names every affected leaf.
    WEIGHT_DEGENERATE = 'degenerate'
    #: The tensor ITSELF contains ``NaN``/``inf``. Statistics are published as
    #: computed (non-finite), never repaired -- see D-001.
    WEIGHT_NON_FINITE = 'non-finite'

class MetricNames:
    """Class holding the standard names of metrics used in spectral analysis"""
    ALPHA = 'alpha'
    NORM = 'norm'
    SPECTRAL_NORM = 'spectral_norm'
    MATRIX_RANK = 'matrix_rank'
    STABLE_RANK = 'stable_rank'
    ENTROPY = 'entropy'
    LOG_NORM = 'log_norm'
    LOG_SPECTRAL_NORM = 'log_spectral_norm'
    ALPHA_WEIGHTED = 'alpha_weighted'
    ALPHA_HAT = 'alpha_hat'
    ALPHA_HAT_NORMALIZED = 'alpha_hat_normalized'
    LOG_ALPHA_NORM = 'log_alpha_norm'
    NUM_EVALS = 'num_evals'
    LAMBDA_MAX = 'lambda_max'
    SV_MAX = 'sv_max'
    SV_MIN = 'sv_min'
    RANK_LOSS = 'rank_loss'
    WEAK_RANK_LOSS = 'weak_rank_loss'
    # True when `compute_eigenvalues` returned only the `n_comp` largest singular
    # values. Every spectrum-wide column on that row is computed over a PARTIAL
    # spectrum; `sv_min`, `rank_loss`, `weak_rank_loss`, `matrix_rank` and
    # `entropy` are NaN there rather than plausible-looking wrong numbers.
    SPECTRUM_TRUNCATED = 'spectrum_truncated'
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
    LEARNING_PHASE = 'learning_phase'
    PL_PVALUE = 'pl_pvalue'
    ERG_LOG_DET = 'erg_log_det'
    ERG_DELTA_LAMBDA_MIN = 'erg_delta_lambda_min'
    ERG_SATISFIED = 'erg_satisfied'
    # Correlation Trap Detection
    HAS_TRAP = 'has_trap'
    NUM_RAND_SPIKES = 'num_rand_spikes'
    TRAP_SEVERITY = 'trap_severity'
    TRAP_SEVERITY_LABEL = 'trap_severity_label'
    MP_LAMBDA_PLUS = 'mp_lambda_plus'
    MP_LAMBDA_MINUS = 'mp_lambda_minus'
    TRAP_THRESHOLD = 'trap_threshold'
    RAND_DISTANCE = 'rand_distance'
    # WW's softrank metric: λ_plus / λ_max (spikes removed first). Reserved for the
    # real WW mp_softrank implementation (Step 4). Replaces the former mis-named
    # softrank constant, which was bound to a non-WW randomization ratio (now
    # RAND_SV_RATIO).
    MP_SOFTRANK = 'mp_softrank'
    # Randomization singular-value ratio: max(rand_evals) / max(evals). NOT WW's
    # mp_softrank — a distinct randomization diagnostic.
    RAND_SV_RATIO = 'rand_sv_ratio'
    RAND_SV_MAX = 'rand_sv_max'

SPECTRAL_DEFAULT_SUMMARY_METRICS = [
    MetricNames.ALPHA, MetricNames.ALPHA_WEIGHTED, MetricNames.ALPHA_HAT,
    MetricNames.STABLE_RANK,
    MetricNames.ENTROPY, MetricNames.LOG_SPECTRAL_NORM, MetricNames.LOG_NORM,
    MetricNames.GINI_COEFFICIENT, MetricNames.DOMINANCE_RATIO,
    MetricNames.PARTICIPATION_RATIO, MetricNames.CONCENTRATION_SCORE
]