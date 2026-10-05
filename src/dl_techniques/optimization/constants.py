"""
Default Configuration Constants for Optimization Module.

This module contains default parameter values for optimizers, learning rate schedules,
and warmup configurations used throughout the dl_techniques optimization system.

The constants are organized by optimizer type and feature:
- General optimization defaults (warmup, optimizer selection)
- Optimizer-specific hyperparameters (Adam, AdamW, RMSprop, Adadelta)
- Learning rate schedule parameters (cosine decay, exponential decay)
"""

# ---------------------------------------------------------------------
# General Optimization Defaults
# ---------------------------------------------------------------------

# Warmup configuration - used to stabilize training in early epochs
DEFAULT_WARMUP_STEPS = 0  # Number of warmup steps (0 = no warmup)
DEFAULT_WARMUP_START_LR = 1e-8  # Starting learning rate during warmup phase
DEFAULT_OPTIMIZER_TYPE = "RMSprop"  # Default optimizer when type not specified

# ---------------------------------------------------------------------
# RMSprop Optimizer Defaults
# ---------------------------------------------------------------------
# RMSprop is effective for RNNs and non-stationary objectives

DEFAULT_RMSPROP_RHO = 0.9  # Decay factor for moving average of squared gradients
DEFAULT_RMSPROP_MOMENTUM = 0.0  # Momentum factor (0.0 = no momentum)
DEFAULT_RMSPROP_EPSILON = 1e-07  # Small constant to prevent division by zero
DEFAULT_RMSPROP_CENTERED = False  # Whether to center the moving averages

# ---------------------------------------------------------------------
# Adam Optimizer Defaults
# ---------------------------------------------------------------------
# Adam combines momentum and adaptive learning rates, good general-purpose optimizer

DEFAULT_ADAM_BETA_1 = 0.9  # Exponential decay rate for first moment estimates (momentum)
DEFAULT_ADAM_BETA_2 = 0.999  # Exponential decay rate for second moment estimates (variance)
DEFAULT_ADAM_EPSILON = 1e-07  # Small constant for numerical stability
DEFAULT_ADAM_AMSGRAD = False  # Whether to use AMSGrad variant (maintains max of past squared gradients)

# ---------------------------------------------------------------------
# AdamW Optimizer Defaults
# ---------------------------------------------------------------------
# AdamW decouples weight decay from gradient-based update, often better for transformers

DEFAULT_ADAMW_BETA_1 = 0.9  # Exponential decay rate for first moment estimates
DEFAULT_ADAMW_BETA_2 = 0.999  # Exponential decay rate for second moment estimates
DEFAULT_ADAMW_EPSILON = 1e-07  # Small constant for numerical stability
DEFAULT_ADAMW_AMSGRAD = False  # Whether to use AMSGrad variant

# ---------------------------------------------------------------------
# Adadelta Optimizer Defaults
# ---------------------------------------------------------------------
# Adadelta adapts learning rates based on window of gradient updates

DEFAULT_ADADELTA_RHO = 0.9  # Decay constant for accumulating squared gradients
DEFAULT_ADADELTA_EPSILON = 1e-07  # Small constant added for numerical stability

# ---------------------------------------------------------------------
# SGD Optimizer Defaults
# ---------------------------------------------------------------------
# Plain (optionally Nesterov-momentum) stochastic gradient descent. Defaults
# mirror keras.optimizers.SGD exactly so the factory cannot silently diverge
# from the Keras class defaults.

DEFAULT_SGD_MOMENTUM = 0.0  # Momentum factor (0.0 = vanilla SGD)
DEFAULT_SGD_NESTEROV = False  # Whether to apply Nesterov momentum

# ---------------------------------------------------------------------
# SGLD Optimizer Defaults
# ---------------------------------------------------------------------
# SGLD (Stochastic Gradient Langevin Dynamics) augments SGD with Gaussian
# noise to enable Bayesian posterior sampling and escape shallow minima.

DEFAULT_SGLD_NOISE_SCALE = 1.0  # Multiplier on canonical Langevin noise (1.0 = temperature 1)
DEFAULT_SGLD_SEED = None  # Optional integer seed for reproducible noise

# ---------------------------------------------------------------------
# Cosine Decay Learning Rate Schedule Defaults
# ---------------------------------------------------------------------
# Cosine decay provides smooth learning rate reduction following cosine curve

DEFAULT_COSINE_ALPHA = 0.0001  # Minimum learning rate as fraction of initial rate

# ---------------------------------------------------------------------
# Cosine Decay with Restarts Schedule Defaults
# ---------------------------------------------------------------------
# Cosine decay with periodic restarts can help escape local minima

DEFAULT_COSINE_RESTARTS_T_MUL = 2.0  # Factor to multiply period length after each restart
DEFAULT_COSINE_RESTARTS_M_MUL = 0.9  # Factor to multiply initial learning rate after each restart
DEFAULT_COSINE_RESTARTS_ALPHA = 0.001  # Minimum learning rate as fraction of initial rate

# ==============================================================================
# VSGD Optimizer Defaults
# ==============================================================================
DEFAULT_VSGD_GHATTG        = 30.0
DEFAULT_VSGD_PS            = 1e-8
DEFAULT_VSGD_TAU1          = 0.81
DEFAULT_VSGD_TAU2          = 0.90
DEFAULT_VSGD_LEARNING_RATE = 0.1
DEFAULT_VSGD_WEIGHT_DECAY  = 0.0
DEFAULT_VSGD_EPS           = 1e-8

# ==============================================================================
# GEFEN Optimizer Defaults
# ==============================================================================
# Gefen-lite (shared-v): AdamW-style update with a block-shared second moment.
# Defaults mirror Gefen.__init__ (gefen_optimizer.py) exactly so the factory
# builder cannot silently diverge from the class defaults.
DEFAULT_GEFEN_LEARNING_RATE = 1e-3
DEFAULT_GEFEN_BETA_1        = 0.9
DEFAULT_GEFEN_BETA_2        = 0.999
DEFAULT_GEFEN_EPSILON       = 1e-8
DEFAULT_GEFEN_WEIGHT_DECAY  = 0.0
DEFAULT_GEFEN_MAX_BLOCK_SIZE = 1024
DEFAULT_GEFEN_MIN_BLOCK_SIZE = 8

# ==============================================================================
# SSP (Spectrum-to-Signal Principle) Defaults
# ==============================================================================
# Defaults mirror ssp/signal.py, ssp/fusion.py and ssp/config.py exactly so the
# factory builder cannot silently diverge from the function and class defaults.
# `schedule.py` does `from .constants import *`, so these must stay module-level.

# -- Spectrum phase (diversity, Pass@K) -------------------------------------
DEFAULT_SSP_PASS_AT_K = 8                    # k for the coverage score
DEFAULT_SSP_ESTIMATOR = "unbiased"           # metrics.pass_at_k estimator
DEFAULT_SSP_FUSION_MODE = "linear"           # 'linear' | 'task_arithmetic'
DEFAULT_SSP_FUSION_SCHEME = "uniform"        # 'uniform' | 'score_softmax'
DEFAULT_SSP_FUSION_TEMPERATURE = 1.0         # score-softmax sharpness
DEFAULT_SSP_FUSION_COEFFICIENT = 1.0         # task-arithmetic scaling
DEFAULT_SSP_SAMPLING_MODE = "max_entropy"    # how Pass@K becomes a sampler

# -- Signal phase (MGPO max-entropy weighting) -------------------------------
DEFAULT_MGPO_LAMBDA = 1.0                    # sharpening; 0.0 disables weighting
DEFAULT_MGPO_P0 = 0.5                        # target success probability
DEFAULT_MGPO_EPS = 1e-6                      # advantage denominator stabilizer
DEFAULT_MGPO_DDOF = 0                        # population std; defined for G == 1
DEFAULT_MGPO_NORMALIZE_WEIGHTS = False       # rescale weights to mean 1
DEFAULT_MGPO_CLIP_EPS = 0.2                  # PPO clip range half-width

# -- SSP shared -------------------------------------------------------------
DEFAULT_SSP_TOKEN_REDUCTION = "per_sequence_mean"   # the paper's two-stage mean
DEFAULT_SSP_ZERO_VARIANCE = "zero"                  # 'zero' | 'raise'
DEFAULT_SSP_ENABLED = False                         # master switch (opt-in)
