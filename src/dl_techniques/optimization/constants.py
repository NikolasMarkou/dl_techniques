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