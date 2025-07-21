"""
Constants for Model Analyzer
============================================================================

Central location for all constants used throughout the analyzer module.
"""

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