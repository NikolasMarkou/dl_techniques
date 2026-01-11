"""
CIFAR-10 Loss Function Comparison: Evaluating Goodhart-Aware Training with Calibration Losses
================================================================================================

This experiment conducts a comprehensive comparison of different loss functions
for image classification on CIFAR-10, with particular emphasis on evaluating
the effectiveness of the GoodhartAwareLoss and calibration-focused losses
(Brier Score and combined approaches) against traditional approaches.

The study addresses a fundamental question in deep learning: how do different
loss formulations affect model robustness, calibration, and generalization?
By comparing standard cross-entropy with more sophisticated loss functions,
this experiment provides insights into the trade-offs between accuracy,
confidence calibration, and resistance to overfitting.

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32x32 RGB images)
- 50,000 training images
- 10,000 test images
- Standard preprocessing with normalization

**Model Architecture**: ResNet-inspired CNN with the following components:
- Initial convolutional layer (32 filters)
- 4 convolutional blocks with residual connections
- Progressive filter scaling: [32, 64, 128, 256]
- Batch normalization and dropout regularization
- Global average pooling
- Dense classification layers with L2 regularization
- Softmax output layer for probability predictions
- Configurable architecture parameters for systematic studies

**Loss Functions Evaluated**:

1. **Standard Cross-Entropy**: The baseline approach for multi-class classification
2. **Label Smoothing**: Cross-entropy with soft targets (alpha=0.1) to reduce overconfidence
3. **Focal Loss**: Addresses class imbalance by down-weighting easy examples (gamma=2.0)
4. **Brier Score Loss**: Direct optimization of calibration via mean squared error
   between predicted probabilities and one-hot labels
5. **Combined Calibration Loss**: Weighted combination of Cross-Entropy and Brier Score
   for joint accuracy and calibration optimization
6. **Goodhart-Aware Loss**: Information-theoretic approach combining:
   - Cross-entropy for task accuracy
   - Entropy regularization to maintain prediction uncertainty
   - Mutual information regularization to compress irrelevant features

Note: SpiegelhalterZLoss is designed for binary classification and is not included
in this multi-class experiment. For multi-class calibration, Brier Score provides
a natural and effective alternative.

Comprehensive Analysis Pipeline
------------------------------

The experiment employs a multi-faceted analysis approach using the ModelAnalyzer:

**Training Analysis**:
- Training and validation curves for all loss functions
- Convergence behavior and stability metrics
- Early stopping based on validation accuracy

**Model Performance Evaluation**:
- Test set accuracy and top-k accuracy
- Loss values and convergence characteristics
- Statistical significance testing across runs

**Calibration Analysis** (via ModelAnalyzer):
- Expected Calibration Error (ECE) with configurable binning
- Brier score for probabilistic prediction quality
- Reliability diagrams and calibration plots
- Confidence histogram analysis

**Weight and Activation Analysis**:
- Layer-wise weight distribution statistics
- Activation pattern analysis across the network
- Information flow characteristics
- Feature representation quality

**Spectral Analysis (WeightWatcher)**:
- Power-law exponent (alpha) for training quality assessment
- Concentration scores for information distribution
- Matrix entropy and stable rank metrics
- Data-free generalization estimates

**Probability Distribution Analysis**:
- Output probability distribution characteristics
- Entropy analysis of predictions
- Uncertainty quantification metrics

**Visual Analysis**:
- Training history comparison plots
- Confusion matrices for each loss function
- Calibration and reliability diagrams
- Weight distribution visualizations
- Comprehensive summary dashboard

Configuration and Customization
-------------------------------

The experiment is highly configurable through the ``ExperimentConfig`` class:

**Architecture Parameters**:
- ``conv_filters``: Filter counts for convolutional layers
- ``dense_units``: Hidden unit counts for dense layers
- ``dropout_rates``: Dropout probabilities per layer
- ``weight_decay``: L2 regularization strength

**Training Parameters**:
- ``epochs``: Number of training epochs
- ``batch_size``: Training batch size
- ``learning_rate``: Adam optimizer learning rate
- ``early_stopping_patience``: Patience for early stopping

**Loss Function Parameters**:
- Easily extensible loss function dictionary
- Configurable hyperparameters for each loss
- Support for custom loss implementations

**Analysis Parameters**:
- ``calibration_bins``: Number of bins for calibration analysis
- Output directory structure and naming
- Visualization and plotting options

Expected Outcomes and Insights
------------------------------

This experiment is designed to reveal:

1. **Accuracy vs. Calibration Trade-offs**: How different loss functions balance
   task performance with prediction reliability

2. **Robustness Characteristics**: Which approaches produce more robust models
   that generalize better to unseen data

3. **Information-Theoretic Benefits**: Whether the Goodhart-Aware Loss's
   information bottleneck principle provides measurable advantages

4. **Calibration Effectiveness**: How direct calibration optimization (Brier Score)
   compares with indirect approaches (label smoothing, entropy regularization)

5. **Training Dynamics**: How different loss formulations affect convergence
   speed, stability, and final performance

Theoretical Foundation
----------------------

This experiment is grounded in several key theoretical frameworks:

**Information Theory**: The Goodhart-Aware Loss leverages information-theoretic
principles to balance task performance with model robustness, drawing from:
- Information Bottleneck Principle (Tishby et al.)
- Entropy regularization for calibration (Pereyra et al.)
- Mutual information constraints for generalization

**Statistical Learning Theory**: The comparison addresses fundamental questions
about the bias-variance trade-off and how different loss formulations affect
generalization bounds.

**Calibration Theory**: The analysis framework evaluates how well predicted
probabilities reflect true confidence, crucial for reliable decision-making
in real-world applications. The Brier Score provides a proper scoring rule
for probabilistic predictions.
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import keras
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable

from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.losses.brier_spiegelhalters_ztest_loss import (
    BrierScoreLoss,
    BrierScoreMetric
)
from dl_techniques.utils.train import TrainingConfig, train_model

from dl_techniques.visualization import (
    VisualizationManager,
    ClassificationResults,
    MultiModelClassification,
    TrainingHistory,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    ROCPRCurves,
    ConvergenceAnalysis,
    OverfittingAnalysis,
    ErrorAnalysisDashboard
)

from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput
)


# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================

@dataclass
class CIFAR10Data:
    """
    Container for CIFAR-10 dataset.

    Attributes:
        x_train: Training images, shape (N, 32, 32, 3), normalized to [0, 1]
        y_train: Training labels, one-hot encoded, shape (N, 10)
        x_test: Test images, shape (M, 32, 32, 3), normalized to [0, 1]
        y_test: Test labels, one-hot encoded, shape (M, 10)
        class_names: List of class names for CIFAR-10
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: List[str] = field(default_factory=lambda: [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ])


def load_and_preprocess_cifar10() -> CIFAR10Data:
    """
    Load and preprocess CIFAR-10 dataset.

    This function loads the raw CIFAR-10 data from Keras datasets, normalizes
    the pixel values to [0, 1] range, and converts labels to one-hot encoding.

    Returns:
        CIFAR10Data object containing preprocessed training and test data.
    """
    logger.info("Loading CIFAR-10 dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten label arrays
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Convert to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    logger.info(
        f"CIFAR-10 loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples"
    )
    logger.info(f"Image shape: {x_train.shape[1:]}, Label shape: {y_train.shape[1:]}")

    return CIFAR10Data(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )


# ==============================================================================
# CUSTOM COMBINED LOSS FOR MULTI-CLASS
# ==============================================================================

@keras.saving.register_keras_serializable()
class CombinedCrossEntropyBrierLoss(keras.losses.Loss):
    """
    Combined loss using Cross-Entropy and Brier Score for multi-class classification.

    This loss function combines standard categorical cross-entropy (for optimizing
    classification accuracy) with the Brier Score (for optimizing calibration).
    The combination provides a balance between discriminative power and
    probabilistic calibration.

    Loss = alpha * CrossEntropy + (1-alpha) * BrierScore

    Args:
        alpha: Weight for the cross-entropy component. The Brier Score component
            has a weight of (1-alpha). Default is 0.5.
        from_logits: Whether the predictions are logits (not passed through
            softmax). Default is False.
        reduction: Type of reduction to apply to the loss.
        name: Optional name for the loss function.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        from_logits: bool = False,
        reduction: str = 'sum_over_batch_size',
        name: str = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the CombinedCrossEntropyBrierLoss.

        Args:
            alpha: Weight for the cross-entropy component (0 to 1).
            from_logits: Whether model outputs raw logits without softmax.
            reduction: Type of reduction to apply to the loss.
            name: Optional name for the loss.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If alpha is not in the range [0, 1].
        """
        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in the range [0, 1], got {alpha}")

        super().__init__(
            reduction=reduction,
            name=name or "combined_ce_brier_loss",
            **kwargs
        )
        self.alpha = alpha
        self.from_logits = from_logits

        # Initialize component losses
        self.ce_loss = keras.losses.CategoricalCrossentropy(
            from_logits=from_logits,
            reduction='none'
        )
        self.brier_loss = BrierScoreLoss(
            from_logits=from_logits,
            reduction='none'
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute the combined calibration loss.

        Args:
            y_true: Ground truth labels (one-hot encoded for multi-class).
            y_pred: Predicted probabilities or logits.

        Returns:
            Combined loss value.
        """
        ce_component = self.ce_loss(y_true, y_pred)
        brier_component = self.brier_loss(y_true, y_pred)

        return self.alpha * ce_component + (1 - self.alpha) * brier_component

    def get_config(self) -> Dict[str, Any]:
        """
        Get loss configuration for serialization.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "from_logits": self.from_logits
        })
        return config


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for the CIFAR-10 loss comparison experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, loss function definitions, and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25, 0.25])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # --- Training Parameters ---
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    monitor_metric: str = 'val_accuracy'

    # --- Loss Functions to Evaluate ---
    loss_functions: Dict[str, Callable] = field(default_factory=lambda: {
        'CrossEntropy': lambda: keras.losses.CategoricalCrossentropy(
            from_logits=False),
        'LabelSmoothing': lambda: keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1, from_logits=False
        ),
        'FocalLoss': lambda: keras.losses.CategoricalFocalCrossentropy(
            gamma=2.0, from_logits=False
        ),
        'BrierScore': lambda: BrierScoreLoss(
            from_logits=False
        ),
        'Combined_CE_Brier_05': lambda: CombinedCrossEntropyBrierLoss(
            alpha=0.5, from_logits=False
        ),
        'Combined_CE_Brier_07': lambda: CombinedCrossEntropyBrierLoss(
            alpha=0.7, from_logits=False
        ),
        'Combined_CE_Brier_03': lambda: CombinedCrossEntropyBrierLoss(
            alpha=0.3, from_logits=False
        ),
        'GAL_0': lambda: GoodhartAwareLoss(
            entropy_weight=0.0, mi_weight=0.01, from_logits=False
        ),
        'GAL_01': lambda: GoodhartAwareLoss(
            entropy_weight=0.1, mi_weight=0.01, from_logits=False
        ),
        'GAL_001': lambda: GoodhartAwareLoss(
            entropy_weight=0.01, mi_weight=0.01, from_logits=False
        ),
        'GAL_0001': lambda: GoodhartAwareLoss(
            entropy_weight=0.001, mi_weight=0.01, from_logits=False
        ),
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_loss_comparison_with_calibration"
    random_seed: int = 42

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        # Enable all main analysis modules
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        analyze_training_dynamics=True,
        analyze_spectral=True,

        # Data sampling configuration
        n_samples=1000,  # Max samples for data-dependent analyses

        # Weight analysis settings
        weight_layer_types=['Dense', 'Conv2D'],
        analyze_biases=False,
        compute_weight_pca=True,

        # Spectral analysis (WeightWatcher) settings
        spectral_min_evals=10,
        spectral_concentration_analysis=True,
        spectral_randomize=False,

        # Calibration settings
        calibration_bins=15,

        # Training dynamics settings
        smooth_training_curves=True,
        smoothing_window=5,

        # Visualization settings
        save_plots=True,
        plot_style='publication',
        save_format='png',
        dpi=300,

        # Performance limits
        max_layers_heatmap=12,
        max_layers_info_flow=8,

        # Verbosity
        verbose=True,
    ))


# ==============================================================================
# MODEL ARCHITECTURE BUILDING UTILITIES
# ==============================================================================

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a residual block with skip connections.

    This function creates a residual block consisting of two convolutional layers
    with batch normalization and ReLU activation, plus a skip connection that
    bypasses the block. If the input and output dimensions don't match, a 1x1
    convolution is used to adjust the skip connection.

    Args:
        inputs: Input tensor to the residual block
        filters: Number of filters in the convolutional layers
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming layers)

    Returns:
        Output tensor after applying the residual block
    """
    # Store the original input for the skip connection
    shortcut = inputs

    # First convolutional layer
    x = keras.layers.Conv2D(
        filters, config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Second convolutional layer
    x = keras.layers.Conv2D(
        filters, config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    # Adjust skip connection if dimensions don't match
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay)
        )(shortcut)

        if config.use_batch_norm:
            shortcut = keras.layers.BatchNormalization()(shortcut)

    # Add skip connection and apply final activation
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    return x


def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a convolutional block with optional residual connections.

    This function creates either a standard convolutional block or a residual
    block based on the configuration. It includes optional max pooling and
    dropout regularization.

    Args:
        inputs: Input tensor to the convolutional block
        filters: Number of filters in the convolutional layers
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming and logic)

    Returns:
        Output tensor after applying the convolutional block
    """
    # Use residual connections for blocks after the first one (if enabled)
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
    else:
        # Standard convolutional block
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{block_index}'
        )(inputs)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

    # Apply max pooling (except for the last convolutional block)
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(config.pool_size)(x)

    # Apply dropout if specified for this layer
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    """
    Build a complete CNN model for CIFAR-10 classification with softmax output.

    This function constructs a ResNet-inspired CNN with configurable architecture
    parameters. The model includes convolutional blocks, global average pooling,
    dense classification layers, and a final softmax layer for probability output.

    Args:
        config: Experiment configuration containing model architecture parameters
        loss_fn: Loss function to use for training
        name: Name prefix for the model and its layers

    Returns:
        Compiled Keras model ready for training with softmax probability outputs
    """
    # Define input layer
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Initial convolutional layer
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Stack of convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

    # Global average pooling to reduce spatial dimensions
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense classification layers
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        # Apply dropout if specified for dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    # Pre-softmax logits layer
    logits = keras.layers.Dense(
        units=config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='logits'
    )(x)

    # Final softmax layer for probability output
    predictions = keras.layers.Activation('softmax', name='predictions')(logits)

    # Create and compile the model
    model = keras.Model(inputs=inputs, outputs=predictions, name=f'{name}_model')

    # Compile with comprehensive metrics including Brier Score
    optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
            BrierScoreMetric(from_logits=False, name='brier_score')
        ]
    )

    return model


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete CIFAR-10 loss function comparison experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing
    2. Model training for each loss function
    3. Comprehensive model analysis using ModelAnalyzer
    4. Visualization generation
    5. Results compilation and reporting

    Args:
        config: Experiment configuration specifying all parameters

    Returns:
        Dictionary containing all experimental results and analysis
    """
    # Set random seed for reproducibility
    keras.utils.set_random_seed(config.random_seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager
    vis_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations"
    )

    # Register visualization templates
    vis_manager.register_template("training_curves", TrainingCurvesVisualization)
    vis_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    vis_manager.register_template("roc_pr_curves", ROCPRCurves)
    vis_manager.register_template("convergence_analysis", ConvergenceAnalysis)
    vis_manager.register_template("overfitting_analysis", OverfittingAnalysis)
    vis_manager.register_template("error_analysis", ErrorAnalysisDashboard)

    # Log experiment start
    logger.info("=" * 80)
    logger.info("CIFAR-10 Loss Comparison Experiment with Calibration Losses")
    logger.info("=" * 80)
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("")

    # ===== DATASET LOADING =====
    logger.info("Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("Dataset loaded successfully")
    logger.info("")

    # ===== MODEL TRAINING PHASE =====
    logger.info("=" * 80)
    logger.info("MODEL TRAINING PHASE")
    logger.info("=" * 80)

    trained_models = {}  # Store trained models: Dict[str, keras.Model]
    training_histories = {}  # Store training histories: Dict[str, Dict[str, List[float]]]
    training_histories_objects = {} # Store standardized TrainingHistory objects

    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"\n--- Training model with {loss_name} loss ---")

        # Build model for this loss function
        model = build_model(config, loss_fn_factory(), loss_name)

        # Log model info
        logger.info(f"Model parameters: {model.count_params():,}")
        logger.info(f"Output layer: {model.output.name}")

        # Configure training parameters
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=loss_name,
            output_dir=experiment_dir / "training_plots" / loss_name
        )

        # Train the model
        history = train_model(
            model,
            cifar10_data.x_train,
            cifar10_data.y_train,
            cifar10_data.x_test,
            cifar10_data.y_test,
            training_config
        )

        # Store results
        # IMPORTANT: Store the .history attribute which is Dict[str, List[float]]
        # This is the correct format for ModelAnalyzer's training_history parameter
        trained_models[loss_name] = model
        training_histories[loss_name] = history.history

        # Standardize history for framework visualizations
        training_histories_objects[loss_name] = TrainingHistory(
            epochs=history.epoch,
            train_loss=history.history['loss'],
            val_loss=history.history['val_loss'] if 'val_loss' in history.history else None,
            train_metrics={k: v for k, v in history.history.items() if 'loss' not in k and 'val_' not in k},
            val_metrics={k: v for k, v in history.history.items() if 'val_' in k and 'loss' not in k}
        )

        logger.info(f"{loss_name} training completed successfully!")

    logger.info("")
    logger.info("=" * 80)
    logger.info("All models trained successfully!")
    logger.info("=" * 80)
    logger.info("")

    # ===== MEMORY MANAGEMENT =====
    logger.info("Triggering garbage collection...")
    gc.collect()
    logger.info("")

    # ===== COMPREHENSIVE MODEL ANALYSIS WITH MODEL ANALYZER =====
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL ANALYSIS (ModelAnalyzer)")
    logger.info("=" * 80)
    logger.info("")

    model_analysis_results = None

    try:
        # Prepare DataInput object for the analyzer
        # According to README: DataInput(x_data=x_test, y_data=y_test)
        test_data_input = DataInput(
            x_data=cifar10_data.x_test,
            y_data=cifar10_data.y_test
        )

        logger.info(f"Test data prepared: {test_data_input.x_data.shape}")
        logger.info(f"Number of models to analyze: {len(trained_models)}")
        logger.info(f"Training histories available for: {list(training_histories.keys())}")
        logger.info("")

        # Initialize the ModelAnalyzer
        # Pass trained_models, training_histories, config, and output_dir
        analyzer = ModelAnalyzer(
            models=trained_models,
            training_history=training_histories,  # Dict[str, Dict[str, List[float]]]
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        logger.info("ModelAnalyzer initialized successfully")
        logger.info("Running comprehensive analysis...")
        logger.info("")

        # Run comprehensive analysis
        # This will generate all plots and metrics
        model_analysis_results = analyzer.analyze(data=test_data_input)

        logger.info("=" * 80)
        logger.info("ModelAnalyzer completed successfully!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Generated analysis outputs:")
        logger.info("  - summary_dashboard.png: High-level overview")
        logger.info("  - spectral_summary.png: Training quality (power-law alpha)")
        logger.info("  - training_dynamics.png: Learning curves and overfitting")
        logger.info("  - weight_learning_journey.png: Weight evolution")
        logger.info("  - confidence_calibration_analysis.png: Calibration metrics")
        logger.info("  - information_flow_analysis.png: Activation patterns")
        logger.info("  - analysis_results.json: Raw metrics")
        logger.info("")

    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)
        logger.warning("Continuing with remaining analyses...")
        logger.info("")

    # ===== ADDITIONAL VISUALIZATION: FRAMEWORK PLOTS =====
    logger.info("=" * 80)
    logger.info("GENERATING FRAMEWORK VISUALIZATIONS")
    logger.info("=" * 80)
    logger.info("")

    try:
        # 1. Training Dynamics (Convergence & Overfitting)
        logger.info("Generating training dynamics visualizations...")
        vis_manager.visualize(
            data=training_histories_objects,
            plugin_name="convergence_analysis",
            show=False
        )
        vis_manager.visualize(
            data=training_histories_objects,
            plugin_name="overfitting_analysis",
            show=False
        )

        # Generate predictions for all models
        raw_predictions = {
            name: model.predict(cifar10_data.x_test, verbose=0)
            for name, model in trained_models.items()
        }
        class_predictions = {
            name: np.argmax(preds, axis=1)
            for name, preds in raw_predictions.items()
        }

        # Convert y_test to class indices for confusion matrix
        y_true_indices = np.argmax(cifar10_data.y_test, axis=1)

        # Create classification results for each model
        all_classification_results = {}
        for model_name, y_pred in class_predictions.items():
            try:
                classification_data = ClassificationResults(
                    y_true=y_true_indices,
                    y_pred=y_pred,
                    y_prob=raw_predictions[model_name],
                    class_names=cifar10_data.class_names,
                    model_name=model_name
                )
                all_classification_results[model_name] = classification_data

                # Individual Error Analysis Dashboard
                # Using unique filenames to prevent overwriting
                vis_manager.visualize(
                    data=classification_data,
                    plugin_name="error_analysis",
                    show=False,
                    filename=f"error_analysis_{model_name}",
                    x_data=cifar10_data.x_test # Pass examples for visualization
                )

            except Exception as e:
                logger.error(f"Failed to prepare classification results for {model_name}: {e}")

        # Create multi-model visualization
        if all_classification_results:
            multi_model_data = MultiModelClassification(
                results=all_classification_results,
                dataset_name="CIFAR-10"
            )

            # Confusion Matrices
            vis_manager.visualize(
                data=multi_model_data,
                plugin_name="confusion_matrix",
                normalize='true',
                show=False
            )

            # ROC/PR Curves
            vis_manager.visualize(
                data=multi_model_data,
                plugin_name="roc_pr_curves",
                plot_type='both',
                show=False
            )

            logger.info("Multi-model visualizations created (Confusion Matrix, ROC/PR)")
            logger.info("")

    except Exception as e:
        logger.error(f"Failed to create framework visualizations: {e}")
        logger.info("")

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("=" * 80)
    logger.info("FINAL PERFORMANCE EVALUATION")
    logger.info("=" * 80)
    logger.info("")

    performance_results = {}

    for name, model in trained_models.items():
        logger.info(f"Evaluating model: {name}")

        # Get model evaluation metrics
        eval_results = model.evaluate(
            cifar10_data.x_test,
            cifar10_data.y_test,
            verbose=0
        )
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        # Calculate manual accuracy verification
        predictions = model.predict(cifar10_data.x_test, verbose=0)
        y_true_indices = np.argmax(cifar10_data.y_test, axis=1)

        # Manual top-1 accuracy
        manual_top1_acc = np.mean(np.argmax(predictions, axis=1) == y_true_indices)

        # Manual top-5 accuracy
        top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]
        manual_top5_acc = np.mean([
            y_true in top5_pred
            for y_true, top5_pred in zip(y_true_indices, top_5_predictions)
        ])

        # Store standardized metrics
        performance_results[name] = {
            'accuracy': metrics_dict.get('accuracy', manual_top1_acc),
            'top_5_accuracy': metrics_dict.get('top_5_accuracy', manual_top5_acc),
            'loss': metrics_dict.get('loss', 0.0),
            'brier_score': metrics_dict.get('brier_score', 0.0)
        }

        # Warn about low accuracy
        final_accuracy = performance_results[name]['accuracy']
        if final_accuracy < 0.2:
            logger.warning(f"Low accuracy detected for {name}: {final_accuracy:.4f}")
            logger.warning("This may indicate training issues")

        logger.info(f"  Accuracy: {performance_results[name]['accuracy']:.4f}")
        logger.info(f"  Top-5 Accuracy: {performance_results[name]['top_5_accuracy']:.4f}")
        logger.info(f"  Loss: {performance_results[name]['loss']:.4f}")
        logger.info(f"  Brier Score: {performance_results[name]['brier_score']:.4f}")
        logger.info("")

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': training_histories,
        'trained_models': trained_models
    }

    # Print comprehensive summary
    print_experiment_summary(results_payload, config.analyzer_config)

    return results_payload


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(
    results: Dict[str, Any],
    analyzer_config: AnalysisConfig
) -> None:
    """
    Print a comprehensive summary of experimental results.

    This function generates a detailed report of all experimental outcomes,
    including performance metrics, calibration analysis, spectral analysis,
    and training progress. The summary is formatted for clear readability
    and easy interpretation.

    Args:
        results: Dictionary containing all experimental results and analysis
        analyzer_config: The analyzer configuration used
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    # ===== PERFORMANCE METRICS SECTION =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("PERFORMANCE METRICS (Test Set)")
        logger.info("-" * 80)
        logger.info(f"{'Model':<30} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12} {'Brier':<12}")
        logger.info("-" * 80)

        for model_name, metrics in results['performance_analysis'].items():
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            brier = metrics.get('brier_score', 0.0)
            logger.info(
                f"{model_name:<30} {accuracy:<12.4f} {top5_acc:<12.4f} "
                f"{loss:<12.4f} {brier:<12.4f}"
            )
        logger.info("")

    # ===== MODEL ANALYZER RESULTS =====
    model_analysis = results.get('model_analysis')
    if model_analysis:

        # --- Calibration Metrics ---
        if analyzer_config.analyze_calibration and model_analysis.calibration_metrics:
            logger.info("CALIBRATION METRICS (ModelAnalyzer)")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'ECE':<12} {'Brier Score':<15} {'Mean Conf':<12} {'Mean Entropy':<12}")
            logger.info("-" * 80)

            for model_name, cal_metrics in model_analysis.calibration_metrics.items():
                conf_metrics = model_analysis.confidence_metrics.get(model_name, {})
                logger.info(
                    f"{model_name:<30} "
                    f"{cal_metrics.get('ece', 0.0):<12.4f} "
                    f"{cal_metrics.get('brier_score', 0.0):<15.4f} "
                    f"{conf_metrics.get('mean_confidence', 0.0):<12.4f} "
                    f"{conf_metrics.get('mean_entropy', 0.0):<12.4f}"
                )
            logger.info("")

        # --- Spectral Analysis Results ---
        if analyzer_config.analyze_spectral and model_analysis.spectral_metrics:
            logger.info("SPECTRAL ANALYSIS (WeightWatcher)")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'Mean alpha':<12} {'Concentration':<15} {'Matrix Entropy':<15}")
            logger.info("-" * 80)

            for model_name, spectral in model_analysis.spectral_metrics.items():
                mean_alpha = spectral.get('mean_alpha', 0.0)
                mean_concentration = spectral.get('mean_concentration_score', 0.0)
                mean_entropy = spectral.get('mean_matrix_entropy', 0.0)

                logger.info(
                    f"{model_name:<30} "
                    f"{mean_alpha:<12.4f} "
                    f"{mean_concentration:<15.4f} "
                    f"{mean_entropy:<15.4f}"
                )
            logger.info("")
            logger.info("Note: Ideal Mean alpha is typically in range [2.0, 6.0]")
            logger.info("      alpha < 2.0 may indicate over-training")
            logger.info("      alpha > 6.0 may indicate under-training")
            logger.info("")

        # --- Training Dynamics ---
        if analyzer_config.analyze_training_dynamics and model_analysis.training_metrics:
            logger.info("TRAINING DYNAMICS (ModelAnalyzer)")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'Epochs to Conv':<15} {'Overfitting':<15} {'Stability':<12}")
            logger.info("-" * 80)

            for model_name, train_metrics in model_analysis.training_metrics.items():
                epochs_conv = train_metrics.get('epochs_to_convergence', 0)
                overfit = train_metrics.get('overfitting_index', 0.0)
                stability = train_metrics.get('training_stability', 0.0)

                logger.info(
                    f"{model_name:<30} "
                    f"{epochs_conv:<15} "
                    f"{overfit:<15.4f} "
                    f"{stability:<12.4f}"
                )
            logger.info("")

    # ===== VALIDATION METRICS FROM TRAINING =====
    if 'histories' in results and results['histories']:
        has_training_data = any(
            history_dict.get('val_accuracy') and len(history_dict['val_accuracy']) > 0
            for history_dict in results['histories'].values()
        )

        if has_training_data:
            logger.info("FINAL VALIDATION METRICS (Last Epoch)")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'Val Accuracy':<15} {'Val Loss':<12} {'Val Brier':<12}")
            logger.info("-" * 80)

            for model_name, history_dict in results['histories'].items():
                if history_dict.get('val_accuracy') and len(history_dict['val_accuracy']) > 0:
                    final_val_acc = history_dict['val_accuracy'][-1]
                    final_val_loss = history_dict['val_loss'][-1]
                    final_val_brier = history_dict.get('val_brier_score', [0.0])[-1]
                    logger.info(
                        f"{model_name:<30} "
                        f"{final_val_acc:<15.4f} "
                        f"{final_val_loss:<12.4f} "
                        f"{final_val_brier:<12.4f}"
                    )
            logger.info("")

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Key Outputs:")
    logger.info("  1. summary_dashboard.png - Start here for high-level comparison")
    logger.info("  2. spectral_summary.png - Training quality assessment (alpha values)")
    logger.info("  3. training_dynamics.png - Learning curves and overfitting")
    logger.info("  4. confidence_calibration_analysis.png - Reliability diagrams")
    logger.info("  5. analysis_results.json - Raw metrics for further analysis")
    logger.info("")
    logger.info("Interpretation Tips:")
    logger.info("  - Lower ECE & Brier Score = Better calibration")
    logger.info("  - Power-law alpha in [2.0, 6.0] = Well-trained model")
    logger.info("  - Lower Overfitting Index = Better generalization")
    logger.info("  - Check Pareto plots for optimal hyperparameter trade-offs")
    logger.info("")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the CIFAR-10 loss comparison experiment.

    This function serves as the entry point for the experiment, handling
    configuration setup, experiment execution, and error handling.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("CIFAR-10 Loss Function Comparison with Calibration Losses")
    logger.info("Enhanced with Comprehensive ModelAnalyzer Integration")
    logger.info("=" * 80)
    logger.info("")

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  Loss Functions: {list(config.loss_functions.keys())}")
    logger.info(f"  Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"  Model Architecture: {len(config.conv_filters)} conv blocks, "
                f"{len(config.dense_units)} dense layers")
    logger.info(f"  Output: Softmax probabilities (from_logits=False)")
    logger.info("")
    logger.info("ANALYSIS MODULES ENABLED:")
    logger.info(f"  - Weight Analysis: {config.analyzer_config.analyze_weights}")
    logger.info(f"  - Calibration Analysis: {config.analyzer_config.analyze_calibration}")
    logger.info(f"  - Information Flow: {config.analyzer_config.analyze_information_flow}")
    logger.info(f"  - Training Dynamics: {config.analyzer_config.analyze_training_dynamics}")
    logger.info(f"  - Spectral Analysis: {config.analyzer_config.analyze_spectral}")
    logger.info("")

    try:
        # Run the complete experiment
        _ = run_experiment(config)

        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("")

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"EXPERIMENT FAILED: {e}")
        logger.error("=" * 80)
        logger.error("", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()