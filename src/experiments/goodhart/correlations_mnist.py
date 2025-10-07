"""
Colored MNIST Spurious Correlation Experiment: GoodhartAwareLoss vs. Baselines
==============================================================================

This experiment evaluates the robustness of different loss functions against
spurious correlations using a synthetic Colored MNIST dataset. It tests whether
models can avoid "gaming the metric" by exploiting dataset artifacts, a key
manifestation of Goodhart's Law in machine learning.

The study systematically compares standard cross-entropy, label smoothing,
focal loss, and the information-theoretic GoodhartAwareLoss to determine which
formulations produce models that generalize based on true features (digit shape)
rather than spurious shortcuts (color).

Experimental Design
-------------------

**Dataset**: Colored MNIST (10 classes, 28×28 RGB images)
- A synthetic dataset with a controlled distribution shift
- **Training Set**: High (95%) spurious correlation between digit class and color
- **Test Set**: Zero correlation; color is random and uninformative
- **Validation Set**: Intermediate (50%) correlation for monitoring

**Model Architecture**: ResNet-inspired CNN with the following components:
- Initial convolutional layer (32 filters)
- 3 convolutional blocks with residual connections
- Progressive filter scaling: [32, 64, 128]
- Batch normalization and dropout regularization
- Global average pooling
- Dense classification layers with L2 regularization
- Softmax output layer for probability predictions

**Loss Functions Evaluated**:

1. **Standard Cross-Entropy**: The baseline, expected to overfit to spurious features
2. **Label Smoothing**: Regularization to reduce overconfidence
3. **Focal Loss**: Down-weights easy examples to focus on hard ones
4. **Goodhart-Aware Loss**: Combines cross-entropy with entropy and mutual
   information regularization to improve robustness

Comprehensive Analysis Pipeline
------------------------------

The experiment employs a multi-faceted analysis approach using the `ModelAnalyzer`
and experiment-specific metrics:

**Training Analysis**:
- Training and validation curves for all loss functions
- Convergence behavior and early stopping

**Model Performance Evaluation**:
- **Test Set Accuracy**: The primary metric for robustness on the uncorrelated test set
- Top-k accuracy and loss values

**Calibration and Distribution Analysis** (via `ModelAnalyzer`):
- Expected Calibration Error (ECE) and Brier score
- Reliability diagrams and confidence histograms
- Entropy and probability distribution analysis of model outputs

**Weight and Activation Analysis** (via `ModelAnalyzer`):
- Layer-wise weight distribution statistics
- Analysis of information flow and feature representations

**Spurious Correlation Analysis** (Custom Metrics):
- **Generalization Gap**: `Train Accuracy - Test Accuracy`, measuring overfitting
  to spurious correlations
- **Color Dependency Score**: Quantifies how much predictions change when image
  colors are shuffled, directly measuring reliance on the spurious feature

Expected Outcomes and Insights
------------------------------

This experiment is designed to reveal:

1. **Robustness to Spurious Correlations**: Which loss functions produce models
   that maintain high accuracy when the spurious correlation is removed

2. **Generalization vs. Shortcut Learning**: The extent to which models learn
   the intended features (digit shape) versus unintended shortcuts (color)

3. **Information-Theoretic Benefits**: Whether the Goodhart-Aware Loss provides
   measurable advantages in preventing overfitting to spurious signals

Theoretical Foundation
----------------------

This experiment tests the practical implications of Goodhart's Law in machine
learning: "When a measure becomes a target, it ceases to be a good measure."
By introducing spurious correlations, we create a scenario where models can
achieve high training accuracy through shortcuts rather than learning robust
features. The information-theoretic regularization in GoodhartAwareLoss aims
to prevent this by encouraging models to compress away spurious signals.
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
from dl_techniques.utils.train import TrainingConfig, train_model

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    ClassificationResults,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization
)

from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput
)


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for the Colored MNIST spurious correlation experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, loss function definitions, and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "colored_mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 3)

    # --- Spurious Correlation Parameters ---
    train_correlation_strength: float = 0.95
    test_correlation_strength: float = 0.0
    validation_correlation_strength: float = 0.5
    validation_split: float = 0.1

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # --- Training Parameters ---
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
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
        'GAL_0': lambda: GoodhartAwareLoss(
            entropy_weight=0.0, mi_weight=0.01, from_logits=False
        ),
        'GAL_01': lambda: GoodhartAwareLoss(
            entropy_weight=0.1, mi_weight=0.01, from_logits=False
        ),
        'GAL_001': lambda: GoodhartAwareLoss(
            entropy_weight=0.01, mi_weight=0.01, from_logits=False
        ),
        'GAL_005': lambda: GoodhartAwareLoss(
            entropy_weight=0.05, mi_weight=0.01, from_logits=False
        ),
        'GAL_0001': lambda: GoodhartAwareLoss(
            entropy_weight=0.001, mi_weight=0.01, from_logits=False
        ),
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "colored_mnist_spurious_correlation"
    random_seed: int = 42

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        calibration_bins=15,
        save_plots=True,
        plot_style='publication',
    ))


# ==============================================================================
# DATASET GENERATION
# ==============================================================================

@dataclass
class ColoredMNISTData:
    """Container for the Colored MNIST dataset splits."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def colorize_mnist(
    images: np.ndarray,
    labels: np.ndarray,
    correlation: float,
    num_classes: int
) -> np.ndarray:
    """
    Colorizes MNIST images with a specified label-color correlation.

    Args:
        images: Grayscale MNIST images
        labels: Integer labels for the images
        correlation: Strength of correlation between labels and colors (0.0 to 1.0)
        num_classes: Number of classes (colors) to use

    Returns:
        RGB images with applied color based on correlation strength
    """
    # Define a distinct color palette for each digit class
    color_palette = [
        (255, 0, 0),    # Red for 0
        (0, 255, 0),    # Green for 1
        (0, 0, 255),    # Blue for 2
        (255, 255, 0),  # Yellow for 3
        (255, 0, 255),  # Magenta for 4
        (0, 255, 255),  # Cyan for 5
        (128, 0, 0),    # Dark red for 6
        (0, 128, 0),    # Dark green for 7
        (0, 0, 128),    # Dark blue for 8
        (128, 128, 0)   # Olive for 9
    ]

    # Normalize grayscale images and add channel dimension
    normalized_images = np.expand_dims(images.astype(np.float32) / 255.0, axis=-1)
    colored_images = np.zeros((*images.shape, 3), dtype=np.float32)

    for i, label in enumerate(labels):
        # Decide whether to use correlated or random color
        if np.random.rand() < correlation:
            color_idx = int(label)  # Use label-correlated color
        else:
            color_idx = np.random.randint(num_classes)  # Use random color

        # Apply the selected color
        color = np.array(color_palette[color_idx], dtype=np.float32) / 255.0
        colored_images[i] = normalized_images[i] * color

    return colored_images


def create_colored_mnist_dataset(config: ExperimentConfig) -> ColoredMNISTData:
    """
    Generates the full Colored MNIST dataset with specified correlations.

    Args:
        config: Experiment configuration containing correlation parameters

    Returns:
        ColoredMNISTData object containing all dataset splits
    """
    logger.info("Generating Colored MNIST dataset...")

    # Set random seed for reproducibility
    np.random.seed(config.random_seed)

    # Load original MNIST data
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.mnist.load_data()

    # Create validation split from training data
    val_size = int(len(x_train_orig) * config.validation_split)
    indices = np.random.permutation(len(x_train_orig))

    x_train_split = x_train_orig[indices[val_size:]]
    y_train_split = y_train_orig[indices[val_size:]]
    x_val_split = x_train_orig[indices[:val_size]]
    y_val_split = y_train_orig[indices[:val_size]]

    # Colorize each split with appropriate correlation
    x_train = colorize_mnist(
        x_train_split, y_train_split,
        config.train_correlation_strength, config.num_classes
    )
    x_val = colorize_mnist(
        x_val_split, y_val_split,
        config.validation_correlation_strength, config.num_classes
    )
    x_test = colorize_mnist(
        x_test_orig, y_test_orig,
        config.test_correlation_strength, config.num_classes
    )

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train_split, config.num_classes)
    y_val = keras.utils.to_categorical(y_val_split, config.num_classes)
    y_test = keras.utils.to_categorical(y_test_orig, config.num_classes)

    logger.info("Dataset generated successfully:")
    logger.info(f"   Train: {len(x_train)} samples, {config.train_correlation_strength:.0%} correlation")
    logger.info(f"   Val:   {len(x_val)} samples, {config.validation_correlation_strength:.0%} correlation")
    logger.info(f"   Test:  {len(x_test)} samples, {config.test_correlation_strength:.0%} correlation")

    return ColoredMNISTData(x_train, y_train, x_val, y_val, x_test, y_test)


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
    Build a complete CNN model for Colored MNIST classification with softmax output.

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

    # Initial convolutional layer
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='initial_conv'
    )(inputs)

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

    # Compile with optimizer and metrics
    optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']  # Use string shorthand for compatibility
    )

    return model


# ==============================================================================
# EXPERIMENT-SPECIFIC ANALYSIS
# ==============================================================================

def compute_color_dependency_score(
    model: keras.Model,
    x_test: np.ndarray,
    n_shuffles: int = 5
) -> float:
    """
    Computes how much a model's predictions change when colors are shuffled.

    This metric directly measures the model's reliance on color information.
    A high score indicates the model depends heavily on color (spurious feature),
    while a low score suggests the model relies on shape (robust feature).

    Args:
        model: Trained Keras model
        x_test: Test images to evaluate
        n_shuffles: Number of times to shuffle colors for averaging

    Returns:
        Average fraction of predictions that change when colors are shuffled
    """
    # Get original predictions
    original_preds = np.argmax(model.predict(x_test, verbose=0), axis=1)

    changes = []
    for _ in range(n_shuffles):
        # Create a copy of test images
        shuffled_images = x_test.copy()

        # Shuffle RGB channels for each image independently
        for i in range(len(shuffled_images)):
            # Randomly permute the color channels
            shuffled_images[i] = shuffled_images[i][:, :, np.random.permutation(3)]

        # Get predictions on shuffled images
        shuffled_preds = np.argmax(model.predict(shuffled_images, verbose=0), axis=1)

        # Calculate fraction of changed predictions
        changes.append(np.mean(original_preds != shuffled_preds))

    return float(np.mean(changes))


def analyze_robustness(
    models: Dict[str, keras.Model],
    data: ColoredMNISTData,
    config: ExperimentConfig
) -> Dict[str, Dict[str, float]]:
    """
    Computes experiment-specific robustness metrics.

    This function evaluates how well each model resists spurious correlations
    by measuring the generalization gap and color dependency.

    Args:
        models: Dictionary of trained models
        data: Colored MNIST dataset
        config: Experiment configuration

    Returns:
        Dictionary of robustness metrics for each model
    """
    logger.info("Computing spurious correlation robustness metrics...")

    robustness_results = {}

    for name, model in models.items():
        logger.info(f"   Analyzing robustness for {name}...")

        # Evaluate on training set (high correlation)
        train_metrics = model.evaluate(
            data.x_train, data.y_train,
            verbose=0, batch_size=512
        )
        train_metrics_dict = dict(zip(model.metrics_names, train_metrics))

        # Evaluate on test set (zero correlation)
        test_metrics = model.evaluate(
            data.x_test, data.y_test,
            verbose=0, batch_size=512
        )
        test_metrics_dict = dict(zip(model.metrics_names, test_metrics))

        # Manual accuracy calculation as fallback
        train_preds = np.argmax(model.predict(data.x_train, verbose=0, batch_size=512), axis=1)
        train_true = np.argmax(data.y_train, axis=1)
        manual_train_acc = np.mean(train_preds == train_true)

        test_preds = np.argmax(model.predict(data.x_test, verbose=0, batch_size=512), axis=1)
        test_true = np.argmax(data.y_test, axis=1)
        manual_test_acc = np.mean(test_preds == test_true)

        # Handle both 'accuracy' and 'categorical_accuracy' metric names, with manual fallback
        train_acc = train_metrics_dict.get('accuracy',
                    train_metrics_dict.get('categorical_accuracy', manual_train_acc))
        test_acc = test_metrics_dict.get('accuracy',
                   test_metrics_dict.get('categorical_accuracy', manual_test_acc))

        # Compute color dependency score
        color_dependency = compute_color_dependency_score(model, data.x_test)

        # Store results
        robustness_results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'generalization_gap': train_acc - test_acc,
            'color_dependency': color_dependency,
            'train_loss': train_metrics_dict.get('loss', 0.0),
            'test_loss': test_metrics_dict.get('loss', 0.0),
        }

        # Log detailed metrics
        logger.info(f"     Train accuracy: {robustness_results[name]['train_accuracy']:.4f}")
        logger.info(f"     Test accuracy:  {robustness_results[name]['test_accuracy']:.4f}")
        logger.info(f"     Gen. gap:       {robustness_results[name]['generalization_gap']:.4f}")
        logger.info(f"     Color dep.:     {robustness_results[name]['color_dependency']:.4f}")

    logger.info("Robustness analysis completed.")

    return robustness_results


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete Colored MNIST spurious correlation experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset generation with controlled spurious correlations
    2. Model training for each loss function
    3. Model analysis and evaluation
    4. Robustness metric computation
    5. Visualization generation
    6. Results compilation and reporting

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

    # Initialize visualization manager (new framework)
    vis_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations"
    )

    # Register visualization templates
    vis_manager.register_template("training_curves", TrainingCurvesVisualization)
    vis_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    # Log experiment start
    logger.info("Starting Colored MNIST Spurious Correlation Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET GENERATION =====
    logger.info("Generating Colored MNIST dataset...")
    dataset = create_colored_mnist_dataset(config)
    logger.info("Dataset generation completed")

    # Debug information about data format
    logger.info(f"Dataset shapes - Train: {dataset.x_train.shape}, {dataset.y_train.shape}")
    logger.info(f"Dataset shapes - Test: {dataset.x_test.shape}, {dataset.y_test.shape}")
    logger.info(f"Sample labels (one-hot): {dataset.y_train[:3]}")
    logger.info(f"Data range - Min: {dataset.x_train.min():.3f}, Max: {dataset.x_train.max():.3f}")

    # ===== MODEL TRAINING PHASE =====
    logger.info("Starting model training phase...")
    trained_models = {}  # Store trained models
    all_histories = {}  # Store training histories

    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"--- Training model with {loss_name} loss ---")

        # Build model for this loss function
        model = build_model(config, loss_fn_factory(), loss_name)

        # Log model architecture info
        logger.info(f"Model {loss_name} output layer: {model.output.name}")
        logger.info(f"Model {loss_name} parameters: {model.count_params():,}")
        logger.info(f"Model {loss_name} metrics: {model.metrics_names}")

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
            dataset.x_train, dataset.y_train,
            dataset.x_val, dataset.y_val,
            training_config
        )

        # Quick evaluation after training to verify
        quick_eval = model.evaluate(dataset.x_val, dataset.y_val, verbose=0)
        logger.info(f"Post-training validation metrics: {dict(zip(model.metrics_names, quick_eval))}")

        # Store results
        trained_models[loss_name] = model
        all_histories[loss_name] = history.history
        logger.info(f"{loss_name} training completed!")

    # ===== MEMORY MANAGEMENT =====
    logger.info("Triggering garbage collection...")
    gc.collect()

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        # Initialize the model analyzer with trained models
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        # Run comprehensive analysis
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(dataset))
        logger.info("Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

    # ===== SPURIOUS CORRELATION ANALYSIS =====
    robustness_results = analyze_robustness(trained_models, dataset, config)

    # ===== VISUALIZATION GENERATION =====
    logger.info("Generating training history and confusion matrix plots...")

    # Convert training histories to TrainingHistory objects
    training_histories = {}
    for name, hist_dict in all_histories.items():
        if len(hist_dict.get('loss', [])) > 0:
            training_histories[name] = TrainingHistory(
                epochs=list(range(len(hist_dict['loss']))),
                train_loss=hist_dict['loss'],
                val_loss=hist_dict.get('val_loss', []),
                train_metrics={
                    'accuracy': hist_dict.get('accuracy', [])
                },
                val_metrics={
                    'accuracy': hist_dict.get('val_accuracy', [])
                }
            )

    # Plot training history comparison
    if training_histories:
        try:
            vis_manager.visualize(
                data=training_histories,
                plugin_name="training_curves",
                metrics_to_plot=['accuracy', 'loss'],
                show=False
            )
            logger.info("Training history visualization created")
        except Exception as e:
            logger.error(f"Failed to create training history visualization: {e}")

    # Generate confusion matrices for model comparison
    raw_predictions = {
        name: model.predict(dataset.x_test, verbose=0)
        for name, model in trained_models.items()
    }
    class_predictions = {
        name: np.argmax(preds, axis=1)
        for name, preds in raw_predictions.items()
    }

    # Convert y_test to class indices for confusion matrix
    y_true_indices = np.argmax(dataset.y_test, axis=1)

    # Create classification results for each model
    for model_name, y_pred in class_predictions.items():
        try:
            classification_data = ClassificationResults(
                y_true=y_true_indices,
                y_pred=y_pred,
                y_prob=raw_predictions[model_name],
                class_names=[str(i) for i in range(config.num_classes)],
                model_name=model_name
            )

            vis_manager.visualize(
                data=classification_data,
                plugin_name="confusion_matrix",
                normalize='true',
                show=False
            )
        except Exception as e:
            logger.error(f"Failed to create confusion matrix for {model_name}: {e}")

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("Evaluating final model performance on test set...")

    performance_results = {}

    for name, model in trained_models.items():
        # Get model evaluation metrics
        eval_results = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        # Debug logging to inspect metrics
        logger.debug(f"Model {name} metrics names: {model.metrics_names}")
        logger.debug(f"Model {name} eval results: {eval_results}")

        # Manual accuracy calculation for verification
        predictions = model.predict(dataset.x_test, verbose=0)
        y_true_indices = np.argmax(dataset.y_test, axis=1)
        y_pred_indices = np.argmax(predictions, axis=1)
        manual_accuracy = np.mean(y_pred_indices == y_true_indices)

        logger.info(f"Model {name} - Manual accuracy: {manual_accuracy:.4f}")

        # Store standardized metrics - handle both 'accuracy' and 'categorical_accuracy'
        accuracy_value = metrics_dict.get('accuracy', metrics_dict.get('categorical_accuracy', manual_accuracy))
        performance_results[name] = {
            'accuracy': accuracy_value,
            'loss': metrics_dict.get('loss', 0.0)
        }

        # Log final metrics for this model
        logger.info(f"Model {name} final test metrics: {performance_results[name]}")

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'config': config,
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'robustness_analysis': robustness_results,
        'histories': all_histories,
        'trained_models': trained_models
    }

    # Print comprehensive summary
    print_experiment_summary(results_payload)

    return results_payload


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of experimental results.

    This function generates a detailed report of all experimental outcomes,
    including performance metrics, robustness analysis, calibration metrics,
    and the final verdict on loss function effectiveness against spurious
    correlations.

    Args:
        results: Dictionary containing all experimental results and analysis
    """
    logger.info("=" * 80)
    logger.info("SPURIOUS CORRELATION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== EXPERIMENT CONFIGURATION =====
    config = results.get('config')
    if config:
        logger.info("EXPERIMENT SETUP:")
        logger.info(f"   Train/Val/Test Correlation: {config.train_correlation_strength:.0%} / "
                   f"{config.validation_correlation_strength:.0%} / {config.test_correlation_strength:.0%}")
        logger.info("")

    # ===== PERFORMANCE METRICS =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("PERFORMANCE METRICS (on Test Set with 0% Color Correlation):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Loss':<12}")
        logger.info("-" * 45)

        for model_name, metrics in results['performance_analysis'].items():
            accuracy = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {loss:<12.4f}")

    # ===== ROBUSTNESS METRICS =====
    if 'robustness_analysis' in results and results['robustness_analysis']:
        logger.info("ROBUSTNESS METRICS:")
        logger.info(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Gen Gap':<12} {'Color Dep':<12}")
        logger.info("-" * 70)

        for model_name, metrics in results['robustness_analysis'].items():
            train_acc = metrics.get('train_accuracy', 0.0)
            test_acc = metrics.get('test_accuracy', 0.0)
            gen_gap = metrics.get('generalization_gap', 0.0)
            color_dep = metrics.get('color_dependency', 0.0)

            logger.info(
                f"{model_name:<20} {train_acc:<12.4f} {test_acc:<12.4f} "
                f"{gen_gap:<12.4f} {color_dep:<12.4f}"
            )

    # ===== CALIBRATION METRICS =====
    model_analysis = results.get('model_analysis')
    if model_analysis and model_analysis.calibration_metrics:
        logger.info("CALIBRATION METRICS (from Model Analyzer):")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)

        for model_name, cal_metrics in model_analysis.calibration_metrics.items():
            # Get the corresponding confidence metrics for the same model
            conf_metrics = model_analysis.confidence_metrics.get(model_name, {})

            logger.info(
                f"{model_name:<20} {cal_metrics.get('ece', 0.0):<12.4f} "
                f"{cal_metrics.get('brier_score', 0.0):<15.4f} "
                f"{conf_metrics.get('mean_entropy', 0.0):<12.4f}"
            )

    # ===== EXPERIMENTAL VERDICT =====
    if 'robustness_analysis' in results and results['robustness_analysis']:
        logger.info("=" * 80)
        logger.info("EXPERIMENTAL VERDICT:")

        rob_res = results['robustness_analysis']

        # Find best models according to different criteria
        best_test_acc = max(rob_res, key=lambda k: rob_res[k]['test_accuracy'])
        most_robust = min(rob_res, key=lambda k: rob_res[k]['color_dependency'])
        smallest_gap = min(rob_res, key=lambda k: rob_res[k]['generalization_gap'])

        logger.info(f"Highest Test Accuracy:      {best_test_acc} ({rob_res[best_test_acc]['test_accuracy']:.4f})")
        logger.info(f"Lowest Color Dependency:    {most_robust} ({rob_res[most_robust]['color_dependency']:.4f})")
        logger.info(f"Smallest Generalization Gap: {smallest_gap} ({rob_res[smallest_gap]['generalization_gap']:.4f})")

        # Compare GoodhartAware loss to baseline
        if 'GoodhartAware' in rob_res and 'CrossEntropy' in rob_res:
            gal_metrics = rob_res['GoodhartAware']
            ce_metrics = rob_res['CrossEntropy']

            acc_diff = gal_metrics['test_accuracy'] - ce_metrics['test_accuracy']
            color_diff = ce_metrics['color_dependency'] - gal_metrics['color_dependency']
            gap_diff = ce_metrics['generalization_gap'] - gal_metrics['generalization_gap']

            logger.info("GoodhartAware vs. CrossEntropy:")
            logger.info(f"   Test Accuracy Improvement:      {acc_diff:+.4f}")
            logger.info(f"   Color Dependency Reduction:     {color_diff:+.4f}")
            logger.info(f"   Generalization Gap Reduction:   {gap_diff:+.4f}")

            # Overall assessment
            positive_indicators = sum([acc_diff > 0.01, color_diff > 0.05, gap_diff > 0.05])

            if positive_indicators >= 3:
                logger.info("   VERDICT: Strong support for Goodhart-Aware loss effectiveness!")
                logger.info("            The model successfully resists spurious correlations.")
            elif positive_indicators >= 2:
                logger.info("   VERDICT: Positive evidence for Goodhart-Aware loss benefits.")
                logger.info("            Some improvement in robustness to spurious features.")
            elif positive_indicators >= 1:
                logger.info("   VERDICT: Mixed results - some benefits observed.")
                logger.info("            Consider tuning hyperparameters for this task.")
            else:
                logger.info("   VERDICT: Limited benefit observed in this configuration.")
                logger.info("            May need different hyperparameters or architecture.")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the Colored MNIST spurious correlation experiment.

    This function serves as the entry point for the experiment, handling
    configuration setup, experiment execution, and error handling.
    """
    logger.info("Colored MNIST Spurious Correlation Experiment")
    logger.info("=" * 80)

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"   Loss Functions: {list(config.loss_functions.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Model Architecture: {len(config.conv_filters)} conv blocks, "
               f"{len(config.dense_units)} dense layers")
    logger.info(f"   Spurious Correlation: Train={config.train_correlation_strength:.0%}, "
               f"Test={config.test_correlation_strength:.0%}")
    logger.info(f"   Output: Softmax probabilities (from_logits=False)")
    logger.info("")

    try:
        # Run the complete experiment
        _ = run_experiment(config)
        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()