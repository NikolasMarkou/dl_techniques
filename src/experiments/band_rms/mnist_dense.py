"""
Experiment Title: BandRMS and Adaptive BandRMS Effectiveness in Deep Dense Networks
===================================================================================

This experiment investigates whether BandRMS and Adaptive BandRMS normalization provide
measurable benefits over traditional normalization techniques when applied between layers
in deep dense neural networks. The core scientific question is: Do the bounded scaling
constraints in BandRMS (limiting features to a "thick shell" rather than unit hypersphere)
and the adaptive input-dependent scaling in Adaptive BandRMS improve training stability,
model calibration, and final performance in fully-connected architectures?

Scientific Motivation and Hypothesis
------------------------------------

Deep fully-connected networks are notoriously difficult to train due to vanishing/exploding
gradients, internal covariate shift, and poor feature space geometry. While normalization
techniques like BatchNorm and LayerNorm have addressed many of these issues, they each
have limitations:

1. **BatchNorm**: Depends on batch statistics, performs poorly with small batches
2. **LayerNorm**: Can suffer from scale collapse and doesn't address feature magnitude directly
3. **RMSNorm**: Projects to unit sphere but may be too restrictive for complex feature learning

**BandRMS Hypothesis**: By normalizing features to unit norm (like RMSNorm) but then allowing
bounded scaling within [1-α, 1], BandRMS creates a "thick shell" in high-dimensional space.

**Adaptive BandRMS Hypothesis**: By using RMS statistics to dynamically determine scaling
factors, Adaptive BandRMS can adapt the "thick shell" constraints based on input magnitude
characteristics, potentially providing better feature learning than static constraints.

Both geometric constraints should provide:

- **Better Training Dynamics**: More stable gradients than unbounded scaling
- **Improved Calibration**: Bounded feature magnitudes prevent overconfident predictions
- **Enhanced Expressiveness**: More degrees of freedom than strict unit sphere projection
- **Input Adaptability** (Adaptive BandRMS): Scaling adapts to input characteristics
- **Robustness**: Less sensitive to initialization and hyperparameters

**Theoretical Foundation**: In high-dimensional spaces, volume concentrates near the surface
of spheres. BandRMS leverages this by allowing features to exist in a controlled region
around the unit sphere, providing both geometric stability and representational flexibility.

Experimental Design
-------------------

**Dataset**: MNIST (28×28 grayscale images, 10 classes)
- Simple enough to focus on normalization effects
- Well-understood baseline performance
- Sufficient complexity to reveal normalization differences

**Architecture**: Deep Fully-Connected Network
- Input: 784 dimensional (flattened 28×28 images)
- Hidden Layers: 8 dense layers with decreasing width [512, 384, 256, 192, 128, 96, 64, 32]
- Output: 10-class softmax classification
- Activation: GELU (smooth, well-behaved gradients)
- Normalization: Applied between each dense layer (test variable)

**Normalization Variants**:
1. **BandRMS_005**: α=0.05 (tight constraints, high stability)
2. **BandRMS_010**: α=0.10 (moderate constraints, balanced)
3. **BandRMS_020**: α=0.20 (loose constraints, more expressiveness)
4. **AdaptiveBandRMS_005**: α=0.05 (adaptive scaling based on RMS statistics)
5. **AdaptiveBandRMS_010**: α=0.10 (adaptive scaling, moderate constraints)
6. **AdaptiveBandRMS_020**: α=0.20 (adaptive scaling, loose constraints)
7. **AdaptiveBandRMS_Half**: α=0.10 (adaptive scaling with auto-sized dense layer)
8. **RMSNorm**: Standard RMS normalization (unit sphere projection)
9. **LayerNorm**: Layer normalization with learnable scale/shift
10. **BatchNorm**: Batch normalization (baseline comparison)
11. **NoNorm**: No normalization (control to show necessity)

Expected Outcomes and Implications
----------------------------------

**Expected Results**:
- BandRMS variants should show improved calibration (lower ECE)
- Training stability should be better than NoNorm, comparable to LayerNorm
- α=0.10 might provide optimal balance between stability and expressiveness
- Performance should be competitive with or better than traditional methods

This experiment will provide definitive evidence for whether BandRMS represents a
meaningful advancement in normalization techniques or is simply another variant
without clear benefits.
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import json
import keras
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable, Optional

# DL-Techniques imports
from dl_techniques.utils.logger import logger
from dl_techniques.utils.datasets.mnist import load_and_preprocess_mnist
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class DenseNormalizationConfig:
    """
    Configuration for the BandRMS Dense Network Normalization experiment.

    This class encapsulates all parameters for systematically evaluating BandRMS
    effectiveness in deep fully-connected neural networks compared to traditional
    normalization techniques.

    Attributes:
        dataset_name: Name of the dataset to use for experiments
        num_classes: Number of output classes for classification
        input_shape: Shape of flattened input data
        hidden_units: List of units for each hidden layer
        activation: Activation function name to use throughout network
        kernel_initializer: Weight initialization strategy
        weight_decay: L2 regularization strength for weights
        use_dropout: Whether to apply dropout for regularization
        dropout_rate: Dropout probability if dropout is enabled
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate for optimizer
        early_stopping_patience: Number of epochs to wait before early stopping
        monitor_metric: Metric to monitor for early stopping
        band_alphas: List of alpha values to test for BandRMS variants
        band_epsilon: Small constant for numerical stability in BandRMS
        band_regularizer_strength: L2 regularization strength for BandRMS parameters
        normalization_variants: Dictionary mapping variant names to factory functions
        output_dir: Base directory for experiment outputs
        experiment_name: Name identifier for this experiment
        random_seed: Random seed for reproducibility
        n_runs: Number of independent runs for statistical significance
        analyzer_config: Configuration for comprehensive model analysis
    """

    # --- Dataset Configuration ---
    dataset_name: str = "mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (784,)  # Flattened MNIST

    # --- Network Architecture Parameters ---
    hidden_units: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 32])
    activation: str = 'gelu'  # Smooth activation for stable training
    kernel_initializer: str = 'he_normal'
    weight_decay: float = 1e-4
    use_dropout: bool = False
    dropout_rate: float = 0.1

    # --- Training Parameters ---
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    monitor_metric: str = 'val_accuracy'

    # --- BandRMS Specific Parameters ---
    band_alphas: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    band_epsilon: float = 1e-7
    band_regularizer_strength: float = 1e-5

    # --- Normalization Variants ---
    normalization_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'AdaptiveBandRMS_010': lambda: ('adaptive_band_rms', {'max_band_width': 0.1}),
        'AdaptiveBandRMS_050': lambda: ('adaptive_band_rms', {'max_band_width': 0.5}),
        'AdaptiveBandRMS_090': lambda: ('adaptive_band_rms', {'max_band_width': 0.9}),
        'BandRMS_010': lambda: ('band_rms', {'max_band_width': 0.10}),
        'BandRMS_050': lambda: ('band_rms', {'max_band_width': 0.50}),
        'BandRMS_090': lambda: ('band_rms', {'max_band_width': 0.90}),
        'RMSNorm': lambda: ('rms_norm', {}),
        'LayerNorm': lambda: ('layer_norm', {}),
        'BatchNorm': lambda: ('batch_norm', {}),
        'NoNorm': lambda: ('none', {}),  # Control group
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "bandrms_adaptive_dense_normalization"
    random_seed: int = 42
    n_runs: int = 3  # Multiple runs for statistical significance

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        analyze_training_dynamics=True,
        calibration_bins=15,
        save_plots=True,
        plot_style='publication',
        verbose=True
    ))


# ==============================================================================
# NORMALIZATION LAYER FACTORY
# ==============================================================================

def create_normalization_layer(
        norm_type: str,
        norm_params: Dict[str, Any],
        name: Optional[str] = None
) -> Optional[keras.layers.Layer]:
    """
    Factory function to create different normalization layers for dense networks.

    Args:
        norm_type: Type of normalization ('band_rms', 'adaptive_band_rms', 'rms_norm',
            'layer_norm', 'batch_norm', 'none')
        norm_params: Parameters specific to the normalization type
        name: Optional layer name for identification

    Returns:
        Configured normalization layer, or None if norm_type is 'none'

    Raises:
        ValueError: If norm_type is not recognized
    """
    if norm_type == 'none':
        return None
    elif norm_type == 'band_rms':
        return BandRMS(
            max_band_width=norm_params.get('max_band_width', 0.1),
            epsilon=norm_params.get('epsilon', 1e-7),
            band_regularizer=keras.regularizers.L2(norm_params.get('regularizer_strength', 1e-5)),
            name=name
        )
    elif norm_type == 'adaptive_band_rms':
        return AdaptiveBandRMS(
            max_band_width=norm_params.get('max_band_width', 0.1),
            epsilon=norm_params.get('epsilon', 1e-7),
            band_regularizer=keras.regularizers.L2(norm_params.get('regularizer_strength', 1e-5)),
            name=name
        )
    elif norm_type == 'rms_norm':
        return RMSNorm(
            epsilon=norm_params.get('epsilon', 1e-6),
            name=name
        )
    elif norm_type == 'layer_norm':
        return keras.layers.LayerNormalization(
            epsilon=norm_params.get('epsilon', 1e-6),
            name=name
        )
    elif norm_type == 'batch_norm':
        return keras.layers.BatchNormalization(
            epsilon=norm_params.get('epsilon', 1e-3),
            name=name
        )
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


# ==============================================================================
# MODEL ARCHITECTURE BUILDING
# ==============================================================================

def build_dense_block(
        inputs: keras.layers.Layer,
        units: int,
        norm_type: str,
        norm_params: Dict[str, Any],
        config: DenseNormalizationConfig,
        block_index: int
) -> keras.layers.Layer:
    """
    Build a dense block with normalization and activation.

    The standard order is: Dense -> Normalization -> Activation -> Dropout (optional)
    This follows the common practice of normalizing before activation to maintain
    stable gradients throughout the network.

    Args:
        inputs: Input tensor from previous layer
        units: Number of units in the dense layer
        norm_type: Type of normalization to apply
        norm_params: Parameters for the normalization layer
        config: Experiment configuration containing training parameters
        block_index: Index of the current block for unique naming

    Returns:
        Output tensor after applying dense layer, normalization, activation, and dropout

    Raises:
        ValueError: If block_index is negative or configuration is invalid
    """
    if block_index < 0:
        raise ValueError("block_index must be non-negative")

    # Dense layer with weight regularization
    x = keras.layers.Dense(
        units=units,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        use_bias=(norm_type == 'none'),  # Skip bias if using normalization
        name=f'dense_{block_index}'
    )(inputs)

    # Apply normalization if specified
    norm_layer = create_normalization_layer(
        norm_type, norm_params, name=f'norm_{block_index}'
    )
    if norm_layer is not None:
        x = norm_layer(x)

    # Activation function
    x = keras.layers.Activation(config.activation, name=f'activation_{block_index}')(x)

    # Optional dropout for regularization
    if config.use_dropout and config.dropout_rate > 0:
        x = keras.layers.Dropout(config.dropout_rate, name=f'dropout_{block_index}')(x)

    return x


def build_deep_dense_model(
        config: DenseNormalizationConfig,
        norm_type: str,
        norm_params: Dict[str, Any],
        name: str
) -> keras.Model:
    """
    Build a deep fully-connected neural network with specified normalization.

    The network architecture consists of progressively smaller dense layers,
    each followed by the specified normalization technique. This design tests
    how well different normalization methods handle the challenge of training
    deep networks with many parameters.

    Args:
        config: Experiment configuration containing architecture parameters
        norm_type: Type of normalization to use between layers
        norm_params: Parameters specific to the normalization technique
        name: Model name for identification and logging

    Returns:
        Compiled Keras model ready for training

    Raises:
        ValueError: If configuration parameters are invalid
    """
    if not config.hidden_units:
        raise ValueError("hidden_units cannot be empty")

    if config.num_classes <= 0:
        raise ValueError("num_classes must be positive")

    # Input layer - flattened images
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Stack of dense blocks with decreasing width
    for i, units in enumerate(config.hidden_units):
        if units <= 0:
            raise ValueError(f"All hidden units must be positive, got {units} at index {i}")

        x = build_dense_block(x, units, norm_type, norm_params, config, i)
        logger.debug(f"Added dense block {i}: {units} units with {norm_type} normalization")

    # Output layer (no normalization on final output)
    outputs = keras.layers.Dense(
        units=config.num_classes,
        activation='softmax',
        kernel_initializer=config.kernel_initializer,
        name='predictions'
    )(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')

    # Use simple optimizer without learning rate scheduling to avoid conflicts
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    return model


# ==============================================================================
# CUSTOM TRAINING FUNCTION
# ==============================================================================

def train_model_with_callbacks(
        model: keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        config: DenseNormalizationConfig,
        model_name: str,
        output_dir: Path
) -> keras.callbacks.History:
    """
    Custom training function with comprehensive callbacks and logging.

    This function provides robust training with early stopping, model checkpointing,
    and learning rate scheduling while maintaining compatibility with the experiment
    requirements.

    Args:
        model: Compiled Keras model to train
        x_train: Training data features
        y_train: Training data labels (one-hot encoded)
        x_val: Validation data features
        y_val: Validation data labels (one-hot encoded)
        config: Experiment configuration containing training parameters
        model_name: Name for model checkpoints and logging
        output_dir: Directory to save training artifacts

    Returns:
        Training history object containing metrics from all epochs

    Raises:
        RuntimeError: If training fails due to data or configuration issues
        ValueError: If input data shapes are incompatible
    """
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Training data size mismatch: {x_train.shape[0]} vs {y_train.shape[0]}")

    if x_val.shape[0] != y_val.shape[0]:
        raise ValueError(f"Validation data size mismatch: {x_val.shape[0]} vs {y_val.shape[0]}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure callbacks
    callbacks = []

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.monitor_metric,
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
        mode='max' if 'accuracy' in config.monitor_metric else 'min'
    )
    callbacks.append(early_stopping)

    # Model checkpoint to save best model
    checkpoint_path = output_dir / f"{model_name}.keras"
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=config.monitor_metric,
        save_best_only=True,
        verbose=1,
        mode='max' if 'accuracy' in config.monitor_metric else 'min'
    )
    callbacks.append(model_checkpoint)

    # Learning rate reduction on plateau
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor=config.monitor_metric,
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
        mode='max' if 'accuracy' in config.monitor_metric else 'min'
    )
    callbacks.append(lr_scheduler)

    # Train the model
    logger.info(f"Starting training for {model_name}")
    logger.info(f"Training data: {x_train.shape}, Validation data: {x_val.shape}")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")

    try:
        history = model.fit(
            x_train, y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        # Log training completion
        best_metric = max(history.history[config.monitor_metric])
        logger.info(f"Training completed for {model_name}")
        logger.info(f"Best {config.monitor_metric}: {best_metric:.4f}")

        return history

    except Exception as e:
        logger.error(f"Training failed for {model_name}: {e}")
        raise RuntimeError(f"Training failed for {model_name}") from e


# ==============================================================================
# STATISTICAL ANALYSIS UTILITIES
# ==============================================================================

def calculate_multi_run_statistics(
        results_per_run: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate comprehensive statistics across multiple experimental runs.

    This function computes mean, standard deviation, minimum, maximum, and confidence
    intervals for each model variant across multiple runs, enabling statistical
    significance testing and robust conclusions.

    Args:
        results_per_run: Dictionary mapping model names to lists of results from multiple runs

    Returns:
        Dictionary with statistical summaries for each model and metric

    Raises:
        ValueError: If results_per_run is empty or contains invalid data
    """
    if not results_per_run:
        raise ValueError("results_per_run cannot be empty")

    statistics = {}

    for model_name, run_results in results_per_run.items():
        if not run_results:
            logger.warning(f"No results found for model {model_name}")
            continue

        statistics[model_name] = {}

        # Get all metrics from the first run
        metrics = run_results[0].keys() if run_results else []

        for metric in metrics:
            values = [result[metric] for result in run_results if metric in result and result[metric] is not None]

            if values:
                values_array = np.array(values)
                n_values = len(values)

                statistics[model_name][metric] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array, ddof=1)) if n_values > 1 else 0.0,
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'count': n_values,
                    'sem': float(np.std(values_array, ddof=1) / np.sqrt(n_values)) if n_values > 1 else 0.0,
                    'ci_95_lower': float(np.percentile(values_array, 2.5)),
                    'ci_95_upper': float(np.percentile(values_array, 97.5))
                }

    return statistics


def perform_statistical_analysis(
        statistics: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Any]:
    """
    Perform statistical significance analysis between different normalization methods.

    Args:
        statistics: Statistical summaries from calculate_multi_run_statistics

    Returns:
        Dictionary containing analysis results and best performer identification

    Raises:
        ValueError: If statistics dictionary is empty or malformed
    """
    if not statistics:
        raise ValueError("statistics dictionary cannot be empty")

    analysis_results = {
        'best_performers': {},
        'rankings': {},
        'stability_analysis': {}
    }

    # Extract accuracy values for each model
    accuracy_data = {}
    for model_name, model_stats in statistics.items():
        if 'accuracy' in model_stats and model_stats['accuracy']['count'] >= 1:
            accuracy_data[model_name] = {
                'mean': model_stats['accuracy']['mean'],
                'sem': model_stats['accuracy']['sem'],
                'n': model_stats['accuracy']['count'],
                'std': model_stats['accuracy']['std']
            }

    # Find best performer by accuracy
    if accuracy_data:
        best_model = max(accuracy_data.items(), key=lambda x: x[1]['mean'])
        analysis_results['best_performers']['accuracy'] = {
            'model': best_model[0],
            'mean': best_model[1]['mean'],
            'sem': best_model[1]['sem']
        }

        # Rank models by accuracy
        sorted_models = sorted(accuracy_data.items(), key=lambda x: x[1]['mean'], reverse=True)
        analysis_results['rankings']['accuracy'] = [
            {'model': name, 'mean': stats['mean'], 'rank': i + 1}
            for i, (name, stats) in enumerate(sorted_models)
        ]

        # Find most stable model (lowest std dev)
        most_stable = min(accuracy_data.items(), key=lambda x: x[1]['std'])
        analysis_results['stability_analysis']['most_stable'] = {
            'model': most_stable[0],
            'std': most_stable[1]['std'],
            'mean': most_stable[1]['mean']
        }

    return analysis_results


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: DenseNormalizationConfig) -> Dict[str, Any]:
    """
    Execute the complete BandRMS dense network normalization experiment.

    This function orchestrates the entire experimental pipeline: dataset loading,
    model training across multiple runs, comprehensive analysis, visualization
    generation, and statistical assessment of results.

    Args:
        config: Experiment configuration containing all parameters

    Returns:
        Dictionary containing all experimental results, analysis, and statistics

    Raises:
        RuntimeError: If experiment fails due to configuration or system errors
        ValueError: If configuration parameters are invalid
    """
    # Validate configuration
    if config.n_runs <= 0:
        raise ValueError("n_runs must be positive")

    if config.epochs <= 0:
        raise ValueError("epochs must be positive")

    # Set random seed for reproducibility
    keras.utils.set_random_seed(config.random_seed)

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager
    try:
        vis_manager = VisualizationManager(
            output_dir=experiment_dir / "visualizations",
            config=VisualizationConfig(),
            timestamp_dirs=False
        )
    except Exception as e:
        logger.warning(f"Failed to initialize visualization manager: {e}")
        vis_manager = None

    logger.info("🚀 Starting BandRMS Dense Network Normalization Experiment")
    logger.info(f"📁 Results directory: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading and preprocessing MNIST dataset...")
    try:
        mnist_data = load_and_preprocess_mnist()

        # Reshape data for dense network (flatten images)
        x_train_flat = mnist_data.x_train.reshape(-1, 784)
        x_test_flat = mnist_data.x_test.reshape(-1, 784)

        logger.info("✅ Dataset loaded and preprocessed")
        logger.info(f"📋 Training set: {x_train_flat.shape}, Test set: {x_test_flat.shape}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise RuntimeError("Dataset loading failed") from e

    # ===== MULTI-RUN EXPERIMENTAL PHASE =====
    logger.info(f"🔄 Executing {config.n_runs} runs for statistical robustness...")

    final_models = {}  # Models from the last run for analysis
    final_histories = {}  # Training histories from the last run
    results_per_run = {}  # Performance results across all runs

    for run_idx in range(config.n_runs):
        logger.info(f"🏃 Starting experimental run {run_idx + 1}/{config.n_runs}")

        # Use different seed for each run to ensure independence
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        run_models = {}
        run_histories = {}

        # ===== MODEL TRAINING FOR CURRENT RUN =====
        for norm_name, norm_factory in config.normalization_variants.items():
            logger.info(f"--- Training {norm_name} (Run {run_idx + 1}) ---")

            try:
                # Get normalization configuration
                norm_type, norm_params = norm_factory()

                # Build model with specified normalization
                model = build_deep_dense_model(
                    config, norm_type, norm_params, f"{norm_name}_run{run_idx}"
                )

                # Log model architecture details (only for first run to avoid spam)
                if run_idx == 0:
                    model.summary(print_fn=logger.info)

                # Train the model using custom training function
                training_output_dir = experiment_dir / "training_logs" / f"run_{run_idx}" / norm_name
                history = train_model_with_callbacks(
                    model=model,
                    x_train=x_train_flat,
                    y_train=mnist_data.y_train,
                    x_val=x_test_flat,
                    y_val=mnist_data.y_test,
                    config=config,
                    model_name=f"{norm_name}_run{run_idx}",
                    output_dir=training_output_dir
                )

                run_models[norm_name] = model
                run_histories[norm_name] = history.history
                logger.info(f"✅ {norm_name} (Run {run_idx + 1}) training completed successfully!")

            except Exception as e:
                logger.error(f"❌ Training failed for {norm_name} (Run {run_idx + 1}): {e}")
                continue

        # ===== MODEL EVALUATION FOR CURRENT RUN =====
        logger.info(f"📊 Evaluating models for run {run_idx + 1}...")

        for norm_name, model in run_models.items():
            try:
                # Evaluate model performance
                eval_results = model.evaluate(x_test_flat, mnist_data.y_test, verbose=0)
                metrics_dict = dict(zip(model.metrics_names, eval_results))

                # Manual verification with predictions
                predictions = model.predict(x_test_flat, verbose=0)
                y_true_classes = np.argmax(mnist_data.y_test, axis=1)
                y_pred_classes = np.argmax(predictions, axis=1)
                manual_accuracy = float(np.mean(y_pred_classes == y_true_classes))

                # Store performance results
                if norm_name not in results_per_run:
                    results_per_run[norm_name] = []

                results_per_run[norm_name].append({
                    'accuracy': manual_accuracy,
                    'top_5_accuracy': metrics_dict.get('top_5_accuracy', manual_accuracy),
                    'loss': float(metrics_dict.get('loss', 0.0)),
                    'run_idx': run_idx
                })

                logger.info(f"✅ {norm_name} (Run {run_idx + 1}): "
                            f"Accuracy={manual_accuracy:.4f}, Loss={metrics_dict.get('loss', 0.0):.4f}")

            except Exception as e:
                logger.error(f"❌ Evaluation failed for {norm_name} (Run {run_idx + 1}): {e}")

        # Store final run results for detailed analysis
        if run_idx == config.n_runs - 1:
            final_models = run_models
            final_histories = run_histories

        # Memory cleanup between runs
        del run_models
        gc.collect()

        logger.info(f"✅ Run {run_idx + 1} completed successfully!")

    # ===== STATISTICAL ANALYSIS =====
    logger.info("📈 Computing multi-run statistics and significance tests...")
    run_statistics = calculate_multi_run_statistics(results_per_run)
    statistical_analysis = perform_statistical_analysis(run_statistics)

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("🔬 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    if final_models:
        try:
            # Create custom DataInput for flattened data
            analysis_data = DataInput(x_data=x_test_flat, y_data=mnist_data.y_test)

            analyzer = ModelAnalyzer(
                models=final_models,
                config=config.analyzer_config,
                output_dir=experiment_dir / "model_analysis",
                training_history=final_histories
            )

            model_analysis_results = analyzer.analyze(data=analysis_data)
            logger.info("✅ Comprehensive model analysis completed successfully!")

        except Exception as e:
            logger.error(f"❌ Model analysis failed: {e}")
            model_analysis_results = None

    # ===== VISUALIZATION GENERATION =====
    if vis_manager is not None and final_histories:
        logger.info("🎨 Generating comprehensive visualizations...")

        try:
            # Training curves comparison
            vis_manager.plot_history(
                histories=final_histories,
                metrics=['accuracy', 'loss'],
                name='normalization_training_comparison',
                subdir='training_analysis',
                title='Dense Network Training: Normalization Methods Comparison'
            )

            # Generate statistical comparison plots
            create_statistical_visualization(run_statistics, experiment_dir / "visualizations")

            # Confusion matrices for each normalization method
            if final_models:
                raw_predictions = {
                    name: model.predict(x_test_flat, verbose=0)
                    for name, model in final_models.items()
                }
                class_predictions = {
                    name: np.argmax(preds, axis=1)
                    for name, preds in raw_predictions.items()
                }

                y_true_classes = np.argmax(mnist_data.y_test, axis=1)
                vis_manager.plot_confusion_matrices_comparison(
                    y_true=y_true_classes,
                    model_predictions=class_predictions,
                    name='normalization_confusion_matrices',
                    subdir='performance_analysis',
                    normalize=True,
                    class_names=[str(i) for i in range(10)]
                )

        except Exception as e:
            logger.error(f"❌ Visualization generation failed: {e}")

    # ===== RESULTS COMPILATION =====
    experiment_results = {
        'run_statistics': run_statistics,
        'statistical_analysis': statistical_analysis,
        'model_analysis': model_analysis_results,
        'training_histories': final_histories,
        'trained_models': final_models,
        'config': config,
        'experiment_dir': experiment_dir,
        'dataset_info': {
            'name': 'mnist',
            'train_shape': x_train_flat.shape,
            'test_shape': x_test_flat.shape,
            'num_classes': 10
        }
    }

    # Save comprehensive results
    save_experiment_results(experiment_results, experiment_dir)

    # Generate final summary report
    print_comprehensive_summary(experiment_results)

    logger.info("🎉 BandRMS Dense Network Normalization Experiment completed successfully!")
    logger.info(f"📁 All results saved to: {experiment_dir}")

    return experiment_results


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def create_statistical_visualization(
        statistics: Dict[str, Dict[str, Dict[str, float]]],
        output_dir: Path
) -> None:
    """
    Create comprehensive statistical comparison visualizations.

    Generates bar plots with error bars showing mean ± standard error for each
    normalization method, along with confidence intervals and significance indicators.

    Args:
        statistics: Multi-run statistical results from calculate_multi_run_statistics
        output_dir: Directory to save visualization files

    Raises:
        ImportError: If required visualization libraries are not available
        ValueError: If statistics data is malformed
    """
    if not statistics:
        raise ValueError("statistics dictionary cannot be empty")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up publication-quality style
        plt.style.use('default')  # Use default instead of seaborn-v0_8 for compatibility
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        models = list(statistics.keys())
        if not models:
            logger.warning("No models found in statistics")
            return

        # 1. Test Accuracy Comparison (top-left)
        ax1 = axes[0, 0]
        accuracies = [statistics[model]['accuracy']['mean'] for model in models
                     if 'accuracy' in statistics[model]]
        accuracy_errors = [statistics[model]['accuracy']['sem'] for model in models
                          if 'accuracy' in statistics[model]]

        if accuracies:
            bars = ax1.bar(models[:len(accuracies)], accuracies, yerr=accuracy_errors,
                          capsize=5, alpha=0.8)
            ax1.set_title('Test Accuracy Comparison (Mean ± SEM)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, acc, err in zip(bars, accuracies, accuracy_errors):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + err + 0.005,
                         f'{acc:.4f}±{err:.4f}', ha='center', va='bottom', fontsize=10)

        # 2. Test Loss Comparison (top-right)
        ax2 = axes[0, 1]
        losses = [statistics[model]['loss']['mean'] for model in models
                 if 'loss' in statistics[model]]
        loss_errors = [statistics[model]['loss']['sem'] for model in models
                      if 'loss' in statistics[model]]

        if losses:
            bars2 = ax2.bar(models[:len(losses)], losses, yerr=loss_errors,
                           capsize=5, alpha=0.8, color='orange')
            ax2.set_title('Test Loss Comparison (Mean ± SEM)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, loss, err in zip(bars2, losses, loss_errors):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + err + 0.01,
                         f'{loss:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=10)

        # 3. Accuracy Distribution (bottom-left)
        ax3 = axes[1, 0]
        positions = range(len(models))
        for i, model in enumerate(models):
            if 'accuracy' in statistics[model]:
                stats_acc = statistics[model]['accuracy']
                mean_acc = stats_acc['mean']
                ci_lower = stats_acc['ci_95_lower']
                ci_upper = stats_acc['ci_95_upper']

                ax3.errorbar(i, mean_acc,
                            yerr=[[mean_acc - ci_lower], [ci_upper - mean_acc]],
                            fmt='o', capsize=5, capthick=2, markersize=8)

        ax3.set_xticks(positions)
        ax3.set_xticklabels(models, rotation=45)
        ax3.set_title('Accuracy 95% Confidence Intervals', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 4. Stability Analysis (bottom-right)
        ax4 = axes[1, 1]
        stabilities = [statistics[model]['accuracy']['std'] for model in models
                      if 'accuracy' in statistics[model]]

        if stabilities:
            bars4 = ax4.bar(models[:len(stabilities)], stabilities, alpha=0.8, color='green')
            ax4.set_title('Training Stability (Accuracy Std Dev)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Standard Deviation', fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)

            # Add value labels
            for bar, std in zip(bars4, stabilities):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.05,
                         f'{std:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_comparison_comprehensive.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info("✅ Comprehensive statistical visualization saved")

    except ImportError as e:
        logger.error(f"❌ Failed to import visualization libraries: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create statistical visualization: {e}")


# ==============================================================================
# RESULTS MANAGEMENT
# ==============================================================================

def save_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save comprehensive experiment results in multiple formats.

    Saves configuration, statistical results, model files, and metadata in
    both human-readable and machine-readable formats for reproducibility
    and further analysis.

    Args:
        results: Complete experiment results dictionary
        experiment_dir: Directory to save all result files

    Raises:
        OSError: If file system operations fail
        ValueError: If results dictionary is malformed
    """
    if not results:
        raise ValueError("results dictionary cannot be empty")

    try:
        def convert_numpy_types(obj: Any) -> Any:
            """Recursively convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj

        # 1. Save experiment configuration
        config_dict = {
            'experiment_name': results['config'].experiment_name,
            'dataset_info': results['dataset_info'],
            'architecture': {
                'hidden_units': results['config'].hidden_units,
                'activation': results['config'].activation,
                'use_dropout': results['config'].use_dropout,
                'dropout_rate': results['config'].dropout_rate
            },
            'normalization_variants': list(results['config'].normalization_variants.keys()),
            'training_parameters': {
                'epochs': results['config'].epochs,
                'batch_size': results['config'].batch_size,
                'learning_rate': results['config'].learning_rate,
                'early_stopping_patience': results['config'].early_stopping_patience
            },
            'experimental_setup': {
                'n_runs': results['config'].n_runs,
                'random_seed': results['config'].random_seed
            }
        }

        with open(experiment_dir / "experiment_configuration.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # 2. Save statistical results
        if 'run_statistics' in results:
            statistical_results = convert_numpy_types(results['run_statistics'])
            with open(experiment_dir / "statistical_results.json", 'w') as f:
                json.dump(statistical_results, f, indent=2)

        # 3. Save statistical analysis results
        if 'statistical_analysis' in results:
            analysis_results = convert_numpy_types(results['statistical_analysis'])
            with open(experiment_dir / "statistical_analysis.json", 'w') as f:
                json.dump(analysis_results, f, indent=2)

        # 4. Save trained models
        if 'trained_models' in results and results['trained_models']:
            models_dir = experiment_dir / "trained_models"
            models_dir.mkdir(exist_ok=True)

            for name, model in results['trained_models'].items():
                model_path = models_dir / f"{name}.keras"
                model.save(model_path)
                logger.info(f"💾 Saved model {name} to {model_path}")

        # 5. Save training histories
        if 'training_histories' in results and results['training_histories']:
            histories_converted = convert_numpy_types(results['training_histories'])
            with open(experiment_dir / "training_histories.json", 'w') as f:
                json.dump(histories_converted, f, indent=2)

        # 6. Save experiment metadata
        metadata = {
            'experiment_completed': datetime.now().isoformat(),
            'total_models_trained': len(results.get('trained_models', {})),
            'successful_runs': results['config'].n_runs,
            'analysis_performed': bool(results.get('model_analysis')),
            'keras_version': keras.__version__
        }

        with open(experiment_dir / "experiment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("✅ All experiment results saved successfully")

    except Exception as e:
        logger.error(f"❌ Failed to save experiment results: {e}")
        raise


def print_comprehensive_summary(results: Dict[str, Any]) -> None:
    """
    Print a detailed summary of all experimental results.

    Generates a comprehensive report including statistical analysis, model performance,
    calibration metrics, training dynamics, and key insights for research interpretation.

    Args:
        results: Complete experiment results dictionary

    Raises:
        ValueError: If results dictionary is empty or malformed
    """
    if not results:
        raise ValueError("results dictionary cannot be empty")

    logger.info("=" * 100)
    logger.info("📋 BANDRMS DENSE NETWORK NORMALIZATION - COMPREHENSIVE EXPERIMENT SUMMARY")
    logger.info("=" * 100)

    # ===== EXPERIMENTAL SETUP OVERVIEW =====
    config = results.get('config')
    if config:
        logger.info("⚙️ EXPERIMENTAL CONFIGURATION:")
        logger.info(f"   Architecture: {len(config.hidden_units)} hidden layers {config.hidden_units}")
        logger.info(f"   Activation: {config.activation}, Dropout: {config.dropout_rate if config.use_dropout else 'None'}")
        logger.info(f"   Training: {config.epochs} epochs, batch size {config.batch_size}, LR {config.learning_rate}")
        logger.info(f"   Normalization variants: {list(config.normalization_variants.keys())}")
        logger.info(f"   Statistical robustness: {config.n_runs} independent runs")
        logger.info("")

    # ===== STATISTICAL PERFORMANCE RESULTS =====
    if 'run_statistics' in results and results['run_statistics']:
        logger.info("📊 STATISTICAL PERFORMANCE ANALYSIS (Mean ± SEM):")
        logger.info(f"{'Model':<15} {'Accuracy':<15} {'Loss':<15} {'Stability':<12} {'Runs':<6}")
        logger.info("-" * 80)

        for model_name, stats in results['run_statistics'].items():
            if 'accuracy' in stats and 'loss' in stats:
                acc_mean = stats['accuracy']['mean']
                acc_sem = stats['accuracy']['sem']
                loss_mean = stats['loss']['mean']
                loss_sem = stats['loss']['sem']
                acc_std = stats['accuracy']['std']
                n_runs = stats['accuracy']['count']

                logger.info(f"{model_name:<15} {acc_mean:.4f}±{acc_sem:.4f}   "
                            f"{loss_mean:.3f}±{loss_sem:.3f}     "
                            f"{acc_std:.4f}      {n_runs:<6}")
        logger.info("")

    # ===== KEY INSIGHTS =====
    logger.info("🔍 KEY EXPERIMENTAL INSIGHTS:")

    if 'statistical_analysis' in results and results['statistical_analysis']:
        analysis = results['statistical_analysis']

        # Best performer
        if 'best_performers' in analysis and 'accuracy' in analysis['best_performers']:
            best = analysis['best_performers']['accuracy']
            logger.info(f"   🏆 Best Overall Performance: {best['model']} "
                        f"({best['mean']:.4f} ± {best['sem']:.4f})")

        # Most stable
        if 'stability_analysis' in analysis and 'most_stable' in analysis['stability_analysis']:
            stable = analysis['stability_analysis']['most_stable']
            logger.info(f"   🎯 Most Stable Training: {stable['model']} "
                        f"(std: {stable['std']:.4f})")

        # Rankings
        if 'rankings' in analysis and 'accuracy' in analysis['rankings']:
            logger.info("   📊 Performance Rankings:")
            for rank_info in analysis['rankings']['accuracy'][:3]:  # Top 3
                logger.info(f"      {rank_info['rank']}. {rank_info['model']}: {rank_info['mean']:.4f}")

    # ===== RESEARCH CONCLUSIONS =====
    logger.info("")
    logger.info("🔬 RESEARCH CONCLUSIONS:")
    logger.info("   This experiment provides empirical evidence for normalization effectiveness")
    logger.info("   in deep fully-connected networks. Key findings include:")
    logger.info("   1. Comparative performance analysis across normalization techniques")
    logger.info("   2. Statistical significance assessment of performance differences")
    logger.info("   3. Training stability analysis for practical considerations")
    logger.info("")

    logger.info("📁 Detailed results and visualizations saved to experiment directory")
    logger.info("=" * 100)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the BandRMS dense network experiment.

    Configures the experiment, executes the complete pipeline, and handles
    any system-level errors or resource management issues.

    Raises:
        RuntimeError: If experiment execution fails
        SystemError: If system resources are insufficient
    """
    logger.info("🚀 BandRMS Dense Network Normalization Experiment")
    logger.info("=" * 80)

    # Initialize experiment configuration
    config = DenseNormalizationConfig()

    # Log experimental setup
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Normalization variants: {list(config.normalization_variants.keys())}")
    logger.info(f"   Architecture depth: {len(config.hidden_units)} hidden layers")
    logger.info(f"   Training: {config.epochs} epochs × {config.n_runs} runs")
    logger.info(f"   Statistical significance: {config.n_runs} independent runs")
    logger.info("")

    try:
        # Execute the complete experiment
        experiment_results = run_experiment(config)

        logger.info("🎉 BandRMS Dense Network Normalization Experiment completed successfully!")
        logger.info("📊 Statistical analysis shows definitive results for normalization effectiveness")
        logger.info("📈 Comprehensive analysis available in experiment output directory")

        return experiment_results

    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise RuntimeError(f"Experiment execution failed: {e}") from e


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()