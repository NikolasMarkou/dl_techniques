"""
Experiment Title: BandRMS Normalization Effectiveness Study on CIFAR-10
========================================================================

This experiment conducts a comprehensive evaluation of BandRMS normalization compared to
traditional normalization techniques (RMSNorm, LayerNorm, BatchNorm) on CIFAR-10 image
classification. The study addresses fundamental questions about constrained normalization:
Does limiting the scaling factor to a bounded range [1-α, 1] improve training stability,
model calibration, and robustness while maintaining competitive performance?

The hypothesis is that BandRMS creates a "thick shell" in feature space that provides
better training dynamics and calibration compared to unbounded normalization methods,
at the potential cost of some expressiveness. This geometric constraint is expected to
be particularly beneficial for deep networks and scenarios requiring reliable confidence
estimates.

Scientific Motivation
--------------------

Modern deep learning relies heavily on normalization techniques to enable stable training
of deep networks. While LayerNorm and BatchNorm have been successful, they can suffer from:
1. Unstable training dynamics in deep networks
2. Poor calibration (overconfident predictions)
3. Sensitivity to hyperparameters and initialization

BandRMS proposes a novel approach: after RMS normalization (which projects features to
unit norm), apply a learnable but bounded scaling factor. This creates a "thick shell"
in high-dimensional feature space where features can exist within [1-α, 1] rather than
being confined to the unit hypersphere or having unbounded scaling.

Theoretical Foundation:
- High-dimensional geometry: Volume concentrates near sphere surface
- Bounded scaling adds degrees of freedom while maintaining geometric constraints
- Sigmoid activation on scaling parameter provides natural regularization
- Should improve calibration by preventing extreme confidence values

Experimental Design
-------------------

**Dataset**: CIFAR-10 (32×32 RGB images, 10 classes)
- 50,000 training images, 10,000 test images
- Standard preprocessing with normalization
- Balanced class distribution for fair calibration analysis

**Model Architecture**: ResNet-inspired CNN with normalization layers
- 4 convolutional blocks with progressive filter scaling [32, 64, 128, 256]
- Residual connections for deep network training
- Global average pooling followed by dense classification
- Identical architecture across all variants (only normalization changes)

**Normalization Techniques Evaluated**:

1. **BandRMS_010**: BandRMS with α=0.1 (tight constraints, high stability)
2. **BandRMS_020**: BandRMS with α=0.2 (moderate constraints)
3. **BandRMS_030**: BandRMS with α=0.3 (loose constraints, more expressiveness)
4. **RMSNorm**: Standard RMS normalization (unconstrained scaling)
5. **LayerNorm**: Layer normalization (mean-variance normalization)
6. **BatchNorm**: Batch normalization (baseline comparison)

**Experimental Variables**:
- **Primary**: Normalization technique (6 variants)
- **Secondary**: BandRMS α parameter (3 values: 0.1, 0.2, 0.3)
- **Control**: Architecture, training procedure, dataset, random seeds

Comprehensive Analysis Pipeline
------------------------------

**Training Dynamics Analysis**:
- Convergence speed and stability metrics
- Loss curve smoothness and early stopping behavior
- Gradient flow characteristics through the network
- Training vs validation performance gap analysis

**Model Performance Evaluation**:
- Test accuracy and top-k accuracy
- Statistical significance testing across multiple runs
- Per-class performance analysis
- Robustness to hyperparameter variations

**Calibration Analysis** (via ModelAnalyzer):
- Expected Calibration Error (ECE) with multiple bin sizes
- Brier score for probabilistic prediction quality
- Reliability diagrams and confidence distribution analysis
- Temperature scaling effectiveness for post-hoc calibration

**Weight and Architecture Analysis**:
- Weight distribution evolution across layers
- Effective scaling factor analysis for BandRMS variants
- Dead neuron detection and activation sparsity
- Layer-wise norm analysis (L1, L2, spectral)

**Information Flow Analysis**:
- Activation pattern analysis across the network
- Feature representation quality metrics
- Information bottleneck characteristics
- Gradient flow stability measures

**Visualization and Reporting**:
- Training dynamics comparison plots
- Calibration reliability diagrams
- Weight distribution heatmaps
- Confusion matrices and error analysis
- Statistical significance tests

Expected Outcomes and Hypotheses
--------------------------------

**Primary Hypotheses**:
1. **Calibration**: BandRMS variants should show lower ECE and better reliability
2. **Stability**: More stable training with less variance across runs
3. **Robustness**: Better generalization with controlled overfitting
4. **Trade-offs**: α parameter should show stability vs expressiveness trade-off

**Scientific Contributions**:
1. Empirical validation of constrained normalization benefits
2. Optimal α parameter selection guidelines
3. Calibration improvements without architectural changes
4. Training stability analysis for deep networks

Usage Example
-------------

Basic usage with default configuration:

    ```python
    config = BandRMSExperimentConfig()
    results = run_experiment(config)

    # Access calibration results
    for model_name, metrics in results['model_analysis'].calibration_metrics.items():
        print(f"{model_name}: ECE={metrics['ece']:.4f}")
    ```

Advanced usage with custom parameters:

    ```python
    config = BandRMSExperimentConfig(
        epochs=100,
        batch_size=128,
        band_alphas=[0.05, 0.1, 0.15, 0.2],  # Custom α values
        calibration_bins=20,  # Fine-grained calibration analysis
        output_dir=Path("bandrms_study")
    )

    results = run_experiment(config)

    # Statistical analysis
    perform_statistical_tests(results)
    ```
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import json
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_cifar10
from dl_techniques.utils.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class BandRMSExperimentConfig:
    """
    Configuration for the BandRMS normalization effectiveness experiment.

    This class encapsulates all configurable parameters for systematically
    evaluating BandRMS against traditional normalization techniques.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_residual: bool = True
    activation: str = 'gelu'

    # --- Training Parameters ---
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 20
    monitor_metric: str = 'val_accuracy'

    # --- BandRMS Specific Parameters ---
    band_alphas: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    band_epsilon: float = 1e-7
    band_regularizer_strength: float = 1e-5

    # --- Normalization Techniques ---
    normalization_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'BRMS_005': lambda: ('band_rms', {'max_band_width': 0.05}),
        'BRMS_010': lambda: ('band_rms', {'max_band_width': 0.10}),
        'BRMS_020': lambda: ('band_rms', {'max_band_width': 0.20}),
        'RMSNorm': lambda: ('rms_norm', {}),
        'LayerNorm': lambda: ('layer_norm', {}),
        'BatchNorm': lambda: ('batch_norm', {}),
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "bandrms_normalization_study"
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
        show_statistical_tests=True,
        show_confidence_intervals=True,
        verbose=True
    ))

# ==============================================================================
# NORMALIZATION LAYER FACTORY
# ==============================================================================

def create_normalization_layer(
    norm_type: str,
    norm_params: Dict[str, Any],
    axis: int = -1,
    name: Optional[str] = None
) -> keras.layers.Layer:
    """
    Factory function to create different normalization layers.

    Args:
        norm_type: Type of normalization ('band_rms', 'rms_norm', 'layer_norm', 'batch_norm')
        norm_params: Parameters specific to the normalization type
        axis: Axis for normalization (default -1 for feature axis)
        name: Optional layer name

    Returns:
        Configured normalization layer
    """
    if norm_type == 'band_rms':
        return BandRMS(
            axis=axis,
            max_band_width=norm_params.get('max_band_width', 0.1),
            epsilon=norm_params.get('epsilon', 1e-7),
            band_regularizer=keras.regularizers.L2(norm_params.get('regularizer_strength', 1e-5)),
            name=name
        )
    elif norm_type == 'rms_norm':
        return RMSNorm(
            axis=axis,
            epsilon=norm_params.get('epsilon', 1e-6),
            name=name
        )
    elif norm_type == 'layer_norm':
        return keras.layers.LayerNormalization(
            axis=axis,
            epsilon=norm_params.get('epsilon', 1e-6),
            name=name
        )
    elif norm_type == 'batch_norm':
        return keras.layers.BatchNormalization(
            axis=axis,
            epsilon=norm_params.get('epsilon', 1e-3),
            name=name
        )
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

# ==============================================================================
# MODEL ARCHITECTURE BUILDING
# ==============================================================================

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    norm_type: str,
    norm_params: Dict[str, Any],
    config: BandRMSExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a residual block with specified normalization technique.

    Args:
        inputs: Input tensor
        filters: Number of filters
        norm_type: Type of normalization layer
        norm_params: Parameters for normalization layer
        config: Experiment configuration
        block_index: Index of the current block

    Returns:
        Output tensor after residual block
    """
    shortcut = inputs

    # First convolution
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)

    # Normalization
    x = create_normalization_layer(
        norm_type, norm_params, axis=-1, name=f'norm{block_index}_1'
    )(x)

    # Activation
    x = keras.layers.Activation(config.activation)(x)

    # Second convolution
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)

    # Normalization
    x = create_normalization_layer(
        norm_type, norm_params, axis=-1, name=f'norm{block_index}_2'
    )(x)

    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters, (1, 1), padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay)
        )(shortcut)

        shortcut = create_normalization_layer(
            norm_type, norm_params, axis=-1, name=f'norm{block_index}_shortcut'
        )(shortcut)

    # Add and activate
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation(config.activation)(x)

    return x

def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    norm_type: str,
    norm_params: Dict[str, Any],
    config: BandRMSExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a convolutional block with specified normalization.

    Args:
        inputs: Input tensor
        filters: Number of filters
        norm_type: Type of normalization layer
        norm_params: Parameters for normalization layer
        config: Experiment configuration
        block_index: Index of the current block

    Returns:
        Output tensor after convolutional block
    """
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, norm_type, norm_params, config, block_index)
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

        # Normalization
        x = create_normalization_layer(
            norm_type, norm_params, axis=-1, name=f'norm{block_index}'
        )(x)

        # Activation
        x = keras.layers.Activation(config.activation)(x)

    # Max pooling (except last block)
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D((2, 2))(x)

    # Dropout
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x

def build_model(
    config: BandRMSExperimentConfig,
    norm_type: str,
    norm_params: Dict[str, Any],
    name: str
) -> keras.Model:
    """
    Build a complete CNN model with specified normalization technique.

    Args:
        config: Experiment configuration
        norm_type: Type of normalization layer
        norm_params: Parameters for normalization layer
        name: Model name

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Initial convolution with normalization
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=(5, 5),
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem_conv'
    )(x)

    x = create_normalization_layer(
        norm_type, norm_params, axis=-1, name='stem_norm'
    )(x)

    x = keras.layers.Activation(config.activation)(x)

    # Convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, norm_type, norm_params, config, i)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers with normalization
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)

        # Apply normalization to dense layers too
        x = create_normalization_layer(
            norm_type, norm_params, axis=-1, name=f'dense_norm_{j}'
        )(x)

        x = keras.layers.Activation(config.activation)(x)

        # Dropout for dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer (no normalization here)
    outputs = keras.layers.Dense(
        config.num_classes, activation='softmax',
        kernel_initializer=config.kernel_initializer,
        name='predictions'
    )(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')

    # Check data format and compile accordingly
    optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate)

    # The model should always use categorical_crossentropy with softmax output
    # We'll handle label format conversion during training/evaluation
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
# STATISTICAL ANALYSIS UTILITIES
# ==============================================================================

def calculate_run_statistics(results_per_run: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics across multiple runs for each model.

    Args:
        results_per_run: Dictionary mapping model names to lists of results across runs

    Returns:
        Dictionary with mean, std, min, max for each model and metric
    """
    statistics = {}

    for model_name, run_results in results_per_run.items():
        if not run_results:
            continue

        # Initialize statistics for this model
        statistics[model_name] = {}

        # Get all metrics from first run
        metrics = run_results[0].keys()

        for metric in metrics:
            values = [result[metric] for result in run_results if metric in result]

            if values:
                statistics[model_name][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

    return statistics

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: BandRMSExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete BandRMS normalization comparison experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experimental results
    """
    # Set random seed
    keras.utils.set_random_seed(config.random_seed)

    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    logger.info("🚀 Starting BandRMS Normalization Effectiveness Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("✅ Dataset loaded successfully")

    # Debug information about data format
    logger.info(f"📋 Dataset Info:")
    logger.info(f"   Training data shape: {cifar10_data.x_train.shape}")
    logger.info(f"   Training labels shape: {cifar10_data.y_train.shape}")
    logger.info(f"   Test data shape: {cifar10_data.x_test.shape}")
    logger.info(f"   Test labels shape: {cifar10_data.y_test.shape}")

    # Check if labels are one-hot encoded
    if len(cifar10_data.y_test.shape) > 1 and cifar10_data.y_test.shape[1] > 1:
        logger.info(f"   Labels are one-hot encoded with {cifar10_data.y_test.shape[1]} classes")
        logger.info(f"   Sample label: {cifar10_data.y_test[0]}")
    else:
        logger.info(f"   Labels are integer encoded")
        logger.info(f"   Sample labels: {cifar10_data.y_test[:5]}")
        logger.info(f"   Unique labels: {np.unique(cifar10_data.y_test)}")

    # ===== MULTIPLE RUNS FOR STATISTICAL SIGNIFICANCE =====
    logger.info(f"🔄 Running {config.n_runs} repetitions for statistical significance...")

    all_trained_models = {}  # Final models from all runs
    all_histories = {}  # Training histories from all runs
    results_per_run = {}  # Performance results per run

    for run_idx in range(config.n_runs):
        logger.info(f"🏃 Starting run {run_idx + 1}/{config.n_runs}")

        # Set different seed for each run
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        run_models = {}
        run_histories = {}

        # ===== TRAINING MODELS FOR THIS RUN =====
        for norm_name, norm_factory in config.normalization_variants.items():
            logger.info(f"--- Training {norm_name} (Run {run_idx + 1}) ---")

            # Get normalization configuration
            norm_type, norm_params = norm_factory()

            # Build model
            model = build_model(config, norm_type, norm_params, f"{norm_name}_run{run_idx}")

            # Log model info
            if run_idx == 0:  # Only log architecture details for first run
                model.summary(print_fn=logger.info)
                logger.info(f"Model {norm_name} parameters: {model.count_params():,}")

            # Configure training
            training_config = TrainingConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
                monitor_metric=config.monitor_metric,
                model_name=f"{norm_name}_run{run_idx}",
                output_dir=experiment_dir / "training_plots" / f"run_{run_idx}" / norm_name
            )

            # Train model
            history = train_model(
                model, cifar10_data.x_train, cifar10_data.y_train,
                cifar10_data.x_test, cifar10_data.y_test, training_config
            )

            run_models[norm_name] = model
            run_histories[norm_name] = history.history

            logger.info(f"✅ {norm_name} (Run {run_idx + 1}) training completed!")

        # ===== EVALUATE MODELS FOR THIS RUN =====
        logger.info(f"📊 Evaluating models for run {run_idx + 1}...")

        for norm_name, model in run_models.items():
            try:
                # Get predictions for manual accuracy calculation
                predictions = model.predict(cifar10_data.x_test, verbose=0)

                # Handle label format - ensure we have proper format for evaluation
                if len(cifar10_data.y_test.shape) > 1 and cifar10_data.y_test.shape[1] > 1:
                    # Labels are one-hot encoded
                    y_true_classes = np.argmax(cifar10_data.y_test, axis=1)
                else:
                    # Labels are integer encoded
                    y_true_classes = cifar10_data.y_test.astype(int)

                # Calculate accuracy manually
                y_pred_classes = np.argmax(predictions, axis=1)
                manual_accuracy = np.mean(y_pred_classes == y_true_classes)

                # Calculate top-5 accuracy
                top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]
                manual_top5_acc = np.mean([
                    y_true in top5_pred
                    for y_true, top5_pred in zip(y_true_classes, top_5_predictions)
                ])

                # Try model.evaluate for loss (but use manual accuracy)
                try:
                    eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
                    metrics_dict = dict(zip(model.metrics_names, eval_results))
                    loss_value = metrics_dict.get('loss', 0.0)
                except Exception as eval_error:
                    logger.warning(f"⚠️ Model evaluation failed for {norm_name}, calculating loss manually: {eval_error}")
                    # Calculate loss manually - ensure labels are one-hot for categorical_crossentropy
                    if len(cifar10_data.y_test.shape) == 1:
                        # Convert integer labels to one-hot
                        y_test_onehot = keras.utils.to_categorical(cifar10_data.y_test, num_classes=10)
                    else:
                        y_test_onehot = cifar10_data.y_test

                    loss_value = float(keras.metrics.categorical_crossentropy(y_test_onehot, predictions).numpy().mean())

                # Store results with manual accuracy
                if norm_name not in results_per_run:
                    results_per_run[norm_name] = []

                results_per_run[norm_name].append({
                    'accuracy': manual_accuracy,
                    'top_5_accuracy': manual_top5_acc,
                    'loss': loss_value,
                    'run_idx': run_idx
                })

                logger.info(f"✅ {norm_name} (Run {run_idx + 1}): Accuracy={manual_accuracy:.4f}, Loss={loss_value:.4f}")

            except Exception as e:
                logger.error(f"❌ Error evaluating {norm_name} (Run {run_idx + 1}): {e}", exc_info=True)

        # Store models and histories from last run for analysis
        if run_idx == config.n_runs - 1:
            all_trained_models = run_models
            all_histories = run_histories

        # Memory cleanup
        del run_models
        gc.collect()

    # ===== STATISTICAL ANALYSIS =====
    logger.info("📈 Calculating statistics across runs...")
    run_statistics = calculate_run_statistics(results_per_run)

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("🔬 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        analyzer = ModelAnalyzer(
            models=all_trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis",
            training_history=all_histories
        )

        model_analysis_results = analyzer.analyze(data=DataInput.from_object(cifar10_data))
        logger.info("✅ Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    # ===== VISUALIZATION GENERATION =====
    logger.info("📊 Generating visualizations...")

    # Training history comparison (from last run)
    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='normalization_training_comparison',
        subdir='training_plots',
        title='Normalization Techniques Training Comparison'
    )

    # Statistical comparison visualization
    create_statistical_comparison_plot(run_statistics, experiment_dir / "visualizations")

    # Confusion matrices
    raw_predictions = {
        name: model.predict(cifar10_data.x_test, verbose=0)
        for name, model in all_trained_models.items()
    }
    class_predictions = {
        name: np.argmax(preds, axis=1)
        for name, preds in raw_predictions.items()
    }

    y_true_classes = (np.argmax(cifar10_data.y_test, axis=1)
                     if len(cifar10_data.y_test.shape) > 1
                     else cifar10_data.y_test)

    vis_manager.plot_confusion_matrices_comparison(
        y_true=y_true_classes,
        model_predictions=class_predictions,
        name='normalization_confusion_matrices',
        subdir='model_comparison',
        normalize=True,
        class_names=[str(i) for i in range(10)]
    )

    # ===== RESULTS COMPILATION =====
    results = {
        'run_statistics': run_statistics,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'trained_models': all_trained_models,
        'config': config,
        'experiment_dir': experiment_dir
    }

    # Save results
    save_experiment_results(results, experiment_dir)

    # Print comprehensive summary
    print_experiment_summary(results)

    return results

# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def create_statistical_comparison_plot(statistics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """
    Create statistical comparison plots showing mean ± std across runs.

    Args:
        statistics: Statistical results from multiple runs
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Setup plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Extract data
        models = list(statistics.keys())
        accuracies = [statistics[model]['accuracy']['mean'] for model in models]
        accuracy_stds = [statistics[model]['accuracy']['std'] for model in models]
        eces = []
        ece_stds = []

        # Plot accuracy comparison
        ax1 = axes[0]
        bars = ax1.bar(models, accuracies, yerr=accuracy_stds, capsize=5)
        ax1.set_title('Test Accuracy Comparison (Mean ± Std)')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, acc, std in zip(bars, accuracies, accuracy_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                    f'{acc:.3f}±{std:.3f}', ha='center', va='bottom')

        # Plot loss comparison
        ax2 = axes[1]
        losses = [statistics[model]['loss']['mean'] for model in models]
        loss_stds = [statistics[model]['loss']['std'] for model in models]
        bars2 = ax2.bar(models, losses, yerr=loss_stds, capsize=5)
        ax2.set_title('Test Loss Comparison (Mean ± Std)')
        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, loss, std in zip(bars2, losses, loss_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{loss:.3f}±{std:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✅ Statistical comparison plot saved")

    except Exception as e:
        logger.error(f"❌ Failed to create statistical comparison plot: {e}")

# ==============================================================================
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save experiment results in multiple formats.

    Args:
        results: Experiment results dictionary
        experiment_dir: Directory to save results
    """
    try:
        # Helper function to convert numpy types to Python native types
        def convert_numpy_to_python(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_to_python(item) for item in obj)
            else:
                return obj

        # Save configuration
        config_dict = {
            'experiment_name': results['config'].experiment_name,
            'normalization_variants': list(results['config'].normalization_variants.keys()),
            'epochs': results['config'].epochs,
            'batch_size': results['config'].batch_size,
            'learning_rate': results['config'].learning_rate,
            'n_runs': results['config'].n_runs,
            'random_seed': results['config'].random_seed,
            'architecture': {
                'conv_filters': results['config'].conv_filters,
                'dense_units': results['config'].dense_units,
                'use_residual': results['config'].use_residual
            }
        }

        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Convert and save statistical results
        statistical_results_converted = convert_numpy_to_python(results['run_statistics'])
        with open(experiment_dir / "statistical_results.json", 'w') as f:
            json.dump(statistical_results_converted, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for name, model in results['trained_models'].items():
            model_path = models_dir / f"{name}.keras"
            model.save(model_path)

        logger.info("✅ Experiment results saved successfully")

    except Exception as e:
        logger.error(f"❌ Failed to save experiment results: {e}", exc_info=True)

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary
    """
    logger.info("=" * 80)
    logger.info("📋 BANDRMS NORMALIZATION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== STATISTICAL RESULTS =====
    if 'run_statistics' in results:
        logger.info("📊 STATISTICAL RESULTS (Mean ± Std across runs):")
        logger.info(f"{'Model':<15} {'Accuracy':<15} {'Loss':<15} {'Runs':<8}")
        logger.info("-" * 60)

        for model_name, stats in results['run_statistics'].items():
            acc_mean = stats['accuracy']['mean']
            acc_std = stats['accuracy']['std']
            loss_mean = stats['loss']['mean']
            loss_std = stats['loss']['std']
            n_runs = stats['accuracy']['count']

            logger.info(f"{model_name:<15} {acc_mean:.3f}±{acc_std:.3f}    "
                       f"{loss_mean:.3f}±{loss_std:.3f}    {n_runs:<8}")

    # ===== CALIBRATION RESULTS =====
    if results.get('model_analysis') and results['model_analysis'].calibration_metrics:
        logger.info("🎯 CALIBRATION ANALYSIS:")
        logger.info(f"{'Model':<15} {'ECE':<12} {'Brier':<12} {'Entropy':<12}")
        logger.info("-" * 55)

        for model_name, cal_metrics in results['model_analysis'].calibration_metrics.items():
            ece = cal_metrics['ece']
            brier = cal_metrics['brier_score']
            entropy = cal_metrics['mean_entropy']

            logger.info(f"{model_name:<15} {ece:<12.4f} {brier:<12.4f} {entropy:<12.4f}")

    # ===== KEY INSIGHTS =====
    logger.info("🔍 KEY INSIGHTS:")

    if 'run_statistics' in results:
        # Find best performing model
        best_model = max(results['run_statistics'].items(),
                        key=lambda x: x[1]['accuracy']['mean'])
        logger.info(f"   🏆 Best Accuracy: {best_model[0]} ({best_model[1]['accuracy']['mean']:.4f})")

        # Find most stable model (lowest std)
        most_stable = min(results['run_statistics'].items(),
                         key=lambda x: x[1]['accuracy']['std'])
        logger.info(f"   🎯 Most Stable: {most_stable[0]} (std: {most_stable[1]['accuracy']['std']:.4f})")

        # Compare BandRMS variants
        band_rms_models = {k: v for k, v in results['run_statistics'].items() if k.startswith('BandRMS')}
        if band_rms_models:
            logger.info("   📐 BandRMS Analysis:")
            for model_name, stats in band_rms_models.items():
                # Extract alpha value from model name (e.g., "BandRMS_010" -> "0.1")
                alpha_str = model_name.split('_')[1]  # "010"
                alpha_value = float(alpha_str) / 100  # Convert "010" to 0.1
                logger.info(f"      α={alpha_value:.1f}: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")

    logger.info("=" * 80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the BandRMS normalization experiment.
    """
    logger.info("🚀 BandRMS Normalization Effectiveness Experiment")
    logger.info("=" * 80)

    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU configuration: {e}")

    # Initialize configuration
    config = BandRMSExperimentConfig()

    # Log configuration
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Normalization variants: {list(config.normalization_variants.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info(f"   Architecture: {len(config.conv_filters)} conv blocks, {len(config.dense_units)} dense layers")
    logger.info(f"   BandRMS α values: {config.band_alphas}")
    logger.info("")

    try:
        # Run experiment
        results = run_experiment(config)
        logger.info("✅ BandRMS normalization experiment completed successfully!")

        # Additional analysis could go here

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        raise

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()