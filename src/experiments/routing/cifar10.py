"""
CIFAR-10 Output Layer Comparison: Hierarchical Routing vs. Softmax
===================================================================

This experiment evaluates the performance and characteristics of the novel
`HierarchicalRoutingLayer` against the standard `Dense` -> `Softmax`
classifier for image classification on the CIFAR-10 dataset.

The study aims to answer a critical question for large-scale classification:
can we replace the computationally expensive softmax layer with a more
efficient alternative without sacrificing accuracy? By directly comparing
these two output layers on an identical base architecture, we can isolate
their effects on training dynamics, final performance, and prediction quality.

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32×32 RGB images)
- 50,000 training images
- 10,000 test images
- Standard preprocessing with normalization

**Model Architecture**: A consistent ResNet-inspired CNN is used for all models,
with only the final output layer differing. The base architecture includes:
- Initial convolutional layer (32 filters)
- 6 convolutional blocks with residual connections
- Progressive filter scaling: [32, 64, 128, 256, 256, 256]
- Batch normalization and dropout regularization
- Global average pooling
- Layer normalization
- Dense classification layers with L2 regularization

**Output Layers Evaluated**:

1. **Standard Softmax**: The baseline approach. A `Dense` layer produces logits,
   followed by a `Softmax` activation to generate a probability distribution.
   Complexity: O(N), where N is the number of classes.

2. **Hierarchical Routing**: A probabilistic binary tree approach. The
   `HierarchicalRoutingLayer` directly produces a probability distribution.
   Complexity: O(log₂N), offering significant computational advantages for
   large N.

3. **Routing Probabilities**: An alternative routing-based approach using
   the `RoutingProbabilitiesLayer` on top of dense logits.

Comprehensive Analysis Pipeline
------------------------------

The experiment employs a multi-faceted analysis approach using ModelAnalyzer:

**Training Analysis**:
- Training and validation curves for accuracy and loss
- Convergence behavior and stability assessment
- Early stopping based on validation accuracy
- Overfitting index and training stability metrics

**Model Performance Evaluation**:
- Test set accuracy and top-k accuracy
- Final loss values
- Statistical significance testing (if multiple runs were performed)

**Calibration and Prediction Analysis** (via ModelAnalyzer):
- Expected Calibration Error (ECE) to measure prediction confidence
- Brier score for probabilistic prediction quality
- Reliability diagrams and calibration plots
- Entropy analysis of the output probability distributions
- Per-class calibration analysis

**Weight and Activation Analysis**:
- Layer-wise weight distribution statistics
- Weight health scores and evolution
- Activation pattern analysis across the network
- Information flow characteristics
- Feature specialization metrics

**Spectral Analysis (WeightWatcher)**:
- Power-law exponent (α) for training quality assessment
- Concentration scores for information distribution
- Matrix entropy and stable rank metrics
- Data-free generalization estimates

**Visual Analysis**:
- Training history comparison plots
- Confusion matrices for each output layer
- Calibration and reliability diagrams
- Weight distribution visualizations
- Comprehensive summary dashboard

Expected Outcomes and Insights
------------------------------

This experiment is designed to reveal:

1. **Performance Trade-offs**: Does the computational efficiency of the
   `HierarchicalRoutingLayer` come at the cost of classification accuracy?

2. **Training Dynamics**: How does the routing-based learning process affect
   convergence speed and stability compared to the standard softmax?

3. **Prediction Quality**: Do the different layers produce differently calibrated
   probability distributions? We will investigate if one is inherently more
   or less confident in its predictions.

4. **Weight Characteristics**: How do the learned weight distributions differ
   between the routing approaches and the standard softmax?

5. **Scalability Implications**: While CIFAR-10 has only 10 classes, this
   experiment provides a crucial proof-of-concept for applying the
   `HierarchicalRoutingLayer` to problems with much larger output spaces,
   such as large-vocabulary language models or fine-grained classification.

Theoretical Foundation
---------------------

This experiment explores the practical implications of replacing the traditional
softmax output with hierarchical routing mechanisms, which can offer:
- Reduced computational complexity (O(log N) vs O(N))
- Potentially different inductive biases in the learned representations
- Alternative paths for gradient flow during backpropagation
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

# ==============================================================================
# LOCAL IMPORTS
# ==============================================================================

from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.layers.hierarchical_routing import HierarchicalRoutingLayer
from dl_techniques.layers.activations.routing_probabilities import RoutingProbabilitiesLayer

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    ClassificationResults,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    MultiModelClassification
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
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for the Hierarchical Routing vs. Softmax experiment.

    This class encapsulates all configurable parameters, including dataset
    info, model architecture, training settings, and analysis options.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 256, 256])
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
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
    early_stopping_patience: int = 30
    monitor_metric: str = 'val_accuracy'
    loss_function: Callable = field(default_factory=lambda: keras.losses.CategoricalCrossentropy(from_logits=False))

    # --- Models to Evaluate ---
    model_types: List[str] = field(default_factory=lambda: [
        'PlainRoutingProbabilities', # Routing
        'RoutingProbabilities',      # Dense -> Routing
        'HierarchicalRouting',       # HierarchicalRouting
        'Softmax',                   # Dense -> Softmax
    ])

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_routing_comparison"
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
        n_samples=1000,

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

    Args:
        inputs: Input tensor to the residual block
        filters: Number of filters in the convolutional layers
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming layers)

    Returns:
        Output tensor after applying the residual block
    """
    shortcut = inputs

    x = keras.layers.Conv2D(
        filters, config.kernel_size, padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

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
            filters=filters, kernel_size=(1, 1), padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay)
        )(shortcut)
        if config.use_batch_norm:
            shortcut = keras.layers.BatchNormalization()(shortcut)

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
    Build a convolutional block, optionally with residual connections.

    Args:
        inputs: Input tensor to the convolutional block
        filters: Number of filters in the convolutional layers
        config: Experiment configuration containing architecture parameters
        block_index: Index of the current block (for naming and logic)

    Returns:
        Output tensor after applying the convolutional block
    """
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
    else:
        x = keras.layers.Conv2D(
            filters=filters, kernel_size=config.kernel_size, padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{block_index}'
        )(inputs)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

    # Apply dropout if specified for this layer
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(config: ExperimentConfig, model_type: str, name: str) -> keras.Model:
    """
    Build a complete CNN model with a specified output layer type.

    This function constructs a ResNet-inspired CNN. The final layer is determined
    by the `model_type` parameter, allowing for a direct comparison between
    a standard Softmax and alternative routing-based output layers.

    Args:
        config: Experiment configuration object.
        model_type: The type of output layer ('Softmax', 'HierarchicalRouting',
            or 'RoutingProbabilities').
        name: Name prefix for the model and its layers.

    Returns:
        A compiled Keras model ready for training.

    Raises:
        ValueError: If an unknown model_type is specified.
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # === Feature Extractor Backbone (Identical for All Models) ===
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0], kernel_size=(4, 4), strides=(2, 2),
        padding='same', kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

    # Global pooling and normalization
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.LayerNormalization()(x)

    # === Interchangeable Output Layer ===
    if model_type == 'Softmax':
        # Standard softmax output
        logits = keras.layers.Dense(
            units=config.num_classes,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name='logits'
        )(x)
        predictions = keras.layers.Activation('softmax', name='predictions')(logits)

    elif model_type == 'HierarchicalRouting':
        # Hierarchical routing layer (O(log N) complexity)
        predictions = HierarchicalRoutingLayer(
            output_dim=config.num_classes,
            name='predictions'
        )(x)

    elif model_type == 'RoutingProbabilities':
        # Alternative routing approach
        logits = keras.layers.Dense(
            units=config.num_classes,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name='logits'
        )(x)
        predictions = RoutingProbabilitiesLayer(name='predictions')(logits)

    elif model_type == 'PlainRoutingProbabilities':
        predictions = (
            RoutingProbabilitiesLayer(
                output_dim=config.num_classes,
                name='predictions')(x)
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choose from {config.model_types}")

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=predictions, name=f'{name}_model')

    optimizer = keras.optimizers.AdamW(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=config.loss_function,
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    return model


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete output layer comparison experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing
    2. Model training for each output layer type
    3. Comprehensive model analysis using ModelAnalyzer
    4. Visualization generation
    5. Results compilation and reporting

    Args:
        config: Experiment configuration object.

    Returns:
        Dictionary containing all experimental results and analysis.
    """
    # Set random seed for reproducibility
    keras.utils.set_random_seed(config.random_seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager (for confusion matrices)
    vis_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations"
    )
    vis_manager.register_template("training_curves", TrainingCurvesVisualization)
    vis_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    # Log experiment start
    logger.info("=" * 80)
    logger.info("CIFAR-10 Output Layer Comparison Experiment")
    logger.info("Hierarchical Routing vs. Softmax")
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

    for model_type in config.model_types:
        logger.info(f"\n--- Training model with {model_type} output layer ---")

        # Build model
        model = build_model(config, model_type, model_type)

        # Log model info
        logger.info(f"Model parameters: {model.count_params():,}")
        model.summary(print_fn=logger.info)

        # Configure training
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=model_type,
            output_dir=experiment_dir / "training_plots" / model_type
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
        # IMPORTANT: Store the .history attribute (Dict[str, List[float]])
        # This is the correct format for ModelAnalyzer
        trained_models[model_type] = model
        training_histories[model_type] = history.history

        logger.info(f"{model_type} training completed successfully!")

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
        model_analysis_results = analyzer.analyze(data=test_data_input)

        logger.info("=" * 80)
        logger.info("ModelAnalyzer completed successfully!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Generated analysis outputs:")
        logger.info("  - summary_dashboard.png: High-level overview")
        logger.info("  - spectral_summary.png: Training quality (power-law α)")
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

    # ===== ADDITIONAL VISUALIZATION: TRAINING CURVES =====
    logger.info("=" * 80)
    logger.info("GENERATING TRAINING CURVE VISUALIZATIONS")
    logger.info("=" * 80)
    logger.info("")

    # Convert training histories to TrainingHistory objects
    training_history_objects = {}
    for name, hist_dict in training_histories.items():
        if len(hist_dict.get('loss', [])) > 0:
            training_history_objects[name] = TrainingHistory(
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
    if training_history_objects:
        try:
            vis_manager.visualize(
                data=training_history_objects,
                plugin_name="training_curves",
                metrics_to_plot=['accuracy', 'loss'],
                show=False
            )
            logger.info("Training history visualization created")
            logger.info("")
        except Exception as e:
            logger.error(f"Failed to create training history visualization: {e}")
            logger.info("")

    # ===== ADDITIONAL VISUALIZATION: CONFUSION MATRICES =====
    logger.info("=" * 80)
    logger.info("GENERATING CONFUSION MATRICES")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Generate predictions for all models
        raw_predictions = {
            name: model.predict(cifar10_data.x_test, verbose=0)
            for name, model in trained_models.items()
        }
        class_predictions = {
            name: np.argmax(preds, axis=1)
            for name, preds in raw_predictions.items()
        }

        # Convert y_test to class indices
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
            except Exception as e:
                logger.error(f"Failed to prepare classification results for {model_name}: {e}")

        # Create multi-model visualization
        if all_classification_results:
            multi_model_data = MultiModelClassification(
                results=all_classification_results,
                dataset_name="CIFAR-10"
            )

            vis_manager.visualize(
                data=multi_model_data,
                plugin_name="confusion_matrix",
                normalize='true',
                show=False
            )
            logger.info("Multi-model confusion matrix visualization created")
            logger.info("")

    except Exception as e:
        logger.error(f"Failed to create confusion matrices: {e}")
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
            'loss': metrics_dict.get('loss', 0.0)
        }

        # Warn about low accuracy
        final_accuracy = performance_results[name]['accuracy']
        if final_accuracy < 0.2:
            logger.warning(f"Low accuracy detected for {name}: {final_accuracy:.4f}")
            logger.warning("This may indicate training issues")

        logger.info(f"  Accuracy: {performance_results[name]['accuracy']:.4f}")
        logger.info(f"  Top-5 Accuracy: {performance_results[name]['top_5_accuracy']:.4f}")
        logger.info(f"  Loss: {performance_results[name]['loss']:.4f}")
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
        logger.info(f"{'Model':<30} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12}")
        logger.info("-" * 80)

        for model_name, metrics in results['performance_analysis'].items():
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(
                f"{model_name:<30} {accuracy:<12.4f} {top5_acc:<12.4f} {loss:<12.4f}"
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
                # Calculate mean confidence from the max_probability array
                max_probs = conf_metrics.get('max_probability', np.array([0.0]))
                mean_confidence = np.mean(max_probs) if max_probs.size > 0 else 0.0

                logger.info(
                    f"{model_name:<30} "
                    f"{cal_metrics.get('ece', 0.0):<12.4f} "
                    f"{cal_metrics.get('brier_score', 0.0):<15.4f} "
                    f"{mean_confidence:<12.4f} "  # Use calculated mean confidence
                    f"{conf_metrics.get('mean_entropy', 0.0):<12.4f}"
                )
            logger.info("")

        # --- Spectral Analysis Results ---
        if analyzer_config.analyze_spectral and \
           model_analysis.spectral_analysis is not None and not model_analysis.spectral_analysis.empty:
            logger.info("SPECTRAL ANALYSIS (WeightWatcher)")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'Mean α':<12} {'Concentration':<15} {'Matrix Entropy':<15}")
            logger.info("-" * 80)

            # CORRECTED LOGIC: The detailed results are in a pandas DataFrame.
            # We need to group by model to get per-model summary statistics.
            spectral_df = model_analysis.spectral_analysis
            per_model_summary = spectral_df.groupby('model_name')[
                ['alpha', 'concentration_score', 'entropy']
            ].mean()

            # Iterate through the performance results keys to maintain a consistent model order
            for model_name in results['performance_analysis'].keys():
                if model_name in per_model_summary.index:
                    model_summary = per_model_summary.loc[model_name]
                    mean_alpha = model_summary.get('alpha', 0.0)
                    mean_concentration = model_summary.get('concentration_score', 0.0)
                    mean_entropy = model_summary.get('entropy', 0.0)

                    logger.info(
                        f"{model_name:<30} {mean_alpha:<12.4f} {mean_concentration:<15.4f} {mean_entropy:<15.4f}"
                    )
            logger.info("")

        # --- Training Dynamics ---
        if analyzer_config.analyze_training_dynamics and model_analysis.training_metrics:
            logger.info("TRAINING DYNAMICS (ModelAnalyzer)")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'Epochs to Conv':<15} {'Overfitting':<15} {'Stability':<12}")
            logger.info("-" * 80)

            # Get the TrainingMetrics object
            tm = model_analysis.training_metrics

            # Iterate through the models from the performance results to ensure we have a model name
            for model_name in results['performance_analysis'].keys():
                # Access metrics from the TrainingMetrics object using the model name
                epochs_conv = tm.epochs_to_convergence.get(model_name, 0)
                overfit = tm.overfitting_index.get(model_name, 0.0)
                stability = tm.training_stability_score.get(model_name, 0.0)

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
            logger.info(f"{'Model':<30} {'Val Accuracy':<15} {'Val Loss':<12}")
            logger.info("-" * 80)

            for model_name, history_dict in results['histories'].items():
                if history_dict.get('val_accuracy') and len(history_dict['val_accuracy']) > 0:
                    final_val_acc = history_dict['val_accuracy'][-1]
                    final_val_loss = history_dict['val_loss'][-1]
                    logger.info(
                        f"{model_name:<30} "
                        f"{final_val_acc:<15.4f} "
                        f"{final_val_loss:<12.4f}"
                    )
                else:
                    logger.info(f"{model_name:<30} {'Not trained':<15} {'Not trained':<12}")
            logger.info("")

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the output layer comparison experiment.

    This function serves as the entry point for the experiment, handling
    configuration setup, experiment execution, and error handling.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("CIFAR-10 Output Layer Comparison Experiment")
    logger.info("Hierarchical Routing vs. Softmax")
    logger.info("Enhanced with Comprehensive ModelAnalyzer Integration")
    logger.info("=" * 80)
    logger.info("")

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  Model Types: {config.model_types}")
    logger.info(f"  Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"  Loss Function: {config.loss_function.name}")
    logger.info(f"  Model Architecture: {len(config.conv_filters)} conv blocks")
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