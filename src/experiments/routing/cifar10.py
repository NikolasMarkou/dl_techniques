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

**Model Architecture**: A consistent ResNet-inspired CNN is used for both models,
with only the final output layer differing. The base architecture includes:
- Initial convolutional layer (32 filters)
- 4 convolutional blocks with residual connections
- Progressive filter scaling: [32, 64, 128, 256]
- Batch normalization and dropout regularization
- Global average pooling
- Dense classification layers with L2 regularization

**Output Layers Evaluated**:

1. **Standard Softmax**: The baseline approach. A `Dense` layer produces logits,
   followed by a `Softmax` activation to generate a probability distribution.
   Complexity: O(N), where N is the number of classes.

2. **Hierarchical Routing**: A probabilistic binary tree approach. The
   `HierarchicalRoutingLayer` directly produces a probability distribution.
   Complexity: O(log₂N), offering significant computational advantages for
   large N.

Comprehensive Analysis Pipeline
------------------------------

The experiment employs a multi-faceted analysis approach to compare the models:

**Training Analysis**:
- Training and validation curves for accuracy and loss.
- Convergence behavior and stability assessment.
- Early stopping based on validation accuracy.

**Model Performance Evaluation**:
- Test set accuracy and top-k accuracy.
- Final loss values.
- Statistical significance testing (if multiple runs were performed).

**Calibration and Prediction Analysis** (via ModelAnalyzer):
- Expected Calibration Error (ECE) to measure prediction confidence.
- Brier score for probabilistic prediction quality.
- Reliability diagrams and calibration plots.
- Entropy analysis of the output probability distributions.

Expected Outcomes and Insights
------------------------------

This experiment is designed to reveal:

1. **Performance Trade-offs**: Does the computational efficiency of the
   `HierarchicalRoutingLayer` come at the cost of classification accuracy?

2. **Training Dynamics**: How does the routing-based learning process affect
   convergence speed and stability compared to the standard softmax?

3. **Prediction Quality**: Do the two layers produce differently calibrated
   probability distributions? We will investigate if one is inherently more
   or less confident in its predictions.

4. **Scalability Implications**: While CIFAR-10 has only 10 classes, this
   experiment provides a crucial proof-of-concept for applying the
   `HierarchicalRoutingLayer` to problems with much larger output spaces,
   such as large-vocabulary language models.
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
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.layers.hierarchical_routing import HierarchicalRoutingLayer
from dl_techniques.layers.routing_probabilities import RoutingProbabilitiesLayer

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
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    monitor_metric: str = 'val_accuracy'
    loss_function: Callable = field(default_factory=lambda: keras.losses.CategoricalCrossentropy(from_logits=False))

    # --- Models to Evaluate ---
    model_types: List[str] = field(default_factory=lambda: ['Softmax', 'HierarchicalRouting', 'RoutingProbabilities'])

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_routing_comparison"
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
# MODEL ARCHITECTURE BUILDING UTILITIES
# ==============================================================================

# Note: build_residual_block and build_conv_block are reused from the example
# as they define the common base architecture for this experiment.

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """Builds a residual block with skip connections."""
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
    """Builds a convolutional block, optionally with residual connections."""
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
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(config.pool_size)(x)
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
    a standard Softmax and the HierarchicalRoutingLayer.

    Args:
        config: Experiment configuration object.
        model_type: The type of output layer ('Softmax' or 'HierarchicalRouting').
        name: Name prefix for the model and its layers.

    Returns:
        A compiled Keras model.
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Feature extractor backbone (identical for all models)
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0], kernel_size=(4, 4), strides=(2, 2),
        padding='same', kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem'
    )(x)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)
    x = keras.layers.GlobalAveragePooling2D()(x)
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units, kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    # --- Interchangeable Output Layer ---
    if model_type == 'Softmax':
        logits = keras.layers.Dense(
            units=config.num_classes,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name='logits'
        )(x)
        predictions = keras.layers.Activation('softmax', name='predictions')(logits)
    elif model_type == 'HierarchicalRouting':
        predictions = HierarchicalRoutingLayer(
            output_dim=config.num_classes,
            name='predictions'
        )(x)
    elif model_type == 'RoutingProbabilities':
        logits = keras.layers.Dense(
            units=config.num_classes,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name='logits'
        )(x)
        predictions = RoutingProbabilitiesLayer()(logits)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choose from {config.model_types}")

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

    Orchestrates the pipeline: data loading, training each model type,
    analysis, visualization, and results reporting.

    Args:
        config: Experiment configuration object.

    Returns:
        A dictionary containing all experimental results.
    """
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    vis_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations"
    )
    vis_manager.register_template("training_curves", TrainingCurvesVisualization)
    vis_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    logger.info("Starting CIFAR-10 Output Layer Comparison Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    cifar10_data = load_and_preprocess_cifar10()

    trained_models = {}
    all_histories = {}

    for model_type in config.model_types:
        logger.info(f"--- Training model with {model_type} output layer ---")
        model = build_model(config, model_type, model_type)
        model.summary(print_fn=logger.info)

        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=model_type,
            output_dir=experiment_dir / "training_plots" / model_type
        )

        history = train_model(
            model, cifar10_data.x_train, cifar10_data.y_train,
            cifar10_data.x_test, cifar10_data.y_test, training_config
        )

        trained_models[model_type] = model
        all_histories[model_type] = history.history
        logger.info(f"{model_type} training completed!")

    gc.collect()

    logger.info("Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )
        model_analysis_results = analyzer.analyze(data=DataInput.from_object(cifar10_data))
        logger.info("Model analysis completed successfully!")
    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

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
        name: model.predict(cifar10_data.x_test, verbose=0)
        for name, model in trained_models.items()
    }
    class_predictions = {
        name: np.argmax(preds, axis=1)
        for name, preds in raw_predictions.items()
    }

    y_true_indices = np.argmax(cifar10_data.y_test, axis=1)
    # 1. Create a dictionary to hold the results for all models
    all_classification_results = {}

    # 2. Loop to populate the dictionary
    for model_name, y_pred in class_predictions.items():
        try:
            # Create the data container for each model
            classification_data = ClassificationResults(
                y_true=y_true_indices,
                y_pred=y_pred,
                y_prob=raw_predictions[model_name],
                class_names=cifar10_data.class_names,
                model_name=model_name
            )
            # Add it to our dictionary
            all_classification_results[model_name] = classification_data
        except Exception as e:
            logger.error(f"Failed to prepare classification results for {model_name}: {e}")

    # 3. Create the multi-model data container
    if all_classification_results:
        try:
            multi_model_data = MultiModelClassification(
                results=all_classification_results,
                dataset_name="CIFAR-10"
            )

            # 4. Make a SINGLE call to the visualizer with the aggregated data
            vis_manager.visualize(
                data=multi_model_data,
                plugin_name="confusion_matrix",
                normalize='true',
                show=False
            )
            logger.info("Multi-model confusion matrix visualization created.")
        except Exception as e:
            logger.error(f"Failed to create multi-model confusion matrix: {e}")

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("Evaluating final model performance on test set...")
    logger.info(f"Test data shape: {cifar10_data.x_test.shape}, {cifar10_data.y_test.shape}")
    logger.info(f"Test labels sample: {cifar10_data.y_test[:5]}")

    performance_results = {}

    for name, model in trained_models.items():
        logger.info(f"Evaluating model {name}...")
        logger.info(f"Model {name} metrics: {model.metrics_names}")

        # Get model evaluation metrics
        eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        # Debug logging for metric inspection
        logger.info(f"Raw evaluation metrics for {name}: {metrics_dict}")

        # Calculate manual accuracy verification
        predictions = model.predict(cifar10_data.x_test, verbose=0)
        y_true_indices = np.argmax(cifar10_data.y_test, axis=1)

        # Manual top-1 accuracy verification
        manual_top1_acc = np.mean(np.argmax(predictions, axis=1) == y_true_indices)
        logger.info(f"Manual top-1 accuracy for {name}: {manual_top1_acc:.4f}")

        # Manual top-5 accuracy calculation
        top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]
        manual_top5_acc = np.mean([
            y_true in top5_pred
            for y_true, top5_pred in zip(y_true_indices, top_5_predictions)
        ])
        logger.info(f"Manual top-5 accuracy for {name}: {manual_top5_acc:.4f}")

        # Store standardized metrics
        performance_results[name] = {
            'accuracy': metrics_dict.get('accuracy', manual_top1_acc),
            'top_5_accuracy': metrics_dict.get('top_5_accuracy', manual_top5_acc),
            'loss': metrics_dict.get('loss', 0.0)
        }

        # Warn about potentially problematic accuracy values
        final_accuracy = performance_results[name]['accuracy']
        if final_accuracy < 0.2:
            logger.warning(f"Low accuracy detected for {name}: {final_accuracy:.4f}")
            logger.warning("This may indicate training issues or model problems")

        # Log final metrics for this model
        logger.info(f"Model {name} final metrics: {performance_results[name]}")

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
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
    including performance metrics, calibration analysis, and training progress.
    The summary is formatted for clear readability and easy interpretation.

    Args:
        results: Dictionary containing all experimental results and analysis
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== PERFORMANCE METRICS SECTION =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("PERFORMANCE METRICS (on Full Test Set):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12}")
        logger.info("-" * 60)

        for model_name, metrics in results['performance_analysis'].items():
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f} {loss:<12.4f}")

    # ===== CALIBRATION METRICS SECTION =====
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

    # ===== TRAINING METRICS SECTION =====
    if 'histories' in results and results['histories']:
        # Check if any model actually has training history data
        has_training_data = False
        for model_name, history_dict in results['histories'].items():
            if (history_dict.get('val_accuracy') and len(history_dict['val_accuracy']) > 0 and
                history_dict.get('val_loss') and len(history_dict['val_loss']) > 0):
                has_training_data = True
                break

        if has_training_data:
            logger.info("FINAL TRAINING METRICS (on Validation Set):")
            logger.info(f"{'Model':<20} {'Val Accuracy':<15} {'Val Loss':<12}")
            logger.info("-" * 50)

            for model_name, history_dict in results['histories'].items():
                # Check if this specific model has actual training data
                if (history_dict.get('val_accuracy') and len(history_dict['val_accuracy']) > 0 and
                    history_dict.get('val_loss') and len(history_dict['val_loss']) > 0):
                    final_val_acc = history_dict['val_accuracy'][-1]
                    final_val_loss = history_dict['val_loss'][-1]
                    logger.info(f"{model_name:<20} {final_val_acc:<15.4f} {final_val_loss:<12.4f}")
                else:
                    logger.info(f"{model_name:<20} {'Not trained':<15} {'Not trained':<12}")
        else:
            logger.info("TRAINING STATUS:")
            logger.info("Models were not trained (epochs=0) - no training metrics available")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for running the experiment."""
    logger.info("CIFAR-10 Output Layer Comparison: Hierarchical Routing vs. Softmax")
    logger.info("=" * 80)

    config = ExperimentConfig()

    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"   Model Types: {config.model_types}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Loss Function: {config.loss_function.name}")
    logger.info("")

    try:
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