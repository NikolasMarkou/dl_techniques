"""
MNIST CNN-RBF Classification Experiment
=======================================

This experiment evaluates a CNN architecture with Radial Basis Function (RBF)
layers for MNIST digit classification. The primary goal is to investigate whether
RBF layers can enhance the feature representation and classification performance
compared to traditional dense layers.

Hypothesis:
-----------
The hypothesis is that RBF layers, with their ability to model non-linear
transformations through radial basis functions, can create more discriminative
feature representations. This is expected to lead to better classification
performance and more robust decision boundaries compared to standard dense layers.

Experimental Design:
--------------------
- **Dataset**: MNIST (28×28 grayscale images, 10 classes), using standardized
  dataset builder with proper augmentation.

- **Model Architecture**: A CNN with three convolutional blocks followed by
  an RBF-Dense classification head:
    - Conv Block 1: 16 filters (5×5 kernel)
    - Conv Block 2: 32 filters (3×3 kernel)
    - Conv Block 3: 64 filters (3×3 kernel)
    - Each block: Conv → BatchNorm → ReLU → MaxPool
    - Classification head: Flatten → RBF Layer → Dense Output

- **Comparative Analysis**: The experiment includes comparisons with baseline
  models using standard dense layers instead of RBF layers.

This version integrates the visualization framework and uses standardized dataset loaders.
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
from typing import Dict, Any, List, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.radial_basis_function import RBFLayer

from dl_techniques.datasets.vision.common import (
    create_dataset_builder,
)

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    ModelComparison,
    ClassificationResults,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ModelComparisonBarChart,
    PerformanceRadarChart,
    ConfusionMatrixVisualization,
)

# Model analyzer
from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput
)


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Minimal training configuration for dataset builder compatibility.

    Attributes:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    num_classes: int = 10
    batch_size: int = 128
    epochs: int = 100


@dataclass
class ExperimentConfig:
    """
    Configuration for the MNIST CNN-RBF experiment.

    This class encapsulates all configurable parameters for the experiment,
    including dataset configuration, model architecture parameters, training
    settings, and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 1)
    use_rgb: bool = False  # Keep MNIST in grayscale

    # --- Model Architecture Parameters ---
    conv1_filters: int = 16
    conv2_filters: int = 32
    conv3_filters: int = 64
    kernel_size: Tuple[int, int] = (3, 3)
    kernel_size_stem: Tuple[int, int] = (5, 5)
    activation: str = "relu"

    # RBF parameters
    rbf_units: int = 32

    # Regularization
    kernel_regularizer_l2: float = 1e-4

    # --- Training Parameters ---
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    lr_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_learning_rate: float = 1e-6
    monitor_metric: str = 'val_accuracy'

    # --- Model Variants ---
    model_variants: Dict[str, str] = field(default_factory=lambda: {
        'CNN_RBF': 'rbf',
        'CNN_Dense': 'dense',
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "mnist_cnn_rbf"
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
# MODEL ARCHITECTURE
# ==============================================================================

class MNISTCNNRBFModel(keras.Model):
    """
    MNIST classification model using CNN layers followed by RBF-Dense block.

    Architecture:
        Conv1 → BatchNorm → ReLU → MaxPool →
        Conv2 → BatchNorm → ReLU → MaxPool →
        Conv3 → BatchNorm → ReLU → MaxPool →
        Flatten → RBF/Dense Block → Output

    Args:
        config: Experiment configuration
        use_rbf: If True, use RBF layer; if False, use standard Dense layer
        name: Model name
    """

    def __init__(
            self,
            config: ExperimentConfig,
            use_rbf: bool = True,
            name: str = 'mnist_cnn_rbf',
            **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.config = config
        self.use_rbf = use_rbf

        # Create regularizer
        kernel_regularizer = keras.regularizers.L2(config.kernel_regularizer_l2)

        # First conv block
        self.conv1 = keras.layers.Conv2D(
            config.conv1_filters,
            config.kernel_size_stem,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer,
            name='conv1'
        )
        self.norm1 = keras.layers.BatchNormalization(name='bn1')
        self.act1 = keras.layers.Activation(config.activation, name='act1')
        self.pool1 = keras.layers.MaxPooling2D(name='pool1')

        # Second conv block
        self.conv2 = keras.layers.Conv2D(
            config.conv2_filters,
            config.kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer,
            name='conv2'
        )
        self.norm2 = keras.layers.BatchNormalization(name='bn2')
        self.act2 = keras.layers.Activation(config.activation, name='act2')
        self.pool2 = keras.layers.MaxPooling2D(name='pool2')

        # Third conv block
        self.conv3 = keras.layers.Conv2D(
            config.conv3_filters,
            config.kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer,
            name='conv3'
        )
        self.norm3 = keras.layers.BatchNormalization(name='bn3')
        self.act3 = keras.layers.Activation(config.activation, name='act3')
        self.pool3 = keras.layers.MaxPooling2D(name='pool3')

        # Flatten
        self.flatten = keras.layers.Flatten(name='flatten')

        # RBF or Dense layer
        if use_rbf:
            self.feature_layer = RBFLayer(
                units=config.rbf_units,
                name='rbf_layer'
            )
        else:
            self.feature_layer = keras.layers.Dense(
                units=config.rbf_units,
                activation=config.activation,
                kernel_initializer='he_normal',
                kernel_regularizer=kernel_regularizer,
                name='dense_layer'
            )

        # Output layer
        self.output_layer = keras.layers.Dense(
            units=config.num_classes,
            activation='softmax',
            kernel_initializer='he_normal',
            name='output'
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.norm3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)

        # Feature extraction
        x = self.flatten(x)
        x = self.feature_layer(x, training=training)

        # Output
        x = self.output_layer(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        config = super().get_config()
        config.update({
            'config': self.config,
            'use_rbf': self.use_rbf
        })
        return config


# ==============================================================================
# MODEL BUILDING UTILITY
# ==============================================================================

def build_model(
        config: ExperimentConfig,
        model_type: str,
        name: str
) -> keras.Model:
    """
    Build a CNN model with RBF or Dense classification head.

    Args:
        config: Experiment configuration
        model_type: 'rbf' or 'dense'
        name: Model name

    Returns:
        Compiled Keras model
    """
    use_rbf = (model_type == 'rbf')

    model = MNISTCNNRBFModel(
        config=config,
        use_rbf=use_rbf,
        name=name
    )

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    return model


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def train_single_model(
        model: keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        config: ExperimentConfig,
        steps_per_epoch: int,
        val_steps: int,
        model_name: str,
        output_dir: Path
) -> Dict[str, List[float]]:
    """
    Train a single model and return its history.

    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        config: Experiment configuration
        steps_per_epoch: Number of steps per epoch
        val_steps: Number of validation steps
        model_name: Name of the model
        output_dir: Directory to save checkpoints

    Returns:
        Training history dictionary
    """
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=config.monitor_metric,
            factor=config.lr_reduction_factor,
            patience=config.lr_patience,
            min_lr=config.min_learning_rate,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f'{model_name}_best.keras'),
            monitor=config.monitor_metric,
            save_best_only=True,
            verbose=0
        )
    ]

    # Train the model
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    return history.history


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete MNIST CNN-RBF experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing using standardized builder
    2. Model training for each variant (RBF vs Dense)
    3. Model analysis and evaluation
    4. Visualization generation using framework
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
    viz_config = PlotConfig(
        style=PlotStyle.SCIENTIFIC,
        color_scheme=ColorScheme(
            primary='#2E86AB',
            secondary='#A23B72',
            accent='#F18F01'
        ),
        title_fontsize=14,
        save_format='png'
    )

    viz_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations",
        config=viz_config
    )

    # Register visualization plugins
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("performance_radar", PerformanceRadarChart)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    # Log experiment start
    logger.info("🚀 Starting MNIST CNN-RBF Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading MNIST dataset using standardized builder...")

    # Create training config for dataset builder
    train_config = TrainingConfig(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    # Create dataset builder with grayscale format
    dataset_builder = create_dataset_builder('mnist', train_config, use_rgb=config.use_rgb)

    # Build datasets
    train_ds, val_ds, steps_per_epoch, val_steps = dataset_builder.build()

    # Get test data for evaluation
    test_data = dataset_builder.get_test_data()
    class_names = dataset_builder.get_class_names()

    logger.info("✅ Dataset loaded successfully")
    logger.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {val_steps}")
    logger.info(f"Test data shape: {test_data.x_data.shape}")
    logger.info(f"Class names: {class_names}")

    # ===== MODEL TRAINING PHASE =====
    logger.info("🏋️ Starting model training phase...")

    trained_models = {}
    all_histories = {}

    for variant_name, model_type in config.model_variants.items():
        logger.info(f"--- Training {variant_name} model ---")

        # Build model
        model = build_model(config, model_type, variant_name)

        # Log model architecture
        model.summary(print_fn=logger.info)
        logger.info(f"Model {variant_name} parameters: {model.count_params():,}")

        # Train the model
        history = train_single_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            config=config,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            model_name=variant_name,
            output_dir=experiment_dir / "checkpoints"
        )

        # Store results
        trained_models[variant_name] = model
        all_histories[variant_name] = history
        logger.info(f"✅ {variant_name} training completed!")

    # ===== MEMORY MANAGEMENT =====
    logger.info("🗑️ Triggering garbage collection...")
    gc.collect()

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("📈 Evaluating final model performance on test set...")

    performance_results = {}
    all_predictions = {}

    for name, model in trained_models.items():
        logger.info(f"Evaluating model {name}...")

        # Get predictions
        predictions = model.predict(test_data.x_data, verbose=0)
        y_pred_classes = np.argmax(predictions, axis=1)

        # Calculate metrics manually for consistency
        y_true = test_data.y_data.astype(int)
        accuracy = np.mean(y_pred_classes == y_true)

        # Calculate top-5 accuracy
        top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]
        top5_acc = np.mean([
            y_true_val in top5_pred
            for y_true_val, top5_pred in zip(y_true, top_5_predictions)
        ])

        # Calculate loss
        loss = -np.mean(
            np.log(predictions[np.arange(len(y_true)), y_true] + 1e-7)
        )

        performance_results[name] = {
            'accuracy': accuracy,
            'top_5_accuracy': top5_acc,
            'loss': loss
        }

        all_predictions[name] = y_pred_classes

        logger.info(f"Model {name} - Accuracy: {accuracy:.4f}, Top-5: {top5_acc:.4f}, Loss: {loss:.4f}")

    # ===== VISUALIZATION GENERATION =====
    logger.info("🖼️ Generating visualizations using framework...")

    # 1. Training curves comparison
    training_histories = {
        name: TrainingHistory(
            epochs=list(range(len(hist['loss']))),
            train_loss=hist['loss'],
            val_loss=hist['val_loss'],
            train_metrics={'accuracy': hist['accuracy']},
            val_metrics={'accuracy': hist['val_accuracy']}
        )
        for name, hist in all_histories.items()
    }

    viz_manager.visualize(
        data=training_histories,
        plugin_name="training_curves",
        show=False
    )

    # 2. Model comparison
    comparison_data = ModelComparison(
        model_names=list(performance_results.keys()),
        metrics={
            name: {
                'accuracy': metrics['accuracy'],
                'top_5_accuracy': metrics['top_5_accuracy']
            }
            for name, metrics in performance_results.items()
        }
    )

    viz_manager.visualize(
        data=comparison_data,
        plugin_name="model_comparison_bars",
        sort_by='accuracy',
        show=False
    )

    viz_manager.visualize(
        data=comparison_data,
        plugin_name="performance_radar",
        normalize=True,
        show=False
    )

    # 3. Confusion matrices for each model
    for name, y_pred in all_predictions.items():
        classification_results = ClassificationResults(
            y_true=test_data.y_data.astype(int),
            y_pred=y_pred,
            class_names=class_names,
            model_name=name
        )

        viz_manager.visualize(
            data=classification_results,
            plugin_name="confusion_matrix",
            normalize='true',
            show=False
        )

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("📊 Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        analyzer = ModelAnalyzer(
            models=trained_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        model_analysis_results = analyzer.analyze(data=test_data)
        logger.info("✅ Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Model analysis failed: {e}", exc_info=True)

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'trained_models': trained_models,
        'predictions': all_predictions,
        'test_data': test_data,
        'class_names': class_names
    }

    # Save and print results
    save_experiment_results(results_payload, experiment_dir, config)
    print_experiment_summary(results_payload)

    return results_payload


# ==============================================================================
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(
        results: Dict[str, Any],
        experiment_dir: Path,
        config: ExperimentConfig
) -> None:
    """
    Save experiment results in multiple formats.

    Args:
        results: Experiment results dictionary
        experiment_dir: Directory to save results
        config: Experiment configuration
    """
    try:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_numpy(i) for i in obj]
            return obj

        # Save configuration
        config_dict = {
            'experiment_name': config.experiment_name,
            'model_variants': config.model_variants,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'random_seed': config.random_seed,
            'architecture': {
                'conv1_filters': config.conv1_filters,
                'conv2_filters': config.conv2_filters,
                'conv3_filters': config.conv3_filters,
                'rbf_units': config.rbf_units
            }
        }
        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save performance results
        perf_converted = convert_numpy(results['performance_analysis'])
        with open(experiment_dir / "performance_results.json", 'w') as f:
            json.dump(perf_converted, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for name, model in results['trained_models'].items():
            model.save(models_dir / f"{name}.keras")

        logger.info("💾 Experiment results saved successfully")

    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}", exc_info=True)


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary
    """
    logger.info("=" * 80)
    logger.info("📋 MNIST CNN-RBF EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Performance metrics
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("\n🎯 PERFORMANCE METRICS (on Test Set):")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12}")
        logger.info("-" * 60)

        # Sort by accuracy
        sorted_results = sorted(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        for model_name, metrics in sorted_results:
            accuracy = metrics.get('accuracy', 0.0)
            top5_acc = metrics.get('top_5_accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{model_name:<20} {accuracy:<12.4f} {top5_acc:<12.4f} {loss:<12.4f}")

    # Calibration metrics
    model_analysis = results.get('model_analysis')
    if model_analysis and model_analysis.calibration_metrics:
        logger.info("\n🎯 CALIBRATION METRICS:")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<12}")
        logger.info("-" * 65)

        for model_name, cal_metrics in model_analysis.calibration_metrics.items():
            logger.info(
                f"{model_name:<20} {cal_metrics['ece']:<12.4f} "
                f"{cal_metrics['brier_score']:<15.4f} {cal_metrics['mean_entropy']:<12.4f}"
            )

    # Best model identification
    if 'performance_analysis' in results:
        best_model = max(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['accuracy']
        )
        logger.info(f"\n🏆 BEST MODEL: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

        # Calculate improvement
        if len(results['performance_analysis']) > 1:
            models = list(results['performance_analysis'].items())
            if models[0][1]['accuracy'] != models[1][1]['accuracy']:
                improvement = abs(models[0][1]['accuracy'] - models[1][1]['accuracy'])
                logger.info(f"📊 Performance difference: {improvement:.4f} ({improvement * 100:.2f}%)")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the MNIST CNN-RBF experiment.
    """
    logger.info("🚀 MNIST CNN-RBF Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(e)

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Model Variants: {list(config.model_variants.keys())}")
    logger.info(f"   Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"   Architecture: {config.conv1_filters}→{config.conv2_filters}→{config.conv3_filters} conv filters")
    logger.info(f"   RBF Units: {config.rbf_units}")
    logger.info(f"   Color Mode: {'RGB' if config.use_rgb else 'Grayscale'}")
    logger.info("")

    try:
        # Run the complete experiment
        results = run_experiment(config)
        logger.info("✅ Experiment completed successfully!")

    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()