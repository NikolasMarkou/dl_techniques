"""
ImageNet-1k Output Layer Comparison: Hierarchical Routing vs. Softmax
======================================================================

This experiment evaluates the performance and characteristics of the novel
`HierarchicalRoutingLayer` against the standard `Dense` -> `Softmax`
classifier for large-scale image classification on the ImageNet (ILSVRC 2012) dataset.

While previous experiments on CIFAR-10 (10 classes) served as a proof-of-concept,
this experiment on ImageNet (1,000 classes) addresses the core theoretical value
proposition of Hierarchical Routing: computational scalability.

The study aims to answer a critical question for large-scale classification:
can we replace the computationally expensive O(N) softmax layer with a more
efficient O(log N) alternative without sacrificing accuracy?

Experimental Design
-------------------

**Dataset**: ImageNet-1k (ILSVRC 2012)
- ~1.28 million training images
- 50,000 validation images
- 1,000 classes
- Preprocessed to 224×224 RGB images

**Model Architecture**: A ResNet-inspired CNN backbone adapted for higher
resolution inputs.
- Initial convolutional stem with downsampling
- Deep convolutional blocks with residual connections
- Periodic pooling to manage spatial dimensionality (224 -> 7)
- Global average pooling
- Feature dimension: 512 or 1024

**Output Layers Evaluated**:

1. **Standard Softmax**:
   - Complexity: O(N) = 1,000 operations per sample.
   - Mechanism: Computes exponentials for all 1,000 logits and normalizes.
   - Bottleneck: Memory bandwidth and compute scale linearly with class count.

2. **Hierarchical Routing**:
   - Complexity: O(log₂N) ≈ 10 operations per sample (depth of tree).
   - Mechanism: A probabilistic binary tree where decisions are made at nodes.
   - Advantage: Massive reduction in final layer compute, crucial for
     extreme classification (e.g., 10k+ classes) or resource-constrained inference.

3. **Routing Probabilities**: An alternative routing-based approach using
   the `RoutingProbabilitiesLayer` on top of dense logits.

Comprehensive Analysis Pipeline
------------------------------

**Scalability & Performance**:
- Validation accuracy (Top-1 and Top-5)
- Training throughput (images/sec) difference between O(N) and O(log N) layers

**ModelAnalyzer Integration**:
- Due to dataset size, analysis is performed on a statistically significant
  stratified subset (N=2048) of the validation set.
- **Calibration**: Does the tree structure lead to better calibrated confidence
  on difficult fine-grained classes (e.g., different breeds of dogs)?
- **Spectral Analysis**: Analyzing the weight matrices of the routing nodes
  versus the massive dense matrix of the Softmax layer.

Theoretical Foundation
---------------------
Moving from 10 classes (CIFAR) to 1,000 classes (ImageNet) increases the
computational cost of the final layer by 100x for Softmax.
Hierarchical Routing, scaling logarithmically, should theoretically see
minimal increase in cost (going from depth ~4 to depth ~10).
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable, Optional

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
class ImageNetData:
    """
    Container for ImageNet-1k dataset pipelines.

    Unlike CIFAR-10, ImageNet is too large to hold in memory as numpy arrays.
    This container holds tf.data.Dataset objects optimized for streaming.

    Attributes:
        train_ds: Training dataset pipeline (shuffled, batched, prefetched)
        val_ds: Validation dataset pipeline (batched, prefetched)
        class_names: List of 1000 class names
        val_sample: Small numpy subset of validation data for intense analysis
    """
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    class_names: List[str]
    # For ModelAnalyzer which usually requires in-memory data
    val_sample: Optional[Tuple[np.ndarray, np.ndarray]] = None


def preprocess_imagenet(features: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Standard ImageNet preprocessing pipeline.
    Resizes images to 224x224 and normalizes to [0, 1].
    """
    image = features['image']
    label = features['label']

    # Resize to standard 224x224
    image = tf.image.resize(image, (224, 224))

    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # One-hot encode labels for 1000 classes
    label = tf.one_hot(label, 1000)

    return image, label


def load_and_preprocess_imagenet(
        batch_size: int = 128,
        data_dir: Optional[str] = None
) -> ImageNetData:
    """
    Load and preprocess ImageNet-1k dataset using TensorFlow Datasets.

    NOTE: ImageNet requires manual download and setup in TFDS.
    If 'imagenet2012' is not found, this will raise a specific error from TFDS.

    Args:
        batch_size: Batch size for training/eval.
        data_dir: Optional custom directory for TFDS data.

    Returns:
        ImageNetData object containing dataset pipelines.
    """
    logger.info("Loading ImageNet-1k (imagenet2012) dataset...")
    logger.info("Note: This assumes ImageNet data is manually downloaded/prepared for TFDS.")

    try:
        # Load the dataset builder
        builder = tfds.builder('imagenet2012', data_dir=data_dir)

        # Download/prepare checks (usually no-op if data exists)
        # builder.download_and_prepare()

        # Get dataset info for class names
        info = builder.info
        class_names = info.features['label'].names

        # Load splits
        train_ds = tfds.load('imagenet2012', split='train', data_dir=data_dir, shuffle_files=True)
        val_ds = tfds.load('imagenet2012', split='validation', data_dir=data_dir, shuffle_files=False)

        # Apply preprocessing
        logger.info("Building data pipelines (Resize 224x224, Normalize)...")

        train_ds = (
            train_ds
            .map(preprocess_imagenet, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = (
            val_ds
            .map(preprocess_imagenet, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Create a small numpy sample for ModelAnalyzer (which needs random access)
        # We take ~1000 images from validation set
        logger.info("Extracting numpy sample for ModelAnalyzer (N=1024)...")
        sample_ds = val_ds.unbatch().take(1024)
        x_sample = []
        y_sample = []
        for img, lbl in sample_ds:
            x_sample.append(img.numpy())
            y_sample.append(lbl.numpy())

        val_sample = (np.array(x_sample), np.array(y_sample))

        logger.info(f"ImageNet loaded. Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")
        logger.info(f"Num classes: {len(class_names)}")

        return ImageNetData(
            train_ds=train_ds,
            val_ds=val_ds,
            class_names=class_names,
            val_sample=val_sample
        )

    except Exception as e:
        logger.error("Failed to load ImageNet. Ensure 'imagenet2012' is prepared in TFDS.")
        logger.error("Error details: " + str(e))
        raise


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for the Hierarchical Routing vs. Softmax experiment on ImageNet.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "imagenet2012"
    num_classes: int = 1000
    input_shape: Tuple[int, ...] = (224, 224, 3)

    # --- Model Architecture Parameters (Scaled for ImageNet) ---
    # Deeper filter progression for larger images
    conv_filters: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 512, 1024])
    # Pooling strategy: Indices of blocks after which to apply pooling/downsampling
    pooling_indices: List[int] = field(default_factory=lambda: [0, 1, 3, 4])

    dense_units: List[int] = field(default_factory=lambda: [1024])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.3, 0.4, 0.4, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # --- Training Parameters ---
    epochs: int = 60  # ImageNet converges slower
    batch_size: int = 64  # Reduced batch size for 224x224 images
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    monitor_metric: str = 'val_accuracy'
    loss_function: Callable = field(default_factory=lambda: keras.losses.CategoricalCrossentropy(from_logits=False))

    # --- Models to Evaluate ---
    model_types: List[str] = field(default_factory=lambda: [
        'HierarchicalRouting',  # O(log N)
        'Softmax',  # O(N)
    ])

    # --- Experiment Configuration ---
    output_dir: Path = Path("results_imagenet")
    experiment_name: str = "imagenet_routing_comparison"
    random_seed: int = 42

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=False,  # Too heavy for ImageNet without massive RAM
        analyze_training_dynamics=True,
        analyze_spectral=True,
        n_samples=1000,
        weight_layer_types=['Dense', 'Conv2D'],
        calibration_bins=20,
        save_plots=True,
        plot_style='publication',
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
    """Build a residual block with skip connections."""
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
    """Build a convolutional block."""
    if config.use_residual:
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

    # Apply dropout
    dropout_rate = (config.dropout_rates[block_index]
                    if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(config: ExperimentConfig, model_type: str, name: str) -> keras.Model:
    """
    Build a ResNet-style model adapted for ImageNet (224x224).

    Includes strided pooling logic to reduce spatial dimensions from
    224x224 down to roughly 7x7 before the dense layers.
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # === Stem (Downsample 224 -> 112) ===
    x = keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2),
        padding='same', kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Output is now 56x56

    # === Convolutional Blocks ===
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

        # Apply periodic downsampling to handle ImageNet resolution
        if i in config.pooling_indices:
            x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name=f'pool_{i}')(x)

    # Global pooling and normalization
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.LayerNormalization()(x)

    # === Interchangeable Output Layer (1000 Classes) ===
    if model_type == 'Softmax':
        # O(N) Complexity: Dense matrix 1000 units wide
        logits = keras.layers.Dense(
            units=config.num_classes,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name='logits'
        )(x)
        predictions = keras.layers.Activation('softmax', name='predictions')(logits)

    elif model_type == 'HierarchicalRouting':
        # O(log N) Complexity: Tree structure
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
        predictions = RoutingProbabilitiesLayer(name='predictions')(logits)

    elif model_type == 'PlainRoutingProbabilities':
        predictions = RoutingProbabilitiesLayer(
            output_dim=config.num_classes,
            name='predictions')(x)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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
    Run the complete output layer comparison experiment on ImageNet.
    """
    # Set random seed
    keras.utils.set_random_seed(config.random_seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    vis_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations"
    )
    vis_manager.register_template("training_curves", TrainingCurvesVisualization)
    vis_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    # Log experiment start
    logger.info("=" * 80)
    logger.info("ImageNet-1k Output Layer Comparison Experiment")
    logger.info(f"Target Classes: {config.num_classes} | Input: {config.input_shape}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    imagenet_data = load_and_preprocess_imagenet(
        batch_size=config.batch_size
    )
    logger.info("")

    # ===== MODEL TRAINING PHASE =====
    logger.info("=" * 80)
    logger.info("MODEL TRAINING PHASE")
    logger.info("=" * 80)

    trained_models = {}
    training_histories = {}

    for model_type in config.model_types:
        logger.info(f"\n--- Training model with {model_type} output layer ---")

        model = build_model(config, model_type, model_type)
        logger.info(f"Model parameters: {model.count_params():,}")

        # Verify output shape for ImageNet models
        logger.info(f"Output shape: {model.output_shape}")

        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=model_type,
            output_dir=experiment_dir / "training_plots" / model_type
        )

        # Train model using dataset pipelines instead of arrays
        history = train_model(
            model,
            imagenet_data.train_ds,
            None,  # y_train is in the dataset
            imagenet_data.val_ds,
            None,  # y_test is in the dataset
            training_config
        )

        trained_models[model_type] = model
        training_histories[model_type] = history.history

        logger.info(f"{model_type} training completed!")

        # Explicitly clear session to free GPU memory between large models
        gc.collect()
        keras.backend.clear_session()

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("=" * 80)
    logger.info("MODEL ANALYSIS (Using Validation Sample)")
    logger.info("=" * 80)

    model_analysis_results = None

    try:
        # Re-load models if they were cleared or for analysis consistency
        # (In this script they are still in memory in `trained_models` dict)

        # Prepare DataInput using the numpy sample we extracted earlier
        # Analysis tools need numpy arrays, not streaming datasets
        x_sample, y_sample = imagenet_data.val_sample

        test_data_input = DataInput(
            x_data=x_sample,
            y_data=y_sample
        )

        logger.info(f"Analysis sample shape: {x_sample.shape}")

        analyzer = ModelAnalyzer(
            models=trained_models,
            training_history=training_histories,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        model_analysis_results = analyzer.analyze(data=test_data_input)

    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

    # ===== ADDITIONAL VISUALIZATION: CONFUSION MATRICES =====
    # Note: Confusion Matrix for 1000 classes is unreadable.
    # We skip full CM visualization or implement Top-K confusion if supported.
    # Here we skip to avoid creating a 1000x1000 image.
    logger.info("Skipping 1000x1000 Confusion Matrix visualization.")

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("=" * 80)
    logger.info("FINAL PERFORMANCE EVALUATION (Full Validation Set)")
    logger.info("=" * 80)

    performance_results = {}

    for name, model in trained_models.items():
        logger.info(f"Evaluating model: {name}")

        # Evaluate on full validation dataset pipeline
        eval_results = model.evaluate(imagenet_data.val_ds, verbose=1)
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        performance_results[name] = {
            'accuracy': metrics_dict.get('accuracy', 0.0),
            'top_5_accuracy': metrics_dict.get('top_5_accuracy', 0.0),
            'loss': metrics_dict.get('loss', 0.0)
        }

        logger.info(f"  Accuracy: {performance_results[name]['accuracy']:.4f}")
        logger.info(f"  Top-5 Acc: {performance_results[name]['top_5_accuracy']:.4f}")
        logger.info(f"  Loss: {performance_results[name]['loss']:.4f}")

    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': training_histories,
    }

    print_experiment_summary(results_payload, config.analyzer_config)
    return results_payload


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(
        results: Dict[str, Any],
        analyzer_config: AnalysisConfig
) -> None:
    """Print summary of experimental results."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("IMAGENET EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Performance
    if 'performance_analysis' in results:
        logger.info("PERFORMANCE METRICS (Validation Set)")
        logger.info("-" * 80)
        logger.info(f"{'Model':<30} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Loss':<12}")
        logger.info("-" * 80)

        for model_name, metrics in results['performance_analysis'].items():
            logger.info(
                f"{model_name:<30} "
                f"{metrics.get('accuracy', 0.0):<12.4f} "
                f"{metrics.get('top_5_accuracy', 0.0):<12.4f} "
                f"{metrics.get('loss', 0.0):<12.4f}"
            )
        logger.info("")

    # Analysis
    model_analysis = results.get('model_analysis')
    if model_analysis and analyzer_config.analyze_calibration:
        logger.info("CALIBRATION (Sampled Subset)")
        logger.info("-" * 80)
        logger.info(f"{'Model':<30} {'ECE':<12} {'Brier Score':<15}")
        logger.info("-" * 80)

        if model_analysis.calibration_metrics:
            for model_name, cal_metrics in model_analysis.calibration_metrics.items():
                logger.info(
                    f"{model_name:<30} "
                    f"{cal_metrics.get('ece', 0.0):<12.4f} "
                    f"{cal_metrics.get('brier_score', 0.0):<15.4f}"
                )

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    logger.info("")
    logger.info("=" * 80)
    logger.info("ImageNet-1k Experiment Runner")
    logger.info("=" * 80)

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("No GPU detected! ImageNet training will be extremely slow.")
    else:
        logger.info(f"GPUs detected: {len(gpus)}")

    config = ExperimentConfig()

    try:
        run_experiment(config)
    except Exception as e:
        logger.error(f"EXPERIMENT FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()