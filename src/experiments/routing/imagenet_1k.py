# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import os
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
    ... (Same as before) ...
    """
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    class_names: List[str]
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
        data_dir: Optional[str] = None,
        manual_dir: Optional[str] = None
) -> ImageNetData:
    """
    Load and preprocess ImageNet-1k dataset using TensorFlow Datasets.

    Args:
        batch_size: Batch size for training/eval.
        data_dir: Directory where prepared TFRecords are/will be stored.
        manual_dir: Directory containing 'ILSVRC2012_img_train.tar' and 'ILSVRC2012_img_val.tar'.

    Returns:
        ImageNetData object containing dataset pipelines.
    """
    logger.info(f"Loading ImageNet-1k (imagenet2012)...")
    logger.info(f"  Data Dir (TFRecords): {data_dir}")
    logger.info(f"  Manual Dir (Tar files): {manual_dir}")

    try:
        # Load the dataset builder
        builder = tfds.builder('imagenet2012', data_dir=data_dir)

        # Configure download options with the specific manual directory
        download_config = tfds.download.DownloadConfig(
            manual_dir=manual_dir
        )

        # Explicitly prepare the data.
        # If dataset is already prepared, this returns immediately.
        # If not, it uses manual_dir to find tarballs and generate TFRecords.
        logger.info("Checking/Preparing ImageNet dataset...")
        try:
            builder.download_and_prepare(download_config=download_config)
        except Exception as e:
            # Provide a helpful error if preparation fails (likely missing files)
            if manual_dir and not os.path.exists(manual_dir):
                logger.error(f"Manual directory does not exist: {manual_dir}")
            else:
                logger.error(
                    f"Preparation failed. Ensure ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar are in {manual_dir}")
            raise e

        # Get dataset info
        info = builder.info
        class_names = info.features['label'].names

        # Load splits
        train_ds = builder.as_dataset(split='train', shuffle_files=True)
        val_ds = builder.as_dataset(split='validation', shuffle_files=False)

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

        # Create a small numpy sample for ModelAnalyzer
        logger.info("Extracting numpy sample for ModelAnalyzer (N=1024)...")
        sample_ds = val_ds.unbatch().take(1024)
        x_sample = []
        y_sample = []
        for img, lbl in sample_ds:
            x_sample.append(img.numpy())
            y_sample.append(lbl.numpy())

        val_sample = (np.array(x_sample), np.array(y_sample))

        logger.info(f"ImageNet loaded successfully.")

        return ImageNetData(
            train_ds=train_ds,
            val_ds=val_ds,
            class_names=class_names,
            val_sample=val_sample
        )

    except Exception as e:
        logger.error(f"Failed to load ImageNet.")
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

    # Path where TFDS will generate/look for the TFRecord files
    data_dir: str = "/media/arxwn/data0_4tb/datasets/tensorflow_datasets/"

    # !!! UPDATED: Path containing the downloaded .tar files
    # Change this to wherever your ILSVRC2012 tar files are located
    # Common default is <data_dir>/downloads/manual
    manual_dir: str = "/media/arxwn/data0_4tb/datasets/tensorflow_datasets/downloads/manual/"

    # --- Model Architecture Parameters (Scaled for ImageNet) ---
    conv_filters: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 512, 1024])
    pooling_indices: List[int] = field(default_factory=lambda: [0, 1, 3, 4])

    dense_units: List[int] = field(default_factory=lambda: [1024])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.3, 0.4, 0.4, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # --- Training Parameters ---
    epochs: int = 60
    batch_size: int = 64
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
        analyze_information_flow=False,
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

# ... (Same build_residual_block, build_conv_block, build_model as before) ...
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

    dropout_rate = (config.dropout_rates[block_index]
                    if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(config: ExperimentConfig, model_type: str, name: str) -> keras.Model:
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Stem
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

    # Blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)
        if i in config.pooling_indices:
            x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name=f'pool_{i}')(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.LayerNormalization()(x)

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
        predictions = RoutingProbabilitiesLayer(name='predictions')(logits)

    elif model_type == 'PlainRoutingProbabilities':
        predictions = RoutingProbabilitiesLayer(
            output_dim=config.num_classes,
            name='predictions')(x)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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

    logger.info("=" * 80)
    logger.info("ImageNet-1k Output Layer Comparison Experiment")
    logger.info(f"Target Classes: {config.num_classes} | Input: {config.input_shape}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    imagenet_data = load_and_preprocess_imagenet(
        batch_size=config.batch_size,
        data_dir=config.data_dir,
        manual_dir=config.manual_dir
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

        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric,
            model_name=model_type,
            output_dir=experiment_dir / "training_plots" / model_type
        )

        history = train_model(
            model,
            imagenet_data.train_ds,
            None,
            imagenet_data.val_ds,
            None,
            training_config
        )

        trained_models[model_type] = model
        training_histories[model_type] = history.history

        logger.info(f"{model_type} training completed!")

        gc.collect()
        keras.backend.clear_session()

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("=" * 80)
    logger.info("MODEL ANALYSIS (Using Validation Sample)")
    logger.info("=" * 80)

    model_analysis_results = None

    try:
        x_sample, y_sample = imagenet_data.val_sample
        test_data_input = DataInput(x_data=x_sample, y_data=y_sample)

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

    # ===== FINAL PERFORMANCE EVALUATION =====
    logger.info("=" * 80)
    logger.info("FINAL PERFORMANCE EVALUATION (Full Validation Set)")
    logger.info("=" * 80)

    performance_results = {}

    for name, model in trained_models.items():
        logger.info(f"Evaluating model: {name}")
        eval_results = model.evaluate(imagenet_data.val_ds, verbose=1)
        metrics_dict = dict(zip(model.metrics_names, eval_results))

        performance_results[name] = {
            'accuracy': metrics_dict.get('accuracy', 0.0),
            'top_5_accuracy': metrics_dict.get('top_5_accuracy', 0.0),
            'loss': metrics_dict.get('loss', 0.0)
        }

        logger.info(f"  Accuracy: {performance_results[name]['accuracy']:.4f}")

    results_payload = {
        'performance_analysis': performance_results,
        'model_analysis': model_analysis_results,
        'histories': training_histories,
    }

    print_experiment_summary(results_payload, config.analyzer_config)
    return results_payload


# ==============================================================================
# RESULTS REPORTING AND MAIN
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any], analyzer_config: AnalysisConfig) -> None:
    logger.info("")
    logger.info("=" * 80)
    logger.info("IMAGENET EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    if 'performance_analysis' in results:
        logger.info("PERFORMANCE METRICS")
        for model_name, metrics in results['performance_analysis'].items():
            logger.info(f"{model_name:<30} Acc: {metrics.get('accuracy', 0.0):.4f}")
    logger.info("=" * 80)


def main() -> None:
    logger.info("ImageNet-1k Experiment Runner")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPUs detected: {len(gpus)}")

    config = ExperimentConfig()
    run_experiment(config)


if __name__ == "__main__":
    main()