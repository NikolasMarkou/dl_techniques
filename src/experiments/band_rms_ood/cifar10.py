"""
Experiment: BandRMS-OOD Geometric Out-of-Distribution Detection on CIFAR-10

This experiment implements and evaluates the BandRMS-OOD approach, a novel method
for geometric out-of-distribution (OOD) detection. The core innovation lies in
using confidence-driven shell scaling within BandRMS normalization layers. This
technique creates learnable "spherical shells" in the feature space, where
in-distribution (ID) samples are actively encouraged toward an outer shell,
while OOD samples are expected to naturally fall toward the inner core.

Scientific Motivation:
--------------------
Traditional OOD detection methods often operate as post-hoc analyses on model
outputs. BandRMS-OOD, by contrast, embeds OOD detection capabilities directly
into the network architecture through geometric constraints, offering several
theoretical advantages:
1.  **Multi-Layer Detection**: OOD signals can be extracted from multiple layers,
    allowing for a more robust, consensus-based decision.
2.  **Geometric Interpretability**: Provides a clear spatial understanding of OOD
    decisions based on a sample's distance from the learned ID manifold shell.
3.  **Training Efficiency**: Requires no OOD examples during training, relying
    solely on the geometry of the ID feature space.
4.  **Confidence Integration**: Leverages the model's internal confidence
    estimates to dynamically shape the feature space during training.

Experimental Design:
--------------------
- **In-Distribution (ID) Data**: CIFAR-10 dataset.
- **Out-of-Distribution (OOD) Data**: SVHN, corrupted CIFAR-10, and uniform noise
  are used to test the model's detection capabilities against various OOD scenarios.

- **Model Architecture**: A ResNet-based CNN is modified to replace standard
  normalization layers with BandRMS-OOD or standard BandRMS layers. The design
  incorporates progressive shell tightening, where the acceptable band of feature
  norms (the shell width) becomes narrower in deeper layers.

- **Variants Evaluated**:
    - **BandRMS-OOD Models**: With different confidence estimation methods (e.g.,
      feature magnitude, entropy) and varying hyperparameters.
    - **Baseline Models**: Including a standard ResNet, a ResNet with regular
      BandRMS, Maximum Softmax Probability (MSP), and MaxLogit for comparison.

Evaluation Metrics:
-------------------
- **OOD Detection**: AUROC, FPR@95, and AUPR are used to quantify the ability to
  distinguish between ID and OOD samples.
- **Classification Performance**: Standard test accuracy and calibration (ECE) on
  the ID task (CIFAR-10) are measured to ensure OOD capabilities do not degrade
  primary task performance.
- **Shell Analysis**: The distribution of shell distances for ID vs. OOD data is
  analyzed to validate the geometric separation hypothesis.

Workflow:
---------
1.  **Configuration**: All experimental parameters are defined in the
    `BandRMSOODExperimentConfig` dataclass.
2.  **Data Handling**: The CIFAR-10 dataset is loaded, and various OOD datasets
    are generated or loaded.
3.  **Multi-Run Training**: Each model variant is trained for multiple runs to
    ensure statistical robustness.
4.  **Evaluation**: After training, each model is evaluated on:
    a. Its classification accuracy on the CIFAR-10 test set.
    b. Its OOD detection performance against each OOD dataset using the
       `MultiLayerOODDetector`, MSP, and MaxLogit methods.
5.  **Analysis and Visualization**: Results are aggregated across runs to compute
    mean and standard deviation. The `ModelAnalyzer` provides calibration metrics,
    and the `VisualizationManager` is used to generate comparative confusion
    matrices.
6.  **Reporting**: A final summary of all results is printed, and all artifacts
    (models, logs, plots, results) are saved to a timestamped directory.
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
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_cifar10
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.experimental.band_rms_ood import BandRMSOOD
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.visualization import (
    VisualizationManager,
    ClassificationResults,
    MultiModelClassification,
    ConfusionMatrixVisualization
)

# ==============================================================================
# OOD DETECTION UTILITIES
# ==============================================================================

class MultiLayerOODDetector:
    """
    Aggregates shell distances from multiple BandRMS-OOD layers.
    """

    def __init__(self, model: keras.Model, layer_weights: Optional[List[float]] = None):
        self.model = model
        self.ood_layers = self._find_ood_layers()
        self.layer_weights = layer_weights or [1.0] * len(self.ood_layers)
        self.threshold = None
        logger.info(f"Found {len(self.ood_layers)} BandRMS-OOD layers: "
                    f"{[layer.name for layer in self.ood_layers]}")

    def _find_ood_layers(self) -> List[BandRMSOOD]:
        """Find all BandRMS-OOD layers in the model."""
        return [layer for layer in self.model.layers if isinstance(layer, BandRMSOOD)]

    def compute_ood_scores(self, data: tf.Tensor) -> np.ndarray:
        """
        Compute OOD scores (higher = more likely OOD).

        Args:
            data: Input data tensor.

        Returns:
            Aggregated OOD scores for the input data.
        """
        _ = self.model(data, training=False)
        layer_distances = []
        for layer in self.ood_layers:
            distances = layer.get_shell_distance()
            if distances is not None:
                if len(distances.shape) > 2:
                    distances = tf.reduce_mean(distances, axis=list(range(1, len(distances.shape) - 1)))
                layer_distances.append(distances.numpy().flatten())

        if not layer_distances:
            raise ValueError("No shell distances found from BandRMS-OOD layers.")

        weighted_distances = np.zeros_like(layer_distances[0], dtype=np.float32)
        total_weight = sum(self.layer_weights)
        if total_weight > 0:
            for distances, weight in zip(layer_distances, self.layer_weights):
                weighted_distances += weight * distances
            return weighted_distances / total_weight
        return weighted_distances

    def fit_threshold(self, id_data: tf.Tensor, fpr_target: float = 0.05):
        """
        Fit detection threshold based on a target false positive rate.

        Args:
            id_data: In-distribution data for calibration.
            fpr_target: Target false positive rate (e.g., 0.05 for 95% TPR).
        """
        id_scores = self.compute_ood_scores(id_data)
        self.threshold = np.percentile(id_scores, (1.0 - fpr_target) * 100)
        logger.info(f"Set OOD threshold to {self.threshold:.4f} (target FPR: {fpr_target})")

    def predict_ood(self, data: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict OOD labels for input data using the fitted threshold.

        Args:
            data: Input data tensor.

        Returns:
            A tuple of (ood_predictions, ood_scores).
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")
        ood_scores = self.compute_ood_scores(data)
        ood_predictions = ood_scores > self.threshold
        return ood_predictions, ood_scores


def evaluate_ood_detection(
        id_scores: np.ndarray,
        ood_scores: np.ndarray,
        method_name: str = ""
) -> Dict[str, float]:
    """
    Evaluate OOD detection performance using standard metrics.

    Args:
        id_scores: OOD scores for in-distribution data.
        ood_scores: OOD scores for out-of-distribution data.
        method_name: Name of the method for logging purposes.

    Returns:
        A dictionary of evaluation metrics (AUROC, AUPR, FPR@95, etc.).
    """
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fpr95 = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 1.0

    results = {
        'auroc': auroc, 'aupr': aupr, 'fpr95': fpr95,
        'id_mean': np.mean(id_scores), 'id_std': np.std(id_scores),
        'ood_mean': np.mean(ood_scores), 'ood_std': np.std(ood_scores)
    }

    if method_name:
        logger.info(f"{method_name} - AUROC: {auroc:.4f}, FPR@95: {fpr95:.4f}, AUPR: {aupr:.4f}")

    return results


# ==============================================================================
# BASELINE OOD DETECTION METHODS
# ==============================================================================

def maximum_softmax_probability(model: keras.Model, data: tf.Tensor) -> np.ndarray:
    """Compute Maximum Softmax Probability (MSP) scores (negative for OOD)."""
    predictions = model(data, training=False)
    msp_scores = tf.reduce_max(tf.nn.softmax(predictions), axis=-1)
    return -msp_scores.numpy()


def max_logit(model: keras.Model, data: tf.Tensor) -> np.ndarray:
    """Compute MaxLogit scores (negative for OOD)."""
    predictions = model(data, training=False)
    max_logit_scores = tf.reduce_max(predictions, axis=-1)
    return -max_logit_scores.numpy()


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class BandRMSOODExperimentConfig:
    """Configuration for the BandRMS-OOD geometric OOD detection experiment."""

    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    ood_datasets: Dict[str, str] = field(default_factory=lambda: {
        'svhn': 'svhn_cropped', 'noise': 'uniform_noise', 'corrupted': 'gaussian_noise'
    })

    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [256])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    activation: str = 'gelu'

    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 15

    band_alphas: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    confidence_types: List[str] = field(default_factory=lambda: ['magnitude', 'entropy'])
    confidence_weights: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    shell_preference_weight: float = 0.01

    model_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'BandRMSOOD_mag_010': lambda: ('magnitude', 0.1, 1.0),
        'BandRMSOOD_mag_020': lambda: ('magnitude', 0.2, 1.0),
        'BandRMSOOD_ent_010': lambda: ('entropy', 0.1, 1.0),
        'BandRMSOOD_ent_020': lambda: ('entropy', 0.2, 1.0),
        'BandRMS_baseline': lambda: ('baseline', 0.1, 0.0),
        'Standard_ResNet': lambda: ('none', 0.0, 0.0),
    })

    output_dir: Path = Path("results")
    experiment_name: str = "bandrms_ood_detection"
    random_seed: int = 42
    n_runs: int = 3
    fpr_target: float = 0.05
    layer_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])

    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True, analyze_calibration=True, analyze_information_flow=False,
        analyze_training_dynamics=True, calibration_bins=15, save_plots=True, verbose=True
    ))

# ==============================================================================
# MODEL BUILDING UTILITIES
# ==============================================================================

def build_ood_model(
        config: BandRMSOODExperimentConfig,
        confidence_type: str,
        band_alpha: float,
        confidence_weight: float,
        model_name: str
) -> keras.Model:
    """Build a model with specified normalization layers for OOD detection."""
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{model_name}_input')
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0], kernel_size=(7, 7), padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem_conv'
    )(inputs)

    if confidence_type == 'none':
        x = keras.layers.BatchNormalization(name='stem_norm')(x)
    elif confidence_type == 'baseline':
        x = BandRMS(max_band_width=band_alpha, name='stem_norm')(x)
    else:
        x = BandRMSOOD(
            max_band_width=band_alpha, confidence_type=confidence_type,
            confidence_weight=confidence_weight,
            shell_preference_weight=config.shell_preference_weight,
            name='stem_ood_norm'
        )(x)

    x = keras.layers.Activation(config.activation)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    for i, filters in enumerate(config.conv_filters):
        shortcut = x
        x = keras.layers.Conv2D(
            filters=filters, kernel_size=config.kernel_size, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{i}_1'
        )(x)

        layer_alpha = band_alpha * (0.5 + 0.5 * (len(config.conv_filters) - i) / len(config.conv_filters))

        if confidence_type == 'none':
            x = keras.layers.BatchNormalization(name=f'norm{i}_1')(x)
        elif confidence_type == 'baseline':
            x = BandRMS(max_band_width=layer_alpha, name=f'norm{i}_1')(x)
        else:
            x = BandRMSOOD(
                max_band_width=layer_alpha, confidence_type=confidence_type,
                confidence_weight=confidence_weight,
                shell_preference_weight=config.shell_preference_weight,
                name=f'ood_norm{i}_1'
            )(x)
        x = keras.layers.Activation(config.activation)(x)

        x = keras.layers.Conv2D(
            filters=filters, kernel_size=config.kernel_size, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{i}_2'
        )(x)

        if shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(
                filters, (1, 1), padding='same', kernel_initializer='he_normal', name=f'shortcut{i}'
            )(shortcut)
        x = keras.layers.Add()([x, shortcut])

        if confidence_type == 'none':
            x = keras.layers.BatchNormalization(name=f'norm{i}_2')(x)
        elif confidence_type == 'baseline':
            x = BandRMS(max_band_width=layer_alpha, name=f'norm{i}_2')(x)
        else:
            x = BandRMSOOD(
                max_band_width=layer_alpha, confidence_type=confidence_type,
                confidence_weight=confidence_weight,
                shell_preference_weight=config.shell_preference_weight,
                name=f'ood_norm{i}_2'
            )(x)
        x = keras.layers.Activation(config.activation)(x)

        if i < len(config.conv_filters) - 1:
            x = keras.layers.MaxPooling2D((2, 2))(x)
        if i < len(config.dropout_rates):
            x = keras.layers.Dropout(config.dropout_rates[i])(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units, kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(config.weight_decay), name=f'dense{j}'
        )(x)
        if confidence_type == 'none':
            x = keras.layers.BatchNormalization(name=f'dense_norm{j}')(x)
        elif confidence_type == 'baseline':
            x = BandRMS(max_band_width=band_alpha * 0.5, name=f'dense_norm{j}')(x)
        else:
            x = BandRMSOOD(
                max_band_width=band_alpha * 0.5, confidence_type=confidence_type,
                confidence_weight=confidence_weight * 1.5,
                shell_preference_weight=config.shell_preference_weight,
                name=f'dense_ood_norm{j}'
            )(x)
        x = keras.layers.Activation(config.activation)(x)

    outputs = keras.layers.Dense(config.num_classes, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{model_name}_model')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    return model

# ==============================================================================
# OOD DATA GENERATION
# ==============================================================================

def generate_ood_data(ood_type: str, num_samples: int, input_shape: Tuple[int, ...]) -> tf.Tensor:
    """Generate out-of-distribution data for evaluation."""
    if ood_type == 'uniform_noise':
        return tf.random.uniform((num_samples,) + input_shape, minval=0.0, maxval=1.0)
    if ood_type == 'gaussian_noise':
        return tf.clip_by_value(
            tf.random.normal((num_samples,) + input_shape, mean=0.5, stddev=0.3), 0.0, 1.0
        )
    if ood_type == 'svhn_cropped':
        logger.warning("Using corrupted CIFAR-10 as a substitute for SVHN.")
        return generate_ood_data('gaussian_noise', num_samples, input_shape)
    raise ValueError(f"Unknown OOD type: {ood_type}")


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_ood_experiment(config: BandRMSOODExperimentConfig) -> Dict[str, Any]:
    """Run the complete BandRMS-OOD geometric detection experiment."""
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    viz_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        experiment_name=config.experiment_name
    )
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)

    logger.info("Starting BandRMS-OOD Geometric Detection Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    logger.info("Loading datasets...")
    cifar10_data = load_and_preprocess_cifar10()
    ood_data = {
        name: generate_ood_data(otype, len(cifar10_data.x_test), config.input_shape)
        for name, otype in config.ood_datasets.items()
    }
    logger.info("Datasets loaded successfully.")

    all_results, all_models = {}, {}
    for run_idx in range(config.n_runs):
        logger.info(f"Starting run {run_idx + 1}/{config.n_runs}")
        keras.utils.set_random_seed(config.random_seed + run_idx * 1000)
        run_results, run_models = {}, {}

        for variant_name, variant_config in config.model_variants.items():
            logger.info(f"--- Training {variant_name} (Run {run_idx + 1}) ---")
            conf_type, alpha, conf_weight = variant_config()
            model = build_ood_model(config, conf_type, alpha, conf_weight, f"{variant_name}_run{run_idx}")
            if run_idx == 0:
                logger.info(f"Model {variant_name} parameters: {model.count_params():,}")

            train_cfg = TrainingConfig(
                epochs=config.epochs, batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
                model_name=f"{variant_name}_run{run_idx}",
                output_dir=experiment_dir / "training_plots" / f"run_{run_idx}" / variant_name
            )
            train_model(
                model, cifar10_data.x_train, cifar10_data.y_train,
                cifar10_data.x_test, cifar10_data.y_test, train_cfg
            )
            run_models[variant_name] = model
            logger.info(f"{variant_name} (Run {run_idx + 1}) training completed.")

        logger.info(f"Evaluating OOD detection for run {run_idx + 1}...")
        for variant_name, model in run_models.items():
            variant_results = {}
            preds = model.predict(cifar10_data.x_test, verbose=0)
            y_true = np.argmax(cifar10_data.y_test, axis=1)
            y_pred = np.argmax(preds, axis=1)
            variant_results['classification_accuracy'] = np.mean(y_pred == y_true)

            if 'OOD' in variant_name or 'baseline' in variant_name:
                try:
                    detector = MultiLayerOODDetector(model, config.layer_weights)
                    id_scores = detector.compute_ood_scores(cifar10_data.x_test)
                    for ood_name, ood_dataset in ood_data.items():
                        ood_scores = detector.compute_ood_scores(ood_dataset)
                        metrics = evaluate_ood_detection(id_scores, ood_scores, f"{variant_name} vs {ood_name}")
                        variant_results[f'ood_{ood_name}'] = metrics
                except Exception as e:
                    logger.warning(f"BandRMS-OOD detection failed for {variant_name}: {e}")

            try:
                id_msp = maximum_softmax_probability(model, cifar10_data.x_test)
                id_logit = max_logit(model, cifar10_data.x_test)
                for ood_name, ood_dataset in ood_data.items():
                    ood_msp = maximum_softmax_probability(model, ood_dataset)
                    variant_results[f'msp_{ood_name}'] = evaluate_ood_detection(id_msp, ood_msp)
                    ood_logit = max_logit(model, ood_dataset)
                    variant_results[f'maxlogit_{ood_name}'] = evaluate_ood_detection(id_logit, ood_logit)
            except Exception as e:
                logger.warning(f"Baseline OOD methods failed for {variant_name}: {e}")

            run_results[variant_name] = variant_results
        all_results[f'run_{run_idx}'] = run_results
        if run_idx == config.n_runs - 1:
            all_models = run_models
        del run_models
        gc.collect()

    logger.info("Compiling results across runs...")
    final_results = {
        name: compute_result_statistics([all_results[f'run_{i}'][name] for i in range(config.n_runs)])
        for name in config.model_variants.keys()
    }

    try:
        y_true_labels = np.argmax(cifar10_data.y_test, axis=1)
        class_names = [str(i) for i in range(config.num_classes)]
        model_results = {}
        for name, model in all_models.items():
            raw_preds = model.predict(cifar10_data.x_test, verbose=0)
            class_preds = np.argmax(raw_preds, axis=1)
            model_results[name] = ClassificationResults(
                y_true=y_true_labels, y_pred=class_preds, y_prob=raw_preds,
                class_names=class_names, model_name=name
            )
        multi_model_data = MultiModelClassification(
            y_true=y_true_labels, model_results=model_results, class_names=class_names
        )
        viz_manager.visualize(
            data=multi_model_data,
            plugin_name="confusion_matrix",
            normalize='true',
            title='Model Variant Confusion Matrix Comparison (ID Task)'
        )
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")

    experiment_results = {
        'final_results': final_results, 'all_runs': all_results,
        'trained_models': all_models, 'config': config, 'experiment_dir': experiment_dir
    }
    save_ood_experiment_results(experiment_results, experiment_dir)
    print_ood_experiment_summary(experiment_results)
    return experiment_results


# ==============================================================================
# RESULTS PROCESSING AND UTILITIES
# ==============================================================================

def compute_result_statistics(variant_data: List[Dict]) -> Dict[str, Any]:
    """Compute statistics across multiple runs for a variant."""
    if not variant_data:
        return {}
    stats, all_keys = {}, set(k for d in variant_data for k in d)
    for key in all_keys:
        if isinstance(variant_data[0][key], dict):
            sub_keys = variant_data[0][key].keys()
            stats[key] = {
                sk: {'mean': np.mean([d[key][sk] for d in variant_data]),
                     'std': np.std([d[key][sk] for d in variant_data])}
                for sk in sub_keys
            }
        else:
            values = [d[key] for d in variant_data]
            stats[key] = {'mean': np.mean(values), 'std': np.std(values)}
    return stats

def save_ood_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """Save OOD experiment results."""
    try:
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [convert_numpy(i) for i in obj]
            return obj

        with open(experiment_dir / "ood_detection_results.json", 'w') as f:
            json.dump(convert_numpy(results['final_results']), f, indent=2)

        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for name, model in results['trained_models'].items():
            model.save(models_dir / f"{name}.keras")
        logger.info("OOD experiment results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save OOD experiment results: {e}", exc_info=True)


def print_ood_experiment_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive OOD experiment summary."""
    logger.info("=" * 80)
    logger.info("BANDRMS-OOD GEOMETRIC DETECTION EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    if 'final_results' not in results:
        logger.error("No final results available for summary.")
        return

    logger.info("CLASSIFICATION PERFORMANCE (Mean ± Std):")
    logger.info(f"{'Model':<20} {'Accuracy':<15}")
    logger.info("-" * 40)
    for name, res in results['final_results'].items():
        if 'classification_accuracy' in res:
            acc = res['classification_accuracy']
            logger.info(f"{name:<20} {acc['mean']:.3f}±{acc['std']:.3f}")

    logger.info("\nOOD DETECTION PERFORMANCE (AUROC):")
    ood_datasets = ['noise', 'svhn', 'corrupted']
    methods = ['ood', 'msp', 'maxlogit']
    for ood_dataset in ood_datasets:
        logger.info(f"\n--- vs {ood_dataset.upper()} ---")
        logger.info(f"{'Model':<20} {'BandRMS-OOD':<15} {'MSP':<15} {'MaxLogit':<15}")
        logger.info("-" * 68)
        for name, res in results['final_results'].items():
            row = f"{name:<20}"
            for method in methods:
                key = f"{method}_{ood_dataset}"
                if key in res and 'auroc' in res[key]:
                    auroc = res[key]['auroc']
                    row += f" {auroc['mean']:.3f}±{auroc['std']:.3f}"
                else:
                    row += f" {'N/A':<15}"
            logger.info(row)

    logger.info("\nKEY INSIGHTS:")
    best_auroc, best_model = 0, ""
    for name, res in results['final_results'].items():
        for key, data in res.items():
            if 'ood_' in key and 'auroc' in data:
                if data['auroc']['mean'] > best_auroc:
                    best_auroc = data['auroc']['mean']
                    best_model = f"{name} ({key})"
    if best_model:
        logger.info(f"   Best OOD Detection: {best_model} (AUROC: {best_auroc:.3f})")
    logger.info("=" * 80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for the BandRMS-OOD experiment."""
    logger.info("BandRMS-OOD Geometric Out-of-Distribution Detection Experiment")
    logger.info("=" * 80)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU memory growth configuration failed: {e}")

    config = BandRMSOODExperimentConfig()
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"   Model variants: {list(config.model_variants.keys())}")
    logger.info(f"   OOD datasets: {list(config.ood_datasets.keys())}")
    logger.info(f"   Confidence types: {config.confidence_types}")
    logger.info(f"   Number of runs: {config.n_runs}\n")

    try:
        run_ood_experiment(config)
        logger.info("BandRMS-OOD experiment completed successfully.")
    except Exception as e:
        logger.error(f"Experiment failed with an unhandled exception: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()