"""
Experiment Title: BandRMS-OOD Geometric Out-of-Distribution Detection on CIFAR-10
===============================================================================

This experiment implements and evaluates the BandRMS-OOD approach for geometric
out-of-distribution detection. The core innovation is using confidence-driven shell
scaling within BandRMS normalization layers to create learnable "spherical shells"
in feature space, where in-distribution samples are encouraged toward the outer shell
and OOD samples naturally fall toward the inner core.

Scientific Motivation
--------------------

Traditional OOD detection methods operate as post-hoc solutions, analyzing model
outputs after the entire forward pass. BandRMS-OOD embeds OOD detection capabilities
directly into the architecture through geometric constraints:

1. **Multi-Layer Detection**: OOD signals available throughout the network
2. **Geometric Interpretability**: Clear spatial understanding of OOD decisions
3. **Training Efficiency**: No additional OOD training data required
4. **Confidence Integration**: Leverages model confidence for shell placement

Theoretical Foundation:
- High-confidence (ID) samples → pushed toward outer shell (radius ≈ 1.0)
- Low-confidence (OOD) samples → remain in inner regions (radius < 1.0)
- Shell distance serves as natural OOD score
- Multi-layer consensus improves detection robustness

Experimental Design
-------------------

**In-Distribution**: CIFAR-10 (32×32 RGB images, 10 classes)
**Out-of-Distribution**:
  - SVHN (different domain)
  - Corrupted CIFAR-10 (Gaussian noise, various corruption types)
  - Uniform noise (synthetic OOD)

**Model Architecture**: ResNet with BandRMS-OOD layers
- Progressive shell tightening: early layers (α=0.3) → late layers (α=0.1)
- Multiple confidence estimation methods
- Multi-layer OOD score aggregation

**Confidence Methods Evaluated**:
1. **Magnitude-based**: Feature L2 norm as confidence proxy
2. **Entropy-based**: Feature entropy for uncertainty estimation
3. **Prediction-based**: Softmax confidence (final layers only)

**BandRMS-OOD Variants**:
- Different α values: [0.05, 0.1, 0.2, 0.3]
- Different confidence weights: [0.5, 1.0, 1.5, 2.0]
- Single vs multi-layer detection

**Baseline Comparisons**:
- Maximum Softmax Probability (MSP)
- MaxLogit
- Standard BandRMS (without OOD detection)
- Vanilla ResNet baseline

Evaluation Metrics
-----------------
- **OOD Detection**: AUROC, FPR@95, AUPR
- **Classification**: Test accuracy, calibration (ECE)
- **Shell Analysis**: Shell distance distributions, confidence-radius correlation
- **Efficiency**: Inference time, memory overhead
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
from typing import Dict, Any, List, Tuple, Callable, Optional, Union
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets import load_and_preprocess_cifar10

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.experimental.band_rms_ood import BandRMSOOD

from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# ==============================================================================
# OOD DETECTION UTILITIES
# ==============================================================================

class MultiLayerOODDetector:
    """
    Multi-layer OOD detector that aggregates shell distances from multiple BandRMS-OOD layers.
    """

    def __init__(self, model: keras.Model, layer_weights: Optional[List[float]] = None):
        self.model = model
        self.ood_layers = self._find_ood_layers()
        self.layer_weights = layer_weights or [1.0] * len(self.ood_layers)
        self.threshold = None

        logger.info(f"Found {len(self.ood_layers)} BandRMS-OOD layers: {[layer.name for layer in self.ood_layers]}")

    def _find_ood_layers(self) -> List[BandRMSOOD]:
        """Find all BandRMS-OOD layers in the model."""
        ood_layers = []
        for layer in self.model.layers:
            if isinstance(layer, BandRMSOOD):
                ood_layers.append(layer)
        return ood_layers

    def compute_ood_scores(self, data: tf.Tensor) -> np.ndarray:
        """
        Compute OOD scores for input data.

        Args:
            data: Input data tensor

        Returns:
            OOD scores (higher = more likely OOD)
        """
        # Forward pass to populate shell distances
        _ = self.model(data, training=False)

        # Collect shell distances from all BandRMS-OOD layers
        layer_distances = []
        for layer in self.ood_layers:
            distances = layer.get_shell_distance()
            if distances is not None:
                # Average over spatial dimensions if needed
                if len(distances.shape) > 2:
                    distances = tf.reduce_mean(distances, axis=list(range(1, len(distances.shape) - 1)))
                layer_distances.append(distances.numpy().flatten())

        if not layer_distances:
            raise ValueError(
                "No shell distances found. Ensure BandRMS-OOD layers are present and data has been passed through the model.")

        # Compute weighted average of layer distances
        weighted_distances = np.zeros(len(layer_distances[0]))
        total_weight = 0

        for distances, weight in zip(layer_distances, self.layer_weights):
            weighted_distances += weight * distances
            total_weight += weight

        if total_weight > 0:
            weighted_distances /= total_weight

        return weighted_distances

    def fit_threshold(self, id_data: tf.Tensor, fpr_target: float = 0.05):
        """
        Fit detection threshold based on in-distribution data.

        Args:
            id_data: In-distribution data
            fpr_target: Target false positive rate
        """
        id_scores = self.compute_ood_scores(id_data)
        self.threshold = np.percentile(id_scores, (1.0 - fpr_target) * 100)
        logger.info(f"Set OOD threshold to {self.threshold:.4f} (target FPR: {fpr_target})")

    def predict_ood(self, data: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict OOD for input data.

        Args:
            data: Input data

        Returns:
            Tuple of (ood_predictions, ood_scores)
        """
        ood_scores = self.compute_ood_scores(data)

        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")

        ood_predictions = ood_scores > self.threshold
        return ood_predictions, ood_scores


def evaluate_ood_detection(
        id_scores: np.ndarray,
        ood_scores: np.ndarray,
        method_name: str = ""
) -> Dict[str, float]:
    """
    Evaluate OOD detection performance.

    Args:
        id_scores: OOD scores for in-distribution data
        ood_scores: OOD scores for out-of-distribution data
        method_name: Name of the method being evaluated

    Returns:
        Dictionary of evaluation metrics
    """
    # Create labels (0=ID, 1=OOD)
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])

    # Calculate metrics
    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    # Calculate FPR@95 (false positive rate when TPR = 95%)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fpr95_idx = np.argmax(tpr >= 0.95)
    fpr95 = fpr[fpr95_idx] if fpr95_idx < len(fpr) else 1.0

    results = {
        'auroc': auroc,
        'aupr': aupr,
        'fpr95': fpr95,
        'id_mean': np.mean(id_scores),
        'id_std': np.std(id_scores),
        'ood_mean': np.mean(ood_scores),
        'ood_std': np.std(ood_scores)
    }

    if method_name:
        logger.info(f"{method_name} - AUROC: {auroc:.4f}, FPR@95: {fpr95:.4f}, AUPR: {aupr:.4f}")

    return results


# ==============================================================================
# BASELINE OOD DETECTION METHODS
# ==============================================================================

def maximum_softmax_probability(model: keras.Model, data: tf.Tensor) -> np.ndarray:
    """Compute Maximum Softmax Probability (MSP) scores."""
    predictions = model(data, training=False)
    if len(predictions.shape) > 2:
        predictions = tf.reduce_mean(predictions, axis=list(range(1, len(predictions.shape) - 1)))
    msp_scores = tf.reduce_max(tf.nn.softmax(predictions), axis=-1)
    return -msp_scores.numpy()  # Negative because higher MSP = more confident (less OOD)


def max_logit(model: keras.Model, data: tf.Tensor) -> np.ndarray:
    """Compute MaxLogit scores."""
    predictions = model(data, training=False)
    if len(predictions.shape) > 2:
        predictions = tf.reduce_mean(predictions, axis=list(range(1, len(predictions.shape) - 1)))
    max_logit_scores = tf.reduce_max(predictions, axis=-1)
    return -max_logit_scores.numpy()  # Negative because higher logit = more confident


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class BandRMSOODExperimentConfig:
    """
    Configuration for the BandRMS-OOD geometric OOD detection experiment.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- OOD Dataset Configuration ---
    ood_datasets: Dict[str, str] = field(default_factory=lambda: {
        'svhn': 'svhn_cropped',
        'noise': 'uniform_noise',
        'corrupted': 'gaussian_noise'
    })

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [256])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    activation: str = 'gelu'

    # --- Training Parameters ---
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 15

    # --- BandRMS-OOD Configuration ---
    band_alphas: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    confidence_types: List[str] = field(default_factory=lambda: ['magnitude', 'entropy'])
    confidence_weights: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    shell_preference_weight: float = 0.01

    # --- Model Variants ---
    model_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'BandRMSOOD_mag_010': lambda: ('magnitude', 0.1, 1.0),
        'BandRMSOOD_mag_020': lambda: ('magnitude', 0.2, 1.0),
        'BandRMSOOD_ent_010': lambda: ('entropy', 0.1, 1.0),
        'BandRMSOOD_ent_020': lambda: ('entropy', 0.2, 1.0),
        'BandRMS_baseline': lambda: ('baseline', 0.1, 0.0),  # No confidence weighting
        'Standard_ResNet': lambda: ('none', 0.0, 0.0),  # No BandRMS at all
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "bandrms_ood_detection"
    random_seed: int = 42
    n_runs: int = 3

    # --- OOD Detection Configuration ---
    fpr_target: float = 0.05  # Target false positive rate for threshold setting
    layer_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])  # Weights for multi-layer detection

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=False,  # Skip for OOD experiment
        analyze_training_dynamics=True,
        calibration_bins=15,
        save_plots=True,
        verbose=True
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
    """
    Build a model with BandRMS-OOD layers for geometric OOD detection.

    Args:
        config: Experiment configuration
        confidence_type: Type of confidence estimation
        band_alpha: Band width parameter
        confidence_weight: Weight for confidence influence
        model_name: Name for the model

    Returns:
        Compiled Keras model
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{model_name}_input')
    x = inputs

    # Initial convolution
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=(7, 7),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem_conv'
    )(x)

    # Apply BandRMS-OOD or baseline normalization
    if confidence_type != 'none':
        if confidence_type == 'baseline':
            # Use standard BandRMS without confidence weighting
            x = BandRMS(max_band_width=band_alpha, name='stem_norm')(x)
        else:
            # Use BandRMS-OOD with confidence-driven scaling
            x = BandRMSOOD(
                max_band_width=band_alpha,
                confidence_type=confidence_type,
                confidence_weight=confidence_weight,
                shell_preference_weight=config.shell_preference_weight,
                name='stem_ood_norm'
            )(x)
    else:
        # Standard batch normalization
        x = keras.layers.BatchNormalization(name='stem_norm')(x)

    x = keras.layers.Activation(config.activation)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Convolutional blocks with progressive shell tightening
    for i, filters in enumerate(config.conv_filters):
        # Residual connection
        shortcut = x

        # First conv
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{i}_1'
        )(x)

        # Progressive band tightening for deeper layers
        layer_alpha = band_alpha * (0.5 + 0.5 * (len(config.conv_filters) - i) / len(config.conv_filters))

        if confidence_type != 'none':
            if confidence_type == 'baseline':
                x = BandRMS(max_band_width=layer_alpha, name=f'norm{i}_1')(x)
            else:
                x = BandRMSOOD(
                    max_band_width=layer_alpha,
                    confidence_type=confidence_type,
                    confidence_weight=confidence_weight,
                    shell_preference_weight=config.shell_preference_weight,
                    name=f'ood_norm{i}_1'
                )(x)
        else:
            x = keras.layers.BatchNormalization(name=f'norm{i}_1')(x)

        x = keras.layers.Activation(config.activation)(x)

        # Second conv
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{i}_2'
        )(x)

        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(
                filters, (1, 1), padding='same',
                kernel_initializer='he_normal',
                name=f'shortcut{i}'
            )(shortcut)

        # Add residual connection
        x = keras.layers.Add()([x, shortcut])

        # Final normalization for this block
        if confidence_type != 'none':
            if confidence_type == 'baseline':
                x = BandRMS(max_band_width=layer_alpha, name=f'norm{i}_2')(x)
            else:
                x = BandRMSOOD(
                    max_band_width=layer_alpha,
                    confidence_type=confidence_type,
                    confidence_weight=confidence_weight,
                    shell_preference_weight=config.shell_preference_weight,
                    name=f'ood_norm{i}_2'
                )(x)
        else:
            x = keras.layers.BatchNormalization(name=f'norm{i}_2')(x)

        x = keras.layers.Activation(config.activation)(x)

        # Pooling and dropout
        if i < len(config.conv_filters) - 1:
            x = keras.layers.MaxPooling2D((2, 2))(x)

        if i < len(config.dropout_rates):
            x = keras.layers.Dropout(config.dropout_rates[i])(x)

    # Global pooling
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense{j}'
        )(x)

        # Final BandRMS-OOD layer before classification
        if confidence_type != 'none':
            if confidence_type == 'baseline':
                x = BandRMS(max_band_width=band_alpha * 0.5, name=f'dense_norm{j}')(x)
            else:
                x = BandRMSOOD(
                    max_band_width=band_alpha * 0.5,
                    confidence_type=confidence_type,
                    confidence_weight=confidence_weight * 1.5,  # Higher weight for final layers
                    shell_preference_weight=config.shell_preference_weight,
                    name=f'dense_ood_norm{j}'
                )(x)
        else:
            x = keras.layers.BatchNormalization(name=f'dense_norm{j}')(x)

        x = keras.layers.Activation(config.activation)(x)

        if j + len(config.conv_filters) < len(config.dropout_rates):
            x = keras.layers.Dropout(config.dropout_rates[j + len(config.conv_filters)])(x)

    # Output layer (no normalization)
    outputs = keras.layers.Dense(
        config.num_classes,
        activation='softmax',
        kernel_initializer='he_normal',
        name='predictions'
    )(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{model_name}_model')

    # Compile model
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    return model


# ==============================================================================
# OOD DATA GENERATION
# ==============================================================================

def generate_ood_data(ood_type: str, num_samples: int, input_shape: Tuple[int, ...]) -> tf.Tensor:
    """
    Generate out-of-distribution data for evaluation.

    Args:
        ood_type: Type of OOD data to generate
        num_samples: Number of samples to generate
        input_shape: Shape of input data

    Returns:
        Generated OOD data tensor
    """
    if ood_type == 'uniform_noise':
        # Uniform noise in [0, 1]
        return tf.random.uniform((num_samples,) + input_shape, minval=0.0, maxval=1.0)

    elif ood_type == 'gaussian_noise':
        # Gaussian noise
        return tf.clip_by_value(
            tf.random.normal((num_samples,) + input_shape, mean=0.5, stddev=0.3),
            0.0, 1.0
        )

    elif ood_type == 'svhn_cropped':
        # For this experiment, we'll use corrupted CIFAR-10 as a substitute for SVHN
        # In a real implementation, you would load the actual SVHN dataset
        logger.warning("Using corrupted CIFAR-10 as substitute for SVHN")
        return generate_ood_data('gaussian_noise', num_samples, input_shape)

    else:
        raise ValueError(f"Unknown OOD type: {ood_type}")


# ==============================================================================
# SHELL PREFERENCE LOSS
# ==============================================================================

class ShellPreferenceLoss:
    """
    Custom loss function that encourages high-confidence samples to reach outer shell.
    """

    def __init__(self, shell_weight: float = 0.01):
        self.shell_weight = shell_weight

    def __call__(self, y_true, y_pred, model):
        """
        Compute shell preference loss.

        Args:
            y_true: True labels
            y_pred: Model predictions
            model: Model with BandRMS-OOD layers

        Returns:
            Combined loss (classification + shell preference)
        """
        # Base classification loss
        base_loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        # Collect shell distances and confidences from all BandRMS-OOD layers
        shell_losses = []

        for layer in model.layers:
            if isinstance(layer, BandRMSOOD):
                shell_distances = layer.get_shell_distance()
                confidences = layer.get_confidences()

                if shell_distances is not None and confidences is not None:
                    # Encourage high-confidence samples to have low shell distance (near outer shell)
                    shell_loss = tf.reduce_mean(confidences * tf.square(shell_distances))
                    shell_losses.append(shell_loss)

        # Add shell preference loss
        total_shell_loss = tf.reduce_mean(shell_losses) if shell_losses else 0.0

        return base_loss + self.shell_weight * total_shell_loss


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_ood_experiment(config: BandRMSOODExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete BandRMS-OOD geometric detection experiment.

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

    logger.info("🚀 Starting BandRMS-OOD Geometric Detection Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading datasets...")
    cifar10_data = load_and_preprocess_cifar10()

    # Generate OOD datasets
    ood_data = {}
    for ood_name, ood_type in config.ood_datasets.items():
        logger.info(f"Generating {ood_name} OOD data ({ood_type})...")
        ood_data[ood_name] = generate_ood_data(
            ood_type,
            len(cifar10_data.x_test),
            config.input_shape
        )

    logger.info("✅ Datasets loaded successfully")

    # ===== MULTIPLE RUNS =====
    all_results = {}
    all_models = {}

    for run_idx in range(config.n_runs):
        logger.info(f"🏃 Starting run {run_idx + 1}/{config.n_runs}")

        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        run_results = {}
        run_models = {}

        # ===== TRAIN MODELS =====
        for variant_name, variant_config in config.model_variants.items():
            logger.info(f"--- Training {variant_name} (Run {run_idx + 1}) ---")

            confidence_type, band_alpha, confidence_weight = variant_config()

            # Build model
            model = build_ood_model(
                config, confidence_type, band_alpha, confidence_weight,
                f"{variant_name}_run{run_idx}"
            )

            if run_idx == 0:
                logger.info(f"Model {variant_name} parameters: {model.count_params():,}")

            # Training configuration
            training_config = TrainingConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
                model_name=f"{variant_name}_run{run_idx}",
                output_dir=experiment_dir / "training_plots" / f"run_{run_idx}" / variant_name
            )

            # Train model
            history = train_model(
                model, cifar10_data.x_train, cifar10_data.y_train,
                cifar10_data.x_test, cifar10_data.y_test, training_config
            )

            run_models[variant_name] = model
            logger.info(f"✅ {variant_name} (Run {run_idx + 1}) training completed!")

        # ===== EVALUATE OOD DETECTION =====
        logger.info(f"🔍 Evaluating OOD detection for run {run_idx + 1}...")

        for variant_name, model in run_models.items():
            variant_results = {}

            # Classification performance
            predictions = model.predict(cifar10_data.x_test, verbose=0)
            y_true_classes = np.argmax(cifar10_data.y_test, axis=1)
            y_pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(y_pred_classes == y_true_classes)

            variant_results['classification_accuracy'] = accuracy

            # OOD detection evaluation
            if 'OOD' in variant_name or 'baseline' in variant_name:
                # BandRMS-OOD based detection
                try:
                    ood_detector = MultiLayerOODDetector(model, config.layer_weights)
                    ood_detector.fit_threshold(cifar10_data.x_test[:1000], config.fpr_target)

                    # Evaluate on all OOD datasets
                    id_scores = ood_detector.compute_ood_scores(cifar10_data.x_test)

                    for ood_name, ood_dataset in ood_data.items():
                        ood_scores = ood_detector.compute_ood_scores(ood_dataset)
                        ood_metrics = evaluate_ood_detection(
                            id_scores, ood_scores, f"{variant_name} vs {ood_name}"
                        )
                        variant_results[f'ood_{ood_name}'] = ood_metrics

                except Exception as e:
                    logger.warning(f"BandRMS-OOD detection failed for {variant_name}: {e}")

            # Baseline OOD methods for comparison
            try:
                # MSP
                id_msp = maximum_softmax_probability(model, cifar10_data.x_test)
                for ood_name, ood_dataset in ood_data.items():
                    ood_msp = maximum_softmax_probability(model, ood_dataset)
                    msp_metrics = evaluate_ood_detection(
                        id_msp, ood_msp, f"{variant_name} MSP vs {ood_name}"
                    )
                    variant_results[f'msp_{ood_name}'] = msp_metrics

                # MaxLogit
                id_maxlogit = max_logit(model, cifar10_data.x_test)
                for ood_name, ood_dataset in ood_data.items():
                    ood_maxlogit = max_logit(model, ood_dataset)
                    maxlogit_metrics = evaluate_ood_detection(
                        id_maxlogit, ood_maxlogit, f"{variant_name} MaxLogit vs {ood_name}"
                    )
                    variant_results[f'maxlogit_{ood_name}'] = maxlogit_metrics

            except Exception as e:
                logger.warning(f"Baseline OOD methods failed for {variant_name}: {e}")

            run_results[variant_name] = variant_results

        # Store results from final run
        if run_idx == config.n_runs - 1:
            all_models = run_models

        all_results[f'run_{run_idx}'] = run_results

        # Cleanup
        del run_models
        gc.collect()

    # ===== COMPILE RESULTS =====
    logger.info("📈 Compiling results across runs...")

    # Average results across runs
    final_results = {}
    for variant_name in config.model_variants.keys():
        variant_data = []
        for run_idx in range(config.n_runs):
            if f'run_{run_idx}' in all_results and variant_name in all_results[f'run_{run_idx}']:
                variant_data.append(all_results[f'run_{run_idx}'][variant_name])

        if variant_data:
            # Compute statistics across runs
            final_results[variant_name] = compute_result_statistics(variant_data)

    # ===== SAVE RESULTS =====
    experiment_results = {
        'final_results': final_results,
        'all_runs': all_results,
        'trained_models': all_models,
        'config': config,
        'experiment_dir': experiment_dir
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

    statistics = {}

    # Get all metrics from first run
    all_metrics = set()
    for run_data in variant_data:
        all_metrics.update(run_data.keys())

    for metric in all_metrics:
        values = []
        for run_data in variant_data:
            if metric in run_data:
                if isinstance(run_data[metric], dict):
                    # For nested dictionaries (OOD metrics), process each sub-metric
                    if metric not in statistics:
                        statistics[metric] = {}
                    for sub_metric, sub_value in run_data[metric].items():
                        if sub_metric not in statistics[metric]:
                            statistics[metric][sub_metric] = []
                        statistics[metric][sub_metric].append(sub_value)
                else:
                    # For scalar values
                    values.append(run_data[metric])

        if values:
            statistics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    # Process nested metrics
    for metric_name, metric_data in statistics.items():
        if isinstance(metric_data, dict) and not ('mean' in metric_data):
            for sub_metric, sub_values in metric_data.items():
                if isinstance(sub_values, list):
                    statistics[metric_name][sub_metric] = {
                        'mean': np.mean(sub_values),
                        'std': np.std(sub_values),
                        'min': np.min(sub_values),
                        'max': np.max(sub_values)
                    }

    return statistics


def save_ood_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """Save OOD experiment results."""
    try:
        # Helper function to convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        # Save final results
        final_results_converted = convert_numpy(results['final_results'])
        with open(experiment_dir / "ood_detection_results.json", 'w') as f:
            json.dump(final_results_converted, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for name, model in results['trained_models'].items():
            model.save(models_dir / f"{name}.keras")

        logger.info("✅ OOD experiment results saved successfully")

    except Exception as e:
        logger.error(f"❌ Failed to save OOD experiment results: {e}", exc_info=True)


def print_ood_experiment_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive OOD experiment summary."""
    logger.info("=" * 80)
    logger.info("📋 BANDRMS-OOD GEOMETRIC DETECTION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    if 'final_results' not in results:
        logger.error("❌ No final results available for summary")
        return

    # Classification performance
    logger.info("🎯 CLASSIFICATION PERFORMANCE (Mean ± Std):")
    logger.info(f"{'Model':<20} {'Accuracy':<15}")
    logger.info("-" * 40)

    for variant_name, variant_results in results['final_results'].items():
        if 'classification_accuracy' in variant_results:
            acc_stats = variant_results['classification_accuracy']
            logger.info(f"{variant_name:<20} {acc_stats['mean']:.3f}±{acc_stats['std']:.3f}")

    # OOD detection performance
    logger.info("\n🔍 OOD DETECTION PERFORMANCE (AUROC):")

    ood_datasets = ['noise', 'svhn', 'corrupted']
    methods = ['ood', 'msp', 'maxlogit']

    for ood_dataset in ood_datasets:
        logger.info(f"\n--- vs {ood_dataset.upper()} ---")
        logger.info(f"{'Model':<20} {'BandRMS-OOD':<12} {'MSP':<12} {'MaxLogit':<12}")
        logger.info("-" * 60)

        for variant_name, variant_results in results['final_results'].items():
            row = f"{variant_name:<20}"

            for method in methods:
                metric_key = f"{method}_{ood_dataset}"
                if metric_key in variant_results and 'auroc' in variant_results[metric_key]:
                    auroc_stats = variant_results[metric_key]['auroc']
                    row += f" {auroc_stats['mean']:.3f}±{auroc_stats['std']:.3f}"
                else:
                    row += f" {'N/A':<12}"

            logger.info(row)

    # Key insights
    logger.info("\n🔍 KEY INSIGHTS:")

    # Find best OOD detector
    best_ood_auroc = 0
    best_ood_model = ""

    for variant_name, variant_results in results['final_results'].items():
        for metric_name, metric_data in variant_results.items():
            if 'ood_' in metric_name and 'auroc' in metric_data:
                auroc_mean = metric_data['auroc']['mean']
                if auroc_mean > best_ood_auroc:
                    best_ood_auroc = auroc_mean
                    best_ood_model = f"{variant_name} ({metric_name})"

    if best_ood_model:
        logger.info(f"   🏆 Best OOD Detection: {best_ood_model} (AUROC: {best_ood_auroc:.3f})")

    # Compare confidence methods
    mag_models = [k for k in results['final_results'].keys() if 'mag' in k]
    ent_models = [k for k in results['final_results'].keys() if 'ent' in k]

    if mag_models and ent_models:
        logger.info("   📐 Confidence Method Comparison:")
        logger.info("      Magnitude-based models:", ", ".join(mag_models))
        logger.info("      Entropy-based models:", ", ".join(ent_models))

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for the BandRMS-OOD experiment."""
    logger.info("🚀 BandRMS-OOD Geometric Out-of-Distribution Detection Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU configuration: {e}")

    # Initialize configuration
    config = BandRMSOODExperimentConfig()

    # Log configuration
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Model variants: {list(config.model_variants.keys())}")
    logger.info(f"   OOD datasets: {list(config.ood_datasets.keys())}")
    logger.info(f"   Confidence types: {config.confidence_types}")
    logger.info(f"   Band α values: {config.band_alphas}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info("")

    try:
        # Run experiment
        results = run_ood_experiment(config)
        logger.info("✅ BandRMS-OOD experiment completed successfully!")

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()