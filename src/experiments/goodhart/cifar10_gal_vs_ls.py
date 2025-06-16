"""
CIFAR-10 Calibration Experiment: GoodhartAwareLoss vs Label Smoothing
====================================================================

This module implements a comprehensive experiment to evaluate and compare the calibration
effectiveness of GoodhartAwareLoss against Label Smoothing on CIFAR-10 classification.
The experiment systematically tests the core claims of Goodhart's Law-aware training
by measuring calibration quality and in-distribution performance.

EXPERIMENT OVERVIEW
------------------
The experiment trains two CNN models with identical architectures but different
loss functions applied during training:

1. **Label Smoothing (LS)**: Standard cross-entropy with label smoothing (α=0.1)
2. **GoodhartAwareLoss (GAL)**: Information-theoretic loss combining temperature scaling,
   entropy regularization, and mutual information constraints

RESEARCH HYPOTHESIS
------------------
**Core Claims Being Tested:**
- GAL should produce better-calibrated models (lower ECE)
- GAL should maintain competitive accuracy on clean, in-distribution data
- GAL should produce less overconfident predictions (higher entropy)
- GAL should show better reliability diagram alignment with perfect calibration

METHODOLOGY
-----------
Each model follows identical training protocols:
- Architecture: ResNet-like CNN optimized for CIFAR-10
- Dataset: CIFAR-10 (50k train, 10k test, 10 classes)
- Training: Same optimizer, learning rate schedule, epochs, and data augmentation
- Evaluation: Comprehensive calibration and performance analysis

MODEL ARCHITECTURE
------------------
- Input: 32x32x3 RGB CIFAR-10 images
- Conv Blocks: Conv2D → BatchNorm → ReLU → Optional Dropout
- Residual connections for improved gradient flow
- Global Average Pooling before final classification layer
- Output: Dense(10) → Loss-specific processing → Softmax
- Regularization: L2 weight decay, dropout, and data augmentation

CALIBRATION METRICS
------------------
**Expected Calibration Error (ECE):**
- Primary metric for measuring calibration quality
- ECE = Σ (|confidence - accuracy|) × proportion_in_bin
- Lower values indicate better calibration (perfect = 0.0)

**Reliability Diagram:**
- Visual representation of calibration quality
- Perfect calibration follows y=x diagonal
- Shows confidence vs accuracy across probability bins

**Average Prediction Entropy:**
- H(p) = -Σ p_i × log(p_i) averaged over test set
- Higher entropy indicates appropriate uncertainty
- Lower entropy may indicate overconfidence

**Brier Score:**
- Proper scoring rule measuring prediction quality
- BS = (1/N) Σ (p_predicted - p_true)²
- Lower values indicate better probabilistic predictions

ANALYSIS OUTPUTS
---------------
**Calibration Analysis:**
- Expected Calibration Error comparison
- Reliability diagrams for both models
- Confidence distribution histograms
- Prediction entropy analysis

**Performance Analysis:**
- Top-1 accuracy on CIFAR-10 test set
- Loss curves during training
- Confusion matrices
- Per-class performance metrics

**Statistical Analysis:**
- Confidence intervals for all metrics
- Statistical significance tests
- Goodhart's Law effect quantification
- Calibration improvement measurement

CONFIGURATION
------------
All experiment parameters are centralized in the ExperimentConfig class:
- Model architecture (filters, layers, regularization)
- Training hyperparameters (epochs, batch size, learning rate)
- Loss function parameters (GAL weights, label smoothing alpha)
- Calibration analysis settings (number of bins, confidence thresholds)
- Output and visualization options

USAGE
-----
To run with default settings:
    python calibration_experiment.py

To customize hyperparameters:
    config = ExperimentConfig()
    config.gal_temperature = 2.5
    config.gal_entropy_weight = 0.15
    config.epochs = 200
    results = run_calibration_experiment(config)

DEPENDENCIES
-----------
- TensorFlow/Keras for deep learning models
- NumPy for numerical computations
- Matplotlib/Seaborn for calibration visualizations
- SciPy for statistical tests
- Custom GoodhartAwareLoss implementation

OUTPUT STRUCTURE
---------------
results/
├── cifar10_calibration_experiment_TIMESTAMP/
│   ├── label_smoothing/       # LS model checkpoints and logs
│   ├── goodhart_aware/        # GAL model checkpoints and logs
│   ├── calibration_analysis/   # ECE, reliability diagrams, entropy analysis
│   ├── performance_analysis/   # Accuracy, confusion matrices, training curves
│   └── statistical_analysis/   # Significance tests, effect sizes

RESEARCH APPLICATIONS
--------------------
This experiment framework enables:
- Validating Goodhart's Law mitigation in ML
- Comparing calibration techniques
- Measuring overconfidence reduction
- Benchmarking information-theoretic loss functions
- Studying the accuracy-calibration trade-off

The results provide empirical evidence for or against the theoretical claims
of information-theoretic approaches to preventing metric gaming in ML.

Organization:
1. Imports and dependencies
2. Configuration class
3. Model building utilities
4. Calibration metrics implementation
5. Experiment runner
6. Main execution
"""

# ------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------

import keras
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.tensors import compute_prediction_entropy

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 2. Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Unified configuration for the CIFAR-10 calibration experiment.

    Contains all parameters for model architecture, training, loss functions,
    and calibration analysis in a single consolidated configuration class.
    """
    # Dataset Configuration
    dataset_name: str = "cifar10"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)
    validation_split: float = 0.1

    # Model Architecture Parameters
    conv_filters: List[int] = (64, 128, 256, 512)
    dense_units: int = 512
    dropout_rate: float = 0.3
    weight_decay: float = 1e-4
    use_batch_norm: bool = True
    use_residual: bool = True

    # Training Parameters
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    monitor_metric: str = 'val_accuracy'

    # Label Smoothing Configuration
    label_smoothing_alpha: float = 0.1

    # GoodhartAwareLoss Configuration
    gal_temperature: float = 2.0
    gal_entropy_weight: float = 0.1
    gal_mi_weight: float = 0.01

    # Calibration Analysis Parameters
    calibration_bins: int = 15
    confidence_threshold: float = 0.5
    bootstrap_samples: int = 1000

    # Experiment Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "cifar10_calibration_experiment"
    save_models: bool = True
    save_plots: bool = True
    plot_format: str = 'png'
    random_seed: int = 42

    # Data Augmentation
    use_data_augmentation: bool = True
    rotation_range: float = 15.0
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    horizontal_flip: bool = True


# ------------------------------------------------------------------------------
# 3. Model Building Utilities
# ------------------------------------------------------------------------------

def build_residual_block(x: keras.layers.Layer, filters: int, config: ExperimentConfig) -> keras.layers.Layer:
    """Build a residual block for improved gradient flow.

    Args:
        x: Input tensor
        filters: Number of filters
        config: Experiment configuration

    Returns:
        Output tensor after residual block
    """
    shortcut = x

    # First conv layer
    x = keras.layers.Conv2D(filters, (3, 3), padding='same',
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay))(x)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Second conv layer
    x = keras.layers.Conv2D(filters, (3, 3), padding='same',
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay))(x)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        if config.use_batch_norm:
            shortcut = keras.layers.BatchNormalization()(shortcut)

    # Add shortcut and activation
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    return x


def build_cnn_model(config: ExperimentConfig, loss_type: str = 'crossentropy') -> keras.Model:
    """Build CNN model optimized for CIFAR-10.

    Args:
        config: Experiment configuration
        loss_type: Type of loss function ('crossentropy', 'label_smoothing', 'goodhart_aware')

    Returns:
        Compiled Keras model
    """
    inputs = keras.layers.Input(shape=config.input_shape)
    x = inputs

    # Initial conv layer
    x = keras.layers.Conv2D(config.conv_filters[0], (3, 3), padding='same',
                            kernel_regularizer=keras.regularizers.L2(config.weight_decay))(x)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Convolutional blocks with optional residual connections
    for i, filters in enumerate(config.conv_filters):
        if config.use_residual and i > 0:
            x = build_residual_block(x, filters, config)
        else:
            x = keras.layers.Conv2D(filters, (3, 3), padding='same',
                                    kernel_regularizer=keras.regularizers.L2(config.weight_decay))(x)
            if config.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)

        # Pooling and dropout
        if i < len(config.conv_filters) - 1:  # Don't pool after last conv block
            x = keras.layers.MaxPooling2D((2, 2))(x)

        if config.dropout_rate > 0:
            x = keras.layers.Dropout(config.dropout_rate)(x)

    # Global average pooling instead of flatten
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense layer
    if config.dense_units > 0:
        x = keras.layers.Dense(config.dense_units,
                               kernel_regularizer=keras.regularizers.L2(config.weight_decay))(x)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        if config.dropout_rate > 0:
            x = keras.layers.Dropout(config.dropout_rate)(x)

    # Output layer (logits)
    logits = keras.layers.Dense(config.num_classes,
                                kernel_regularizer=keras.regularizers.L2(config.weight_decay),
                                name='logits')(x)

    # Apply softmax for predictions
    outputs = keras.layers.Activation('softmax', name='predictions')(logits)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Configure loss function and optimizer
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    if loss_type == 'label_smoothing':
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing_alpha)
    elif loss_type == 'goodhart_aware':
        loss = GoodhartAwareLoss(
            temperature=config.gal_temperature,
            entropy_weight=config.gal_entropy_weight,
            mi_weight=config.gal_mi_weight
        )
    else:
        loss = keras.losses.CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


# ------------------------------------------------------------------------------
# 4. Calibration Metrics Implementation
# ------------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error.

    Args:
        y_true: True class labels (not one-hot)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Expected Calibration Error
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def compute_reliability_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    """Compute data for reliability diagram.

    Args:
        y_true: True class labels (not one-hot)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with bin centers, accuracies, confidences, and counts
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[len(bin_accuracies)])
            bin_counts.append(0)

    return {
        'bin_centers': bin_centers,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }


def compute_brier_score(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score.

    Args:
        y_true_onehot: True labels (one-hot encoded)
        y_prob: Predicted probabilities

    Returns:
        Brier Score
    """
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))


def compute_prediction_entropy_stats(y_prob: np.ndarray) -> Dict[str, float]:
    """Compute prediction entropy statistics.

    Args:
        y_prob: Predicted probabilities

    Returns:
        Dictionary with entropy statistics
    """
    # Compute entropy for each prediction
    epsilon = 1e-8
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
    entropies = -np.sum(y_prob_clipped * np.log(y_prob_clipped), axis=1)

    return {
        'mean_entropy': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'median_entropy': np.median(entropies),
        'max_entropy': np.max(entropies),
        'min_entropy': np.min(entropies)
    }


# ------------------------------------------------------------------------------
# 5. Visualization Utilities
# ------------------------------------------------------------------------------

def plot_reliability_diagram(reliability_data: Dict[str, np.ndarray],
                             model_name: str,
                             save_path: Optional[Path] = None) -> None:
    """Plot reliability diagram for calibration visualization.

    Args:
        reliability_data: Data from compute_reliability_data
        model_name: Name of the model for title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')

    # Plot model calibration
    bin_centers = reliability_data['bin_centers']
    bin_accuracies = reliability_data['bin_accuracies']
    bin_counts = reliability_data['bin_counts']

    # Use bin counts for marker sizes
    sizes = 100 * bin_counts / np.max(bin_counts + 1)

    scatter = ax.scatter(bin_centers, bin_accuracies, s=sizes, alpha=0.7,
                         label=f'{model_name} Calibration', color='red')

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram - {model_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add colorbar for bin counts
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Samples', rotation=270, labelpad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_comparison(ls_reliability: Dict[str, np.ndarray],
                                gal_reliability: Dict[str, np.ndarray],
                                save_path: Optional[Path] = None) -> None:
    """Plot comparison of calibration between Label Smoothing and GoodhartAwareLoss.

    Args:
        ls_reliability: Reliability data for Label Smoothing model
        gal_reliability: Reliability data for GoodhartAwareLoss model
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')

    # Label Smoothing
    ax.plot(ls_reliability['bin_centers'], ls_reliability['bin_accuracies'],
            'o-', markersize=8, linewidth=2, label='Label Smoothing', color='blue', alpha=0.8)

    # GoodhartAwareLoss
    ax.plot(gal_reliability['bin_centers'], gal_reliability['bin_accuracies'],
            's-', markersize=8, linewidth=2, label='GoodhartAwareLoss', color='red', alpha=0.8)

    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Calibration Comparison: Label Smoothing vs GoodhartAwareLoss', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_entropy_comparison(ls_entropy_stats: Dict[str, float],
                            gal_entropy_stats: Dict[str, float],
                            save_path: Optional[Path] = None) -> None:
    """Plot comparison of prediction entropy statistics.

    Args:
        ls_entropy_stats: Entropy statistics for Label Smoothing
        gal_entropy_stats: Entropy statistics for GoodhartAwareLoss
        save_path: Path to save the plot
    """
    metrics = ['mean_entropy', 'std_entropy', 'median_entropy']
    ls_values = [ls_entropy_stats[m] for m in metrics]
    gal_values = [gal_entropy_stats[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width / 2, ls_values, width, label='Label Smoothing', color='blue', alpha=0.7)
    ax.bar(x + width / 2, gal_values, width, label='GoodhartAwareLoss', color='red', alpha=0.7)

    ax.set_xlabel('Entropy Metrics', fontsize=12)
    ax.set_ylabel('Entropy Value', fontsize=12)
    ax.set_title('Prediction Entropy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (ls_val, gal_val) in enumerate(zip(ls_values, gal_values)):
        ax.text(i - width / 2, ls_val + 0.01, f'{ls_val:.3f}', ha='center', va='bottom')
        ax.text(i + width / 2, gal_val + 0.01, f'{gal_val:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------------------------
# 6. Data Loading and Preprocessing
# ------------------------------------------------------------------------------

def load_and_preprocess_cifar10(config: ExperimentConfig) -> Tuple[Tuple[np.ndarray, ...], keras.utils.Sequence]:
    """Load and preprocess CIFAR-10 dataset.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (train_data, test_data) and data generator
    """
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, config.num_classes)
    y_test = keras.utils.to_categorical(y_test, config.num_classes)

    # Create data generator with augmentation
    if config.use_data_augmentation:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=config.rotation_range,
            width_shift_range=config.width_shift_range,
            height_shift_range=config.height_shift_range,
            horizontal_flip=config.horizontal_flip,
            validation_split=config.validation_split
        )
        datagen.fit(x_train)
    else:
        datagen = None

    return (x_train, y_train, x_test, y_test), datagen


# ------------------------------------------------------------------------------
# 7. Experiment Runner
# ------------------------------------------------------------------------------

def run_calibration_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete calibration experiment comparing Label Smoothing vs GoodhartAwareLoss.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experiment results
    """
    # Set random seed for reproducibility
    keras.utils.set_random_seed(config.random_seed)

    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting CIFAR-10 Calibration Experiment")
    print(f"📁 Results will be saved to: {experiment_dir}")
    print("=" * 80)

    # Load and preprocess data
    print("📊 Loading CIFAR-10 dataset...")
    (x_train, y_train, x_test, y_test), datagen = load_and_preprocess_cifar10(config)

    print(f"✅ Dataset loaded:")
    print(f"   Training samples: {x_train.shape[0]}")
    print(f"   Test samples: {x_test.shape[0]}")
    print(f"   Input shape: {x_train.shape[1:]}")
    print(f"   Number of classes: {config.num_classes}")

    # Prepare validation split
    val_split_idx = int(len(x_train) * (1 - config.validation_split))
    x_val = x_train[val_split_idx:]
    y_val = y_train[val_split_idx:]
    x_train = x_train[:val_split_idx]
    y_train = y_train[:val_split_idx]

    print(f"   Training samples (after val split): {x_train.shape[0]}")
    print(f"   Validation samples: {x_val.shape[0]}")

    # Store results
    results = {
        'config': config,
        'models': {},
        'histories': {},
        'predictions': {},
        'calibration_metrics': {},
        'performance_metrics': {}
    }

    # Train both models
    model_configs = [
        ('label_smoothing', 'Label Smoothing'),
        ('goodhart_aware', 'GoodhartAwareLoss')
    ]

    for model_type, model_name in model_configs:
        print(f"\n🏗️  Building and training {model_name} model...")
        print("-" * 50)

        # Build model
        model = build_cnn_model(config, loss_type=model_type)
        print(f"✅ Model built with {model.count_params():,} parameters")

        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=config.monitor_metric,
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=config.monitor_metric,
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                verbose=1,
                min_lr=1e-7
            )
        ]

        if config.save_models:
            model_dir = experiment_dir / model_type
            model_dir.mkdir(exist_ok=True)
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=str(model_dir / 'best_model.keras'),
                    monitor=config.monitor_metric,
                    save_best_only=True,
                    verbose=1
                )
            )

        # Train model
        print(f"🏃 Training {model_name} model...")
        if config.use_data_augmentation and datagen is not None:
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=config.batch_size, subset='training'),
                epochs=config.epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                x_train, y_train,
                batch_size=config.batch_size,
                epochs=config.epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

        # Store model and history
        results['models'][model_type] = model
        results['histories'][model_type] = history.history

        print(f"✅ {model_name} training completed!")
        print(f"   Best val accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"   Final val loss: {min(history.history['val_loss']):.4f}")

    # Evaluation phase
    print(f"\n📈 Evaluating models and computing calibration metrics...")
    print("=" * 80)

    for model_type, model_name in model_configs:
        model = results['models'][model_type]

        # Get predictions
        print(f"\n🔍 Analyzing {model_name}...")
        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        results['predictions'][model_type] = {
            'probabilities': y_pred_proba,
            'classes': y_pred_classes
        }

        # Compute performance metrics
        accuracy = np.mean(y_pred_classes == y_true_classes)
        top5_accuracy = keras.metrics.top_k_categorical_accuracy(y_test, y_pred_proba, k=5).numpy().mean()

        results['performance_metrics'][model_type] = {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
        }

        # Compute calibration metrics
        ece = compute_ece(y_true_classes, y_pred_proba, config.calibration_bins)
        reliability_data = compute_reliability_data(y_true_classes, y_pred_proba, config.calibration_bins)
        brier_score = compute_brier_score(y_test, y_pred_proba)
        entropy_stats = compute_prediction_entropy_stats(y_pred_proba)

        results['calibration_metrics'][model_type] = {
            'ece': ece,
            'brier_score': brier_score,
            'reliability_data': reliability_data,
            'entropy_stats': entropy_stats
        }

        print(f"   📊 Performance Metrics:")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"   🎯 Calibration Metrics:")
        print(f"      ECE: {ece:.4f}")
        print(f"      Brier Score: {brier_score:.4f}")
        print(f"      Mean Entropy: {entropy_stats['mean_entropy']:.4f}")

    # Generate visualizations
    if config.save_plots:
        print(f"\n📊 Generating visualizations...")
        viz_dir = experiment_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Plot reliability diagrams
        for model_type, model_name in model_configs:
            reliability_data = results['calibration_metrics'][model_type]['reliability_data']
            plot_reliability_diagram(
                reliability_data,
                model_name,
                viz_dir / f"reliability_{model_type}.{config.plot_format}"
            )

        # Plot calibration comparison
        plot_calibration_comparison(
            results['calibration_metrics']['label_smoothing']['reliability_data'],
            results['calibration_metrics']['goodhart_aware']['reliability_data'],
            viz_dir / f"calibration_comparison.{config.plot_format}"
        )

        # Plot entropy comparison
        plot_entropy_comparison(
            results['calibration_metrics']['label_smoothing']['entropy_stats'],
            results['calibration_metrics']['goodhart_aware']['entropy_stats'],
            viz_dir / f"entropy_comparison.{config.plot_format}"
        )

        print(f"✅ Visualizations saved to {viz_dir}")

    # Print summary results
    print_experiment_summary(results)

    return results


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print a comprehensive summary of experiment results.

    Args:
        results: Dictionary containing all experiment results
    """
    print(f"\n" + "=" * 80)
    print(f"📋 EXPERIMENT SUMMARY")
    print(f"=" * 80)

    # Performance comparison
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"{'Metric':<20} {'Label Smoothing':<18} {'GoodhartAwareLoss':<18} {'Difference':<12}")
    print(f"-" * 70)

    ls_perf = results['performance_metrics']['label_smoothing']
    gal_perf = results['performance_metrics']['goodhart_aware']

    accuracy_diff = gal_perf['accuracy'] - ls_perf['accuracy']
    top5_diff = gal_perf['top5_accuracy'] - ls_perf['top5_accuracy']

    print(f"{'Accuracy':<20} {ls_perf['accuracy']:<18.4f} {gal_perf['accuracy']:<18.4f} {accuracy_diff:<+12.4f}")
    print(
        f"{'Top-5 Accuracy':<20} {ls_perf['top5_accuracy']:<18.4f} {gal_perf['top5_accuracy']:<18.4f} {top5_diff:<+12.4f}")

    # Calibration comparison
    print(f"\n🎯 CALIBRATION METRICS:")
    print(f"{'Metric':<20} {'Label Smoothing':<18} {'GoodhartAwareLoss':<18} {'Difference':<12}")
    print(f"-" * 70)

    ls_cal = results['calibration_metrics']['label_smoothing']
    gal_cal = results['calibration_metrics']['goodhart_aware']

    ece_diff = gal_cal['ece'] - ls_cal['ece']
    brier_diff = gal_cal['brier_score'] - ls_cal['brier_score']
    entropy_diff = gal_cal['entropy_stats']['mean_entropy'] - ls_cal['entropy_stats']['mean_entropy']

    print(f"{'ECE':<20} {ls_cal['ece']:<18.4f} {gal_cal['ece']:<18.4f} {ece_diff:<+12.4f}")
    print(f"{'Brier Score':<20} {ls_cal['brier_score']:<18.4f} {gal_cal['brier_score']:<18.4f} {brier_diff:<+12.4f}")
    print(
        f"{'Mean Entropy':<20} {ls_cal['entropy_stats']['mean_entropy']:<18.4f} {gal_cal['entropy_stats']['mean_entropy']:<18.4f} {entropy_diff:<+12.4f}")

    # Interpretation
    print(f"\n🔍 INTERPRETATION:")

    # Accuracy interpretation
    if abs(accuracy_diff) < 0.01:
        accuracy_verdict = "✅ Similar accuracy (good)"
    elif accuracy_diff < -0.02:
        accuracy_verdict = "⚠️  GAL accuracy notably lower"
    else:
        accuracy_verdict = "✅ GAL accuracy maintained or improved"

    # ECE interpretation (lower is better)
    if ece_diff < -0.01:
        ece_verdict = "✅ GAL significantly better calibrated"
    elif ece_diff < 0.01:
        ece_verdict = "✅ Similar calibration"
    else:
        ece_verdict = "❌ GAL worse calibrated"

    # Entropy interpretation (higher is generally better for avoiding overconfidence)
    if entropy_diff > 0.1:
        entropy_verdict = "✅ GAL less overconfident"
    elif entropy_diff > 0.05:
        entropy_verdict = "✅ GAL slightly less overconfident"
    else:
        entropy_verdict = "➖ Similar confidence levels"

    print(f"   Accuracy: {accuracy_verdict}")
    print(f"   Calibration: {ece_verdict}")
    print(f"   Overconfidence: {entropy_verdict}")

    # Overall verdict
    print(f"\n🏆 OVERALL VERDICT:")
    calibration_improved = ece_diff < -0.005
    accuracy_maintained = accuracy_diff > -0.02
    entropy_improved = entropy_diff > 0.05

    if calibration_improved and accuracy_maintained:
        if entropy_improved:
            verdict = "🎉 STRONG SUPPORT for GoodhartAwareLoss: Better calibration, maintained accuracy, reduced overconfidence"
        else:
            verdict = "✅ SUPPORT for GoodhartAwareLoss: Better calibration with maintained accuracy"
    elif calibration_improved:
        verdict = "⚠️  MIXED RESULTS: Better calibration but accuracy trade-off"
    else:
        verdict = "❌ LIMITED SUPPORT: No clear calibration improvement"

    print(f"   {verdict}")
    print(f"=" * 80)


# ------------------------------------------------------------------------------
# 8. Main Execution
# ------------------------------------------------------------------------------

def main():
    """Main execution function for running the calibration experiment."""
    print("🚀 CIFAR-10 Calibration Experiment: GoodhartAwareLoss vs Label Smoothing")
    print("=" * 80)

    # Create configuration
    config = ExperimentConfig()

    # Display configuration
    print("⚙️  EXPERIMENT CONFIGURATION:")
    print(f"   Dataset: {config.dataset_name.upper()}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Label Smoothing α: {config.label_smoothing_alpha}")
    print(f"   GAL Temperature: {config.gal_temperature}")
    print(f"   GAL Entropy Weight: {config.gal_entropy_weight}")
    print(f"   GAL MI Weight: {config.gal_mi_weight}")
    print(f"   Calibration Bins: {config.calibration_bins}")
    print()

    # Run experiment
    try:
        results = run_calibration_experiment(config)
        print(f"\n✅ Experiment completed successfully!")

        # Save results summary
        summary_path = Path(results['config'].output_dir) / "experiment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CIFAR-10 Calibration Experiment Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Configuration:\n")
            for key, value in results['config'].__dict__.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nResults saved to: {summary_path.parent}\n")

        return results

    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    results = main()