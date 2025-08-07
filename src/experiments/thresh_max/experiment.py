"""
Experiment Title: ThreshMax vs Softmax: Comprehensive Evaluation of Sparse Activation Functions
==================================================================================================

This experiment conducts a comprehensive evaluation of the novel ThreshMax sparse activation function
against traditional Softmax activation on image classification tasks. The study investigates whether
ThreshMax's confidence thresholding mechanism provides superior calibration, sparsity, and robustness
compared to standard softmax probability distributions.

The core hypothesis is that ThreshMax, which creates sparsity by subtracting uniform probability and
clipping negative values, will demonstrate improved confidence calibration, better out-of-distribution
detection, and more robust predictions while maintaining competitive classification accuracy.

Scientific Motivation
--------------------

Traditional softmax activation often produces overconfident predictions and struggles with:
1. **Calibration Issues**: Models are often overconfident on incorrect predictions
2. **Dense Distributions**: All classes receive non-zero probability, even for clear cases
3. **OOD Detection**: Difficulty distinguishing in-distribution from out-of-distribution samples
4. **Robustness**: Sensitivity to input perturbations and adversarial examples

ThreshMax addresses these challenges through:
1. **Confidence Thresholding**: Subtracting uniform probability (1/N) acts as confidence filter
2. **Sparsity Induction**: Clipping negative values creates sparse probability distributions
3. **Graceful Degradation**: Falls back to softmax in maximum entropy (uniform) cases
4. **Calibration Enhancement**: Threshold mechanism should improve probability calibration

Theoretical Foundation:
- **ThreshMax Algorithm**: softmax(x) → subtract 1/N → clip negatives → renormalize
- **Degenerate Case Handling**: Automatic fallback to softmax for uniform inputs
- **Sparsity Metrics**: Average number of zero probabilities per prediction
- **Calibration Theory**: ECE measures alignment between confidence and accuracy

Experimental Design
-------------------

**Datasets**:
- **In-Distribution**: CIFAR-10 (32×32 RGB images, 10 classes)
- **Out-of-Distribution**: Fashion-MNIST (grayscale converted to RGB for fair comparison)

**Model Architecture**: Identical CNN architectures with only final activation differing
- Progressive convolutional blocks: [32, 64, 128, 256] filters
- Batch normalization and dropout for regularization
- Global average pooling and dense classification head
- **Critical**: Everything identical except final activation layer

**Activation Functions Evaluated**:
1. **Standard Softmax**: Baseline reference implementation
2. **ThreshMax Variants**: Different epsilon values for numerical stability

**Key Evaluation Metrics**:
- **Primary**: Classification accuracy, F1-score, log loss
- **Calibration**: Expected Calibration Error (ECE), Brier score, reliability diagrams
- **Sparsity**: Average prediction sparsity, entropy analysis
- **Robustness**: OOD detection performance, noise robustness
- **Statistical**: Multiple runs for significance testing
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from dl_techniques.utils.logger import logger
from dl_techniques.utils.convert import convert_numpy_to_python
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets.cifar10 import load_and_preprocess_cifar10
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

from dl_techniques.layers.activations.thresh_max import ThreshMax


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ThreshMaxExperimentConfig:
    """
    Configuration for the ThreshMax vs Softmax comparative experiment.

    This class encapsulates all configurable parameters for systematically
    evaluating ThreshMax against standard softmax activation functions.

    Attributes:
        dataset_name: Name of the primary dataset to use
        ood_dataset_name: Name of the out-of-distribution dataset
        num_classes: Number of classification classes
        input_shape: Shape of input tensors (H, W, C)
        conv_filters: Number of filters in each convolutional block
        dense_units: Number of units in dense layers
        dropout_rates: Dropout rates for each layer
        kernel_size: Size of convolutional kernels
        weight_decay: L2 regularization strength
        kernel_initializer: Weight initialization method
        activation: Activation function for hidden layers
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        early_stopping_patience: Early stopping patience
        monitor_metric: Metric to monitor for early stopping
        threshmax_variants: Dictionary of ThreshMax configurations to test
        noise_levels: Noise levels for robustness testing
        output_dir: Output directory for results
        experiment_name: Name of the experiment
        random_seed: Random seed for reproducibility
        n_runs: Number of experimental runs for statistical significance
        analyzer_config: Configuration for model analysis
    """

    # --- Dataset Configuration ---
    dataset_name: str = "cifar10"
    ood_dataset_name: str = "fashion_mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (32, 32, 3)

    # --- Model Architecture Parameters ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    kernel_size: Tuple[int, int] = (3, 3)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    activation: str = 'relu'

    # --- Training Parameters ---
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 25
    monitor_metric: str = 'val_accuracy'

    # --- ThreshMax Specific Parameters ---
    threshmax_variants: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'ThreshMax_1e12': {'epsilon': 1e-12},
        'ThreshMax_1e10': {'epsilon': 1e-10},
        'ThreshMax_1e8': {'epsilon': 1e-8},
    })

    # --- Robustness Testing Parameters ---
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "threshmax_vs_softmax_comprehensive"
    random_seed: int = 42
    n_runs: int = 1  # Multiple runs for statistical significance

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
# DATA LOADING UTILITIES
# ==============================================================================

def load_fashion_mnist_as_ood() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST as out-of-distribution data, converting to RGB format.

    Returns:
        Tuple of (x_test, y_test) arrays in CIFAR-10 compatible format
    """
    # Load Fashion-MNIST
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize to [0, 1]
    x_test = x_test.astype('float32') / 255.0

    # Resize from 28x28 to 32x32 to match CIFAR-10
    x_test_resized = np.zeros((x_test.shape[0], 32, 32, 1))
    for i in range(x_test.shape[0]):
        x_test_resized[i, :, :, 0] = tf.image.resize(
            x_test[i:i + 1, :, :, np.newaxis],
            [32, 32],
            method='bilinear'
        ).numpy().squeeze()

    # Convert grayscale to RGB by replicating channels
    x_test_rgb = np.repeat(x_test_resized, 3, axis=-1)

    logger.info(f"Fashion-MNIST OOD data shape: {x_test_rgb.shape}")
    return x_test_rgb, y_test


def add_gaussian_noise(x: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add Gaussian noise to input data.

    Args:
        x: Input data array
        noise_level: Standard deviation of Gaussian noise

    Returns:
        Noisy data array
    """
    if noise_level == 0.0:
        return x

    noise = np.random.normal(0, noise_level, x.shape)
    noisy_x = x + noise
    return np.clip(noisy_x, 0.0, 1.0)


# ==============================================================================
# SPARSITY AND CONFIDENCE ANALYSIS UTILITIES
# ==============================================================================

def calculate_prediction_sparsity(predictions: np.ndarray, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Calculate sparsity metrics for model predictions.

    Args:
        predictions: Prediction probabilities array (N, C)
        threshold: Threshold below which probabilities are considered zero

    Returns:
        Dictionary containing sparsity metrics
    """
    # Count near-zero predictions
    near_zero = predictions < threshold

    # Calculate metrics
    total_predictions = predictions.shape[0] * predictions.shape[1]
    zero_predictions = np.sum(near_zero)
    sparsity_ratio = zero_predictions / total_predictions

    # Average number of zero predictions per sample
    zeros_per_sample = np.mean(np.sum(near_zero, axis=1))

    # Percentage of samples with at least one zero
    samples_with_zeros = np.mean(np.any(near_zero, axis=1))

    return {
        'sparsity_ratio': float(sparsity_ratio),
        'avg_zeros_per_sample': float(zeros_per_sample),
        'samples_with_zeros_pct': float(samples_with_zeros * 100),
        'total_zero_predictions': int(zero_predictions),
        'total_predictions': int(total_predictions)
    }


def calculate_confidence_metrics(predictions: np.ndarray) -> Dict[str, float]:
    """
    Calculate confidence-related metrics for predictions.

    Args:
        predictions: Prediction probabilities array (N, C)

    Returns:
        Dictionary containing confidence metrics
    """
    # Maximum probability (confidence)
    max_probs = np.max(predictions, axis=1)

    # Entropy (lower = more confident)
    entropy = -np.sum(predictions * np.log(predictions + 1e-12), axis=1)

    # Margin (difference between top two predictions)
    sorted_probs = np.sort(predictions, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    return {
        'mean_confidence': float(np.mean(max_probs)),
        'std_confidence': float(np.std(max_probs)),
        'mean_entropy': float(np.mean(entropy)),
        'std_entropy': float(np.std(entropy)),
        'mean_margin': float(np.mean(margins)),
        'std_margin': float(np.std(margins))
    }


def evaluate_ood_detection(
        id_predictions: np.ndarray,
        ood_predictions: np.ndarray,
        method: str = 'max_prob'
) -> Dict[str, float]:
    """
    Evaluate out-of-distribution detection performance.

    Args:
        id_predictions: In-distribution predictions
        ood_predictions: Out-of-distribution predictions
        method: Method for OOD scoring ('max_prob', 'entropy')

    Returns:
        Dictionary containing OOD detection metrics
    """
    if method == 'max_prob':
        id_scores = np.max(id_predictions, axis=1)
        ood_scores = np.max(ood_predictions, axis=1)
        # For max_prob, higher scores indicate in-distribution
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
    elif method == 'entropy':
        id_scores = -np.sum(id_predictions * np.log(id_predictions + 1e-12), axis=1)
        ood_scores = -np.sum(ood_predictions * np.log(ood_predictions + 1e-12), axis=1)
        # For entropy, lower scores indicate in-distribution
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_scores = np.concatenate([-id_scores, -ood_scores])
    else:
        raise ValueError(f"Unknown OOD detection method: {method}")

    # Calculate AUROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5  # Random performance if all scores are identical

    return {
        f'ood_auroc_{method}': float(auroc),
        f'id_mean_{method}': float(np.mean(id_scores if method == 'max_prob' else -id_scores)),
        f'ood_mean_{method}': float(np.mean(ood_scores if method == 'max_prob' else -ood_scores))
    }


# ==============================================================================
# MODEL BUILDING UTILITIES
# ==============================================================================

def build_base_model(
        config: ThreshMaxExperimentConfig,
        model_name: str
) -> keras.Model:
    """
    Build the base CNN model without final activation layer.

    Args:
        config: Experiment configuration
        model_name: Name for the model

    Returns:
        Keras model without final activation
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{model_name}_input')
    x = inputs

    # Progressive convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'{model_name}_conv{i}'
        )(x)

        x = keras.layers.BatchNormalization(name=f'{model_name}_bn{i}')(x)
        x = keras.layers.Activation(config.activation, name=f'{model_name}_act{i}')(x)

        # Max pooling except for last layer
        if i < len(config.conv_filters) - 1:
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'{model_name}_pool{i}'
            )(x)

        # Dropout
        if i < len(config.dropout_rates):
            x = keras.layers.Dropout(
                config.dropout_rates[i],
                name=f'{model_name}_dropout{i}'
            )(x)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling2D(name=f'{model_name}_gap')(x)

    # Dense layers
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'{model_name}_dense{j}'
        )(x)
        x = keras.layers.Activation(config.activation, name=f'{model_name}_dense_act{j}')(x)

        # Dropout for dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            x = keras.layers.Dropout(
                config.dropout_rates[dense_dropout_idx],
                name=f'{model_name}_dense_dropout{j}'
            )(x)

    # Final dense layer (logits) without activation
    logits = keras.layers.Dense(
        config.num_classes,
        kernel_initializer=config.kernel_initializer,
        name=f'{model_name}_logits'
    )(x)

    return keras.Model(inputs=inputs, outputs=logits, name=f'{model_name}_base')


def create_softmax_model(config: ThreshMaxExperimentConfig, base_model: keras.Model) -> keras.Model:
    """
    Create model with softmax activation.

    Args:
        config: Experiment configuration
        base_model: Base model producing logits

    Returns:
        Complete model with softmax activation
    """
    logits = base_model.output
    probabilities = keras.layers.Softmax(name='softmax_activation')(logits)

    model = keras.Model(inputs=base_model.input, outputs=probabilities, name='SoftmaxModel')

    # Compile model
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    return model


def create_threshmax_model(
        config: ThreshMaxExperimentConfig,
        base_model: keras.Model,
        threshmax_params: Dict[str, Any],
        model_name: str
) -> keras.Model:
    """
    Create model with ThreshMax activation.

    Args:
        config: Experiment configuration
        base_model: Base model producing logits
        threshmax_params: Parameters for ThreshMax layer
        model_name: Name for the model

    Returns:
        Complete model with ThreshMax activation
    """
    logits = base_model.output
    probabilities = ThreshMax(
        axis=-1,
        epsilon=threshmax_params['epsilon'],
        name='threshmax_activation'
    )(logits)

    model = keras.Model(inputs=base_model.input, outputs=probabilities, name=model_name)

    # Compile model
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    return model


# ==============================================================================
# COMPREHENSIVE EVALUATION UTILITIES
# ==============================================================================

def evaluate_model_comprehensive(
        model: keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_ood: np.ndarray,
        y_ood: np.ndarray,
        config: ThreshMaxExperimentConfig,
        model_name: str
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of a model.

    Args:
        model: Trained Keras model
        x_test: Test data (in-distribution)
        y_test: Test labels (in-distribution)
        x_ood: OOD test data
        y_ood: OOD test labels
        config: Experiment configuration
        model_name: Name of the model

    Returns:
        Dictionary containing comprehensive evaluation results
    """
    results = {'model_name': model_name}

    # === Basic Classification Metrics ===
    predictions = model.predict(x_test, verbose=0)

    # Convert labels if needed
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true_classes = np.argmax(y_test, axis=1)
        y_test_categorical = y_test
    else:
        y_true_classes = y_test.astype(int)
        y_test_categorical = keras.utils.to_categorical(y_test, num_classes=config.num_classes)

    y_pred_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(y_pred_classes == y_true_classes)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted', zero_division=0
    )

    # Log loss
    log_loss = float(keras.metrics.categorical_crossentropy(y_test_categorical, predictions).numpy().mean())

    results.update({
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'log_loss': float(log_loss)
    })

    # === Sparsity Analysis ===
    sparsity_metrics = calculate_prediction_sparsity(predictions)
    results.update({f'sparsity_{k}': v for k, v in sparsity_metrics.items()})

    # === Confidence Analysis ===
    confidence_metrics = calculate_confidence_metrics(predictions)
    results.update({f'confidence_{k}': v for k, v in confidence_metrics.items()})

    # === Out-of-Distribution Detection ===
    ood_predictions = model.predict(x_ood, verbose=0)

    # OOD detection using max probability
    ood_metrics_maxprob = evaluate_ood_detection(predictions, ood_predictions, method='max_prob')
    results.update(ood_metrics_maxprob)

    # OOD detection using entropy
    ood_metrics_entropy = evaluate_ood_detection(predictions, ood_predictions, method='entropy')
    results.update(ood_metrics_entropy)

    # === Noise Robustness ===
    noise_results = {}
    for noise_level in config.noise_levels:
        if noise_level == 0.0:
            continue  # Skip clean data, already evaluated

        x_noisy = add_gaussian_noise(x_test, noise_level)
        noisy_predictions = model.predict(x_noisy, verbose=0)
        noisy_pred_classes = np.argmax(noisy_predictions, axis=1)
        noisy_accuracy = np.mean(noisy_pred_classes == y_true_classes)

        noise_results[f'accuracy_noise_{noise_level}'] = float(noisy_accuracy)

        # Confidence degradation under noise
        clean_confidence = calculate_confidence_metrics(predictions)
        noisy_confidence = calculate_confidence_metrics(noisy_predictions)
        confidence_drop = clean_confidence['mean_confidence'] - noisy_confidence['mean_confidence']
        noise_results[f'confidence_drop_noise_{noise_level}'] = float(confidence_drop)

    results.update(noise_results)

    # === Additional Metrics ===
    # Top-3 accuracy
    top_3_predictions = np.argsort(predictions, axis=1)[:, -3:]
    top3_accuracy = np.mean([
        y_true in top3_pred
        for y_true, top3_pred in zip(y_true_classes, top_3_predictions)
    ])
    results['top_3_accuracy'] = float(top3_accuracy)

    # Maximum entropy case detection (for ThreshMax)
    max_entropy_threshold = np.log(config.num_classes) - 0.1  # Near maximum entropy
    entropies = -np.sum(predictions * np.log(predictions + 1e-12), axis=1)
    high_entropy_samples = np.mean(entropies > max_entropy_threshold)
    results['high_entropy_samples_pct'] = float(high_entropy_samples * 100)

    logger.info(f"✅ Comprehensive evaluation completed for {model_name}")
    return results


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def create_threshmax_comparison_plots(
        all_results: Dict[str, List[Dict[str, Any]]],
        output_dir: Path
) -> None:
    """
    Create comprehensive comparison plots for ThreshMax vs Softmax.

    Args:
        all_results: Results from all model runs
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle

        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")

        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ThreshMax vs Softmax: Comprehensive Comparison', fontsize=16, fontweight='bold')

        # Prepare data
        model_names = list(all_results.keys())
        colors = ['#1f77b4' if 'Softmax' in name else '#ff7f0e' for name in model_names]

        # Calculate statistics
        stats = {}
        for name, results_list in all_results.items():
            stats[name] = {}
            for key in results_list[0].keys():
                if isinstance(results_list[0][key], (int, float)):
                    values = [r[key] for r in results_list if key in r]
                    stats[name][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }

        # Plot 1: Accuracy Comparison
        ax = axes[0, 0]
        accuracies = [stats[name]['accuracy']['mean'] for name in model_names]
        accuracy_stds = [stats[name]['accuracy']['std'] for name in model_names]

        bars = ax.bar(range(len(model_names)), accuracies, yerr=accuracy_stds,
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Classification Accuracy\n(Mean ± Std)', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, acc, std in zip(bars, accuracies, accuracy_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot 2: Sparsity Analysis
        ax = axes[0, 1]
        sparsity_ratios = [stats[name]['sparsity_sparsity_ratio']['mean'] for name in model_names]
        sparsity_stds = [stats[name]['sparsity_sparsity_ratio']['std'] for name in model_names]

        bars = ax.bar(range(len(model_names)), sparsity_ratios, yerr=sparsity_stds,
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Prediction Sparsity\n(Fraction of Near-Zero Probabilities)', fontweight='bold')
        ax.set_ylabel('Sparsity Ratio')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, sparse, std in zip(bars, sparsity_ratios, sparsity_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.01,
                    f'{sparse:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot 3: Calibration (ECE would come from ModelAnalyzer)
        ax = axes[0, 2]
        confidences = [stats[name]['confidence_mean_confidence']['mean'] for name in model_names]
        confidence_stds = [stats[name]['confidence_mean_confidence']['std'] for name in model_names]

        bars = ax.bar(range(len(model_names)), confidences, yerr=confidence_stds,
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Mean Prediction Confidence', fontweight='bold')
        ax.set_ylabel('Mean Confidence')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Plot 4: OOD Detection (AUROC)
        ax = axes[1, 0]
        ood_aucs = [stats[name]['ood_auroc_max_prob']['mean'] for name in model_names]
        ood_stds = [stats[name]['ood_auroc_max_prob']['std'] for name in model_names]

        bars = ax.bar(range(len(model_names)), ood_aucs, yerr=ood_stds,
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Out-of-Distribution Detection\n(AUROC using Max Probability)', fontweight='bold')
        ax.set_ylabel('AUROC')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 5: Noise Robustness
        ax = axes[1, 1]
        noise_level = 0.1  # Choose one noise level for visualization
        noise_key = f'accuracy_noise_{noise_level}'
        if all(noise_key in stats[name] for name in model_names):
            noise_accs = [stats[name][noise_key]['mean'] for name in model_names]
            noise_acc_stds = [stats[name][noise_key]['std'] for name in model_names]

            bars = ax.bar(range(len(model_names)), noise_accs, yerr=noise_acc_stds,
                          capsize=5, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(f'Robustness to Gaussian Noise\n(σ={noise_level})', fontweight='bold')
            ax.set_ylabel('Accuracy under Noise')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

        # Plot 6: Entropy Analysis
        ax = axes[1, 2]
        entropies = [stats[name]['confidence_mean_entropy']['mean'] for name in model_names]
        entropy_stds = [stats[name]['confidence_mean_entropy']['std'] for name in model_names]

        bars = ax.bar(range(len(model_names)), entropies, yerr=entropy_stds,
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Mean Prediction Entropy\n(Lower = More Confident)', fontweight='bold')
        ax.set_ylabel('Mean Entropy')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Create legend
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='#1f77b4', label='Softmax'),
            Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', label='ThreshMax')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)

        plt.tight_layout()
        plt.savefig(output_dir / 'threshmax_comprehensive_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✅ ThreshMax comparison plots created successfully")

    except Exception as e:
        logger.error(f"❌ Failed to create comparison plots: {e}")


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_threshmax_experiment(config: ThreshMaxExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete ThreshMax vs Softmax comparison experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experimental results and analysis
    """
    # Set random seed
    keras.utils.set_random_seed(config.random_seed)

    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting ThreshMax vs Softmax Comprehensive Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # === DATA LOADING ===
    logger.info("📊 Loading datasets...")

    # Load CIFAR-10 (in-distribution)
    cifar10_data = load_and_preprocess_cifar10()
    logger.info(f"✅ CIFAR-10 loaded: {cifar10_data.x_train.shape} train, {cifar10_data.x_test.shape} test")

    # Load Fashion-MNIST (out-of-distribution)
    x_ood, y_ood = load_fashion_mnist_as_ood()
    logger.info(f"✅ Fashion-MNIST (OOD) loaded: {x_ood.shape}")

    # === MULTIPLE RUNS FOR STATISTICAL SIGNIFICANCE ===
    logger.info(f"🔄 Running {config.n_runs} repetitions for statistical significance...")

    all_results = {}  # Results from all runs
    all_trained_models = {}  # Final models
    all_histories = {}  # Training histories

    for run_idx in range(config.n_runs):
        logger.info(f"🏃 Starting run {run_idx + 1}/{config.n_runs}")

        # Set different seed for each run
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        run_models = {}
        run_histories = {}

        # Build base model (shared architecture)
        base_model = build_base_model(config, f'base_run{run_idx}')
        logger.info(f"Base model parameters: {base_model.count_params():,}")

        # === CREATE AND TRAIN SOFTMAX MODEL ===
        logger.info("--- Training Softmax Model ---")
        try:
            softmax_model = create_softmax_model(config, base_model)

            # Training configuration
            training_config = TrainingConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
                monitor_metric=config.monitor_metric,
                model_name=f"SoftmaxModel_run{run_idx}",
                output_dir=experiment_dir / "training_plots" / f"run_{run_idx}" / "softmax"
            )

            # Train model
            history = train_model(
                softmax_model,
                cifar10_data.x_train, cifar10_data.y_train,
                cifar10_data.x_test, cifar10_data.y_test,
                training_config
            )

            run_models['SoftmaxModel'] = softmax_model
            run_histories['SoftmaxModel'] = history.history
            logger.info("✅ Softmax model training completed!")

        except Exception as e:
            logger.error(f"❌ Error training Softmax model: {e}")
            continue

        # === CREATE AND TRAIN THRESHMAX MODELS ===
        for variant_name, threshmax_params in config.threshmax_variants.items():
            logger.info(f"--- Training {variant_name} Model ---")
            try:
                # Create fresh base model for each variant to avoid interference
                variant_base = build_base_model(config, f'{variant_name}_base_run{run_idx}')

                threshmax_model = create_threshmax_model(
                    config, variant_base, threshmax_params, f'{variant_name}Model'
                )

                # Training configuration
                training_config = TrainingConfig(
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    early_stopping_patience=config.early_stopping_patience,
                    monitor_metric=config.monitor_metric,
                    model_name=f"{variant_name}Model_run{run_idx}",
                    output_dir=experiment_dir / "training_plots" / f"run_{run_idx}" / variant_name.lower()
                )

                # Train model
                history = train_model(
                    threshmax_model,
                    cifar10_data.x_train, cifar10_data.y_train,
                    cifar10_data.x_test, cifar10_data.y_test,
                    training_config
                )

                run_models[f'{variant_name}Model'] = threshmax_model
                run_histories[f'{variant_name}Model'] = history.history
                logger.info(f"✅ {variant_name} model training completed!")

            except Exception as e:
                logger.error(f"❌ Error training {variant_name} model: {e}")
                continue

        # === EVALUATE MODELS FOR THIS RUN ===
        logger.info(f"📊 Evaluating models for run {run_idx + 1}...")

        for model_name, model in run_models.items():
            try:
                results = evaluate_model_comprehensive(
                    model,
                    cifar10_data.x_test, cifar10_data.y_test,
                    x_ood, y_ood,
                    config,
                    model_name
                )

                # Store results
                if model_name not in all_results:
                    all_results[model_name] = []
                all_results[model_name].append(results)

                logger.info(f"✅ {model_name} evaluation: Acc={results['accuracy']:.4f}, "
                            f"Sparsity={results['sparsity_sparsity_ratio']:.4f}, "
                            f"OOD_AUROC={results['ood_auroc_max_prob']:.4f}")

            except Exception as e:
                logger.error(f"❌ Error evaluating {model_name}: {e}")

        # Store models from last run for analysis
        if run_idx == config.n_runs - 1:
            all_trained_models = run_models
            all_histories = run_histories

        # Cleanup
        del run_models
        gc.collect()

    # === COMPREHENSIVE ANALYSIS ===
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
        logger.info("✅ ModelAnalyzer analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ ModelAnalyzer failed: {e}")

    # === VISUALIZATION ===
    logger.info("📊 Generating visualizations...")

    # Initialize visualization manager
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    # Training history comparison
    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='threshmax_vs_softmax_training',
        subdir='training_comparison',
        title='ThreshMax vs Softmax: Training Comparison'
    )

    # Comprehensive comparison plots
    create_threshmax_comparison_plots(all_results, experiment_dir / "visualizations")

    # === COMPILE RESULTS ===
    final_results = {
        'all_results': all_results,
        'trained_models': all_trained_models,
        'histories': all_histories,
        'model_analysis': model_analysis_results,
        'config': config,
        'experiment_dir': experiment_dir
    }

    # Save results
    save_experiment_results(final_results, experiment_dir)

    # Print summary
    print_experiment_summary(final_results)

    return final_results


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
        # Save configuration
        config_dict = {
            'experiment_name': results['config'].experiment_name,
            'threshmax_variants': results['config'].threshmax_variants,
            'epochs': results['config'].epochs,
            'batch_size': results['config'].batch_size,
            'learning_rate': results['config'].learning_rate,
            'n_runs': results['config'].n_runs,
            'noise_levels': results['config'].noise_levels,
        }

        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save evaluation results
        evaluation_results = convert_numpy_to_python(results['all_results'])
        with open(experiment_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for name, model in results['trained_models'].items():
            model_path = models_dir / f"{name}.keras"
            model.save(model_path)

        logger.info("✅ Experiment results saved successfully")

    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary
    """
    logger.info("=" * 80)
    logger.info("📋 THRESHMAX VS SOFTMAX EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Calculate statistics across runs
    all_results = results['all_results']

    logger.info("📊 PERFORMANCE SUMMARY (Mean ± Std across runs):")
    logger.info(f"{'Model':<18} {'Accuracy':<12} {'Sparsity':<12} {'OOD_AUROC':<12} {'Entropy':<12}")
    logger.info("-" * 80)

    for model_name, result_list in all_results.items():
        if not result_list:
            continue

        # Calculate statistics
        accuracy_vals = [r['accuracy'] for r in result_list]
        sparsity_vals = [r['sparsity_sparsity_ratio'] for r in result_list]
        ood_vals = [r['ood_auroc_max_prob'] for r in result_list]
        entropy_vals = [r['confidence_mean_entropy'] for r in result_list]

        acc_mean, acc_std = np.mean(accuracy_vals), np.std(accuracy_vals)
        sparse_mean, sparse_std = np.mean(sparsity_vals), np.std(sparsity_vals)
        ood_mean, ood_std = np.mean(ood_vals), np.std(ood_vals)
        entropy_mean, entropy_std = np.mean(entropy_vals), np.std(entropy_vals)

        logger.info(f"{model_name:<18} {acc_mean:.3f}±{acc_std:.3f}  "
                    f"{sparse_mean:.3f}±{sparse_std:.3f}  "
                    f"{ood_mean:.3f}±{ood_std:.3f}  "
                    f"{entropy_mean:.3f}±{entropy_std:.3f}")

    # Key findings
    logger.info("\n💡 KEY FINDINGS:")

    # Find best models
    best_accuracy = max(
        [(name, np.mean([r['accuracy'] for r in results]))
         for name, results in all_results.items()],
        key=lambda x: x[1]
    )

    best_sparsity = max(
        [(name, np.mean([r['sparsity_sparsity_ratio'] for r in results]))
         for name, results in all_results.items()],
        key=lambda x: x[1]
    )

    best_ood = max(
        [(name, np.mean([r['ood_auroc_max_prob'] for r in results]))
         for name, results in all_results.items()],
        key=lambda x: x[1]
    )

    logger.info(f"   🏆 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]:.4f})")
    logger.info(f"   🎯 Most Sparse: {best_sparsity[0]} ({best_sparsity[1]:.4f})")
    logger.info(f"   🛡️ Best OOD Detection: {best_ood[0]} ({best_ood[1]:.4f})")

    # Compare softmax vs threshmax
    softmax_results = all_results.get('SoftmaxModel', [])
    threshmax_results = {k: v for k, v in all_results.items() if 'ThreshMax' in k}

    if softmax_results and threshmax_results:
        logger.info("\n📊 THRESHMAX VS SOFTMAX COMPARISON:")

        softmax_acc = np.mean([r['accuracy'] for r in softmax_results])
        softmax_sparsity = np.mean([r['sparsity_sparsity_ratio'] for r in softmax_results])
        softmax_ood = np.mean([r['ood_auroc_max_prob'] for r in softmax_results])

        logger.info(f"   Softmax - Accuracy: {softmax_acc:.4f}, "
                    f"Sparsity: {softmax_sparsity:.4f}, OOD: {softmax_ood:.4f}")

        for name, results_list in threshmax_results.items():
            thresh_acc = np.mean([r['accuracy'] for r in results_list])
            thresh_sparsity = np.mean([r['sparsity_sparsity_ratio'] for r in results_list])
            thresh_ood = np.mean([r['ood_auroc_max_prob'] for r in results_list])

            logger.info(f"   {name} - Accuracy: {thresh_acc:.4f} "
                        f"({'↑' if thresh_acc > softmax_acc else '↓'}{abs(thresh_acc - softmax_acc):.4f}), "
                        f"Sparsity: {thresh_sparsity:.4f} "
                        f"({'↑' if thresh_sparsity > softmax_sparsity else '↓'}{abs(thresh_sparsity - softmax_sparsity):.4f}), "
                        f"OOD: {thresh_ood:.4f} "
                        f"({'↑' if thresh_ood > softmax_ood else '↓'}{abs(thresh_ood - softmax_ood):.4f})")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the ThreshMax vs Softmax experiment.
    """
    logger.info("🚀 ThreshMax vs Softmax Comprehensive Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ Configured {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU configuration warning: {e}")

    # Initialize configuration
    config = ThreshMaxExperimentConfig()

    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   ThreshMax variants: {list(config.threshmax_variants.keys())}")
    logger.info(f"   Training epochs: {config.epochs}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info(f"   Noise levels: {config.noise_levels}")

    try:
        # Run experiment
        results = run_threshmax_experiment(config)
        logger.info("✅ ThreshMax experiment completed successfully!")

        experiment_dir = results.get('experiment_dir')
        if experiment_dir:
            logger.info(f"📊 Results: {experiment_dir}")
            logger.info(f"📈 Visualizations: {experiment_dir / 'visualizations'}")

        return results

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()