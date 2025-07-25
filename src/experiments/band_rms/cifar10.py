"""
Experiment Title: Advanced BandRMS Normalization Study: Static vs Adaptive Approaches on CIFAR-10
===================================================================================================

This experiment conducts a comprehensive evaluation of advanced BandRMS normalization techniques,
comparing traditional static BandRMS with the novel AdaptiveBandRMS approach against conventional
normalization methods on CIFAR-10 image classification. The study investigates whether adaptive
scaling based on log-transformed RMS statistics provides superior training dynamics, model
calibration, and robustness compared to fixed-parameter approaches.

The core hypothesis is that AdaptiveBandRMS, which dynamically adjusts scaling factors based on
input magnitude characteristics, will demonstrate improved training stability and calibration
while maintaining or exceeding the performance benefits of static BandRMS normalization.

Scientific Motivation
--------------------

Modern deep learning normalization techniques face challenges in balancing expressiveness with
training stability. While static BandRMS provides controlled scaling within bounded ranges,
AdaptiveBandRMS introduces input-dependent adaptation that could offer:

1. **Dynamic Responsiveness**: Scaling factors adapt to input magnitude patterns
2. **Improved Numerical Stability**: Log-transformed statistics handle extreme values better
3. **Enhanced Calibration**: Adaptive constraints may improve confidence reliability
4. **Architecture Independence**: Better support for both dense and convolutional layers

Theoretical Foundation:
- **Static BandRMS**: Fixed "thick shell" constraints with learnable but bounded scaling
- **Adaptive BandRMS**: Input-dependent scaling using log-transformed RMS statistics
- **Log Transformation**: Variance stabilization and symmetric magnitude distribution
- **Dense Projection**: Learned mapping from magnitude patterns to optimal scaling factors

Experimental Design
-------------------

**Dataset**: CIFAR-10 (32×32 RGB images, 10 classes)
- Standard preprocessing with data augmentation
- Balanced evaluation for fair calibration analysis

**Model Architecture**: ResNet-inspired CNN with progressive complexity
- 4 convolutional blocks: [32, 64, 128, 256] filters
- Residual connections for gradient flow
- Global average pooling and dense classification head

**Normalization Techniques Evaluated**:

1. **AdaptiveBandRMS**: Novel adaptive approach with band widths [0.1, 0.5, 0.9]
2. **Static BandRMS**: Traditional fixed approach with α values [0.1, 0.2, 0.3]
3. **Baseline Methods**: RMSNorm, LayerNorm, BatchNorm for comparison

**Key Metrics**:
- Classification accuracy and loss
- Calibration metrics (ECE, Brier score)
- Training dynamics and convergence
- Statistical significance across multiple runs
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
from dl_techniques.utils.convert import convert_numpy_to_python
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.datasets.cifar10 import load_and_preprocess_cifar10
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class AdvancedBandRMSExperimentConfig:
    """
    Configuration for the advanced BandRMS normalization effectiveness experiment.

    This class encapsulates all configurable parameters for systematically
    evaluating AdaptiveBandRMS against traditional normalization techniques.

    Attributes:
        dataset_name: Name of the dataset to use
        num_classes: Number of classification classes
        input_shape: Shape of input tensors (H, W, C)
        conv_filters: Number of filters in each convolutional block
        dense_units: Number of units in dense layers
        dropout_rates: Dropout rates for each layer
        kernel_size: Size of convolutional kernels
        weight_decay: L2 regularization strength
        kernel_initializer: Weight initialization method
        use_residual: Whether to use residual connections
        activation: Activation function to use
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        early_stopping_patience: Early stopping patience
        monitor_metric: Metric to monitor for early stopping
        adaptive_band_widths: Band widths for AdaptiveBandRMS testing
        static_band_alphas: Alpha values for static BandRMS testing
        band_epsilon: Epsilon for numerical stability
        band_regularizer_strength: Regularization strength for band parameters
        normalization_variants: Dictionary of normalization configurations
        output_dir: Output directory for results
        experiment_name: Name of the experiment
        random_seed: Random seed for reproducibility
        n_runs: Number of experimental runs for statistical significance
        analyzer_config: Configuration for model analysis
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
    epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 50
    monitor_metric: str = 'val_accuracy'

    # --- AdaptiveBandRMS Specific Parameters ---
    adaptive_band_widths: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    adaptive_epsilon: float = 1e-7
    adaptive_regularizer_strength: float = 1e-5

    # --- Static BandRMS Specific Parameters ---
    static_band_alphas: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    static_epsilon: float = 1e-7
    static_regularizer_strength: float = 1e-5

    # --- Normalization Techniques ---
    normalization_variants: Dict[str, Callable] = field(default_factory=lambda: {
        # Adaptive BandRMS variants
        'ABRMS_01': lambda: ('adaptive_band_rms', {'max_band_width': 0.1}),
        'ABRMS_05': lambda: ('adaptive_band_rms', {'max_band_width': 0.5}),
        'ABRMS_09': lambda: ('adaptive_band_rms', {'max_band_width': 0.9}),

        # Static BandRMS variants
        'BRMS_01': lambda: ('static_band_rms', {'max_band_width': 0.1}),
        'BRMS_02': lambda: ('static_band_rms', {'max_band_width': 0.5}),
        'BRMS_03': lambda: ('static_band_rms', {'max_band_width': 0.9}),

        # Baseline normalization methods
        'RMSNorm': lambda: ('rms_norm', {}),
        'LayerNorm': lambda: ('layer_norm', {}),
        'BatchNorm': lambda: ('batch_norm', {}),
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "advanced_bandrms_normalization_study"
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
    config: AdvancedBandRMSExperimentConfig,
    axis: int = -1,
    name: Optional[str] = None
) -> keras.layers.Layer:
    """
    Factory function to create different normalization layers.

    Args:
        norm_type: Type of normalization layer to create
        norm_params: Parameters specific to the normalization type
        config: Experiment configuration containing default parameters
        axis: Axis for normalization (default -1 for feature axis)
        name: Optional layer name

    Returns:
        Configured normalization layer

    Raises:
        ValueError: If unknown normalization type is specified
    """
    if norm_type == 'adaptive_band_rms':
        return AdaptiveBandRMS(
            max_band_width=norm_params.get('max_band_width', 0.1),
            axis=axis,
            epsilon=config.adaptive_epsilon,
            band_regularizer=keras.regularizers.L2(config.adaptive_regularizer_strength),
            name=name
        )
    elif norm_type == 'static_band_rms':
        return BandRMS(
            max_band_width=norm_params.get('max_band_width', 0.1),
            axis=axis,
            epsilon=config.static_epsilon,
            band_regularizer=keras.regularizers.L2(config.static_regularizer_strength),
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
    config: AdvancedBandRMSExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a residual block with specified normalization technique.

    This function creates a standard residual block with two convolutional layers,
    normalization, and a skip connection. The normalization technique is configurable.

    Args:
        inputs: Input tensor
        filters: Number of convolutional filters
        norm_type: Type of normalization layer
        norm_params: Parameters for normalization layer
        config: Experiment configuration
        block_index: Index of the current block for naming

    Returns:
        Output tensor after residual block processing
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

    # First normalization
    x = create_normalization_layer(
        norm_type, norm_params, config, axis=-1, name=f'norm{block_index}_1'
    )(x)

    # First activation
    x = keras.layers.Activation(config.activation, name=f'act{block_index}_1')(x)

    # Second convolution
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)

    # Second normalization
    x = create_normalization_layer(
        norm_type, norm_params, config, axis=-1, name=f'norm{block_index}_2'
    )(x)

    # Adjust shortcut if needed (dimension mismatch)
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters, (1, 1),
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'shortcut_conv{block_index}'
        )(shortcut)

        shortcut = create_normalization_layer(
            norm_type, norm_params, config, axis=-1, name=f'shortcut_norm{block_index}'
        )(shortcut)

    # Add residual connection and activate
    x = keras.layers.Add(name=f'add{block_index}')([x, shortcut])
    x = keras.layers.Activation(config.activation, name=f'act{block_index}_final')(x)

    return x

def build_conv_block(
    inputs: keras.layers.Layer,
    filters: int,
    norm_type: str,
    norm_params: Dict[str, Any],
    config: AdvancedBandRMSExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """
    Build a convolutional block with specified normalization.

    This function creates either a residual block (if configured and not the first block)
    or a standard convolutional block with normalization, activation, pooling, and dropout.

    Args:
        inputs: Input tensor
        filters: Number of convolutional filters
        norm_type: Type of normalization layer
        norm_params: Parameters for normalization layer
        config: Experiment configuration
        block_index: Index of the current block for naming

    Returns:
        Output tensor after convolutional block processing
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
            norm_type, norm_params, config, axis=-1, name=f'norm{block_index}'
        )(x)

        # Activation
        x = keras.layers.Activation(config.activation, name=f'act{block_index}')(x)

    # Max pooling (except for last block to preserve spatial resolution)
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            name=f'pool{block_index}'
        )(x)

    # Dropout for regularization
    dropout_rate = (config.dropout_rates[block_index]
                   if block_index < len(config.dropout_rates) else 0.0)
    if dropout_rate > 0:
        x = keras.layers.Dropout(
            dropout_rate,
            name=f'dropout{block_index}'
        )(x)

    return x

def build_model(
    config: AdvancedBandRMSExperimentConfig,
    norm_type: str,
    norm_params: Dict[str, Any],
    name: str
) -> keras.Model:
    """
    Build a complete CNN model with specified normalization technique.

    This function constructs a ResNet-inspired CNN architecture with configurable
    normalization layers, suitable for CIFAR-10 classification.

    Args:
        config: Experiment configuration
        norm_type: Type of normalization layer to use
        norm_params: Parameters specific to the normalization type
        name: Model name for identification

    Returns:
        Compiled Keras model ready for training
    """
    # Input layer
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Initial stem convolution with larger kernel
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=(5, 5),
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='stem_conv'
    )(x)

    x = create_normalization_layer(
        norm_type, norm_params, config, axis=-1, name='stem_norm'
    )(x)

    x = keras.layers.Activation(config.activation, name='stem_activation')(x)

    # Progressive convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, norm_type, norm_params, config, i)

    # Global average pooling
    x = keras.layers.Flatten(name="flatten")(x)

    # Dense classification layers with normalization
    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)

        # Apply normalization to dense layers as well
        x = create_normalization_layer(
            norm_type, norm_params, config, axis=-1, name=f'dense_norm_{j}'
        )(x)

        x = keras.layers.Activation(config.activation, name=f'dense_act_{j}')(x)

        # Dropout for dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(
                    dropout_rate,
                    name=f'dense_dropout_{j}'
                )(x)

    # Output layer (no normalization on final predictions)
    outputs = keras.layers.Dense(
        config.num_classes,
        activation='softmax',
        kernel_initializer=config.kernel_initializer,
        name='predictions'
    )(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')

    # Compile with AdamW optimizer and appropriate loss function
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

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

def calculate_run_statistics(
    results_per_run: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate comprehensive statistics across multiple runs for each model.

    This function computes mean, standard deviation, minimum, and maximum values
    for each metric across all experimental runs, enabling statistical analysis
    of model performance consistency.

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
        if run_results:
            metrics = run_results[0].keys()

            for metric in metrics:
                values = [result[metric] for result in run_results if metric in result]

                if values:
                    statistics[model_name][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }

    return statistics

def analyze_normalization_comparison(
    statistics: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Any]:
    """
    Analyze and compare different normalization approaches.

    This function performs comparative analysis between adaptive and static BandRMS
    approaches, identifying performance patterns and statistical significance.

    Args:
        statistics: Statistical results from multiple runs

    Returns:
        Dictionary containing comparative analysis results
    """
    analysis = {
        'adaptive_performance': {},
        'static_performance': {},
        'baseline_performance': {},
        'best_performers': {},
        'stability_analysis': {}
    }

    # Categorize models
    for model_name, model_stats in statistics.items():
        if 'AdaptiveBandRMS' in model_name:
            analysis['adaptive_performance'][model_name] = model_stats
        elif 'StaticBandRMS' in model_name:
            analysis['static_performance'][model_name] = model_stats
        else:
            analysis['baseline_performance'][model_name] = model_stats

    # Find best performers by accuracy
    if statistics:
        best_accuracy = max(
            statistics.items(),
            key=lambda x: x[1].get('accuracy', {}).get('mean', 0.0)
        )
        analysis['best_performers']['accuracy'] = best_accuracy

        # Find most stable model (lowest standard deviation)
        most_stable = min(
            statistics.items(),
            key=lambda x: x[1].get('accuracy', {}).get('std', float('inf'))
        )
        analysis['best_performers']['stability'] = most_stable

        # Stability analysis
        for model_name, model_stats in statistics.items():
            acc_stats = model_stats.get('accuracy', {})
            if acc_stats:
                cv = acc_stats.get('std', 0) / max(acc_stats.get('mean', 1), 1e-8)
                analysis['stability_analysis'][model_name] = {
                    'coefficient_of_variation': cv,
                    'mean_accuracy': acc_stats.get('mean', 0),
                    'std_accuracy': acc_stats.get('std', 0)
                }

    return analysis

# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def create_comprehensive_comparison_plots(
    statistics: Dict[str, Dict[str, Dict[str, float]]],
    analysis: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Create comprehensive comparison plots for normalization techniques.

    Args:
        statistics: Statistical results from multiple runs
        analysis: Comparative analysis results
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle

        # Set up the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # Create comprehensive comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced BandRMS Normalization: Comprehensive Analysis', fontsize=16, fontweight='bold')

        # Extract data for plotting
        models = list(statistics.keys())
        accuracies = [statistics[model]['accuracy']['mean'] for model in models]
        accuracy_stds = [statistics[model]['accuracy']['std'] for model in models]
        losses = [statistics[model]['loss']['mean'] for model in models]
        loss_stds = [statistics[model]['loss']['std'] for model in models]

        # Color coding by normalization type
        colors = []
        for model in models:
            if 'AdaptiveBandRMS' in model:
                colors.append('#FF6B6B')  # Red for Adaptive
            elif 'StaticBandRMS' in model:
                colors.append('#4ECDC4')  # Teal for Static
            elif 'RMSNorm' in model:
                colors.append('#45B7D1')  # Blue for RMS
            elif 'LayerNorm' in model:
                colors.append('#96CEB4')  # Green for Layer
            else:
                colors.append('#FFEAA7')  # Yellow for Batch

        # Plot 1: Accuracy Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(models)), accuracies, yerr=accuracy_stds,
                        capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('Test Accuracy Comparison\n(Mean ± Std)', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, acc, std) in enumerate(zip(bars1, accuracies, accuracy_stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                    f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Loss Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(models)), losses, yerr=loss_stds,
                        capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_title('Test Loss Comparison\n(Mean ± Std)', fontweight='bold')
        ax2.set_ylabel('Loss')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, loss, std) in enumerate(zip(bars2, losses, loss_stds)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{loss:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 3: Stability Analysis (Coefficient of Variation)
        ax3 = axes[1, 0]
        stability_data = analysis.get('stability_analysis', {})
        if stability_data:
            stability_models = list(stability_data.keys())
            cv_values = [stability_data[model]['coefficient_of_variation'] for model in stability_models]
            stability_colors = [colors[models.index(model)] for model in stability_models]

            bars3 = ax3.bar(range(len(stability_models)), cv_values,
                           color=stability_colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax3.set_title('Training Stability Analysis\n(Coefficient of Variation)', fontweight='bold')
            ax3.set_ylabel('CV (Lower = More Stable)')
            ax3.set_xticks(range(len(stability_models)))
            ax3.set_xticklabels(stability_models, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bar, cv in zip(bars3, cv_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{cv:.4f}', ha='center', va='bottom', fontsize=8)

        # Plot 4: Band Width Analysis for Adaptive and Static BandRMS
        ax4 = axes[1, 1]

        # Extract band width information and performance
        band_analysis = {}
        for model_name, model_stats in statistics.items():
            if 'BandRMS' in model_name:
                # Extract band width from model name
                if 'AdaptiveBandRMS' in model_name:
                    band_width = float(model_name.split('_')[-1]) / 10  # Convert 01->0.1, 05->0.5, etc.
                    approach = 'Adaptive'
                elif 'StaticBandRMS' in model_name:
                    band_width = float(model_name.split('_')[-1]) / 10
                    approach = 'Static'
                else:
                    continue

                if approach not in band_analysis:
                    band_analysis[approach] = {'widths': [], 'accuracies': [], 'stds': []}

                band_analysis[approach]['widths'].append(band_width)
                band_analysis[approach]['accuracies'].append(model_stats['accuracy']['mean'])
                band_analysis[approach]['stds'].append(model_stats['accuracy']['std'])

        # Plot band width vs performance
        for approach, data in band_analysis.items():
            marker = 'o' if approach == 'Adaptive' else 's'
            color = '#FF6B6B' if approach == 'Adaptive' else '#4ECDC4'
            ax4.errorbar(data['widths'], data['accuracies'], yerr=data['stds'],
                        marker=marker, linestyle='-', linewidth=2, markersize=8,
                        color=color, label=f'{approach} BandRMS', capsize=5)

        ax4.set_title('Band Width vs Performance Analysis', fontweight='bold')
        ax4.set_xlabel('Band Width')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Create legend for normalization types
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='#FF6B6B', label='Adaptive BandRMS'),
            Rectangle((0, 0), 1, 1, facecolor='#4ECDC4', label='Static BandRMS'),
            Rectangle((0, 0), 1, 1, facecolor='#45B7D1', label='RMSNorm'),
            Rectangle((0, 0), 1, 1, facecolor='#96CEB4', label='LayerNorm'),
            Rectangle((0, 0), 1, 1, facecolor='#FFEAA7', label='BatchNorm')
        ]

        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5)

        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_normalization_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✅ Comprehensive comparison plots saved")

    except Exception as e:
        logger.error(f"❌ Failed to create comprehensive comparison plots: {e}")

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_advanced_bandrms_experiment(config: AdvancedBandRMSExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete advanced BandRMS normalization comparison experiment.

    This function orchestrates the entire experimental pipeline, including data loading,
    model training, evaluation, analysis, and visualization generation.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experimental results and analysis

    Raises:
        Exception: If critical experimental steps fail
    """
    # Set random seed for reproducibility
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

    logger.info("🚀 Starting Advanced BandRMS Normalization Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("📊 Loading CIFAR-10 dataset...")
    cifar10_data = load_and_preprocess_cifar10()
    logger.info("✅ Dataset loaded successfully")

    # Log dataset information
    logger.info(f"📋 Dataset Info:")
    logger.info(f"   Training data shape: {cifar10_data.x_train.shape}")
    logger.info(f"   Training labels shape: {cifar10_data.y_train.shape}")
    logger.info(f"   Test data shape: {cifar10_data.x_test.shape}")
    logger.info(f"   Test labels shape: {cifar10_data.y_test.shape}")

    # ===== MULTIPLE RUNS FOR STATISTICAL SIGNIFICANCE =====
    logger.info(f"🔄 Running {config.n_runs} repetitions for statistical significance...")

    all_trained_models = {}  # Final models from all runs
    all_histories = {}  # Training histories from all runs
    results_per_run = {}  # Performance results per run

    for run_idx in range(config.n_runs):
        logger.info(f"🏃 Starting run {run_idx + 1}/{config.n_runs}")

        # Set different seed for each run to ensure independent trials
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        run_models = {}
        run_histories = {}

        # ===== TRAINING MODELS FOR THIS RUN =====
        for norm_name, norm_factory in config.normalization_variants.items():
            logger.info(f"--- Training {norm_name} (Run {run_idx + 1}) ---")

            try:
                # Get normalization configuration
                norm_type, norm_params = norm_factory()

                # Build model
                model = build_model(config, norm_type, norm_params, f"{norm_name}_run{run_idx}")

                # Log model info for first run
                if run_idx == 0:
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
                    model,
                    cifar10_data.x_train, cifar10_data.y_train,
                    cifar10_data.x_test, cifar10_data.y_test,
                    training_config
                )

                run_models[norm_name] = model
                run_histories[norm_name] = history.history

                logger.info(f"✅ {norm_name} (Run {run_idx + 1}) training completed!")

            except Exception as e:
                logger.error(f"❌ Error training {norm_name} (Run {run_idx + 1}): {e}", exc_info=True)
                continue

        # ===== EVALUATE MODELS FOR THIS RUN =====
        logger.info(f"📊 Evaluating models for run {run_idx + 1}...")

        for norm_name, model in run_models.items():
            try:
                # Get predictions for comprehensive evaluation
                predictions = model.predict(cifar10_data.x_test, verbose=0)

                # Handle label format
                if len(cifar10_data.y_test.shape) > 1 and cifar10_data.y_test.shape[1] > 1:
                    y_true_classes = np.argmax(cifar10_data.y_test, axis=1)
                else:
                    y_true_classes = cifar10_data.y_test.astype(int)

                # Calculate accuracy metrics
                y_pred_classes = np.argmax(predictions, axis=1)
                accuracy = np.mean(y_pred_classes == y_true_classes)

                # Calculate top-5 accuracy
                top_5_predictions = np.argsort(predictions, axis=1)[:, -5:]
                top5_accuracy = np.mean([
                    y_true in top5_pred
                    for y_true, top5_pred in zip(y_true_classes, top_5_predictions)
                ])

                # Calculate loss
                try:
                    eval_results = model.evaluate(cifar10_data.x_test, cifar10_data.y_test, verbose=0)
                    metrics_dict = dict(zip(model.metrics_names, eval_results))
                    loss_value = metrics_dict.get('loss', 0.0)
                except Exception as eval_error:
                    logger.warning(f"⚠️ Model evaluation failed for {norm_name}, calculating loss manually: {eval_error}")
                    # Calculate loss manually
                    if len(cifar10_data.y_test.shape) == 1:
                        y_test_onehot = keras.utils.to_categorical(cifar10_data.y_test, num_classes=10)
                    else:
                        y_test_onehot = cifar10_data.y_test

                    loss_value = float(keras.metrics.categorical_crossentropy(y_test_onehot, predictions).numpy().mean())

                # Store results
                if norm_name not in results_per_run:
                    results_per_run[norm_name] = []

                results_per_run[norm_name].append({
                    'accuracy': accuracy,
                    'top_5_accuracy': top5_accuracy,
                    'loss': loss_value,
                    'run_idx': run_idx
                })

                logger.info(f"✅ {norm_name} (Run {run_idx + 1}): Accuracy={accuracy:.4f}, Loss={loss_value:.4f}")

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
    logger.info("📈 Calculating comprehensive statistics...")
    run_statistics = calculate_run_statistics(results_per_run)
    comparative_analysis = analyze_normalization_comparison(run_statistics)

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
    logger.info("📊 Generating comprehensive visualizations...")

    # Training history comparison
    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='advanced_normalization_training_comparison',
        subdir='training_plots',
        title='Advanced BandRMS vs Traditional Normalization: Training Comparison'
    )

    # Comprehensive statistical comparison plots
    create_comprehensive_comparison_plots(
        run_statistics,
        comparative_analysis,
        experiment_dir / "visualizations"
    )

    # Confusion matrices comparison
    try:
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
            name='advanced_normalization_confusion_matrices',
            subdir='model_comparison',
            normalize=True,
            class_names=[f'Class_{i}' for i in range(10)]
        )

    except Exception as e:
        logger.error(f"❌ Failed to generate confusion matrices: {e}")

    # ===== RESULTS COMPILATION =====
    results = {
        'run_statistics': run_statistics,
        'comparative_analysis': comparative_analysis,
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
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save experiment results in multiple formats for analysis and archival.

    Args:
        results: Experiment results dictionary
        experiment_dir: Directory to save results
    """
    try:
        # Save experiment configuration
        config_dict = {
            'experiment_name': results['config'].experiment_name,
            'normalization_variants': list(results['config'].normalization_variants.keys()),
            'adaptive_band_widths': results['config'].adaptive_band_widths,
            'static_band_alphas': results['config'].static_band_alphas,
            'epochs': results['config'].epochs,
            'batch_size': results['config'].batch_size,
            'learning_rate': results['config'].learning_rate,
            'n_runs': results['config'].n_runs,
            'random_seed': results['config'].random_seed,
            'architecture': {
                'conv_filters': results['config'].conv_filters,
                'dense_units': results['config'].dense_units,
                'use_residual': results['config'].use_residual,
                'activation': results['config'].activation
            }
        }

        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save statistical results
        statistical_results = convert_numpy_to_python(results['run_statistics'])
        with open(experiment_dir / "statistical_results.json", 'w') as f:
            json.dump(statistical_results, f, indent=2)

        # Save comparative analysis
        comparative_analysis = convert_numpy_to_python(results['comparative_analysis'])
        with open(experiment_dir / "comparative_analysis.json", 'w') as f:
            json.dump(comparative_analysis, f, indent=2)

        # Save trained models
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
    Print comprehensive experiment summary with key findings and insights.

    Args:
        results: Experiment results dictionary
    """
    logger.info("=" * 80)
    logger.info("📋 ADVANCED BANDRMS NORMALIZATION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== STATISTICAL RESULTS =====
    if 'run_statistics' in results:
        logger.info("📊 STATISTICAL RESULTS (Mean ± Std across runs):")
        logger.info(f"{'Model':<20} {'Accuracy':<15} {'Loss':<15} {'Runs':<8}")
        logger.info("-" * 65)

        for model_name, stats in results['run_statistics'].items():
            acc_mean = stats.get('accuracy', {}).get('mean', 0.0)
            acc_std = stats.get('accuracy', {}).get('std', 0.0)
            loss_mean = stats.get('loss', {}).get('mean', 0.0)
            loss_std = stats.get('loss', {}).get('std', 0.0)
            n_runs = stats.get('accuracy', {}).get('count', 0)

            logger.info(f"{model_name:<20} {acc_mean:.3f}±{acc_std:.3f}    "
                        f"{loss_mean:.3f}±{loss_std:.3f}    {n_runs:<8}")

    # ===== COMPARATIVE ANALYSIS INSIGHTS =====
    if 'comparative_analysis' in results:
        analysis = results['comparative_analysis']
        logger.info("🔍 COMPARATIVE ANALYSIS:")

        # Best performers
        if 'best_performers' in analysis:
            best_acc = analysis['best_performers'].get('accuracy')
            if best_acc:
                logger.info(f"   🏆 Best Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']['mean']:.4f})")

            most_stable = analysis['best_performers'].get('stability')
            if most_stable:
                logger.info(f"   🎯 Most Stable: {most_stable[0]} (std: {most_stable[1]['accuracy']['std']:.4f})")

        # Adaptive vs Static BandRMS comparison
        adaptive_models = analysis.get('adaptive_performance', {})
        static_models = analysis.get('static_performance', {})

        if adaptive_models and static_models:
            logger.info("   📐 BandRMS APPROACH COMPARISON:")

            # Calculate average performance for each approach
            adaptive_avg = np.mean([
                stats['accuracy']['mean'] for stats in adaptive_models.values()
            ])
            static_avg = np.mean([
                stats['accuracy']['mean'] for stats in static_models.values()
            ])

            logger.info(f"      Adaptive BandRMS Average: {adaptive_avg:.4f}")
            logger.info(f"      Static BandRMS Average:   {static_avg:.4f}")

            if adaptive_avg > static_avg:
                logger.info("      → Adaptive approach shows superior performance")
            else:
                logger.info("      → Static approach shows competitive performance")

            # Band width analysis
            logger.info("   📊 BAND WIDTH ANALYSIS:")
            logger.info("      Adaptive BandRMS:")
            for model_name, stats in adaptive_models.items():
                band_width = float(model_name.split('_')[-1]) / 10
                logger.info(f"        α={band_width:.1f}: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")

            logger.info("      Static BandRMS:")
            for model_name, stats in static_models.items():
                band_width = float(model_name.split('_')[-1]) / 10
                logger.info(f"        α={band_width:.1f}: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")

    # ===== CALIBRATION RESULTS =====
    if (results.get('model_analysis') and
        results['model_analysis'].calibration_metrics and
        results['model_analysis'].confidence_metrics):

        logger.info("🎯 CALIBRATION ANALYSIS:")
        logger.info(f"{'Model':<20} {'ECE':<12} {'Brier':<12} {'Entropy':<12}")
        logger.info("-" * 60)

        for model_name in results['model_analysis'].calibration_metrics.keys():
            cal_metrics = results['model_analysis'].calibration_metrics[model_name]
            ece = cal_metrics.get('ece', 0.0)
            brier = cal_metrics.get('brier_score', 0.0)

            confidence_metrics = results['model_analysis'].confidence_metrics.get(model_name, {})
            entropy = confidence_metrics.get('mean_entropy', 0.0)

            logger.info(f"{model_name:<20} {ece:<12.4f} {brier:<12.4f} {entropy:<12.4f}")

    # ===== KEY INSIGHTS AND RECOMMENDATIONS =====
    logger.info("💡 KEY INSIGHTS AND RECOMMENDATIONS:")

    if 'run_statistics' in results and 'comparative_analysis' in results:
        adaptive_models = results['comparative_analysis'].get('adaptive_performance', {})
        static_models = results['comparative_analysis'].get('static_performance', {})
        baseline_models = results['comparative_analysis'].get('baseline_performance', {})

        if adaptive_models:
            best_adaptive = max(adaptive_models.items(),
                              key=lambda x: x[1]['accuracy']['mean'])
            logger.info(f"   1. Best Adaptive Approach: {best_adaptive[0]} demonstrates the effectiveness")
            logger.info(f"      of log-transformed RMS statistics for dynamic scaling")

        if static_models:
            best_static = max(static_models.items(),
                            key=lambda x: x[1]['accuracy']['mean'])
            logger.info(f"   2. Best Static Approach: {best_static[0]} provides reliable performance")
            logger.info(f"      with simpler parameter management")

        if baseline_models:
            best_baseline = max(baseline_models.items(),
                              key=lambda x: x[1]['accuracy']['mean'])
            logger.info(f"   3. Best Baseline: {best_baseline[0]} serves as the comparative reference")

    logger.info("=" * 80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the advanced BandRMS normalization experiment.

    This function initializes the experimental setup, configures GPU memory,
    and orchestrates the complete experimental pipeline.
    """
    logger.info("🚀 Advanced BandRMS Normalization Effectiveness Experiment")
    logger.info("=" * 80)

    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ Configured {len(gpus)} GPU(s) for memory growth")
        except RuntimeError as e:
            logger.warning(f"GPU configuration warning: {e}")
    else:
        logger.info("ℹ️ No GPUs detected, using CPU")

    # Initialize configuration
    config = AdvancedBandRMSExperimentConfig()

    # Log experimental configuration
    logger.info("⚙️ EXPERIMENT CONFIGURATION:")
    logger.info(f"   Normalization variants: {list(config.normalization_variants.keys())}")
    logger.info(f"   Adaptive band widths: {config.adaptive_band_widths}")
    logger.info(f"   Static band alphas: {config.static_band_alphas}")
    logger.info(f"   Training epochs: {config.epochs}")
    logger.info(f"   Batch size: {config.batch_size}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info(f"   Architecture: {len(config.conv_filters)} conv blocks, {len(config.dense_units)} dense layers")
    logger.info("")

    try:
        # Execute experiment
        results = run_advanced_bandrms_experiment(config)
        logger.info("✅ Advanced BandRMS normalization experiment completed successfully!")

        # Log final summary path for easy access
        experiment_dir = results.get('experiment_dir')
        if experiment_dir:
            logger.info(f"📊 Full results available at: {experiment_dir}")
            logger.info(f"📈 Visualizations available at: {experiment_dir / 'visualizations'}")

        return results

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}", exc_info=True)
        raise

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()