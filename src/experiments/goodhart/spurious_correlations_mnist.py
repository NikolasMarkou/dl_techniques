"""
Colored MNIST Spurious Correlation Experiment: GoodhartAwareLoss vs. Baselines
==============================================================================

This module implements a comprehensive experiment to evaluate and compare the robustness
of different loss functions against spurious correlations. The experiment uses a synthetic
Colored MNIST dataset to test whether models can avoid "gaming the metric" by exploiting
dataset artifacts, a key manifestation of Goodhart's Law in machine learning.

The experiment systematically compares loss function variants using the project's
standard analysis framework, focusing on metrics that quantify robustness to
distributional shifts and spurious feature exploitation.

EXPERIMENT OVERVIEW
------------------
The experiment trains multiple CNN models with identical architectures but different
loss functions applied during training:

1. **Standard Cross-Entropy**: A baseline model expected to overfit to spurious features.
2. **Label Smoothing**: A common regularization technique tested for its robustness benefits.
3. **GoodhartAwareLoss**: An information-theoretic loss designed to prevent metric gaming
   by regularizing the model's internal information flow and output uncertainty.

SPURIOUS CORRELATION DESIGN
---------------------------
A synthetic Colored MNIST dataset is generated with a controlled distribution shift
between the training and test sets:

- **Training Set (High Correlation)**: A strong, spurious correlation (e.g., 95%) is
  introduced between the digit's class and its color. For example, the digit '7' is
  colored red 95% of the time. A model can achieve high training accuracy by simply
  learning this color-to-class mapping.

- **Test Set (Zero Correlation)**: The spurious correlation is completely removed.
  Colors are assigned randomly, providing no information about the digit's class.
  A model's performance on this set reveals whether it learned the true underlying
  feature (digit shape) or relied on the spurious shortcut (color).

RESEARCH HYPOTHESIS
------------------
**Core Claims Being Tested:**
- GoodhartAwareLoss (GAL) should maintain higher test accuracy on the uncorrelated
  test set, demonstrating superior robustness.
- GAL should exhibit a smaller "generalization gap" (the drop in accuracy from the
  training set to the test set), indicating less overfitting to spurious patterns.
- GAL should force the model to learn the true, causal features (digit morphology)
  instead of the easy, spurious ones (color).

METHODOLOGY
-----------
Each model follows an identical training and evaluation protocol to ensure a fair comparison:

- **Architecture**: A CNN optimized for 28x28 colored images with residual connections.
  All models share the exact same architecture and weight initialization. The model
  outputs raw logits for numerical stability.
- **Dataset**: The generated Colored MNIST dataset with a 95% train / 0% test correlation.
- **Training**: All models are trained using the project's standard `train_model` utility,
  ensuring identical optimizers, learning rate schedules, and epochs.
- **Evaluation**: Comprehensive analysis including robustness metrics, weight distributions,
  calibration analysis, and spurious correlation specific visualizations.

ROBUSTNESS METRICS
------------------
**Test Set Accuracy (Primary Metric):**
- The model's accuracy on the "fair" test set where the color correlation is broken.
- This is the single most important measure of a model's robustness. Higher is better.

**Generalization Gap:**
- Defined as: `Train Accuracy - Test Accuracy`.
- Measures the degree to which a model has overfitted to the spurious correlations
  present only in the training data. A smaller gap indicates more robust learning.

**Color Dependency Score:**
- A novel metric measuring how much the model's predictions change when colors
  are shuffled, indicating reliance on spurious color features.

ANALYSIS OUTPUTS
---------------
The experiment produces comprehensive analyses and visualizations:

**Robustness Analysis:**
- Bar charts comparing Test Accuracy, Generalization Gap, and Color Dependency
- Spurious correlation specific metrics and visualizations

**Weight Analysis:**
- Weight distribution comparisons across loss functions
- Layer-wise analysis of learned representations

**Calibration Analysis:**
- Reliability diagrams and calibration metrics
- Analysis of prediction confidence patterns

**Training Analysis:**
- Training and validation accuracy/loss curves for all models
- Comprehensive model comparison visualizations
"""

# ------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable, Optional

import keras
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.calibration_analyzer import CalibrationAnalyzer
from dl_techniques.utils.weight_analyzer import WeightAnalyzerConfig, WeightAnalyzer
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig


# ------------------------------------------------------------------------------
# 2. Experiment Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Unified configuration for the Colored MNIST spurious correlation experiment."""
    # Dataset Configuration
    dataset_name: str = "colored_mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 3)

    # Spurious Correlation Parameters
    train_correlation_strength: float = 0.95
    test_correlation_strength: float = 0.0
    validation_correlation_strength: float = 0.5  # Intermediate correlation for validation
    validation_split: float = 0.1

    # Model Architecture Parameters
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    dense_units: List[int] = field(default_factory=lambda: [128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.5])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    weight_decay: float = 1e-4
    kernel_initializer: str = 'he_normal'
    use_batch_norm: bool = True
    use_residual: bool = True

    # Training Parameters
    epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    monitor_metric: str = 'val_accuracy'

    # Loss Functions to Test
    loss_functions: Dict[str, Callable] = field(default_factory=lambda: {
        'goodhart_aware': lambda: GoodhartAwareLoss(
            entropy_weight=0.15,
            mi_weight=0.02,
            from_logits=True
        ),
        'cross_entropy': lambda: keras.losses.CategoricalCrossentropy(from_logits=True),
        'label_smoothing': lambda: keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1,
            from_logits=True
        ),
        'focal_loss': lambda: keras.losses.CategoricalFocalCrossentropy(
            gamma=2.0,
            from_logits=True
        )
    })

    # Loss Function Display Names
    loss_display_names: Dict[str, str] = field(default_factory=lambda: {
        'goodhart_aware': 'GoodhartAware Loss',
        'cross_entropy': 'Cross-Entropy',
        'label_smoothing': 'Label Smoothing',
        'focal_loss': 'Focal Loss'
    })

    # Loss Function Specific Parameters
    label_smoothing_alpha: float = 0.1
    gal_entropy_weight: float = 0.15
    gal_mi_weight: float = 0.02
    focal_gamma: float = 2.0

    # Analysis Parameters
    calibration_bins: int = 15
    analyze_calibration: bool = True
    analyze_weights: bool = True
    analyze_spurious_features: bool = True

    # Experiment Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "colored_mnist_spurious_correlation"
    random_seed: int = 42


# ------------------------------------------------------------------------------
# 3. Colored MNIST Dataset Generation
# ------------------------------------------------------------------------------

def get_color_palette() -> List[Tuple[int, int, int]]:
    """
    Defines the color palette for each digit class.

    Returns:
        List of RGB tuples for each digit (0-9).
    """
    return [
        (255, 0, 0),    # 0: Red
        (0, 255, 0),    # 1: Green
        (0, 0, 255),    # 2: Blue
        (255, 255, 0),  # 3: Yellow
        (255, 0, 255),  # 4: Magenta
        (0, 255, 255),  # 5: Cyan
        (128, 0, 0),    # 6: Dark Red
        (0, 128, 0),    # 7: Dark Green
        (0, 0, 128),    # 8: Dark Blue
        (128, 128, 0)   # 9: Olive
    ]


def colorize_mnist(
    images: np.ndarray,
    labels: np.ndarray,
    correlation: float,
    config: ExperimentConfig,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Colorize MNIST images with specified correlation strength.

    Args:
        images: Grayscale MNIST images (N, 28, 28).
        labels: Digit labels (N,).
        correlation: Correlation strength between digit and color (0.0 to 1.0).
        config: Experiment configuration.
        random_state: Random seed for reproducibility.

    Returns:
        Colored images (N, 28, 28, 3).
    """
    if random_state is not None:
        np.random.seed(random_state)

    colors = get_color_palette()
    normalized_images = np.expand_dims(images.astype(np.float32) / 255.0, axis=-1)
    colored_images = np.zeros((len(images), 28, 28, 3), dtype=np.float32)

    for i, label in enumerate(labels):
        # Decide whether to use correlated or random color
        if np.random.rand() < correlation:
            color_idx = int(label)  # Use digit-correlated color
        else:
            color_idx = np.random.randint(config.num_classes)  # Random color

        color = np.array(colors[color_idx], dtype=np.float32) / 255.0
        colored_images[i] = normalized_images[i] * color

    return colored_images


def create_colored_mnist_dataset(config: ExperimentConfig) -> Tuple[np.ndarray, ...]:
    """
    Generate the complete Colored MNIST dataset with specified correlations.

    Args:
        config: Experiment configuration containing correlation parameters.

    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test) arrays.
    """
    logger.info("🎨 Generating Colored MNIST dataset...")
    np.random.seed(config.random_seed)

    # Load original MNIST
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.mnist.load_data()

    # Create validation split from training data
    val_size = int(len(x_train_orig) * config.validation_split)
    indices = np.random.permutation(len(x_train_orig))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_train_split = x_train_orig[train_indices]
    y_train_split = y_train_orig[train_indices]
    x_val_split = x_train_orig[val_indices]
    y_val_split = y_train_orig[val_indices]

    # Colorize each split with different correlation strengths
    x_train_colored = colorize_mnist(
        x_train_split, y_train_split,
        config.train_correlation_strength, config,
        random_state=config.random_seed
    )

    x_val_colored = colorize_mnist(
        x_val_split, y_val_split,
        config.validation_correlation_strength, config,
        random_state=config.random_seed + 1
    )

    x_test_colored = colorize_mnist(
        x_test_orig, y_test_orig,
        config.test_correlation_strength, config,
        random_state=config.random_seed + 2
    )

    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train_split, config.num_classes)
    y_val_cat = keras.utils.to_categorical(y_val_split, config.num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_orig, config.num_classes)

    logger.info(f"✅ Dataset generated:")
    logger.info(f"   Train: {len(x_train_colored)} samples, {config.train_correlation_strength:.0%} correlation")
    logger.info(f"   Val: {len(x_val_colored)} samples, {config.validation_correlation_strength:.0%} correlation")
    logger.info(f"   Test: {len(x_test_colored)} samples, {config.test_correlation_strength:.0%} correlation")

    return x_train_colored, y_train_cat, x_val_colored, y_val_cat, x_test_colored, y_test_cat


# ------------------------------------------------------------------------------
# 4. Model Building Utilities
# ------------------------------------------------------------------------------

def build_residual_block(
    inputs: keras.layers.Layer,
    filters: int,
    config: ExperimentConfig,
    block_index: int
) -> keras.layers.Layer:
    """Build a residual block adapted for MNIST-sized images."""
    shortcut = inputs

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_1'
    )(inputs)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name=f'conv{block_index}_2'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    # Adjust shortcut dimensions if necessary
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding='same',
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
    """Build a convolutional block with optional residual connections."""
    if config.use_residual and block_index > 0:
        x = build_residual_block(inputs, filters, config, block_index)
    else:
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=config.kernel_size,
            padding='same',
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'conv{block_index}'
        )(inputs)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

    # Add pooling except for the last conv layer
    if block_index < len(config.conv_filters) - 1:
        x = keras.layers.MaxPooling2D(config.pool_size)(x)

    # Add dropout if specified
    dropout_rate = config.dropout_rates[block_index] if block_index < len(config.dropout_rates) else 0.0
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    """
    Build CNN model optimized for Colored MNIST, outputting raw logits.

    Args:
        config: Experiment configuration.
        loss_fn: Loss function to use.
        name: Model name for identification.

    Returns:
        Compiled Keras model.
    """
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs

    # Initial convolution
    x = keras.layers.Conv2D(
        filters=config.conv_filters[0],
        kernel_size=config.kernel_size,
        padding='same',
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='initial_conv'
    )(x)

    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Convolutional blocks
    for i, filters in enumerate(config.conv_filters):
        x = build_conv_block(x, filters, config, i)

    # Global pooling and dense layers
    x = keras.layers.GlobalAveragePooling2D()(x)

    for j, units in enumerate(config.dense_units):
        x = keras.layers.Dense(
            units=units,
            kernel_initializer=config.kernel_initializer,
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            name=f'dense_{j}'
        )(x)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        # Apply dropout to dense layers
        dense_dropout_idx = len(config.conv_filters) + j
        if dense_dropout_idx < len(config.dropout_rates):
            dropout_rate = config.dropout_rates[dense_dropout_idx]
            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer (logits)
    logits = keras.layers.Dense(
        config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay),
        name='logits'
    )(x)

    model = keras.Model(inputs=inputs, outputs=logits, name=f'{name}_model')

    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model


# ------------------------------------------------------------------------------
# 5. Spurious Correlation Analysis
# ------------------------------------------------------------------------------

class SpuriousCorrelationAnalyzer:
    """Analyzer for spurious correlation specific metrics and visualizations."""

    def __init__(self, models: Dict[str, keras.Model], config: ExperimentConfig):
        """
        Initialize the spurious correlation analyzer.

        Args:
            models: Dictionary of trained models (with softmax outputs).
            config: Experiment configuration.
        """
        self.models = models
        self.config = config
        self.results = {}

    def compute_color_dependency_score(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        n_shuffles: int = 5
    ) -> Dict[str, float]:
        """
        Compute how much model predictions change when colors are shuffled.

        Args:
            x_test: Test images.
            y_test: Test labels.
            n_shuffles: Number of color shuffle iterations.

        Returns:
            Dictionary of color dependency scores for each model.
        """
        logger.info("🎨 Computing color dependency scores...")

        color_dependency_scores = {}

        for name, model in self.models.items():
            logger.info(f"   Analyzing color dependency for {name}...")

            # Get original predictions
            original_preds = model.predict(x_test, verbose=0)
            original_classes = np.argmax(original_preds, axis=1)

            # Shuffle colors multiple times and measure prediction changes
            changes = []

            for shuffle_idx in range(n_shuffles):
                # Create color-shuffled version
                shuffled_images = x_test.copy()

                # Shuffle color channels independently
                for i in range(len(shuffled_images)):
                    # Randomly permute the color channels
                    perm = np.random.permutation(3)
                    shuffled_images[i] = shuffled_images[i][:, :, perm]

                # Get predictions on shuffled images
                shuffled_preds = model.predict(shuffled_images, verbose=0)
                shuffled_classes = np.argmax(shuffled_preds, axis=1)

                # Compute percentage of changed predictions
                change_rate = np.mean(original_classes != shuffled_classes)
                changes.append(change_rate)

            # Average across shuffles
            color_dependency_scores[name] = np.mean(changes)

        logger.info("✅ Color dependency analysis completed")
        return color_dependency_scores

    def analyze_robustness_metrics(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive robustness metrics.

        Args:
            x_train: Training images.
            y_train: Training labels.
            x_test: Test images.
            y_test: Test labels.

        Returns:
            Dictionary of robustness metrics for each model.
        """
        logger.info("🛡️  Computing robustness metrics...")

        robustness_results = {}

        for name, model in self.models.items():
            logger.info(f"   Analyzing robustness for {name}...")

            # Basic accuracy metrics
            train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

            robustness_results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'generalization_gap': train_acc - test_acc,
                'loss_gap': test_loss - train_loss
            }

        # Add color dependency scores
        color_scores = self.compute_color_dependency_score(x_test, y_test)
        for name in robustness_results:
            robustness_results[name]['color_dependency'] = color_scores[name]

        self.results = robustness_results
        logger.info("✅ Robustness analysis completed")
        return robustness_results

    def plot_robustness_comparison(self, output_dir: Path) -> None:
        """Plot comprehensive robustness comparison."""
        if not self.results:
            logger.warning("No results to plot. Run analyze_robustness_metrics first.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for plotting
        model_names = list(self.results.keys())
        display_names = [self.config.loss_display_names.get(name, name) for name in model_names]

        metrics = ['test_accuracy', 'generalization_gap', 'color_dependency']
        metric_labels = ['Test Accuracy', 'Generalization Gap', 'Color Dependency Score']

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Spurious Correlation Robustness Analysis', fontsize=16, weight='bold')

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            values = [self.results[name][metric] for name in model_names]

            bars = ax.bar(display_names, values, color=colors[:len(display_names)])
            ax.set_title(label, fontsize=14, weight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save plot
        save_path = output_dir / "spurious_correlation_robustness.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Robustness comparison plot saved to {save_path}")

    def plot_color_analysis_grid(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path,
        n_samples: int = 5
    ) -> None:
        """Plot grid showing model predictions on original vs color-shuffled images."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select random samples
        indices = np.random.choice(len(x_test), n_samples, replace=False)

        for name, model in self.models.items():
            logger.info(f"   Creating color analysis grid for {name}...")

            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
            fig.suptitle(f'Color Dependency Analysis - {self.config.loss_display_names.get(name, name)}',
                        fontsize=16, weight='bold')

            for i, idx in enumerate(indices):
                original_img = x_test[idx]
                true_label = np.argmax(y_test[idx])

                # Create color-shuffled version
                shuffled_img = original_img.copy()
                perm = np.random.permutation(3)
                shuffled_img = shuffled_img[:, :, perm]

                # Get predictions
                orig_pred = model.predict(original_img[np.newaxis, ...], verbose=0)[0]
                shuf_pred = model.predict(shuffled_img[np.newaxis, ...], verbose=0)[0]

                orig_class = np.argmax(orig_pred)
                shuf_class = np.argmax(shuf_pred)

                # Plot original image
                axes[i, 0].imshow(original_img)
                axes[i, 0].set_title(f'Original\nTrue: {true_label}, Pred: {orig_class}\nConf: {orig_pred[orig_class]:.3f}')
                axes[i, 0].axis('off')

                # Plot shuffled image
                axes[i, 1].imshow(shuffled_img)
                axes[i, 1].set_title(f'Color Shuffled\nPred: {shuf_class}\nConf: {shuf_pred[shuf_class]:.3f}')
                axes[i, 1].axis('off')

                # Plot prediction comparison
                x_pos = np.arange(10)
                width = 0.35

                axes[i, 2].bar(x_pos - width/2, orig_pred, width, label='Original', alpha=0.8)
                axes[i, 2].bar(x_pos + width/2, shuf_pred, width, label='Shuffled', alpha=0.8)
                axes[i, 2].set_xlabel('Digit Class')
                axes[i, 2].set_ylabel('Prediction Probability')
                axes[i, 2].set_title('Prediction Comparison')
                axes[i, 2].legend()
                axes[i, 2].set_xticks(x_pos)

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            # Save plot
            save_path = output_dir / f"color_analysis_grid_{name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info("✅ Color analysis grids saved")


# ------------------------------------------------------------------------------
# 6. Experiment Runner
# ------------------------------------------------------------------------------

def run_spurious_correlation_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete spurious correlation experiment with comprehensive analysis."""
    # Setup
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    logger.info("🚀 Starting Colored MNIST Spurious Correlation Experiment")
    logger.info(f"📁 Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # Generate dataset
    x_train, y_train, x_val, y_val, x_test, y_test = create_colored_mnist_dataset(config)

    # Train models
    logger.info("🏗️  Training models with different loss functions...")
    models = {}
    all_histories = {'accuracy': {}, 'loss': {}, 'val_accuracy': {}, 'val_loss': {}}

    for loss_name, loss_fn_factory in config.loss_functions.items():
        logger.info(f"🏗️  Building and training {config.loss_display_names[loss_name]} model...")

        loss_fn = loss_fn_factory()
        model = build_model(config, loss_fn, loss_name)

        logger.info(f"✅ Model built with {model.count_params():,} parameters (outputs logits)")

        # Training configuration
        training_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            reduce_lr_patience=config.reduce_lr_patience,
            reduce_lr_factor=config.reduce_lr_factor,
            monitor_metric=config.monitor_metric,
            model_name=loss_name,
            output_dir=experiment_dir / loss_name
        )

        logger.info(f"🏃 Training {config.loss_display_names[loss_name]} model...")

        # Train model
        history = train_model(
            model, x_train, y_train, x_val, y_val, training_config
        )

        models[loss_name] = model

        # Store history
        for metric in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
            all_histories[metric][loss_name] = history.history[metric]

        logger.info(f"✅ {config.loss_display_names[loss_name]} training completed! "
                   f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")

    # --- ANALYSIS PHASE ---
    logger.info("=" * 80)
    logger.info("🔬 Starting Comprehensive Analysis Phase...")

    # Create prediction models with softmax for analysis
    logger.info("🛠️  Creating prediction models with softmax outputs...")
    prediction_models = {}
    for name, trained_model in models.items():
        logits_output = trained_model.output
        probs_output = keras.layers.Activation('softmax', name='predictions')(logits_output)
        prediction_model = keras.Model(
            inputs=trained_model.input,
            outputs=probs_output,
            name=f"{name}_prediction"
        )
        prediction_model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        prediction_models[name] = prediction_model

    # Weight Analysis
    if config.analyze_weights:
        logger.info("📊 Performing weight distribution analysis...")
        try:
            analysis_config = WeightAnalyzerConfig(
                compute_l1_norm=True,
                compute_l2_norm=True,
                compute_rms_norm=True,
                analyze_biases=True,
                save_plots=True,
                export_format="png"
            )

            weight_analyzer = WeightAnalyzer(
                models=models,
                config=analysis_config,
                output_dir=experiment_dir / "weight_analysis"
            )

            if weight_analyzer.has_valid_analysis():
                weight_analyzer.plot_comprehensive_dashboard()
                weight_analyzer.plot_norm_distributions()
                weight_analyzer.plot_layer_comparisons(['mean', 'std', 'l2_norm'])
                weight_analyzer.save_analysis_results()
                logger.info("✅ Weight analysis completed successfully!")
            else:
                logger.warning("❌ No valid weight data found for analysis")

        except Exception as e:
            logger.error(f"Weight analysis failed: {e}")
            logger.info("Continuing with experiment without weight analysis...")

    # Calibration Analysis
    if config.analyze_calibration:
        logger.info("🎯 Performing calibration effectiveness analysis...")
        try:
            calibration_analyzer = CalibrationAnalyzer(
                models=prediction_models,
                calibration_bins=config.calibration_bins
            )
            calibration_metrics = calibration_analyzer.analyze_calibration(
                x_test=x_test, y_test=y_test
            )
            calibration_dir = experiment_dir / "calibration_analysis"
            calibration_analyzer.plot_reliability_diagrams(calibration_dir)
            calibration_analyzer.plot_calibration_metrics_comparison(calibration_dir)
            logger.info("✅ Calibration analysis completed successfully!")
        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")
            calibration_metrics = {}

    # Spurious Correlation Analysis
    if config.analyze_spurious_features:
        logger.info("🎨 Performing spurious correlation analysis...")
        spurious_analyzer = SpuriousCorrelationAnalyzer(prediction_models, config)

        # Compute robustness metrics
        robustness_results = spurious_analyzer.analyze_robustness_metrics(
            x_train, y_train, x_test, y_test
        )

        # Create visualizations
        spurious_dir = experiment_dir / "spurious_analysis"
        spurious_analyzer.plot_robustness_comparison(spurious_dir)
        spurious_analyzer.plot_color_analysis_grid(x_test, y_test, spurious_dir)

        logger.info("✅ Spurious correlation analysis completed!")

    # Training History Visualizations
    logger.info("📈 Generating training history visualizations...")
    for metric in ['accuracy', 'loss']:
        combined_histories = {
            config.loss_display_names[name]: {
                metric: all_histories[metric][name],
                f'val_{metric}': all_histories[f'val_{metric}'][name]
            }
            for name in models.keys()
        }

        vis_manager.plot_history(
            combined_histories,
            [metric],
            f'training_{metric}_comparison',
            title=f'Loss Functions {metric.capitalize()} Comparison - Spurious Correlation Experiment'
        )

    # Confusion Matrices
    logger.info("🔍 Generating confusion matrix comparisons...")
    model_predictions = {
        config.loss_display_names[name]: model.predict(x_test, verbose=0)
        for name, model in prediction_models.items()
    }

    class_names = [str(i) for i in range(10)]
    vis_manager.plot_confusion_matrices_comparison(
        y_true=np.argmax(y_test, axis=1),
        model_predictions=model_predictions,
        name='spurious_correlation_confusion_matrices',
        subdir='model_comparison',
        normalize=True,
        class_names=class_names
    )

    # Final Performance Analysis
    logger.info("📊 Computing final performance metrics...")
    performance_results = {}
    for name, model in prediction_models.items():
        logger.info(f"   Evaluating {config.loss_display_names[name]}...")
        eval_results = model.evaluate(x_test, y_test, verbose=0)
        performance_results[name] = {
            'loss': eval_results[0],
            'accuracy': eval_results[1]
        }

    # Compile results
    results = {
        'models': models,
        'prediction_models': prediction_models,
        'histories': all_histories,
        'performance_analysis': performance_results,
        'experiment_config': config,
        'robustness_analysis': robustness_results if config.analyze_spurious_features else {},
        'calibration_analysis': calibration_metrics if config.analyze_calibration else {},
        'dataset': {
            'x_train': x_train, 'y_train': y_train,
            'x_val': x_val, 'y_val': y_val,
            'x_test': x_test, 'y_test': y_test
        }
    }

    # Print summary
    print_experiment_summary(results)

    return results


# ------------------------------------------------------------------------------
# 7. Summary and Results Display
# ------------------------------------------------------------------------------

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print a comprehensive summary of experiment results."""
    config = results['experiment_config']

    logger.info("=" * 80)
    logger.info("📋 SPURIOUS CORRELATION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Performance metrics
    logger.info("🎯 PERFORMANCE METRICS (on Test Set with 0% Color Correlation):")
    logger.info(f"{'Model':<25} {'Accuracy':<12} {'Loss':<12}")
    logger.info("-" * 50)

    for model_name, metrics in results.get('performance_analysis', {}).items():
        display_name = config.loss_display_names.get(model_name, model_name)
        if isinstance(metrics, dict):
            accuracy = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            logger.info(f"{display_name:<25} {accuracy:<12.4f} {loss:<12.4f}")

    # Robustness metrics
    if 'robustness_analysis' in results and results['robustness_analysis']:
        logger.info("\n🛡️  ROBUSTNESS METRICS:")
        logger.info(f"{'Model':<25} {'Test Acc':<12} {'Gen Gap':<12} {'Color Dep':<12}")
        logger.info("-" * 65)

        for model_name, rob_metrics in results.get('robustness_analysis', {}).items():
            display_name = config.loss_display_names.get(model_name, model_name)
            test_acc = rob_metrics.get('test_accuracy', 0.0)
            gen_gap = rob_metrics.get('generalization_gap', 0.0)
            color_dep = rob_metrics.get('color_dependency', 0.0)
            logger.info(f"{display_name:<25} {test_acc:<12.4f} {gen_gap:<12.4f} {color_dep:<12.4f}")

    # Calibration metrics
    if 'calibration_analysis' in results and results['calibration_analysis']:
        logger.info("\n🎯 CALIBRATION METRICS:")
        logger.info(f"{'Model':<25} {'ECE':<12} {'Brier Score':<15}")
        logger.info("-" * 55)

        for model_name, cal_metrics in results.get('calibration_analysis', {}).items():
            display_name = config.loss_display_names.get(model_name, model_name)
            ece = cal_metrics.get('ece', 0.0)
            brier = cal_metrics.get('brier_score', 0.0)
            logger.info(f"{display_name:<25} {ece:<12.4f} {brier:<15.4f}")

    # Training summary
    logger.info("\n🏁 FINAL TRAINING METRICS (on Validation Set with 50% Color Correlation):")
    logger.info(f"{'Model':<25} {'Val Accuracy':<15} {'Val Loss':<12}")
    logger.info("-" * 55)

    for model_name in results.get('models', {}).keys():
        display_name = config.loss_display_names.get(model_name, model_name)
        final_val_acc = results['histories']['val_accuracy'][model_name][-1]
        final_val_loss = results['histories']['val_loss'][model_name][-1]
        logger.info(f"{display_name:<25} {final_val_acc:<15.4f} {final_val_loss:<12.4f}")

    # Verdict
    logger.info("\n" + "=" * 80)
    logger.info("🔍 EXPERIMENTAL VERDICT:")

    if results.get('robustness_analysis'):
        # Find best performing models
        best_test_acc = 0
        best_model = ""
        lowest_color_dep = float('inf')
        most_robust_model = ""

        for model_name, metrics in results['robustness_analysis'].items():
            if metrics['test_accuracy'] > best_test_acc:
                best_test_acc = metrics['test_accuracy']
                best_model = config.loss_display_names.get(model_name, model_name)

            if metrics['color_dependency'] < lowest_color_dep:
                lowest_color_dep = metrics['color_dependency']
                most_robust_model = config.loss_display_names.get(model_name, model_name)

        logger.info(f"🏆 Highest Test Accuracy: {best_model} ({best_test_acc:.4f})")
        logger.info(f"🛡️  Lowest Color Dependency: {most_robust_model} ({lowest_color_dep:.4f})")

        # Compare GoodhartAware vs others
        if 'goodhart_aware' in results['robustness_analysis']:
            gal_metrics = results['robustness_analysis']['goodhart_aware']
            gal_test_acc = gal_metrics['test_accuracy']
            gal_color_dep = gal_metrics['color_dependency']

            # Compare with cross-entropy
            if 'cross_entropy' in results['robustness_analysis']:
                ce_metrics = results['robustness_analysis']['cross_entropy']
                acc_improvement = gal_test_acc - ce_metrics['test_accuracy']
                dep_improvement = ce_metrics['color_dependency'] - gal_color_dep

                logger.info(f"\n📊 GOODHART AWARE LOSS vs CROSS-ENTROPY:")
                logger.info(f"   Test Accuracy Improvement: {acc_improvement:+.4f}")
                logger.info(f"   Color Dependency Reduction: {dep_improvement:+.4f}")

                if acc_improvement > 0.02 and dep_improvement > 0.05:
                    verdict = "🎉 STRONG SUPPORT for GoodhartAware Loss effectiveness!"
                elif acc_improvement > 0.01:
                    verdict = "✅ POSITIVE results for GoodhartAware Loss."
                elif acc_improvement > -0.01:
                    verdict = "➖ NEUTRAL results - similar performance."
                else:
                    verdict = "⚠️  NEGATIVE results for this configuration."

                logger.info(f"   VERDICT: {verdict}")

    logger.info("=" * 80)


# ------------------------------------------------------------------------------
# 8. Main Execution
# ------------------------------------------------------------------------------

def main() -> None:
    """Main execution function for running the spurious correlation experiment."""
    logger.info("🚀 Colored MNIST Spurious Correlation Experiment")
    logger.info("=" * 80)

    config = ExperimentConfig()

    logger.info("⚙️  EXPERIMENT CONFIGURATION:")
    logger.info(f"   Dataset: {config.dataset_name.upper()}")
    logger.info(f"   Train Correlation: {config.train_correlation_strength:.0%}")
    logger.info(f"   Val Correlation: {config.validation_correlation_strength:.0%}")
    logger.info(f"   Test Correlation: {config.test_correlation_strength:.0%}")
    logger.info(f"   Epochs: {config.epochs}")
    logger.info(f"   Batch Size: {config.batch_size}")
    logger.info(f"   Learning Rate: {config.learning_rate}")
    logger.info(f"   Loss Functions: {list(config.loss_display_names.values())}")
    logger.info(f"   Architecture: CNN with {config.conv_filters} conv filters")
    logger.info(f"   Residual Connections: {config.use_residual}")
    logger.info(f"   Batch Normalization: {config.use_batch_norm}")
    logger.info("")

    try:
        results = run_spurious_correlation_experiment(config)
        logger.info("✅ Experiment completed successfully!")
        return results

    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)
        raise


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()