"""
Colored MNIST Spurious Correlation Experiment: GoodhartAwareLoss vs Label Smoothing
==================================================================================

This module implements a comprehensive experiment to evaluate and compare the robustness
of GoodhartAwareLoss against Label Smoothing when facing spurious correlations in the
training data. The experiment uses Colored MNIST to test whether models can avoid
"gaming the metric" by exploiting dataset artifacts.

EXPERIMENT OVERVIEW
------------------
The experiment creates a synthetic dataset where digit classification can be "gamed"
by exploiting spurious color correlations in training, then tests generalization
when these correlations are removed:

1. **Label Smoothing (LS)**: Standard cross-entropy with label smoothing (α=0.1)
2. **GoodhartAwareLoss (GAL)**: Information-theoretic loss designed to prevent
   metric gaming through spurious pattern exploitation

SPURIOUS CORRELATION DESIGN
---------------------------
**Colored MNIST Construction:**
- **Training Set**: 95% correlation between digit class and color
  - Digit '0': 95% black, 5% random colors
  - Digit '1': 95% red, 5% random colors
  - Digit '2': 95% green, 5% random colors
  - ... (each digit gets a dominant color)
  - **"Cheater" strategy**: Model can achieve ~95% accuracy by learning color alone

- **Test Set**: Colors assigned completely randomly (0% correlation)
  - Same digits, but color provides no information about class
  - **Robust model**: Maintains accuracy by learning actual digit shapes
  - **"Cheater" model**: Fails dramatically when color correlation breaks

RESEARCH HYPOTHESIS
------------------
**Core Claims Being Tested:**
- GAL should show smaller generalization gap (train_acc - test_acc)
- GAL should maintain higher test accuracy when spurious correlations are removed
- GAL should be less susceptible to exploiting dataset artifacts for metric gaming
- GAL should learn more robust feature representations focused on digit morphology

METHODOLOGY
-----------
Each model follows identical training protocols:
- Architecture: CNN optimized for 28x28x3 RGB Colored MNIST
- Dataset: Colored MNIST with controlled spurious correlations
- Training: Same optimizer, learning rate schedule, epochs, and augmentation
- Evaluation: Performance gap analysis between correlated and uncorrelated data

MODEL ARCHITECTURE
------------------
- Input: 28x28x3 RGB Colored MNIST images
- Conv Blocks: Conv2D → BatchNorm → ReLU → Optional Dropout
- Residual connections for improved gradient flow
- Global Average Pooling before final classification layer
- Output: Dense(10) → Loss-specific processing → Softmax
- Regularization: L2 weight decay, dropout, and data augmentation

SPURIOUS CORRELATION METRICS
----------------------------
**Test Set Accuracy (Primary Metric):**
- Accuracy on "fair" test set where color correlation is removed
- Higher values indicate robust learning of actual digit features
- Lower values suggest over-reliance on spurious color correlations

**Generalization Gap:**
- Gap = Training Accuracy - Test Accuracy
- Measures overfitting to spurious correlations
- Smaller gaps indicate more robust generalization

**Feature Attribution Analysis:**
- Gradient-based attribution to visualize what features models focus on
- Robust models should focus on digit edges/shapes
- "Cheater" models should focus on color regions

**Correlation Exploitation Score:**
- Custom metric measuring model's reliance on color vs shape
- Computed by comparing performance on color-consistent vs color-inconsistent samples
- Lower scores indicate less exploitation of spurious correlations

ANALYSIS OUTPUTS
---------------
**Robustness Analysis:**
- Generalization gap comparison between LS and GAL
- Test accuracy comparison on uncorrelated data
- Feature importance visualization and analysis
- Spurious correlation exploitation quantification

**Performance Analysis:**
- Training curves on correlated data
- Test performance on uncorrelated data
- Confusion matrices for both training and test sets
- Per-class robustness analysis

**Feature Visualization:**
- Gradient-based attribution maps
- Model attention visualization
- Comparison of learned feature representations
- Color vs shape sensitivity analysis

CONFIGURATION
------------
All experiment parameters are centralized in the ExperimentConfig class:
- Dataset generation parameters (correlation strength, color assignments)
- Model architecture (filters, layers, regularization)
- Training hyperparameters (epochs, batch size, learning rate)
- Loss function parameters (GAL weights, label smoothing alpha)
- Analysis settings (attribution methods, visualization options)

USAGE
-----
To run with default settings:
    python spurious_correlation_experiment.py

To customize correlation strength:
    config = ExperimentConfig()
    config.spurious_correlation_strength = 0.9  # 90% correlation
    config.epochs = 50
    config.gal_entropy_weight = 0.2
    results = run_spurious_correlation_experiment(config)

DEPENDENCIES
-----------
- TensorFlow/Keras for deep learning models
- NumPy for numerical computations and dataset generation
- OpenCV for image processing and color manipulation
- Matplotlib/Seaborn for visualization
- SciPy for statistical analysis
- Custom GoodhartAwareLoss implementation

OUTPUT STRUCTURE
---------------
results/
├── colored_mnist_spurious_correlation_TIMESTAMP/
│   ├── dataset/              # Generated colored MNIST dataset
│   ├── label_smoothing/      # LS model checkpoints and logs
│   ├── goodhart_aware/       # GAL model checkpoints and logs
│   ├── robustness_analysis/  # Generalization gap, feature attribution
│   ├── feature_visualization/# Attribution maps, attention visualization
│   └── performance_analysis/ # Accuracy curves, confusion matrices

RESEARCH APPLICATIONS
--------------------
This experiment framework enables:
- Testing robustness to spurious correlations
- Validating Goodhart's Law mitigation in practice
- Measuring susceptibility to dataset artifacts
- Comparing feature learning robustness
- Benchmarking generalization across distribution shifts

The results provide empirical evidence for whether information-theoretic loss
functions can prevent models from "gaming" metrics through spurious patterns.

Organization:
1. Imports and dependencies
2. Configuration class
3. Colored MNIST dataset generation
4. Model building utilities
5. Robustness metrics implementation
6. Feature attribution and visualization
7. Experiment runner
8. Main execution
"""

# ------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------

import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Custom imports - assuming the GoodhartAwareLoss is available
from goodhart_aware_loss import GoodhartAwareLoss, analyze_loss_components, compute_prediction_entropy


# ------------------------------------------------------------------------------
# 2. Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Unified configuration for the Colored MNIST spurious correlation experiment.

    Contains all parameters for dataset generation, model architecture, training,
    and robustness analysis in a single consolidated configuration class.
    """
    # Dataset Configuration
    dataset_name: str = "colored_mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 3)  # RGB for colored MNIST
    validation_split: float = 0.1

    # Spurious Correlation Parameters
    spurious_correlation_strength: float = 0.95  # 95% correlation in training
    test_correlation_strength: float = 0.0  # 0% correlation in test (random)
    color_palette: str = "bright"  # "bright", "pastel", or "custom"
    background_color: Tuple[int, int, int] = (255, 255, 255)  # White background

    # Model Architecture Parameters
    conv_filters: List[int] = (32, 64, 128, 256)
    dense_units: int = 256
    dropout_rate: float = 0.3
    weight_decay: float = 1e-4
    use_batch_norm: bool = True
    use_residual: bool = True

    # Training Parameters
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    monitor_metric: str = 'val_accuracy'

    # Label Smoothing Configuration
    label_smoothing_alpha: float = 0.1

    # GoodhartAwareLoss Configuration
    gal_temperature: float = 2.0
    gal_entropy_weight: float = 0.15  # Higher for spurious correlation resistance
    gal_mi_weight: float = 0.02

    # Robustness Analysis Parameters
    compute_feature_attribution: bool = True
    attribution_method: str = "integrated_gradients"  # "gradients", "integrated_gradients"
    num_attribution_samples: int = 100

    # Experiment Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "colored_mnist_spurious_correlation"
    save_models: bool = True
    save_dataset: bool = True
    save_plots: bool = True
    plot_format: str = 'png'
    random_seed: int = 42

    # Data Augmentation (careful not to destroy color information)
    use_data_augmentation: bool = True
    rotation_range: float = 10.0
    width_shift_range: float = 0.05
    height_shift_range: float = 0.05
    zoom_range: float = 0.05


# ------------------------------------------------------------------------------
# 3. Colored MNIST Dataset Generation
# ------------------------------------------------------------------------------

def get_color_palette(palette_name: str) -> List[Tuple[int, int, int]]:
    """Get predefined color palette for digit classes.

    Args:
        palette_name: Name of the color palette

    Returns:
        List of RGB color tuples for each digit class
    """
    palettes = {
        "bright": [
            (0, 0, 0),  # 0: Black
            (255, 0, 0),  # 1: Red
            (0, 255, 0),  # 2: Green
            (0, 0, 255),  # 3: Blue
            (255, 255, 0),  # 4: Yellow
            (255, 0, 255),  # 5: Magenta
            (0, 255, 255),  # 6: Cyan
            (255, 128, 0),  # 7: Orange
            (128, 0, 255),  # 8: Purple
            (255, 192, 203),  # 9: Pink
        ],
        "pastel": [
            (64, 64, 64),  # 0: Dark Gray
            (255, 182, 193),  # 1: Light Pink
            (144, 238, 144),  # 2: Light Green
            (173, 216, 230),  # 3: Light Blue
            (255, 255, 224),  # 4: Light Yellow
            (221, 160, 221),  # 5: Plum
            (175, 238, 238),  # 6: Pale Turquoise
            (255, 218, 185),  # 7: Peach Puff
            (221, 160, 221),  # 8: Plum
            (255, 182, 193),  # 9: Light Pink
        ]
    }

    if palette_name not in palettes:
        raise ValueError(f"Unknown palette: {palette_name}. Available: {list(palettes.keys())}")

    return palettes[palette_name]


def colorize_mnist_image(image: np.ndarray, color: Tuple[int, int, int],
                         background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Colorize a grayscale MNIST image with specified foreground and background colors.

    Args:
        image: Grayscale MNIST image (28, 28) with values in [0, 255]
        color: RGB color tuple for the digit
        background_color: RGB color tuple for the background

    Returns:
        Colored image (28, 28, 3) with values in [0, 255]
    """
    # Normalize image to [0, 1]
    image_norm = image.astype(np.float32) / 255.0

    # Create RGB image
    colored_image = np.zeros((28, 28, 3), dtype=np.float32)

    # Apply colors based on pixel intensity
    # Background pixels (low intensity) get background color
    # Foreground pixels (high intensity) get foreground color
    for c in range(3):
        colored_image[:, :, c] = (
                background_color[c] * (1 - image_norm) +
                color[c] * image_norm
        )

    return colored_image.astype(np.uint8)


def generate_colored_mnist(config: ExperimentConfig) -> Tuple[Tuple[np.ndarray, ...], Dict[str, Any]]:
    """Generate Colored MNIST dataset with controlled spurious correlations.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (train_data, test_data) and dataset metadata
    """
    print("🎨 Generating Colored MNIST dataset...")

    # Set random seed for reproducible dataset generation
    np.random.seed(config.random_seed)

    # Load original MNIST
    (x_train_gray, y_train), (x_test_gray, y_test) = keras.datasets.mnist.load_data()

    # Get color palette
    colors = get_color_palette(config.color_palette)

    print(f"   Dataset parameters:")
    print(f"     Training correlation: {config.spurious_correlation_strength:.1%}")
    print(f"     Test correlation: {config.test_correlation_strength:.1%}")
    print(f"     Color palette: {config.color_palette}")
    print(f"     Classes and colors:")
    for i, color in enumerate(colors):
        print(f"       {i}: RGB{color}")

    # Generate training set with spurious correlations
    x_train_colored = []

    print("   🔧 Generating training set with spurious correlations...")
    for i in range(len(x_train_gray)):
        digit_class = y_train[i]
        image = x_train_gray[i]

        # Apply spurious correlation
        if np.random.random() < config.spurious_correlation_strength:
            # Use the "correct" color for this digit class
            color = colors[digit_class]
        else:
            # Use a random color
            color = colors[np.random.randint(0, len(colors))]

        colored_image = colorize_mnist_image(image, color, config.background_color)
        x_train_colored.append(colored_image)

        if (i + 1) % 10000 == 0:
            print(f"     Processed {i + 1}/{len(x_train_gray)} training images...")

    x_train_colored = np.array(x_train_colored)

    # Generate test set without spurious correlations (random colors)
    x_test_colored = []

    print("   🔧 Generating test set with random colors...")
    for i in range(len(x_test_gray)):
        image = x_test_gray[i]

        # Apply random color (no correlation)
        if np.random.random() < config.test_correlation_strength:
            # Use the "correct" color (usually 0% so this rarely happens)
            color = colors[y_test[i]]
        else:
            # Use a random color
            color = colors[np.random.randint(0, len(colors))]

        colored_image = colorize_mnist_image(image, color, config.background_color)
        x_test_colored.append(colored_image)

        if (i + 1) % 2000 == 0:
            print(f"     Processed {i + 1}/{len(x_test_gray)} test images...")

    x_test_colored = np.array(x_test_colored)

    # Normalize to [0, 1]
    x_train_colored = x_train_colored.astype('float32') / 255.0
    x_test_colored = x_test_colored.astype('float32') / 255.0

    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, config.num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, config.num_classes)

    # Create metadata
    metadata = {
        'train_correlation': config.spurious_correlation_strength,
        'test_correlation': config.test_correlation_strength,
        'colors': colors,
        'color_palette': config.color_palette,
        'background_color': config.background_color,
        'train_samples': len(x_train_colored),
        'test_samples': len(x_test_colored),
        'classes': config.num_classes
    }

    print(f"   ✅ Colored MNIST generated:")
    print(f"     Training samples: {len(x_train_colored):,}")
    print(f"     Test samples: {len(x_test_colored):,}")
    print(f"     Image shape: {x_train_colored.shape[1:]}")

    return (x_train_colored, y_train_cat, x_test_colored, y_test_cat), metadata


def visualize_colored_mnist_samples(x_train: np.ndarray, y_train: np.ndarray,
                                    x_test: np.ndarray, y_test: np.ndarray,
                                    metadata: Dict[str, Any], save_path: Optional[Path] = None) -> None:
    """Visualize samples from the colored MNIST dataset.

    Args:
        x_train: Training images
        y_train: Training labels (categorical)
        x_test: Test images
        y_test: Test labels (categorical)
        metadata: Dataset metadata
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))

    # Show training samples (first 2 rows)
    for i in range(20):
        row = i // 10
        col = i % 10

        if i < len(x_train):
            axes[row, col].imshow(x_train[i])
            axes[row, col].set_title(f'Train: {np.argmax(y_train[i])}', fontsize=10)
        axes[row, col].axis('off')

    # Show test samples (last 2 rows)
    for i in range(20):
        row = (i // 10) + 2
        col = i % 10

        if i < len(x_test):
            axes[row, col].imshow(x_test[i])
            axes[row, col].set_title(f'Test: {np.argmax(y_test[i])}', fontsize=10)
        axes[row, col].axis('off')

    # Add overall title
    fig.suptitle(f'Colored MNIST Samples\n'
                 f'Training Correlation: {metadata["train_correlation"]:.1%}, '
                 f'Test Correlation: {metadata["test_correlation"]:.1%}',
                 fontsize=16)

    # Add row labels
    fig.text(0.02, 0.75, 'Training\n(Correlated)', rotation=90, fontsize=14,
             verticalalignment='center', weight='bold')
    fig.text(0.02, 0.25, 'Test\n(Random)', rotation=90, fontsize=14,
             verticalalignment='center', weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------------------------
# 4. Model Building Utilities
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
    """Build CNN model optimized for Colored MNIST.

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
# 5. Robustness Metrics Implementation
# ------------------------------------------------------------------------------

def compute_generalization_gap(train_accuracy: float, test_accuracy: float) -> float:
    """Compute generalization gap as a measure of overfitting to spurious correlations.

    Args:
        train_accuracy: Accuracy on training set
        test_accuracy: Accuracy on test set

    Returns:
        Generalization gap (train - test)
    """
    return train_accuracy - test_accuracy


def compute_correlation_exploitation_score(model: keras.Model,
                                           x_correlated: np.ndarray,
                                           y_correlated: np.ndarray,
                                           x_uncorrelated: np.ndarray,
                                           y_uncorrelated: np.ndarray) -> Dict[str, float]:
    """Compute how much the model exploits spurious correlations.

    Args:
        model: Trained model
        x_correlated: Images with spurious correlations
        y_correlated: Labels for correlated images
        x_uncorrelated: Images without spurious correlations
        y_uncorrelated: Labels for uncorrelated images

    Returns:
        Dictionary with exploitation metrics
    """
    # Get predictions
    pred_correlated = model.predict(x_correlated, verbose=0)
    pred_uncorrelated = model.predict(x_uncorrelated, verbose=0)

    # Compute accuracies
    acc_correlated = np.mean(
        np.argmax(pred_correlated, axis=1) == np.argmax(y_correlated, axis=1)
    )
    acc_uncorrelated = np.mean(
        np.argmax(pred_uncorrelated, axis=1) == np.argmax(y_uncorrelated, axis=1)
    )

    # Exploitation score: higher values indicate more reliance on spurious correlations
    exploitation_score = acc_correlated - acc_uncorrelated

    # Robustness score: higher values indicate more robust learning
    robustness_score = acc_uncorrelated / max(acc_correlated, 1e-8)

    return {
        'correlated_accuracy': acc_correlated,
        'uncorrelated_accuracy': acc_uncorrelated,
        'exploitation_score': exploitation_score,
        'robustness_score': robustness_score
    }


def compute_per_class_robustness(model: keras.Model,
                                 x_test: np.ndarray,
                                 y_test: np.ndarray) -> Dict[int, float]:
    """Compute per-class robustness scores.

    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels (categorical)

    Returns:
        Dictionary mapping class index to robustness score
    """
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    per_class_scores = {}
    for class_idx in range(10):
        class_mask = true_classes == class_idx
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(pred_classes[class_mask] == true_classes[class_mask])
            per_class_scores[class_idx] = class_accuracy
        else:
            per_class_scores[class_idx] = 0.0

    return per_class_scores


# ------------------------------------------------------------------------------
# 6. Feature Attribution and Visualization
# ------------------------------------------------------------------------------

def compute_integrated_gradients(model: keras.Model,
                                 images: np.ndarray,
                                 class_indices: np.ndarray,
                                 steps: int = 50) -> np.ndarray:
    """Compute integrated gradients for feature attribution.

    Args:
        model: Trained model
        images: Input images
        class_indices: Target class indices
        steps: Number of integration steps

    Returns:
        Attribution maps for each image
    """
    # Create baseline (black image)
    baseline = np.zeros_like(images)

    # Generate interpolated images
    alphas = np.linspace(0, 1, steps)

    attributions = []

    for i, (image, class_idx) in enumerate(zip(images, class_indices)):
        image_attributions = []

        for alpha in alphas:
            # Interpolate between baseline and image
            interpolated = baseline[i] + alpha * (image - baseline[i])

            # Compute gradients
            with keras.utils.custom_object_scope({'GoodhartAwareLoss': GoodhartAwareLoss}):
                with tf.GradientTape() as tape:
                    tape.watch(interpolated)
                    predictions = model(interpolated[np.newaxis, ...])
                    target_output = predictions[0, class_idx]

                gradients = tape.gradient(target_output, interpolated)
                image_attributions.append(gradients.numpy())

        # Integrate gradients
        avg_gradients = np.mean(image_attributions, axis=0)
        integrated_gradients = (image - baseline[i]) * avg_gradients
        attributions.append(integrated_gradients)

    return np.array(attributions)


def visualize_feature_attribution(images: np.ndarray,
                                  attributions: np.ndarray,
                                  true_labels: np.ndarray,
                                  pred_labels: np.ndarray,
                                  model_name: str,
                                  save_path: Optional[Path] = None,
                                  num_samples: int = 8) -> None:
    """Visualize feature attribution maps.

    Args:
        images: Original images
        attributions: Attribution maps
        true_labels: True class labels
        pred_labels: Predicted class labels
        model_name: Name of the model
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))

    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}', fontsize=10)
        axes[0, i].axis('off')

        # Attribution map (summed across channels for visualization)
        attr_sum = np.sum(np.abs(attributions[i]), axis=2)
        im = axes[1, i].imshow(attr_sum, cmap='hot', interpolation='bilinear')
        axes[1, i].set_title('Attribution', fontsize=10)
        axes[1, i].axis('off')

        # Overlay attribution on original image
        # Normalize attribution for overlay
        attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)

        # Create overlay
        overlay = images[i].copy()
        for c in range(3):
            overlay[:, :, c] = overlay[:, :, c] * 0.7 + attr_norm * 0.3

        axes[2, i].imshow(overlay)
        axes[2, i].set_title('Overlay', fontsize=10)
        axes[2, i].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Original', rotation=90, fontsize=12, labelpad=20)
    axes[1, 0].set_ylabel('Attribution', rotation=90, fontsize=12, labelpad=20)
    axes[2, 0].set_ylabel('Overlay', rotation=90, fontsize=12, labelpad=20)

    plt.suptitle(f'Feature Attribution Analysis - {model_name}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------------------------
# 7. Visualization Utilities
# ------------------------------------------------------------------------------

def plot_robustness_comparison(ls_metrics: Dict[str, float],
                               gal_metrics: Dict[str, float],
                               save_path: Optional[Path] = None) -> None:
    """Plot comparison of robustness metrics between models.

    Args:
        ls_metrics: Robustness metrics for Label Smoothing
        gal_metrics: Robustness metrics for GoodhartAwareLoss
        save_path: Path to save the plot
    """
    metrics = ['test_accuracy', 'generalization_gap', 'robustness_score']
    ls_values = [ls_metrics.get(m, 0) for m in metrics]
    gal_values = [gal_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, ls_values, width, label='Label Smoothing',
                   color='blue', alpha=0.7)
    bars2 = ax.bar(x + width / 2, gal_values, width, label='GoodhartAwareLoss',
                   color='red', alpha=0.7)

    ax.set_xlabel('Robustness Metrics', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Spurious Correlation Robustness Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Test Accuracy', 'Generalization Gap', 'Robustness Score'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_class_robustness(ls_scores: Dict[int, float],
                              gal_scores: Dict[int, float],
                              save_path: Optional[Path] = None) -> None:
    """Plot per-class robustness comparison.

    Args:
        ls_scores: Per-class scores for Label Smoothing
        gal_scores: Per-class scores for GoodhartAwareLoss
        save_path: Path to save the plot
    """
    classes = list(range(10))
    ls_values = [ls_scores.get(c, 0) for c in classes]
    gal_values = [gal_scores.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, ls_values, width, label='Label Smoothing',
           color='blue', alpha=0.7)
    ax.bar(x + width / 2, gal_values, width, label='GoodhartAwareLoss',
           color='red', alpha=0.7)

    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Per-Class Robustness on Uncorrelated Test Set', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------------------------
# 8. Experiment Runner
# ------------------------------------------------------------------------------

def run_spurious_correlation_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete spurious correlation experiment.

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

    print(f"🚀 Starting Colored MNIST Spurious Correlation Experiment")
    print(f"📁 Results will be saved to: {experiment_dir}")
    print("=" * 80)

    # Generate Colored MNIST dataset
    (x_train, y_train, x_test, y_test), dataset_metadata = generate_colored_mnist(config)

    # Save dataset if requested
    if config.save_dataset:
        dataset_dir = experiment_dir / "dataset"
        dataset_dir.mkdir(exist_ok=True)

        # Visualize samples
        visualize_colored_mnist_samples(
            x_train[:20], y_train[:20],
            x_test[:20], y_test[:20],
            dataset_metadata,
            dataset_dir / f"colored_mnist_samples.{config.plot_format}"
        )

        print(f"✅ Dataset visualization saved to {dataset_dir}")

    # Prepare validation split
    val_split_idx = int(len(x_train) * (1 - config.validation_split))
    x_val = x_train[val_split_idx:]
    y_val = y_train[val_split_idx:]
    x_train = x_train[:val_split_idx]
    y_train = y_train[:val_split_idx]

    print(f"📊 Dataset splits:")
    print(f"   Training samples: {x_train.shape[0]:,}")
    print(f"   Validation samples: {x_val.shape[0]:,}")
    print(f"   Test samples: {x_test.shape[0]:,}")

    # Store results
    results = {
        'config': config,
        'dataset_metadata': dataset_metadata,
        'models': {},
        'histories': {},
        'predictions': {},
        'robustness_metrics': {},
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

        # Setup data generator if using augmentation
        if config.use_data_augmentation:
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=config.rotation_range,
                width_shift_range=config.width_shift_range,
                height_shift_range=config.height_shift_range,
                zoom_range=config.zoom_range,
                fill_mode='nearest'
            )
            datagen.fit(x_train)

        # Train model
        print(f"🏃 Training {model_name} model...")
        if config.use_data_augmentation:
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=config.batch_size),
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
    print(f"\n📈 Evaluating models and computing robustness metrics...")
    print("=" * 80)

    for model_type, model_name in model_configs:
        model = results['models'][model_type]

        print(f"\n🔍 Analyzing {model_name}...")

        # Get predictions
        train_pred = model.predict(x_train, verbose=0)
        test_pred = model.predict(x_test, verbose=0)

        results['predictions'][model_type] = {
            'train_probabilities': train_pred,
            'test_probabilities': test_pred,
            'train_classes': np.argmax(train_pred, axis=1),
            'test_classes': np.argmax(test_pred, axis=1)
        }

        # Compute performance metrics
        train_accuracy = np.mean(
            np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1)
        )
        test_accuracy = np.mean(
            np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1)
        )

        # Compute robustness metrics
        generalization_gap = compute_generalization_gap(train_accuracy, test_accuracy)
        per_class_scores = compute_per_class_robustness(model, x_test, y_test)

        # Compute correlation exploitation (using train as correlated, test as uncorrelated)
        exploitation_metrics = compute_correlation_exploitation_score(
            model, x_train, y_train, x_test, y_test
        )

        results['performance_metrics'][model_type] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
        }

        results['robustness_metrics'][model_type] = {
            'generalization_gap': generalization_gap,
            'per_class_scores': per_class_scores,
            **exploitation_metrics
        }

        print(f"   📊 Performance Metrics:")
        print(f"      Train Accuracy: {train_accuracy:.4f}")
        print(f"      Test Accuracy: {test_accuracy:.4f}")
        print(f"   🛡️  Robustness Metrics:")
        print(f"      Generalization Gap: {generalization_gap:.4f}")
        print(f"      Robustness Score: {exploitation_metrics['robustness_score']:.4f}")
        print(f"      Exploitation Score: {exploitation_metrics['exploitation_score']:.4f}")

    # Generate visualizations
    if config.save_plots:
        print(f"\n📊 Generating visualizations...")
        viz_dir = experiment_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Plot robustness comparison
        plot_robustness_comparison(
            {
                'test_accuracy': results['performance_metrics']['label_smoothing']['test_accuracy'],
                'generalization_gap': results['robustness_metrics']['label_smoothing']['generalization_gap'],
                'robustness_score': results['robustness_metrics']['label_smoothing']['robustness_score']
            },
            {
                'test_accuracy': results['performance_metrics']['goodhart_aware']['test_accuracy'],
                'generalization_gap': results['robustness_metrics']['goodhart_aware']['generalization_gap'],
                'robustness_score': results['robustness_metrics']['goodhart_aware']['robustness_score']
            },
            viz_dir / f"robustness_comparison.{config.plot_format}"
        )

        # Plot per-class robustness
        plot_per_class_robustness(
            results['robustness_metrics']['label_smoothing']['per_class_scores'],
            results['robustness_metrics']['goodhart_aware']['per_class_scores'],
            viz_dir / f"per_class_robustness.{config.plot_format}"
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
    print(f"📋 SPURIOUS CORRELATION EXPERIMENT SUMMARY")
    print(f"=" * 80)

    # Dataset info
    metadata = results['dataset_metadata']
    print(f"\n📊 DATASET INFORMATION:")
    print(f"   Training correlation: {metadata['train_correlation']:.1%}")
    print(f"   Test correlation: {metadata['test_correlation']:.1%}")
    print(f"   Training samples: {metadata['train_samples']:,}")
    print(f"   Test samples: {metadata['test_samples']:,}")

    # Performance comparison
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"{'Metric':<20} {'Label Smoothing':<18} {'GoodhartAwareLoss':<18} {'Difference':<12}")
    print(f"-" * 70)

    ls_perf = results['performance_metrics']['label_smoothing']
    gal_perf = results['performance_metrics']['goodhart_aware']

    train_acc_diff = gal_perf['train_accuracy'] - ls_perf['train_accuracy']
    test_acc_diff = gal_perf['test_accuracy'] - ls_perf['test_accuracy']

    print(
        f"{'Train Accuracy':<20} {ls_perf['train_accuracy']:<18.4f} {gal_perf['train_accuracy']:<18.4f} {train_acc_diff:<+12.4f}")
    print(
        f"{'Test Accuracy':<20} {ls_perf['test_accuracy']:<18.4f} {gal_perf['test_accuracy']:<18.4f} {test_acc_diff:<+12.4f}")

    # Robustness comparison
    print(f"\n🛡️  ROBUSTNESS METRICS:")
    print(f"{'Metric':<20} {'Label Smoothing':<18} {'GoodhartAwareLoss':<18} {'Difference':<12}")
    print(f"-" * 70)

    ls_rob = results['robustness_metrics']['label_smoothing']
    gal_rob = results['robustness_metrics']['goodhart_aware']

    gap_diff = gal_rob['generalization_gap'] - ls_rob['generalization_gap']
    robust_diff = gal_rob['robustness_score'] - ls_rob['robustness_score']
    exploit_diff = gal_rob['exploitation_score'] - ls_rob['exploitation_score']

    print(
        f"{'Generalization Gap':<20} {ls_rob['generalization_gap']:<18.4f} {gal_rob['generalization_gap']:<18.4f} {gap_diff:<+12.4f}")
    print(
        f"{'Robustness Score':<20} {ls_rob['robustness_score']:<18.4f} {gal_rob['robustness_score']:<18.4f} {robust_diff:<+12.4f}")
    print(
        f"{'Exploitation Score':<20} {ls_rob['exploitation_score']:<18.4f} {gal_rob['exploitation_score']:<18.4f} {exploit_diff:<+12.4f}")

    # Interpretation
    print(f"\n🔍 INTERPRETATION:")

    # Test accuracy interpretation (higher is better)
    if test_acc_diff > 0.05:
        test_verdict = "✅ GAL significantly more robust to spurious correlations"
    elif test_acc_diff > 0.02:
        test_verdict = "✅ GAL moderately more robust"
    elif test_acc_diff > -0.02:
        test_verdict = "➖ Similar robustness"
    else:
        test_verdict = "❌ GAL less robust"

    # Generalization gap interpretation (smaller is better)
    if gap_diff < -0.05:
        gap_verdict = "✅ GAL much less prone to overfitting spurious correlations"
    elif gap_diff < -0.02:
        gap_verdict = "✅ GAL less prone to overfitting"
    elif gap_diff < 0.02:
        gap_verdict = "➖ Similar overfitting behavior"
    else:
        gap_verdict = "❌ GAL more prone to overfitting"

    # Robustness score interpretation (higher is better)
    if robust_diff > 0.1:
        robust_verdict = "✅ GAL significantly more robust"
    elif robust_diff > 0.05:
        robust_verdict = "✅ GAL moderately more robust"
    else:
        robust_verdict = "➖ Similar robustness levels"

    print(f"   Test Accuracy: {test_verdict}")
    print(f"   Generalization Gap: {gap_verdict}")
    print(f"   Robustness: {robust_verdict}")

    # Overall verdict
    print(f"\n🏆 OVERALL VERDICT:")
    robust_improvement = test_acc_diff > 0.02
    gap_improvement = gap_diff < -0.02
    no_accuracy_loss = train_acc_diff > -0.05

    if robust_improvement and gap_improvement and no_accuracy_loss:
        verdict = "🎉 STRONG SUPPORT for GoodhartAwareLoss: Better robustness to spurious correlations"
    elif robust_improvement and no_accuracy_loss:
        verdict = "✅ SUPPORT for GoodhartAwareLoss: Improved robustness with maintained performance"
    elif robust_improvement:
        verdict = "⚠️  MIXED RESULTS: Better robustness but potential accuracy trade-off"
    else:
        verdict = "❌ LIMITED SUPPORT: No clear robustness improvement"

    print(f"   {verdict}")
    print(f"=" * 80)


# ------------------------------------------------------------------------------
# 9. Main Execution
# ------------------------------------------------------------------------------

def main():
    """Main execution function for running the spurious correlation experiment."""
    print("🚀 Colored MNIST Spurious Correlation Experiment: GoodhartAwareLoss vs Label Smoothing")
    print("=" * 80)

    # Create configuration
    config = ExperimentConfig()

    # Display configuration
    print("⚙️  EXPERIMENT CONFIGURATION:")
    print(f"   Dataset: {config.dataset_name.upper()}")
    print(
        f"   Spurious Correlation: {config.spurious_correlation_strength:.1%} (train) → {config.test_correlation_strength:.1%} (test)")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Label Smoothing α: {config.label_smoothing_alpha}")
    print(f"   GAL Temperature: {config.gal_temperature}")
    print(f"   GAL Entropy Weight: {config.gal_entropy_weight}")
    print(f"   GAL MI Weight: {config.gal_mi_weight}")
    print(f"   Color Palette: {config.color_palette}")
    print()

    # Run experiment
    try:
        results = run_spurious_correlation_experiment(config)
        print(f"\n✅ Experiment completed successfully!")

        # Save results summary
        summary_dir = Path(results['config'].output_dir) / "experiment_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_path = summary_dir / "spurious_correlation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Colored MNIST Spurious Correlation Experiment Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Configuration:\n")
            for key, value in results['config'].__dict__.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nResults saved to: {summary_path.parent.parent}\n")

        return results

    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    results = main()