"""
CIFAR-10 Window Attention Comparison Experiment
===============================================

This experiment evaluates and compares three windowed attention mechanisms
on the CIFAR-10 image classification task:

1. **WindowAttention**: Standard windowed multi-head self-attention from Swin Transformer
2. **WindowZigZagAttention**: Windowed attention with zigzag position bias and optional
   adaptive normalization strategies (adaptive softmax, hierarchical routing)
3. **WindowKAN**: Windowed attention with KAN (Kolmogorov-Arnold Network)
   projections that learn B-spline activation functions

Research Questions
------------------

1. **Accuracy**: Does the sophisticated KAN-based attention or zigzag ordering provide
   benefits over standard windowed attention on CIFAR-10?

2. **Training Dynamics**: How do the different attention mechanisms affect convergence
   speed, stability, and overfitting behavior?

3. **Efficiency**: Given similar model sizes (<1M parameters), which attention mechanism
   provides the best accuracy-efficiency trade-off?

4. **Calibration**: Do the different attention mechanisms produce differently calibrated
   probability distributions? (Especially relevant for WindowZigZagAttention with
   adaptive normalization)

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32×32 RGB images)
- 50,000 training images
- 10,000 test images
- Standard preprocessing with normalization

**Model Architecture**: Lightweight CNN-Transformer hybrid architecture:

```
Input (32×32×3)
      ↓
Conv2D (32 filters, 3×3) + BN + ReLU
      ↓
Conv2D (64 filters, 3×3, stride=2) + BN + ReLU  → [16×16×64]
      ↓
Attention Layer (window_size=4, num_heads=4)  → [256, 64]
      ↓
Conv2D (128 filters, 3×3, stride=2) + BN + ReLU → [8×8×128]
      ↓
Attention Layer (window_size=4, num_heads=4)  → [64, 128]
      ↓
GlobalAveragePooling
      ↓
Dense(256) + Dropout + ReLU
      ↓
Dense(10) + Softmax
```

**Attention Configurations**:

1. **WindowAttention**:
   - Standard Swin-style windowed attention
   - Relative position bias
   - Efficient O(M²) complexity per window

2. **WindowZigZagAttention**:
   - Zigzag-ordered relative position bias
   - Standard softmax normalization (baseline)
   - Comparison with adaptive temperature softmax variant

3. **WindowKAN**:
   - KAN-based Q, K, V projections
   - Learnable B-spline activations
   - Grid size: 5, Spline order: 3

**Training Configuration**:
- Optimizer: AdamW with weight decay
- Learning rate: 0.001 with cosine decay
- Batch size: 128
- Epochs: 50
- Early stopping on validation accuracy
- Data augmentation: Random flips and crops

**Analysis Pipeline**:

1. **Performance Metrics**:
   - Test accuracy and top-5 accuracy
   - Training/validation curves
   - Convergence speed

2. **Calibration Analysis**:
   - Expected Calibration Error (ECE)
   - Brier score
   - Confidence distributions

3. **Weight Analysis**:
   - Layer-wise weight distributions
   - Spectral properties (via WeightWatcher)
   - Parameter efficiency

4. **Training Dynamics**:
   - Overfitting indices
   - Training stability scores
   - Epochs to convergence

Expected Outcomes
-----------------

1. **WindowAttention** (baseline): Should provide solid performance as a proven
   architecture from Swin Transformer

2. **WindowZigZagAttention**: May show improved performance if zigzag ordering
   provides better inductive bias for CIFAR-10's image patterns

3. **WindowKAN**: Higher expressiveness from learnable activations may
   lead to better accuracy but potentially slower training and higher risk of
   overfitting given the small dataset

Visual Analysis
---------------

The experiment generates:
- Training history comparison plots
- Confusion matrices for each attention type
- Calibration curves and reliability diagrams
- Weight distribution visualizations
- Comprehensive summary dashboard
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import keras
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

# ==============================================================================
# LOCAL IMPORTS
# ==============================================================================

from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model

from dl_techniques.layers.attention import (
    WindowAttention,
    WindowZigZagAttention,
    WindowAttentionKAN
)

from sklearn.model_selection import train_test_split

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    ClassificationResults,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    MultiModelClassification,
    ROCPRCurves
)

from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for the window attention comparison experiment.

    Attributes:
        # Model configurations
        attention_types: List of attention types to compare
        conv_filters: Filter progression for CNN backbone
        attention_dim: Embedding dimension for attention layers
        window_size: Window size for all attention mechanisms
        num_heads: Number of attention heads
        dropout_rate: Dropout rate for regularization

        # KAN-specific parameters
        kan_grid_size: Grid size for KAN B-spline basis
        kan_spline_order: Order of B-spline basis functions
        kan_activation: Base activation for KAN layers

        # Training parameters
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW optimizer
        validation_split: Fraction of training data for validation

        # Analysis configuration
        analyzer_config: Configuration for ModelAnalyzer
        output_dir: Directory for saving results
    """
    # Model architecture
    attention_types: List[str] = field(default_factory=lambda: [
        'window_kan',
        'window_zigzag_adaptive',
        'window_zigzag',
        'window',
    ])
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    attention_dim: int = 64
    window_size: int = 4  # For 16×16 and 8×8 feature maps
    num_heads: int = 4
    dropout_rate: float = 0.3

    # KAN-specific parameters
    kan_grid_size: int = 5
    kan_spline_order: int = 3
    kan_activation: str = 'gelu'
    kan_regularization: float = 0.01

    # Training configuration
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_split: float = 0.1  # For manual splitting

    # Training callbacks configuration
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 15
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    monitor_metric: str = 'val_accuracy'

    # Analysis configuration
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=False,  # Skip for speed
        analyze_training_dynamics=True,
        analyze_spectral=True,
    ))

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path(
        f"window_attention_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ))


# ==============================================================================
# DATA LOADING
# ==============================================================================

@dataclass
class CIFAR10Data:
    """
    Container for CIFAR-10 dataset.

    Attributes:
        x_train: Training images, shape (N, 32, 32, 3), normalized to [0, 1]
        y_train: Training labels, one-hot encoded, shape (N, 10)
        x_test: Test images, shape (M, 32, 32, 3), normalized to [0, 1]
        y_test: Test labels, one-hot encoded, shape (M, 10)
        class_names: List of class names for CIFAR-10
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: List[str] = field(default_factory=lambda: [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ])


def load_and_preprocess_cifar10() -> CIFAR10Data:
    """
    Load and preprocess CIFAR-10 dataset.

    Returns:
        CIFAR10Data object containing preprocessed training and test data.
    """
    logger.info("Loading CIFAR-10 dataset...")

    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    logger.info(f"Training samples: {len(x_train)}")
    logger.info(f"Test samples: {len(x_test)}")
    logger.info(f"Image shape: {x_train.shape[1:]}")

    return CIFAR10Data(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )


# ==============================================================================
# MODEL BUILDERS
# ==============================================================================

def create_attention_layer(
    index: int,
    attention_type: str,
    dim: int,
    window_size: int,
    num_heads: int,
    dropout_rate: float,
    config: ExperimentConfig
) -> keras.layers.Layer:
    """
    Factory function to create the appropriate attention layer.

    Args:
        attention_type: Type of attention ('window', 'window_zigzag',
                       'window_zigzag_adaptive', 'window_kan')
        dim: Embedding dimension
        window_size: Window size for local attention
        num_heads: Number of attention heads
        dropout_rate: Dropout rate for attention
        config: Experiment configuration

    Returns:
        Configured attention layer
    """
    if attention_type == 'window':
        return WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_dropout_rate=dropout_rate,
            proj_dropout_rate=dropout_rate,
            name=f'window_attn_{dim}_{index}'
        )

    elif attention_type == 'window_zigzag':
        return WindowZigZagAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_dropout_rate=dropout_rate,
            proj_dropout_rate=dropout_rate,
            use_hierarchical_routing=False,
            use_adaptive_softmax=False,
            name=f'window_zigzag_attn_{dim}_{index}'
        )

    elif attention_type == 'window_zigzag_adaptive':
        return WindowZigZagAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_dropout_rate=dropout_rate,
            proj_dropout_rate=dropout_rate,
            use_hierarchical_routing=False,
            use_adaptive_softmax=True,
            adaptive_softmax_config={
                'min_temp': 0.1,
                'max_temp': 2.0,
                'entropy_threshold': 0.5
            },
            name=f'window_zigzag_adaptive_attn_{dim}_{index}'
        )

    elif attention_type == 'window_kan':
        return WindowAttentionKAN(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            kan_grid_size=config.kan_grid_size,
            kan_spline_order=config.kan_spline_order,
            kan_activation=config.kan_activation,
            kan_regularization_factor=config.kan_regularization,
            attn_dropout_rate=dropout_rate,
            proj_dropout_rate=dropout_rate,
            name=f'kan_window_attn_{dim}_{index}'
        )

    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def build_attention_model(
    attention_type: str,
    config: ExperimentConfig,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10
) -> keras.Model:
    """
    Build a CNN-Transformer hybrid model with specified attention mechanism.

    Architecture:
        Conv Block 1 (32×32 → 16×16) → Attention →
        Conv Block 2 (16×16 → 8×8) → Attention →
        Global Pool → Dense → Output

    Args:
        attention_type: Type of attention mechanism
        config: Experiment configuration
        input_shape: Input image shape
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs

    # Initial conv block (32×32 → 32×32)
    x = keras.layers.Conv2D(
        config.conv_filters[0], 3, padding='same',
        kernel_initializer='he_normal',
        name='conv1'
    )(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.Activation('relu', name='relu1')(x)

    # Conv block with stride (32×32 → 16×16)
    x = keras.layers.Conv2D(
        config.conv_filters[1], 3, strides=2, padding='same',
        kernel_initializer='he_normal',
        name='conv2'
    )(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.Activation('relu', name='relu2')(x)

    # Project to attention dimension
    x = keras.layers.Conv2D(
        config.attention_dim, 1, padding='same',
        kernel_initializer='he_normal',
        name='proj1'
    )(x)
    x = keras.layers.BatchNormalization(name='bn_proj1')(x)

    # First attention block (16×16 = 256 tokens)
    B = keras.ops.shape(x)[0]
    H, W, C = 16, 16, config.attention_dim
    x_reshaped = keras.layers.Reshape((H * W, C), name='reshape1')(x)

    attn_layer_1 = create_attention_layer(
        index=0,
        attention_type=attention_type,
        dim=config.attention_dim,
        window_size=config.window_size,
        num_heads=config.num_heads,
        dropout_rate=config.dropout_rate,
        config=config
    )
    x_attn = attn_layer_1(x_reshaped)
    x_attn = keras.layers.Dropout(config.dropout_rate, name='drop1')(x_attn)

    # Reshape back to spatial
    x = keras.layers.Reshape((H, W, C), name='reshape_back1')(x_attn)

    # Conv block with stride (16×16 → 8×8)
    x = keras.layers.Conv2D(
        config.conv_filters[2], 3, strides=2, padding='same',
        kernel_initializer='he_normal',
        name='conv3'
    )(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.Activation('relu', name='relu3')(x)

    # Project to attention dimension
    x = keras.layers.Conv2D(
        config.attention_dim, 1, padding='same',
        kernel_initializer='he_normal',
        name='proj2'
    )(x)
    x = keras.layers.BatchNormalization(name='bn_proj2')(x)

    # Second attention block (8×8 = 64 tokens)
    H2, W2 = 8, 8
    x_reshaped = keras.layers.Reshape((H2 * W2, config.attention_dim), name='reshape2')(x)

    attn_layer_2 = create_attention_layer(
        index=1,
        attention_type=attention_type,
        dim=config.attention_dim,
        window_size=config.window_size,
        num_heads=config.num_heads,
        dropout_rate=config.dropout_rate,
        config=config
    )
    x_attn = attn_layer_2(x_reshaped)
    x_attn = keras.layers.Dropout(config.dropout_rate, name='drop2')(x_attn)

    # Reshape back to spatial
    x = keras.layers.Reshape((H2, W2, config.attention_dim), name='reshape_back2')(x_attn)

    # Global pooling and classification head
    x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = keras.layers.Dense(
        256,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(config.weight_decay),
        name='dense1'
    )(x)
    x = keras.layers.Dropout(config.dropout_rate, name='drop_dense')(x)
    x = keras.layers.Activation('relu', name='relu_dense')(x)

    # Output layer
    logits = keras.layers.Dense(
        num_classes,
        activation='linear',
        kernel_initializer='glorot_uniform',
        name='logits'
    )(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=logits, name=f'model_{attention_type}')

    return model


def get_model_name(attention_type: str) -> str:
    """Get a human-readable model name for the attention type."""
    name_mapping = {
        'window': 'WindowAttention',
        'window_zigzag': 'WindowZigZag',
        'window_zigzag_adaptive': 'WindowZigZag+Adaptive',
        'window_kan': 'WindowKAN'
    }
    return name_mapping.get(attention_type, attention_type)


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_and_evaluate_model(
    model: keras.Model,
    model_name: str,
    data: CIFAR10Data,
    config: ExperimentConfig
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """
    Train and evaluate a single model.

    Args:
        model: Keras model to train
        model_name: Name for logging and saving
        data: CIFAR10Data object
        config: Experiment configuration

    Returns:
        Tuple of (trained_model, history_dict)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Training: {model_name}")
    logger.info("=" * 80)

    # Print model summary
    param_count = model.count_params()
    logger.info(f"Total parameters: {param_count:,}")

    if param_count > 1_000_000:
        logger.warning(f"Model has {param_count:,} parameters (target: <1M)")

    # Split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        data.x_train,
        data.y_train,
        test_size=config.validation_split,
        random_state=42,
        stratify=np.argmax(data.y_train, axis=1)
    )

    logger.info(f"Training samples: {len(x_train)}")
    logger.info(f"Validation samples: {len(x_val)}")

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        ),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Configure training
    training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        model_name=model_name.replace(' ', '_').replace('+', '_'),
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        reduce_lr_factor=config.reduce_lr_factor,
        min_learning_rate=config.min_learning_rate,
        monitor_metric=config.monitor_metric
    )

    # Train model
    history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_val,
        y_test=y_val,
        config=training_config
    )

    # Evaluate on test set
    logger.info("")
    logger.info(f"Evaluating {model_name} on test set...")
    test_results = model.evaluate(
        data.x_test,
        data.y_test,
        batch_size=config.batch_size,
        verbose=0
    )

    logger.info(f"Test Loss: {test_results[0]:.4f}")
    logger.info(f"Test Accuracy: {test_results[1]:.4f}")

    return model, history.history


# ==============================================================================
# ANALYSIS AND VISUALIZATION
# ==============================================================================

def analyze_models(
    models: Dict[str, keras.Model],
    histories: Dict[str, Dict[str, List[float]]],
    data: CIFAR10Data,
    config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on all trained models.

    Args:
        models: Dictionary mapping model names to trained models
        histories: Dictionary mapping model names to training histories
        data: CIFAR10Data object
        config: Experiment configuration

    Returns:
        Dictionary containing all analysis results
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL ANALYSIS")
    logger.info("=" * 80)

    results = {
        'models': models,
        'histories': histories,
        'performance_analysis': {},
        'model_analysis': None
    }

    # ===== Performance Analysis =====
    logger.info("")
    logger.info("Computing performance metrics...")

    for model_name, model in models.items():
        # Test set evaluation
        test_loss, test_acc = model.evaluate(
            x=data.x_test,
            y=data.y_test,
            batch_size=config.batch_size,
            verbose=0
        )

        # Top-5 accuracy
        predictions = model.predict(data.x_test, batch_size=config.batch_size, verbose=0)
        top5_acc = keras.metrics.top_k_categorical_accuracy(
            data.y_test,
            predictions,
            k=5
        )
        top5_acc = float(keras.ops.mean(top5_acc))

        results['performance_analysis'][model_name] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'top_5_accuracy': top5_acc
        }

        logger.info(f"{model_name}: Acc={test_acc:.4f}, Top-5={top5_acc:.4f}, Loss={test_loss:.4f}")

    # ===== Model Analyzer Integration =====
    logger.info("")
    logger.info("Running ModelAnalyzer...")

    # Prepare data input for analyzer as per the README documentation.
    data_input = DataInput(
        x_data=data.x_test,
        y_data=data.y_test
    )

    # Initialize ModelAnalyzer with all required arguments.
    analyzer = ModelAnalyzer(
        models=models,
        training_history=histories,
        config=config.analyzer_config,
        output_dir=config.output_dir
    )

    # Analyze all models using the correct method.
    analysis_results = analyzer.analyze(data=data_input)

    results['model_analysis'] = analysis_results

    # ===== Visualization =====
    logger.info("")
    logger.info("Generating visualizations...")

    viz_manager = VisualizationManager(
        experiment_name="window_attention_comparison",
        output_dir=config.output_dir
    )

    # Register the visualization templates to be used, as per the new API.
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("roc_pr_curves", ROCPRCurves)

    # --- 1. Training Curves Comparison ---
    training_histories_data = {
        name: TrainingHistory(
            epochs=list(range(len(hist.get('loss', [])))),
            train_loss=hist.get('loss', []),
            val_loss=hist.get('val_loss', []),
            train_metrics={'accuracy': hist.get('accuracy', [])},
            val_metrics={'accuracy': hist.get('val_accuracy', [])}
        )
        for name, hist in histories.items()
    }
    viz_manager.visualize(
        data=training_histories_data,
        plugin_name="training_curves",
        title="Training Curves: Attention Mechanisms Comparison"
    )

    # Pre-calculate predictions for all models to avoid redundant calls
    all_predictions = {
        name: model.predict(data.x_test, verbose=0)
        for name, model in models.items()
    }

    # Convert y_true from one-hot to class indices for compatibility
    y_true_indices = np.argmax(data.y_test, axis=1)

    # --- 2. Individual Confusion Matrices ---
    for model_name, predictions in all_predictions.items():
        classification_results = ClassificationResults(
            y_true=y_true_indices,
            y_pred=np.argmax(predictions, axis=1),
            y_prob=predictions,
            class_names=data.class_names,
            model_name=model_name
        )
        viz_manager.visualize(
            data=classification_results,
            plugin_name="confusion_matrix",
            title=f"Confusion Matrix: {model_name}"
        )

    # --- 3. Multi-Model Comparison (ROC & PR Curves) ---
    multi_model_data = MultiModelClassification(
        results={
            name: ClassificationResults(
                y_true=y_true_indices,
                y_pred=np.argmax(preds, axis=1),
                y_prob=preds,
                class_names=data.class_names,
                model_name=name
            ) for name, preds in all_predictions.items()
        },
        dataset_name="CIFAR-10"
    )
    viz_manager.visualize(
        data=multi_model_data,
        plugin_name="roc_pr_curves",
        title="ROC & PR Curves: Attention Mechanisms Comparison"
    )

    logger.info(f"Visualizations saved to: {config.output_dir}")

    return results

# ==============================================================================
# SUMMARY REPORTING
# ==============================================================================

def print_experiment_summary(
    results: Dict[str, Any],
    config: ExperimentConfig
) -> None:
    """
    Print a comprehensive summary of experiment results.

    Args:
        results: Dictionary containing all experimental results
        config: Experiment configuration
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    # ===== Architecture Overview =====
    logger.info("ARCHITECTURE OVERVIEW")
    logger.info("-" * 80)
    logger.info(f"Attention Dimension: {config.attention_dim}")
    logger.info(f"Window Size: {config.window_size}")
    logger.info(f"Number of Heads: {config.num_heads}")
    logger.info(f"Dropout Rate: {config.dropout_rate}")
    logger.info(f"CNN Filters: {config.conv_filters}")
    logger.info("")

    # ===== Performance Comparison =====
    logger.info("PERFORMANCE COMPARISON (Test Set)")
    logger.info("-" * 80)
    logger.info(f"{'Model':<30} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12} {'Params':<12}")
    logger.info("-" * 80)

    # Sort by accuracy
    perf_items = sorted(
        results['performance_analysis'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )

    for model_name, metrics in perf_items:
        model = results['models'][model_name]
        param_count = model.count_params()

        logger.info(
            f"{model_name:<30} "
            f"{metrics['accuracy']:<12.4f} "
            f"{metrics['top_5_accuracy']:<12.4f} "
            f"{metrics['loss']:<12.4f} "
            f"{param_count:<12,}"
        )
    logger.info("")

    # ===== Model Analysis Results =====
    model_analysis = results.get('model_analysis')
    if model_analysis:

        # Calibration Metrics
        if config.analyzer_config.analyze_calibration and model_analysis.calibration_metrics:
            logger.info("CALIBRATION METRICS")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<15}")
            logger.info("-" * 80)

            for model_name in results['performance_analysis'].keys():
                cal_metrics = model_analysis.calibration_metrics.get(model_name, {})
                conf_metrics = model_analysis.confidence_metrics.get(model_name, {})

                logger.info(
                    f"{model_name:<30} "
                    f"{cal_metrics.get('ece', 0.0):<12.4f} "
                    f"{cal_metrics.get('brier_score', 0.0):<15.4f} "
                    f"{conf_metrics.get('mean_entropy', 0.0):<15.4f}"
                )
            logger.info("")

        # Training Dynamics
        if config.analyzer_config.analyze_training_dynamics and model_analysis.training_metrics:
            logger.info("TRAINING DYNAMICS")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'Epochs to Conv':<15} {'Overfitting':<15} {'Stability':<12}")
            logger.info("-" * 80)

            tm = model_analysis.training_metrics

            for model_name in results['performance_analysis'].keys():
                epochs_conv = tm.epochs_to_convergence.get(model_name, 0)
                overfit = tm.overfitting_index.get(model_name, 0.0)
                stability = tm.training_stability_score.get(model_name, 0.0)

                logger.info(
                    f"{model_name:<30} "
                    f"{epochs_conv:<15} "
                    f"{overfit:<15.4f} "
                    f"{stability:<12.4f}"
                )
            logger.info("")

        # Spectral Analysis
        if config.analyzer_config.analyze_spectral and \
           model_analysis.spectral_analysis is not None and not model_analysis.spectral_analysis.empty:
            logger.info("SPECTRAL ANALYSIS (WeightWatcher)")
            logger.info("-" * 80)
            logger.info(f"{'Model':<30} {'Mean α':<12} {'Concentration':<15} {'Matrix Entropy':<15}")
            logger.info("-" * 80)

            spectral_df = model_analysis.spectral_analysis
            per_model_summary = spectral_df.groupby('model_name')[
                ['alpha', 'concentration_score', 'entropy']
            ].mean()

            for model_name in results['performance_analysis'].keys():
                if model_name in per_model_summary.index:
                    model_summary = per_model_summary.loc[model_name]
                    mean_alpha = model_summary.get('alpha', 0.0)
                    mean_concentration = model_summary.get('concentration_score', 0.0)
                    mean_entropy = model_summary.get('entropy', 0.0)

                    logger.info(
                        f"{model_name:<30} "
                        f"{mean_alpha:<12.4f} "
                        f"{mean_concentration:<15.4f} "
                        f"{mean_entropy:<15.4f}"
                    )
            logger.info("")

    # ===== Key Findings =====
    logger.info("KEY FINDINGS")
    logger.info("-" * 80)

    # Best performing model
    best_model = perf_items[0]
    logger.info(f"✓ Best Accuracy: {best_model[0]} ({best_model[1]['accuracy']:.4f})")

    # Parameter efficiency
    param_efficiency = {
        name: metrics['accuracy'] / results['models'][name].count_params() * 1e6
        for name, metrics in results['performance_analysis'].items()
    }
    best_efficiency = max(param_efficiency.items(), key=lambda x: x[1])
    logger.info(f"✓ Most Efficient: {best_efficiency[0]} ({best_efficiency[1]:.2f} acc/M params)")

    # Convergence speed
    if model_analysis and model_analysis.training_metrics:
        fastest_conv = min(
            model_analysis.training_metrics.epochs_to_convergence.items(),
            key=lambda x: x[1]
        )
        logger.info(f"✓ Fastest Convergence: {fastest_conv[0]} ({fastest_conv[1]} epochs)")

    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete window attention comparison experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary containing all experimental results
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    data = load_and_preprocess_cifar10()

    # Build and train models
    logger.info("")
    logger.info("=" * 80)
    logger.info("BUILDING AND TRAINING MODELS")
    logger.info("=" * 80)

    models = {}
    histories = {}

    for attention_type in config.attention_types:
        model_name = get_model_name(attention_type)

        # Build model
        logger.info(f"Building {model_name}...")
        model = build_attention_model(
            attention_type=attention_type,
            config=config,
            input_shape=(32, 32, 3),
            num_classes=10
        )

        # Train model
        trained_model, history = train_and_evaluate_model(
            model=model,
            model_name=model_name,
            data=data,
            config=config
        )

        models[model_name] = trained_model
        histories[model_name] = history

        # Clean up
        gc.collect()

    # Analyze and visualize results
    results = analyze_models(
        models=models,
        histories=histories,
        data=data,
        config=config
    )

    # Print summary
    print_experiment_summary(results, config)

    return results


def main() -> None:
    """
    Main execution function for the window attention comparison experiment.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("CIFAR-10 Window Attention Comparison Experiment")
    logger.info("WindowAttention vs WindowZigZag vs WindowKAN")
    logger.info("=" * 80)
    logger.info("")

    # Initialize configuration
    config = ExperimentConfig()

    # Log configuration
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  Attention Types: {config.attention_types}")
    logger.info(f"  Attention Dim: {config.attention_dim}")
    logger.info(f"  Window Size: {config.window_size}")
    logger.info(f"  Num Heads: {config.num_heads}")
    logger.info(f"  Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}, Weight Decay: {config.weight_decay}")
    logger.info("")
    logger.info("ANALYSIS MODULES ENABLED:")
    logger.info(f"  - Weight Analysis: {config.analyzer_config.analyze_weights}")
    logger.info(f"  - Calibration Analysis: {config.analyzer_config.analyze_calibration}")
    logger.info(f"  - Training Dynamics: {config.analyzer_config.analyze_training_dynamics}")
    logger.info(f"  - Spectral Analysis: {config.analyzer_config.analyze_spectral}")
    logger.info("")

    try:
        # Run experiment
        _ = run_experiment(config)

        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("")

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"EXPERIMENT FAILED: {e}")
        logger.error("=" * 80)
        logger.error("", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
