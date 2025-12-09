"""
CIFAR-10 Anchor Attention Comparison Experiment
================================================

This experiment evaluates the AnchorAttention mechanism with different
probability output configurations on the CIFAR-10 image classification task.

Anchor Attention is a hierarchical, memory-efficient attention mechanism that
creates an information bottleneck through a small, fixed set of "anchor" tokens.
This reduces the quadratic complexity of attention for long sequences while
preserving the model's ability to access global context.

Research Questions
------------------

1. **Probability Output Impact**: How do different probability functions
   (softmax, sparsemax, threshmax, adaptive) affect attention sparsity
   and model performance?

2. **Anchor Token Count**: What is the optimal number of anchor tokens
   for the CIFAR-10 task? How does performance scale with anchor count?

3. **Efficiency-Accuracy Trade-off**: Does the hierarchical attention
   structure provide computational benefits while maintaining accuracy?

4. **Calibration**: Do sparse attention mechanisms (sparsemax, threshmax)
   produce better-calibrated probability distributions?

Experimental Design
-------------------

**Dataset**: CIFAR-10 (10 classes, 32x32 RGB images)
- 50,000 training images
- 10,000 test images
- Standard preprocessing with normalization

**Model Architecture**: Lightweight Vision Transformer (ViT) architecture:

```
Input (32x32x3)
      |
Conv2D (5x5, stride=2) as Patch Embedding  -> [16x16xdim] -> [256, dim]
      |
Transformer Encoder Block x N:
  - LayerNorm
  - AnchorAttention (with configurable probability output)
  - Residual
  - LayerNorm
  - MLP (4*dim hidden)
  - Residual
      |
Layer Normalization
      |
GlobalAveragePooling1D
      |
Dense(10) + Softmax
```

**Probability Output Configurations**:

1. **Softmax** (Baseline): Standard exponential normalization
2. **Sparsemax**: Euclidean projection onto simplex (sparse outputs)
3. **ThreshMax**: Differentiable confidence thresholding
4. **Adaptive Softmax**: Entropy-based temperature scaling

**Training Configuration**:
- Optimizer: AdamW with weight decay
- Learning rate: 0.001 with cosine decay
- Batch size: 64
- Epochs: 100
- Early stopping on validation accuracy

Expected Outcomes
-----------------
- Sparse attention (sparsemax, threshmax) may improve interpretability
- Adaptive softmax may improve calibration on uncertain predictions
- Different anchor counts will show accuracy-efficiency trade-offs
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
from typing import Dict, Any, List, Tuple, Optional

# ==============================================================================
# LOCAL IMPORTS
# ==============================================================================

from dl_techniques.utils.logger import logger
from dl_techniques.utils.train import TrainingConfig, train_model

from dl_techniques.layers.attention import AnchorAttention

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
class ProbabilityConfig:
    """
    Configuration for a probability output type.

    Attributes:
        name: Human-readable name for the configuration.
        probability_type: Type string for ProbabilityOutput layer.
        probability_config: Optional dict of type-specific parameters.
    """
    name: str
    probability_type: str
    probability_config: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """
    Configuration for the anchor attention comparison experiment.

    Attributes:
        probability_configs: List of probability output configurations to test.
        num_anchor_tokens: Number of anchor tokens for hierarchical attention.
        attention_dim: Embedding dimension for attention layers.
        num_attention_layers: Number of transformer encoder blocks.
        mlp_dim_multiplier: Multiplier for the MLP hidden layer size.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for regularization.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW optimizer.
        validation_split: Fraction of training data for validation.
        early_stopping_patience: Patience for early stopping callback.
        reduce_lr_patience: Patience for learning rate reduction.
        reduce_lr_factor: Factor for learning rate reduction.
        min_learning_rate: Minimum learning rate.
        monitor_metric: Metric to monitor for callbacks.
        analyzer_config: Configuration for ModelAnalyzer.
        output_dir: Directory for saving results.
    """
    # Probability output configurations to compare
    probability_configs: List[ProbabilityConfig] = field(default_factory=lambda: [
        ProbabilityConfig(
            name="Sparsemax",
            probability_type="sparsemax",
            probability_config={"axis": -1}
        ),
        ProbabilityConfig(
            name="Softmax",
            probability_type="softmax",
            probability_config={"axis": -1}
        ),
        ProbabilityConfig(
            name="ThreshMax",
            probability_type="threshmax",
            probability_config={"axis": -1, "slope": 10.0, "trainable_slope": True}
        ),
        ProbabilityConfig(
            name="Adaptive",
            probability_type="adaptive",
            probability_config={"min_temp": 0.1, "max_temp": 2.0, "entropy_threshold": 0.5}
        ),
    ])

    # Anchor attention configuration
    num_anchor_tokens: int = 16  # ~6% of 256 tokens for hierarchical structure

    # Model architecture
    attention_dim: int = 128
    num_attention_layers: int = 6
    mlp_dim_multiplier: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1

    # Training configuration
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_split: float = 0.1

    # Training callbacks configuration
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    monitor_metric: str = 'val_accuracy'

    # Analysis configuration
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=False,
        analyze_training_dynamics=True,
        analyze_spectral=True,
    ))

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path(
        f"results/anchor_attention_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ))


# ==============================================================================
# DATA LOADING
# ==============================================================================

@dataclass
class CIFAR10Data:
    """
    Container for CIFAR-10 dataset.

    Attributes:
        x_train: Training images, shape (N, 32, 32, 3), normalized to [0, 1].
        y_train: Training labels, one-hot encoded, shape (N, 10).
        x_test: Test images, shape (M, 32, 32, 3), normalized to [0, 1].
        y_test: Test labels, one-hot encoded, shape (M, 10).
        class_names: List of class names for CIFAR-10.
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

def create_anchor_attention_layer(
    index: int,
    dim: int,
    num_heads: int,
    dropout_rate: float,
    probability_config: ProbabilityConfig
) -> keras.layers.Layer:
    """
    Factory function to create an AnchorAttention layer with specified
    probability output configuration.

    Args:
        index: Layer index (for naming).
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention.
        probability_config: Probability output configuration.

    Returns:
        Configured AnchorAttention layer.
    """
    layer_name = f"anchor_attn_{probability_config.name.lower()}_{index}"

    return AnchorAttention(
        dim=dim,
        num_heads=num_heads,
        head_dim=None,  # Use default dim // num_heads
        dropout_rate=dropout_rate,
        use_bias=True,
        probability_type=probability_config.probability_type,
        probability_config=probability_config.probability_config,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        name=layer_name
    )


def build_anchor_attention_model(
    probability_config: ProbabilityConfig,
    config: ExperimentConfig,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10
) -> keras.Model:
    """
    Build a Transformer-style model with AnchorAttention and a CNN stem.

    Architecture:
        Conv Stem (32x32 -> 16x16) ->
        [Transformer Block (AnchorAttention + MLP)] x N ->
        Layer Norm -> Global Pool -> Dense -> Output

    Args:
        probability_config: Probability output configuration for attention.
        config: Experiment configuration.
        input_shape: Input image shape.
        num_classes: Number of output classes.

    Returns:
        A Keras model.
    """
    inputs = keras.Input(shape=input_shape, name='input')

    # 1. Patch Embedding using a Conv layer
    # 32x32 -> 16x16, with embedding dimension 'attention_dim'
    x = keras.layers.Conv2D(
        filters=config.attention_dim,
        kernel_size=5,
        strides=2,
        padding='same',
        kernel_initializer='he_normal',
        name='patch_embed_conv'
    )(inputs)

    # Reshape to sequence: [batch, 16*16, dim] = [batch, 256, dim]
    x = keras.layers.Reshape(
        target_shape=(-1, config.attention_dim),
        name='flatten_patches'
    )(x)

    # 2. Transformer Blocks with AnchorAttention
    for i in range(config.num_attention_layers):
        # --- Attention Sub-block ---
        x_res = x
        x = keras.layers.LayerNormalization(name=f'ln1_block{i}')(x)

        attn_layer = create_anchor_attention_layer(
            index=i,
            dim=config.attention_dim,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            probability_config=probability_config
        )

        # Apply anchor attention with hierarchical mode
        x = attn_layer(x, num_anchor_tokens=config.num_anchor_tokens)
        x = keras.layers.Dropout(config.dropout_rate, name=f'drop_attn_block{i}')(x)
        x = keras.layers.Add(name=f'add_attn_block{i}')([x_res, x])

        # --- MLP Sub-block ---
        x_res = x
        x = keras.layers.LayerNormalization(name=f'ln2_block{i}')(x)

        mlp_dim = config.attention_dim * config.mlp_dim_multiplier
        x = keras.layers.Dense(
            mlp_dim,
            activation='gelu',
            kernel_initializer='he_normal',
            name=f'mlp_dense1_block{i}'
        )(x)
        x = keras.layers.Dropout(config.dropout_rate, name=f'drop_mlp_block{i}')(x)
        x = keras.layers.Dense(
            config.attention_dim,
            kernel_initializer='he_normal',
            name=f'mlp_dense2_block{i}'
        )(x)
        x = keras.layers.Add(name=f'add_mlp_block{i}')([x_res, x])

    # 3. Final Layer Normalization
    x = keras.layers.LayerNormalization(name='final_ln')(x)

    # 4. Global Average Pooling over the sequence dimension
    x = keras.layers.GlobalAveragePooling1D(name='gap')(x)

    # 5. Output layer
    logits = keras.layers.Dense(
        num_classes,
        activation='linear',
        kernel_initializer='glorot_uniform',
        name='logits'
    )(x)

    # Create model
    model_name = f"anchor_attn_{probability_config.name.lower()}"
    model = keras.Model(inputs=inputs, outputs=logits, name=model_name)

    return model


def get_model_name(probability_config: ProbabilityConfig) -> str:
    """
    Get a human-readable model name for the probability configuration.

    Args:
        probability_config: Probability output configuration.

    Returns:
        Human-readable model name string.
    """
    return f"AnchorAttn ({probability_config.name})"


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
        model: Keras model to train.
        model_name: Name for logging and saving.
        data: CIFAR10Data object.
        config: Experiment configuration.

    Returns:
        Tuple of (trained_model, history_dict).
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Training: {model_name}")
    logger.info("=" * 80)

    model.summary()

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
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        ),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Configure training
    safe_model_name = (
        model_name.replace(' ', '_')
        .replace('+', '_')
        .replace('(', '')
        .replace(')', '')
    )
    training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        model_name=safe_model_name,
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
        models: Dictionary mapping model names to trained models.
        histories: Dictionary mapping model names to training histories.
        data: CIFAR10Data object.
        config: Experiment configuration.

    Returns:
        Dictionary containing all analysis results.
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

        logger.info(
            f"{model_name}: Acc={test_acc:.4f}, Top-5={top5_acc:.4f}, Loss={test_loss:.4f}"
        )

    # ===== Model Analyzer Integration =====
    logger.info("")
    logger.info("Running ModelAnalyzer...")

    # Prepare data input for analyzer
    data_input = DataInput(
        x_data=data.x_test,
        y_data=data.y_test
    )

    # Initialize ModelAnalyzer
    analyzer = ModelAnalyzer(
        models=models,
        training_history=histories,
        config=config.analyzer_config,
        output_dir=config.output_dir
    )

    # Analyze all models
    analysis_results = analyzer.analyze(data=data_input)
    results['model_analysis'] = analysis_results

    # ===== Visualization =====
    logger.info("")
    logger.info("Generating visualizations...")

    viz_manager = VisualizationManager(
        experiment_name="anchor_attention_comparison",
        output_dir=config.output_dir
    )

    # Register visualization templates
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
        title="Training Curves: Anchor Attention Probability Output Comparison"
    )

    # Pre-calculate predictions for all models
    all_predictions = {
        name: model.predict(data.x_test, verbose=0)
        for name, model in models.items()
    }

    # Convert y_true from one-hot to class indices
    y_true_indices = np.argmax(data.y_test, axis=1)

    # --- 2. Individual Confusion Matrices ---
    all_classification_results = {}
    for model_name, predictions in all_predictions.items():
        try:
            classification_data = ClassificationResults(
                y_true=y_true_indices,
                y_pred=np.argmax(predictions, axis=1),
                y_prob=predictions,
                class_names=data.class_names,
                model_name=model_name
            )
            all_classification_results[model_name] = classification_data
        except Exception as e:
            logger.error(f"Failed to prepare classification results for {model_name}: {e}")

    # --- 3. Multi-Model Comparison ---
    if all_classification_results:
        multi_model_data = MultiModelClassification(
            results=all_classification_results,
            dataset_name="CIFAR-10"
        )

        viz_manager.visualize(
            data=multi_model_data,
            plugin_name="confusion_matrix",
            normalize='true',
            show=False
        )
        logger.info("Multi-model confusion matrix visualization created")
        logger.info("")

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
        results: Dictionary containing all experimental results.
        config: Experiment configuration.
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
    logger.info(f"Num Attention Layers: {config.num_attention_layers}")
    logger.info(f"Number of Heads: {config.num_heads}")
    logger.info(f"Num Anchor Tokens: {config.num_anchor_tokens}")
    logger.info(f"Dropout Rate: {config.dropout_rate}")
    logger.info("")

    # ===== Performance Comparison =====
    logger.info("PERFORMANCE COMPARISON (Test Set)")
    logger.info("-" * 80)
    logger.info(
        f"{'Model':<30} {'Accuracy':<12} {'Top-5 Acc':<12} {'Loss':<12} {'Params':<12}"
    )
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
        if (config.analyzer_config.analyze_calibration and
                model_analysis.calibration_metrics):
            logger.info("CALIBRATION METRICS")
            logger.info("-" * 80)
            logger.info(
                f"{'Model':<30} {'ECE':<12} {'Brier Score':<15} {'Mean Entropy':<15}"
            )
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
        if (config.analyzer_config.analyze_training_dynamics and
                model_analysis.training_metrics):
            logger.info("TRAINING DYNAMICS")
            logger.info("-" * 80)
            logger.info(
                f"{'Model':<30} {'Epochs to Conv':<15} {'Overfitting':<15} {'Stability':<12}"
            )
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
        if (config.analyzer_config.analyze_spectral and
                model_analysis.spectral_analysis is not None and
                not model_analysis.spectral_analysis.empty):
            logger.info("SPECTRAL ANALYSIS (WeightWatcher)")
            logger.info("-" * 80)
            logger.info(
                f"{'Model':<30} {'Mean alpha':<12} {'Concentration':<15} {'Matrix Entropy':<15}"
            )
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
    logger.info(f"Best Accuracy: {best_model[0]} ({best_model[1]['accuracy']:.4f})")

    # Parameter efficiency
    param_efficiency = {
        name: metrics['accuracy'] / results['models'][name].count_params() * 1e6
        for name, metrics in results['performance_analysis'].items()
    }
    best_efficiency = max(param_efficiency.items(), key=lambda x: x[1])
    logger.info(
        f"Most Efficient: {best_efficiency[0]} ({best_efficiency[1]:.2f} acc/M params)"
    )

    # Convergence speed
    if model_analysis and model_analysis.training_metrics:
        fastest_conv = min(
            model_analysis.training_metrics.epochs_to_convergence.items(),
            key=lambda x: x[1]
        )
        logger.info(f"Fastest Convergence: {fastest_conv[0]} ({fastest_conv[1]} epochs)")

    # Best calibration
    if (model_analysis and model_analysis.calibration_metrics and
            config.analyzer_config.analyze_calibration):
        best_calibration = min(
            model_analysis.calibration_metrics.items(),
            key=lambda x: x[1].get('ece', float('inf'))
        )
        logger.info(
            f"Best Calibration (ECE): {best_calibration[0]} "
            f"({best_calibration[1].get('ece', 0.0):.4f})"
        )

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
    Run the complete anchor attention comparison experiment.

    Args:
        config: Experiment configuration.

    Returns:
        Dictionary containing all experimental results.
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

    for prob_config in config.probability_configs:
        model_name = get_model_name(prob_config)

        # Build model
        logger.info(f"Building {model_name}...")
        model = build_anchor_attention_model(
            probability_config=prob_config,
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
    Main execution function for the anchor attention comparison experiment.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("CIFAR-10 Anchor Attention Comparison Experiment")
    logger.info("Probability Output: Softmax vs Sparsemax vs ThreshMax vs Adaptive")
    logger.info("=" * 80)
    logger.info("")

    # Initialize configuration
    config = ExperimentConfig()

    # Log configuration
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  Probability Configs: {[p.name for p in config.probability_configs]}")
    logger.info(f"  Num Anchor Tokens: {config.num_anchor_tokens}")
    logger.info(f"  Attention Dim: {config.attention_dim}")
    logger.info(f"  Num Attention Layers: {config.num_attention_layers}")
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