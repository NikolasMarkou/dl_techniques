"""
BandRMS Evaluation in Transformer Language Models - Comprehensive Study
=====================================================================

This experiment conducts a rigorous evaluation of BandRMS normalization compared to
standard normalization techniques in Transformer architectures for language modeling.
The study addresses key questions about bounded RMS normalization: Does constraining
the RMS magnitude within learnable spherical shells improve training stability,
model calibration, and generalization in language modeling tasks?

The hypothesis is that BandRMS's magnitude constraints create more stable training
dynamics and better-calibrated predictions compared to unconstrained normalization
methods, potentially at the cost of some model expressiveness.

Experimental Design
------------------

**Task**: Autoregressive language modeling on synthetic structured sequences
- 10,000 training sequences with n-gram-like patterns
- 128 sequence length with 8,000 vocabulary size
- Cross-entropy loss with teacher forcing

**Model Architecture**: Lightweight Transformer (6 layers, 256 hidden, 8 heads)
- Identical architecture across all variants (only normalization changes)
- Pre-normalization configuration for training stability
- Positional embeddings with learned token embeddings

**BandRMS Variants Evaluated**:

1. **BandRMS_Tight**: Very tight constraints (max_band_width=0.05)
2. **BandRMS_Medium**: Medium constraints (max_band_width=0.1)
3. **BandRMS_Loose**: Loose constraints (max_band_width=0.2)
4. **BandRMS_Adaptive**: Adaptive band width during training
5. **RMSNorm**: Standard RMS normalization (baseline)
6. **LayerNorm**: Standard layer normalization
7. **NoNorm**: No normalization (control)
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
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class BandRMSExperimentConfig:
    """
    Configuration for the BandRMS effectiveness experiment.

    This class encapsulates all configurable parameters for systematically
    evaluating BandRMS against traditional normalization techniques in Transformers.
    """

    # --- Dataset Configuration ---
    vocab_size: int = 8000
    seq_len: int = 128
    num_samples: int = 10000
    validation_split: float = 0.1
    test_split: float = 0.2

    # --- Model Architecture Parameters ---
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size_factor: int = 4
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    # --- Training Parameters ---
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    early_stopping_patience: int = 10
    monitor_metric: str = 'val_loss'

    # --- BandRMS Specific Parameters ---
    band_width_variants: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    adaptive_schedule_config: Dict[str, Any] = field(default_factory=lambda: {
        'initial_band_width': 0.2,
        'final_band_width': 0.05,
        'schedule_type': 'cosine'
    })

    # --- Normalization Variants ---
    normalization_variants: Dict[str, Callable] = field(default_factory=lambda: {
        'BandRMS_Tight': lambda config: ('band_rms', {'max_band_width': 0.05}),
        'BandRMS_Medium': lambda config: ('band_rms', {'max_band_width': 0.1}),
        'BandRMS_Loose': lambda config: ('band_rms', {'max_band_width': 0.2}),
        'BandRMS_Wide': lambda config: ('band_rms', {'max_band_width': 0.3}),
        'RMSNorm': lambda config: ('rms_norm', {}),
        'LayerNorm': lambda config: ('layer_norm', {}),
        'NoNorm': lambda config: ('none', {})
    })

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "bandrms_transformer_study"
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
        dpi=300
    ))

# ==============================================================================
# NORMALIZATION FACTORY
# ==============================================================================

def create_normalization_layer(
    norm_type: str,
    norm_params: Dict[str, Any],
    name: Optional[str] = None
) -> keras.layers.Layer:
    """
    Factory function to create different normalization layers.

    Args:
        norm_type: Type of normalization ('band_rms', 'rms_norm', 'layer_norm', 'none')
        norm_params: Parameters specific to the normalization type
        name: Optional layer name

    Returns:
        Configured normalization layer
    """
    if norm_type == 'band_rms':
        return BandRMS(
            max_band_width=norm_params.get('max_band_width', 0.1),
            axis=norm_params.get('axis', -1),
            epsilon=norm_params.get('epsilon', 1e-7),
            name=name
        )
    elif norm_type == 'rms_norm':
        return RMSNorm(
            axis=norm_params.get('axis', -1),
            epsilon=norm_params.get('epsilon', 1e-6),
            use_scale=norm_params.get('use_scale', True),
            name=name
        )
    elif norm_type == 'layer_norm':
        return keras.layers.LayerNormalization(
            axis=norm_params.get('axis', -1),
            epsilon=norm_params.get('epsilon', 1e-6),
            name=name
        )
    elif norm_type == 'none':
        return keras.layers.Lambda(lambda x: x, name=name)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

# ==============================================================================
# LIGHTWEIGHT TRANSFORMER IMPLEMENTATION
# ==============================================================================

@keras.saving.register_keras_serializable()
class LightweightTransformer(keras.Model):
    """
    Lightweight Transformer for language modeling experiments.

    Args:
        config: Experiment configuration
        normalization_type: Type of normalization to use
        norm_params: Parameters for the normalization layer
    """

    def __init__(
        self,
        config: BandRMSExperimentConfig,
        normalization_type: str,
        norm_params: Dict[str, Any],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.normalization_type = normalization_type
        self.norm_params = norm_params

        # Token and position embeddings
        self.token_embedding = keras.layers.Embedding(
            config.vocab_size,
            config.embed_dim,
            name='token_embedding'
        )
        self.position_embedding = PositionalEmbedding(
            config.seq_len,
            config.embed_dim,
            name='position_embedding'
        )

        # Transformer layers with specified normalization
        self.transformer_layers = []
        for i in range(config.num_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    hidden_size=config.embed_dim,
                    num_heads=config.num_heads,
                    intermediate_size=config.embed_dim * config.intermediate_size_factor,
                    normalization_type=self._get_transformer_norm_type(),
                    normalization_position='pre',
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    name=f'transformer_layer_{i}',
                    **self._get_transformer_norm_kwargs()
                )
            )

        # Final normalization
        self.final_norm = self._create_norm_layer('final_norm')

        # Output projection
        self.output_projection = keras.layers.Dense(
            config.vocab_size,
            name='output_projection'
        )

    def _get_transformer_norm_type(self) -> str:
        """Convert normalization type for TransformerLayer."""
        if self.normalization_type == 'band_rms':
            return 'band_rms'
        elif self.normalization_type == 'rms_norm':
            return 'rms_norm'
        elif self.normalization_type == 'layer_norm':
            return 'layer_norm'
        else:  # none
            return 'layer_norm'  # TransformerLayer will handle appropriately

    def _get_transformer_norm_kwargs(self) -> Dict[str, Any]:
        """Get normalization kwargs for TransformerLayer."""
        # TransformerLayer handles normalization internally
        # No need to pass BandRMS-specific parameters as kwargs
        return {}

    def _create_norm_layer(self, name: str) -> keras.layers.Layer:
        """Create normalization layer based on type."""
        return create_normalization_layer(
            self.normalization_type,
            self.norm_params,
            name
        )

    def call(self, inputs: keras.KerasTensor, training: bool = None) -> keras.KerasTensor:
        """Forward pass through the transformer."""
        # Embeddings
        x = self.token_embedding(inputs)
        x = self.position_embedding(x)

        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)

        # Final normalization and projection
        x = self.final_norm(x, training=training)
        logits = self.output_projection(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'normalization_type': self.normalization_type,
            'norm_params': self.norm_params,
            # Note: We don't serialize the full config object as it contains non-serializable items
        })
        return config

# ==============================================================================
# BANDRMS TRACKING CALLBACK
# ==============================================================================

class BandRMSTracker(keras.callbacks.Callback):
    """Callback to track BandRMS-specific metrics during training."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.band_rms_layers = []
        self.band_width_history = []
        self.rms_magnitude_history = []
        self.constraint_violation_history = []

    def on_train_begin(self, logs=None):
        """Find all BandRMS layers in the model."""
        def find_band_rms_layers(layer):
            layers_found = []
            if isinstance(layer, BandRMS):
                layers_found.append(layer)
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    layers_found.extend(find_band_rms_layers(sublayer))
            return layers_found

        self.band_rms_layers = find_band_rms_layers(self.model)
        logger.info(f"Found {len(self.band_rms_layers)} BandRMS layers in {self.model_name}")

    def calculate_rms_magnitude(self, layer: BandRMS) -> float:
        """Calculate current RMS magnitude for a layer."""
        if not hasattr(layer, 'built') or not layer.built:
            return 0.0

        # Get the scale parameter if it exists
        if hasattr(layer, 'scale') and layer.scale is not None:
            scales = keras.ops.convert_to_numpy(layer.scale)
            return float(np.sqrt(np.mean(np.square(scales))))
        return 0.0

    def calculate_constraint_violation(self, layer: BandRMS) -> float:
        """Calculate how much the layer violates its band constraints."""
        if not hasattr(layer, 'built') or not layer.built:
            return 0.0

        # This is a simplified metric - in practice, you'd need to track
        # the actual RMS values during forward passes
        if hasattr(layer, 'scale') and layer.scale is not None:
            scales = keras.ops.convert_to_numpy(layer.scale)
            rms = np.sqrt(np.mean(np.square(scales)))
            max_allowed = layer.max_band_width
            violation = max(0.0, rms - max_allowed) / max_allowed
            return float(violation)
        return 0.0

    def on_epoch_end(self, epoch, logs=None):
        """Track BandRMS metrics at the end of each epoch."""
        epoch_band_widths = {}
        epoch_rms_magnitudes = {}
        epoch_violations = {}

        for i, layer in enumerate(self.band_rms_layers):
            layer_name = f'layer_{i}'

            # Track band width (if adaptive)
            epoch_band_widths[layer_name] = layer.max_band_width

            # Track RMS magnitude
            epoch_rms_magnitudes[layer_name] = self.calculate_rms_magnitude(layer)

            # Track constraint violations
            epoch_violations[layer_name] = self.calculate_constraint_violation(layer)

        self.band_width_history.append(epoch_band_widths)
        self.rms_magnitude_history.append(epoch_rms_magnitudes)
        self.constraint_violation_history.append(epoch_violations)

# ==============================================================================
# DATASET GENERATION
# ==============================================================================

def create_synthetic_language_dataset(config: BandRMSExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic language modeling dataset with structured patterns.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    logger.info("Creating synthetic language modeling dataset")

    # Set seed for reproducible dataset
    np.random.seed(config.random_seed)

    sequences = []
    for _ in range(config.num_samples):
        # Create sequences with n-gram-like patterns
        seq = [np.random.randint(1, config.vocab_size)]

        for i in range(1, config.seq_len):
            if i > 2:
                # Pattern-based generation with structure
                pattern_base = (seq[i-1] + seq[i-2]) % (config.vocab_size // 2)
                pattern_token = pattern_base + (config.vocab_size // 2)

                # Add randomness but keep structure
                if np.random.random() < 0.7:
                    next_token = pattern_token
                else:
                    next_token = np.random.randint(1, config.vocab_size)
            else:
                next_token = np.random.randint(1, config.vocab_size)

            seq.append(next_token)

        sequences.append(seq)

    sequences = np.array(sequences, dtype=np.int32)

    # Split into train/test
    test_size = int(len(sequences) * config.test_split)
    train_size = len(sequences) - test_size

    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    # For language modeling: input = seq[:-1], target = seq[1:]
    x_train = train_sequences[:, :-1]
    y_train = train_sequences[:, 1:]
    x_test = test_sequences[:, :-1]
    y_test = test_sequences[:, 1:]

    logger.info(f"Created dataset: {len(x_train)} train, {len(x_test)} test sequences")
    logger.info(f"Sequence length: {x_train.shape[1]}, Vocab size: {config.vocab_size}")

    return x_train, y_train, x_test, y_test

# ==============================================================================
# MODEL TRAINING WITH TRACKING
# ==============================================================================

def train_model_with_bandrms_tracking(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: BandRMSExperimentConfig,
    model_name: str,
    output_dir: Path
) -> Tuple[keras.callbacks.History, BandRMSTracker]:
    """
    Train model with BandRMS tracking.

    Args:
        model: Keras model to train
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        config: Experiment configuration
        model_name: Name of the model
        output_dir: Output directory for saving

    Returns:
        Tuple of (training history, BandRMS tracker)
    """
    # Setup optimizer and learning rate schedule
    lr_config = {
        "type": "cosine_decay",
        "warmup_steps": config.warmup_steps,
        "warmup_start_lr": 1e-6,
        "learning_rate": config.learning_rate,
        "decay_steps": len(x_train) // config.batch_size * config.epochs,
        "alpha": 0.1
    }

    opt_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "gradient_clipping_by_norm": 1.0
    }

    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(opt_config, lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        ]
    )

    # Setup callbacks
    band_rms_tracker = BandRMSTracker(model_name)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / f"{model_name}_best.keras",
            monitor=config.monitor_metric,
            save_best_only=True,
            verbose=0
        ),
        band_rms_tracker
    ]

    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(output_dir / f"{model_name}_final.keras")

    return history, band_rms_tracker

# ==============================================================================
# STATISTICAL ANALYSIS UTILITIES
# ==============================================================================

def calculate_run_statistics(results_per_run: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics across multiple runs for each model.

    Args:
        results_per_run: Dictionary mapping model names to lists of results across runs

    Returns:
        Dictionary with mean, std, min, max for each model and metric
    """
    statistics = {}

    for model_name, run_results in results_per_run.items():
        if not run_results:
            continue

        statistics[model_name] = {}
        metrics = run_results[0].keys()

        for metric in metrics:
            values = [result[metric] for result in run_results if metric in result]

            if values:
                statistics[model_name][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

    return statistics

def calculate_perplexity_from_loss(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return np.exp(loss)

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_bandrms_experiment(config: BandRMSExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete BandRMS effectiveness experiment.

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

    logger.info("Starting BandRMS Transformer Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET CREATION =====
    logger.info("Creating synthetic language dataset...")
    x_train, y_train, x_test, y_test = create_synthetic_language_dataset(config)

    # Further split training data for validation
    val_size = int(len(x_train) * config.validation_split)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    logger.info(f"Dataset splits - Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # ===== MULTIPLE RUNS FOR STATISTICAL SIGNIFICANCE =====
    logger.info(f"Running {config.n_runs} repetitions for statistical significance...")

    all_trained_models = {}
    all_histories = {}
    all_band_rms_trackers = {}
    results_per_run = {variant_name: [] for variant_name in config.normalization_variants.keys()}

    for run_idx in range(config.n_runs):
        logger.info(f"Starting run {run_idx + 1}/{config.n_runs}")
        run_seed = config.random_seed + run_idx * 1000
        keras.utils.set_random_seed(run_seed)

        current_run_models = {}
        current_run_histories = {}
        current_run_trackers = {}

        for variant_name, variant_factory in config.normalization_variants.items():
            logger.info(f"--- Training {variant_name} (Run {run_idx + 1}) ---")

            norm_type, norm_params = variant_factory(config)

            # Create model
            model = LightweightTransformer(
                config=config,
                normalization_type=norm_type,
                norm_params=norm_params,
                name=f"{variant_name}_run{run_idx}"
            )

            # Build model
            dummy_input = np.zeros((1, config.seq_len - 1), dtype=np.int32)
            _ = model(dummy_input)

            if run_idx == 0:
                model.summary(print_fn=logger.info)

            # Train model
            run_output_dir = experiment_dir / "models" / f"run_{run_idx}" / variant_name
            run_output_dir.mkdir(parents=True, exist_ok=True)

            history, band_rms_tracker = train_model_with_bandrms_tracking(
                model, x_train, y_train, x_val, y_val,
                config, f"{variant_name}_run{run_idx}", run_output_dir
            )

            # Evaluate model with manual accuracy calculation
            try:
                logger.info(f"Evaluating {variant_name} (Run {run_idx + 1})...")

                # Get model predictions
                predictions = model.predict(x_test, verbose=0)

                # Calculate accuracy manually
                y_pred_classes = np.argmax(predictions, axis=-1).flatten()
                y_true_classes = y_test.flatten()
                manual_accuracy = np.mean(y_pred_classes == y_true_classes)

                # Calculate loss manually
                loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                manual_loss = float(loss_fn(y_test, predictions).numpy())

                # Calculate perplexity
                perplexity = calculate_perplexity_from_loss(manual_loss)

                # Store results
                results_per_run[variant_name].append({
                    'accuracy': manual_accuracy,
                    'loss': manual_loss,
                    'perplexity': perplexity,
                    'best_val_loss': float(min(history.history['val_loss'])),
                    'best_val_accuracy': float(max(history.history['val_accuracy'])),
                    'epochs_trained': len(history.history['loss'])
                })

                logger.info(f"{variant_name} (Run {run_idx + 1}): "
                          f"Accuracy={manual_accuracy:.4f}, "
                          f"Loss={manual_loss:.4f}, "
                          f"Perplexity={perplexity:.2f}")

            except Exception as e:
                logger.error(f"Error evaluating {variant_name} (Run {run_idx + 1}): {e}")
                results_per_run[variant_name].append({'error': str(e)})

            current_run_models[variant_name] = model
            current_run_histories[variant_name] = history.history
            current_run_trackers[variant_name] = band_rms_tracker

        # Store final run data
        if run_idx == config.n_runs - 1:
            all_trained_models = current_run_models
            all_histories = current_run_histories
            all_band_rms_trackers = current_run_trackers

        # Clean up to save memory
        del current_run_models, current_run_histories, current_run_trackers
        gc.collect()

    # ===== STATISTICAL ANALYSIS =====
    logger.info("Calculating statistics across runs...")
    run_statistics = calculate_run_statistics(results_per_run)

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("Performing comprehensive analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        # Prepare data for analyzer
        test_data = DataInput(x_data=x_test, y_data=y_test)

        analyzer = ModelAnalyzer(
            models=all_trained_models,
            training_history=all_histories,
            config=config.analyzer_config,
            output_dir=str(experiment_dir / "model_analysis")
        )

        model_analysis_results = analyzer.analyze(data=test_data)
        logger.info("Model analysis completed successfully!")

    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

    # ===== VISUALIZATION =====
    logger.info("Generating visualizations...")
    create_bandrms_comparison_plots(run_statistics, all_histories, experiment_dir / "visualizations")
    create_bandrms_analysis_plots(all_band_rms_trackers, experiment_dir / "visualizations")

    # ===== COMPILE RESULTS =====
    results = {
        'run_statistics': run_statistics,
        'model_analysis': model_analysis_results,
        'histories': all_histories,
        'band_rms_trackers': all_band_rms_trackers,
        'trained_models': all_trained_models,
        'config': config,
        'experiment_dir': experiment_dir
    }

    # ===== SAVE RESULTS =====
    save_experiment_results(results, experiment_dir)
    print_experiment_summary(results)

    return results

# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def create_bandrms_comparison_plots(
    statistics: Dict[str, Dict[str, float]],
    histories: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create comparison plots for BandRMS experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Statistical comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        models = list(statistics.keys())

        # Accuracy comparison
        accuracies = [statistics[model]['accuracy']['mean'] for model in models]
        accuracy_stds = [statistics[model]['accuracy']['std'] for model in models]

        colors = ['red' if 'BandRMS' in model else 'blue' if 'RMS' in model else 'gray'
                 for model in models]

        ax1 = axes[0, 0]
        bars = ax1.bar(models, accuracies, yerr=accuracy_stds, capsize=5, color=colors, alpha=0.7)
        ax1.set_title('Test Accuracy Comparison (Mean ± Std)')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)

        for bar, acc, std in zip(bars, accuracies, accuracy_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                    f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        # Perplexity comparison
        ax2 = axes[0, 1]
        perplexities = [statistics[model]['perplexity']['mean'] for model in models]
        perplexity_stds = [statistics[model]['perplexity']['std'] for model in models]

        bars2 = ax2.bar(models, perplexities, yerr=perplexity_stds, capsize=5, color=colors, alpha=0.7)
        ax2.set_title('Test Perplexity Comparison (Mean ± Std)')
        ax2.set_ylabel('Perplexity')
        ax2.tick_params(axis='x', rotation=45)

        # Training curves comparison
        ax3 = axes[1, 0]
        for model_name, history in histories.items():
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                color = 'red' if 'BandRMS' in model_name else 'blue' if 'RMS' in model_name else 'gray'
                ax3.plot(epochs, history['val_loss'], label=model_name, color=color, alpha=0.7)

        ax3.set_title('Validation Loss During Training')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Validation Loss')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Training accuracy curves
        ax4 = axes[1, 1]
        for model_name, history in histories.items():
            if 'val_accuracy' in history:
                epochs = range(1, len(history['val_accuracy']) + 1)
                color = 'red' if 'BandRMS' in model_name else 'blue' if 'RMS' in model_name else 'gray'
                ax4.plot(epochs, history['val_accuracy'], label=model_name, color=color, alpha=0.7)

        ax4.set_title('Validation Accuracy During Training')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Validation Accuracy')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'bandrms_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("BandRMS comparison plots saved")

    except Exception as e:
        logger.error(f"Failed to create BandRMS comparison plots: {e}")

def create_bandrms_analysis_plots(
    trackers: Dict[str, BandRMSTracker],
    output_dir: Path
) -> None:
    """Create BandRMS-specific analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not trackers:
            logger.info("No BandRMS trackers found, skipping BandRMS analysis plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: RMS magnitude evolution
        ax1 = axes[0, 0]
        for model_name, tracker in trackers.items():
            if 'BandRMS' in model_name and tracker.rms_magnitude_history:
                epochs = range(len(tracker.rms_magnitude_history))
                rms_values = [metrics.get('layer_0', 0) for metrics in tracker.rms_magnitude_history]
                ax1.plot(epochs, rms_values, label=model_name, marker='o', markersize=2)

        ax1.set_title('RMS Magnitude Evolution During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMS Magnitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Constraint violation evolution
        ax2 = axes[0, 1]
        for model_name, tracker in trackers.items():
            if 'BandRMS' in model_name and tracker.constraint_violation_history:
                epochs = range(len(tracker.constraint_violation_history))
                violations = [metrics.get('layer_0', 0) for metrics in tracker.constraint_violation_history]
                ax2.plot(epochs, violations, label=model_name, marker='o', markersize=2)

        ax2.set_title('Constraint Violation During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Violation Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Band width comparison (if adaptive)
        ax3 = axes[1, 0]
        band_widths = []
        model_names = []
        for model_name, tracker in trackers.items():
            if 'BandRMS' in model_name and tracker.band_width_history:
                band_width = tracker.band_width_history[0].get('layer_0', 0)
                band_widths.append(band_width)
                model_names.append(model_name)

        if band_widths:
            colors = plt.cm.viridis(np.linspace(0, 1, len(band_widths)))
            bars = ax3.bar(model_names, band_widths, color=colors)
            ax3.set_title('Band Width Settings Comparison')
            ax3.set_ylabel('Max Band Width')
            ax3.tick_params(axis='x', rotation=45)

            for bar, bw in zip(bars, band_widths):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{bw:.3f}', ha='center', va='bottom', fontsize=9)

        # Plot 4: Final constraint adherence
        ax4 = axes[1, 1]
        final_violations = []
        for model_name, tracker in trackers.items():
            if 'BandRMS' in model_name and tracker.constraint_violation_history:
                final_violation = tracker.constraint_violation_history[-1].get('layer_0', 0)
                final_violations.append(final_violation)

        if final_violations and len(final_violations) == len(model_names):
            bars = ax4.bar(model_names, final_violations, color=colors)
            ax4.set_title('Final Constraint Violations')
            ax4.set_ylabel('Final Violation Ratio')
            ax4.tick_params(axis='x', rotation=45)

            for bar, viol in zip(bars, final_violations):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{viol:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'bandrms_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("BandRMS analysis plots saved")

    except Exception as e:
        logger.error(f"Failed to create BandRMS analysis plots: {e}")

# ==============================================================================
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(results: Dict[str, Any], experiment_dir: Path) -> None:
    """Save experiment results in multiple formats."""
    try:
        def convert_numpy_to_python(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_to_python(item) for item in obj)
            else:
                return obj

        # Save configuration
        config = results['config']
        config_dict = {
            'experiment_name': config.experiment_name,
            'normalization_variants': list(config.normalization_variants.keys()),
            'vocab_size': config.vocab_size,
            'seq_len': config.seq_len,
            'embed_dim': config.embed_dim,
            'num_layers': config.num_layers,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'n_runs': config.n_runs,
            'band_width_variants': config.band_width_variants
        }

        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save statistical results
        statistical_results_converted = convert_numpy_to_python(results['run_statistics'])
        with open(experiment_dir / "statistical_results.json", 'w') as f:
            json.dump(statistical_results_converted, f, indent=2)

        # Save BandRMS tracking data
        if results['band_rms_trackers']:
            band_rms_data = {}
            for model_name, tracker in results['band_rms_trackers'].items():
                band_rms_data[model_name] = {
                    'band_width_history': convert_numpy_to_python(tracker.band_width_history),
                    'rms_magnitude_history': convert_numpy_to_python(tracker.rms_magnitude_history),
                    'constraint_violation_history': convert_numpy_to_python(tracker.constraint_violation_history)
                }

            with open(experiment_dir / "bandrms_tracking_data.json", 'w') as f:
                json.dump(band_rms_data, f, indent=2)

        # Save models
        models_dir = experiment_dir / "models" / "final"
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, model in results['trained_models'].items():
            model_path = models_dir / f"{name}.keras"
            model.save(model_path)

        logger.info("Experiment results saved successfully")

    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}", exc_info=True)

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive experiment summary."""
    logger.info("=" * 80)
    logger.info("BANDRMS TRANSFORMER EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Statistical results
    if 'run_statistics' in results and results['run_statistics']:
        logger.info("STATISTICAL RESULTS (Mean ± Std across runs):")
        logger.info(f"{'Model':<20} {'Accuracy':<18} {'Perplexity':<15} {'Loss':<15} {'Runs':<5}")
        logger.info("-" * 80)

        for model_name, stats in results['run_statistics'].items():
            acc_mean = stats['accuracy']['mean']
            acc_std = stats['accuracy']['std']
            ppl_mean = stats['perplexity']['mean']
            ppl_std = stats['perplexity']['std']
            loss_mean = stats['loss']['mean']
            loss_std = stats['loss']['std']
            n_runs = stats['accuracy']['count']

            logger.info(f"{model_name:<20} {acc_mean:.4f}±{acc_std:.4f}    "
                       f"{ppl_mean:.2f}±{ppl_std:.2f}    "
                       f"{loss_mean:.4f}±{loss_std:.4f}   {n_runs:<5}")

    # Key insights
    logger.info("\nKEY INSIGHTS:")

    if 'run_statistics' in results and results['run_statistics']:
        # Best performing model
        best_model = max(results['run_statistics'].items(),
                        key=lambda x: x[1]['accuracy']['mean'])
        logger.info(f"   Best Accuracy: {best_model[0]} ({best_model[1]['accuracy']['mean']:.4f})")

        # Most stable model
        most_stable = min(results['run_statistics'].items(),
                         key=lambda x: x[1]['accuracy']['std'])
        logger.info(f"   Most Stable: {most_stable[0]} (Accuracy Std: {most_stable[1]['accuracy']['std']:.4f})")

        # BandRMS analysis
        band_rms_models = {k: v for k, v in results['run_statistics'].items() if 'BandRMS' in k}
        other_models = {k: v for k, v in results['run_statistics'].items() if 'BandRMS' not in k}

        if band_rms_models:
            logger.info("   BandRMS Variants:")
            for model_name, stats in band_rms_models.items():
                logger.info(f"      {model_name:<18}: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f} "
                           f"(Perplexity: {stats['perplexity']['mean']:.2f})")

        if other_models:
            logger.info("   Baseline Methods:")
            for model_name, stats in other_models.items():
                logger.info(f"      {model_name:<18}: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f} "
                           f"(Perplexity: {stats['perplexity']['mean']:.2f})")

    logger.info("=" * 80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for the BandRMS experiment."""
    logger.info("BandRMS Transformer Effectiveness Experiment")
    logger.info("=" * 80)

    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(f"GPU configuration: {e}")

    # Initialize configuration
    config = BandRMSExperimentConfig()

    # Log configuration
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"   Normalization variants: {list(config.normalization_variants.keys())}")
    logger.info(f"   Architecture: {config.num_layers} layers, {config.embed_dim} hidden, {config.num_heads} heads")
    logger.info(f"   Training: {config.epochs} epochs, {config.batch_size} batch size")
    logger.info(f"   Dataset: {config.num_samples} samples, {config.vocab_size} vocab, {config.seq_len} length")
    logger.info(f"   Band widths tested: {config.band_width_variants}")
    logger.info(f"   Number of runs: {config.n_runs}")
    logger.info("")

    try:
        # Run experiment
        _ = run_bandrms_experiment(config)
        logger.info("BandRMS experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()