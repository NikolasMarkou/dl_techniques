"""
N-BEATS Forecasting with Scientific Forecasting Layers Integration

This module orchestrates end-to-end training of N-BEATS (Neural Basis Expansion Analysis
for Time Series) enhanced with Valeriy Manokhin's scientific forecasting principles
through two specialized architectural layers: NaiveResidual and ForecastabilityGate.

What This Code Does
-------------------
1. **Creates Enhanced N-BEATS Models**: Wraps the base N-BEATS architecture with
   forecasting layers that enforce scientific forecasting best practices at the
   architectural level.

2. **Trains on Synthetic Multi-Pattern Data**: Uses a comprehensive time series generator
   to create diverse training patterns (80+ types) covering trends, seasonality,
   volatility, regimes, and domain-specific behaviors (financial, weather, industrial, etc.).

3. **Implements Streaming Data Pipeline**: Generates infinite batches on-the-fly with
   per-instance normalization to [-1, 1] range, enabling meta-learning across pattern types.

4. **Provides Architectural Guarantees**:
   - NaiveResidual ensures graceful degradation to Random Walk baseline
   - ForecastabilityGate learns to suppress predictions on unforecastable inputs

Architecture Enhancement
------------------------
The code modifies standard N-BEATS by wrapping its output with forecasting layers:

**Standard N-BEATS Flow:**
    Input → N-BEATS Stacks → Deep Forecast

**Enhanced Flow (this implementation):**
    Input → N-BEATS Stacks → Deep Forecast
                          ↓
                    NaiveResidual Layer (adds Random Walk baseline)
                          ↓
                    ForecastabilityGate (weights deep vs naive based on complexity)
                          ↓
                    Final Forecast

**Mathematical Formulation:**
    naive_forecast = last_observed_value (repeated for forecast horizon)

    If NaiveResidual only:
        output = deep_forecast + naive_forecast

    If NaiveResidual + ForecastabilityGate:
        α = sigmoid(ComplexityAnalyzer(input))  # Learned forecastability score
        output = α · deep_forecast + (1 - α) · naive_forecast

Key Components
--------------
1. **NBeatsTrainer**: Main orchestration class
   - Initializes time series generator with 80+ pattern types
   - Creates data processor for weighted pattern sampling
   - Builds enhanced model with optional forecasting layers
   - Manages training loop with callbacks and analysis

2. **MultiPatternDataProcessor**: Streaming data pipeline
   - Generates batches from weighted pattern distribution
   - Applies per-instance min-max normalization to [-1, 1]
   - Samples random windows (backcast + forecast) from long series
   - Handles both single-output and dual-output (forecast + residual) targets

3. **Model Creation (create_model)**:
   - Instantiates base N-BEATS from factory
   - Wraps outputs with NaiveResidual layer (if enabled)
   - Adds ForecastabilityGate for adaptive weighting (if enabled)
   - Compiles with MASE loss or MSE, plus optional reconstruction loss

4. **PatternPerformanceCallback**: Visualization
   - Generates prediction plots across pattern categories
   - Tests model on unseen series from each category
   - Saves visualizations at regular intervals

Training Data Strategy
----------------------
**Pattern Diversity**: 80+ time series patterns across 11 categories:
- Basic: trend (8 types), seasonal (7 types), composite (4 types)
- Stochastic: AR/MA/ARMA processes (7 types)
- Domain-specific: financial (5), weather (4), biomedical (4), industrial (4)
- Advanced: intermittent (3), volatility/GARCH (3), regime-switching (2)
- Anomalies: structural breaks (3), outliers (3), chaotic systems (4)

**Weighted Sampling**: Categories have configurable weights to balance learning:
- Higher weights (1.5): financial patterns (more challenging)
- Medium weights (1.2-1.3): composite, weather, biomedical, industrial
- Base weights (1.0): trend, seasonal, intermittent

**Per-Instance Normalization**: Each series normalized independently to [-1, 1]
to enable meta-learning across vastly different scales and dynamics.

Configuration Options
---------------------
**Forecasting Layers** (controllable via CLI/config):
- `use_naive_residual`: Enable structural naive baseline (default: True)
- `use_forecastability_gate`: Enable complexity gating (default: True)
- `gate_hidden_units`: Capacity of complexity analyzer (default: 16)
- `gate_activation`: Gate activation function (default: 'relu')

**N-BEATS Architecture**:
- `stack_types`: Block types ['trend', 'seasonality', 'generic']
- `nb_blocks_per_stack`: Blocks per stack (default: 3)
- `hidden_layer_units`: MLP capacity (default: 128)
- `reconstruction_loss_weight`: Backcast regularization (default: 0.5)

**Training**:
- `learning_rate`: Base LR with cosine decay + warmup (default: 1e-4)
- `batch_size`: Samples per batch (default: 128)
- `steps_per_epoch`: Training iterations per epoch (default: 1000)
- `optimizer`: 'adam' or 'adamw' (default: 'adam')

Foundational Mathematics
------------------------
**N-BEATS Basis Expansion** (unchanged from original):
For block l, input x_l produces expansion coefficients θ via MLP:

    θ_l = MLP(x_l)
    backcast_l = Σ θ_l^b · g^b(t)  # Reconstruct input
    forecast_l = Σ θ_l^f · g^f(t)  # Predict future

    x_{l+1} = x_l - backcast_l  # Residual for next block

Final forecast = Σ forecast_l (sum across all blocks)

Where basis functions g(t) are:
- Trend blocks: Polynomials [1, t, t², t³, ...]
- Seasonality blocks: Fourier series [sin(2πkt/T), cos(2πkt/T), ...]
- Generic blocks: Learned linear projections

**Enhanced Output** (this implementation):
    y_naive = x[-1]  # Random Walk: repeat last value
    y_deep = NBeatsNet(x)  # Standard N-BEATS forecast

    # Complexity assessment
    α = σ(Dense_16(ReLU(Dense(Flatten(x)))))  # α ∈ [0,1]

    # Adaptive combination
    y_final = α · y_deep + (1 - α) · y_naive

When α → 1: Model trusts deep forecast (forecastable input)
When α → 0: Model falls back to naive (unforecastable/noisy input)

Usage Examples
--------------
Basic training with all defaults:
```bash
    python train.py --epochs 200 --batch_size 128
```

Train without forecasting layers (vanilla N-BEATS):
```bash
    python train.py --no-naive-residual --no-forecastability-gate
```

Train with only NaiveResidual (no adaptive gating):
```bash
    python train.py --no-forecastability-gate
```

Custom forecasting layer configuration:
```bash
    python train.py \
        --gate_hidden_units 32 \
        --gate_activation gelu \
        --backcast_length 336 \
        --forecast_length 48
```

Output Structure
----------------
The experiment saves to `results/nbeats_forecasting_layers_<timestamp>/`:
- `best_model.keras`: Best model checkpoint
- `results.json`: Training metrics and configuration
- `visualizations/`: Prediction plots per epoch
- `deep_analysis/`: Weight spectral analysis (if enabled)

References
----------
1. Oreshkin, B. N., et al. (2019). N-BEATS: Neural basis expansion analysis
   for interpretable time series forecasting. ICLR.

2. Manokhin, V. (2024). Scientific Framework for Time Series Forecasting.
   - Section 2: Forecastability Assessment
   - Section 8: Naive Benchmark Principle
"""

import os
import json
import math
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.models.nbeats import create_nbeats_model
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.datasets.time_series import TimeSeriesConfig, TimeSeriesGenerator

from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

from dl_techniques.layers.time_series.forecasting_layers import (
    NaiveResidual, ForecastabilityGate)

# ---------------------------------------------------------------------

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for major libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


# ---------------------------------------------------------------------


@dataclass
class NBeatsTrainingConfig:
    """Configuration dataclass for N-BEATS training with forecasting layers."""
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats_forecasting_layers"

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific configuration
    backcast_length: int = 168
    forecast_length: int = 24
    input_dim: int = 1

    # Model architecture
    stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 128
    use_normalization: bool = True  # Model internal normalization layers
    use_bias: bool = True
    activation: str = "gelu"

    # Forecasting layers configuration
    use_naive_residual: bool = True  # Enable NaiveResidual layer
    use_forecastability_gate: bool = True  # Enable ForecastabilityGate
    gate_hidden_units: int = 16  # Hidden units in forecastability gate
    gate_activation: str = "relu"  # Activation for forecastability gate

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 1000
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adam'
    primary_loss: Union[str, keras.losses.Loss] = "mase_loss"
    mase_seasonal_periods: int = 1

    # Learning rate schedule with warmup
    use_warmup: bool = True
    warmup_steps: int = 5000
    warmup_start_lr: float = 1e-6

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.1

    # Pattern selection & Data Processing
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 100
    normalize_per_instance: bool = True  # Enforce [-1, 1] scaling

    # Category weights for balanced sampling
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 1.0, "seasonal": 1.0, "composite": 1.2,
        "financial": 1.5, "weather": 1.3, "biomedical": 1.2,
        "industrial": 1.3, "intermittent": 1.0, "volatility": 1.1,
        "regime": 1.2, "structural": 1.1
    })

    # Visualization configuration
    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 12
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # DEEP ANALYSIS CONFIGURATION
    perform_deep_analysis: bool = True
    analysis_frequency: int = 10
    analysis_start_epoch: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


class MultiPatternDataProcessor:
    """Streams multi-pattern time series data using Python generators."""

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            generator: TimeSeriesGenerator,
            selected_patterns: List[str],
            pattern_to_category: Dict[str, str]
    ):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_category = pattern_to_category
        self.weighted_patterns, self.weights = self._prepare_weighted_sampling()

    def _prepare_weighted_sampling(self) -> Tuple[List[str], List[float]]:
        """Prepare lists for weighted random pattern selection."""
        patterns, weights = [], []
        for pattern_name in self.selected_patterns:
            category = self.pattern_to_category[pattern_name]
            weight = self.config.category_weights.get(category, 1.0)
            patterns.append(pattern_name)
            weights.append(weight)

        weights_sum = sum(weights)
        normalized_weights = [w / weights_sum for w in weights]
        return patterns, normalized_weights

    def _normalize_sequence(
            self,
            series: np.ndarray,
            epsilon: float = 1e-8
    ) -> Tuple[np.ndarray, float, float]:
        """
        Normalize sequence to [-1, 1] range using min-max scaling.

        Args:
            series: Input time series with shape (n_samples, features).
            epsilon: Small constant for numerical stability.

        Returns:
            Tuple of (normalized_series, series_min, series_max).
        """
        series_min = np.min(series)
        series_max = np.max(series)
        series_range = series_max - series_min

        if series_range < epsilon:
            return np.zeros_like(series), series_min, series_max

        normalized = 2.0 * (series - series_min) / series_range - 1.0
        return normalized, series_min, series_max

    def _sample_window(
            self,
            series: np.ndarray,
            backcast_length: int,
            forecast_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random window from the series for training.

        Args:
            series: Complete time series with shape (n_samples, features).
            backcast_length: Length of the input window.
            forecast_length: Length of the forecast window.

        Returns:
            Tuple of (x_window, y_window) for model input/output.
        """
        total_length = backcast_length + forecast_length
        if len(series) < total_length:
            raise ValueError(
                f"Series length {len(series)} is less than required {total_length}"
            )

        max_start_idx = len(series) - total_length
        start_idx = np.random.randint(0, max_start_idx + 1)

        x_window = series[start_idx:start_idx + backcast_length]
        y_window = series[start_idx + backcast_length:start_idx + total_length]

        return x_window, y_window

    def _generate_data(self, batch_size: int) -> Generator[Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]], None, None]:
        """
        Infinite generator for streaming time series data.

        Args:
            batch_size: Number of samples per batch.

        Yields:
            Tuples of (X_batch, y_structure) matching the model output.
        """
        while True:
            X_batch = np.zeros(
                (batch_size, self.config.backcast_length, self.config.input_dim),
                dtype=np.float32
            )
            y_batch = np.zeros(
                (batch_size, self.config.forecast_length, self.config.input_dim),
                dtype=np.float32
            )

            for i in range(batch_size):
                pattern_name = np.random.choice(
                    self.weighted_patterns,
                    p=self.weights
                )

                # Use the existing generator instance instead of re-initializing
                series = self.ts_generator.generate_task_data(pattern_name)

                if self.config.normalize_per_instance:
                    series, _, _ = self._normalize_sequence(series)

                x_window, y_window = self._sample_window(
                    series,
                    self.config.backcast_length,
                    self.config.forecast_length
                )

                X_batch[i] = x_window
                y_batch[i] = y_window

            if self.config.reconstruction_loss_weight > 0.0:
                # Provide zero target for reconstruction residual
                # Shape: (batch, backcast_length * input_dim) - flattened in N-BEATS output
                residual_batch = np.zeros(
                    (batch_size, self.config.backcast_length * self.config.input_dim),
                    dtype=np.float32
                )
                yield X_batch, (y_batch, residual_batch)
            else:
                yield X_batch, y_batch

    def prepare_datasets(self) -> Dict[str, Any]:
        """
        Prepare TensorFlow datasets for training, validation, and testing.

        Returns:
            Dictionary containing datasets and step counts.
        """
        val_steps = max(1, int(self.config.steps_per_epoch * self.config.val_ratio))
        test_steps = max(1, int(self.config.steps_per_epoch * self.config.test_ratio))

        train_gen = self._generate_data(self.config.batch_size)
        val_gen = self._generate_data(self.config.batch_size)
        test_gen = self._generate_data(self.config.batch_size)

        x_spec = tf.TensorSpec(
            shape=(self.config.batch_size, self.config.backcast_length, self.config.input_dim),
            dtype=tf.float32
        )
        y_spec = tf.TensorSpec(
            shape=(self.config.batch_size, self.config.forecast_length, self.config.input_dim),
            dtype=tf.float32
        )

        if self.config.reconstruction_loss_weight > 0.0:
            residual_spec = tf.TensorSpec(
                shape=(self.config.batch_size, self.config.backcast_length * self.config.input_dim),
                dtype=tf.float32
            )
            output_signature = (x_spec, (y_spec, residual_spec))
        else:
            output_signature = (x_spec, y_spec)

        train_ds = tf.data.Dataset.from_generator(
            lambda: train_gen,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_generator(
            lambda: test_gen,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        logger.info("Data pipeline created:")
        logger.info(f"  Training: {self.config.steps_per_epoch} steps/epoch")
        logger.info(f"  Validation: {val_steps} steps")
        logger.info(f"  Test: {test_steps} steps")

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': val_steps,
            'test_steps': test_steps
        }


class PatternPerformanceCallback(keras.callbacks.Callback):
    """Callback for monitoring and visualizing performance on a fixed test set."""

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            processor: MultiPatternDataProcessor,
            viz_dir: str,
            model_name: str
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.viz_dir = viz_dir
        self.model_name = model_name
        self.training_history = {
            'loss': [], 'val_loss': [], 'forecast_mae': [], 'val_forecast_mae': []
        }
        os.makedirs(self.viz_dir, exist_ok=True)
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a GUARANTEED DIVERSE visualization test set.
        Instead of using the sequential generator, we pick 1 sample from N different patterns.
        """
        logger.info("Creating a diverse, fixed visualization test set...")

        x_list, y_list = [], []

        # Get list of patterns and shuffle them to pick random ones
        available_patterns = self.processor.selected_patterns.copy()
        random.shuffle(available_patterns)

        # Ratio for test split
        start_ratio = self.config.train_ratio + self.config.val_ratio

        # Iterate through distinct patterns to ensure variety
        for pattern_name in available_patterns:
            if len(x_list) >= self.config.plot_top_k_patterns:
                break

            data = self.processor.ts_generator.generate_task_data(pattern_name)

            # Normalize entire sequence first (consistent with train_advanced philosophy)
            if self.config.normalize_per_instance:
                data, _, _ = self.processor._normalize_sequence(data)

            # Slice test split
            start_idx_split = int(start_ratio * len(data))
            test_data = data[start_idx_split:]

            total_len = self.config.backcast_length + self.config.forecast_length
            max_start = len(test_data) - total_len

            if max_start <= 0:
                continue

            # Pick a random window within this specific pattern's test data
            rand_idx = random.randint(0, max_start)

            x = test_data[rand_idx : rand_idx + self.config.backcast_length]
            y = test_data[rand_idx + self.config.backcast_length : rand_idx + total_len]

            # Ensure shapes
            x = x.reshape(self.config.backcast_length, self.config.input_dim)
            y = y.reshape(self.config.forecast_length, self.config.input_dim)

            x_list.append(x)
            y_list.append(y)

        if not x_list:
            logger.warning("Could not generate viz samples.")
            return np.array([]), np.array([])

        logger.info(f"Created {len(x_list)} diverse samples from different patterns.")
        return np.array(x_list), np.array(y_list)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Generate visualizations at specified intervals."""
        logs = logs or {}
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs != 0:
            return

        if self.config.create_learning_curves:
            self._plot_learning_curves(epoch)
        if self.config.create_prediction_plots and len(self.viz_test_data[0]) > 0:
            self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_history['loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Val Loss')
        plt.title(f'Loss History (Epoch {epoch+1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'learning_curves_epoch_{epoch+1:03d}.png'))
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        """Generate and save prediction plots for the fixed visualization set."""
        test_x, test_y = self.viz_test_data
        predictions_tuple = self.model(test_x, training=False)

        # Handle different output formats (forecast only or [forecast, residual])
        if isinstance(predictions_tuple, (list, tuple)):
            predictions = predictions_tuple[0]
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
        else:
            predictions = predictions_tuple.numpy()

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            # Create indices relative to the forecast point (0)
            backcast_steps = np.arange(-self.config.backcast_length, 0)
            forecast_steps = np.arange(0, self.config.forecast_length)

            ax.plot(backcast_steps, test_x[i].flatten(), label='Backcast', color='blue', alpha=0.6)
            ax.plot(forecast_steps, test_y[i].flatten(), label='True Future', color='green')
            ax.plot(forecast_steps, predictions[i].flatten(), label='Pred Future', color='red', linestyle='--')

            ax.set_title(f'Sample {i+1}')
            if i == 0: ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f'{self.model_name} Predictions - Epoch {epoch + 1}',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        save_path = os.path.join(
            self.viz_dir,
            f'{self.model_name}_predictions_epoch_{epoch + 1:03d}.png'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved prediction plot to {save_path}")


class NBeatsTrainer:
    """Main trainer class for N-BEATS with forecasting layers."""

    def __init__(self, config: NBeatsTrainingConfig, ts_config: TimeSeriesConfig):
        self.config = config
        self.ts_config = ts_config

        generator = TimeSeriesGenerator(ts_config)
        all_patterns = generator.get_task_names()
        pattern_to_category = {
            pattern: generator.task_definitions[pattern]["category"]
            for pattern in all_patterns
        }

        if config.target_categories:
            selected_patterns = [
                p for p in all_patterns
                if pattern_to_category[p] in config.target_categories
            ]
        else:
            selected_patterns = all_patterns

        if config.max_patterns:
            selected_patterns = selected_patterns[:config.max_patterns]

        self.processor = MultiPatternDataProcessor(
            config, generator, selected_patterns, pattern_to_category
        )

        logger.info(f"Initialized trainer with {len(selected_patterns)} patterns")
        logger.info(f"Categories: {set(pattern_to_category.values())}")

    def create_model(self) -> keras.Model:
        """
        Create N-BEATS model with forecasting layers integration.

        Returns:
            Compiled Keras model with forecasting layers.
        """
        # Create base N-BEATS model
        base_model = create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=self.config.forecast_length,
            stack_types=self.config.stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            activation=self.config.activation,
            use_normalization=self.config.use_normalization,
            dropout_rate=self.config.dropout_rate,
            reconstruction_weight=self.config.reconstruction_loss_weight,
            input_dim=self.config.input_dim,
            output_dim=self.config.input_dim,
            use_bias=self.config.use_bias,
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2)
        )

        # Build the model to get its structure
        base_model.build((None, self.config.backcast_length, self.config.input_dim))

        # Create functional model with forecasting layers
        inputs = keras.Input(
            shape=(self.config.backcast_length, self.config.input_dim),
            name='input'
        )

        # Get N-BEATS outputs
        nbeats_outputs = base_model(inputs)

        # Handle different output formats (forecast only or [forecast, residual])
        if isinstance(nbeats_outputs, (list, tuple)):
            deep_forecast = nbeats_outputs[0]
            residual = nbeats_outputs[1]
        else:
            deep_forecast = nbeats_outputs
            residual = None

        # Apply forecasting layers
        if self.config.use_naive_residual:
            logger.info("Adding NaiveResidual layer for Naive Benchmark Principle")
            naive_layer = NaiveResidual(
                forecast_length=self.config.forecast_length,
                name='naive_residual'
            )
            # Get pure naive forecast
            pure_naive = naive_layer(inputs, keras.ops.zeros_like(deep_forecast))

            if self.config.use_forecastability_gate:
                logger.info("Adding ForecastabilityGate for complexity assessment")
                gate = ForecastabilityGate(
                    hidden_units=self.config.gate_hidden_units,
                    activation=self.config.gate_activation,
                    name='forecastability_gate'
                )
                final_forecast = gate(inputs, deep_forecast, pure_naive)
            else:
                # Just use NaiveResidual without gating
                final_forecast = naive_layer(inputs, deep_forecast)
        else:
            # No forecasting layers, use raw N-BEATS output
            final_forecast = deep_forecast

        # Create final model
        if residual is not None and self.config.reconstruction_loss_weight > 0.0:
            model = keras.Model(
                inputs=inputs,
                outputs=[final_forecast, residual],
                name="nbeats_with_forecasting_layers"
            )
        else:
            model = keras.Model(
                inputs=inputs,
                outputs=final_forecast,
                name="nbeats_with_forecasting_layers"
            )

        # Create optimizer
        if self.config.use_warmup:
            primary_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=self.config.steps_per_epoch * self.config.epochs,
                alpha=0.1
            )
            schedule = WarmupSchedule(
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
                primary_schedule=primary_schedule
            )
            logger.info(
                f"Using warmup schedule: {self.config.warmup_steps} steps, "
                f"{self.config.warmup_start_lr} → {self.config.learning_rate}"
            )
        else:
            schedule = self.config.learning_rate

        if self.config.optimizer.lower() == 'adam':
            optimizer = keras.optimizers.Adam(
                learning_rate=schedule,
                clipnorm=self.config.gradient_clip_norm
            )
        elif self.config.optimizer.lower() == 'adamw':
            optimizer = keras.optimizers.AdamW(
                learning_rate=schedule,
                clipnorm=self.config.gradient_clip_norm
            )
        else:
            optimizer = keras.optimizers.get(self.config.optimizer)

        # Compile model
        if residual is not None and self.config.reconstruction_loss_weight > 0.0:
            if self.config.primary_loss == 'mase_loss':
                forecast_loss = MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
            else:
                forecast_loss = keras.losses.get(self.config.primary_loss)

            losses = [forecast_loss, 'mse']
            loss_weights = [1.0, self.config.reconstruction_loss_weight]
            
            # USE model.output_names TO MATCH OUTPUTS CORRECTLY
            metrics = {
                model.output_names[0]: [
                    keras.metrics.MeanAbsoluteError(name="forecast_mae")
                ]
            }
        else:
            # Model returns only forecast
            raw_output = model.output
            if isinstance(raw_output, (list, tuple)):
                forecast = raw_output[0]
            else:
                forecast = raw_output

            model = keras.Model(inputs=inputs, outputs=forecast, name="nbeats_forecast_only")

            if self.config.primary_loss == 'mase_loss':
                losses = MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
            else:
                losses = keras.losses.get(self.config.primary_loss)

            loss_weights = None
            metrics = [keras.metrics.MeanAbsoluteError(name="forecast_mae")]

        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete training experiment."""
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting N-BEATS Experiment with Forecasting Layers: {exp_dir}")

        data_pipeline = self.processor.prepare_datasets()
        results = self._train_model(data_pipeline, exp_dir)

        self._save_results(results, exp_dir)
        return {"results_dir": exp_dir, "results": results}

    def _train_model(self, data_pipeline: Dict, exp_dir: str) -> Dict[str, Any]:
        """Train the model and return results."""
        model = self.create_model()
        model.build((None, self.config.backcast_length, self.config.input_dim))
        model.summary(print_fn=logger.info)

        viz_dir = os.path.join(exp_dir, 'visualizations')
        performance_cb = PatternPerformanceCallback(
            self.config, self.processor, viz_dir, "nbeats_forecasting"
        )

        model_path = os.path.join(exp_dir, 'best_model.keras')

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1
            ),
            performance_cb,
            keras.callbacks.TerminateOnNaN()
        ]

        # --- DEEP MODEL ANALYSIS CALLBACK ---
        if self.config.perform_deep_analysis:
            logger.info("Adding Deep Model Analysis (Weights/Spectral) callback.")

            # Configure analysis (Data-free for speed during training loops)
            analysis_config = AnalysisConfig(
                analyze_weights=True,
                analyze_spectral=True,  # Spectral analysis (WeightWatcher)
                analyze_calibration=False,  # Skip data-dependent analyses
                analyze_information_flow=False,
                analyze_training_dynamics=False,
                verbose=False
            )

            analysis_dir = os.path.join(exp_dir, 'deep_analysis')

            analyzer_cb = EpochAnalyzerCallback(
                output_dir=analysis_dir,
                analysis_config=analysis_config,
                start_epoch=self.config.analysis_start_epoch,
                epoch_frequency=self.config.analysis_frequency,
                model_name="N-BEATS-Forecasting"
            )
            callbacks.append(analyzer_cb)

        history = model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Evaluating on test set...")
        test_results = model.evaluate(
            data_pipeline['test_ds'],
            steps=data_pipeline['test_steps'],
            verbose=1,
            return_dict=True
        )

        return {
            'history': history.history,
            'test_metrics': {k: float(v) for k, v in test_results.items()},
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        """Save experiment results to JSON."""
        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__
        }

        def default(o):
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            return str(o)

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=default)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for N-BEATS training."""
    parser = argparse.ArgumentParser(
        description="N-BEATS Training with Scientific Forecasting Layers"
    )

    parser.add_argument("--experiment_name", type=str, default="nbeats_forecasting_layers",
                        help="Name for logging.")
    parser.add_argument("--backcast_length", type=int, default=168)
    parser.add_argument("--forecast_length", type=int, default=24)

    # Training Loop
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # Data Processing
    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false",
                        help="Disable [-1, 1] normalization.")
    parser.set_defaults(normalize_per_instance=True)

    # N-BEATS Structure
    parser.add_argument("--stack_types", nargs='+', default=["trend", "seasonality", "generic"])
    parser.add_argument("--hidden_layer_units", type=int, default=256)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.5,
                        help="Weight for backcast reconstruction loss (default: 0.5)")

    # Forecasting Layers
    parser.add_argument("--no-naive-residual", dest="use_naive_residual", action="store_false",
                        help="Disable NaiveResidual layer.")
    parser.set_defaults(use_naive_residual=True)
    parser.add_argument("--no-forecastability-gate", dest="use_forecastability_gate",
                        action="store_false",
                        help="Disable ForecastabilityGate layer.")
    parser.set_defaults(use_forecastability_gate=True)
    parser.add_argument("--gate_hidden_units", type=int, default=16,
                        help="Hidden units in forecastability gate.")
    parser.add_argument("--gate_activation", type=str, default="relu",
                        help="Activation for forecastability gate.")

    # Deep Analysis (ModelAnalyzer)
    parser.add_argument("--no-deep-analysis", dest="perform_deep_analysis", action="store_false",
                        help="Disable periodic deep model analysis (WeightWatcher, etc).")
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10,
                        help="Frequency (in epochs) to run deep model analysis.")
    parser.add_argument("--analysis_start_epoch", type=int, default=1,
                        help="Epoch to start running deep model analysis.")

    return parser.parse_args()


def main() -> None:
    """Main function to configure and run the N-BEATS training experiment."""
    args = parse_args()

    config = NBeatsTrainingConfig(
        experiment_name=args.experiment_name,
        backcast_length=args.backcast_length,
        forecast_length=args.forecast_length,
        stack_types=args.stack_types,
        hidden_layer_units=args.hidden_layer_units,
        # Forecasting layers config
        use_naive_residual=args.use_naive_residual,
        use_forecastability_gate=args.use_forecastability_gate,
        gate_hidden_units=args.gate_hidden_units,
        gate_activation=args.gate_activation,
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        gradient_clip_norm=args.gradient_clip_norm,
        # Data
        normalize_per_instance=args.normalize_per_instance,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        # Analysis config
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch
    )

    ts_config = TimeSeriesConfig(
        n_samples=5000,
        random_seed=42
    )

    try:
        trainer = NBeatsTrainer(config, ts_config)
        results = trainer.run_experiment()
        logger.info(
            f"Experiment completed! Results saved to: {results['results_dir']}"
        )
        logger.info(
            f"Forecasting layers enabled: "
            f"NaiveResidual={config.use_naive_residual}, "
            f"ForecastabilityGate={config.use_forecastability_gate}"
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
