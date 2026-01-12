"""
Multi-Task Time Series Forecasting with Large-Scale MDN.

This implementation demonstrates a comprehensive multi-task learning approach using a single
large MDN model trained on diverse time series patterns. The system includes:

- 20+ different time series generation patterns via TimeSeriesGenerator
- Enhanced model architecture with attention mechanisms
- Improved uncertainty calibration for tighter confidence intervals
- Long sequence modeling capabilities
- Extensive parameter space for complex pattern learning
- Category-specific scaling for better financial time series handling
- Integration with dl_techniques visualization framework
"""

import os
import json
import keras
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from datetime import datetime
from keras.api import regularizers
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------
# Local imports - Visualization Framework
# ---------------------------------------------------------------------

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    TrainingCurvesVisualization,
    TimeSeriesEvaluationResults,
    ForecastVisualization,
    ModelComparison,
    ModelComparisonBarChart,
    PerformanceRadarChart,
)

# ---------------------------------------------------------------------
# Local imports - Time Series Dataset Module
# ---------------------------------------------------------------------

from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    TimeSeriesNormalizer,
    NormalizationMethod,
)

# ---------------------------------------------------------------------
# Local imports - Logger and Models
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.mdn_model.model import MDNModel

# ---------------------------------------------------------------------
# Set random seeds for reproducibility
# ---------------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class MultiTaskMDNConfig:
    """Configuration for large-scale multi-task MDN forecasting."""

    # General experiment config
    result_dir: str = "results"
    save_results: bool = True

    # Data config - Much larger scale
    n_samples: int = 10000
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Windowing config - Much longer sequences
    window_size: int = 120
    pred_horizon: int = 1
    stride: int = 1

    # Model config - Much larger model
    num_mixtures: int = 12
    hidden_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    l2_regularization: float = 1e-6
    task_embedding_dim: int = 32

    # Attention mechanism
    use_attention: bool = True
    attention_heads: int = 8
    attention_dim: int = 128
    fallback_without_attention: bool = True

    # Calibration enhancement
    use_temperature_scaling: bool = True
    initial_temperature: float = 1.0
    calibration_weight: float = 0.1

    # Training config - More intensive training
    epochs: int = 100
    batch_size: int = 256
    early_stopping_patience: int = 30
    learning_rate: float = 5e-4
    optimizer: str = "adamw"
    weight_decay: float = 1e-4

    # Multi-task training config
    task_balance_sampling: bool = True
    task_weight_decay: float = 0.98
    use_curriculum_learning: bool = True

    # Prediction config
    confidence_level: float = 0.9
    num_forecast_samples: int = 100

    # Visualization config
    max_plot_points: int = 300


@dataclass
class TaskMetrics:
    """Metrics for task evaluation."""

    task_name: str
    task_category: str
    mse: float
    rmse: float
    mae: float
    coverage: float
    interval_width: float
    avg_aleatoric: float
    avg_epistemic: float
    calibration_error: float
    sharpness: float
    samples_count: int


# ---------------------------------------------------------------------
# Task Category Mapping for TimeSeriesGenerator
# ---------------------------------------------------------------------


def get_task_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Define task mapping to TimeSeriesGenerator patterns.

    :return: Dictionary mapping task names to generator patterns and categories.
    """
    tasks = {
        # Harmonic patterns
        "sine_freq_0.02": {"category": "harmonic", "pattern": "sine_low_freq"},
        "sine_freq_0.05": {"category": "harmonic", "pattern": "sine_medium_freq"},
        "sine_freq_0.10": {"category": "harmonic", "pattern": "sine_high_freq"},
        "harmonic_mix_2": {"category": "harmonic", "pattern": "multi_harmonic_2"},
        "harmonic_mix_3": {"category": "harmonic", "pattern": "multi_harmonic_3"},
        # Trend patterns
        "linear_trend_up": {"category": "trend", "pattern": "linear_trend_strong"},
        "linear_trend_down": {"category": "trend", "pattern": "linear_trend_weak"},
        "exponential_trend": {"category": "trend", "pattern": "exponential_growth"},
        "polynomial_trend": {"category": "trend", "pattern": "quadratic_trend"},
        # Seasonal patterns
        "daily_seasonal": {"category": "seasonal", "pattern": "daily_seasonal"},
        "weekly_seasonal": {"category": "seasonal", "pattern": "weekly_seasonal"},
        "monthly_seasonal": {"category": "seasonal", "pattern": "annual_seasonal"},
        # Stochastic processes
        "random_walk": {"category": "stochastic", "pattern": "random_walk_drift"},
        "ar_process_1": {"category": "stochastic", "pattern": "ar1_process"},
        "ar_process_2": {"category": "stochastic", "pattern": "ar2_process"},
        "ma_process": {"category": "stochastic", "pattern": "ma2_process"},
        # Composite patterns
        "trend_seasonal": {"category": "composite", "pattern": "trend_plus_seasonal"},
        "sine_with_trend": {"category": "composite", "pattern": "damped_oscillation"},
        "noisy_harmonic": {"category": "composite", "pattern": "noisy_sine"},
        # Financial patterns
        "gbm_conservative": {"category": "financial", "pattern": "stock_returns_normal"},
        "gbm_volatile": {"category": "financial", "pattern": "stock_returns_volatile"},
        "mean_reverting": {"category": "financial", "pattern": "mean_reverting_ou"},
        # Discrete patterns
        "step_function": {"category": "discrete", "pattern": "regime_switching"},
        "ramp_function": {"category": "discrete", "pattern": "sawtooth_wave"},
        # Chaotic patterns
        "lorenz_x": {"category": "chaotic", "pattern": "lorenz_x"},
        "henon_map": {"category": "chaotic", "pattern": "henon_map"},
    }
    return tasks


# ---------------------------------------------------------------------
# Multi-Task MDN Model with Attention
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiTaskMDNModel(keras.Model):
    """
    Multi-task MDN model with attention and better calibration.

    :param num_tasks: Number of distinct tasks.
    :param task_embedding_dim: Dimension of task embedding vectors.
    :param hidden_layers: List of hidden layer sizes for MDN.
    :param output_dimension: Output dimension (typically 1 for time series).
    :param num_mixtures: Number of Gaussian mixture components.
    :param dropout_rate: Dropout rate for regularization.
    :param use_batch_norm: Whether to use batch normalization.
    :param use_attention: Whether to use multi-head attention.
    :param attention_heads: Number of attention heads.
    :param attention_dim: Dimension of attention mechanism.
    :param use_temperature_scaling: Whether to use temperature scaling.
    :param initial_temperature: Initial temperature value.
    :param kernel_regularizer: Regularizer for kernel weights.
    """

    def __init__(
        self,
        num_tasks: int,
        task_embedding_dim: int,
        hidden_layers: List[int],
        output_dimension: int,
        num_mixtures: int,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_attention: bool = True,
        attention_heads: int = 8,
        attention_dim: int = 128,
        use_temperature_scaling: bool = True,
        initial_temperature: float = 1.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.hidden_layers_sizes = hidden_layers
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.use_temperature_scaling = use_temperature_scaling
        self.kernel_regularizer = kernel_regularizer

        # Task embedding layer
        self.task_embedding = keras.layers.Embedding(
            input_dim=num_tasks,
            output_dim=task_embedding_dim,
            name="task_embedding",
        )

        # Sequence processing layers
        self.sequence_layers = []
        self.attention_layer = None
        self.mdn_model = None

        # Temperature scaling for calibration
        if use_temperature_scaling:
            self.temperature = self.add_weight(
                name="temperature",
                shape=(),
                initializer=keras.initializers.Constant(initial_temperature),
                trainable=True,
            )

        self._build_input_shape = None

        logger.info(f"Initialized MultiTaskMDNModel with {num_tasks} tasks")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the multi-task model.

        :param input_shape: Tuple of input shapes for sequence and task inputs.
        """
        sequence_shape, task_shape = input_shape
        self._build_input_shape = input_shape

        # Build task embedding
        self.task_embedding.build(task_shape)

        # Sequence processing layers (1D convolutions for feature extraction)
        self.sequence_layers = [
            keras.layers.Conv1D(64, 7, activation=None, padding="same", name="conv1d_1"),
            keras.layers.LayerNormalization(),
            keras.layers.Activation("gelu"),
            keras.layers.Conv1D(
                128, 5, activation=None, padding="same", name="conv1d_2"
            ),
            keras.layers.LayerNormalization(),
            keras.layers.Activation("gelu"),
            keras.layers.Conv1D(
                256, 3, activation=None, padding="same", name="conv1d_3"
            ),
            keras.layers.LayerNormalization(),
            keras.layers.Activation("gelu"),
        ]

        # Attention mechanism
        if self.use_attention:
            try:
                self.attention_layer = keras.layers.MultiHeadAttention(
                    num_heads=self.attention_heads,
                    key_dim=self.attention_dim // self.attention_heads,
                    name="sequence_attention",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create attention layer: {e}. Disabling attention."
                )
                self.use_attention = False
                self.attention_layer = None

        # Build sequence layers
        current_shape = sequence_shape
        for i, layer in enumerate(self.sequence_layers):
            try:
                layer.build(current_shape)
                current_shape = layer.compute_output_shape(current_shape)
                logger.debug(f"Conv1D layer {i + 1} output shape: {current_shape}")
            except Exception as e:
                logger.error(f"Failed to build Conv1D layer {i + 1}: {e}")
                raise

        # Build attention layer
        if self.attention_layer:
            try:
                self.attention_layer.build(
                    query_shape=current_shape,
                    value_shape=current_shape,
                    key_shape=current_shape,
                )
                logger.debug(f"Attention layer built with shape: {current_shape}")
            except Exception as e:
                logger.warning(
                    f"Failed to build attention layer: {e}. Disabling attention."
                )
                self.use_attention = False
                self.attention_layer = None

        # Calculate final feature size
        sequence_features = current_shape[-1] * current_shape[-2]
        combined_features = sequence_features + self.task_embedding_dim

        # Create MDN model
        self.mdn_model = MDNModel(
            hidden_layers=self.hidden_layers_sizes,
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            kernel_regularizer=self.kernel_regularizer,
        )

        # Build MDN model
        self.mdn_model.build((None, combined_features))

        super().build(input_shape)
        logger.info(f"MultiTaskMDNModel built with input shapes: {input_shape}")

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass with attention and temperature scaling.

        :param inputs: Tuple of (sequence_input, task_input).
        :param training: Whether in training mode.
        :return: MDN output tensor.
        """
        sequence_input, task_input = inputs

        # Process sequence through convolutions
        x = sequence_input
        for i, layer in enumerate(self.sequence_layers):
            try:
                x = layer(x, training=training)
            except Exception as e:
                logger.error(f"Error in Conv1D layer {i + 1}: {e}")
                raise

        # Apply attention if enabled and available
        if self.use_attention and self.attention_layer is not None:
            try:
                attended = self.attention_layer(x, x, x, training=training)
                x = x + attended
            except Exception as e:
                logger.warning(
                    f"Attention mechanism failed: {e}. Continuing without attention."
                )

        # Flatten sequence features
        batch_size = keras.ops.shape(x)[0]
        sequence_flat = keras.ops.reshape(x, (batch_size, -1))

        # Get task embeddings
        task_emb = self.task_embedding(task_input)

        # Combine sequence features with task embeddings
        combined_features = keras.ops.concatenate([sequence_flat, task_emb], axis=-1)

        # Pass through MDN model
        mdn_output = self.mdn_model(combined_features, training=training)

        # Apply temperature scaling if enabled
        if self.use_temperature_scaling:
            try:
                mu_end = self.num_mix * self.output_dim
                sigma_end = mu_end + (self.num_mix * self.output_dim)

                out_mu = mdn_output[..., :mu_end]
                out_sigma = mdn_output[..., mu_end:sigma_end]
                out_pi = mdn_output[..., sigma_end:]

                out_pi_scaled = out_pi / keras.ops.maximum(self.temperature, 0.1)
                mdn_output = keras.ops.concatenate(
                    [out_mu, out_sigma, out_pi_scaled], axis=-1
                )
            except Exception as e:
                logger.warning(
                    f"Temperature scaling failed: {e}. Using original output."
                )

        return mdn_output

    def get_mdn_layer(self):
        """
        Get the MDN layer for loss computation.

        :return: MDN layer instance.
        """
        return self.mdn_model.mdn_layer

    def sample(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        num_samples: int = 1,
        temp: float = 1.0,
    ) -> keras.KerasTensor:
        """
        Generate samples from the predicted distribution.

        :param inputs: Tuple of (sequence_input, task_input).
        :param num_samples: Number of samples to generate.
        :param temp: Sampling temperature.
        :return: Sampled values.
        """
        predictions = self(inputs, training=False)
        return self.mdn_model.mdn_layer.sample(predictions, temperature=temp)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        :return: Configuration dictionary.
        """
        return {
            "num_tasks": self.num_tasks,
            "task_embedding_dim": self.task_embedding_dim,
            "hidden_layers": self.hidden_layers_sizes,
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "use_attention": self.use_attention,
            "attention_heads": self.attention_heads,
            "attention_dim": self.attention_dim,
            "use_temperature_scaling": self.use_temperature_scaling,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        }


# ---------------------------------------------------------------------
# Data Processing Utilities
# ---------------------------------------------------------------------


def create_windows_with_tasks(
    data: np.ndarray, task_id: int, config: MultiTaskMDNConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windowed data with task identifiers.

    :param data: Time series data array.
    :param task_id: Task identifier integer.
    :param config: Configuration object.
    :return: Tuple of (X windows, y targets, task_ids).
    """
    X, y, task_ids = [], [], []

    for i in range(len(data) - config.window_size - config.pred_horizon + 1):
        X.append(data[i : i + config.window_size])
        y.append(data[i + config.window_size + config.pred_horizon - 1])
        task_ids.append(task_id)

    return np.array(X), np.array(y), np.array(task_ids)


def combine_task_data(
    task_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: MultiTaskMDNConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Combine data from multiple tasks.

    :param task_data: Dictionary of task data tuples.
    :param config: Configuration object.
    :return: Combined arrays and task indices mapping.
    """
    all_X, all_y, all_task_ids = [], [], []
    task_indices = {}

    start_idx = 0
    for task_name, (X, y, task_ids) in task_data.items():
        all_X.append(X)
        all_y.append(y)
        all_task_ids.append(task_ids)

        end_idx = start_idx + len(X)
        task_indices[task_name] = np.arange(start_idx, end_idx)
        start_idx = end_idx

    combined_X = np.concatenate(all_X, axis=0)
    combined_y = np.concatenate(all_y, axis=0)
    combined_task_ids = np.concatenate(all_task_ids, axis=0)

    return combined_X, combined_y, combined_task_ids, task_indices


# ---------------------------------------------------------------------
# Multi-Task Trainer
# ---------------------------------------------------------------------


class MultiTaskTrainer:
    """
    Trainer for large-scale multi-task MDN model.

    :param config: Configuration object.
    """

    def __init__(self, config: MultiTaskMDNConfig):
        self.config = config

        # Initialize time series generator from dl_techniques
        gen_config = TimeSeriesGeneratorConfig(
            n_samples=config.n_samples,
            random_seed=42,
            default_noise_level=0.1,
        )
        self.generator = TimeSeriesGenerator(gen_config)

        # Get task definitions
        self.task_definitions = get_task_definitions()
        self.task_names = list(self.task_definitions.keys())
        self.task_categories = list(
            set(td["category"] for td in self.task_definitions.values())
        )

        # Task-specific normalizers using dl_techniques normalizer
        self.task_normalizers: Dict[str, TimeSeriesNormalizer] = {}

        # Initialize visualization manager
        self.viz_manager = VisualizationManager(
            experiment_name="multitask_mdn",
            output_dir=config.result_dir,
        )
        self._register_visualizations()

        logger.info(
            f"Initialized trainer with {len(self.task_names)} tasks "
            f"across {len(self.task_categories)} categories"
        )

    def _register_visualizations(self) -> None:
        """Register visualization templates with the manager."""
        self.viz_manager.register_template(
            "training_curves", TrainingCurvesVisualization
        )
        self.viz_manager.register_template(
            "forecast_visualization", ForecastVisualization
        )
        self.viz_manager.register_template(
            "model_comparison_bars", ModelComparisonBarChart
        )
        self.viz_manager.register_template("performance_radar", PerformanceRadarChart)

    def _get_normalizer_for_category(self, category: str) -> TimeSeriesNormalizer:
        """
        Get appropriate normalizer based on task category.

        :param category: Task category string.
        :return: Configured TimeSeriesNormalizer.
        """
        if category == "financial":
            return TimeSeriesNormalizer(method=NormalizationMethod.STANDARD)
        elif category in ["stochastic", "chaotic"]:
            return TimeSeriesNormalizer(method=NormalizationMethod.STANDARD)
        else:
            return TimeSeriesNormalizer(method=NormalizationMethod.ROBUST)

    def prepare_data(
        self,
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Prepare multi-task training data with enhanced scaling.

        :return: Dictionary of task data splits.
        """
        logger.info("Preparing multi-task data...")

        # Generate raw data for all tasks
        raw_data = {}

        for task_name, task_def in self.task_definitions.items():
            pattern_name = task_def["pattern"]
            category = task_def["category"]

            logger.info(
                f"Generating data for task: {task_name} (pattern: {pattern_name})"
            )

            try:
                # Try to generate using the pattern name
                data = self.generator.generate_task_data(pattern_name)
                raw_data[task_name] = data.reshape(-1, 1)
            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Pattern {pattern_name} not found, using custom generation: {e}"
                )
                # Fallback to custom pattern generation
                data = self._generate_fallback_pattern(task_name, category)
                raw_data[task_name] = data.reshape(-1, 1)

            # Initialize normalizer for this task
            self.task_normalizers[task_name] = self._get_normalizer_for_category(
                category
            )

        # Split and scale data for each task
        task_data = {}
        for task_id, task_name in enumerate(self.task_names):
            data = raw_data[task_name]
            category = self.task_definitions[task_name]["category"]

            # Split data
            train_size = int(self.config.train_ratio * len(data))
            val_size = int(self.config.val_ratio * len(data))

            train_data = data[:train_size]
            val_data = data[train_size : train_size + val_size]
            test_data = data[train_size + val_size :]

            # Fit normalizer on training data and transform all splits
            normalizer = self.task_normalizers[task_name]
            train_scaled = normalizer.fit_transform(train_data)
            val_scaled = normalizer.transform(val_data)
            test_scaled = normalizer.transform(test_data)

            # Create windows
            train_windows = create_windows_with_tasks(
                train_scaled, task_id, self.config
            )
            val_windows = create_windows_with_tasks(val_scaled, task_id, self.config)
            test_windows = create_windows_with_tasks(test_scaled, task_id, self.config)

            task_data[task_name] = {
                "train": train_windows,
                "val": val_windows,
                "test": test_windows,
            }

            logger.info(
                f"Task {task_name} ({category}): "
                f"train={train_windows[0].shape[0]}, "
                f"val={val_windows[0].shape[0]}, test={test_windows[0].shape[0]}"
            )

        return task_data

    def _generate_fallback_pattern(self, task_name: str, category: str) -> np.ndarray:
        """
        Generate fallback pattern when main generator doesn't have pattern.

        :param task_name: Name of the task.
        :param category: Category of the task.
        :return: Generated time series data.
        """
        n_samples = self.config.n_samples
        t = np.linspace(0, n_samples / 100, n_samples)
        noise = np.random.normal(0, 0.1, n_samples)

        if "sine" in task_name:
            freq = float(task_name.split("_")[-1]) if "freq" in task_name else 0.1
            return np.sin(2 * np.pi * freq * t) + noise
        elif "trend" in task_name:
            slope = 0.001 if "up" in task_name else -0.0005
            return slope * np.arange(n_samples) + noise
        elif "random_walk" in task_name:
            return np.cumsum(np.random.normal(0.001, 0.02, n_samples))
        elif "ar" in task_name:
            return self._generate_ar_process([0.7], n_samples) + noise
        elif "ma" in task_name:
            return self._generate_ma_process([0.8, 0.3], n_samples)
        elif category == "financial":
            return np.random.normal(0, 0.02, n_samples)
        elif category == "chaotic":
            return self._generate_lorenz_attractor(n_samples)
        else:
            return np.sin(2 * np.pi * 0.1 * t) + noise

    def _generate_ar_process(
        self, ar_coeffs: List[float], n_samples: int, noise_std: float = 0.1
    ) -> np.ndarray:
        """Generate AR process."""
        p = len(ar_coeffs)
        y = np.zeros(n_samples)
        y[:p] = np.random.normal(0, noise_std, p)
        for t in range(p, n_samples):
            y[t] = sum(ar_coeffs[i] * y[t - 1 - i] for i in range(p))
            y[t] += np.random.normal(0, noise_std)
        return y

    def _generate_ma_process(
        self, ma_coeffs: List[float], n_samples: int, noise_std: float = 0.1
    ) -> np.ndarray:
        """Generate MA process."""
        q = len(ma_coeffs)
        noise = np.random.normal(0, noise_std, n_samples + q)
        y = np.zeros(n_samples)
        for t in range(n_samples):
            y[t] = noise[t + q] + sum(
                ma_coeffs[i] * noise[t + q - 1 - i] for i in range(q)
            )
        return y

    def _generate_lorenz_attractor(self, n_samples: int) -> np.ndarray:
        """Generate Lorenz attractor x-component."""
        dt = 0.01
        sigma, rho, beta = 10.0, 28.0, 8 / 3
        x, y, z = 1.0, 1.0, 1.0
        trajectory = []
        for _ in range(n_samples * 10):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            trajectory.append(x)
        return np.array(trajectory[::10][:n_samples])

    def create_model(self) -> MultiTaskMDNModel:
        """
        Create the multi-task MDN model.

        :return: Configured MultiTaskMDNModel.
        """
        logger.info("Creating multi-task MDN model...")
        logger.info(
            f"Configuration: {len(self.config.hidden_units)} hidden layers, "
            f"{self.config.num_mixtures} mixtures, "
            f"attention: {self.config.use_attention}, "
            f"temperature scaling: {self.config.use_temperature_scaling}"
        )

        model = MultiTaskMDNModel(
            num_tasks=len(self.task_names),
            task_embedding_dim=self.config.task_embedding_dim,
            hidden_layers=self.config.hidden_units,
            output_dimension=1,
            num_mixtures=self.config.num_mixtures,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm,
            use_attention=self.config.use_attention,
            attention_heads=self.config.attention_heads,
            attention_dim=self.config.attention_dim,
            use_temperature_scaling=self.config.use_temperature_scaling,
            initial_temperature=self.config.initial_temperature,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization),
        )

        return model

    def train_model(
        self, model: MultiTaskMDNModel, task_data: Dict
    ) -> Dict[str, Any]:
        """
        Train the multi-task model with advanced techniques.

        :param model: The model to train.
        :param task_data: Dictionary of task data.
        :return: Training results dictionary.
        """
        logger.info("Training multi-task MDN model...")

        # Combine training data
        train_data = {name: data["train"] for name, data in task_data.items()}
        val_data = {name: data["val"] for name, data in task_data.items()}

        X_train, y_train, task_ids_train, train_indices = combine_task_data(
            train_data, self.config
        )
        X_val, y_val, task_ids_val, val_indices = combine_task_data(
            val_data, self.config
        )

        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

        # Build model
        model.build([(None, self.config.window_size, 1), (None,)])

        # Create optimizer with weight decay
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Loss function with calibration
        def loss_fn(y_true, y_pred):
            base_loss = model.get_mdn_layer().loss_func(y_true, y_pred)
            if model.use_temperature_scaling:
                temp_penalty = (
                    keras.ops.square(model.temperature - 1.0)
                    * self.config.calibration_weight
                )
                return base_loss + temp_penalty
            return base_loss

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_fn)

        # Print model summary
        model.summary(print_fn=lambda x: logger.info(x))

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                min_delta=1e-6,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.7,
                patience=15,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.TerminateOnNaN(),
        ]

        # Add model checkpoint if saving results
        if self.config.save_results:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    "best_model.keras",
                    save_best_only=True,
                    monitor="val_loss",
                    verbose=1,
                )
            )

        # Train model
        history = model.fit(
            [X_train, task_ids_train],
            y_train,
            validation_data=([X_val, task_ids_val], y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return {
            "history": history.history,
            "train_indices": train_indices,
            "val_indices": val_indices,
        }

    def evaluate_model(
        self, model: MultiTaskMDNModel, task_data: Dict
    ) -> Dict[str, TaskMetrics]:
        """
        Enhanced evaluation with calibration metrics.

        :param model: Trained model to evaluate.
        :param task_data: Dictionary of task data.
        :return: Dictionary of TaskMetrics per task.
        """
        logger.info("Evaluating multi-task model...")

        task_metrics = {}

        for task_id, (task_name, data) in enumerate(task_data.items()):
            logger.info(f"Evaluating task: {task_name}")

            X_test, y_test, task_ids_test = data["test"]

            # Make predictions
            predictions = model.predict([X_test, task_ids_test])

            # Get uncertainty estimates with calibration
            mdn_layer = model.get_mdn_layer()
            mu, sigma, pi = mdn_layer.split_mixture_params(predictions)

            # Convert to probabilities
            pi_probs = keras.activations.softmax(pi, axis=-1)

            # Calculate point estimates (weighted mean)
            pi_expanded = keras.ops.expand_dims(pi_probs, axis=-1)
            point_estimates = keras.ops.sum(mu * pi_expanded, axis=1)

            # Calculate variances
            point_expanded = keras.ops.expand_dims(point_estimates, axis=1)
            aleatoric_var = keras.ops.sum(pi_expanded * sigma**2, axis=1)
            epistemic_var = keras.ops.sum(
                pi_expanded * (mu - point_expanded) ** 2, axis=1
            )
            total_var = aleatoric_var + epistemic_var

            # Convert to numpy and inverse transform
            point_estimates_np = keras.ops.convert_to_numpy(point_estimates)
            total_var_np = keras.ops.convert_to_numpy(total_var)
            aleatoric_var_np = keras.ops.convert_to_numpy(aleatoric_var)
            epistemic_var_np = keras.ops.convert_to_numpy(epistemic_var)

            # Inverse transform predictions using normalizer
            normalizer = self.task_normalizers[task_name]
            point_estimates_orig = normalizer.inverse_transform(point_estimates_np)
            y_test_orig = normalizer.inverse_transform(y_test)

            # Scale variances
            if hasattr(normalizer, "std_") and normalizer.std_ is not None:
                scale_factor = normalizer.std_**2
            elif hasattr(normalizer, "scale_") and normalizer.scale_ is not None:
                scale_factor = normalizer.scale_**2
            else:
                scale_factor = 1.0

            total_var_orig = total_var_np * scale_factor
            aleatoric_var_orig = aleatoric_var_np * scale_factor
            epistemic_var_orig = epistemic_var_np * scale_factor

            # Calculate prediction intervals
            z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
            std_dev = np.sqrt(total_var_orig)
            lower_bound = point_estimates_orig - z_score * std_dev
            upper_bound = point_estimates_orig + z_score * std_dev

            # Calculate metrics
            mse = np.mean((y_test_orig - point_estimates_orig) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_orig - point_estimates_orig))
            coverage = np.mean(
                (y_test_orig >= lower_bound) & (y_test_orig <= upper_bound)
            )
            interval_width = np.mean(upper_bound - lower_bound)

            # Calibration error calculation
            is_covered = (y_test_orig >= lower_bound) & (y_test_orig <= upper_bound)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            max_interval_width = np.max(upper_bound - lower_bound)
            confidence_scores = 1 - (upper_bound - lower_bound) / (
                max_interval_width + 1e-8
            )
            confidence_scores = np.clip(confidence_scores, 0, 1)

            calibration_error = 0
            total_samples = len(confidence_scores)

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidence_scores > bin_lower) & (
                    confidence_scores <= bin_upper
                )
                if in_bin.sum() > 0:
                    accuracy_in_bin = is_covered[in_bin].mean()
                    avg_confidence_in_bin = confidence_scores[in_bin].mean()
                    bin_weight = in_bin.sum() / total_samples
                    calibration_error += (
                        np.abs(avg_confidence_in_bin - accuracy_in_bin) * bin_weight
                    )

            sharpness = interval_width
            task_category = self.task_definitions[task_name]["category"]

            task_metrics[task_name] = TaskMetrics(
                task_name=task_name,
                task_category=task_category,
                mse=float(mse),
                rmse=float(rmse),
                mae=float(mae),
                coverage=float(coverage),
                interval_width=float(interval_width),
                avg_aleatoric=float(np.mean(aleatoric_var_orig)),
                avg_epistemic=float(np.mean(epistemic_var_orig)),
                calibration_error=float(calibration_error),
                sharpness=float(sharpness),
                samples_count=len(y_test_orig),
            )

            logger.info(
                f"Task {task_name} ({task_category}) - RMSE: {rmse:.4f}, "
                f"Coverage: {coverage:.4f}, Calibration Error: {calibration_error:.4f}"
            )

        return task_metrics

    def create_visualizations(
        self,
        model: MultiTaskMDNModel,
        task_data: Dict,
        training_results: Dict[str, Any],
        task_metrics: Dict[str, TaskMetrics],
        save_dir: str,
    ) -> None:
        """
        Create visualizations using the visualization framework.

        :param model: Trained model.
        :param task_data: Dictionary of task data.
        :param training_results: Training results with history.
        :param task_metrics: Dictionary of task metrics.
        :param save_dir: Directory to save visualizations.
        """
        logger.info("Creating visualizations...")
        os.makedirs(save_dir, exist_ok=True)

        # 1. Training curves visualization
        history = training_results["history"]
        training_history = TrainingHistory(
            epochs=list(range(len(history["loss"]))),
            train_loss=history["loss"],
            val_loss=history["val_loss"],
            train_metrics={},
            val_metrics={},
        )

        self.viz_manager.visualize(
            data=training_history,
            plugin_name="training_curves",
            show=False,
            filename="training_curves",
        )
        logger.info("Saved training curves visualization")

        # 2. Forecast visualizations by category
        for category in self.task_categories:
            category_tasks = [
                name
                for name in self.task_names
                if self.task_definitions[name]["category"] == category
            ]

            if not category_tasks:
                continue

            # Pick first task in category for visualization
            task_name = category_tasks[0]
            task_id = self.task_names.index(task_name)
            X_test, y_test, task_ids_test = task_data[task_name]["test"]

            # Make predictions
            predictions = model.predict([X_test, task_ids_test])
            mdn_layer = model.get_mdn_layer()
            mu, sigma, pi = mdn_layer.split_mixture_params(predictions)
            pi_probs = keras.activations.softmax(pi, axis=-1)
            pi_expanded = keras.ops.expand_dims(pi_probs, axis=-1)
            point_estimates = keras.ops.sum(mu * pi_expanded, axis=1)

            # Convert to numpy
            point_estimates_np = keras.ops.convert_to_numpy(point_estimates)

            # Inverse transform
            normalizer = self.task_normalizers[task_name]
            point_estimates_orig = normalizer.inverse_transform(point_estimates_np)
            y_test_orig = normalizer.inverse_transform(y_test)

            # Limit samples for visualization
            num_samples = min(100, len(y_test_orig))

            # Create TimeSeriesEvaluationResults for forecast visualization
            forecast_data = TimeSeriesEvaluationResults(
                all_inputs=X_test[:num_samples],
                all_true_forecasts=y_test_orig[:num_samples].reshape(num_samples, 1),
                all_predicted_forecasts=point_estimates_orig[:num_samples].reshape(
                    num_samples, 1
                ),
                model_name=f"MDN_{category}",
            )

            self.viz_manager.visualize(
                data=forecast_data,
                plugin_name="forecast_visualization",
                show=False,
                filename=f"forecast_{category}",
            )
            logger.info(f"Saved forecast visualization for {category} category")

        # 3. Model comparison bar chart
        comparison_metrics = {}
        for category in self.task_categories:
            category_tasks = [
                name
                for name in self.task_names
                if self.task_definitions[name]["category"] == category
            ]
            if category_tasks:
                avg_rmse = np.mean(
                    [task_metrics[t].rmse for t in category_tasks if t in task_metrics]
                )
                avg_coverage = np.mean(
                    [
                        task_metrics[t].coverage
                        for t in category_tasks
                        if t in task_metrics
                    ]
                )
                avg_calibration = np.mean(
                    [
                        task_metrics[t].calibration_error
                        for t in category_tasks
                        if t in task_metrics
                    ]
                )
                comparison_metrics[category] = {
                    "rmse": float(avg_rmse),
                    "coverage": float(avg_coverage),
                    "calibration_error": float(avg_calibration),
                }

        if comparison_metrics:
            model_comparison = ModelComparison(
                model_names=list(comparison_metrics.keys()),
                metrics=comparison_metrics,
            )

            self.viz_manager.visualize(
                data=model_comparison,
                plugin_name="model_comparison_bars",
                show=False,
                filename="category_comparison_bars",
            )

            self.viz_manager.visualize(
                data=model_comparison,
                plugin_name="performance_radar",
                show=False,
                filename="category_comparison_radar",
            )
            logger.info("Saved model comparison visualizations")


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------


def main() -> None:
    """Main function to run the multi-task MDN experiment."""
    config = MultiTaskMDNConfig()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.result_dir, f"multitask_mdn_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    logger.info("Starting multi-task MDN experiment")
    logger.info(f"Results will be saved to: {exp_dir}")
    logger.info(
        f"Configuration: {len(config.hidden_units)} hidden layers, "
        f"{config.num_mixtures} mixtures, {config.window_size} window size"
    )

    # Initialize trainer
    trainer = MultiTaskTrainer(config)
    logger.info(
        f"Training on {len(trainer.task_names)} tasks "
        f"across {len(trainer.task_categories)} categories"
    )

    try:
        # Prepare data
        task_data = trainer.prepare_data()
        logger.info(f"Data prepared for {len(task_data)} tasks")

        # Create model
        model = trainer.create_model()
        logger.info("Multi-task MDN model created")

        # Train model
        training_results = trainer.train_model(model, task_data)
        logger.info("Model training completed")

        # Evaluate model on each task
        task_metrics = trainer.evaluate_model(model, task_data)

        # Create visualizations using the framework
        trainer.create_visualizations(
            model, task_data, training_results, task_metrics, exp_dir
        )

        # Log results
        logger.info("=" * 100)
        logger.info("MULTI-TASK MDN RESULTS")
        logger.info("=" * 100)

        # Create results dataframe with categories
        results_data = []
        for task_name, metrics in task_metrics.items():
            results_data.append(
                {
                    "Task": task_name.replace("_", " ").title(),
                    "Category": metrics.task_category.title(),
                    "RMSE": metrics.rmse,
                    "MAE": metrics.mae,
                    "Coverage": metrics.coverage,
                    "Interval Width": metrics.interval_width,
                    "Calibration Error": metrics.calibration_error,
                    "Aleatoric Unc.": metrics.avg_aleatoric,
                    "Epistemic Unc.": metrics.avg_epistemic,
                    "Samples": metrics.samples_count,
                }
            )

        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values(["Category", "Task"])

        logger.info("Per-Task Performance:")
        logger.info(f"\n{results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")

        # Category-wise summary
        logger.info("=" * 100)
        logger.info("CATEGORY-WISE SUMMARY")
        logger.info("=" * 100)

        category_summary = (
            results_df.groupby("Category")
            .agg(
                {
                    "RMSE": ["mean", "std"],
                    "Coverage": ["mean", "std"],
                    "Calibration Error": ["mean", "std"],
                    "Samples": "sum",
                }
            )
            .round(4)
        )

        logger.info(f"\n{category_summary}")

        # Overall aggregate metrics
        total_samples = sum(m.samples_count for m in task_metrics.values())
        weighted_rmse = (
            sum(m.rmse * m.samples_count for m in task_metrics.values()) / total_samples
        )
        weighted_coverage = (
            sum(m.coverage * m.samples_count for m in task_metrics.values())
            / total_samples
        )
        weighted_calibration = (
            sum(m.calibration_error * m.samples_count for m in task_metrics.values())
            / total_samples
        )
        avg_aleatoric = np.mean([m.avg_aleatoric for m in task_metrics.values()])
        avg_epistemic = np.mean([m.avg_epistemic for m in task_metrics.values()])

        logger.info("=" * 100)
        logger.info("OVERALL AGGREGATE METRICS")
        logger.info("=" * 100)
        logger.info(f"Total Tasks: {len(task_metrics)}")
        logger.info(f"Total Categories: {len(trainer.task_categories)}")
        logger.info(f"Weighted RMSE: {weighted_rmse:.4f}")
        logger.info(f"Weighted Coverage: {weighted_coverage:.4f}")
        logger.info(f"Weighted Calibration Error: {weighted_calibration:.4f}")
        logger.info(f"Average Aleatoric Uncertainty: {avg_aleatoric:.4f}")
        logger.info(f"Average Epistemic Uncertainty: {avg_epistemic:.4f}")
        logger.info(f"Total Parameters: {model.count_params():,}")

        if config.use_temperature_scaling:
            final_temperature = float(model.temperature.numpy())
            logger.info(f"Final Temperature: {final_temperature:.4f}")

        # Save results
        if config.save_results:
            # Save detailed metrics
            results_df.to_csv(os.path.join(exp_dir, "task_metrics.csv"), index=False)

            # Save category summary
            category_summary.to_csv(os.path.join(exp_dir, "category_summary.csv"))

            # Save aggregate metrics
            with open(os.path.join(exp_dir, "aggregate_metrics.txt"), "w") as f:
                f.write(f"Total Tasks: {len(task_metrics)}\n")
                f.write(f"Total Categories: {len(trainer.task_categories)}\n")
                f.write(f"Weighted RMSE: {weighted_rmse:.4f}\n")
                f.write(f"Weighted Coverage: {weighted_coverage:.4f}\n")
                f.write(f"Weighted Calibration Error: {weighted_calibration:.4f}\n")
                f.write(f"Average Aleatoric Uncertainty: {avg_aleatoric:.4f}\n")
                f.write(f"Average Epistemic Uncertainty: {avg_epistemic:.4f}\n")
                f.write(f"Total Parameters: {model.count_params()}\n")
                if config.use_temperature_scaling:
                    f.write(f"Final Temperature: {final_temperature:.4f}\n")

            # Save model
            model.save(os.path.join(exp_dir, "multitask_mdn_model.keras"))

            # Save training history
            history_df = pd.DataFrame(training_results["history"])
            history_df.to_csv(os.path.join(exp_dir, "training_history.csv"), index=False)

            # Save task definitions
            with open(os.path.join(exp_dir, "task_definitions.json"), "w") as f:
                json.dump(trainer.task_definitions, f, indent=2, default=str)

            logger.info(f"Results saved to {exp_dir}")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()