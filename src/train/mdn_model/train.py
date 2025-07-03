"""
Multi-Task Time Series Forecasting with Large-Scale MDN

This implementation demonstrates a comprehensive multi-task learning approach using a single
large MDN model trained on diverse time series patterns. The system includes:

- 20+ different time series generation patterns
- Enhanced model architecture with attention mechanisms
- Improved uncertainty calibration for tighter confidence intervals
- Long sequence modeling capabilities
- Extensive parameter space for complex pattern learning
- Category-specific scaling for better financial time series handling
"""

import os
import json
import keras
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from keras.api import regularizers
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.mdn_model import MDNModel

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
    n_samples: int = 10000           # 10x more samples
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Windowing config - Much longer sequences
    window_size: int = 120          # 4x longer input windows
    pred_horizon: int = 1
    stride: int = 1

    # Model config - Much larger model
    num_mixtures: int = 12          # More mixture components for better calibration
    hidden_units: List[int] = field(default_factory=lambda: [256, 128, 64])  # Larger network
    dropout_rate: float = 0.3       # Higher dropout for regularization
    use_batch_norm: bool = True
    l2_regularization: float = 1e-6
    task_embedding_dim: int = 32    # Richer task embeddings

    # Attention mechanism
    use_attention: bool = True
    attention_heads: int = 8
    attention_dim: int = 128
    fallback_without_attention: bool = True  # Fallback to simple model if attention fails

    # Calibration enhancement
    use_temperature_scaling: bool = True
    initial_temperature: float = 1.0
    calibration_weight: float = 0.1

    # Training config - More intensive training
    epochs: int = 100              # Much longer training
    batch_size: int = 256          # Larger batches
    early_stopping_patience: int = 30
    learning_rate: float = 5e-4    # Slightly lower for stability
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4     # Weight decay for better generalization

    # Multi-task training config
    task_balance_sampling: bool = True
    task_weight_decay: float = 0.98
    use_curriculum_learning: bool = True  # Start with easier tasks

    # Prediction config
    confidence_level: float = 0.9
    num_forecast_samples: int = 100

    # Visualization config
    max_plot_points: int = 300

    # Time series generation config
    base_length: int = 5000
    noise_std_range: Tuple[float, float] = (0.01, 0.2)
    trend_strength_range: Tuple[float, float] = (0.1, 2.0)
    seasonal_strength_range: Tuple[float, float] = (0.5, 3.0)
    frequency_range: Tuple[float, float] = (0.01, 0.5)

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
# Multi-Task Data Generation
# ---------------------------------------------------------------------

class TimeSeriesGenerator:
    """Generator for diverse time series patterns."""

    def __init__(self, config: MultiTaskMDNConfig):
        self.config = config
        self.task_definitions = self._define_tasks()

    def _define_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Define comprehensive set of time series tasks."""
        tasks = {}

        # === HARMONIC PATTERNS ===
        # Simple sine waves with different frequencies
        for i, freq in enumerate([0.02, 0.05, 0.1, 0.15, 0.2]):
            tasks[f"sine_freq_{freq:.2f}"] = {
                "category": "harmonic",
                "generator": self._generate_sine_wave,
                "params": {"frequency": freq, "amplitude": 1.0, "phase": 0}
            }

        # Multiple harmonic components
        tasks["harmonic_mix_2"] = {
            "category": "harmonic",
            "generator": self._generate_harmonic_mix,
            "params": {"frequencies": [0.05, 0.15], "amplitudes": [1.0, 0.5]}
        }

        tasks["harmonic_mix_3"] = {
            "category": "harmonic",
            "generator": self._generate_harmonic_mix,
            "params": {"frequencies": [0.03, 0.1, 0.25], "amplitudes": [1.0, 0.7, 0.3]}
        }

        # === TREND PATTERNS ===
        tasks["linear_trend_up"] = {
            "category": "trend",
            "generator": self._generate_trend,
            "params": {"trend_type": "linear", "slope": 0.001}
        }

        tasks["linear_trend_down"] = {
            "category": "trend",
            "generator": self._generate_trend,
            "params": {"trend_type": "linear", "slope": -0.0005}
        }

        tasks["exponential_trend"] = {
            "category": "trend",
            "generator": self._generate_trend,
            "params": {"trend_type": "exponential", "growth_rate": 0.0002}
        }

        tasks["polynomial_trend"] = {
            "category": "trend",
            "generator": self._generate_trend,
            "params": {"trend_type": "polynomial", "coefficients": [0, 0.001, -1e-7]}
        }

        # === SEASONAL PATTERNS ===
        tasks["daily_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal,
            "params": {"period": 24, "amplitude": 1.0}
        }

        tasks["weekly_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal,
            "params": {"period": 168, "amplitude": 0.8}
        }

        tasks["monthly_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal,
            "params": {"period": 720, "amplitude": 1.2}
        }

        # === STOCHASTIC PROCESSES ===
        tasks["random_walk"] = {
            "category": "stochastic",
            "generator": self._generate_random_walk,
            "params": {"drift": 0.001, "volatility": 0.02}
        }

        tasks["ar_process_1"] = {
            "category": "stochastic",
            "generator": self._generate_ar_process,
            "params": {"ar_coeffs": [0.7], "noise_std": 0.1}
        }

        tasks["ar_process_2"] = {
            "category": "stochastic",
            "generator": self._generate_ar_process,
            "params": {"ar_coeffs": [0.6, -0.2], "noise_std": 0.1}
        }

        tasks["ma_process"] = {
            "category": "stochastic",
            "generator": self._generate_ma_process,
            "params": {"ma_coeffs": [0.8, 0.3], "noise_std": 0.1}
        }

        # === COMPOSITE PATTERNS ===
        tasks["trend_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {"trend_slope": 0.0005, "seasonal_period": 50, "seasonal_amp": 0.8}
        }

        tasks["sine_with_trend"] = {
            "category": "composite",
            "generator": self._generate_sine_with_trend,
            "params": {"frequency": 0.1, "trend_slope": 0.0008, "amplitude": 1.0}
        }

        tasks["noisy_harmonic"] = {
            "category": "composite",
            "generator": self._generate_noisy_harmonic,
            "params": {"frequencies": [0.08, 0.2], "amplitudes": [1.0, 0.6], "noise_level": 0.2}
        }

        # === FINANCIAL PATTERNS ===
        tasks["gbm_conservative"] = {
            "category": "financial",
            "generator": self._generate_gbm,
            "params": {"drift": 0.05, "volatility": 0.15, "initial_price": 100}
        }

        tasks["gbm_volatile"] = {
            "category": "financial",
            "generator": self._generate_gbm,
            "params": {"drift": 0.08, "volatility": 0.35, "initial_price": 100}
        }

        tasks["mean_reverting"] = {
            "category": "financial",
            "generator": self._generate_mean_reverting,
            "params": {"theta": 0.1, "mu": 0, "sigma": 0.2}
        }

        # === DISCRETE PATTERNS ===
        tasks["step_function"] = {
            "category": "discrete",
            "generator": self._generate_step_function,
            "params": {"n_steps": 8, "step_size_range": (-2, 2)}
        }

        tasks["ramp_function"] = {
            "category": "discrete",
            "generator": self._generate_ramp_function,
            "params": {"n_ramps": 5, "ramp_duration": 200}
        }

        # === CHAOTIC PATTERNS ===
        tasks["lorenz_x"] = {
            "category": "chaotic",
            "generator": self._generate_lorenz_attractor,
            "params": {"component": "x", "sigma": 10, "rho": 28, "beta": 8/3}
        }

        tasks["henon_map"] = {
            "category": "chaotic",
            "generator": self._generate_henon_map,
            "params": {"a": 1.4, "b": 0.3}
        }

        return tasks

    def get_task_names(self) -> List[str]:
        """Get list of all task names."""
        return list(self.task_definitions.keys())

    def get_task_categories(self) -> List[str]:
        """Get list of unique task categories."""
        return list(set(task["category"] for task in self.task_definitions.values()))

    def generate_task_data(self, task_name: str) -> np.ndarray:
        """Generate data for a specific task."""
        if task_name not in self.task_definitions:
            raise ValueError(f"Unknown task: {task_name}")

        task_def = self.task_definitions[task_name]
        generator = task_def["generator"]
        params = task_def.get("params", {})

        return generator(**params)

    # === GENERATOR FUNCTIONS ===

    def _generate_sine_wave(self, frequency: float, amplitude: float = 1.0, phase: float = 0) -> np.ndarray:
        """Generate a simple sine wave."""
        t = np.linspace(0, self.config.base_length / 100, self.config.n_samples)
        y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_harmonic_mix(self, frequencies: List[float], amplitudes: List[float]) -> np.ndarray:
        """Generate mixture of harmonic components."""
        t = np.linspace(0, self.config.base_length / 100, self.config.n_samples)
        y = np.zeros_like(t)

        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.uniform(0, 2*np.pi)
            y += amp * np.sin(2 * np.pi * freq * t + phase)

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_trend(self, trend_type: str, **kwargs) -> np.ndarray:
        """Generate various trend patterns."""
        t = np.arange(self.config.n_samples)

        if trend_type == "linear":
            slope = kwargs.get("slope", 0.001)
            y = slope * t
        elif trend_type == "exponential":
            growth_rate = kwargs.get("growth_rate", 0.0002)
            y = np.exp(growth_rate * t) - 1
        elif trend_type == "polynomial":
            coeffs = kwargs.get("coefficients", [0, 0.001, -1e-7])
            y = np.polyval(coeffs[::-1], t)
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_seasonal(self, period: int, amplitude: float = 1.0) -> np.ndarray:
        """Generate seasonal patterns."""
        t = np.arange(self.config.n_samples)
        y = amplitude * np.sin(2 * np.pi * t / period)

        # Add some harmonics for realism
        y += 0.3 * amplitude * np.sin(4 * np.pi * t / period + np.pi/3)
        y += 0.1 * amplitude * np.sin(6 * np.pi * t / period + np.pi/6)

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_random_walk(self, drift: float = 0, volatility: float = 0.02) -> np.ndarray:
        """Generate random walk with drift."""
        innovations = np.random.normal(drift, volatility, self.config.n_samples)
        y = np.cumsum(innovations)
        return y.reshape(-1, 1)

    def _generate_ar_process(self, ar_coeffs: List[float], noise_std: float = 0.1) -> np.ndarray:
        """Generate autoregressive process."""
        p = len(ar_coeffs)
        y = np.zeros(self.config.n_samples)

        # Initialize with random values
        y[:p] = np.random.normal(0, noise_std, p)

        for t in range(p, self.config.n_samples):
            y[t] = sum(ar_coeffs[i] * y[t-1-i] for i in range(p))
            y[t] += np.random.normal(0, noise_std)

        return y.reshape(-1, 1)

    def _generate_ma_process(self, ma_coeffs: List[float], noise_std: float = 0.1) -> np.ndarray:
        """Generate moving average process."""
        q = len(ma_coeffs)
        noise = np.random.normal(0, noise_std, self.config.n_samples + q)
        y = np.zeros(self.config.n_samples)

        for t in range(self.config.n_samples):
            y[t] = noise[t + q] + sum(ma_coeffs[i] * noise[t + q - 1 - i] for i in range(q))

        return y.reshape(-1, 1)

    def _generate_trend_seasonal(self, trend_slope: float, seasonal_period: int, seasonal_amp: float) -> np.ndarray:
        """Generate trend + seasonal pattern."""
        t = np.arange(self.config.n_samples)
        trend = trend_slope * t
        seasonal = seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)
        y = trend + seasonal

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_sine_with_trend(self, frequency: float, trend_slope: float, amplitude: float) -> np.ndarray:
        """Generate sine wave with linear trend."""
        t = np.linspace(0, self.config.base_length / 100, self.config.n_samples)
        trend = trend_slope * np.arange(self.config.n_samples)
        sine = amplitude * np.sin(2 * np.pi * frequency * t)
        y = trend + sine

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_noisy_harmonic(self, frequencies: List[float], amplitudes: List[float], noise_level: float) -> np.ndarray:
        """Generate noisy harmonic with heteroscedastic noise."""
        t = np.linspace(0, self.config.base_length / 100, self.config.n_samples)
        y = np.zeros_like(t)

        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.uniform(0, 2*np.pi)
            y += amp * np.sin(2 * np.pi * freq * t + phase)

        # Heteroscedastic noise (noise level varies with signal)
        noise_std = noise_level * (1 + 0.5 * np.abs(y))
        noise = np.array([np.random.normal(0, std) for std in noise_std])

        return (y + noise).reshape(-1, 1)

    def _generate_gbm(self, drift: float, volatility: float, initial_price: float) -> np.ndarray:
        """Generate GBM log-returns for better stationarity."""
        dt = 1/252
        n_steps = self.config.n_samples

        # Generate log-returns directly (more stationary than prices)
        log_returns = np.random.normal(
            (drift - 0.5 * volatility**2) * dt,
            volatility * np.sqrt(dt),
            n_steps
        )

        # Return log-returns directly (stationary time series)
        return log_returns.reshape(-1, 1)

    def _generate_mean_reverting(self, theta: float, mu: float, sigma: float) -> np.ndarray:
        """Generate mean-reverting process returns, not levels."""
        dt = 0.01
        n_steps = self.config.n_samples

        y = np.zeros(n_steps)
        y[0] = np.random.normal(0, sigma)  # Start at 0, not mu

        for t in range(1, n_steps):
            # This generates the increments/returns
            dy = -theta * y[t-1] * dt + sigma * np.random.normal(0, np.sqrt(dt))
            y[t] = y[t-1] + dy

        return y.reshape(-1, 1)

    def _generate_step_function(self, n_steps: int, step_size_range: Tuple[float, float]) -> np.ndarray:
        """Generate step function pattern."""
        y = np.zeros(self.config.n_samples)
        step_points = np.linspace(0, self.config.n_samples-1, n_steps+1, dtype=int)

        current_level = 0
        for i in range(n_steps):
            step_size = np.random.uniform(*step_size_range)
            current_level += step_size
            start, end = step_points[i], step_points[i+1]
            y[start:end] = current_level

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_ramp_function(self, n_ramps: int, ramp_duration: int) -> np.ndarray:
        """Generate ramp function pattern."""
        y = np.zeros(self.config.n_samples)
        ramp_starts = np.random.choice(self.config.n_samples - ramp_duration, n_ramps, replace=False)

        for start in ramp_starts:
            end = min(start + ramp_duration, self.config.n_samples)
            ramp_height = np.random.uniform(-2, 2)
            y[start:end] += np.linspace(0, ramp_height, end - start)

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_lorenz_attractor(self, component: str, sigma: float, rho: float, beta: float) -> np.ndarray:
        """Generate Lorenz attractor projection."""
        def lorenz_equations(state, t):
            x, y, z = state
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

        # Use simple Euler integration
        dt = 0.01
        n_steps = self.config.n_samples * 10  # Higher resolution then downsample

        # Initial conditions
        x, y, z = 1.0, 1.0, 1.0
        trajectory = []

        for _ in range(n_steps):
            dx, dy, dz = lorenz_equations([x, y, z], 0)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            trajectory.append([x, y, z])

        trajectory = np.array(trajectory)

        # Select component and downsample
        if component == "x":
            data = trajectory[::10, 0]
        elif component == "y":
            data = trajectory[::10, 1]
        else:
            data = trajectory[::10, 2]

        # Trim to required length
        data = data[:self.config.n_samples]

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(data))
        return (data + noise).reshape(-1, 1)

    def _generate_henon_map(self, a: float, b: float) -> np.ndarray:
        """Generate Henon map chaotic sequence."""
        x, y = 0.1, 0.1
        trajectory = []

        for _ in range(self.config.n_samples + 1000):  # Extra for stabilization
            x_new = 1 - a * x**2 + y
            y_new = b * x
            x, y = x_new, y_new
            trajectory.append(x)

        # Skip initial transient
        data = np.array(trajectory[1000:1000 + self.config.n_samples])

        noise = np.random.normal(0, np.random.uniform(*self.config.noise_std_range), len(data))
        return (data + noise).reshape(-1, 1)

# ---------------------------------------------------------------------
# Enhanced Multi-Task Data Processing
# ---------------------------------------------------------------------

class StandardScaler:
    """Z-score normalization (mean=0, std=1)."""

    def __init__(self):
        self.mean_, self.std_ = None, None

    def fit(self, data: np.ndarray):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.std_

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        return scaled_data * self.std_ + self.mean_

class RobustScaler:
    """Robust scaler using median and IQR."""

    def __init__(self):
        self.median_, self.scale_ = None, None

    def fit(self, data: np.ndarray):
        self.median_ = np.median(data, axis=0)
        q75, q25 = np.percentile(data, [75, 25], axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.median_) / self.scale_

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        return scaled_data * self.scale_ + self.median_

class TaskScaler:
    """Enhanced scaler with category-specific scaling strategies."""

    def __init__(self):
        self.scalers = {}
        self.task_categories = {}

    def fit(self, data: Dict[str, np.ndarray], task_categories: Dict[str, str]):
        """Fit scalers with category-specific strategies."""
        self.task_categories = task_categories

        for task_name, task_data in data.items():
            category = task_categories[task_name]

            if category == "financial":
                # Use StandardScaler for financial data
                scaler = StandardScaler()
                logger.info(f"Using StandardScaler for financial task: {task_name}")
            elif category in ["stochastic", "chaotic"]:
                # Use StandardScaler for stochastic/chaotic processes
                scaler = StandardScaler()
                logger.info(f"Using StandardScaler for {category} task: {task_name}")
            else:
                # Use RobustScaler for periodic/deterministic patterns
                scaler = RobustScaler()
                logger.info(f"Using RobustScaler for {category} task: {task_name}")

            scaler.fit(task_data)
            self.scalers[task_name] = scaler

    def transform(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data for a specific task."""
        return self.scalers[task_name].transform(data)

    def inverse_transform(self, task_name: str, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform data for a specific task."""
        return self.scalers[task_name].inverse_transform(scaled_data)

    def get_scale_factor(self, task_name: str) -> float:
        """Get the scale factor for variance scaling."""
        scaler = self.scalers[task_name]
        if isinstance(scaler, StandardScaler):
            return scaler.std_**2
        else:  # RobustScaler
            return scaler.scale_**2

# ---------------------------------------------------------------------
# Multi-Task MDN Model with Attention
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiTaskMDNModel(keras.Model):
    """Multi-task MDN model with attention and better calibration."""

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
        **kwargs
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
            name="task_embedding"
        )

        # Sequence processing layers
        self.sequence_layers = []
        self.attention_layer = None
        self.feature_layers = []
        self.mdn_model = None

        # Temperature scaling for calibration
        if use_temperature_scaling:
            self.temperature = self.add_weight(
                name="temperature",
                shape=(),
                initializer=keras.initializers.Constant(initial_temperature),
                trainable=True
            )

        self._build_input_shape = None

        logger.info(f"Initialized MultiTaskMDNModel with {num_tasks} tasks")

    def build(self, input_shape: Tuple[Optional[int], ...]):
        """Build the multi-task model."""
        # input_shape: [(batch, window_size, features), (batch,)]
        sequence_shape, task_shape = input_shape
        self._build_input_shape = input_shape

        window_size = sequence_shape[-2]
        input_features = sequence_shape[-1]

        # Build task embedding
        self.task_embedding.build(task_shape)

        # Sequence processing layers (1D convolutions for feature extraction)
        self.sequence_layers = [
            keras.layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d_1'),
            keras.layers.Conv1D(128, 3, activation='relu', padding='same', name='conv1d_2'),
            keras.layers.Conv1D(256, 3, activation='relu', padding='same', name='conv1d_3'),
        ]

        # Attention mechanism
        if self.use_attention:
            try:
                self.attention_layer = keras.layers.MultiHeadAttention(
                    num_heads=self.attention_heads,
                    key_dim=self.attention_dim // self.attention_heads,
                    name="sequence_attention"
                )
            except Exception as e:
                logger.warning(f"Failed to create attention layer: {e}. Disabling attention.")
                self.use_attention = False
                self.attention_layer = None

        # Build sequence layers
        current_shape = sequence_shape
        for i, layer in enumerate(self.sequence_layers):
            try:
                layer.build(current_shape)
                current_shape = layer.compute_output_shape(current_shape)
                logger.debug(f"Conv1D layer {i+1} output shape: {current_shape}")
            except Exception as e:
                logger.error(f"Failed to build Conv1D layer {i+1}: {e}")
                raise

        # Build attention layer
        if self.attention_layer:
            try:
                self.attention_layer.build(
                    query_shape=current_shape,
                    value_shape=current_shape,
                    key_shape=current_shape
                )
                logger.debug(f"Attention layer built with shape: {current_shape}")
            except Exception as e:
                logger.warning(f"Failed to build attention layer: {e}. Disabling attention.")
                self.use_attention = False
                self.attention_layer = None

        # Calculate final feature size
        sequence_features = current_shape[-1] * current_shape[-2]  # Flattened
        combined_features = sequence_features + self.task_embedding_dim

        # Create MDN model
        self.mdn_model = MDNModel(
            hidden_layers=self.hidden_layers_sizes,
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            kernel_regularizer=self.kernel_regularizer
        )

        # Build MDN model
        self.mdn_model.build((None, combined_features))

        super().build(input_shape)
        logger.info(f"MultiTaskMDNModel built with input shapes: {input_shape}")

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ):
        """Forward pass with attention and temperature scaling."""
        sequence_input, task_input = inputs

        # Process sequence through convolutions
        x = sequence_input
        for i, layer in enumerate(self.sequence_layers):
            try:
                x = layer(x, training=training)
            except Exception as e:
                logger.error(f"Error in Conv1D layer {i+1}: {e}")
                raise

        # Apply attention if enabled and available
        if self.use_attention and self.attention_layer is not None:
            try:
                # Self-attention on the sequence
                attended = self.attention_layer(x, x, x, training=training)
                # Add residual connection
                x = x + attended
            except Exception as e:
                logger.warning(f"Attention mechanism failed: {e}. Continuing without attention.")
                # Continue without attention
                pass

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
                # Split MDN output to apply temperature to mixture weights only
                mu_end = self.num_mix * self.output_dim
                sigma_end = mu_end + (self.num_mix * self.output_dim)

                out_mu = mdn_output[..., :mu_end]
                out_sigma = mdn_output[..., mu_end:sigma_end]
                out_pi = mdn_output[..., sigma_end:]

                # Apply temperature scaling to mixture weights
                out_pi_scaled = out_pi / keras.ops.maximum(self.temperature, 0.1)

                # Recombine
                mdn_output = keras.ops.concatenate([out_mu, out_sigma, out_pi_scaled], axis=-1)
            except Exception as e:
                logger.warning(f"Temperature scaling failed: {e}. Using original output.")
                # Continue with original output
                pass

        return mdn_output

    def get_mdn_layer(self):
        """Get the MDN layer for loss computation."""
        return self.mdn_model.mdn_layer

    def sample(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor], num_samples: int = 1, temp: float = 1.0):
        """Generate samples from the predicted distribution."""
        predictions = self(inputs, training=False)
        return self.mdn_model.mdn_layer.sample(predictions, temperature=temp)

    def get_config(self):
        """Get model configuration."""
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
# Training with Calibration
# ---------------------------------------------------------------------

def create_windows_with_tasks(
    data: np.ndarray,
    task_id: int,
    config: MultiTaskMDNConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windowed data with task identifiers."""
    X, y, task_ids = [], [], []

    for i in range(len(data) - config.window_size - config.pred_horizon + 1):
        X.append(data[i : i + config.window_size])
        y.append(data[i + config.window_size + config.pred_horizon - 1])
        task_ids.append(task_id)

    return np.array(X), np.array(y), np.array(task_ids)

def combine_task_data(
    task_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: MultiTaskMDNConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Combine data from multiple tasks."""
    all_X, all_y, all_task_ids = [], [], []
    task_indices = {}

    start_idx = 0
    for task_name, (X, y, task_ids) in task_data.items():
        all_X.append(X)
        all_y.append(y)
        all_task_ids.append(task_ids)

        # Store indices for this task
        end_idx = start_idx + len(X)
        task_indices[task_name] = np.arange(start_idx, end_idx)
        start_idx = end_idx

    combined_X = np.concatenate(all_X, axis=0)
    combined_y = np.concatenate(all_y, axis=0)
    combined_task_ids = np.concatenate(all_task_ids, axis=0)

    return combined_X, combined_y, combined_task_ids, task_indices

def calibration_loss(y_true, y_pred, mdn_layer, temperature=None):
    """Calibration loss for better uncertainty quantification."""
    # Get standard MDN loss
    mdn_loss = mdn_layer.loss_func(y_true, y_pred)

    # Add calibration term (simplified version)
    if temperature is not None:
        # Encourage temperature to stay close to 1 for good calibration
        temp_penalty = keras.ops.square(temperature - 1.0) * 0.01
        return mdn_loss + temp_penalty

    return mdn_loss

# ---------------------------------------------------------------------
# Multi-Task Trainer
# ---------------------------------------------------------------------

class MultiTaskTrainer:
    """Trainer for large-scale multi-task MDN model."""

    def __init__(self, config: MultiTaskMDNConfig):
        self.config = config
        self.generator = TimeSeriesGenerator(config)
        self.task_names = self.generator.get_task_names()
        self.task_categories = self.generator.get_task_categories()
        self.task_scalers = TaskScaler()

        logger.info(f"Initialized trainer with {len(self.task_names)} tasks across {len(self.task_categories)} categories")

    def prepare_data(self) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Prepare multi-task training data with enhanced scaling."""
        logger.info("Preparing multi-task data...")

        # Generate raw data for all tasks
        raw_data = {}
        task_categories = {}

        for task_name in self.task_names:
            logger.info(f"Generating data for task: {task_name}")
            raw_data[task_name] = self.generator.generate_task_data(task_name)
            task_categories[task_name] = self.generator.task_definitions[task_name]["category"]

        # Fit scalers with category information
        self.task_scalers.fit(raw_data, task_categories)

        # Split and scale data for each task
        task_data = {}
        for task_id, task_name in enumerate(self.task_names):
            data = raw_data[task_name]

            # Split data
            train_size = int(self.config.train_ratio * len(data))
            val_size = int(self.config.val_ratio * len(data))

            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]

            # Scale data using category-specific scaling
            train_scaled = self.task_scalers.transform(task_name, train_data)
            val_scaled = self.task_scalers.transform(task_name, val_data)
            test_scaled = self.task_scalers.transform(task_name, test_data)

            # Create windows
            train_windows = create_windows_with_tasks(train_scaled, task_id, self.config)
            val_windows = create_windows_with_tasks(val_scaled, task_id, self.config)
            test_windows = create_windows_with_tasks(test_scaled, task_id, self.config)

            task_data[task_name] = {
                "train": train_windows,
                "val": val_windows,
                "test": test_windows
            }

            logger.info(f"Task {task_name} ({task_categories[task_name]}): "
                       f"train={train_windows[0].shape[0]}, "
                       f"val={val_windows[0].shape[0]}, test={test_windows[0].shape[0]}")

        return task_data

    def create_model(self) -> MultiTaskMDNModel:
        """Create the multi-task MDN model."""
        logger.info("Creating multi-task MDN model...")
        logger.info(f"Configuration: {len(self.config.hidden_units)} hidden layers, "
                   f"{self.config.num_mixtures} mixtures, "
                   f"attention: {self.config.use_attention}, "
                   f"temperature scaling: {self.config.use_temperature_scaling}")

        model = MultiTaskMDNModel(
            num_tasks=len(self.task_names),
            task_embedding_dim=self.config.task_embedding_dim,
            hidden_layers=self.config.hidden_units,
            output_dimension=1,  # Single output dimension for time series
            num_mixtures=self.config.num_mixtures,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm,
            use_attention=self.config.use_attention,
            attention_heads=self.config.attention_heads,
            attention_dim=self.config.attention_dim,
            use_temperature_scaling=self.config.use_temperature_scaling,
            initial_temperature=self.config.initial_temperature,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )

        return model

    def train_model(self, model: MultiTaskMDNModel, task_data: Dict) -> Dict[str, Any]:
        """Train the multi-task model with advanced techniques."""
        logger.info("Training multi-task MDN model...")

        # Combine training data
        train_data = {name: data["train"] for name, data in task_data.items()}
        val_data = {name: data["val"] for name, data in task_data.items()}

        X_train, y_train, task_ids_train, train_indices = combine_task_data(train_data, self.config)
        X_val, y_val, task_ids_val, val_indices = combine_task_data(val_data, self.config)

        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

        # Build model
        model.build([(None, self.config.window_size, 1), (None,)])

        # Create optimizer with weight decay
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Loss function with calibration
        def loss_fn(y_true, y_pred):
            base_loss = model.get_mdn_layer().loss_func(y_true, y_pred)
            if model.use_temperature_scaling:
                # Add temperature regularization
                temp_penalty = keras.ops.square(model.temperature - 1.0) * self.config.calibration_weight
                return base_loss + temp_penalty
            return base_loss

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn
        )

        # Print model summary
        model.summary(print_fn=lambda x: logger.info(x))

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                min_delta=1e-6
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=15,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TerminateOnNaN(),
        ]

        # Add model checkpoint if saving results
        if self.config.save_results:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    'best_model.keras',
                    save_best_only=True,
                    monitor='val_loss',
                    verbose=1
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
            verbose=1
        )

        return {
            "history": history.history,
            "train_indices": train_indices,
            "val_indices": val_indices
        }

    def evaluate_model(self, model: MultiTaskMDNModel, task_data: Dict) -> Dict[str, TaskMetrics]:
        """Enhanced evaluation with calibration metrics."""
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
            epistemic_var = keras.ops.sum(pi_expanded * (mu - point_expanded)**2, axis=1)
            total_var = aleatoric_var + epistemic_var

            # Convert to numpy and inverse transform
            point_estimates_np = keras.ops.convert_to_numpy(point_estimates)
            total_var_np = keras.ops.convert_to_numpy(total_var)
            aleatoric_var_np = keras.ops.convert_to_numpy(aleatoric_var)
            epistemic_var_np = keras.ops.convert_to_numpy(epistemic_var)

            # Inverse transform predictions using category-specific scaling
            point_estimates_orig = self.task_scalers.inverse_transform(task_name, point_estimates_np)
            y_test_orig = self.task_scalers.inverse_transform(task_name, y_test)

            # Scale variances using category-specific scaling
            scale_factor = self.task_scalers.get_scale_factor(task_name)
            total_var_orig = total_var_np * scale_factor
            aleatoric_var_orig = aleatoric_var_np * scale_factor
            epistemic_var_orig = epistemic_var_np * scale_factor

            # Calculate prediction intervals
            z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
            std_dev = np.sqrt(total_var_orig)
            lower_bound = point_estimates_orig - z_score * std_dev
            upper_bound = point_estimates_orig + z_score * std_dev

            # Calculate metrics
            mse = np.mean((y_test_orig - point_estimates_orig)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_orig - point_estimates_orig))
            coverage = np.mean((y_test_orig >= lower_bound) & (y_test_orig <= upper_bound))
            interval_width = np.mean(upper_bound - lower_bound)

            # Create binary coverage array for calibration error calculation
            is_covered = (y_test_orig >= lower_bound) & (y_test_orig <= upper_bound)

            # Calibration error (Expected Calibration Error approximation)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            # Calculate confidence scores (based on prediction interval width)
            max_interval_width = np.max(upper_bound - lower_bound)
            confidence_scores = 1 - (upper_bound - lower_bound) / (max_interval_width + 1e-8)
            confidence_scores = np.clip(confidence_scores, 0, 1)

            calibration_error = 0
            total_samples = len(confidence_scores)

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
                if in_bin.sum() > 0:
                    accuracy_in_bin = is_covered[in_bin].mean()
                    avg_confidence_in_bin = confidence_scores[in_bin].mean()
                    bin_weight = in_bin.sum() / total_samples
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * bin_weight

            # Sharpness (average prediction interval width)
            sharpness = interval_width

            # Get task category
            task_category = self.generator.task_definitions[task_name]["category"]

            task_metrics[task_name] = TaskMetrics(
                task_name=task_name,
                task_category=task_category,
                mse=mse,
                rmse=rmse,
                mae=mae,
                coverage=coverage,
                interval_width=interval_width,
                avg_aleatoric=np.mean(aleatoric_var_orig),
                avg_epistemic=np.mean(epistemic_var_orig),
                calibration_error=calibration_error,
                sharpness=sharpness,
                samples_count=len(y_test_orig)
            )

            logger.info(f"Task {task_name} ({task_category}) - RMSE: {rmse:.4f}, "
                       f"Coverage: {coverage:.4f}, Calibration Error: {calibration_error:.4f}")

        return task_metrics

    def create_visualizations(self, model: MultiTaskMDNModel, task_data: Dict, save_dir: str):
        """Create visualizations with category grouping."""
        logger.info("Creating visualizations...")

        os.makedirs(save_dir, exist_ok=True)

        # Create visualizations by category
        for category in self.task_categories:
            category_tasks = [name for name in self.task_names
                            if self.generator.task_definitions[name]["category"] == category]

            if len(category_tasks) > 4:
                category_tasks = category_tasks[:4]  # Limit to 4 for visualization

            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            axes = axes.flatten()
            fig.suptitle(f'Multi-Task MDN - {category.title()} Category', fontsize=16)

            for i, task_name in enumerate(category_tasks):
                if i >= 4:
                    break

                task_id = self.task_names.index(task_name)
                X_test, y_test, task_ids_test = task_data[task_name]["test"]

                # Make predictions
                predictions = model.predict([X_test, task_ids_test])

                # Get point estimates and uncertainties
                mdn_layer = model.get_mdn_layer()
                mu, sigma, pi = mdn_layer.split_mixture_params(predictions)
                pi_probs = keras.activations.softmax(pi, axis=-1)
                pi_expanded = keras.ops.expand_dims(pi_probs, axis=-1)
                point_estimates = keras.ops.sum(mu * pi_expanded, axis=1)

                # Calculate total variance
                point_expanded = keras.ops.expand_dims(point_estimates, axis=1)
                total_var = keras.ops.sum(pi_expanded * (sigma**2 + (mu - point_expanded)**2), axis=1)

                # Convert to numpy and inverse transform
                point_estimates_np = keras.ops.convert_to_numpy(point_estimates)
                total_var_np = keras.ops.convert_to_numpy(total_var)

                point_estimates_orig = self.task_scalers.inverse_transform(task_name, point_estimates_np)
                y_test_orig = self.task_scalers.inverse_transform(task_name, y_test)

                # Scale variance
                scale_factor = self.task_scalers.get_scale_factor(task_name)
                total_var_orig = total_var_np * scale_factor

                # Calculate prediction intervals
                z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
                std_dev = np.sqrt(total_var_orig)
                lower_bound = point_estimates_orig - z_score * std_dev
                upper_bound = point_estimates_orig + z_score * std_dev

                # Create plot
                plot_len = min(len(y_test_orig), self.config.max_plot_points)
                x_indices = np.arange(plot_len)

                axes[i].plot(x_indices, y_test_orig[:plot_len], 'b-', label='True Values', linewidth=2, alpha=0.8)
                axes[i].plot(x_indices, point_estimates_orig[:plot_len], 'r-', label='Predictions', linewidth=2)
                axes[i].fill_between(
                    x_indices,
                    lower_bound[:plot_len].flatten(),
                    upper_bound[:plot_len].flatten(),
                    color='red', alpha=0.2, label=f'{int(self.config.confidence_level*100)}% PI'
                )

                axes[i].set_title(f'{task_name.replace("_", " ").title()}', fontsize=12)
                axes[i].set_xlabel('Time Steps')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

            # Hide unused subplots
            for i in range(len(category_tasks), 4):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{category}_predictions.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved visualization for {category} category")

# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main():
    """Main function to run the multi-task MDN experiment."""
    config = MultiTaskMDNConfig()

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(config.result_dir, f"multitask_mdn_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"Starting multi-task MDN experiment")
    logger.info(f"Results will be saved to: {exp_dir}")
    logger.info(f"Configuration: {len(config.hidden_units)} hidden layers, "
               f"{config.num_mixtures} mixtures, {config.window_size} window size")

    # Initialize trainer
    trainer = MultiTaskTrainer(config)
    logger.info(f"Training on {len(trainer.task_names)} tasks across {len(trainer.task_categories)} categories")

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

        # Create visualizations
        trainer.create_visualizations(model, task_data, exp_dir)

        # Print results
        print("\n" + "="*100)
        print("MULTI-TASK MDN RESULTS")
        print("="*100)

        # Create results dataframe with categories
        results_data = []
        for task_name, metrics in task_metrics.items():
            results_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Category': metrics.task_category.title(),
                'RMSE': metrics.rmse,
                'MAE': metrics.mae,
                'Coverage': metrics.coverage,
                'Interval Width': metrics.interval_width,
                'Calibration Error': metrics.calibration_error,
                'Aleatoric Unc.': metrics.avg_aleatoric,
                'Epistemic Unc.': metrics.avg_epistemic,
                'Samples': metrics.samples_count
            })

        results_df = pd.DataFrame(results_data)

        # Sort by category for better readability
        results_df = results_df.sort_values(['Category', 'Task'])

        print("\nPer-Task Performance:")
        print(results_df.to_string(index=False, float_format='%.4f'))

        # Category-wise summary
        print("\n" + "="*100)
        print("CATEGORY-WISE SUMMARY")
        print("="*100)

        category_summary = results_df.groupby('Category').agg({
            'RMSE': ['mean', 'std'],
            'Coverage': ['mean', 'std'],
            'Calibration Error': ['mean', 'std'],
            'Samples': 'sum'
        }).round(4)

        print(category_summary)

        # Overall aggregate metrics
        total_samples = sum(m.samples_count for m in task_metrics.values())
        weighted_rmse = sum(m.rmse * m.samples_count for m in task_metrics.values()) / total_samples
        weighted_coverage = sum(m.coverage * m.samples_count for m in task_metrics.values()) / total_samples
        weighted_calibration = sum(m.calibration_error * m.samples_count for m in task_metrics.values()) / total_samples
        avg_aleatoric = np.mean([m.avg_aleatoric for m in task_metrics.values()])
        avg_epistemic = np.mean([m.avg_epistemic for m in task_metrics.values()])

        print(f"\n" + "="*100)
        print("OVERALL AGGREGATE METRICS")
        print("="*100)
        print(f"Total Tasks: {len(task_metrics)}")
        print(f"Total Categories: {len(trainer.task_categories)}")
        print(f"Weighted RMSE: {weighted_rmse:.4f}")
        print(f"Weighted Coverage: {weighted_coverage:.4f}")
        print(f"Weighted Calibration Error: {weighted_calibration:.4f}")
        print(f"Average Aleatoric Uncertainty: {avg_aleatoric:.4f}")
        print(f"Average Epistemic Uncertainty: {avg_epistemic:.4f}")
        print(f"Total Parameters: {model.count_params():,}")

        if config.use_temperature_scaling:
            final_temperature = float(model.temperature.numpy())
            print(f"Final Temperature: {final_temperature:.4f}")

        # Save results
        if config.save_results:
            # Save detailed metrics
            results_df.to_csv(os.path.join(exp_dir, 'task_metrics.csv'), index=False)

            # Save category summary
            category_summary.to_csv(os.path.join(exp_dir, 'category_summary.csv'))

            # Save aggregate metrics
            with open(os.path.join(exp_dir, 'aggregate_metrics.txt'), 'w') as f:
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
            model.save(os.path.join(exp_dir, 'multitask_mdn_model.keras'))

            # Save training history
            history_df = pd.DataFrame(training_results['history'])
            history_df.to_csv(os.path.join(exp_dir, 'training_history.csv'), index=False)

            # Save task definitions
            with open(os.path.join(exp_dir, 'task_definitions.json'), 'w') as f:
                json.dump(trainer.generator.task_definitions, f, indent=2, default=str)

            logger.info(f"Results saved to {exp_dir}")

        # Create summary visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Task MDN Performance Summary', fontsize=16)

        # RMSE by category
        category_rmse = results_df.groupby('Category')['RMSE'].mean()
        axes[0, 0].bar(category_rmse.index, category_rmse.values)
        axes[0, 0].set_title('Average RMSE by Category')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Coverage by category
        category_coverage = results_df.groupby('Category')['Coverage'].mean()
        axes[0, 1].bar(category_coverage.index, category_coverage.values)
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Average Coverage by Category')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Calibration error by category
        category_cal = results_df.groupby('Category')['Calibration Error'].mean()
        axes[0, 2].bar(category_cal.index, category_cal.values)
        axes[0, 2].set_title('Average Calibration Error by Category')
        axes[0, 2].set_ylabel('Calibration Error')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Training history
        axes[1, 0].plot(training_results['history']['loss'], label='Training Loss')
        axes[1, 0].plot(training_results['history']['val_loss'], label='Validation Loss')
        axes[1, 0].set_title('Training History')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Uncertainty by category
        category_uncertainty = results_df.groupby('Category')[['Aleatoric Unc.', 'Epistemic Unc.']].mean()
        x = np.arange(len(category_uncertainty))
        width = 0.35
        axes[1, 1].bar(x - width/2, category_uncertainty['Aleatoric Unc.'], width, label='Aleatoric')
        axes[1, 1].bar(x + width/2, category_uncertainty['Epistemic Unc.'], width, label='Epistemic')
        axes[1, 1].set_title('Average Uncertainty by Category')
        axes[1, 1].set_ylabel('Uncertainty')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(category_uncertainty.index, rotation=45)
        axes[1, 1].legend()

        # Task count by category
        task_counts = results_df['Category'].value_counts()
        axes[1, 2].bar(task_counts.index, task_counts.values)
        axes[1, 2].set_title('Number of Tasks by Category')
        axes[1, 2].set_ylabel('Task Count')
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()