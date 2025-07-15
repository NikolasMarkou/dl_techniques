"""
Multi-Task Time Series Forecasting with N-BEATS

This implementation demonstrates a comprehensive multi-task learning approach using N-BEATS
models trained on diverse time series patterns. The system includes:

- 25+ different time series generation patterns
- Multiple N-BEATS architectures (Generic, Interpretable, Hybrid)
- Advanced forecasting evaluation metrics
- Multi-horizon forecasting capabilities
- Extensive parameter space for complex pattern learning
- Category-specific scaling for better time series handling
- Ensemble forecasting with uncertainty quantification
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.nbeats import NBeatsNet, create_nbeats_model
from dl_techniques.losses.smape_loss import SMAPELoss, MASELoss
from dl_techniques.utils.datasets.nbeats import SyntheticDataGenerator, TimeSeriesNormalizer

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
class NBeatsConfig:
    """Configuration for large-scale multi-task N-BEATS forecasting."""
    # General experiment config
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats_multitask"

    # Data config - Large scale
    n_samples: int = 8000  # 8k samples per task
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific config
    backcast_length: int = 96  # 4 days of hourly data
    forecast_length: int = 24  # 1 day ahead

    # Multiple forecast horizons for comprehensive evaluation
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24, 48])

    # Model architectures to test
    model_types: List[str] = field(default_factory=lambda: ["interpretable", "generic", "hybrid"])

    # Model config - Larger models
    stack_types: Dict[str, List[str]] = field(default_factory=lambda: {
        "interpretable": ["trend", "seasonality"],
        "generic": ["generic", "generic", "generic"],
        "hybrid": ["trend", "seasonality", "generic"]
    })

    nb_blocks_per_stack: int = 4  # More blocks for better capacity
    thetas_dim: Dict[str, List[int]] = field(default_factory=lambda: {
        "interpretable": [4, 8],
        "generic": [128, 128, 128],
        "hybrid": [4, 8, 128]
    })

    hidden_layer_units: int = 512  # Larger hidden layers
    share_weights_in_stack: bool = False

    # Training config
    epochs: int = 200  # Longer training
    batch_size: int = 128  # Larger batches
    early_stopping_patience: int = 25
    learning_rate: float = 1e-3
    optimizer: str = 'adam'

    # Loss function config
    loss_functions: List[str] = field(default_factory=lambda: ["smape", "mase", "mae"])
    primary_loss: str = "smape"

    # Ensemble config
    use_ensemble: bool = True
    ensemble_size: int = 5
    ensemble_method: str = "averaging"  # "averaging", "stacking"

    # Evaluation config
    confidence_levels: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])
    num_bootstrap_samples: int = 1000

    # Visualization config
    max_plot_points: int = 200
    plot_samples: int = 5  # Number of samples to plot per task

    # Time series generation config
    base_length: int = 8000
    noise_std_range: Tuple[float, float] = (0.01, 0.15)
    seasonal_strength_range: Tuple[float, float] = (0.5, 2.0)
    trend_strength_range: Tuple[float, float] = (0.1, 1.5)


@dataclass
class ForecastMetrics:
    """Comprehensive forecasting metrics."""
    task_name: str
    task_category: str
    model_type: str
    horizon: int

    # Point forecast metrics
    mse: float
    rmse: float
    mae: float
    mape: float
    smape: float
    mase: float

    # Directional accuracy
    directional_accuracy: float

    # Distributional metrics
    coverage_80: float
    coverage_90: float
    coverage_95: float
    interval_width_80: float
    interval_width_90: float
    interval_width_95: float

    # Forecast quality
    forecast_bias: float
    forecast_variance: float

    # Sample information
    samples_count: int


# ---------------------------------------------------------------------
# Enhanced Time Series Generator
# ---------------------------------------------------------------------

class NBeatsTimeSeriesGenerator:
    """Generator for diverse time series patterns optimized for N-BEATS."""

    def __init__(self, config: NBeatsConfig):
        self.config = config
        self.task_definitions = self._define_tasks()
        logger.info(f"Initialized N-BEATS generator with {len(self.task_definitions)} tasks")

    def _define_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Define comprehensive set of time series tasks for N-BEATS."""
        tasks = {}

        # === TREND PATTERNS ===
        tasks["linear_trend_strong"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {"trend_type": "linear", "strength": 0.002, "noise_level": 0.05}
        }

        tasks["linear_trend_weak"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {"trend_type": "linear", "strength": 0.0005, "noise_level": 0.1}
        }

        tasks["exponential_growth"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {"trend_type": "exponential", "strength": 0.0001, "noise_level": 0.08}
        }

        tasks["polynomial_trend"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {"trend_type": "polynomial", "coefficients": [0, 0.001, -2e-7], "noise_level": 0.06}
        }

        tasks["logistic_growth"] = {
            "category": "trend",
            "generator": self._generate_logistic_growth,
            "params": {"carrying_capacity": 10, "growth_rate": 0.01, "noise_level": 0.1}
        }

        # === SEASONAL PATTERNS ===
        tasks["daily_seasonality"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {"periods": [24], "amplitudes": [1.0], "noise_level": 0.08}
        }

        tasks["weekly_seasonality"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {"periods": [168], "amplitudes": [1.2], "noise_level": 0.06}
        }

        tasks["multi_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {"periods": [24, 168], "amplitudes": [1.0, 0.8], "noise_level": 0.1}
        }

        tasks["complex_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {"periods": [12, 24, 168], "amplitudes": [0.6, 1.0, 0.7], "noise_level": 0.12}
        }

        # === TREND + SEASONAL COMBINATIONS ===
        tasks["trend_daily_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {
                "trend_type": "linear", "trend_strength": 0.001,
                "periods": [24], "seasonal_amplitudes": [1.0], "noise_level": 0.08
            }
        }

        tasks["trend_weekly_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {
                "trend_type": "linear", "trend_strength": 0.0008,
                "periods": [168], "seasonal_amplitudes": [1.2], "noise_level": 0.1
            }
        }

        tasks["exp_trend_multi_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {
                "trend_type": "exponential", "trend_strength": 0.0001,
                "periods": [24, 168], "seasonal_amplitudes": [1.0, 0.6], "noise_level": 0.12
            }
        }

        # === STOCHASTIC PROCESSES ===
        tasks["random_walk"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {"process_type": "random_walk", "drift": 0.001, "volatility": 0.05}
        }

        tasks["ar_process"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {"process_type": "ar", "ar_coeffs": [0.7, -0.2], "noise_std": 0.1}
        }

        tasks["ma_process"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {"process_type": "ma", "ma_coeffs": [0.8, 0.3], "noise_std": 0.1}
        }

        tasks["arma_process"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {"process_type": "arma", "ar_coeffs": [0.6], "ma_coeffs": [0.4], "noise_std": 0.08}
        }

        tasks["mean_reverting"] = {
            "category": "stochastic",
            "generator": self._generate_mean_reverting,
            "params": {"theta": 0.05, "mu": 0, "sigma": 0.2}
        }

        # === INTERMITTENT PATTERNS ===
        tasks["intermittent_demand"] = {
            "category": "intermittent",
            "generator": self._generate_intermittent_series,
            "params": {"demand_prob": 0.3, "demand_mean": 2.0, "demand_std": 0.5}
        }

        tasks["lumpy_demand"] = {
            "category": "intermittent",
            "generator": self._generate_intermittent_series,
            "params": {"demand_prob": 0.1, "demand_mean": 5.0, "demand_std": 1.0}
        }

        # === VOLATILITY CLUSTERING ===
        tasks["garch_low_vol"] = {
            "category": "volatility",
            "generator": self._generate_garch_series,
            "params": {"alpha": 0.1, "beta": 0.8, "omega": 0.01}
        }

        tasks["garch_high_vol"] = {
            "category": "volatility",
            "generator": self._generate_garch_series,
            "params": {"alpha": 0.2, "beta": 0.7, "omega": 0.05}
        }

        # === REGIME SWITCHING ===
        tasks["regime_switching"] = {
            "category": "regime",
            "generator": self._generate_regime_switching,
            "params": {"regimes": 2, "switch_prob": 0.02, "regime_params": [(0.001, 0.05), (0.005, 0.15)]}
        }

        # === STRUCTURAL BREAKS ===
        tasks["level_shift"] = {
            "category": "structural",
            "generator": self._generate_structural_break,
            "params": {"break_type": "level", "break_magnitude": 2.0, "break_points": [0.5]}
        }

        tasks["trend_change"] = {
            "category": "structural",
            "generator": self._generate_structural_break,
            "params": {"break_type": "trend", "break_magnitude": 0.001, "break_points": [0.4, 0.7]}
        }

        # === OUTLIER PATTERNS ===
        tasks["additive_outliers"] = {
            "category": "outliers",
            "generator": self._generate_outlier_series,
            "params": {"outlier_type": "additive", "outlier_prob": 0.05, "outlier_magnitude": 3.0}
        }

        tasks["innovation_outliers"] = {
            "category": "outliers",
            "generator": self._generate_outlier_series,
            "params": {"outlier_type": "innovation", "outlier_prob": 0.03, "outlier_magnitude": 2.0}
        }

        # === CHAOTIC PATTERNS ===
        tasks["henon_map"] = {
            "category": "chaotic",
            "generator": self._generate_chaotic_series,
            "params": {"system": "henon", "a": 1.4, "b": 0.3}
        }

        tasks["lorenz_x"] = {
            "category": "chaotic",
            "generator": self._generate_chaotic_series,
            "params": {"system": "lorenz", "component": "x", "sigma": 10, "rho": 28, "beta": 8 / 3}
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

    def _generate_trend_series(self, trend_type: str, noise_level: float, **kwargs) -> np.ndarray:
        """Generate trend-based time series."""
        t = np.arange(self.config.n_samples)

        if trend_type == "linear":
            strength = kwargs.get("strength", 0.001)
            y = strength * t
        elif trend_type == "exponential":
            strength = kwargs.get("strength", 0.0001)
            y = np.exp(strength * t) - 1
        elif trend_type == "polynomial":
            coeffs = kwargs.get("coefficients", [0, 0.001, -1e-7])
            y = np.polyval(coeffs[::-1], t)
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")

        # Add noise
        noise = np.random.normal(0, noise_level, len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_seasonal_series(self, periods: List[int], amplitudes: List[float], noise_level: float) -> np.ndarray:
        """Generate seasonal time series."""
        t = np.arange(self.config.n_samples)
        y = np.zeros_like(t, dtype=float)

        for period, amplitude in zip(periods, amplitudes):
            # Primary seasonal component
            y += amplitude * np.sin(2 * np.pi * t / period)
            # Add harmonics for realism
            y += 0.3 * amplitude * np.sin(4 * np.pi * t / period + np.pi / 3)
            y += 0.1 * amplitude * np.sin(6 * np.pi * t / period + np.pi / 6)

        # Add noise
        noise = np.random.normal(0, noise_level, len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_trend_seasonal(self, trend_type: str, trend_strength: float,
                                 periods: List[int], seasonal_amplitudes: List[float],
                                 noise_level: float, **kwargs) -> np.ndarray:
        """Generate trend + seasonal combination."""
        t = np.arange(self.config.n_samples)

        # Generate trend
        if trend_type == "linear":
            trend = trend_strength * t
        elif trend_type == "exponential":
            trend = np.exp(trend_strength * t) - 1
        elif trend_type == "polynomial":
            coeffs = kwargs.get("coefficients", [0, trend_strength, -trend_strength / 1000])
            trend = np.polyval(coeffs[::-1], t)
        else:
            trend = np.zeros_like(t)

        # Generate seasonal components
        seasonal = np.zeros_like(t, dtype=float)
        for period, amplitude in zip(periods, seasonal_amplitudes):
            seasonal += amplitude * np.sin(2 * np.pi * t / period)
            seasonal += 0.2 * amplitude * np.sin(4 * np.pi * t / period + np.pi / 4)

        y = trend + seasonal

        # Add noise
        noise = np.random.normal(0, noise_level, len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_stochastic_series(self, process_type: str, **kwargs) -> np.ndarray:
        """Generate stochastic time series."""
        n = self.config.n_samples

        if process_type == "random_walk":
            drift = kwargs.get("drift", 0)
            volatility = kwargs.get("volatility", 0.02)
            innovations = np.random.normal(drift, volatility, n)
            y = np.cumsum(innovations)

        elif process_type == "ar":
            ar_coeffs = kwargs.get("ar_coeffs", [0.7])
            noise_std = kwargs.get("noise_std", 0.1)
            p = len(ar_coeffs)
            y = np.zeros(n)
            y[:p] = np.random.normal(0, noise_std, p)

            for t in range(p, n):
                y[t] = sum(ar_coeffs[i] * y[t - 1 - i] for i in range(p))
                y[t] += np.random.normal(0, noise_std)

        elif process_type == "ma":
            ma_coeffs = kwargs.get("ma_coeffs", [0.8])
            noise_std = kwargs.get("noise_std", 0.1)
            q = len(ma_coeffs)
            noise = np.random.normal(0, noise_std, n + q)
            y = np.zeros(n)

            for t in range(n):
                y[t] = noise[t + q] + sum(ma_coeffs[i] * noise[t + q - 1 - i] for i in range(q))

        elif process_type == "arma":
            ar_coeffs = kwargs.get("ar_coeffs", [0.6])
            ma_coeffs = kwargs.get("ma_coeffs", [0.4])
            noise_std = kwargs.get("noise_std", 0.1)
            p, q = len(ar_coeffs), len(ma_coeffs)
            max_lag = max(p, q)

            y = np.zeros(n)
            noise = np.random.normal(0, noise_std, n + max_lag)

            for t in range(max_lag, n):
                # AR part
                if p > 0:
                    y[t] = sum(ar_coeffs[i] * y[t - 1 - i] for i in range(p))
                # MA part
                if q > 0:
                    y[t] += sum(ma_coeffs[i] * noise[t + max_lag - 1 - i] for i in range(q))
                # Current innovation
                y[t] += noise[t + max_lag]

        return y.reshape(-1, 1)

    def _generate_mean_reverting(self, theta: float, mu: float, sigma: float) -> np.ndarray:
        """Generate mean-reverting process."""
        dt = 0.01
        n = self.config.n_samples
        y = np.zeros(n)
        y[0] = np.random.normal(mu, sigma)

        for t in range(1, n):
            dy = theta * (mu - y[t - 1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
            y[t] = y[t - 1] + dy

        return y.reshape(-1, 1)

    def _generate_intermittent_series(self, demand_prob: float, demand_mean: float, demand_std: float) -> np.ndarray:
        """Generate intermittent demand series."""
        n = self.config.n_samples
        y = np.zeros(n)

        # Generate demand occurrences
        demand_occurs = np.random.binomial(1, demand_prob, n)

        # Generate demand sizes when demand occurs
        demand_sizes = np.random.normal(demand_mean, demand_std, n)
        demand_sizes = np.maximum(demand_sizes, 0)  # Ensure non-negative

        y = demand_occurs * demand_sizes

        return y.reshape(-1, 1)

    def _generate_garch_series(self, alpha: float, beta: float, omega: float) -> np.ndarray:
        """Generate GARCH series."""
        n = self.config.n_samples
        y = np.zeros(n)
        sigma2 = np.zeros(n)

        # Initialize
        sigma2[0] = omega / (1 - alpha - beta)
        y[0] = np.random.normal(0, np.sqrt(sigma2[0]))

        for t in range(1, n):
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
            y[t] = np.random.normal(0, np.sqrt(sigma2[t]))

        return y.reshape(-1, 1)

    def _generate_regime_switching(self, regimes: int, switch_prob: float,
                                   regime_params: List[Tuple[float, float]]) -> np.ndarray:
        """Generate regime-switching series."""
        n = self.config.n_samples
        y = np.zeros(n)
        current_regime = 0

        for t in range(n):
            # Check for regime switch
            if np.random.random() < switch_prob:
                current_regime = (current_regime + 1) % regimes

            # Generate observation from current regime
            drift, volatility = regime_params[current_regime]
            if t == 0:
                y[t] = np.random.normal(0, volatility)
            else:
                y[t] = y[t - 1] + drift + np.random.normal(0, volatility)

        return y.reshape(-1, 1)

    def _generate_structural_break(self, break_type: str, break_magnitude: float,
                                   break_points: List[float]) -> np.ndarray:
        """Generate series with structural breaks."""
        n = self.config.n_samples
        t = np.arange(n)
        y = np.zeros(n)

        # Base trend
        y = 0.0005 * t + np.random.normal(0, 0.1, n)

        # Apply structural breaks
        for break_point in break_points:
            break_index = int(break_point * n)

            if break_type == "level":
                y[break_index:] += break_magnitude
            elif break_type == "trend":
                y[break_index:] += break_magnitude * np.arange(len(y[break_index:]))

        return y.reshape(-1, 1)

    def _generate_outlier_series(self, outlier_type: str, outlier_prob: float, outlier_magnitude: float) -> np.ndarray:
        """Generate series with outliers."""
        n = self.config.n_samples
        t = np.arange(n)

        # Base series (trend + seasonal)
        y = 0.001 * t + np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, n)

        # Add outliers
        outlier_locations = np.random.binomial(1, outlier_prob, n)
        outlier_magnitudes = np.random.normal(0, outlier_magnitude, n)

        if outlier_type == "additive":
            y += outlier_locations * outlier_magnitudes
        elif outlier_type == "innovation":
            # Innovation outliers affect subsequent observations
            for t in range(1, n):
                if outlier_locations[t]:
                    y[t:] += outlier_magnitudes[t] * np.exp(-0.1 * np.arange(len(y[t:])))

        return y.reshape(-1, 1)

    def _generate_chaotic_series(self, system: str, **kwargs) -> np.ndarray:
        """Generate chaotic time series."""
        n = self.config.n_samples

        if system == "henon":
            a, b = kwargs.get("a", 1.4), kwargs.get("b", 0.3)
            x, y = 0.1, 0.1
            trajectory = []

            for _ in range(n + 1000):  # Extra for stabilization
                x_new = 1 - a * x ** 2 + y
                y_new = b * x
                x, y = x_new, y_new
                trajectory.append(x)

            data = np.array(trajectory[1000:1000 + n])

        elif system == "lorenz":
            component = kwargs.get("component", "x")
            sigma, rho, beta = kwargs.get("sigma", 10), kwargs.get("rho", 28), kwargs.get("beta", 8 / 3)

            dt = 0.01
            x, y, z = 1.0, 1.0, 1.0
            trajectory = []

            for _ in range(n * 10):
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z

                x += dx * dt
                y += dy * dt
                z += dz * dt

                if component == "x":
                    trajectory.append(x)
                elif component == "y":
                    trajectory.append(y)
                else:
                    trajectory.append(z)

            data = np.array(trajectory[::10][:n])

        # Add small amount of noise
        noise = np.random.normal(0, 0.01, len(data))
        return (data + noise).reshape(-1, 1)

    def _generate_logistic_growth(self, carrying_capacity: float, growth_rate: float, noise_level: float) -> np.ndarray:
        """Generate logistic growth series."""
        t = np.arange(self.config.n_samples)
        y = carrying_capacity / (1 + np.exp(-growth_rate * (t - self.config.n_samples / 2)))

        # Add noise
        noise = np.random.normal(0, noise_level, len(y))
        return (y + noise).reshape(-1, 1)


# ---------------------------------------------------------------------
# Enhanced Data Processing
# ---------------------------------------------------------------------

class NBeatsDataProcessor:
    """Enhanced data processor for N-BEATS with category-specific scaling."""

    def __init__(self, config: NBeatsConfig):
        self.config = config
        self.scalers = {}
        self.task_categories = {}

    def create_sequences(self, data: np.ndarray, backcast_length: int, forecast_length: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """Create N-BEATS sequences (backcast, forecast)."""
        X, y = [], []

        for i in range(len(data) - backcast_length - forecast_length + 1):
            X.append(data[i:i + backcast_length])
            y.append(data[i + backcast_length:i + backcast_length + forecast_length])

        return np.array(X), np.array(y)

    def fit_scalers(self, task_data: Dict[str, np.ndarray], task_categories: Dict[str, str]):
        """Fit scalers with category-specific strategies."""
        self.task_categories = task_categories

        for task_name, data in task_data.items():
            category = task_categories[task_name]

            # Choose scaling strategy based on category
            if category in ["stochastic", "volatility", "chaotic"]:
                # Use robust scaling for volatile series
                scaler = TimeSeriesNormalizer(method='robust')
            elif category in ["intermittent", "outliers"]:
                # Use robust scaling for series with outliers
                scaler = TimeSeriesNormalizer(method='robust')
            else:
                # Use standard scaling for trend/seasonal patterns
                scaler = TimeSeriesNormalizer(method='standard')

            # Fit scaler on training portion
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size].reshape(1, -1, 1)
            scaler.fit(train_data)

            self.scalers[task_name] = scaler
            logger.info(f"Fitted {scaler.method} scaler for {task_name} ({category})")

    def transform_data(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")

        data_3d = data.reshape(1, -1, 1)
        scaled = self.scalers[task_name].transform(data_3d)
        return scaled.reshape(-1, 1)

    def inverse_transform_data(self, task_name: str, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform data using fitted scaler."""
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")

        if len(scaled_data.shape) == 3:
            # Handle sequence data
            batch_size, seq_len, features = scaled_data.shape
            reshaped = scaled_data.reshape(1, -1, features)
            inverse_scaled = self.scalers[task_name].inverse_transform(reshaped)
            return inverse_scaled.reshape(batch_size, seq_len, features)
        else:
            # Handle point data
            data_3d = scaled_data.reshape(1, -1, 1)
            inverse_scaled = self.scalers[task_name].inverse_transform(data_3d)
            return inverse_scaled.reshape(-1, 1)


# ---------------------------------------------------------------------
# N-BEATS Trainer
# ---------------------------------------------------------------------

class NBeatsTrainer:
    """Comprehensive trainer for N-BEATS multi-task forecasting."""

    def __init__(self, config: NBeatsConfig):
        self.config = config
        self.generator = NBeatsTimeSeriesGenerator(config)
        self.processor = NBeatsDataProcessor(config)
        self.task_names = self.generator.get_task_names()
        self.task_categories = self.generator.get_task_categories()

        logger.info(
            f"Initialized N-BEATS trainer with {len(self.task_names)} tasks across {len(self.task_categories)} categories")

    def prepare_data(self) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """Prepare multi-task data for all horizons and model types."""
        logger.info("Preparing N-BEATS multi-task data...")

        # Generate raw data for all tasks
        raw_data = {}
        task_categories = {}

        for task_name in self.task_names:
            logger.info(f"Generating data for task: {task_name}")
            raw_data[task_name] = self.generator.generate_task_data(task_name)
            task_categories[task_name] = self.generator.task_definitions[task_name]["category"]

        # Fit scalers
        self.processor.fit_scalers(raw_data, task_categories)

        # Prepare data for all horizons
        prepared_data = {}

        for horizon in self.config.forecast_horizons:
            prepared_data[horizon] = {}

            for task_name in self.task_names:
                data = raw_data[task_name]

                # Split data
                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))

                train_data = data[:train_size]
                val_data = data[train_size:train_size + val_size]
                test_data = data[train_size + val_size:]

                # Scale data
                train_scaled = self.processor.transform_data(task_name, train_data)
                val_scaled = self.processor.transform_data(task_name, val_data)
                test_scaled = self.processor.transform_data(task_name, test_data)

                # Create sequences for this horizon
                train_X, train_y = self.processor.create_sequences(train_scaled, self.config.backcast_length, horizon)
                val_X, val_y = self.processor.create_sequences(val_scaled, self.config.backcast_length, horizon)
                test_X, test_y = self.processor.create_sequences(test_scaled, self.config.backcast_length, horizon)

                prepared_data[horizon][task_name] = {
                    "train": (train_X, train_y),
                    "val": (val_X, val_y),
                    "test": (test_X, test_y)
                }

                logger.info(f"Task {task_name} (horizon {horizon}): "
                            f"train={train_X.shape[0]}, val={val_X.shape[0]}, test={test_X.shape[0]}")

        return prepared_data

    def create_model(self, model_type: str, forecast_length: int) -> NBeatsNet:
        """Create N-BEATS model based on type."""
        logger.info(f"Creating {model_type} N-BEATS model for horizon {forecast_length}")

        stack_types = self.config.stack_types[model_type]
        thetas_dim = self.config.thetas_dim[model_type]

        model = NBeatsNet(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            thetas_dim=thetas_dim,
            hidden_layer_units=self.config.hidden_layer_units,
            share_weights_in_stack=self.config.share_weights_in_stack,
            input_dim=1,
            output_dim=1
        )

        return model

    def train_model(self, model: NBeatsNet, train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    val_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                    horizon: int, model_type: str) -> Dict[str, Any]:
        """Train N-BEATS model on multi-task data."""
        logger.info(f"Training {model_type} model for horizon {horizon}")

        # Combine data from all tasks
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []

        for task_name, (X, y) in train_data.items():
            X_train_list.append(X)
            y_train_list.append(y)

        for task_name, (X, y) in val_data.items():
            X_val_list.append(X)
            y_val_list.append(y)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        logger.info(f"Combined training data: {X_train.shape}, validation: {X_val.shape}")

        # Choose loss function
        if self.config.primary_loss == "smape":
            loss_fn = SMAPELoss()
        elif self.config.primary_loss == "mase":
            loss_fn = MASELoss()
        else:
            loss_fn = "mae"

        # Create optimizer
        if self.config.optimizer == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == "adamw":
            optimizer = keras.optimizers.AdamW(learning_rate=self.config.learning_rate)
        else:
            optimizer = self.config.optimizer

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'mse']
        )

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
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TerminateOnNaN()
        ]

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return {
            "history": history.history,
            "model": model
        }

    def evaluate_model(self, model: NBeatsNet, test_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       horizon: int, model_type: str) -> Dict[str, ForecastMetrics]:
        """Comprehensive evaluation of N-BEATS model."""
        logger.info(f"Evaluating {model_type} model for horizon {horizon}")

        task_metrics = {}

        for task_name, (X_test, y_test) in test_data.items():
            logger.info(f"Evaluating task: {task_name}")

            # Make predictions
            predictions = model.predict(X_test, verbose=0)

            # Inverse transform predictions and targets
            pred_orig = self.processor.inverse_transform_data(task_name, predictions)
            y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

            # Calculate comprehensive metrics
            metrics = self._calculate_forecast_metrics(
                y_test_orig, pred_orig, task_name, model_type, horizon
            )

            task_metrics[task_name] = metrics

            logger.info(f"Task {task_name}: RMSE={metrics.rmse:.4f}, "
                        f"MAPE={metrics.mape:.4f}, Coverage_90={metrics.coverage_90:.4f}")

        return task_metrics

    def _calculate_forecast_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    task_name: str, model_type: str, horizon: int) -> ForecastMetrics:
        """Calculate comprehensive forecasting metrics."""

        # Flatten for easier computation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        # Basic metrics
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))

        # MAPE (handle zero values)
        non_zero_mask = np.abs(y_true_flat) > 1e-8
        if np.any(non_zero_mask):
            mape = np.mean(
                np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) / y_true_flat[non_zero_mask])) * 100
        else:
            mape = 0.0

        # SMAPE
        denominator = (np.abs(y_true_flat) + np.abs(y_pred_flat)) / 2
        smape = np.mean(np.abs(y_true_flat - y_pred_flat) / (denominator + 1e-8)) * 100

        # MASE (using naive seasonal forecast)
        if len(y_true_flat) > horizon:
            naive_errors = np.abs(y_true_flat[horizon:] - y_true_flat[:-horizon])
            mae_naive = np.mean(naive_errors)
            mase = mae / (mae_naive + 1e-8)
        else:
            mase = 0.0

        # Directional accuracy
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            # Multi-step forecast
            y_true_diff = np.diff(y_true, axis=1)
            y_pred_diff = np.diff(y_pred, axis=1)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        else:
            directional_accuracy = 0.0

        # Bootstrap confidence intervals (simplified)
        n_bootstrap = min(self.config.num_bootstrap_samples, len(y_pred_flat))
        bootstrap_errors = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_pred_flat), len(y_pred_flat), replace=True)
            bootstrap_pred = y_pred_flat[indices]
            bootstrap_true = y_true_flat[indices]
            bootstrap_errors.append(np.abs(bootstrap_pred - bootstrap_true))

        bootstrap_errors = np.array(bootstrap_errors)

        # Calculate prediction intervals
        coverage_80 = coverage_90 = coverage_95 = 0.0
        interval_width_80 = interval_width_90 = interval_width_95 = 0.0

        for i, confidence in enumerate([0.8, 0.9, 0.95]):
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            # Use bootstrap distribution for intervals
            lower_bounds = np.percentile(bootstrap_errors, lower_percentile, axis=0)
            upper_bounds = np.percentile(bootstrap_errors, upper_percentile, axis=0)

            # Calculate coverage
            in_interval = (np.abs(y_true_flat - y_pred_flat) >= lower_bounds) & \
                          (np.abs(y_true_flat - y_pred_flat) <= upper_bounds)
            coverage = np.mean(in_interval)

            # Calculate interval width
            interval_width = np.mean(upper_bounds - lower_bounds)

            if confidence == 0.8:
                coverage_80 = coverage
                interval_width_80 = interval_width
            elif confidence == 0.9:
                coverage_90 = coverage
                interval_width_90 = interval_width
            elif confidence == 0.95:
                coverage_95 = coverage
                interval_width_95 = interval_width

        # Forecast bias and variance
        forecast_bias = np.mean(y_pred_flat - y_true_flat)
        forecast_variance = np.var(y_pred_flat)

        # Get task category
        task_category = self.generator.task_definitions[task_name]["category"]

        return ForecastMetrics(
            task_name=task_name,
            task_category=task_category,
            model_type=model_type,
            horizon=horizon,
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            smape=smape,
            mase=mase,
            directional_accuracy=directional_accuracy,
            coverage_80=coverage_80,
            coverage_90=coverage_90,
            coverage_95=coverage_95,
            interval_width_80=interval_width_80,
            interval_width_90=interval_width_90,
            interval_width_95=interval_width_95,
            forecast_bias=forecast_bias,
            forecast_variance=forecast_variance,
            samples_count=len(y_true_flat)
        )

    def create_visualizations(self, models: Dict[str, Dict[str, NBeatsNet]],
                              test_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                              save_dir: str):
        """Create comprehensive visualizations."""
        logger.info("Creating N-BEATS visualizations...")

        os.makedirs(save_dir, exist_ok=True)

        # Create visualizations by category
        for category in self.task_categories:
            category_tasks = [name for name in self.task_names
                              if self.generator.task_definitions[name]["category"] == category]

            if len(category_tasks) > 4:
                category_tasks = category_tasks[:4]

            for horizon in self.config.forecast_horizons:
                fig, axes = plt.subplots(2, 2, figsize=(20, 12))
                axes = axes.flatten()
                fig.suptitle(f'N-BEATS Forecasting - {category.title()} (Horizon {horizon})', fontsize=16)

                for i, task_name in enumerate(category_tasks):
                    if i >= 4:
                        break

                    # Get test data
                    X_test, y_test = test_data[horizon][task_name]

                    # Make predictions with different models
                    predictions = {}
                    for model_type in self.config.model_types:
                        if model_type in models[horizon]:
                            pred = models[horizon][model_type].predict(X_test[:self.config.plot_samples], verbose=0)
                            predictions[model_type] = self.processor.inverse_transform_data(task_name, pred)

                    # Inverse transform true values
                    y_test_orig = self.processor.inverse_transform_data(task_name, y_test[:self.config.plot_samples])

                    # Create plot
                    ax = axes[i]

                    # Plot a few sample forecasts
                    for sample_idx in range(min(3, len(y_test_orig))):
                        # Plot backcast context (if available)
                        backcast_start = max(0, sample_idx * 10)
                        forecast_start = self.config.backcast_length

                        x_backcast = np.arange(backcast_start, forecast_start)
                        x_forecast = np.arange(forecast_start, forecast_start + horizon)

                        # True values
                        y_true_sample = y_test_orig[sample_idx].flatten()
                        ax.plot(x_forecast, y_true_sample, 'b-', alpha=0.7,
                                label='True' if sample_idx == 0 else "", linewidth=2)

                        # Predictions from different models
                        colors = ['red', 'green', 'orange']
                        for j, (model_type, pred) in enumerate(predictions.items()):
                            pred_sample = pred[sample_idx].flatten()
                            ax.plot(x_forecast, pred_sample, '--', color=colors[j % len(colors)],
                                    alpha=0.7, label=f'{model_type.title()}' if sample_idx == 0 else "", linewidth=2)

                    ax.set_title(f'{task_name.replace("_", " ").title()}')
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                # Hide unused subplots
                for i in range(len(category_tasks), 4):
                    axes[i].set_visible(False)

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{category}_horizon_{horizon}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved visualization for {category} category, horizon {horizon}")

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete N-BEATS experiment."""
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(self.config.result_dir, f"{self.config.experiment_name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

        logger.info(f"Starting N-BEATS experiment: {exp_dir}")

        # Prepare data
        prepared_data = self.prepare_data()

        # Train models for each horizon and type
        trained_models = {}
        all_metrics = {}

        for horizon in self.config.forecast_horizons:
            trained_models[horizon] = {}
            all_metrics[horizon] = {}

            for model_type in self.config.model_types:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Training {model_type} model for horizon {horizon}")
                logger.info(f"{'=' * 60}")

                # Create model
                model = self.create_model(model_type, horizon)

                # Prepare training data
                train_data = {name: data["train"] for name, data in prepared_data[horizon].items()}
                val_data = {name: data["val"] for name, data in prepared_data[horizon].items()}
                test_data = {name: data["test"] for name, data in prepared_data[horizon].items()}

                # Train model
                training_results = self.train_model(model, train_data, val_data, horizon, model_type)
                trained_models[horizon][model_type] = training_results["model"]

                # Evaluate model
                task_metrics = self.evaluate_model(training_results["model"], test_data, horizon, model_type)
                all_metrics[horizon][model_type] = task_metrics

                # Save model
                if self.config.save_results:
                    model_path = os.path.join(exp_dir, f"{model_type}_h{horizon}_model.keras")
                    training_results["model"].save(model_path)
                    logger.info(f"Saved model: {model_path}")

        # Create visualizations
        test_data_dict = {}
        for horizon in self.config.forecast_horizons:
            test_data_dict[horizon] = {name: data["test"] for name, data in prepared_data[horizon].items()}

        self.create_visualizations(trained_models, test_data_dict, exp_dir)

        # Generate comprehensive results
        results = self._generate_results(all_metrics, exp_dir)

        return {
            "results": results,
            "models": trained_models,
            "experiment_dir": exp_dir
        }

    def _generate_results(self, all_metrics: Dict, exp_dir: str) -> Dict[str, Any]:
        """Generate comprehensive results and save them."""
        logger.info("Generating comprehensive results...")

        # Create results dataframe
        results_data = []

        for horizon in self.config.forecast_horizons:
            for model_type in self.config.model_types:
                if model_type in all_metrics[horizon]:
                    for task_name, metrics in all_metrics[horizon][model_type].items():
                        results_data.append({
                            'Task': task_name.replace('_', ' ').title(),
                            'Category': metrics.task_category.title(),
                            'Model': model_type.title(),
                            'Horizon': horizon,
                            'RMSE': metrics.rmse,
                            'MAE': metrics.mae,
                            'MAPE': metrics.mape,
                            'SMAPE': metrics.smape,
                            'MASE': metrics.mase,
                            'Directional_Acc': metrics.directional_accuracy,
                            'Coverage_90': metrics.coverage_90,
                            'Interval_Width_90': metrics.interval_width_90,
                            'Bias': metrics.forecast_bias,
                            'Variance': metrics.forecast_variance,
                            'Samples': metrics.samples_count
                        })

        results_df = pd.DataFrame(results_data)

        # Print detailed results
        print("\n" + "=" * 120)
        print("N-BEATS MULTI-TASK FORECASTING RESULTS")
        print("=" * 120)

        print("\nDetailed Results:")
        print(results_df.to_string(index=False, float_format='%.4f'))

        # Summary by model type and horizon
        print("\n" + "=" * 120)
        print("SUMMARY BY MODEL TYPE AND HORIZON")
        print("=" * 120)

        summary_by_model = results_df.groupby(['Model', 'Horizon']).agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'MAPE': 'mean',
            'SMAPE': 'mean',
            'Coverage_90': 'mean',
            'Samples': 'sum'
        }).round(4)

        print(summary_by_model)

        # Summary by category
        print("\n" + "=" * 120)
        print("SUMMARY BY CATEGORY")
        print("=" * 120)

        summary_by_category = results_df.groupby('Category').agg({
            'RMSE': ['mean', 'std'],
            'SMAPE': ['mean', 'std'],
            'Coverage_90': ['mean', 'std'],
            'Samples': 'sum'
        }).round(4)

        print(summary_by_category)

        # Best model per task
        print("\n" + "=" * 120)
        print("BEST MODEL PER TASK (by RMSE)")
        print("=" * 120)

        best_models = results_df.loc[results_df.groupby(['Task', 'Horizon'])['RMSE'].idxmin()]
        best_summary = best_models.groupby('Model').size().sort_values(ascending=False)
        print(best_summary)

        # Save results
        if self.config.save_results:
            results_df.to_csv(os.path.join(exp_dir, 'detailed_results.csv'), index=False)
            summary_by_model.to_csv(os.path.join(exp_dir, 'summary_by_model.csv'))
            summary_by_category.to_csv(os.path.join(exp_dir, 'summary_by_category.csv'))
            best_models.to_csv(os.path.join(exp_dir, 'best_models.csv'), index=False)

            # Save aggregate metrics
            with open(os.path.join(exp_dir, 'experiment_summary.txt'), 'w') as f:
                f.write(f"N-BEATS Multi-Task Experiment Summary\n")
                f.write(f"Total Tasks: {len(self.task_names)}\n")
                f.write(f"Total Categories: {len(self.task_categories)}\n")
                f.write(f"Model Types: {self.config.model_types}\n")
                f.write(f"Forecast Horizons: {self.config.forecast_horizons}\n")
                f.write(f"Best Overall Model: {best_summary.index[0]}\n")
                f.write(f"Average RMSE: {results_df['RMSE'].mean():.4f}\n")
                f.write(f"Average Coverage: {results_df['Coverage_90'].mean():.4f}\n")

            logger.info(f"Results saved to {exp_dir}")

        return {
            "detailed_results": results_df,
            "summary_by_model": summary_by_model,
            "summary_by_category": summary_by_category,
            "best_models": best_models
        }


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main():
    """Main function to run the N-BEATS multi-task experiment."""
    config = NBeatsConfig()

    logger.info("Starting N-BEATS multi-task forecasting experiment")
    logger.info(f"Configuration: {len(config.model_types)} model types, "
                f"{len(config.forecast_horizons)} horizons, "
                f"{config.backcast_length} backcast length")

    try:
        # Initialize trainer
        trainer = NBeatsTrainer(config)

        # Run experiment
        results = trainer.run_experiment()

        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {results['experiment_dir']}")

        return results

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()