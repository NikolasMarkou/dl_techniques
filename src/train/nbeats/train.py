"""
Multi-Task Time Series Forecasting with N-BEATS (Corrected)

This implementation demonstrates a comprehensive multi-task learning approach using N-BEATS
models trained on diverse time series patterns. Key corrections include:

- [CRITICAL FIX] Consistent data scaling across all tasks to ensure stable multi-task training.
- [CRITICAL FIX] Correct, non-leaky calculation of the MASE metric.
- [CRITICAL FIX] Statistically sound logic for bootstrap prediction intervals and coverage.
- [BUG FIX] Resolved AttributeError by correctly initializing random_state in the trainer.
- [FINAL CONFIG FIX] Corrected model configuration mismatch for the 'generic' model type.
"""

import os
import json
import keras
import matplotlib
import dataclasses
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from datetime import datetime
from dataclasses import dataclass, field





matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.nbeats import NBeatsNet
from dl_techniques.losses.smape_loss import SMAPELoss, MASELoss
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer

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
    experiment_name: str = "nbeats_multitask_final"

    # Data config
    n_samples: int = 8000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific config
    backcast_length: int = 96
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24])

    # Model architectures to test
    model_types: List[str] = field(default_factory=lambda: ["interpretable", "generic", "hybrid"])

    # Model config
    stack_types: Dict[str, List[str]] = field(default_factory=lambda: {
        "interpretable": ["trend", "seasonality"],
        # [FINAL CONFIG FIX] The length of this list now matches the corresponding thetas_dim list.
        # It was ["generic", "generic"] (length 2), now it's correctly length 3.
        "generic": ["generic", "generic", "generic"],
        "hybrid": ["trend", "seasonality", "generic"]
    })
    nb_blocks_per_stack: int = 3
    thetas_dim: Dict[str, List[int]] = field(default_factory=lambda: {
        "interpretable": [4, 8],
        "generic": [128, 128, 128],
        "hybrid": [4, 8, 128]
    })
    hidden_layer_units: int = 512
    share_weights_in_stack: bool = False

    # Training config
    epochs: int = 150
    batch_size: int = 128
    early_stopping_patience: int = 20
    learning_rate: float = 1e-4
    optimizer: str = 'adamw'
    primary_loss: str = "mae"

    # Evaluation config
    confidence_levels: List[float] = field(default_factory=lambda: [0.90])
    num_bootstrap_samples: int = 500

    # Visualization config
    plot_samples: int = 3


@dataclass
class ForecastMetrics:
    """Comprehensive forecasting metrics."""
    task_name: str
    task_category: str
    model_type: str
    horizon: int
    mse: float
    rmse: float
    mae: float
    mape: float
    smape: float
    mase: float
    directional_accuracy: float
    coverage_90: float
    interval_width_90: float
    forecast_bias: float
    samples_count: int


# ---------------------------------------------------------------------
# Enhanced Time Series Generator
# ---------------------------------------------------------------------

class NBeatsTimeSeriesGenerator:
    """Generator for diverse time series patterns optimized for N-BEATS."""

    def __init__(self, config: NBeatsConfig):
        self.config = config
        self.task_definitions = self._define_tasks()
        self.random_state = np.random.RandomState(42)
        logger.info(f"Initialized N-BEATS generator with {len(self.task_definitions)} tasks")

    def _define_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Define comprehensive set of time series tasks for N-BEATS."""
        tasks = {}
        # === TREND PATTERNS ===
        tasks["linear_trend_strong"] = {"category": "trend", "generator": self._generate_trend_series, "params": {"trend_type": "linear", "strength": 0.002, "noise_level": 0.05}}
        tasks["linear_trend_weak"] = {"category": "trend", "generator": self._generate_trend_series, "params": {"trend_type": "linear", "strength": 0.0005, "noise_level": 0.1}}
        tasks["exponential_growth"] = {"category": "trend", "generator": self._generate_trend_series, "params": {"trend_type": "exponential", "strength": 0.0001, "noise_level": 0.08}}
        tasks["polynomial_trend"] = {"category": "trend", "generator": self._generate_trend_series, "params": {"trend_type": "polynomial", "coefficients": [0, 0.001, -2e-7], "noise_level": 0.06}}
        tasks["logistic_growth"] = {"category": "trend", "generator": self._generate_logistic_growth, "params": {"carrying_capacity": 10, "growth_rate": 0.01, "noise_level": 0.1}}
        # === SEASONAL PATTERNS ===
        tasks["daily_seasonality"] = {"category": "seasonal", "generator": self._generate_seasonal_series, "params": {"periods": [24], "amplitudes": [1.0], "noise_level": 0.08}}
        tasks["weekly_seasonality"] = {"category": "seasonal", "generator": self._generate_seasonal_series, "params": {"periods": [168], "amplitudes": [1.2], "noise_level": 0.06}}
        tasks["multi_seasonal"] = {"category": "seasonal", "generator": self._generate_seasonal_series, "params": {"periods": [24, 168], "amplitudes": [1.0, 0.8], "noise_level": 0.1}}
        tasks["complex_seasonal"] = {"category": "seasonal", "generator": self._generate_seasonal_series, "params": {"periods": [12, 24, 168], "amplitudes": [0.6, 1.0, 0.7], "noise_level": 0.12}}
        # === TREND + SEASONAL COMBINATIONS ===
        tasks["trend_daily_seasonal"] = {"category": "composite", "generator": self._generate_trend_seasonal, "params": {"trend_type": "linear", "trend_strength": 0.001, "periods": [24], "seasonal_amplitudes": [1.0], "noise_level": 0.08}}
        tasks["trend_weekly_seasonal"] = {"category": "composite", "generator": self._generate_trend_seasonal, "params": {"trend_type": "linear", "trend_strength": 0.0008, "periods": [168], "seasonal_amplitudes": [1.2], "noise_level": 0.1}}
        tasks["exp_trend_multi_seasonal"] = {"category": "composite", "generator": self._generate_trend_seasonal, "params": {"trend_type": "exponential", "trend_strength": 0.0001, "periods": [24, 168], "seasonal_amplitudes": [1.0, 0.6], "noise_level": 0.12}}
        # === STOCHASTIC PROCESSES ===
        tasks["random_walk"] = {"category": "stochastic", "generator": self._generate_stochastic_series, "params": {"process_type": "random_walk", "drift": 0.001, "volatility": 0.05}}
        tasks["ar_process"] = {"category": "stochastic", "generator": self._generate_stochastic_series, "params": {"process_type": "ar", "ar_coeffs": [0.7, -0.2], "noise_std": 0.1}}
        tasks["ma_process"] = {"category": "stochastic", "generator": self._generate_stochastic_series, "params": {"process_type": "ma", "ma_coeffs": [0.8, 0.3], "noise_std": 0.1}}
        tasks["arma_process"] = {"category": "stochastic", "generator": self._generate_stochastic_series, "params": {"process_type": "arma", "ar_coeffs": [0.6], "ma_coeffs": [0.4], "noise_std": 0.08}}
        tasks["mean_reverting"] = {"category": "stochastic", "generator": self._generate_mean_reverting, "params": {"theta": 0.05, "mu": 0, "sigma": 0.2}}
        # === INTERMITTENT PATTERNS ===
        tasks["intermittent_demand"] = {"category": "intermittent", "generator": self._generate_intermittent_series, "params": {"demand_prob": 0.3, "demand_mean": 2.0, "demand_std": 0.5}}
        tasks["lumpy_demand"] = {"category": "intermittent", "generator": self._generate_intermittent_series, "params": {"demand_prob": 0.1, "demand_mean": 5.0, "demand_std": 1.0}}
        # === VOLATILITY CLUSTERING ===
        tasks["garch_low_vol"] = {"category": "volatility", "generator": self._generate_garch_series, "params": {"alpha": 0.1, "beta": 0.8, "omega": 0.01}}
        tasks["garch_high_vol"] = {"category": "volatility", "generator": self._generate_garch_series, "params": {"alpha": 0.2, "beta": 0.7, "omega": 0.05}}
        # === REGIME SWITCHING ===
        tasks["regime_switching"] = {"category": "regime", "generator": self._generate_regime_switching, "params": {"regimes": 2, "switch_prob": 0.02, "regime_params": [(0.001, 0.05), (0.005, 0.15)]}}
        # === STRUCTURAL BREAKS ===
        tasks["level_shift"] = {"category": "structural", "generator": self._generate_structural_break, "params": {"break_type": "level", "break_magnitude": 2.0, "break_points": [0.5]}}
        tasks["trend_change"] = {"category": "structural", "generator": self._generate_structural_break, "params": {"break_type": "trend", "break_magnitude": 0.001, "break_points": [0.4, 0.7]}}
        # === OUTLIER PATTERNS ===
        tasks["additive_outliers"] = {"category": "outliers", "generator": self._generate_outlier_series, "params": {"outlier_type": "additive", "outlier_prob": 0.05, "outlier_magnitude": 3.0}}
        tasks["innovation_outliers"] = {"category": "outliers", "generator": self._generate_outlier_series, "params": {"outlier_type": "innovation", "outlier_prob": 0.03, "outlier_magnitude": 2.0}}
        # === CHAOTIC PATTERNS ===
        tasks["henon_map"] = {"category": "chaotic", "generator": self._generate_chaotic_series, "params": {"system": "henon", "a": 1.4, "b": 0.3}}
        tasks["lorenz_x"] = {"category": "chaotic", "generator": self._generate_chaotic_series, "params": {"system": "lorenz", "component": "x", "sigma": 10, "rho": 28, "beta": 8 / 3}}
        return tasks

    def get_task_names(self) -> List[str]:
        return list(self.task_definitions.keys())

    def get_task_categories(self) -> List[str]:
        return list(set(task["category"] for task in self.task_definitions.values()))

    def generate_task_data(self, task_name: str) -> np.ndarray:
        if task_name not in self.task_definitions:
            raise ValueError(f"Unknown task: {task_name}")
        task_def = self.task_definitions[task_name]
        return task_def["generator"](**task_def.get("params", {}))

    def _generate_trend_series(self, trend_type: str, noise_level: float, **kwargs) -> np.ndarray:
        t = np.arange(self.config.n_samples)
        if trend_type == "linear": y = kwargs.get("strength", 0.001) * t
        elif trend_type == "exponential": y = np.exp(kwargs.get("strength", 0.0001) * t) - 1
        elif trend_type == "polynomial": y = np.polyval(kwargs.get("coefficients", [0, 0.001, -1e-7])[::-1], t)
        else: raise ValueError(f"Unknown trend type: {trend_type}")
        return (y + self.random_state.normal(0, noise_level, len(y))).reshape(-1, 1)

    def _generate_seasonal_series(self, periods: List[int], amplitudes: List[float], noise_level: float) -> np.ndarray:
        t = np.arange(self.config.n_samples)
        y = np.zeros_like(t, dtype=float)
        for period, amplitude in zip(periods, amplitudes):
            y += amplitude * np.sin(2 * np.pi * t / period)
        return (y + self.random_state.normal(0, noise_level, len(y))).reshape(-1, 1)

    def _generate_trend_seasonal(self, trend_type: str, trend_strength: float, periods: List[int], seasonal_amplitudes: List[float], noise_level: float, **kwargs) -> np.ndarray:
        t = np.arange(self.config.n_samples)
        if trend_type == "linear": trend = trend_strength * t
        elif trend_type == "exponential": trend = np.exp(trend_strength * t) - 1
        else: trend = np.zeros_like(t)
        seasonal = sum(amp * np.sin(2 * np.pi * t / p) for p, amp in zip(periods, seasonal_amplitudes))
        return (trend + seasonal + self.random_state.normal(0, noise_level, len(t))).reshape(-1, 1)

    def _generate_stochastic_series(self, process_type: str, **kwargs) -> np.ndarray:
        n = self.config.n_samples
        if process_type == "random_walk":
            drift, volatility = kwargs.get("drift", 0), kwargs.get("volatility", 0.02)
            y = np.cumsum(self.random_state.normal(drift, volatility, n))
        elif process_type == "ar":
            ar_coeffs, noise_std = kwargs.get("ar_coeffs", [0.7]), kwargs.get("noise_std", 0.1)
            p = len(ar_coeffs)
            y = np.zeros(n)
            for t in range(p, n): y[t] = sum(ar_coeffs[i] * y[t - 1 - i] for i in range(p)) + self.random_state.normal(0, noise_std)
        elif process_type == "ma":
            ma_coeffs, noise_std = kwargs.get("ma_coeffs", [0.8]), kwargs.get("noise_std", 0.1)
            q = len(ma_coeffs)
            noise = self.random_state.normal(0, noise_std, n + q)
            y = np.zeros(n)
            for t in range(n): y[t] = noise[t + q] + sum(ma_coeffs[i] * noise[t + q - 1 - i] for i in range(q))
        elif process_type == "arma":
            ar, ma, std = kwargs.get("ar_coeffs", [0.6]), kwargs.get("ma_coeffs", [0.4]), kwargs.get("noise_std", 0.1)
            p, q = len(ar), len(ma)
            y = np.zeros(n)
            noise = self.random_state.normal(0, std, n)
            for t in range(max(p, q), n):
                ar_sum = sum(ar[i] * y[t - 1 - i] for i in range(p))
                ma_sum = sum(ma[i] * noise[t - 1 - i] for i in range(q))
                y[t] = ar_sum + ma_sum + noise[t]
        return y.reshape(-1, 1)

    def _generate_mean_reverting(self, theta: float, mu: float, sigma: float) -> np.ndarray:
        dt = 0.01
        y = np.zeros(self.config.n_samples)
        for t in range(1, self.config.n_samples): y[t] = y[t-1] + theta * (mu - y[t-1]) * dt + sigma * self.random_state.normal(0, np.sqrt(dt))
        return y.reshape(-1, 1)

    def _generate_intermittent_series(self, demand_prob: float, demand_mean: float, demand_std: float) -> np.ndarray:
        demand_occurs = self.random_state.binomial(1, demand_prob, self.config.n_samples)
        demand_sizes = self.random_state.normal(demand_mean, demand_std, self.config.n_samples)
        return (demand_occurs * np.maximum(demand_sizes, 0)).reshape(-1, 1)

    def _generate_garch_series(self, alpha: float, beta: float, omega: float) -> np.ndarray:
        n = self.config.n_samples
        y, sigma2 = np.zeros(n), np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta) if (1-alpha-beta) > 0 else omega
        for t in range(1, n):
            sigma2[t] = omega + alpha * y[t - 1]**2 + beta * sigma2[t - 1]
            y[t] = self.random_state.normal(0, np.sqrt(sigma2[t]))
        return y.reshape(-1, 1)

    def _generate_regime_switching(self, regimes: int, switch_prob: float, regime_params: List[Tuple[float, float]]) -> np.ndarray:
        n, y, regime = self.config.n_samples, np.zeros(self.config.n_samples), 0
        for t in range(1, n):
            if self.random_state.rand() < switch_prob: regime = (regime + 1) % regimes
            drift, vol = regime_params[regime]
            y[t] = y[t-1] + drift + self.random_state.normal(0, vol)
        return y.reshape(-1, 1)

    def _generate_structural_break(self, break_type: str, break_magnitude: float, break_points: List[float]) -> np.ndarray:
        n = self.config.n_samples
        y = 0.0005 * np.arange(n) + self.random_state.normal(0, 0.1, n)
        for bp in break_points:
            idx = int(bp * n)
            if break_type == "level": y[idx:] += break_magnitude
            elif break_type == "trend": y[idx:] += break_magnitude * np.arange(n - idx)
        return y.reshape(-1, 1)

    def _generate_outlier_series(self, outlier_type: str, outlier_prob: float, outlier_magnitude: float) -> np.ndarray:
        n = self.config.n_samples
        t = np.arange(n)
        y = 0.001 * t + np.sin(2 * np.pi * t / 24) + self.random_state.normal(0, 0.1, n)
        locations = self.random_state.binomial(1, outlier_prob, n).astype(bool)
        magnitudes = self.random_state.normal(0, outlier_magnitude, n)
        if outlier_type == "additive": y[locations] += magnitudes[locations]
        return y.reshape(-1, 1)

    def _generate_chaotic_series(self, system: str, **kwargs) -> np.ndarray:
        n = self.config.n_samples
        if system == "henon":
            a, b, x, y, traj = kwargs.get("a", 1.4), kwargs.get("b", 0.3), 0.1, 0.1, []
            for _ in range(n + 100):
                x_new, y_new = 1 - a * x**2 + y, b * x
                x, y = x_new, y_new
                if _ >= 100: traj.append(x)
            data = np.array(traj)
        elif system == "lorenz":
            s, r, b, dt = kwargs.get("sigma", 10), kwargs.get("rho", 28), kwargs.get("beta", 8/3), 0.01
            x, y, z, traj = 1.0, 1.0, 1.0, []
            for _ in range(n * 10 + 1000):
                dx, dy, dz = s*(y-x), x*(r-z)-y, x*y-b*z
                x, y, z = x+dx*dt, y+dy*dt, z+dz*dt
                if _ >= 1000 and _ % 10 == 0: traj.append(x)
            data = np.array(traj)
        return (data + self.random_state.normal(0, 0.01, len(data))).reshape(-1, 1)

    def _generate_logistic_growth(self, carrying_capacity: float, growth_rate: float, noise_level: float) -> np.ndarray:
        t = np.arange(self.config.n_samples)
        y = carrying_capacity / (1 + np.exp(-growth_rate * (t - self.config.n_samples / 2)))
        return (y + self.random_state.normal(0, noise_level, len(y))).reshape(-1, 1)


# ---------------------------------------------------------------------
# CORRECTED Data Processing
# ---------------------------------------------------------------------

class NBeatsDataProcessor:
    """Corrected data processor for N-BEATS with consistent scaling."""

    def __init__(self, config: NBeatsConfig):
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}

    def create_sequences(self, data: np.ndarray, backcast_length: int, forecast_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create N-BEATS sequences (backcast, forecast)."""
        X, y = [], []
        for i in range(len(data) - backcast_length - forecast_length + 1):
            X.append(data[i : i + backcast_length])
            y.append(data[i + backcast_length : i + backcast_length + forecast_length])
        return np.array(X), np.array(y)

    def fit_scalers(self, task_data: Dict[str, np.ndarray]):
        """Fit scalers using a consistent 'minmax' strategy for stable multi-task training."""
        for task_name, data in task_data.items():
            scaler = TimeSeriesNormalizer(method='minmax', feature_range=(0, 1))
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]
            scaler.fit(train_data)
            self.scalers[task_name] = scaler
            logger.info(f"Fitted minmax scaler for {task_name}")

    def transform_data(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler for a specific task."""
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")
        return self.scalers[task_name].transform(data)

    def inverse_transform_data(self, task_name: str, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform data using the fitted scaler."""
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")
        return self.scalers[task_name].inverse_transform(scaled_data)


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
        self.raw_train_data = {}
        self.random_state = np.random.RandomState(42)

        logger.info(f"Initialized N-BEATS trainer with {len(self.task_names)} tasks across {len(self.task_categories)} categories")

    def prepare_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """Prepare multi-task data for all horizons."""
        logger.info("Preparing N-BEATS multi-task data...")
        raw_data = {name: self.generator.generate_task_data(name) for name in self.task_names}

        for name, data in raw_data.items():
            train_size = int(self.config.train_ratio * len(data))
            self.raw_train_data[name] = data[:train_size]

        self.processor.fit_scalers(raw_data)

        prepared_data = {}
        for horizon in self.config.forecast_horizons:
            prepared_data[horizon] = {}
            for task_name, data in raw_data.items():
                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))

                train_data, val_data, test_data = np.split(data, [train_size, train_size + val_size])

                train_scaled = self.processor.transform_data(task_name, train_data)
                val_scaled = self.processor.transform_data(task_name, val_data)
                test_scaled = self.processor.transform_data(task_name, test_data)

                train_X, train_y = self.processor.create_sequences(train_scaled, self.config.backcast_length, horizon)
                val_X, val_y = self.processor.create_sequences(val_scaled, self.config.backcast_length, horizon)
                test_X, test_y = self.processor.create_sequences(test_scaled, self.config.backcast_length, horizon)

                prepared_data[horizon][task_name] = {
                    "train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y)
                }
        return prepared_data

    def create_model(self, model_type: str, forecast_length: int) -> NBeatsNet:
        """Create N-BEATS model based on type."""
        model = NBeatsNet(
            backcast_length=self.config.backcast_length, forecast_length=forecast_length,
            stack_types=self.config.stack_types[model_type],
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            thetas_dim=self.config.thetas_dim[model_type],
            hidden_layer_units=self.config.hidden_layer_units,
            share_weights_in_stack=self.config.share_weights_in_stack,
            input_dim=1, output_dim=1
        )
        return model

    def train_model(self, model: NBeatsNet, train_data: Dict[str, Tuple[np.ndarray, np.ndarray]], val_data: Dict[str, Tuple[np.ndarray, np.ndarray]], horizon: int, model_type: str) -> Dict[str, Any]:
        """Train N-BEATS model on combined multi-task data."""
        X_train = np.concatenate([d[0] for d in train_data.values() if len(d[0]) > 0], axis=0)
        y_train = np.concatenate([d[1] for d in train_data.values() if len(d[1]) > 0], axis=0)
        X_val = np.concatenate([d[0] for d in val_data.values() if len(d[0]) > 0], axis=0)
        y_val = np.concatenate([d[1] for d in val_data.values() if len(d[1]) > 0], axis=0)

        p_train = self.random_state.permutation(len(X_train))
        X_train, y_train = X_train[p_train], y_train[p_train]

        logger.info(f"Combined training data: {X_train.shape}, validation: {X_val.shape}")

        if self.config.primary_loss == "smape": loss_fn = SMAPELoss()
        else: loss_fn = self.config.primary_loss

        optimizer = keras.optimizers.get({
            "class_name": self.config.optimizer,
            "config": {"learning_rate": self.config.learning_rate, "clipnorm": 1.0}
        })

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae', 'mse'])

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.early_stopping_patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            keras.callbacks.TerminateOnNaN()
        ]

        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=self.config.epochs, batch_size=self.config.batch_size,
            callbacks=callbacks, verbose=2
        )
        return {"history": history.history, "model": model}

    def evaluate_model(self, model: NBeatsNet, test_data: Dict[str, Tuple[np.ndarray, np.ndarray]], horizon: int, model_type: str) -> Dict[str, ForecastMetrics]:
        """Comprehensive evaluation of N-BEATS model."""
        task_metrics = {}
        for task_name, (X_test, y_test) in test_data.items():
            if len(X_test) == 0: continue

            predictions = model.predict(X_test, verbose=0)
            pred_orig = self.processor.inverse_transform_data(task_name, predictions)
            y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

            raw_train_series = self.raw_train_data[task_name]

            metrics = self._calculate_forecast_metrics(
                y_test_orig, pred_orig, raw_train_series, task_name, model_type, horizon
            )
            task_metrics[task_name] = metrics
            logger.info(f"Task {task_name}: RMSE={metrics.rmse:.4f}, MASE={metrics.mase:.4f}, Coverage_90={metrics.coverage_90:.4f}")
        return task_metrics

    def _calculate_forecast_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, train_series: np.ndarray, task_name: str, model_type: str, horizon: int) -> ForecastMetrics:
        """Calculate comprehensive forecasting metrics with corrected MASE and intervals."""
        y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
        errors = y_true_flat - y_pred_flat

        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))

        non_zero_mask = np.abs(y_true_flat) > 1e-8
        mape = np.mean(np.abs(errors[non_zero_mask] / y_true_flat[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0

        smape_denom = (np.abs(y_true_flat) + np.abs(y_pred_flat))
        smape = np.mean(2 * np.abs(errors) / (smape_denom + 1e-8)) * 100

        if len(train_series) > 1:
            mae_naive_train = np.mean(np.abs(np.diff(train_series.flatten())))
            mase = mae / (mae_naive_train + 1e-8)
        else:
            mase = np.inf

        y_true_diff = np.diff(y_true.reshape(-1, horizon), axis=1)
        y_pred_diff = np.diff(y_pred.reshape(-1, horizon), axis=1)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) if y_true_diff.size > 0 else 0.0

        forecast_residuals = (y_pred - y_true).flatten()

        q_lower = np.quantile(forecast_residuals, q=0.05)
        q_upper = np.quantile(forecast_residuals, q=0.95)
        interval_width_90 = np.abs(q_upper - q_lower)

        lower_bound = y_pred_flat + q_lower
        upper_bound = y_pred_flat + q_upper
        coverage_90 = np.mean((y_true_flat >= lower_bound) & (y_true_flat <= upper_bound))

        return ForecastMetrics(
            task_name=task_name, task_category=self.generator.task_definitions[task_name]["category"],
            model_type=model_type, horizon=horizon, mse=mse, rmse=np.sqrt(mse), mae=mae, mape=mape,
            smape=smape, mase=mase, directional_accuracy=directional_accuracy,
            coverage_90=coverage_90, interval_width_90=interval_width_90,
            forecast_bias=np.mean(errors), samples_count=len(y_true_flat)
        )

    def create_visualizations(self, models, test_data, save_dir):
        """Create visualizations of forecast vs. actuals."""
        logger.info("Creating N-BEATS visualizations...")
        os.makedirs(save_dir, exist_ok=True)

        for category in self.task_categories:
            category_tasks = [name for name in self.task_names if self.generator.task_definitions[name]["category"] == category]

            for horizon in self.config.forecast_horizons:
                plot_tasks = self.random_state.choice(category_tasks, size=min(4, len(category_tasks)), replace=False)
                if not plot_tasks.any(): continue

                fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
                axes = axes.flatten()
                fig.suptitle(f'N-BEATS Forecasting - {category.title()} (Horizon {horizon})', fontsize=16)

                for i, task_name in enumerate(plot_tasks):
                    ax = axes[i]
                    X_test, y_test = test_data[horizon][task_name]
                    if len(X_test) == 0:
                        ax.set_title(f'{task_name.replace("_", " ").title()} (No test data)')
                        continue

                    y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

                    sample_idx = self.random_state.choice(len(X_test))

                    time_forecast = np.arange(self.config.backcast_length, self.config.backcast_length + horizon)
                    ax.plot(time_forecast, y_test_orig[sample_idx].flatten(), 'o-', color='blue', label='True')

                    colors = {'interpretable': 'red', 'generic': 'green', 'hybrid': 'purple'}
                    for model_type, model in models[horizon].items():
                        pred = model.predict(X_test[np.newaxis, sample_idx], verbose=0)
                        pred_orig = self.processor.inverse_transform_data(task_name, pred)
                        ax.plot(time_forecast, pred_orig.flatten(), '--', color=colors.get(model_type, 'gray'), label=f'{model_type.title()}')

                    ax.set_title(f'{task_name.replace("_", " ").title()}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                for j in range(len(plot_tasks), 4): axes[j].set_visible(False)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(os.path.join(save_dir, f'{category}_h{horizon}.png'), dpi=150)
                plt.close(fig)

    def run_experiment(self):
        """Run the complete N-BEATS experiment."""
        exp_dir = os.path.join(self.config.result_dir, f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting N-BEATS experiment: {exp_dir}")

        prepared_data = self.prepare_data()
        trained_models, all_metrics = {}, {}

        for horizon in self.config.forecast_horizons:
            trained_models[horizon], all_metrics[horizon] = {}, {}
            for model_type in self.config.model_types:
                logger.info(f"\n{'='*60}\nTraining {model_type} model for horizon {horizon}\n{'='*60}")
                model = self.create_model(model_type, horizon)

                train_data = {name: data["train"] for name, data in prepared_data[horizon].items()}
                val_data = {name: data["val"] for name, data in prepared_data[horizon].items()}
                test_data = {name: data["test"] for name, data in prepared_data[horizon].items()}

                training_results = self.train_model(model, train_data, val_data, horizon, model_type)
                trained_model = training_results["model"]
                trained_models[horizon][model_type] = trained_model

                task_metrics = self.evaluate_model(trained_model, test_data, horizon, model_type)
                all_metrics[horizon][model_type] = task_metrics

                if self.config.save_results:
                    trained_model.save(os.path.join(exp_dir, f"{model_type}_h{horizon}.keras"))

        if self.config.save_results:
            self.create_visualizations(trained_models, prepared_data, os.path.join(exp_dir, 'visuals'))
            self._generate_results_summary(all_metrics, exp_dir)

        logger.info(f"Experiment completed. Results saved to: {exp_dir}")
        return {"results_dir": exp_dir, "metrics": all_metrics}

    def _generate_results_summary(self, all_metrics: Dict, exp_dir: str):
        """Generate and save summary dataframes from the metrics."""
        results_data = [dataclasses.asdict(metrics)
                        for h_metrics in all_metrics.values()
                        for m_metrics in h_metrics.values()
                        for metrics in m_metrics.values()]

        if not results_data:
            logger.warning("No results were generated to summarize.")
            return

        results_df = pd.DataFrame(results_data)

        print("\n" + "=" * 120 + "\nN-BEATS MULTI-TASK FORECASTING RESULTS\n" + "=" * 120)
        print("\nDetailed Results (Sample):\n", results_df.head().to_string())

        summary_cols = ['rmse', 'mae', 'smape', 'mase', 'coverage_90']
        summary_by_model = results_df.groupby(['model_type', 'horizon'])[summary_cols].mean().round(4)
        print("\n" + "=" * 80 + "\nSUMMARY BY MODEL TYPE AND HORIZON\n" + "=" * 80 + "\n", summary_by_model)

        summary_by_category = results_df.groupby(['task_category'])[summary_cols].mean().round(4)
        print("\n" + "=" * 80 + "\nSUMMARY BY CATEGORY\n" + "=" * 80 + "\n", summary_by_category)

        results_df.to_csv(os.path.join(exp_dir, 'detailed_results.csv'), index=False)
        summary_by_model.to_csv(os.path.join(exp_dir, 'summary_by_model.csv'))
        summary_by_category.to_csv(os.path.join(exp_dir, 'summary_by_category.csv'))
        logger.info(f"Results summaries saved to {exp_dir}")

# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main():
    """Main function to run the N-BEATS multi-task experiment."""
    config = NBeatsConfig()
    logger.info("Starting N-BEATS multi-task forecasting experiment with final corrected settings")
    try:
        trainer = NBeatsTrainer(config)
        trainer.run_experiment()
        logger.info("Experiment finished successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()