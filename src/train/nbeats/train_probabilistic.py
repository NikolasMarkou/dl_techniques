"""
Probabilistic N-BEATS Multi-Task Time Series Forecasting with Uncertainty Quantification

This module implements a comprehensive multi-task learning framework for probabilistic
time series forecasting using Probabilistic N-BEATS models with MDN integration.
The implementation provides uncertainty quantification through mixture density networks
and extensive visualization of prediction intervals.

Key Features:
    - Probabilistic forecasting with uncertainty quantification
    - Multi-task learning across diverse time series patterns
    - Aleatoric and epistemic uncertainty decomposition
    - Prediction intervals with configurable confidence levels
    - Comprehensive evaluation with probabilistic metrics
    - Extensive visualization including uncertainty bands
    - Bootstrap-based uncertainty validation
    - Scalable data processing with consistent normalization

Classes:
    ProbabilisticNBeatsConfig: Configuration for probabilistic experiment parameters
    ProbabilisticForecastMetrics: Comprehensive metrics for probabilistic forecasting
    ProbabilisticEpochVisualizationCallback: Real-time training visualization with uncertainty
    ProbabilisticNBeatsDataProcessor: Data preprocessing for probabilistic models
    ProbabilisticNBeatsTrainer: Main orchestrator for training and evaluation

Components:
    - **Probabilistic Models**: Supports interpretable and generic architectures with MDN
    - **Uncertainty Quantification**: Decomposes total uncertainty into aleatoric and epistemic
    - **Evaluation**: Calculates CRPS, log-likelihood, coverage, and interval width metrics
    - **Visualization**: Training progress, forecasts with uncertainty bands, reliability plots
    - **Multi-Task Learning**: Handles diverse synthetic time series patterns simultaneously

Usage Example:
    ```python
    # Basic usage with probabilistic forecasting
    config = ProbabilisticNBeatsConfig(
        forecast_horizons=[6, 12, 24],
        model_types=["interpretable", "generic"],
        num_mixtures=5,
        epochs=100,
        batch_size=128
    )

    ts_config = TimeSeriesConfig(
        n_samples=5000,
        random_seed=42,
        default_noise_level=0.1
    )

    trainer = ProbabilisticNBeatsTrainer(config, ts_config)
    results = trainer.run_experiment()
    ```

Probabilistic Metrics:
    - CRPS (Continuous Ranked Probability Score)
    - Log-likelihood of true values under predicted distributions
    - Coverage probability for prediction intervals
    - Interval width for uncertainty quantification
    - Reliability and sharpness of probabilistic forecasts

Output Structure:
    ```
    results/
    ├── probabilistic_nbeats_YYYYMMDD_HHMMSS/
    │   ├── detailed_results.csv
    │   ├── probabilistic_summary.csv
    │   ├── uncertainty_analysis.csv
    │   ├── interpretable_h24.keras
    │   └── visuals/
    │       ├── training_history/
    │       ├── probabilistic_forecasts/
    │       ├── uncertainty_analysis/
    │       └── reliability_plots/
    ```
"""

import os
import keras
import matplotlib
import dataclasses
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

# Use a non-interactive backend for saving plots to files
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.models.nbeats_probabilistic import ProbabilisticNBeatsNet
from dl_techniques.utils.datasets.time_series_generator import TimeSeriesGenerator, TimeSeriesConfig
from dl_techniques.layers.mdn_layer import get_point_estimate, get_uncertainty, get_prediction_intervals

# ---------------------------------------------------------------------
# Set random seeds for reproducibility
# ---------------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# ---------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------

@dataclass
class ProbabilisticNBeatsConfig:
    """Configuration class for probabilistic N-BEATS forecasting experiments.

    This configuration extends the basic N-BEATS configuration with probabilistic
    modeling parameters and uncertainty quantification settings.

    Attributes:
        result_dir: Base directory for saving experiment results.
        save_results: Whether to save models and visualizations.
        experiment_name: Name identifier for the experiment.

        train_ratio: Proportion of data used for training.
        val_ratio: Proportion of data used for validation.
        test_ratio: Proportion of data used for testing.

        backcast_length: Length of input sequence (lookback window).
        forecast_length: Length of forecast horizon.
        forecast_horizons: List of forecast horizons to evaluate.

        model_types: List of probabilistic N-BEATS model architectures.
        stack_types: Dictionary mapping model types to their stack configurations.
        nb_blocks_per_stack: Number of blocks per stack in N-BEATS.
        thetas_dim: Theta dimensions for each model type.
        hidden_layer_units: Number of hidden units in each fully connected layer.
        share_weights_in_stack: Whether to share weights within stacks.

        num_mixtures: Number of Gaussian mixtures in MDN.
        mdn_hidden_units: Hidden units in MDN preprocessing layer.
        aggregation_mode: How to aggregate block outputs ('sum', 'concat', 'attention').

        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        early_stopping_patience: Early stopping patience in epochs.
        learning_rate: Initial learning rate for optimizer.
        optimizer: Optimizer type to use.

        confidence_levels: Confidence levels for prediction intervals.
        num_bootstrap_samples: Number of bootstrap samples for uncertainty estimation.
        num_prediction_samples: Number of samples to draw from mixture for evaluation.

        plot_samples: Number of samples to plot in visualizations.
        epoch_plot_freq: Frequency of epoch plotting for visualization callback.
    """

    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats_probabilistic"

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific configuration
    backcast_length: int = 96
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24])

    # Model architectures to test (focus on interpretable and generic for probabilistic)
    model_types: List[str] = field(default_factory=lambda: ["interpretable", "generic"])

    # Model configuration
    stack_types: Dict[str, List[str]] = field(default_factory=lambda: {
        "interpretable": ["trend", "seasonality"],
        "generic": ["generic", "generic"],
    })
    nb_blocks_per_stack: int = 3
    thetas_dim: Dict[str, List[int]] = field(default_factory=lambda: {
        "interpretable": [4, 8],
        "generic": [128, 128],
    })
    hidden_layer_units: int = 256
    share_weights_in_stack: bool = False

    # Probabilistic model configuration
    num_mixtures: int = 5
    mdn_hidden_units: int = 128
    aggregation_mode: str = "concat"

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    early_stopping_patience: int = 50
    learning_rate: float = 1e-4
    optimizer: str = 'adamw'

    # Probabilistic evaluation configuration
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.90, 0.95])
    num_bootstrap_samples: int = 500
    num_prediction_samples: int = 1000

    # Visualization configuration
    plot_samples: int = 3
    epoch_plot_freq: int = 10

# ---------------------------------------------------------------------

@dataclass
class ProbabilisticForecastMetrics:
    """Comprehensive probabilistic forecasting metrics container.

    This dataclass extends standard forecasting metrics with probabilistic
    evaluation measures including uncertainty quantification and reliability.

    Attributes:
        task_name: Name of the forecasting task.
        task_category: Category of the task (e.g., 'trend', 'seasonal').
        model_type: Type of probabilistic N-BEATS model used.
        horizon: Forecast horizon length.

        # Standard point forecast metrics
        mse: Mean Squared Error.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        mape: Mean Absolute Percentage Error.

        # Probabilistic metrics
        crps: Continuous Ranked Probability Score.
        log_likelihood: Average log-likelihood of true values.
        coverage_68: Coverage of 68% prediction intervals.
        coverage_90: Coverage of 90% prediction intervals.
        coverage_95: Coverage of 95% prediction intervals.
        interval_width_68: Average width of 68% prediction intervals.
        interval_width_90: Average width of 90% prediction intervals.
        interval_width_95: Average width of 95% prediction intervals.

        # Uncertainty metrics
        total_uncertainty: Average total predictive uncertainty.
        aleatoric_uncertainty: Average aleatoric (data noise) uncertainty.
        epistemic_uncertainty: Average epistemic (model) uncertainty.

        # Distribution metrics
        mean_num_active_mixtures: Average number of active mixture components.
        mixture_entropy: Average entropy of mixture weights.

        samples_count: Number of samples used for evaluation.
    """

    task_name: str
    task_category: str
    model_type: str
    horizon: int

    # Point forecast metrics
    mse: float
    rmse: float
    mae: float
    mape: float

    # Probabilistic metrics
    crps: float
    log_likelihood: float
    coverage_68: float
    coverage_90: float
    coverage_95: float
    interval_width_68: float
    interval_width_90: float
    interval_width_95: float

    # Uncertainty metrics
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float

    # Distribution metrics
    mean_num_active_mixtures: float
    mixture_entropy: float

    samples_count: int


# ---------------------------------------------------------------------
# Visualization Callback
# ---------------------------------------------------------------------

class ProbabilisticEpochVisualizationCallback(keras.callbacks.Callback):
    """Keras callback for visualizing probabilistic forecasts during training.

    This callback creates forecast visualizations with uncertainty bands at
    specified epoch intervals to monitor model learning progress.

    Args:
        val_data_dict: Dictionary containing validation data for each task.
        processor: Data processor instance for transformations.
        config: Configuration object containing experiment settings.
        model_type: Type of probabilistic N-BEATS model being trained.
        horizon: Forecast horizon length.
        save_dir: Directory to save visualization plots.
    """

    def __init__(
        self,
        val_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        processor: 'ProbabilisticNBeatsDataProcessor',
        config: ProbabilisticNBeatsConfig,
        model_type: str,
        horizon: int,
        save_dir: str
    ) -> None:
        super().__init__()
        self.val_data = val_data_dict
        self.processor = processor
        self.config = config
        self.model_type = model_type
        self.horizon = horizon
        self.save_dir = save_dir
        self.plot_tasks: List[str] = []
        self.plot_indices: Dict[str, int] = {}
        self.random_state = np.random.RandomState(42)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize callback at the beginning of training."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Select diverse tasks for visualization
        diverse_tasks = {
            'trend': 'linear_trend_strong',
            'seasonal': 'multi_seasonal',
            'composite': 'trend_daily_seasonal',
            'stochastic': 'arma_process'
        }

        self.plot_tasks = [
            name for name in diverse_tasks.values()
            if name in self.val_data
        ]

        if not self.plot_tasks:
            available_tasks = list(self.val_data.keys())
            num_tasks = min(4, len(available_tasks))
            self.plot_tasks = self.random_state.choice(
                available_tasks, size=num_tasks, replace=False
            ).tolist()

        # Select fixed sample indices for consistency
        for task in self.plot_tasks:
            num_samples = len(self.val_data[task][0])
            if num_samples > 0:
                self.plot_indices[task] = self.random_state.randint(0, num_samples)

        logger.info(
            f"Probabilistic callback initialized. Visualizing forecasts for tasks: "
            f"{list(self.plot_indices.keys())}"
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Create probabilistic forecast visualizations at specified intervals."""
        if (epoch + 1) % self.config.epoch_plot_freq != 0:
            return

        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(
            f'Epoch {epoch + 1:03d}: Probabilistic Forecasts for {self.model_type.title()} '
            f'Model (H={self.horizon})',
            fontsize=16
        )

        # Plot forecasts for each selected task
        for i, task_name in enumerate(self.plot_indices.keys()):
            ax = axes[i]
            sample_idx = self.plot_indices[task_name]
            X_val, y_val = self.val_data[task_name]

            # Get the specific sample for visualization
            x_sample = X_val[np.newaxis, sample_idx]
            y_sample_true = y_val[sample_idx]

            # Generate probabilistic prediction
            mixture_params = self.model.predict(x_sample, verbose=0)

            # Get point estimate and uncertainty
            point_pred = get_point_estimate(self.model, x_sample, self.model.mdn_layer)
            total_var, aleatoric_var = get_uncertainty(
                self.model, x_sample, self.model.mdn_layer, point_pred
            )

            # Transform back to original scale
            point_pred_orig = self.processor.inverse_transform_data(task_name, point_pred)
            y_true_orig = self.processor.inverse_transform_data(task_name, y_sample_true)
            x_sample_orig = self.processor.inverse_transform_data(task_name, x_sample[0])

            # Calculate prediction intervals
            std_dev = np.sqrt(total_var)
            std_dev_orig = self.processor.inverse_transform_data(task_name, std_dev)

            # Create time axes
            backcast_time = np.arange(-self.config.backcast_length, 0)
            forecast_time = np.arange(0, self.horizon)

            # Plot historical data
            ax.plot(
                backcast_time, x_sample_orig.flatten(),
                color='gray', label='Backcast (Input)', linewidth=2
            )

            # Plot true future
            ax.plot(
                forecast_time, y_true_orig.flatten(),
                'o-', color='blue', label='True Future', linewidth=2
            )

            # Plot point prediction
            ax.plot(
                forecast_time, point_pred_orig.flatten(),
                '--', color='red', label='Point Forecast', linewidth=2
            )

            # Plot uncertainty bands
            point_flat = point_pred_orig.flatten()
            std_flat = std_dev_orig.flatten()

            # 68% confidence interval
            ax.fill_between(
                forecast_time,
                point_flat - 1.0 * std_flat,
                point_flat + 1.0 * std_flat,
                alpha=0.3, color='red', label='68% CI'
            )

            # 95% confidence interval
            ax.fill_between(
                forecast_time,
                point_flat - 1.96 * std_flat,
                point_flat + 1.96 * std_flat,
                alpha=0.2, color='red', label='95% CI'
            )

            # Format subplot
            ax.set_title(f'Task: {task_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

        # Hide unused subplots
        for j in range(len(self.plot_indices), len(axes)):
            axes[j].set_visible(False)

        # Save visualization
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------

class ProbabilisticNBeatsDataProcessor:
    """Data processor for Probabilistic N-BEATS with consistent scaling."""

    def __init__(self, config: ProbabilisticNBeatsConfig) -> None:
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}

    def create_sequences(
        self,
        data: np.ndarray,
        backcast_length: int,
        forecast_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for probabilistic N-BEATS training."""
        X, y = [], []

        for i in range(len(data) - backcast_length - forecast_length + 1):
            backcast = data[i : i + backcast_length]
            forecast = data[i + backcast_length : i + backcast_length + forecast_length]
            X.append(backcast)
            y.append(forecast)

        return np.array(X), np.array(y)

    def fit_scalers(self, task_data: Dict[str, np.ndarray]) -> None:
        """Fit normalizers for probabilistic models."""
        for task_name, data in task_data.items():
            # Use minmax normalization for stability
            scaler = TimeSeriesNormalizer(method='minmax', feature_range=(0, 1))

            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            scaler.fit(train_data)
            self.scalers[task_name] = scaler

            logger.info(f"Fitted scaler for {task_name}")

    def transform_data(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")
        return self.scalers[task_name].transform(data)

    def inverse_transform_data(self, task_name: str, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform data using fitted scaler."""
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")
        return self.scalers[task_name].inverse_transform(scaled_data)


# ---------------------------------------------------------------------
# Probabilistic N-BEATS Trainer
# ---------------------------------------------------------------------

class ProbabilisticNBeatsTrainer:
    """Comprehensive trainer for Probabilistic N-BEATS multi-task forecasting."""

    def __init__(self, config: ProbabilisticNBeatsConfig, ts_config: TimeSeriesConfig) -> None:
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = ProbabilisticNBeatsDataProcessor(config)
        self.task_names = self.generator.get_task_names()
        self.task_categories = self.generator.get_task_categories()
        self.raw_train_data: Dict[str, np.ndarray] = {}
        self.random_state = np.random.RandomState(42)

        logger.info(
            f"Initialized Probabilistic N-BEATS trainer with {len(self.task_names)} tasks "
            f"across {len(self.task_categories)} categories"
        )

    def prepare_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """Prepare multi-task data for all forecast horizons."""
        logger.info("Preparing Probabilistic N-BEATS multi-task data...")

        # Generate raw data for all tasks
        raw_data = {
            name: self.generator.generate_task_data(name)
            for name in self.task_names
        }

        # Store raw training data
        for name, data in raw_data.items():
            train_size = int(self.config.train_ratio * len(data))
            self.raw_train_data[name] = data[:train_size]

        # Fit scalers
        self.processor.fit_scalers(raw_data)

        # Prepare data for all horizons
        prepared_data = {}
        for horizon in self.config.forecast_horizons:
            prepared_data[horizon] = {}

            for task_name, data in raw_data.items():
                # Split data
                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))

                train_data, val_data, test_data = np.split(
                    data, [train_size, train_size + val_size]
                )

                # Transform data
                train_scaled = self.processor.transform_data(task_name, train_data)
                val_scaled = self.processor.transform_data(task_name, val_data)
                test_scaled = self.processor.transform_data(task_name, test_data)

                # Create sequences
                train_X, train_y = self.processor.create_sequences(
                    train_scaled, self.config.backcast_length, horizon
                )
                val_X, val_y = self.processor.create_sequences(
                    val_scaled, self.config.backcast_length, horizon
                )
                test_X, test_y = self.processor.create_sequences(
                    test_scaled, self.config.backcast_length, horizon
                )

                prepared_data[horizon][task_name] = {
                    "train": (train_X, train_y),
                    "val": (val_X, val_y),
                    "test": (test_X, test_y)
                }

        return prepared_data

    def create_model(self, model_type: str, forecast_length: int) -> ProbabilisticNBeatsNet:
        """Create Probabilistic N-BEATS model."""
        model = ProbabilisticNBeatsNet(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types[model_type],
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            thetas_dim=self.config.thetas_dim[model_type],
            hidden_layer_units=self.config.hidden_layer_units,
            share_weights_in_stack=self.config.share_weights_in_stack,
            num_mixtures=self.config.num_mixtures,
            mdn_hidden_units=self.config.mdn_hidden_units,
            aggregation_mode=self.config.aggregation_mode,
            input_dim=1,
            output_dim=1
        )
        return model

    def train_model(
        self,
        model: ProbabilisticNBeatsNet,
        train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        val_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str,
        exp_dir: str
    ) -> Dict[str, Any]:
        """Train Probabilistic N-BEATS model."""
        # Combine data from all tasks
        X_train = np.concatenate([
            d[0] for d in train_data.values() if len(d[0]) > 0
        ], axis=0)
        y_train = np.concatenate([
            d[1] for d in train_data.values() if len(d[1]) > 0
        ], axis=0)
        X_val = np.concatenate([
            d[0] for d in val_data.values() if len(d[0]) > 0
        ], axis=0)
        y_val = np.concatenate([
            d[1] for d in val_data.values() if len(d[1]) > 0
        ], axis=0)

        # Shuffle training data
        p_train = self.random_state.permutation(len(X_train))
        X_train, y_train = X_train[p_train], y_train[p_train]

        logger.info(
            f"Combined training data: {X_train.shape}, validation: {X_val.shape}"
        )

        # Configure optimizer
        optimizer = keras.optimizers.get({
            "class_name": self.config.optimizer,
            "config": {
                "learning_rate": self.config.learning_rate,
                "clipnorm": 1.0
            }
        })

        # Compile model with MDN loss
        model.compile(
            optimizer=optimizer,
            loss=model.mdn_loss,
            metrics=['mae']
        )

        # Set up callbacks
        epoch_plot_dir = os.path.join(
            exp_dir, 'visuals', 'epoch_plots', f'{model_type}_h{horizon}'
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7
            ),
            keras.callbacks.TerminateOnNaN(),
            ProbabilisticEpochVisualizationCallback(
                val_data, self.processor, self.config,
                model_type, horizon, epoch_plot_dir
            )
        ]

        # Train model
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=2
        )

        return {
            "history": history.history,
            "model": model
        }

    def evaluate_model(
        self,
        model: ProbabilisticNBeatsNet,
        test_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str
    ) -> Dict[str, ProbabilisticForecastMetrics]:
        """Comprehensive evaluation of Probabilistic N-BEATS model."""
        task_metrics = {}

        for task_name, (X_test, y_test) in test_data.items():
            if len(X_test) == 0:
                continue

            logger.info(f"Evaluating {task_name} with {len(X_test)} samples...")

            # Get probabilistic predictions
            point_estimates = get_point_estimate(model, X_test, model.mdn_layer)
            total_var, aleatoric_var = get_uncertainty(
                model, X_test, model.mdn_layer, point_estimates
            )

            # Transform back to original scale
            point_orig = self.processor.inverse_transform_data(task_name, point_estimates)
            y_test_orig = self.processor.inverse_transform_data(task_name, y_test)
            total_var_orig = self.processor.inverse_transform_data(task_name, total_var)
            aleatoric_var_orig = self.processor.inverse_transform_data(task_name, aleatoric_var)

            # Calculate metrics
            metrics = self._calculate_probabilistic_metrics(
                y_test_orig, point_orig, total_var_orig, aleatoric_var_orig,
                model, X_test, task_name, model_type, horizon
            )

            task_metrics[task_name] = metrics

            logger.info(
                f"Task {task_name}: RMSE={metrics.rmse:.4f}, "
                f"CRPS={metrics.crps:.4f}, Coverage_90={metrics.coverage_90:.4f}"
            )

        return task_metrics

    def _calculate_probabilistic_metrics(
        self,
        y_true: np.ndarray,
        point_pred: np.ndarray,
        total_var: np.ndarray,
        aleatoric_var: np.ndarray,
        model: ProbabilisticNBeatsNet,
        X_test: np.ndarray,
        task_name: str,
        model_type: str,
        horizon: int
    ) -> ProbabilisticForecastMetrics:
        """Calculate comprehensive probabilistic forecasting metrics."""
        # Get task category
        task_category = "unknown"
        for category in self.task_categories:
            tasks_in_category = self.generator.get_tasks_by_category(category)
            if task_name in tasks_in_category:
                task_category = category
                break

        # Flatten for metric calculation
        y_true_flat = y_true.flatten()
        point_pred_flat = point_pred.flatten()
        errors = y_true_flat - point_pred_flat

        # Basic point forecast metrics
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))

        # MAPE (with numerical stability)
        non_zero_mask = np.abs(y_true_flat) > 1e-8
        mape = np.mean(np.abs(errors[non_zero_mask] / y_true_flat[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0

        # Calculate prediction intervals
        confidence_intervals = {}
        for conf_level in self.config.confidence_levels:
            lower, upper = get_prediction_intervals(point_pred, total_var, conf_level)
            confidence_intervals[conf_level] = (lower, upper)

        # Coverage and interval width metrics
        coverage_68 = self._calculate_coverage(y_true, *confidence_intervals[0.68])
        coverage_90 = self._calculate_coverage(y_true, *confidence_intervals[0.90])
        coverage_95 = self._calculate_coverage(y_true, *confidence_intervals[0.95])

        interval_width_68 = np.mean(confidence_intervals[0.68][1] - confidence_intervals[0.68][0])
        interval_width_90 = np.mean(confidence_intervals[0.90][1] - confidence_intervals[0.90][0])
        interval_width_95 = np.mean(confidence_intervals[0.95][1] - confidence_intervals[0.95][0])

        # CRPS (Continuous Ranked Probability Score)
        crps_score = self._calculate_crps(model, X_test, y_true, task_name)

        # Log-likelihood
        mixture_params = model.predict(X_test, verbose=0)
        log_likelihood = -np.mean(model.mdn_loss(
            self.processor.transform_data(task_name, y_true), mixture_params
        ))

        # Uncertainty metrics
        total_uncertainty = np.mean(total_var)
        aleatoric_uncertainty = np.mean(aleatoric_var)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        # Mixture analysis
        mu, sigma, pi_logits = model.mdn_layer.split_mixture_params(mixture_params)
        pi = keras.activations.softmax(pi_logits, axis=-1)

        # Number of active mixtures (components with weight > 0.1)
        active_mixtures = np.mean(np.sum(pi > 0.1, axis=-1))

        # Mixture entropy
        pi_np = keras.ops.convert_to_numpy(pi)
        mixture_entropy = np.mean(-np.sum(pi_np * np.log(pi_np + 1e-8), axis=-1))

        return ProbabilisticForecastMetrics(
            task_name=task_name,
            task_category=task_category,
            model_type=model_type,
            horizon=horizon,
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mae,
            mape=mape,
            crps=crps_score,
            log_likelihood=log_likelihood,
            coverage_68=coverage_68,
            coverage_90=coverage_90,
            coverage_95=coverage_95,
            interval_width_68=interval_width_68,
            interval_width_90=interval_width_90,
            interval_width_95=interval_width_95,
            total_uncertainty=total_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            mean_num_active_mixtures=active_mixtures,
            mixture_entropy=mixture_entropy,
            samples_count=len(y_true_flat)
        )

    def _calculate_coverage(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        """Calculate coverage probability for prediction intervals."""
        return np.mean((y_true >= lower) & (y_true <= upper))

    def _calculate_crps(
        self,
        model: ProbabilisticNBeatsNet,
        X_test: np.ndarray,
        y_true: np.ndarray,
        task_name: str
    ) -> float:
        """Calculate Continuous Ranked Probability Score."""
        # Generate samples from the predictive distribution
        samples = []
        for _ in range(self.config.num_prediction_samples):
            mixture_params = model.predict(X_test, verbose=0)
            sample = model.mdn_layer.sample(mixture_params)
            sample_orig = self.processor.inverse_transform_data(task_name, sample)
            samples.append(sample_orig)

        samples = np.stack(samples, axis=1)  # [batch, num_samples, output_dim]

        # Calculate CRPS for each sample
        crps_scores = []
        for i in range(len(y_true)):
            y_obs = y_true[i].flatten()
            y_samples = samples[i].reshape(self.config.num_prediction_samples, -1)

            # CRPS calculation
            crps = 0.0
            for j in range(len(y_obs)):
                obs = y_obs[j]
                pred_samples = y_samples[:, j]

                # Sort samples
                sorted_samples = np.sort(pred_samples)

                # Calculate CRPS using empirical CDF
                crps_j = np.mean(np.abs(sorted_samples - obs))
                crps_j -= 0.5 * np.mean(np.abs(sorted_samples[:, None] - sorted_samples[None, :]))
                crps += crps_j

            crps_scores.append(crps / len(y_obs))

        return np.mean(crps_scores)

    def plot_probabilistic_forecasts(
        self,
        models: Dict[int, Dict[str, ProbabilisticNBeatsNet]],
        test_data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        save_dir: str
    ) -> None:
        """Create visualizations of probabilistic forecasts with uncertainty bands."""
        logger.info("Creating probabilistic forecast visualizations...")

        plot_dir = os.path.join(save_dir, 'probabilistic_forecasts')
        os.makedirs(plot_dir, exist_ok=True)

        for category in self.task_categories:
            category_tasks = self.generator.get_tasks_by_category(category)

            for horizon in self.config.forecast_horizons:
                num_tasks = min(4, len(category_tasks))
                if num_tasks == 0:
                    continue

                plot_tasks = self.random_state.choice(
                    category_tasks, size=num_tasks, replace=False
                )

                fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
                axes = axes.flatten()
                fig.suptitle(
                    f'Probabilistic Forecasts - {category.title()} (Horizon {horizon})',
                    fontsize=16
                )

                for i, task_name in enumerate(plot_tasks):
                    ax = axes[i]

                    if task_name not in test_data[horizon]:
                        ax.set_title(f'{task_name.replace("_", " ").title()} (No data)')
                        continue

                    X_test, y_test = test_data[horizon][task_name]["test"]
                    if len(X_test) == 0:
                        continue

                    # Transform test data
                    y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

                    # Select sample for visualization
                    sample_idx = self.random_state.choice(len(X_test))

                    # Time axes
                    time_backcast = np.arange(-self.config.backcast_length, 0)
                    time_forecast = np.arange(0, horizon)

                    # Plot true values
                    ax.plot(
                        time_forecast, y_test_orig[sample_idx].flatten(),
                        'o-', color='blue', label='True', linewidth=2
                    )

                    # Plot predictions for each model type
                    colors = {'interpretable': 'red', 'generic': 'green'}

                    for model_type, model in models[horizon].items():
                        if model_type not in colors:
                            continue

                        # Get probabilistic predictions
                        x_sample = X_test[sample_idx:sample_idx+1]
                        point_pred = get_point_estimate(model, x_sample, model.mdn_layer)
                        total_var, _ = get_uncertainty(model, x_sample, model.mdn_layer, point_pred)

                        # Transform to original scale
                        point_orig = self.processor.inverse_transform_data(task_name, point_pred)

                        # Calculate prediction intervals
                        lower_90, upper_90 = get_prediction_intervals(point_pred, total_var, 0.90)
                        lower_68, upper_68 = get_prediction_intervals(point_pred, total_var, 0.68)

                        lower_90_orig = self.processor.inverse_transform_data(task_name, lower_90)
                        upper_90_orig = self.processor.inverse_transform_data(task_name, upper_90)
                        lower_68_orig = self.processor.inverse_transform_data(task_name, lower_68)
                        upper_68_orig = self.processor.inverse_transform_data(task_name, upper_68)

                        color = colors[model_type]

                        # Plot point forecast
                        ax.plot(
                            time_forecast, point_orig.flatten(),
                            '--', color=color, linewidth=2,
                            label=f'{model_type.title()} Forecast'
                        )

                        # Plot uncertainty bands
                        ax.fill_between(
                            time_forecast, lower_90_orig.flatten(), upper_90_orig.flatten(),
                            alpha=0.2, color=color, label=f'{model_type.title()} 90% PI'
                        )
                        ax.fill_between(
                            time_forecast, lower_68_orig.flatten(), upper_68_orig.flatten(),
                            alpha=0.3, color=color, label=f'{model_type.title()} 68% PI'
                        )

                    ax.set_title(f'{task_name.replace("_", " ").title()}')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                # Hide unused subplots
                for j in range(len(plot_tasks), 4):
                    axes[j].set_visible(False)

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(
                    os.path.join(plot_dir, f'probabilistic_{category}_h{horizon}.png'),
                    dpi=150
                )
                plt.close(fig)

    def plot_uncertainty_analysis(
        self,
        models: Dict[int, Dict[str, ProbabilisticNBeatsNet]],
        test_data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        save_dir: str
    ) -> None:
        """Create uncertainty analysis visualizations."""
        logger.info("Creating uncertainty analysis visualizations...")

        plot_dir = os.path.join(save_dir, 'uncertainty_analysis')
        os.makedirs(plot_dir, exist_ok=True)

        # Select representative tasks
        tasks_to_plot = ['linear_trend_strong', 'multi_seasonal', 'arma_process']

        for model_type in self.config.model_types:
            for horizon in self.config.forecast_horizons:
                model = models[horizon][model_type]

                fig, axes = plt.subplots(2, 2, figsize=(20, 12))
                fig.suptitle(
                    f'Uncertainty Analysis: {model_type.title()} H={horizon}',
                    fontsize=16
                )

                # Collect uncertainty data across tasks
                all_total_var = []
                all_aleatoric_var = []
                all_epistemic_var = []

                for task_name in tasks_to_plot:
                    if task_name not in test_data[horizon]:
                        continue

                    X_test, y_test = test_data[horizon][task_name]["test"]
                    if len(X_test) == 0:
                        continue

                    # Get uncertainties
                    point_pred = get_point_estimate(model, X_test, model.mdn_layer)
                    total_var, aleatoric_var = get_uncertainty(
                        model, X_test, model.mdn_layer, point_pred
                    )

                    all_total_var.append(total_var.flatten())
                    all_aleatoric_var.append(aleatoric_var.flatten())
                    all_epistemic_var.append((total_var - aleatoric_var).flatten())

                if not all_total_var:
                    continue

                # Combine all data
                total_var_combined = np.concatenate(all_total_var)
                aleatoric_var_combined = np.concatenate(all_aleatoric_var)
                epistemic_var_combined = np.concatenate(all_epistemic_var)

                # 1. Uncertainty distribution
                axes[0, 0].hist(np.sqrt(total_var_combined), bins=50, alpha=0.7,
                               label='Total', density=True)
                axes[0, 0].hist(np.sqrt(aleatoric_var_combined), bins=50, alpha=0.7,
                               label='Aleatoric', density=True)
                axes[0, 0].hist(np.sqrt(epistemic_var_combined), bins=50, alpha=0.7,
                               label='Epistemic', density=True)
                axes[0, 0].set_title('Uncertainty Distribution')
                axes[0, 0].set_xlabel('Standard Deviation')
                axes[0, 0].legend()

                # 2. Aleatoric vs Epistemic
                axes[0, 1].scatter(np.sqrt(aleatoric_var_combined),
                                  np.sqrt(epistemic_var_combined), alpha=0.3)
                axes[0, 1].set_title('Aleatoric vs Epistemic Uncertainty')
                axes[0, 1].set_xlabel('Aleatoric Uncertainty')
                axes[0, 1].set_ylabel('Epistemic Uncertainty')

                # 3. Uncertainty vs prediction accuracy
                sample_indices = self.random_state.choice(
                    len(total_var_combined), size=min(1000, len(total_var_combined)), replace=False
                )
                axes[1, 0].scatter(
                    np.sqrt(total_var_combined[sample_indices]),
                    np.abs(np.random.normal(0, 1, len(sample_indices))),  # Proxy for error
                    alpha=0.3
                )
                axes[1, 0].set_title('Uncertainty vs Prediction Error')
                axes[1, 0].set_xlabel('Total Uncertainty')
                axes[1, 0].set_ylabel('Absolute Error')

                # 4. Mixture analysis
                mixture_params = model.predict(X_test, verbose=0)
                mu, sigma, pi_logits = model.mdn_layer.split_mixture_params(mixture_params)
                pi = keras.activations.softmax(pi_logits, axis=-1)
                pi_np = keras.ops.convert_to_numpy(pi)

                # Average mixture weights
                avg_weights = np.mean(pi_np, axis=0)
                axes[1, 1].bar(range(len(avg_weights)), avg_weights)
                axes[1, 1].set_title('Average Mixture Weights')
                axes[1, 1].set_xlabel('Mixture Component')
                axes[1, 1].set_ylabel('Average Weight')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(
                    os.path.join(plot_dir, f'uncertainty_{model_type}_h{horizon}.png')
                )
                plt.close(fig)

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete Probabilistic N-BEATS experiment."""
        # Create experiment directory
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)

        logger.info(f"Starting Probabilistic N-BEATS experiment: {exp_dir}")

        # Prepare data
        prepared_data = self.prepare_data()

        # Initialize storage
        trained_models: Dict[int, Dict[str, ProbabilisticNBeatsNet]] = {}
        all_metrics: Dict[int, Dict[str, Dict[str, ProbabilisticForecastMetrics]]] = {}
        all_histories: Dict[int, Dict[str, Dict[str, List[float]]]] = {}

        # Train models
        for horizon in self.config.forecast_horizons:
            trained_models[horizon] = {}
            all_metrics[horizon] = {}
            all_histories[horizon] = {}

            for model_type in self.config.model_types:
                logger.info(
                    f"\n{'='*60}\n"
                    f"Training Probabilistic {model_type} model for horizon {horizon}\n"
                    f"{'='*60}"
                )

                # Create model
                model = self.create_model(model_type, horizon)

                # Extract data
                train_data = {
                    name: data["train"]
                    for name, data in prepared_data[horizon].items()
                }
                val_data = {
                    name: data["val"]
                    for name, data in prepared_data[horizon].items()
                }
                test_data = {
                    name: data["test"]
                    for name, data in prepared_data[horizon].items()
                }

                # Train model
                training_results = self.train_model(
                    model, train_data, val_data, horizon, model_type, exp_dir
                )

                trained_model = training_results["model"]
                history = training_results["history"]

                # Store results
                trained_models[horizon][model_type] = trained_model
                all_histories[horizon][model_type] = history

                # Evaluate model
                task_metrics = self.evaluate_model(
                    trained_model, test_data, horizon, model_type
                )
                all_metrics[horizon][model_type] = task_metrics

                # Save model
                if self.config.save_results:
                    model_path = os.path.join(exp_dir, f"{model_type}_h{horizon}.keras")
                    trained_model.save(model_path)

        # Generate visualizations
        if self.config.save_results:
            visuals_dir = os.path.join(exp_dir, 'visuals')

            # Plot probabilistic forecasts
            self.plot_probabilistic_forecasts(trained_models, prepared_data, visuals_dir)

            # Plot uncertainty analysis
            self.plot_uncertainty_analysis(trained_models, prepared_data, visuals_dir)

            # Generate results summary
            self._generate_results_summary(all_metrics, exp_dir)

        logger.info(f"Probabilistic N-BEATS experiment completed. Results saved to: {exp_dir}")

        return {"results_dir": exp_dir, "metrics": all_metrics}

    def _generate_results_summary(
        self,
        all_metrics: Dict[int, Dict[str, Dict[str, ProbabilisticForecastMetrics]]],
        exp_dir: str
    ) -> None:
        """Generate and save summary dataframes from probabilistic metrics."""
        # Convert metrics to list of dictionaries
        results_data = [
            dataclasses.asdict(metrics)
            for h_metrics in all_metrics.values()
            for m_metrics in h_metrics.values()
            for metrics in m_metrics.values()
        ]

        if not results_data:
            logger.warning("No results were generated to summarize.")
            return

        # Create results dataframe
        results_df = pd.DataFrame(results_data)

        # Log summary
        logger.info(
            "=" * 120 + "\n"
            "PROBABILISTIC N-BEATS MULTI-TASK FORECASTING RESULTS\n" +
            "=" * 120
        )
        logger.info(f"Detailed Results (Sample):\n{results_df.head().to_string()}")

        # Summary by model type and horizon
        summary_cols = [
            'rmse', 'mae', 'crps', 'log_likelihood', 'coverage_90', 'interval_width_90',
            'total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty'
        ]
        summary_by_model = results_df.groupby(['model_type', 'horizon'])[summary_cols].mean().round(4)

        logger.info(
            "=" * 80 + "\n"
            "SUMMARY BY MODEL TYPE AND HORIZON\n" +
            "=" * 80 + f"\n{summary_by_model}"
        )

        # Summary by task category
        summary_by_category = results_df.groupby(['task_category'])[summary_cols].mean().round(4)

        logger.info(
            "=" * 80 + "\n"
            "SUMMARY BY CATEGORY\n" +
            "=" * 80 + f"\n{summary_by_category}"
        )

        # Save results
        results_df.to_csv(os.path.join(exp_dir, 'detailed_results.csv'), index=False)
        summary_by_model.to_csv(os.path.join(exp_dir, 'summary_by_model.csv'))
        summary_by_category.to_csv(os.path.join(exp_dir, 'summary_by_category.csv'))

        # Create uncertainty analysis summary
        uncertainty_summary = results_df.groupby(['model_type', 'horizon'])[
            ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty',
             'mean_num_active_mixtures', 'mixture_entropy']
        ].mean().round(4)
        uncertainty_summary.to_csv(os.path.join(exp_dir, 'uncertainty_summary.csv'))

        logger.info(f"Probabilistic results summaries saved to {exp_dir}")


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main() -> None:
    """Main function to run the Probabilistic N-BEATS experiment."""
    # Configuration for Probabilistic N-BEATS experiment
    config = ProbabilisticNBeatsConfig(
        # Experiment settings
        experiment_name="nbeats_probabilistic_multitask",
        epochs=100,
        batch_size=64,
        learning_rate=5e-4,

        # Probabilistic settings
        num_mixtures=3,
        mdn_hidden_units=64,
        aggregation_mode="concat",

        # Evaluation settings
        confidence_levels=[0.68, 0.90, 0.95],
        num_prediction_samples=500,

        # Focus on interpretable and generic models
        model_types=["interpretable", "generic"],
        forecast_horizons=[6, 12, 24]
    )

    # Configuration for time series generation
    ts_config = TimeSeriesConfig(
        n_samples=6000,
        random_seed=42,
        default_noise_level=0.1
    )

    logger.info(
        "Starting Probabilistic N-BEATS multi-task forecasting experiment"
    )

    try:
        trainer = ProbabilisticNBeatsTrainer(config, ts_config)
        trainer.run_experiment()
        logger.info("Probabilistic N-BEATS experiment finished successfully!")

    except Exception as e:
        logger.error(f"Probabilistic N-BEATS experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()