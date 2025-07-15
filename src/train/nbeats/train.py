"""
Multi-Task Time Series Forecasting with N-BEATS

This implementation demonstrates a comprehensive multi-task learning approach using N-BEATS
models trained on diverse time series patterns with enhanced documentation and type safety.
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
from dl_techniques.models.nbeats import NBeatsNet
from dl_techniques.losses.smape_loss import SMAPELoss
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.utils.datasets.time_series_generator import TimeSeriesGenerator, TimeSeriesConfig

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
class NBeatsConfig:
    """Configuration class for large-scale multi-task N-BEATS forecasting experiments.

    This configuration class contains all hyperparameters and settings needed for
    running comprehensive N-BEATS multi-task forecasting experiments.

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

        model_types: List of N-BEATS model architectures to test.
        stack_types: Dictionary mapping model types to their stack configurations.
        nb_blocks_per_stack: Number of blocks per stack in N-BEATS.
        thetas_dim: Theta dimensions for each model type.
        hidden_layer_units: Number of hidden units in each layer.
        share_weights_in_stack: Whether to share weights within stacks.

        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        early_stopping_patience: Early stopping patience in epochs.
        learning_rate: Initial learning rate for optimizer.
        optimizer: Optimizer type to use.
        primary_loss: Primary loss function for training.

        confidence_levels: Confidence levels for prediction intervals.
        num_bootstrap_samples: Number of bootstrap samples for uncertainty estimation.

        plot_samples: Number of samples to plot in visualizations.
        epoch_plot_freq: Frequency of epoch plotting for visualization callback.
    """

    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats_multitask"

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific configuration
    backcast_length: int = 96
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24])

    # Model architectures to test
    model_types: List[str] = field(default_factory=lambda: ["interpretable", "generic", "hybrid"])

    # Model configuration
    stack_types: Dict[str, List[str]] = field(default_factory=lambda: {
        "interpretable": ["trend", "seasonality"],
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

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    early_stopping_patience: int = 50
    learning_rate: float = 1e-4
    optimizer: str = 'adamw'
    primary_loss: str = "mae"

    # Evaluation configuration
    confidence_levels: List[float] = field(default_factory=lambda: [0.90])
    num_bootstrap_samples: int = 500

    # Visualization configuration
    plot_samples: int = 3
    epoch_plot_freq: int = 10  # Plot every 10 epochs to avoid clutter


@dataclass
class ForecastMetrics:
    """Comprehensive forecasting metrics container.

    This dataclass stores all relevant metrics for evaluating time series
    forecasting performance across different tasks and models.

    Attributes:
        task_name: Name of the forecasting task.
        task_category: Category of the task (e.g., 'trend', 'seasonal').
        model_type: Type of N-BEATS model used.
        horizon: Forecast horizon length.
        mse: Mean Squared Error.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        mape: Mean Absolute Percentage Error.
        smape: Symmetric Mean Absolute Percentage Error.
        mase: Mean Absolute Scaled Error.
        directional_accuracy: Directional accuracy of forecasts.
        coverage_90: Coverage of 90% prediction intervals.
        interval_width_90: Width of 90% prediction intervals.
        forecast_bias: Bias in forecasts.
        samples_count: Number of samples used for evaluation.
    """

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
# Visualization Callback
# ---------------------------------------------------------------------

class EpochVisualizationCallback(keras.callbacks.Callback):
    """Keras callback for visualizing model forecasts during training.

    This callback creates forecast visualizations at specified epoch intervals
    to monitor model learning progress across different time series tasks.

    Args:
        val_data_dict: Dictionary containing validation data for each task.
        processor: Data processor instance for transformations.
        config: Configuration object containing experiment settings.
        model_type: Type of N-BEATS model being trained.
        horizon: Forecast horizon length.
        save_dir: Directory to save visualization plots.
    """

    def __init__(
        self,
        val_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        processor: 'NBeatsDataProcessor',
        config: NBeatsConfig,
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
        """Initialize callback at the beginning of training.

        Selects diverse tasks for visualization and creates save directory.

        Args:
            logs: Dictionary containing training logs (unused).
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Select diverse tasks to visualize throughout training
        diverse_tasks = {
            'trend': 'linear_trend_strong',
            'seasonal': 'multi_seasonal',
            'composite': 'trend_daily_seasonal',
            'stochastic': 'arma_process'
        }

        # Filter tasks that exist in validation data
        self.plot_tasks = [
            name for name in diverse_tasks.values()
            if name in self.val_data
        ]

        # Fallback if specific tasks aren't present
        if not self.plot_tasks:
            available_tasks = list(self.val_data.keys())
            num_tasks = min(4, len(available_tasks))
            self.plot_tasks = self.random_state.choice(
                available_tasks, size=num_tasks, replace=False
            ).tolist()

        # Select one fixed sample index per task for consistency across epochs
        for task in self.plot_tasks:
            num_samples = len(self.val_data[task][0])
            if num_samples > 0:
                self.plot_indices[task] = self.random_state.randint(0, num_samples)

        logger.info(
            f"Callback initialized. Visualizing forecasts for tasks: "
            f"{list(self.plot_indices.keys())}"
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Create forecast visualizations at specified epoch intervals.

        Args:
            epoch: Current epoch number.
            logs: Dictionary containing training logs (unused).
        """
        # Only plot at specified intervals
        if (epoch + 1) % self.config.epoch_plot_freq != 0:
            return

        # Create subplot grid for visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(
            f'Epoch {epoch + 1:03d}: Forecasts for {self.model_type.title()} '
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

            # Generate prediction and inverse transform to original scale
            y_sample_pred_scaled = self.model.predict(x_sample, verbose=0)
            y_sample_pred_orig = self.processor.inverse_transform_data(
                task_name, y_sample_pred_scaled
            )
            y_sample_true_orig = self.processor.inverse_transform_data(
                task_name, y_sample_true
            )
            x_sample_orig = self.processor.inverse_transform_data(
                task_name, x_sample[0]
            )

            # Create time axes for plotting
            backcast_time = np.arange(-self.config.backcast_length, 0)
            forecast_time = np.arange(0, self.horizon)

            # Plot historical data, true future, and forecast
            ax.plot(
                backcast_time, x_sample_orig.flatten(),
                color='gray', label='Backcast (Input)'
            )
            ax.plot(
                forecast_time, y_sample_true_orig.flatten(),
                'o-', color='blue', label='True Future'
            )
            ax.plot(
                forecast_time, y_sample_pred_orig.flatten(),
                'x--', color='red', label='Forecast'
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

class NBeatsDataProcessor:
    """Data processor for N-BEATS with consistent scaling across tasks.

    This class handles data preprocessing including normalization and
    sequence creation for N-BEATS training.

    Args:
        config: Configuration object containing experiment settings.

    Attributes:
        config: Configuration object.
        scalers: Dictionary mapping task names to their fitted scalers.
    """

    def __init__(self, config: NBeatsConfig) -> None:
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}

    def create_sequences(
        self,
        data: np.ndarray,
        backcast_length: int,
        forecast_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create N-BEATS sequences (backcast, forecast) from time series data.

        Args:
            data: Time series data.
            backcast_length: Length of input sequence.
            forecast_length: Length of forecast sequence.

        Returns:
            Tuple of (backcast_sequences, forecast_sequences).
        """
        X, y = [], []

        # Create sliding window sequences
        for i in range(len(data) - backcast_length - forecast_length + 1):
            # Extract backcast (input) and forecast (target) windows
            backcast = data[i : i + backcast_length]
            forecast = data[i + backcast_length : i + backcast_length + forecast_length]

            X.append(backcast)
            y.append(forecast)

        return np.array(X), np.array(y)

    def fit_scalers(self, task_data: Dict[str, np.ndarray]) -> None:
        """Fit normalizers using consistent 'minmax' strategy for stable multi-task training.

        Args:
            task_data: Dictionary mapping task names to their time series data.
        """
        for task_name, data in task_data.items():
            # Use minmax normalization for stability across diverse tasks
            scaler = TimeSeriesNormalizer(method='minmax', feature_range=(0, 1))

            # Fit on training portion only
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            scaler.fit(train_data)
            self.scalers[task_name] = scaler

            logger.info(f"Fitted minmax scaler for {task_name}")

    def transform_data(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler for a specific task.

        Args:
            task_name: Name of the task.
            data: Data to transform.

        Returns:
            Transformed data.

        Raises:
            ValueError: If scaler not fitted for the task.
        """
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")

        return self.scalers[task_name].transform(data)

    def inverse_transform_data(self, task_name: str, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform data using the fitted scaler.

        Args:
            task_name: Name of the task.
            scaled_data: Scaled data to inverse transform.

        Returns:
            Data in original scale.

        Raises:
            ValueError: If scaler not fitted for the task.
        """
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")

        return self.scalers[task_name].inverse_transform(scaled_data)


# ---------------------------------------------------------------------
# N-BEATS Trainer
# ---------------------------------------------------------------------

class NBeatsTrainer:
    """Comprehensive trainer for N-BEATS multi-task forecasting.

    This class orchestrates the entire N-BEATS training process including
    data preparation, model training, evaluation, and visualization.

    Args:
        config: Configuration object containing experiment settings.
        ts_config: Time series generator configuration.

    Attributes:
        config: Configuration object.
        ts_config: Time series generator configuration.
        generator: Time series generator instance.
        processor: Data processor instance.
        task_names: List of all available task names.
        task_categories: List of all task categories.
        raw_train_data: Dictionary storing raw training data for each task.
        random_state: Random state for reproducible experiments.
    """

    def __init__(self, config: NBeatsConfig, ts_config: TimeSeriesConfig) -> None:
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = NBeatsDataProcessor(config)
        self.task_names = self.generator.get_task_names()
        self.task_categories = self.generator.get_task_categories()
        self.raw_train_data: Dict[str, np.ndarray] = {}
        self.random_state = np.random.RandomState(42)

        logger.info(
            f"Initialized N-BEATS trainer with {len(self.task_names)} tasks "
            f"across {len(self.task_categories)} categories"
        )

    def prepare_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """Prepare multi-task data for all forecast horizons.

        Returns:
            Nested dictionary with structure:
            {horizon: {task_name: {split: (X, y)}}}
        """
        logger.info("Preparing N-BEATS multi-task data...")

        # Generate raw data for all tasks
        raw_data = {
            name: self.generator.generate_task_data(name)
            for name in self.task_names
        }

        # Store raw training data for MASE calculation
        for name, data in raw_data.items():
            train_size = int(self.config.train_ratio * len(data))
            self.raw_train_data[name] = data[:train_size]

        # Fit scalers on raw data
        self.processor.fit_scalers(raw_data)

        # Prepare data for all horizons
        prepared_data = {}
        for horizon in self.config.forecast_horizons:
            prepared_data[horizon] = {}

            for task_name, data in raw_data.items():
                # Split data into train/validation/test
                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))

                train_data, val_data, test_data = np.split(
                    data, [train_size, train_size + val_size]
                )

                # Transform data using fitted scalers
                train_scaled = self.processor.transform_data(task_name, train_data)
                val_scaled = self.processor.transform_data(task_name, val_data)
                test_scaled = self.processor.transform_data(task_name, test_data)

                # Create sequences for current horizon
                train_X, train_y = self.processor.create_sequences(
                    train_scaled, self.config.backcast_length, horizon
                )
                val_X, val_y = self.processor.create_sequences(
                    val_scaled, self.config.backcast_length, horizon
                )
                test_X, test_y = self.processor.create_sequences(
                    test_scaled, self.config.backcast_length, horizon
                )

                # Store prepared data
                prepared_data[horizon][task_name] = {
                    "train": (train_X, train_y),
                    "val": (val_X, val_y),
                    "test": (test_X, test_y)
                }

        return prepared_data

    def create_model(self, model_type: str, forecast_length: int) -> NBeatsNet:
        """Create N-BEATS model based on specified type.

        Args:
            model_type: Type of N-BEATS model ('interpretable', 'generic', 'hybrid').
            forecast_length: Length of forecast horizon.

        Returns:
            Configured N-BEATS model instance.
        """
        model = NBeatsNet(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types[model_type],
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            thetas_dim=self.config.thetas_dim[model_type],
            hidden_layer_units=self.config.hidden_layer_units,
            share_weights_in_stack=self.config.share_weights_in_stack,
            input_dim=1,
            output_dim=1
        )
        return model

    def train_model(
        self,
        model: NBeatsNet,
        train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        val_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str,
        exp_dir: str
    ) -> Dict[str, Any]:
        """Train N-BEATS model on combined multi-task data.

        Args:
            model: N-BEATS model instance to train.
            train_data: Training data for all tasks.
            val_data: Validation data for all tasks.
            horizon: Forecast horizon length.
            model_type: Type of N-BEATS model.
            exp_dir: Experiment directory for saving results.

        Returns:
            Dictionary containing training history and trained model.
        """
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

        # Shuffle training data for better learning
        p_train = self.random_state.permutation(len(X_train))
        X_train, y_train = X_train[p_train], y_train[p_train]

        logger.info(
            f"Combined training data: {X_train.shape}, "
            f"validation: {X_val.shape}"
        )

        # Configure loss function
        if self.config.primary_loss == "smape":
            loss_fn = SMAPELoss()
        else:
            loss_fn = self.config.primary_loss

        # Configure optimizer
        optimizer = keras.optimizers.get({
            "class_name": self.config.optimizer,
            "config": {
                "learning_rate": self.config.learning_rate,
                "clipnorm": 1.0  # Gradient clipping for stability
            }
        })

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'mse']
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
                patience=10,
                min_lr=1e-7
            ),
            keras.callbacks.TerminateOnNaN(),
            EpochVisualizationCallback(
                val_data, self.processor, self.config,
                model_type, horizon, epoch_plot_dir
            )
        ]

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=2
        )

        return {"history": history.history, "model": model}

    def evaluate_model(
        self,
        model: NBeatsNet,
        test_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str
    ) -> Dict[str, ForecastMetrics]:
        """Comprehensive evaluation of N-BEATS model.

        Args:
            model: Trained N-BEATS model.
            test_data: Test data for all tasks.
            horizon: Forecast horizon length.
            model_type: Type of N-BEATS model.

        Returns:
            Dictionary mapping task names to their forecast metrics.
        """
        task_metrics = {}

        for task_name, (X_test, y_test) in test_data.items():
            if len(X_test) == 0:
                continue

            # Generate predictions
            predictions = model.predict(X_test, verbose=0)

            # Transform back to original scale
            pred_orig = self.processor.inverse_transform_data(task_name, predictions)
            y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

            # Get training series for MASE calculation
            raw_train_series = self.raw_train_data[task_name]

            # Calculate comprehensive metrics
            metrics = self._calculate_forecast_metrics(
                y_test_orig, pred_orig, raw_train_series,
                task_name, model_type, horizon
            )

            task_metrics[task_name] = metrics

            logger.info(
                f"Task {task_name}: RMSE={metrics.rmse:.4f}, "
                f"MASE={metrics.mase:.4f}, Coverage_90={metrics.coverage_90:.4f}"
            )

        return task_metrics

    def _calculate_forecast_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        train_series: np.ndarray,
        task_name: str,
        model_type: str,
        horizon: int
    ) -> ForecastMetrics:
        """Calculate comprehensive forecasting metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            train_series: Training series for MASE calculation.
            task_name: Name of the task.
            model_type: Type of model.
            horizon: Forecast horizon.

        Returns:
            ForecastMetrics object containing all calculated metrics.
        """
        # Get task category from generator
        task_category = "unknown"
        for category in self.task_categories:
            tasks_in_category = self.generator.get_tasks_by_category(category)
            if task_name in tasks_in_category:
                task_category = category
                break

        # Flatten arrays for metric calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        errors = y_true_flat - y_pred_flat

        # Basic metrics
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))

        # MAPE (avoiding division by zero)
        non_zero_mask = np.abs(y_true_flat) > 1e-8
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / y_true_flat[non_zero_mask])) * 100
        else:
            mape = 0.0

        # SMAPE
        smape_denom = (np.abs(y_true_flat) + np.abs(y_pred_flat))
        smape = np.mean(2 * np.abs(errors) / (smape_denom + 1e-8)) * 100

        # MASE (Mean Absolute Scaled Error)
        if len(train_series) > 1:
            mae_naive_train = np.mean(np.abs(np.diff(train_series.flatten())))
            mase = mae / (mae_naive_train + 1e-8)
        else:
            mase = np.inf

        # Directional accuracy
        y_true_reshaped = y_true.reshape(-1, horizon)
        y_pred_reshaped = y_pred.reshape(-1, horizon)

        if y_true_reshaped.shape[1] > 1:
            y_true_diff = np.diff(y_true_reshaped, axis=1)
            y_pred_diff = np.diff(y_pred_reshaped, axis=1)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        else:
            directional_accuracy = 0.0

        # Prediction intervals using empirical residuals
        forecast_residuals = (y_pred - y_true).flatten()
        q_lower = np.quantile(forecast_residuals, q=0.05)
        q_upper = np.quantile(forecast_residuals, q=0.95)

        # Interval width
        interval_width_90 = np.abs(q_upper - q_lower)

        # Coverage calculation
        lower_bound = y_pred_flat + q_lower
        upper_bound = y_pred_flat + q_upper
        coverage_90 = np.mean(
            (y_true_flat >= lower_bound) & (y_true_flat <= upper_bound)
        )

        return ForecastMetrics(
            task_name=task_name,
            task_category=task_category,
            model_type=model_type,
            horizon=horizon,
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mae,
            mape=mape,
            smape=smape,
            mase=mase,
            directional_accuracy=directional_accuracy,
            coverage_90=coverage_90,
            interval_width_90=interval_width_90,
            forecast_bias=np.mean(errors),
            samples_count=len(y_true_flat)
        )

    def plot_final_forecasts(
        self,
        models: Dict[int, Dict[str, NBeatsNet]],
        test_data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        save_dir: str
    ) -> None:
        """Create visualizations of forecast vs. actuals with prediction intervals.

        Args:
            models: Dictionary of trained models for each horizon and type.
            test_data: Test data for all tasks and horizons.
            save_dir: Directory to save visualizations.
        """
        logger.info("Creating final N-BEATS forecast visualizations with prediction intervals...")

        plot_dir = os.path.join(save_dir, 'final_forecasts')
        os.makedirs(plot_dir, exist_ok=True)

        for category in self.task_categories:
            # Get tasks for this category
            category_tasks = self.generator.get_tasks_by_category(category)

            for horizon in self.config.forecast_horizons:
                # Select tasks to plot
                num_tasks = min(4, len(category_tasks))
                if num_tasks == 0:
                    continue

                plot_tasks = self.random_state.choice(
                    category_tasks, size=num_tasks, replace=False
                )

                # Create subplot grid
                fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
                axes = axes.flatten()
                fig.suptitle(
                    f'Final Forecasts - {category.title()} (Horizon {horizon})',
                    fontsize=16
                )

                for i, task_name in enumerate(plot_tasks):
                    ax = axes[i]

                    # Check if test data exists for this task
                    if task_name not in test_data[horizon]:
                        ax.set_title(f'{task_name.replace("_", " ").title()} (No test data)')
                        continue

                    X_test, y_test = test_data[horizon][task_name]
                    if len(X_test) == 0:
                        ax.set_title(f'{task_name.replace("_", " ").title()} (No test data)')
                        continue

                    # Transform test data back to original scale
                    y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

                    # Select random sample for visualization
                    sample_idx = self.random_state.choice(len(X_test))

                    # Create time axes
                    time_backcast = np.arange(-self.config.backcast_length, 0)
                    time_forecast = np.arange(
                        self.config.backcast_length,
                        self.config.backcast_length + horizon
                    )

                    # Plot true values
                    ax.plot(
                        time_forecast, y_test_orig[sample_idx].flatten(),
                        'o-', color='blue', label='True'
                    )

                    # Plot predictions for each model type
                    colors = {'interpretable': 'red', 'generic': 'green', 'hybrid': 'purple'}

                    for model_type, model in models[horizon].items():
                        # Get prediction intervals based on all test residuals
                        all_preds_scaled = model.predict(X_test, verbose=0)
                        residuals = y_test - all_preds_scaled
                        q_lower = np.quantile(residuals.flatten(), q=0.05)
                        q_upper = np.quantile(residuals.flatten(), q=0.95)

                        # Get prediction for selected sample
                        pred_sample_scaled = all_preds_scaled[sample_idx]
                        pred_sample_orig = self.processor.inverse_transform_data(
                            task_name, pred_sample_scaled
                        )

                        # Calculate prediction intervals
                        lower_bound = self.processor.inverse_transform_data(
                            task_name, pred_sample_scaled + q_lower
                        )
                        upper_bound = self.processor.inverse_transform_data(
                            task_name, pred_sample_scaled + q_upper
                        )

                        # Plot forecast and prediction intervals
                        color = colors.get(model_type, 'gray')
                        ax.plot(
                            time_forecast, pred_sample_orig.flatten(),
                            '--', color=color,
                            label=f'{model_type.title()} Forecast'
                        )
                        ax.fill_between(
                            time_forecast, lower_bound.flatten(), upper_bound.flatten(),
                            color=color, alpha=0.2,
                            label=f'{model_type.title()} 90% PI'
                        )

                    # Format subplot
                    ax.set_title(f'{task_name.replace("_", " ").title()}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                # Hide unused subplots
                for j in range(len(plot_tasks), 4):
                    axes[j].set_visible(False)

                # Save plot
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(
                    os.path.join(plot_dir, f'forecast_{category}_h{horizon}.png'),
                    dpi=150
                )
                plt.close(fig)

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        model_type: str,
        horizon: int,
        save_dir: str
    ) -> None:
        """Plot and save training and validation loss/metrics.

        Args:
            history: Training history dictionary.
            model_type: Type of model.
            horizon: Forecast horizon.
            save_dir: Directory to save plots.
        """
        plot_dir = os.path.join(save_dir, 'training_history')
        os.makedirs(plot_dir, exist_ok=True)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(
            f'Training History for {model_type.title()} Model (Horizon {horizon})',
            fontsize=16
        )

        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MAE if available
        if 'mae' in history and 'val_mae' in history:
            ax2.plot(history['mae'], label='Training MAE')
            ax2.plot(history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.legend()
            ax2.grid(True)

        # Save plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            os.path.join(plot_dir, f'history_{model_type}_h{horizon}.png')
        )
        plt.close(fig)

    def plot_error_analysis(
        self,
        models: Dict[int, Dict[str, NBeatsNet]],
        test_data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        save_dir: str
    ) -> None:
        """Create and save detailed error analysis plots.

        Args:
            models: Dictionary of trained models.
            test_data: Test data for all tasks and horizons.
            save_dir: Directory to save plots.
        """
        logger.info("Creating error analysis visualizations...")

        plot_dir = os.path.join(save_dir, 'error_analysis')
        os.makedirs(plot_dir, exist_ok=True)

        # Select representative tasks for analysis
        tasks_to_plot = [
            'linear_trend_strong', 'multi_seasonal',
            'arma_process', 'level_shift'
        ]

        for model_type in self.config.model_types:
            for horizon in self.config.forecast_horizons:
                model = models[horizon][model_type]

                for task_name in tasks_to_plot:
                    # Check if task data exists
                    if task_name not in test_data[horizon]:
                        continue

                    X_test, y_test = test_data[horizon][task_name]
                    if len(X_test) == 0:
                        continue

                    # Generate predictions and transform to original scale
                    preds_scaled = model.predict(X_test, verbose=0)
                    preds_orig = self.processor.inverse_transform_data(
                        task_name, preds_scaled
                    )
                    y_test_orig = self.processor.inverse_transform_data(
                        task_name, y_test
                    )

                    # Calculate errors
                    errors = y_test_orig - preds_orig

                    # Create error analysis plots
                    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
                    fig.suptitle(
                        f'Error Analysis: {model_type.title()} H={horizon} '
                        f'on {task_name.replace("_", " ").title()}',
                        fontsize=16
                    )

                    # 1. Residuals histogram
                    axes[0, 0].hist(errors.flatten(), bins=50, density=True)
                    axes[0, 0].set_title('Distribution of Forecast Errors (Residuals)')
                    axes[0, 0].set_xlabel('Error')
                    axes[0, 0].set_ylabel('Density')
                    axes[0, 0].grid(True, alpha=0.5)

                    # 2. Residuals vs. predictions
                    axes[0, 1].scatter(preds_orig.flatten(), errors.flatten(), alpha=0.3)
                    axes[0, 1].axhline(0, color='red', linestyle='--')
                    axes[0, 1].set_title('Residuals vs. Predicted Values')
                    axes[0, 1].set_xlabel('Predicted Value')
                    axes[0, 1].set_ylabel('Error')
                    axes[0, 1].grid(True, alpha=0.5)

                    # 3. MAE per forecast step
                    mae_per_step = np.mean(np.abs(errors), axis=0).flatten()
                    axes[1, 0].bar(range(1, horizon + 1), mae_per_step)
                    axes[1, 0].set_title('MAE per Forecast Step')
                    axes[1, 0].set_xlabel('Forecast Horizon Step')
                    axes[1, 0].set_ylabel('Mean Absolute Error')
                    axes[1, 0].grid(True, axis='y', alpha=0.5)
                    axes[1, 0].set_xticks(range(1, horizon + 1))

                    # 4. Actual vs. predicted scatter
                    axes[1, 1].scatter(
                        y_test_orig.flatten(), preds_orig.flatten(), alpha=0.3
                    )

                    # Add diagonal reference line
                    lims = [
                        np.min([axes[1,1].get_xlim(), axes[1,1].get_ylim()]),
                        np.max([axes[1,1].get_xlim(), axes[1,1].get_ylim()])
                    ]
                    axes[1, 1].plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x')
                    axes[1, 1].set_title('Actual vs. Predicted Values')
                    axes[1, 1].set_xlabel('Actual Value')
                    axes[1, 1].set_ylabel('Predicted Value')
                    axes[1, 1].grid(True, alpha=0.5)
                    axes[1, 1].legend()

                    # Save plot
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(
                        os.path.join(
                            plot_dir,
                            f'errors_{model_type}_h{horizon}_{task_name}.png'
                        )
                    )
                    plt.close(fig)

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete N-BEATS experiment.

        Returns:
            Dictionary containing experiment results and metrics.
        """
        # Create experiment directory
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)

        logger.info(f"Starting N-BEATS experiment: {exp_dir}")

        # Prepare data
        prepared_data = self.prepare_data()

        # Initialize storage for results
        trained_models: Dict[int, Dict[str, NBeatsNet]] = {}
        all_metrics: Dict[int, Dict[str, Dict[str, ForecastMetrics]]] = {}
        all_histories: Dict[int, Dict[str, Dict[str, List[float]]]] = {}

        # Train models for each horizon and type
        for horizon in self.config.forecast_horizons:
            trained_models[horizon] = {}
            all_metrics[horizon] = {}
            all_histories[horizon] = {}

            for model_type in self.config.model_types:
                logger.info(
                    f"\n{'='*60}\n"
                    f"Training {model_type} model for horizon {horizon}\n"
                    f"{'='*60}"
                )

                # Create model
                model = self.create_model(model_type, horizon)

                # Extract data for current horizon
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

                # Save model if requested
                if self.config.save_results:
                    model_path = os.path.join(exp_dir, f"{model_type}_h{horizon}.keras")
                    trained_model.save(model_path)

        # Generate visualizations and summaries
        if self.config.save_results:
            visuals_dir = os.path.join(exp_dir, 'visuals')

            # Plot training histories
            for horizon, histories in all_histories.items():
                for model_type, history in histories.items():
                    self.plot_training_history(
                        history, model_type, horizon, visuals_dir
                    )

            # Plot final forecasts
            self.plot_final_forecasts(trained_models, prepared_data, visuals_dir)

            # Plot error analysis
            self.plot_error_analysis(trained_models, prepared_data, visuals_dir)

            # Generate results summary
            self._generate_results_summary(all_metrics, exp_dir)

        logger.info(f"Experiment completed. Results saved to: {exp_dir}")

        return {"results_dir": exp_dir, "metrics": all_metrics}

    def _generate_results_summary(
        self,
        all_metrics: Dict[int, Dict[str, Dict[str, ForecastMetrics]]],
        exp_dir: str
    ) -> None:
        """Generate and save summary dataframes from the metrics.

        Args:
            all_metrics: Dictionary containing all calculated metrics.
            exp_dir: Experiment directory for saving results.
        """
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

        # Log summary information
        logger.info(
            "=" * 120 + "\n"
            "N-BEATS MULTI-TASK FORECASTING RESULTS\n" +
            "=" * 120
        )
        logger.info(f"Detailed Results (Sample):\n{results_df.head().to_string()}")

        # Summary by model type and horizon
        summary_cols = ['rmse', 'mae', 'smape', 'mase', 'coverage_90']
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

        # Save results to CSV files
        results_df.to_csv(os.path.join(exp_dir, 'detailed_results.csv'), index=False)
        summary_by_model.to_csv(os.path.join(exp_dir, 'summary_by_model.csv'))
        summary_by_category.to_csv(os.path.join(exp_dir, 'summary_by_category.csv'))

        logger.info(f"Results summaries saved to {exp_dir}")


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main() -> None:
    """Main function to run the N-BEATS multi-task experiment.

    Raises:
        Exception: If experiment fails with detailed error information.
    """
    # Configuration for N-BEATS experiment
    config = NBeatsConfig()

    # Configuration for time series generation
    ts_config = TimeSeriesConfig(
        n_samples=8000,
        random_seed=42,
        default_noise_level=0.1
    )

    logger.info(
        "Starting N-BEATS multi-task forecasting experiment using Time Series Generator"
    )

    try:
        trainer = NBeatsTrainer(config, ts_config)
        trainer.run_experiment()
        logger.info("Experiment finished successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()