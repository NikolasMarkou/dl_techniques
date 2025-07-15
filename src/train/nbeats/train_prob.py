"""
Multi-Task Probabilistic Time Series Forecasting with N-BEATS + MDN

This implementation demonstrates a comprehensive multi-task learning approach using
Probabilistic N-BEATS models that combine the interpretable structure of N-BEATS
with the uncertainty quantification capabilities of Mixture Density Networks.

Cleaned and simplified version based on the classic N-BEATS structure.
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
from dl_techniques.models.nbeats_mdn import ProbabilisticNBeatsNet
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.utils.datasets.time_series_generator import (
    TimeSeriesGenerator,
    TimeSeriesConfig
)

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
    """Configuration class for Probabilistic N-BEATS forecasting experiments.

    This configuration extends the classic N-BEATS config with probabilistic modeling
    parameters for uncertainty quantification.

    Attributes:
        result_dir: Base directory for saving experiment results.
        save_results: Whether to save models and visualizations.
        experiment_name: Name identifier for the experiment.

        # Data configuration
        train_ratio: Proportion of data used for training.
        val_ratio: Proportion of data used for validation.
        test_ratio: Proportion of data used for testing.

        # N-BEATS configuration
        backcast_length: Length of input sequence (lookback window).
        forecast_length: Length of forecast horizon.
        forecast_horizons: List of forecast horizons to evaluate.

        # Model architectures
        model_types: List of Probabilistic N-BEATS model architectures to test.
        stack_types: Dictionary mapping model types to their stack configurations.
        nb_blocks_per_stack: Number of blocks per stack in N-BEATS.
        thetas_dim: Theta dimensions for each model type.
        hidden_layer_units: Number of hidden units in each layer.
        share_weights_in_stack: Whether to share weights within stacks.

        # Probabilistic parameters
        num_mixtures: Number of Gaussian mixtures in MDN layer.
        mdn_hidden_units: Hidden units in MDN preprocessing layer.
        aggregation_mode: How to aggregate N-BEATS block outputs.

        # Training configuration
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        early_stopping_patience: Early stopping patience in epochs.
        learning_rate: Initial learning rate for optimizer.
        optimizer: Optimizer type to use.

        # Evaluation configuration
        confidence_levels: Confidence levels for prediction intervals.
        num_samples_uncertainty: Number of samples for uncertainty estimation.

        # Visualization configuration
        plot_samples: Number of samples to plot in visualizations.
        epoch_plot_freq: Frequency of epoch plotting for visualization callback.
    """

    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "probabilistic_nbeats_multitask"

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS configuration
    backcast_length: int = 96
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24])

    # Model architectures
    model_types: List[str] = field(default_factory=lambda: ["interpretable", "generic", "hybrid"])
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

    # Probabilistic parameters
    num_mixtures: int = 3
    mdn_hidden_units: int = 128
    aggregation_mode: str = 'concat'

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    early_stopping_patience: int = 50
    learning_rate: float = 5e-5
    optimizer: str = 'adamw'

    # Evaluation configuration
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.90, 0.95])
    num_samples_uncertainty: int = 100

    # Visualization configuration
    plot_samples: int = 3
    epoch_plot_freq: int = 10


@dataclass
class ProbabilisticForecastMetrics:
    """Probabilistic forecasting metrics container.

    Attributes:
        task_name: Name of the forecasting task.
        task_category: Category of the task.
        model_type: Type of N-BEATS model used.
        horizon: Forecast horizon length.

        # Point forecast metrics
        mse: Mean Squared Error.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        mape: Mean Absolute Percentage Error.
        smape: Symmetric Mean Absolute Percentage Error.
        mase: Mean Absolute Scaled Error.
        directional_accuracy: Directional accuracy of forecasts.

        # Probabilistic metrics
        log_likelihood: Average log-likelihood of observations.
        crps: Continuous Ranked Probability Score.
        coverage_68: Coverage of 68% prediction intervals.
        coverage_90: Coverage of 90% prediction intervals.
        coverage_95: Coverage of 95% prediction intervals.
        interval_width_90: Average width of 90% intervals.
        calibration_error: Calibration error of prediction intervals.

        # Uncertainty measures
        predictive_variance: Average predictive variance.
        forecast_bias: Bias in forecasts.
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
    smape: float
    mase: float
    directional_accuracy: float

    # Probabilistic metrics
    log_likelihood: float
    crps: float
    coverage_68: float
    coverage_90: float
    coverage_95: float
    interval_width_90: float
    calibration_error: float

    # Uncertainty measures
    predictive_variance: float
    forecast_bias: float
    samples_count: int


# ---------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------

class ProbabilisticNBeatsDataProcessor:
    """Data processor for Probabilistic N-BEATS with consistent scaling.

    This class handles data preprocessing including normalization and
    sequence creation for Probabilistic N-BEATS training.

    Args:
        config: Configuration object containing experiment settings.
    """

    def __init__(self, config: ProbabilisticNBeatsConfig) -> None:
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}

    def create_sequences(
        self,
        data: np.ndarray,
        backcast_length: int,
        forecast_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create N-BEATS sequences from time series data.

        Args:
            data: Time series data.
            backcast_length: Length of input sequence.
            forecast_length: Length of forecast sequence.

        Returns:
            Tuple of (backcast_sequences, forecast_sequences).
        """
        X, y = [], []

        for i in range(len(data) - backcast_length - forecast_length + 1):
            backcast = data[i : i + backcast_length]
            forecast = data[i + backcast_length : i + backcast_length + forecast_length]

            X.append(backcast)
            y.append(forecast)

        return np.array(X), np.array(y)

    def fit_scalers(self, task_data: Dict[str, np.ndarray]) -> None:
        """Fit normalizers for all tasks.

        Args:
            task_data: Dictionary mapping task names to their time series data.
        """
        for task_name, data in task_data.items():
            scaler = TimeSeriesNormalizer(method='minmax', feature_range=(0, 1))

            # Fit on training portion only
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            scaler.fit(train_data)
            self.scalers[task_name] = scaler

            logger.info(f"Fitted scaler for {task_name}")

    def transform_data(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler.

        Args:
            task_name: Name of the task.
            data: Data to transform.

        Returns:
            Transformed data.
        """
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")

        return self.scalers[task_name].transform(data)

    def inverse_transform_data(self, task_name: str, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform data using fitted scaler.

        Args:
            task_name: Name of the task.
            scaled_data: Scaled data to inverse transform.

        Returns:
            Data in original scale.
        """
        if task_name not in self.scalers:
            raise ValueError(f"Scaler not fitted for task: {task_name}")

        return self.scalers[task_name].inverse_transform(scaled_data)


# ---------------------------------------------------------------------
# Visualization Callback
# ---------------------------------------------------------------------

class ProbabilisticVisualizationCallback(keras.callbacks.Callback):
    """Keras callback for visualizing probabilistic forecasts during training.

    Args:
        val_data_dict: Dictionary containing validation data for each task.
        processor: Data processor instance.
        config: Configuration object.
        model_type: Type of model being trained.
        horizon: Forecast horizon length.
        save_dir: Directory to save plots.
    """

    def __init__(
        self,
        val_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        processor: ProbabilisticNBeatsDataProcessor,
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
        """Initialize callback at training start."""
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

        # Fallback to any available tasks
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

        logger.info(f"Visualization callback initialized for tasks: {list(self.plot_indices.keys())}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Create probabilistic forecast visualizations."""
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
            if i >= len(axes):
                break

            ax = axes[i]
            sample_idx = self.plot_indices[task_name]
            X_val, y_val = self.val_data[task_name]

            # Get sample data
            x_sample = X_val[np.newaxis, sample_idx]
            y_sample_true = y_val[sample_idx]

            # Generate probabilistic prediction
            try:
                predictions = self.model.predict_probabilistic(
                    x_sample,
                    num_samples=50,
                    return_components=False
                )

                point_estimate = predictions['point_estimate'][0]
                total_variance = predictions['total_variance'][0]

                # Transform to original scale
                point_estimate_orig = self.processor.inverse_transform_data(
                    task_name, point_estimate
                )
                y_sample_true_orig = self.processor.inverse_transform_data(
                    task_name, y_sample_true
                )
                x_sample_orig = self.processor.inverse_transform_data(
                    task_name, x_sample[0]
                )

                # Calculate confidence intervals
                std_dev = np.sqrt(total_variance)
                lower_68 = self.processor.inverse_transform_data(
                    task_name, point_estimate - std_dev
                )
                upper_68 = self.processor.inverse_transform_data(
                    task_name, point_estimate + std_dev
                )
                lower_95 = self.processor.inverse_transform_data(
                    task_name, point_estimate - 1.96 * std_dev
                )
                upper_95 = self.processor.inverse_transform_data(
                    task_name, point_estimate + 1.96 * std_dev
                )

                # Create time axes
                backcast_time = np.arange(-self.config.backcast_length, 0)
                forecast_time = np.arange(0, self.horizon)

                # Plot data
                ax.plot(
                    backcast_time, x_sample_orig.flatten(),
                    color='gray', label='Historical', alpha=0.7
                )
                ax.plot(
                    forecast_time, y_sample_true_orig.flatten(),
                    'o-', color='blue', label='True Future', linewidth=2
                )
                ax.plot(
                    forecast_time, point_estimate_orig.flatten(),
                    'x--', color='red', label='Point Forecast', linewidth=2
                )

                # Plot uncertainty bands
                ax.fill_between(
                    forecast_time, lower_95.flatten(), upper_95.flatten(),
                    color='red', alpha=0.2, label='95% Confidence'
                )
                ax.fill_between(
                    forecast_time, lower_68.flatten(), upper_68.flatten(),
                    color='red', alpha=0.4, label='68% Confidence'
                )

            except Exception as e:
                logger.warning(f"Failed to generate prediction for {task_name}: {e}")
                # Plot only historical and true data
                ax.plot(
                    np.arange(-self.config.backcast_length, 0),
                    x_sample[0].flatten(),
                    color='gray', label='Historical', alpha=0.7
                )
                ax.plot(
                    np.arange(0, self.horizon),
                    y_sample_true.flatten(),
                    'o-', color='blue', label='True Future', linewidth=2
                )

            # Format subplot
            ax.set_title(f'Task: {task_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

        # Hide unused subplots
        for j in range(len(self.plot_indices), len(axes)):
            axes[j].set_visible(False)

        # Save visualization
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}_probabilistic.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------------
# Probabilistic N-BEATS Trainer
# ---------------------------------------------------------------------

class ProbabilisticNBeatsTrainer:
    """Comprehensive trainer for Probabilistic N-BEATS multi-task forecasting.

    This trainer follows the same clean structure as the classic N-BEATS trainer
    but includes probabilistic modeling capabilities.

    Args:
        config: Configuration object containing experiment settings.
        ts_config: Time series generator configuration.
    """

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
        """Prepare multi-task data for all forecast horizons.

        Returns:
            Nested dictionary: {horizon: {task_name: {split: (X, y)}}}
        """
        logger.info("Preparing Probabilistic N-BEATS multi-task data...")

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
        """Create Probabilistic N-BEATS model.

        Args:
            model_type: Type of model ('interpretable', 'generic', 'hybrid').
            forecast_length: Length of forecast horizon.

        Returns:
            Configured Probabilistic N-BEATS model.
        """
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
            output_dim=forecast_length
        )
        return model

    def _create_mdn_point_estimate_metric(self, model: ProbabilisticNBeatsNet):
        """Create a custom metric that computes MAE on point estimates from MDN output."""
        def mdn_point_mae(y_true, y_pred):
            """Calculate MAE between true values and MDN point estimates."""
            # Extract mixture parameters
            mu, sigma, pi_logits = model.mdn_layer.split_mixture_params(y_pred)

            # Convert to probabilities
            pi = keras.activations.softmax(pi_logits, axis=-1)

            # Calculate point estimate (weighted mean)
            pi_expanded = keras.ops.expand_dims(pi, axis=-1)
            point_estimate = keras.ops.sum(pi_expanded * mu, axis=1)

            # Calculate MAE
            return keras.ops.mean(keras.ops.abs(y_true - point_estimate))

        return mdn_point_mae

    def train_model(
        self,
        model: ProbabilisticNBeatsNet,
        train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        val_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str,
        exp_dir: str
    ) -> Dict[str, Any]:
        """Train Probabilistic N-BEATS model.

        Args:
            model: Model to train.
            train_data: Training data for all tasks.
            val_data: Validation data for all tasks.
            horizon: Forecast horizon.
            model_type: Type of model.
            exp_dir: Experiment directory.

        Returns:
            Training results dictionary.
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

        # Ensure y data has correct shape [batch, forecast_length]
        if len(y_train.shape) == 3 and y_train.shape[-1] == 1:
            y_train = np.squeeze(y_train, axis=-1)
        if len(y_val.shape) == 3 and y_val.shape[-1] == 1:
            y_val = np.squeeze(y_val, axis=-1)

        # Shuffle training data
        p_train = self.random_state.permutation(len(X_train))
        X_train, y_train = X_train[p_train], y_train[p_train]

        logger.info(f"Training data: {X_train.shape}, validation: {X_val.shape}")
        logger.info(f"Target shape: y_train={y_train.shape}, y_val={y_val.shape}")

        # Debug: Test model prediction to check output shape
        test_pred = model.predict(X_train[:1], verbose=0)
        logger.info(f"Model output shape: {test_pred.shape}")
        expected_mdn_size = (2 * model.num_mixtures * model.output_dim) + model.num_mixtures
        logger.info(f"Expected MDN output size: {expected_mdn_size}")

        # Configure optimizer
        optimizer = keras.optimizers.get({
            "class_name": self.config.optimizer,
            "config": {
                "learning_rate": self.config.learning_rate,
                "clipnorm": 1.0
            }
        })

        # Create custom metric for point estimate MAE
        point_mae_metric = self._create_mdn_point_estimate_metric(model)

        # Compile model with MDN loss and custom metric
        model.compile(
            optimizer=optimizer,
            loss=model.mdn_loss,
            metrics=[point_mae_metric]
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
                factor=0.7,
                patience=15,
                min_lr=1e-8
            ),
            keras.callbacks.TerminateOnNaN(),
            ProbabilisticVisualizationCallback(
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
        model: ProbabilisticNBeatsNet,
        test_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str
    ) -> Dict[str, ProbabilisticForecastMetrics]:
        """Evaluate Probabilistic N-BEATS model.

        Args:
            model: Trained model.
            test_data: Test data for all tasks.
            horizon: Forecast horizon.
            model_type: Type of model.

        Returns:
            Dictionary of metrics for each task.
        """
        task_metrics = {}

        for task_name, (X_test, y_test) in test_data.items():
            if len(X_test) == 0:
                continue

            # Ensure y_test has correct shape
            if len(y_test.shape) == 3 and y_test.shape[-1] == 1:
                y_test = np.squeeze(y_test, axis=-1)

            # Generate probabilistic predictions
            predictions = model.predict_probabilistic(
                X_test,
                num_samples=self.config.num_samples_uncertainty,
                return_components=True
            )

            # Transform back to original scale
            point_estimates_orig = self.processor.inverse_transform_data(
                task_name, predictions['point_estimate']
            )
            y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

            # Calculate metrics
            metrics = self._calculate_probabilistic_metrics(
                y_test_orig, point_estimates_orig, predictions,
                task_name, model_type, horizon
            )

            task_metrics[task_name] = metrics

            logger.info(
                f"Task {task_name}: RMSE={metrics.rmse:.4f}, "
                f"Coverage_90={metrics.coverage_90:.4f}"
            )

        return task_metrics

    def _get_task_category(self, task_name: str) -> str:
        """Get task category safely."""
        try:
            for category in self.task_categories:
                tasks_in_category = self.generator.get_tasks_by_category(category)
                if task_name in tasks_in_category:
                    return category
        except (AttributeError, KeyError):
            pass
        return 'unknown'

    def _calculate_probabilistic_metrics(
        self,
        y_true: np.ndarray,
        point_estimates: np.ndarray,
        predictions: Dict[str, np.ndarray],
        task_name: str,
        model_type: str,
        horizon: int
    ) -> ProbabilisticForecastMetrics:
        """Calculate comprehensive probabilistic metrics.

        Args:
            y_true: True values in original scale.
            point_estimates: Point estimates in original scale.
            predictions: Dictionary of predictions from model.
            task_name: Name of the task.
            model_type: Type of model.
            horizon: Forecast horizon.

        Returns:
            ProbabilisticForecastMetrics object.
        """
        # Flatten arrays
        y_true_flat = y_true.flatten()
        point_flat = point_estimates.flatten()
        errors = y_true_flat - point_flat

        # Basic point forecast metrics
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))

        # MAPE
        non_zero_mask = np.abs(y_true_flat) > 1e-8
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / y_true_flat[non_zero_mask])) * 100
        else:
            mape = 0.0

        # SMAPE
        smape_denom = (np.abs(y_true_flat) + np.abs(point_flat))
        smape = np.mean(2 * np.abs(errors) / (smape_denom + 1e-8)) * 100

        # MASE
        raw_train_series = self.raw_train_data[task_name]
        if len(raw_train_series) > 1:
            mae_naive_train = np.mean(np.abs(np.diff(raw_train_series.flatten())))
            mase = mae / (mae_naive_train + 1e-8)
        else:
            mase = np.inf

        # Directional accuracy
        if horizon > 1:
            y_true_reshaped = y_true.reshape(-1, horizon)
            point_reshaped = point_estimates.reshape(-1, horizon)
            y_true_diff = np.diff(y_true_reshaped, axis=1)
            point_diff = np.diff(point_reshaped, axis=1)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(point_diff))
        else:
            directional_accuracy = 0.0

        # Probabilistic metrics using samples
        samples = predictions.get('samples', np.expand_dims(point_estimates, axis=1))
        if len(samples.shape) == 3:
            samples_flat = samples.reshape(len(y_true_flat), -1)
        else:
            samples_flat = samples

        # Coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(y_true_flat, samples_flat)

        # CRPS (simplified)
        crps = self._calculate_crps(y_true_flat, samples_flat)

        # Predictive variance
        predictive_variance = np.mean(predictions.get('total_variance', 0.0))

        return ProbabilisticForecastMetrics(
            task_name=task_name,
            task_category=self._get_task_category(task_name),
            model_type=model_type,
            horizon=horizon,

            # Point forecast metrics
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mae,
            mape=mape,
            smape=smape,
            mase=mase,
            directional_accuracy=directional_accuracy,

            # Probabilistic metrics
            log_likelihood=0.0,  # Placeholder
            crps=crps,
            coverage_68=coverage_metrics['coverage_68'],
            coverage_90=coverage_metrics['coverage_90'],
            coverage_95=coverage_metrics['coverage_95'],
            interval_width_90=coverage_metrics['width_90'],
            calibration_error=coverage_metrics['calibration_error'],

            # Uncertainty measures
            predictive_variance=predictive_variance,
            forecast_bias=np.mean(errors),
            samples_count=len(y_true_flat)
        )

    def _calculate_crps(self, observations: np.ndarray, samples: np.ndarray) -> float:
        """Calculate Continuous Ranked Probability Score."""
        crps_values = []

        for i, obs in enumerate(observations):
            if i >= len(samples):
                break
            sample_set = samples[i]
            sorted_samples = np.sort(sample_set)
            n_samples = len(sorted_samples)

            # Simplified CRPS calculation
            crps_val = 0.0
            for j, sample in enumerate(sorted_samples):
                if obs <= sample:
                    crps_val += (obs - sample) * (2 * j / n_samples - 1)
                else:
                    crps_val += (sample - obs) * (2 * j / n_samples - 1)

            crps_values.append(crps_val / n_samples)

        return np.mean(crps_values) if crps_values else 0.0

    def _calculate_coverage_metrics(
        self,
        observations: np.ndarray,
        samples: np.ndarray
    ) -> Dict[str, float]:
        """Calculate coverage and interval metrics."""
        if len(samples) == 0:
            return {
                'coverage_68': 0.0,
                'coverage_90': 0.0,
                'coverage_95': 0.0,
                'width_90': 0.0,
                'calibration_error': 1.0
            }

        # Calculate quantiles
        q_16 = np.percentile(samples, 16, axis=1)
        q_84 = np.percentile(samples, 84, axis=1)
        q_5 = np.percentile(samples, 5, axis=1)
        q_95 = np.percentile(samples, 95, axis=1)
        q_2_5 = np.percentile(samples, 2.5, axis=1)
        q_97_5 = np.percentile(samples, 97.5, axis=1)

        # Coverage calculation
        coverage_68 = np.mean((observations >= q_16) & (observations <= q_84))
        coverage_90 = np.mean((observations >= q_5) & (observations <= q_95))
        coverage_95 = np.mean((observations >= q_2_5) & (observations <= q_97_5))

        # Interval width
        width_90 = np.mean(q_95 - q_5)

        # Calibration error
        calibration_error = abs(coverage_90 - 0.90)

        return {
            'coverage_68': coverage_68,
            'coverage_90': coverage_90,
            'coverage_95': coverage_95,
            'width_90': width_90,
            'calibration_error': calibration_error
        }

    def plot_final_forecasts(
        self,
        models: Dict[int, Dict[str, ProbabilisticNBeatsNet]],
        test_data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        save_dir: str
    ) -> None:
        """Create final forecast visualizations with uncertainty bands.

        Args:
            models: Dictionary of trained models.
            test_data: Test data for all tasks.
            save_dir: Directory to save plots.
        """
        logger.info("Creating final probabilistic forecast visualizations...")

        plot_dir = os.path.join(save_dir, 'final_forecasts')
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

                # Create subplot grid
                fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
                axes = axes.flatten()
                fig.suptitle(
                    f'Final Probabilistic Forecasts - {category.title()} (Horizon {horizon})',
                    fontsize=16
                )

                for i, task_name in enumerate(plot_tasks):
                    ax = axes[i]

                    if task_name not in test_data[horizon]:
                        ax.set_title(f'{task_name.replace("_", " ").title()} (No test data)')
                        continue

                    X_test, y_test = test_data[horizon][task_name]
                    if len(X_test) == 0:
                        ax.set_title(f'{task_name.replace("_", " ").title()} (No test data)')
                        continue

                    # Transform test data to original scale
                    y_test_orig = self.processor.inverse_transform_data(task_name, y_test)

                    # Select random sample
                    sample_idx = self.random_state.choice(len(X_test))
                    x_sample = X_test[np.newaxis, sample_idx]
                    y_sample_true = y_test_orig[sample_idx]

                    # Create time axes
                    backcast_time = np.arange(-self.config.backcast_length, 0)
                    forecast_time = np.arange(0, horizon)

                    # Plot true values
                    ax.plot(
                        forecast_time, y_sample_true.flatten(),
                        'o-', color='blue', label='True Future', linewidth=2
                    )

                    # Plot predictions for each model type
                    colors = {'interpretable': 'red', 'generic': 'green', 'hybrid': 'purple'}

                    for model_type, model in models[horizon].items():
                        try:
                            # Generate probabilistic predictions
                            predictions = model.predict_probabilistic(
                                x_sample,
                                num_samples=100,
                                return_components=False
                            )

                            # Transform to original scale
                            point_est_orig = self.processor.inverse_transform_data(
                                task_name, predictions['point_estimate'][0]
                            )

                            # Calculate confidence intervals
                            std_dev = np.sqrt(predictions['total_variance'][0])
                            lower_68_scaled = predictions['point_estimate'][0] - std_dev
                            upper_68_scaled = predictions['point_estimate'][0] + std_dev
                            lower_95_scaled = predictions['point_estimate'][0] - 1.96 * std_dev
                            upper_95_scaled = predictions['point_estimate'][0] + 1.96 * std_dev

                            # Transform to original scale
                            lower_68_orig = self.processor.inverse_transform_data(task_name, lower_68_scaled)
                            upper_68_orig = self.processor.inverse_transform_data(task_name, upper_68_scaled)
                            lower_95_orig = self.processor.inverse_transform_data(task_name, lower_95_scaled)
                            upper_95_orig = self.processor.inverse_transform_data(task_name, upper_95_scaled)

                            color = colors.get(model_type, 'gray')

                            # Plot point forecast
                            ax.plot(
                                forecast_time, point_est_orig.flatten(),
                                '--', color=color, linewidth=2,
                                label=f'{model_type.title()} Point'
                            )

                            # Plot uncertainty bands
                            ax.fill_between(
                                forecast_time, lower_95_orig.flatten(), upper_95_orig.flatten(),
                                color=color, alpha=0.2,
                                label=f'{model_type.title()} 95% PI'
                            )
                            ax.fill_between(
                                forecast_time, lower_68_orig.flatten(), upper_68_orig.flatten(),
                                color=color, alpha=0.4,
                                label=f'{model_type.title()} 68% PI'
                            )

                        except Exception as e:
                            logger.warning(f"Failed to generate prediction for {task_name} with {model_type}: {e}")

                    # Format subplot
                    ax.set_title(f'{task_name.replace("_", " ").title()}')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Value')

                # Hide unused subplots
                for j in range(len(plot_tasks), 4):
                    axes[j].set_visible(False)

                # Save plot
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(
                    os.path.join(plot_dir, f'probabilistic_{category}_h{horizon}.png'),
                    dpi=150
                )
                plt.close(fig)

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete Probabilistic N-BEATS experiment.

        Returns:
            Dictionary containing experiment results.
        """
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

        # Train models for each horizon and type
        for horizon in self.config.forecast_horizons:
            trained_models[horizon] = {}
            all_metrics[horizon] = {}
            all_histories[horizon] = {}

            for model_type in self.config.model_types:
                logger.info(
                    f"\n{'='*60}\n"
                    f"Training {model_type} Probabilistic N-BEATS for horizon {horizon}\n"
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
                    model_path = os.path.join(exp_dir, f"probabilistic_{model_type}_h{horizon}.keras")
                    trained_model.save(model_path)

        # Generate visualizations
        if self.config.save_results:
            visuals_dir = os.path.join(exp_dir, 'visuals')
            self.plot_final_forecasts(trained_models, prepared_data, visuals_dir)
            self._generate_results_summary(all_metrics, exp_dir)

        logger.info(f"Probabilistic N-BEATS experiment completed: {exp_dir}")
        return {"results_dir": exp_dir, "metrics": all_metrics}

    def _generate_results_summary(
        self,
        all_metrics: Dict[int, Dict[str, Dict[str, ProbabilisticForecastMetrics]]],
        exp_dir: str
    ) -> None:
        """Generate and save results summary.

        Args:
            all_metrics: Dictionary of all calculated metrics.
            exp_dir: Experiment directory.
        """
        # Convert metrics to list of dictionaries
        results_data = [
            dataclasses.asdict(metrics)
            for h_metrics in all_metrics.values()
            for m_metrics in h_metrics.values()
            for metrics in m_metrics.values()
        ]

        if not results_data:
            logger.warning("No results to summarize.")
            return

        # Create results dataframe
        results_df = pd.DataFrame(results_data)

        # Log summary
        logger.info(
            "=" * 120 + "\n"
            "PROBABILISTIC N-BEATS MULTI-TASK FORECASTING RESULTS\n" +
            "=" * 120
        )

        # Summary by model type and horizon
        summary_cols = [
            'rmse', 'mae', 'smape', 'mase', 'crps',
            'coverage_68', 'coverage_90', 'coverage_95',
            'interval_width_90', 'predictive_variance'
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
        results_df.to_csv(os.path.join(exp_dir, 'detailed_probabilistic_results.csv'), index=False)
        summary_by_model.to_csv(os.path.join(exp_dir, 'summary_by_model_probabilistic.csv'))
        summary_by_category.to_csv(os.path.join(exp_dir, 'summary_by_category_probabilistic.csv'))

        logger.info(f"Results summaries saved to {exp_dir}")


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main() -> None:
    """Main function to run the Probabilistic N-BEATS experiment."""
    # Configuration for Probabilistic N-BEATS
    config = ProbabilisticNBeatsConfig()

    # Configuration for time series generation
    ts_config = TimeSeriesConfig(
        n_samples=8000,
        random_seed=42,
        default_noise_level=0.1
    )

    logger.info("Starting Probabilistic N-BEATS multi-task forecasting experiment")

    try:
        trainer = ProbabilisticNBeatsTrainer(config, ts_config)
        trainer.run_experiment()
        logger.info("Probabilistic N-BEATS experiment finished successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()