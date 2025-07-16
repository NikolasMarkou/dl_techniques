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
    - Optional Hybrid Loss combining NLL and Point Estimate MAE for faster convergence
    - Prediction intervals with configurable confidence levels
    - Comprehensive evaluation with probabilistic metrics
    - Extensive visualization including uncertainty bands
    - Scalable data processing with consistent normalization for stable training
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
from scipy import stats

# Use a non-interactive backend for saving plots to files
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.models.nbeats_probabilistic import ProbabilisticNBeatsNet
from dl_techniques.utils.datasets.time_series_generator import (
    TimeSeriesGenerator,
    TimeSeriesConfig
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

RANDOM_SEED = 42
DEFAULT_AUGMENTATION_NOISE = 0.05
MIXTURE_COLLAPSE_THRESHOLD = 0.95
MIN_SAMPLES_FOR_PLOT = 10
CRPS_SAMPLE_LIMIT = 1000
PLOTTING_SAMPLE_LIMIT = 1000
MAX_CATEGORIES_TO_PLOT = 4
MIXTURE_ACTIVE_THRESHOLD = 0.1
EPSILON_LOG = 1e-8
EPSILON_ZERO_DIVISION = 1e-8

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)

# ---------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------

@dataclass
class ProbabilisticNBeatsConfig:
    """Configuration class for probabilistic N-BEATS forecasting experiments.

    This class contains all hyperparameters and settings needed for training
    and evaluating probabilistic N-BEATS models across multiple tasks, including
    support for hybrid loss functions.
    """

    # Experiment settings
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "probabilistic_nbeats_hybrid"

    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Model architecture parameters
    backcast_length: int = 96
    forecast_horizons: List[int] = field(default_factory=lambda: [12, 24])
    model_types: List[str] = field(default_factory=lambda: ["interpretable", "generic"])

    stack_types: Dict[str, List[str]] = field(default_factory=lambda: {
        "interpretable": ["trend", "seasonality"],
        "generic": ["generic", "generic"]
    })

    nb_blocks_per_stack: int = 2

    thetas_dim: Dict[str, List[int]] = field(default_factory=lambda: {
        "interpretable": [3, 6],
        "generic": [64, 64]
    })

    hidden_layer_units: int = 128
    share_weights_in_stack: bool = False

    # MDN parameters
    num_mixtures: int = 5
    mdn_hidden_units: int = 128
    aggregation_mode: str = "concat"
    diversity_regularizer_strength: float = 0.02
    min_sigma: float = 0.01

    # Training parameters
    epochs: int = 150
    batch_size: int = 128
    early_stopping_patience: int = 25
    learning_rate: float = 1e-3
    optimizer: str = 'adamw'
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-4

    # Uncertainty quantification parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.90, 0.95])
    num_prediction_samples: int = 1000

    # Visualization parameters
    epoch_plot_freq: int = 10

    # Hybrid loss configuration
    use_hybrid_loss: bool = True
    hybrid_loss_alpha: float = 0.7  # 70% MDN loss, 30% Point Estimate MAE


@dataclass
class ProbabilisticForecastMetrics:
    """Comprehensive probabilistic forecasting metrics container.

    This class stores all evaluation metrics for a single task/model/horizon
    combination, including point forecasting metrics, probabilistic metrics,
    and uncertainty decomposition.
    """

    # Task identification
    task_name: str
    task_category: str
    model_type: str
    horizon: int

    # Point forecasting metrics
    mse: float
    rmse: float
    mae: float
    mape: float

    # Probabilistic metrics
    crps: float
    log_likelihood: float

    # Coverage metrics
    coverage_68: float
    coverage_90: float
    coverage_95: float

    # Interval width metrics
    interval_width_68: float
    interval_width_90: float
    interval_width_95: float

    # Uncertainty decomposition
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float

    # Mixture analysis
    mean_num_active_mixtures: float
    mixture_entropy: float

    # Sample information
    samples_count: int


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

class MixtureMonitoringCallback(keras.callbacks.Callback):
    """Callback to monitor mixture weights and detect model collapse.

    This callback tracks the behavior of mixture components during training
    to detect potential mode collapse where one mixture dominates others.
    """

    def __init__(
        self,
        val_data_sample: np.ndarray,
        processor,
        task_name: str,
        save_dir: str
    ):
        """Initialize the mixture monitoring callback.

        Args:
            val_data_sample: Sample of validation data for monitoring
            processor: Data processor for transformations
            task_name: Name of the task being monitored
            save_dir: Directory to save monitoring results
        """
        super().__init__()
        self.val_data_sample = val_data_sample
        self.processor = processor
        self.task_name = task_name
        self.save_dir = save_dir

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Monitor mixture weights and sigma values at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Training logs (unused)
        """
        if (epoch + 1) % 5 != 0:
            return

        try:
            # Sample a subset of validation data for efficient monitoring
            sample_size = min(MIN_SAMPLES_FOR_PLOT, len(self.val_data_sample))
            X_sample = self.val_data_sample[:sample_size]

            # Get mixture parameters from the model
            mixture_params = self.model.predict(X_sample, verbose=0)
            mu, sigma, pi_logits = self.model.mdn_layer.split_mixture_params(mixture_params)

            # Convert to numpy arrays for analysis
            mu_np = keras.ops.convert_to_numpy(mu)
            sigma_np = keras.ops.convert_to_numpy(sigma)
            pi_weights = keras.ops.convert_to_numpy(
                keras.activations.softmax(pi_logits, axis=-1)
            )

            # Calculate mixture statistics
            mean_weights = np.mean(pi_weights, axis=0)
            max_weight = np.max(pi_weights)
            entropy = -np.sum(mean_weights * np.log(mean_weights + EPSILON_LOG))

            # Calculate sigma statistics
            mean_sigma = np.mean(sigma_np)
            min_sigma = np.min(sigma_np)
            max_sigma = np.max(sigma_np)

            # Log mixture and sigma information periodically
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"MixWeights={np.round(mean_weights, 3)}, "
                    f"MaxWeight={max_weight:.3f}, "
                    f"Entropy={entropy:.3f}"
                )
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Sigma - Mean={mean_sigma:.4f}, "
                    f"Min={min_sigma:.4f}, "
                    f"Max={max_sigma:.4f}"
                )

            # Warn about potential issues
            if max_weight > MIXTURE_COLLAPSE_THRESHOLD:
                logger.warning(
                    f"Epoch {epoch+1}: Potential mixture collapse! "
                    f"Max weight: {max_weight:.3f}"
                )

            if min_sigma < 0.001:  # Very small sigma values
                logger.warning(
                    f"Epoch {epoch+1}: Very small sigma values detected! "
                    f"Min sigma: {min_sigma:.6f} - This may cause over-confidence"
                )

        except Exception as e:
            logger.warning(f"Mixture monitoring failed at epoch {epoch+1}: {e}")


class ProbabilisticEpochVisualizationCallback(keras.callbacks.Callback):
    """Keras callback for visualizing probabilistic forecasts during training.

    This callback creates visualizations of the model's predictions at regular
    intervals during training to monitor progress and detect issues.
    """

    def __init__(
        self,
        val_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        processor,
        config: ProbabilisticNBeatsConfig,
        model_type: str,
        horizon: int,
        save_dir: str
    ):
        """Initialize the visualization callback.

        Args:
            val_data_dict: Dictionary of validation data by task
            processor: Data processor for transformations
            config: Configuration object
            model_type: Type of model being trained
            horizon: Forecast horizon
            save_dir: Directory to save visualizations
        """
        super().__init__()
        self.val_data = val_data_dict
        self.processor = processor
        self.config = config
        self.model_type = model_type
        self.horizon = horizon
        self.save_dir = save_dir
        self.plot_indices = {}
        self.random_state = np.random.RandomState(RANDOM_SEED)

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Set up visualization tasks and sample indices.

        Args:
            logs: Training logs (unused)
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Select tasks that have validation data
        available_tasks = [
            name for name, data in self.val_data.items()
            if len(data[0]) > 0
        ]

        # Randomly select up to 4 tasks for visualization
        num_tasks_to_plot = min(MAX_CATEGORIES_TO_PLOT, len(available_tasks))
        plot_tasks = self.random_state.choice(
            available_tasks,
            num_tasks_to_plot,
            replace=False
        )

        # Select random sample indices for each task
        for task in plot_tasks:
            max_idx = len(self.val_data[task][0]) - 1
            self.plot_indices[task] = self.random_state.randint(0, max_idx)

        logger.info(f"Callback will visualize: {list(self.plot_indices.keys())}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Create visualizations at specified intervals.

        Args:
            epoch: Current epoch number
            logs: Training logs (unused)
        """
        if (epoch + 1) % self.config.epoch_plot_freq != 0:
            return

        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
        fig.suptitle(
            f'Epoch {epoch+1:03d}: Forecasts for {self.model_type.title()} '
            f'(H={self.horizon})',
            fontsize=16
        )
        axes = axes.flatten()

        # Plot forecasts for each selected task
        for i, (task_name, sample_idx) in enumerate(self.plot_indices.items()):
            ax = axes[i]
            self._plot_task_forecast(ax, task_name, sample_idx)

        # Hide unused subplots
        for j in range(len(self.plot_indices), MAX_CATEGORIES_TO_PLOT):
            axes[j].set_visible(False)

        # Save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}.png")
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)

    def _plot_task_forecast(self, ax, task_name: str, sample_idx: int) -> None:
        """Plot forecast for a specific task and sample.

        Args:
            ax: Matplotlib axis to plot on
            task_name: Name of the task
            sample_idx: Index of the sample to plot
        """
        # Get validation data for this task
        X_val_scaled, y_val_scaled = self.val_data[task_name]
        x_sample_scaled = X_val_scaled[np.newaxis, sample_idx]
        y_true_scaled = y_val_scaled[sample_idx]

        # Generate probabilistic predictions
        preds = self.model.predict_probabilistic(x_sample_scaled, num_samples=50)
        point_pred_scaled = preds['point_estimate']
        total_var_scaled = preds['total_variance']

        # Transform predictions back to original scale
        point_pred_orig = self.processor.inverse_transform_data(task_name, point_pred_scaled)
        y_true_orig = self.processor.inverse_transform_data(task_name, y_true_scaled)
        x_input_orig = self.processor.inverse_transform_data(task_name, x_sample_scaled[0])

        # Calculate uncertainty in original scale
        scaler = self.processor.scalers[task_name]
        scale_factor = self._get_scale_factor(scaler)
        std_dev_orig = np.sqrt(total_var_scaled) * scale_factor

        # Prepare data for plotting
        point_flat = point_pred_orig.flatten()
        std_flat = std_dev_orig.flatten()

        # Calculate confidence intervals
        lower_68 = point_flat - std_flat
        upper_68 = point_flat + std_flat
        lower_95 = point_flat - 1.96 * std_flat
        upper_95 = point_flat + 1.96 * std_flat

        # Create time axes
        backcast_time = np.arange(-self.config.backcast_length, 0)
        forecast_time = np.arange(self.horizon)

        # Plot the components
        ax.plot(backcast_time, x_input_orig.flatten(),
               color='gray', label='Backcast', linewidth=2)
        ax.plot(forecast_time, y_true_orig.flatten(),
               'o-', color='blue', label='True Future', linewidth=2)
        ax.plot(forecast_time, point_flat,
               '--', color='red', label='Point Forecast', linewidth=2)

        # Add uncertainty bands
        ax.fill_between(
            forecast_time, lower_68, upper_68,
            alpha=0.3, color='red', label='68% PI'
        )
        ax.fill_between(
            forecast_time, lower_95, upper_95,
            alpha=0.2, color='red', label='95% PI'
        )

        # Format the plot
        ax.set_title(f'Task: {task_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    def _get_scale_factor(self, scaler: TimeSeriesNormalizer) -> float:
        """Get the appropriate scale factor for uncertainty transformation.

        Args:
            scaler: The fitted scaler for the task

        Returns:
            Scale factor for transforming uncertainty to original scale
        """
        if scaler.method == 'standard' and scaler.std_val is not None:
            return scaler.std_val
        elif scaler.method == 'minmax' and scaler.max_val is not None:
            return scaler.max_val - scaler.min_val
        else:
            return 1.0


# ---------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------

class ProbabilisticNBeatsDataProcessor:
    """Data processor for probabilistic N-BEATS experiments.

    This class handles data normalization, sequence creation, and transformations
    needed for training and evaluation of probabilistic N-BEATS models.
    """

    def __init__(self, config: ProbabilisticNBeatsConfig):
        """Initialize the data processor.

        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}

    def create_sequences(
        self,
        data: np.ndarray,
        backcast_length: int,
        forecast_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences from time series data.

        Args:
            data: Time series data array
            backcast_length: Length of input sequences
            forecast_length: Length of output sequences

        Returns:
            Tuple of (input_sequences, output_sequences)
        """
        X, y = [], []

        # Calculate the range of valid starting positions
        total_length = backcast_length + forecast_length
        valid_range = range(len(data) - total_length + 1)

        # Create sequences
        for i in valid_range:
            backcast_end = i + backcast_length
            forecast_end = backcast_end + forecast_length

            X.append(data[i:backcast_end])
            y.append(data[backcast_end:forecast_end])

        return np.array(X), np.array(y)

    def fit_scalers(self, task_data: Dict[str, np.ndarray]) -> None:
        """Fit normalization scalers on training data for each task.

        Args:
            task_data: Dictionary mapping task names to time series data
        """
        for task_name, data in task_data.items():
            # Only fit on training portion of the data
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            # Create and fit scaler (using minmax normalization)
            scaler = TimeSeriesNormalizer(method='minmax')
            scaler.fit(train_data)
            self.scalers[task_name] = scaler

    def transform_data(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler for a specific task.

        Args:
            task_name: Name of the task
            data: Data to transform

        Returns:
            Transformed data
        """
        return self.scalers[task_name].transform(data)

    def inverse_transform_data(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Inverse transform data back to original scale.

        Args:
            task_name: Name of the task
            data: Data to inverse transform

        Returns:
            Data in original scale
        """
        return self.scalers[task_name].inverse_transform(data)


# ---------------------------------------------------------------------
# Probabilistic N-BEATS Trainer
# ---------------------------------------------------------------------

class ProbabilisticNBeatsTrainer:
    """Main trainer class for probabilistic N-BEATS multi-task experiments.

    This class orchestrates the entire experimental pipeline including data
    preparation, model training, evaluation, and visualization. It supports
    both standard MDN loss and hybrid loss functions.
    """

    def __init__(self, config: ProbabilisticNBeatsConfig, ts_config: TimeSeriesConfig):
        """Initialize the trainer.

        Args:
            config: Probabilistic N-BEATS configuration
            ts_config: Time series generation configuration
        """
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = ProbabilisticNBeatsDataProcessor(config)
        self.task_names = self.generator.get_task_names()
        self.task_categories = {
            category: self.generator.get_tasks_by_category(category)
            for category in self.generator.get_task_categories()
        }
        self.random_state = np.random.RandomState(RANDOM_SEED)

    def prepare_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """Prepare multi-task data for training and evaluation.

        Returns:
            Nested dictionary structure:
            {horizon: {task_name: {split: (X, y)}}}
        """
        logger.info("Preparing multi-task data...")

        # Generate raw data for all tasks
        raw_data = {
            name: self.generator.generate_task_data(name)
            for name in self.task_names
        }

        # Fit scalers on training data
        self.processor.fit_scalers(raw_data)

        # Prepare data for each forecast horizon
        prepared_data = {}
        for horizon in self.config.forecast_horizons:
            prepared_data[horizon] = {}

            for task_name, data in raw_data.items():
                # Split data into train/val/test
                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))

                train_data = data[:train_size]
                val_data = data[train_size:train_size + val_size]
                test_data = data[train_size + val_size:]

                # Scale each split
                train_scaled = self.processor.transform_data(task_name, train_data)
                val_scaled = self.processor.transform_data(task_name, val_data)
                test_scaled = self.processor.transform_data(task_name, test_data)

                # Create sequences for each split
                prepared_data[horizon][task_name] = {
                    "train": self.processor.create_sequences(
                        train_scaled, self.config.backcast_length, horizon
                    ),
                    "val": self.processor.create_sequences(
                        val_scaled, self.config.backcast_length, horizon
                    ),
                    "test": self.processor.create_sequences(
                        test_scaled, self.config.backcast_length, horizon
                    ),
                }

        return prepared_data

    def create_model(self, model_type: str, forecast_length: int) -> ProbabilisticNBeatsNet:
        """Create a probabilistic N-BEATS model with specified configuration.

        Args:
            model_type: Type of model ("interpretable" or "generic")
            forecast_length: Length of forecast horizon

        Returns:
            Configured probabilistic N-BEATS model
        """
        return ProbabilisticNBeatsNet(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types[model_type],
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            thetas_dim=self.config.thetas_dim[model_type],
            hidden_layer_units=self.config.hidden_layer_units,
            num_mixtures=self.config.num_mixtures,
            mdn_hidden_units=self.config.mdn_hidden_units,
            aggregation_mode=self.config.aggregation_mode,
            diversity_regularizer_strength=self.config.diversity_regularizer_strength,
            min_sigma=self.config.min_sigma,
            input_dim=1
        )

    def train_model(
        self,
        model: ProbabilisticNBeatsNet,
        train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        val_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str,
        exp_dir: str
    ) -> Dict[str, Any]:
        """Train a probabilistic N-BEATS model.

        Args:
            model: Model to train
            train_data: Training data by task
            val_data: Validation data by task
            horizon: Forecast horizon
            model_type: Type of model
            exp_dir: Experiment directory for saving results

        Returns:
            Dictionary containing training history and model
        """
        # Aggregate data from all tasks
        valid_train_data = [data for data in train_data.values() if len(data[0]) > 0]
        valid_val_data = [data for data in val_data.values() if len(data[0]) > 0]

        if not valid_train_data or not valid_val_data:
            raise ValueError("No valid training or validation data found")

        # Concatenate all training data
        X_train = np.concatenate([data[0] for data in valid_train_data], axis=0)
        y_train = np.concatenate([data[1] for data in valid_train_data], axis=0)

        # Concatenate all validation data
        X_val = np.concatenate([data[0] for data in valid_val_data], axis=0)
        y_val = np.concatenate([data[1] for data in valid_val_data], axis=0)

        # Shuffle training data
        train_indices = self.random_state.permutation(len(X_train))
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        # Add data augmentation noise
        X_train = X_train + self.random_state.normal(
            0, DEFAULT_AUGMENTATION_NOISE, X_train.shape
        )

        logger.info(
            f"Training data shape: {X_train.shape}, "
            f"Validation shape: {X_val.shape}, "
            f"Augmentation noise: {DEFAULT_AUGMENTATION_NOISE}"
        )

        # Configure optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            clipnorm=self.config.gradient_clip_norm
        )

        # Configure loss function
        loss_fn = self._get_loss_function(model)

        # Configure model with min_sigma parameter
        if hasattr(model, 'mdn_layer') and model.mdn_layer is not None:
            # Ensure min_sigma is properly configured
            logger.info(f"MDN layer min_sigma constraint: {self.config.min_sigma}")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss_fn)

        # Set up callbacks
        callbacks = self._create_training_callbacks(
            val_data, model_type, horizon, exp_dir, X_val
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
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

    def _get_loss_function(self, model: ProbabilisticNBeatsNet):
        """Get the configured loss function for the model.

        Args:
            model: The model instance

        Returns:
            Loss function (either hybrid or standard MDN loss)
        """
        if self.config.use_hybrid_loss:
            logger.info(f"Using Hybrid Loss with alpha = {self.config.hybrid_loss_alpha}")
            return model.create_hybrid_loss(alpha=self.config.hybrid_loss_alpha)
        else:
            logger.info("Using standard MDN Negative Log-Likelihood Loss.")
            return model.mdn_loss

    def _log_loss_interpretation(self, loss_value: float, epoch: int) -> None:
        """Log interpretation of loss values for better understanding.

        Args:
            loss_value: Current loss value
            epoch: Current epoch number
        """
        if epoch % 10 == 0:  # Log every 10 epochs
            if loss_value < 0:
                logger.info(
                    f"Epoch {epoch}: Negative loss ({loss_value:.4f}) indicates "
                    f"high-confidence predictions. Model is learning well."
                )
            else:
                logger.info(
                    f"Epoch {epoch}: Positive loss ({loss_value:.4f}) indicates "
                    f"lower-confidence predictions."
                )

            # Additional interpretations based on loss magnitude
            if loss_value < -15:
                logger.warning(
                    f"Epoch {epoch}: Very negative loss ({loss_value:.4f}) may "
                    f"indicate over-confidence. Monitor mixture components."
                )
            elif loss_value < -10:
                logger.info(
                    f"Epoch {epoch}: Strongly negative loss ({loss_value:.4f}) "
                    f"suggests good model confidence."
                )

    def _create_training_callbacks(
        self,
        val_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        model_type: str,
        horizon: int,
        exp_dir: str,
        X_val: np.ndarray
    ) -> List[keras.callbacks.Callback]:
        """Create training callbacks.

        Args:
            val_data: Validation data
            model_type: Type of model
            horizon: Forecast horizon
            exp_dir: Experiment directory
            X_val: Validation input data

        Returns:
            List of configured callbacks
        """
        # Visualization directory
        vis_dir = os.path.join(
            exp_dir, 'visuals', 'epoch_plots', f'{model_type}_h{horizon}'
        )

        # Custom callback to log loss interpretations
        class LossInterpretationCallback(keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer

            def on_epoch_end(self, epoch, logs=None):
                if logs and 'loss' in logs:
                    self.trainer._log_loss_interpretation(logs['loss'], epoch + 1)

        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-6
            ),
            keras.callbacks.TerminateOnNaN(),
            ProbabilisticEpochVisualizationCallback(
                val_data, self.processor, self.config,
                model_type, horizon, vis_dir
            ),
            MixtureMonitoringCallback(
                X_val, self.processor, f"{model_type}_h{horizon}", exp_dir
            ),
            LossInterpretationCallback(self)
        ]

    def evaluate_model(
        self,
        model: ProbabilisticNBeatsNet,
        test_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        model_type: str
    ) -> Dict[str, ProbabilisticForecastMetrics]:
        """Evaluate a trained model on test data.

        Args:
            model: Trained model to evaluate
            test_data: Test data by task
            horizon: Forecast horizon
            model_type: Type of model

        Returns:
            Dictionary of metrics by task name
        """
        all_task_metrics = {}

        for task_name, (X_test, y_test) in test_data.items():
            if len(X_test) > 0:
                metrics = self._calculate_probabilistic_metrics(
                    model, X_test, y_test, task_name, model_type, horizon
                )
                all_task_metrics[task_name] = metrics

        return all_task_metrics

    def _calculate_probabilistic_metrics(
        self,
        model: ProbabilisticNBeatsNet,
        X_test: np.ndarray,
        y_test_scaled: np.ndarray,
        task_name: str,
        model_type: str,
        horizon: int
    ) -> ProbabilisticForecastMetrics:
        """Calculate comprehensive probabilistic metrics for a single task.

        Args:
            model: Trained model
            X_test: Test input data
            y_test_scaled: Test target data (scaled)
            task_name: Name of the task
            model_type: Type of model
            horizon: Forecast horizon

        Returns:
            Comprehensive metrics object
        """
        # Generate probabilistic predictions
        preds = model.predict_probabilistic(
            X_test, num_samples=self.config.num_prediction_samples
        )

        point_scaled = preds['point_estimate']
        total_var_scaled = preds['total_variance']
        aleatoric_var_scaled = preds['aleatoric_variance']

        # Transform to original scale
        scaler = self.processor.scalers[task_name]
        point_orig = self.processor.inverse_transform_data(task_name, point_scaled)
        y_true_orig = self.processor.inverse_transform_data(task_name, y_test_scaled)

        # Calculate variance scaling factor
        scale_squared = self._get_variance_scale_factor(scaler)

        # Transform variances to original scale
        total_var_orig = total_var_scaled * scale_squared
        aleatoric_var_orig = aleatoric_var_scaled * scale_squared

        # Ensure consistent shapes for metric calculation
        if y_true_orig.ndim == 3 and y_true_orig.shape[-1] == 1:
            y_true_orig = np.squeeze(y_true_orig, axis=-1)

        # Calculate point forecasting metrics
        errors = y_true_orig - point_orig
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))

        # Calculate MAPE (avoiding division by zero)
        non_zero_mask = np.abs(y_true_orig) > EPSILON_ZERO_DIVISION
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / y_true_orig[non_zero_mask])) * 100
        else:
            mape = 0.0

        # Calculate log-likelihood
        mixture_params = model.predict(X_test, verbose=0)
        try:
            # Note: For hybrid loss, true log-likelihood is only part of the loss.
            # This calculates the pure NLL part for evaluation.
            log_likelihood = -float(model.mdn_loss(y_test_scaled, mixture_params))
        except Exception as e:
            logger.warning(f"Log-likelihood calculation failed: {e}")
            log_likelihood = -np.inf

        # Calculate prediction intervals and coverage
        std_dev_orig = np.sqrt(total_var_orig)
        confidence_intervals = {}
        coverage = {}
        widths = {}

        for confidence_level in self.config.confidence_levels:
            # Calculate z-score for the confidence level
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

            # Calculate interval bounds
            lower_bound = point_orig - z_score * std_dev_orig
            upper_bound = point_orig + z_score * std_dev_orig
            confidence_intervals[confidence_level] = (lower_bound, upper_bound)

            # Calculate coverage
            in_interval = (y_true_orig >= lower_bound) & (y_true_orig <= upper_bound)
            coverage[confidence_level] = np.mean(in_interval)

            # Calculate average interval width
            widths[confidence_level] = np.mean(upper_bound - lower_bound)

        # Calculate CRPS
        crps = self._calculate_crps(model, X_test, y_true_orig, task_name)

        # Analyze mixture behavior
        _, _, pi_logits = model.mdn_layer.split_mixture_params(mixture_params)
        pi_weights = keras.ops.convert_to_numpy(keras.activations.softmax(pi_logits))

        mean_num_active_mixtures = np.mean(
            np.sum(pi_weights > MIXTURE_ACTIVE_THRESHOLD, axis=-1)
        )
        mixture_entropy = np.mean(
            -np.sum(pi_weights * np.log(pi_weights + EPSILON_LOG), axis=-1)
        )

        # Get task category
        task_category = self._get_task_category(task_name)

        return ProbabilisticForecastMetrics(
            task_name=task_name,
            task_category=task_category,
            model_type=model_type,
            horizon=horizon,
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mae,
            mape=mape,
            crps=crps,
            log_likelihood=log_likelihood,
            coverage_68=coverage.get(0.68, 0),
            coverage_90=coverage.get(0.90, 0),
            coverage_95=coverage.get(0.95, 0),
            interval_width_68=widths.get(0.68, 0),
            interval_width_90=widths.get(0.90, 0),
            interval_width_95=widths.get(0.95, 0),
            total_uncertainty=np.mean(total_var_orig),
            aleatoric_uncertainty=np.mean(aleatoric_var_orig),
            epistemic_uncertainty=np.mean(total_var_orig - aleatoric_var_orig),
            mean_num_active_mixtures=mean_num_active_mixtures,
            mixture_entropy=mixture_entropy,
            samples_count=y_true_orig.size
        )

    def _get_variance_scale_factor(self, scaler: TimeSeriesNormalizer) -> float:
        """Get the variance scaling factor for transforming uncertainty to original scale.

        Args:
            scaler: The fitted scaler for the task

        Returns:
            Variance scaling factor
        """
        if scaler.method == 'standard' and scaler.std_val is not None:
            return scaler.std_val ** 2
        elif scaler.max_val is not None:
            return (scaler.max_val - scaler.min_val) ** 2
        else:
            return 1.0

    def _get_task_category(self, task_name: str) -> str:
        """Get the category for a given task name.

        Args:
            task_name: Name of the task

        Returns:
            Task category name
        """
        # Check each category to find which one contains this task
        for category in self.generator.get_task_categories():
            tasks_in_category = self.generator.get_tasks_by_category(category)
            if task_name in tasks_in_category:
                return category
        return ""

    def _calculate_crps(
        self,
        model: ProbabilisticNBeatsNet,
        X_test: np.ndarray,
        y_true_orig: np.ndarray,
        task_name: str
    ) -> float:
        """Calculate Continuous Ranked Probability Score (CRPS) with improved efficiency.

        Args:
            model: Trained model
            X_test: Test input data
            y_true_orig: True values in original scale
            task_name: Name of the task

        Returns:
            CRPS value
        """
        # Generate prediction samples
        preds = model.predict_probabilistic(
            X_test, num_samples=self.config.num_prediction_samples
        )

        # Transform samples to original scale
        samples_orig = np.zeros_like(preds['samples'])
        for i in range(preds['samples'].shape[1]):
            samples_orig[:, i, :] = self.processor.inverse_transform_data(
                task_name, preds['samples'][:, i, :]
            )

        # Ensure consistent shapes
        if y_true_orig.ndim == 3 and y_true_orig.shape[-1] == 1:
            y_true_orig = np.squeeze(y_true_orig, axis=-1)

        # Sort samples along the sample dimension for efficient calculation
        samples_orig.sort(axis=1)

        # Reshape for broadcasting
        y_reshaped = y_true_orig[:, np.newaxis, :]  # (n_instances, 1, horizon)

        # Calculate first term: E[|Y - X|]
        term1 = np.mean(np.abs(samples_orig - y_reshaped), axis=1)  # (n_instances, horizon)

        # Calculate second term more efficiently
        # For each instance, calculate the expected absolute difference between samples
        term2_vals = []
        for i in range(samples_orig.shape[0]):  # Loop over instances
            sample_instance = samples_orig[i, :, :]  # (n_samples, horizon)
            # Calculate pairwise absolute differences
            pairwise_diffs = np.abs(
                sample_instance[:, :, np.newaxis] - sample_instance[:, np.newaxis, :]
            )
            # Average over all pairs
            term2_vals.append(np.mean(pairwise_diffs))

        term2 = np.mean(term2_vals)

        # CRPS formula: E[|Y - X|] - 0.5 * E[|X - X'|]
        crps_result = np.mean(term1) - 0.5 * term2

        return crps_result

    def plot_probabilistic_forecasts(
        self,
        models: Dict[int, Dict[str, ProbabilisticNBeatsNet]],
        prepared_data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
        save_dir: str
    ) -> None:
        """Create probabilistic forecast visualizations by category.

        Args:
            models: Trained models by horizon and type
            prepared_data: Prepared data structure
            save_dir: Directory to save visualizations
        """
        logger.info("Creating probabilistic forecast visualizations...")

        plot_dir = os.path.join(save_dir, 'probabilistic_forecasts')
        os.makedirs(plot_dir, exist_ok=True)

        # Create plots for each category and horizon
        for category in self.generator.get_task_categories():
            tasks = self.generator.get_tasks_by_category(category)
            for horizon in self.config.forecast_horizons:
                self._plot_category_forecasts(
                    category, tasks, horizon, models[horizon],
                    prepared_data[horizon], plot_dir
                )

    def _plot_category_forecasts(
        self,
        category: str,
        tasks: List[str],
        horizon: int,
        horizon_models: Dict[str, ProbabilisticNBeatsNet],
        horizon_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        plot_dir: str
    ) -> None:
        """Plot forecasts for a specific category and horizon.

        Args:
            category: Task category name
            tasks: List of task names in the category
            horizon: Forecast horizon
            horizon_models: Models for this horizon
            horizon_data: Data for this horizon
            plot_dir: Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
        fig.suptitle(f'Forecasts - {category.title()} (H={horizon})', fontsize=16)

        # Select up to 4 tasks randomly from the category
        available_tasks = [task for task in tasks if task in horizon_data]
        if not available_tasks:
            plt.close(fig)
            return

        selected_tasks = self.random_state.choice(
            available_tasks,
            min(MAX_CATEGORIES_TO_PLOT, len(available_tasks)),
            replace=False
        )

        for i, task_name in enumerate(selected_tasks):
            ax = axes.flatten()[i]

            # Check if test data exists for this task
            if (task_name not in horizon_data or
                len(horizon_data[task_name]["test"][0]) == 0):
                continue

            # Get test data and select random sample
            X_test, y_test = horizon_data[task_name]["test"]
            sample_idx = self.random_state.randint(len(X_test))

            # Transform true values to original scale
            y_true_orig = self.processor.inverse_transform_data(task_name, y_test[sample_idx])

            # Plot true values
            ax.plot(
                np.arange(horizon), y_true_orig.flatten(),
                'o-', color='blue', label='True', linewidth=2
            )

            # Plot predictions from each model type
            for model_type, model in horizon_models.items():
                # Generate probabilistic predictions
                preds = model.predict_probabilistic(X_test[np.newaxis, sample_idx])

                # Transform to original scale
                point_orig = self.processor.inverse_transform_data(
                    task_name, preds['point_estimate']
                )

                # Calculate uncertainty in original scale
                scaler = self.processor.scalers[task_name]
                scale_factor = self._get_scale_factor_for_plotting(scaler)
                std_orig = np.sqrt(preds['total_variance']) * scale_factor

                # Plot point prediction
                ax.plot(
                    np.arange(horizon), point_orig.flatten(),
                    '--', label=f'{model_type} Pred', linewidth=2
                )

                # Plot 95% prediction interval
                lower_bound = (point_orig - 1.96 * std_orig).flatten()
                upper_bound = (point_orig + 1.96 * std_orig).flatten()
                ax.fill_between(
                    np.arange(horizon), lower_bound, upper_bound,
                    alpha=0.2, label=f'{model_type} 95% PI'
                )

            # Format subplot
            ax.set_title(task_name.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.4)

        # Save plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(plot_dir, f'forecasts_{category}_h{horizon}.png')
        plt.savefig(plot_path)
        plt.close(fig)

    def _get_scale_factor_for_plotting(self, scaler: TimeSeriesNormalizer) -> float:
        """Get scale factor for plotting uncertainty bands.

        Args:
            scaler: The fitted scaler for the task

        Returns:
            Scale factor for plotting
        """
        if scaler.method == 'standard' and scaler.std_val is not None:
            return scaler.std_val
        elif scaler.method == 'minmax' and scaler.max_val is not None:
            return scaler.max_val - scaler.min_val
        else:
            return 1.0

    def plot_uncertainty_analysis(
        self,
        models: Dict[int, Dict[str, ProbabilisticNBeatsNet]],
        prepared_data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
        save_dir: str
    ) -> None:
        """Create uncertainty analysis visualizations.

        Args:
            models: Trained models by horizon and type
            prepared_data: Prepared data structure
            save_dir: Directory to save visualizations
        """
        logger.info("Creating uncertainty analysis visualizations...")

        plot_dir = os.path.join(save_dir, 'uncertainty_analysis')
        os.makedirs(plot_dir, exist_ok=True)

        # Create uncertainty plots for each model type and horizon
        for model_type in self.config.model_types:
            for horizon in self.config.forecast_horizons:
                self._plot_model_uncertainty_analysis(
                    model_type, horizon, models[horizon][model_type],
                    prepared_data[horizon], plot_dir
                )

    def _plot_model_uncertainty_analysis(
        self,
        model_type: str,
        horizon: int,
        model: ProbabilisticNBeatsNet,
        horizon_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        plot_dir: str
    ) -> None:
        """Create uncertainty analysis for a specific model.

        Args:
            model_type: Type of model
            horizon: Forecast horizon
            model: Trained model
            horizon_data: Data for this horizon
            plot_dir: Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Uncertainty Analysis: {model_type.title()} H={horizon}', fontsize=16)

        # Collect uncertainty and error data across all tasks
        all_total_std = []
        all_aleatoric_std = []
        all_epistemic_std = []
        all_errors = []

        for task_name, data_splits in horizon_data.items():
            if len(data_splits["test"][0]) == 0:
                continue

            X_test, y_test = data_splits["test"]

            # Generate predictions
            preds = model.predict_probabilistic(X_test)

            # Get scaling factor
            scaler = self.processor.scalers[task_name]
            scale_factor = self._get_scale_factor_for_plotting(scaler)

            # Transform predictions to original scale
            point_orig = self.processor.inverse_transform_data(task_name, preds['point_estimate'])
            y_true_orig = self.processor.inverse_transform_data(task_name, y_test)

            # Handle shape consistency
            if y_true_orig.ndim == 3 and y_true_orig.shape[-1] == 1:
                y_true_orig = np.squeeze(y_true_orig, axis=-1)

            # Calculate uncertainties in original scale
            total_std = np.sqrt(preds['total_variance'].flatten()) * scale_factor
            aleatoric_std = np.sqrt(preds['aleatoric_variance'].flatten()) * scale_factor
            epistemic_var = preds['total_variance'] - preds['aleatoric_variance']
            epistemic_std = np.sqrt(epistemic_var.flatten()) * scale_factor

            # Calculate errors
            errors = np.abs(y_true_orig - point_orig).flatten()

            # Collect data
            all_total_std.append(total_std)
            all_aleatoric_std.append(aleatoric_std)
            all_epistemic_std.append(epistemic_std)
            all_errors.append(errors)

        if not all_total_std:
            plt.close(fig)
            return

        # Concatenate all data
        total_std = np.concatenate(all_total_std)
        aleatoric_std = np.concatenate(all_aleatoric_std)
        epistemic_std = np.concatenate(all_epistemic_std)
        errors = np.concatenate(all_errors)

        # Plot 1: Uncertainty distributions
        axes[0, 0].hist(total_std, bins=50, alpha=0.7, density=True, label='Total')
        axes[0, 0].hist(aleatoric_std, bins=50, alpha=0.7, density=True, label='Aleatoric')
        axes[0, 0].legend()
        axes[0, 0].set_title('Uncertainty Distribution')
        axes[0, 0].set_xlabel('Standard Deviation')
        axes[0, 0].set_ylabel('Density')

        # Plot 2: Aleatoric vs Epistemic uncertainty
        axes[0, 1].scatter(aleatoric_std, epistemic_std, alpha=0.1)
        axes[0, 1].set_title('Aleatoric vs Epistemic Uncertainty')
        axes[0, 1].set_xlabel('Aleatoric Std')
        axes[0, 1].set_ylabel('Epistemic Std')

        # Plot 3: Uncertainty vs Error relationship
        sample_indices = self.random_state.choice(
            len(errors),
            min(PLOTTING_SAMPLE_LIMIT, len(errors)),
            replace=False
        )
        axes[1, 0].scatter(
            total_std[sample_indices],
            errors[sample_indices],
            alpha=0.1
        )
        axes[1, 0].set_title('Total Uncertainty vs. Prediction Error')
        axes[1, 0].set_xlabel('Total Std')
        axes[1, 0].set_ylabel('Absolute Error')

        # Plot 4: Average mixture weights
        self._plot_mixture_weights(axes[1, 1], model, horizon_data)

        # Save plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(plot_dir, f'uncertainty_{model_type}_h{horizon}.png')
        plt.savefig(plot_path)
        plt.close(fig)

    def _plot_mixture_weights(
        self,
        ax,
        model: ProbabilisticNBeatsNet,
        horizon_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
    ) -> None:
        """Plot average mixture weights for the model.

        Args:
            ax: Matplotlib axis to plot on
            model: Trained model
            horizon_data: Data for this horizon
        """
        # Get sample data for mixture analysis
        X_sample = np.concatenate([
            data["test"][0] for data in horizon_data.values()
            if len(data["test"][0]) > 0
        ])[:100]

        if len(X_sample) > 0:
            mixture_params = model.predict(X_sample, verbose=0)
            _, _, pi_logits = model.mdn_layer.split_mixture_params(mixture_params)
            pi_weights = keras.ops.convert_to_numpy(keras.activations.softmax(pi_logits))
            mean_weights = np.mean(pi_weights, axis=0)

            ax.bar(range(self.config.num_mixtures), mean_weights)
            ax.set_title('Average Mixture Weights')
            ax.set_xlabel('Mixture Component')
            ax.set_ylabel('Weight')
        else:
            ax.set_title('Average Mixture Weights (No Data)')

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete probabilistic N-BEATS experiment.

        Returns:
            Dictionary containing experiment results and metadata
        """
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_{timestamp}"
        )
        os.makedirs(exp_dir, exist_ok=True)

        logger.info(f"Starting experiment: {exp_dir}")

        # Prepare data
        prepared_data = self.prepare_data()

        # Train models and collect metrics
        trained_models = {}
        all_metrics = {}

        for horizon in self.config.forecast_horizons:
            trained_models[horizon] = {}
            all_metrics[horizon] = {}

            for model_type in self.config.model_types:
                logger.info(
                    f"\n{'='*60}\n"
                    f"Training {model_type} model for horizon {horizon}\n"
                    f"{'='*60}"
                )

                # Create and train model
                model = self.create_model(model_type, horizon)

                # Organize data splits
                data_splits = {
                    split_name: {
                        task_name: task_data[split_name]
                        for task_name, task_data in prepared_data[horizon].items()
                    }
                    for split_name in ["train", "val", "test"]
                }

                # Train model
                training_result = self.train_model(
                    model,
                    data_splits["train"],
                    data_splits["val"],
                    horizon,
                    model_type,
                    exp_dir
                )

                # Store trained model
                trained_models[horizon][model_type] = training_result["model"]

                # Evaluate model
                task_metrics = self.evaluate_model(
                    training_result["model"],
                    data_splits["test"],
                    horizon,
                    model_type
                )
                all_metrics[horizon][model_type] = task_metrics

                # Save model if requested
                if self.config.save_results:
                    model_path = os.path.join(exp_dir, f"{model_type}_h{horizon}.keras")
                    training_result["model"].save(model_path)

        # Create visualizations and summary
        if self.config.save_results:
            visuals_dir = os.path.join(exp_dir, 'visuals')

            self.plot_probabilistic_forecasts(
                trained_models, prepared_data, visuals_dir
            )
            self.plot_uncertainty_analysis(
                trained_models, prepared_data, visuals_dir
            )
            self._generate_results_summary(all_metrics, exp_dir)

        logger.info(f"Experiment complete. Results saved to: {exp_dir}")

        return {
            "results_dir": exp_dir,
            "metrics": all_metrics
        }

    def _generate_results_summary(
        self,
        all_metrics: Dict[int, Dict[str, Dict[str, ProbabilisticForecastMetrics]]],
        exp_dir: str
    ) -> None:
        """Generate and save results summary.

        Args:
            all_metrics: All computed metrics
            exp_dir: Experiment directory
        """
        # Convert metrics to list of dictionaries
        results = []
        for horizon_metrics in all_metrics.values():
            for model_type_metrics in horizon_metrics.values():
                for metrics in model_type_metrics.values():
                    results.append(dataclasses.asdict(metrics))

        if not results:
            logger.warning("No results to summarize")
            return

        # Create DataFrame
        df = pd.DataFrame(results)

        # Create summary by model type and horizon
        summary_columns = [
            'rmse', 'crps', 'log_likelihood', 'coverage_95',
            'interval_width_95', 'total_uncertainty'
        ]

        summary = df.groupby(['model_type', 'horizon'])[summary_columns].mean().round(4)

        # Log summary
        logger.info(
            f"\n{'='*80}\n"
            f"SUMMARY BY MODEL AND HORIZON\n"
            f"{'='*80}\n"
            f"{summary}"
        )

        # Save detailed results
        df.to_csv(os.path.join(exp_dir, 'detailed_results.csv'), index=False)
        summary.to_csv(os.path.join(exp_dir, 'summary_by_model.csv'))


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main() -> None:
    """Main function to run the Probabilistic N-BEATS experiment."""
    # Configuration
    config = ProbabilisticNBeatsConfig(
        epochs=150,
        batch_size=128,
        learning_rate=1e-3,
        early_stopping_patience=50,
        # Increase diversity regularization to prevent mixture collapse
        diversity_regularizer_strength=0.001,  # Increased from 0.001
        # Increase min_sigma to prevent over-confidence
        min_sigma=0.01,  # Increased from 0.01 to prevent very small sigmas
        use_hybrid_loss=True,
        hybrid_loss_alpha=0.5
    )

    ts_config = TimeSeriesConfig(
        n_samples=5000,
        random_seed=RANDOM_SEED,
        # Slightly increase noise to prevent over-fitting
        default_noise_level=0.05
    )

    # Log configuration for debugging
    logger.info(f"Training configuration:")
    logger.info(f"  diversity_regularizer_strength: {config.diversity_regularizer_strength}")
    logger.info(f"  min_sigma: {config.min_sigma}")
    logger.info(f"  use_hybrid_loss: {config.use_hybrid_loss}")
    logger.info(f"  hybrid_loss_alpha: {config.hybrid_loss_alpha}")
    logger.info(f"  default_noise_level: {ts_config.default_noise_level}")

    try:
        # Create and run trainer
        trainer = ProbabilisticNBeatsTrainer(config, ts_config)
        trainer.run_experiment()
        logger.info("Experiment finished successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()