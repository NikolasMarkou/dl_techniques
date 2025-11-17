"""
Comprehensive N-BEATS Training Framework for Multiple Time Series Patterns

This module provides a sophisticated, production-ready training framework for
N-BEATS models. It leverages a streaming data pipeline with tf.data.Dataset
to train a single model on an arbitrary number of dynamically generated time
series patterns, ensuring memory efficiency and scalability.

Classes
-------
NBeatsTrainingConfig
    Comprehensive configuration dataclass for all training parameters.

MultiPatternDataProcessor
    Advanced data processing pipeline that uses Python generators to stream
    sequences for training and evaluation, handling on-the-fly sampling and
    balancing.

PatternPerformanceCallback
    Callback for monitoring performance and visualizing prediction samples on a
    fixed test set.

NBeatsTrainer
    Main training orchestrator for multi-pattern N-BEATS training with
    comprehensive experiment management and performance analysis.
"""

import os
import json
import keras
import random
import math
import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Generator, Union

# Use a non-interactive backend for matplotlib to prevent issues on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.nbeats.model import create_nbeats_model, NBeatsNet
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator, TimeSeriesConfig
)

# ---------------------------------------------------------------------


plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for major libraries to ensure reproducibility.

    :param seed: The integer seed value to use.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)

# ---------------------------------------------------------------------


@dataclass
class NBeatsTrainingConfig:
    """
    Configuration dataclass for N-BEATS training on multiple patterns.

    This class centralizes all hyperparameters and settings for the experiment,
    from data generation and splitting to model architecture and training
    procedures.

    :ivar result_dir: Directory to save experiment results.
    :ivar save_results: If True, saves results and artifacts.
    :ivar experiment_name: A unique name for the experiment.
    :ivar target_categories: Optional list of pattern categories to train on.
    :ivar train_ratio: Fraction of data for training (e.g., 0.8).
    :ivar val_ratio: Fraction of data for validation (e.g., 0.1).
    :ivar test_ratio: Fraction of data for testing (e.g., 0.1).
    :ivar backcast_length: Length of the input sequence (lookback window).
    :ivar forecast_length: Length of the output sequence to predict.
    :ivar forecast_horizons: List of forecast horizons to train models for.
    :ivar stack_types: N-BEATS stack architecture (e.g., trend, seasonality).
    :ivar nb_blocks_per_stack: Number of blocks within each N-BEATS stack.
    :ivar hidden_layer_units: Number of units in N-BEATS block hidden layers.
    :ivar use_revin: Whether to use Reversible Instance Normalization.
    :ivar use_bias: Whether to use bias in the N-BEATS block layers.
    :ivar epochs: Maximum number of training epochs.
    :ivar batch_size: Number of samples per training batch.
    :ivar steps_per_epoch: Number of batches per epoch for generator training.
    :ivar learning_rate: Initial learning rate for the optimizer.
    :ivar gradient_clip_norm: Maximum norm for gradient clipping.
    :ivar optimizer: Name of the Keras optimizer (e.g., 'adamw').
    :ivar primary_loss: Primary loss function (e.g., 'mae').
    :ivar kernel_regularizer_l2: L2 regularization factor for kernel weights.
    :ivar dropout_rate: Dropout rate for regularization.
    :ivar max_patterns: Optional cap on the total number of patterns to use.
    :ivar max_patterns_per_category: Max patterns to select from each category.
    :ivar min_data_length: Minimum length required for a generated time series.
    :ivar category_weights: Dictionary mapping categories to sampling weights.
    :ivar visualize_every_n_epochs: Frequency of generating visualization plots.
    :ivar save_interim_plots: If True, saves plots during training.
    :ivar plot_top_k_patterns: Number of prediction samples to plot.
    :ivar create_learning_curves: If True, generates learning curve plots.
    :ivar create_prediction_plots: If True, generates prediction sample plots.
    :ivar multiplicative_noise_std: Std dev for multiplicative noise augment.
    :ivar additive_noise_std: Std dev for additive noise augmentation.
    :ivar enable_multiplicative_noise: If True, enables multiplicative noise.
    :ivar enable_additive_noise: If True, enables additive noise.
    """
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats_multi_pattern"

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # N-BEATS specific configuration
    backcast_length: int = 168
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [24])

    # Model architecture
    stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 256
    use_revin: bool = True
    use_bias: bool = True

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 500  # Required for generator-based training
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    primary_loss: Union[str, keras.losses.Loss] = keras.losses.MeanAbsoluteError(reduction="mean")

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    dropout_rate: float = 0.15

    # Pattern selection
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    min_data_length: int = 2000
    normalize_per_instance: bool = True  # If True, standardizes each sample (x-mean)/std

    # Category weights for balanced sampling
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 1.0,
        "seasonal": 1.0,
        "composite": 1.2,
        "stochastic": 1.0,
        "financial": 1.5,
        "weather": 1.3,
        "network": 1.4,
        "biomedical": 1.2,
        "industrial": 1.3,
        "intermittent": 1.0,
        "volatility": 1.1,
        "regime": 1.2,
        "structural": 1.1,
        "outliers": 1.0,
        "chaotic": 1.1
    })

    # Visualization configuration
    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 8
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Data augmentation configuration
    multiplicative_noise_std: float = 0.01
    additive_noise_std: float = 0.01
    enable_multiplicative_noise: bool = True
    enable_additive_noise: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.

        :raises ValueError: If data ratios do not sum to 1, or if
                            backcast/forecast lengths are not positive.
        """
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(
                f"Data ratios must sum to 1.0, got {total_ratio}"
            )
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError(
                "backcast_length and forecast_length must be positive"
            )
        if not self.use_revin:
            logger.warning("RevIN is disabled. External normalization is required.")


class MultiPatternDataProcessor:
    """
    Streams multi-pattern time series data using Python generators.

    This class handles the creation of `tf.data.Dataset` objects for
    training, validation, and testing. It uses an on-the-fly data generation
    approach to handle large numbers of time series patterns without loading
    everything into memory. It now supports optional per-instance
    standardization to ensure all time series, regardless of their original
    scale, contribute equally to the loss function.

    :param config: The training configuration object.
    :type config: NBeatsTrainingConfig
    :param generator: An instance of `TimeSeriesGenerator` to produce data.
    :type generator: TimeSeriesGenerator
    :param selected_patterns: A list of pattern names to be used.
    :type selected_patterns: List[str]
    :param pattern_to_category: Mapping from pattern name to its category.
    :type pattern_to_category: Dict[str, str]
    """
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
        """
        Prepare lists for weighted random pattern selection during training.

        :return: A tuple containing the list of pattern names and their
                 corresponding normalized weights.
        :rtype: Tuple[List[str], List[float]]
        """
        patterns = []
        weights = []
        for pattern_name in self.selected_patterns:
            category = self.pattern_to_category.get(pattern_name, "unknown")
            weight = self.config.category_weights.get(category, 1.0)
            patterns.append(pattern_name)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(patterns)] * len(patterns)
        return patterns, normalized_weights

    def _training_generator(
        self, forecast_length: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Create an infinite generator for training data with weighted sampling.

        This generator continuously samples patterns based on category weights,
        generates a fresh time series for that pattern, and yields a single
        randomly selected (backcast, forecast) pair from its training split.

        :param forecast_length: The length of the forecast horizon.
        :type forecast_length: int
        :yield: A tuple containing a backcast and a forecast sequence.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        while True:
            pattern_name = random.choices(
                self.weighted_patterns, self.weights, k=1
            )[0]
            data = self.ts_generator.generate_task_data(pattern_name)
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            max_start_idx = (
                len(train_data) -
                self.config.backcast_length - forecast_length
            )
            if max_start_idx > 0:
                start_idx = random.randint(0, max_start_idx)
                backcast = train_data[
                    start_idx: start_idx + self.config.backcast_length
                ]
                forecast = train_data[
                    start_idx + self.config.backcast_length:
                    start_idx + self.config.backcast_length + forecast_length
                ]

                if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                    yield (
                        backcast.astype(np.float32),
                        forecast.astype(np.float32)
                    )

    def _evaluation_generator(
        self, forecast_length: int, split: str
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Create a finite generator for validation or test data.

        This generator iterates through every selected pattern once, generates
        the time series, and yields all possible (backcast, forecast) pairs
        from the specified data split ('val' or 'test').

        :param forecast_length: The length of the forecast horizon.
        :type forecast_length: int
        :param split: The data split to generate from ('val' or 'test').
        :type split: str
        :raises ValueError: If the split is not 'val' or 'test'.
        :yield: A tuple containing a backcast and a forecast sequence.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            raise ValueError("Split must be 'val' or 'test'")

        for pattern_name in self.selected_patterns:
            data = self.ts_generator.generate_task_data(pattern_name)
            start_idx = int(start_ratio * len(data))
            end_idx = int(end_ratio * len(data))
            split_data = data[start_idx:end_idx]

            for i in range(
                len(split_data) -
                self.config.backcast_length - forecast_length + 1
            ):
                backcast = split_data[i: i + self.config.backcast_length]
                forecast = split_data[
                    i + self.config.backcast_length:
                    i + self.config.backcast_length + forecast_length
                ]
                if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                    yield (
                        backcast.astype(np.float32),
                        forecast.astype(np.float32)
                    )

    def get_evaluation_steps(self, forecast_length: int, split: str) -> int:
        """
        Calculate the number of steps for a full evaluation pass.

        :param forecast_length: The length of the forecast horizon.
        :type forecast_length: int
        :param split: The data split ('val' or 'test').
        :type split: str
        :return: The total number of batches for the evaluation dataset.
        :rtype: int
        """
        total_samples = 0
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            return 0

        for _ in self.selected_patterns:
            data_len = self.ts_generator.config.n_samples
            start_idx = int(start_ratio * data_len)
            end_idx = int(end_ratio * data_len)
            split_len = end_idx - start_idx
            num_sequences = (
                split_len - self.config.backcast_length - forecast_length + 1
            )
            if num_sequences > 0:
                total_samples += num_sequences

        return math.ceil(total_samples / self.config.batch_size)

    def _normalize_instance(
        self, backcast: tf.Tensor, forecast: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Standardizes an individual time series instance (z-score normalization).

        The mean and standard deviation are calculated from the `backcast`
        only and then applied to both the `backcast` and `forecast`. This
        prevents data leakage from the future (forecast) and simulates the
        real-world scenario where we only have past data for normalization.
        This ensures that series with different scales contribute equally to
        the loss function.

        :param backcast: The input tensor (lookback window).
        :type backcast: tf.Tensor
        :param forecast: The target tensor (horizon).
        :type forecast: tf.Tensor
        :return: A tuple of the normalized (backcast, forecast).
        :rtype: Tuple[tf.Tensor, tf.Tensor]
        """
        # Calculate statistics from the backcast
        mean = tf.math.reduce_mean(backcast)
        std = tf.math.reduce_std(backcast)

        # Add a small epsilon to prevent division by zero for flat series
        epsilon = 1e-6

        # Normalize both backcast and forecast with backcast stats
        normalized_backcast = (backcast - mean) / (std + epsilon)
        normalized_forecast = (forecast - mean) / (std + epsilon)

        return normalized_backcast, normalized_forecast

    def _apply_noise_augmentation(
        self, x: tf.Tensor, y: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply additive and/or multiplicative noise augmentation to the input.

        :param x: The input tensor (backcast).
        :type x: tf.Tensor
        :param y: The target tensor (forecast).
        :type y: tf.Tensor
        :return: A tuple of the augmented input and original target.
        :rtype: Tuple[tf.Tensor, tf.Tensor]
        """
        augmented_x = x
        if (
            self.config.enable_multiplicative_noise and
            self.config.multiplicative_noise_std > 0
        ):
            mult_noise = tf.random.normal(
                tf.shape(x), mean=1.0,
                stddev=self.config.multiplicative_noise_std, dtype=x.dtype
            )
            augmented_x = augmented_x * mult_noise
        if (
            self.config.enable_additive_noise and
            self.config.additive_noise_std > 0
        ):
            add_noise = tf.random.normal(
                tf.shape(augmented_x), mean=0.0,
                stddev=self.config.additive_noise_std,
                dtype=augmented_x.dtype
            )
            augmented_x = augmented_x + add_noise
        return augmented_x, y

    def prepare_datasets(self, forecast_length: int) -> Dict[str, Any]:
        """
        Create the complete tf.data pipeline for a given forecast horizon.

        This method constructs and returns the training, validation, and test
        `tf.data.Dataset` objects, along with the calculated steps for
        validation and testing. It incorporates optional per-instance
        standardization and data augmentation into the pipeline.

        :param forecast_length: The forecast horizon for the datasets.
        :type forecast_length: int
        :return: A dictionary containing the datasets and step counts.
        :rtype: Dict[str, Any]
        """
        output_signature = (
            tf.TensorSpec(
                shape=(self.config.backcast_length, 1), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(forecast_length, 1), dtype=tf.float32)
        )

        # --- Create base datasets from generators ---
        train_ds = tf.data.Dataset.from_generator(
            lambda: self._training_generator(forecast_length),
            output_signature=output_signature
        )
        val_ds = tf.data.Dataset.from_generator(
            lambda: self._evaluation_generator(forecast_length, 'val'),
            output_signature=output_signature
        )
        test_ds = tf.data.Dataset.from_generator(
            lambda: self._evaluation_generator(forecast_length, 'test'),
            output_signature=output_signature
        )

        # --- Conditionally apply per-instance normalization ---
        # This is applied BEFORE batching to normalize each sample individually.
        if self.config.normalize_per_instance:
            logger.info("Applying per-instance standardization to all datasets.")
            train_ds = train_ds.map(
                self._normalize_instance, num_parallel_calls=tf.data.AUTOTUNE
            )
            val_ds = val_ds.map(
                self._normalize_instance, num_parallel_calls=tf.data.AUTOTUNE
            )
            test_ds = test_ds.map(
                self._normalize_instance, num_parallel_calls=tf.data.AUTOTUNE
            )

        # --- Configure training dataset pipeline ---
        train_ds = train_ds.prefetch(10000).shuffle(
            self.config.batch_size * 100
        ).batch(self.config.batch_size)
        if (
            self.config.enable_additive_noise or
            self.config.enable_multiplicative_noise
        ):
            train_ds = train_ds.map(
                self._apply_noise_augmentation,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # --- Configure evaluation dataset pipelines ---
        val_steps = self.get_evaluation_steps(forecast_length, 'val')
        val_ds = val_ds.batch(
            self.config.batch_size
        ).prefetch(tf.data.AUTOTUNE)

        test_steps = self.get_evaluation_steps(forecast_length, 'test')
        test_ds = test_ds.batch(
            self.config.batch_size
        ).prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': val_steps,
            'test_steps': test_steps
        }

class PatternPerformanceCallback(keras.callbacks.Callback):
    """
    Callback for monitoring and visualizing performance.

    This callback logs training history and, at specified epoch intervals,
    generates and saves plots for learning curves and prediction samples.

    A key feature is the creation of a fixed, diverse set of visualization
    samples during initialization. This ensures that the prediction plots
    at each interval show the model's improving performance on the exact
    same inputs, providing a consistent and comparable view of learning progress.

    :param config: The training configuration object.
    :type config: NBeatsTrainingConfig
    :param data_processor: The data processor instance.
    :type data_processor: MultiPatternDataProcessor
    :param forecast_length: The forecast horizon length.
    :type forecast_length: int
    :param save_dir: Directory to save visualization plots.
    :type save_dir: str
    :param model_name: Name of the model, used for logging.
    :type model_name: str
    """

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            data_processor: MultiPatternDataProcessor,
            forecast_length: int,
            save_dir: str,
            model_name: str = "model"
    ):
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.forecast_length = forecast_length
        self.save_dir = save_dir
        self.model_name = model_name
        self.training_history = {'epoch': [], 'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        os.makedirs(save_dir, exist_ok=True)
        # Create a fixed, diverse test set for consistent visualization
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a small, fixed, and diverse test set for visualizations.

        To ensure a representative and varied set of examples, this method
        explicitly samples one sequence from N different time series patterns,
        where N is defined by `config.plot_top_k_patterns`. The selected
        patterns are shuffled to provide a random sample for each experiment run.
        This set is cached upon initialization and reused for all subsequent
        plotting, allowing for consistent comparison of model performance over
        epochs.

        :return: A tuple of NumPy arrays (X, y) for visualization.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        logger.info("Creating a diverse, fixed visualization test set...")
        num_samples_to_plot = self.config.plot_top_k_patterns
        patterns_to_sample_from = self.data_processor.selected_patterns.copy()
        random.shuffle(patterns_to_sample_from)

        X, y = [], []
        patterns_sampled = 0

        # Define the test split range
        start_ratio = self.config.train_ratio + self.config.val_ratio
        end_ratio = 1.0

        # Iterate through shuffled patterns to gather one sample from each
        for pattern_name in patterns_to_sample_from:
            if patterns_sampled >= num_samples_to_plot:
                break

            # 1. Generate the full time series for the current pattern
            data = self.data_processor.ts_generator.generate_task_data(pattern_name)

            # 2. Isolate the test portion
            start_idx = int(start_ratio * len(data))
            end_idx = int(end_ratio * len(data))
            test_data = data[start_idx:end_idx]

            # 3. Check if we can extract at least one sequence
            sequence_length = self.config.backcast_length + self.forecast_length
            if len(test_data) >= sequence_length:
                # 4. Pick a random valid start index within the test data
                max_start = len(test_data) - sequence_length
                sample_start_idx = random.randint(0, max_start)

                backcast = test_data[sample_start_idx: sample_start_idx + self.config.backcast_length]
                forecast = test_data[sample_start_idx + self.config.backcast_length: sample_start_idx + sequence_length]

                if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                    X.append(backcast.astype(np.float32))
                    y.append(forecast.astype(np.float32))
                    patterns_sampled += 1

        if not X:
            logger.warning("Could not generate any visualization samples.")
            return np.array([]), np.array([])

        logger.info(f"Successfully created visualization set with {len(X)} diverse samples.")
        return np.array(X), np.array(y)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if logs is None:
            logs = {}
        for key in self.training_history.keys():
            if key != 'epoch':
                self.training_history[key].append(logs.get(key, 0.0))
        self.training_history['epoch'].append(epoch)

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Creating visualizations for {self.model_name} at epoch {epoch + 1}")
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        try:
            if self.config.create_learning_curves: self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots: self._plot_prediction_samples(epoch)
        except Exception as e:
            logger.warning(f"Failed to create interim plots for {self.model_name}: {e}")

    def _plot_learning_curves(self, epoch: int) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = self.training_history['epoch']
        axes[0].plot(epochs, self.training_history['loss'], label='Training Loss')
        axes[0].plot(epochs, self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'Loss Curves (Epoch {epoch + 1})');
        axes[0].legend();
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, self.training_history['mae'], label='Training MAE')
        axes[1].plot(epochs, self.training_history['val_mae'], label='Validation MAE')
        axes[1].set_title(f'MAE Curves (Epoch {epoch + 1})');
        axes[1].legend();
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png'))
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        if self.viz_test_data[0].shape[0] == 0:
            return

        test_X, test_y = self.viz_test_data
        predictions = self.model(test_X, training=False)

        num_plots = min(len(test_X), self.config.plot_top_k_patterns)
        # Dynamic grid size based on number of plots
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            backcast_x = np.arange(-self.config.backcast_length, 0)
            forecast_x = np.arange(0, self.forecast_length)
            ax.plot(backcast_x, test_X[i], label='Input', color='blue')
            ax.plot(forecast_x, test_y[i], label='True', color='green')
            ax.plot(forecast_x, predictions[i], label='Predicted', color='red', linestyle='--')
            ax.set_title(f'Sample {i + 1}');
            ax.legend();
            ax.grid(True, alpha=0.3)

        # Turn off any unused subplots
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Prediction Samples (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'))
        plt.close()


class NBeatsTrainer:
    """
    Orchestrates the training, evaluation, and reporting of N-BEATS models.

    This class encapsulates the entire experiment lifecycle, including
    pattern selection, model creation, training loop execution for multiple
    horizons, and final result aggregation and saving.

    :param config: The main training configuration object.
    :type config: NBeatsTrainingConfig
    :param ts_config: Configuration for the time series generator.
    :type ts_config: TimeSeriesConfig
    """
    def __init__(self, config: NBeatsTrainingConfig, ts_config: TimeSeriesConfig) -> None:
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        # Create a pattern-to-category mapping for efficient lookup
        self.pattern_to_category = {
            task: category
            for category in self.pattern_categories
            for task in self.generator.get_tasks_by_category(category)
        }
        self.selected_patterns = self._select_patterns()
        self.processor = MultiPatternDataProcessor(
            config, self.generator, self.selected_patterns, self.pattern_to_category
        )

    def _select_patterns(self) -> List[str]:
        """
        Select time series patterns based on the configuration.

        This method filters patterns by target categories and applies limits
        per category and overall, ensuring a reproducible selection.

        :return: A list of selected pattern names for the experiment.
        :rtype: List[str]
        """
        selected = []
        if self.config.target_categories:
            patterns_to_consider = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            patterns_to_consider = self.all_patterns

        category_counts = {}
        for pattern in sorted(patterns_to_consider):  # Sort for reproducibility
            category = self.pattern_to_category.get(pattern)
            if category:
                if category_counts.get(category, 0) < self.config.max_patterns_per_category:
                    selected.append(pattern)
                    category_counts[category] = category_counts.get(category, 0) + 1

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = random.sample(selected, self.config.max_patterns)

        logger.info(f"Selected {len(selected)} patterns for training.")
        return selected

    def create_model(self, forecast_length: int) -> NBeatsNet:
        """
        Create and compile an N-BEATS Keras model.

        :param forecast_length: The forecast horizon for the model.
        :type forecast_length: int
        :return: A compiled NBeatsNet Keras model.
        :rtype: NBeatsNet
        """
        if self.config.kernel_regularizer_l2 > 0:
            kernel_regularizer = keras.regularizers.L2(
                self.config.kernel_regularizer_l2
            )
        else:
            kernel_regularizer = None

        return create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            use_revin=self.config.use_revin,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=self.config.dropout_rate,
            optimizer=self.config.optimizer,
            loss=self.config.primary_loss,
            learning_rate=self.config.learning_rate,
            gradient_clip_norm=self.config.gradient_clip_norm
        )

    def run_experiment(self) -> Dict[str, Any]:
        """
        Execute the full training and evaluation experiment.

        This method iterates through all specified forecast horizons, training
        a separate model for each. It manages result directories and saves
        all artifacts.

        :return: A dictionary containing the results directory and a nested
                 dictionary of results per horizon.
        :rtype: Dict[str, Any]
        """
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting N-BEATS Experiment: {exp_dir}")

        results = {}
        for horizon in self.config.forecast_horizons:
            logger.info(f"{'=' * 50}\nTraining Model for Horizon={horizon}\n{'=' * 50}")
            data_pipeline = self.processor.prepare_datasets(horizon)
            horizon_results = self._train_model(data_pipeline, horizon, exp_dir)
            results[horizon] = horizon_results

        self._save_results(results, exp_dir)
        logger.info("N-BEATS Experiment completed successfully.")
        return {"results_dir": exp_dir, "results": results}

    def _train_model(
        self, data_pipeline: Dict, horizon: int, exp_dir: str
    ) -> Dict[str, Any]:
        """
        Train and evaluate a single model for a specific horizon.

        :param data_pipeline: Dictionary of datasets and step counts.
        :type data_pipeline: Dict
        :param horizon: The forecast horizon for this model.
        :type horizon: int
        :param exp_dir: The main experiment directory for saving artifacts.
        :type exp_dir: str
        :return: A dictionary containing training history and test metrics.
        :rtype: Dict[str, Any]
        """
        model = self.create_model(horizon)
        model.build((None, self.config.backcast_length, 1))

        viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
        performance_cb = PatternPerformanceCallback(
            self.config, self.processor, horizon, viz_dir, "nbeats_model"
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(exp_dir, f'best_model_h{horizon}.keras'),
                save_best_only=True
            ),
            performance_cb
        ]

        logger.info(
            f"Training with {self.config.steps_per_epoch} steps/epoch and "
            f"validating with {data_pipeline['validation_steps']} steps."
        )
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
            verbose=0
        )
        test_loss = test_results[0]
        test_mae = test_results[1] if len(test_results) > 1 else None

        return {
            'history': history.history,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        """
        Save the final experiment results to a JSON file.

        :param results: A dictionary containing results for each horizon.
        :type results: Dict
        :param exp_dir: The directory to save the results file in.
        :type exp_dir: str
        """
        logger.info(f"Saving results to {exp_dir}")
        serializable_results = {}
        for h, r in results.items():
            # Convert numpy types to native python types for JSON
            history_serializable = {
                k: [float(v) for v in val] for k, val in r['history'].items()
            }
            serializable_results[h] = {
                'history': history_serializable,
                'test_loss': float(r['test_loss']),
                'test_mae': float(r['test_mae']) if r['test_mae'] else None,
                'final_epoch': r['final_epoch']
            }

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)


def main() -> None:
    """
    Main function to configure and run the N-BEATS training experiment.
    """
    config = NBeatsTrainingConfig(
        experiment_name="nbeats_streaming",
        backcast_length=104,
        forecast_horizons=[4],
        stack_types=["trend", "seasonality"],
        nb_blocks_per_stack=2,
        hidden_layer_units=256,
        use_revin=False,
        max_patterns_per_category=100,
        epochs=100,
        batch_size=64,
        steps_per_epoch=5000,
        learning_rate=1e-5,
        dropout_rate=0.1,
        kernel_regularizer_l2=1e-5,
    )
    ts_config = TimeSeriesConfig(n_samples=10000, random_seed=42)

    try:
        trainer = NBeatsTrainer(config, ts_config)
        trainer.run_experiment()
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()