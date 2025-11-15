"""
Comprehensive N-BEATS Training Framework for Multiple Time Series Patterns

This module provides a sophisticated, production-ready training framework for
N-BEATS models trained on multiple time series patterns. It enables training
N-BEATS models across diverse time series patterns with comprehensive monitoring,
visualization, and performance analysis.

Classes
-------
NBeatsTrainingConfig
    Comprehensive configuration dataclass containing all training parameters,
    including model architecture, training parameters, data management,
    regularization, and visualization settings.

MultiPatternDataProcessor
    Advanced data processing pipeline handling multi-pattern data preparation,
    sequence generation, balanced sampling, and proper train/validation/test
    splitting with temporal integrity. RevIN handles normalization.

PatternPerformanceCallback
    Comprehensive monitoring callback tracking performance across all patterns,
    creating detailed visualizations of training progress, pattern-specific
    performance, and learning dynamics.

NBeatsTrainer
    Main training orchestrator for multi-pattern N-BEATS training with
    comprehensive experiment management and performance analysis.

Training Approach
-----------------
The framework trains a single N-BEATS model on mixed patterns from multiple
time series categories. This approach:
* Learns general forecasting representations across diverse patterns
* Provides efficient deployment with a single model
* Achieves good generalization across unseen patterns
* Optimizes computational resources
* Applies configurable noise augmentation for improved robustness
* Shuffles data at every epoch for enhanced training stability
"""

import os
import json
import keras
import random
import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss, mase_metric
from dl_techniques.models.nbeats.model import create_nbeats_model, NBeatsNet
# NOTE: TimeSeriesNormalizer is no longer needed as RevIN handles normalization
from dl_techniques.datasets.time_series import TimeSeriesGenerator, TimeSeriesConfig

# ---------------------------------------------------------------------


plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed value, by default 42
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
    Configuration for N-BEATS training with multiple patterns.

    This dataclass contains comprehensive configuration options for training
    N-BEATS models on diverse time series patterns with various regularization,
    optimization, and visualization settings.
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
    stack_types: List[str] = field(default_factory=lambda: ["trend", "seasonality", "generic"])
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 256
    use_revin: bool = True
    use_bias: bool = True

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    primary_loss: str = "mae"

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    dropout_rate: float = 0.15

    # Pattern selection and balancing
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    min_data_length: int = 2000
    balance_patterns: bool = True
    samples_per_pattern: int = 25000

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

    # Evaluation configuration
    eval_during_training: bool = True
    eval_every_n_epochs: int = 10

    # Data augmentation configuration
    multiplicative_noise_std: float = 0.01
    additive_noise_std: float = 0.01
    enable_multiplicative_noise: bool = True
    enable_additive_noise: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.

        Raises
        ------
        ValueError
            If configuration parameters are invalid
        """
        # Validate data ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        # Validate basic parameters
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")

        if not self.use_revin:
            logger.warning("RevIN is disabled. Consider enabling it for better performance. If disabled, external normalization is required.")

        if self.val_ratio < 0.1:
            logger.warning(f"Validation ratio {self.val_ratio} might be too small for reliable validation")

        logger.info(f"N-BEATS Training Configuration:")
        logger.info(f"  ✅ Data split: {self.train_ratio:.1f}/{self.val_ratio:.1f}/{self.test_ratio:.1f}")
        logger.info(f"  ✅ Model: {self.nb_blocks_per_stack} blocks, {self.hidden_layer_units} units")
        logger.info(f"  ✅ Training: {self.epochs} epochs, batch {self.batch_size}, lr {self.learning_rate}")
        logger.info(f"  ✅ Regularization: dropout {self.dropout_rate}, L2 {self.kernel_regularizer_l2}")
        logger.info(f"  ✅ Augmentation: mult_noise {self.multiplicative_noise_std if self.enable_multiplicative_noise else 'disabled'}, add_noise {self.additive_noise_std if self.enable_additive_noise else 'disabled'}")

        if self.target_categories:
            logger.info(f"  ✅ Target categories: {self.target_categories}")


class MultiPatternDataProcessor:
    """
    Advanced data processor for multiple pattern training.

    This class handles the preparation of time series data from multiple patterns,
    including sequence generation, and proper train/validation/test
    splitting while maintaining temporal integrity. Normalization is handled by RevIN.

    Parameters
    ----------
    config : NBeatsTrainingConfig
        Configuration object containing data processing parameters

    Attributes
    ----------
    config : NBeatsTrainingConfig
        Configuration object
    pattern_to_id : Dict[str, int]
        Mapping from pattern names to integer IDs
    id_to_pattern : Dict[int, str]
        Mapping from integer IDs to pattern names
    """

    def __init__(self, config: NBeatsTrainingConfig) -> None:
        self.config = config
        self.pattern_to_id: Dict[str, int] = {}
        self.id_to_pattern: Dict[int, str] = {}

    def prepare_multi_pattern_data(
            self,
            raw_pattern_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Prepare multi-pattern data for training.

        This method combines data from multiple time series patterns into
        a single training dataset while maintaining pattern diversity and
        temporal structure.

        Parameters
        ----------
        raw_pattern_data : Dict[str, np.ndarray]
            Dictionary mapping pattern names to time series data arrays

        Returns
        -------
        Dict[str, Any]
            Dictionary containing prepared training data for each forecast horizon
        """
        logger.info(f"Preparing data for multi-pattern training...")

        # Create pattern ID mapping
        self.pattern_to_id = {pattern: idx for idx, pattern in enumerate(raw_pattern_data.keys())}
        self.id_to_pattern = {idx: pattern for pattern, idx in self.pattern_to_id.items()}

        # NOTE: Scaler fitting is removed because RevIN handles normalization.

        prepared_data = {}

        for horizon in self.config.forecast_horizons:
            logger.info(f"Preparing data for horizon {horizon}")
            prepared_data[horizon] = self._prepare_mixed_pattern_data(raw_pattern_data, horizon)

        return prepared_data

    def _prepare_mixed_pattern_data(
            self,
            raw_pattern_data: Dict[str, np.ndarray],
            horizon: int
    ) -> Dict[str, tf.data.Dataset]:
        """
        Prepare data by mixing multiple patterns with a proper shuffle.

        This version includes a full shuffle of the training data at the NumPy
        level before creating the TensorFlow dataset. This prevents the loss
        artifact where each epoch starts with "easy" patterns from the shuffle
        buffer.

        Parameters
        ----------
        raw_pattern_data : Dict[str, np.ndarray]
            Dictionary of pattern data
        horizon : int
            Forecast horizon length

        Returns
        -------
        Dict[str, tf.data.Dataset]
            Dictionary containing combined training datasets
        """
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []

        for pattern_name, data in raw_pattern_data.items():
            try:
                min_length = self.config.backcast_length + horizon + 100
                if len(data) < min_length:
                    continue

                # Split data temporally
                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))

                train_data = data[:train_size]
                val_data = data[train_size:train_size + val_size]
                test_data = data[train_size + val_size:]

                # Create sequences from raw data splits
                train_X, train_y = self._create_sequences(train_data, horizon, stride=1)
                val_X, val_y = self._create_sequences(val_data, horizon, stride=horizon // 2)
                test_X, test_y = self._create_sequences(test_data, horizon, stride=horizon // 2)

                # Balance data if needed
                if self.config.balance_patterns and len(train_X) > self.config.samples_per_pattern:
                    step = max(1, len(train_X) // self.config.samples_per_pattern)
                    indices = np.arange(0, len(train_X), step)[:self.config.samples_per_pattern]
                    train_X = train_X[indices]
                    train_y = train_y[indices]

                all_train_X.append(train_X)
                all_train_y.append(train_y)
                all_val_X.append(val_X)
                all_val_y.append(val_y)
                all_test_X.append(test_X)
                all_test_y.append(test_y)

            except Exception as e:
                logger.warning(f"Failed to prepare {pattern_name} H={horizon}: {e}")
                continue

        if not all_train_X:
            raise ValueError(f"No data prepared for horizon {horizon}")

        # Combine all patterns
        combined_train_X = np.concatenate(all_train_X, axis=0)
        combined_train_y = np.concatenate(all_train_y, axis=0)
        combined_val_X = np.concatenate(all_val_X, axis=0)
        combined_val_y = np.concatenate(all_val_y, axis=0)
        combined_test_X = np.concatenate(all_test_X, axis=0)
        combined_test_y = np.concatenate(all_test_y, axis=0)

        # --------------------------------------------------------------------
        # --- FIX: SHUFFLE THE TRAINING DATASET AT THE NUMPY LEVEL ---
        # This is the crucial step to ensure the tf.data.Dataset shuffle buffer
        # is filled with a representative sample of the data at the start of
        # each epoch, preventing the observed loss-reset behavior.
        logger.info(f"Shuffling {len(combined_train_X)} training samples before creating dataset.")

        # Create a single random permutation of indices
        p = np.random.permutation(len(combined_train_X))

        # Apply the same permutation to both X and y to maintain alignment
        combined_train_X = combined_train_X[p]
        combined_train_y = combined_train_y[p]
        # --------------------------------------------------------------------

        # Create TensorFlow datasets
        train_dataset = self._create_tf_dataset(
            combined_train_X, combined_train_y,
            batch_size=self.config.batch_size,
            shuffle=True,  # Still recommended to keep tf.data shuffle for extra randomness
            apply_augmentation=True
        )

        val_dataset = self._create_tf_dataset(
            combined_val_X, combined_val_y,
            batch_size=self.config.batch_size,
            shuffle=False,  # Validation data should not be shuffled
            apply_augmentation=False
        )

        test_dataset = self._create_tf_dataset(
            combined_test_X, combined_test_y,
            batch_size=self.config.batch_size,
            shuffle=False,  # Test data should not be shuffled
            apply_augmentation=False
        )

        return {
            'mixed_patterns': {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset,
                # Keep raw arrays for callback visualization
                'test_arrays': (combined_test_X, combined_test_y)
            }
        }

    def _create_sequences(
            self,
            data: np.ndarray,
            forecast_length: int,
            stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with specified stride.

        Parameters
        ----------
        data : np.ndarray
            Time series data
        forecast_length : int
            Length of forecast horizon
        stride : int, optional
            Stride for sequence creation, by default 1

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Input sequences and target sequences
        """
        X, y = [], []

        for i in range(0, len(data) - self.config.backcast_length - forecast_length + 1, stride):
            backcast = data[i: i + self.config.backcast_length]
            forecast = data[i + self.config.backcast_length: i + self.config.backcast_length + forecast_length]

            if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                X.append(backcast)
                y.append(forecast)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _apply_noise_augmentation(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply multiplicative and additive noise augmentation to input data.

        Parameters
        ----------
        x : tf.Tensor
            Input sequences
        y : tf.Tensor
            Target sequences

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Augmented input and target sequences
        """
        # Apply noise only to input sequences (x), not targets (y)
        augmented_x = x

        if self.config.enable_multiplicative_noise and self.config.multiplicative_noise_std > 0:
            # Multiplicative noise: x * (1 + noise)
            mult_noise = tf.random.normal(
                tf.shape(x),
                mean=1.0,
                stddev=self.config.multiplicative_noise_std,
                dtype=x.dtype
            )
            augmented_x = augmented_x * mult_noise

        if self.config.enable_additive_noise and self.config.additive_noise_std > 0:
            # Additive noise: x + noise
            add_noise = tf.random.normal(
                tf.shape(augmented_x),
                mean=0.0,
                stddev=self.config.additive_noise_std,
                dtype=augmented_x.dtype
            )
            augmented_x = augmented_x + add_noise

        return augmented_x, y

    def _create_tf_dataset(
            self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            shuffle: bool = True,
            apply_augmentation: bool = False
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with optional augmentation.

        Parameters
        ----------
        X : np.ndarray
            Input sequences
        y : np.ndarray
            Target sequences
        batch_size : int
            Batch size
        shuffle : bool, optional
            Whether to shuffle data, by default True
        apply_augmentation : bool, optional
            Whether to apply noise augmentation, by default False

        Returns
        -------
        tf.data.Dataset
            Configured TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            # Use large buffer size for thorough shuffling
            buffer_size = min(10000, len(X))
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size, drop_remainder=False)

        if apply_augmentation:
            dataset = dataset.map(
                self._apply_noise_augmentation,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


class PatternPerformanceCallback(keras.callbacks.Callback):
    """
    Comprehensive callback for monitoring pattern-specific performance.

    This callback tracks training progress across multiple patterns and creates
    detailed visualizations of learning dynamics, prediction quality, and
    performance metrics.

    Parameters
    ----------
    config : NBeatsTrainingConfig
        Configuration object containing visualization settings
    data_processor : MultiPatternDataProcessor
        Data processor containing pattern information
    test_data : Dict[int, Any]
        Test data for visualization
    save_dir : str
        Directory to save visualization plots
    model_name : str, optional
        Name of the model for identification, by default "model"

    Attributes
    ----------
    config : NBeatsTrainingConfig
        Configuration object
    data_processor : MultiPatternDataProcessor
        Data processor instance
    test_data : Dict[int, Any]
        Test data dictionary
    save_dir : str
        Directory for saving plots
    model_name : str
        Model identifier
    training_history : Dict[str, List]
        Training history tracking
    """

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            data_processor: MultiPatternDataProcessor,
            test_data: Dict[int, Any],
            save_dir: str,
            model_name: str = "model"
    ) -> None:
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.test_data = test_data
        self.save_dir = save_dir
        self.model_name = model_name

        self.training_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': []
        }

        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Track training progress and create visualizations.

        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : Optional[Dict[str, float]], optional
            Training logs, by default None
        """
        if logs is None:
            logs = {}

        self.training_history['epoch'].append(epoch)
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))
        self.training_history['mae'].append(logs.get('mae', 0.0))
        self.training_history['val_mae'].append(logs.get('val_mae', 0.0))

        # Create visualizations at specified intervals
        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Creating visualizations for {self.model_name} at epoch {epoch + 1}")
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        """
        Create comprehensive interim plots.

        Parameters
        ----------
        epoch : int
            Current epoch number
        """
        try:
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)

            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)

        except Exception as e:
            logger.warning(f"Failed to create interim plots for {self.model_name}: {e}")

    def _plot_learning_curves(self, epoch: int) -> None:
        """
        Plot training and validation curves.

        Parameters
        ----------
        epoch : int
            Current epoch number
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        epochs = self.training_history['epoch']

        # Loss curves
        axes[0].plot(epochs, self.training_history['loss'], label='Training Loss', color='blue', linewidth=2)
        axes[0].plot(epochs, self.training_history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0].set_title(f'{self.model_name} - Loss Curves (Epoch {epoch + 1})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE curves
        axes[1].plot(epochs, self.training_history['mae'], label='Training MAE', color='green', linewidth=2)
        axes[1].plot(epochs, self.training_history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
        axes[1].set_title(f'{self.model_name} - MAE Curves (Epoch {epoch + 1})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        """
        Plot sample predictions.

        Parameters
        ----------
        epoch : int
            Current epoch number
        """
        horizon = self.config.forecast_horizons[0]

        if horizon not in self.test_data:
            return

        test_data = self.test_data[horizon]
        mixed_data = test_data['mixed_patterns']

        # Use test arrays for visualization
        if 'test_arrays' not in mixed_data:
            return

        test_X, test_y = mixed_data['test_arrays']

        if len(test_X) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Sample different parts of the test set
        sample_indices = np.linspace(0, len(test_X) - 1, 6, dtype=int)

        for i, sample_idx in enumerate(sample_indices):
            sample_X = test_X[sample_idx:sample_idx + 1]
            sample_y = test_y[sample_idx:sample_idx + 1]

            # Get prediction
            pred_y = self.model(sample_X, training=False)

            # Plot
            backcast_x = np.arange(-self.config.backcast_length, 0)
            forecast_x = np.arange(0, self.config.forecast_length)

            axes[i].plot(backcast_x, sample_X[0], label='Input', color='blue', alpha=0.7)
            axes[i].plot(forecast_x, sample_y[0].flatten(), label='True', color='green', linewidth=2)
            axes[i].plot(forecast_x, pred_y[0].numpy().flatten(), label='Predicted', color='red', linewidth=2)

            axes[i].set_title(f'Sample {i + 1}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.5)

        plt.suptitle(f'Prediction Samples (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


class NBeatsTrainer:
    """
    Comprehensive trainer for N-BEATS with multiple pattern support.

    This class orchestrates the complete training process for N-BEATS models
    on multiple time series patterns, including data preparation, model training,
    evaluation, and result visualization.

    Parameters
    ----------
    config : NBeatsTrainingConfig
        Configuration object containing all training parameters
    ts_config : TimeSeriesConfig
        Time series generation configuration

    Attributes
    ----------
    config : NBeatsTrainingConfig
        Training configuration
    ts_config : TimeSeriesConfig
        Time series configuration
    generator : TimeSeriesGenerator
        Time series pattern generator
    processor : MultiPatternDataProcessor
        Data processing pipeline
    all_patterns : List[str]
        All available pattern names
    pattern_categories : List[str]
        All available pattern categories
    selected_patterns : List[str]
        Selected patterns for training
    """

    def __init__(self, config: NBeatsTrainingConfig, ts_config: TimeSeriesConfig) -> None:
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = MultiPatternDataProcessor(config)

        # Get patterns
        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        self.selected_patterns = self._select_patterns()

        logger.info(f"N-BEATS Trainer initialized:")
        logger.info(f"  - Available categories: {len(self.pattern_categories)}")
        logger.info(f"  - Total patterns available: {len(self.all_patterns)}")
        logger.info(f"  - Selected {len(self.selected_patterns)} patterns")
        logger.info(f"  - Category distribution: {self._get_category_distribution()}")

    def _select_patterns(self) -> List[str]:
        """
        Select patterns based on configuration.

        Returns
        -------
        List[str]
            List of selected pattern names
        """
        selected = []

        # Handle target categories
        if self.config.target_categories:
            for category in self.config.target_categories:
                category_patterns = self.generator.get_tasks_by_category(category)
                max_patterns = self.config.max_patterns_per_category

                if len(category_patterns) <= max_patterns:
                    selected.extend(category_patterns)
                else:
                    weight = self.config.category_weights.get(category, 1.0)
                    adjusted_max = min(int(max_patterns * weight), len(category_patterns))
                    selected_from_category = np.random.choice(
                        category_patterns, size=adjusted_max, replace=False
                    ).tolist()
                    selected.extend(selected_from_category)
        else:
            # General pattern selection
            for category in self.pattern_categories:
                category_patterns = self.generator.get_tasks_by_category(category)
                weight = self.config.category_weights.get(category, 1.0)
                max_patterns = min(int(self.config.max_patterns_per_category * weight), len(category_patterns))

                if len(category_patterns) <= max_patterns:
                    selected.extend(category_patterns)
                else:
                    selected_from_category = np.random.choice(
                        category_patterns, size=max_patterns, replace=False
                    ).tolist()
                    selected.extend(selected_from_category)

        # Apply global pattern limit if specified
        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = np.random.choice(selected, size=self.config.max_patterns, replace=False).tolist()

        return selected

    def _get_category_distribution(self) -> Dict[str, int]:
        """
        Get distribution of selected patterns by category.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping categories to pattern counts
        """
        distribution = {}
        for pattern in self.selected_patterns:
            category = None
            for cat in self.pattern_categories:
                if pattern in self.generator.get_tasks_by_category(cat):
                    category = cat
                    break
            if category:
                distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare training data for the model.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing prepared data and metadata
        """
        logger.info("Generating data for selected patterns...")

        raw_pattern_data = {}
        generation_failures = []

        for pattern_name in self.selected_patterns:
            try:
                data = self.generator.generate_task_data(pattern_name)
                if len(data) >= self.config.min_data_length:
                    raw_pattern_data[pattern_name] = data
                else:
                    logger.warning(
                        f"Generated data for {pattern_name} too short: {len(data)} < {self.config.min_data_length}")
                    generation_failures.append(pattern_name)
            except Exception as e:
                logger.warning(f"Failed to generate {pattern_name}: {e}")
                generation_failures.append(pattern_name)

        logger.info(f"Successfully generated data for {len(raw_pattern_data)} patterns")
        if generation_failures:
            logger.info(f"Failed to generate {len(generation_failures)} patterns: {generation_failures[:5]}...")

        prepared_data = self.processor.prepare_multi_pattern_data(raw_pattern_data)

        return {
            'prepared_data': prepared_data,
            'raw_pattern_data': raw_pattern_data,
            'num_patterns': len(raw_pattern_data),
            'pattern_to_id': self.processor.pattern_to_id,
            'generation_failures': generation_failures,
            'category_distribution': self._get_final_category_distribution(raw_pattern_data)
        }

    def _get_final_category_distribution(self, raw_pattern_data: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Get final category distribution after data generation.

        Parameters
        ----------
        raw_pattern_data : Dict[str, np.ndarray]
            Generated pattern data

        Returns
        -------
        Dict[str, int]
            Final category distribution
        """
        distribution = {}
        for pattern in raw_pattern_data.keys():
            category = None
            for cat in self.pattern_categories:
                if pattern in self.generator.get_tasks_by_category(cat):
                    category = cat
                    break
            if category:
                distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def create_model(self, forecast_length: int) -> NBeatsNet:
        """
        Create N-BEATS model with enhanced configuration.

        Parameters
        ----------
        forecast_length : int
            Length of forecast horizon

        Returns
        -------
        NBeatsNet
            Configured N-BEATS model instance
        """
        kernel_regularizer = keras.regularizers.L2(
            self.config.kernel_regularizer_l2) if self.config.kernel_regularizer_l2 > 0 else None

        return create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types.copy(),
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
        Run the multi-pattern experiment.

        Returns
        -------
        Dict[str, Any]
            Comprehensive experiment results

        Raises
        ------
        ValueError
            If no data is prepared for training
        """
        try:
            exp_dir = os.path.join(
                self.config.result_dir,
                f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(exp_dir, exist_ok=True)

            logger.info(f"🚀 Starting N-BEATS Experiment: {exp_dir}")

            # Prepare data
            data_info = self.prepare_data()
            prepared_data = data_info['prepared_data']

            if not prepared_data:
                raise ValueError("No data prepared for training")

            results = {}

            for horizon in self.config.forecast_horizons:
                if horizon not in prepared_data:
                    continue

                logger.info(f"{'=' * 50}")
                logger.info(f"🎯 Training Model H={horizon}")
                logger.info(f"{'=' * 50}")

                horizon_results = self._train_model(prepared_data[horizon], horizon, exp_dir)
                results[horizon] = horizon_results

            # Save comprehensive results
            self._save_results(results, exp_dir, data_info)

            logger.info("🎉 N-BEATS Experiment completed successfully!")
            return {
                "results_dir": exp_dir,
                "results": results,
                "num_patterns": data_info['num_patterns'],
                "pattern_mapping": data_info['pattern_to_id'],
                "category_distribution": data_info['category_distribution']
            }

        except Exception as e:
            logger.error(f"💥 experiment failed: {e}", exc_info=True)
            raise

    def _train_model(self, pattern_data: Dict, horizon: int, exp_dir: str) -> Dict[str, Any]:
        """
        Train model on mixed patterns.

        Parameters
        ----------
        pattern_data : Dict
            Prepared pattern data
        horizon : int
            Forecast horizon
        exp_dir : str
            Experiment directory

        Returns
        -------
        Dict[str, Any]
            Training results
        """
        logger.info("Training model on mixed patterns")

        data = pattern_data['mixed_patterns']

        # Create model
        model = self.create_model(horizon)

        # Build model with sample data from test arrays
        test_X, _ = data['test_arrays']
        model(test_X[:1])  # Build with sample data

        # Create callbacks
        viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
        callback = PatternPerformanceCallback(
            config=self.config,
            data_processor=self.processor,
            test_data={horizon: pattern_data},
            save_dir=viz_dir,
            model_name="nbeats_model"
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=30,
                min_lr=1e-6,
                verbose=1
            ),
            callback,
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(exp_dir, f'best_model_h{horizon}.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model using tf.data.Dataset
        start_time = datetime.now()

        train_dataset = data['train']
        val_dataset = data['val']

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate using test dataset
        test_dataset = data['test']
        test_results = model.evaluate(test_dataset, verbose=0)
        test_loss = test_results[0] if isinstance(test_results, list) else test_results
        test_mae = test_results[1] if len(test_results) > 1 else None

        logger.info(f"✅ Model: Test Loss = {test_loss:.4f}, Time = {training_time:.1f}s")

        return {
            'model_results': {
                'history': history.history,
                'training_time': training_time,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'final_epoch': len(history.history['loss'])
            }
        }

    def _save_results(self, results: Dict, exp_dir: str, data_info: Dict) -> None:
        """
        Save comprehensive experiment results.

        Parameters
        ----------
        results : Dict
            Experiment results
        exp_dir : str
            Experiment directory
        data_info : Dict
            Data information
        """
        try:
            # Save JSON results
            with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
                json_results = {}
                for horizon, result in results.items():
                    json_results[str(horizon)] = self._serialize_results(result)
                json.dump(json_results, f, indent=2)

            # Save experiment information
            with open(os.path.join(exp_dir, 'experiment_info.json'), 'w') as f:
                json.dump({
                    'num_patterns': data_info['num_patterns'],
                    'pattern_to_id': data_info['pattern_to_id'],
                    'selected_patterns': self.selected_patterns,
                    'category_distribution': data_info['category_distribution'],
                    'category_weights': self.config.category_weights,
                    'generation_failures': data_info.get('generation_failures', []),
                    'config': {
                        'backcast_length': self.config.backcast_length,
                        'forecast_horizons': self.config.forecast_horizons,
                        'stack_types': self.config.stack_types,
                        'nb_blocks_per_stack': self.config.nb_blocks_per_stack,
                        'hidden_layer_units': self.config.hidden_layer_units,
                        'epochs': self.config.epochs,
                        'batch_size': self.config.batch_size,
                        'learning_rate': self.config.learning_rate,
                        'optimizer': self.config.optimizer,
                        'primary_loss': self.config.primary_loss
                    }
                }, f, indent=2)

            # Create comprehensive summary plots
            self._create_comprehensive_summary_plots(results, exp_dir)

            # Create detailed experiment report
            self._create_detailed_experiment_report(results, exp_dir, data_info)

            logger.info(f"Results saved to {exp_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _serialize_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize results for JSON storage.

        Parameters
        ----------
        result : Dict[str, Any]
            Results to serialize

        Returns
        -------
        Dict[str, Any]
            Serialized results
        """
        serialized = {}

        for key, value in result.items():
            if key == 'model_results':
                serialized[key] = {}
                for metric_key, metric_value in value.items():
                    if metric_key == 'history':
                        serialized[key][metric_key] = {
                            hist_key: [float(v) for v in hist_value] if isinstance(hist_value,
                                                                                   (list, np.ndarray)) else hist_value
                            for hist_key, hist_value in metric_value.items()
                        }
                    else:
                        serialized[key][metric_key] = float(metric_value) if isinstance(metric_value, (np.floating,
                                                                                                       np.integer)) else metric_value
            else:
                serialized[key] = value

        return serialized

    def _create_comprehensive_summary_plots(self, results: Dict, exp_dir: str) -> None:
        """
        Create comprehensive summary visualization.

        Parameters
        ----------
        results : Dict
            Experiment results
        exp_dir : str
            Experiment directory
        """
        try:
            for horizon, result in results.items():
                self._plot_model_summary(result, horizon, exp_dir)
        except Exception as e:
            logger.warning(f"Failed to create summary plots: {e}")

    def _plot_model_summary(self, results: Dict, horizon: int, exp_dir: str) -> None:
        """
        Plot summary for model.

        Parameters
        ----------
        results : Dict
            Model results
        horizon : int
            Forecast horizon
        exp_dir : str
            Experiment directory
        """
        model_results = results.get('model_results', {})

        if not model_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        history = model_results.get('history', {})

        # Training curves
        if 'loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['loss']) + 1)
            axes[0, 0].plot(epochs, history['loss'], label='Training Loss', linewidth=2)
            axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].set_title(f'Training Curves (H={horizon})')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # MAE curves
        if 'mae' in history and 'val_mae' in history:
            axes[0, 1].plot(epochs, history['mae'], label='Training MAE', linewidth=2)
            axes[0, 1].plot(epochs, history['val_mae'], label='Validation MAE', linewidth=2)
            axes[0, 1].set_title(f'MAE Curves (H={horizon})')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Performance metrics
        test_loss = model_results.get('test_loss', 0)
        test_mae = model_results.get('test_mae', 0)
        training_time = model_results.get('training_time', 0)
        final_epoch = model_results.get('final_epoch', 0)

        metrics = ['Test Loss', 'Test MAE', 'Training Time (s)', 'Final Epoch']
        values = [test_loss, test_mae, training_time, final_epoch]

        axes[1, 0].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], alpha=0.7,
                       edgecolor='black')
        axes[1, 0].set_title(f'Performance Metrics (H={horizon})')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[1, 0].text(i, v, f'{v:.3f}' if v < 100 else f'{v:.0f}', ha='center', va='bottom')

        # Summary text
        axes[1, 1].text(0.1, 0.8, f'Model Summary (H={horizon})', fontsize=14, fontweight='bold',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Test Loss: {test_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Test MAE: {test_mae:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Training Time: {training_time:.1f}s', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Final Epoch: {final_epoch}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f'Multi-Pattern Training', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

        plt.suptitle(f'Model Summary (H={horizon})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'model_summary_h{horizon}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _create_detailed_experiment_report(self, results: Dict, exp_dir: str, data_info: Dict) -> None:
        """
        Create detailed text report of experiment results.

        Parameters
        ----------
        results : Dict
            Experiment results
        exp_dir : str
            Experiment directory
        data_info : Dict
            Data information
        """
        try:
            report_path = os.path.join(exp_dir, 'detailed_experiment_report.txt')

            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("N-BEATS MULTI-PATTERN EXPERIMENT REPORT\n")
                f.write("=" * 80 + "\n\n")

                # Experiment overview
                f.write("EXPERIMENT OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Experiment Name: {self.config.experiment_name}\n")
                f.write(f"Number of Patterns: {data_info['num_patterns']}\n")
                f.write(f"Available Pattern Categories: {len(self.pattern_categories)}\n")
                f.write(f"Time Series Length: {self.ts_config.n_samples}\n")
                f.write(f"Forecast Horizons: {self.config.forecast_horizons}\n")
                f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if self.config.target_categories:
                    f.write("TARGET CATEGORIES\n")
                    f.write("-" * 17 + "\n")
                    f.write(f"Target Categories: {self.config.target_categories}\n\n")

                # Pattern distribution
                f.write("PATTERN DISTRIBUTION BY CATEGORY\n")
                f.write("-" * 33 + "\n")
                for category, count in data_info['category_distribution'].items():
                    weight = self.config.category_weights.get(category, 1.0)
                    f.write(f"{category:15s}: {count:2d} patterns (weight: {weight:.1f})\n")
                f.write(f"Total Categories Used: {len(data_info['category_distribution'])}\n")
                f.write(f"Total Patterns Generated: {sum(data_info['category_distribution'].values())}\n\n")

                # Model configuration
                f.write("MODEL CONFIGURATION\n")
                f.write("-" * 19 + "\n")
                f.write(f"Backcast Length: {self.config.backcast_length}\n")
                f.write(f"Stack Types: {self.config.stack_types}\n")
                f.write(f"Blocks per Stack: {self.config.nb_blocks_per_stack}\n")
                f.write(f"Hidden Layer Units: {self.config.hidden_layer_units}\n")
                f.write(f"RevIN Normalization: {'Enabled' if self.config.use_revin else 'Disabled'}\n")
                f.write(f"Dropout Rate: {self.config.dropout_rate}\n")
                f.write(f"L2 Regularization: {self.config.kernel_regularizer_l2}\n\n")

                # Training configuration
                f.write("TRAINING CONFIGURATION\n")
                f.write("-" * 21 + "\n")
                f.write(
                    f"Data Split: {self.config.train_ratio:.2f}/{self.config.val_ratio:.2f}/{self.config.test_ratio:.2f}\n")
                f.write(f"Epochs: {self.config.epochs}\n")
                f.write(f"Batch Size: {self.config.batch_size}\n")
                f.write(f"Learning Rate: {self.config.learning_rate}\n")
                f.write(f"Optimizer: {self.config.optimizer}\n")
                f.write(f"Primary Loss: {self.config.primary_loss}\n")
                f.write(f"Gradient Clipping: {self.config.gradient_clip_norm}\n\n")

                # Results for each horizon
                for horizon, result in results.items():
                    f.write(f"RESULTS FOR HORIZON {horizon}\n")
                    f.write("-" * 25 + "\n")

                    model_results = result.get('model_results', {})
                    if model_results:
                        f.write(f"Test Loss: {model_results['test_loss']:.6f}\n")
                        f.write(f"Test MAE: {model_results.get('test_mae', 'N/A')}\n")
                        f.write(f"Training Time: {model_results['training_time']:.1f} seconds\n")
                        f.write(f"Final Epoch: {model_results['final_epoch']}\n")

                    f.write("\n")

                # Pattern mapping (sample)
                f.write("PATTERN MAPPING (Sample)\n")
                f.write("-" * 22 + "\n")
                sample_patterns = list(data_info['pattern_to_id'].items())[:20]
                for pattern_name, pattern_id in sample_patterns:
                    f.write(f"Pattern {pattern_id:2d}: {pattern_name}\n")
                if len(data_info['pattern_to_id']) > 20:
                    f.write(f"... and {len(data_info['pattern_to_id']) - 20} more patterns\n")
                f.write("\n")

                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                f.write("✅ N-BEATS training completed successfully\n")
                f.write(f"✅ Successfully processed {len(data_info['category_distribution'])} pattern categories\n")
                f.write(
                    f"✅ Generated {sum(data_info['category_distribution'].values())} diverse time series patterns\n")
                f.write("✅ Model provides efficient deployment with good generalization\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")

            logger.info(f"Detailed experiment report saved to {report_path}")

        except Exception as e:
            logger.warning(f"Failed to create detailed experiment report: {e}")


def main() -> None:
    """Run the N-BEATS experiment."""

    config = NBeatsTrainingConfig(
        experiment_name="nbeats",

        backcast_length=104,
        forecast_length=4,
        forecast_horizons=[4],

        stack_types=["trend", "seasonality", "generic"],
        nb_blocks_per_stack=3,
        hidden_layer_units=256,
        use_revin=True,

        max_patterns_per_category=15,
        min_data_length=2000,
        balance_patterns=True,
        samples_per_pattern=25000,

        epochs=200,
        batch_size=128,
        learning_rate=1e-3,
        dropout_rate=0.15,
        kernel_regularizer_l2=1e-5,
        gradient_clip_norm=1.0,
        optimizer='adamw',
        primary_loss="mae",

        visualize_every_n_epochs=10,
        save_interim_plots=True,
        plot_top_k_patterns=6,
        create_learning_curves=True,
        create_prediction_plots=True,

        # Data augmentation
        multiplicative_noise_std=0.01,
        additive_noise_std=0.001,
        enable_multiplicative_noise=True,
        enable_additive_noise=True
    )

    ts_config = TimeSeriesConfig(
        n_samples=20000,
        random_seed=42,
        default_noise_level=0.01
    )

    try:
        logger.info("🚀 Running N-BEATS Multi-Pattern Experiment")
        trainer = NBeatsTrainer(config, ts_config)
        results = trainer.run_experiment()
        logger.info(f"✅ Experiment completed: {results['results_dir']}")
    except Exception as e:
        logger.error(f"💥 Experiment failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()