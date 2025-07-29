"""
Comprehensive Base N-BEATS Training Framework for Multiple Time Series Patterns

This module provides a sophisticated, production-ready training framework for base
N-BEATS models trained on multiple time series patterns. It enables training
individual or ensemble N-BEATS models across diverse time series patterns
with comprehensive monitoring, visualization, and performance analysis.

Classes
-------
BaseNBeatsTrainingConfig
    Comprehensive configuration dataclass containing all training parameters,
    including model architecture, training strategies, data management,
    regularization, and visualization settings.

MultiPatternDataProcessor
    Advanced data processing pipeline handling multi-pattern data preparation,
    sequence generation, normalization, balanced sampling, and proper
    train/validation/test splitting with temporal integrity.

PatternPerformanceCallback
    Comprehensive monitoring callback tracking performance across all patterns,
    creating detailed visualizations of training progress, pattern-specific
    performance, and learning dynamics.

BaseNBeatsTrainer
    Main training orchestrator supporting multiple training strategies including
    individual models per pattern, unified multi-pattern models, and ensemble
    approaches with comprehensive experiment management.

Training Strategies
-------------------
**Individual Models Strategy:**
* Train separate N-BEATS models for each time series pattern
* Optimal for pattern-specific optimization
* Enables specialized forecasting for each pattern type
* Supports ensemble predictions across models

**Unified Model Strategy:**
* Train single N-BEATS model on mixed patterns
* Learns general forecasting representations
* Efficient deployment with single model
* Good generalization across unseen patterns

**Ensemble Strategy:**
* Train multiple models with different configurations
* Combine predictions for improved robustness
* Reduces prediction variance
* Optimal for critical forecasting applications

**Pattern-Specific Strategy:**
* Focus training on specific pattern categories
* Ideal for domain-specific applications
* Enhanced performance on target pattern types
* Reduced model complexity

Usage Examples
--------------
Basic Multi-Pattern Training:
    >>> # Configure training
    >>> config = NBeatsTrainingConfig(
    ...     backcast_length=168,
    ...     forecast_length=24,
    ...     training_strategy='unified',
    ...     epochs=100,
    ...     batch_size=128
    ... )
    >>>
    >>> # Configure time series generation
    >>> ts_config = TimeSeriesConfig(n_samples=5000, random_seed=42)
    >>>
    >>> # Create trainer and run experiment
    >>> trainer = BaseNBeatsTrainer(config, ts_config)
    >>> results = trainer.run_experiment()
    >>> print(f"Experiment completed: {results['results_dir']}")

Individual Models Strategy:
    >>> # Train separate models for each pattern
    >>> config = NBeatsTrainingConfig(
    ...     training_strategy='individual',
    ...     backcast_length=168,
    ...     forecast_length=24,
    ...     max_patterns=20,  # Limit for manageable training
    ...     epochs=150,
    ...     batch_size=64
    ... )
    >>>
    >>> trainer = BaseNBeatsTrainer(config, ts_config)
    >>> results = trainer.run_experiment()
    >>>
    >>> # Access individual model results
    >>> for pattern_name, result in results['pattern_results'].items():
    ...     print(f"{pattern_name}: Test Loss = {result['test_loss']:.4f}")

Ensemble Strategy:
    >>> # Train ensemble of models with different configurations
    >>> config = NBeatsTrainingConfig(
    ...     training_strategy='ensemble',
    ...     ensemble_size=5,
    ...     ensemble_diversity='architecture',  # or 'data', 'hyperparams'
    ...     backcast_length=168,
    ...     forecast_length=24
    ... )
    >>>
    >>> trainer = BaseNBeatsTrainer(config, ts_config)
    >>> results = trainer.run_experiment()
    >>>
    >>> # Ensemble predictions are automatically combined
    >>> ensemble_loss = results['ensemble_performance']['test_loss']
    >>> print(f"Ensemble Test Loss: {ensemble_loss:.4f}")

Pattern-Specific Training:
    >>> # Focus on specific pattern categories
    >>> config = NBeatsTrainingConfig(
    ...     training_strategy='pattern_specific',
    ...     target_categories=['financial', 'weather', 'industrial'],
    ...     category_weights={'financial': 2.0, 'weather': 1.5, 'industrial': 1.5},
    ...     max_patterns_per_category=15
    ... )
    >>>
    >>> trainer = BaseNBeatsTrainer(config, ts_config)
    >>> results = trainer.run_experiment()

Multi-Horizon Training:
    >>> # Train for multiple forecast horizons
    >>> config = NBeatsTrainingConfig(
    ...     forecast_horizons=[6, 12, 24, 48],
    ...     training_strategy='unified',
    ...     backcast_length=168
    ... )
    >>>
    >>> trainer = BaseNBeatsTrainer(config, ts_config)
    >>> results = trainer.run_experiment()
    >>>
    >>> # Access results for each horizon
    >>> for horizon in [6, 12, 24, 48]:
    ...     if horizon in results['horizon_results']:
    ...         print(f"Horizon {horizon}: {results['horizon_results'][horizon]['test_loss']:.4f}")

Configuration Parameters
------------------------
**Training Strategy:**
* `training_strategy`: Strategy type ('individual', 'unified', 'ensemble', 'pattern_specific')
* `ensemble_size`: Number of models in ensemble (default: 5)
* `ensemble_diversity`: Diversity source ('architecture', 'data', 'hyperparams')
* `target_categories`: Specific categories for pattern-specific training

**Data Configuration:**
* `train_ratio`: Training data fraction (default: 0.7)
* `val_ratio`: Validation data fraction (default: 0.15)
* `test_ratio`: Test data fraction (default: 0.15)
* `max_patterns`: Maximum number of patterns to train on
* `max_patterns_per_category`: Maximum patterns per category
* `samples_per_pattern`: Samples per pattern for balanced training

**Model Architecture:**
* `backcast_length`: Input sequence length (default: 168)
* `forecast_length`: Prediction horizon length (default: 24)
* `stack_types`: N-BEATS stack types (default: ["trend", "seasonality", "generic"])
* `nb_blocks_per_stack`: Blocks per stack (default: 3)
* `hidden_layer_units`: Hidden layer size (default: 256)
* `use_revin`: Enable RevIN normalization (default: True)

**Training Parameters:**
* `epochs`: Maximum training epochs (default: 150)
* `batch_size`: Training batch size (default: 128)
* `learning_rate`: Initial learning rate (default: 1e-4)
* `optimizer`: Optimizer type ('adam' or 'adamw', default: 'adamw')
* `primary_loss`: Primary loss function ('mae', 'mse', 'smape', default: 'mae')
* `gradient_clip_norm`: Gradient clipping norm (default: 1.0)

**Regularization:**
* `dropout_rate`: Dropout probability (default: 0.15)
* `kernel_regularizer_l2`: L2 regularization strength (default: 1e-5)

Technical Details
-----------------
**Mathematical Foundation:**
The base N-BEATS training uses the standard N-BEATS architecture:
* Doubly residual stacking: residual = residual - backcast
* Forecast accumulation: forecast_sum = Σ forecast_i
* RevIN normalization for improved performance
* Pattern-aware data balancing for robust training

**Training Strategies Comparison:**
1. **Individual**: Optimal pattern-specific performance, higher computational cost
2. **Unified**: Good generalization, efficient deployment, moderate performance
3. **Ensemble**: Best overall performance, highest computational cost
4. **Pattern-Specific**: Domain-optimized, efficient for specific use cases

References
----------
**N-BEATS Architecture:**
* Oreshkin, B. N., et al. (2019). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." ICLR 2020.

**Ensemble Methods:**
* Breiman, L. (1996). "Bagging predictors." Machine learning, 24(2), 123-140.
* Hansen, L. K., & Salamon, P. (1990). "Neural network ensembles." IEEE transactions on pattern analysis and machine intelligence, 12(10), 993-1001.
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
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_techniques.utils.logger import logger
from dl_techniques.losses.smape_loss import SMAPELoss
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.utils.datasets.time_series_generator import TimeSeriesGenerator, TimeSeriesConfig
from dl_techniques.models.nbeats import create_nbeats_model, NBeatsNet

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


class TrainingStrategy(Enum):
    """Training strategy enumeration."""
    INDIVIDUAL = "individual"
    UNIFIED = "unified"
    ENSEMBLE = "ensemble"
    PATTERN_SPECIFIC = "pattern_specific"


class EnsembleDiversity(Enum):
    """Ensemble diversity source enumeration."""
    ARCHITECTURE = "architecture"
    DATA = "data"
    HYPERPARAMS = "hyperparams"


@dataclass
class NBeatsTrainingConfig:
    """Configuration for base N-BEATS training with multiple strategies."""

    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "base_nbeats_multi_pattern"

    # Training strategy configuration
    training_strategy: Union[TrainingStrategy, str] = TrainingStrategy.UNIFIED
    ensemble_size: int = 5
    ensemble_diversity: Union[EnsembleDiversity, str] = EnsembleDiversity.ARCHITECTURE
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

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
    samples_per_pattern: int = 15000

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

    def __post_init__(self) -> None:
        """Validation and configuration processing."""
        # Convert string enums to enum objects
        if isinstance(self.training_strategy, str):
            self.training_strategy = TrainingStrategy(self.training_strategy)
        if isinstance(self.ensemble_diversity, str):
            self.ensemble_diversity = EnsembleDiversity(self.ensemble_diversity)

        # Validate data ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        # Validate basic parameters
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")

        if self.val_ratio < 0.1:
            logger.warning(f"Validation ratio {self.val_ratio} might be too small for reliable validation")

        # Validate ensemble configuration
        if self.training_strategy == TrainingStrategy.ENSEMBLE:
            if self.ensemble_size < 2:
                raise ValueError("ensemble_size must be at least 2")
            if self.ensemble_size > 10:
                logger.warning(f"Large ensemble size ({self.ensemble_size}) may be computationally expensive")

        # Validate pattern-specific configuration
        if self.training_strategy == TrainingStrategy.PATTERN_SPECIFIC:
            if not self.target_categories:
                logger.warning("No target_categories specified for pattern-specific training")

        logger.info(f"Base N-BEATS Training Configuration:")
        logger.info(f"  ✅ Strategy: {self.training_strategy.value}")
        logger.info(f"  ✅ Data split: {self.train_ratio:.1f}/{self.val_ratio:.1f}/{self.test_ratio:.1f}")
        logger.info(f"  ✅ Model: {self.nb_blocks_per_stack} blocks, {self.hidden_layer_units} units")
        logger.info(f"  ✅ Training: {self.epochs} epochs, batch {self.batch_size}, lr {self.learning_rate}")
        logger.info(f"  ✅ Regularization: dropout {self.dropout_rate}, L2 {self.kernel_regularizer_l2}")

        if self.training_strategy == TrainingStrategy.ENSEMBLE:
            logger.info(f"  ✅ Ensemble: {self.ensemble_size} models, diversity: {self.ensemble_diversity.value}")
        if self.training_strategy == TrainingStrategy.PATTERN_SPECIFIC and self.target_categories:
            logger.info(f"  ✅ Target categories: {self.target_categories}")


class MultiPatternDataProcessor:
    """Advanced data processor for multiple pattern training strategies."""

    def __init__(self, config: NBeatsTrainingConfig):
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}
        self.pattern_to_id: Dict[str, int] = {}
        self.id_to_pattern: Dict[int, str] = {}

    def prepare_multi_pattern_data(
            self,
            raw_pattern_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Prepare multi-pattern data for different training strategies."""

        logger.info(f"Preparing data for {self.config.training_strategy.value} strategy...")

        # Create pattern ID mapping
        self.pattern_to_id = {pattern: idx for idx, pattern in enumerate(raw_pattern_data.keys())}
        self.id_to_pattern = {idx: pattern for pattern, idx in self.pattern_to_id.items()}

        # Fit scalers for each pattern
        self._fit_scalers(raw_pattern_data)

        prepared_data = {}

        for horizon in self.config.forecast_horizons:
            logger.info(f"Preparing data for horizon {horizon}")

            if self.config.training_strategy == TrainingStrategy.INDIVIDUAL:
                prepared_data[horizon] = self._prepare_individual_data(raw_pattern_data, horizon)
            elif self.config.training_strategy == TrainingStrategy.UNIFIED:
                prepared_data[horizon] = self._prepare_unified_data(raw_pattern_data, horizon)
            elif self.config.training_strategy == TrainingStrategy.ENSEMBLE:
                prepared_data[horizon] = self._prepare_ensemble_data(raw_pattern_data, horizon)
            elif self.config.training_strategy == TrainingStrategy.PATTERN_SPECIFIC:
                prepared_data[horizon] = self._prepare_pattern_specific_data(raw_pattern_data, horizon)

        return prepared_data

    def _prepare_individual_data(
            self,
            raw_pattern_data: Dict[str, np.ndarray],
            horizon: int
    ) -> Dict[str, Tuple]:
        """Prepare data for individual model training strategy."""
        pattern_datasets = {}

        for pattern_name, data in raw_pattern_data.items():
            try:
                min_length = self.config.backcast_length + horizon + 100
                if len(data) < min_length:
                    logger.warning(f"Insufficient data for {pattern_name} H={horizon}: {len(data)} < {min_length}")
                    continue

                # Split data temporally
                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))

                train_data = data[:train_size]
                val_data = data[train_size:train_size + val_size]
                test_data = data[train_size + val_size:]

                # Transform data
                train_scaled = self.scalers[pattern_name].transform(train_data)
                val_scaled = self.scalers[pattern_name].transform(val_data)
                test_scaled = self.scalers[pattern_name].transform(test_data)

                # Create sequences
                train_X, train_y = self._create_sequences(train_scaled, horizon, stride=1)
                val_X, val_y = self._create_sequences(val_scaled, horizon, stride=horizon // 2)
                test_X, test_y = self._create_sequences(test_scaled, horizon, stride=horizon // 2)

                # Balance data if needed
                if self.config.balance_patterns and len(train_X) > self.config.samples_per_pattern:
                    step = max(1, len(train_X) // self.config.samples_per_pattern)
                    indices = np.arange(0, len(train_X), step)[:self.config.samples_per_pattern]
                    train_X = train_X[indices]
                    train_y = train_y[indices]

                pattern_datasets[pattern_name] = {
                    'train': (train_X, train_y),
                    'val': (val_X, val_y),
                    'test': (test_X, test_y)
                }

                logger.info(f"Pattern {pattern_name}: train={len(train_X)}, val={len(val_X)}, test={len(test_X)}")

            except Exception as e:
                logger.warning(f"Failed to prepare {pattern_name} H={horizon}: {e}")
                continue

        return pattern_datasets

    def _prepare_unified_data(
            self,
            raw_pattern_data: Dict[str, np.ndarray],
            horizon: int
    ) -> Dict[str, Tuple]:
        """Prepare data for unified model training strategy."""
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

                # Transform data
                train_scaled = self.scalers[pattern_name].transform(train_data)
                val_scaled = self.scalers[pattern_name].transform(val_data)
                test_scaled = self.scalers[pattern_name].transform(test_data)

                # Create sequences
                train_X, train_y = self._create_sequences(train_scaled, horizon, stride=1)
                val_X, val_y = self._create_sequences(val_scaled, horizon, stride=horizon // 2)
                test_X, test_y = self._create_sequences(test_scaled, horizon, stride=horizon // 2)

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

        # Shuffle training data
        train_indices = np.random.permutation(len(combined_train_X))
        combined_train_X = combined_train_X[train_indices]
        combined_train_y = combined_train_y[train_indices]

        return {
            'unified': {
                'train': (combined_train_X, combined_train_y),
                'val': (combined_val_X, combined_val_y),
                'test': (combined_test_X, combined_test_y)
            }
        }

    def _prepare_ensemble_data(
            self,
            raw_pattern_data: Dict[str, np.ndarray],
            horizon: int
    ) -> Dict[str, Any]:
        """Prepare data for ensemble training strategy."""
        if self.config.ensemble_diversity == EnsembleDiversity.DATA:
            # Create different data subsets for each ensemble member
            ensemble_datasets = {}
            base_data = self._prepare_unified_data(raw_pattern_data, horizon)['unified']

            for i in range(self.config.ensemble_size):
                # Create bootstrap sample
                train_X, train_y = base_data['train']
                n_samples = len(train_X)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

                ensemble_datasets[f'model_{i}'] = {
                    'train': (train_X[bootstrap_indices], train_y[bootstrap_indices]),
                    'val': base_data['val'],
                    'test': base_data['test']
                }

            return ensemble_datasets
        else:
            # Use same data for all ensemble members (diversity comes from architecture/hyperparams)
            base_data = self._prepare_unified_data(raw_pattern_data, horizon)
            ensemble_datasets = {}

            for i in range(self.config.ensemble_size):
                ensemble_datasets[f'model_{i}'] = base_data['unified']

            return ensemble_datasets

    def _prepare_pattern_specific_data(
            self,
            raw_pattern_data: Dict[str, np.ndarray],
            horizon: int
    ) -> Dict[str, Tuple]:
        """Prepare data for pattern-specific training strategy."""
        if not self.config.target_categories:
            logger.warning("No target categories specified, using all patterns")
            return self._prepare_unified_data(raw_pattern_data, horizon)

        # Filter patterns by target categories
        from dl_techniques.utils.datasets.time_series_generator import TimeSeriesGenerator
        generator = TimeSeriesGenerator()

        filtered_data = {}
        for pattern_name, data in raw_pattern_data.items():
            for category in self.config.target_categories:
                if pattern_name in generator.get_tasks_by_category(category):
                    filtered_data[pattern_name] = data
                    break

        logger.info(f"Filtered to {len(filtered_data)} patterns from target categories")
        return self._prepare_unified_data(filtered_data, horizon)

    def _fit_scalers(self, pattern_data: Dict[str, np.ndarray]) -> None:
        """Fit scalers for each pattern."""
        for pattern_name, data in pattern_data.items():
            if len(data) < self.config.min_data_length:
                continue

            scaler = TimeSeriesNormalizer(method='standard')
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            scaler.fit(train_data)
            self.scalers[pattern_name] = scaler

    def _create_sequences(
            self,
            data: np.ndarray,
            forecast_length: int,
            stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with specified stride."""
        X, y = [], []

        for i in range(0, len(data) - self.config.backcast_length - forecast_length + 1, stride):
            backcast = data[i: i + self.config.backcast_length]
            forecast = data[i + self.config.backcast_length: i + self.config.backcast_length + forecast_length]

            if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                X.append(backcast)
                y.append(forecast)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class PatternPerformanceCallback(keras.callbacks.Callback):
    """Comprehensive callback for monitoring pattern-specific performance."""

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            data_processor: MultiPatternDataProcessor,
            test_data: Dict[int, Any],
            save_dir: str,
            model_name: str = "model"
    ):
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

    def on_epoch_end(self, epoch, logs=None):
        """Track training progress and create visualizations."""
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

    def _create_interim_plots(self, epoch: int):
        """Create comprehensive interim plots."""
        try:
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)

            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)

        except Exception as e:
            logger.warning(f"Failed to create interim plots for {self.model_name}: {e}")

    def _plot_learning_curves(self, epoch: int):
        """Plot training and validation curves."""
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
        plt.savefig(os.path.join(self.save_dir, f'{self.model_name}_learning_curves_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_prediction_samples(self, epoch: int):
        """Plot sample predictions."""
        horizon = self.config.forecast_horizons[0]

        if horizon not in self.test_data:
            return

        test_data = self.test_data[horizon]

        if self.config.training_strategy == TrainingStrategy.INDIVIDUAL:
            self._plot_individual_predictions(epoch, test_data)
        else:
            self._plot_unified_predictions(epoch, test_data)

    def _plot_individual_predictions(self, epoch: int, test_data: Dict):
        """Plot predictions for individual models."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        plot_idx = 0
        pattern_names = list(test_data.keys())[:6]  # Top 6 patterns

        for pattern_name in pattern_names:
            if plot_idx >= len(axes):
                break

            pattern_test_data = test_data[pattern_name]
            test_X, test_y = pattern_test_data['test']

            if len(test_X) == 0:
                continue

            # Take first sample
            sample_X = test_X[0:1]
            sample_y = test_y[0:1]

            # Get prediction
            pred_y = self.model(sample_X, training=False)

            # Plot
            backcast_x = np.arange(-self.config.backcast_length, 0)
            forecast_x = np.arange(0, self.config.forecast_length)

            axes[plot_idx].plot(backcast_x, sample_X[0], label='Input', color='blue', alpha=0.7)
            axes[plot_idx].plot(forecast_x, sample_y[0].flatten(), label='True', color='green', linewidth=2)
            axes[plot_idx].plot(forecast_x, pred_y[0].numpy().flatten(), label='Predicted', color='red', linewidth=2)

            axes[plot_idx].set_title(f'{pattern_name}')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].axvline(x=0, color='black', linestyle='-', alpha=0.5)

            plot_idx += 1

        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'Prediction Samples (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_unified_predictions(self, epoch: int, test_data: Dict):
        """Plot predictions for unified model."""
        unified_data = test_data['unified']
        test_X, test_y = unified_data['test']

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


class BaseNBeatsTrainer:
    """Comprehensive trainer for base N-BEATS with multiple training strategies."""

    def __init__(self, config: NBeatsTrainingConfig, ts_config: TimeSeriesConfig):
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = MultiPatternDataProcessor(config)

        # Get patterns
        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        self.selected_patterns = self._select_patterns()

        logger.info(f"Base N-BEATS Trainer initialized:")
        logger.info(f"  - Strategy: {self.config.training_strategy.value}")
        logger.info(f"  - Available categories: {len(self.pattern_categories)}")
        logger.info(f"  - Total patterns available: {len(self.all_patterns)}")
        logger.info(f"  - Selected {len(self.selected_patterns)} patterns")
        logger.info(f"  - Category distribution: {self._get_category_distribution()}")

    def _select_patterns(self) -> List[str]:
        """Select patterns based on strategy and configuration."""
        selected = []

        # Handle pattern-specific strategy
        if self.config.training_strategy == TrainingStrategy.PATTERN_SPECIFIC and self.config.target_categories:
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
        """Get distribution of selected patterns by category."""
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
        """Prepare training data for the selected strategy."""
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
        """Get final category distribution after data generation."""
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

    def create_model(self, forecast_length: int, model_id: str = "base") -> NBeatsNet:
        """Create N-BEATS model with enhanced configuration."""

        # Generate diverse configurations for ensemble
        if self.config.training_strategy == TrainingStrategy.ENSEMBLE and self.config.ensemble_diversity == EnsembleDiversity.ARCHITECTURE:
            return self._create_diverse_model(forecast_length, model_id)
        elif self.config.training_strategy == TrainingStrategy.ENSEMBLE and self.config.ensemble_diversity == EnsembleDiversity.HYPERPARAMS:
            return self._create_hyperparameter_diverse_model(forecast_length, model_id)
        else:
            return self._create_standard_model(forecast_length)

    def _create_standard_model(self, forecast_length: int) -> NBeatsNet:
        """Create standard N-BEATS model."""
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

    def _create_diverse_model(self, forecast_length: int, model_id: str) -> NBeatsNet:
        """Create architecturally diverse model for ensemble."""
        # Different stack configurations for ensemble diversity
        stack_configs = [
            ["trend", "seasonality"],
            ["trend", "seasonality", "generic"],
            ["seasonality", "generic"],
            ["trend", "generic", "generic"],
            ["generic", "generic", "generic"]
        ]

        block_configs = [2, 3, 4]
        hidden_configs = [128, 256, 512]

        # Select configuration based on model_id
        model_idx = int(model_id.split('_')[-1]) if '_' in model_id else 0

        stack_types = stack_configs[model_idx % len(stack_configs)]
        nb_blocks = block_configs[model_idx % len(block_configs)]
        hidden_units = hidden_configs[model_idx % len(hidden_configs)]

        kernel_regularizer = keras.regularizers.L2(
            self.config.kernel_regularizer_l2) if self.config.kernel_regularizer_l2 > 0 else None

        return create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=stack_types,
            nb_blocks_per_stack=nb_blocks,
            hidden_layer_units=hidden_units,
            use_revin=self.config.use_revin,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=self.config.dropout_rate,
            optimizer=self.config.optimizer,
            loss=self.config.primary_loss,
            learning_rate=self.config.learning_rate,
            gradient_clip_norm=self.config.gradient_clip_norm
        )

    def _create_hyperparameter_diverse_model(self, forecast_length: int, model_id: str) -> NBeatsNet:
        """Create hyperparameter diverse model for ensemble."""
        # Different hyperparameter configurations
        lr_configs = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
        dropout_configs = [0.0, 0.1, 0.15, 0.2, 0.25]
        l2_configs = [0, 1e-6, 1e-5, 1e-4, 1e-3]

        model_idx = int(model_id.split('_')[-1]) if '_' in model_id else 0

        learning_rate = lr_configs[model_idx % len(lr_configs)]
        dropout_rate = dropout_configs[model_idx % len(dropout_configs)]
        l2_reg = l2_configs[model_idx % len(l2_configs)]

        kernel_regularizer = keras.regularizers.L2(l2_reg) if l2_reg > 0 else None

        return create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=forecast_length,
            stack_types=self.config.stack_types.copy(),
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            use_revin=self.config.use_revin,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=dropout_rate,
            optimizer=self.config.optimizer,
            loss=self.config.primary_loss,
            learning_rate=learning_rate,
            gradient_clip_norm=self.config.gradient_clip_norm
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Run the multi-pattern experiment with the selected strategy."""
        try:
            exp_dir = os.path.join(
                self.config.result_dir,
                f"{self.config.experiment_name}_{self.config.training_strategy.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(exp_dir, exist_ok=True)

            logger.info(f"🚀 Starting Base N-BEATS Experiment: {exp_dir}")

            # Prepare data
            data_info = self.prepare_data()
            prepared_data = data_info['prepared_data']

            if not prepared_data:
                raise ValueError("No data prepared for training")

            results = {}

            for horizon in self.config.forecast_horizons:
                if horizon not in prepared_data:
                    continue

                logger.info(f"{'=' * 50}\n🎯 Training Models H={horizon}\n{'=' * 50}")

                if self.config.training_strategy == TrainingStrategy.INDIVIDUAL:
                    horizon_results = self._train_individual_models(prepared_data[horizon], horizon, exp_dir)
                elif self.config.training_strategy == TrainingStrategy.UNIFIED:
                    horizon_results = self._train_unified_model(prepared_data[horizon], horizon, exp_dir)
                elif self.config.training_strategy == TrainingStrategy.ENSEMBLE:
                    horizon_results = self._train_ensemble_models(prepared_data[horizon], horizon, exp_dir)
                elif self.config.training_strategy == TrainingStrategy.PATTERN_SPECIFIC:
                    horizon_results = self._train_pattern_specific_model(prepared_data[horizon], horizon, exp_dir)

                results[horizon] = horizon_results

            # Save comprehensive results
            self._save_results(results, exp_dir, data_info)

            logger.info("🎉 Base N-BEATS Experiment completed successfully!")
            return {
                "results_dir": exp_dir,
                "results": results,
                "num_patterns": data_info['num_patterns'],
                "pattern_mapping": data_info['pattern_to_id'],
                "category_distribution": data_info['category_distribution'],
                "training_strategy": self.config.training_strategy.value
            }

        except Exception as e:
            logger.error(f"💥 experiment failed: {e}", exc_info=True)
            raise

    def _train_individual_models(self, pattern_data: Dict, horizon: int, exp_dir: str) -> Dict[str, Any]:
        """Train individual models for each pattern."""
        logger.info(f"Training individual models for {len(pattern_data)} patterns")

        pattern_results = {}

        for pattern_name, data in pattern_data.items():
            logger.info(f"Training model for pattern: {pattern_name}")

            try:
                # Create model
                model = self.create_model(horizon, f"individual_{pattern_name}")

                # Build model
                train_X, train_y = data['train']
                model(train_X[:1])  # Build with sample data

                # Create callbacks
                viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
                callback = PatternPerformanceCallback(
                    config=self.config,
                    data_processor=self.processor,
                    test_data={horizon: {pattern_name: data}},
                    save_dir=viz_dir,
                    model_name=f"individual_{pattern_name}"
                )

                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=30,
                        restore_best_weights=True,
                        verbose=0
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=1e-6,
                        verbose=0
                    ),
                    callback,
                    keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(exp_dir, f'best_model_{pattern_name}_h{horizon}.keras'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=0
                    )
                ]

                # Train model
                start_time = datetime.now()

                val_X, val_y = data['val']
                history = model.fit(
                    train_X, train_y,
                    validation_data=(val_X, val_y),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    callbacks=callbacks,
                    verbose=0
                )

                training_time = (datetime.now() - start_time).total_seconds()

                # Evaluate
                test_X, test_y = data['test']
                test_results = model.evaluate(test_X, test_y, verbose=0)
                test_loss = test_results[0] if isinstance(test_results, list) else test_results
                test_mae = test_results[1] if len(test_results) > 1 else None

                pattern_results[pattern_name] = {
                    'history': history.history,
                    'training_time': training_time,
                    'test_loss': test_loss,
                    'test_mae': test_mae,
                    'final_epoch': len(history.history['loss'])
                }

                logger.info(f"✅ {pattern_name}: Test Loss = {test_loss:.4f}, Time = {training_time:.1f}s")

            except Exception as e:
                logger.warning(f"Failed to train model for {pattern_name}: {e}")
                continue

        return {'pattern_results': pattern_results}

    def _train_unified_model(self, pattern_data: Dict, horizon: int, exp_dir: str) -> Dict[str, Any]:
        """Train unified model on mixed patterns."""
        logger.info("Training unified model on mixed patterns")

        data = pattern_data['unified']

        # Create model
        model = self.create_model(horizon, "unified")

        # Build model
        train_X, train_y = data['train']
        model(train_X[:1])  # Build with sample data

        # Create callbacks
        viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
        callback = PatternPerformanceCallback(
            config=self.config,
            data_processor=self.processor,
            test_data={horizon: pattern_data},
            save_dir=viz_dir,
            model_name="unified"
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
                patience=15,
                min_lr=1e-6,
                verbose=1
            ),
            callback,
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(exp_dir, f'best_unified_model_h{horizon}.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        start_time = datetime.now()

        val_X, val_y = data['val']
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        test_X, test_y = data['test']
        test_results = model.evaluate(test_X, test_y, verbose=0)
        test_loss = test_results[0] if isinstance(test_results, list) else test_results
        test_mae = test_results[1] if len(test_results) > 1 else None

        logger.info(f"✅ Unified model: Test Loss = {test_loss:.4f}, Time = {training_time:.1f}s")

        return {
            'unified_results': {
                'history': history.history,
                'training_time': training_time,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'final_epoch': len(history.history['loss'])
            }
        }

    def _train_ensemble_models(self, pattern_data: Dict, horizon: int, exp_dir: str) -> Dict[str, Any]:
        """Train ensemble of models."""
        logger.info(f"Training ensemble of {self.config.ensemble_size} models")

        ensemble_results = {}
        ensemble_predictions = []

        for model_id, data in pattern_data.items():
            logger.info(f"Training ensemble member: {model_id}")

            try:
                # Create model
                model = self.create_model(horizon, model_id)

                # Build model
                train_X, train_y = data['train']
                model(train_X[:1])  # Build with sample data

                # Create callbacks
                viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
                callback = PatternPerformanceCallback(
                    config=self.config,
                    data_processor=self.processor,
                    test_data={horizon: {model_id: data}},
                    save_dir=viz_dir,
                    model_name=model_id
                )

                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=30,
                        restore_best_weights=True,
                        verbose=0
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=1e-6,
                        verbose=0
                    ),
                    callback,
                    keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(exp_dir, f'best_{model_id}_h{horizon}.keras'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=0
                    )
                ]

                # Train model
                start_time = datetime.now()

                val_X, val_y = data['val']
                history = model.fit(
                    train_X, train_y,
                    validation_data=(val_X, val_y),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    callbacks=callbacks,
                    verbose=0
                )

                training_time = (datetime.now() - start_time).total_seconds()

                # Evaluate
                test_X, test_y = data['test']
                test_results = model.evaluate(test_X, test_y, verbose=0)
                test_loss = test_results[0] if isinstance(test_results, list) else test_results
                test_mae = test_results[1] if len(test_results) > 1 else None

                # Get predictions for ensemble
                predictions = model.predict(test_X, verbose=0)
                ensemble_predictions.append(predictions)

                ensemble_results[model_id] = {
                    'history': history.history,
                    'training_time': training_time,
                    'test_loss': test_loss,
                    'test_mae': test_mae,
                    'final_epoch': len(history.history['loss'])
                }

                logger.info(f"✅ {model_id}: Test Loss = {test_loss:.4f}, Time = {training_time:.1f}s")

            except Exception as e:
                logger.warning(f"Failed to train {model_id}: {e}")
                continue

        # Compute ensemble performance
        if ensemble_predictions:
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
            ensemble_loss = np.mean(np.abs(test_y - ensemble_pred))
            ensemble_mae = ensemble_loss  # Same for MAE

            logger.info(f"✅ Ensemble: Test Loss = {ensemble_loss:.4f}")

            ensemble_results['ensemble_performance'] = {
                'test_loss': ensemble_loss,
                'test_mae': ensemble_mae,
                'individual_losses': [result['test_loss'] for result in ensemble_results.values() if
                                      'test_loss' in result]
            }

        return {'ensemble_results': ensemble_results}

    def _train_pattern_specific_model(self, pattern_data: Dict, horizon: int, exp_dir: str) -> Dict[str, Any]:
        """Train pattern-specific model."""
        logger.info("Training pattern-specific model")

        # Use unified training on filtered patterns
        return self._train_unified_model(pattern_data, horizon, exp_dir)

    def _save_results(self, results: Dict, exp_dir: str, data_info: Dict):
        """Save comprehensive experiment results."""
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
                    'training_strategy': self.config.training_strategy.value,
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
        """Serialize results for JSON storage."""
        serialized = {}

        for key, value in result.items():
            if key in ['pattern_results', 'ensemble_results']:
                serialized[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        serialized[key][sub_key] = {}
                        for metric_key, metric_value in sub_value.items():
                            if metric_key == 'history':
                                # Convert numpy arrays to lists for JSON serialization
                                serialized[key][sub_key][metric_key] = {
                                    hist_key: [float(v) for v in hist_value] if isinstance(hist_value, (list,
                                                                                                        np.ndarray)) else hist_value
                                    for hist_key, hist_value in metric_value.items()
                                }
                            else:
                                serialized[key][sub_key][metric_key] = float(metric_value) if isinstance(metric_value,
                                                                                                         (np.floating,
                                                                                                          np.integer)) else metric_value
                    else:
                        serialized[key][sub_key] = float(sub_value) if isinstance(sub_value, (np.floating,
                                                                                              np.integer)) else sub_value
            elif key == 'unified_results':
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

    def _create_comprehensive_summary_plots(self, results: Dict, exp_dir: str):
        """Create comprehensive summary visualization."""
        try:
            for horizon, result in results.items():
                if self.config.training_strategy == TrainingStrategy.INDIVIDUAL:
                    self._plot_individual_summary(result, horizon, exp_dir)
                elif self.config.training_strategy == TrainingStrategy.UNIFIED:
                    self._plot_unified_summary(result, horizon, exp_dir)
                elif self.config.training_strategy == TrainingStrategy.ENSEMBLE:
                    self._plot_ensemble_summary(result, horizon, exp_dir)
                elif self.config.training_strategy == TrainingStrategy.PATTERN_SPECIFIC:
                    self._plot_pattern_specific_summary(result, horizon, exp_dir)

        except Exception as e:
            logger.warning(f"Failed to create summary plots: {e}")

    def _plot_individual_summary(self, results: Dict, horizon: int, exp_dir: str):
        """Plot summary for individual models strategy."""
        pattern_results = results.get('pattern_results', {})

        if not pattern_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Test losses comparison
        pattern_names = list(pattern_results.keys())
        test_losses = [pattern_results[name]['test_loss'] for name in pattern_names]

        axes[0, 0].bar(range(len(pattern_names)), test_losses, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'Test Loss by Pattern (H={horizon})')
        axes[0, 0].set_xlabel('Pattern')
        axes[0, 0].set_ylabel('Test Loss')
        axes[0, 0].set_xticks(range(len(pattern_names)))
        axes[0, 0].set_xticklabels(pattern_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)

        # Training times
        training_times = [pattern_results[name]['training_time'] for name in pattern_names]

        axes[0, 1].bar(range(len(pattern_names)), training_times, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'Training Time by Pattern (H={horizon})')
        axes[0, 1].set_xlabel('Pattern')
        axes[0, 1].set_ylabel('Training Time (s)')
        axes[0, 1].set_xticks(range(len(pattern_names)))
        axes[0, 1].set_xticklabels(pattern_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)

        # Training epochs
        final_epochs = [pattern_results[name]['final_epoch'] for name in pattern_names]

        axes[1, 0].bar(range(len(pattern_names)), final_epochs, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title(f'Training Epochs by Pattern (H={horizon})')
        axes[1, 0].set_xlabel('Pattern')
        axes[1, 0].set_ylabel('Final Epoch')
        axes[1, 0].set_xticks(range(len(pattern_names)))
        axes[1, 0].set_xticklabels(pattern_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        avg_loss = np.mean(test_losses)
        std_loss = np.std(test_losses)
        min_loss = np.min(test_losses)
        max_loss = np.max(test_losses)

        axes[1, 1].text(0.1, 0.8, f'Summary Statistics (H={horizon})', fontsize=14, fontweight='bold',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Average Test Loss: {avg_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Std Test Loss: {std_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Min Test Loss: {min_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Max Test Loss: {max_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f'Total Patterns: {len(pattern_names)}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

        plt.suptitle(f'Individual Models Summary (H={horizon})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'individual_summary_h{horizon}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_unified_summary(self, results: Dict, horizon: int, exp_dir: str):
        """Plot summary for unified model strategy."""
        unified_results = results.get('unified_results', {})

        if not unified_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        history = unified_results.get('history', {})

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
        test_loss = unified_results.get('test_loss', 0)
        test_mae = unified_results.get('test_mae', 0)
        training_time = unified_results.get('training_time', 0)
        final_epoch = unified_results.get('final_epoch', 0)

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
        axes[1, 1].text(0.1, 0.8, f'Unified Model Summary (H={horizon})', fontsize=14, fontweight='bold',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Test Loss: {test_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Test MAE: {test_mae:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Training Time: {training_time:.1f}s', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Final Epoch: {final_epoch}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f'Strategy: Unified', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

        plt.suptitle(f'Unified Model Summary (H={horizon})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'unified_summary_h{horizon}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_ensemble_summary(self, results: Dict, horizon: int, exp_dir: str):
        """Plot summary for ensemble strategy."""
        ensemble_results = results.get('ensemble_results', {})

        if not ensemble_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract individual model results
        individual_results = {k: v for k, v in ensemble_results.items() if k != 'ensemble_performance'}
        ensemble_performance = ensemble_results.get('ensemble_performance', {})

        if individual_results:
            model_names = list(individual_results.keys())
            test_losses = [individual_results[name]['test_loss'] for name in model_names]

            # Individual model performance
            axes[0, 0].bar(range(len(model_names)), test_losses, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title(f'Individual Model Performance (H={horizon})')
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('Test Loss')
            axes[0, 0].set_xticks(range(len(model_names)))
            axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)

            # Ensemble vs individual comparison
            if ensemble_performance:
                ensemble_loss = ensemble_performance.get('test_loss', 0)
                avg_individual = np.mean(test_losses)

                comparison_names = ['Ensemble', 'Avg Individual']
                comparison_values = [ensemble_loss, avg_individual]

                axes[0, 1].bar(comparison_names, comparison_values,
                               color=['gold', 'lightcoral'], alpha=0.7, edgecolor='black')
                axes[0, 1].set_title(f'Ensemble vs Individual (H={horizon})')
                axes[0, 1].set_ylabel('Test Loss')
                axes[0, 1].grid(True, alpha=0.3)

                # Add value labels
                for i, v in enumerate(comparison_values):
                    axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

            # Training times
            training_times = [individual_results[name]['training_time'] for name in model_names]

            axes[1, 0].bar(range(len(model_names)), training_times, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title(f'Training Times (H={horizon})')
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('Training Time (s)')
            axes[1, 0].set_xticks(range(len(model_names)))
            axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        if ensemble_performance and individual_results:
            ensemble_loss = ensemble_performance.get('test_loss', 0)
            avg_individual = np.mean([individual_results[name]['test_loss'] for name in individual_results])
            improvement = ((avg_individual - ensemble_loss) / avg_individual) * 100

            axes[1, 1].text(0.1, 0.8, f'Ensemble Summary (H={horizon})', fontsize=14, fontweight='bold',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.7, f'Ensemble Loss: {ensemble_loss:.4f}', fontsize=12,
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f'Avg Individual: {avg_individual:.4f}', fontsize=12,
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.5, f'Improvement: {improvement:.2f}%', fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f'Ensemble Size: {len(individual_results)}', fontsize=12,
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.3, f'Diversity: {self.config.ensemble_diversity.value}', fontsize=12,
                            transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')

        plt.suptitle(f'Ensemble Summary (H={horizon})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'ensemble_summary_h{horizon}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_pattern_specific_summary(self, results: Dict, horizon: int, exp_dir: str):
        """Plot summary for pattern-specific strategy."""
        # Similar to unified summary but with pattern-specific context
        self._plot_unified_summary(results, horizon, exp_dir)

    def _create_detailed_experiment_report(self, results: Dict, exp_dir: str, data_info: Dict):
        """Create detailed text report of experiment results."""
        try:
            report_path = os.path.join(exp_dir, 'detailed_experiment_report.txt')

            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("BASE N-BEATS MULTI-PATTERN EXPERIMENT REPORT\n")
                f.write("=" * 80 + "\n\n")

                # Experiment overview
                f.write("EXPERIMENT OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Experiment Name: {self.config.experiment_name}\n")
                f.write(f"Training Strategy: {self.config.training_strategy.value}\n")
                f.write(f"Number of Patterns: {data_info['num_patterns']}\n")
                f.write(f"Available Pattern Categories: {len(self.pattern_categories)}\n")
                f.write(f"Time Series Length: {self.ts_config.n_samples}\n")
                f.write(f"Forecast Horizons: {self.config.forecast_horizons}\n")
                f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Strategy-specific details
                if self.config.training_strategy == TrainingStrategy.ENSEMBLE:
                    f.write("ENSEMBLE CONFIGURATION\n")
                    f.write("-" * 22 + "\n")
                    f.write(f"Ensemble Size: {self.config.ensemble_size}\n")
                    f.write(f"Diversity Source: {self.config.ensemble_diversity.value}\n\n")

                if self.config.training_strategy == TrainingStrategy.PATTERN_SPECIFIC:
                    f.write("PATTERN-SPECIFIC CONFIGURATION\n")
                    f.write("-" * 32 + "\n")
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
                    f"Data Split: {self.config.train_ratio:.1f}/{self.config.val_ratio:.1f}/{self.config.test_ratio:.1f}\n")
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

                    if self.config.training_strategy == TrainingStrategy.INDIVIDUAL:
                        pattern_results = result.get('pattern_results', {})
                        if pattern_results:
                            avg_loss = np.mean([r['test_loss'] for r in pattern_results.values()])
                            std_loss = np.std([r['test_loss'] for r in pattern_results.values()])
                            min_loss = np.min([r['test_loss'] for r in pattern_results.values()])
                            max_loss = np.max([r['test_loss'] for r in pattern_results.values()])
                            total_time = sum([r['training_time'] for r in pattern_results.values()])

                            f.write(f"Strategy: Individual Models\n")
                            f.write(f"Number of Models: {len(pattern_results)}\n")
                            f.write(f"Average Test Loss: {avg_loss:.6f}\n")
                            f.write(f"Std Test Loss: {std_loss:.6f}\n")
                            f.write(f"Min Test Loss: {min_loss:.6f}\n")
                            f.write(f"Max Test Loss: {max_loss:.6f}\n")
                            f.write(f"Total Training Time: {total_time:.1f} seconds\n")

                    elif self.config.training_strategy == TrainingStrategy.UNIFIED:
                        unified_results = result.get('unified_results', {})
                        if unified_results:
                            f.write(f"Strategy: Unified Model\n")
                            f.write(f"Test Loss: {unified_results['test_loss']:.6f}\n")
                            f.write(f"Test MAE: {unified_results.get('test_mae', 'N/A')}\n")
                            f.write(f"Training Time: {unified_results['training_time']:.1f} seconds\n")
                            f.write(f"Final Epoch: {unified_results['final_epoch']}\n")

                    elif self.config.training_strategy == TrainingStrategy.ENSEMBLE:
                        ensemble_results = result.get('ensemble_results', {})
                        if ensemble_results:
                            ensemble_perf = ensemble_results.get('ensemble_performance', {})
                            individual_results = {k: v for k, v in ensemble_results.items() if
                                                  k != 'ensemble_performance'}

                            f.write(f"Strategy: Ensemble\n")
                            f.write(f"Ensemble Size: {len(individual_results)}\n")
                            if ensemble_perf:
                                f.write(f"Ensemble Test Loss: {ensemble_perf['test_loss']:.6f}\n")
                                if individual_results:
                                    avg_individual = np.mean([r['test_loss'] for r in individual_results.values()])
                                    improvement = ((avg_individual - ensemble_perf['test_loss']) / avg_individual) * 100
                                    f.write(f"Average Individual Loss: {avg_individual:.6f}\n")
                                    f.write(f"Ensemble Improvement: {improvement:.2f}%\n")
                                total_time = sum([r['training_time'] for r in individual_results.values()])
                                f.write(f"Total Training Time: {total_time:.1f} seconds\n")

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
                f.write("✅ Base N-BEATS training completed successfully\n")
                f.write(f"✅ Trained using {self.config.training_strategy.value} strategy\n")
                f.write(f"✅ Successfully processed {len(data_info['category_distribution'])} pattern categories\n")
                f.write(
                    f"✅ Generated {sum(data_info['category_distribution'].values())} diverse time series patterns\n")

                if self.config.training_strategy == TrainingStrategy.ENSEMBLE:
                    f.write("✅ Ensemble approach provides robust predictions with reduced variance\n")
                elif self.config.training_strategy == TrainingStrategy.INDIVIDUAL:
                    f.write("✅ Individual models provide pattern-specific optimization\n")
                elif self.config.training_strategy == TrainingStrategy.UNIFIED:
                    f.write("✅ Unified model provides efficient deployment with good generalization\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")

            logger.info(f"Detailed experiment report saved to {report_path}")

        except Exception as e:
            logger.warning(f"Failed to create detailed experiment report: {e}")


def main():
    """Run the base N-BEATS experiment with different training strategies."""

    # Unified strategy (single model on mixed patterns)
    config = NBeatsTrainingConfig(
        training_strategy=TrainingStrategy.UNIFIED,
        experiment_name="base_nbeats",

        backcast_length=168,
        forecast_length=4,
        forecast_horizons=[4, 8],

        stack_types=["trend", "seasonality", "generic"],
        nb_blocks_per_stack=3,
        hidden_layer_units=256,
        use_revin=True,

        max_patterns_per_category=8,
        min_data_length=2000,
        balance_patterns=True,
        samples_per_pattern=12000,

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
        create_prediction_plots=True
    )

    ts_config = TimeSeriesConfig(
        n_samples=5000,
        random_seed=42,
        default_noise_level=0.01
    )

    try:
        logger.info("🚀 Running Unified Strategy Experiment")
        trainer_unified = BaseNBeatsTrainer(config, ts_config)
        results_unified = trainer_unified.run_experiment()
        logger.info(f"✅ Unified experiment completed: {results_unified['results_dir']}")
    except Exception as e:
        logger.error(f"💥 Unified experiment failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()