"""
Comprehensive TiRex Training Framework for Multiple Time Series Patterns

This module provides a sophisticated, production-ready training framework for
TiRex models trained on multiple time series patterns. It enables training
TiRex models across diverse time series patterns with comprehensive monitoring,
visualization, and performance analysis, supporting both point and probabilistic
forecasting objectives via configurable loss functions.

Classes
-------
TiRexTrainingConfig
    Comprehensive configuration dataclass containing all training parameters,
    including model architecture, training objective (loss function), data
    management, and visualization settings.

TiRexDataProcessor
    Advanced data processing pipeline handling multi-pattern data preparation,
    sequence generation, normalization, balanced sampling, and proper
    train/validation/test splitting with temporal integrity.

TiRexPerformanceCallback
    Context-aware monitoring callback that creates detailed visualizations of
    training progress and prediction samples, adapting its plots for either
    point forecasts (MASE) or probabilistic forecasts (Quantile Loss).

TiRexTrainer
    Main training orchestrator for multi-pattern TiRex training with
    comprehensive experiment management and performance analysis.

Training Approach
-----------------
The framework trains a single TiRex model on mixed patterns from multiple
time series categories. The training objective can be configured to:
1.  **Minimize MASE:** Trains the model to produce a point forecast (the median)
    that outperforms a naive baseline.
2.  **Minimize Quantile Loss:** Trains the model to produce probabilistic
    forecasts to quantify uncertainty using a correctly vectorized loss function.
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
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.losses.mase_loss import MASELoss, mase_metric
from dl_techniques.models.tirex.model import create_tirex_by_variant, TiRexCore
from dl_techniques.datasets.time_series import (
    TimeSeriesNormalizer, TimeSeriesGenerator, TimeSeriesConfig
)

# ---------------------------------------------------------------------

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


# ---------------------------------------------------------------------

@dataclass
class TiRexTrainingConfig:
    """
    Configuration for TiRex training with multiple patterns.

    This dataclass contains comprehensive configuration options for training
    TiRex models on diverse time series patterns with various optimization
    and visualization settings.
    """

    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "tirex_multi_pattern"

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # TiRex specific configuration
    input_length: int = 56
    prediction_length: int = 24
    prediction_horizons: List[int] = field(default_factory=lambda: [24])

    # Model architecture
    variant: str = "small"  # 'tiny', 'small', 'medium', 'large'
    patch_size: int = 12
    embed_dim: int = 128
    num_blocks: int = 6
    num_heads: int = 8
    block_types: List[str] = field(default_factory=lambda: ["mixed"])
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # Training configuration
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    loss_function: str = "quantile"  # 'mase' or 'quantile'
    mase_seasonal_periods: int = 1

    # Regularization
    dropout_rate: float = 0.15

    # Pattern selection and balancing
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    min_data_length: int = 2000
    balance_patterns: bool = True
    samples_per_pattern: int = 25000

    # Category weights for balanced sampling
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 1.0, "seasonal": 1.0, "composite": 1.2, "stochastic": 1.0,
        "financial": 1.5, "weather": 1.3, "network": 1.4, "biomedical": 1.2,
        "industrial": 1.3, "intermittent": 1.0, "volatility": 1.1,
        "regime": 1.2, "structural": 1.1, "outliers": 1.0, "chaotic": 1.1
    })

    # Visualization and evaluation
    visualize_every_n_epochs: int = 1
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Data augmentation
    multiplicative_noise_std: float = 0.01
    additive_noise_std: float = 0.01
    enable_multiplicative_noise: bool = True
    enable_additive_noise: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(
                f"Data ratios must sum to 1.0, got {total_ratio}"
            )

        if self.loss_function not in ['mase', 'quantile']:
            raise ValueError(
                f"loss_function must be 'mase' or 'quantile', "
                f"got {self.loss_function}"
            )

        if self.loss_function == 'quantile' and 0.5 not in self.quantile_levels:
            logger.warning(
                "For best results with quantile loss, it's recommended to "
                "include 0.5 (the median) in quantile_levels."
            )

        logger.info("TiRex Training Configuration:")
        logger.info(
            f"  - Data split: {self.train_ratio:.1f}/{self.val_ratio:.1f}/"
            f"{self.test_ratio:.1f}"
        )
        logger.info(
            f"  - Model: variant='{self.variant}', {self.num_blocks} blocks, "
            f"embed_dim={self.embed_dim}"
        )
        logger.info(
            f"  - Training: {self.epochs} epochs, batch {self.batch_size}, "
            f"lr {self.learning_rate}, loss={self.loss_function}"
        )
        mult_noise = self.multiplicative_noise_std if self.enable_multiplicative_noise else 'disabled'
        add_noise = self.additive_noise_std if self.enable_additive_noise else 'disabled'
        logger.info(
            f"  - Augmentation: mult_noise {mult_noise}, "
            f"add_noise {add_noise}"
        )


class TiRexDataProcessor:
    """Advanced data processor for multiple pattern training for TiRex."""

    def __init__(self, config: TiRexTrainingConfig) -> None:
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}
        self.pattern_to_id: Dict[str, int] = {}
        self.id_to_pattern: Dict[int, str] = {}

    def prepare_multi_pattern_data(
            self, raw_pattern_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Prepare multi-pattern data for training."""
        logger.info("Preparing data for multi-pattern training...")
        self.pattern_to_id = {
            pattern: idx for idx, pattern in enumerate(raw_pattern_data.keys())
        }
        self.id_to_pattern = {
            idx: pattern for pattern, idx in self.pattern_to_id.items()
        }
        self._fit_scalers(raw_pattern_data)
        prepared_data = {}
        for horizon in self.config.prediction_horizons:
            logger.info(f"Preparing data for horizon {horizon}")
            prepared_data[horizon] = self._prepare_mixed_pattern_data(
                raw_pattern_data, horizon
            )
        return prepared_data

    def _prepare_mixed_pattern_data(
            self, raw_pattern_data: Dict[str, np.ndarray], horizon: int
    ) -> Dict[str, Any]:
        """Prepare data by mixing multiple patterns."""
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []

        for pattern_name, data in raw_pattern_data.items():
            try:
                min_len = self.config.input_length + horizon + 100
                if len(data) < min_len:
                    continue

                train_size = int(self.config.train_ratio * len(data))
                val_size = int(self.config.val_ratio * len(data))
                train_data = data[:train_size]
                val_data = data[train_size:train_size + val_size]
                test_data = data[train_size + val_size:]

                train_scaled = self.scalers[pattern_name].transform(train_data)
                val_scaled = self.scalers[pattern_name].transform(val_data)
                test_scaled = self.scalers[pattern_name].transform(test_data)

                train_X, train_y = self._create_sequences(train_scaled, horizon, stride=1)
                val_X, val_y = self._create_sequences(val_scaled, horizon, stride=horizon // 2)
                test_X, test_y = self._create_sequences(test_scaled, horizon, stride=horizon // 2)

                if self.config.balance_patterns and len(train_X) > self.config.samples_per_pattern:
                    step = max(1, len(train_X) // self.config.samples_per_pattern)
                    indices = np.arange(0, len(train_X), step)[:self.config.samples_per_pattern]
                    train_X, train_y = train_X[indices], train_y[indices]

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

        combined_train_X = np.concatenate(all_train_X)
        combined_train_y = np.concatenate(all_train_y)
        combined_val_X = np.concatenate(all_val_X)
        combined_val_y = np.concatenate(all_val_y)
        combined_test_X = np.concatenate(all_test_X)
        combined_test_y = np.concatenate(all_test_y)

        return {
            'mixed_patterns': {
                'train': self._create_tf_dataset(
                    combined_train_X, combined_train_y, self.config.batch_size,
                    shuffle=True, augment=True
                ),
                'val': self._create_tf_dataset(
                    combined_val_X, combined_val_y, self.config.batch_size,
                    shuffle=False, augment=False
                ),
                'test': self._create_tf_dataset(
                    combined_test_X, combined_test_y, self.config.batch_size,
                    shuffle=False, augment=False
                ),
                'test_arrays': (combined_test_X, combined_test_y)
            }
        }

    def _fit_scalers(self, pattern_data: Dict[str, np.ndarray]) -> None:
        """Fit scalers for each pattern."""
        for pattern_name, data in pattern_data.items():
            if len(data) >= self.config.min_data_length:
                scaler = TimeSeriesNormalizer(method='standard')
                train_end = int(self.config.train_ratio * len(data))
                scaler.fit(data[:train_end])
                self.scalers[pattern_name] = scaler

    def _create_sequences(
            self, data: np.ndarray, prediction_length: int, stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input/output sequences."""
        X, y = [], []
        total_len = self.config.input_length + prediction_length
        for i in range(0, len(data) - total_len + 1, stride):
            context = data[i: i + self.config.input_length]
            target_start = i + self.config.input_length
            target_end = target_start + prediction_length
            target = data[target_start:target_end].flatten()

            if not (np.isnan(context).any() or np.isnan(target).any()):
                X.append(context)  # Context shape: (input_length, 1)
                y.append(target)  # Target shape: (prediction_length,)

        # Final y shape: (num_samples, prediction_length)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _apply_noise_augmentation(
            self, x: tf.Tensor, y: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply multiplicative and additive noise augmentation."""
        augmented_x = x
        if self.config.enable_multiplicative_noise and self.config.multiplicative_noise_std > 0:
            mult_noise = tf.random.normal(
                tf.shape(x), mean=1.0,
                stddev=self.config.multiplicative_noise_std, dtype=x.dtype
            )
            augmented_x *= mult_noise
        if self.config.enable_additive_noise and self.config.additive_noise_std > 0:
            add_noise = tf.random.normal(
                tf.shape(augmented_x), mean=0.0,
                stddev=self.config.additive_noise_std, dtype=augmented_x.dtype
            )
            augmented_x += add_noise
        return augmented_x, y

    def _create_tf_dataset(
            self, X: np.ndarray, y: np.ndarray, batch_size: int,
            shuffle: bool, augment: bool
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            buffer_size = min(10000, len(X))
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        if augment:
            dataset = dataset.map(
                self._apply_noise_augmentation,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        return dataset.prefetch(tf.data.AUTOTUNE)


class TiRexPerformanceCallback(keras.callbacks.Callback):
    """Context-aware callback for monitoring TiRex performance."""

    def __init__(
            self, config: TiRexTrainingConfig, test_data: Dict[int, Any],
            save_dir: str, model_name: str = "model"
    ) -> None:
        super().__init__()
        self.config = config
        self.test_data = test_data
        self.save_dir = save_dir
        self.model_name = model_name
        self.history = {'epoch': [], 'loss': [], 'val_loss': []}
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(
            self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """Track progress and create visualizations."""
        logs = logs or {}
        self.history['epoch'].append(epoch)
        self.history['loss'].append(logs.get('loss', 0.0))
        self.history['val_loss'].append(logs.get('val_loss', 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(
                f"Creating visualizations for {self.model_name} at "
                f"epoch {epoch + 1}"
            )
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        """Create comprehensive interim plots."""
        try:
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)
        except Exception as e:
            logger.warning(
                f"Failed to create interim plots for {self.model_name}: {e}"
            )

    def _plot_learning_curves(self, epoch: int) -> None:
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epoch'], self.history['loss'],
                 label='Training Loss', color='blue', linewidth=2)
        plt.plot(self.history['epoch'], self.history['val_loss'],
                 label='Validation Loss', color='red', linewidth=2)
        title = (
            f'{self.model_name} - Loss Curves ({self.config.loss_function}, '
            f'Epoch {epoch + 1})'
        )
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(
            self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png'
        )
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        """Plot prediction samples, adapting to the loss function."""
        horizon = self.config.prediction_horizons[0]
        if horizon not in self.test_data:
            return

        test_X, test_y = self.test_data[horizon]['mixed_patterns']['test_arrays']
        if len(test_X) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        sample_indices = np.linspace(0, len(test_X) - 1, 6, dtype=int)

        for i, sample_idx in enumerate(sample_indices):
            sample_X = test_X[sample_idx:sample_idx + 1]
            sample_y_true = test_y[sample_idx]

            # --- With this block of code ---
            # The model call normalizes the input internally and produces normalized predictions.
            preds_norm = self.model(sample_X, training=False)

            # Retrieve statistics from the model's internal scaler to reverse the instance normalization.
            # This is crucial for visualizing predictions on the same scale as the input data.
            scaler = self.model.scaler
            last_mean = scaler._last_mean
            last_std = scaler._last_std

            # Denormalize the predictions. Broadcasting handles the shape alignment.
            preds = (preds_norm * last_std) + last_mean
            preds = preds[0]  # Remove the batch dimension for plotting
            # --- End of replacement ---

            context_x = np.arange(-self.config.input_length, 0)
            horizon_x = np.arange(0, horizon)

            axes[i].plot(context_x, sample_X[0], label='Input',
                         color='blue', alpha=0.7)
            axes[i].plot(horizon_x, sample_y_true, label='True Future',
                         color='green', linewidth=2)

            if self.config.loss_function == 'quantile':
                quantiles = self.model.quantile_levels
                try:
                    median_idx = quantiles.index(0.5)
                except ValueError:
                    median_idx = len(quantiles) // 2  # Fallback

                lower_q, upper_q = (0.1, 0.9)
                lower_idx = min(range(len(quantiles)), key=lambda j: abs(quantiles[j] - lower_q))
                upper_idx = min(range(len(quantiles)), key=lambda j: abs(quantiles[j] - upper_q))

                axes[i].plot(horizon_x, preds[median_idx],
                             label='Median Prediction', color='red',
                             linewidth=2, linestyle='--')
                fill_label = f'{quantiles[lower_idx]*100:.0f}-{quantiles[upper_idx]*100:.0f}% Range'
                axes[i].fill_between(
                    horizon_x, preds[lower_idx], preds[upper_idx],
                    color='red', alpha=0.2, label=fill_label
                )
                plot_title = f'Probabilistic Sample {i + 1}'
            else:  # 'mase'
                axes[i].plot(horizon_x, preds[0], label='Median Prediction',
                             color='red', linewidth=2, linestyle='--')
                plot_title = f'Point Forecast Sample {i + 1}'

            axes[i].set_title(plot_title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)

        plt.suptitle(
            f'Prediction Samples (Epoch {epoch + 1}, '
            f'Loss: {self.config.loss_function})', fontsize=16
        )
        plt.tight_layout()
        save_path = os.path.join(
            self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'
        )
        plt.savefig(save_path, dpi=150)
        plt.close()


class TiRexTrainer:
    """Comprehensive trainer for TiRex with multiple pattern support."""

    def __init__(
            self, config: TiRexTrainingConfig, ts_config: TimeSeriesConfig
    ) -> None:
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = TiRexDataProcessor(config)
        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        self.selected_patterns = self._select_patterns()
        logger.info(
            f"TiRex Trainer initialized: {len(self.selected_patterns)} "
            "patterns selected."
        )

    def _select_patterns(self) -> List[str]:
        """Select patterns based on configuration."""
        selected = []
        categories = self.config.target_categories or self.pattern_categories
        for category in categories:
            patterns = self.generator.get_tasks_by_category(category)
            weight = self.config.category_weights.get(category, 1.0)
            max_p = min(int(self.config.max_patterns_per_category * weight),
                        len(patterns))
            if len(patterns) > max_p:
                selected.extend(
                    np.random.choice(patterns, size=max_p, replace=False)
                )
            else:
                selected.extend(patterns)

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = np.random.choice(
                selected, size=self.config.max_patterns, replace=False
            ).tolist()
        return selected

    def prepare_data(self) -> Dict[str, Any]:
        """Prepare training data for the model."""
        logger.info("Generating data for selected patterns...")
        raw_data = {
            name: self.generator.generate_task_data(name)
            for name in self.selected_patterns
            if len(self.generator.generate_task_data(name)) >= self.config.min_data_length
        }
        prepared_data = self.processor.prepare_multi_pattern_data(raw_data)
        return {'prepared_data': prepared_data, 'num_patterns': len(raw_data)}

    def create_model(
            self, prediction_length: int
    ) -> Tuple[TiRexCore, List[Any]]:
        """Create and compile the TiRex model based on configuration."""
        loss, metrics, quantiles_for_model = None, [], []

        if self.config.loss_function == 'mase':
            logger.info("Configuring model for MASE loss (median point forecast).")
            loss = MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
            metrics = [
                mase_metric(seasonal_periods=self.config.mase_seasonal_periods),
                'mae', 'mse'
            ]
            quantiles_for_model = [0.5]
        elif self.config.loss_function == 'quantile':
            logger.info("Configuring model for Quantile loss (probabilistic forecast).")
            loss = QuantileLoss(quantiles=self.config.quantile_levels)
            try:
                median_idx = self.config.quantile_levels.index(0.5)

                def mae_of_median(y_true, y_pred):
                    return keras.metrics.mean_absolute_error(
                        y_true, y_pred[:, median_idx, :]
                    )
                mae_of_median.__name__ = 'mae_of_median'
                metrics.append(mae_of_median)
            except ValueError:
                logger.warning(
                    "0.5 not in quantile_levels, cannot compute MAE of "
                    "median metric."
                )
            quantiles_for_model = self.config.quantile_levels
        else:
            raise ValueError(
                f"Unsupported loss function: {self.config.loss_function}"
            )

        model = create_tirex_by_variant(
            variant=self.config.variant,
            input_length=self.config.input_length,
            prediction_length=prediction_length,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            num_blocks=self.config.num_blocks,
            num_heads=self.config.num_heads,
            quantile_levels=quantiles_for_model,
            dropout_rate=self.config.dropout_rate
        )

        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = self.config.learning_rate
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=True
        )
        return model, metrics

    def run_experiment(self) -> Dict[str, Any]:
        """Run the multi-pattern experiment."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_{timestamp}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting TiRex Experiment: {exp_dir}")
        data_info = self.prepare_data()
        prepared_data = data_info['prepared_data']
        if not prepared_data:
            raise ValueError("No data prepared for training")

        results = {}
        for horizon in self.config.prediction_horizons:
            if horizon in prepared_data:
                logger.info(f"{'='*50}\nTraining Model H={horizon}\n{'='*50}")
                results[horizon] = self._train_model(
                    prepared_data[horizon], horizon, exp_dir
                )

        self._save_results(results, exp_dir, data_info)
        logger.info("TiRex Experiment completed successfully!")
        return {"results_dir": exp_dir, "results": results}

    def _train_model(
            self, pattern_data: Dict, horizon: int, exp_dir: str
    ) -> Dict[str, Any]:
        """Train model on mixed patterns."""
        data = pattern_data['mixed_patterns']
        model, metrics = self.create_model(horizon)
        model.summary(print_fn=logger.info)

        viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
        log_dir = os.path.join(exp_dir, 'logs', f'h{horizon}')
        checkpoint_path = os.path.join(exp_dir, f'best_model_h{horizon}.keras')

        callbacks = [
            TiRexPerformanceCallback(
                self.config, {horizon: pattern_data}, viz_dir, "tirex_model"
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=50,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=30,
                min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True, verbose=1
            ),
            keras.callbacks.TensorBoard(log_dir=log_dir)
        ]

        start_time = datetime.now()
        history = model.fit(
            data['train'],
            validation_data=data['val'],
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        training_time = (datetime.now() - start_time).total_seconds()

        test_results = model.evaluate(data['test'], verbose=0, return_dict=True)

        logger.info(
            f"Model H={horizon}: Test Loss = {test_results['loss']:.4f}, "
            f"Time = {training_time:.1f}s"
        )

        return {
            'history': history.history,
            'training_time': training_time,
            'test_metrics': test_results,
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(
            self, results: Dict, exp_dir: str, data_info: Dict
    ) -> None:
        """Save comprehensive experiment results and report."""
        try:
            def default_serializer(o):
                if isinstance(o, (np.floating, np.integer)):
                    return str(o)
                return '<not_serializable>'

            with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2, default=default_serializer)

            exp_info = {
                'num_patterns': data_info['num_patterns'],
                'config': self.config.__dict__
            }
            with open(os.path.join(exp_dir, 'experiment_info.json'), 'w') as f:
                json.dump(exp_info, f, indent=2, default=str)

            self._create_detailed_experiment_report(results, exp_dir, data_info)
            logger.info(f"Results saved to {exp_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _create_detailed_experiment_report(
            self, results: Dict, exp_dir: str, data_info: Dict
    ) -> None:
        """Create a detailed text report of the experiment."""
        report_path = os.path.join(exp_dir, 'detailed_experiment_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\nTIREX MULTI-PATTERN EXPERIMENT REPORT\n" +
                    "=" * 80 + "\n\n")
            f.write("EXPERIMENT OVERVIEW\n" + "-" * 20 + "\n")
            f.write(f"Experiment Name: {self.config.experiment_name}\n")
            f.write(f"Number of Patterns: {data_info['num_patterns']}\n\n")

            f.write("MODEL CONFIGURATION\n" + "-" * 19 + "\n")
            f.write(f"Variant: {self.config.variant}\n")
            f.write(f"Input Length: {self.config.input_length}\n\n")

            f.write("TRAINING CONFIGURATION\n" + "-" * 21 + "\n")
            f.write(f"Loss Function: {self.config.loss_function}\n")
            if self.config.loss_function == 'mase':
                f.write(
                    f"MASE Seasonal Periods: {self.config.mase_seasonal_periods}\n"
                )
            else:
                f.write(f"Quantile Levels: {self.config.quantile_levels}\n")
            f.write(f"Optimizer: {self.config.optimizer} "
                    f"(LR: {self.config.learning_rate})\n\n")

            for horizon, result in results.items():
                f.write(f"RESULTS FOR HORIZON {horizon}\n" + "-" * 25 + "\n")
                metrics = result['test_metrics']
                f.write(f"Test Loss ({self.config.loss_function}): "
                        f"{metrics['loss']:.6f}\n")
                for name, value in metrics.items():
                    if name != 'loss':
                        f.write(f"Test Metric ({name}): {value:.6f}\n")
                f.write(f"Training Time: {result['training_time']:.1f} seconds\n")
                f.write(f"Final Epoch: {result['final_epoch']}\n\n")

            f.write("=" * 80 + "\nEnd of Report\n" + "=" * 80 + "\n")
        logger.info(f"Detailed experiment report saved to {report_path}")


def main() -> None:
    """Run the TiRex experiment."""
    # Example 1: Training for a point forecast using MASE loss
    # config = TiRexTrainingConfig(
    #     experiment_name="tirex_small_mase_h4",
    #     input_length=112,
    #     patch_size=12,
    #     prediction_horizons=[12],
    #     loss_function='mase',
    #     mase_seasonal_periods=1,
    #     variant="small",
    #     epochs=200,
    #     batch_size=128,
    #     learning_rate=5e-4,
    # )

    # Example 2: Training for a probabilistic forecast using Quantile loss
    config = TiRexTrainingConfig(
        experiment_name="tirex_small",
        input_length=112,
        patch_size=12,
        prediction_horizons=[12],
        loss_function='quantile',
        variant="small",
        epochs=200,
        batch_size=128,
        learning_rate=5e-4,
    )

    ts_config = TimeSeriesConfig(n_samples=2000, random_seed=42)

    try:
        logger.info(
            f"Running TiRex Multi-Pattern Experiment with loss: "
            f"{config.loss_function}"
        )
        trainer = TiRexTrainer(config, ts_config)
        results = trainer.run_experiment()
        logger.info(f"Experiment completed. Results in: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()