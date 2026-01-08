"""
Orchestrate the end-to-end training and evaluation of the TiRex probabilistic forecasting framework.

Architecture
------------
The TiRex (Time-series Representation EXchange) framework implements a patch-based
Transformer architecture designed to capture long-range dependencies in time series data
while maintaining computational efficiency. The architecture consists of three distinct stages:

1.  **Patching and Embedding**: The input time series history is segmented into
    non-overlapping patches. This reduces the sequence length from $L$ to $L/P$ (where $P$
    is stride/patch size), effectively increasing the receptive field and forcing the
    model to learn semantic local patterns rather than point-wise correlations.

2.  **Transformer Encoder**: A stack of standard Transformer encoder layers processes
    the sequence of patch embeddings. This stage utilizes multi-head self-attention to
    capture temporal dependencies between different time segments.

3.  **Variant-Specific Decoding**: The framework supports two distinct decoding strategies
    to project the latent representation into the forecast horizon:
    *   **TiRexCore (Global Pooling)**: Aggregates the temporal dimension via mean pooling
        to create a single global context vector, which is then projected via an MLP to
        the forecast horizon. This is computationally efficient and assumes the history's
        sentiment is globally relevant.
    *   **TiRexExtended (Query-Based)**: Utilizes a set of learnable "Query Tokens"
        (representing the future horizon) that attend to the encoded history via
        cross-attention (or interactions within the sequence). This allows the model to
        selectively extract historical information relevant to specific future time steps.

Integration
-----------
Updated to use the unified `dl_techniques.datasets.time_series` module for
synthetic data generation and normalization.

References
----------
1.  Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
    A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
    In International Conference on Learning Representations (ICLR).
2.  Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2023).
    Long-term Forecasting with TiDE: Time-series Dense Encoder.
    arXiv preprint arXiv:2304.08424.
3.  Koenker, R., & Bassett Jr, G. (1978). Regression quantiles.
    Econometrica: journal of the Econometric Society, 33-50.
"""

import os
import sys
import json
import math
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.models.tirex.model import create_tirex_by_variant, TiRexCore
from dl_techniques.models.tirex.model_extended import create_tirex_extended, TiRexExtended

from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    TimeSeriesNormalizer,
    NormalizationMethod
)

from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

# ---------------------------------------------------------------------

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for major libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)

# ---------------------------------------------------------------------


@dataclass
class TiRexTrainingConfig:
    """Configuration dataclass for TiRex training on multiple patterns."""
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "tirex_probabilistic"

    # Model type selection
    model_type: str = "core"  # 'core' or 'extended'

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # TiRex specific configuration
    input_length: int = 168
    prediction_length: int = 24

    # Model Architecture
    variant: str = "small"  # 'tiny', 'small', 'medium', 'large'
    patch_size: int = 12

    dropout_rate: float = 0.1
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 500
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'

    # Learning rate schedule with warmup
    use_warmup: bool = True
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-6

    # Pattern selection and Normalization
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    min_data_length: int = 2000
    normalize_per_instance: bool = True

    # Category weights for balanced sampling
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 1.0, "seasonal": 1.0, "composite": 1.2,
        "financial": 1.5, "weather": 1.3, "biomedical": 1.2,
        "industrial": 1.3, "intermittent": 1.0, "volatility": 1.1,
        "regime": 1.2, "structural": 1.1
    })

    # Visualization configuration
    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 12
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Deep Analysis Configuration
    perform_deep_analysis: bool = True
    analysis_frequency: int = 10
    analysis_start_epoch: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.model_type not in ['core', 'extended']:
            raise ValueError(
                f"model_type must be 'core' or 'extended', got '{self.model_type}'"
            )

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")

        if 0.5 not in self.quantile_levels:
            logger.warning(
                "Recommended to include 0.5 (median) in quantile_levels."
            )


def _fill_nans(data: np.ndarray) -> np.ndarray:
    """
    Forward fill NaNs in numpy array.

    :param data: Input array potentially containing NaNs.
    :return: Array with NaNs forward-filled.
    """
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = data[idx]
    out[np.isnan(out)] = 0
    return out


class TiRexDataProcessor:
    """
    Robust data processor for TiRex using the unified TimeSeriesGenerator and Normalizer.

    Features:
    - Infinite streaming generator for Training (maximum diversity).
    - Pre-computed in-memory arrays for Validation/Test (eliminates end-of-epoch lag).
    - Robust per-instance normalization using TimeSeriesNormalizer.
    """

    def __init__(
            self,
            config: TiRexTrainingConfig,
            generator: TimeSeriesGenerator,
            selected_patterns: List[str],
            pattern_to_category: Dict[str, str],
            num_features: int = 1
    ):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_category = pattern_to_category
        self.num_features = num_features
        self.weighted_patterns, self.weights = self._prepare_weighted_sampling()

    def _prepare_weighted_sampling(self) -> Tuple[List[str], List[float]]:
        """Prepare lists for weighted random pattern selection."""
        patterns, weights = [], []
        for pattern_name in self.selected_patterns:
            category = self.pattern_to_category.get(pattern_name, "unknown")
            weight = self.config.category_weights.get(category, 1.0)
            patterns.append(pattern_name)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            num_patterns = len(patterns)
            normalized_weights = [1.0 / num_patterns] * num_patterns
        return patterns, normalized_weights

    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        """
        Normalize series instance using TimeSeriesNormalizer.
        Includes clipping to prevent extreme outliers from destabilizing training.

        :param series: Input numpy array (n_timesteps, features) or (n_timesteps,).
        :return: Normalized array as float32.
        """
        series = np.clip(series, -1e6, 1e6)

        if self.config.normalize_per_instance:
            normalizer = TimeSeriesNormalizer(method=NormalizationMethod.STANDARD)
            if np.isnan(series).any():
                series = _fill_nans(series)
            series = normalizer.fit_transform(series)

        series = np.clip(series, -10.0, 10.0)
        return series.astype(np.float32)

    def _training_generator(
        self
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Infinite generator for training data with high diversity.
        Uses a buffer to mix patterns and reduce generator overhead.
        """
        patterns_to_mix = 50
        windows_per_pattern = 5
        buffer: List[Tuple[np.ndarray, np.ndarray]] = []

        total_window_len = self.config.input_length + self.config.prediction_length

        while True:
            if not buffer:
                selected_patterns = random.choices(
                    self.weighted_patterns, self.weights, k=patterns_to_mix
                )

                for pattern_name in selected_patterns:
                    data = self.ts_generator.generate_task_data(pattern_name)

                    if len(data) < total_window_len or not np.isfinite(data).all():
                        continue

                    train_size = int(self.config.train_ratio * len(data))
                    train_data = data[:train_size]

                    max_start_idx = len(train_data) - total_window_len
                    if max_start_idx <= 0:
                        continue

                    for _ in range(windows_per_pattern):
                        start_idx = random.randint(0, max_start_idx)
                        window = train_data[start_idx: start_idx + total_window_len]

                        window = self._safe_normalize(window)

                        context = window[:self.config.input_length].reshape(
                            -1, self.num_features
                        )
                        # Target: (prediction_length,) flat for QuantileLoss
                        target = window[self.config.input_length:].flatten()

                        buffer.append((context, target))

                random.shuffle(buffer)

            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(
        self, split: str, num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-generate a fixed dataset in memory to prevent generator lag.

        :param split: 'val' or 'test'.
        :param num_samples: Total number of samples to generate.
        :return: Tuple of (Contexts, Targets) as numpy arrays.
        """
        logger.info(f"Pre-computing {split} dataset ({num_samples} samples)...")

        contexts: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        total_window_len = self.config.input_length + self.config.prediction_length
        samples_collected = 0
        pattern_cycle = 0

        while samples_collected < num_samples:
            pattern_idx = pattern_cycle % len(self.selected_patterns)
            pattern_name = self.selected_patterns[pattern_idx]
            pattern_cycle += 1

            data = self.ts_generator.generate_task_data(pattern_name)

            if len(data) < total_window_len or not np.isfinite(data).all():
                continue

            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))

            if split == 'val':
                split_data = data[train_end:val_end]
            else:
                split_data = data[val_end:]

            max_start_idx = len(split_data) - total_window_len
            if max_start_idx <= 0:
                continue

            start_idx = random.randint(0, max_start_idx)
            window = split_data[start_idx: start_idx + total_window_len]
            window = self._safe_normalize(window)

            ctx = window[:self.config.input_length].reshape(-1, self.num_features)
            tgt = window[self.config.input_length:].flatten()

            contexts.append(ctx)
            targets.append(tgt)
            samples_collected += 1

        return np.array(contexts, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _test_generator_raw(
        self
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Helper specifically for visualization callback to pull fresh test samples.
        """
        total_window_len = self.config.input_length + self.config.prediction_length

        viz_patterns = self.selected_patterns.copy()
        random.shuffle(viz_patterns)

        for pattern_name in viz_patterns:
            data = self.ts_generator.generate_task_data(pattern_name)

            if len(data) < total_window_len or not np.isfinite(data).all():
                continue

            start_test = int(
                (self.config.train_ratio + self.config.val_ratio) * len(data)
            )
            test_data = data[start_test:]

            if len(test_data) < total_window_len:
                continue

            max_start_idx = len(test_data) - total_window_len
            start_idx = random.randint(0, max_start_idx)

            window = test_data[start_idx: start_idx + total_window_len]
            window = self._safe_normalize(window)

            yield (
                window[:self.config.input_length].reshape(-1, self.num_features),
                window[self.config.input_length:].flatten()
            )

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create tf.data.Datasets."""
        output_sig = (
            tf.TensorSpec(
                shape=(self.config.input_length, self.num_features), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(self.config.prediction_length,), dtype=tf.float32)
        )

        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=output_sig
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        validation_steps = max(50, len(self.selected_patterns))
        test_steps = max(20, len(self.selected_patterns))

        num_val_samples = validation_steps * self.config.batch_size
        num_test_samples = test_steps * self.config.batch_size

        val_x, val_y = self._generate_fixed_dataset('val', num_val_samples)
        val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_ds = val_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        test_x, test_y = self._generate_fixed_dataset('test', num_test_samples)
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_ds = test_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': validation_steps,
            'test_steps': test_steps
        }


class TiRexPerformanceCallback(keras.callbacks.Callback):
    """Custom callback for tracking and visualizing TiRex performance."""

    def __init__(
            self,
            config: TiRexTrainingConfig,
            processor: TiRexDataProcessor,
            save_dir: str,
            model_name: str = "tirex"
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.save_dir = save_dir
        self.model_name = model_name

        os.makedirs(self.save_dir, exist_ok=True)

        self.viz_test_data = self._prepare_viz_data()

        self.training_history: Dict[str, List[float]] = {
            'loss': [], 'val_loss': [],
            'mae_median': [], 'val_mae_median': [],
            'lr': []
        }

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare a fixed batch of test samples for consistent visualization."""
        viz_samples_x: List[np.ndarray] = []
        viz_samples_y: List[np.ndarray] = []
        test_gen = self.processor._test_generator_raw()

        for _ in range(self.config.plot_top_k_patterns):
            try:
                x, y = next(test_gen)
                viz_samples_x.append(x)
                viz_samples_y.append(y)
            except StopIteration:
                break

        if not viz_samples_x:
            return np.array([]), np.array([])

        return np.array(viz_samples_x), np.array(viz_samples_y)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Track metrics and create visualizations at epoch end."""
        logs = logs or {}

        self.training_history['loss'].append(logs.get('loss', 0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0))
        self.training_history['mae_median'].append(logs.get('mae_of_median', 0))
        self.training_history['val_mae_median'].append(logs.get('val_mae_of_median', 0))

        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        if hasattr(self.model.optimizer.learning_rate, '__call__'):
            lr = float(keras.ops.convert_to_numpy(
                self.model.optimizer.learning_rate(self.model.optimizer.iterations)
            ))
        self.training_history['lr'].append(lr)

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Creating visualizations for epoch {epoch + 1}...")
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        epochs_range = range(1, len(self.training_history['loss']) + 1)

        axes[0].plot(epochs_range, self.training_history['loss'], label='Train')
        axes[0].plot(epochs_range, self.training_history['val_loss'], label='Val')
        axes[0].set_title('Quantile Loss')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        if self.training_history['mae_median']:
            axes[1].plot(epochs_range, self.training_history['mae_median'], label='Train')
            axes[1].plot(
                epochs_range, self.training_history['val_mae_median'], label='Val'
            )
            axes[1].set_title('MAE (Median Forecast)')
        axes[1].set_ylabel('MAE')
        axes[1].legend()

        axes[2].plot(epochs_range, self.training_history['lr'])
        axes[2].set_title('Learning Rate')
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png')
        )
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data

        if len(test_x) == 0:
            return

        # Predictions shape: (batch, prediction_length, num_quantiles)
        preds = self.model(test_x, training=False)
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()

        quantiles = self.config.quantile_levels
        try:
            median_idx = quantiles.index(0.5)
        except ValueError:
            median_idx = len(quantiles) // 2

        low_idx = 0
        high_idx = len(quantiles) - 1 if len(quantiles) >= 3 else -1

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            input_x = np.arange(-self.config.input_length, 0)
            pred_x = np.arange(0, self.config.prediction_length)

            ax.plot(
                input_x, test_x[i].flatten(),
                label='Input', color='blue', alpha=0.7
            )
            ax.plot(
                pred_x, test_y[i].flatten(),
                label='True', color='green', linewidth=2
            )

            ax.plot(
                pred_x, preds[i, :, median_idx],
                label='Median', color='red', linestyle='--'
            )

            ax.fill_between(
                pred_x,
                preds[i, :, low_idx],
                preds[i, :, high_idx],
                color='red', alpha=0.2,
                label=f'{quantiles[low_idx]}-{quantiles[high_idx]} Q'
            )

            ax.set_title(f'Sample {i + 1}')
            ax.legend(loc='upper left', fontsize='small')
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Probabilistic Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png')
        )
        plt.close()


class TiRexTrainer:
    """Orchestrates TiRex training with Quantile Loss."""

    def __init__(
            self,
            config: TiRexTrainingConfig,
            generator_config: TimeSeriesGeneratorConfig
    ) -> None:
        self.config = config
        self.generator_config = generator_config
        self.generator = TimeSeriesGenerator(generator_config)

        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        self.pattern_to_category = {
            task: cat
            for cat in self.pattern_categories
            for task in self.generator.get_tasks_by_category(cat)
        }
        self.selected_patterns = self._select_patterns()

        self.processor = TiRexDataProcessor(
            config, self.generator, self.selected_patterns, self.pattern_to_category
        )

        self.model: Optional[Union[TiRexCore, TiRexExtended]] = None
        self.exp_dir: Optional[str] = None

    def _select_patterns(self) -> List[str]:
        if self.config.target_categories:
            patterns_to_consider = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            patterns_to_consider = self.all_patterns

        selected: List[str] = []
        category_counts: Dict[str, int] = {}

        for pattern in sorted(patterns_to_consider):
            category = self.pattern_to_category.get(pattern)
            if (category and
                    category_counts.get(category, 0) <
                    self.config.max_patterns_per_category):
                selected.append(pattern)
                category_counts[category] = category_counts.get(category, 0) + 1

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = random.sample(selected, self.config.max_patterns)

        logger.info(f"Selected {len(selected)} patterns for training.")
        return selected

    def create_model(self) -> Union[TiRexCore, TiRexExtended]:
        """Create and compile the TiRex model based on configuration."""
        if self.config.model_type == 'core':
            logger.info(f"Creating TiRexCore ({self.config.variant}) model...")
            model = create_tirex_by_variant(
                variant=self.config.variant,
                input_length=self.config.input_length,
                prediction_length=self.config.prediction_length,
                patch_size=self.config.patch_size,
                quantile_levels=self.config.quantile_levels,
                dropout_rate=self.config.dropout_rate
            )
        elif self.config.model_type == 'extended':
            logger.info(f"Creating TiRexExtended ({self.config.variant}) model...")
            model = create_tirex_extended(
                variant=self.config.variant,
                input_length=self.config.input_length,
                prediction_length=self.config.prediction_length,
                patch_size=self.config.patch_size,
                quantile_levels=self.config.quantile_levels,
                dropout_rate=self.config.dropout_rate
            )
        else:
            raise ValueError(
                f"Unknown model_type: {self.config.model_type}. "
                f"Expected 'core' or 'extended'."
            )

        if self.config.use_warmup:
            total_steps = self.config.epochs * self.config.steps_per_epoch
            primary_steps = max(1, total_steps - self.config.warmup_steps)

            primary_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=primary_steps,
                alpha=0.01
            )
            lr_schedule = WarmupSchedule(
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
                primary_schedule=primary_schedule
            )
            logger.info("Using Warmup + CosineDecay schedule.")
        else:
            lr_schedule = self.config.learning_rate

        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = lr_schedule
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        loss = QuantileLoss(quantiles=self.config.quantile_levels)

        metrics = []
        if 0.5 in self.config.quantile_levels:
            median_idx = self.config.quantile_levels.index(0.5)

            def mae_of_median(y_true, y_pred):
                # y_true: (batch, horizon)
                # y_pred: (batch, horizon, quantiles)
                return keras.metrics.mean_absolute_error(
                    y_true, y_pred[:, :, median_idx]
                )
            mae_of_median.__name__ = 'mae_of_median'
            metrics.append(mae_of_median)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=True
        )
        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete training experiment."""
        logger.info("Starting TiRex training experiment")
        self.exp_dir = self._create_experiment_dir()
        logger.info(f"Results will be saved to: {self.exp_dir}")

        data_pipeline = self.processor.prepare_datasets()

        self.model = self.create_model()

        dummy_in = np.zeros(
            (1, self.config.input_length, 1), dtype='float32'
        )
        self.model(dummy_in)
        logger.info("Model created successfully")
        logger.info(f"Model parameters: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        training_results = self._train_model(data_pipeline, self.exp_dir)

        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config,
            'experiment_dir': self.exp_dir,
            'training_results': training_results,
            'results_dir': self.exp_dir
        }

    def _create_experiment_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.experiment_name}_{self.config.model_type}_{timestamp}"
        exp_dir = os.path.join(self.config.result_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _train_model(self, data_pipeline: Dict, exp_dir: str) -> Dict[str, Any]:
        logger.info("Starting model training...")

        viz_dir = os.path.join(exp_dir, 'visualizations')
        callbacks = [
            TiRexPerformanceCallback(self.config, self.processor, viz_dir, "tirex"),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=30, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(exp_dir, 'best_model.keras'),
                monitor='val_loss', save_best_only=True, verbose=1
            ),
            keras.callbacks.TerminateOnNaN()
        ]

        if not self.config.use_warmup:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
            ))

        if self.config.perform_deep_analysis:
            logger.info("Adding Deep Model Analysis callback.")

            analysis_config = AnalysisConfig(
                analyze_weights=True,
                analyze_spectral=True,
                analyze_calibration=False,
                analyze_information_flow=False,
                analyze_training_dynamics=False,
                verbose=False
            )

            analysis_dir = os.path.join(exp_dir, 'deep_analysis')

            analyzer_cb = EpochAnalyzerCallback(
                output_dir=analysis_dir,
                analysis_config=analysis_config,
                start_epoch=self.config.analysis_start_epoch,
                epoch_frequency=self.config.analysis_frequency,
                model_name="TiRex"
            )
            callbacks.append(analyzer_cb)

        history = self.model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Evaluating on test set...")
        test_metrics = self.model.evaluate(
            data_pipeline['test_ds'],
            steps=data_pipeline['test_steps'],
            verbose=1,
            return_dict=True
        )

        return {
            'history': history.history,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__
        }

        def json_convert(o: Any) -> str:
            if isinstance(o, (np.floating, np.integer)):
                return str(o)
            return str(o)

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_convert)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for TiRex training configuration."""
    parser = argparse.ArgumentParser(
        description="TiRex Training Framework (Unified TimeSeriesGenerator)"
    )

    parser.add_argument(
        "--experiment_name", type=str, default="tirex",
        help="Name of the experiment for logging."
    )

    parser.add_argument(
        "--model_type", type=str, default="core",
        choices=['core', 'extended'],
        help="Model architecture type: 'core' (mean pooling) or 'extended' (query tokens)."
    )

    parser.add_argument(
        "--variant", type=str, default="small",
        choices=['tiny', 'small', 'medium', 'large'],
        help="Model variant/size."
    )
    parser.add_argument("--input_length", type=int, default=256)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw")

    parser.add_argument(
        "--no-warmup", dest="use_warmup", action="store_false",
        help="Disable learning rate warmup."
    )
    parser.set_defaults(use_warmup=True)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)

    parser.add_argument(
        "--no-normalize", dest="normalize_per_instance", action="store_false",
        help="Disable per-instance normalization."
    )
    parser.set_defaults(normalize_per_instance=True)
    parser.add_argument("--max_patterns_per_category", type=int, default=100)

    parser.add_argument("--visualize_every_n_epochs", type=int, default=5)
    parser.add_argument("--plot_top_k_patterns", type=int, default=12)

    parser.add_argument(
        "--no-deep-analysis", dest="perform_deep_analysis", action="store_false",
        help="Disable periodic deep model analysis."
    )
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10)
    parser.add_argument("--analysis_start_epoch", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    """Main function to configure and run the TiRex training experiment."""
    args = parse_args()

    config = TiRexTrainingConfig(
        experiment_name=args.experiment_name,
        model_type=args.model_type,
        variant=args.variant,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        patch_size=args.patch_size,
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        normalize_per_instance=args.normalize_per_instance,
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=10000,
        random_seed=42,
        default_noise_level=0.1
    )

    trainer = TiRexTrainer(config, generator_config)
    results = trainer.run_experiment()
    logger.info(f"Completed. Results: {results['results_dir']}")
    sys.exit(0)


if __name__ == "__main__":
    main()