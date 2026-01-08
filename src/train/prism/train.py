"""
Orchestrate end-to-end training and evaluation of the PRISM forecasting framework.

Architecture
------------
The PRISM (Progressive Refinement via Integrated Segmentation and Mixing) framework
implements hierarchical time-frequency decomposition with routing mechanisms designed
to capture both local patterns and long-range dependencies.

This script implements a robust training pipeline supporting both:
1. Point Forecasting (MSE Loss)
2. Probabilistic Forecasting (Quantile Loss)

It uses pre-computed in-memory validation datasets to prevent CPU-bound generation
lags at the end of epochs.

Integration
-----------
Updated to use the unified `dl_techniques.datasets.time_series` module for
synthetic data generation and normalization.

References
----------
1.  Time Series Decomposition Methods in Deep Learning: A Review
2.  Hierarchical Temporal Modeling with Wavelet Transforms
3.  Router Networks for Mixture-of-Experts in Time Series Forecasting
"""

import os
import json
import math
import random
import argparse
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

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
from dl_techniques.analyzer import AnalysisConfig

from dl_techniques.models.prism.model import PRISMModel
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    TimeSeriesNormalizer,
    NormalizationMethod
)

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
class PRISMTrainingConfig:
    """Configuration dataclass for PRISM training."""
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "prism_forecasting"

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # PRISM specific configuration
    context_len: int = 168
    forecast_len: int = 24

    # Model Architecture
    preset: str = "small"  # 'tiny', 'small', 'base', 'large'
    hidden_dim: Optional[int] = None  # If None, uses num_features
    num_layers: int = 2
    tree_depth: int = 2
    overlap_ratio: float = 0.25
    num_wavelet_levels: int = 3
    router_hidden_dim: int = 64
    router_temperature: float = 1.0
    dropout_rate: float = 0.1
    ffn_expansion: int = 4

    # Probabilistic / Quantile Configuration
    use_quantile_head: bool = False
    enforce_monotonicity: bool = True
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # Training configuration
    epochs: int = 150
    batch_size: int = 64
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
        """Validate configuration."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        if self.context_len <= 0 or self.forecast_len <= 0:
            raise ValueError("context_len and forecast_len must be positive")

        if self.use_quantile_head and 0.5 not in self.quantile_levels:
            logger.warning("Recommended to include 0.5 (median) in quantile_levels for evaluation.")


class PRISMDataProcessor:
    """
    Robust data processor for PRISM using the new TimeSeriesGenerator and Normalizer.

    Features:
    - Infinite streaming generator for Training (maximum diversity).
    - Pre-computed in-memory arrays for Validation/Test (eliminates end-of-epoch lag).
    - Robust per-instance normalization using TimeSeriesNormalizer.
    """

    def __init__(
            self,
            config: PRISMTrainingConfig,
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

        :param series: Input numpy array (n_timesteps, features) or (n_timesteps,)
        :return: Normalized array as float32
        """
        # 1. Clip extreme outliers first (stability)
        series = np.clip(series, -1e6, 1e6) # clip infinities

        if self.config.normalize_per_instance:
            # Use the module's Normalizer for robust standard scaling
            normalizer = TimeSeriesNormalizer(method=NormalizationMethod.STANDARD)
            # Check for NaNs before fitting
            if np.isnan(series).any():
                # Simple forward fill for safety inside generator
                series = _fill_nans(series)

            series = normalizer.fit_transform(series)

        # 2. Hard Clip for neural network stability (-10 to 10 sigmas is usually plenty)
        series = np.clip(series, -10.0, 10.0)

        return series.astype(np.float32)

    def _training_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Infinite generator for training data.
        Uses a buffer to mix patterns and reduce generator overhead.
        """
        patterns_to_mix = 50
        windows_per_pattern = 5
        buffer = []

        total_window_len = self.config.context_len + self.config.forecast_len

        while True:
            if not buffer:
                # 1. Select a mix of patterns
                selected_patterns = random.choices(
                    self.weighted_patterns, self.weights, k=patterns_to_mix
                )

                for pattern_name in selected_patterns:
                    # 2. Generate full series using the new TimeSeriesGenerator API
                    # Note: Returns (n_samples, 1)
                    data = self.ts_generator.generate_task_data(pattern_name)

                    # Safety Check
                    if len(data) < total_window_len or not np.isfinite(data).all():
                        continue

                    train_size = int(self.config.train_ratio * len(data))
                    train_data = data[:train_size]

                    max_start_idx = len(train_data) - total_window_len
                    if max_start_idx <= 0:
                        continue

                    # 3. Extract multiple windows from this series
                    for _ in range(windows_per_pattern):
                        start_idx = random.randint(0, max_start_idx)
                        window = train_data[start_idx : start_idx + total_window_len]

                        # 4. Normalize and Clip
                        window = self._safe_normalize(window)

                        # 5. Split Context/Target
                        context = window[:self.config.context_len].reshape(-1, self.num_features)
                        target = window[self.config.context_len:].reshape(-1, self.num_features)

                        buffer.append((context, target))

                random.shuffle(buffer)

            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(self, split: str, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-generate a fixed dataset in memory to prevent generator lag during validation/test.

        Args:
            split: 'val' or 'test'
            num_samples: Total number of samples to generate

        Returns:
            Tuple of (Contexts, Targets) as numpy arrays
        """
        logger.info(f"Pre-computing {split} dataset ({num_samples} samples)...")

        contexts = []
        targets = []
        total_window_len = self.config.context_len + self.config.forecast_len

        samples_collected = 0
        pattern_cycle = 0

        while samples_collected < num_samples:
            # Deterministic cycling through patterns for fair representation
            pattern_idx = pattern_cycle % len(self.selected_patterns)
            pattern_name = self.selected_patterns[pattern_idx]
            pattern_cycle += 1

            # Generate full series
            data = self.ts_generator.generate_task_data(pattern_name)

            if len(data) < total_window_len or not np.isfinite(data).all():
                continue

            # Slice based on split
            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))

            if split == 'val':
                split_data = data[train_end:val_end]
            else: # test
                split_data = data[val_end:]

            max_start_idx = len(split_data) - total_window_len
            if max_start_idx <= 0:
                continue

            # Grab one random window from this pattern's split
            start_idx = random.randint(0, max_start_idx)
            window = split_data[start_idx : start_idx + total_window_len]
            window = self._safe_normalize(window)

            ctx = window[:self.config.context_len].reshape(-1, self.num_features)
            tgt = window[self.config.context_len:].reshape(-1, self.num_features)

            contexts.append(ctx)
            targets.append(tgt)
            samples_collected += 1

        return np.array(contexts, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _test_generator_raw(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Helper specifically for the visualization callback to pull fresh test samples
        on demand without affecting the main pipeline.
        """
        total_window_len = self.config.context_len + self.config.forecast_len

        # Shuffle patterns for viz variety
        viz_patterns = self.selected_patterns.copy()
        random.shuffle(viz_patterns)

        for pattern_name in viz_patterns:
            data = self.ts_generator.generate_task_data(pattern_name)

            if len(data) < total_window_len or not np.isfinite(data).all():
                continue

            # Use test split
            start_test = int((self.config.train_ratio + self.config.val_ratio) * len(data))
            test_data = data[start_test:]

            if len(test_data) < total_window_len:
                continue

            max_start_idx = len(test_data) - total_window_len
            start_idx = random.randint(0, max_start_idx)

            window = test_data[start_idx : start_idx + total_window_len]
            window = self._safe_normalize(window)

            yield (
                window[:self.config.context_len].reshape(-1, self.num_features),
                window[self.config.context_len:].reshape(-1, self.num_features)
            )

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create tf.data.Datasets."""

        # 1. Training Dataset (Infinite Stream)
        output_sig = (
            tf.TensorSpec(shape=(self.config.context_len, self.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(self.config.forecast_len, self.num_features), dtype=tf.float32)
        )

        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=output_sig
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        # 2. Calculate sizes for Val/Test
        # We ensure enough steps to cover all patterns at least once or twice
        validation_steps = max(50, len(self.selected_patterns))
        test_steps = max(20, len(self.selected_patterns))

        num_val_samples = validation_steps * self.config.batch_size
        num_test_samples = test_steps * self.config.batch_size

        # 3. Validation Dataset (Pre-computed Memory Cache)
        val_x, val_y = self._generate_fixed_dataset('val', num_val_samples)
        val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_ds = val_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        # 4. Test Dataset (Pre-computed Memory Cache)
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

def _fill_nans(data: np.ndarray) -> np.ndarray:
    """Helper to forward fill NaNs in numpy array."""
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = data[idx]
    # If leading NaNs, fill with 0
    out[np.isnan(out)] = 0
    return out


class PRISMPerformanceCallback(keras.callbacks.Callback):
    """Custom callback for tracking and visualizing PRISM performance."""

    def __init__(
            self,
            config: PRISMTrainingConfig,
            processor: PRISMDataProcessor,
            save_dir: str,
            model_name: str = "prism"
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.save_dir = save_dir
        self.model_name = model_name

        os.makedirs(self.save_dir, exist_ok=True)

        # Prepare fixed test data for visualization
        self.viz_test_data = self._prepare_viz_data()

        # Track metrics across epochs
        self.training_history = {
            'loss': [], 'val_loss': [],
            'metric': [], 'val_metric': [],
            'lr': []
        }

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare a fixed batch of test samples for consistent visualization."""
        viz_samples_x, viz_samples_y = [], []
        # Use the specialized raw generator for viz samples
        test_gen = self.processor._test_generator_raw()

        for _ in range(self.config.plot_top_k_patterns):
            try:
                x, y = next(test_gen)
                viz_samples_x.append(x)
                viz_samples_y.append(y)
            except StopIteration:
                break

        if not viz_samples_x:
            return np.zeros((0, self.config.context_len, 1)), np.zeros((0, self.config.forecast_len, 1))

        return np.array(viz_samples_x), np.array(viz_samples_y)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Track metrics and create visualizations at epoch end."""
        logs = logs or {}

        # Track history
        self.training_history['loss'].append(logs.get('loss', 0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0))

        # Handle different metrics for Point vs Quantile
        if self.config.use_quantile_head:
            self.training_history['metric'].append(logs.get('mae_of_median', 0))
            self.training_history['val_metric'].append(logs.get('val_mae_of_median', 0))
            metric_name = "MAE (Median)"
        else:
            self.training_history['metric'].append(logs.get('mae', 0))
            self.training_history['val_metric'].append(logs.get('val_mae', 0))
            metric_name = "MAE"

        # Track learning rate
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        if hasattr(self.model.optimizer.learning_rate, '__call__'):
            lr = float(keras.ops.convert_to_numpy(
                self.model.optimizer.learning_rate(self.model.optimizer.iterations)
            ))
        self.training_history['lr'].append(lr)

        # Visualize every N epochs
        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Creating visualizations for epoch {epoch + 1}...")
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch, metric_name)
            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int, metric_name: str) -> None:
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        epochs_range = range(1, len(self.training_history['loss']) + 1)

        # Loss
        axes[0].plot(epochs_range, self.training_history['loss'], label='Train')
        axes[0].plot(epochs_range, self.training_history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Metric
        axes[1].plot(epochs_range, self.training_history['metric'], label='Train')
        axes[1].plot(epochs_range, self.training_history['val_metric'], label='Val')
        axes[1].set_title(metric_name)
        axes[1].set_ylabel(metric_name)
        axes[1].legend()

        # Learning Rate
        axes[2].plot(epochs_range, self.training_history['lr'])
        axes[2].set_title('Learning Rate')
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png'))
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        context, target = self.viz_test_data

        if len(context) == 0:
            return

        # Make predictions
        predictions = self.model.predict(context, verbose=0)
        # predictions shape:
        # Point: (batch, forecast_len, num_features)
        # Quantile: (batch, forecast_len, num_features, num_quantiles)

        num_samples = min(self.config.plot_top_k_patterns, len(context))
        n_cols = 3
        n_rows = math.ceil(num_samples / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        axes = np.array(axes).flatten()

        # Setup quantile indices if needed
        if self.config.use_quantile_head:
            quantiles = self.config.quantile_levels
            try:
                median_idx = quantiles.index(0.5)
            except ValueError:
                median_idx = len(quantiles) // 2

            # Use outer quantiles for confidence interval (e.g., 10th and 90th)
            low_idx = 0
            high_idx = -1

        for i in range(num_samples):
            ax = axes[i]

            # Flatten to 1D for plotting (assuming 1 feature or taking 0-th feature)
            ctx_data = context[i, :, 0].flatten()
            tgt_data = target[i, :, 0].flatten()

            x_ctx = np.arange(len(ctx_data))
            x_tgt = np.arange(len(ctx_data), len(ctx_data) + len(tgt_data))

            ax.plot(x_ctx, ctx_data, label='Context', color='blue')
            ax.plot(x_tgt, tgt_data, label='Target', color='green', linestyle='--')

            if self.config.use_quantile_head:
                # Quantile Plotting
                pred_median = predictions[i, :, 0, median_idx].flatten()
                pred_low = predictions[i, :, 0, low_idx].flatten()
                pred_high = predictions[i, :, 0, high_idx].flatten()

                ax.plot(x_tgt, pred_median, label='Median', color='red', alpha=0.9)
                ax.fill_between(
                    x_tgt, pred_low, pred_high,
                    color='red', alpha=0.2,
                    label=f'{quantiles[low_idx]}-{quantiles[high_idx]} Q'
                )
            else:
                # Point Plotting
                pred_data = predictions[i, :, 0].flatten()
                ax.plot(x_tgt, pred_data, label='Pred', color='red', alpha=0.7)

            if i == 0:
                ax.legend(loc='upper left', fontsize='small')
            ax.set_title(f'Sample {i}')

        for j in range(num_samples, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'PRISM Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'))
        plt.close()


class PRISMTrainer:
    """Trainer for PRISM using robust data handling."""

    def __init__(
        self,
        config: PRISMTrainingConfig,
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

        # Initialize Data Processor
        self.processor = PRISMDataProcessor(
            config, self.generator, self.selected_patterns, self.pattern_to_category
        )

        self.model: Optional[keras.Model] = None
        self.exp_dir: Optional[str] = None

    def _select_patterns(self) -> List[str]:
        if self.config.target_categories:
            patterns_to_consider = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            patterns_to_consider = self.all_patterns

        selected, category_counts = [], {}
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

    def run_experiment(self) -> Dict[str, Any]:
        logger.info("Starting PRISM training experiment")
        self.exp_dir = self._create_experiment_dir()
        logger.info(f"Results will be saved to: {self.exp_dir}")

        # Data Pipeline
        data_pipeline = self.processor.prepare_datasets()

        # Build Model
        num_features = 1
        self.model = self._build_model(num_features)

        # Test build/summary
        dummy_input = np.zeros((1, self.config.context_len, num_features), dtype='float32')
        self.model(dummy_input)
        logger.info("Model created successfully")
        logger.info(f"Model parameters: {self.model.count_params():,}")

        # Train
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
        mode = "quantile" if self.config.use_quantile_head else "point"
        exp_name = f"{self.config.experiment_name}_{self.config.preset}_{mode}_{timestamp}"
        exp_dir = os.path.join(self.config.result_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _build_model(self, num_features: int) -> keras.Model:
        logger.info(f"Building PRISM model (preset={self.config.preset})")

        model_kwargs = {
            "preset": self.config.preset,
            "context_len": self.config.context_len,
            "forecast_len": self.config.forecast_len,
            "num_features": num_features,
            "num_layers": self.config.num_layers,
            "tree_depth": self.config.tree_depth,
            "overlap_ratio": self.config.overlap_ratio,
            "num_wavelet_levels": self.config.num_wavelet_levels,
            "router_hidden_dim": self.config.router_hidden_dim,
            "router_temperature": self.config.router_temperature,
            "dropout_rate": self.config.dropout_rate,
            "ffn_expansion": self.config.ffn_expansion,

            # Quantile/Probabilistic arguments
            "use_quantile_head": self.config.use_quantile_head,
            "num_quantiles": len(self.config.quantile_levels) if self.config.use_quantile_head else 3,
            "quantile_levels": self.config.quantile_levels if self.config.use_quantile_head else None,
            "enforce_monotonicity": self.config.enforce_monotonicity
        }

        if self.config.hidden_dim is not None:
            model_kwargs["hidden_dim"] = self.config.hidden_dim

        model = PRISMModel.from_preset(**model_kwargs)

        # Optimizer & Schedule
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
        else:
            lr_schedule = self.config.learning_rate

        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = lr_schedule
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        # Explicit Build
        model.build((None, self.config.context_len, num_features))

        # ---------------------------------------------------------------------
        # Compilation: Switch between Point (MSE) and Quantile Loss
        # ---------------------------------------------------------------------
        if self.config.use_quantile_head:
            logger.info("Compiling with QuantileLoss")
            loss = QuantileLoss(quantiles=self.config.quantile_levels)

            # Custom metric to track MAE of the Median (0.5 quantile)
            if 0.5 in self.config.quantile_levels:
                median_idx = self.config.quantile_levels.index(0.5)

                def mae_of_median(y_true, y_pred):
                    # y_true: (batch, forecast_len, num_features)
                    # y_pred: (batch, forecast_len, num_features, quantiles)
                    return keras.metrics.mean_absolute_error(
                        y_true, y_pred[:, :, :, median_idx]
                    )
                mae_of_median.__name__ = 'mae_of_median'
                metrics = [mae_of_median]
            else:
                metrics = []
        else:
            logger.info("Compiling with MSE Loss")
            loss = 'mse'
            metrics = ['mae', 'mse']

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
            # jit_compile=True  <-- REMOVED TO PREVENT NANs IN ROUTER/WAVELET
        )

        return model

    def _train_model(self, data_pipeline: Dict[str, Any], exp_dir: str) -> Dict[str, Any]:
        logger.info("Starting model training...")

        viz_dir = os.path.join(exp_dir, 'visualizations')
        callbacks = [
            PRISMPerformanceCallback(self.config, self.processor, viz_dir, "prism"),
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
                analyze_weights=True, analyze_spectral=True,
                analyze_calibration=False, analyze_information_flow=False,
                analyze_training_dynamics=False, verbose=False
            )
            analyzer_cb = EpochAnalyzerCallback(
                output_dir=os.path.join(exp_dir, 'deep_analysis'),
                analysis_config=analysis_config,
                start_epoch=self.config.analysis_start_epoch,
                epoch_frequency=self.config.analysis_frequency,
                model_name="PRISM"
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
            'test_metrics': test_metrics,
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        def json_convert(o):
            if isinstance(o, (np.floating, np.integer)): return str(o)
            return str(o)

        serializable = {
            'history': results['history'],
            'test_metrics': {k: float(v) for k, v in results['test_metrics'].items()},
            'config': self.config.__dict__
        }

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_convert)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PRISM Training Framework")

    parser.add_argument("--experiment_name", type=str, default="prism", help="Exp name")
    parser.add_argument("--preset", type=str, default="small", choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument("--context_len", type=int, default=168)
    parser.add_argument("--forecast_len", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--tree_depth", type=int, default=2)
    parser.add_argument("--num_wavelet_levels", type=int, default=3)

    # Probabilistic args
    parser.add_argument("--use_quantile_head", action="store_true", help="Enable probabilistic forecasting")
    parser.add_argument("--no_monotonicity", dest="enforce_monotonicity", action="store_false")
    parser.set_defaults(enforce_monotonicity=True)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=500)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw")

    parser.add_argument("--no-warmup", dest="use_warmup", action="store_false")
    parser.set_defaults(use_warmup=True)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)

    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false")
    parser.set_defaults(normalize_per_instance=True)
    parser.add_argument("--max_patterns_per_category", type=int, default=10)

    parser.add_argument("--visualize_every_n_epochs", type=int, default=5)
    parser.add_argument("--plot_top_k_patterns", type=int, default=12)

    parser.add_argument("--no-deep-analysis", dest="perform_deep_analysis", action="store_false")
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10)
    parser.add_argument("--analysis_start_epoch", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PRISMTrainingConfig(
        experiment_name=args.experiment_name,
        preset=args.preset,
        context_len=args.context_len,
        forecast_len=args.forecast_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        tree_depth=args.tree_depth,
        num_wavelet_levels=args.num_wavelet_levels,
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
        analysis_start_epoch=args.analysis_start_epoch,

        # Quantile config
        use_quantile_head=args.use_quantile_head,
        enforce_monotonicity=args.enforce_monotonicity
    )

    # Use the new config dataclass from the generator module
    generator_config = TimeSeriesGeneratorConfig(
        n_samples=10000,
        random_seed=42,
        default_noise_level=0.1
    )

    trainer = PRISMTrainer(config, generator_config)
    results = trainer.run_experiment()
    logger.info(f"Completed. Results: {results['results_dir']}")
    sys.exit(0)


if __name__ == "__main__":
    main()