"""
Comprehensive TiRex Training Framework using N-BEATS Style Infrastructure.
"""

import os
import json
import math
import random
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
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.models.tirex.model import create_tirex_by_variant, TiRexCore
from dl_techniques.datasets.time_series import TimeSeriesConfig, TimeSeriesGenerator

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
    embed_dim: int = 128
    num_blocks: int = 6
    num_heads: int = 8
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
    normalize_per_instance: bool = False

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

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")
        if 0.5 not in self.quantile_levels:
            logger.warning("Recommended to include 0.5 (median) in quantile_levels.")


class TiRexDataProcessor:
    """Streams multi-pattern time series data using Python generators."""

    def __init__(
            self,
            config: TiRexTrainingConfig,
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

    def _training_generator(
        self
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Create an infinite generator for training data with high diversity.
        Strategy: Select patterns -> Extract windows -> Shuffle -> Yield.
        """
        patterns_to_mix = 50
        windows_per_pattern = 5
        buffer = []

        while True:
            if not buffer:
                # Select a large mix of patterns based on weights
                selected_patterns = random.choices(
                    self.weighted_patterns, self.weights, k=patterns_to_mix
                )
                new_samples = []

                for pattern_name in selected_patterns:
                    data = self.ts_generator.generate_task_data(pattern_name)
                    train_size = int(self.config.train_ratio * len(data))
                    train_data = data[:train_size]

                    max_start_idx = max(
                        len(train_data) - self.config.input_length - self.config.prediction_length, 0
                    )

                    if max_start_idx <= 0:
                        continue

                    for _ in range(windows_per_pattern):
                        start_idx = random.randint(0, max_start_idx)

                        # Inputs: (input_length, 1)
                        inputs = train_data[
                            start_idx: start_idx + self.config.input_length
                        ].astype(np.float32).reshape(-1, 1)

                        # Targets: (prediction_length,) -> Flat vector for QuantileLoss
                        targets = train_data[
                            start_idx + self.config.input_length:
                            start_idx + self.config.input_length + self.config.prediction_length
                        ].astype(np.float32).flatten()

                        new_samples.append((inputs, targets))

                random.shuffle(new_samples)
                buffer.extend(new_samples)

                if not buffer:
                    continue

            yield buffer.pop(0)

    def _evaluation_generator(
            self,
            split: str
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Create an infinite repeating generator for validation or test data.
        """
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            raise ValueError("Split must be 'val' or 'test'")

        all_sequences = []
        for pattern_name in self.selected_patterns:
            data = self.ts_generator.generate_task_data(pattern_name)
            start_idx = int(start_ratio * len(data))
            end_idx = int(end_ratio * len(data))
            split_data = data[start_idx:end_idx]

            seq_len = self.config.input_length + self.config.prediction_length

            # Extract all possible sequences
            for i in range(len(split_data) - seq_len + 1):
                inputs = split_data[
                    i: i + self.config.input_length
                ].astype(np.float32).reshape(-1, 1)

                targets = split_data[
                    i + self.config.input_length: i + seq_len
                ].astype(np.float32).flatten()

                all_sequences.append((inputs, targets))

        if not all_sequences:
            logger.warning(f"No sequences found for {split} split!")
            dummy_in = np.zeros((self.config.input_length, 1), dtype=np.float32)
            dummy_out = np.zeros((self.config.prediction_length,), dtype=np.float32)
            while True:
                yield dummy_in, dummy_out
        else:
            logger.info(f"Created {len(all_sequences)} sequences for {split} split")
            random.shuffle(all_sequences)
            idx = 0
            while True:
                yield all_sequences[idx]
                idx = (idx + 1) % len(all_sequences)
                if idx == 0:
                    random.shuffle(all_sequences)

    def get_evaluation_steps(self, split: str) -> int:
        """Calculate the number of steps for a full evaluation pass."""
        total_samples = 0
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            return 1

        for _ in self.selected_patterns:
            data_len = self.ts_generator.config.n_samples
            start_idx = int(start_ratio * data_len)
            end_idx = int(end_ratio * data_len)
            split_len = end_idx - start_idx
            num_sequences = (
                split_len - self.config.input_length - self.config.prediction_length + 1
            )
            if num_sequences > 0:
                total_samples += num_sequences

        steps = max(1, math.ceil(total_samples / self.config.batch_size))
        return steps

    def _normalize_instance(
            self,
            inputs: tf.Tensor,
            targets: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Standardize an individual time series instance (z-score normalization).
        Calculates stats from inputs ONLY, applies to inputs and targets.
        """
        # inputs: (input_length, 1)
        # targets: (prediction_length,)

        mean = tf.reduce_mean(inputs)
        std = tf.maximum(tf.math.reduce_std(inputs), 1e-7)

        normalized_inputs = (inputs - mean) / std
        normalized_targets = (targets - mean) / std

        return normalized_inputs, normalized_targets

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create the complete tf.data pipeline."""
        output_signature = (
            tf.TensorSpec(shape=(self.config.input_length, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(self.config.prediction_length,), dtype=tf.float32)
        )

        train_ds = tf.data.Dataset.from_generator(
            generator=self._training_generator,
            output_signature=output_signature
        )
        val_ds = tf.data.Dataset.from_generator(
            generator=lambda: self._evaluation_generator('val'),
            output_signature=output_signature
        )
        test_ds = tf.data.Dataset.from_generator(
            generator=lambda: self._evaluation_generator('test'),
            output_signature=output_signature
        )

        if self.config.normalize_per_instance:
            logger.info("Applying per-instance standardization.")
            train_ds = train_ds.map(self._normalize_instance, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(self._normalize_instance, num_parallel_calls=tf.data.AUTOTUNE)
            test_ds = test_ds.map(self._normalize_instance, num_parallel_calls=tf.data.AUTOTUNE)

        # Deep shuffling pipeline
        train_ds = (
            train_ds.shuffle(buffer_size=6343, reshuffle_each_iteration=True)
            .batch(self.config.batch_size)
            .shuffle(buffer_size=163, reshuffle_each_iteration=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': self.get_evaluation_steps('val'),
            'test_steps': self.get_evaluation_steps('test')
        }


class TiRexPerformanceCallback(keras.callbacks.Callback):
    """Callback for monitoring and visualizing performance on a fixed test set."""

    def __init__(
            self,
            config: TiRexTrainingConfig,
            data_processor: TiRexDataProcessor,
            save_dir: str,
            model_name: str = "model"
    ):
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.save_dir = save_dir
        self.model_name = model_name
        self.training_history = {
            'epoch': [], 'lr': [],
            'loss': [], 'val_loss': [],
            'mae_median': [], 'val_mae_median': []
        }
        os.makedirs(save_dir, exist_ok=True)
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a diverse, fixed test set for visualizations."""
        logger.info("Creating a diverse, fixed visualization test set...")
        num_samples = self.config.plot_top_k_patterns
        patterns_to_sample = self.data_processor.selected_patterns.copy()
        random.shuffle(patterns_to_sample)

        x_list, y_list, patterns_sampled = [], [], 0
        start_ratio = self.config.train_ratio + self.config.val_ratio
        end_ratio = 1.0

        for pattern_name in patterns_to_sample:
            if patterns_sampled >= num_samples:
                break
            data = self.data_processor.ts_generator.generate_task_data(pattern_name)
            start_idx = int(start_ratio * len(data))
            end_idx = int(end_ratio * len(data))
            test_data = data[start_idx:end_idx]

            total_len = self.config.input_length + self.config.prediction_length
            if len(test_data) >= total_len:
                max_start = len(test_data) - total_len
                sample_start_idx = random.randint(0, max_start)

                inputs = test_data[
                    sample_start_idx: sample_start_idx + self.config.input_length
                ].astype(np.float32).reshape(-1, 1)

                targets = test_data[
                    sample_start_idx + self.config.input_length:
                    sample_start_idx + total_len
                ].astype(np.float32).flatten()

                # Apply normalization if config enabled, to match model expectation
                if self.config.normalize_per_instance:
                    mean = np.mean(inputs)
                    std = max(np.std(inputs), 1e-7)
                    inputs = (inputs - mean) / std
                    targets = (targets - mean) / std

                x_list.append(inputs)
                y_list.append(targets)
                patterns_sampled += 1

        if not x_list:
            return (
                np.array([]).reshape(0, self.config.input_length, 1),
                np.array([]).reshape(0, self.config.prediction_length)
            )

        return np.array(x_list), np.array(y_list)

    def on_epoch_end(
            self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """Actions to perform at the end of each epoch."""
        logs = logs or {}
        self.training_history['epoch'].append(epoch)
        self.training_history['lr'].append(self.model.optimizer.learning_rate)
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))

        # Track MAE of median if available
        mae_key = next((k for k in logs.keys() if 'mae_of_median' in k and 'val' not in k), None)
        val_mae_key = next((k for k in logs.keys() if 'mae_of_median' in k and 'val' in k), None)

        if mae_key:
            self.training_history['mae_median'].append(logs.get(mae_key, 0.0))
            self.training_history['val_mae_median'].append(logs.get(val_mae_key, 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        try:
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots and self.viz_test_data[0].shape[0] > 0:
                self._plot_prediction_samples(epoch)
        except Exception as e:
            logger.warning(f"Failed to create interim plots: {e}")

    def _plot_learning_curves(self, epoch: int) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs_range = self.training_history['epoch']

        # Loss
        axes[0].plot(epochs_range, self.training_history['loss'], label='Train')
        axes[0].plot(epochs_range, self.training_history['val_loss'], label='Val')
        axes[0].set_title('Quantile Loss')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # MAE Median
        if self.training_history['mae_median']:
            axes[1].plot(epochs_range, self.training_history['mae_median'], label='Train')
            axes[1].plot(epochs_range, self.training_history['val_mae_median'], label='Val')
            axes[1].set_title('MAE (Median Forecast)')
        axes[1].set_ylabel('MAE')
        axes[1].legend()

        # Learning Rate
        axes[2].plot(epochs_range, self.training_history['lr'])
        axes[2].set_title('Learning Rate')
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png'))
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data
        # Predictions shape: (batch, num_quantiles, prediction_length)
        preds = self.model(test_x, training=False)
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()

        quantiles = self.config.quantile_levels
        try:
            median_idx = quantiles.index(0.5)
        except ValueError:
            median_idx = len(quantiles) // 2

        # Get bounds for 80% confidence interval (0.1 to 0.9 typically)
        low_idx = 0
        high_idx = -1
        if len(quantiles) >= 3:
             # Assuming sorted quantiles like [0.1, ... 0.9]
            low_idx = 0
            high_idx = len(quantiles) - 1

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            input_x = np.arange(-self.config.input_length, 0)
            pred_x = np.arange(0, self.config.prediction_length)

            # Input history
            ax.plot(input_x, test_x[i].flatten(), label='Input', color='blue', alpha=0.7)

            # True Future
            ax.plot(pred_x, test_y[i].flatten(), label='True', color='green', linewidth=2)

            # Median Prediction
            ax.plot(pred_x, preds[i, median_idx, :], label='Median', color='red', linestyle='--')

            # Uncertainty Interval
            ax.fill_between(
                pred_x,
                preds[i, low_idx, :],
                preds[i, high_idx, :],
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
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'))
        plt.close()


class TiRexTrainer:
    """Orchestrates TiRex training with Quantile Loss."""

    def __init__(
            self, config: TiRexTrainingConfig, ts_config: TimeSeriesConfig
    ) -> None:
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
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

    def create_model(self) -> TiRexCore:
        """Create and compile the TiRex model."""
        model = create_tirex_by_variant(
            variant=self.config.variant,
            input_length=self.config.input_length,
            prediction_length=self.config.prediction_length,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            num_blocks=self.config.num_blocks,
            num_heads=self.config.num_heads,
            quantile_levels=self.config.quantile_levels,
            dropout_rate=self.config.dropout_rate
        )

        # Learning Rate Schedule
        if self.config.use_warmup:
            primary_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=self.config.steps_per_epoch * self.config.epochs,
                alpha=0.1
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

        # Loss and Metrics
        loss = QuantileLoss(quantiles=self.config.quantile_levels)

        metrics = []
        if 0.5 in self.config.quantile_levels:
            median_idx = self.config.quantile_levels.index(0.5)
            def mae_of_median(y_true, y_pred):
                # y_true: (batch, horizon)
                # y_pred: (batch, quantiles, horizon)
                return keras.metrics.mean_absolute_error(
                    y_true, y_pred[:, median_idx, :]
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
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting TiRex Experiment: {exp_dir}")

        results = self._train_model(exp_dir)
        self._save_results(results, exp_dir)
        return {"results_dir": exp_dir, "results": results}

    def _train_model(self, exp_dir: str) -> Dict[str, Any]:
        data_pipeline = self.processor.prepare_datasets()
        model = self.create_model()

        # Build explicitly
        dummy_in = np.zeros((1, self.config.input_length, 1), dtype='float32')
        model(dummy_in)
        model.summary(print_fn=logger.info)

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
        test_metrics = model.evaluate(
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
        serializable = {
            'history': results['history'],
            'test_metrics': {k: float(v) for k, v in results['test_metrics'].items()},
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__
        }

        # Helper for JSON serialization
        def json_convert(o):
            if isinstance(o, (np.floating, np.integer)): return str(o)
            return str(o)

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_convert)


def main() -> None:
    config = TiRexTrainingConfig(
        experiment_name="tirex",
        variant="small",
        input_length=104,
        prediction_length=12,
        patch_size=4,
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        epochs=200,
        batch_size=128,
        steps_per_epoch=1000,
        learning_rate=1e-4,
        use_warmup=True,
        warmup_steps=2000,
        warmup_start_lr=1e-6,
        gradient_clip_norm=1.0,
        optimizer='adamw',
        normalize_per_instance=False,
        max_patterns_per_category=100,
        visualize_every_n_epochs=5,
        plot_top_k_patterns=12,
    )

    ts_config = TimeSeriesConfig(n_samples=5000, random_seed=42)

    try:
        trainer = TiRexTrainer(config, ts_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()