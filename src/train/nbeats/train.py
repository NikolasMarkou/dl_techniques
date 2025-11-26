"""
Comprehensive N-BEATS Training Framework - Normalized & Unified

This framework trains the N-BEATS architecture with:
1. Robust Instance Normalization (-1, +1 scaling based on backcast).
2. High-entropy reservoir sampling (Shuffling like TiRex).
3. Infinite Data Streams (Prevents OUT_OF_RANGE errors).
4. Diverse Visualization (Ensures plots show different patterns).
"""

import os
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
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.models.nbeats.model import create_nbeats_model
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
class NBeatsTrainingConfig:
    """Configuration dataclass for N-BEATS training."""
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats"

    # Pattern selection configuration
    target_categories: Optional[List[str]] = None

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific configuration
    backcast_length: int = 168
    forecast_length: int = 24
    input_dim: int = 1

    # Model architecture
    stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 128
    use_normalization: bool = True  # Model internal normalization layers
    use_bias: bool = True
    activation: str = "relu"

    # Training configuration
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 1000
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adam'
    primary_loss: Union[str, keras.losses.Loss] = "mae"
    mase_seasonal_periods: int = 1

    # Learning rate schedule with warmup
    use_warmup: bool = True
    warmup_steps: int = 5000
    warmup_start_lr: float = 1e-6

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.1

    # Pattern selection & Data Processing
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 100
    normalize_per_instance: bool = True  # Enforce [-1, 1] scaling

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
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


class MultiPatternDataProcessor:
    """Streams multi-pattern time series data using Python generators."""

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

    def _normalize_window(
        self, backcast: np.ndarray, forecast: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the instance to the range [-1, +1] using Backcast statistics.
        Formula: X_norm = (X - min) / (max - min) * 2 - 1
        """
        if not self.config.normalize_per_instance:
            return backcast, forecast

        # Calculate statistics from the backcast (history) ONLY
        min_val = np.min(backcast)
        max_val = np.max(backcast)
        scale = max_val - min_val

        # Avoid division by zero
        if scale < 1e-7:
            scale = 1.0

        # Apply normalization to match range [-1, +1]
        backcast_norm = ((backcast - min_val) / scale) * 2.0 - 1.0
        forecast_norm = ((forecast - min_val) / scale) * 2.0 - 1.0

        return backcast_norm.astype(np.float32), forecast_norm.astype(np.float32)

    def _training_generator(
        self
    ) -> Generator[Tuple[np.ndarray, Union[np.ndarray, Tuple]], None, None]:
        """
        Create an infinite generator for training data with high diversity.
        """
        patterns_to_mix = 50
        windows_per_pattern = 5
        buffer = []

        while True:
            if not buffer:
                # 1. Select a mix of patterns
                try:
                    selected_patterns = random.choices(
                        self.weighted_patterns, self.weights, k=patterns_to_mix
                    )
                except IndexError:
                    selected_patterns = self.selected_patterns

                new_samples = []

                for pattern_name in selected_patterns:
                    data = self.ts_generator.generate_task_data(pattern_name)
                    train_size = int(self.config.train_ratio * len(data))
                    train_data = data[:train_size]

                    total_len = self.config.backcast_length + self.config.forecast_length
                    max_start_idx = len(train_data) - total_len

                    if max_start_idx <= 0:
                        continue

                    # 2. Extract random windows
                    for _ in range(windows_per_pattern):
                        start_idx = random.randint(0, max_start_idx)

                        backcast_raw = train_data[
                            start_idx : start_idx + self.config.backcast_length
                        ].reshape(-1, 1)

                        forecast_raw = train_data[
                            start_idx + self.config.backcast_length :
                            start_idx + total_len
                        ].reshape(-1, 1)

                        # 3. Apply Instance Normalization [-1, 1]
                        x, y = self._normalize_window(backcast_raw, forecast_raw)

                        if self.config.reconstruction_loss_weight > 0.0:
                            residual_target = x.flatten()
                            new_samples.append((x, (y, residual_target)))
                        else:
                            new_samples.append((x, y))

                if not new_samples:
                    continue

                # 4. Shuffle the refill batch thoroughly
                random.shuffle(new_samples)
                buffer.extend(new_samples)

            yield buffer.pop(0)

    def _evaluation_generator(
            self, split: str
    ) -> Generator[Tuple[np.ndarray, Union[np.ndarray, Tuple]], None, None]:
        """
        Create an INFINITE generator for validation or test data.
        """
        if split == 'val':
            start_ratio = self.config.train_ratio
            end_ratio = self.config.train_ratio + self.config.val_ratio
        elif split == 'test':
            start_ratio = self.config.train_ratio + self.config.val_ratio
            end_ratio = 1.0
        else:
            raise ValueError("Split must be 'val' or 'test'")

        task_list = self.selected_patterns.copy()

        while True:
            # Shuffle tasks every pass for validation to keep batches statistically consistent
            if split == 'val':
                random.shuffle(task_list)

            samples_yielded_in_pass = 0

            for pattern_name in task_list:
                data = self.ts_generator.generate_task_data(pattern_name)
                start_idx = int(start_ratio * len(data))
                end_idx = int(end_ratio * len(data))
                split_data = data[start_idx:end_idx]

                total_len = self.config.backcast_length + self.config.forecast_length
                stride = self.config.forecast_length // 2
                if stride < 1: stride = 1

                max_start = len(split_data) - total_len

                if max_start <= 0:
                    continue

                for i in range(0, max_start + 1, stride):
                    backcast_raw = split_data[i : i + self.config.backcast_length].reshape(-1, 1)
                    forecast_raw = split_data[
                        i + self.config.backcast_length : i + total_len
                    ].reshape(-1, 1)

                    x, y = self._normalize_window(backcast_raw, forecast_raw)

                    samples_yielded_in_pass += 1
                    if self.config.reconstruction_loss_weight > 0.0:
                        yield x, (y, x.flatten())
                    else:
                        yield x, y

            if samples_yielded_in_pass == 0:
                logger.warning(f"No samples found for {split} split. Yielding zero-sample.")
                x_dummy = np.zeros((self.config.backcast_length, 1), dtype=np.float32)
                y_dummy = np.zeros((self.config.forecast_length, 1), dtype=np.float32)
                if self.config.reconstruction_loss_weight > 0.0:
                    yield x_dummy, (y_dummy, x_dummy.flatten())
                else:
                    yield x_dummy, y_dummy

    def get_evaluation_steps(self, split: str) -> int:
        return max(10, len(self.selected_patterns) * 5)

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create the complete tf.data pipeline."""
        x_shape = (self.config.backcast_length, 1)
        y_shape = (self.config.forecast_length, 1)

        if self.config.reconstruction_loss_weight > 0.0:
            rec_shape = (self.config.backcast_length * self.config.input_dim,)
            output_signature = (
                tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                (
                    tf.TensorSpec(shape=y_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=rec_shape, dtype=tf.float32)
                )
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                tf.TensorSpec(shape=y_shape, dtype=tf.float32)
            )

        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=output_signature
        ).repeat().shuffle(1603).batch(self.config.batch_size).shuffle(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: self._evaluation_generator('val'), output_signature=output_signature
        ).repeat().batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_generator(
            lambda: self._evaluation_generator('test'), output_signature=output_signature
        ).repeat().batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': self.get_evaluation_steps('val'),
            'test_steps': self.get_evaluation_steps('test')
        }


class PatternPerformanceCallback(keras.callbacks.Callback):
    """Callback for monitoring and visualizing performance on a fixed test set."""

    def __init__(
            self,
            config: NBeatsTrainingConfig,
            data_processor: MultiPatternDataProcessor,
            save_dir: str,
            model_name: str = "model"
    ):
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.save_dir = save_dir
        self.model_name = model_name
        self.training_history = {
            'loss': [], 'val_loss': [], 'forecast_mae': [], 'val_forecast_mae': []
        }
        os.makedirs(save_dir, exist_ok=True)
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a GUARANTEED DIVERSE visualization test set.
        Instead of using the sequential generator, we pick 1 sample from N different patterns.
        """
        logger.info("Creating a diverse, fixed visualization test set...")

        x_list, y_list = [], []

        # Get list of patterns and shuffle them to pick random ones
        available_patterns = self.data_processor.selected_patterns.copy()
        random.shuffle(available_patterns)

        # Ratio for test split
        start_ratio = self.config.train_ratio + self.config.val_ratio

        # Iterate through distinct patterns to ensure variety
        for pattern_name in available_patterns:
            if len(x_list) >= self.config.plot_top_k_patterns:
                break

            data = self.data_processor.ts_generator.generate_task_data(pattern_name)

            # Slice test split
            start_idx_split = int(start_ratio * len(data))
            test_data = data[start_idx_split:]

            total_len = self.config.backcast_length + self.config.forecast_length
            max_start = len(test_data) - total_len

            if max_start <= 0:
                continue

            # Pick a random window within this specific pattern's test data
            rand_idx = random.randint(0, max_start)

            backcast_raw = test_data[rand_idx : rand_idx + self.config.backcast_length].reshape(-1, 1)
            forecast_raw = test_data[
                rand_idx + self.config.backcast_length :
                rand_idx + total_len
            ].reshape(-1, 1)

            # Normalize
            x, y = self.data_processor._normalize_window(backcast_raw, forecast_raw)

            x_list.append(x)
            y_list.append(y)

        if not x_list:
            logger.warning("Could not generate viz samples.")
            return np.array([]), np.array([])

        logger.info(f"Created {len(x_list)} diverse samples from different patterns.")
        return np.array(x_list), np.array(y_list)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        logs = logs or {}
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))
        self.training_history['forecast_mae'].append(logs.get('forecast_mae', 0.0))
        self.training_history['val_forecast_mae'].append(logs.get('val_forecast_mae', 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int) -> None:
        if self.config.create_learning_curves:
            self._plot_learning_curves(epoch)
        if self.config.create_prediction_plots and len(self.viz_test_data[0]) > 0:
            self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.training_history['loss'], label='Train Loss')
        axes[0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0].set_title(f'Total Loss (Epoch {epoch+1})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.training_history['forecast_mae'], label='Train MAE')
        axes[1].plot(self.training_history['val_forecast_mae'], label='Val MAE')
        axes[1].set_title('Forecast MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch+1:03d}.png'))
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data
        predictions_tuple = self.model(test_x, training=False)

        if isinstance(predictions_tuple, (tuple, list)):
            predictions = predictions_tuple[0]
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
        else:
            predictions = predictions_tuple.numpy()

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            backcast_steps = np.arange(-self.config.backcast_length, 0)
            forecast_steps = np.arange(0, self.config.forecast_length)

            ax.plot(backcast_steps, test_x[i].flatten(), label='Backcast', color='blue', alpha=0.6)
            ax.plot(forecast_steps, test_y[i].flatten(), label='True Future', color='green')
            ax.plot(forecast_steps, predictions[i].flatten(), label='Pred Future', color='red', linestyle='--')

            ax.set_title(f'Sample {i+1}')
            if i == 0: ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'N-BEATS Predictions (Epoch {epoch + 1})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'))
        plt.close()


class NBeatsTrainer:
    """Orchestrates the training, evaluation, and reporting of N-BEATS models."""

    def __init__(
            self, config: NBeatsTrainingConfig, ts_config: TimeSeriesConfig
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
        self.processor = MultiPatternDataProcessor(
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

    def create_model(self) -> keras.Model:
        """Create and compile an N-BEATS Keras model."""
        kernel_regularizer = None
        if self.config.kernel_regularizer_l2 > 0:
            kernel_regularizer = keras.regularizers.L2(
                self.config.kernel_regularizer_l2
            )

        # Instantiate the base N-BEATS architecture
        base_model = create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=self.config.forecast_length,
            stack_types=self.config.stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            activation=self.config.activation,
            use_normalization=self.config.use_normalization,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=self.config.dropout_rate,
            reconstruction_weight=self.config.reconstruction_loss_weight,
            input_dim=self.config.input_dim,
            output_dim=1
        )

        # Configure Learning Rate
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
        else:
            lr_schedule = self.config.learning_rate

        # Configure Optimizer
        optimizer_cls = keras.optimizers.get(self.config.optimizer).__class__
        optimizer = optimizer_cls(
            learning_rate=lr_schedule,
            clipnorm=self.config.gradient_clip_norm
        )

        # Prepare Loss Functions and Model Wrapping
        if self.config.reconstruction_loss_weight > 0.0:
            # Case A: Reconstruction Enabled
            model = base_model

            if self.config.primary_loss == 'mase_loss':
                forecast_loss = MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
            else:
                forecast_loss = keras.losses.get(self.config.primary_loss)

            losses = [
                forecast_loss,
                keras.losses.MeanAbsoluteError(name="residual_loss")
            ]
            loss_weights = [1.0, self.config.reconstruction_loss_weight]
            metrics = [
                [keras.metrics.MeanAbsoluteError(name="forecast_mae")],
                [keras.metrics.MeanAbsoluteError(name="residual_mae")]
            ]
        else:
            # Case B: Forecast Only - Wrap base_model to select first output
            inputs = keras.Input(shape=(self.config.backcast_length, self.config.input_dim))
            raw_output = base_model(inputs)

            if isinstance(raw_output, (list, tuple)):
                forecast = raw_output[0]
            else:
                forecast = raw_output

            model = keras.Model(inputs=inputs, outputs=forecast, name="nbeats_forecast_only")

            if self.config.primary_loss == 'mase_loss':
                losses = MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
            else:
                losses = keras.losses.get(self.config.primary_loss)

            loss_weights = None
            metrics = [keras.metrics.MeanAbsoluteError(name="forecast_mae")]

        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        return model

    def run_experiment(self) -> Dict[str, Any]:
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting N-BEATS Experiment: {exp_dir}")

        data_pipeline = self.processor.prepare_datasets()
        results = self._train_model(data_pipeline, exp_dir)

        self._save_results(results, exp_dir)
        return {"results_dir": exp_dir, "results": results}

    def _train_model(self, data_pipeline: Dict, exp_dir: str) -> Dict[str, Any]:
        model = self.create_model()
        model.build((None, self.config.backcast_length, self.config.input_dim))
        model.summary(print_fn=logger.info)

        viz_dir = os.path.join(exp_dir, 'visualizations')
        performance_cb = PatternPerformanceCallback(
            self.config, self.processor, viz_dir, "nbeats"
        )

        model_path = os.path.join(exp_dir, 'best_model.keras')

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1
            ),
            performance_cb,
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
        test_results = model.evaluate(
            data_pipeline['test_ds'],
            steps=data_pipeline['test_steps'],
            verbose=1,
            return_dict=True
        )

        return {
            'history': history.history,
            'test_metrics': {k: float(v) for k, v in test_results.items()},
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__
        }

        def default(o):
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            return str(o)

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=default)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for N-BEATS training."""
    parser = argparse.ArgumentParser(description="N-BEATS Training (Normalized & Unified)")

    parser.add_argument("--experiment_name", type=str, default="nbeats",
                        help="Name for logging.")
    parser.add_argument("--backcast_length", type=int, default=168)
    parser.add_argument("--forecast_length", type=int, default=24)

    # Training Loop
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # Data Processing
    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false",
                        help="Disable [-1, 1] normalization.")
    parser.set_defaults(normalize_per_instance=True)

    # N-BEATS Structure
    parser.add_argument("--stack_types", nargs='+', default=["trend", "seasonality", "generic"])
    parser.add_argument("--hidden_layer_units", type=int, default=128)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.5,
                        help="Weight for backcast reconstruction loss (default: 0.5)")

    return parser.parse_args()


def main() -> None:
    """Main function to configure and run the N-BEATS training experiment."""
    args = parse_args()

    config = NBeatsTrainingConfig(
        experiment_name=args.experiment_name,
        backcast_length=args.backcast_length,
        forecast_length=args.forecast_length,
        stack_types=args.stack_types,
        hidden_layer_units=args.hidden_layer_units,
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        gradient_clip_norm=args.gradient_clip_norm,
        # Data
        normalize_per_instance=args.normalize_per_instance,
        reconstruction_loss_weight=args.reconstruction_loss_weight
    )

    ts_config = TimeSeriesConfig(
        n_samples=5000,
        random_seed=42
    )

    try:
        trainer = NBeatsTrainer(config, ts_config)
        results = trainer.run_experiment()
        logger.info(f"Experiment completed! Results saved to: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()