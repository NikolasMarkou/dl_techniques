"""N-BEATS training with scientific forecasting layers (NaiveResidual + ForecastabilityGate)."""

import os
import json
import math
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from train.common import setup_gpu, create_callbacks as create_common_callbacks
from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.models.nbeats import create_nbeats_model
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.datasets.time_series import TimeSeriesConfig, TimeSeriesGenerator
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.layers.time_series.forecasting_layers import (
    NaiveResidual, ForecastabilityGate)

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


@dataclass
class NBeatsTrainingConfig:
    """Configuration for N-BEATS training with forecasting layers."""
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats_forecasting_layers"

    target_categories: Optional[List[str]] = None

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_realizations_per_pattern: int = 10

    backcast_length: int = 168
    forecast_length: int = 24
    input_dim: int = 1

    stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 128
    use_normalization: bool = True
    use_bias: bool = True
    activation: str = "gelu"

    use_naive_residual: bool = True
    use_forecastability_gate: bool = True
    gate_hidden_units: int = 16
    gate_activation: str = "relu"

    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 1000
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adam'
    primary_loss: Union[str, keras.losses.Loss] = "mase_loss"
    mase_seasonal_periods: int = 1

    use_warmup: bool = True
    warmup_steps: int = 5000
    warmup_start_lr: float = 1e-6

    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.25

    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 100
    normalize_per_instance: bool = True

    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 1.0, "seasonal": 1.0, "composite": 1.2,
        "financial": 1.5, "weather": 1.3, "biomedical": 1.2,
        "industrial": 1.3, "intermittent": 1.0, "volatility": 1.1,
        "regime": 1.2, "structural": 1.1
    })

    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 12
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    perform_deep_analysis: bool = True
    analysis_frequency: int = 10
    analysis_start_epoch: int = 1

    def __post_init__(self) -> None:
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


class MultiPatternDataProcessor:
    """Manages data loading using pre-generated Tensors and TF Graph sampling."""

    def __init__(self, config: NBeatsTrainingConfig, generator: TimeSeriesGenerator,
                 selected_patterns: List[str], pattern_to_category: Dict[str, str]):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_category = pattern_to_category
        self.pattern_names, self.sampling_weights = self._prepare_weights()
        self.datasets = self._preload_data_tensors()

    def _prepare_weights(self) -> Tuple[List[str], np.ndarray]:
        """Prepare normalized probability weights for each pattern."""
        patterns, weights = [], []
        for pattern_name in self.selected_patterns:
            category = self.pattern_to_category.get(pattern_name, "unknown")
            weight = self.config.category_weights.get(category, 1.0)
            patterns.append(pattern_name)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = np.array([w / total_weight for w in weights], dtype=np.float32)
        else:
            normalized_weights = np.ones(len(weights), dtype=np.float32) / len(weights)
        return patterns, normalized_weights

    def _normalize_sequence(self, series: np.ndarray) -> np.ndarray:
        """Normalize sequence to [-1, 1] range."""
        if not self.config.normalize_per_instance:
            return series
        s_min = np.min(series)
        s_max = np.max(series)
        rng = s_max - s_min
        if rng < 1e-7:
            return np.zeros_like(series)
        return 2.0 * (series - s_min) / rng - 1.0

    def _preload_data_tensors(self) -> Dict[str, tf.Tensor]:
        """Pre-generate and cache data tensors. Shape: (NumPatterns, NumRealizations, TimeSteps, 1)."""
        logger.info("Pre-loading and normalizing dataset into memory Tensors...")

        num_patterns = len(self.pattern_names)
        num_realizations = self.config.num_realizations_per_pattern

        dummy = self.ts_generator.generate_task_data(self.pattern_names[0])
        total_len = len(dummy)

        idx_train = int(total_len * self.config.train_ratio)
        idx_val = int(total_len * (self.config.train_ratio + self.config.val_ratio))

        train_buffer = np.zeros((num_patterns, num_realizations, idx_train, 1), dtype=np.float32)
        val_buffer = np.zeros((num_patterns, num_realizations, idx_val - idx_train, 1), dtype=np.float32)
        test_buffer = np.zeros((num_patterns, num_realizations, total_len - idx_val, 1), dtype=np.float32)

        for p_idx, pattern_name in enumerate(self.pattern_names):
            for r_idx in range(num_realizations):
                raw_series = self.ts_generator.generate_task_data(pattern_name)
                norm_series = self._normalize_sequence(raw_series).reshape(-1, 1)
                train_buffer[p_idx, r_idx] = norm_series[:idx_train]
                val_buffer[p_idx, r_idx] = norm_series[idx_train:idx_val]
                test_buffer[p_idx, r_idx] = norm_series[idx_val:]

        logger.info(f"Data tensors ready. Train shape: {train_buffer.shape}, Size: {train_buffer.nbytes / 1e6:.1f} MB")
        return {
            'train': tf.constant(train_buffer),
            'val': tf.constant(val_buffer),
            'test': tf.constant(test_buffer)
        }

    def _create_tf_dataset(self, data_tensor: tf.Tensor, is_training: bool = True) -> tf.data.Dataset:
        """Create a pure TF dataset pipeline using graph-based sampling."""
        data_const = data_tensor
        weights_const = tf.constant(self.sampling_weights)
        log_weights = tf.math.log(tf.reshape(weights_const, (1, -1)))

        num_patterns = data_tensor.shape[0]
        num_realizations = data_tensor.shape[1]
        time_steps = data_tensor.shape[2]

        backcast_len = self.config.backcast_length
        forecast_len = self.config.forecast_length
        window_size = backcast_len + forecast_len
        input_dim = self.config.input_dim
        use_reconstruction = (self.config.reconstruction_loss_weight > 0.0)

        @tf.function
        def sample_window(_):
            p_idx = tf.random.categorical(log_weights, 1, dtype=tf.int32)[0, 0]
            r_idx = tf.random.uniform((), maxval=num_realizations, dtype=tf.int32)
            max_start = time_steps - window_size
            if max_start <= 0:
                return (tf.zeros((backcast_len, input_dim)),
                        tf.zeros((forecast_len, input_dim)))

            start_idx = tf.random.uniform((), maxval=max_start + 1, dtype=tf.int32)
            full_window = data_const[p_idx, r_idx, start_idx : start_idx + window_size]
            x = full_window[:backcast_len]
            y = full_window[backcast_len:]

            x.set_shape([backcast_len, input_dim])
            y.set_shape([forecast_len, input_dim])

            if use_reconstruction:
                rec_target = tf.reshape(x, (-1,))
                rec_target.set_shape([backcast_len * input_dim])
                return x, (y, rec_target)
            else:
                return x, y

        dataset = tf.data.Dataset.from_tensors(0).repeat()
        dataset = dataset.map(sample_window, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def prepare_datasets(self) -> Dict[str, Any]:
        """Prepare TF datasets for training, validation, and testing."""
        val_steps = max(1, int(self.config.steps_per_epoch * self.config.val_ratio))
        test_steps = max(1, int(self.config.steps_per_epoch * self.config.test_ratio))

        train_ds = self._create_tf_dataset(self.datasets['train'], is_training=True)
        val_ds = self._create_tf_dataset(self.datasets['val'], is_training=False)
        test_ds = self._create_tf_dataset(self.datasets['test'], is_training=False)

        logger.info("TF Graph Data Pipeline ready.")
        return {
            'train_ds': train_ds, 'val_ds': val_ds, 'test_ds': test_ds,
            'validation_steps': val_steps, 'test_steps': test_steps
        }


class PatternPerformanceCallback(keras.callbacks.Callback):
    """Callback for monitoring and visualizing performance on a fixed test set."""

    def __init__(self, config: NBeatsTrainingConfig, processor: MultiPatternDataProcessor,
                 viz_dir: str, model_name: str):
        super().__init__()
        self.config = config
        self.processor = processor
        self.viz_dir = viz_dir
        self.model_name = model_name
        self.training_history = {
            'loss': [], 'val_loss': [], 'forecast_mae': [], 'val_forecast_mae': []
        }
        os.makedirs(self.viz_dir, exist_ok=True)
        self.viz_test_data = self._create_viz_test_set()

    def _create_viz_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a diverse visualization test set with 1 sample from N different patterns."""
        logger.info("Creating a diverse, fixed visualization test set...")
        x_list, y_list = [], []
        available_patterns = self.processor.selected_patterns.copy()
        random.shuffle(available_patterns)
        start_ratio = self.config.train_ratio + self.config.val_ratio

        for pattern_name in available_patterns:
            if len(x_list) >= self.config.plot_top_k_patterns:
                break

            data = self.processor.ts_generator.generate_task_data(pattern_name)
            if self.config.normalize_per_instance:
                data = self.processor._normalize_sequence(data)

            start_idx_split = int(start_ratio * len(data))
            test_data = data[start_idx_split:]
            total_len = self.config.backcast_length + self.config.forecast_length
            max_start = len(test_data) - total_len

            if max_start <= 0:
                continue

            rand_idx = random.randint(0, max_start)
            x = test_data[rand_idx : rand_idx + self.config.backcast_length]
            y = test_data[rand_idx + self.config.backcast_length : rand_idx + total_len]
            x = x.reshape(self.config.backcast_length, self.config.input_dim)
            y = y.reshape(self.config.forecast_length, self.config.input_dim)
            x_list.append(x)
            y_list.append(y)

        if not x_list:
            logger.warning("Could not generate viz samples.")
            return np.array([]), np.array([])

        logger.info(f"Created {len(x_list)} diverse samples from different patterns.")
        return np.array(x_list), np.array(y_list)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))

        if (epoch + 1) % self.config.visualize_every_n_epochs != 0:
            return
        if self.config.create_learning_curves:
            self._plot_learning_curves(epoch)
        if self.config.create_prediction_plots and len(self.viz_test_data[0]) > 0:
            self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_history['loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Val Loss')
        plt.title(f'Loss History (Epoch {epoch+1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'learning_curves_epoch_{epoch+1:03d}.png'))
        plt.close()

    def _plot_prediction_samples(self, epoch: int) -> None:
        """Generate and save prediction plots for the fixed visualization set."""
        test_x, test_y = self.viz_test_data
        predictions_tuple = self.model(test_x, training=False)

        if isinstance(predictions_tuple, (list, tuple)):
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

        plt.suptitle(f'{self.model_name} Predictions - Epoch {epoch + 1}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.viz_dir, f'{self.model_name}_predictions_epoch_{epoch + 1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction plot to {save_path}")


class NBeatsTrainer:
    """Main trainer class for N-BEATS with forecasting layers."""

    def __init__(self, config: NBeatsTrainingConfig, ts_config: TimeSeriesConfig):
        self.config = config
        self.ts_config = ts_config

        generator = TimeSeriesGenerator(ts_config)
        all_patterns = generator.get_task_names()
        pattern_to_category = {
            pattern: generator.task_definitions[pattern]["category"]
            for pattern in all_patterns
        }

        if config.target_categories:
            selected_patterns = [
                p for p in all_patterns
                if pattern_to_category[p] in config.target_categories
            ]
        else:
            selected_patterns = all_patterns

        if config.max_patterns:
            selected_patterns = selected_patterns[:config.max_patterns]

        self.processor = MultiPatternDataProcessor(
            config, generator, selected_patterns, pattern_to_category
        )
        logger.info(f"Initialized trainer with {len(selected_patterns)} patterns")
        logger.info(f"Categories: {set(pattern_to_category.values())}")

    def create_model(self) -> keras.Model:
        """Create N-BEATS model with optional forecasting layers."""
        base_model = create_nbeats_model(
            backcast_length=self.config.backcast_length,
            forecast_length=self.config.forecast_length,
            stack_types=self.config.stack_types,
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            activation=self.config.activation,
            use_normalization=self.config.use_normalization,
            dropout_rate=self.config.dropout_rate,
            reconstruction_weight=self.config.reconstruction_loss_weight,
            input_dim=self.config.input_dim,
            output_dim=self.config.input_dim,
            use_bias=self.config.use_bias,
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2)
        )
        base_model.build((None, self.config.backcast_length, self.config.input_dim))

        inputs = keras.Input(
            shape=(self.config.backcast_length, self.config.input_dim), name='input'
        )
        nbeats_outputs = base_model(inputs)

        if isinstance(nbeats_outputs, (list, tuple)):
            deep_forecast = nbeats_outputs[0]
            residual = nbeats_outputs[1]
        else:
            deep_forecast = nbeats_outputs
            residual = None

        if self.config.use_naive_residual:
            logger.info("Adding NaiveResidual layer")
            naive_layer = NaiveResidual(
                forecast_length=self.config.forecast_length, name='naive_residual'
            )
            pure_naive = naive_layer(inputs, keras.ops.zeros_like(deep_forecast))

            if self.config.use_forecastability_gate:
                logger.info("Adding ForecastabilityGate")
                gate = ForecastabilityGate(
                    hidden_units=self.config.gate_hidden_units,
                    activation=self.config.gate_activation,
                    name='forecastability_gate'
                )
                final_forecast = gate(inputs, deep_forecast, pure_naive)
            else:
                final_forecast = naive_layer(inputs, deep_forecast)
        else:
            final_forecast = deep_forecast

        if residual is not None and self.config.reconstruction_loss_weight > 0.0:
            model = keras.Model(
                inputs=inputs, outputs=[final_forecast, residual],
                name="nbeats_with_forecasting_layers"
            )
        else:
            model = keras.Model(
                inputs=inputs, outputs=final_forecast,
                name="nbeats_with_forecasting_layers"
            )

        # Optimizer
        if self.config.use_warmup:
            primary_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=self.config.steps_per_epoch * self.config.epochs,
                alpha=0.1
            )
            schedule = WarmupSchedule(
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
                primary_schedule=primary_schedule
            )
            logger.info(
                f"Warmup schedule: {self.config.warmup_steps} steps, "
                f"{self.config.warmup_start_lr} -> {self.config.learning_rate}"
            )
        else:
            schedule = self.config.learning_rate

        if self.config.optimizer.lower() == 'adam':
            optimizer = keras.optimizers.Adam(
                learning_rate=schedule, clipnorm=self.config.gradient_clip_norm
            )
        elif self.config.optimizer.lower() == 'adamw':
            optimizer = keras.optimizers.AdamW(
                learning_rate=schedule, clipnorm=self.config.gradient_clip_norm
            )
        else:
            optimizer = keras.optimizers.get(self.config.optimizer)

        # Compile
        if residual is not None and self.config.reconstruction_loss_weight > 0.0:
            if self.config.primary_loss == 'mase_loss':
                forecast_loss = MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
            else:
                forecast_loss = keras.losses.get(self.config.primary_loss)

            losses = [forecast_loss, 'mse']
            loss_weights = [1.0, self.config.reconstruction_loss_weight]
            metrics = {
                model.output_names[0]: [
                    keras.metrics.MeanAbsoluteError(name="forecast_mae")
                ]
            }
        else:
            raw_output = model.output
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
            optimizer=optimizer, loss=losses,
            loss_weights=loss_weights, metrics=metrics
        )
        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete training experiment."""
        exp_dir = os.path.join(
            self.config.result_dir,
            f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Starting N-BEATS Experiment: {exp_dir}")

        data_pipeline = self.processor.prepare_datasets()
        results = self._train_model(data_pipeline, exp_dir)
        self._save_results(results, exp_dir)
        return {"results_dir": exp_dir, "results": results}

    def _train_model(self, data_pipeline: Dict, exp_dir: str) -> Dict[str, Any]:
        """Train the model and return results."""
        model = self.create_model()
        model.build((None, self.config.backcast_length, self.config.input_dim))
        model.summary(print_fn=logger.info)

        viz_dir = os.path.join(exp_dir, 'visualizations')

        callbacks, _ = create_common_callbacks(
            model_name="N-BEATS-Forecasting",
            results_dir_prefix=exp_dir,
            monitor="val_loss",
            patience=25,
            use_lr_schedule=self.config.use_warmup,
            include_terminate_on_nan=True,
            include_analyzer=self.config.perform_deep_analysis,
            analyzer_config=AnalysisConfig(
                analyze_weights=True, analyze_spectral=True,
                analyze_calibration=False, analyze_information_flow=False,
                analyze_training_dynamics=False, verbose=False),
            analyzer_start_epoch=self.config.analysis_start_epoch,
            analyzer_epoch_frequency=self.config.analysis_frequency,
        )
        callbacks.append(PatternPerformanceCallback(
            self.config, self.processor, viz_dir, "nbeats_forecasting"
        ))

        history = model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks, verbose=1
        )

        logger.info("Evaluating on test set...")
        test_results = model.evaluate(
            data_pipeline['test_ds'], steps=data_pipeline['test_steps'],
            verbose=1, return_dict=True
        )

        return {
            'history': history.history,
            'test_metrics': {k: float(v) for k, v in test_results.items()},
            'final_epoch': len(history.history['loss'])
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        """Save experiment results to JSON."""
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
    parser = argparse.ArgumentParser(
        description="N-BEATS Training with Scientific Forecasting Layers"
    )

    parser.add_argument("--experiment_name", type=str, default="nbeats_forecasting_layers")
    parser.add_argument("--backcast_length", type=int, default=168)
    parser.add_argument("--forecast_length", type=int, default=24)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false")
    parser.set_defaults(normalize_per_instance=True)

    parser.add_argument("--stack_types", nargs='+', default=["trend", "seasonality", "generic"])
    parser.add_argument("--hidden_layer_units", type=int, default=256)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.5)

    parser.add_argument("--no-naive-residual", dest="use_naive_residual", action="store_false")
    parser.set_defaults(use_naive_residual=True)
    parser.add_argument("--no-forecastability-gate", dest="use_forecastability_gate", action="store_false")
    parser.set_defaults(use_forecastability_gate=True)
    parser.add_argument("--gate_hidden_units", type=int, default=16)
    parser.add_argument("--gate_activation", type=str, default="relu")

    parser.add_argument("--no-deep-analysis", dest="perform_deep_analysis", action="store_false")
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10)
    parser.add_argument("--analysis_start_epoch", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    """Configure and run the N-BEATS training experiment."""
    args = parse_args()

    config = NBeatsTrainingConfig(
        experiment_name=args.experiment_name,
        backcast_length=args.backcast_length,
        forecast_length=args.forecast_length,
        stack_types=args.stack_types,
        hidden_layer_units=args.hidden_layer_units,
        use_naive_residual=args.use_naive_residual,
        use_forecastability_gate=args.use_forecastability_gate,
        gate_hidden_units=args.gate_hidden_units,
        gate_activation=args.gate_activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        gradient_clip_norm=args.gradient_clip_norm,
        normalize_per_instance=args.normalize_per_instance,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch
    )

    ts_config = TimeSeriesConfig(n_samples=5000, random_seed=42)

    try:
        trainer = NBeatsTrainer(config, ts_config)
        results = trainer.run_experiment()
        logger.info(f"Experiment completed! Results saved to: {results['results_dir']}")
        logger.info(
            f"Forecasting layers: NaiveResidual={config.use_naive_residual}, "
            f"ForecastabilityGate={config.use_forecastability_gate}"
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
