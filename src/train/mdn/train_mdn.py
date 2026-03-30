"""
Training pipeline for the Multi-Task MDN (Mixture Density Network) framework.

Combines task-aware embeddings with a deep feature extractor and probabilistic
output layer for multi-task time series forecasting with uncertainty quantification.

References:
    Bishop (1994) - Mixture Density Networks
    Gal (2016) - Uncertainty in Deep Learning
"""

import os
import sys
import json
import random
import argparse
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
from scipy import stats

from train.common import setup_gpu, create_callbacks as create_common_callbacks
from dl_techniques.utils.logger import logger
from dl_techniques.models.mdn import MDNModel
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    TimeSeriesNormalizer,
    NormalizationMethod
)

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
class MDNTrainingConfig:
    """Configuration for Multi-Task MDN training."""
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "mdn_multitask"

    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Sequence
    window_size: int = 120
    pred_horizon: int = 1
    stride: int = 1

    # Model architecture
    num_mixtures: int = 12
    hidden_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    task_embedding_dim: int = 32
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    use_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 64

    # Calibration
    use_temperature_scaling: bool = True
    initial_temperature: float = 1.0
    calibration_weight: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 256
    steps_per_epoch: int = 200
    learning_rate: float = 5e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4

    # Pattern selection
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    normalize_per_instance: bool = True

    # Visualization
    confidence_level: float = 0.95
    num_forecast_samples: int = 100
    visualize_every_n_epochs: int = 5
    plot_top_k_patterns: int = 9

    def __post_init__(self) -> None:
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


def _fill_nans(data: np.ndarray) -> np.ndarray:
    """Forward fill NaNs in numpy array."""
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return data[idx]


@keras.saving.register_keras_serializable()
class MultiTaskMDNModel(keras.Model):
    """Multi-task wrapper around MDNModel with task embeddings, Conv1D, and attention."""

    def __init__(self, num_tasks: int, config: MDNTrainingConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_tasks = num_tasks

        self.task_embedding = keras.layers.Embedding(
            input_dim=num_tasks, output_dim=config.task_embedding_dim, name="task_embedding")
        self.conv1 = keras.layers.Conv1D(64, 7, padding="same", activation="gelu")
        self.norm1 = keras.layers.LayerNormalization()
        self.conv2 = keras.layers.Conv1D(128, 5, padding="same", activation="gelu")
        self.norm2 = keras.layers.LayerNormalization()

        if config.use_attention:
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=config.attention_heads, key_dim=config.attention_dim, name="seq_attention")
            self.att_norm = keras.layers.LayerNormalization()

        self.flatten = keras.layers.Flatten()
        self.mdn_core = MDNModel(
            hidden_layers=config.hidden_units, output_dimension=1,
            num_mixtures=config.num_mixtures, dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm)

        if config.use_temperature_scaling:
            self.temperature = self.add_weight(
                name="temperature", shape=(),
                initializer=keras.initializers.Constant(config.initial_temperature),
                trainable=True)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        sequence_input, task_input = inputs

        x = self.conv1(sequence_input)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.config.use_attention:
            att = self.attention(x, x)
            x = self.att_norm(x + att)

        seq_features = self.flatten(x)
        task_emb = self.task_embedding(task_input)
        if len(task_emb.shape) == 3:
            task_emb = keras.ops.squeeze(task_emb, axis=1)

        combined = keras.ops.concatenate([seq_features, task_emb], axis=-1)
        return self.mdn_core(combined, training=training)

    def get_mdn_layer(self):
        return self.mdn_core.mdn_layer


class MDNDataProcessor:
    """Data processor for Multi-Task MDN with task ID mapping and streaming generation."""

    def __init__(self, config: MDNTrainingConfig, generator: TimeSeriesGenerator,
                 selected_patterns: List[str]):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_id = {name: i for i, name in enumerate(selected_patterns)}
        self.id_to_pattern = {i: name for name, i in self.pattern_to_id.items()}
        self.num_tasks = len(selected_patterns)
        logger.info(f"Initialized processor with {self.num_tasks} tasks")

    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        series = np.clip(series, -1e6, 1e6)
        if self.config.normalize_per_instance:
            normalizer = TimeSeriesNormalizer(method=NormalizationMethod.ROBUST)
            if np.isnan(series).any():
                series = _fill_nans(series)
            series = normalizer.fit_transform(series)
        series = np.clip(series, -10.0, 10.0)
        return series.astype(np.float32)

    def _training_generator(self) -> Generator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray], None, None]:
        """Infinite generator yielding ((sequence, task_id), target)."""
        patterns_to_mix, windows_per_pattern = 50, 10
        buffer = []
        total_len = self.config.window_size + self.config.pred_horizon

        while True:
            if not buffer:
                for name in random.choices(self.selected_patterns, k=patterns_to_mix):
                    task_id = self.pattern_to_id[name]
                    try:
                        data = self.ts_generator.generate_task_data(name)
                    except Exception:
                        continue
                    if len(data) < total_len * 2:
                        continue
                    train_data = data[:int(self.config.train_ratio * len(data))]
                    if len(train_data) < total_len:
                        continue
                    max_start = len(train_data) - total_len
                    for _ in range(windows_per_pattern):
                        start = random.randint(0, max_start)
                        window = self._safe_normalize(train_data[start:start + total_len])
                        x_seq = window[:self.config.window_size].reshape(-1, 1)
                        y_target = window[self.config.window_size:].reshape(-1)
                        buffer.append(((x_seq, np.array([task_id], dtype=np.int32)), y_target))
                random.shuffle(buffer)
            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(self, split: str, num_samples: int
                                ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Pre-compute dataset for val/test."""
        logger.info(f"Pre-computing {split} dataset ({num_samples} samples)")
        x_seqs, x_tasks, y_targets = [], [], []
        total_len = self.config.window_size + self.config.pred_horizon
        collected, cycle = 0, 0

        while collected < num_samples:
            name = self.selected_patterns[cycle % self.num_tasks]
            cycle += 1
            task_id = self.pattern_to_id[name]
            try:
                data = self.ts_generator.generate_task_data(name)
            except Exception:
                continue

            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))
            split_data = data[train_end:val_end] if split == 'val' else data[val_end:]
            if len(split_data) < total_len:
                continue

            start = random.randint(0, len(split_data) - total_len)
            window = self._safe_normalize(split_data[start:start + total_len])
            x_seqs.append(window[:self.config.window_size].reshape(-1, 1))
            x_tasks.append([task_id])
            y_targets.append(window[self.config.window_size:].reshape(-1))
            collected += 1

        return (
            (np.array(x_seqs, dtype=np.float32), np.array(x_tasks, dtype=np.int32)),
            np.array(y_targets, dtype=np.float32)
        )

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create tf.data.Datasets for train/val/test."""
        output_sig = (
            (tf.TensorSpec(shape=(self.config.window_size, 1), dtype=tf.float32),
             tf.TensorSpec(shape=(1,), dtype=tf.int32)),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=output_sig
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        val_steps = max(50, self.num_tasks)
        test_steps = max(20, self.num_tasks)

        val_inputs, val_y = self._generate_fixed_dataset('val', val_steps * self.config.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_y)).batch(
            self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        test_inputs, test_y = self._generate_fixed_dataset('test', test_steps * self.config.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_y)).batch(
            self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds, 'val_ds': val_ds, 'test_ds': test_ds,
            'validation_steps': val_steps, 'test_steps': test_steps,
            'test_data_raw': (test_inputs, test_y)
        }


class MDNPerformanceCallback(keras.callbacks.Callback):
    """Tracks and visualizes MDN probabilistic forecast performance."""

    def __init__(self, config: MDNTrainingConfig, save_dir: str,
                 viz_data: Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]):
        super().__init__()
        self.config = config
        self.save_dir = save_dir
        self.viz_inputs, self.viz_targets = viz_data
        os.makedirs(self.save_dir, exist_ok=True)
        self.history_log = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        self.history_log['loss'].append(logs.get('loss', 0))
        self.history_log['val_loss'].append(logs.get('val_loss', 0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Visualizations for epoch {epoch + 1}")
            self._plot_learning_curves(epoch)
            self._plot_probabilistic_predictions(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.history_log['loss']) + 1)
        plt.plot(epochs, self.history_log['loss'], label='Train Loss')
        plt.plot(epochs, self.history_log['val_loss'], label='Val Loss')
        plt.title('MDN Negative Log-Likelihood')
        plt.xlabel('Epochs'); plt.ylabel('Loss (NLL)'); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch+1:03d}.png'))
        plt.close()

    def _plot_probabilistic_predictions(self, epoch: int) -> None:
        """Visualize MDN outputs: context, target, and predicted distribution."""
        total_samples = len(self.viz_targets)
        indices = np.random.choice(total_samples, min(self.config.plot_top_k_patterns, total_samples), replace=False)

        sample_seq = self.viz_inputs[0][indices]
        sample_task = self.viz_inputs[1][indices]
        sample_target = self.viz_targets[indices]

        params = self.model.predict((sample_seq, sample_task), verbose=0)
        mdn_layer = self.model.get_mdn_layer()
        mus, sigmas, pis = mdn_layer.split_mixture_params(params)

        mus = keras.ops.convert_to_numpy(mus)
        sigmas = keras.ops.convert_to_numpy(sigmas)
        pis = keras.ops.convert_to_numpy(pis)
        pis = np.exp(pis) / np.sum(np.exp(pis), axis=1, keepdims=True)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(indices):
                break

            ctx = sample_seq[i].flatten()
            tgt = sample_target[i]
            time_steps = np.arange(len(ctx))
            future_step = len(ctx)

            ax.plot(time_steps, ctx, label='Context', color='blue', alpha=0.6)
            ax.scatter([future_step], [tgt], label='True Target', color='green', marker='x', s=100, zorder=5)

            # Reconstruct mixture PDF and compute percentiles
            y_min = min(ctx.min(), float(tgt)) - 2.0
            y_max = max(ctx.max(), float(tgt)) + 2.0
            y_grid = np.linspace(y_min, y_max, 200)

            pdf_values = np.zeros_like(y_grid)
            for k in range(self.config.num_mixtures):
                pdf_values += pis[i, k] * stats.norm.pdf(y_grid, mus[i, k], sigmas[i, k])

            cdf_values = np.cumsum(pdf_values)
            if cdf_values[-1] > 0:
                cdf_values /= cdf_values[-1]
                idx_05 = np.clip(np.searchsorted(cdf_values, 0.05), 0, len(y_grid) - 1)
                idx_50 = np.clip(np.searchsorted(cdf_values, 0.50), 0, len(y_grid) - 1)
                idx_95 = np.clip(np.searchsorted(cdf_values, 0.95), 0, len(y_grid) - 1)

                lower, median_pred, upper = y_grid[idx_05], y_grid[idx_50], y_grid[idx_95]
                ax.scatter([future_step], [median_pred], label='Median Pred', color='red', alpha=0.8)
                ax.errorbar([future_step], [median_pred],
                            yerr=[[median_pred - lower], [upper - median_pred]],
                            fmt='none', ecolor='red', alpha=0.3, capsize=5, label='90% CI')

            ax.set_title(f'Sample {i} (Task {sample_task[i][0]})')
            if i == 0:
                ax.legend(loc='upper left', fontsize='small')

        plt.suptitle(f'MDN Probabilistic Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch+1:03d}.png'))
        plt.close()


class MDNTrainer:
    """Trainer for Multi-Task MDN."""

    def __init__(self, config: MDNTrainingConfig,
                 generator_config: TimeSeriesGeneratorConfig) -> None:
        self.config = config
        self.generator = TimeSeriesGenerator(generator_config)
        all_patterns = self.generator.get_task_names()

        if config.max_patterns:
            self.selected_patterns = random.sample(all_patterns, config.max_patterns)
        else:
            self.selected_patterns = all_patterns

        self.processor = MDNDataProcessor(config, self.generator, self.selected_patterns)
        self.model: Optional[MultiTaskMDNModel] = None

    def run_experiment(self) -> Dict[str, Any]:
        logger.info("Starting Multi-Task MDN experiment")
        self.exp_dir = self._create_experiment_dir()
        data_pipeline = self.processor.prepare_datasets()
        self.model = self._build_model(self.processor.num_tasks)
        training_results = self._train_model(data_pipeline, self.exp_dir)

        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config, 'experiment_dir': self.exp_dir,
            'results': training_results
        }

    def _create_experiment_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(self.config.result_dir, f"{self.config.experiment_name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _build_model(self, num_tasks: int) -> MultiTaskMDNModel:
        logger.info(f"Building Multi-Task MDN for {num_tasks} tasks")
        model = MultiTaskMDNModel(num_tasks, self.config)

        dummy_seq = tf.zeros((1, self.config.window_size, 1))
        dummy_task = tf.zeros((1, 1), dtype=tf.int32)
        model((dummy_seq, dummy_task))

        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = self.config.learning_rate
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        def mdn_loss_wrapper(y_true, y_pred):
            base_loss = model.get_mdn_layer().loss_func(y_true, y_pred)
            if self.config.use_temperature_scaling:
                temp_penalty = keras.ops.square(model.temperature - 1.0) * self.config.calibration_weight
                return base_loss + temp_penalty
            return base_loss

        model.compile(optimizer=optimizer, loss=mdn_loss_wrapper)
        model.summary(print_fn=logger.info)
        return model

    def _train_model(self, data_pipeline: Dict[str, Any], exp_dir: str) -> Dict[str, Any]:
        viz_dir = os.path.join(exp_dir, 'visualizations')

        callbacks, _ = create_common_callbacks(
            model_name="MDN",
            results_dir_prefix=exp_dir,
            monitor="val_loss",
            patience=15,
            use_lr_schedule=False,
            include_terminate_on_nan=True,
            include_analyzer=False,
        )
        callbacks.append(MDNPerformanceCallback(self.config, viz_dir, data_pipeline['test_data_raw']))

        history = self.model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks, verbose=1
        )

        logger.info("Evaluating on test set")
        test_metrics = self.model.evaluate(
            data_pipeline['test_ds'], steps=data_pipeline['test_steps'], return_dict=True)

        return {'history': history.history, 'test_metrics': test_metrics}

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        def json_convert(o):
            if isinstance(o, (np.floating, np.integer)):
                return str(o)
            return str(o)

        serializable = {
            'history': results['history'],
            'test_metrics': {k: float(v) for k, v in results['test_metrics'].items()},
            'config': self.config.__dict__
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_convert)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Task MDN Training")
    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--num_mixtures", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--visualize_every_n_epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_gpu(args.gpu)

    config = MDNTrainingConfig(
        window_size=args.window_size,
        num_mixtures=args.num_mixtures,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        visualize_every_n_epochs=args.visualize_every_n_epochs
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=5000, random_seed=42, default_noise_level=0.1
    )

    try:
        trainer = MDNTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['experiment_dir']}")
        keras.backend.clear_session()
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

    os._exit(0)


if __name__ == "__main__":
    main()
