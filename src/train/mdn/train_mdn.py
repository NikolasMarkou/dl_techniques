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

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    generate_training_curves,
    set_seeds,
    json_numpy_default,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
)
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


@dataclass
class MDNTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for Multi-Task MDN training.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags)
    from :class:`BaseTimeSeriesTrainingConfig` and adds the MDN architecture
    fields below. mdn is multi-task / uniform-sampling, so the inherited
    ``category_weights`` / ``use_warmup`` / ``target_categories`` / ``warmup_*``
    fields are unused (harmless) and intentionally not re-declared. A handful of
    inherited defaults are re-declared because the MDN originals differ from the
    base: ``batch_size`` (256 vs 128), ``steps_per_epoch`` (200 vs 500),
    ``learning_rate`` (5e-4 vs 1e-4), ``plot_top_k_patterns`` (9 vs 12). The
    ``optimizer`` ('adamw'), ``gradient_clip_norm`` (1.0),
    ``max_patterns_per_category`` (10), and ``normalize_per_instance`` (True)
    match the base defaults and are dropped.
    """

    experiment_name: str = "mdn_multitask"

    # Re-declared: MDN originals differ from the base defaults.
    batch_size: int = 256            # base default: 128
    steps_per_epoch: int = 200       # base default: 500
    learning_rate: float = 5e-4      # base default: 1e-4
    plot_top_k_patterns: int = 9     # base default: 12

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
    weight_decay: float = 1e-4

    # Visualization / forecasting
    confidence_level: float = 0.95
    num_forecast_samples: int = 100
    visualize_every_n_epochs: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


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


class MDNDataProcessor(WindowedTimeSeriesProcessor):
    """Multi-task MDN data processor: subclass of :class:`WindowedTimeSeriesProcessor`.

    mdn is the divergent (multi-task) call site: uniform pattern sampling
    (``pattern_to_category=None``), ROBUST per-instance normalization, and a
    nested ``((sequence, task_id), target)`` sample structure. Both axes are
    expressed via the two base hooks (:meth:`_make_sample` emits the task id and
    the base ``tf.nest`` stacking handles the nested structure) plus ctor params
    (``windows_per_pattern=10``, ``min_length_multiplier=2``,
    ``require_finite=False`` preserving the original no-skip behavior).
    """

    def __init__(self, config: MDNTrainingConfig, generator: TimeSeriesGenerator,
                 selected_patterns: List[str]):
        super().__init__(
            config,
            generator,
            selected_patterns,
            pattern_to_category=None,  # uniform sampling
            context_len=config.window_size,
            horizon_len=config.pred_horizon,
            num_features=1,
            normalize=True,
            normalize_method=NormalizationMethod.ROBUST,
            windows_per_pattern=10,
            min_length_multiplier=2,
            require_finite=False,
        )
        logger.info(f"Initialized processor with {self.num_tasks} tasks")

    def _make_sample(self, window: np.ndarray, pattern_name: str) -> Tuple[Any, Any]:
        ctx = self.context_len
        task_id = self.pattern_to_id[pattern_name]
        x_seq = window[:ctx].reshape(-1, 1).astype(np.float32)
        y = window[ctx:].reshape(-1).astype(np.float32)
        return (x_seq, np.array([task_id], dtype=np.int32)), y

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        return (
            (tf.TensorSpec(shape=(self.context_len, 1), dtype=tf.float32),
             tf.TensorSpec(shape=(1,), dtype=tf.int32)),
            tf.TensorSpec(shape=(self.horizon_len,), dtype=tf.float32),
        )


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
        generate_training_curves(
            history=self.history_log,
            results_dir=self.save_dir,
            filename=f"learning_curves_epoch_{epoch+1:03d}",
        )

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
        serializable = {
            'history': results['history'],
            'test_metrics': {k: float(v) for k, v in results['test_metrics'].items()},
            'config': self.config.__dict__
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_numpy_default)


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
    set_seeds(42)
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
