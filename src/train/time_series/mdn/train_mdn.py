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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    set_seeds,
    json_numpy_default,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
    BaseTimeSeriesTrainer,
    create_ts_argument_parser,
)
from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.mdn import MDNModel
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    NormalizationMethod,
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


class MDNPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks and visualizes MDN probabilistic forecast performance.

    Thin subclass of :class:`TimeSeriesPerformanceCallback`. The base owns the
    scaffolding (``__init__`` + makedirs, ``loss``/``val_loss`` accumulation, the
    ``visualize_every_n_epochs`` gate, learning-curve delegation to
    ``generate_training_curves``). mdn is the divergent (A4 / D-002) call site:
    its callback is seeded with a PRE-BUILT ``viz_data`` tuple
    (``((seq_stack, task_stack), y_stack)`` = the ``test_data_raw`` produced by
    ``prepare_datasets``), NOT a processor — so :meth:`_prepare_viz_data` simply
    returns the stored tuple, bypassing the processor path the trio use. mdn
    tracks NO extra history keys and NO learning rate, so :meth:`_extend_history`
    is left as the base no-op. mdn's config lacks ``create_learning_curves`` /
    ``create_prediction_plots``, so the base's ``getattr(..., True)`` flag checks
    default True → the original "always plot both" behavior is preserved.
    """

    def __init__(self, config: MDNTrainingConfig,
                 viz_data: Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
                 save_dir: str, model_name: str = "model"):
        # viz_data must be stored BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() which returns self._viz_data.
        self._viz_data = viz_data
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[Any, Any]:
        # A4 / D-002: return the pre-built (inputs, targets) tuple verbatim; the
        # base no-op default (empty arrays / processor path) is intentionally
        # bypassed — mdn never touches a processor in its callback.
        return self._viz_data

    @property
    def viz_inputs(self):
        return self.viz_test_data[0]

    @property
    def viz_targets(self):
        return self.viz_test_data[1]

    def _plot_predictions(self, epoch: int) -> None:
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
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch+1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


class MDNTrainer(BaseTimeSeriesTrainer):
    """Orchestrates Multi-Task MDN training.

    Thin subclass of :class:`BaseTimeSeriesTrainer`. The base owns the skeleton
    (``__init__`` generator/pattern setup, ``_create_experiment_dir``,
    ``_train_model`` fit+evaluate, ``run_experiment``). mdn is the most divergent
    call site (multi-task, uniform sampling, pre-built ``viz_data``, no ONNX) and
    overrides:

    - :meth:`_select_patterns` — UNIFORM ``random.sample`` (D-002), NOT the base
      category-balanced selection. The base ``__init__`` still builds
      ``pattern_to_category`` (harmless; the processor passes ``None``).
    - :meth:`_build_processor` — the multi-task :class:`MDNDataProcessor` (3-arg
      signature, ``pattern_to_category=None`` internally).
    - :meth:`_build_model` — bespoke :class:`MultiTaskMDNModel` + dummy-input
      warmup + ``mdn_loss_wrapper`` compile.
    - :meth:`_make_callbacks` — ``include_analyzer=False`` / ``patience=15`` /
      ``model_name="MDN"`` (mdn never runs the deep analyzer), and seeds the
      performance callback with ``self._test_data_raw`` (the ``test_data_raw``
      the base ``_train_model`` exposes before calling this).
    - :meth:`_save_results` — mdn's original key set is ``{history, test_metrics,
      config}`` (NO ``final_epoch``, NO ``onnx_path``), so this overrides the
      base 4-key write to match exactly.
    - :meth:`_build_performance_callback` — abstract base hook; mdn builds it
      inside :meth:`_make_callbacks` instead (it needs the viz tuple), so this
      raises if ever called via the base path.
    """

    def _select_patterns(self) -> List[str]:
        # DECISION plan_2026-06-09_a3c7304c/D-002: uniform random.sample, NOT the
        # base category-balanced _select_patterns. mdn is multi-task with uniform
        # sampling (pattern_to_category=None in its processor). Do NOT replace this
        # with the base body — that would re-introduce category weighting mdn never
        # had. See decisions.md D-002.
        all_patterns = self.all_patterns
        if self.config.max_patterns:
            return random.sample(all_patterns, self.config.max_patterns)
        return all_patterns

    def _build_processor(self) -> MDNDataProcessor:
        return MDNDataProcessor(self.config, self.generator, self.selected_patterns)

    def _build_model(self) -> MultiTaskMDNModel:
        num_tasks = self.processor.num_tasks
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
        return model

    def _build_performance_callback(self, viz_dir: str):
        # mdn builds its callback inside _make_callbacks (it needs the pre-built
        # viz_data tuple). This base hook is intentionally unused.
        raise NotImplementedError("MDN builds its callback in _make_callbacks")

    def _make_callbacks(self, exp_dir: Optional[str] = None) -> List:
        """Override: mdn uses ``include_analyzer=False`` / ``patience=15`` /
        ``model_name="MDN"`` and seeds the perf callback with the pre-built
        ``test_data_raw`` tuple (exposed on ``self._test_data_raw`` by the base
        ``_train_model``). ``use_lr_schedule=False`` (mdn uses a constant LR).
        """
        # DECISION plan_2026-06-10_39646d39/D-002
        # Pass a BARE prefix (self._build_results_prefix()) to
        # create_common_callbacks and adopt its RETURNED results_dir as
        # self.exp_dir -- the D-009 bare-prefix contract, mirroring prism/tirex.
        # Do NOT pass the pre-created full exp_dir path as results_dir_prefix and
        # do NOT rely on the base _create_experiment_dir here: passing the full
        # path as prefix built a SEPARATE doubly-nested
        # results/{full}_MDN_{ts2}/ that received CSVLogger/ModelCheckpoint while
        # results.json/visualizations landed in the discarded first dir.
        # The exp_dir param is ignored on purpose. See decisions.md D-009.
        callbacks, results_dir = create_common_callbacks(
            model_name="MDN",
            results_dir_prefix=self._build_results_prefix(),
            monitor="val_loss",
            patience=15,
            use_lr_schedule=False,
            include_terminate_on_nan=True,
            include_analyzer=False,
        )
        self.exp_dir = results_dir
        viz_dir = os.path.join(self.exp_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        callbacks.append(
            MDNPerformanceCallback(self.config, self._test_data_raw, viz_dir, "MDN"))
        return callbacks

    def run_experiment(self) -> Dict[str, Any]:
        """Base skeleton with the D-009 dir resolution (mirrors prism/tirex).

        Overridden so ``self.exp_dir`` is resolved from
        ``create_common_callbacks``' returned dir (inside ``_train_model`` ->
        ``_make_callbacks``) instead of the base ``_create_experiment_dir``.
        Passing ``exp_dir=None`` to ``_train_model`` means no first dir is
        pre-built; CSVLogger/ModelCheckpoint, results.json, and visualizations
        all land in the single returned dir.
        """
        logger.info(f"Starting {self.config.experiment_name} training experiment")

        data_pipeline = self.processor.prepare_datasets()
        self.model = self._build_model()
        logger.info(f"Model params: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        # _train_model -> _make_callbacks sets self.exp_dir (D-009).
        training_results = self._train_model(data_pipeline, exp_dir=None)
        logger.info(f"Results: {self.exp_dir}")

        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config, 'experiment_dir': self.exp_dir,
            'training_results': training_results, 'results_dir': self.exp_dir
        }

    def _save_results(self, results: Dict[str, Any], exp_dir: str,
                      extra_fields: Optional[Dict[str, Any]] = None) -> None:
        # mdn's original results.json key set is {history, test_metrics, config}
        # ONLY (no final_epoch, no onnx_path). Override the base 4-key write to
        # match exactly. (extra_fields is part of the base signature; mdn never
        # passes any.)
        serializable = {
            'history': results['history'],
            'test_metrics': {k: float(v) for k, v in results['test_metrics'].items()},
            # Schema parity (plan_2026-06-09_49c73926 N1): include final_epoch like the
            # other TS trainers. Robust fallback to len(history.loss) if absent.
            'final_epoch': results.get('final_epoch', len(results['history']['loss'])),
            'config': self.config.__dict__,
        }
        if extra_fields:
            serializable.update(extra_fields)
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_numpy_default)


def build_parser() -> argparse.ArgumentParser:
    """Build the MDN CLI parser on top of the shared TS argument parser.

    Starts from :func:`create_ts_argument_parser` (the shared TS args), restores
    mdn's own defaults via ``set_defaults`` (``experiment_name="mdn_multitask"``,
    ``epochs=100``, ``batch_size=256``, ``steps_per_epoch=200``,
    ``learning_rate=5e-4``, ``plot_top_k_patterns=9``), then adds mdn's
    architecture-specific flags. The shared parser additionally exposes warmup /
    analysis flags mdn's original CLI lacked — the harmless superset pattern used
    by the prism / tirex / nbeats migrations (mdn's config ignores warmup).
    """
    parser = create_ts_argument_parser("Multi-Task MDN Training")
    parser.set_defaults(
        experiment_name="mdn_multitask",
        epochs=100,
        batch_size=256,
        steps_per_epoch=200,
        learning_rate=5e-4,
        plot_top_k_patterns=9,
    )
    # MDN architecture-specific arguments.
    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--num_mixtures", type=int, default=12)
    parser.add_argument("--task_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--no-attention", dest="use_attention", action="store_false")
    parser.set_defaults(use_attention=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(42)
    setup_gpu(args.gpu)

    config = MDNTrainingConfig(
        experiment_name=args.experiment_name,
        window_size=args.window_size,
        num_mixtures=args.num_mixtures,
        task_embedding_dim=args.task_embedding_dim,
        dropout_rate=args.dropout_rate,
        use_attention=args.use_attention,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=5000, random_seed=42, default_noise_level=0.1
    )

    # Preserve mdn's original hard-exit teardown (os._exit(0)): a deliberate
    # force-exit that skips Python/atexit cleanup to avoid TF prefetch-thread
    # hangs on shutdown. Do NOT normalize to a sys.exit-only/finally template
    # (per-script choice, not consolidatable duplication). See
    # plan_2026-06-09_a3c7304c/decisions.md D-008.
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
