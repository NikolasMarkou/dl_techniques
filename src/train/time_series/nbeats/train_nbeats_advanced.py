"""N-BEATS training with scientific forecasting layers (NaiveResidual + ForecastabilityGate).

Step-10 migration (plan_2026-06-11_84296249): the callback now subclasses
:class:`train.common.timeseries.TimeSeriesPerformanceCallback` and the trainer subclasses
:class:`train.common.timeseries.BaseTimeSeriesTrainer`; the hand-rolled ``argparse`` is replaced
by :func:`train.common.create_ts_argument_parser` + ``set_defaults``; ``--seed`` is wired into
``config.seed`` + ``set_seeds`` and the synthetic-data generator config is built via
:func:`build_generator_config`. The D-004 pre-materialized 4-D MIN-MAX data engine
(:class:`MultiPatternDataProcessor`) is KEPT byte-identical (INV-4) — only the callback, the
trainer skeleton, and the parser migrate. ``backcast_length``/``forecast_length`` (paper
terminology) are preserved.
"""

import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from train.common import (
    setup_gpu, set_seeds, BaseTimeSeriesTrainingConfig,
    TimeSeriesPerformanceCallback, BaseTimeSeriesTrainer,
    create_ts_argument_parser,
)
from train.common import create_callbacks as create_common_callbacks
from train.common.args import build_generator_config
from train.common.timeseries import compute_post_hoc_forecast_metrics
from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.models.time_series.nbeats import create_nbeats_model
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.datasets.time_series import TimeSeriesGenerator
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.layers.time_series.forecasting_layers import (
    NaiveResidual, ForecastabilityGate)

plt.style.use('default')
sns.set_palette("husl")


@dataclass
class NBeatsTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for N-BEATS training with forecasting layers.

    Subclasses :class:`BaseTimeSeriesTrainingConfig`, which contributes the
    shared time-series fields (data splits, optimizer/warmup knobs, pattern
    selection, category weights, visualization + deep-analysis flags, ``seed``)
    at their standard defaults. Only the fields whose defaults DIVERGE from the
    base (``experiment_name``, ``steps_per_epoch``, ``optimizer``,
    ``warmup_steps``, ``max_patterns_per_category``) are re-declared here, plus
    all the N-BEATS-architecture-specific fields that the base does not carry.
    The base ``category_weights`` default already equals this script's former
    inline map, so it is inherited unchanged.
    """

    # --- Divergent-default overrides of base fields (kept to preserve behavior) ---
    experiment_name: str = "nbeats_forecasting_layers"
    steps_per_epoch: int = 1000
    optimizer: str = 'adam'
    warmup_steps: int = 5000
    max_patterns_per_category: int = 100

    # --- N-BEATS architecture-specific fields (not in the base) ---
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

    primary_loss: Union[str, keras.losses.Loss] = "mase_loss"
    mase_seasonal_periods: int = 1

    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.25

    def __post_init__(self) -> None:
        # Base enforces the train/val/test ratio-sum invariant.
        super().__post_init__()
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


# DECISION plan_2026-06-11_84296249/D-010: this processor is the D-004-exempt data
# engine and is KEPT byte-identical (INV-4). It is intentionally NOT migrated onto
# ``train.common.WindowedTimeSeriesProcessor`` (see plans/DECISIONS.md D-004 and the
# prior plan_2026-06-09_49c73926). It is a structurally different engine, not a copy
# of the streaming trio's processor:
#   - it pre-materializes a dense ``(NumPatterns, NumRealizations, T, 1)`` tensor
#     and samples windows with an ``@tf.function`` ``tf.random.categorical`` /
#     ``tf.random.uniform`` GRAPH sampler (vs. the base's Python buffered streaming),
#   - it normalizes per instance to MIN-MAX [-1, 1] (vs. the base's STANDARD/ROBUST
#     z-score), and
#   - it builds uncached, infinitely-resampled val/test pipelines sized by
#     ``steps_per_epoch * ratio`` (vs. the base's pre-computed fixed cached splits).
# The base's two reshape hooks cannot replace this sampling engine, normalization,
# or val/test construction. Forcing it would change runtime numerics. Only the
# CALLBACK, the TRAINER SKELETON, and the PARSER were migrated onto the shared base
# (step-10); this engine stays. See decisions.md D-010. Do NOT migrate this onto
# WindowedTimeSeriesProcessor.
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


class PatternPerformanceCallback(TimeSeriesPerformanceCallback):
    """Per-epoch multi-pattern reconstruction/forecast plots for N-BEATS.

    Thin subclass of :class:`TimeSeriesPerformanceCallback`. The base owns the
    scaffolding (``__init__`` + makedirs, ``loss``/``val_loss`` accumulation, the
    ``visualize_every_n_epochs`` gate, learning-curve delegation to
    ``generate_training_curves``, and — critically — the D-001 non-fatal
    try/except guard around viz-data preparation so a viz-prep failure can never
    abort training, I1). This subclass overrides only the two genuinely bespoke
    hooks:

    - :meth:`_prepare_viz_data` — builds a diverse fixed test set of 1 window from
      N different patterns directly from the D-004 generator/processor (the
      bespoke MIN-MAX engine, not a :class:`WindowedTimeSeriesProcessor`), under
      the base non-fatal guard.
    - :meth:`_plot_predictions` — the multi-pattern backcast/forecast grid stays
      bespoke (it titles per pattern sample and handles the recon-on tuple output).

    Per SYSTEM.md, Keras callbacks are NOT
    ``@keras.saving.register_keras_serializable``.
    """

    def __init__(self, config: NBeatsTrainingConfig, processor: MultiPatternDataProcessor,
                 save_dir: str, model_name: str = "nbeats_forecasting"):
        # processor must be stored BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() (inside its non-fatal guard), which reads it.
        self.processor = processor
        super().__init__(config, save_dir, model_name=model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a diverse visualization test set: 1 sample from N patterns."""
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

    def _plot_predictions(self, epoch: int) -> None:
        """Generate and save prediction plots for the fixed visualization set."""
        test_x, test_y = self.viz_test_data
        if len(test_x) == 0:
            return

        predictions_tuple = self.model(test_x, training=False)
        if isinstance(predictions_tuple, (list, tuple)):
            predictions = predictions_tuple[0]
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
        else:
            predictions = predictions_tuple
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()

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
            if i == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'{self.model_name} Predictions - Epoch {epoch + 1}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{self.model_name}_predictions_epoch_{epoch + 1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction plot to {save_path}")


class NBeatsTrainer(BaseTimeSeriesTrainer):
    """Orchestrates N-BEATS-with-forecasting-layers training.

    Subclass of :class:`BaseTimeSeriesTrainer` (step-10 migration). The base owns
    the skeleton (``__init__`` generator/pattern setup, ``_select_patterns``,
    ``_train_model`` ``model.fit`` + ``model.evaluate``). N-BEATS-advanced keeps
    these genuine divergences as overrides:

    - :meth:`_build_processor` — returns the D-004 pre-materialized 4-D MIN-MAX
      :class:`MultiPatternDataProcessor` UNCHANGED (INV-4). This is NOT a
      :class:`WindowedTimeSeriesProcessor`; the base only consumes its
      ``prepare_datasets()`` output, which it already produces.
    - :meth:`_build_model` — bespoke functional wrapper (NaiveResidual +
      ForecastabilityGate around ``create_nbeats_model``) with a WarmupSchedule.
    - :meth:`_build_performance_callback` — the multi-pattern reconstruction
      callback.
    - :meth:`_make_callbacks` / :meth:`run_experiment` — the D-009 bare-prefix dir
      contract (``create_common_callbacks`` mints its own dir; the base
      ``_create_experiment_dir`` would yield a doubly-nested dir).
    - :meth:`_compute_post_hoc_metrics` — KEPT: the model is an anonymous functional
      wrapper (NOT a :class:`ForecastMixin`), so the base post-hoc block returns
      ``{}``; this override actually computes point-forecast metrics off the same
      D-004 ``test_ds``.
    - :meth:`_save_results` — KEPT (D-005): ``config.primary_loss`` may be a
      ``keras.losses.Loss``, which ``json_numpy_default`` raises on; the local
      ``default`` degrades it to ``str``.
    """

    def _build_processor(self) -> MultiPatternDataProcessor:
        # INV-4 / D-010: return the D-004 engine UNCHANGED. The base stores
        # self.generator / self.selected_patterns / self.pattern_to_category in
        # __init__ before calling this hook.
        return MultiPatternDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> PatternPerformanceCallback:
        return PatternPerformanceCallback(
            self.config, self.processor, viz_dir, "nbeats_forecasting"
        )

    def _make_callbacks(self, exp_dir: Optional[str] = None) -> List:
        """Override: N-BEATS-advanced uses ``model_name="N-BEATS-Forecasting"`` and
        the D-009 bare-prefix dir contract.
        """
        # DECISION plan_2026-06-11_84296249/D-011: pass a BARE prefix
        # (self._build_results_prefix()) to create_common_callbacks and ADOPT its
        # RETURNED results_dir as self.exp_dir -- the D-009 bare-prefix contract,
        # mirroring train_nbeats.py / adaptive_ema. Do NOT pass the pre-created full
        # exp_dir as results_dir_prefix and do NOT rely on the base
        # _create_experiment_dir: create_common_callbacks ALWAYS mints its own
        # results/{prefix}_{name}_{ts} dir and writes best_model.keras there, so the
        # base default would produce a SECOND doubly-nested dir that receives
        # CSVLogger/ModelCheckpoint while results.json/visualizations land in the
        # discarded first dir. The exp_dir param is ignored on purpose.
        # See decisions.md D-011.
        callbacks, results_dir = create_common_callbacks(
            model_name="N-BEATS-Forecasting",
            results_dir_prefix=self._build_results_prefix(),
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
        self.exp_dir = results_dir
        viz_dir = os.path.join(self.exp_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        callbacks.append(self._build_performance_callback(viz_dir))
        return callbacks

    def run_experiment(self) -> Dict[str, Any]:
        """Base skeleton with the D-011 dir resolution (mirrors train_nbeats.py).

        Overridden so ``self.exp_dir`` is resolved from
        ``create_common_callbacks``' returned dir (inside ``_train_model`` ->
        ``_make_callbacks``) instead of the base ``_create_experiment_dir``.
        Passing ``exp_dir=None`` means no first dir is pre-built; CSVLogger /
        ModelCheckpoint, results.json, and visualizations all land in the single
        returned dir.
        """
        logger.info(f"Starting {self.config.experiment_name} training experiment")

        data_pipeline = self.processor.prepare_datasets()
        self.model = self._build_model()
        self.model.build((None, self.config.backcast_length, self.config.input_dim))
        logger.info(f"Model params: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        # _train_model -> _make_callbacks sets self.exp_dir (D-011).
        training_results = self._train_model(data_pipeline, exp_dir=None)
        logger.info(f"Results: {self.exp_dir}")

        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config, 'experiment_dir': self.exp_dir,
            'training_results': training_results, 'results_dir': self.exp_dir
        }

    def _build_model(self) -> keras.Model:
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

    def _compute_post_hoc_metrics(self, data_pipeline: Dict[str, Any]) -> Dict[str, float]:
        """Point-only post-hoc forecast metrics for the trained model.

        KEPT as an override: this model is the anonymous functional wrapper around
        ``create_nbeats_model`` + forecasting layers, so it is NOT a
        :class:`ForecastMixin` and the base ``_compute_post_hoc_metrics`` returns
        ``{}``. This version computes a real block via the SHARED
        :func:`compute_post_hoc_forecast_metrics`, re-using the SAME uncached
        resampled ``test_ds`` (the D-004 processor untouched, INV-4). The point
        forecast is ``model.predict(x)`` (``[0]`` when the recon-on path returns a
        ``(forecast, residual)`` tuple).

        Kept ADDITIVE and NON-FATAL (I1): any failure logs a warning and yields an
        empty block so the run still completes and writes results.json.
        """
        try:
            x_batches: List[np.ndarray] = []
            y_batches: List[np.ndarray] = []
            for x_b, y_b in data_pipeline['test_ds'].take(data_pipeline['test_steps']):
                # recon-on path yields y_b as a (forecast, rec_target) tuple.
                if isinstance(y_b, (list, tuple)):
                    y_b = y_b[0]
                x_batches.append(np.asarray(x_b, dtype=np.float32))
                y_batches.append(np.asarray(y_b, dtype=np.float32))

            if not x_batches:
                logger.warning("post_hoc: no test batches collected; skipping block.")
                return {}

            backcast = np.concatenate(x_batches, axis=0)
            y_true = np.concatenate(y_batches, axis=0)

            point = self.model.predict(backcast, verbose=0)
            if isinstance(point, (list, tuple)):
                point = point[0]
            point = np.asarray(point, dtype=np.float32)

            block = compute_post_hoc_forecast_metrics(
                y_true=y_true, point=point, backcast=backcast,
                quantiles=None, quantile_levels=None,
            )
            logger.info(f"post_hoc_metrics: {block}")
            return {k: float(v) for k, v in block.items()}
        except Exception as exc:
            logger.warning(
                f"post_hoc_metrics computation failed ({exc!r}); "
                f"emitting empty block, run continues."
            )
            return {}

    def _save_results(self, results: Dict[str, Any], exp_dir: str,
                      extra_fields: Optional[Dict[str, Any]] = None) -> None:
        # DECISION plan_2026-06-11_84296249/D-012: use this LOCAL serializer, NOT
        # the base's train.common.json_numpy_default. config.__dict__ carries
        # `primary_loss: Union[str, keras.losses.Loss]`; json_numpy_default RAISES
        # TypeError on a Loss object, whereas this local default() degrades a Loss
        # to str(o). Mirrors train_nbeats.py's D-005. Do NOT swap this for
        # json_numpy_default. See decisions.md D-012.
        def default(o: Any) -> Any:
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            return str(o)

        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'post_hoc_metrics': results.get('post_hoc_metrics', {}),
            'config': self.config.__dict__,
        }
        if extra_fields:
            serializable.update(extra_fields)
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=default)


def build_parser() -> argparse.ArgumentParser:
    """Build the N-BEATS-advanced CLI on top of the shared TS argument parser.

    Starts from :func:`create_ts_argument_parser` (the shared TS args:
    ``--experiment_name``/``--seed``/``--n_samples``/``--noise_level``/
    ``--epochs``/``--batch_size``/``--steps_per_epoch``/``--learning_rate``/
    ``--gradient_clip_norm``/``--optimizer``/warmup/
    ``--max_patterns_per_category``/``--visualize_every_n_epochs``/
    ``--plot_top_k_patterns``/``--no-deep-analysis``/analysis/``--gpu`` etc.),
    restores N-BEATS-advanced's tuned defaults via ``set_defaults`` for the args
    whose default DIVERGES from the shared parser, then adds N-BEATS-advanced's
    architecture-specific flags.

    ``backcast_length``/``forecast_length`` (paper terminology) are KEPT — they
    are NOT renamed to ``input_length``/``prediction_length`` (the canonical names
    used by the streaming trio). Only the flags the trainer/config ACTUALLY read
    are surfaced: the shared parser already provides ``--seed``/``--n_samples``/
    ``--noise_level``/``--result_dir``/``--no-warmup``/``--warmup_steps``/``--gpu``,
    all of which become live now that the trainer subclasses the base.
    """
    parser = create_ts_argument_parser("N-BEATS Training with Scientific Forecasting Layers")

    # Restore N-BEATS-advanced's tuned defaults where they differ from the shared
    # parser. The original hand-rolled CLI used:
    #   experiment_name="nbeats_forecasting_layers" (shared default "timeseries"),
    #   optimizer="adamw" (shared default "adamw" -> no change needed),
    #   steps_per_epoch=1000 (== shared default), epochs=200 (== shared default),
    #   batch_size=128 (== shared default), learning_rate=1e-4 (== shared default),
    #   gradient_clip_norm=1.0 (== shared default), analysis_frequency=10 (==),
    #   analysis_start_epoch=1 (==). The config's own divergent defaults
    #   (warmup_steps=5000, max_patterns_per_category=100) are restored here so the
    #   default invocation matches the pre-migration behavior now that main() WIRES
    #   the shared flags into the config (no silent no-ops).
    parser.set_defaults(
        experiment_name="nbeats_forecasting_layers",
        optimizer="adamw",
        warmup_steps=5000,
        max_patterns_per_category=100,
        # Original hand-rolled main() built TimeSeriesGeneratorConfig(n_samples=5000);
        # restore that (shared parser default is 10000). It omitted noise -> the
        # build_generator_config / shared default of 0.1 reproduces the prior behavior.
        n_samples=5000,
    )

    # N-BEATS-advanced architecture-specific arguments (KEEP backcast/forecast).
    parser.add_argument("--backcast_length", type=int, default=168)
    parser.add_argument("--forecast_length", type=int, default=24)
    parser.add_argument("--stack_types", nargs='+', default=["trend", "seasonality", "generic"])
    parser.add_argument("--hidden_layer_units", type=int, default=256)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.5)
    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false")
    parser.set_defaults(normalize_per_instance=True)
    parser.add_argument("--no-naive-residual", dest="use_naive_residual", action="store_false")
    parser.set_defaults(use_naive_residual=True)
    parser.add_argument("--no-forecastability-gate", dest="use_forecastability_gate", action="store_false")
    parser.set_defaults(use_forecastability_gate=True)
    parser.add_argument("--gate_hidden_units", type=int, default=16)
    parser.add_argument("--gate_activation", type=str, default="relu")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    """Configure and run the N-BEATS-advanced training experiment."""
    args = parse_args()
    set_seeds(args.seed)
    setup_gpu(args.gpu)

    config = NBeatsTrainingConfig(
        experiment_name=args.experiment_name,
        seed=args.seed,
        result_dir=args.result_dir,
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
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch,
    )

    generator_config = build_generator_config(args)

    try:
        trainer = NBeatsTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
        logger.info(
            f"Forecasting layers: NaiveResidual={config.use_naive_residual}, "
            f"ForecastabilityGate={config.use_forecastability_gate}"
        )
        keras.backend.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
