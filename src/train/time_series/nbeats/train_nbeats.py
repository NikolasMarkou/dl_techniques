"""
Training pipeline for the N-BEATS forecasting architecture.

N-BEATS uses a doubly residual topology of stacks and blocks to decompose time
series into trend, seasonality, and generic components. Supports both interpretable
and generic stack configurations with optional backcast reconstruction loss.

References:
    Oreshkin et al. (2019) - N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting (ICLR)
"""

import os
import sys
import json
import math
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
    setup_gpu,
    set_seeds,
    create_learning_rate_schedule,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
    BaseTimeSeriesTrainer,
    create_ts_argument_parser,
    _prepare_viz_data_from_processor,
)
from dl_techniques.utils.logger import logger
from dl_techniques.losses.mase_loss import MASELoss
from dl_techniques.models.time_series.nbeats import create_nbeats_model
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    NormalizationMethod,
)

plt.style.use('default')
sns.set_palette("husl")


@dataclass
class NBeatsTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for N-BEATS training.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags)
    from :class:`BaseTimeSeriesTrainingConfig` and adds the N-BEATS architecture
    fields below. A handful of inherited defaults are re-declared because the
    N-BEATS originals differ from the base: ``steps_per_epoch`` (1000 vs 500),
    ``max_patterns_per_category`` (100 vs 10), and ``optimizer``
    ('adam' vs 'adamw').

    Note on ``warmup_steps``: this dataclass re-declares it as 5000, but that
    value only takes effect for a *programmatic* config built without an explicit
    ``warmup_steps``. The EFFECTIVE default under the normal CLI path is 1000:
    ``main()`` always passes ``args.warmup_steps`` and the shared parser's default
    is 1000 (matching the original N-BEATS CLI). So at default invocation the
    runtime value is 1000, not 5000. ``category_weights`` and
    ``normalize_per_instance`` match the base defaults and are dropped.
    """

    experiment_name: str = "nbeats"

    # Re-declared: N-BEATS originals differ from the base defaults.
    steps_per_epoch: int = 1000      # base default: 500
    warmup_steps: int = 5000         # programmatic only; CLI path -> 1000 (see docstring)
    max_patterns_per_category: int = 100  # base default: 10
    optimizer: str = 'adam'          # base default: 'adamw'

    # N-BEATS architecture
    backcast_length: int = 168
    forecast_length: int = 24
    input_dim: int = 1
    stack_types: List[str] = field(default_factory=lambda: ["trend", "seasonality", "generic"])
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 128
    use_normalization: bool = True
    use_bias: bool = True
    activation: str = "gelu"
    primary_loss: Union[str, keras.losses.Loss] = "mase_loss"
    mase_seasonal_periods: int = 1

    # Regularization
    kernel_regularizer_l2: float = 1e-5
    reconstruction_loss_weight: float = 0.5
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")


class MultiPatternDataProcessor(WindowedTimeSeriesProcessor):
    """N-BEATS data processor: subclass of :class:`WindowedTimeSeriesProcessor`.

    Overrides BOTH hooks to emit the optional backcast-reconstruction target.
    With ``reconstruction_loss_weight > 0`` the target is a 2-tuple
    ``(forecast, backcast.flatten())``; otherwise it is the single forecast
    tensor (matching the base default). STANDARD per-instance normalization.
    """

    def __init__(self, config: NBeatsTrainingConfig, generator: TimeSeriesGenerator,
                 selected_patterns: List[str], pattern_to_category: Dict[str, str],
                 num_features: int = 1):
        super().__init__(
            config,
            generator,
            selected_patterns,
            pattern_to_category=pattern_to_category,
            context_len=config.backcast_length,
            horizon_len=config.forecast_length,
            num_features=num_features,
            normalize=True,
            normalize_method=NormalizationMethod.STANDARD,
        )

    def _make_sample(self, window: np.ndarray, pattern_name: str) -> Tuple[Any, Any]:
        ctx = self.context_len
        backcast = window[:ctx].reshape(-1, self.num_features).astype(np.float32)
        forecast = window[ctx:].reshape(-1, self.num_features).astype(np.float32)
        if self.config.reconstruction_loss_weight > 0.0:
            return backcast, (forecast, backcast.flatten().astype(np.float32))
        return backcast, forecast

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        x_spec = tf.TensorSpec(shape=(self.context_len, self.num_features), dtype=tf.float32)
        y_spec = tf.TensorSpec(shape=(self.horizon_len, self.num_features), dtype=tf.float32)
        if self.config.reconstruction_loss_weight > 0.0:
            rec_spec = tf.TensorSpec(shape=(self.context_len * self.num_features,), dtype=tf.float32)
            return (x_spec, (y_spec, rec_spec))
        return (x_spec, y_spec)


class PatternPerformanceCallback(TimeSeriesPerformanceCallback):
    """Monitors and visualizes N-BEATS performance on a fixed test set.

    Thin subclass of :class:`TimeSeriesPerformanceCallback`. The base owns the
    scaffolding (``__init__`` + makedirs, ``loss``/``val_loss`` accumulation, the
    ``visualize_every_n_epochs`` gate, learning-curve delegation). N-BEATS keeps
    only the three model-specific pieces: viz-data prep from its processor (the
    backcast ``x`` / forecast ``y`` collection — same loop the original used),
    the ``forecast_mae``/``val_forecast_mae`` extra-history tracking (N-BEATS
    tracks NO learning rate, so no :meth:`_track_lr` call), and the
    backcast/forecast prediction-plot body that unpacks ``predictions_tuple[0]``
    from a ``model(x, training=False)`` call.
    """

    def __init__(self, config: NBeatsTrainingConfig, data_processor: MultiPatternDataProcessor,
                 save_dir: str, model_name: str = "model"):
        # data_processor must be set BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() which reads self.data_processor.
        self.data_processor = data_processor
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return _prepare_viz_data_from_processor(self.data_processor, self.config.plot_top_k_patterns)

    def _extend_history(self, logs: dict) -> None:
        self.training_history.setdefault('forecast_mae', []).append(logs.get('forecast_mae', 0.0))
        self.training_history.setdefault('val_forecast_mae', []).append(logs.get('val_forecast_mae', 0.0))

    def _plot_predictions(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data
        predictions_tuple = self.model(test_x, training=False)

        if isinstance(predictions_tuple, (tuple, list)):
            predictions = predictions_tuple[0]
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
        else:
            predictions = predictions_tuple.numpy()

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols, n_rows = 3, math.ceil(num_plots / 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            bk_steps = np.arange(-self.config.backcast_length, 0)
            fc_steps = np.arange(0, self.config.forecast_length)
            ax.plot(bk_steps, test_x[i].flatten(), label='Backcast', color='blue', alpha=0.6)
            ax.plot(fc_steps, test_y[i].flatten(), label='True Future', color='green')
            ax.plot(fc_steps, predictions[i].flatten(), label='Pred Future', color='red', linestyle='--')
            ax.set_title(f'Sample {i+1}')
            if i == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'N-BEATS Predictions (Epoch {epoch + 1})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


class NBeatsTrainer(BaseTimeSeriesTrainer):
    """Orchestrates N-BEATS training, evaluation, and reporting.

    Thin subclass of :class:`BaseTimeSeriesTrainer`. The base owns the skeleton
    (``__init__`` generator/pattern setup, ``_select_patterns``,
    ``_create_experiment_dir``, ``_make_callbacks``, ``_train_model``,
    ``run_experiment``). N-BEATS overrides only the genuine divergences: the
    processor, the model build (+ compile + build-from-input-shape), the
    performance callback, the ``model_name="N-BEATS"`` callback set, and
    :meth:`_save_results` (D-005: the ``primary_loss: Union[str, Loss]`` field
    needs a str-fallback serializer, NOT the base's ``json_numpy_default``).
    N-BEATS has NO ONNX export, so it uses the base ``run_experiment`` directly.
    """

    def _build_processor(self) -> MultiPatternDataProcessor:
        return MultiPatternDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> PatternPerformanceCallback:
        return PatternPerformanceCallback(self.config, self.processor, viz_dir, "nbeats")

    def _make_callbacks(self, exp_dir: Optional[str] = None) -> List:
        """Override: N-BEATS uses ``model_name="N-BEATS"`` (base default is the
        experiment name). ``patience=25`` matches the base default; everything
        else (monitor, terminate-on-nan, analyzer config) matches the base body.
        """
        from dl_techniques.analyzer import AnalysisConfig
        from train.common import create_callbacks as create_common_callbacks

        # DECISION plan_2026-06-10_39646d39/D-002
        # Pass a BARE prefix (self._build_results_prefix()) to
        # create_common_callbacks and adopt its RETURNED results_dir as
        # self.exp_dir -- the D-009 bare-prefix contract, mirroring prism/tirex.
        # Do NOT pass the pre-created full exp_dir path as results_dir_prefix and
        # do NOT rely on the base _create_experiment_dir here: passing the full
        # path as prefix built a SEPARATE doubly-nested
        # results/{full}_N-BEATS_{ts2}/ that received CSVLogger/ModelCheckpoint
        # while results.json/visualizations landed in the discarded first dir.
        # The exp_dir param is ignored on purpose. See decisions.md D-009.
        callbacks, results_dir = create_common_callbacks(
            model_name="N-BEATS",
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

    def _build_model(self) -> keras.Model:
        """Create and compile an N-BEATS model (+ build from input shape)."""
        kernel_regularizer = None
        if self.config.kernel_regularizer_l2 > 0:
            kernel_regularizer = keras.regularizers.L2(self.config.kernel_regularizer_l2)

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

        # Learning rate schedule
        if self.config.use_warmup:
            lr_schedule = create_learning_rate_schedule(
                self.config.learning_rate, 'cosine',
                total_epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
            )
        else:
            lr_schedule = self.config.learning_rate

        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = lr_schedule
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        # Compile with appropriate loss configuration
        if self.config.reconstruction_loss_weight > 0.0:
            model = base_model
            forecast_loss = (MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
                            if self.config.primary_loss == 'mase_loss'
                            else keras.losses.get(self.config.primary_loss))
            losses = [forecast_loss, keras.losses.MeanAbsoluteError(name="residual_loss", reduction="mean")]
            loss_weights = [1.0, self.config.reconstruction_loss_weight]
            metrics = [
                [keras.metrics.MeanAbsoluteError(name="forecast_mae")],
                [keras.metrics.MeanAbsoluteError(name="residual_mae")]
            ]
        else:
            inputs = keras.Input(shape=(self.config.backcast_length, self.config.input_dim))
            raw_output = base_model(inputs)
            forecast = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
            model = keras.Model(inputs=inputs, outputs=forecast, name="nbeats_forecast_only")
            losses = (MASELoss(seasonal_periods=self.config.mase_seasonal_periods)
                      if self.config.primary_loss == 'mase_loss'
                      else keras.losses.get(self.config.primary_loss))
            loss_weights = None
            metrics = [keras.metrics.MeanAbsoluteError(name="forecast_mae")]

        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        model.build((None, self.config.backcast_length, self.config.input_dim))
        return model

    def _save_results(self, results: Dict[str, Any], exp_dir: str,
                      extra_fields: Optional[Dict[str, Any]] = None) -> None:
        # DECISION plan_2026-06-09_a3c7304c/D-005: use this LOCAL serializer, NOT
        # the base's train.common.json_numpy_default. config.__dict__ carries
        # `primary_loss: Union[str, keras.losses.Loss]`; json_numpy_default RAISES
        # TypeError on a Loss object (prior-plan D-003 forbids restoring its
        # str-coercion), whereas this local default() degrades a Loss to str(o).
        # Do NOT swap this for json_numpy_default. See decisions.md D-005.
        def default(o: Any) -> Any:
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            return str(o)

        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__,
        }
        if extra_fields:
            serializable.update(extra_fields)
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=default)


def build_parser() -> argparse.ArgumentParser:
    """Build the N-BEATS CLI parser on top of the shared TS argument parser.

    Starts from :func:`create_ts_argument_parser` (the shared TS args:
    ``--epochs``/``--batch_size``/``--steps_per_epoch``/``--learning_rate``/
    warmup/analysis/``--gpu`` etc.), restores N-BEATS's own defaults for the args
    whose default differs from the shared parser via ``set_defaults`` (
    ``experiment_name`` "nbeats"; the original N-BEATS CLI's ``optimizer`` was
    "adamw" and ``warmup_steps`` 1000 — both already the shared-parser defaults,
    so only ``experiment_name`` needs restoring), then adds N-BEATS's
    architecture-specific flags (backcast/forecast lengths, stack types,
    hidden units, reconstruction-loss weight, per-instance normalization toggle).
    """
    parser = create_ts_argument_parser("N-BEATS Training")

    # Restore N-BEATS's per-arg defaults where they differ from the shared parser.
    # The original N-BEATS CLI used experiment_name="nbeats"; its epochs(200)/
    # batch_size(128)/steps_per_epoch(1000)/learning_rate(1e-4)/optimizer("adamw")/
    # gradient_clip_norm(1.0)/warmup_steps(1000)/warmup_start_lr(1e-6)/
    # analysis_frequency(10)/analysis_start_epoch(1) already match the shared parser.
    # max_patterns_per_category=100 restores the N-BEATS config default (shared
    # parser default is 10) so that, now that main() WIRES the flag into the
    # config (review #5: no silent no-op), default invocation still selects 100.
    parser.set_defaults(experiment_name="nbeats", max_patterns_per_category=100)

    # N-BEATS architecture-specific arguments.
    parser.add_argument("--backcast_length", type=int, default=168)
    parser.add_argument("--forecast_length", type=int, default=24)
    parser.add_argument("--stack_types", nargs='+', default=["trend", "seasonality", "generic"])
    parser.add_argument("--hidden_layer_units", type=int, default=256)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.5)
    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false")
    parser.set_defaults(normalize_per_instance=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(42)
    setup_gpu(args.gpu)

    config = NBeatsTrainingConfig(
        experiment_name=args.experiment_name,
        backcast_length=args.backcast_length,
        forecast_length=args.forecast_length,
        stack_types=args.stack_types,
        hidden_layer_units=args.hidden_layer_units,
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
        # Wire the shared-parser pattern/viz flags so they are not silent no-ops
        # (review #5). max_patterns_per_category default is restored to 100 in
        # build_parser; visualize_every_n_epochs/plot_top_k_patterns parser
        # defaults (5/12) already equal the N-BEATS config defaults.
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=10000, random_seed=42, default_noise_level=0.1
    )

    # Preserve nbeats's original hard-exit teardown (os._exit(0)): a deliberate
    # force-exit that skips Python/atexit cleanup to avoid TF prefetch-thread
    # hangs on shutdown. Do NOT normalize to sys.exit/finally (per-script choice,
    # not consolidatable duplication). See plan_2026-06-09_a3c7304c/decisions.md D-008.
    try:
        trainer = NBeatsTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
        keras.backend.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)

    os._exit(0)


if __name__ == "__main__":
    main()
