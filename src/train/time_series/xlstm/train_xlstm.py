"""
Training pipeline for the :class:`xLSTMForecaster` probabilistic forecaster.

xLSTMForecaster is the continuous-input forecasting sibling of the language-model
``xLSTM``: it reuses the mLSTM/sLSTM residual block stack with a continuous
input projection, a global mean-pool, and a quantile (or point) forecasting head,
plugging into the unified ``Forecast`` contract via ``ForecastMixin``.

This Pattern-2 trainer mirrors the TiRex trainer (the exact template) but with one
deliberate omission: it does NOT override ``_compute_post_hoc_metrics``. The model
emits clean ``[B, context, F]`` tensor input, so the base
``BaseTimeSeriesTrainer._compute_post_hoc_metrics`` (ForecastMixin-gated) populates
the post-hoc forecast metric block unchanged — the contract-native payoff.

.. warning::
    **EXPERIMENTAL — known training instability (NaN) as of plan_2026-06-10_c6197fb1.**
    The ``xLSTMForecaster`` MODEL is correct and fully tested (eager forward /
    ``_forecast`` / single-step gradient on real batches are all finite; 11 model
    tests pass; ``get_config`` round-trips). However, ``model.fit()`` through this
    trainer produces ``NaN`` loss from ~step 1 under every configuration tried
    (``jit_compile`` True/False, ``run_eagerly=True``, both normalization paths).
    A single manual eager train step on a real batch is finite, so the NaN is a
    fit-loop/data interaction not yet pinned — prime suspect: a near-constant
    synthetic window hitting the base ``_safe_normalize`` STANDARD per-instance
    divide-by-~0-std (no epsilon). DEFERRED to a follow-up. Do NOT treat this
    trainer as production-ready until the NaN is root-caused and fixed. The
    structural pieces (config/processor/callback/trainer, NO post_hoc override —
    the contract-native payoff) are otherwise complete and correct.

References:
    Beck, M., et al. (2024) - xLSTM: Extended Long Short-Term Memory (arXiv:2405.04517)
    Kim, T., et al. (2022) - Reversible Instance Normalization (ICLR)
    Koenker & Bassett (1978) - Regression quantiles
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    set_seeds,
    create_learning_rate_schedule,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
    BaseTimeSeriesTrainer,
    create_ts_argument_parser,
    _prepare_viz_data_from_processor,
)
from train.common.timeseries import _plot_ts_forecast
from dl_techniques.utils.logger import logger
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.models.time_series.xlstm.forecaster import (
    xLSTMForecaster,
    create_xlstm_forecaster,
)
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
)

plt.style.use('default')
sns.set_palette("husl")


@dataclass
class XLSTMForecasterTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for xLSTMForecaster training on multiple patterns.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags, and
    the promoted probabilistic fields ``use_quantile_head`` / ``quantile_levels`` /
    ``enforce_monotonicity``) from :class:`BaseTimeSeriesTrainingConfig` and adds
    the xLSTM architecture fields below.

    The point/quantile head is driven by the PROMOTED base ``use_quantile_head``
    flag (NOT a local ``output_mode``).
    """

    experiment_name: str = "xlstm"

    # xLSTM architecture
    input_length: int = 168
    prediction_length: int = 24
    embed_dim: int = 128
    num_layers: int = 4
    mlstm_ratio: float = 0.5
    mlstm_num_heads: int = 4
    dropout_rate: float = 0.1

    # ONNX export (xLSTM recurrence may not be tf2onnx-traceable; off by default)
    export_onnx: bool = False
    onnx_opset_version: int = 17

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant + promoted-field defaults
        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")
        if self.embed_dim % self.mlstm_num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"mlstm_num_heads ({self.mlstm_num_heads})"
            )
        if self.use_quantile_head and 0.5 not in self.quantile_levels:
            logger.warning("Recommended to include 0.5 (median) in quantile_levels.")


class XLSTMForecasterDataProcessor(WindowedTimeSeriesProcessor):
    """xLSTMForecaster data processor: subclass of :class:`WindowedTimeSeriesProcessor`.

    Context is always ``[context, F]``. The TARGET shape is mode-dependent so it
    matches the model output + loss:
    - quantile mode: target is FLATTENED to ``[H]`` (like tirex) — `QuantileLoss`
      expects ``y_true [B,H]`` and internally expands to broadcast against the
      ``y_pred [B,H,Q]`` quantile axis; a ``[B,H,1]`` target would rank-mismatch.
    - point mode: target stays ``[H, F]`` to match the point head's ``[B,H,F]``
      MeanAbsoluteError target.
    Per-instance STANDARD normalization is ON (base default).
    """

    def __init__(
            self,
            config: XLSTMForecasterTrainingConfig,
            generator: TimeSeriesGenerator,
            selected_patterns: List[str],
            pattern_to_category: Dict[str, str],
            num_features: int = 1,
    ):
        super().__init__(
            config,
            generator,
            selected_patterns,
            pattern_to_category=pattern_to_category,
            context_len=config.input_length,
            horizon_len=config.prediction_length,
            num_features=num_features,
            # normalize=True: the processor owns normalization (mdn/prism pattern).
            # This is the NaN-safe path: base normalization runs `_fill_nans` so
            # raw synthetic NaN windows never reach the model. The model's own RevIN
            # is DISABLED (use_normalization=False in _build_model) to avoid double
            # normalization; targets are already standardized, so QuantileLoss
            # (normalize=True) divides by a healthy mean(|target|) ~= 0.8, not ~0.
            normalize=True,
        )

    def _make_sample(self, window: np.ndarray, pattern_name: str) -> Tuple[Any, Any]:
        inputs = window[:self.context_len].reshape(-1, self.num_features).astype(np.float32)
        targets = window[self.context_len:].reshape(-1, self.num_features).astype(np.float32)
        if self.config.use_quantile_head:
            # QuantileLoss expects y_true [H]; flatten the trailing feature axis.
            targets = targets.reshape(self.horizon_len).astype(np.float32)
        return inputs, targets

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        ctx_spec = tf.TensorSpec(shape=(self.context_len, self.num_features), dtype=tf.float32)
        if self.config.use_quantile_head:
            tgt_spec = tf.TensorSpec(shape=(self.horizon_len,), dtype=tf.float32)
        else:
            tgt_spec = tf.TensorSpec(shape=(self.horizon_len, self.num_features), dtype=tf.float32)
        return (ctx_spec, tgt_spec)


class XLSTMForecasterPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks and visualizes xLSTMForecaster forecast performance.

    Thin subclass of :class:`TimeSeriesPerformanceCallback`. The base owns the
    scaffolding (``__init__`` + makedirs, ``loss``/``val_loss`` accumulation, the
    ``visualize_every_n_epochs`` gate, learning-curve delegation). This callback
    keeps only the model-specific pieces: viz-data prep from its processor, the
    extra-history tracking (``mae_median`` in quantile mode + ``lr``), and the
    prediction-plot body routed through :func:`_plot_ts_forecast`. In quantile mode
    a median line + low/high band are drawn; in point mode just the point line.
    """

    def __init__(self, config: XLSTMForecasterTrainingConfig,
                 processor: XLSTMForecasterDataProcessor,
                 save_dir: str, model_name: str = "xlstm"):
        # processor must be set BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() which reads self.processor.
        self.processor = processor
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return _prepare_viz_data_from_processor(self.processor, self.config.plot_top_k_patterns)

    def _extend_history(self, logs: dict) -> None:
        if self.config.use_quantile_head:
            self.training_history.setdefault('mae_median', []).append(logs.get('mae_of_median', 0))
            self.training_history.setdefault('val_mae_median', []).append(logs.get('val_mae_of_median', 0))
        self._track_lr(logs)

    def _plot_predictions(self, epoch: int) -> None:
        test_x, test_y = self.viz_test_data
        if len(test_x) == 0:
            return

        preds = self.model(test_x, training=False)
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols, n_rows = 3, math.ceil(num_plots / 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        if self.config.use_quantile_head:
            quantiles = self.config.quantile_levels
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            low_idx = 0
            high_idx = len(quantiles) - 1 if len(quantiles) >= 3 else -1
            for i in range(num_plots):
                _plot_ts_forecast(
                    axes[i],
                    test_x[i].flatten(),
                    test_y[i].flatten(),
                    preds[i, :, median_idx],
                    lower=preds[i, :, low_idx],
                    upper=preds[i, :, high_idx],
                    title=f'Sample {i + 1}',
                    context_label='Input',
                    target_label='True',
                    point_label='Median',
                    band_label=f'{quantiles[low_idx]}-{quantiles[high_idx]} Q',
                )
            suptitle = f'Probabilistic Forecasts (Epoch {epoch + 1})'
        else:
            for i in range(num_plots):
                _plot_ts_forecast(
                    axes[i],
                    test_x[i].flatten(),
                    test_y[i].flatten(),
                    preds[i].reshape(-1),
                    title=f'Sample {i + 1}',
                    context_label='Input',
                    target_label='True',
                    point_label='Forecast',
                )
            suptitle = f'Point Forecasts (Epoch {epoch + 1})'

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


class XLSTMForecasterTrainer(BaseTimeSeriesTrainer):
    """Orchestrates xLSTMForecaster training (quantile or point).

    Thin subclass of :class:`BaseTimeSeriesTrainer`. The base owns the skeleton
    (generator/pattern setup, ``_select_patterns``, ``_train_model``,
    ``_save_results``, ``_export_to_onnx``, AND ``_compute_post_hoc_metrics``).
    This trainer overrides only the genuine divergences: the processor, the model
    build (+compile), the performance callback, the head-mode results prefix, the
    ``patience=30`` / ``model_name="xLSTMForecaster"`` callback set, and a minimal
    ``run_experiment`` to fold ``onnx_path`` into ``results.json``.

    NOTE: ``_compute_post_hoc_metrics`` is DELIBERATELY NOT overridden. The model
    feeds the base a clean ``[B, context, F]`` tensor, so the base ForecastMixin
    path populates the metric block unchanged (the contract-native payoff).
    """

    def _build_processor(self) -> XLSTMForecasterDataProcessor:
        return XLSTMForecasterDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> XLSTMForecasterPerformanceCallback:
        return XLSTMForecasterPerformanceCallback(self.config, self.processor, viz_dir, "xlstm")

    def _build_results_prefix(self) -> str:
        mode = 'quantile' if self.config.use_quantile_head else 'point'
        return f"{self.config.experiment_name}_{mode}"

    def _make_callbacks(self, exp_dir: Optional[str] = None) -> List:
        """Override: xLSTMForecaster uses ``patience=30`` /
        ``model_name="xLSTMForecaster"`` AND owns its experiment dir (D-009)."""
        # DECISION plan_2026-06-09_a3c7304c/D-009
        # Pass a BARE prefix (self._build_results_prefix()) to
        # create_common_callbacks and adopt its RETURNED results_dir as
        # self.exp_dir -- matching the tirex flow. Do NOT pass the pre-created
        # full exp_dir as results_dir_prefix and do NOT use the base
        # _create_experiment_dir here: that built a SEPARATE doubly-nested
        # results dir while the ONNX read path used the first dir -> checkpoint
        # not found -> silent None when export_onnx=True. The exp_dir param is
        # ignored on purpose. See decisions.md D-009.
        callbacks, results_dir = create_common_callbacks(
            model_name="xLSTMForecaster",
            results_dir_prefix=self._build_results_prefix(),
            monitor="val_loss",
            patience=30,
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

    def _build_model(self) -> xLSTMForecaster:
        """Create and compile the xLSTMForecaster."""
        logger.info(
            f"Creating xLSTMForecaster "
            f"({'quantile' if self.config.use_quantile_head else 'point'} head)"
        )

        model = create_xlstm_forecaster(
            input_length=self.config.input_length,
            prediction_length=self.config.prediction_length,
            num_features=1,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            mlstm_ratio=self.config.mlstm_ratio,
            mlstm_num_heads=self.config.mlstm_num_heads,
            use_quantile_head=self.config.use_quantile_head,
            quantile_levels=self.config.quantile_levels,
            enforce_monotonicity=self.config.enforce_monotonicity,
            dropout_rate=self.config.dropout_rate,
            # Processor already normalizes (normalize=True, NaN-safe). Disable the
            # model's RevIN to avoid double normalization (which collapses targets
            # and NaNs QuantileLoss). Single normalization owned by the processor.
            use_normalization=False,
        )

        if self.config.use_warmup:
            lr_schedule = create_learning_rate_schedule(
                self.config.learning_rate, 'cosine',
                total_epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
                warmup_steps=self.config.warmup_steps,
                warmup_start_lr=self.config.warmup_start_lr,
            )
            logger.info("Using Warmup + CosineDecay schedule")
        else:
            lr_schedule = self.config.learning_rate

        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = lr_schedule
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        if self.config.use_quantile_head:
            loss = QuantileLoss(quantiles=self.config.quantile_levels, normalize=True)
            metrics = []
            if 0.5 in self.config.quantile_levels:
                median_idx = self.config.quantile_levels.index(0.5)
                def mae_of_median(y_true, y_pred):
                    return keras.metrics.mean_absolute_error(y_true, y_pred[:, :, median_idx])
                mae_of_median.__name__ = 'mae_of_median'
                metrics.append(mae_of_median)
        else:
            loss = keras.losses.MeanAbsoluteError()
            metrics = [keras.metrics.MeanSquaredError(name='mse')]

        # jit_compile=False (STOP-IF-1 fired): the gated mLSTM/sLSTM recurrence
        # produces NaN under XLA on real data while EAGER forward+loss+gradient are
        # all finite (diagnosed: eager train step finite, jit=True NaN at step 1).
        # XLA-GPU has no faithful kernel for these ops here -- same class of issue
        # as the vMF keras.random.beta XLA incompatibility (SYSTEM.md). Disabled.
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)

        # Build the subclassed model with a dummy forward pass so downstream
        # `count_params()` / `summary()` (run_experiment) work before fit.
        dummy = keras.ops.zeros((1, self.config.input_length, 1), dtype="float32")
        _ = model(dummy, training=False)
        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Base skeleton + optional ONNX export folded into ``results.json``.

        Overridden (not the bare base ``run_experiment``) because xLSTMForecaster
        resolves ``self.exp_dir`` from ``create_common_callbacks``' returned dir
        (D-009) rather than the base ``_create_experiment_dir``, and may carry an
        ``onnx_path`` key. ``self.exp_dir`` is set INSIDE ``_train_model`` (via its
        ``_make_callbacks`` call); the ONNX read + ``_save_results`` below run after
        it so checkpoint, viz, results.json, and the ONNX read path share one dir.
        """
        logger.info("Starting xLSTMForecaster training experiment")

        data_pipeline = self.processor.prepare_datasets()
        self.model = self._build_model()
        logger.info(f"Model params: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        # _train_model -> _make_callbacks sets self.exp_dir (D-009).
        training_results = self._train_model(data_pipeline, exp_dir=None)
        logger.info(f"Results: {self.exp_dir}")

        best_model_path = os.path.join(self.exp_dir, 'best_model.keras')
        onnx_path = self._export_to_onnx(best_model_path, self.exp_dir)

        if self.config.save_results:
            self._save_results(training_results, self.exp_dir,
                               extra_fields={'onnx_path': onnx_path})

        return {
            'config': self.config, 'experiment_dir': self.exp_dir,
            'training_results': training_results, 'results_dir': self.exp_dir
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the xLSTMForecaster CLI parser on top of the shared TS argument parser.

    Starts from :func:`create_ts_argument_parser` (the shared TS args), restores
    xLSTMForecaster's defaults via ``set_defaults`` (experiment_name "xlstm"),
    then adds the xLSTM architecture-specific flags and the head/ONNX toggles.
    """
    parser = create_ts_argument_parser("xLSTMForecaster Training")

    parser.set_defaults(
        experiment_name="xlstm",
        max_patterns_per_category=100,
    )

    # xLSTM architecture-specific arguments.
    parser.add_argument("--input_length", type=int, default=168)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mlstm_ratio", type=float, default=0.5)
    parser.add_argument("--mlstm_num_heads", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Head mode (point by default; --use_quantile_head switches to quantile).
    parser.add_argument("--use_quantile_head", action="store_true",
                        help="Use the quantile head (probabilistic mode)")
    parser.set_defaults(use_quantile_head=False)

    # ONNX export.
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false")
    parser.set_defaults(export_onnx=False)
    parser.add_argument("--onnx_opset_version", type=int, default=17)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(42)
    setup_gpu(args.gpu)

    config = XLSTMForecasterTrainingConfig(
        experiment_name=args.experiment_name,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        mlstm_ratio=args.mlstm_ratio,
        mlstm_num_heads=args.mlstm_num_heads,
        dropout_rate=args.dropout_rate,
        use_quantile_head=args.use_quantile_head,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch,
        export_onnx=args.export_onnx,
        onnx_opset_version=args.onnx_opset_version,
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=10000, random_seed=42, default_noise_level=0.1
    )

    try:
        trainer = XLSTMForecasterTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        keras.backend.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
