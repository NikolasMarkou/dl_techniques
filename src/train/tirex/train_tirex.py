"""
Training pipeline for the TiRex probabilistic forecasting framework.

TiRex (Time-series Representation EXchange) uses a patch-based Transformer
architecture with two decoding strategies:
- TiRexCore: Global pooling + MLP projection
- TiRexExtended: Learnable query tokens with cross-attention

References:
    Nie et al. (2023) - A Time Series is Worth 64 Words (ICLR)
    Das et al. (2023) - Long-term Forecasting with TiDE
    Koenker & Bassett (1978) - Regression quantiles
"""

import os
import sys
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
from dl_techniques.utils.logger import logger
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.models.tirex.model import create_tirex_by_variant, TiRexCore
from dl_techniques.models.tirex.model_extended import create_tirex_extended, TiRexExtended
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
)

plt.style.use('default')
sns.set_palette("husl")


@dataclass
class TiRexTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for TiRex training on multiple patterns.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags)
    from :class:`BaseTimeSeriesTrainingConfig` and adds the TiRex architecture
    fields below. All inherited defaults (``batch_size=128``, ``steps_per_epoch=500``,
    ``warmup_steps=1000``, ``category_weights``) match the base, so none are
    re-declared.
    """

    experiment_name: str = "tirex_probabilistic"
    model_type: str = "core"  # 'core' or 'extended'

    # TiRex architecture
    input_length: int = 168
    prediction_length: int = 24
    variant: str = "small"  # 'tiny', 'small', 'medium', 'large'
    patch_size: int = 12
    dropout_rate: float = 0.1
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )

    # TiRex-specific pattern selection
    min_data_length: int = 2000

    # ONNX export
    export_onnx: bool = False
    onnx_opset_version: int = 17

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.model_type not in ['core', 'extended']:
            raise ValueError(f"model_type must be 'core' or 'extended', got '{self.model_type}'")
        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")
        if 0.5 not in self.quantile_levels:
            logger.warning("Recommended to include 0.5 (median) in quantile_levels.")


class TiRexDataProcessor(WindowedTimeSeriesProcessor):
    """TiRex data processor: thin subclass of :class:`WindowedTimeSeriesProcessor`.

    Two TiRex-specific differences from the trio default:
    1. ``normalize=False`` — the model handles normalization, so the base only
       clips + NaN-fills + casts to float32 (matching the original
       ``_safe_normalize``: no per-instance normalize, no clip(10)).
    2. the target is FLATTENED to ``(prediction_length,)`` rather than reshaped
       to ``(horizon_len, num_features)``, so ``_make_sample`` and
       ``output_signature`` are overridden accordingly.
    """

    def __init__(
            self,
            config: TiRexTrainingConfig,
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
            normalize=False,
        )

    def _make_sample(self, window: np.ndarray, pattern_name: str) -> Tuple[np.ndarray, np.ndarray]:
        ctx = self.context_len
        x = window[:ctx].reshape(-1, self.num_features).astype(np.float32)
        y = window[ctx:].flatten().astype(np.float32)
        return x, y

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        import tensorflow as tf
        return (
            tf.TensorSpec(shape=(self.context_len, self.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(self.horizon_len,), dtype=tf.float32),
        )


class TiRexPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks and visualizes TiRex quantile forecast performance.

    Thin subclass of :class:`TimeSeriesPerformanceCallback`. The base owns the
    scaffolding (``__init__`` + makedirs, ``loss``/``val_loss`` accumulation, the
    ``visualize_every_n_epochs`` gate, learning-curve delegation). TiRex keeps
    only the three model-specific pieces: viz-data prep from its processor, the
    ``mae_median``/``val_mae_median``/``lr`` extra-history tracking, and the
    3D-quantile prediction-plot body (``preds[i, :, q]``) rendered from a
    ``model(x, training=False)`` inference call.
    """

    def __init__(self, config: TiRexTrainingConfig, processor: TiRexDataProcessor,
                 save_dir: str, model_name: str = "tirex"):
        # processor must be set BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() which reads self.processor.
        self.processor = processor
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return _prepare_viz_data_from_processor(self.processor, self.config.plot_top_k_patterns)

    def _extend_history(self, logs: dict) -> None:
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

        quantiles = self.config.quantile_levels
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        low_idx, high_idx = 0, len(quantiles) - 1 if len(quantiles) >= 3 else -1

        num_plots = min(len(test_x), self.config.plot_top_k_patterns)
        n_cols, n_rows = 3, math.ceil(num_plots / 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            input_x = np.arange(-self.config.input_length, 0)
            pred_x = np.arange(0, self.config.prediction_length)

            ax.plot(input_x, test_x[i].flatten(), label='Input', color='blue', alpha=0.7)
            ax.plot(pred_x, test_y[i].flatten(), label='True', color='green', linewidth=2)
            ax.plot(pred_x, preds[i, :, median_idx], label='Median', color='red', linestyle='--')
            ax.fill_between(pred_x, preds[i, :, low_idx], preds[i, :, high_idx],
                            color='red', alpha=0.2, label=f'{quantiles[low_idx]}-{quantiles[high_idx]} Q')
            ax.set_title(f'Sample {i + 1}')
            if i == 0:
                ax.legend(loc='upper left', fontsize='small')
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Probabilistic Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'))
        plt.close()


class TiRexTrainer(BaseTimeSeriesTrainer):
    """Orchestrates TiRex training with Quantile Loss.

    Thin subclass of :class:`BaseTimeSeriesTrainer`. The base owns the skeleton
    (``__init__`` generator/pattern setup, ``_select_patterns``,
    ``_create_experiment_dir``, ``_train_model``, ``_save_results``,
    ``_export_to_onnx``). TiRex overrides only the genuine divergences: the
    processor, the model build (+compile), the performance callback, the
    ``model_type`` results prefix, the ``patience=30``/``model_name="TiRex"``
    callback set, and a minimal ``run_experiment`` to fold ``onnx_path`` into
    ``results.json``.
    """

    def _build_processor(self) -> TiRexDataProcessor:
        return TiRexDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> TiRexPerformanceCallback:
        return TiRexPerformanceCallback(self.config, self.processor, viz_dir, "tirex")

    def _build_results_prefix(self) -> str:
        return f"{self.config.experiment_name}_{self.config.model_type}"

    def _make_callbacks(self, exp_dir: Optional[str] = None) -> List:
        """Override: TiRex uses ``patience=30`` / ``model_name="TiRex"`` AND owns
        its experiment dir (D-009)."""
        # DECISION plan_2026-06-09_a3c7304c/D-009
        # Pass a BARE prefix (self._build_results_prefix()) to
        # create_common_callbacks and adopt its RETURNED results_dir as
        # self.exp_dir -- matching the git-original tirex flow (c3fbacef:511,
        # `self.exp_dir = results_dir`). Do NOT pass the pre-created full
        # exp_dir path as results_dir_prefix and do NOT use the base
        # _create_experiment_dir here: that built a SEPARATE doubly-nested
        # results/results/{prefix}_TiRex_{ts2}/best_model.keras while the ONNX
        # read path used the first dir -> checkpoint not found -> silent None
        # when export_onnx=True, and lost the _TiRex dir infix.
        # The exp_dir param is ignored on purpose. See decisions.md D-009.
        callbacks, results_dir = create_common_callbacks(
            model_name="TiRex",
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

    def _build_model(self) -> Union[TiRexCore, TiRexExtended]:
        """Create and compile the TiRex model."""
        factory = create_tirex_by_variant if self.config.model_type == 'core' else create_tirex_extended
        logger.info(f"Creating TiRex{self.config.model_type.title()} ({self.config.variant})")

        model = factory(
            variant=self.config.variant,
            input_length=self.config.input_length,
            prediction_length=self.config.prediction_length,
            patch_size=self.config.patch_size,
            quantile_levels=self.config.quantile_levels,
            dropout_rate=self.config.dropout_rate
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

        loss = QuantileLoss(quantiles=self.config.quantile_levels, normalize=True)
        metrics = []
        if 0.5 in self.config.quantile_levels:
            median_idx = self.config.quantile_levels.index(0.5)
            def mae_of_median(y_true, y_pred):
                return keras.metrics.mean_absolute_error(y_true, y_pred[:, :, median_idx])
            mae_of_median.__name__ = 'mae_of_median'
            metrics.append(mae_of_median)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=True)
        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Base skeleton + TiRex's ONNX export folded into ``results.json``.

        Overridden (not using the bare base ``run_experiment``) for two reasons:
        TiRex's original ``results.json`` carried a 5th ``onnx_path`` key, AND
        TiRex resolves ``self.exp_dir`` from ``create_common_callbacks``'
        returned dir (D-009) rather than the base ``_create_experiment_dir``.
        ``self.exp_dir`` is therefore set INSIDE ``_train_model`` (via its
        ``_make_callbacks`` call); the ONNX read + ``_save_results`` below run
        after it and read the now-coincident ``self.exp_dir`` so checkpoint,
        viz, results.json, and the ONNX read path all live in one dir.
        """
        logger.info("Starting TiRex training experiment")

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
    """Build the TiRex CLI parser on top of the shared TS argument parser.

    Starts from :func:`create_ts_argument_parser` (the shared TS args:
    ``--epochs``/``--batch_size``/``--steps_per_epoch``/``--learning_rate``/
    warmup/analysis/``--gpu`` etc.), restores TiRex's own defaults for the args
    whose default differs from the shared parser via ``set_defaults``
    (experiment_name "tirex", max_patterns_per_category 100), then adds TiRex's
    architecture-specific flags (model_type, variant, input/prediction lengths,
    patch_size, ONNX).
    """
    parser = create_ts_argument_parser("TiRex Training Framework")

    # Restore TiRex's per-arg defaults where they differ from the shared parser.
    # Shared parser already matches TiRex on epochs(200)/batch_size(128)/
    # steps_per_epoch(1000)/learning_rate(1e-4); only experiment_name and
    # max_patterns_per_category differ.
    parser.set_defaults(
        experiment_name="tirex",
        max_patterns_per_category=100,
    )

    # TiRex architecture-specific arguments.
    parser.add_argument("--model_type", type=str, default="core", choices=['core', 'extended'])
    parser.add_argument("--variant", type=str, default="small", choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument("--input_length", type=int, default=256)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=4)
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

    config = TiRexTrainingConfig(
        experiment_name=args.experiment_name,
        model_type=args.model_type,
        variant=args.variant,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        patch_size=args.patch_size,
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
        onnx_opset_version=args.onnx_opset_version
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=10000, random_seed=42, default_noise_level=0.1
    )

    try:
        trainer = TiRexTrainer(config, generator_config)
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
