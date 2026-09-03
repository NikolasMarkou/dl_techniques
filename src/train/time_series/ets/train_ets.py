"""
Training pipeline for the pure-additive ETS state-space forecaster.

This trainer is small on purpose. ``ETSModel`` has at most THREE trainable
scalars (``alpha``, ``beta``, ``gamma``), so there is no architecture to sweep
and no representation to analyse -- the deep-analysis pass is off by default
because a weight analyser over three scalars reports nothing.

What this pipeline IS for is the comparison the multistep losses exist to make.
The same model, the same data, one flag apart::

    # conventional: minimise the ONE-step-ahead error
    ... --multistep_loss mseh --multistep_h 1

    # multistep: align the objective with a 24-step lead time
    ... --multistep_loss gtmse

``--track_per_horizon`` adds the h = 1..H error profile to the training logs, so
the effect of changing the loss is visible per step rather than only in an
aggregate that hides it.

References:
    Svetunkov, I., Kourentzes, N., & Killick, R. (2023). "Multi-step Estimators
        and Shrinkage Effect in Time Series Models". Computational Statistics.
    Hyndman, R.J., Koehler, A.B., Ord, J.K., & Snyder, R.D. (2008). Forecasting
        with Exponential Smoothing: The State Space Approach. Springer.
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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    set_seeds,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
    BaseTimeSeriesTrainer,
    create_ts_argument_parser,
    _prepare_viz_data_from_processor,
)
from train.common.args import build_generator_config
from train.common.timeseries import _plot_ts_forecast
from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.ets.model import ETS_VARIANTS, ETSModel
from dl_techniques.metrics.time_series_metrics import per_horizon_metrics
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    NormalizationMethod,
)

plt.style.use('default')
sns.set_palette("husl")


# ---------------------------------------------------------------------

@dataclass
class ETSTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for ETS training.

    Inherits the shared time-series fields from
    :class:`BaseTimeSeriesTrainingConfig` -- including ``multistep_loss`` /
    ``multistep_h``, which are the point of this trainer -- and adds the ETS
    state-space fields below.

    Several inherited defaults are re-declared because a three-parameter model
    wants different ones: a much larger learning rate (the parameters are
    smoothing coefficients in ``[0, 1]``, not weights), no warmup, fewer epochs,
    and deep analysis off.
    """

    experiment_name: str = "ets_forecasting"

    # Re-declared: three sigmoid-bounded scalars want a big step and no warmup.
    learning_rate: float = 5e-2
    use_warmup: bool = False
    epochs: int = 50
    steps_per_epoch: int = 100
    perform_deep_analysis: bool = False

    # Windowing
    input_length: int = 168
    prediction_length: int = 24

    # ETS state space
    variant: str = "ANN"
    seasonal_period: Optional[int] = None
    alpha_init: float = 0.3
    beta_init: float = 0.1
    gamma_init: float = 0.1

    # Diagnostics: adds H scalar metrics to the logs (one per horizon step).
    track_per_horizon: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant + multistep validation
        if self.variant not in ETS_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(ETS_VARIANTS)}, got "
                f"{self.variant!r}. Multiplicative and mixed ETS forms are "
                f"deliberately not implemented; see the model README."
            )
        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")
        if self.variant == "AAA":
            if not self.seasonal_period or self.seasonal_period < 2:
                raise ValueError("variant='AAA' requires --seasonal_period > 1")
            if self.input_length < self.seasonal_period + 1:
                raise ValueError(
                    f"input_length={self.input_length} is too short for "
                    f"seasonal_period={self.seasonal_period}"
                )
        if self.multistep_h is not None and self.multistep_h > self.prediction_length:
            raise ValueError(
                f"multistep_h={self.multistep_h} exceeds "
                f"prediction_length={self.prediction_length}"
            )


class ETSDataProcessor(WindowedTimeSeriesProcessor):
    """ETS data processor: a thin subclass of the shared windowed processor.

    Uses the base's default reshape-both hooks (context ->
    ``(input_length, 1)``, horizon -> ``(prediction_length, 1)``), which is
    exactly the shape ``ETSModel`` consumes and emits. Univariate only.
    """

    def __init__(
            self,
            config: ETSTrainingConfig,
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
            normalize=True,
            normalize_method=NormalizationMethod.STANDARD,
        )


class ETSPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks ETS forecast performance and the smoothing parameters themselves.

    The extra history this adds over the base is the thing worth watching here:
    ``alpha`` / ``beta`` / ``gamma`` per epoch. Shrinkage is a statement about
    where those land, so a run that does not record them cannot be compared to
    another.
    """

    def __init__(self, config: ETSTrainingConfig, processor: ETSDataProcessor,
                 save_dir: str, model_name: str = "ets"):
        # processor must be set BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data(), which reads self.processor.
        self.processor = processor
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return _prepare_viz_data_from_processor(
            self.processor, self.config.plot_top_k_patterns)

    def _extend_history(self, logs: dict) -> None:
        self.training_history.setdefault('metric', [])
        self.training_history.setdefault('val_metric', [])
        self.training_history['metric'].append(logs.get('mae', 0))
        self.training_history['val_metric'].append(logs.get('val_mae', 0))

        for name in ('alpha', 'beta', 'gamma'):
            self.training_history.setdefault(name, [])
            self.training_history[name].append(getattr(self.model, name))

        self._track_lr(logs)

    def _plot_predictions(self, epoch: int) -> None:
        context, target = self.viz_test_data
        if len(context) == 0:
            return

        predictions = self.model.predict(context, verbose=0)
        num_samples = min(self.config.plot_top_k_patterns, len(context))
        n_cols, n_rows = 3, math.ceil(num_samples / 3)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_samples):
            _plot_ts_forecast(
                axes[i],
                context[i, :, 0].flatten(),
                target[i, :, 0].flatten(),
                predictions[i, :, 0].flatten(),
                title=f'Sample {i}',
                context_label='Context',
                target_label='Target',
                point_label='Pred',
            )

        for j in range(num_samples, len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f'ETS({self.config.variant}) Forecasts '
            f'(Epoch {epoch + 1}, alpha={self.model.alpha:.3f})',
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


class ETSTrainer(BaseTimeSeriesTrainer):
    """Orchestrates ETS training.

    The base owns the whole skeleton; this overrides only the processor, the
    model build, the performance callback and the results prefix.

    ``INCLUDE_ANALYZER = False`` is a deliberate force-off rather than a
    default: the deep-analysis pass inspects weight distributions and spectra,
    and this model's entire weight set is one to three scalars.
    """

    MODEL_DISPLAY_NAME = "ETS"
    EARLY_STOPPING_PATIENCE = 15
    INCLUDE_ANALYZER = False

    def _build_processor(self) -> ETSDataProcessor:
        return ETSDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> ETSPerformanceCallback:
        return ETSPerformanceCallback(self.config, self.processor, viz_dir, "ets")

    def _build_results_prefix(self) -> str:
        loss_tag = self.config.multistep_loss or "mse"
        return f"{self.config.experiment_name}_{self.config.variant}_{loss_tag}"

    def _build_model(self) -> ETSModel:
        """Create and compile the ETS model."""
        logger.info(
            f"Building ETSModel(variant={self.config.variant}, "
            f"horizon={self.config.prediction_length})"
        )
        model = ETSModel(
            variant=self.config.variant,
            horizon=self.config.prediction_length,
            seasonal_period=self.config.seasonal_period,
            alpha_init=self.config.alpha_init,
            beta_init=self.config.beta_init,
            gamma_init=self.config.gamma_init,
        )

        optimizer = self._build_optimizer()

        # Unlike the other four trainers, `None` here means plain MSE rather
        # than "keep a pre-existing loss": this pipeline is new, so there is no
        # prior behaviour to preserve.
        multistep = self.config.build_multistep_loss()
        if multistep is not None:
            logger.info(
                f"Compiling with MultistepLoss({self.config.multistep_loss}, "
                f"h={self.config.multistep_h})"
            )
            loss: Any = multistep
        else:
            logger.info("Compiling with MeanSquaredError")
            loss = keras.losses.MeanSquaredError()

        metrics: List[Any] = ['mae', 'mse']
        if self.config.track_per_horizon:
            metrics = metrics + per_horizon_metrics(self.config.prediction_length)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.build((None, self.config.input_length, self.processor.num_features))
        return model


def build_parser() -> argparse.ArgumentParser:
    """Build the ETS CLI parser on top of the shared TS argument parser.

    Every flag added here is forwarded explicitly in :func:`main`. That is not
    decoration: ``main()`` constructs the config field by field, so a flag added
    to the parser and forgotten in ``main()`` parses cleanly and then does
    nothing at all.
    """
    parser = create_ts_argument_parser("ETS Training Framework")

    parser.set_defaults(
        experiment_name="ets",
        epochs=50,
        batch_size=128,
        steps_per_epoch=100,
        learning_rate=5e-2,
    )

    parser.add_argument("--variant", type=str, default="ANN", choices=list(ETS_VARIANTS))
    parser.add_argument("--input_length", type=int, default=168)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument(
        "--seasonal_period", type=int, default=None,
        help="Season length m. Required (>1) for --variant AAA.",
    )
    parser.add_argument("--alpha_init", type=float, default=0.3)
    parser.add_argument("--beta_init", type=float, default=0.1)
    parser.add_argument("--gamma_init", type=float, default=0.1)
    parser.add_argument(
        "--multistep_loss", type=str, default=None,
        choices=["mseh", "tmse", "gtmse", "msce"],
        help="Multistep loss aggregation. Omit for plain MSE over the horizon. "
             "'mseh' with --multistep_h 1 reproduces conventional one-step "
             "estimation.",
    )
    parser.add_argument(
        "--multistep_h", type=int, default=None,
        help="Horizon the multistep loss is evaluated over. Defaults to the "
             "full --prediction_length.",
    )
    parser.add_argument(
        "--track_per_horizon", action="store_true",
        help="Log the h = 1..H error profile (adds H scalar metrics).",
    )
    parser.add_argument("--no-normalize", dest="normalize_per_instance", action="store_false")
    parser.set_defaults(normalize_per_instance=True)
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false")
    parser.set_defaults(export_onnx=False)
    parser.add_argument("--onnx_opset_version", type=int, default=17)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    setup_gpu(args.gpu)

    config = ETSTrainingConfig(
        seed=args.seed,
        # Forwarded deliberately: the shared parser has offered
        # --result_dir all along, but the sibling trainers' main()
        # functions never passed it through, so it silently no-opped and
        # every run landed in the repo-root `results/`.
        result_dir=args.result_dir,
        experiment_name=args.experiment_name,
        variant=args.variant,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        seasonal_period=args.seasonal_period,
        alpha_init=args.alpha_init,
        beta_init=args.beta_init,
        gamma_init=args.gamma_init,
        multistep_loss=args.multistep_loss,
        multistep_h=args.multistep_h,
        track_per_horizon=args.track_per_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        normalize_per_instance=args.normalize_per_instance,
        max_patterns_per_category=args.max_patterns_per_category,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch,
        export_onnx=args.export_onnx,
        onnx_opset_version=args.onnx_opset_version,
    )

    generator_config = build_generator_config(args)

    try:
        trainer = ETSTrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(
            f"Completed. alpha={trainer.model.alpha:.4f} "
            f"beta={trainer.model.beta:.4f} gamma={trainer.model.gamma:.4f}"
        )
        logger.info(f"Results: {results['results_dir']}")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        keras.utils.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
