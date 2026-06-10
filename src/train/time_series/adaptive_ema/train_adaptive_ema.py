"""
Training pipeline for the AdaptiveEMASlopeFilterModel.

This is a Pattern-2 (time-series / probabilistic) trainer that mirrors
`src/train/tirex/`. It exposes two training modes:

- ``--mode classification``: trains the ``signal_between`` head against a
  binary "in regime" target derived from the realized future slope, using
  ``BinaryCrossentropy``. Only meaningful when ``--learnable-thresholds``
  is set (otherwise the head is zero-gradient).
- ``--mode quantile``: trains the optional ``slope_quantiles`` head against
  the realized future slope, using ``QuantileLoss``.

The model is wrapped at training time in a thin ``keras.Model`` that selects
a single tensor from the dict output so that standard ``model.compile``
+ ``model.fit`` work without a custom training loop.

References:
    Charles LeBeau & David Lucas, 1992 — EMA-slope regime filtering.
    Koenker & Bassett, 1978 — Regression quantiles (used by QuantileLoss).
"""

import os
import sys
import json
import math
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    generate_training_curves,
    set_seeds,
    json_numpy_default,
    create_learning_rate_schedule,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    _fill_nans,
)
from dl_techniques.utils.logger import logger
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.models.time_series.adaptive_ema.model import AdaptiveEMASlopeFilterModel
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    NormalizationMethod,
)
from dl_techniques.analyzer import AnalysisConfig

# ---------------------------------------------------------------------

plt.style.use('default')
sns.set_palette("husl")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, Keras and TF."""
    set_seeds(seed)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class AdaptiveEMATrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for AdaptiveEMASlopeFilterModel training.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags,
    quantile-head toggle, ONNX export) from :class:`BaseTimeSeriesTrainingConfig`
    and adds the AdaptiveEMA-specific fields below.

    Re-declared inherited defaults that diverge from the base: ``epochs`` (50 vs
    150), ``steps_per_epoch`` (200 vs 500), ``batch_size`` (64 vs 128),
    ``learning_rate`` (1e-3 vs 1e-4), ``warmup_steps`` (500 vs 1000),
    ``max_patterns_per_category`` (6 vs 10), ``perform_deep_analysis`` (False vs
    True — too few params to be interesting), ``plot_top_k_patterns`` (6 vs 12),
    and ``use_quantile_head`` (True vs False — the slope-quantile head is the
    core output here). ``target_categories`` is re-declared with a non-None
    default (``["trend", "composite", "financial"]``) because
    :meth:`AdaptiveEMATrainer._select_patterns` iterates it directly.

    Field renames vs the pre-migration standalone config (base names adopted):
    ``plot_top_k_samples`` -> ``plot_top_k_patterns``,
    ``enable_quantile_head`` -> ``use_quantile_head``,
    ``dataset_patterns`` -> ``target_categories``. ``prediction_horizon`` is kept
    as an AdaptiveEMA-specific field (the future-slope horizon the training
    wrapper slices to); the base has no equivalent.
    """

    # Bookkeeping
    experiment_name: str = "adaptive_ema"
    mode: str = "quantile"  # "classification" | "quantile"

    # Re-declared inherited defaults that diverge from the base.
    epochs: int = 50                       # base default: 150
    steps_per_epoch: int = 200             # base default: 500
    batch_size: int = 64                   # base default: 128
    learning_rate: float = 1e-3            # base default: 1e-4
    warmup_steps: int = 500                # base default: 1000
    max_patterns_per_category: int = 6     # base default: 10
    perform_deep_analysis: bool = False    # base default: True
    plot_top_k_patterns: int = 6           # base default: 12
    use_quantile_head: bool = True         # base default: False
    target_categories: Optional[List[str]] = field(  # base default: None
        default_factory=lambda: ["trend", "composite", "financial"]
    )

    # Model (AdaptiveEMA-specific)
    ema_period: int = 25
    lookback_period: int = 25
    initial_upper_threshold: float = 1.5
    initial_lower_threshold: float = -1.5
    learnable_thresholds: bool = True
    adjust_ema: bool = True
    num_quantiles: int = 5
    quantile_dropout_rate: float = 0.1

    # Data (AdaptiveEMA-specific; horizon is the future-slope length)
    input_length: int = 128
    prediction_horizon: int = 24
    num_features: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.mode not in ("classification", "quantile"):
            raise ValueError(
                f"mode must be 'classification' or 'quantile', got '{self.mode}'"
            )
        if self.input_length <= 0 or self.prediction_horizon <= 0:
            raise ValueError(
                "input_length and prediction_horizon must be positive"
            )
        if self.mode == "quantile" and not self.use_quantile_head:
            logger.warning(
                "mode='quantile' but use_quantile_head=False — forcing it on."
            )
            self.use_quantile_head = True
        if self.mode == "classification" and not self.learnable_thresholds:
            logger.warning(
                "mode='classification' with learnable_thresholds=False has "
                "zero gradient — forcing learnable_thresholds=True."
            )
            self.learnable_thresholds = True


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

class AdaptiveEMADataProcessor(WindowedTimeSeriesProcessor):
    """
    Streaming data processor for AdaptiveEMA training.

    Subclass of :class:`WindowedTimeSeriesProcessor` (Pattern-2 convention):
    pulls plausibly price-like synthetic series from ``TimeSeriesGenerator``
    (trend / composite / financial categories), windows them into
    ``(context, future)`` pairs, and produces a binary "in regime" target
    (classification mode) or a continuous future-slope target (quantile mode).

    Sampling is uniform (``pattern_to_category=None``). The base supplies the
    buffered streaming generator, the fixed val/test pre-computation, the raw
    test generator, and ``tf.data`` assembly; this subclass overrides only the
    two hooks (:meth:`_make_sample` / :attr:`output_signature`) for the
    slope/regime target, plus :meth:`_safe_normalize` for AdaptiveEMA's
    centered/unit-variance transform (which the EMA thresholds depend on).
    """

    def __init__(
        self,
        config: AdaptiveEMATrainingConfig,
        generator: TimeSeriesGenerator,
        selected_patterns: List[str],
    ) -> None:
        super().__init__(
            config,
            generator,
            selected_patterns,
            pattern_to_category=None,  # uniform sampling
            context_len=config.input_length,
            horizon_len=config.prediction_horizon,
            num_features=config.num_features,
            normalize=False,  # AdaptiveEMA uses its own _safe_normalize below
            patterns_to_mix=16,
            windows_per_pattern=8,
            min_length_multiplier=1,
            require_finite=True,
        )

    # -- normalization ------------------------------------------------

    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        # DECISION plan_2026-06-10_31eed970/D-004: subclass WindowedTimeSeriesProcessor
        # so NaN handling flows through the shared `_fill_nans` (timeseries.py:147),
        # which carries the D-007 fix. The pre-D-007 in-place form here was
        # `idx = np.where(~mask, np.arange(mask.shape[0]), 0); series = series[idx]`
        # — on a 2-D `(T, 1)` window that BROADCASTS to `(T, T, 1)` (arange is `(T,)`,
        # mask is `(T, 1)`), blowing up the sample shape. It was only ever reached on
        # NaN-containing input (guarded by `if isnan.any()`), so the bug stayed latent.
        # Do NOT reinstate the `np.arange(mask.shape[0])` / `series[idx]` form or an
        # in-place patch of it; rely on `_fill_nans` via the base. See decisions.md D-004.
        # The centered/unit-variance transform below is AdaptiveEMA-specific (keeps the
        # learnable thresholds in [-1.5, 1.5] meaningful) so `normalize=False` is passed
        # to the base and this override owns the whole transform.
        series = np.clip(series, -1e6, 1e6)
        series = _fill_nans(series)
        mean = float(np.mean(series))
        std = float(np.std(series)) + 1e-6
        return ((series - mean) / std).astype(np.float32)

    # -- target generation --------------------------------------------

    def _compute_future_slope(self, future: np.ndarray) -> np.ndarray:
        """Approximate slope of the future window via finite difference."""
        L = max(1, min(self.config.lookback_period, len(future) - 1))
        slope = np.zeros_like(future, dtype=np.float32)
        slope[L:] = future[L:] - future[:-L]
        return slope

    def _target_for_context(
        self, future: np.ndarray
    ) -> np.ndarray:
        """Build training target from the realized future segment.

        The synthetic ``TimeSeriesGenerator`` returns ``(T, 1)`` arrays, so
        ``future`` may be 2D. Targets are 1D ``(prediction_horizon,)`` per the
        :attr:`output_signature`, so we squeeze any trailing singleton feature
        axis defensively.
        """
        if future.ndim == 2 and future.shape[-1] == 1:
            future = future.reshape(-1)
        future_slope = self._compute_future_slope(future)
        if self.config.mode == "classification":
            lo = self.config.initial_lower_threshold
            hi = self.config.initial_upper_threshold
            mask = ((future_slope >= lo) & (future_slope <= hi)).astype(np.float32)
            return mask.astype(np.float32)
        # quantile mode → predict the slope itself
        return future_slope.astype(np.float32)

    # -- hooks (override WindowedTimeSeriesProcessor) -----------------

    def _make_sample(
        self, window: np.ndarray, pattern_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split a normalized window into ``(context, slope/regime target)``."""
        context = window[: self.context_len].reshape(
            -1, self.num_features
        ).astype(np.float32)
        future = window[self.context_len:]
        target = self._target_for_context(future)
        return context, target

    @property
    def output_signature(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        return (
            tf.TensorSpec(
                shape=(self.context_len, self.num_features), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(self.horizon_len,), dtype=tf.float32),
        )

    def _viz_samples(
        self, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self._generate_fixed_dataset("test", n)
        return x, y


# ---------------------------------------------------------------------
# Single-head training wrapper
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AdaptiveEMATrainingWrapper(keras.Model):
    """
    Wraps ``AdaptiveEMASlopeFilterModel`` to expose a single tensor output
    (one of the dict keys) so that ``model.compile(loss=...)`` works with
    the standard Keras training loop.
    """

    def __init__(
        self,
        base: AdaptiveEMASlopeFilterModel,
        output_key: str,
        prediction_horizon: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base = base
        self.output_key = output_key
        self.prediction_horizon = prediction_horizon

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        outputs = self.base(inputs, training=training)
        tensor = outputs[self.output_key]
        # Slice the trailing `prediction_horizon` timesteps so the target
        # of shape (B, H) lines up.
        # tensor shape: (B, T) or (B, T, F) or (B, T, K)
        if len(tensor.shape) == 2:
            return tensor[:, -self.prediction_horizon:]
        return tensor[:, -self.prediction_horizon:, ...]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "output_key": self.output_key,
                "prediction_horizon": self.prediction_horizon,
                "base": keras.saving.serialize_keras_object(self.base),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AdaptiveEMATrainingWrapper":
        base = keras.saving.deserialize_keras_object(config.pop("base"))
        return cls(base=base, **config)


# ---------------------------------------------------------------------
# Visualization callback
# ---------------------------------------------------------------------

class AdaptiveEMAPerformanceCallback(keras.callbacks.Callback):
    """Visualizes predictions vs. realized future slope or in-regime mask."""

    def __init__(
        self,
        config: AdaptiveEMATrainingConfig,
        processor: AdaptiveEMADataProcessor,
        save_dir: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.processor = processor
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.viz_x, self.viz_y = self.processor._viz_samples(
            self.config.plot_top_k_patterns
        )
        self.training_history: Dict[str, List[float]] = {
            "loss": [], "val_loss": [], "lr": []
        }

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        logs = logs or {}
        self.training_history["loss"].append(float(logs.get("loss", 0.0)))
        self.training_history["val_loss"].append(
            float(logs.get("val_loss", 0.0))
        )
        try:
            lr = float(
                keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)
            )
        except Exception:
            lr = 0.0
        self.training_history["lr"].append(lr)

        if (epoch + 1) % self.config.visualize_every_n_epochs != 0:
            return

        logger.info(f"Visualizations for epoch {epoch + 1}")
        if self.config.create_learning_curves:
            try:
                generate_training_curves(
                    history=self.training_history,
                    results_dir=self.save_dir,
                    filename=f"learning_curves_epoch_{epoch + 1:03d}",
                )
            except Exception as exc:
                logger.warning(f"Learning-curve plot failed: {exc}")

        if self.config.create_prediction_plots and len(self.viz_x) > 0:
            self._plot_predictions(epoch)

    def _plot_predictions(self, epoch: int) -> None:
        preds = self.model(self.viz_x, training=False)
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        num_plots = min(len(self.viz_x), self.config.plot_top_k_patterns)
        n_cols = 2
        n_rows = math.ceil(num_plots / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False
        )
        axes = axes.flatten()

        for i in range(num_plots):
            ax = axes[i]
            input_x = np.arange(-self.config.input_length, 0)
            pred_x = np.arange(0, self.config.prediction_horizon)
            ax.plot(
                input_x, self.viz_x[i].flatten(), label="Input price",
                color="blue", alpha=0.6,
            )
            ax.plot(
                pred_x, self.viz_y[i].flatten(), label="Target",
                color="green", linewidth=2,
            )

            if self.config.mode == "quantile" and preds.ndim == 3:
                quantiles = self.config.quantile_levels
                median_idx = (
                    quantiles.index(0.5) if 0.5 in quantiles
                    else len(quantiles) // 2
                )
                low_idx = 0
                high_idx = len(quantiles) - 1
                ax.plot(
                    pred_x, preds[i, :, median_idx], label="Median",
                    color="red", linestyle="--",
                )
                ax.fill_between(
                    pred_x, preds[i, :, low_idx], preds[i, :, high_idx],
                    color="red", alpha=0.2,
                    label=f"{quantiles[low_idx]}-{quantiles[high_idx]} Q",
                )
            else:
                ax.plot(
                    pred_x, preds[i].flatten(), label="Predicted",
                    color="red", linestyle="--",
                )
            ax.set_title(f"Sample {i + 1}")
            if i == 0:
                ax.legend(loc="upper left", fontsize="small")
            ax.grid(True, alpha=0.3)

        for j in range(num_plots, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            f"AdaptiveEMA predictions (Epoch {epoch + 1})", fontsize=14
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"predictions_epoch_{epoch + 1:03d}.png")
        )
        plt.close()


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------

class AdaptiveEMATrainer:
    """Orchestrates the full training experiment."""

    def __init__(
        self,
        config: AdaptiveEMATrainingConfig,
        generator_config: TimeSeriesGeneratorConfig,
    ) -> None:
        self.config = config
        self.generator = TimeSeriesGenerator(generator_config)
        self.selected_patterns = self._select_patterns()
        self.processor = AdaptiveEMADataProcessor(
            config, self.generator, self.selected_patterns
        )
        self.model: Optional[AdaptiveEMATrainingWrapper] = None
        self.exp_dir: Optional[str] = None

    def _select_patterns(self) -> List[str]:
        all_categories = self.generator.get_task_categories()
        selected: List[str] = []
        for cat in (self.config.target_categories or []):
            if cat not in all_categories:
                logger.warning(
                    f"Requested category '{cat}' not in generator categories; "
                    f"available: {all_categories}"
                )
                continue
            tasks = self.generator.get_tasks_by_category(cat)
            selected.extend(
                sorted(tasks)[: self.config.max_patterns_per_category]
            )
        if not selected:
            selected = list(self.generator.get_task_names())[:6]
            logger.warning(
                f"No matching categories — falling back to first 6 tasks: "
                f"{selected}"
            )
        logger.info(f"Selected {len(selected)} patterns: {selected}")
        return selected

    def create_model(self) -> AdaptiveEMATrainingWrapper:
        quantile_head_config: Optional[Dict[str, Any]] = None
        if self.config.use_quantile_head:
            quantile_head_config = {
                "num_quantiles": self.config.num_quantiles,
                "dropout_rate": self.config.quantile_dropout_rate,
                "enforce_monotonicity": True,
                "use_bias": True,
            }

        base = AdaptiveEMASlopeFilterModel(
            ema_period=self.config.ema_period,
            lookback_period=self.config.lookback_period,
            initial_upper_threshold=self.config.initial_upper_threshold,
            initial_lower_threshold=self.config.initial_lower_threshold,
            learnable_thresholds=self.config.learnable_thresholds,
            adjust_ema=self.config.adjust_ema,
            quantile_head_config=quantile_head_config,
        )

        if self.config.mode == "classification":
            output_key = "signal_between"
            loss: Any = keras.losses.BinaryCrossentropy()
            metrics: List[Any] = [keras.metrics.BinaryAccuracy(name="acc")]
        else:
            output_key = "slope_quantiles"
            loss = QuantileLoss(
                quantiles=self.config.quantile_levels, normalize=True
            )
            metrics = []
            if 0.5 in self.config.quantile_levels:
                median_idx = self.config.quantile_levels.index(0.5)

                def mae_of_median(y_true, y_pred):
                    return keras.metrics.mean_absolute_error(
                        y_true, y_pred[:, :, median_idx]
                    )

                mae_of_median.__name__ = "mae_of_median"
                metrics.append(mae_of_median)

        wrapper = AdaptiveEMATrainingWrapper(
            base=base,
            output_key=output_key,
            prediction_horizon=self.config.prediction_horizon,
        )

        # LR schedule
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

        wrapper.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return wrapper

    def _build_results_prefix(self) -> str:
        return f"{self.config.experiment_name}_{self.config.mode}"

    def _train_model(
        self, data_pipeline: Dict[str, Any], prefix: str
    ) -> Dict[str, Any]:
        callbacks, results_dir = create_common_callbacks(
            model_name="AdaptiveEMA",
            results_dir_prefix=prefix,
            monitor="val_loss",
            patience=25,
            use_lr_schedule=self.config.use_warmup,
            include_terminate_on_nan=True,
            include_analyzer=self.config.perform_deep_analysis,
            analyzer_config=AnalysisConfig(
                analyze_weights=True,
                analyze_spectral=False,
                analyze_calibration=False,
                analyze_information_flow=False,
                analyze_training_dynamics=False,
                verbose=False,
            ),
            analyzer_start_epoch=self.config.analysis_start_epoch,
            analyzer_epoch_frequency=self.config.analysis_frequency,
        )
        self.exp_dir = results_dir
        viz_dir = os.path.join(self.exp_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        callbacks.append(
            AdaptiveEMAPerformanceCallback(self.config, self.processor, viz_dir)
        )

        history = self.model.fit(
            data_pipeline["train_ds"],
            validation_data=data_pipeline["val_ds"],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline["validation_steps"],
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Evaluating on test set")
        test_metrics = self.model.evaluate(
            data_pipeline["test_ds"],
            steps=data_pipeline["test_steps"],
            verbose=1,
            return_dict=True,
        )

        best_model_path = os.path.join(self.exp_dir, "best_model.keras")
        onnx_path = self._export_to_onnx(best_model_path, self.exp_dir)

        return {
            "history": history.history,
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "final_epoch": len(history.history["loss"]),
            "onnx_path": onnx_path,
        }

    def _save_results(
        self, results: Dict[str, Any], exp_dir: str
    ) -> None:
        serializable = {
            "history": results["history"],
            "test_metrics": results["test_metrics"],
            "final_epoch": results["final_epoch"],
            "onnx_path": results.get("onnx_path"),
            "config": self.config.__dict__,
        }
        with open(os.path.join(exp_dir, "results.json"), "w") as f:
            json.dump(serializable, f, indent=4, default=json_numpy_default)

    def _export_to_onnx(
        self, model_path: str, exp_dir: str
    ) -> Optional[str]:
        if not self.config.export_onnx:
            return None
        onnx_path = os.path.join(exp_dir, "model.onnx")
        try:
            logger.info(f"Exporting to ONNX: {onnx_path}")
            best_model = keras.saving.load_model(model_path, compile=False)
            input_signature = [
                keras.InputSpec(
                    shape=(None, self.config.input_length, self.config.num_features),
                    dtype="float32",
                )
            ]
            best_model.export(
                onnx_path,
                format="onnx",
                input_signature=input_signature,
                opset_version=self.config.onnx_opset_version,
                verbose=True,
            )
            logger.info(f"ONNX export successful: {onnx_path}")
            return onnx_path
        except Exception as exc:
            logger.error(f"ONNX export failed: {exc}", exc_info=True)
            return None

    def run_experiment(self) -> Dict[str, Any]:
        logger.info("Starting AdaptiveEMA training experiment")
        prefix = self._build_results_prefix()

        data_pipeline = self.processor.prepare_datasets()
        self.model = self.create_model()
        # Build the wrapper with a dummy forward pass so params count is real
        dummy = np.zeros(
            (1, self.config.input_length, self.config.num_features),
            dtype=np.float32,
        )
        _ = self.model(dummy, training=False)
        logger.info(f"Model params: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        training_results = self._train_model(data_pipeline, prefix)
        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)
        return {
            "config": self.config,
            "experiment_dir": self.exp_dir,
            "training_results": training_results,
            "results_dir": self.exp_dir,
        }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdaptiveEMA Slope Filter Training"
    )
    parser.add_argument("--experiment_name", type=str, default="adaptive_ema")
    parser.add_argument(
        "--mode", type=str, default="quantile",
        choices=["classification", "quantile"],
    )
    parser.add_argument("--ema_period", type=int, default=25)
    parser.add_argument("--lookback_period", type=int, default=25)
    parser.add_argument("--initial_upper_threshold", type=float, default=1.5)
    parser.add_argument("--initial_lower_threshold", type=float, default=-1.5)
    parser.add_argument(
        "--no-learnable-thresholds", dest="learnable_thresholds",
        action="store_false",
    )
    parser.set_defaults(learnable_thresholds=True)
    parser.add_argument(
        "--no-quantile-head", dest="use_quantile_head",
        action="store_false",
    )
    parser.set_defaults(use_quantile_head=True)
    parser.add_argument("--num_quantiles", type=int, default=5)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--prediction_horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument(
        "--no-warmup", dest="use_warmup", action="store_false"
    )
    parser.set_defaults(use_warmup=True)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)
    parser.add_argument("--visualize_every_n_epochs", type=int, default=5)
    parser.add_argument(
        "--plot_top_k_samples", dest="plot_top_k_patterns", type=int, default=6
    )
    parser.add_argument(
        "--deep-analysis", dest="perform_deep_analysis", action="store_true",
        help="Enable EpochAnalyzerCallback (off by default — model has few params).",
    )
    parser.set_defaults(perform_deep_analysis=False)
    parser.add_argument("--analysis_frequency", type=int, default=10)
    parser.add_argument("--analysis_start_epoch", type=int, default=1)
    parser.add_argument(
        "--onnx", dest="export_onnx", action="store_true",
        help="Export to ONNX at end of training.",
    )
    parser.set_defaults(export_onnx=False)
    parser.add_argument("--onnx_opset_version", type=int, default=17)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)
    setup_gpu(args.gpu)

    config = AdaptiveEMATrainingConfig(
        experiment_name=args.experiment_name,
        mode=args.mode,
        ema_period=args.ema_period,
        lookback_period=args.lookback_period,
        initial_upper_threshold=args.initial_upper_threshold,
        initial_lower_threshold=args.initial_lower_threshold,
        learnable_thresholds=args.learnable_thresholds,
        use_quantile_head=args.use_quantile_head,
        num_quantiles=args.num_quantiles,
        input_length=args.input_length,
        prediction_horizon=args.prediction_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.warmup_start_lr,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        plot_top_k_patterns=args.plot_top_k_patterns,
        perform_deep_analysis=args.perform_deep_analysis,
        analysis_frequency=args.analysis_frequency,
        analysis_start_epoch=args.analysis_start_epoch,
        export_onnx=args.export_onnx,
        onnx_opset_version=args.onnx_opset_version,
    )

    generator_config = TimeSeriesGeneratorConfig(
        n_samples=4000, random_seed=args.seed, default_noise_level=0.1
    )

    try:
        trainer = AdaptiveEMATrainer(config, generator_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['results_dir']}")
    except Exception as exc:
        logger.error(f"Failed: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        keras.backend.clear_session()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
