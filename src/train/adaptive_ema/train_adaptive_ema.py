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
from typing import Any, Dict, Generator, List, Optional, Tuple

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
)
from dl_techniques.utils.logger import logger
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.models.time_series.adaptive_ema.model import AdaptiveEMASlopeFilterModel
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
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
class AdaptiveEMATrainingConfig:
    """Configuration for AdaptiveEMASlopeFilterModel training."""

    # Bookkeeping
    experiment_name: str = "adaptive_ema"
    result_dir: str = "results"
    save_results: bool = True
    mode: str = "quantile"  # "classification" | "quantile"

    # Model
    ema_period: int = 25
    lookback_period: int = 25
    initial_upper_threshold: float = 1.5
    initial_lower_threshold: float = -1.5
    learnable_thresholds: bool = True
    adjust_ema: bool = True
    enable_quantile_head: bool = True
    num_quantiles: int = 5
    quantile_dropout_rate: float = 0.1
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )

    # Data
    input_length: int = 128
    prediction_horizon: int = 24
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_features: int = 1
    dataset_patterns: List[str] = field(
        default_factory=lambda: ["trend", "composite", "financial"]
    )
    max_patterns_per_category: int = 6

    # Training
    epochs: int = 50
    batch_size: int = 64
    steps_per_epoch: int = 200
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    optimizer: str = "adamw"
    use_warmup: bool = True
    warmup_steps: int = 500
    warmup_start_lr: float = 1e-6

    # Visualization
    visualize_every_n_epochs: int = 5
    plot_top_k_samples: int = 6
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Deep analysis (off by default — too few params to be interesting)
    perform_deep_analysis: bool = False
    analysis_frequency: int = 10
    analysis_start_epoch: int = 1

    # ONNX export
    export_onnx: bool = False
    onnx_opset_version: int = 17

    def __post_init__(self) -> None:
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
        if self.mode not in ("classification", "quantile"):
            raise ValueError(
                f"mode must be 'classification' or 'quantile', got '{self.mode}'"
            )
        if self.input_length <= 0 or self.prediction_horizon <= 0:
            raise ValueError(
                "input_length and prediction_horizon must be positive"
            )
        if self.mode == "quantile" and not self.enable_quantile_head:
            logger.warning(
                "mode='quantile' but enable_quantile_head=False — forcing it on."
            )
            self.enable_quantile_head = True
        if self.mode == "classification" and not self.learnable_thresholds:
            logger.warning(
                "mode='classification' with learnable_thresholds=False has "
                "zero gradient — forcing learnable_thresholds=True."
            )
            self.learnable_thresholds = True


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

class AdaptiveEMADataProcessor:
    """
    Streaming data processor for AdaptiveEMA training.

    Pulls plausibly price-like synthetic series from ``TimeSeriesGenerator``
    (trend / composite / financial categories), windows them into
    ``(context, future)`` pairs, and produces a binary "in regime" target
    (classification mode) or a continuous future-slope target (quantile mode).
    """

    def __init__(
        self,
        config: AdaptiveEMATrainingConfig,
        generator: TimeSeriesGenerator,
        selected_patterns: List[str],
    ) -> None:
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns

    # -- helpers ------------------------------------------------------

    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        series = np.clip(series, -1e6, 1e6)
        if np.isnan(series).any():
            mask = np.isnan(series)
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            series = series[idx]
            series[np.isnan(series)] = 0.0
        # Centered, unit-variance — keeps thresholds in [-1.5, 1.5] meaningful.
        mean = float(np.mean(series))
        std = float(np.std(series)) + 1e-6
        return ((series - mean) / std).astype(np.float32)

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
        ``future`` may be 2D. Targets are 1D ``(prediction_horizon,)`` per
        the output signature in :meth:`prepare_datasets`, so we squeeze any
        trailing singleton feature axis defensively.
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

    # -- generators ---------------------------------------------------

    def _training_generator(
        self,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        total_len = self.config.input_length + self.config.prediction_horizon
        buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        windows_per_pattern = 8
        patterns_to_mix = 16

        while True:
            if not buffer:
                for name in random.choices(self.selected_patterns, k=patterns_to_mix):
                    data = self.ts_generator.generate_task_data(name)
                    if len(data) < total_len or not np.isfinite(data).all():
                        continue
                    train_end = int(self.config.train_ratio * len(data))
                    train_data = data[:train_end]
                    max_start = len(train_data) - total_len
                    if max_start <= 0:
                        continue
                    for _ in range(windows_per_pattern):
                        start = random.randint(0, max_start)
                        window = self._safe_normalize(
                            train_data[start:start + total_len]
                        )
                        context = window[: self.config.input_length].reshape(
                            -1, self.config.num_features
                        )
                        future = window[self.config.input_length:]
                        target = self._target_for_context(future)
                        buffer.append((context, target))
                random.shuffle(buffer)
            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(
        self, split: str, num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(
            f"Pre-computing {split} dataset ({num_samples} samples)"
        )
        total_len = self.config.input_length + self.config.prediction_horizon
        contexts: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        collected = 0
        cycle = 0
        while collected < num_samples:
            name = self.selected_patterns[cycle % len(self.selected_patterns)]
            cycle += 1
            data = self.ts_generator.generate_task_data(name)
            if len(data) < total_len or not np.isfinite(data).all():
                continue
            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))
            split_data = (
                data[train_end:val_end] if split == "val" else data[val_end:]
            )
            max_start = len(split_data) - total_len
            if max_start <= 0:
                continue
            start = random.randint(0, max_start)
            window = self._safe_normalize(split_data[start:start + total_len])
            contexts.append(
                window[: self.config.input_length].reshape(
                    -1, self.config.num_features
                )
            )
            future = window[self.config.input_length:]
            targets.append(self._target_for_context(future))
            collected += 1
        return (
            np.asarray(contexts, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
        )

    def _viz_samples(
        self, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self._generate_fixed_dataset("test", n)
        return x, y

    def prepare_datasets(self) -> Dict[str, Any]:
        target_len = self.config.prediction_horizon
        output_sig = (
            tf.TensorSpec(
                shape=(self.config.input_length, self.config.num_features),
                dtype=tf.float32,
            ),
            tf.TensorSpec(shape=(target_len,), dtype=tf.float32),
        )
        train_ds = (
            tf.data.Dataset.from_generator(
                self._training_generator, output_signature=output_sig
            )
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_steps = max(20, len(self.selected_patterns))
        test_steps = max(10, len(self.selected_patterns))
        val_x, val_y = self._generate_fixed_dataset(
            "val", val_steps * self.config.batch_size
        )
        test_x, test_y = self._generate_fixed_dataset(
            "test", test_steps * self.config.batch_size
        )

        val_ds = (
            tf.data.Dataset.from_tensor_slices((val_x, val_y))
            .batch(self.config.batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = (
            tf.data.Dataset.from_tensor_slices((test_x, test_y))
            .batch(self.config.batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "validation_steps": val_steps,
            "test_steps": test_steps,
        }


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
            self.config.plot_top_k_samples
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

        num_plots = min(len(self.viz_x), self.config.plot_top_k_samples)
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
        for cat in self.config.dataset_patterns:
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
        if self.config.enable_quantile_head:
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
        "--no-quantile-head", dest="enable_quantile_head",
        action="store_false",
    )
    parser.set_defaults(enable_quantile_head=True)
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
    parser.add_argument("--plot_top_k_samples", type=int, default=6)
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
        enable_quantile_head=args.enable_quantile_head,
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
        plot_top_k_samples=args.plot_top_k_samples,
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
