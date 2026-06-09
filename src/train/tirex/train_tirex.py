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
import json
import math
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

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
    generate_training_curves,
    set_seeds,
    json_numpy_default,
    create_learning_rate_schedule,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
)
from dl_techniques.utils.logger import logger
from dl_techniques.losses.quantile_loss import QuantileLoss
from dl_techniques.models.tirex.model import create_tirex_by_variant, TiRexCore
from dl_techniques.models.tirex.model_extended import create_tirex_extended, TiRexExtended
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    NormalizationMethod,
)
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

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
        return (
            tf.TensorSpec(shape=(self.context_len, self.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(self.horizon_len,), dtype=tf.float32),
        )


class TiRexPerformanceCallback(keras.callbacks.Callback):
    """Tracks and visualizes TiRex quantile forecast performance."""

    def __init__(self, config: TiRexTrainingConfig, processor: TiRexDataProcessor,
                 save_dir: str, model_name: str = "tirex"):
        super().__init__()
        self.config = config
        self.processor = processor
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.viz_test_data = self._prepare_viz_data()
        self.training_history: Dict[str, List[float]] = {
            'loss': [], 'val_loss': [], 'mae_median': [], 'val_mae_median': [], 'lr': []
        }

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        viz_x, viz_y = [], []
        for x, y in self.processor._test_generator_raw():
            viz_x.append(x)
            viz_y.append(y)
            if len(viz_x) >= self.config.plot_top_k_patterns:
                break
        if not viz_x:
            return np.array([]), np.array([])
        return np.array(viz_x), np.array(viz_y)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        self.training_history['loss'].append(logs.get('loss', 0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0))
        self.training_history['mae_median'].append(logs.get('mae_of_median', 0))
        self.training_history['val_mae_median'].append(logs.get('val_mae_of_median', 0))

        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        if hasattr(self.model.optimizer.learning_rate, '__call__'):
            lr = float(keras.ops.convert_to_numpy(
                self.model.optimizer.learning_rate(self.model.optimizer.iterations)
            ))
        self.training_history['lr'].append(lr)

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Visualizations for epoch {epoch + 1}")
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)
            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        generate_training_curves(
            history=self.training_history,
            results_dir=self.save_dir,
            filename=f"learning_curves_epoch_{epoch + 1:03d}",
        )

    def _plot_prediction_samples(self, epoch: int) -> None:
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


class TiRexTrainer:
    """Orchestrates TiRex training with Quantile Loss."""

    def __init__(self, config: TiRexTrainingConfig,
                 generator_config: TimeSeriesGeneratorConfig) -> None:
        self.config = config
        self.generator_config = generator_config
        self.generator = TimeSeriesGenerator(generator_config)

        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        self.pattern_to_category = {
            task: cat
            for cat in self.pattern_categories
            for task in self.generator.get_tasks_by_category(cat)
        }
        self.selected_patterns = self._select_patterns()
        self.processor = TiRexDataProcessor(
            config, self.generator, self.selected_patterns, self.pattern_to_category
        )
        self.model: Optional[Union[TiRexCore, TiRexExtended]] = None
        self.exp_dir: Optional[str] = None

    def _select_patterns(self) -> List[str]:
        if self.config.target_categories:
            candidates = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            candidates = self.all_patterns

        selected: List[str] = []
        cat_counts: Dict[str, int] = {}
        for pattern in sorted(candidates):
            cat = self.pattern_to_category.get(pattern)
            if cat and cat_counts.get(cat, 0) < self.config.max_patterns_per_category:
                selected.append(pattern)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = random.sample(selected, self.config.max_patterns)

        logger.info(f"Selected {len(selected)} patterns for training")
        return selected

    def create_model(self) -> Union[TiRexCore, TiRexExtended]:
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
        """Run the complete training experiment."""
        logger.info("Starting TiRex training experiment")
        prefix = self._build_results_prefix()

        data_pipeline = self.processor.prepare_datasets()
        self.model = self.create_model()
        logger.info(f"Model params: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        training_results = self._train_model(data_pipeline, prefix)
        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config, 'experiment_dir': self.exp_dir,
            'training_results': training_results, 'results_dir': self.exp_dir
        }

    def _build_results_prefix(self) -> str:
        return f"{self.config.experiment_name}_{self.config.model_type}"

    def _train_model(self, data_pipeline: Dict, prefix: str) -> Dict[str, Any]:
        callbacks, results_dir = create_common_callbacks(
            model_name="TiRex",
            results_dir_prefix=prefix,
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
        callbacks.append(TiRexPerformanceCallback(self.config, self.processor, viz_dir, "tirex"))

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
            data_pipeline['test_ds'], steps=data_pipeline['test_steps'],
            verbose=1, return_dict=True
        )

        best_model_path = os.path.join(self.exp_dir, 'best_model.keras')
        onnx_path = self._export_to_onnx(best_model_path, self.exp_dir)

        return {
            'history': history.history,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'final_epoch': len(history.history['loss']),
            'onnx_path': onnx_path
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'onnx_path': results.get('onnx_path'),
            'config': self.config.__dict__
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_numpy_default)

    def _export_to_onnx(self, model_path: str, exp_dir: str) -> Optional[str]:
        """Export trained model to ONNX format."""
        if not self.config.export_onnx:
            return None

        onnx_path = os.path.join(exp_dir, 'model.onnx')
        try:
            logger.info(f"Exporting to ONNX: {onnx_path}")
            best_model = keras.saving.load_model(model_path, compile=False)
            input_signature = [
                keras.InputSpec(
                    shape=(None, self.config.input_length, self.processor.num_features),
                    dtype="float32"
                )
            ]
            best_model.export(
                onnx_path, format="onnx",
                input_signature=input_signature,
                opset_version=self.config.onnx_opset_version, verbose=True
            )
            logger.info(f"ONNX export successful: {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"ONNX export failed: {e}", exc_info=True)
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TiRex Training Framework")
    parser.add_argument("--experiment_name", type=str, default="tirex")
    parser.add_argument("--model_type", type=str, default="core", choices=['core', 'extended'])
    parser.add_argument("--variant", type=str, default="small", choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument("--input_length", type=int, default=256)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--no-warmup", dest="use_warmup", action="store_false")
    parser.set_defaults(use_warmup=True)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)
    parser.add_argument("--max_patterns_per_category", type=int, default=100)
    parser.add_argument("--visualize_every_n_epochs", type=int, default=5)
    parser.add_argument("--plot_top_k_patterns", type=int, default=12)
    parser.add_argument("--no-deep-analysis", dest="perform_deep_analysis", action="store_false")
    parser.set_defaults(perform_deep_analysis=True)
    parser.add_argument("--analysis_frequency", type=int, default=10)
    parser.add_argument("--analysis_start_epoch", type=int, default=1)
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false")
    parser.set_defaults(export_onnx=False)
    parser.add_argument("--onnx_opset_version", type=int, default=17)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    return parser.parse_args()


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
