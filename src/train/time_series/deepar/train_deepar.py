"""
Training pipeline for the DeepAR probabilistic forecasting model.

DeepAR (Salinas et al., 2019) is an autoregressive LSTM that produces
probabilistic forecasts through Monte-Carlo sampling. It learns a global model
across related series, handling diverse scales via a per-series scale factor,
and supports Gaussian (real-valued) or Negative-Binomial (count) likelihoods.

This module rides the shared Pattern-2 time-series scaffolding
(`train.common.timeseries`) and joins the unified `Forecast`/`ForecastMixin`
contract. DeepAR is the first parametric -> Monte-Carlo-sampled ->
empirical-quantile model on the contract and the only one that populates
`Forecast.samples`.

The file is built in stages: this module currently provides the FOUNDATION
(config + training wrapper + data processor). The trainer, callbacks, parser,
and `main` are added by later steps (5-11).

Reference:
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Salinas et al., 2019 - https://arxiv.org/abs/1704.04110
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
import tensorflow as tf
import seaborn as sns

from train.common import (
    setup_gpu,
    set_seeds,
    create_callbacks as create_common_callbacks,
    create_learning_rate_schedule,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
    BaseTimeSeriesTrainer,
    create_ts_argument_parser,
    _prepare_viz_data_from_processor,
)
from train.common.args import build_generator_config
from train.common.timeseries import (
    compute_post_hoc_forecast_metrics,
    _plot_ts_forecast,
)
from dl_techniques.utils.logger import logger
from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.models.time_series.deepar.model import DeepAR
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    NormalizationMethod,
)

plt.style.use('default')
sns.set_palette("husl")


# ---------------------------------------------------------------------
# Configuration (Step 3)
# ---------------------------------------------------------------------


@dataclass
class DeepARTrainingConfig(BaseTimeSeriesTrainingConfig):
    """Configuration for DeepAR probabilistic-forecasting training.

    Inherits the shared time-series fields (data splits, optimizer/warmup knobs,
    pattern selection, category weights, visualization + deep-analysis flags)
    from :class:`BaseTimeSeriesTrainingConfig` and adds the DeepAR architecture
    fields below.

    DeepAR's eager Monte-Carlo sampling path is O(num_samples x prediction_len)
    per batch, so the TS defaults here (``batch_size=64``, ``steps_per_epoch=200``)
    are deliberately lighter than the base classification-scale defaults to keep
    smoke runs fast; ``num_samples`` defaults LOW for the same reason.

    Args:
        input_length: Conditioning (context) length fed to the encoder.
        prediction_length: Forecast horizon length.
        num_features: Target feature dimension (``target_dim``); 1 = univariate.
        covariate_dim: Number of synthetic positional covariate channels.
        num_layers: Number of stacked LSTM layers.
        hidden_dim: LSTM hidden width.
        dropout: LSTM output dropout rate.
        recurrent_dropout: LSTM recurrent dropout rate.
        likelihood: Observation distribution, ``'gaussian'`` or
            ``'negative_binomial'``.
        num_samples: Monte-Carlo sample count for prediction. Keep LOW for
            smokes (eager sampling perf cliff).
        scale_epsilon: Constant added in scale computation; ``>=1.0`` guards
            against near-zero / negative mean -> NaN scale.
    """

    experiment_name: str = "deepar"

    # Training cadence (lighter than base: DeepAR sampling is expensive)
    batch_size: int = 64
    steps_per_epoch: int = 200

    # DeepAR architecture
    input_length: int = 96          # conditioning length
    prediction_length: int = 24
    num_features: int = 1
    covariate_dim: int = 4
    num_layers: int = 3
    hidden_dim: int = 40
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    likelihood: str = "gaussian"
    num_samples: int = 20
    scale_epsilon: float = 1.0

    # ONNX export
    export_onnx: bool = False
    onnx_opset_version: int = 17

    def __post_init__(self) -> None:
        super().__post_init__()  # ratio-sum invariant
        if self.likelihood not in {"gaussian", "negative_binomial"}:
            raise ValueError(
                f"likelihood must be 'gaussian' or 'negative_binomial', "
                f"got '{self.likelihood}'"
            )
        if self.input_length <= 0 or self.prediction_length <= 0:
            raise ValueError("input_length and prediction_length must be positive")
        if self.covariate_dim <= 0:
            raise ValueError("covariate_dim must be positive")


# ---------------------------------------------------------------------
# Training wrapper (Step 2)
# ---------------------------------------------------------------------


# DECISION plan_2026-06-10_7036cab1/D-001 -- add_loss wrapper is the chosen
# resolution: DeepAR.call returns a DICT and its static NLL reads
# y_pred['target'], so compile(loss=DeepAR.gaussian_loss) is structurally broken
# (Keras passes the dataset label as y_pred). Do NOT switch to compile(loss=) or
# force a tensor output from DeepAR.call -- that would alter the shared model's
# contract. Compute NLL in call() + add_loss, compile with loss=None. See
# decisions.md D-001.
@keras.saving.register_keras_serializable()
class DeepARTrainingWrapper(keras.Model, ForecastMixin):
    """Serializable training wrapper that wires DeepAR's NLL via ``add_loss``.

    WHY this wrapper exists (D-001): the base :class:`DeepAR` returns a DICT in
    training mode (``{'mu', 'sigma'/'alpha', 'target'}``) and its static NLL
    losses read ``y_pred['target']``. Keras' ``compile(loss=...)`` path passes
    the *dataset label* as ``y_pred`` (and the model output as ``y_true`` only in
    the functional sense), so ``compile(loss=DeepAR.gaussian_loss)`` is
    structurally broken. The fix (mirroring ``AdaptiveEMATrainingWrapper``) is to
    compute the NLL inside ``call`` and register it with ``self.add_loss(...)``,
    then compile the wrapper with ``loss=None``.

    The wrapper also DELIBERATELY does NOT define ``predict_step`` so that
    ``model.evaluate`` runs the wrapper's ``call`` (the fast training-mode
    add_loss path) instead of DeepAR's slow sampling ``predict_step``.

    Forecasting delegates straight to the base's :meth:`DeepAR._forecast`, so the
    wrapper inherits the full ``ForecastMixin`` contract (``predict_forecast``).

    Args:
        base: A built :class:`DeepAR` instance.
        **kwargs: Forwarded to :class:`keras.Model`.
    """

    def __init__(self, base: DeepAR, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base = base

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the base in training mode, register the NLL, return ``mu``.

        Args:
            inputs: Training-mode dict with ``'target'`` and ``'covariates'``.
            training: Keras training flag, forwarded to the base.

        Returns:
            The ``mu`` parameter tensor ``(batch, seq_len, target_dim)``. The
            loss is attached via ``add_loss`` (the returned tensor is only the
            nominal output for Keras plumbing).
        """
        params = self.base(inputs, training=training)

        # Select the NLL by the base's likelihood. The static losses read the
        # ground-truth from params['target'], so y_true is unused -> pass None.
        if self.base.likelihood == "gaussian":
            loss = type(self.base).gaussian_loss(None, params)
        else:  # negative_binomial
            loss = type(self.base).negative_binomial_loss(None, params)

        self.add_loss(loss)
        return params["mu"]

    def _forecast(self, x: Dict[str, keras.KerasTensor], **kwargs: Any) -> Forecast:
        """Delegate to the base DeepAR's sampling-based forecast."""
        return self.base._forecast(x, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["base"] = keras.saving.serialize_keras_object(self.base)
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepARTrainingWrapper":
        base = keras.saving.deserialize_keras_object(config.pop("base"))
        return cls(base=base, **config)


# ---------------------------------------------------------------------
# Data processor (Step 4)
# ---------------------------------------------------------------------


class DeepARDataProcessor(WindowedTimeSeriesProcessor):
    """DeepAR data processor emitting the model's dict input contract.

    DeepAR's ``call()`` REQUIRES a dict with ``'target'`` and ``'covariates'``
    keys (no ``.get`` guard for covariates; invariant I1). Since the synthetic
    generators produce no real covariates, this processor SYNTHESIZES positional
    sin/cos covariates spanning the full window.

    Unlike the trio default, the FULL window (``context_len + horizon_len``) is
    the teacher-forced ``target`` ``[T, D]`` (DeepAR lags it internally), paired
    with covariates of equal length ``T``. Both ``_make_sample`` and
    ``output_signature`` are overridden to emit the nested dict structure; the
    label is an unused zero placeholder (the NLL lives in the wrapper).
    """

    def __init__(
            self,
            config: DeepARTrainingConfig,
            generator: TimeSeriesGenerator,
            selected_patterns: List[str],
            pattern_to_category: Dict[str, str],
    ) -> None:
        super().__init__(
            config,
            generator,
            selected_patterns,
            pattern_to_category=pattern_to_category,
            context_len=config.input_length,
            horizon_len=config.prediction_length,
            num_features=config.num_features,
            # DECISION plan_2026-06-11_84296249/D-005
            # KEEP normalize=True (STANDARD z-score). The D-002 double-norm was
            # diagnosed and the candidate fix (normalize=False, letting DeepAR's
            # per-series scale own normalization) was EMPIRICALLY FALSIFIED: with
            # raw-magnitude targets (mean~25, max~325) the gaussian NLL goes
            # invalid on batch 1 and TerminateOnNaN fires immediately (the LESSONS
            # "DeepAR scale hazard": mean-based scale on raw inputs -> near-zero/
            # unstable scale -> NaN). The z-scored path trains finitely (loss
            # ~17, no NaN). The model's per-series scale IS rendered a ~no-op by
            # the z-score (|scale-epsilon|~1.2e-4), but with scale_epsilon=1.0 it
            # is a BENIGN identity, not a destabilizer. Do NOT set normalize=False
            # without a model-side scale fix (out of this trainers-only scope).
            # See decisions.md D-005.
            normalize=True,
            normalize_method=NormalizationMethod.STANDARD,
        )
        self.covariate_dim = config.covariate_dim

    def _make_covariates(self, length: int) -> np.ndarray:
        """Build deterministic positional sin/cos covariates.

        This is the SINGLE covariate generator reused by the training path and
        (in later steps) the post-hoc / viz prediction path, so the covariates a
        window was trained on are reproduced exactly at prediction time. Keeping
        it one method avoids the I3 hazard of a divergent prediction-time
        covariate span.

        For ``covariate_dim = C``, it emits ``C // 2`` (sin, cos) frequency pairs
        over ``arange(length)``; if ``C`` is odd, the last channel is zero-padded.

        Args:
            length: Number of timesteps (rows).

        Returns:
            ``float32`` array of shape ``(length, covariate_dim)``.
        """
        positions = np.arange(length, dtype=np.float32)
        n_pairs = self.covariate_dim // 2
        channels: List[np.ndarray] = []
        for k in range(n_pairs):
            # Geometric frequency schedule (period halves each pair).
            freq = 1.0 / (10000.0 ** (k / max(n_pairs, 1)))
            channels.append(np.sin(positions * freq))
            channels.append(np.cos(positions * freq))

        if not channels:
            cov = np.zeros((length, self.covariate_dim), dtype=np.float32)
            return cov

        cov = np.stack(channels, axis=-1).astype(np.float32)  # (length, 2*n_pairs)
        if cov.shape[-1] < self.covariate_dim:  # odd C -> pad trailing channel(s)
            pad = self.covariate_dim - cov.shape[-1]
            cov = np.concatenate(
                [cov, np.zeros((length, pad), dtype=np.float32)], axis=-1
            )
        return cov

    def _make_sample(
            self, window: np.ndarray, pattern_name: str
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Emit DeepAR's training dict from a full normalized window.

        Args:
            window: Normalized window of length ``T = context_len + horizon_len``.
            pattern_name: Source pattern name (unused).

        Returns:
            ``({'target': [T, D], 'covariates': [T, C]}, zeros[(1,)])``. The label
            is a zero placeholder; the NLL is computed by the wrapper.
        """
        T = self.context_len + self.horizon_len
        target = window.reshape(T, self.num_features).astype(np.float32)
        cov = self._make_covariates(T)
        return ({"target": target, "covariates": cov},
                np.zeros((1,), np.float32))

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        """``tf.TensorSpec`` structure matching :meth:`_make_sample`."""
        T = self.context_len + self.horizon_len
        return (
            {
                "target": tf.TensorSpec(
                    shape=(T, self.num_features), dtype=tf.float32),
                "covariates": tf.TensorSpec(
                    shape=(T, self.covariate_dim), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )


# ---------------------------------------------------------------------
# Performance callback (Step 7)
# ---------------------------------------------------------------------


class DeepARPerformanceCallback(TimeSeriesPerformanceCallback):
    """Tracks and visualizes DeepAR probabilistic-forecast performance.

    Thin subclass of :class:`TimeSeriesPerformanceCallback` (mirrors the tirex
    callback). The base owns the scaffolding (``__init__`` + makedirs,
    ``loss``/``val_loss`` accumulation, the ``visualize_every_n_epochs`` gate, the
    non-fatal viz guard, learning-curve delegation). DeepAR keeps only the
    model-specific pieces: viz-data prep from its processor, optional ``lr``
    tracking, and the fan-chart prediction body rendered through the SAMPLING
    forecast path (``predict_forecast``).

    Each viz window has full length ``T = L + H``; the conditioning prefix
    ``window[:L]`` and the synthetic ``full_covariates`` spanning ``L + H`` (I3)
    are assembled into the prediction-mode dict, then the empirical median +
    low/high quantile band are routed through :func:`_plot_ts_forecast`.
    """

    def __init__(self, config: DeepARTrainingConfig, processor: DeepARDataProcessor,
                 save_dir: str, model_name: str = "deepar"):
        # processor must be set BEFORE super().__init__: the base ctor calls
        # _prepare_viz_data() which reads self.processor.
        self.processor = processor
        super().__init__(config, save_dir, model_name)

    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # DeepAR's processor yields dict inputs ({'target','covariates'}), so the
        # generic `_prepare_viz_data_from_processor` would stack a list of dicts
        # into a 1-D object array. Extract and stack the 'target' window
        # [T, D] per sample into a proper [B, T, D] array for the fan-chart split.
        viz_targets = []
        for x, _y in self.processor._test_generator_raw():
            tgt = x['target'] if isinstance(x, dict) else x
            viz_targets.append(np.asarray(tgt, dtype=np.float32))
            if len(viz_targets) >= self.config.plot_top_k_patterns:
                break
        if not viz_targets:
            return np.array([]), np.array([])
        return np.stack(viz_targets, axis=0), np.zeros((len(viz_targets),), np.float32)

    def _extend_history(self, logs: dict) -> None:
        self._track_lr(logs)

    def _plot_predictions(self, epoch: int) -> None:
        # viz_test_data holds full windows: x is the dict-mode sample's 'target'
        # [T, D] (T = L + H); y is the unused zero placeholder. We split each
        # window into conditioning [L, D] and true future [H, D].
        test_x, _ = self.viz_test_data
        if len(test_x) == 0:
            return

        L = self.config.input_length
        H = self.config.prediction_length

        windows = np.asarray(test_x)              # [B, T, D] (or [B, T])
        if windows.ndim == 2:
            windows = windows[..., np.newaxis]    # -> [B, T, 1]
        B = windows.shape[0]

        cond = windows[:, :L, :]                  # [B, L, D]
        true_future = windows[:, L:L + H, :]      # [B, H, D]

        # ONE covariate generator (step 4) spanning the WHOLE L+H horizon (I3).
        full_cov = np.stack(
            [self.processor._make_covariates(L + H)] * B, axis=0)  # [B, L+H, C]

        pred_dict = {'conditioning_target': cond, 'full_covariates': full_cov}
        fc = self.model.predict_forecast(pred_dict)

        # Empirical low/high band from the trained quantile levels.
        lower, upper = fc.interval(fc.quantile_levels[0], fc.quantile_levels[-1])

        num_plots = min(B, self.config.plot_top_k_patterns)
        n_cols, n_rows = 3, math.ceil(num_plots / 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_plots):
            _plot_ts_forecast(
                axes[i],
                cond[i, :, 0],
                true_future[i, :, 0],
                fc.point[i, :, 0],
                lower=lower[i, :, 0],
                upper=upper[i, :, 0],
                title=f'Sample {i + 1}',
                context_label='Input',
                target_label='True',
                point_label='Median',
                band_label=f'{fc.quantile_levels[0]}-{fc.quantile_levels[-1]} Q',
            )

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'DeepAR Probabilistic Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


# ---------------------------------------------------------------------
# Trainer (Steps 5, 6, 8, 9, 10)
# ---------------------------------------------------------------------


class DeepARTrainer(BaseTimeSeriesTrainer):
    """Orchestrates DeepAR probabilistic-forecast training.

    Thin subclass of :class:`BaseTimeSeriesTrainer`. The base owns the skeleton
    (``__init__`` generator/pattern setup, ``_select_patterns``,
    ``_create_experiment_dir``, ``_train_model``, ``_save_results``,
    ``_export_to_onnx``). DeepAR overrides only the genuine divergences: the
    processor, the wrapped model build (+compile ``loss=None``, D-001), the
    dict-input post-hoc metrics (I4), the performance callback, the
    ``likelihood`` results prefix, and the ``MODEL_DISPLAY_NAME`` class attr.
    """

    MODEL_DISPLAY_NAME = "DeepAR"

    def _build_processor(self) -> DeepARDataProcessor:
        return DeepARDataProcessor(
            self.config, self.generator, self.selected_patterns,
            self.pattern_to_category,
        )

    def _build_performance_callback(self, viz_dir: str) -> DeepARPerformanceCallback:
        return DeepARPerformanceCallback(self.config, self.processor, viz_dir, "deepar")

    def _build_results_prefix(self) -> str:
        return f"{self.config.experiment_name}_{self.config.likelihood}"

    def _build_model(self) -> DeepARTrainingWrapper:
        """Construct DeepAR, wrap it for the add_loss NLL, and compile.

        Returns the wrapper (the base assigns ``self.model = self._build_model()``,
        matching tirex). The base is BUILT with a training-mode dummy dict forward
        pass; the wrapper is then built by one call on the same dummy dict before
        ``compile(loss=None)`` (D-001 / I2). NO ``jit_compile`` — DeepAR's eager
        sampling path is not XLA-traceable.
        """
        logger.info(
            f"Creating DeepAR (likelihood={self.config.likelihood}, "
            f"layers={self.config.num_layers}, hidden={self.config.hidden_dim})")

        base = DeepAR(
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            recurrent_dropout=self.config.recurrent_dropout,
            likelihood=self.config.likelihood,
            target_dim=self.config.num_features,
            num_samples=self.config.num_samples,
            scale_epsilon=self.config.scale_epsilon,
        )

        # Build the base + wrapper with a training-mode dummy dict (T = L + H).
        T = self.config.input_length + self.config.prediction_length
        D = self.config.num_features
        C = self.config.covariate_dim
        dummy = {
            'target': np.zeros((1, T, D), dtype=np.float32),
            'covariates': np.zeros((1, T, C), dtype=np.float32),
        }
        base(dummy, training=False)

        wrapper = DeepARTrainingWrapper(base)
        wrapper(dummy, training=False)  # build the wrapper's weights/loss graph

        optimizer = self._build_optimizer()

        # loss=None: the NLL is registered inside the wrapper via add_loss (D-001).
        wrapper.compile(optimizer=optimizer, loss=None)
        return wrapper

    # DECISION plan_2026-06-10_7036cab1/D-001 -- the dict-input post-hoc assembly
    # is the non-obvious resolution of I4: the base extractor does inputs[0]
    # (timeseries.py:1050), incompatible with DeepAR's dict test_data_raw, AND it
    # must build the PREDICTION-mode dict {conditioning_target, full_covariates}
    # from the teacher-forced training window. Do NOT fall back to the base
    # extractor and do NOT let full_covariates span only H -- it MUST span L+H
    # (I3) or DeepAR silently mis-derives pred_len. One covariate generator
    # (processor._make_covariates) is reused for train + this path. See D-001.
    def _compute_post_hoc_metrics(self, data_pipeline: Dict[str, Any]) -> Dict[str, float]:
        """Override: assemble DeepAR's prediction-mode dict from the test window.

        Keeps the base contract's ADDITIVE / NON-FATAL / ForecastMixin-GATED
        structure: a new ``post_hoc_metrics`` key only, wrapped in try/except
        (warning + ``{}`` on failure), run only when the model is a
        :class:`ForecastMixin`.
        """
        try:
            test_data_raw = data_pipeline.get('test_data_raw')
            if test_data_raw is None:
                return {}

            if not isinstance(self.model, ForecastMixin):
                logger.info(
                    "post_hoc_metrics: model is not a ForecastMixin; skipping")
                return {}

            # DeepAR processor emits ({'target':[T,D],'covariates':[T,C]}, zeros).
            # After stacking, test_data_raw == (inputs_dict, zeros) where
            # inputs_dict['target'] is [B, T, D] (T = L + H).
            inputs, _ = test_data_raw
            target_stack = np.asarray(inputs['target'])  # [B, T, D]
            if target_stack.ndim == 2:
                target_stack = target_stack[..., np.newaxis]

            L = self.config.input_length
            H = self.config.prediction_length
            B = target_stack.shape[0]

            cond = target_stack[:, :L, :]            # [B, L, D]
            y_true = target_stack[:, L:L + H, :]     # [B, H, D]

            full_cov = np.stack(
                [self.processor._make_covariates(L + H)] * B, axis=0)  # [B, L+H, C]
            assert full_cov.shape[1] == L + H, (
                f"full_covariates must span L+H={L + H}, got {full_cov.shape[1]}")

            fc = self.model.predict_forecast(
                {'conditioning_target': cond, 'full_covariates': full_cov})

            quantiles = fc.quantiles if fc.has_quantiles() else None
            quantile_levels = fc.quantile_levels if fc.has_quantiles() else None

            return compute_post_hoc_forecast_metrics(
                np.asarray(y_true),
                np.asarray(fc.point),
                backcast=np.asarray(cond),
                quantiles=None if quantiles is None else np.asarray(quantiles),
                quantile_levels=quantile_levels,
            )
        except Exception as e:
            logger.warning(f"post_hoc_metrics computation failed: {e}")
            return {}

# ---------------------------------------------------------------------
# CLI parser + main (Step 11)
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the DeepAR CLI parser on top of the shared TS argument parser.

    Starts from :func:`create_ts_argument_parser` (shared TS args:
    ``--epochs``/``--batch_size``/``--steps_per_epoch``/``--learning_rate``/
    warmup/analysis/``--gpu``), restores DeepAR's own defaults via
    ``set_defaults`` (``experiment_name="deepar"``, lighter cadence), then adds
    DeepAR's architecture-specific flags. ``--input_length`` is the conditioning
    (context) length fed to the encoder.
    """
    parser = create_ts_argument_parser("DeepAR Training")
    parser.set_defaults(
        experiment_name="deepar",
        batch_size=64,
        steps_per_epoch=200,
    )

    # DeepAR architecture-specific arguments.
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=40)
    parser.add_argument("--likelihood", type=str, default="gaussian",
                        choices=['gaussian', 'negative_binomial'])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--covariate_dim", type=int, default=4)
    parser.add_argument("--input_length", type=int, default=96,
                        help="Conditioning (context) length fed to the encoder.")
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--num_features", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--recurrent_dropout", type=float, default=0.0)
    parser.add_argument("--scale_epsilon", type=float, default=1.0)
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

    config = DeepARTrainingConfig(
        experiment_name=args.experiment_name,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        num_features=args.num_features,
        covariate_dim=args.covariate_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        likelihood=args.likelihood,
        num_samples=args.num_samples,
        scale_epsilon=args.scale_epsilon,
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
        seed=args.seed,
    )

    generator_config = build_generator_config(args)

    try:
        trainer = DeepARTrainer(config, generator_config)
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
