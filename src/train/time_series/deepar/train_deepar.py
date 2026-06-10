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
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
    BaseTimeSeriesTrainer,
    create_ts_argument_parser,
    _prepare_viz_data_from_processor,
)
from train.common.timeseries import (
    compute_post_hoc_forecast_metrics,
    _plot_ts_forecast,
)
from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.deepar.model import DeepAR
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
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
# Trainer / callbacks / parser / main: added by steps 5-11.
# ---------------------------------------------------------------------
