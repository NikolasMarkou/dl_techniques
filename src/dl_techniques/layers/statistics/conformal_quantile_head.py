"""
``ConformalQuantileHead`` is the quantile output layer of Conformalized Quantile
Regression (CQR) for forecasting.

The layer projects encoded features to three quantiles per time step and feature, a lower
quantile (typically ``alpha/2``), the median and an upper quantile (typically
``1 - alpha/2``). The learned quantiles alone carry no coverage guarantee. A held-out
calibration set measures how far the true values fall outside them, and a single scalar
offset ``q_hat`` widens both bounds by that amount. ``call()`` never applies ``q_hat``;
only :meth:`ConformalQuantileHead.predict_intervals` does, after
:meth:`ConformalQuantileHead.calibrate` has fitted it with the finite-sample order
statistic of the scores (the same rule as
:func:`dl_techniques.utils.conformal_denoiser_intervals.conformal_quantile`).

It is the learned-quantile sibling of
:class:`~dl_techniques.layers.statistics.conformal_interval.ConformalIntervalLayer`,
which widens a point estimate by a fixed radius, and of
:class:`~dl_techniques.layers.statistics.conformal_prediction_set.ConformalPredictionSetLayer`,
which turns classifier logits into a prediction set.

References:
    - Romano, Patterson and Candes, 2019. Conformalized Quantile Regression.
      NeurIPS. (https://arxiv.org/abs/1905.03222)
    - Angelopoulos and Bates, 2021. A Gentle Introduction to Conformal
      Prediction and Distribution-Free Uncertainty Quantification.
      (https://arxiv.org/abs/2107.07511)

"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.utils.conformal_denoiser_intervals import conformal_quantile
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

#: Channels of the last axis of ``call()``.
LOWER, MEDIAN, UPPER = 0, 1, 2
NUM_QUANTILES = 3


@register_dl_technique("dl_techniques.layers.statistics.conformal_quantile_head")
class ConformalQuantileHead(keras.layers.Layer):
    """Quantile output layer for Conformalized Quantile Regression (CQR).

    **Architecture Overview:**

    .. code-block:: text

        This is the call() path. No conformal offset here.

        input: encoded features (B, input_dim)
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  Dense(Tf * output_dim * 3)         │
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │  reshape to (B, Tf, output_dim, 3)  │
        └──────────────────┬──────────────────┘
                           ▼
             output (B, Tf, output_dim, 3)

        B = batch, Tf = forecast_length. On the last axis,
        0 = lower, 1 = median, 2 = upper.

    **Calibration path:**

    .. code-block:: text

        calibrate() writes the non-trainable scalar q_hat:

            q_hat = the ceil((n + 1)(1 - alpha))-th smallest score
                    (+inf when n is too small for alpha)

        predict_intervals() then forks the three channels:

                      preds (B, Tf, out, 3)
                                │
                ┌───────────────┴───────────────┐
                ▼               ▼               ▼
           preds[..., 0]   preds[..., 1]   preds[..., 2]
                │               │               │
                ▼               │               ▼
        ┌───────────────┐       │       ┌───────────────┐
        │ lower - q_hat │       │       │ upper + q_hat │
        └───────┬───────┘       │       └───────┬───────┘
                ▼               ▼               ▼
              lower          median           upper

        The returned tuple is (median, lower, upper), in that
        order. Each element has shape (B, Tf, output_dim).

    :param forecast_length: Number of time steps to forecast. Must be positive.
    :type forecast_length: int
    :param output_dim: Number of features to predict. Must be positive.
    :type output_dim: int
    :param kernel_initializer: Initializer of the projection kernel, as a name, a
        serialized dict or an instance (stored as a spec). Defaults to
        ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer of the projection kernel.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param q_hat_init: Initial conformal offset. Defaults to ``0.0`` (uncalibrated:
        the intervals are the raw learned quantiles). The real value is fitted by
        :meth:`calibrate`; ``get_config`` stores the current one, so a config-only
        reload keeps it.
    :type q_hat_init: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        ``(batch, input_dim)``.

    Output shape:
        ``(batch, forecast_length, output_dim, 3)``.

    :raises ValueError: If ``forecast_length`` or ``output_dim`` is not positive, or
        ``q_hat_init`` is NaN.
    """

    def __init__(
            self,
            forecast_length: int,
            output_dim: int,
            kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
            q_hat_init: float = 0.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if forecast_length <= 0:
            raise ValueError(f"forecast_length must be positive, got {forecast_length}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if q_hat_init != q_hat_init:
            raise ValueError(f"q_hat_init must not be NaN, got {q_hat_init}")

        self.forecast_length = forecast_length
        self.output_dim = output_dim
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.q_hat_init = float(q_hat_init)

        # One Dense for all quantiles: (Tf * output_dim * 3) outputs.
        self.projection = keras.layers.Dense(
            forecast_length * output_dim * NUM_QUANTILES,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="quantile_projection",
        )

        # Created in build().
        self.q_hat = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the projection and create the non-trainable offset weight.

        :param input_shape: Shape of the encoded features, ``(batch, input_dim)``.
        :raises ValueError: If the input is not rank 2 or ``input_dim`` is unknown.
        """
        if len(input_shape) != 2 or input_shape[-1] is None:
            raise ValueError(
                f"ConformalQuantileHead takes (batch, input_dim) features with a "
                f"defined input_dim, got shape {tuple(input_shape)}"
            )
        self.projection.build(input_shape)
        self.q_hat = self.add_weight(
            name="conformal_q",
            shape=(),
            initializer=keras.initializers.Constant(self.q_hat_init),
            trainable=False,
            dtype=self.variable_dtype,
        )
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Project encoded features to the three quantiles.

        :param inputs: Encoded features ``(batch, input_dim)``.
        :param training: Whether in training mode.
        :return: ``(batch, forecast_length, output_dim, 3)`` with ``[lower, median,
            upper]`` on the last axis. ``q_hat`` is not applied.
        """
        x = self.projection(inputs, training=training)
        return keras.ops.reshape(
            x, (-1, self.forecast_length, self.output_dim, NUM_QUANTILES)
        )

    def calibrate(self, calibration_scores: np.ndarray, alpha: float = 0.1) -> None:
        """Fit ``q_hat`` from the held-out nonconformity scores.

        Each score is ``max(q_low - y_true, y_true - q_high)`` on a calibration set the
        model did not train on. ``q_hat`` is the ``ceil((n + 1)(1 - alpha))``-th
        smallest score, the order statistic that gives coverage ``>= 1 - alpha``;
        when ``n`` is too small for ``alpha`` it is ``+inf`` (unbounded intervals).
        Runs in NumPy and writes a Keras weight, so it is not part of a traced graph.

        :param calibration_scores: Scores with shape ``(n_calibration_samples,)``.
        :param alpha: Miscoverage level in ``(0, 1)``. Defaults to 0.1 (90% coverage).
        :raises ValueError: If the layer is not built, the scores are empty, or
            ``alpha`` is outside ``(0, 1)``.
        """
        if self.q_hat is None:
            raise ValueError(
                f"ConformalQuantileHead '{self.name}' is not built: call the layer "
                f"(or build it) before calibrate()"
            )
        q_value = conformal_quantile(np.asarray(calibration_scores), alpha)
        self.q_hat.assign(q_value)
        logger.info(f"ConformalQuantileHead '{self.name}' calibrated q_hat={q_value:.6g}")

    def predict_intervals(
            self,
            inputs: Union[keras.KerasTensor, np.ndarray],
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Compute calibrated prediction intervals.

        Runs the layer with ``training=False`` (building it on first use), then widens
        the interval by ``q_hat``: ``lower - q_hat`` and ``upper + q_hat``. The median
        is returned unchanged. Before :meth:`calibrate` ``q_hat`` is 0 and the bounds
        are the raw learned quantiles.

        :param inputs: Encoded features ``(batch, input_dim)``.
        :return: ``(median, lower, upper)``, each ``(batch, forecast_length, output_dim)``.
            Note the order: the median comes first.
        """
        preds = self(inputs, training=False)
        # q_hat is a variable_dtype weight read outside call(), so it is not autocast:
        # bring it to the output dtype (they differ under mixed_float16).
        q_hat = keras.ops.cast(self.q_hat, preds.dtype)
        median = preds[..., MEDIAN]
        lower = preds[..., LOWER] - q_hat
        upper = preds[..., UPPER] + q_hat
        return median, lower, upper

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], int, int, int]:
        """Only the batch axis is read from ``input_shape``."""
        return (input_shape[0], self.forecast_length, self.output_dim, NUM_QUANTILES)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        Stores the current calibrated ``q_hat`` as ``q_hat_init``, so a config-only
        reload (``from_config`` / ``clone_model``) keeps the calibration even without
        the ``.keras`` weights archive.
        """
        config = super().get_config()
        config.update({
            "forecast_length": self.forecast_length,
            "output_dim": self.output_dim,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "q_hat_init": float(keras.ops.convert_to_numpy(self.q_hat))
            if self.q_hat is not None else self.q_hat_init,
        })
        return config
