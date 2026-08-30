"""
Residual autocorrelation (ACF) diagnostics and regularization for forecasts.

`ResidualACFLayer` sits between a forecasting model and its loss. It takes
``[predictions, targets]``, forms the residuals ``r = predictions - targets``,
and measures their autocorrelation up to ``max_lag``. Predictions come back out
unchanged, so the layer drops into an existing model without altering what that
model computes.

A well-fitted forecaster leaves residuals that look like white noise. Any
autocorrelation left in them is structure the model failed to use.

**What it does:**

- Reports ACF statistics for diagnostics, without touching the forward pass.
- Optionally adds a regularization loss that penalizes autocorrelation at
  chosen lags, during training only.
- Ships `ACFMonitorCallback`, which logs those statistics while training runs.

**Mathematical Foundation:**

The autocorrelation at lag ``k`` is::

    ACF(k) = Cov(r_t, r_{t-k}) / Var(r_t)

For white noise, ACF(k) is about 0 for every k > 0. ACF(0) is 1 by definition,
and the layer emits an exact 1.0 for it rather than computing it.

The two terms average over different sample counts. The covariance at lag ``k``
averages over the ``T - k`` overlapping positions; the variance averages over
all ``T``. The more common estimator divides both by ``T``, which shrinks values
at large lags, so this layer reports slightly larger magnitudes there.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.statistics.residual_acf")
class ResidualACFLayer(keras.layers.Layer):
    """
    Measures the autocorrelation of forecast residuals and can penalize it.

    The layer takes two tensors of the same shape, ``[predictions, targets]``,
    and subtracts them to get residuals. It computes the residual ACF for lags
    0 through ``max_lag`` and keeps the result on ``acf_values``. It then
    returns the predictions untouched.

    Set ``regularization_weight`` and the layer also adds a loss during
    training::

        weight * (mean ACF^2 + mean max(0, |ACF| - threshold)^2)

    Both means call ``ops.mean`` with no axis, so each averages over every
    element of the selected-lag slice: batch, lags and features. That slice is
    shaped ``(batch, len(target_lags), features)``, so each term is a single
    scalar, not one value per lag. The first term pushes every targeted
    autocorrelation toward zero. The second term adds a hinge that only bites
    once a lag exceeds ``acf_threshold``.

    The layer owns no weights and adds nothing to the forward computation, so
    monitoring-only mode (``regularization_weight=None``) is free apart from the
    ACF arithmetic.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────┐        ┌─────────────────────┐
        │ inputs[0]           │        │ inputs[1]           │
        │ predictions         │        │ targets             │
        │ (..., T, F) tensor  │        │ (..., T, F) tensor  │
        └─────┬─────────┬─────┘        └──────────┬──────────┘
              │         │                         │
              │         └────────────┬────────────┘
              │                      ▼
              │      ┌────────────────────────────────┐
              │      │ residuals = predictions        │
              │      │           - targets            │
              │      └───────────────┬────────────────┘
              │                      ▼ (..., T, F)
              │      ┌────────────────────────────────┐
              │      │ compute_acf                    │
              │      └───────────────┬────────────────┘
              │                      │ acf
              │                      │ (..., max_lag+1, F)
              │            ┌─────────┴─────────┐
              │            ▼                   ▼
              │  ┌──────────────────┐ ┌──────────────────┐
              │  │ self.acf_values  │ │ add_loss(...)    │
              │  │ = acf            │ │ (optional: only  │
              │  │ (plain attribute)│ │  when training)  │
              │  └──────────────────┘ └──────────────────┘
              ▼
        ┌─────────────────────┐
        │ return predictions  │
        │ (..., T, F)         │
        └─────────────────────┘

    Both boxes at the top are inputs, not weights; the layer creates no
    variables at all. The pass-through branch on the left never touches the ACF
    branch, so removing the loss does not change the output.

    **Inside compute_acf:**

    .. code-block:: text

        residuals (..., T, F)
                 ▼
        ┌────────────────────────────────────────────┐
        │ mean     = mean(residuals, axis=-2)        │
        │ centered = residuals - mean                │
        └──────────────────────┬─────────────────────┘
                               ▼ (..., T, F)
        ┌────────────────────────────────────────────┐
        │ variance = mean(square(centered), axis=-2) │
        │          + epsilon             (..., 1, F) │
        │ slot     = centered[..., :1, :](..., 1, F) │
        └──────────────────────┬─────────────────────┘
                               ▼
        ┌────────────────────────────────────────────┐
        │ acf_list = [ones_like(slot)]   lag 0 = 1.0 │
        └──────────────────────┬─────────────────────┘
                               ▼
                    for lag in 1..max_lag
                    lag >= seq_length ?
              ┌────────────────┴────────────────┐
             yes                               no
              ▼                                 ▼
        ┌──────────────────┐  ┌──────────────────────────┐
        │ zeros_like(slot) │  │ segment1 =               │
        │ (no overlap)     │  │   centered[..., :-lag, :]│
        │                  │  │ segment2 =               │
        │                  │  │   centered[..., lag:, :] │
        │                  │  │ mean(segment1 * segment2,│
        │                  │  │      axis=-2) / variance │
        └────────┬─────────┘  └────────────┬─────────────┘
                 └─────────────┬───────────┘
                               ▼ append (..., 1, F)
        ┌────────────────────────────────────────────┐
        │ concatenate(acf_list, axis=-2)             │
        │ acf  (..., max_lag + 1, F)                 │
        └────────────────────────────────────────────┘

    ``segment1`` holds the earlier values ``r_{t-lag}`` and ``segment2`` the
    later values ``r_t``; both are ``T - lag`` long, so their product averages
    over fewer positions than ``variance`` does. Every lag contributes exactly
    one ``(..., 1, F)`` slice, which is why the concatenation has ``max_lag + 1``
    entries.

    :param max_lag: Highest lag the ACF is computed for. Must be >= 1.
        Defaults to 40.
    :type max_lag: int
    :param regularization_weight: Weight on the ACF regularization loss. If
        "None", the layer only monitors and adds no loss. Defaults to "None".
    :type regularization_weight: float | None
    :param target_lags: Lags the regularization targets. If "None", every lag
        from 1 to ``max_lag`` is targeted. Defaults to "None".
    :type target_lags: list[int] | None
    :param acf_threshold: Level above which an ACF value picks up the extra
        hinge penalty. Defaults to 0.1.
    :type acf_threshold: float
    :param use_absolute_acf: Whether to take absolute ACF values before the
        penalty terms. Defaults to "True". See the Note below: this flag does
        not change the loss.
    :type use_absolute_acf: bool
    :param epsilon: Added to the variance so the ACF divide cannot blow up on a
        constant residual. Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar acf_values: ACF tensor from the most recent ``call``, or ``None``
        before the first call. A plain attribute, not a weight, so it is not
        serialized.
    :vartype acf_values: keras.KerasTensor | None

    :raises ValueError: If ``max_lag`` is less than 1, if
        ``regularization_weight`` is negative, if ``acf_threshold`` is negative,
        or if any entry of ``target_lags`` falls outside ``[1, max_lag]``.

    Input shape:
        A list of two tensors, ``[predictions, targets]``, of identical shape
        ``(..., sequence_length, features)``. The time axis is -2.

    Output shape:
        Same as ``predictions``.

    Example:
        >>> acf = ResidualACFLayer(max_lag=10, regularization_weight=0.01)
        >>> out = acf([predictions, targets])
        >>> stats = acf.get_acf_summary()

    Note:
        ``use_absolute_acf`` has no effect on the loss. Both penalty terms
        already square or take the absolute value of their argument, so an
        ``abs`` in front of them cancels. The flag is still stored and
        serialized.

    Note:
        Lags at or beyond the sequence length have no overlapping samples, so
        the layer reports exactly 0 for them instead of dividing by nothing.
    """

    def __init__(
        self,
        max_lag: int = 40,
        regularization_weight: Optional[float] = None,
        target_lags: Optional[List[int]] = None,
        acf_threshold: float = 0.1,
        use_absolute_acf: bool = True,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Initialize the ResidualACFLayer.

        The layer creates no weights, here or in ``build``. See the class
        docstring for the parameters.

        :raises ValueError: If ``max_lag`` is less than 1, if
            ``regularization_weight`` is negative, if ``acf_threshold`` is
            negative, or if any entry of ``target_lags`` falls outside
            ``[1, max_lag]``.
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        if regularization_weight is not None and regularization_weight < 0:
            raise ValueError(f"regularization_weight must be >= 0, got {regularization_weight}")
        if acf_threshold < 0:
            raise ValueError(f"acf_threshold must be >= 0, got {acf_threshold}")

        # Store configuration parameters
        self.max_lag = max_lag
        self.regularization_weight = regularization_weight
        self.target_lags = target_lags if target_lags is not None else list(range(1, max_lag + 1))
        self.acf_threshold = acf_threshold
        self.use_absolute_acf = use_absolute_acf
        self.epsilon = epsilon

        # Validate target lags
        for lag in self.target_lags:
            if lag < 1 or lag > max_lag:
                raise ValueError(f"target_lags must be between 1 and {max_lag}, got {lag}")

        # Holds the ACF tensor from the last call. Not a weight and not
        # serialized: its shape follows the dynamic batch dimension, so storing
        # it as a weight would pin the layer to one batch size.
        self.acf_values = None

        # Static sequence length, filled in by ``build``. A Python int, or None
        # when the time axis is dynamic.
        self._seq_length = None

        logger.debug(f"Initialized ResidualACFLayer with max_lag={max_lag}, "
                     f"regularization_weight={regularization_weight}")

    def build(self, input_shape: Union[List[Tuple], Tuple[Tuple]]) -> None:
        """Validate the two input shapes and record the sequence length.

        No weights are created. The one thing this does that matters later is
        capture the static length of the time axis (-2) into ``_seq_length``,
        which ``compute_acf`` uses to bound its lag loop.

        :param input_shape: Pair of shapes for ``(predictions, targets)``.
        :type input_shape: list[tuple] | tuple[tuple]
        :raises ValueError: If ``input_shape`` is not a pair, or if the two
            shapes differ.
        """
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError(
                f"ResidualACFLayer expects a list of 2 input shapes "
                f"[predictions, targets], got a {type(input_shape).__name__}"
            )
        if len(input_shape) != 2:
            raise ValueError(
                f"ResidualACFLayer expects a list of 2 input shapes "
                f"[predictions, targets], got {len(input_shape)}: {input_shape}"
            )

        pred_shape, target_shape = input_shape

        # Validate that prediction and target shapes match
        if pred_shape != target_shape:
            raise ValueError(f"Predictions shape {pred_shape} must match targets shape {target_shape}")

        # Capture the static sequence length when it is known. It stays None for
        # a fully dynamic time axis.
        if isinstance(pred_shape, (list, tuple)) and len(pred_shape) >= 2:
            self._seq_length = pred_shape[-2]
        else:
            self._seq_length = None

        logger.debug(f"Built ResidualACFLayer with input shapes: {input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def compute_acf(self, residuals: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the autocorrelation function of the residuals.

        Returns one slice per lag, from lag 0 up to ``max_lag``, concatenated
        along the time axis. Lag 0 is an exact 1.0. Lags with no overlapping
        samples are an exact 0.0. See the class docstring diagram for the
        per-lag arithmetic.

        :param residuals: Residuals of shape ``(..., sequence_length, features)``.
        :type residuals: keras.KerasTensor
        :return: ACF values of shape ``(..., max_lag + 1, features)``.
        :rtype: keras.KerasTensor
        """
        # Center the residuals by removing mean
        mean = ops.mean(residuals, axis=-2, keepdims=True)
        centered = residuals - mean

        # Compute variance (lag 0 autocovariance) with numerical stability
        variance = ops.mean(ops.square(centered), axis=-2, keepdims=True) + self.epsilon

        # A (..., 1, features) template with the right dynamic batch and feature
        # extent. The lag-0 ones and the out-of-range zeros are built from it
        # with ``*_like``, so no shape vector is ever assembled from a symbolic
        # ``ops.shape``.
        slot = centered[..., :1, :]

        # Initialize list to collect ACF values for each lag
        acf_list = []

        # ACF at lag 0 is always exactly 1.0 by definition.
        acf_list.append(ops.ones_like(slot))

        # DECISION plan_2026-06-08_a5f40f4f/D-003: never branch on a symbolic
        # shape here. Slice with Python-int lag offsets and decide in-range vs
        # out-of-range from the STATIC sequence length. A symbolic
        # ``if lag < seq_length`` raises under tf.function or is always true,
        # slicing wrongly when the series is shorter than max_lag.
        seq_length = self._seq_length
        if seq_length is None:
            static_seq = residuals.shape[-2]
            seq_length = int(static_seq) if static_seq is not None else None

        for lag in range(1, self.max_lag + 1):
            if seq_length is not None and lag >= seq_length:
                # Out-of-range lag for this (statically-known) sequence length:
                # ACF is deterministically 0 (no overlapping samples).
                acf_list.append(ops.zeros_like(slot))
                continue

            # Overlapping segments via Python-int slicing (graph-safe).
            # segment1 is the earlier segment r_{t-k}, segment2 the later r_t.
            segment1 = centered[..., :-lag, :]
            segment2 = centered[..., lag:, :]

            # Autocovariance at this lag, normalized by lag-0 variance.
            autocovariance = ops.mean(segment1 * segment2, axis=-2, keepdims=True)
            acf_list.append(autocovariance / variance)

        # Concatenate all ACF values along the lag dimension -> (..., max_lag+1, features)
        acf = ops.concatenate(acf_list, axis=-2)

        return acf

    def call(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute the residual ACF and return the predictions unchanged.

        Stores the ACF on ``acf_values``. Adds the regularization loss only when
        ``regularization_weight`` is set and ``training`` is exactly ``True``.

        :param inputs: List of ``[predictions, targets]`` tensors of matching
            shape.
        :type inputs: list[keras.KerasTensor] | tuple[keras.KerasTensor]
        :param training: Boolean for training mode.
        :type training: bool | None
        :return: The predictions tensor, unchanged.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a pair of tensors.
        """
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                f"ResidualACFLayer expects a list of 2 inputs "
                f"[predictions, targets], got a {type(inputs).__name__}"
            )
        if len(inputs) != 2:
            raise ValueError(
                f"ResidualACFLayer expects a list of 2 inputs "
                f"[predictions, targets], got {len(inputs)}"
            )

        predictions, targets = inputs

        # Compute residuals (prediction errors)
        residuals = predictions - targets

        # Compute autocorrelation function
        acf = self.compute_acf(residuals)

        # Store ACF values for monitoring and diagnostics
        self.acf_values = acf

        # The test is ``training is True``, not ``if training``. The default at
        # inference is training=None, which must not fire the loss, and identity
        # keeps any truthy non-bool from counting as training.
        if self.regularization_weight is not None and training is True:
            # Extract ACF values at target lags (excluding lag 0 which is always 1)
            target_acf_list = []
            for lag in self.target_lags:
                # Use tensor slicing to extract specific lag values
                lag_acf = acf[..., lag:lag + 1, :]
                target_acf_list.append(lag_acf)

            # Concatenate ACF values for all target lags
            target_acf = ops.concatenate(target_acf_list, axis=-2)

            # Apply absolute value transformation if specified
            if self.use_absolute_acf:
                target_acf = ops.abs(target_acf)

            # Compute regularization loss components
            # L2 penalty: quadratic penalty on ACF values
            l2_loss = ops.mean(ops.square(target_acf))

            # Threshold penalty: additional penalty for significant ACF values
            excess = ops.maximum(ops.abs(target_acf) - self.acf_threshold, 0.0)
            threshold_penalty = ops.mean(ops.square(excess))

            # Combined regularization loss
            reg_loss = self.regularization_weight * (l2_loss + threshold_penalty)

            # Add regularization loss to the layer's losses
            # DECISION plan-2026-08-19T163559-499b6f0e/D-084: no logging on the
            # forward path (R-033/R-041). The removed line interpolated the
            # SYMBOLIC `reg_loss` tensor into an f-string, so under
            # `tf.function` it printed a graph node once and nothing thereafter.
            # `add_loss` already surfaces this value to the training loop.
            self.add_loss(reg_loss)

        # Return predictions unchanged (pass-through design)
        return predictions

    def get_acf_summary(self) -> Optional[Dict[str, float]]:
        """Summarize the ACF from the most recent call.

        Reports the mean and max absolute autocorrelation over lags 1 to
        ``max_lag``, how many entries exceed ``acf_threshold``, and the value at
        each of the first five ``target_lags``.

        .. note::
            This reads ``acf_values``, which only a preceding eager ``call()``
            fills in and which is not serialized. Before any forward pass, or
            after ``keras.models.load_model(...)``, it returns ``None``.

        :return: Dictionary with ACF statistics, or ``None`` if no call has
            populated ``acf_values`` yet.
        :rtype: dict[str, float] | None
        """
        if self.acf_values is None:
            return None

        # Convert to NumPy for statistical computations
        acf_np = ops.convert_to_numpy(self.acf_values)

        # Drop lag 0, which is always 1.0. What remains has shape
        # (..., max_lag, features).
        acf_lags = acf_np[..., 1:, :]

        # Compute summary statistics
        summary = {
            "mean_abs_acf": float(np.mean(np.abs(acf_lags))),
            "max_abs_acf": float(np.max(np.abs(acf_lags))),
            "significant_lags": int(np.sum(np.abs(acf_lags) > self.acf_threshold)),
        }

        # Add specific lag values for the first few target lags
        for i, lag in enumerate(self.target_lags[:5]):
            if 1 <= lag <= self.max_lag:
                # Average ACF value across batch and features for this lag
                lag_value = float(np.mean(acf_np[..., lag, :]))
                summary[f"acf_lag_{lag}"] = lag_value

        logger.debug(f"ACF Summary: {summary}")
        return summary

    def compute_output_shape(
        self,
        input_shape: Union[List[Tuple], Tuple[Tuple]]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        The layer passes the predictions through, so the output shape is the
        first of the two input shapes.

        :param input_shape: Pair of shapes for ``(predictions, targets)``.
        :type input_shape: list[tuple] | tuple[tuple]
        :return: Output shape, identical to the predictions shape.
        :rtype: tuple[int | None, ...]
        """
        # Return the shape of predictions (first input) - pass-through behavior
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "max_lag": self.max_lag,
            "regularization_weight": self.regularization_weight,
            "target_lags": self.target_lags,
            "acf_threshold": self.acf_threshold,
            "use_absolute_acf": self.use_absolute_acf,
            "epsilon": self.epsilon,
        })
        return config


# ---------------------------------------------------------------------

class ACFMonitorCallback(keras.callbacks.Callback):
    """
    Logs the ACF statistics of a `ResidualACFLayer` while training runs.

    Every ``log_frequency`` training batches, the callback looks up the layer
    named ``layer_name`` in the model, asks it for ``get_acf_summary()`` and
    writes the result to the logger. Nothing is stored and nothing is returned.

    The whole body runs inside a ``try``. If the layer is missing, is not a
    `ResidualACFLayer`, or has no ACF yet, the callback logs a warning and
    training continues.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │ on_train_batch_end(batch, logs)      │
        │ batch_count += 1                     │
        └──────────────────┬───────────────────┘
                           ▼
                batch_count % log_frequency
              ┌────────────┴────────────┐
           not 0                        0
              ▼                         ▼
        ┌────────────┐  ┌──────────────────────────┐
        │ return     │  │ model.get_layer(         │
        │ (no work)  │  │     layer_name)          │
        └────────────┘  └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │ isinstance               │
                        │ ResidualACFLayer ?       │
                        └────────────┬─────────────┘
                                     ▼ yes
                        ┌──────────────────────────┐
                        │ get_acf_summary()        │
                        └────────────┬─────────────┘
                                     ▼ not None
                        ┌──────────────────────────┐
                        │ logger.info per statistic│
                        │ and per acf_lag_* entry  │
                        └──────────────────────────┘

    Any exception raised on the right-hand path is caught and logged as a
    warning instead of propagating.

    :param layer_name: Name of the `ResidualACFLayer` to monitor. Must match the
        layer's ``name`` in the model.
    :type layer_name: str
    :param log_frequency: How many training batches between logs.
        Defaults to 100.
    :type log_frequency: int

    :ivar batch_count: Training batches seen since construction.
    :vartype batch_count: int

    Example:
        >>> model.fit(x, y, callbacks=[ACFMonitorCallback("residual_acf", 50)])
    """

    def __init__(
        self,
        layer_name: str,
        log_frequency: int = 100
    ) -> None:
        """Initialize the ACF monitor callback.

        See the class docstring for the parameters. The batch counter starts
        at 0 and is never reset between epochs.

        :raises ValueError: If ``layer_name`` is not a non-empty string, or if
            ``log_frequency`` is less than 1.
        """
        super().__init__()

        # Validated here rather than left to the first batch. log_frequency=0
        # used to construct fine and then die with ZeroDivisionError inside
        # on_train_batch_end, in the middle of a training run.
        if not isinstance(layer_name, str) or not layer_name.strip():
            raise ValueError(
                f"layer_name must be a non-empty string naming the "
                f"ResidualACFLayer to monitor, got {layer_name!r}"
            )
        if not isinstance(log_frequency, int) or log_frequency < 1:
            raise ValueError(
                f"log_frequency must be an integer >= 1, got {log_frequency!r}"
            )

        self.layer_name = layer_name
        self.log_frequency = log_frequency
        self.batch_count = 0

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Log ACF statistics every ``log_frequency`` batches.

        Does nothing on the other batches. Never raises: a failure to reach the
        layer or its summary is logged as a warning.

        :param batch: Batch index reported by Keras. Unused; the callback counts
            batches itself.
        :type batch: int
        :param logs: Training metrics dictionary. Unused.
        :type logs: dict | None
        :return: Nothing.
        :rtype: None
        """
        self.batch_count += 1

        if self.batch_count % self.log_frequency == 0:
            try:
                # Retrieve the ACF layer from the model
                acf_layer = self.model.get_layer(self.layer_name)

                if isinstance(acf_layer, ResidualACFLayer):
                    summary = acf_layer.get_acf_summary()

                    if summary is not None:
                        logger.info(f"Batch {self.batch_count} - ACF Statistics:")
                        logger.info(f"  Mean |ACF|: {summary['mean_abs_acf']:.4f}")
                        logger.info(f"  Max |ACF|: {summary['max_abs_acf']:.4f}")
                        logger.info(f"  Significant lags: {summary['significant_lags']}")

                        # Log individual lag values
                        for key, value in summary.items():
                            if key.startswith("acf_lag_"):
                                logger.info(f"  {key}: {value:.4f}")

            except Exception as e:
                logger.warning(f"Could not monitor ACF layer '{self.layer_name}': {e}")


# ---------------------------------------------------------------------
