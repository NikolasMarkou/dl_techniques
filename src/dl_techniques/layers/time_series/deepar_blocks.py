"""
Building blocks for the DeepAR probabilistic forecaster.

Four layers live here. ``ScaleLayer`` divides inputs by a per-item scale and
multiplies outputs back up. ``GaussianLikelihoodHead`` and
``NegativeBinomialLikelihoodHead`` turn LSTM hidden states into the parameters
of a predictive distribution. ``DeepARCell`` wraps a Keras ``LSTMCell`` so it
can be driven step by step.

Both likelihood heads share the module-level constant ``MIN_LIKELIHOOD_PARAM``,
the floor applied to every parameter that has to stay strictly positive.

Reference:
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Salinas et al., 2019
    https://arxiv.org/abs/1704.04110
"""

import keras
from keras import ops, layers
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# Floor for every likelihood parameter that must stay strictly positive: the
# Gaussian sigma, and the negative-binomial mu and alpha. Softplus underflows
# to exactly 0.0 in float32 for logits below about -103. At 0 every downstream
# use is an inf or a NaN: log(sigma), division by sigma, 1/alpha, and
# lgamma(1/alpha). 1e-6 is small enough not to distort a trained model and
# large enough to survive float16.
MIN_LIKELIHOOD_PARAM: float = 1e-6

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.deepar_blocks")
class ScaleLayer(keras.layers.Layer):
    """
    Divides inputs by an item-dependent scale, or multiplies outputs by it.

    DeepAR trains on series whose magnitudes differ by orders of magnitude.
    This layer is how one model copes with all of them: autoregressive inputs
    are divided by a per-item scale, and the predicted likelihood parameters
    are multiplied back up.

    The scale is

        ``nu = mean(conditioning_range) + epsilon``

    Forward scaling divides by ``nu``. Inverse scaling multiplies by whatever
    ``scale`` the caller passes. The layer applies exactly one division or one
    multiplication and holds no per-parameter policy, so the call site picks
    the scale:

    - Gaussian ``mu`` **and** ``sigma``: pass ``nu``. The forward path divides
      ``z`` by ``nu`` exactly once, so ``mu = nu * mu~`` and
      ``sigma = nu * sigma~``. Here ``sigma`` scales like a first moment, not
      like a variance.
    - Negative-binomial shape ``alpha``: pass ``1 / sqrt(nu)``.

    **Architecture Overview:**

    .. code-block:: text

        Input: x [B, T, F]
        Optional input: scale [B, 1, F], passed by the caller
                    │
                    ▼
        ┌──────────────────────────────────┐
        │ if scale is None and             │
        │    scale_per_sample:             │
        │      scale = mean(x, axis=1)+eps │
        └───────────────────┬──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │  forward  │ │  inverse  │ │  no scale │
        │ x / scale │ │ x * scale │ │  x as-is  │
        └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
              ▼             ▼             ▼
           Scaled       Descaled     Passthrough

    The third leaf is reached when ``scale`` is None and ``scale_per_sample``
    is False. The layer then returns its input untouched.

    Note:
        Do not restore ``sqrt(nu)`` for the Gaussian ``sigma``. This docstring
        claimed that until 2026-08-15. No call site ever did it, and the
        diagram above (``x * scale``) always contradicted it.

    :param scale_per_sample: If True, compute the scale from the input when the
        caller passes none. If False, use only the provided scale. Defaults to
        True.
    :type scale_per_sample: bool
    :param epsilon: Constant added to the computed mean. Keeps the scale away
        from zero for a near-empty series. Defaults to 1.0.
    :type epsilon: float
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            scale_per_sample: bool = True,
            epsilon: float = 1.0,
            **kwargs: Any
    ) -> None:
        """
        Initialize the ScaleLayer.

        :param scale_per_sample: If True, compute the scale per sample when the
            caller passes none.
        :type scale_per_sample: bool
        :param epsilon: Constant added to the computed mean.
        :type epsilon: float
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.scale_per_sample = scale_per_sample
        self.epsilon = epsilon

    def call(
            self,
            inputs: keras.KerasTensor,
            scale: Optional[keras.KerasTensor] = None,
            inverse: bool = False
    ) -> keras.KerasTensor:
        """
        Apply scaling or inverse scaling.

        With no usable scale the input is returned unchanged. That happens when
        ``scale`` is None and ``scale_per_sample`` is False.

        :param inputs: Tensor to scale.
        :type inputs: keras.KerasTensor
        :param scale: Pre-computed scale. If None and ``scale_per_sample`` is
            True, it is computed from ``inputs``.
        :type scale: keras.KerasTensor or None
        :param inverse: If True, multiply by the scale. If False, divide by it.
        :type inverse: bool
        :return: The scaled, inverse-scaled or unchanged tensor.
        :rtype: keras.KerasTensor
        """
        if scale is None and self.scale_per_sample:
            # Mean over the time axis, plus epsilon.
            scale = ops.mean(inputs, axis=1, keepdims=True) + self.epsilon

        if scale is None:
            return inputs

        if inverse:
            return inputs * scale
        else:
            return inputs / scale

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the input shape.

        Scaling is elementwise, so no dimension changes.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        :return: The same shape.
        :rtype: tuple[int or None, ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'scale_per_sample': self.scale_per_sample,
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.time_series.deepar_blocks")
class GaussianLikelihoodHead(keras.layers.Layer):
    """
    Projects hidden states to Gaussian parameters: a mean and a std deviation.

    Two independent Dense layers read the same hidden state. One produces
    ``mu``, the distribution mean, with no activation, so it may be negative.
    The other produces logits for ``sigma``, the standard deviation, which
    softplus and a floor keep strictly positive.

    The operations are:
        ``mu(h) = W_mu^T h + b_mu``
        ``sigma(h) = max(softplus(W_sigma^T h + b_sigma), 1e-6)``

    ``sigma`` is a standard deviation, not a variance. The negative
    log-likelihood that consumes it takes ``log(sigma)`` and divides the
    residual by ``sigma``.

    **Architecture Overview:**

    .. code-block:: text

        Input: h_t [B, (T,) hidden_dim]
                    │
                    ├─────────────────────┐
                    ▼                     ▼
            ┌──────────────┐     ┌──────────────────┐
            │ Dense(units) │     │ Dense(units)     │
            │ (no activ.)  │     │ (sigma logits)   │
            └───────┬──────┘     └────────┬─────────┘
                    │                     ▼
                    │            ┌──────────────────┐
                    │            │ softplus, then   │
                    │            │ max(., 1e-6)     │
                    │            └────────┬─────────┘
                    ▼                     ▼
              mu [B, (T,) units]   sigma > 0, same shape
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    Output: (mu, sigma) tuple

    :param units: Width of both outputs. Use 1 for a univariate series.
    :type units: int
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            units: int = 1,
            **kwargs: Any
    ) -> None:
        """
        Initialize the GaussianLikelihoodHead.

        :param units: Width of both outputs.
        :type units: int
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units

        # One Dense per distribution parameter.
        self.mu_projection = layers.Dense(
            units,
            name='mu_projection'
        )
        self.sigma_projection = layers.Dense(
            units,
            name='sigma_projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build both projections explicitly, then the layer itself.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        """
        self.mu_projection.build(input_shape)
        self.sigma_projection.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute the Gaussian mean and standard deviation from hidden states.

        :param inputs: Hidden states from the LSTM.
        :type inputs: keras.KerasTensor
        :return: Tuple of (mu, sigma) tensors, both of width ``units``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        mu = self.mu_projection(inputs)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-037: keep `ops.softplus`
        # and keep the `maximum` floor. Do NOT respell softplus as
        # `ops.log(1.0 + ops.exp(x))`: a float32 logit above ~88 makes `exp`
        # inf, so sigma is inf and the loss NaN. The floor is a second guard --
        # softplus itself underflows to 0.0 below about -103. See D-037.
        sigma_logits = self.sigma_projection(inputs)
        sigma = ops.maximum(ops.softplus(sigma_logits), MIN_LIKELIHOOD_PARAM)

        return mu, sigma

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Return one shape per output. Both are the input with ``units`` last.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        :return: The shapes of mu and sigma, which are equal.
        :rtype: tuple[tuple[int or None, ...], tuple[int or None, ...]]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)
        return output_shape, output_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.time_series.deepar_blocks")
class NegativeBinomialLikelihoodHead(keras.layers.Layer):
    """
    Projects hidden states to negative-binomial parameters: a mean and a
    dispersion.

    Use this head for count data that is overdispersed, meaning its variance
    exceeds its mean. Two Dense layers read the same hidden state. Both outputs
    have to be positive, so both go through softplus and the same floor.

    The operations are:
        ``mu(h) = max(softplus(W_mu^T h + b_mu), 1e-6)``
        ``alpha(h) = max(softplus(W_alpha^T h + b_alpha), 1e-6)``

    ``mu`` is the mean and ``alpha`` is the dispersion, not a shape count. The
    distribution is:
        ``E[z] = mu``
        ``Var[z] = mu + mu^2 * alpha``

    So ``alpha`` at 0 would be a Poisson. The likelihood that consumes these
    uses ``r = 1 / alpha`` as the shape, which is why ``alpha`` must never
    reach 0.

    **Architecture Overview:**

    .. code-block:: text

        Input: h_t [B, (T,) hidden_dim]
                    │
                    ├─────────────────────┐
                    ▼                     ▼
            ┌──────────────┐     ┌──────────────────┐
            │ Dense(units) │     │ Dense(units)     │
            │ (mu logits)  │     │ (alpha logits)   │
            └───────┬──────┘     └────────┬─────────┘
                    ▼                     ▼
            ┌──────────────┐     ┌──────────────────┐
            │ softplus,    │     │ softplus,        │
            │ max(., 1e-6) │     │ max(., 1e-6)     │
            └───────┬──────┘     └────────┬─────────┘
                    ▼                     ▼
             mu > 0 (mean)        alpha > 0 (dispersion)
             [B, (T,) units]      same shape
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    Output: (mu, alpha) tuple

    :param units: Width of both outputs. Use 1 for a univariate series.
    :type units: int
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            units: int = 1,
            **kwargs: Any
    ) -> None:
        """
        Initialize the NegativeBinomialLikelihoodHead.

        :param units: Width of both outputs.
        :type units: int
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units

        # One Dense per distribution parameter.
        self.mu_projection = layers.Dense(
            units,
            name='mu_projection'
        )
        self.alpha_projection = layers.Dense(
            units,
            name='alpha_projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build both projections explicitly, then the layer itself.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        """
        self.mu_projection.build(input_shape)
        self.alpha_projection.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute the negative-binomial mean and dispersion from hidden states.

        :param inputs: Hidden states from the LSTM.
        :type inputs: keras.KerasTensor
        :return: Tuple of (mu, alpha) tensors, both of width ``units``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Softplus plus the floor, for the reason recorded at the anchor in
        # GaussianLikelihoodHead.call. `alpha` matters even more than `sigma`
        # does there: the negative-binomial loss divides by it and takes
        # `lgamma(1 / alpha)`, so an `alpha` of exactly 0 is an immediate inf.
        mu_logits = self.mu_projection(inputs)
        mu = ops.maximum(ops.softplus(mu_logits), MIN_LIKELIHOOD_PARAM)

        alpha_logits = self.alpha_projection(inputs)
        alpha = ops.maximum(ops.softplus(alpha_logits), MIN_LIKELIHOOD_PARAM)

        return mu, alpha

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Return one shape per output. Both are the input with ``units`` last.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        :return: The shapes of mu and alpha, which are equal.
        :rtype: tuple[tuple[int or None, ...], tuple[int or None, ...]]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)
        return output_shape, output_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.deepar_blocks")
class DeepARCell(keras.layers.Layer):
    """
    One autoregressive DeepAR step, as a thin wrapper over a Keras LSTMCell.

    ``call`` forwards its input and states straight to the LSTM cell and
    returns what the cell returns. The recurrence DeepAR describes is

        ``h_t = h(h_{t-1}, z_{t-1}, x_t, Theta)``

    but this cell does not build ``[z_{t-1}, x_t]`` itself. The caller
    concatenates the previous observation with the covariates and passes the
    result as one tensor. Drive the cell with ``keras.layers.RNN``, or step it
    by hand for autoregressive sampling.

    **Architecture Overview:**

    .. code-block:: text

        Input: x_t [B, input_dim]   States: (h_{t-1}, c_{t-1})
                    │                        │
                    └───────────┬────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ keras LSTMCell(units)         │
                │ dropout, recurrent_dropout    │
                └───────────────┬───────────────┘
                                ▼
                Output: h_t [B, units]
                States: (h_t, c_t), each [B, units]

    Nothing else happens in ``call``. There is no concatenation, no projection
    and no scaling stage inside this cell.

    :param units: Number of LSTM units. This is also the hidden width.
    :type units: int
    :param dropout_rate: Dropout on the LSTM inputs. Defaults to 0.0.
    :type dropout_rate: float
    :param recurrent_dropout_rate: Dropout on the recurrent connections.
        Defaults to 0.0.
    :type recurrent_dropout_rate: float
    :param kwargs: Additional arguments for the Layer base class.

    :ivar state_size: Width of one state, so ``units``. The cell itself carries
        two states of that width, and ``get_initial_state`` returns both.
    :vartype state_size: int
    """

    def __init__(
            self,
            units: int,
            dropout_rate: float = 0.0,
            recurrent_dropout_rate: float = 0.0,
            **kwargs: Any
    ) -> None:
        """
        Initialize the DeepARCell.

        :param units: Number of LSTM units.
        :type units: int
        :param dropout_rate: Dropout on the LSTM inputs.
        :type dropout_rate: float
        :param recurrent_dropout_rate: Dropout on the recurrent connections.
        :type recurrent_dropout_rate: float
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.state_size = units

        # The whole computation of this cell.
        self.lstm_cell = layers.LSTMCell(
            units,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name='lstm_cell'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the LSTM sub-layer explicitly, then the cell itself.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        """
        self.lstm_cell.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            states: Tuple[keras.KerasTensor, ...],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """
        Process one time step through the LSTM cell.

        :param inputs: Input at time t, shape ``(batch_size, input_dim)``. The
            caller has already concatenated the previous observation with the
            covariates.
        :type inputs: keras.KerasTensor
        :param states: State tensors from the previous step, ``(h, c)``.
        :type states: tuple[keras.KerasTensor, ...]
        :param training: Whether in training mode. Both dropout rates are
            applied only when this is True.
        :type training: bool or None
        :return: Tuple of (output, new_states), exactly as the LSTM cell
            returns them.
        :rtype: tuple[keras.KerasTensor, tuple[keras.KerasTensor, ...]]
        """
        output, new_states = self.lstm_cell(inputs, states, training=training)
        return output, new_states

    def get_initial_state(
            self,
            batch_size: Optional[int] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Get the initial state, delegated to the LSTM cell.

        Two tensors come back, the hidden state and the carry state, each of
        shape ``(batch_size, units)``.

        :param batch_size: Batch size for the initial state tensors.
        :type batch_size: int or None
        :return: Tuple of initial hidden and cell state tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        return self.lstm_cell.get_initial_state(batch_size=batch_size)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the per-time-step output shape of the cell.

        :param input_shape: Shape of the per-step input ``(batch_size, input_dim)``.
        :type input_shape: tuple[int or None, ...]
        :return: Per-step output shape ``(batch_size, units)``.
        :rtype: tuple[int or None, ...]
        """
        return (input_shape[0], self.units)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate,
        })
        return config

# ---------------------------------------------------------------------
