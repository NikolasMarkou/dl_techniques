"""
Likelihood heads for the DeepAR probabilistic forecaster.

``GaussianLikelihoodHead`` and ``NegativeBinomialLikelihoodHead`` turn LSTM
hidden states into the parameters of a predictive distribution.

Both subclass the private ``_LikelihoodHead``, which owns the two projections,
and share ``_floored_softplus`` with its module-level floor
``MIN_LIKELIHOOD_PARAM``, applied to every parameter that has to stay strictly
positive.

Reference:
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Salinas et al., 2019
    https://arxiv.org/abs/1704.04110
"""

import keras
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


def _floored_softplus(logits: keras.KerasTensor) -> keras.KerasTensor:
    """
    Map logits to a strictly positive likelihood parameter.

    :param logits: Unconstrained projection output.
    :type logits: keras.KerasTensor
    :return: ``max(softplus(logits), MIN_LIKELIHOOD_PARAM)``.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-037: keep `keras.ops.softplus`
    # and keep the `maximum` floor. Do NOT respell softplus as
    # `keras.ops.log(1.0 + keras.ops.exp(x))`: a float32 logit above ~88 makes `exp`
    # inf, so the parameter is inf and the loss NaN. The floor is a second
    # guard -- softplus itself underflows to 0.0 below about -103. See D-037.
    return keras.ops.maximum(keras.ops.softplus(logits), MIN_LIKELIHOOD_PARAM)


class _LikelihoodHead(keras.layers.Layer):
    """
    Shared body of the two-parameter DeepAR likelihood heads.

    Two independent Dense layers read the same hidden state. ``mu_projection``
    produces the mean. A second projection, named after the parameter it
    produces, gives a strictly positive second parameter through
    :func:`_floored_softplus`. Subclasses set the two class attributes below
    and inherit everything else. The class is not registered, because it is
    never serialized on its own.

    :cvar SECOND_PARAMETER: Name of the second parameter. The second
        projection is stored as ``<SECOND_PARAMETER>_projection`` and carries
        that layer name.
    :cvar POSITIVE_MEAN: If True, ``mu`` also goes through
        :func:`_floored_softplus`. If False, ``mu`` is the raw projection.
    :param units: Width of both outputs. Use 1 for a univariate series.
    :type units: int
    :param kwargs: Additional arguments for the Layer base class.
    """

    SECOND_PARAMETER: str
    POSITIVE_MEAN: bool

    def __init__(
            self,
            units: int = 1,
            **kwargs: Any
    ) -> None:
        """
        Create one Dense projection per distribution parameter.

        :param units: Width of both outputs.
        :type units: int
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units

        second_name = f'{self.SECOND_PARAMETER}_projection'
        self.mu_projection = keras.layers.Dense(units, name='mu_projection')
        setattr(self, second_name, keras.layers.Dense(units, name=second_name))

    @property
    def _second_projection(self) -> keras.layers.Dense:
        """The projection that produces the second parameter."""
        return getattr(self, f'{self.SECOND_PARAMETER}_projection')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build both projections explicitly, then the layer itself.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        """
        self.mu_projection.build(input_shape)
        self._second_projection.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute the mean and the positive second parameter.

        :param inputs: Hidden states from the LSTM.
        :type inputs: keras.KerasTensor
        :return: Tuple of (mu, second parameter), both of width ``units``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        mu = self.mu_projection(inputs)
        if self.POSITIVE_MEAN:
            mu = _floored_softplus(mu)
        second = _floored_softplus(self._second_projection(inputs))
        return mu, second

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Return one shape per output. Both are the input with ``units`` last.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        :return: The two output shapes, which are equal.
        :rtype: tuple[tuple[int or None, ...], tuple[int or None, ...]]
        """
        output_shape = (*input_shape[:-1], self.units)
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


@register_dl_technique("dl_techniques.layers.heads.time_series.likelihood_heads")
class GaussianLikelihoodHead(_LikelihoodHead):
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
    :ivar mu_projection: Dense layer producing ``mu``.
    :vartype mu_projection: keras.layers.Dense
    :ivar sigma_projection: Dense layer producing the ``sigma`` logits.
    :vartype sigma_projection: keras.layers.Dense
    """

    SECOND_PARAMETER = 'sigma'
    POSITIVE_MEAN = False

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.heads.time_series.likelihood_heads")
class NegativeBinomialLikelihoodHead(_LikelihoodHead):
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
    uses ``r = 1 / alpha`` as the shape, and takes ``lgamma(1 / alpha)``, which
    is why ``alpha`` must never reach 0: at exactly 0 the loss is an immediate
    inf.

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
    :ivar mu_projection: Dense layer producing the ``mu`` logits.
    :vartype mu_projection: keras.layers.Dense
    :ivar alpha_projection: Dense layer producing the ``alpha`` logits.
    :vartype alpha_projection: keras.layers.Dense
    """

    SECOND_PARAMETER = 'alpha'
    POSITIVE_MEAN = True

