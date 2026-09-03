"""
Autoregressive recurrent forecaster. ``DeepAR`` emits a likelihood, Gaussian
or negative binomial, over each horizon step instead of a point prediction.

The model factorizes the joint distribution over the horizon into one-step
conditionals, `p(z_{t0:T} | z_{1:t0-1}, x) = prod_t p(z_t | theta(h_t))` with
`h_t = LSTM([z_{t-1} / nu, x_t], h_{t-1})`, and trains the head to output
distribution parameters `theta` under negative log-likelihood. Training is
teacher-forced and runs as one symbolic forward pass (`call`); inference is
ancestral, drawing `num_samples` independent trajectories one step at a time
and reading quantiles off the empirical percentiles, since there is no closed
form for the multi-step predictive distribution.

Each series is scaled by `nu = mean(z over the conditioning range) +
scale_epsilon` before entering the network, and predictions are scaled back;
this lets one shared network fit series whose magnitudes differ by orders of
magnitude. `conditioning_length` controls which window `nu` is computed over
during training; leaving it unset computes `nu` over the full target window,
which leaks the prediction range into training and disagrees with inference.

The sampling loop replays the whole conditioning window plus every draw made
so far at each horizon step, since a stacked `keras.layers.LSTM` does not
carry state across separate calls here; this costs `num_samples *
prediction_len` full-sequence passes.

References:
    - Salinas et al., 2020. DeepAR: Probabilistic Forecasting with Autoregressive
      Recurrent Networks. (https://arxiv.org/abs/1704.04110)
    - Hochreiter and Schmidhuber, 1997. Long Short-Term Memory. Neural Computation
      9(8): 1735-1780.
    - Bengio et al., 2015. Scheduled Sampling for Sequence Prediction with Recurrent
      Neural Networks. (https://arxiv.org/abs/1506.03099)
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Optional, Union, Literal, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import log_gamma
from dl_techniques.layers.time_series.deepar_blocks import (
    ScaleLayer,
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
)
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.deepar.model")
class DeepAR(keras.Model, ForecastMixin):
    """
    DeepAR: Probabilistic forecasting with autoregressive recurrent networks.

    Learns a global model across many related time series, producing
    calibrated probabilistic forecasts instead of point predictions.

    Architecture:

    .. code-block:: text

        Conditioning range [t=1..t0-1]:
            [z_{t-1}, x_t] -> scale (z/v) -> LSTM encoder -> head -> theta_t
                                                            -> loss -log p(z_t|theta_t)

        Prediction range [t=t0..T]:
            [z_sampled_{t-1}, x_t] -> scale -> LSTM decoder -> head -> theta_t
                                                    -> sample z_t ~ p(z|theta_t*v)
                                                    -> feed z_t back as input

    The encoder and decoder share one LSTM stack. Supports Gaussian
    (real-valued) and negative binomial (count) likelihoods, and produces
    multiple sample paths via ancestral sampling for quantile estimation.

    :param num_layers: Number of LSTM layers.
    :type num_layers: int
    :param hidden_dim: Hidden dimension of the LSTM layers.
    :type hidden_dim: int
    :param dropout_rate: Dropout rate for the LSTM layers.
    :type dropout_rate: float
    :param recurrent_dropout_rate: Recurrent dropout rate for the LSTM layers.
    :type recurrent_dropout_rate: float
    :param likelihood: ``'gaussian'`` for real-valued data or
        ``'negative_binomial'`` for count data.
    :type likelihood: str
    :param target_dim: Dimensionality of the target variable, typically 1 for
        univariate forecasting.
    :type target_dim: int
    :param num_samples: Number of Monte Carlo trajectories drawn during
        prediction. The sampling loop unrolls `num_samples * prediction_len`
        stacked-LSTM passes into one traced graph, so this cost cannot be
        routed around by any path other than lowering it or the horizon;
        quantile precision degrades as `1/sqrt(num_samples)`.
    :type num_samples: int
    :param scale_epsilon: Constant added to the scale computation, keeping it
        away from zero for near-empty series.
    :type scale_epsilon: float
    :param conditioning_length: Number of leading steps used to compute the
        per-series scale during training; set this to the context length. If
        ``None``, the scale is computed over the full target window, which
        leaks the prediction range into it and disagrees with inference,
        where the scale comes from the conditioning range only — a warning
        is emitted in that case. A precomputed ``inputs['scale']`` bypasses
        this entirely.
    :type conditioning_length: Optional[int]
    :param kwargs: Additional arguments for the ``keras.Model`` base class.

    Input shape:
        During training:
        - target: `(batch_size, seq_len, target_dim)` - Target time series
        - covariates: `(batch_size, seq_len, covariate_dim)` - Covariates
        - scale: Optional `(batch_size, 1, target_dim)` - Pre-computed scales

        During prediction:
        - conditioning_target: `(batch_size, conditioning_len, target_dim)`
        - full_covariates: `(batch_size, conditioning_len + prediction_len, covariate_dim)`
        - scale: Optional pre-computed scales

    Output shape:
        Training:
        - Dictionary with 'mu', 'sigma' (Gaussian) or 'mu', 'alpha' (NegBin)
          Each: `(batch_size, seq_len, target_dim)`

        Prediction:
        - Samples: `(num_samples, batch_size, prediction_len, target_dim)`

    Example:
        ```python
        # Training
        model = DeepAR(
            num_layers=3,
            hidden_dim=128,
            likelihood='gaussian',
            dropout_rate=0.1
        )

        model.compile(
            optimizer='adam',
            loss=model.gaussian_loss  # or model.negative_binomial_loss
        )

        # Prepare data
        target = keras.random.normal((32, 100, 1))
        covariates = keras.random.normal((32, 100, 10))

        # Train
        model.fit({'target': target, 'covariates': covariates})

        # Prediction
        conditioning_target = keras.random.normal((32, 50, 1))
        full_covariates = keras.random.normal((32, 100, 10))

        samples = model.predict({
            'conditioning_target': conditioning_target,
            'full_covariates': full_covariates
        })  # Shape: (100, 32, 50, 1)

        # Compute quantiles
        quantiles = np.percentile(samples, [10, 50, 90], axis=0)
        ```

    Note:
        The model uses teacher forcing during training (feeding true values)
        and autoregressive sampling during prediction (feeding sampled values).
        This is standard for sequence-to-sequence models and does not typically
        cause issues in forecasting, unlike in some NLP tasks.
    """

    # Single source of truth for the constructor validation below.
    SUPPORTED_LIKELIHOODS = ('gaussian', 'negative_binomial')

    def __init__(
            self,
            num_layers: int = 3,
            hidden_dim: int = 40,
            dropout_rate: float = 0.0,
            recurrent_dropout_rate: float = 0.0,
            likelihood: Literal['gaussian', 'negative_binomial'] = 'gaussian',
            target_dim: int = 1,
            num_samples: int = 100,
            scale_epsilon: float = 1.0,
            conditioning_length: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate hyperparameters (fail fast on nonsensical geometry).
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if target_dim <= 0:
            raise ValueError(f"target_dim must be > 0, got {target_dim}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if likelihood not in self.SUPPORTED_LIKELIHOODS:
            raise ValueError(
                f"Unknown likelihood: {likelihood}. "
                f"Must be one of {self.SUPPORTED_LIKELIHOODS}"
            )
        if conditioning_length is not None and conditioning_length <= 0:
            raise ValueError(
                f"conditioning_length must be > 0, got {conditioning_length}")

        # Store configuration
        self.conditioning_length = conditioning_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.likelihood = likelihood
        self.target_dim = target_dim
        self.num_samples = num_samples
        self.scale_epsilon = scale_epsilon

        self.scale_layer = ScaleLayer(
            scale_per_sample=False,  # scale is computed externally, in compute_scale
            epsilon=scale_epsilon,
            name='scale_layer'
        )

        # A plain Python list attribute is tracked by Keras 3, verified empirically.
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.LSTM(
                hidden_dim,
                return_sequences=True,
                return_state=False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                name=f'lstm_{i}'
            )
            self.lstm_layers.append(lstm)

        if likelihood == 'gaussian':
            self.likelihood_head = GaussianLikelihoodHead(
                units=target_dim,
                name='gaussian_head'
            )
        else:  # negative_binomial
            self.likelihood_head = NegativeBinomialLikelihoodHead(
                units=target_dim,
                name='negbin_head'
            )

    def build(self, input_shape: Any) -> None:
        """Build the model and its sublayers explicitly.

        DeepAR takes a dict input, ``{'target': (B, T, target_dim),
        'covariates': (B, T, covariate_dim), ...}``. On ``load_model``, Keras
        calls ``build`` and then restores weights, so the sublayers must
        already exist to receive them.

        # DECISION plan_2026-06-11_fe7401f4/D-002: build sublayers here, not
        # lazily in call(), or load_model restores weights into unbuilt layers.

        :param input_shape: A dict with ``'target'`` and ``'covariates'``
            shapes, or a non-dict/None shape (e.g. a bare ``model.build(None)``),
            in which case sublayer builds are deferred to the first ``call``.
        :type input_shape: Any
        """
        if isinstance(input_shape, dict) and 'covariates' in input_shape:
            covariate_dim = input_shape['covariates'][-1]
            seq_len = input_shape['target'][1]

            lstm_input_shape = (None, seq_len, self.target_dim + covariate_dim)
            for lstm in self.lstm_layers:
                lstm.build(lstm_input_shape)
                # Subsequent layers consume the prior LSTM's hidden sequence.
                lstm_input_shape = (None, seq_len, self.hidden_dim)

            self.likelihood_head.build((None, seq_len, self.hidden_dim))

        super().build(input_shape)

    def compute_scale(
            self,
            target: keras.KerasTensor,
            conditioning_length: Optional[int] = None
    ) -> keras.KerasTensor:
        """Compute the per-series scale as the mean over a window, plus ``scale_epsilon``.

        :param target: Target time series.
        :type target: keras.KerasTensor, shape (batch, seq_len, target_dim)
        :param conditioning_length: If given, only the first N steps are used.
        :type conditioning_length: Optional[int]
        :return: Scale tensor.
        :rtype: keras.KerasTensor, shape (batch, 1, target_dim)
        """
        if conditioning_length is not None:
            target_for_scale = target[:, :conditioning_length, :]
        else:
            target_for_scale = target

        scale = ops.mean(target_for_scale, axis=1, keepdims=True) + self.scale_epsilon
        return scale

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Run the training-mode, teacher-forced path and return the likelihood parameters.

        The Monte Carlo sampling path is not routed through ``call``: it
        lives in :meth:`predict_step`, which invokes :meth:`_prediction_mode`
        directly, keeping ``call`` a single symbolic, dict-returning function.

        :param inputs: A dict with ``'target'`` (batch, seq_len, target_dim),
            ``'covariates'`` (batch, seq_len, covariate_dim), and optionally
            ``'scale'`` (batch, 1, target_dim).
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: ``{'mu', 'sigma', 'target'}`` for Gaussian, or
            ``{'mu', 'alpha', 'target'}`` for negative binomial.
        :rtype: Dict[str, keras.KerasTensor]
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "Inputs must be a dictionary with 'target' and 'covariates' keys"
            )
        return self._training_mode(inputs, training=training)

    def _training_mode(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Run the teacher-forced training pass and return the likelihood parameters.

        :param inputs: A dict with ``'target'``, ``'covariates'`` and
            optionally ``'scale'``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: Likelihood parameters for each time step.
        :rtype: Dict[str, keras.KerasTensor]
        """
        target = inputs['target']  # (batch, seq_len, target_dim)
        covariates = inputs['covariates']  # (batch, seq_len, covariate_dim)
        scale = inputs.get('scale', None)

        # conditioning_length must be honoured here, or the scale is a function
        # of the steps being predicted and disagrees with _prediction_mode.
        if scale is None:
            if self.conditioning_length is None:
                logger.warning(
                    "DeepAR: computing the scale over the FULL target window "
                    "because conditioning_length is None. This leaks the "
                    "prediction range into the scale and does not match "
                    "_prediction_mode, which scales from the conditioning range "
                    "only. Pass conditioning_length=<context steps> to the "
                    "constructor, or supply a precomputed inputs['scale']."
                )
            scale = self.compute_scale(
                target, conditioning_length=self.conditioning_length)

        target_scaled = self.scale_layer(target, scale=scale, inverse=False)

        batch_size = ops.shape(target)[0]
        seq_len = ops.shape(target)[1]
        target_dim = ops.shape(target)[2]

        zeros = ops.zeros((batch_size, 1, target_dim))
        lagged_target = ops.concatenate([zeros, target_scaled[:, :-1, :]], axis=1)

        inputs_combined = ops.concatenate([lagged_target, covariates], axis=-1)

        hidden = inputs_combined
        for lstm in self.lstm_layers:
            hidden = lstm(hidden, training=training)

        if self.likelihood == 'gaussian':
            mu_scaled, sigma_scaled = self.likelihood_head(hidden)

            # sigma uses `scale`, not sqrt(scale): target_scaled divides by nu
            # exactly once, so mu = nu*mu~ and sigma = nu*sigma~ (Salinas et al.
            # section 3.2). The sqrt belongs only to the negative-binomial branch.
            mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
            sigma = self.scale_layer(sigma_scaled, scale=scale, inverse=True)

            return {'mu': mu, 'sigma': sigma, 'target': target}

        else:  # negative_binomial
            mu_scaled, alpha_scaled = self.likelihood_head(hidden)

            mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
            alpha = self.scale_layer(
                alpha_scaled,
                scale=ops.reciprocal(ops.sqrt(scale)),
                inverse=True
            )

            return {'mu': mu, 'alpha': alpha, 'target': target}

    def _prediction_mode(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run ancestral sampling over the prediction horizon.

        :param inputs: A dict with ``'conditioning_target'``
            (batch, cond_len, target_dim), ``'full_covariates'``
            (batch, total_len, covariate_dim), and optionally ``'scale'``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Whether the call is in training mode; typically
            ``False`` for prediction.
        :type training: Optional[bool]
        :return: Sampled trajectories.
        :rtype: keras.KerasTensor, shape (num_samples, batch, pred_len, target_dim)
        """
        conditioning_target = inputs['conditioning_target']
        full_covariates = inputs['full_covariates']
        scale = inputs.get('scale', None)

        batch_size = ops.shape(conditioning_target)[0]
        conditioning_len = ops.shape(conditioning_target)[1]
        total_len = ops.shape(full_covariates)[1]
        prediction_len = total_len - conditioning_len

        if scale is None:
            scale = self.compute_scale(conditioning_target)

        conditioning_scaled = self.scale_layer(
            conditioning_target,
            scale=scale,
            inverse=False
        )

        conditioning_covariates = full_covariates[:, :conditioning_len, :]
        prediction_covariates = full_covariates[:, conditioning_len:, :]

        zeros = ops.zeros((batch_size, 1, self.target_dim))
        lagged_conditioning = ops.concatenate(
            [zeros, conditioning_scaled[:, :-1, :]],
            axis=1
        )

        encoder_inputs = ops.concatenate(
            [lagged_conditioning, conditioning_covariates],
            axis=-1
        )

        all_samples = []

        for sample_idx in range(self.num_samples):
            current_value = conditioning_scaled[:, -1:, :]  # (batch, 1, target_dim)

            sample_trajectory = []
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-038: keep every drawn
            # step, not just the most recent, or an H-step forecast degenerates
            # into H nearly independent one-step forecasts. See decisions.md.
            decoder_history = []

            for t in range(prediction_len):
                current_covariates = prediction_covariates[:, t:t + 1, :]

                decoder_input = ops.concatenate(
                    [current_value, current_covariates],
                    axis=-1
                )
                decoder_history.append(decoder_input)

                # A stacked LSTM does not carry state across separate calls
                # here, so the prefix is replayed rather than resumed.
                decoder_input_seq = ops.concatenate(
                    [encoder_inputs] + decoder_history,
                    axis=1
                )

                hidden_t = decoder_input_seq
                for lstm in self.lstm_layers:
                    hidden_t = lstm(hidden_t, training=training)

                hidden_t_current = hidden_t[:, -1:, :]

                if self.likelihood == 'gaussian':
                    mu_scaled, sigma_scaled = self.likelihood_head(hidden_t_current)

                    # sigma uses `scale`, not sqrt(scale) — must agree with
                    # _training_mode, or sampling would not match the trained
                    # likelihood.
                    mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
                    sigma = self.scale_layer(sigma_scaled, scale=scale, inverse=True)

                    epsilon = keras.random.normal(ops.shape(mu))
                    sampled_value = mu + sigma * epsilon

                else:  # negative_binomial
                    mu_scaled, alpha_scaled = self.likelihood_head(hidden_t_current)

                    mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
                    alpha = self.scale_layer(
                        alpha_scaled,
                        scale=ops.reciprocal(ops.sqrt(scale)),
                        inverse=True
                    )

                    # Approximated by moment-matching to a Gaussian clipped at
                    # zero, not a Gamma-Poisson draw.
                    variance = mu + ops.square(mu) * alpha
                    std = ops.sqrt(variance)
                    epsilon = keras.random.normal(ops.shape(mu))
                    sampled_value = mu + std * epsilon
                    sampled_value = ops.maximum(sampled_value, 0.0)

                sample_trajectory.append(sampled_value)

                current_value = self.scale_layer(
                    sampled_value,
                    scale=scale,
                    inverse=False
                )

            # Stack trajectory
            trajectory = ops.concatenate(sample_trajectory, axis=1)  # (batch, pred_len, target_dim)
            all_samples.append(trajectory)

        # Stack all samples: (num_samples, batch, pred_len, target_dim)
        samples = ops.stack(all_samples, axis=0)

        return samples

    def predict_step(self, data):
        """Run the sampling path instead of the training-mode ``call``.

        Returns ``(S, B, H, D)`` with axis 0 as ``num_samples``, not batch;
        :meth:`_forecast` depends on this axis order.
        """
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        return self._prediction_mode(x, training=False)

    def _forecast(
            self,
            x: Dict[str, "keras.KerasTensor"],
            quantile_levels: Optional[list] = None,
            **kwargs: Any
    ) -> Forecast:
        """Run Monte Carlo sampling and reduce it into a unified :class:`Forecast`.

        This is the ``ForecastMixin`` hook for DeepAR. It runs the
        autoregressive sampling path and reduces the resulting trajectories
        to an empirical-mean point forecast plus empirical-percentile
        quantiles. DeepAR is the only model on the contract that populates
        :attr:`Forecast.samples`, since it is the only sampler.

        Prediction is routed through ``self.predict``, not ``self(x)``, so it
        hits the :meth:`predict_step` override.

        :param x: Prediction-mode input dict with ``conditioning_target``
            (B, conditioning_len, target_dim) and ``full_covariates``
            (B, conditioning_len + prediction_len, covariate_dim), and
            optionally ``scale``.
        :type x: Dict[str, keras.KerasTensor]
        :param quantile_levels: Quantile levels to extract. Defaults to
            ``[0.1, 0.5, 0.9]``.
        :type quantile_levels: Optional[list]
        :param kwargs: Forwarded to ``self.predict`` (e.g. ``batch_size``,
            ``verbose``).
        :return: A :class:`Forecast` with ``point`` shape ``[B, H, D]``,
            ``quantiles`` shape ``[B, H, D, Q]``, and ``samples`` shape
            ``[S, B, H, D]``.
        :rtype: Forecast
        """
        # DECISION plan_2026-06-10_7036cab1/D-001: force one predict batch, or
        # Keras concatenates per-batch outputs along the sample axis. See decisions.md.
        cond = x['conditioning_target'] if isinstance(x, dict) else x
        kwargs.setdefault('batch_size', int(np.asarray(cond).shape[0]))
        kwargs.setdefault('verbose', 0)
        samples = np.asarray(self.predict(x, **kwargs))      # (S, B, H, D)
        point = samples.mean(axis=0)                         # (B, H, D)
        levels = list(quantile_levels) if quantile_levels is not None else [0.1, 0.5, 0.9]
        q = np.percentile(samples, [lvl * 100 for lvl in levels], axis=0)  # (Q, B, H, D)
        q = np.moveaxis(q, 0, -1)                            # (B, H, D, Q)
        return Forecast(
            point=point,
            quantiles=q,
            quantile_levels=levels,
            samples=samples,
        )

    @staticmethod
    def gaussian_loss(
            y_true: keras.KerasTensor,
            y_pred: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Compute the Gaussian negative log-likelihood.

        `nll = 0.5 * log(2*pi) + log(sigma) + 0.5 * ((y - mu) / sigma)^2`

        :param y_true: Not used; the target is read from ``y_pred``.
        :type y_true: keras.KerasTensor
        :param y_pred: A dict with ``'mu'``, ``'sigma'``, ``'target'``.
        :type y_pred: Dict[str, keras.KerasTensor]
        :return: Mean negative log-likelihood.
        :rtype: keras.KerasTensor
        """
        mu = y_pred['mu']
        sigma = y_pred['sigma']
        target = y_pred['target']

        two_pi = 2.0 * np.pi
        nll = 0.5 * ops.log(two_pi) + ops.log(sigma) + \
              0.5 * ops.square((target - mu) / sigma)

        return ops.mean(nll)

    @staticmethod
    def negative_binomial_loss(
            y_true: keras.KerasTensor,
            y_pred: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Compute the negative binomial negative log-likelihood, up to an additive constant.

        With ``r = 1 / alpha`` and ``p = r / (r + mu) = 1 / (1 + alpha * mu)``:

        `log p(z) = lgamma(z + r) - lgamma(r) - lgamma(z + 1) + r*log(p) + z*log(1-p)`

        This returns the mean of the negation, omitting the parameter-free
        ``-lgamma(z + 1)`` term, which shifts the value by a constant and
        leaves the gradient exact; it is dropped because it is ``nan`` for
        any target below -1.

        :param y_true: Not used; the target is read from ``y_pred``.
        :type y_true: keras.KerasTensor
        :param y_pred: A dict with ``'mu'``, ``'alpha'``, ``'target'``.
        :type y_pred: Dict[str, keras.KerasTensor]
        :return: Mean negative log-likelihood, up to an additive constant.
        :rtype: keras.KerasTensor
        """
        mu = y_pred['mu']
        alpha = y_pred['alpha']
        target = y_pred['target']

        eps = 1e-7
        p = 1.0 / (1.0 + alpha * mu + eps)
        r = 1.0 / (alpha + eps)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-039: lgamma(z+r) - lgamma(r)
        # depends on alpha through r; dropping it as "constant" made the alpha
        # gradient wrong, not merely offset. See decisions.md.
        log_binomial = log_gamma(target + r) - log_gamma(r)

        nll = -(log_binomial
                + r * ops.log(p + eps)
                + target * ops.log(1.0 - p + eps))

        return ops.mean(nll)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for serialization.

        :return: All constructor parameters needed to recreate this instance.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate,
            'likelihood': self.likelihood,
            'target_dim': self.target_dim,
            'num_samples': self.num_samples,
            'scale_epsilon': self.scale_epsilon,
            'conditioning_length': self.conditioning_length,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepAR":
        """Reconstruct a DeepAR from its config.

        All constructor arguments are plain JSON-serializable scalars/strings
        (no nested Keras objects), so a direct ``cls(**config)`` round-trips.
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_deepar(
        num_layers: int = 3,
        hidden_dim: int = 40,
        dropout_rate: float = 0.0,
        recurrent_dropout_rate: float = 0.0,
        likelihood: Literal['gaussian', 'negative_binomial'] = 'gaussian',
        target_dim: int = 1,
        num_samples: int = 100,
        scale_epsilon: float = 1.0,
        covariate_dim: int = 1,
        **kwargs: Any
) -> DeepAR:
    """Construct a :class:`DeepAR` model and build it with a dummy forward pass.

    Runs a tiny training-mode dummy forward pass so the returned model has
    its weights materialized, ready for ``.summary()``, ``.save()``, or
    weight transfer without a separate warmup call.

    :param num_layers: Number of stacked LSTM layers.
    :type num_layers: int
    :param hidden_dim: LSTM hidden width.
    :type hidden_dim: int
    :param dropout_rate: LSTM output dropout rate.
    :type dropout_rate: float
    :param recurrent_dropout_rate: LSTM recurrent dropout rate.
    :type recurrent_dropout_rate: float
    :param likelihood: ``'gaussian'`` or ``'negative_binomial'``.
    :type likelihood: str
    :param target_dim: Target feature dimension.
    :type target_dim: int
    :param num_samples: Monte Carlo sample count for prediction.
    :type num_samples: int
    :param scale_epsilon: Constant added in the scale computation.
    :type scale_epsilon: float
    :param covariate_dim: Covariate width used only for the dummy build
        forward pass; the model is covariate-width agnostic at construction.
    :type covariate_dim: int
    :param kwargs: Forwarded to :class:`DeepAR` (e.g. ``name``).
    :return: A built :class:`DeepAR` instance.
    :rtype: DeepAR
    """
    model = DeepAR(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        likelihood=likelihood,
        target_dim=target_dim,
        num_samples=num_samples,
        scale_epsilon=scale_epsilon,
        recurrent_dropout_rate=recurrent_dropout_rate,
        **kwargs
    )

    # Build via a tiny training-mode dummy dict forward pass.
    dummy = {
        'target': np.zeros((1, 4, target_dim), dtype='float32'),
        'covariates': np.zeros((1, 4, covariate_dim), dtype='float32'),
    }
    model(dummy, training=False)
    return model


# ---------------------------------------------------------------------
