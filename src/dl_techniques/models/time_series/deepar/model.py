"""
Autoregressive recurrent forecaster that emits a likelihood, Gaussian or negative
binomial, instead of a point prediction.

The problem DeepAR addresses is that a forecast is a decision input, and a decision
needs the spread as much as the level. A network trained on squared error returns
`E[z_t | history]` and nothing else, so the width of the outcome is unavailable
exactly where it matters — inventory, capacity, risk. DeepAR instead factorizes the
joint over the horizon into one-step conditionals and models each one explicitly:

`p(z_{t0:T} | z_{1:t0-1}, x) = prod_t p(z_t | theta(h_t))`,
`h_t = LSTM([z_{t-1} / nu, x_t], h_{t-1})`

The recurrent state is the whole of the conditioning; the head maps it to
distribution parameters `theta` and the loss is the negative log-likelihood of the
observed value under those parameters. Nothing about this is Gaussian by necessity:
swapping the head swaps the assumed observation process, which is why the count case
gets a negative binomial (mean `mu`, variance `mu + alpha * mu^2`) rather than a
Gaussian that would put mass on negative counts.

Training and inference run the same recurrence under two different input policies.
Training is teacher-forced: the true lagged target is fed in, every step is scored
in parallel, and the pass is one symbolic forward — this is what `call` does, and it
returns the parameter dict rather than a prediction. Inference is ancestral: at each
horizon step a value is drawn from the current predictive distribution and fed back
as the next input, so uncertainty compounds across the horizon the way it actually
does. There is no closed form for the multi-step predictive distribution under that
recurrence; `num_samples` independent trajectories are drawn and quantiles are read
off as empirical percentiles. The sampling path is deliberately kept out of `call`
and lives in `predict_step` / `_prediction_mode`, so `call` stays a single symbolic
dict-returning function instead of a mode-switched one.

Scale is the second thing the architecture handles explicitly. Across a large panel
the magnitudes of individual series follow a power law, and a shared network cannot
fit both a series in the single digits and one in the millions with the same weights.
Each series gets `nu = mean(z over the conditioning range) + scale_epsilon`; inputs
are divided by it and likelihood parameters are multiplied back. `scale_epsilon`
defaults to 1.0 and is not a numerical fudge — it keeps `nu` away from zero for
near-empty series and makes `nu >> 1` the normal case. The Gaussian `sigma` is
un-scaled by `nu`, NOT `sqrt(nu)`: the forward path divides `z` by `nu` exactly once,
so if `z~ = z / nu` then both `mu = nu * mu~` and `sigma = nu * sigma~` — `sigma` is
a first-moment-scale quantity here, not a variance. The `sqrt` belongs only to the
negative-binomial shape, where `alpha = alpha~ / sqrt(nu)`. Training and sampling
must agree on this, so both sites carry the same reasoning inline.

Which window `nu` is computed over is a correctness question, not a detail. Inference
only ever sees the conditioning range, so the scale must come from there. If
`conditioning_length` is left `None` the training-time scale is the mean of the full
target window, which makes every teacher-forced step a function of the steps being
predicted and simultaneously puts training and serving on different `nu`
distributions. The constructor accepts `conditioning_length` for exactly this, and
the model emits a warning rather than silently proceeding when it is absent; passing
a precomputed `inputs['scale']` bypasses the question entirely.

Two implementation properties are worth knowing before trusting long horizons. The
sampling loop re-runs the whole stacked LSTM over the conditioning window plus every
draw made so far, once per horizon step per sample, because a stacked
`keras.layers.LSTM` does not carry state across separate invocations here, so the
prefix has to be replayed rather than resumed. That costs `num_samples *
prediction_len` full-sequence passes over a sequence that grows from
`conditioning_len + 1` to `conditioning_len + prediction_len`, which is the price of
being ancestral: until 2026-08-15 the context was the conditioning range plus only
the *single most recent* draw, which made an H-step forecast H nearly independent
one-step-ahead forecasts and multi-step quantiles too narrow. Separately, the negative
binomial path samples by moment matching to a Gaussian clipped at zero rather than by
a Gamma-Poisson draw — an approximation, marked as such at the call site.
`negative_binomial_loss` is no longer among the approximations: it dropped
`lgamma(z + r) - lgamma(r)`, and since `r = 1 / alpha` those terms are not constant
in the parameters, so the gradient with respect to `alpha` was wrong rather than
merely offset and dispersion was mis-estimated for every count model. Only the
parameter-free `-lgamma(z + 1)` normalizer is now omitted, which shifts the value by
a constant and leaves the gradient exact.

Both loss functions ignore `y_true` and read the target out of `y_pred`, because
`call` returns the target alongside the parameters. That keeps the likelihood
parameters and the value they are scored against in one structure and makes
`model.compile(loss=model.gaussian_loss)` work with any placeholder `y`. And
`_forecast` forces a single `predict` batch: `predict_step` returns `(S, B, H, D)`
with samples on axis 0, and Keras' default multi-batch concatenation would append
along that same axis, scrambling the result.

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
    DeepARCell
)
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DeepAR(keras.Model, ForecastMixin):
    """
    DeepAR: Probabilistic forecasting with autoregressive recurrent networks.

    DeepAR is a methodology for producing accurate probabilistic forecasts based
    on training an autoregressive recurrent network on multiple related time series.
    It addresses challenges in forecasting datasets with widely-varying scales
    and provides calibrated probabilistic predictions.

    **Intent**: Enable probabilistic forecasting at scale by learning a global
    model across thousands or millions of related time series, handling diverse
    scales, and producing calibrated forecast distributions.

    **Architecture**:
    ```
    Conditioning Range [t=1...t0-1]:
        Input: [z_{t-1}, x_t]
              ↓
        Scale: z_scaled = z / ν
              ↓
        LSTM Encoder: h_t = LSTM([z_scaled_{t-1}, x_t], h_{t-1})
              ↓
        Likelihood Head: θ_t = Head(h_t)
              ↓
        Loss: -log p(z_t | θ_t)

    Prediction Range [t=t0...T]:
        Input: [z_sampled_{t-1}, x_t]
              ↓
        Scale: z_scaled = z_sampled / ν
              ↓
        LSTM Decoder: h_t = LSTM([z_scaled_{t-1}, x_t], h_{t-1})
              ↓
        Likelihood Head: θ_t = Head(h_t)
              ↓
        Sample: z_sampled_t ~ p(z | θ_t * ν)
              ↓
        Iterate: Feed z_sampled_t back as input
    ```

    **Key Features**:
    1. **Scale Handling**: Normalizes inputs and denormalizes outputs to handle
       power-law scale distributions.
    2. **Flexible Likelihoods**: Supports Gaussian (real-valued) and Negative
       Binomial (count data) distributions.
    3. **Probabilistic Forecasts**: Generates multiple sample paths via ancestral
       sampling for quantile estimation.
    4. **Shared Weights**: Uses same LSTM for encoding and decoding.

    Args:
        num_layers: Number of LSTM layers. Defaults to 3.
        hidden_dim: Hidden dimension of LSTM layers. Defaults to 40.
        dropout_rate: Dropout rate for LSTM layers. Defaults to 0.0.
        recurrent_dropout_rate: Recurrent dropout rate. Defaults to 0.0.
        likelihood: Distribution for modeling observations. Either 'gaussian'
            for real-valued data or 'negative_binomial' for count data.
            Defaults to 'gaussian'.
        target_dim: Dimensionality of target variable (typically 1 for
            univariate forecasting). Defaults to 1.
        num_samples: Number of Monte Carlo samples to draw during prediction.
            Defaults to 100.
        scale_epsilon: Small constant added to scale computation. Defaults to 1.0.
        conditioning_length: Number of leading steps used to compute the
            per-series scale nu during training. Set this to the context length.
            If None (default) the scale is computed over the FULL target window,
            which leaks the prediction range into nu and disagrees with
            inference, where the scale comes from the conditioning range only;
            a warning is emitted in that case. Passing a precomputed
            ``inputs['scale']`` bypasses this entirely.
        **kwargs: Additional arguments for Model base class.

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

    # Likelihoods this model can construct a head for. Single source of truth
    # for the ctor validation below (do NOT duplicate the string set inline).
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

        # Create scale layer
        self.scale_layer = ScaleLayer(
            scale_per_sample=False,  # We'll compute scale externally
            epsilon=scale_epsilon,
            name='scale_layer'
        )

        # Create LSTM layers (stacked).
        # NOTE: a plain Python list attribute IS tracked by Keras 3 — verified
        # empirically (model.weights non-empty incl. lstm_*/lstm_cell/* paths
        # after a forward pass), so no ListWrapper/tracked-storage is needed.
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

        # Create likelihood head (likelihood already validated above).
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

        DeepAR consumes a DICT input (``{'target': (B, T, target_dim),
        'covariates': (B, T, covariate_dim), ...}``). Keras captures this dict
        shape in the build-config and replays it on ``load_model``. We MUST build
        the sublayers here (not lazily in ``call``): on load, Keras calls
        ``build`` then restores weights, so the sublayers must already exist to
        receive the saved values. A minimal ``super().build()``-only body left
        the LSTM/head unbuilt at restore time -> kernels were re-initialized on
        the next forward and the saved weights were silently discarded.

        # DECISION plan_2026-06-11_fe7401f4/D-002

        Args:
            input_shape: Dict with ``'target'`` and ``'covariates'`` shapes, OR a
                non-dict/None shape (e.g. from a bare ``model.build(None)``), in
                which case sublayer builds are deferred to the first ``call``.
        """
        if isinstance(input_shape, dict) and 'covariates' in input_shape:
            covariate_dim = input_shape['covariates'][-1]
            seq_len = input_shape['target'][1]

            # LSTM stack input: concat([lagged_target (target_dim), covariates]).
            lstm_input_shape = (None, seq_len, self.target_dim + covariate_dim)
            for lstm in self.lstm_layers:
                lstm.build(lstm_input_shape)
                # Subsequent layers consume the prior LSTM's hidden sequence.
                lstm_input_shape = (None, seq_len, self.hidden_dim)

            # Likelihood head consumes the final LSTM hidden state.
            self.likelihood_head.build((None, seq_len, self.hidden_dim))

        super().build(input_shape)

    def compute_scale(
            self,
            target: keras.KerasTensor,
            conditioning_length: Optional[int] = None
    ) -> keras.KerasTensor:
        """
        Compute scale factor for each time series.

        Args:
            target: Target time series, shape (batch, seq_len, target_dim).
            conditioning_length: If provided, only use first N steps for scale.

        Returns:
            Scale tensor, shape (batch, 1, target_dim).
        """
        if conditioning_length is not None:
            target_for_scale = target[:, :conditioning_length, :]
        else:
            target_for_scale = target

        # Scale = mean + epsilon
        scale = ops.mean(target_for_scale, axis=1, keepdims=True) + self.scale_epsilon
        return scale

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through DeepAR (training mode).

        ``call`` is the training-mode (teacher-forced) path and returns the
        likelihood-parameter dict. The Monte-Carlo SAMPLING path is NOT routed
        through ``call``: it lives in :meth:`predict_step`, which invokes
        :meth:`_prediction_mode` directly. This keeps ``call`` a single,
        symbolic, dict-returning function (canonical Keras 3 contract) instead
        of overloading it with a mode flag.

        Args:
            inputs: Dictionary with keys:
                - 'target': Target time series (batch, seq_len, target_dim)
                - 'covariates': Covariates (batch, seq_len, covariate_dim)
                - 'scale': Optional pre-computed scale (batch, 1, target_dim)
            training: Whether in training mode.

        Returns:
            Dictionary with likelihood parameters
            (``{'mu', 'sigma', 'target'}`` for Gaussian,
            ``{'mu', 'alpha', 'target'}`` for negative-binomial).
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
        """
        Training mode: Teacher forcing with true observations.

        Args:
            inputs: Dictionary with 'target', 'covariates', optional 'scale'.
            training: Whether in training mode.

        Returns:
            Dictionary with likelihood parameters for each time step.
        """
        target = inputs['target']  # (batch, seq_len, target_dim)
        covariates = inputs['covariates']  # (batch, seq_len, covariate_dim)
        scale = inputs.get('scale', None)

        # Compute scale if not provided.
        #
        # `conditioning_length` MUST be honoured here. Without it the scale is
        # mean(z_1..z_T) over the whole window -- including the steps being
        # predicted -- so every teacher-forced step t is divided by a nu that is
        # a function of z_{t+1..T}, and the likelihood parameters are multiplied
        # back by that same nu. Both the inputs and the output scale then carry
        # information about the future of the window.
        #
        # _prediction_mode has always done this correctly (it passes only the
        # conditioning target), so leaving it out here also made training and
        # inference use different nu distributions -- leakage AND train/serve
        # skew. The `conditioning_length` argument of compute_scale existed for
        # exactly this purpose and no caller ever supplied it.
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

        # Scale the target
        target_scaled = self.scale_layer(target, scale=scale, inverse=False)

        # Lag the target by 1 time step (shift right, pad with zeros)
        batch_size = ops.shape(target)[0]
        seq_len = ops.shape(target)[1]
        target_dim = ops.shape(target)[2]

        # Create lagged target: [0, z_1, z_2, ..., z_{T-1}]
        zeros = ops.zeros((batch_size, 1, target_dim))
        lagged_target = ops.concatenate([zeros, target_scaled[:, :-1, :]], axis=1)

        # Concatenate lagged target with covariates
        inputs_combined = ops.concatenate([lagged_target, covariates], axis=-1)

        # Pass through LSTM layers
        hidden = inputs_combined
        for lstm in self.lstm_layers:
            hidden = lstm(hidden, training=training)

        # Compute likelihood parameters
        if self.likelihood == 'gaussian':
            mu_scaled, sigma_scaled = self.likelihood_head(hidden)

            # Inverse scale for mu and sigma.
            #
            # sigma uses `scale`, NOT sqrt(scale). The forward path divides the
            # target by nu exactly once (`target_scaled = z / nu` above), so if
            # z~ = z/nu then mu = nu*mu~ AND sigma = nu*sigma~ -- sigma is a
            # first-moment-scale quantity, not a variance. Salinas et al. §3.2
            # gives mu = nu*mu~, sigma = nu*softplus(sigma~).
            #
            # The sqrt belongs only to the NegBinomial branch below, where
            # alpha = alpha~/sqrt(nu) is correct; it was copied across to the
            # Gaussian head, where it under-scaled sigma by sqrt(nu) and left
            # every predictive interval mis-calibrated by a scale-dependent
            # factor (nu = mean(z)+1, so nu >> 1 is the normal case).
            mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
            sigma = self.scale_layer(sigma_scaled, scale=scale, inverse=True)

            return {'mu': mu, 'sigma': sigma, 'target': target}

        else:  # negative_binomial
            mu_scaled, alpha_scaled = self.likelihood_head(hidden)

            # Inverse scale: mu * scale, alpha / sqrt(scale)
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
        """
        Prediction mode: Autoregressive sampling.

        Args:
            inputs: Dictionary with:
                - 'conditioning_target': (batch, cond_len, target_dim)
                - 'full_covariates': (batch, total_len, covariate_dim)
                - 'scale': Optional pre-computed scale
            training: Whether in training mode (typically False for prediction).

        Returns:
            Samples tensor: (num_samples, batch, pred_len, target_dim).
        """
        conditioning_target = inputs['conditioning_target']
        full_covariates = inputs['full_covariates']
        scale = inputs.get('scale', None)

        batch_size = ops.shape(conditioning_target)[0]
        conditioning_len = ops.shape(conditioning_target)[1]
        total_len = ops.shape(full_covariates)[1]
        prediction_len = total_len - conditioning_len

        # Compute scale if not provided
        if scale is None:
            scale = self.compute_scale(conditioning_target)

        # Scale conditioning target
        conditioning_scaled = self.scale_layer(
            conditioning_target,
            scale=scale,
            inverse=False
        )

        # Get conditioning covariates
        conditioning_covariates = full_covariates[:, :conditioning_len, :]
        prediction_covariates = full_covariates[:, conditioning_len:, :]

        # Encode conditioning range
        # Lag the conditioning target
        zeros = ops.zeros((batch_size, 1, self.target_dim))
        lagged_conditioning = ops.concatenate(
            [zeros, conditioning_scaled[:, :-1, :]],
            axis=1
        )

        # Combine with covariates
        encoder_inputs = ops.concatenate(
            [lagged_conditioning, conditioning_covariates],
            axis=-1
        )

        # No encoder-only pass here: the decoder loop below replays
        # `encoder_inputs` as the prefix of every sequence it runs, so a
        # separate pass would be a whole stacked-LSTM forward whose result is
        # discarded. One was computed here (`last_hidden`) and never read.

        # Generate samples
        all_samples = []

        for sample_idx in range(self.num_samples):
            # Initialize with last observed value (scaled)
            current_value = conditioning_scaled[:, -1:, :]  # (batch, 1, target_dim)

            sample_trajectory = []
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-038
            # The decoder steps consumed SO FAR, kept so step t's context is the
            # conditioning window plus draws 1..t. Until 2026-08-15 the context
            # was `[encoder_inputs, decoder_input]` -- the window plus only the
            # single most recent draw -- which made an H-step forecast H nearly
            # independent one-step-ahead forecasts and multi-step quantiles too
            # narrow. Do NOT drop this list to shorten the sequence: ancestral
            # sampling is defined by conditioning on the whole drawn prefix.
            # See decisions.md D-038.
            decoder_history = []

            for t in range(prediction_len):
                # Get covariates for this time step
                current_covariates = prediction_covariates[:, t:t + 1, :]

                # Combine current value with covariates
                decoder_input = ops.concatenate(
                    [current_value, current_covariates],
                    axis=-1
                )
                decoder_history.append(decoder_input)

                # Re-run the stacked LSTM over the conditioning window plus
                # every decoder step drawn so far. A stacked `keras.layers.LSTM`
                # does not carry state across separate invocations here, so the
                # prefix has to be replayed rather than resumed.
                decoder_input_seq = ops.concatenate(
                    [encoder_inputs] + decoder_history,
                    axis=1
                )

                # Pass through LSTM
                hidden_t = decoder_input_seq
                for lstm in self.lstm_layers:
                    hidden_t = lstm(hidden_t, training=training)

                # Get output for current time step
                hidden_t_current = hidden_t[:, -1:, :]

                # Compute likelihood parameters
                if self.likelihood == 'gaussian':
                    mu_scaled, sigma_scaled = self.likelihood_head(hidden_t_current)

                    # Inverse scale. sigma uses `scale`, not sqrt(scale) --
                    # same reasoning as _training_mode; the two sites must agree
                    # or sampling would not match the trained likelihood.
                    mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
                    sigma = self.scale_layer(sigma_scaled, scale=scale, inverse=True)

                    # Sample from Gaussian
                    epsilon = keras.random.normal(ops.shape(mu))
                    sampled_value = mu + sigma * epsilon

                else:  # negative_binomial
                    mu_scaled, alpha_scaled = self.likelihood_head(hidden_t_current)

                    # Inverse scale
                    mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
                    alpha = self.scale_layer(
                        alpha_scaled,
                        scale=ops.reciprocal(ops.sqrt(scale)),
                        inverse=True
                    )

                    # Sample from Negative Binomial (approximation via Gamma-Poisson)
                    # For simplicity, we use Gaussian approximation
                    # In practice, you'd want proper NegBin sampling
                    variance = mu + ops.square(mu) * alpha
                    std = ops.sqrt(variance)
                    epsilon = keras.random.normal(ops.shape(mu))
                    sampled_value = mu + std * epsilon
                    sampled_value = ops.maximum(sampled_value, 0.0)  # Ensure non-negative

                sample_trajectory.append(sampled_value)

                # Update current value (scaled) for next iteration
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
        """Override predict_step to use the sampling (prediction) mode.

        ``call`` returns training-mode params only; the sampling path is invoked
        here by calling :meth:`_prediction_mode` directly. Returns ``(S, B, H, D)``
        with axis-0 = num_samples (NOT batch) — the ``_forecast`` reducer and the
        batch-size-forcing logic in ``_forecast`` depend on this axis order.
        """
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        return self._prediction_mode(x, training=False)

    def _forecast(
            self,
            x: Dict[str, "keras.KerasTensor"],
            quantile_levels: Optional[list] = None,
            **kwargs: Any
    ) -> Forecast:
        """Produce a unified :class:`Forecast` via Monte-Carlo sampling.

        This is the ``ForecastMixin`` hook for DeepAR. It runs the model's
        autoregressive sampling path and reduces the resulting trajectories into
        the shared contract: an empirical-mean point forecast plus
        empirical-percentile quantiles. DeepAR is the ONLY model on the contract
        that populates :attr:`Forecast.samples` (the raw Monte-Carlo paths),
        because it is the only sampler — point and quantile-tensor models have no
        per-sample trajectories to expose.

        Prediction is routed through ``self.predict`` (NOT ``self(x)``) so it
        hits the :meth:`predict_step` override, which calls
        :meth:`_prediction_mode` and returns the sampled trajectories of shape
        ``(S, B, H, D)``.

        Args:
            x: Prediction-mode input dict with keys ``conditioning_target``
                ``(B, conditioning_len, target_dim)`` and ``full_covariates``
                ``(B, conditioning_len + prediction_len, covariate_dim)``
                (optionally ``scale``). The forecast horizon ``H`` is derived as
                ``total_len - conditioning_len`` inside the prediction path.
            quantile_levels: Quantile levels to extract; defaults to
                ``[0.1, 0.5, 0.9]``.
            **kwargs: Forwarded to ``self.predict`` (e.g. ``batch_size``,
                ``verbose``).

        Returns:
            A :class:`Forecast` with ``point`` shape ``[B, H, D]``, ``quantiles``
            shape ``[B, H, D, Q]`` ordered to match ``quantile_levels``, and
            ``samples`` shape ``[S, B, H, D]`` (non-``None``).
        """
        # DECISION plan_2026-06-10_7036cab1/D-001
        # Force a SINGLE predict batch. `predict_step` returns (S, B, H, D) with
        # axis-0 = num_samples (NOT batch). Keras' default multi-batch predict
        # concatenates per-batch outputs along axis 0 (the sample axis), which
        # scrambles the result to the last partial batch's B (observed: y_true
        # (928,8,1) vs point (32,8,1)). Passing batch_size = B keeps it one batch
        # (correct shapes) and is also faster: the sampling loop is vectorized
        # over B, so one big batch beats many re-batched calls.
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
        """
        Gaussian negative log-likelihood loss.

        Args:
            y_true: Not used (target is in y_pred).
            y_pred: Dictionary with 'mu', 'sigma', 'target'.

        Returns:
            Negative log-likelihood.
        """
        mu = y_pred['mu']
        sigma = y_pred['sigma']
        target = y_pred['target']

        # Gaussian NLL: 0.5 * log(2π) + log(σ) + 0.5 * ((y - μ) / σ)^2
        two_pi = 2.0 * np.pi
        nll = 0.5 * ops.log(two_pi) + ops.log(sigma) + \
              0.5 * ops.square((target - mu) / sigma)

        return ops.mean(nll)

    @staticmethod
    def negative_binomial_loss(
            y_true: keras.KerasTensor,
            y_pred: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Negative Binomial negative log-likelihood loss.

        With shape ``r = 1 / alpha`` and success probability
        ``p = r / (r + mu) = 1 / (1 + alpha * mu)``, the log-pmf is::

            log p(z) = lgamma(z + r) - lgamma(r) - lgamma(z + 1)
                       + r * log(p) + z * log(1 - p)

        and this returns the mean of its negation, less the parameter-free
        ``-lgamma(z + 1)`` normalizer. Dropping a term that does not depend on
        ``mu`` or ``alpha`` shifts the value by a constant and leaves the
        gradient exact; it is dropped because it is ``nan`` for any target
        below ``-1``, and a mis-specified target should not turn the loss into
        a ``nan`` when it can simply be a large number.

        Args:
            y_true: Not used (target is in y_pred).
            y_pred: Dictionary with 'mu', 'alpha', 'target'.

        Returns:
            Negative log-likelihood, up to an additive constant.
        """
        mu = y_pred['mu']
        alpha = y_pred['alpha']
        target = y_pred['target']

        eps = 1e-7
        p = 1.0 / (1.0 + alpha * mu + eps)
        r = 1.0 / (alpha + eps)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-039
        # The `lgamma(z + r) - lgamma(r)` pair is NOT a constant that may be
        # dropped for training: `r = 1 / alpha`, so both terms depend on alpha
        # and their omission makes d(loss)/d(alpha) systematically wrong rather
        # than merely offset -- dispersion is mis-estimated for every count
        # model. It was omitted here as "simplified, ignoring Gamma terms".
        # Do NOT drop it again. `log_gamma` exists because `keras.ops` has no
        # `lgamma` in Keras 3.8. See decisions.md D-039.
        log_binomial = log_gamma(target + r) - log_gamma(r)

        nll = -(log_binomial
                + r * ops.log(p + eps)
                + target * ops.log(1.0 - p + eps))

        return ops.mean(nll)

    def get_config(self) -> Dict[str, Any]:
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
    """Construct and build a :class:`DeepAR` model.

    Convenience factory mirroring ``create_tirex_model``: it instantiates the
    model and runs a tiny training-mode dummy forward pass so the returned model
    is already BUILT (weights materialized), ready for ``.summary()``,
    ``.save()``, or weight transfer without a separate warmup call.

    Args:
        num_layers: Number of stacked LSTM layers.
        hidden_dim: LSTM hidden width.
        dropout_rate: LSTM output dropout rate.
        recurrent_dropout_rate: LSTM recurrent dropout rate.
        likelihood: Observation distribution, ``'gaussian'`` or
            ``'negative_binomial'``.
        target_dim: Target feature dimension.
        num_samples: Monte-Carlo sample count for prediction.
        scale_epsilon: Constant added in scale computation.
        covariate_dim: Covariate channel width used only for the dummy build
            forward pass (the model is covariate-width agnostic at construction).
        **kwargs: Forwarded to :class:`DeepAR` (e.g. ``name``).

    Returns:
        A built :class:`DeepAR` instance.
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
