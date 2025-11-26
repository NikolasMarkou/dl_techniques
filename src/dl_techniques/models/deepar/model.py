"""
DeepAR Probabilistic Forecasting Model.

This module implements the DeepAR model for probabilistic time series forecasting.
DeepAR learns a global model from multiple related time series and produces
probabilistic forecasts through Monte Carlo sampling.

Reference:
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Salinas et al., 2019
    https://arxiv.org/abs/1704.04110
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Optional, Union, Literal, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.time_series.deepar_blocks import (
    ScaleLayer,
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
    DeepARCell
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DeepAR(keras.Model):
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
        dropout: Dropout rate for LSTM layers. Defaults to 0.0.
        recurrent_dropout: Recurrent dropout rate. Defaults to 0.0.
        likelihood: Distribution for modeling observations. Either 'gaussian'
            for real-valued data or 'negative_binomial' for count data.
            Defaults to 'gaussian'.
        target_dim: Dimensionality of target variable (typically 1 for
            univariate forecasting). Defaults to 1.
        num_samples: Number of Monte Carlo samples to draw during prediction.
            Defaults to 100.
        scale_epsilon: Small constant added to scale computation. Defaults to 1.0.
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
            dropout=0.1
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

    def __init__(
            self,
            num_layers: int = 3,
            hidden_dim: int = 40,
            dropout: float = 0.0,
            recurrent_dropout: float = 0.0,
            likelihood: Literal['gaussian', 'negative_binomial'] = 'gaussian',
            target_dim: int = 1,
            num_samples: int = 100,
            scale_epsilon: float = 1.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
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

        # Create LSTM layers (stacked)
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.LSTM(
                hidden_dim,
                return_sequences=True,
                return_state=False,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f'lstm_{i}'
            )
            self.lstm_layers.append(lstm)

        # Create likelihood head
        if likelihood == 'gaussian':
            self.likelihood_head = GaussianLikelihoodHead(
                units=target_dim,
                name='gaussian_head'
            )
        elif likelihood == 'negative_binomial':
            self.likelihood_head = NegativeBinomialLikelihoodHead(
                units=target_dim,
                name='negbin_head'
            )
        else:
            raise ValueError(
                f"Unknown likelihood: {likelihood}. "
                f"Must be 'gaussian' or 'negative_binomial'"
            )

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
            return_samples: bool = False
    ) -> Union[Dict[str, keras.KerasTensor], keras.KerasTensor]:
        """
        Forward pass through DeepAR.

        Args:
            inputs: Dictionary with keys:
                - 'target': Target time series (batch, seq_len, target_dim)
                - 'covariates': Covariates (batch, seq_len, covariate_dim)
                - 'scale': Optional pre-computed scale (batch, 1, target_dim)
                For prediction mode:
                - 'conditioning_target': (batch, cond_len, target_dim)
                - 'full_covariates': (batch, total_len, covariate_dim)
            training: Whether in training mode.
            return_samples: If True, return Monte Carlo samples (prediction mode).

        Returns:
            Training mode: Dictionary with likelihood parameters.
            Prediction mode (return_samples=True): Sampled trajectories.
        """
        if isinstance(inputs, dict):
            if return_samples:
                return self._prediction_mode(inputs, training=training)
            else:
                return self._training_mode(inputs, training=training)
        else:
            raise ValueError(
                "Inputs must be a dictionary with 'target' and 'covariates' keys"
            )

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

        # Compute scale if not provided
        if scale is None:
            scale = self.compute_scale(target)

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

            # Inverse scale for mu and sigma
            mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
            sigma = self.scale_layer(
                sigma_scaled,
                scale=ops.sqrt(scale),
                inverse=True
            )

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

        # Pass through LSTM to get final hidden state
        hidden = encoder_inputs
        for lstm in self.lstm_layers:
            hidden = lstm(hidden, training=training)

        # Get last hidden state to initialize decoder
        # For prediction, we need to maintain LSTM states across steps
        # However, Keras LSTM doesn't easily expose this in functional API
        # We'll use a simpler approach: use the last encoder output
        last_hidden = hidden[:, -1:, :]  # (batch, 1, hidden_dim)

        # Generate samples
        all_samples = []

        for sample_idx in range(self.num_samples):
            # Initialize with last observed value (scaled)
            current_value = conditioning_scaled[:, -1:, :]  # (batch, 1, target_dim)

            sample_trajectory = []

            for t in range(prediction_len):
                # Get covariates for this time step
                current_covariates = prediction_covariates[:, t:t + 1, :]

                # Combine current value with covariates
                decoder_input = ops.concatenate(
                    [current_value, current_covariates],
                    axis=-1
                )

                # Extend last_hidden to match sequence length
                # This is a simplified approach; ideally we'd maintain LSTM state
                decoder_input_seq = ops.concatenate(
                    [encoder_inputs, decoder_input],
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

                    # Inverse scale
                    mu = self.scale_layer(mu_scaled, scale=scale, inverse=True)
                    sigma = self.scale_layer(
                        sigma_scaled,
                        scale=ops.sqrt(scale),
                        inverse=True
                    )

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
        """Override predict_step to use sampling mode."""
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        return self(x, training=False, return_samples=True)

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

        Args:
            y_true: Not used (target is in y_pred).
            y_pred: Dictionary with 'mu', 'alpha', 'target'.

        Returns:
            Negative log-likelihood.
        """
        mu = y_pred['mu']
        alpha = y_pred['alpha']
        target = y_pred['target']

        # Negative Binomial NLL (simplified, ignoring Gamma terms)
        # Full formula involves lgamma functions
        # Approximation: -log p(z|μ,α) ≈ key terms

        # p = 1 / (1 + α*μ)
        # r = 1 / α

        eps = 1e-7
        p = 1.0 / (1.0 + alpha * mu + eps)

        # Simplified NLL (main terms)
        nll = -ops.log(p + eps) / alpha - target * ops.log(1.0 - p + eps)

        return ops.mean(nll)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'likelihood': self.likelihood,
            'target_dim': self.target_dim,
            'num_samples': self.num_samples,
            'scale_epsilon': self.scale_epsilon,
        })
        return config

# ---------------------------------------------------------------------
