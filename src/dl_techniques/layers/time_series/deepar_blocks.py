"""
DeepAR Custom Layers.

This module implements the core building blocks for the DeepAR probabilistic
forecasting model, including scale handling, likelihood parameter computation,
and the autoregressive LSTM architecture.

Reference:
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Salinas et al., 2019
    https://arxiv.org/abs/1704.04110
"""

import keras
from keras import ops, layers
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ScaleLayer(keras.layers.Layer):
    """
    Applies item-dependent scaling to inputs and inverse scaling to outputs.

    This layer addresses the challenge of learning from time series with widely
    varying magnitudes by normalizing autoregressive inputs and denormalizing
    likelihood parameters. It is critical for datasets exhibiting power-law
    scale distributions.

    Scale is computed as:
        ``nu = mean(conditioning_range) + epsilon``

    Forward scaling divides by ``nu``, while inverse scaling multiplies by
    ``nu`` (for means) or scales by ``sqrt(nu)`` (for standard deviations in
    the Gaussian case) or ``1/sqrt(nu)`` (for shape parameters in the Negative
    Binomial case).

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, seq_len, features)
                    │
                    ▼
        ┌───────────────────────────────┐
        │  Scale Computation            │
        │  nu = mean(x, axis=1) + eps   │
        └──────────────┬────────────────┘
                       │
               ┌───────┴───────┐
               ▼               ▼
        ┌────────────┐  ┌─────────────┐
        │  Forward:  │  │  Inverse:   │
        │  x / nu    │  │  x * nu     │
        └─────┬──────┘  └──────┬──────┘
              │                │
              ▼                ▼
        Scaled Output    Descaled Output

    :param scale_per_sample: If True, compute scale per sample in batch.
        If False, use provided scale. Defaults to True.
    :type scale_per_sample: bool
    :param epsilon: Small constant for numerical stability. Defaults to 1.0.
    :type epsilon: float
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            scale_per_sample: bool = True,
            epsilon: float = 1.0,
            **kwargs: Any
    ) -> None:
        """
        Initialize the ScaleLayer.

        :param scale_per_sample: If True, compute scale per sample.
        :type scale_per_sample: bool
        :param epsilon: Small constant for numerical stability.
        :type epsilon: float
        :param kwargs: Additional arguments for Layer base class.
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

        :param inputs: Input tensor to scale.
        :type inputs: keras.KerasTensor
        :param scale: Pre-computed scale values. If None and scale_per_sample
            is True, computes from inputs.
        :type scale: keras.KerasTensor or None
        :param inverse: If True, apply inverse scaling (multiply). If False,
            apply forward scaling (divide).
        :type inverse: bool
        :return: Scaled or inverse-scaled tensor.
        :rtype: keras.KerasTensor
        """
        if scale is None and self.scale_per_sample:
            # Compute scale as mean over sequence dimension plus epsilon
            scale = ops.mean(inputs, axis=1, keepdims=True) + self.epsilon

        if scale is None:
            return inputs

        if inverse:
            return inputs * scale
        else:
            return inputs / scale

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


@keras.saving.register_keras_serializable()
class GaussianLikelihoodHead(keras.layers.Layer):
    """
    Computes Gaussian likelihood parameters (mean, std) from hidden states.

    This layer projects LSTM hidden states to Gaussian distribution parameters
    using affine transformations with appropriate activations to ensure valid
    parameter values (positive standard deviation).

    The mathematical operations are:
        ``mu(h) = W_mu^T h + b_mu``
        ``sigma(h) = log(1 + exp(W_sigma^T h + b_sigma))``

    **Architecture Overview:**

    .. code-block:: text

        Input: h_t (batch, [seq_len,] hidden_dim)
                    │
                    ├─────────────────────┐
                    ▼                     ▼
            ┌──────────────┐     ┌───────────────────┐
            │ Dense(units) │     │  Dense(units)     │
            │  (linear)    │     │  (linear logits)  │
            └──────┬───────┘     └────────┬──────────┘
                   │                      │
                   ▼                      ▼
                mu (mean)         ┌──────────────┐
                                  │  Softplus    │
                                  │  log(1+exp)  │
                                  └──────┬───────┘
                                         │
                                         ▼
                                   sigma (std > 0)
                   │                      │
                   ▼                      ▼
            Output: (mu, sigma)

    :param units: Dimensionality of output (typically 1 for univariate time series).
    :type units: int
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            units: int = 1,
            **kwargs: Any
    ) -> None:
        """
        Initialize the GaussianLikelihoodHead.

        :param units: Dimensionality of output.
        :type units: int
        :param kwargs: Additional arguments for Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units

        # Create projection layers
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
        Build the layer by explicitly building sub-layers.

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
        Compute Gaussian distribution parameters from hidden states.

        :param inputs: Hidden states from LSTM.
        :type inputs: keras.KerasTensor
        :return: Tuple of (mu, sigma) tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        mu = self.mu_projection(inputs)

        # Softplus activation to ensure sigma > 0
        sigma_logits = self.sigma_projection(inputs)
        sigma = ops.log(1.0 + ops.exp(sigma_logits))

        return mu, sigma

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute output shapes for both mu and sigma.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        :return: Tuple of two shapes for mu and sigma.
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


@keras.saving.register_keras_serializable()
class NegativeBinomialLikelihoodHead(keras.layers.Layer):
    """
    Computes Negative Binomial likelihood parameters (mu, alpha) from hidden states.

    This layer projects LSTM hidden states to Negative Binomial distribution
    parameters, suitable for modeling count data with overdispersion. Both
    parameters must be positive, enforced via softplus activation.

    The mathematical operations are:
        ``mu(h) = log(1 + exp(W_mu^T h + b_mu))``
        ``alpha(h) = log(1 + exp(W_alpha^T h + b_alpha))``

    Distribution properties:
        ``E[z] = mu``
        ``Var[z] = mu + mu^2 * alpha``

    **Architecture Overview:**

    .. code-block:: text

        Input: h_t (batch, [seq_len,] hidden_dim)
                    │
                    ├─────────────────────┐
                    ▼                     ▼
            ┌──────────────┐     ┌──────────────────┐
            │ Dense(units) │     │  Dense(units)     │
            │ (linear)     │     │  (linear logits)  │
            └──────┬───────┘     └────────┬──────────┘
                   │                      │
                   ▼                      ▼
            ┌──────────────┐     ┌──────────────────┐
            │  Softplus    │     │  Softplus         │
            │  log(1+exp)  │     │  log(1+exp)       │
            └──────┬───────┘     └────────┬──────────┘
                   │                      │
                   ▼                      ▼
              mu (mean > 0)       alpha (shape > 0)
                   │                      │
                   ▼                      ▼
            Output: (mu, alpha)

    :param units: Dimensionality of output (typically 1 for univariate time series).
    :type units: int
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            units: int = 1,
            **kwargs: Any
    ) -> None:
        """
        Initialize the NegativeBinomialLikelihoodHead.

        :param units: Dimensionality of output.
        :type units: int
        :param kwargs: Additional arguments for Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units

        # Create projection layers
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
        Build the layer by explicitly building sub-layers.

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
        Compute Negative Binomial distribution parameters from hidden states.

        :param inputs: Hidden states from LSTM.
        :type inputs: keras.KerasTensor
        :return: Tuple of (mu, alpha) tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Softplus activation to ensure both params > 0
        mu_logits = self.mu_projection(inputs)
        mu = ops.log(1.0 + ops.exp(mu_logits))

        alpha_logits = self.alpha_projection(inputs)
        alpha = ops.log(1.0 + ops.exp(alpha_logits))

        return mu, alpha

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute output shapes for both mu and alpha.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        :return: Tuple of two shapes for mu and alpha.
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


@keras.saving.register_keras_serializable()
class DeepARCell(keras.layers.Layer):
    """
    Autoregressive recurrent cell for DeepAR.

    This layer implements the core autoregressive recurrent computation of DeepAR,
    combining the previous observation, covariates, and hidden state to produce
    the next hidden state. It is designed to work with RNN layers.

    The mathematical operation is:
        ``h_t = h(h_{t-1}, z_{t-1}, x_t, Theta)``

    where ``h`` is implemented as an LSTM cell.

    **Architecture Overview:**

    .. code-block:: text

        Inputs: [z_{t-1}, x_t]       States: h_{t-1}
                    │                       │
                    ▼                       │
            ┌───────────────┐               │
            │  Concatenate  │               │
            │  [z_{t-1}, x_t]               │
            └───────┬───────┘               │
                    │                       │
                    ▼                       ▼
            ┌───────────────────────────────────┐
            │           LSTM Cell               │
            │  h_t = LSTM([z_{t-1}, x_t], h_{t-1})│
            └───────────────┬───────────────────┘
                            │
                            ▼
                    Output: h_t (batch, units)
                    States: (h_t, c_t)

    :param units: Number of LSTM units (hidden dimension).
    :type units: int
    :param dropout: Dropout rate for LSTM. Defaults to 0.0.
    :type dropout: float
    :param recurrent_dropout: Recurrent dropout rate for LSTM. Defaults to 0.0.
    :type recurrent_dropout: float
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            units: int,
            dropout: float = 0.0,
            recurrent_dropout: float = 0.0,
            **kwargs: Any
    ) -> None:
        """
        Initialize the DeepARCell.

        :param units: Number of LSTM units.
        :type units: int
        :param dropout: Dropout rate for LSTM.
        :type dropout: float
        :param recurrent_dropout: Recurrent dropout rate for LSTM.
        :type recurrent_dropout: float
        :param kwargs: Additional arguments for Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.state_size = units

        # Create LSTM cell
        self.lstm_cell = layers.LSTMCell(
            units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name='lstm_cell'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the cell by building the LSTM sub-layer.

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

        :param inputs: Input tensor at time t, shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param states: State tensors from previous time step.
        :type states: tuple[keras.KerasTensor, ...]
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Tuple of (output, new_states).
        :rtype: tuple[keras.KerasTensor, tuple[keras.KerasTensor, ...]]
        """
        output, new_states = self.lstm_cell(inputs, states, training=training)
        return output, new_states

    def get_initial_state(
            self,
            batch_size: Optional[int] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Get initial state for the cell.

        :param batch_size: Batch size for the initial state tensors.
        :type batch_size: int or None
        :return: Tuple of initial hidden and cell state tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        return self.lstm_cell.get_initial_state(batch_size=batch_size)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
        })
        return config
