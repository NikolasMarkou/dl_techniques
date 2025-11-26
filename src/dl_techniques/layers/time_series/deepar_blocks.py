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

    **Intent**: Enable DeepAR to learn effectively from time series spanning
    multiple orders of magnitude by bringing all series into a common scale
    during processing, then restoring the original scale for predictions.

    **Architecture**:
    ```
    Input: (batch, seq_len, features)
           ↓
    Scale Computation: ν = mean(conditioning_range) + 1
           ↓
    Forward: x_scaled = x / ν
    Inverse: μ_scaled * ν, σ_scaled * √ν (for Gaussian)
             μ_scaled * ν, α_scaled / √ν (for NegBin)
           ↓
    Output: scaled or descaled values
    ```

    Args:
        scale_per_sample: If True, compute scale per sample in batch.
            If False, use provided scale. Defaults to True.
        epsilon: Small constant for numerical stability. Defaults to 1.0.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, seq_len, features)`.

    Output shape:
        Same as input shape when scaling forward.

    Example:
        ```python
        # Scale time series inputs
        scale_layer = ScaleLayer()

        # During encoding (conditioning range)
        conditioning_data = keras.random.normal((32, 50, 1))
        scale = ops.mean(conditioning_data, axis=1, keepdims=True) + 1.0
        scaled_data = conditioning_data / scale

        # During decoding (apply inverse to likelihood params)
        mu_scaled = model_output[..., 0:1]
        sigma_scaled = model_output[..., 1:2]
        mu = mu_scaled * scale
        sigma = sigma_scaled * ops.sqrt(scale)
        ```
    """

    def __init__(
            self,
            scale_per_sample: bool = True,
            epsilon: float = 1.0,
            **kwargs: Any
    ) -> None:
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

        Args:
            inputs: Input tensor to scale.
            scale: Pre-computed scale values. If None and scale_per_sample=True,
                compute from inputs.
            inverse: If True, apply inverse scaling (multiply). If False,
                apply forward scaling (divide).

        Returns:
            Scaled or inverse-scaled tensor.
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

    **Intent**: Convert recurrent network outputs into parameters for a
    Gaussian distribution over the next time step prediction.

    **Architecture**:
    ```
    Input: h_t (hidden state)
           ↓
    Linear(units) → μ (mean)
           ↓
    Linear(units) → Softplus → σ (std > 0)
           ↓
    Output: (μ, σ)
    ```

    **Mathematical Operation**:
        μ(h) = W_μ^T h + b_μ
        σ(h) = log(1 + exp(W_σ^T h + b_σ))

    Args:
        units: Dimensionality of output (typically 1 for univariate time series).
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        2D/3D tensor with shape: `(batch_size, [seq_len,] hidden_dim)`.

    Output shape:
        Tuple of two tensors with shape: `(batch_size, [seq_len,] units)`.

    Example:
        ```python
        # Create likelihood head
        likelihood_head = GaussianLikelihoodHead(units=1)

        # Compute parameters from LSTM hidden states
        hidden_states = keras.random.normal((32, 50, 128))
        mu, sigma = likelihood_head(hidden_states)

        # Use for probabilistic forecasting
        # log_likelihood = -0.5 * log(2π) - log(σ) - 0.5 * ((y - μ) / σ)^2
        ```
    """

    def __init__(
            self,
            units: int = 1,
            **kwargs: Any
    ) -> None:
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
        """Build the layer by explicitly building sub-layers."""
        self.mu_projection.build(input_shape)
        self.sigma_projection.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute Gaussian parameters.

        Args:
            inputs: Hidden states from LSTM.

        Returns:
            Tuple of (mu, sigma) tensors.
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
        """Compute output shapes for both mu and sigma."""
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)
        return output_shape, output_shape

    def get_config(self) -> Dict[str, Any]:
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
    parameters must be positive.

    **Intent**: Convert recurrent network outputs into parameters for a
    Negative Binomial distribution, enabling accurate modeling of count data
    with varying dispersion levels.

    **Architecture**:
    ```
    Input: h_t (hidden state)
           ↓
    Linear(units) → Softplus → μ (mean > 0)
           ↓
    Linear(units) → Softplus → α (shape > 0)
           ↓
    Output: (μ, α)
    ```

    **Mathematical Operation**:
        μ(h) = log(1 + exp(W_μ^T h + b_μ))
        α(h) = log(1 + exp(W_α^T h + b_α))

    **Distribution Properties**:
        E[z] = μ
        Var[z] = μ + μ²α

    Args:
        units: Dimensionality of output (typically 1 for univariate time series).
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        2D/3D tensor with shape: `(batch_size, [seq_len,] hidden_dim)`.

    Output shape:
        Tuple of two tensors with shape: `(batch_size, [seq_len,] units)`.

    Example:
        ```python
        # Create likelihood head
        likelihood_head = NegativeBinomialLikelihoodHead(units=1)

        # Compute parameters from LSTM hidden states
        hidden_states = keras.random.normal((32, 50, 128))
        mu, alpha = likelihood_head(hidden_states)

        # Use for count data forecasting
        # NB distribution can model overdispersed count data
        ```
    """

    def __init__(
            self,
            units: int = 1,
            **kwargs: Any
    ) -> None:
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
        """Build the layer by explicitly building sub-layers."""
        self.mu_projection.build(input_shape)
        self.alpha_projection.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute Negative Binomial parameters.

        Args:
            inputs: Hidden states from LSTM.

        Returns:
            Tuple of (mu, alpha) tensors.
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
        """Compute output shapes for both mu and alpha."""
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)
        return output_shape, output_shape

    def get_config(self) -> Dict[str, Any]:
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
    the next hidden state. It's designed to work with RNN layers.

    **Intent**: Provide the autoregressive computation unit that processes
    sequences in DeepAR, taking into account both temporal dependencies
    (via hidden state) and the previous actual/predicted value.

    **Architecture**:
    ```
    Inputs: [z_{t-1}, x_t, h_{t-1}]
           ↓
    Concatenate: [z_{t-1}, x_t]
           ↓
    LSTM Cell: h_t = LSTM([z_{t-1}, x_t], h_{t-1})
           ↓
    Output: h_t
    ```

    **Mathematical Operation**:
        h_t = h(h_{t-1}, z_{t-1}, x_t, Θ)

    Where h is implemented as an LSTM cell.

    Args:
        units: Number of LSTM units (hidden dimension).
        dropout: Dropout rate for LSTM. Defaults to 0.0.
        recurrent_dropout: Recurrent dropout rate for LSTM. Defaults to 0.0.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)` where input_dim
        includes both the lagged target and covariates.

    Output shape:
        2D tensor with shape: `(batch_size, units)`.

    Example:
        ```python
        # Create cell
        cell = DeepARCell(units=128, dropout=0.1)

        # Wrap in RNN for sequence processing
        rnn = keras.layers.RNN(cell, return_sequences=True)

        # Process sequence
        inputs = keras.random.normal((32, 50, 10))
        outputs = rnn(inputs)  # Shape: (32, 50, 128)
        ```
    """

    def __init__(
            self,
            units: int,
            dropout: float = 0.0,
            recurrent_dropout: float = 0.0,
            **kwargs: Any
    ) -> None:
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
        """Build the cell by building the LSTM sub-layer."""
        self.lstm_cell.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            states: Tuple[keras.KerasTensor, ...],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """
        Process one time step.

        Args:
            inputs: Input tensor at time t, shape (batch_size, input_dim).
            states: List of state tensors from previous time step.
            training: Boolean or None, whether in training mode.

        Returns:
            Tuple of (output, new_states).
        """
        output, new_states = self.lstm_cell(inputs, states, training=training)
        return output, new_states

    def get_initial_state(
            self,
            batch_size: Optional[int] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Get initial state for the cell."""
        return self.lstm_cell.get_initial_state(batch_size=batch_size)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
        })
        return config