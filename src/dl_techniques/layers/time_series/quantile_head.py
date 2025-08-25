"""
TiRex-inspired time series forecasting components for Keras.

This module implements time series forecasting layers inspired by the TiRex architecture,
adapted to work with Keras and our project's available components.
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class QuantileHead(keras.layers.Layer):
    """
    Quantile prediction head for probabilistic time series forecasting.

    This layer implements a neural network head for predicting multiple quantiles
    of a time series distribution, enabling probabilistic forecasting with uncertainty
    quantification. It takes encoded features and projects them to quantile predictions
    across a specified forecast horizon.

    **Intent**: Enable probabilistic time series forecasting by predicting multiple
    quantiles (e.g., 10th, 50th, 90th percentiles) simultaneously, providing both
    point predictions and uncertainty estimates essential for robust forecasting
    applications.

    **Architecture**:
    ```
    Input(shape=[..., feature_dim])
           ↓
    Dropout(rate=dropout_rate) ← (optional, if dropout_rate > 0)
           ↓
    Dense(num_quantiles × output_length, activation=None)
           ↓
    Reshape(shape=[batch, num_quantiles, output_length])
           ↓
    Output(shape=[batch, num_quantiles, output_length])
    ```

    **Mathematical Operation**:
        For each quantile τ ∈ {τ₁, τ₂, ..., τₖ}:
        quantile_τ = W_τ @ features + b_τ

    Where:
    - W_τ, b_τ are learnable parameters for quantile τ
    - Output represents predicted values at different probability levels
    - Common quantiles: [0.1, 0.5, 0.9] for 80% prediction intervals

    **Use Cases**:
    - Financial forecasting with risk assessment
    - Weather prediction with confidence intervals
    - Demand forecasting with supply planning margins
    - Any scenario requiring uncertainty quantification

    Args:
        num_quantiles: Integer, number of quantiles to predict simultaneously.
            Common choices: 3 for [0.1, 0.5, 0.9], 5 for [0.05, 0.25, 0.5, 0.75, 0.95].
            Must be positive.
        output_length: Integer, length of the forecast horizon (number of future steps).
            Must be positive. This determines how many time steps ahead to predict.
        dropout_rate: Float between 0 and 1, fraction of features randomly set to 0
            during training for regularization. When 0, no dropout is applied.
            Defaults to 0.1.
        use_bias: Boolean, whether to include learnable bias terms in the projection.
            Defaults to True.
        kernel_initializer: String or Initializer instance for projection layer weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer instance for projection layer biases.
            Only used when use_bias=True. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., feature_dim)`.
        Most common: 2D tensor `(batch_size, feature_dim)` from encoder output.

    Output shape:
        3D tensor with shape: `(batch_size, num_quantiles, output_length)`.
        Each [i, j, k] represents the j-th quantile prediction for the k-th future
        time step for the i-th sample in the batch.

    Attributes:
        projection: Dense layer that projects features to flattened quantile predictions.
        dropout: Dropout layer for regularization (None if dropout_rate=0).

    Example:
        ```python
        # Basic quantile head for 3 quantiles, 24-hour forecast
        head = QuantileHead(num_quantiles=3, output_length=24)
        features = keras.Input(shape=(256,))  # From encoder
        quantiles = head(features)  # Shape: (batch, 3, 24)

        # With higher dropout for regularization
        head = QuantileHead(
            num_quantiles=5,
            output_length=48,
            dropout_rate=0.2
        )

        # In a complete forecasting model
        encoder_output = encoder(time_series_input)
        quantile_predictions = QuantileHead(
            num_quantiles=9,  # Deciles
            output_length=forecast_horizon
        )(encoder_output)

        # Extract median (50th percentile) predictions
        median_forecast = quantile_predictions[:, num_quantiles//2, :]
        ```

    Note:
        This layer outputs raw quantile predictions. During training, these should
        be used with quantile loss functions (e.g., pinball loss) that enforce
        the proper ordering of quantiles. The layer itself does not enforce
        quantile ordering constraints.

    References:
        - Quantile Regression: Koenker, R. & Bassett Jr, G. (1978)
        - Neural Network Quantile Regression: Taylor, J. W. (2000)
        - TiRex Architecture: Time series forecasting with quantile predictions
    """

    def __init__(
        self,
        num_quantiles: int,
        output_length: int,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_quantiles <= 0:
            raise ValueError(f"num_quantiles must be positive, got {num_quantiles}")
        if output_length <= 0:
            raise ValueError(f"output_length must be positive, got {output_length}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.num_quantiles = num_quantiles
        self.output_length = output_length
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.projection = keras.layers.Dense(
            units=self.num_quantiles * self.output_length,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="quantile_projection"
        )

        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="quantile_dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        # Build sub-layers in computational order
        if self.dropout is not None:
            self.dropout.build(input_shape)

        self.projection.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Predict quantiles from the input feature vector.

        Args:
            inputs: Input tensor with encoded features.
            training: Boolean indicating training mode for dropout.

        Returns:
            Quantile predictions tensor of shape (batch_size, num_quantiles, output_length).
        """
        x = inputs

        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Project features to flattened quantile predictions
        quantile_preds = self.projection(x, training=training)

        # Reshape to [batch_size, num_quantiles, output_length]
        # Using -1 for batch dimension handles dynamic batch sizes
        quantiles = ops.reshape(
            quantile_preds,
            (-1, self.num_quantiles, self.output_length)
        )

        return quantiles

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        return (batch_size, self.num_quantiles, self.output_length)

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_quantiles": self.num_quantiles,
            "output_length": self.output_length,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

# ---------------------------------------------------------------------