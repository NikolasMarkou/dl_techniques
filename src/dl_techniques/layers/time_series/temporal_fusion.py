"""
Temporal Fusion Layer for Advanced Time Series Forecasting

This layer provides a sophisticated mechanism for fusing information from a deep
contextual model (like an LSTM) with a dynamic autoregressive model based on
temporal lags. It is designed for robustness, interpretability, and flexibility.

Theory:
    The layer operates on two parallel forecasting pathways which are then
    intelligently blended:

    1. Contextual Pathway: A rich `context_tensor` (e.g., the output of an
       LSTM) is passed through a dense layer to produce a `context_forecast`.
       This represents the model's understanding of the current state, including
       any exogenous variables or complex patterns.

    2. Autoregressive Pathway: The same `context_tensor` is used to generate:
       a) Independent attention weights (via sigmoid) for a set of historical
          `lag_tensor` values.
       b) A `lag_forecast` is computed from the attention-weighted sum of these lags.

    3. Fusion Gating: The context also generates a scalar `fusion_gate` (g)
       between 0 and 1. This gate dynamically interpolates between the two
       forecasts, deciding how much to trust each pathway for the final prediction:

       `Final Output = (1 - g) * context_forecast + g * lag_forecast`

    This architecture allows the model to learn complex strategies, such as relying
    on historical patterns during stable periods (gate ≈ 1) but switching to a
    context-driven forecast during anomalous events (gate ≈ 0).

Applications:
    - Financial forecasting (cash flow, revenue) requiring both trend-following
      and adaptation to market shocks.
    - Demand forecasting with complex seasonalities and promotions.
    - Any sequence modeling task where a blend of deep feature extraction and
      simple autoregression is beneficial.
"""

import keras
from keras import ops
from typing import List, Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TemporalFusionLayer(keras.layers.Layer):
    """A layer that fuses a context-based forecast with an attention-based autoregressive forecast.

    This layer implements a sophisticated temporal fusion mechanism that combines
    deep contextual understanding with dynamic autoregressive modeling. It operates
    on two parallel forecasting pathways that are intelligently blended using a
    learned fusion gate.

    **Architecture & Flow**:
    ```
    Inputs: [context_tensor, lag_tensor]
             ↓
    Context → Dense(attention_weights) → sigmoid
             ↓
    Context → Dense(fusion_gate) → sigmoid
             ↓
    Context → Dense(context_forecast)
             ↓
    Lags → [Optional: Dense(projection)] → Attention-weighted sum → Dense(lag_forecast)
             ↓
    Output = (1 - gate) * context_forecast + gate * lag_forecast
    ```

    **Mathematical Operations**:
    1. **Attention**: α = sigmoid(Dense_att(context))
    2. **Gate**: g = sigmoid(Dense_gate(context))
    3. **Context Path**: fc = Dense_ctx(context)
    4. **AR Path**: fl = Dense_lag(sum(α ⊙ lags))
    5. **Fusion**: output = (1 - g) ⊙ fc + g ⊙ fl

    Args:
        output_dim: Integer, the dimensionality of the final output forecast.
            Must be positive.
        num_lags: Integer, the number of past time series values (lags) to consider.
            Must be positive.
        project_lags: Boolean, if True, an internal Dense layer will transform the
            raw lag values into a richer feature space before attention is applied.
            Defaults to False.
        kernel_initializer: String name of initializer or initializer instance
            for all internal sublayers. Defaults to "glorot_uniform".
        bias_initializer: String name of initializer or initializer instance
            for the bias vectors of sublayers. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for the kernel weights of all sublayers.
        bias_regularizer: Optional regularizer for the bias vectors of all sublayers.
        activity_regularizer: Optional regularizer function for the output.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        A list of two tensors:
        - Context tensor: A 2D tensor with shape `(batch_size, context_dim)`
        - Lag tensor: A 2D tensor with shape `(batch_size, num_lags)`

    Output shape:
        A 2D tensor with shape `(batch_size, output_dim)`

    Attributes:
        attention_generator: Dense layer generating attention weights for lags.
        gate_generator: Dense layer generating fusion gate values.
        context_forecaster: Dense layer creating forecast from context.
        lag_projector: Optional Dense layer for lag feature projection.
        lag_forecaster: Dense layer creating forecast from weighted lags.

    Returns:
        A tensor representing the final fused forecast combining contextual and
        autoregressive pathways.

    Raises:
        ValueError: If output_dim or num_lags is not a positive integer.
        ValueError: If inputs is not a list of two tensors during call.
        ValueError: If lag tensor's last dimension doesn't match num_lags.

    Examples:
        ```python
        # Basic usage with LSTM context
        context_dim = 128
        num_lags = 10
        output_dim = 1

        context_input = keras.Input(shape=(context_dim,), name="context_input")
        lag_input = keras.Input(shape=(num_lags,), name="lag_input")

        fusion_layer = TemporalFusionLayer(
            output_dim=output_dim,
            num_lags=num_lags,
            project_lags=True
        )
        output = fusion_layer([context_input, lag_input])
        model = keras.Model(inputs=[context_input, lag_input], outputs=output)

        # Multi-dimensional forecasting
        multi_output = TemporalFusionLayer(
            output_dim=5,  # Forecast 5 variables
            num_lags=20,
            project_lags=False
        )([context_input, keras.Input(shape=(20,))])
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and explicitly built in build() for proper serialization.
        The fusion gate enables dynamic switching between context-driven and lag-driven
        forecasting strategies.
    """

    def __init__(
        self,
        output_dim: int,
        num_lags: int,
        project_lags: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the TemporalFusionLayer."""
        super().__init__(**kwargs)

        # Validate parameters
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(num_lags, int) or num_lags <= 0:
            raise ValueError(f"num_lags must be a positive integer, got {num_lags}")

        # Store configuration parameters
        self.output_dim = output_dim
        self.num_lags = num_lags
        self.project_lags = project_lags
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (modern pattern)
        # --- Control Pathway Sublayers (driven by context) ---
        self.attention_generator = keras.layers.Dense(
            units=self.num_lags,
            activation='sigmoid',
            name='attention_generator',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        self.gate_generator = keras.layers.Dense(
            units=self.output_dim,
            activation='sigmoid',
            name='gate_generator',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # --- Contextual Forecast Pathway Sublayer ---
        self.context_forecaster = keras.layers.Dense(
            units=self.output_dim,
            name='context_forecaster',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # --- Autoregressive Forecast Pathway Sublayers ---
        if self.project_lags:
            # Optional layer to enrich the lag features
            self.lag_projector = keras.layers.Dense(
                units=self.num_lags,
                activation='relu',
                name='lag_projector',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )
        else:
            self.lag_projector = None

        # This layer creates the final AR forecast from the weighted sum
        self.lag_forecaster = keras.layers.Dense(
            units=self.output_dim,
            name='lag_forecaster',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        logger.debug(f"TemporalFusionLayer initialized with output_dim={output_dim}, num_lags={num_lags}")

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer weights and sublayers based on input shape.

        CRITICAL: Explicitly build each sub-layer for robust serialization.

        Args:
            input_shape: A list of two tuples representing the shapes of
                the context tensor and lag tensor inputs.

        Raises:
            ValueError: If input_shape is not a list of two tensors.
            ValueError: If lag tensor's last dimension doesn't match num_lags.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "TemporalFusionLayer expects a list of two inputs: "
                "[context_tensor, lag_tensor]. "
                f"Received input_shape: {input_shape}"
            )

        context_shape, lag_shape = input_shape

        # Validate shapes
        if len(context_shape) < 2:
            raise ValueError(
                f"Context tensor must be at least 2D, got shape: {context_shape}"
            )
        if len(lag_shape) < 2 or lag_shape[-1] != self.num_lags:
            raise ValueError(
                f"The last dimension of the lag_tensor input ({lag_shape[-1] if len(lag_shape) >= 2 else 'unknown'}) "
                f"does not match `num_lags` ({self.num_lags})."
            )

        # Build sub-layers in order they'll be used
        # All context-driven layers use the same context shape
        self.attention_generator.build(context_shape)
        self.gate_generator.build(context_shape)
        self.context_forecaster.build(context_shape)

        # Lag processing layers
        if self.lag_projector is not None:
            self.lag_projector.build(lag_shape)

        # The lag_forecaster receives a weighted sum, which is (batch_size, 1)
        weighted_sum_shape = (context_shape[0], 1)
        self.lag_forecaster.build(weighted_sum_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug(f"TemporalFusionLayer built with context_shape={context_shape}, lag_shape={lag_shape}")

    def call(self, inputs: List[keras.KerasTensor], training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the temporal fusion mechanism.

        Args:
            inputs: A list containing two tensors:
                - inputs[0]: The context tensor, shape `(batch_size, context_dim)`.
                - inputs[1]: The lag values tensor, shape `(batch_size, num_lags)`.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            The final fused forecast tensor with shape `(batch_size, output_dim)`.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Expected a list of two inputs, got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        context_tensor, lag_tensor = inputs

        # --- Pathway 1: Contextual Forecast ---
        context_forecast = self.context_forecaster(context_tensor, training=training)

        # --- Pathway 2: Autoregressive Forecast ---
        # Generate attention weights and the fusion gate from the context
        attention_weights = self.attention_generator(context_tensor, training=training)
        fusion_gate = self.gate_generator(context_tensor, training=training)

        # Optionally project lags into a richer feature space
        if self.lag_projector is not None:
            processed_lags = self.lag_projector(lag_tensor, training=training)
        else:
            processed_lags = lag_tensor

        # Calculate the weighted sum of lags (attention mechanism)
        weighted_sum = ops.sum(attention_weights * processed_lags, axis=-1, keepdims=True)
        lag_forecast = self.lag_forecaster(weighted_sum, training=training)

        # --- Pathway 3: Fusion ---
        # Blend the two forecasts using the learned gate
        final_forecast = (
            (1.0 - fusion_gate) * context_forecast +
            fusion_gate * lag_forecast
        )

        # Apply activity regularization if specified
        if self.activity_regularizer is not None:
            self.add_loss(self.activity_regularizer(final_forecast))

        return final_forecast

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: A list of two tuples representing the input shapes.

        Returns:
            A tuple representing the output shape.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Expected input_shape to be a list of two tuples, "
                f"got {type(input_shape)} with length {len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        context_shape, _ = input_shape

        # Convert to list for manipulation, then back to tuple
        output_shape = list(context_shape)
        output_shape[-1] = self.output_dim

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_lags": self.num_lags,
            "project_lags": self.project_lags,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------