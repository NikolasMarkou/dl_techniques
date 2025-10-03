"""
A context-aware, gated attention mechanism for autoregression.

This layer performs a dynamic, context-dependent autoregressive forecast. It
is designed to be a robust and interpretable component within a larger time
series model, allowing the model to learn when and how to rely on historical
values.

Architecture and Design Philosophy:
The layer's architecture is based on separating the control logic from the
data flow. A `context_tensor`, typically from a deep encoder like an LSTM,
acts as the controller. It does not directly participate in the forecast but
instead generates two distinct control signals that modulate a separate
`lag_tensor` containing historical values.

1.  **Attention Weights**: The context is passed through a dense layer with a
    `sigmoid` activation to produce a set of independent attention weights,
    one for each lag. These weights determine the relative importance of
    each historical value for the current time step.

2.  **Master Gate**: In parallel, the context is passed through a second dense
    layer, also with a `sigmoid` activation, to produce a single scalar
    gate value. This gate acts as a master switch, controlling the overall
    contribution of the entire autoregressive component to the final model output.

The final output is the weighted sum of the lags, multiplicatively controlled
by the master gate. This design allows the model to learn complex temporal
strategies, such as ignoring history entirely (gate ≈ 0) during anomalous
periods or focusing on specific seasonalities (high weights on corresponding
lags).

Foundational Mathematics:
The two key mathematical choices distinguish this layer from standard
attention mechanisms and provide its flexibility:

1.  **Independent Sigmoid Attention**: Unlike the `softmax` function used in
    Transformers, which forces a competitive probability distribution where
    `sum(weights) = 1`, this layer uses a `sigmoid` activation. This yields
    independent weights `wᵢ ∈ (0, 1)` for each lag. This is a critical
    distinction: it allows the model to recognize that multiple historical
    points are simultaneously important (e.g., both 7 days ago and 365 days
    ago could have high weights), or conversely, that *no* historical points
    are relevant. `softmax` would be forced to assign high weights to some
    lags even if all are irrelevant.

2.  **Multiplicative Gating**: The final output is computed as:
    `output = g * (Σᵢ wᵢ * lᵢ)`
    where `g` is the master gate, `wᵢ` are the attention weights, and `lᵢ` are
    the lag values. The gate `g` provides a mechanism to learn the utility of
    the entire autoregressive feature. If the context suggests that history
    is not predictive (e.g., due to a structural break), the model can learn
    to set `g` close to zero, effectively nullifying the layer's output and
    allowing other parts of the model to dominate the forecast.

References:
    The concept of gating to control information flow is a foundational
    principle in modern deep learning, most famously used in LSTMs and GRUs.
    - [Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
      Neural Computation.](
      https://www.bioinf.jku.at/publications/older/2604.pdf)
    - [Cho, K., et al. (2014). Learning Phrase Representations using RNN
      Encoder-Decoder for Statistical Machine Translation. In EMNLP.](
      https://arxiv.org/abs/1406.1078)
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
class AdaptiveLagAttentionLayer(keras.layers.Layer):
    """
    Advanced attention layer for dynamically weighting temporal lags with gating control.

    This layer uses a context tensor to generate independent attention weights
    and a master gate for a set of provided lag values. The attention mechanism
    uses sigmoid activation to allow independent weighting of multiple lags,
    while a master gate controls the overall contribution of the autoregressive
    component.

    **Intent**: Enable sophisticated time series forecasting by learning to
    dynamically weight historical values based on context, with explicit control
    over when to rely on autoregressive information versus other features.

    **Architecture**:
    ```
    Context(shape=[batch, context_dim]) ┌─────────┐ Lags(shape=[batch, num_lags])
                    ↓                   │         │             ↓
    Dense(num_lags, sigmoid) → Attention Weights  │     (no processing)
                    ↓                   │         │             ↓
    Dense(1, sigmoid) → Gate Value      │         │    Element-wise Multiply
                    ↓                   │         │             ↓
              Master Gating ←───────────┴─────────┘    Weighted Lags Sum
                    ↓                                           ↓
                Final Output ←─────── Element-wise Multiply ────┘
                    ↓
    Output(shape=[batch, 1])
    ```

    **Mathematical Operations**:
    1. **Attention Weights**: w = σ(W_a × context + b_a) where σ is sigmoid
    2. **Gate Value**: g = σ(W_g × context + b_g)
    3. **Weighted Sum**: s = Σ(w_i × lag_i) for i in [1, num_lags]
    4. **Final Output**: output = g × s

    This design allows the model to:
    - Attend to multiple important lags simultaneously (sigmoid vs softmax)
    - Completely shut off autoregressive behavior when not useful (master gate)
    - Handle multi-scale temporal patterns through dilated lag support

    Args:
        num_lags: Integer, the number of past time series values (lags) to consider.
            This must match the last dimension of the lag input tensor. Must be positive.
        kernel_initializer: String name of initializer or initializer instance
            for the weight-generating sublayers. Defaults to "glorot_uniform".
        bias_initializer: String name of initializer or initializer instance
            for the bias of the sublayers. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for the kernel weights of sublayers.
            Can be string name or regularizer instance.
        bias_regularizer: Optional regularizer for the bias vectors of sublayers.
            Can be string name or regularizer instance.
        activity_regularizer: Optional regularizer function for the output.
            Can be string name or regularizer instance.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        A list of two tensors:
        - Context tensor: A 2D tensor with shape `(batch_size, context_dim)`
        - Lag tensor: A 2D tensor with shape `(batch_size, num_lags)`

    Output shape:
        A 2D tensor with shape `(batch_size, 1)`

    Attributes:
        attention_generator: Dense layer that maps context to attention weights.
        gate_generator: Dense layer that maps context to master gate value.

    Examples:
        ```python
        # Standard lags example
        num_lags = 30
        context_input = keras.Input(shape=(64,), name="context_input")
        lag_input = keras.Input(shape=(num_lags,), name="lag_input")
        layer = AdaptiveLagAttentionLayer(num_lags=num_lags)
        output = layer([context_input, lag_input])
        model = keras.Model(inputs=[context_input, lag_input], outputs=output)

        # Dilated lags for multi-scale analysis
        dilated_lags = [1, 7, 14, 30]  # Daily, weekly, bi-weekly, monthly
        num_dilated_lags = len(dilated_lags)
        context_input = keras.Input(shape=(128,), name="context_input")
        lag_input = keras.Input(shape=(num_dilated_lags,), name="lag_input")
        layer = AdaptiveLagAttentionLayer(
            num_lags=num_dilated_lags,
            kernel_regularizer='l2',
            activity_regularizer='l1'
        )
        output = layer([context_input, lag_input])

        # In a complete forecasting model
        inputs = [
            keras.Input(shape=(64,), name="context"),
            keras.Input(shape=(12,), name="lags")
        ]
        x = AdaptiveLagAttentionLayer(num_lags=12)(inputs)
        outputs = keras.layers.Dense(1, activation='linear', name='forecast')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        ```

    Raises:
        ValueError: If num_lags is not a positive integer.
        ValueError: If input format is incorrect during call.
        ValueError: If lag tensor's last dimension doesn't match num_lags during build.

    Note:
        This implementation follows modern Keras 3 patterns where sub-layers are
        created in __init__ and built explicitly in build() for robust serialization.
        The layer is designed to handle variable batch sizes and integrates seamlessly
        with Keras training pipelines.
    """

    def __init__(
        self,
        num_lags: int,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the AdaptiveLagAttentionLayer."""
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(num_lags, int) or num_lags <= 0:
            raise ValueError(f"num_lags must be a positive integer, got {num_lags}")

        # Store ALL configuration parameters
        self.num_lags = num_lags
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Sublayer for attention weights: maps context -> independent weights
        self.attention_generator = keras.layers.Dense(
            units=self.num_lags,
            activation='sigmoid',  # Sigmoid for independent [0, 1] weights
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='attention_generator'
        )

        # Sublayer for master gate: maps context -> single gate value
        self.gate_generator = keras.layers.Dense(
            units=1,
            activation='sigmoid',  # Sigmoid for a [0, 1] gate value
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='gate_generator'
        )

        logger.debug(f"AdaptiveLagAttentionLayer initialized with num_lags={num_lags}")

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer and all its sub-layers.

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
                "This layer expects a list of two inputs: [context_tensor, lag_tensor]. "
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

        # Build sub-layers in computational order
        self.attention_generator.build(context_shape)
        self.gate_generator.build(context_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug(f"AdaptiveLagAttentionLayer built with context_shape={context_shape}, lag_shape={lag_shape}")

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        Args:
            inputs: A list containing two tensors:
                - inputs[0]: The context tensor, shape `(batch_size, context_dim)`.
                - inputs[1]: The lag values tensor, shape `(batch_size, num_lags)`.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            The predicted value tensor with shape `(batch_size, 1)`.

        Raises:
            ValueError: If inputs is not a list of exactly two tensors.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Expected a list of two inputs, got {type(inputs)} with length "
                f"{len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        context_tensor, lag_tensor = inputs

        # 1. Generate independent attention weights from the context
        # Shape: (batch_size, num_lags)
        attention_weights = self.attention_generator(context_tensor, training=training)

        # 2. Generate the master gate value from the context
        # Shape: (batch_size, 1)
        gate = self.gate_generator(context_tensor, training=training)

        # 3. Compute the weighted sum of the lags
        # Shape: (batch_size,)
        weighted_sum_of_lags = ops.sum(attention_weights * lag_tensor, axis=-1)

        # 4. Apply the master gate
        # Shape: (batch_size,)
        gated_output = ops.squeeze(gate, axis=-1) * weighted_sum_of_lags

        # 5. Reshape for a consistent output shape
        # Shape: (batch_size, 1)
        output = ops.expand_dims(gated_output, axis=-1)

        # Apply activity regularization if specified
        if self.activity_regularizer is not None:
            self.add_loss(self.activity_regularizer(output))

        return output

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: A list of two tuples representing the input shapes.

        Returns:
            A tuple representing the output shape (batch_size, 1).

        Raises:
            ValueError: If input_shape is not a list of two tuples.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Expected input_shape to be a list of two tuples, "
                f"got {type(input_shape)} with length "
                f"{len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        context_shape, _ = input_shape

        # Output shape: (batch_size, 1)
        return (context_shape[0], 1)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration with ALL __init__ parameters.
        """
        config = super().get_config()
        config.update({
            "num_lags": self.num_lags,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------