"""
Adaptive Lag Attention Layer

This layer improves upon a simple dynamic lag model by incorporating more robust
attention and gating mechanisms. It calculates a forecast by attending to a set
of historical values (lags), which can be standard or dilated.

Key Features & Improvements:
    - Independent Attention Weights: Uses a sigmoid activation instead of softmax.
      This allows the model to assign high importance to multiple lags simultaneously
      or to ignore all lags if necessary, avoiding the "forced competition" of softmax.
    - Explicit Gating: A separate gate neuron learns to control the overall
      contribution of the autoregressive component. This allows the model to
      explicitly "turn off" its reliance on past values when they are not predictive.
    - Multi-Scale Support: Natively supports dilated lags, enabling the model to
      attend to different time scales (e.g., daily, weekly, monthly) within a
      single layer.

Theory:
    For each input, the layer receives a context vector and a set of lag values.
    It then computes:

    1. Attention Weights (w): A set of independent weights between 0 and 1, one
       for each lag, determined by the context.
       w_i = sigmoid(f_w(context))
    2. Gating Value (g): A single scalar value between 0 and 1 that controls the
       overall influence of the historical data.
       g = sigmoid(f_g(context))
    3. Weighted Lags: The weighted sum of the historical values.
       s = Σ w_i * y_{t-i}
    4. Gated Output: The final prediction is the gated weighted sum.
       prediction = g * s

Applications:
    - Robust financial forecasting where the importance of history can vary dramatically.
    - Modeling time series with complex, multiple, and non-stationary seasonalities.
    - Any scenario where a model needs to dynamically decide whether to be
      autoregressive or to rely on other exogenous features.
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
    """An advanced attention layer for dynamically weighting temporal lags.

    This layer uses a context tensor to generate independent attention weights
    and a master gate for a set of provided lag values. The attention mechanism
    uses sigmoid activation to allow independent weighting of multiple lags,
    while a master gate controls the overall contribution of the autoregressive
    component.

    Args:
        num_lags: Integer, the number of past time series values (lags) to consider.
            This must match the last dimension of the lag input tensor.
        kernel_initializer: String name of initializer or initializer instance
            for the weight-generating sublayers. Defaults to "glorot_uniform".
        bias_initializer: String name of initializer or initializer instance
            for the bias of the sublayers. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for the kernel weights of sublayers.
        bias_regularizer: Optional regularizer for the bias vectors of sublayers.
        activity_regularizer: Optional regularizer function for the output.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        A list of two tensors:
        - Context tensor: A 2D tensor with shape `(batch_size, context_dim)`
        - Lag tensor: A 2D tensor with shape `(batch_size, num_lags)`

    Output shape:
        A 2D tensor with shape `(batch_size, 1)`

    Returns:
        A tensor representing the gated weighted sum of the lag values.

    Raises:
        ValueError: If num_lags is not a positive integer.
        ValueError: If input_shape is not a list of two tensors.
        ValueError: If lag tensor's last dimension doesn't match num_lags.

    Examples:
        >>> # Standard lags example
        >>> num_lags = 30
        >>> context_input = keras.Input(shape=(64,), name="context_input")
        >>> lag_input = keras.Input(shape=(num_lags,), name="lag_input")
        >>> output = AdaptiveLagAttentionLayer(num_lags=num_lags)([context_input, lag_input])
        >>> model = keras.Model(inputs=[context_input, lag_input], outputs=output)
        >>>
        >>> # Dilated lags for multi-scale analysis
        >>> dilated_lags = [1, 7, 14, 30]  # Daily, weekly, bi-weekly, monthly
        >>> num_dilated_lags = len(dilated_lags)
        >>> context_input_dilated = keras.Input(shape=(64,), name="context_input")
        >>> lag_input_dilated = keras.Input(shape=(num_dilated_lags,), name="lag_input")
        >>> output_dilated = AdaptiveLagAttentionLayer(num_lags=num_dilated_lags)(
        ...     [context_input_dilated, lag_input_dilated]
        ... )
        >>> model_dilated = keras.Model(
        ...     inputs=[context_input_dilated, lag_input_dilated],
        ...     outputs=output_dilated
        ... )
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

        if not isinstance(num_lags, int) or num_lags <= 0:
            raise ValueError(f"num_lags must be a positive integer, got {num_lags}")

        # Store configuration parameters
        self.num_lags = num_lags
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Will be initialized in build()
        self.attention_generator = None
        self.gate_generator = None
        self._build_input_shape = None

        logger.debug(f"AdaptiveLagAttentionLayer initialized with num_lags={num_lags}")

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer weights and sublayers based on input shape.

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
        self._build_input_shape = input_shape

        # Validate lag shape
        if len(lag_shape) < 2 or lag_shape[-1] != self.num_lags:
            raise ValueError(
                f"The last dimension of the lag_tensor input ({lag_shape[-1] if len(lag_shape) >= 2 else 'unknown'}) "
                f"does not match `num_lags` ({self.num_lags})."
            )

        # Validate context shape
        if len(context_shape) < 2:
            raise ValueError(
                f"Context tensor must be at least 2D, got shape: {context_shape}"
            )

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

        # Build sublayers
        self.attention_generator.build(context_shape)
        self.gate_generator.build(context_shape)

        super().build(input_shape)
        logger.debug(f"AdaptiveLagAttentionLayer built with context_shape={context_shape}, lag_shape={lag_shape}")

    def call(self, inputs: List[keras.KerasTensor], training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: A list containing two tensors:
                - inputs[0]: The context tensor, shape `(batch_size, context_dim)`.
                - inputs[1]: The lag values tensor, shape `(batch_size, num_lags)`.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            The predicted value tensor with shape `(batch_size, 1)`.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Expected a list of two inputs, got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
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
        context_shape_list = list(context_shape)
        output_shape_list = context_shape_list[:-1] + [1]

        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a build configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AdaptiveLagAttentionLayer":
        """Creates a layer from its configuration.

        Args:
            config: Dictionary containing the layer configuration.

        Returns:
            An AdaptiveLagAttentionLayer instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------
