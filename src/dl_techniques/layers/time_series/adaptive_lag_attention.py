"""
Context-aware, gated attention mechanism for autoregression.

This layer performs a dynamic, context-dependent autoregressive forecast. It
is designed to be a robust and interpretable component within a larger time
series model, allowing the model to learn when and how to rely on historical
values.

The layer's architecture separates control logic from data flow. A
``context_tensor``, typically from a deep encoder like an LSTM, acts as the
controller. It generates two distinct control signals that modulate a separate
``lag_tensor`` containing historical values:

1. **Attention Weights**: The context is passed through a dense layer with a
   ``sigmoid`` activation to produce a set of independent attention weights,
   one for each lag. These weights determine the relative importance of
   each historical value for the current time step.

2. **Master Gate**: In parallel, the context is passed through a second dense
   layer, also with a ``sigmoid`` activation, to produce a single scalar
   gate value. This gate acts as a master switch, controlling the overall
   contribution of the entire autoregressive component to the final model output.

The final output is the weighted sum of the lags, multiplicatively controlled
by the master gate. This design allows the model to learn complex temporal
strategies, such as ignoring history entirely (gate ~ 0) during anomalous
periods or focusing on specific seasonalities (high weights on corresponding
lags).

**Independent Sigmoid Attention**: Unlike the ``softmax`` function used in
Transformers, which forces a competitive probability distribution where
``sum(weights) = 1``, this layer uses a ``sigmoid`` activation. This yields
independent weights ``w_i in (0, 1)`` for each lag, allowing the model to
recognize that multiple historical points are simultaneously important (e.g.,
both 7 days ago and 365 days ago could have high weights), or conversely, that
*no* historical points are relevant.

**Multiplicative Gating**: The final output is computed as:
    ``output = g * (sum_i w_i * l_i)``
where ``g`` is the master gate, ``w_i`` are the attention weights, and ``l_i``
are the lag values.

References:
    The concept of gating to control information flow is a foundational
    principle in modern deep learning, most famously used in LSTMs and GRUs.

    - Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
      Neural Computation.
      https://www.bioinf.jku.at/publications/older/2604.pdf
    - Cho, K., et al. (2014). Learning Phrase Representations using RNN
      Encoder-Decoder for Statistical Machine Translation. In EMNLP.
      https://arxiv.org/abs/1406.1078
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

    The two key mathematical operations are:

    1. **Attention Weights**: ``w = sigma(W_a * context + b_a)`` where sigma is sigmoid
    2. **Gate Value**: ``g = sigma(W_g * context + b_g)``
    3. **Weighted Sum**: ``s = sum(w_i * lag_i)`` for i in [1, num_lags]
    4. **Final Output**: ``output = g * s``

    **Architecture Overview:**

    .. code-block:: text

        Context (batch, context_dim)          Lags (batch, num_lags)
                │                                     │
                ├──────────────┐                      │
                ▼              ▼                      │
        ┌──────────────┐ ┌──────────┐                 │
        │Dense(num_lags│ │ Dense(1) │                 │
        │  sigmoid)    │ │ sigmoid  │                 │
        └──────┬───────┘ └────┬─────┘                 │
               │              │                       │
               ▼              │                       ▼
        Attention Weights     │              ┌────────────────┐
        (batch, num_lags)     │              │  Element-wise  │
               │              │              │  Multiply      │
               └──────────────┼──► w * lags ─┘
                              │         │
                              │         ▼
                              │    ops.sum(axis=-1)
                              │         │
                              │         ▼
                              │   Weighted Sum (batch,)
                              │         │
                              ▼         ▼
                         Gate (batch,1) │
                              │         │
                              ▼         ▼
                        ┌───────────────────┐
                        │  g * weighted_sum │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        Output (batch, 1)

    :param num_lags: The number of past time series values (lags) to consider.
        This must match the last dimension of the lag input tensor. Must be positive.
    :type num_lags: int
    :param kernel_initializer: Initializer for the weight-generating sublayers.
        Defaults to ``"glorot_uniform"``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for the bias of the sublayers.
        Defaults to ``"zeros"``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the kernel weights of sublayers.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for the bias vectors of sublayers.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param activity_regularizer: Optional regularizer function for the output.
    :type activity_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If num_lags is not a positive integer.
    :raises ValueError: If input format is incorrect during call.
    :raises ValueError: If lag tensor's last dimension doesn't match num_lags during build.
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
        """
        Initialize the AdaptiveLagAttentionLayer.

        :param num_lags: Number of past time series values (lags) to consider.
        :type num_lags: int
        :param kernel_initializer: Initializer for kernel weights.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param bias_initializer: Initializer for bias vectors.
        :type bias_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Optional regularizer for kernel weights.
        :type kernel_regularizer: str or keras.regularizers.Regularizer or None
        :param bias_regularizer: Optional regularizer for bias vectors.
        :type bias_regularizer: str or keras.regularizers.Regularizer or None
        :param activity_regularizer: Optional regularizer for the output.
        :type activity_regularizer: str or keras.regularizers.Regularizer or None
        :param kwargs: Additional keyword arguments for the Layer parent class.
        """
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

        Explicitly builds each sub-layer for robust serialization.

        :param input_shape: A list of two tuples representing the shapes of
            the context tensor and lag tensor inputs.
        :type input_shape: list[tuple[int or None, ...]]
        :raises ValueError: If input_shape is not a list of two tensors.
        :raises ValueError: If lag tensor's last dimension doesn't match num_lags.
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

        :param inputs: A list containing two tensors: inputs[0] is the context
            tensor of shape ``(batch_size, context_dim)``, and inputs[1] is
            the lag values tensor of shape ``(batch_size, num_lags)``.
        :type inputs: list[keras.KerasTensor]
        :param training: Whether the layer should behave in training mode.
        :type training: bool or None
        :return: The predicted value tensor with shape ``(batch_size, 1)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If inputs is not a list of exactly two tensors.
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

        :param input_shape: A list of two tuples representing the input shapes.
        :type input_shape: list[tuple[int or None, ...]]
        :return: The output shape ``(batch_size, 1)``.
        :rtype: tuple[int or None, ...]
        :raises ValueError: If input_shape is not a list of two tuples.
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
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration with all init parameters.
        :rtype: dict[str, Any]
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
