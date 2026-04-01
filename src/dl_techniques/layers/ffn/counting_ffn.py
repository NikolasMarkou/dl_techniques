"""
A Feed-Forward Network that learns to count features in a sequence.

This layer provides a mechanism to augment token representations with explicit
information about feature frequencies within a sequence. It operates on the
principle of "soft counting," where the model first learns to identify
semantically meaningful, countable "events" and then aggregates their
occurrences to inform each token's final representation. This is particularly
useful for tasks where understanding feature repetition, enumeration, or
relative position is critical.

Architectural Overview:
The layer's architecture consists of three conceptual stages:

1.  **Event Identification**: A "key" projection (`key_projection`) with a
    sigmoid activation is applied to each input token. This projection learns
    to identify a set of `count_dim` distinct, abstract features. The sigmoid
    output for each feature can be interpreted as a soft probability or a
    degree of presence for that "event" at that specific token position.

2.  **Count Aggregation**: The identified events are aggregated across the
    sequence dimension. The `counting_scope` parameter dictates the nature of
    this aggregation, allowing the layer to capture different types of
    contextual frequency information:
    - 'global': A simple sum across the entire sequence. Every token receives
      the same total count for each feature, providing a global summary.
    - 'causal': A cumulative sum (prefix scan) from the beginning of the
      sequence. This is suitable for autoregressive tasks, as each token's
      count only includes information from past and present positions.
    - 'local': A bidirectional cumulative sum. This is achieved by
      concatenating a forward cumulative sum with a backward (reversed)
      cumulative sum, providing each token with a rich positional signal based
      on counts before and after it.

3.  **Gated Integration**: The aggregated counts are first passed through a
    trainable linear transformation (`count_transform`) with a non-linearity.
    This projects the raw counts into a meaningful feature space. A learned
    gating mechanism then dynamically blends this count-derived information
    with the original input sequence. If the layer's `output_dim` matches the
    `input_dim`, this blending takes the form of a residual connection, where
    the gate controls the interpolation between the input and the transformed
    counts. Otherwise, the gate simply scales the transformed counts.

Foundational Mathematics:
Let `x_t` be the input vector for a token at position `t`.

1.  The "event" vector `k_t` is computed as:
    `k_t = sigmoid(W_k @ x_t + b_k)`
    where `W_k` and `b_k` are the weights and bias of the key projection. Each
    element of `k_t` represents the soft occurrence of a specific feature.

2.  The aggregated count vector `C_t` depends on the scope:
    - Global: `C_t = sum_{i=1 to T} k_i`
    - Causal: `C_t = sum_{i=1 to t} k_i`
    - Local: `C_t = concat[sum_{i=1 to t} k_i, sum_{i=t to T} k_i]`

3.  The final output `y_t` is produced by a gated update. First, the gate
    `g_t` and transformed counts `C'_t` are calculated:
    `g_t = sigmoid(W_g @ x_t + b_g)`
    `C'_t = activation(W_c @ C_t + b_c)`

    The final output is a gated mixture. For the residual case (`output_dim`
    == `input_dim`):
    `y_t = (1 - g_t) * x_t + g_t * C'_t`

This formulation allows the network to learn not only *what* to count but
also *how* to use those counts to refine its understanding of the sequence.

References:
This layer synthesizes several established concepts in deep learning. The
gating mechanism is inspired by its successful use in recurrent architectures
like LSTMs and GRUs for controlling information flow. The use of cumulative
sums (prefix scans) as a primitive for sequence modeling has been explored in
modern, non-attentional architectures designed for long-range dependency
modeling.

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural
  Computation.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for
  Image Recognition. CVPR.
- Poli, M., et al. (2023). Hyena Hierarchy: Towards Larger Convolutional
  Language Models. ICML.

"""

import keras
from typing import Literal, Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CountingFFN(keras.layers.Layer):
    """
    Feed-Forward Network that learns to count events in a sequence.

    This layer identifies "countable" features via a sigmoid key projection
    (``k_t = sigmoid(W_k @ x_t)``), aggregates their counts according to the
    ``counting_scope`` ('global' sum, 'causal' cumsum, or 'local' bidirectional
    cumsum), transforms the counts through a dense layer with activation, and
    integrates count information back into each token via a learned gate
    producing residual-style blending when ``output_dim == input_dim``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │ Input (batch, seq_len, input_dim)     │
        └──────────────────┬───────────────────┘
                           │
                     ┌─────┴─────┐
                     ▼           ▼
        ┌────────────────┐ ┌────────────────┐
        │ key_projection │ │     gate       │
        │ Dense(sigmoid) │ │ Dense(sigmoid) │
        └───────┬────────┘ └───────┬────────┘
                ▼                  │
        ┌────────────────┐         │
        │  Count Agg.    │         │
        │ (scope-based)  │         │
        └───────┬────────┘         │
                ▼                  │
        ┌────────────────┐         │
        │count_transform │         │
        │Dense(activation)│        │
        └───────┬────────┘         │
                │                  │
                └────────┬─────────┘
                         ▼
        ┌──────────────────────────────────────┐
        │  Gated Blend / Residual Integration   │
        │  y = (1-g)*x + g*C'  (if dims match) │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ Output (batch, seq_len, output_dim)   │
        └──────────────────────────────────────┘

    :param output_dim: Integer, the final output dimension of the layer. For residual
        architectures, this should match the input dimension to enable
        residual-style blending. Must be positive.
    :type output_dim: int
    :param count_dim: Integer, the intermediate dimension for the counting projection.
        Controls the complexity of features that can be counted. Must be positive.
    :type count_dim: int
    :param counting_scope: The scope of counting. Must be one of
        'global', 'local', or 'causal'. Defaults to 'local'.
    :type counting_scope: str
    :param activation: Activation function name or callable to use
        for the count transformation layer. Defaults to 'gelu'.
    :type activation: Union[str, callable]
    :param use_bias: Whether to use bias terms in dense layers.
        Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If output_dim or count_dim is not positive.
    :raises ValueError: If counting_scope is not one of 'global', 'local', 'causal'.

    Note:
        This layer is particularly effective for tasks requiring an understanding
        of feature frequency, enumeration, or relative positioning within a
        sequence. The ``counting_scope`` parameter is crucial for tailoring the
        layer's behavior to the specific task (e.g., 'causal' for auto-regressive
        models). When output_dim matches input_dim, the layer performs residual-style
        blending; otherwise, it acts as a dimension-changing transformation layer.
    """

    def __init__(
        self,
        output_dim: int,
        count_dim: int,
        counting_scope: Literal["global", "local", "causal"] = "local",
        activation: Union[str, callable] = "gelu",
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        # Pop 'max_count' if it exists to avoid passing it to super(), for test compatibility
        kwargs.pop("max_count", None)
        super().__init__(**kwargs)

        # Validate inputs
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if count_dim <= 0:
            raise ValueError(f"count_dim must be positive, got {count_dim}")
        if counting_scope not in ["global", "local", "causal"]:
            raise ValueError(
                f"counting_scope must be one of 'global', 'local', 'causal', "
                f"but got {counting_scope}"
            )

        # Store configuration parameters
        self.output_dim = output_dim
        self.count_dim = count_dim
        self.counting_scope = counting_scope
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Store original activation for serialization
        self._activation_identifier = activation

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        # Layer to identify "countable events"
        self.key_projection = keras.layers.Dense(
            self.count_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="key_projection",
        )

        # Layer to transform aggregated counts with configurable activation
        # Note: The actual input dimension is determined dynamically based on counting_scope
        # For 'local' scope, we concatenate forward and backward counts (2 * count_dim)
        # For 'global' and 'causal' scopes, it's just count_dim
        count_transform_input_dim = self.count_dim * 2 if self.counting_scope == "local" else self.count_dim

        self.count_transform = keras.layers.Dense(
            self.output_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="count_transform",
        )

        # Gating layer to blend count info with original input
        self.gate = keras.layers.Dense(
            self.output_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="gate",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the Counting FFN and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"Input must be at least 2D, got {len(input_shape)}D: {input_shape}"
            )
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified")

        logger.info(
            f"Building CountingFFN: input_dim={input_dim}, output_dim={self.output_dim}, "
            f"count_dim={self.count_dim}, counting_scope='{self.counting_scope}', "
            f"activation='{self._activation_identifier}'"
        )

        if self.output_dim != input_dim:
            logger.warning(
                f"output_dim ({self.output_dim}) does not match input_dim ({input_dim}). "
                "The layer will use gated count transformation instead of residual-style blending."
            )

        # Build sub-layers in computational order for robust serialization
        # 1. Build key projection (takes original input)
        self.key_projection.build(input_shape)

        # 2. Build gate (takes original input)
        self.gate.build(input_shape)

        # 3. Build count transform (takes aggregated counts)
        count_input_dim = self.count_dim
        if self.counting_scope == "local":
            count_input_dim *= 2  # Forward and backward counts are concatenated

        count_transform_input_shape = tuple(input_shape[:-1]) + (count_input_dim,)
        self.count_transform.build(count_transform_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation.

        :param inputs: Input tensor with shape (batch_size, sequence_length, input_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: Optional[bool]
        :return: Output tensor with shape (batch_size, sequence_length, output_dim).
        :rtype: keras.KerasTensor
        """
        # 1. Identify what to count for each token
        # Shape: (batch, seq, count_dim)
        countable_events = self.key_projection(inputs, training=training)

        # 2. Aggregate counts based on the specified scope
        if self.counting_scope == "global":
            # Sum across the sequence and broadcast back
            global_sum = keras.ops.sum(countable_events, axis=1, keepdims=True)
            aggregated_counts = keras.ops.broadcast_to(
                global_sum, keras.ops.shape(countable_events)
            )
        elif self.counting_scope == "causal":
            # Count everything up to the current token
            aggregated_counts = keras.ops.cumsum(countable_events, axis=1)
        else:  # 'local'
            # Forward pass: count up to current token
            forward_counts = keras.ops.cumsum(countable_events, axis=1)
            # Backward pass: count from current token to the end
            reversed_events = keras.ops.flip(countable_events, axis=1)
            backward_counts_rev = keras.ops.cumsum(reversed_events, axis=1)
            backward_counts = keras.ops.flip(backward_counts_rev, axis=1)
            # Combine both directions
            aggregated_counts = keras.ops.concatenate(
                [forward_counts, backward_counts], axis=-1
            )

        # 3. Transform the aggregated counts with configurable activation
        # Shape: (batch, seq, output_dim)
        transformed_counts = self.count_transform(aggregated_counts, training=training)

        # 4. Create a gate to blend the information
        # Shape: (batch, seq, output_dim)
        gate_values = self.gate(inputs, training=training)

        # 5. Blend the count information based on dimensions compatibility
        input_dim = keras.ops.shape(inputs)[-1]

        if self.output_dim == input_dim:
            # When dimensions match, perform residual-style blending with original input
            # The gate decides how much count information vs original input to use
            output = (gate_values * transformed_counts) + ((1 - gate_values) * inputs)
        else:
            # When dimensions don't match, we can't blend with original input
            # Instead, gate controls how much of the transformed counts to use
            # Gate of 1.0 = full transformed counts, gate of 0.0 = zeros
            output = gate_values * transformed_counts

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "count_dim": self.count_dim,
            "counting_scope": self.counting_scope,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
