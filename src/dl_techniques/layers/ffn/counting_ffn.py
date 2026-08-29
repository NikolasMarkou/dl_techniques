"""
A feed-forward network that counts features across a sequence.

This layer adds frequency information to each token. It first learns which
"events" are worth counting, then aggregates how often they occur, then
blends that count back into each position. Use it when repetition,
enumeration or relative position matters.

**Architecture Overview:**
Three stages:

1.  **Event identification**. ``key_projection`` is a Dense layer with a
    sigmoid, applied to every token. It learns ``count_dim`` abstract
    features. Its output at a position is how strongly that event is present
    there, between 0 and 1.

2.  **Count aggregation**. The events are summed along the sequence axis.
    ``counting_scope`` picks how:
    - 'global': one sum over the whole sequence, broadcast back to every
      position. Every token sees the same totals.
    - 'causal': a cumulative sum from the start. Position ``t`` sees only
      positions 1..t, so this is safe for autoregressive models.
    - 'local': a forward cumulative sum concatenated with a backward one.
      Position ``t`` sees counts before AND after it, so this looks ahead.
      This is the default, and it doubles the aggregated width.

3.  **Gated integration**. ``count_transform`` is a Dense layer with an
    activation that maps the aggregated counts to ``output_dim``. A separate
    sigmoid gate, computed from the original input, decides how much of that
    to use. When ``output_dim`` equals the input width the gate interpolates
    between the input and the transformed counts, which is a residual-style
    blend. Otherwise the gate just scales the transformed counts.

**Mathematics:**
Let ``x_t`` be the input vector at position ``t``, and ``T`` the sequence
length.

1.  The event vector:
    ``k_t = sigmoid(W_k @ x_t + b_k)``
    Each element of ``k_t`` is the soft occurrence of one feature.

2.  The aggregated count depends on the scope:
    - Global: ``C_t = sum_{i=1..T} k_i``
    - Causal: ``C_t = sum_{i=1..t} k_i``
    - Local:  ``C_t = concat[sum_{i=1..t} k_i, sum_{i=t..T} k_i]``

3.  The gate and the transformed counts:
    ``g_t = sigmoid(W_g @ x_t + b_g)``
    ``C'_t = activation(W_c @ C_t + b_c)``

4.  The output, when ``output_dim`` equals the input width:
    ``y_t = g_t * C'_t + (1 - g_t) * x_t``
    and otherwise:
    ``y_t = g_t * C'_t``

So the network learns both what to count and how much the counts should
change its view of the sequence.

References:
The gating comes from recurrent architectures such as LSTMs and GRUs, which
use gates to control information flow. Cumulative sums as a sequence-modeling
primitive appear in recent non-attentional long-range architectures.

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
    Feed-forward network that counts events across a sequence.

    ``key_projection`` finds countable features with a sigmoid
    (``k_t = sigmoid(W_k @ x_t)``). ``counting_scope`` says how those are
    aggregated along the sequence axis. ``count_transform`` maps the
    aggregate to ``output_dim``, and a sigmoid gate blends it back into each
    position.

    The counting happens on axis 1, so the input must be a sequence with the
    sequence on that axis.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │ Input x  [B, T, input_dim]           │
        └──────────────────┬───────────────────┘
                           │
                     ┌─────┴─────┬──────────────┐
                     ▼           ▼              │
        ┌────────────────┐ ┌────────────────┐   │
        │ key_projection │ │      gate      │   │
        │ Dense(C), sigm │ │ Dense(O), sigm │   │
        └───────┬────────┘ └───────┬────────┘   │
                ▼                  │            │
        ┌────────────────┐         │            │
        │ count aggreg.  │         │            │
        │ (scope-based)  │         │            │
        └───────┬────────┘         │            │
                ▼                  │            │
        ┌─────────────────┐        │            │
        │ count_transform │        │            │
        │ Dense(O), activ │        │            │
        └───────┬─────────┘        │            │
                └────────┬─────────┘            │
                         ▼                      │
        ┌──────────────────────────────────┐    │
        │  gated blend  (fork below)  ◄────┼────┘
        └──────────────────┬───────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ Output  [B, T, output_dim]           │
        └──────────────────────────────────────┘

        C = count_dim, O = output_dim, T = sequence length.
        The input x reaches the blend directly, but only on the
        residual leaf of the fork.

    **The counting_scope fork:**

    .. code-block:: text

              k = key_projection(x)  [B, T, C]
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
         'global'       'causal'        'local'
             │              │              │
             ▼              ▼              ▼
         sum over T     cumsum over    cumsum(k) and
         keepdims,      axis 1         flip(cumsum(flip(k)))
         broadcast                     concat on axis -1
             │              │              │
             ▼              ▼              ▼
         [B, T, C]      [B, T, C]      [B, T, 2C]

        'local' is the DEFAULT, and it is the only scope that
        doubles the width. That is why count_transform is built
        on 2C for 'local' and on C for the other two.

        'global' gives every position the same vector. 'causal'
        gives position t the sum over 1..t, so it never looks
        ahead. 'local' concatenates the forward prefix sum with
        the suffix sum from t to T, so it looks BOTH ways. Do not
        use 'local' in an autoregressive model - it leaks the
        future into every position.

    **The output blend fork:**

    .. code-block:: text

        g  = gate(x)                  [B, T, O]
        C' = count_transform(counts)  [B, T, O]

               output_dim == input width ?
               (a Python int comparison, fixed
                in build() from the static shape)
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
                True                 False
          y = g*C' + (1-g)*x       y = g*C'
          residual blend           gated counts only
          x is an input here,      x is not added at
          not a weight             all on this leaf

        On the False leaf the input reaches the output only
        through gate(x), never additively, so at g = 0 the output
        is exactly zero. build() logs a warning when this leaf is
        the one that will run.

    :param output_dim: Width of the output. Match it to the input width to
        get the residual blend. Must be positive.
    :type output_dim: int
    :param count_dim: Number of countable events ``key_projection`` learns.
        Must be positive.
    :type count_dim: int
    :param counting_scope: 'global', 'local' or 'causal'. See the scope fork
        above for what each does. Defaults to 'local'.
    :type counting_scope: Literal["global", "local", "causal"]
    :param activation: Activation for ``count_transform``. A name or a
        callable. Defaults to 'gelu'.
    :type activation: Union[str, callable]
    :param use_bias: Whether the three Dense layers carry a bias. Defaults to
        True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels. The same instance
        goes to all three Dense layers. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on). A ``max_count`` key is accepted and DISCARDED
        for backward compatibility with older configs; it changes nothing.
    :type kwargs: Any

    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar count_dim: The stored number of countable events.
    :vartype count_dim: int
    :ivar counting_scope: The stored scope name.
    :vartype counting_scope: str
    :ivar activation: The resolved activation, a callable after
        ``keras.activations.get``.
    :vartype activation: Callable
    :ivar use_bias: Whether the Dense layers carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar key_projection: ``Dense(count_dim, activation='sigmoid')``.
    :vartype key_projection: keras.layers.Dense
    :ivar count_transform: ``Dense(output_dim, activation=activation)``,
        applied to the aggregated counts.
    :vartype count_transform: keras.layers.Dense
    :ivar gate: ``Dense(output_dim, activation='sigmoid')``, read from the
        original input.
    :vartype gate: keras.layers.Dense
    :ivar _activation_identifier: The activation argument exactly as passed.
        Only the ``build()`` log line reads it; ``get_config()`` serializes
        ``self.activation`` instead.
    :vartype _activation_identifier: Union[str, callable]
    :ivar _input_last_dim: The input's last axis, captured in ``build()`` as a
        Python int so ``call()`` can pick its branch without inspecting a
        dynamic shape. ``None`` before ``build()``, and not serialized.
    :vartype _input_last_dim: Optional[int]

    :raises ValueError: If ``output_dim`` or ``count_dim`` is not positive.
    :raises ValueError: If ``counting_scope`` is not 'global', 'local' or
        'causal'.
    :raises ValueError: From ``build()``, if the input has rank < 2 or if its
        last axis is ``None``.

    Input shape:
        3D tensor ``(batch, sequence_length, input_dim)``. Rank 2 is accepted
        by ``build()``, but the counting reduces over axis 1, so a rank-2
        input counts over its feature axis rather than over time.

    Output shape:
        Same leading axes as the input, last axis ``output_dim``.

    Example:
        .. code-block:: python

            ffn = CountingFFN(output_dim=32, count_dim=16)
            y = ffn(keras.random.normal((2, 10, 32)))
            y.shape                 # (2, 10, 32) -- residual blend

    Note:
        ``counting_scope='local'`` is the default and it is bidirectional.
        Pick 'causal' for anything autoregressive.

    Warning:
        All three Dense layers share one initializer instance. When
        ``count_transform`` and ``gate`` end up the same shape - which happens
        whenever the aggregated width equals the input width, for example
        ``count_dim=8`` with a 16-wide input under the 'local' default - they
        start with bit-identical kernels. MEASURED: ``max|delta|`` = 0.0 at
        build time. They see different inputs, so they do diverge under
        training, unlike the same pattern in ``gated_mlp.py``.
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
        """
        Validate the configuration and create the three Dense layers.

        Every argument is documented on the class. A ``max_count`` keyword is
        dropped before ``super().__init__`` so old configs that carry it still
        load; it has no effect.

        :raises ValueError: If ``output_dim`` or ``count_dim`` is not
            positive, or if ``counting_scope`` is not one of 'global',
            'local', 'causal'.
        """
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

        # The activation argument exactly as passed. get_config() serializes
        # self.activation instead; only the build() log line reads this.
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

        # Layer to transform aggregated counts with configurable activation.
        # Its INPUT width depends on counting_scope -- 2 * count_dim for
        # 'local', which concatenates forward and backward counts, and
        # count_dim for 'global' and 'causal'. That rule is written down once,
        # in build()'s count_input_dim, which is where the width is actually
        # needed; a Dense layer takes only its output width here.
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

        # Static input feature dim, resolved in build() (graph-safe branch in call()).
        self._input_last_dim: Optional[int] = None

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

        # Capture the static feature dim so call() can decide the residual-vs-gated
        # branch with a Python int comparison instead of branching on a dynamic
        # keras.ops.shape() tensor (graph-unsafe under @tf.function). Derived from
        # input_shape -> not serialized in get_config.
        self._input_last_dim = int(input_dim)

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
            # Forward and backward counts are concatenated.
            count_input_dim *= 2

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
        else:
            # 'local' scope.
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

        # 5. Blend the count information based on dimensions compatibility.
        # Decided at build time from the static feature dim (graph-safe: Python
        # int vs Python int, no branch on a dynamic ops.shape() tensor).
        if self.output_dim == self._input_last_dim:
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
