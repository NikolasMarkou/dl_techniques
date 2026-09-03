"""
A feed-forward network, ``CountingFFN``, that counts features across a
sequence and blends the counts back into each position.

A Dense-plus-sigmoid layer learns which abstract "events" are worth
counting at each token, a scope-dependent sum aggregates how often they
occur ('global' sums the whole sequence, 'causal' sums up to each position
for autoregressive use, 'local' concatenates forward and backward sums and
looks both ways), and a sigmoid gate blends the transformed counts back
into each position. Counting reduces over axis 1, so the input must carry
the sequence there. When ``output_dim`` equals the input width the gate
interpolates between the input and the transformed counts; otherwise it
only scales the transformed counts, and ``build()`` logs a warning for
that case.

References:
    - Hochreiter & Schmidhuber, 1997. Long Short-Term Memory.
    - He et al., 2016. Deep Residual Learning for Image Recognition.
    - Poli et al., 2023. Hyena Hierarchy: Towards Larger Convolutional
      Language Models.
"""

import keras
from typing import Literal, Tuple, Optional, Union, Any, Dict

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.ffn.counting_ffn")
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

    Architecture:

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

    The counting_scope fork:

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

        'local' is the default, and it is the only scope that
        doubles the width. That is why count_transform is built
        on 2C for 'local' and on C for the other two.

        'global' gives every position the same vector. 'causal'
        gives position t the sum over 1..t, so it never looks
        ahead. 'local' concatenates the forward prefix sum with
        the suffix sum from t to T, so it looks both ways. Do not
        use 'local' in an autoregressive model - it leaks the
        future into every position.

    The output blend fork:

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
    :param kernel_initializer: Initializer for the kernels. Each of the three
        Dense layers gets its own clone of it. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases. Each of the three
        Dense layers gets its own clone of it too, which only shows with a
        non-zero initializer. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on). A ``max_count`` key is accepted and discarded
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

    Note:
        Each of the three Dense layers receives its own clone of
        ``kernel_initializer`` and ``bias_initializer``, so
        ``count_transform`` and ``gate`` start as two different functions even
        at the shapes where they coincide - which happens whenever the
        aggregated width equals the input width, for example ``count_dim=8``
        with a 16-wide input under the 'local' default. A seeded initializer
        still gives them the same weights, which is what asking for a seed
        means.
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
        kwargs.pop("max_count", None)
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if count_dim <= 0:
            raise ValueError(f"count_dim must be positive, got {count_dim}")
        if counting_scope not in ["global", "local", "causal"]:
            raise ValueError(
                f"counting_scope must be one of 'global', 'local', 'causal', "
                f"but got {counting_scope}"
            )

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

        # DECISION plan-2026-08-29T043546-e97b34d8/D-005: clone_initializer per
        # Dense, never the shared instance -- a shared instance drew bit-identical weights when shapes coincide. See decisions.md.
        self.key_projection = keras.layers.Dense(
            self.count_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="key_projection",
        )

        # count_transform's input width (2*count_dim for 'local', count_dim
        # otherwise) is resolved once in build()'s count_input_dim.
        self.count_transform = keras.layers.Dense(
            self.output_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="count_transform",
        )

        self.gate = keras.layers.Dense(
            self.output_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
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

        if len(input_shape) < 2:
            raise ValueError(
                f"Input must be at least 2D, got {len(input_shape)}D: {input_shape}"
            )
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified")

        # Captured as a Python int so call() branches without a dynamic
        # keras.ops.shape() tensor, which is graph-unsafe under @tf.function.
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

        self.key_projection.build(input_shape)
        self.gate.build(input_shape)

        count_input_dim = self.count_dim
        if self.counting_scope == "local":
            # Forward and backward counts are concatenated.
            count_input_dim *= 2

        count_transform_input_shape = tuple(input_shape[:-1]) + (count_input_dim,)
        self.count_transform.build(count_transform_input_shape)

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
        countable_events = self.key_projection(inputs, training=training)

        if self.counting_scope == "global":
            global_sum = keras.ops.sum(countable_events, axis=1, keepdims=True)
            aggregated_counts = keras.ops.broadcast_to(
                global_sum, keras.ops.shape(countable_events)
            )
        elif self.counting_scope == "causal":
            aggregated_counts = keras.ops.cumsum(countable_events, axis=1)
        else:
            forward_counts = keras.ops.cumsum(countable_events, axis=1)
            reversed_events = keras.ops.flip(countable_events, axis=1)
            backward_counts_rev = keras.ops.cumsum(reversed_events, axis=1)
            backward_counts = keras.ops.flip(backward_counts_rev, axis=1)
            aggregated_counts = keras.ops.concatenate(
                [forward_counts, backward_counts], axis=-1
            )

        transformed_counts = self.count_transform(aggregated_counts, training=training)
        gate_values = self.gate(inputs, training=training)

        # Decided from the static feature dim captured in build(), not a
        # dynamic ops.shape() tensor.
        if self.output_dim == self._input_last_dim:
            output = (gate_values * transformed_counts) + ((1 - gate_values) * inputs)
        else:
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
