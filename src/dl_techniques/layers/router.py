"""
A dynamic routing mechanism for conditional computation.

This layer introduces adaptive-depth routing into a neural network by
wrapping a standard computational block, such as a Transformer layer. For
each input it produces a routing decision -- skip, execute, or repeat the
wrapped block -- via a lightweight router network, exposing a per-input
signal for a compute-aware policy that would allocate more depth to complex
inputs and less to simpler ones.

.. note::
    **Compute cost as implemented.** This layer does NOT reduce FLOPs at
    inference. All three paths are computed unconditionally on every forward
    pass: SKIP is free (identity), EXECUTE runs the wrapped transformer once,
    and REPEAT runs it a second time on the executed output. The chosen action
    then selects among the three via a one-hot mask. Because EXECUTE and
    REPEAT are always materialised, the layer costs approximately 2x a single
    transformer application on every input. It provides the *routing signal*
    for a compute-aware policy; realising actual compute savings would require
    true per-item conditional dispatch (gather/scatter), which is
    intentionally NOT implemented here -- an accelerator-friendly
    select-after-compute scheme was chosen instead.

Architectural and Mathematical Underpinnings:

The layer's architecture decouples the decision-making (routing) from the
primary computation. A small router network operates on a compressed summary
of the input to make an efficient, low-cost decision. This decision then
governs the application of the main, computationally expensive transformer
block on the full, uncompressed input.

1.  **Input Summarization via Windowed Pooling**: To make a routing decision
    that is independent of sequence length, the input hidden states are
    first summarized. The sequence of token embeddings is partitioned into a
    fixed number of contiguous windows. The embeddings within each window are
    then averaged (mean-pooled). This process transforms a variable-length
    sequence `H` of shape `(L, D)` into a fixed-size summary tensor `S` of
    shape `(W, D)`, where `W` is the number of windows. This summary serves as
    a fixed-dimensional "glance" at the input sequence.

2.  **Routing Policy Network**: The core of the router is a simple two-layer
    Multi-Layer Perceptron (MLP). This network learns a routing policy `π`
    that maps the input summary `S` to a probability distribution over the
    three possible actions: {SKIP, EXECUTE, REPEAT}.

        logits = MLP(S)
        action = argmax(logits)

    During training, this policy network is typically trained with explicit
    supervision derived from a search algorithm (like MCTS) that determines
    the optimal computational path for a given task and budget.

3.  **Select-after-compute execution**: All three candidate outputs are
    computed for the original input `H`, and the router's decision selects
    one of them via a one-hot mask:
    -   **SKIP**: An identity function, `f(H) = H`.
    -   **EXECUTE**: A single application of the wrapped transformer block,
        `f(H) = Transformer(H)`.
    -   **REPEAT**: A sequential double application of the block,
        `f(H) = Transformer(Transformer(H))`.

    EXECUTE and REPEAT are evaluated on every forward pass regardless of the
    decision (see the compute-cost note above), so the selection produces the
    routing behaviour but does not, by itself, yield a reduction in
    computational cost.

References:
    - Heakl, A., et al. (2025). Dr.LLM: Dynamic Layer Routing in LLMs.
      *arXiv preprint arXiv:2510.12773*.
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple
from keras import layers, initializers, regularizers, ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .transformers.transformer import TransformerLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RouterLayer(keras.layers.Layer):
    """Dynamic router wrapping a TransformerLayer for adaptive computation.

    This layer adds a lightweight routing MLP that decides, on a
    per-sequence basis, whether to **skip** (identity), **execute** once,
    or **repeat** twice the wrapped ``TransformerLayer``. The decision
    is based on a windowed mean-pooling summary of the input sequence
    passed through a two-layer bottleneck MLP producing logits for
    ``{SKIP=0, EXECUTE=1, REPEAT=2}``. The decision uses the supplied
    ``layer_decision`` whenever it is not ``None`` (regardless of
    training/inference); otherwise ``argmax(logits)`` is used. All three
    computational paths are always evaluated and selected via a one-hot
    mask for hardware-accelerator-friendly execution, so the layer costs
    approximately 2x a single transformer application on every forward pass
    and does not by itself reduce inference FLOPs.

    .. warning::
        **The router MLP is not trained by task loss alone.** Action
        selection is non-differentiable (``argmax`` -> one-hot), so no
        gradient flows to the router weights from the primary task loss via
        the returned output. The router MLP trains ONLY if the caller attaches
        an explicit loss to the returned ``logits`` (e.g. cross-entropy
        against teacher decisions). A caller that trains on task loss alone --
        without supplying ``layer_decision`` and without a loss on ``logits``
        -- leaves the router MLP at its initialization.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input [B, seq_len, hidden_size]       │
        └───────────┬────────────────────────────┘
                    │
            ┌───────┴──────────────────────┐
            ▼                              ▼
        ┌────────────────┐       ┌──────────────────┐
        │ Windowed Mean  │       │ (pass-through    │
        │ Pool → summary │       │  for execution)  │
        └───────┬────────┘       └──────────────────┘
                │
                ▼
        ┌────────────────────────────────┐
        │  Router MLP:                   │
        │  Dense(bottleneck) → GELU →    │
        │  Dense(3) → logits             │
        └───────┬────────────────────────┘
                │
        ┌───────┴───────┬───────────────┐
        ▼               ▼               ▼
    ┌────────┐   ┌────────────┐   ┌─────────────┐
    │ SKIP   │   │ EXECUTE x1 │   │ REPEAT x2   │
    │(ident.)│   │ Transformer│   │ Transformer │
    └───┬────┘   └─────┬──────┘   └──────┬──────┘
        │              │                 │
        └──────────────┼─────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Select path via one-hot mask          │
        │  Output [B, seq_len, hidden_size]      │
        │  + logits [B, 3]                       │
        └────────────────────────────────────────┘

    :param transformer_layer: Instantiated ``TransformerLayer`` to wrap.
    :type transformer_layer: TransformerLayer
    :param router_bottleneck_dim: Hidden dimension of the router MLP.
    :type router_bottleneck_dim: int
    :param num_windows: Number of windows for input summarisation.
    :type num_windows: int
    :param kernel_initializer: Initializer for router kernel weights.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for router bias weights.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for router kernels.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for router biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""
    def __init__(
        self,
        transformer_layer: TransformerLayer,
        router_bottleneck_dim: int = 128,
        num_windows: int = 8,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        # --- Input Validation ---
        if not isinstance(transformer_layer, TransformerLayer):
            raise TypeError(
                f"transformer_layer must be an instance of TransformerLayer, "
                f"got {type(transformer_layer)}"
            )
        if router_bottleneck_dim <= 0:
            raise ValueError(f"router_bottleneck_dim must be positive, got {router_bottleneck_dim}")
        if num_windows <= 0:
            raise ValueError(f"num_windows must be positive, got {num_windows}")

        # --- Configuration Storage ---
        # Store ALL __init__ arguments as attributes
        self.transformer_layer = transformer_layer
        self.hidden_size = transformer_layer.hidden_size
        self.router_bottleneck_dim = router_bottleneck_dim
        self.num_windows = num_windows
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # --- Sub-layer Creation (in __init__) ---
        # Create all sub-layers here; they are unbuilt until the build() method is called.
        self.router_bottleneck = layers.Dense(
            units=self.router_bottleneck_dim,
            activation='gelu',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="router_bottleneck"
        )
        self.router_output = layers.Dense(
            units=3,  # Logits for {SKIP, EXECUTE, REPEAT}
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="router_output"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        # --- Build Sub-layers ---
        # 1. Build the wrapped transformer layer with the main input shape
        self.transformer_layer.build(input_shape)

        # 2. Build the router's MLP layers by calculating their specific input shapes
        # The input to the router MLP is the summarized windowed representation.
        # Shape: (batch_size, num_windows, hidden_size)
        router_input_shape = (input_shape[0], self.num_windows, self.hidden_size)
        self.router_bottleneck.build(router_input_shape)

        # The input to the final router layer is the output of the bottleneck.
        # Shape: (batch_size, num_windows, router_bottleneck_dim)
        router_output_input_shape = (input_shape[0], self.num_windows, self.router_bottleneck_dim)
        self.router_output.build(router_output_input_shape)

        # --- Finalize Build ---
        # Always call parent's build() at the end.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        layer_idx: int = 0,
        layer_decision: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass: route input through skip, execute, or repeat.

        :param inputs: Input tensor ``(batch, seq_len, hidden_size)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask. **Must be
            multiplicative** (``1``/``True`` = keep, ``0``/``False`` = mask),
            matching this repo's ``MultiHeadAttention`` convention. **Additive
            masks are NOT supported**: an additive mask (``0`` / ``-inf`` or
            ``0`` / ``-1e9``) inverts the keep/mask sense and is multiplied into
            the windowed sum, silently corrupting the routing summary (garbage
            logits, no error, no NaN) — a caller holding an additive mask
            elsewhere must convert it to a multiplicative/boolean mask before
            passing it here. Accepted ranks are ``(B, S)``, ``(B, S, S)``, and
            ``(B, H, S, S)``; rank-3/4 masks are reduced to a per-key keep
            vector ``(B, S)`` by "keep a key position if ANY query/head attends
            to it" (``ops.max`` over the query/head axes). The SAME mask is also
            forwarded unchanged to the wrapped transformer.
        :type attention_mask: Optional[keras.KerasTensor]
        :param layer_idx: Layer index passed to the transformer.
        :type layer_idx: int
        :param layer_decision: Optional teacher-forced decisions ``(batch,)``.
        :type layer_decision: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Tuple of (output tensor, router logits).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]

        .. note::
            **jit / static-shape caveat.** The static-shape (JAX-``jit`` /
            XLA-safe) windowed-pooling path applies ONLY when the sequence
            length is statically known -- i.e. ``inputs.shape[1]`` is a concrete
            Python int, the normal ``keras.Input(shape=(SEQ, HID))`` case. A
            symbolic-seq input (``keras.Input(shape=(None, HID))``) falls back to
            the dynamic path, whose reshape target carries data-dependent tensor
            dims and therefore does NOT carry the static-shape jit guarantee
            (it may hit the JAX concretization error / forced XLA recompile that
            the static path avoids)."""
        # --- 1. Router Logic: Generate decision logits ---
        batch_size = ops.shape(inputs)[0]

        # DECISION plan-2026-07-24T091356-f29927b4/D-001
        # Do NOT collapse this back to a single dynamic reshape whose target
        # carries two data-dependent tensor dims (num_windows_eff, window_len):
        # that is the JAX-jit/XLA hazard being fixed here (concretization error /
        # forced recompile). Prefer the STATIC sequence length so the reshape
        # uses Python-int dims; fall back to the dynamic ``ops.shape`` path only
        # when the static seq dim is unknown (``None``, symbolic seq length).
        # See decisions.md D-001.
        static_seq_len = inputs.shape[1]

        if static_seq_len is not None:
            # Static path: every window dim is a Python int; ``-1`` carries the
            # batch axis so no data-dependent dim enters the reshape target.
            seq_len = int(static_seq_len)
            num_windows_eff = min(self.num_windows, seq_len)
            window_len = seq_len // num_windows_eff
            truncated_len = window_len * num_windows_eff
            # NOTE: the trailing ``seq_len % num_windows_eff`` tokens are dropped
            # from the routing SUMMARY only; the full ``inputs`` sequence still
            # reaches the transformer at the EXECUTE/REPEAT calls below.
            pooled_input = inputs[:, :truncated_len, :]
            reshaped = ops.reshape(
                pooled_input, (-1, num_windows_eff, window_len, self.hidden_size)
            )
            mask_window_shape = (-1, num_windows_eff, window_len)
        else:
            # Dynamic fallback: symbolic seq length; dims stay tensor-valued.
            seq_len = ops.shape(inputs)[1]
            num_windows_eff = ops.minimum(self.num_windows, seq_len)
            window_len = seq_len // num_windows_eff
            truncated_len = window_len * num_windows_eff
            # NOTE: trailing ``seq_len % num_windows_eff`` tokens are dropped from
            # the routing SUMMARY only; the full sequence still reaches the
            # transformer at the EXECUTE/REPEAT calls below.
            pooled_input = inputs[:, :truncated_len, :]
            reshaped = ops.reshape(
                pooled_input, (batch_size, num_windows_eff, window_len, self.hidden_size)
            )
            mask_window_shape = (batch_size, num_windows_eff, window_len)

        # Summarize each window. Without a mask this is a plain windowed mean;
        # with a mask, padded key positions are excluded so the routing decision
        # is not biased by padding length/content.
        if attention_mask is None:
            window_summary = ops.mean(reshaped, axis=2)
        else:
            # Reduce the attention mask (MHA convention: 1=keep, 0=mask; ranks
            # (B, S), (B, S, S), or (B, H, S, S)) to a per-key keep-vector
            # (B, seq_len): a key is kept if ANY query/head attends to it.
            keep = attention_mask
            while len(keep.shape) > 2:
                keep = ops.max(keep, axis=1)
            keep = ops.cast(keep, reshaped.dtype)
            keep = keep[:, :truncated_len]
            mask_windows = ops.reshape(keep, mask_window_shape)
            masked_sum = ops.sum(reshaped * mask_windows[..., None], axis=2)
            counts = ops.sum(mask_windows, axis=2)[..., None]
            # Safe denominator: a fully-masked window yields a zero summary vector.
            window_summary = masked_sum / ops.maximum(counts, 1.0)

        # Pass summary through router MLP to get logits per window
        hidden = self.router_bottleneck(window_summary, training=training)
        logits_per_window = self.router_output(hidden, training=training)

        # Aggregate logits by averaging across windows, as per the paper
        logits = ops.mean(logits_per_window, axis=1)

        # --- 2. Decision Making ---
        if layer_decision is not None:
            # Teacher-forcing during training
            decisions = ops.cast(layer_decision, dtype="int32")
        else:
            # Inference mode: use argmax of the router's logits
            decisions = ops.cast(ops.argmax(logits, axis=-1), dtype="int32")

        # --- 3. Conditional Execution using Batch-wise Selection ---
        # This approach computes all paths and selects the correct one per batch item,
        # which is more efficient on hardware accelerators than native conditionals.

        # Path 0: SKIP (output is the same as input)
        output_skip = inputs

        # Path 1: EXECUTE (apply transformer once)
        output_execute = self.transformer_layer(
            inputs,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            training=training
        )

        # Path 2: REPEAT (apply transformer twice)
        output_repeat = self.transformer_layer(
            output_execute,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            training=training
        )

        # Stack all possible outputs along a new dimension
        # Shape: (batch, 3, seq_len, hidden_size)
        stacked_outputs = ops.stack([output_skip, output_execute, output_repeat], axis=1)

        # Create a one-hot mask from decisions to select the correct path
        # Shape: (batch, 3)
        selection_mask = ops.one_hot(decisions, num_classes=3, dtype=stacked_outputs.dtype)

        # Reshape mask for broadcasting: (batch, 3, 1, 1)
        reshaped_mask = ops.reshape(selection_mask, (batch_size, 3, 1, 1))

        # Select the output by element-wise multiplication and summing
        final_output = ops.sum(stacked_outputs * reshaped_mask, axis=1)

        return final_output, logits

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Tuple of (tensor shape, logits shape).
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]"""
        tensor_shape = tuple(input_shape)
        logits_shape = (input_shape[0], 3)
        return tensor_shape, logits_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'transformer_layer': keras.saving.serialize_keras_object(self.transformer_layer),
            'router_bottleneck_dim': self.router_bottleneck_dim,
            'num_windows': self.num_windows,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RouterLayer':
        """Create a layer from its config, deserialising sub-layers.

        :param config: Configuration dictionary from ``get_config()``.
        :type config: Dict[str, Any]
        :return: New ``RouterLayer`` instance.
        :rtype: RouterLayer"""
        config['transformer_layer'] = keras.saving.deserialize_keras_object(config['transformer_layer'])
        return cls(**config)

# ---------------------------------------------------------------------
