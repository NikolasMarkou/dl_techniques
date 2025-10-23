"""
A dynamic routing mechanism for conditional computation.

This layer introduces adaptive-depth processing into a neural network by
wrapping a standard computational block, such as a Transformer layer. It
addresses the inefficiency of static network architectures where every input
is processed with the same amount of computation. By adding a lightweight
router, this layer dynamically decides for each input whether to skip,
execute, or repeat the wrapped block, allowing the model to allocate more
resources to complex inputs while saving compute on simpler ones.

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

3.  **Conditional Execution**: Based on the router's decision, one of three
    computational graphs is executed for the original input `H`:
    -   **SKIP**: An identity function, `f(H) = H`.
    -   **EXECUTE**: A single application of the wrapped transformer block,
        `f(H) = Transformer(H)`.
    -   **REPEAT**: A sequential double application of the block,
        `f(H) = Transformer(Transformer(H))`.

    This conditional application of computation based on input characteristics
    is the fundamental principle that enables the model to achieve a better
    trade-off between accuracy and computational cost.

References:
    - Heakl, A., et al. (2025). Dr.LLM: Dynamic Layer Routing in LLMs.
      *arXiv preprint arXiv:2510.12773*.
"""

import keras
import tensorflow as tf
from typing import Optional, Union, Any, Dict, Tuple
from keras import layers, initializers, regularizers, ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .transformers.transformer import TransformerLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RouterLayer(keras.layers.Layer):
    """
    Wraps a TransformerLayer with a Dr.LLM-style dynamic routing mechanism.

    This layer adds a lightweight router that decides on a per-sequence basis
    whether to skip, execute once, or repeat the wrapped TransformerLayer.
    This enables adaptive computation, saving resources on simpler inputs while
    allowing deeper processing for more complex ones.

    **Intent**: To provide a retrofittable routing module that follows modern Keras
    3 best practices, ensuring robust serialization and clear separation of concerns
    as outlined in the comprehensive guide.

    **Architecture**:
    ```
    Input(shape=[..., seq_len, hidden_size])
           │
           ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
           │                                                                 │
           ↓                                                                 ↓
    Windowed Mean Pooling (Summarize input)                             (Pass-through for execution)
           ↓
    Router MLP: Dense(bottleneck) -> GELU -> Dense(3)
           ↓
    Logits for {SKIP, EXECUTE, REPEAT}
           ↓
    Decision (argmax during inference, or teacher-forced)
           │
           ├──────────────────────────┬──────────────────────────┐
           ↓                          ↓                          ↓
    SKIP (Identity)         EXECUTE (x1 Transformer)    REPEAT (x2 Transformer)
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      ↓
                      Output(shape=[..., seq_len, hidden_size])
    ```

    **Data Flow**:
    1.  **Summarization**: The input sequence is divided into N windows, and the
        hidden states in each window are mean-pooled to create a compact summary.
    2.  **Decision**: The summary is passed through a small bottleneck MLP to
        produce logits for three actions: {SKIP, EXECUTE, REPEAT}.
    3.  **Execution**: Based on the chosen action, the wrapped TransformerLayer
        is applied 0, 1, or 2 times to the original, full-sequence input.

    Args:
        transformer_layer: An instantiated `TransformerLayer` to be wrapped.
            This layer's configuration is preserved and serialized.
        router_bottleneck_dim: Integer, the hidden dimension of the router's MLP.
            Defaults to 128 as suggested in the Dr.LLM paper.
        num_windows: Integer, the number of windows for pooling the input sequence
            to generate the router's summary. Defaults to 8.
        kernel_initializer: Initializer for the router's kernel weights.
        bias_initializer: Initializer for the router's bias weights.
        kernel_regularizer: Optional regularizer for the router's kernel weights.
        bias_regularizer: Optional regularizer for the router's bias weights.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        - `inputs`: 3D tensor `(batch_size, sequence_length, hidden_size)`.
        - `attention_mask`: Optional mask passed to the inner transformer.
        - `layer_idx`: Integer layer index, passed to the inner transformer.
        - `layer_decision`: Optional 1D int tensor `(batch_size,)` for
          teacher-forcing. Values in {0: SKIP, 1: EXECUTE, 2: REPEAT}.
        - `training`: Boolean flag for training mode.

    Output:
        A tuple containing:
        - `layer_output`: 3D tensor `(batch_size, sequence_length, hidden_size)`.
        - `logits`: 2D tensor `(batch_size, 3)` for the router's decision.

    Attributes:
        transformer_layer: The wrapped `TransformerLayer` instance.
        router_bottleneck: The first Dense layer of the router MLP.
        router_output: The final Dense layer of the router MLP producing logits.

    Example:
        ```python
        # 1. Create the base transformer layer
        transformer_block = TransformerLayer(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048
        )

        # 2. Wrap it with the RouterLayer
        router_block = RouterLayer(
            transformer_layer=transformer_block,
            router_bottleneck_dim=128,
            num_windows=8
        )

        # 3. Use in a model
        inputs = keras.Input(shape=(256, 512))
        # The layer returns two outputs
        processed_output, decision_logits = router_block(inputs)
        model = keras.Model(inputs, [processed_output, decision_logits])

        # During training, you might provide teacher-forced decisions
        # and use the logits for a cross-entropy loss.
        ```

    Note:
        This implementation follows the "Composite Layer" pattern. All sub-layers
        are created in `__init__` and explicitly built in `build()` to guarantee
        correct behavior during serialization and weight restoration.
    """
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
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization,
        following the guide's "Composite Layer" pattern.
        """
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
        """Forward pass of the RouterLayer."""
        # --- 1. Router Logic: Generate decision logits ---
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Use tf ops for dynamic shape handling during pooling
        num_windows_eff = tf.minimum(self.num_windows, seq_len)
        window_len = seq_len // num_windows_eff
        truncated_len = window_len * num_windows_eff
        pooled_input = inputs[:, :truncated_len, :]

        # Reshape for pooling: (batch, num_windows, window_len, hidden_size)
        reshaped = ops.reshape(
            pooled_input, (batch_size, num_windows_eff, window_len, self.hidden_size)
        )
        # Summarize by mean-pooling over tokens in each window
        window_summary = ops.mean(reshaped, axis=2)

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
        """Computes the output shape of the layer."""
        tensor_shape = tuple(input_shape)
        logits_shape = (input_shape[0], 3)
        return tensor_shape, logits_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer for serialization."""
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
        """Creates a layer from its config, properly deserializing sub-layers."""
        config['transformer_layer'] = keras.saving.deserialize_keras_object(config['transformer_layer'])
        return cls(**config)

# ---------------------------------------------------------------------
