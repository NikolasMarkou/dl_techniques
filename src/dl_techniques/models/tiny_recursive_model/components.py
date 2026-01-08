"""
Core reasoning modules for the Tiny Recursive Model (TRM).

This file implements the computational heart of the TRM architecture, which
is designed for recursive, multi-step reasoning. It contains two key Keras
Layer components:

1.  `TRMReasoningModule`: A fundamental building block composed of a stack of
    Transformer layers. It represents a deep, sequential transformation
    applied to a latent state. This module is used to construct the more
    complex hierarchical reasoning structures.

2.  `TRMInner`: The primary engine for a single reasoning step within the
    broader TRM model. It orchestrates multiple `TRMReasoningModule`
    instances to manage a hierarchical state (`z_H`, `z_L`). Crucially, it
    also generates the halting signals required by the Adaptive Computation
    Time (ACT) algorithm, allowing the parent model to dynamically decide
    when to stop processing.

These components are designed following modern Keras best practices, ensuring
they are robust, serializable, and clearly documented. The separation of
concerns—with `TRMReasoningModule` providing the raw computational power and
`TRMInner` orchestrating the reasoning flow—enables a modular and
understandable implementation of a complex reasoning architecture.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformers import (
    TransformerLayer,
    FFNType,
    AttentionType,
    NormalizationType,
    NormalizationPositionType
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TRMReasoningModule(keras.layers.Layer):
    """A module that stacks multiple TransformerLayers for deep reasoning.

    This layer serves as a core computational engine by applying a sequence of
    identical transformations to its input. It is composed of `num_layers`
    instances of `TransformerLayer`, demonstrating the composite layer
    pattern where complex operations are built by orchestrating sub-layers.

    **Intent**: To create a deep, sequential processing block that forms the
    main reasoning component of the TRM architecture, allowing for multiple
    rounds of feature refinement within a single module.

    **Architecture**:
    ```
    Input(shape=[B, S, H])
           ↓
    TransformerBlock_0(Input)
           ↓
    TransformerBlock_1(Output_0)
           ↓
          ...
           ↓
    TransformerBlock_L-1(Output_L-2)
           ↓
    Output(shape=[B, S, H])
    ```
    - B: Batch size, S: Sequence length, H: Hidden size.
    - Each `TransformerBlock` is a `dl_techniques.layers.TransformerLayer`.

    Args:
        hidden_size (int): The dimensionality of the hidden states.
        num_heads (int): Number of attention heads in each transformer layer.
        expansion (float): Factor to determine the FFN intermediate size. The
            intermediate size is `int(hidden_size * expansion)`.
        num_layers (int): The number of `TransformerLayer` instances to stack.
        seq_len (int): The length of the input sequence.
        puzzle_emb_len (int): Length of the puzzle embedding prefix.
        rope_theta (float): Theta value for RoPE (Rotary Position Embedding).
        attention_type (str): Type of attention mechanism to use.
        ffn_type (str): Type of feed-forward network to use.
        normalization_type (str): Type of normalization layer to use.
        normalization_position (str): Position of normalization ('pre'/'post').
        dropout_rate (float): Dropout rate for transformer layers.
        attention_dropout_rate (float): Dropout rate for attention weights.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    Output shape:
        3D tensor with the same shape as the input.

    Attributes:
        layers_list (list): The list of `TransformerLayer` instances.

    Example:
        ```python
        # Create a deep reasoning block with 4 transformer layers
        reasoning_block = TRMReasoningModule(
            hidden_size=512,
            num_heads=8,
            expansion=4.0,
            num_layers=4,
            seq_len=128
        )
        # Input tensor
        inputs = keras.ops.random.normal((2, 128, 512))
        # Another tensor for injection (e.g., from another state)
        injection = keras.ops.random.normal((2, 128, 512))
        # The output has the same shape as the input
        output = reasoning_block(inputs, injection)
        ```

    Note:
        This layer follows the "Composite Layer" pattern. Sub-layers are
        created in `__init__` and explicitly built in the `build` method. This
        is critical for ensuring that Keras can correctly save and load models
        containing this layer without "layer has not been built" errors.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        num_layers: int,
        seq_len: int,
        puzzle_emb_len: int = 16,
        rope_theta: float = 10000.0,
        attention_type: AttentionType = 'multi_head',
        ffn_type: FFNType = 'swiglu',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        """Initialize the TRMReasoningModule.

        This method creates all sub-layers and stores the configuration. The
        sub-layers are instantiated but remain unbuilt until the `build`
        method is called.

        Args:
            hidden_size (int): The dimensionality of the hidden states.
            num_heads (int): Number of attention heads in each transformer layer.
            expansion (float): Factor to determine the FFN intermediate size.
            num_layers (int): The number of `TransformerLayer` instances to stack.
            seq_len (int): The length of the input sequence.
            puzzle_emb_len (int): Length of the puzzle embedding prefix.
            rope_theta (float): Theta value for RoPE.
            attention_type (str): Type of attention mechanism.
            ffn_type (str): Type of feed-forward network.
            normalization_type (str): Type of normalization layer.
            normalization_position (str): Position of normalization.
            dropout_rate (float): Dropout rate for transformer layers.
            attention_dropout_rate (float): Dropout rate for attention weights.
            **kwargs: Additional arguments for the base Layer class.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.puzzle_emb_len = puzzle_emb_len
        self.rope_theta = rope_theta
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        intermediate_size = int(hidden_size * expansion)

        # CREATE sub-layers in __init__ as per the Golden Rule.
        self.layers_list = [
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                attention_type=attention_type,
                attention_args={
                    'max_seq_len': seq_len + puzzle_emb_len,
                    'rope_theta': rope_theta
                },
                normalization_type=normalization_type,
                normalization_position=normalization_position,
                ffn_type=ffn_type,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                name=f"transformer_block_{i}"
            ) for i in range(num_layers)
        ]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all constituent TransformerLayer instances.

        This method follows the "Composite Layer" pattern by explicitly
        building each sub-layer. This is a critical step that ensures weight
        variables are created for all sub-components before Keras attempts to
        restore them during model loading, thus preventing serialization errors.

        Args:
            input_shape (Tuple[Optional[int], ...]): The shape of the input tensor,
                e.g., `(batch_size, seq_len, hidden_size)`.
        """
        for layer in self.layers_list:
            if not layer.built:
                layer.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            hidden_states: keras.KerasTensor,
            input_injection: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Perform the forward pass through the stack of TransformerLayers.

        Args:
            hidden_states (keras.KerasTensor): The primary latent state tensor
                to be processed.
            input_injection (keras.KerasTensor): An external tensor (e.g., an
                embedding or another latent state) to be added to the hidden
                state at the beginning of the process.
            training (Optional[bool]): A boolean flag indicating whether the layer
                should behave in training mode (e.g., applying dropout).

        Returns:
            keras.KerasTensor: The transformed hidden_states tensor, with the
                same shape as the input.
        """
        hidden_states = hidden_states + input_injection
        for layer in self.layers_list:
            hidden_states = layer(hidden_states, training=training)
        return hidden_states

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Since this module only applies transformations that preserve the input
        shape, the output shape is identical to the input shape.

        Args:
            input_shape (Tuple[Optional[int], ...]): Shape of the input tensor.

        Returns:
            Tuple[Optional[int], ...]: The identical output shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        This method includes all `__init__` parameters, allowing the layer to
        be fully reconstructed from its config.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration.
        """
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'expansion': self.expansion,
            'num_layers': self.num_layers,
            'seq_len': self.seq_len,
            'puzzle_emb_len': self.puzzle_emb_len,
            'rope_theta': self.rope_theta,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TRMInner(keras.layers.Layer):
    """The inner computational core of the TRM model.

    This layer orchestrates the computation within a single step of the outer
    Adaptive Computation Time (ACT) loop. It maintains and updates two
    hierarchical latent states (`z_H`, `z_L`) and produces both task-specific
    predictions (logits) and the halting probabilities required by ACT.

    **Intent**: To encapsulate a fixed-cycle, hierarchical reasoning process.
    Each call represents a focused burst of computation that refines the
    model's latent states and informs the outer loop's decision to continue
    or terminate processing for a given sample.

    **Architecture**:
    ```
    Inputs (carry['z_H'], carry['z_L'], data['inputs'])
                                │
                      ┌─────────┴─────────┐
                      │    TRMInner       │
                      │                   │
            (data) ───► Token Embedding   │ ──► Pad
                      │        │          │
            (z_L) ────► L_level Module ◄──┘
                      │        │          │
            (z_H) ────► H_level Module ◄──┘
                      │        │          │
                      │   ┌────┴────┐     │
                      │   ▼         ▼     │
                      │ lm_head   q_head  │
                      └─────┬───────┬─────┘
                            │       │
                            ▼       ▼
                        logits    (q_halt, q_continue)
    ```

    **Data Flow**:
    1. Input tokens from `data` are embedded.
    2. The low-level state `z_L` is updated by the `L_level` module, using
       the previous `z_L` and the new token embeddings as input.
    3. The high-level state `z_H` is updated by the `H_level` module, using
       the previous `z_H` and the updated `z_L` as input.
    4. The final `z_H` is passed to two heads: `lm_head` produces prediction
       logits, and `q_head` produces the halting and continuation logits.

    Args:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Dimensionality of all hidden states.
        num_heads (int): Number of attention heads in transformer layers.
        expansion (float): Factor for FFN intermediate size.
        seq_len (int): Length of the input token sequence.
        puzzle_emb_len (int): Length of the puzzle embedding prefix.
        h_layers (int): Number of layers in the high-level reasoning module.
        l_layers (int): Number of layers in the low-level reasoning module.
        rope_theta (float): Theta value for RoPE.
        attention_type (str): Type of attention mechanism.
        ffn_type (str): Type of feed-forward network.
        normalization_type (str): Type of normalization layer.
        normalization_position (str): Position of normalization.
        dropout_rate (float): Dropout rate for transformer layers.
        attention_dropout_rate (float): Dropout rate for attention weights.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.

    Attributes:
        token_emb (Embedding): Layer for token embeddings.
        H_level (TRMReasoningModule): Module for high-level state updates.
        L_level (TRMReasoningModule): Module for low-level state updates.
        lm_head (Dense): Head for generating final prediction logits.
        q_head (Dense): Head for generating ACT halting logits.
        H_init (tf.Variable): Learnable initial state for `z_H`.
        L_init (tf.Variable): Learnable initial state for `z_L`.

    Note:
        This layer is a "Composite Layer" that also creates its own weights
        (`H_init`, `L_init`). The `build` method is responsible for both
        building all sub-layers and creating these learnable initial state
        weights, ensuring full serializability.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        seq_len: int,
        puzzle_emb_len: int = 16,
        h_layers: int = 2,
        l_layers: int = 2,
        rope_theta: float = 10000.0,
        attention_type: AttentionType = 'multi_head',
        ffn_type: FFNType = 'swiglu',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        """Initialize the TRMInner module.

        This method creates all sub-layers and stores the configuration. The
        sub-layers are instantiated but remain unbuilt until the `build`
        method is called.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            num_heads (int): Number of attention heads.
            expansion (float): FFN expansion factor.
            seq_len (int): Length of the input sequence.
            puzzle_emb_len (int): Length of the puzzle embedding prefix.
            h_layers (int): Number of layers in the H_level module.
            l_layers (int): Number of layers in the L_level module.
            rope_theta (float): Theta value for RoPE.
            attention_type (str): Type of attention mechanism.
            ffn_type (str): Type of feed-forward network.
            normalization_type (str): Type of normalization.
            normalization_position (str): Position of normalization.
            dropout_rate (float): General dropout rate.
            attention_dropout_rate (float): Attention-specific dropout.
            **kwargs: Additional arguments for the base Layer class.
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.seq_len = seq_len
        self.puzzle_emb_len = puzzle_emb_len
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.rope_theta = rope_theta
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            name="token_embedding"
        )
        self.H_level = TRMReasoningModule(
            hidden_size=hidden_size, num_heads=num_heads, expansion=expansion,
            num_layers=h_layers, seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len, rope_theta=rope_theta,
            attention_type=attention_type, ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate, name="H_level"
        )
        self.L_level = TRMReasoningModule(
            hidden_size=hidden_size, num_heads=num_heads, expansion=expansion,
            num_layers=l_layers, seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len, rope_theta=rope_theta,
            attention_type=attention_type, ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate, name="L_level"
        )
        self.lm_head = keras.layers.Dense(
            vocab_size, use_bias=False, name="lm_head"
        )
        self.q_head = keras.layers.Dense(2, name="q_head")

    def build(self, input_shape: Optional[Any] = None) -> None:
        """Build sub-layers and create initial state weights.

        This method performs two critical functions for serializability:
        1.  It explicitly builds all composite sub-layers (`H_level`, `L_level`)
            to ensure their weights are created before restoration.
        2.  It creates the layer's own learnable weights (`H_init`, `L_init`)
            which serve as the initial states for the reasoning process.

        Args:
            input_shape (Optional[Any]): Not used directly, as shapes are
                inferred from the configuration.
        """
        full_seq_len = self.seq_len + self.puzzle_emb_len
        latent_shape = (None, full_seq_len, self.hidden_size)

        if not self.token_emb.built:
            self.token_emb.build((None, self.seq_len))
        if not self.H_level.built:
            self.H_level.build(latent_shape)
        if not self.L_level.built:
            self.L_level.build(latent_shape)
        if not self.lm_head.built:
            self.lm_head.build(latent_shape)
        if not self.q_head.built:
            self.q_head.build((None, self.hidden_size))

        self.H_init = self.add_weight(
            name="H_init",
            shape=(1, full_seq_len, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.L_init = self.add_weight(
            name="L_init",
            shape=(1, full_seq_len, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(
            self,
            carry: Dict[str, keras.KerasTensor],
            data: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Perform a single inner reasoning step.

        Args:
            carry (Dict[str, keras.KerasTensor]): A dictionary containing the
                recurrent latent states from the previous step:
                - `z_H`: High-level latent state.
                - `z_L`: Low-level latent state.
            data (Dict[str, keras.KerasTensor]): A dictionary containing the
                input data for the current step:
                - `inputs`: Input token IDs of shape (batch_size, seq_len).
            training (Optional[bool]): Boolean flag for training mode.

        Returns:
            A tuple containing:
            - new_carry (Dict[str, keras.KerasTensor]): Updated latent states.
              Gradients are stopped to prevent backpropagation through time
              across ACT steps.
            - logits (keras.KerasTensor): Output logits for sequence prediction.
            - (q_halt, q_continue) (Tuple): Logits for halting and continuing.
        """
        z_H = carry["z_H"]
        z_L = carry["z_L"]

        input_emb = self.token_emb(data["inputs"])

        batch_size = ops.shape(input_emb)[0]
        puzzle_emb_padding = ops.zeros(
            (batch_size, self.puzzle_emb_len, self.hidden_size),
            dtype=input_emb.dtype
        )
        input_emb_padded = ops.concatenate([puzzle_emb_padding, input_emb], axis=1)

        z_L = self.L_level(z_L, input_emb_padded, training=training)
        z_H = self.H_level(z_H, z_L, training=training)

        new_carry = {
            "z_H": keras.ops.stop_gradient(z_H),
            "z_L": keras.ops.stop_gradient(z_L)
        }

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0])
        q_halt, q_continue = q_logits[..., 0], q_logits[..., 1]

        return new_carry, logits, (q_halt, q_continue)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        This method includes all `__init__` parameters, allowing the layer to
        be fully reconstructed from its config.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration.
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'expansion': self.expansion,
            'seq_len': self.seq_len,
            'puzzle_emb_len': self.puzzle_emb_len,
            'h_layers': self.h_layers,
            'l_layers': self.l_layers,
            'rope_theta': self.rope_theta,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
        })
        return config

# ---------------------------------------------------------------------
