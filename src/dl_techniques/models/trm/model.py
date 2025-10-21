"""
Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

This model performs recursive reasoning to solve complex tasks by repeatedly
applying a small, shared neural network over a variable number of steps. It
integrates the principles of Adaptive Computation Time (ACT) to learn how
many computational steps are necessary for a given problem, allowing it to
dynamically allocate resources. The model's `call` method encapsulates
a single step of this adaptive, recursive process.

The architecture is conceptually divided into two nested levels of recursion:
1.  An outer ACT loop, managed externally by the training script, which
    calls this model's `call` method repeatedly. This loop allows the
    model to progressively refine its solution over multiple "thought" steps.
2.  An inner, fixed-cycle reasoning process within the `TRMInner` submodule.
    During each single call to this model's `call` method, the inner
    module performs a multi-step update of its latent states (`z_H`, `z_L`),
    representing a focused burst of computation.

The core mathematical principle is Adaptive Computation Time (ACT). At each
step of the outer loop, the model produces not only a prediction but also a
halting probability via a dedicated `q_head`. This probability is used to
decide whether to continue processing or to terminate for a given example
in the batch. The model's state (the `carry` object) is passed from one
step to the next. For sequences that have not halted, their latent states
are updated by the `TRMInner` module. This design makes the model highly
parameter-efficient, as the same small network is reused across all steps,
and computationally efficient, as it can stop processing easier inputs
early.

References:
    - Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning
      with Tiny Networks. arXiv preprint arXiv:2510.04871.
    - Graves, A. (2016). Adaptive Computation Time for Recurrent Neural
      Networks. Advances in Neural Information Processing Systems, 29.
      https://proceedings.neurips.cc/paper/2016/file/bf69145244511593c662c12aee4608c0-Paper.pdf
    - Wang, G., et al. (2025). Hierarchical Reasoning Model. arXiv
      preprint arXiv:2506.21734. (Inspirational basis for TRM).
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformer import (
    TransformerLayer,
    FFNType,
    AttentionType,
    NormalizationType,
    NormalizationPositionType
)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TRMReasoningModule(keras.layers.Layer):
    """
    A module that stacks multiple TransformerLayers for the reasoning process.

    This layer serves as the core computational engine of the TRM, applying a
    sequence of transformations to the latent states. It is composed of `num_layers`
    instances of the `TransformerLayer`, demonstrating the composite layer pattern
    where sub-layers are orchestrated.

    **Intent**: To create a deep, sequential processing block by stacking identical
    transformer layers, forming the main reasoning component of the TRM architecture.

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
    - Each `TransformerBlock` is a `TransformerLayer` instance.

    Args:
        hidden_size (int): The dimensionality of the hidden states.
        num_heads (int): Number of attention heads in each transformer layer.
        expansion (float): Factor to determine the FFN intermediate size. The
            intermediate size is computed as `int(hidden_size * expansion)`.
        num_layers (int): The number of `TransformerLayer` instances to stack.
        seq_len (int): The length of the input sequence.
        puzzle_emb_len (int): Length of the puzzle embedding prefix. Default is 16.
        rope_theta (float): Theta value for RoPE (Rotary Position Embedding).
            Default is 10000.0.
        attention_type (str): Type of attention mechanism to use. Default is 'multi_head'.
        ffn_type (str): Type of feed-forward network to use. Default is 'swiglu'.
        normalization_type (str): Type of normalization layer to use. Default is 'rms_norm'.
        normalization_position (str): Position of normalization ('pre' or 'post').
            Default is 'post'.
        dropout_rate (float): Dropout rate for transformer layers. Default is 0.0.
        attention_dropout_rate (float): Dropout rate specifically for attention.
            Default is 0.0.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.
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
        super().__init__(**kwargs)

        # Store all configuration parameters as instance attributes
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

        # Calculate intermediate size for the FFN inside the TransformerLayer
        intermediate_size = int(hidden_size * expansion)

        # CREATE sub-layers in __init__ as per the Golden Rule.
        # These layers are instantiated but not yet built.
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
        """
        Build all constituent TransformerLayer instances.

        This method follows the guide's "Composite Layer" pattern by explicitly
        building each sub-layer. This ensures that their weights are created
        before Keras attempts to restore them during model loading, preventing
        "layer has not been built" errors.

        Args:
            input_shape (Tuple[Optional[int], ...]): The shape of the input tensor.
        """
        # Sequentially build each sub-layer. Since each TransformerLayer preserves
        # the shape, we can reuse the initial input_shape for all of them.
        for layer in self.layers_list:
            layer.build(input_shape)
        # Always call the parent's build method at the end.
        super().build(input_shape)

    def call(
            self,
            hidden_states: keras.KerasTensor,
            input_injection: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the stack of TransformerLayers.

        Args:
            hidden_states (keras.KerasTensor): The primary latent state tensor.
            input_injection (keras.KerasTensor): The tensor to be added to the
                hidden_states at the start.
            training (Optional[bool]): Boolean flag for training mode, passed to
                sub-layers.

        Returns:
            keras.KerasTensor: The transformed hidden_states tensor.
        """
        hidden_states = hidden_states + input_injection
        for layer in self.layers_list:
            hidden_states = layer(hidden_states, training=training)
        return hidden_states

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape (Tuple[Optional[int], ...]): Shape of the input tensor.

        Returns:
            Tuple[Optional[int], ...]: The output shape is identical to the input shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dict[str, Any]: Configuration dictionary containing all parameters needed
                to reconstruct this layer.
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
    """
    The inner computational core of the TRM model.

    This layer orchestrates the computation within a single step of the outer
    ACT loop. It maintains two latent states: `z_H` (high-level) and `z_L`
    (low-level), and updates them through a series of reasoning modules. The
    layer also produces output logits and halting probabilities.

    **Intent**: To encapsulate the inner fixed-cycle reasoning process, where
    each call represents a focused burst of computation on the latent states.

    **Architecture**:
    The layer consists of:
    - Token embedding projection
    - Two-level reasoning modules (H_level and L_level)
    - Language modeling head
    - Halting probability head

    Args:
        vocab_size (int): Size of the vocabulary for token embeddings.
        hidden_size (int): Dimensionality of hidden states.
        num_heads (int): Number of attention heads in transformer layers.
        expansion (float): Factor to determine FFN intermediate size.
        seq_len (int): Length of the input sequence (excluding puzzle embedding).
        puzzle_emb_len (int): Length of the puzzle embedding prefix. Default is 16.
        h_layers (int): Number of layers in the H_level reasoning module.
        l_layers (int): Number of layers in the L_level reasoning module.
        rope_theta (float): Theta value for RoPE (Rotary Position Embedding).
            Default is 10000.0.
        attention_type (str): Type of attention mechanism to use. Default is 'multi_head'.
        ffn_type (str): Type of feed-forward network to use. Default is 'swiglu'.
        normalization_type (str): Type of normalization layer to use. Default is 'rms_norm'.
        normalization_position (str): Position of normalization ('pre' or 'post').
            Default is 'post'.
        dropout_rate (float): Dropout rate for transformer layers. Default is 0.0.
        attention_dropout_rate (float): Dropout rate specifically for attention.
            Default is 0.0.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.
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
        super().__init__(**kwargs)

        # Store all configuration parameters as instance attributes
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

        # CREATE sub-layers in __init__ following the Golden Rule
        # Token embedding layer
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            name="token_embedding"
        )

        # Reasoning modules for high and low level processing
        self.H_level = TRMReasoningModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            num_layers=h_layers,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            rope_theta=rope_theta,
            attention_type=attention_type,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            name="H_level"
        )
        self.L_level = TRMReasoningModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            num_layers=l_layers,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            rope_theta=rope_theta,
            attention_type=attention_type,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            name="L_level"
        )

        # Output heads
        self.lm_head = keras.layers.Dense(
            vocab_size,
            use_bias=False,
            name="lm_head"
        )
        self.q_head = keras.layers.Dense(2, name="q_head")

    def build(self, input_shape: Optional[Any] = None) -> None:
        """
        Build the layer and its sub-components.

        This method explicitly builds all sub-layers and creates the initial
        state weights. Following the composite layer pattern ensures all weights
        exist before serialization/deserialization.

        Args:
            input_shape (Optional[Any]): Shape of the input. Can be None since
                this layer creates its own initial states.
        """
        # Compute the full sequence length
        full_seq_len = self.seq_len + self.puzzle_emb_len

        # Shape for latent states: (batch, seq_len, hidden_size)
        # We use (None, ...) to allow variable batch size
        latent_shape = (None, full_seq_len, self.hidden_size)

        # Build token embedding layer
        # Input shape for embedding is (batch, seq_len)
        self.token_emb.build((None, self.seq_len))

        # Build reasoning modules with the full latent shape
        self.H_level.build(latent_shape)
        self.L_level.build(latent_shape)

        # Build output heads
        # lm_head takes (batch, seq_len, hidden_size)
        self.lm_head.build(latent_shape)
        # q_head takes the first token: (batch, hidden_size)
        self.q_head.build((None, self.hidden_size))

        # Create initial state weights
        # These are learnable initial states that will be used to reset
        # the carry when a sequence starts or restarts.
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

        # Call parent build
        super().build(input_shape)

    def call(
            self,
            carry: Dict[str, keras.KerasTensor],
            data: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Perform a single inner reasoning step.

        Args:
            carry (Dict[str, keras.KerasTensor]): Dictionary containing latent states:
                - `z_H` (keras.KerasTensor): High-level latent state.
                - `z_L` (keras.KerasTensor): Low-level latent state.
            data (Dict[str, keras.KerasTensor]): Dictionary containing input data:
                - `inputs` (keras.KerasTensor): Input token IDs.
            training (Optional[bool]): Boolean flag for training mode.

        Returns:
            Tuple containing:
            - new_carry (Dict[str, keras.KerasTensor]): Updated latent states
                with gradients detached.
            - logits (keras.KerasTensor): Output logits for sequence prediction.
            - (q_halt, q_continue) (Tuple[keras.KerasTensor, keras.KerasTensor]):
                Halting probability logits.
        """
        z_H = carry["z_H"]
        z_L = carry["z_L"]

        # Embed the input tokens
        input_emb = self.token_emb(data["inputs"])

        # Pad embedding to match full sequence length (including puzzle embedding)
        batch_size = ops.shape(input_emb)[0]
        puzzle_emb_padding = ops.zeros(
            (batch_size, self.puzzle_emb_len, self.hidden_size),
            dtype=input_emb.dtype
        )
        input_emb_padded = ops.concatenate([puzzle_emb_padding, input_emb], axis=1)

        # Update low-level state
        z_L = self.L_level(z_L, input_emb_padded, training=training)

        # Update high-level state
        z_H = self.H_level(z_H, z_L, training=training)

        # Detach new carry states from the computation graph for the next iteration
        # This prevents gradients from flowing through multiple unrolled steps
        new_carry = {
            "z_H": tf.stop_gradient(z_H),
            "z_L": tf.stop_gradient(z_L)
        }

        # Generate output logits (excluding puzzle embedding positions)
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Generate halting probabilities from the first token
        q_logits = self.q_head(z_H[:, 0])
        q_halt, q_continue = q_logits[..., 0], q_logits[..., 1]

        return new_carry, logits, (q_halt, q_continue)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dict[str, Any]: Configuration dictionary containing all parameters needed
                to reconstruct this layer.
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


@keras.saving.register_keras_serializable()
class TinyRecursiveReasoningModel(keras.Model):
    """
    Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

    This model recursively refines its predictions over a variable number of steps.
    It starts with an initial state and iteratively updates it using a core reasoning
    module (`TRMInner`). The decision to continue or halt is learned, allowing
    the model to allocate more computation to harder problems.

    **Intent**: To solve complex reasoning tasks with a highly parameter-efficient
    model by applying the same small network (`TRMInner`) recursively, progressively
    improving the solution.

    **Architecture**:
    The model's `call` method implements a single step of the ACT loop. An external
    training script is expected to manage the state (`carry`) and loop until all
    sequences in the batch have halted. The core computation is delegated to the
    `TRMInner` layer.

    **State Management**:
    - `carry`: A dictionary holding the state between steps, including:
      - `inner_carry`: The `z_H` and `z_L` latent states for `TRMInner`.
      - `steps`: The current step count for each item in the batch.
      - `halted`: A boolean mask indicating which items have halted.
      - `current_data`: The input data for non-halted items.

    Args:
        vocab_size (int): Size of the vocabulary for token embeddings.
        hidden_size (int): Dimensionality of hidden states.
        num_heads (int): Number of attention heads in transformer layers.
        expansion (float): Factor to determine FFN intermediate size.
        seq_len (int): Length of the input sequence (excluding puzzle embedding).
        puzzle_emb_len (int): Length of the puzzle embedding prefix. Default is 16.
        h_layers (int): Number of layers in the H_level reasoning module. Default is 2.
        l_layers (int): Number of layers in the L_level reasoning module. Default is 2.
        halt_max_steps (int): Maximum number of ACT steps allowed. Default is 10.
        halt_exploration_prob (float): Probability of exploration during halting
            decisions. Default is 0.1.
        no_act_continue (bool): Whether to use simple halting (True) or Q-learning
            based halting (False). Default is True.
        rope_theta (float): Theta value for RoPE (Rotary Position Embedding).
            Default is 10000.0.
        attention_type (str): Type of attention mechanism to use. Default is 'multi_head'.
        ffn_type (str): Type of feed-forward network to use. Default is 'swiglu'.
        normalization_type (str): Type of normalization layer to use. Default is 'rms_norm'.
        normalization_position (str): Position of normalization ('pre' or 'post').
            Default is 'post'.
        dropout_rate (float): Dropout rate for transformer layers. Default is 0.0.
        attention_dropout_rate (float): Dropout rate specifically for attention.
            Default is 0.0.
        **kwargs: Additional arguments for the `keras.Model` base class.
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
        halt_max_steps: int = 10,
        halt_exploration_prob: float = 0.1,
        no_act_continue: bool = True,
        rope_theta: float = 10000.0,
        attention_type: AttentionType = 'multi_head',
        ffn_type: FFNType = 'swiglu',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store all configuration parameters as instance attributes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.seq_len = seq_len
        self.puzzle_emb_len = puzzle_emb_len
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.no_act_continue = no_act_continue
        self.rope_theta = rope_theta
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        # CREATE the main sub-layer in __init__ following the Golden Rule.
        # We will explicitly build it in this model's `build` method to ensure
        # its weights are available before the first `call`.
        self.inner = TRMInner(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            h_layers=h_layers,
            l_layers=l_layers,
            rope_theta=rope_theta,
            attention_type=attention_type,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            name="trm_inner"
        )

    def build(self, input_shape: Optional[Any] = None) -> None:
        """
        Build the model and its inner layer.

        This explicit build call is crucial. It ensures that `self.inner.H_init`
        and `self.inner.L_init` are created before the `call` method tries to
        access them for the state reset logic. Without this, an error occurs
        because the weights don't exist yet on the first call.

        Args:
            input_shape (Optional[Any]): Shape of the input. Not used since the
                inner layer handles its own shape inference.
        """
        if not self.inner.built:
            self.inner.build()
        super().build(input_shape)

    def initial_carry(self, batch: Dict[str, keras.KerasTensor]) -> Dict[str, Any]:
        """
        Create the initial state for the ACT loop.

        This method initializes all state variables needed for the recursive
        reasoning process, including latent states, step counters, halting flags,
        and current data.

        Args:
            batch (Dict[str, keras.KerasTensor]): A batch of input data containing:
                - `inputs` (keras.KerasTensor): Input token IDs with shape
                    (batch_size, seq_len).

        Returns:
            Dict[str, Any]: The initial `carry` dictionary containing:
                - `inner_carry`: Initial latent states (all zeros).
                - `steps`: Step counter initialized to 0.
                - `halted`: Boolean mask initialized to True (triggers reset on first step).
                - `current_data`: Data tensor initialized to zeros.
        """
        batch_size = ops.shape(batch["inputs"])[0]
        full_shape = (
            batch_size,
            self.seq_len + self.puzzle_emb_len,
            self.hidden_size
        )

        return {
            "inner_carry": {
                "z_H": ops.zeros(full_shape, dtype=self.compute_dtype),
                "z_L": ops.zeros(full_shape, dtype=self.compute_dtype),
            },
            "steps": ops.zeros((batch_size,), dtype="int32"),
            # Start with `halted` as True to trigger a reset on the first step.
            "halted": ops.ones((batch_size,), dtype="bool"),
            "current_data": {k: ops.zeros_like(v) for k, v in batch.items()},
        }

    def call(
            self,
            carry: Dict[str, Any],
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], Dict[str, keras.KerasTensor]]:
        """
        Perform one step of the ACT reasoning process.

        This method implements a single iteration of the adaptive computation loop.
        It handles state resetting for newly started sequences, delegates computation
        to the inner layer, and manages the halting logic.

        Args:
            carry (Dict[str, Any]): The state from the previous step containing:
                - `inner_carry`: Latent states from previous step.
                - `steps`: Current step count.
                - `halted`: Boolean mask of halted sequences.
                - `current_data`: Current input data.
            batch (Dict[str, keras.KerasTensor]): The current batch of data containing:
                - `inputs`: Input token IDs.
            training (Optional[bool]): Boolean flag for training mode. Affects halting
                behavior (training uses learned halting, inference uses max steps).

        Returns:
            Tuple containing:
            - new_carry (Dict[str, Any]): The updated state for the next step.
            - outputs (Dict[str, keras.KerasTensor]): The model outputs for this step:
                - `logits`: Prediction logits.
                - `q_halt_logits`: Halting probability logits.
                - `q_continue_logits`: Continuation probability logits.
                - `target_q_continue` (optional): Target Q-value for Bellman update
                    (only present during training with Q-learning).
        """
        inner_carry = carry["inner_carry"]
        halted = carry["halted"]

        # Reset inner state (z_H, z_L) for newly started sequences using
        # the initial state weights from the (now built) `inner` layer.
        # Broadcasting is handled by ops.where with appropriate expansion.
        reset_flag = ops.expand_dims(halted, axis=(-1, -2))
        z_H = ops.where(reset_flag, self.inner.H_init, inner_carry["z_H"])
        z_L = ops.where(reset_flag, self.inner.L_init, inner_carry["z_L"])

        # Reset step counter for newly started sequences
        steps = ops.where(halted, 0, carry["steps"])

        # Update the data for sequences that have not yet halted.
        # For halted sequences, use new batch data; for non-halted, keep current.
        current_data = {}
        for k, v in batch.items():
            # Expand halted mask to match data dimensions
            expand_dims = (1,) * (len(v.shape) - 1)
            halted_expanded = ops.reshape(halted, (-1, *expand_dims))
            current_data[k] = ops.where(halted_expanded, v, carry["current_data"][k])

        # Perform inner reasoning step
        new_inner_carry, logits, (q_halt, q_continue) = self.inner(
            {"z_H": z_H, "z_L": z_L}, current_data, training=training
        )

        # Prepare outputs
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue
        }

        # --- Halting Logic (No Gradients) ---
        # Increment step counter
        steps = steps + 1

        # Check if maximum steps reached
        is_last_step = steps >= self.halt_max_steps
        new_halted = is_last_step

        if training and self.halt_max_steps > 1:
            # Training mode: use learned halting signals
            if self.no_act_continue:
                # Simple halting: halt if q_halt > 0
                halt_signal = q_halt > 0
            else:
                # Q-learning halting: halt if q_halt > q_continue
                halt_signal = q_halt > q_continue
            new_halted = new_halted | halt_signal

            # Exploration: randomly force continuation for some sequences
            rand_val = keras.random.uniform(ops.shape(q_halt))
            explore_halt = rand_val < self.halt_exploration_prob
            min_halt_steps = ops.cast(explore_halt, "int32") * keras.random.randint(
                ops.shape(steps), 2, self.halt_max_steps + 1
            )
            new_halted = new_halted & (steps >= min_halt_steps)

            if not self.no_act_continue:
                # Q-learning: compute target Q-value for Bellman update
                # Lookahead one step to get target Q-value
                _, _, (next_q_halt, next_q_continue) = self.inner(
                    new_inner_carry, current_data, training=training
                )
                # Target is the maximum Q-value at the next state
                target_q = ops.where(
                    is_last_step,
                    next_q_halt,
                    ops.maximum(next_q_halt, next_q_continue)
                )
                outputs["target_q_continue"] = ops.sigmoid(target_q)

        if not training:
            # Inference mode: only halt at the maximum step limit
            new_halted = is_last_step

        # Construct new carry state
        new_carry = {
            "inner_carry": new_inner_carry,
            "steps": steps,
            "halted": new_halted,
            "current_data": current_data,
        }

        return new_carry, outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dict[str, Any]: Configuration dictionary containing all parameters needed
                to reconstruct this model.
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
            'halt_max_steps': self.halt_max_steps,
            'halt_exploration_prob': self.halt_exploration_prob,
            'no_act_continue': self.no_act_continue,
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