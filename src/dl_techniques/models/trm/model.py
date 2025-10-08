"""
Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

This model performs recursive reasoning to solve complex tasks by repeatedly
applying a small, shared neural network over a variable number of steps. It
integrates the principles of Adaptive Computation Time (ACT) to learn how
many computational steps are necessary for a given problem, allowing it to
dynamically allocate resources. The model's `forward` method encapsulates
a single step of this adaptive, recursive process.

The architecture is conceptually divided into two nested levels of recursion:
1.  An outer ACT loop, managed externally by the training script, which
    calls this model's `forward` method repeatedly. This loop allows the
    model to progressively refine its solution over multiple "thought" steps.
2.  An inner, fixed-cycle reasoning process within the `TRMInner` submodule.
    During each single call to this model's `forward` method, the inner
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
import tensorflow as tf
from keras import ops, layers, initializers
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformer import TransformerLayer


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TRMReasoningModule(layers.Layer):
    """
    A module that stacks multiple TransformerLayers for the reasoning process.

    This layer serves as the core computational engine of the TRM, applying a
    sequence of transformations to the latent states. It is composed of `L_layers`
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
        config (Dict[str, Any]): A dictionary containing model configuration parameters.
            Expected keys include:
            - `hidden_size` (int): The dimensionality of the hidden states.
            - `expansion` (float): Factor to determine the FFN intermediate size.
            - `num_heads` (int): Number of attention heads.
            - `seq_len` (int): The length of the input sequence.
            - `puzzle_emb_len` (int): Length of the puzzle embedding prefix.
            - `rope_theta` (float, optional): Theta value for RoPE.
            - `L_layers` (int): The number of `TransformerLayer` instances to stack.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

        # Calculate intermediate size for the FFN inside the TransformerLayer
        intermediate_size = int(config["hidden_size"] * config["expansion"])

        # CREATE sub-layers in __init__ as per the guide's Golden Rule.
        # These layers are instantiated but not yet built.
        self.layers_list = [
            TransformerLayer(
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                intermediate_size=intermediate_size,
                attention_type='multi_head',
                attention_args={
                    'max_seq_len': config["seq_len"] + config.get("puzzle_emb_len", 16),
                    'rope_theta': config.get("rope_theta", 10000.0)
                },
                normalization_type='rms_norm',
                normalization_position='post',
                ffn_type='swiglu',
                dropout_rate=0.0,
                attention_dropout_rate=0.0,
                name=f"transformer_block_{i}"
            ) for i in range(config["L_layers"])
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
            input_injection (keras.KerasTensor): The tensor to be added to the hidden_states at the start.
            training (Optional[bool]): Boolean flag for training mode, passed to sub-layers.

        Returns:
            keras.KerasTensor: The transformed hidden_states tensor.
        """
        hidden_states = hidden_states + input_injection
        for layer in self.layers_list:
            hidden_states = layer(hidden_states, training=training)
        return hidden_states

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """The output shape is identical to the input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        # Store the full configuration dictionary, ensuring all parameters
        # needed for __init__ are present.
        config.update({"config": self.config})
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TRMInner(layers.Layer):
    """
    The inner computational core of the TRM model.

    This layer orchestrates a single, multi-cycle reasoning step. It manages the
    input embeddings and the recursive application of the `TRMReasoningModule`
    to update the latent states `z_H` and `z_L`. It demonstrates a complex
    composite layer with both sub-layers and its own weights.

    **Intent**: To encapsulate one full reasoning "thought process" of the TRM,
    including state updates and output generation, which can be called
    recursively by the parent model.

    **Architecture**:
    1.  Input token IDs and puzzle IDs are converted to embeddings.
    2.  The core logic involves a nested loop structure:
        - Outer loop (`H_cycles`): Updates high-level state `z_H`.
        - Inner loop (`L_cycles`): Refines low-level state `z_L`.
        - Both loops utilize the same `TRMReasoningModule` (`L_level`).
    3.  Final `z_H` is used to produce token logits (`lm_head`) and
        halting Q-values (`q_head`).

    Args:
        config (Dict[str, Any]): A dictionary containing model configuration parameters.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

        self.embed_scale = self.config["hidden_size"] ** 0.5
        embed_init_std = 1.0 / self.embed_scale

        # --- CREATE sub-layers in __init__ (unbuilt) ---
        self.embed_tokens = layers.Embedding(
            self.config["vocab_size"],
            self.config["hidden_size"],
            embeddings_initializer=initializers.TruncatedNormal(stddev=embed_init_std),
            name="embed_tokens",
        )
        self.lm_head = layers.Dense(self.config["vocab_size"], use_bias=False, name="lm_head")
        self.q_head = layers.Dense(2, use_bias=True, name="q_head")

        self.puzzle_emb_len = self.config.get("puzzle_emb_len", 16)
        self.puzzle_emb = layers.Embedding(
            self.config["num_puzzle_identifiers"],
            self.config["puzzle_emb_ndim"],
            embeddings_initializer="zeros",
            name="puzzle_emb",
        )

        self.L_level = TRMReasoningModule(config, name="L_level")

        # --- Initialize weight attributes to None ---
        # Weights will be created in build() as per the guide's Golden Rule.
        self.H_init = None
        self.L_init = None

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Build sub-layers and create initial state weights.

        This method correctly separates weight creation (`add_weight`) and
        sub-layer building from `__init__`, adhering to the guide's core principles.
        """
        if self.built:
            return

        # CREATE the layer's own weights using add_weight().
        self.H_init = self.add_weight(
            name="H_init",
            shape=(self.config["hidden_size"],),
            initializer=initializers.TruncatedNormal(stddev=1.0),
            trainable=True,
        )
        self.L_init = self.add_weight(
            name="L_init",
            shape=(self.config["hidden_size"],),
            initializer=initializers.TruncatedNormal(stddev=1.0),
            trainable=True,
        )

        # BUILD all sub-layers explicitly with their expected input shapes.
        # This is CRITICAL for robust serialization and deserialization.
        seq_plus_puzzle_len = self.config["seq_len"] + self.puzzle_emb_len
        full_shape = (None, seq_plus_puzzle_len, self.config["hidden_size"])
        hidden_state_shape = (None, self.config["hidden_size"])

        self.embed_tokens.build(None)  # Embedding layers can be built with None shape.
        self.puzzle_emb.build(None)
        self.L_level.build(full_shape)
        self.lm_head.build(full_shape)
        self.q_head.build(hidden_state_shape)

        super().build(input_shape)

    def _input_embeddings(
            self,
            inputs: keras.KerasTensor,
            puzzle_identifiers: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Constructs the combined input embeddings for the model.

        This helper method combines token and puzzle embeddings into a single
        tensor that serves as the input injection for the reasoning modules.

        Args:
            inputs (keras.KerasTensor): The input token IDs. Shape: `(batch, seq_len)`.
            puzzle_identifiers (keras.KerasTensor): The puzzle IDs. Shape: `(batch,)`.

        Returns:
            keras.KerasTensor: The combined and scaled embeddings.
                Shape: `(batch, puzzle_emb_len + seq_len, hidden_size)`.
        """
        token_embedding = self.embed_tokens(inputs)
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

        # Pad or reshape the puzzle embedding to match the expected format
        target_puzzle_dim = self.puzzle_emb_len * self.config["hidden_size"]
        pad_count = target_puzzle_dim - self.config["puzzle_emb_ndim"]
        if pad_count > 0:
            puzzle_embedding = ops.pad(puzzle_embedding, [[0, 0], [0, pad_count]])

        puzzle_embedding = ops.reshape(
            puzzle_embedding, (-1, self.puzzle_emb_len, self.config["hidden_size"])
        )

        embedding = ops.concatenate([puzzle_embedding, token_embedding], axis=1)
        return embedding * self.embed_scale

    def call(
            self,
            inner_carry: Dict[str, keras.KerasTensor],
            batch: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """A single step of the recursive reasoning process."""
        z_H, z_L = inner_carry["z_H"], inner_carry["z_L"]
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # H_cycles-1 iterations without gradient flow for state evolution.
        for _ in range(self.config["H_cycles"] - 1):
            z_L_no_grad = tf.stop_gradient(z_L)
            z_H_no_grad = tf.stop_gradient(z_H)

            # Inner L-level cycles
            temp_z_L = z_L_no_grad
            for _ in range(self.config["L_cycles"]):
                temp_z_L = self.L_level(
                    temp_z_L, z_H_no_grad + input_embeddings, training=training
                )
            z_L = temp_z_L

            # H-level update
            z_H = self.L_level(z_H_no_grad, z_L, training=training)

        # Final iteration with gradient flow
        for _ in range(self.config["L_cycles"]):
            z_L = self.L_level(z_L, z_H + input_embeddings, training=training)
        z_H = self.L_level(z_H, z_L, training=training)

        # Detach new carry states from the graph for the next iteration
        new_carry = {"z_H": tf.stop_gradient(z_H), "z_L": tf.stop_gradient(z_L)}

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0])
        q_halt, q_continue = q_logits[..., 0], q_logits[..., 1]

        return new_carry, logits, (q_halt, q_continue)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({"config": self.config})
        return config

# ---------------------------------------------------------------------

# dl_techniques/models/trm/model.py

"""
... (file content above this class is unchanged) ...
"""
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
        config (Dict[str, Any]): Dictionary containing all model hyperparameters.
        **kwargs: Additional arguments for the `keras.Model` base class.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        # CREATE the main sub-layer in __init__. We will explicitly build it
        # in this model's `build` method to ensure its weights are available
        # before the first `call`.
        self.inner = TRMInner(config, name="trm_inner")

    def build(self, input_shape: Any):
        """
        Builds the model and its inner layer.

        This explicit build call is crucial. It ensures that `self.inner.H_init`
        and `self.inner.L_init` are created before the `call` method tries to
        access them for the state reset logic. Without this, an error occurs
        because the weights don't exist yet on the first call.
        """
        if not self.inner.built:
            self.inner.build()
        super().build(input_shape)

    def initial_carry(self, batch: Dict[str, keras.KerasTensor]) -> Dict[str, Any]:
        """
        Creates the initial state for the ACT loop.

        Args:
            batch (Dict[str, keras.KerasTensor]): A batch of input data.

        Returns:
            Dict[str, Any]: The initial `carry` dictionary.
        """
        batch_size = ops.shape(batch["inputs"])[0]
        puzzle_emb_len = self.config.get("puzzle_emb_len", 16)
        full_shape = (batch_size, self.config["seq_len"] + puzzle_emb_len, self.config["hidden_size"])

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
        Performs one step of the ACT reasoning process.

        Args:
            carry (Dict[str, Any]): The state from the previous step.
            batch (Dict[str, keras.KerasTensor]): The current batch of data.
            training (Optional[bool]): Boolean flag for training mode.

        Returns:
            A tuple containing:
            - new_carry (Dict[str, Any]): The updated state for the next step.
            - outputs (Dict[str, keras.KerasTensor]): The model outputs for this step.
        """
        inner_carry = carry["inner_carry"]
        halted = carry["halted"]

        # Reset inner state (z_H, z_L) for newly started sequences using
        # the initial state weights from the (now built) `inner` layer.
        reset_flag = ops.expand_dims(halted, axis=(-1, -2))
        z_H = ops.where(reset_flag, self.inner.H_init, inner_carry["z_H"])
        z_L = ops.where(reset_flag, self.inner.L_init, inner_carry["z_L"])

        steps = ops.where(halted, 0, carry["steps"])

        # Update the data for sequences that have not yet halted.
        current_data = {}
        for k, v in batch.items():
            expand_dims = (1,) * (len(v.shape) - 1)
            halted_expanded = ops.reshape(halted, (-1, *expand_dims))
            current_data[k] = ops.where(halted_expanded, v, carry["current_data"][k])

        new_inner_carry, logits, (q_halt, q_continue) = self.inner(
            {"z_H": z_H, "z_L": z_L}, current_data, training=training
        )

        outputs = {"logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_continue}

        # --- Halting Logic (No Gradients) ---
        steps = steps + 1
        is_last_step = steps >= self.config["halt_max_steps"]
        new_halted = is_last_step

        if training and self.config["halt_max_steps"] > 1:
            if self.config.get("no_ACT_continue", True):
                halt_signal = q_halt > 0
            else:
                halt_signal = q_halt > q_continue
            new_halted = new_halted | halt_signal

            rand_val = keras.random.uniform(ops.shape(q_halt))
            explore_halt = rand_val < self.config["halt_exploration_prob"]
            min_halt_steps = ops.cast(explore_halt, "int32") * keras.random.randint(
                ops.shape(steps), 2, self.config["halt_max_steps"] + 1
            )
            new_halted = new_halted & (steps >= min_halt_steps)

            if not self.config.get("no_ACT_continue", True):
                # Lookahead one step to get target Q-value for Bellman update
                _, _, (next_q_halt, next_q_continue) = self.inner(
                    new_inner_carry, current_data, training=training
                )
                target_q = ops.where(is_last_step, next_q_halt, ops.maximum(next_q_halt, next_q_continue))
                outputs["target_q_continue"] = ops.sigmoid(target_q)

        if not training:
            # During inference, only halt at the maximum step limit.
            new_halted = is_last_step

        new_carry = {
            "inner_carry": new_inner_carry,
            "steps": steps,
            "halted": new_halted,
            "current_data": current_data,
        }
        return new_carry, outputs

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({"config": self.config})
        return config

# ---------------------------------------------------------------------
