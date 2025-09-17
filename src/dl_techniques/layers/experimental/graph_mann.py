import keras
from keras import ops, layers, initializers
from typing import Optional, Tuple, Dict, Any, Literal, List

@keras.saving.register_keras_serializable()
class GraphMannLayer(keras.layers.Layer):
    """
    Graph Memory-Augmented Neural Network (GMANN) layer based on NTM principles.

    This layer implements a variant of a Memory-Augmented Neural Network where the
    external memory is structured as a graph. The controller learns to interact
    with memory nodes through differentiable read, write, and graph traversal
    operations, enabling it to solve tasks requiring relational reasoning.

    **Intent**: Provide a configurable GMANN component for tasks where relationships
    and structure are as important as content, such as knowledge base reasoning,
    scene understanding, or structured data processing. This implementation follows
    modern Keras 3 best practices for robust serialization and composability.

    **Architecture**:
    ```
          ┌───────────────────┐
    Input─► Controller (RNN)  ├─► Head Parameter Generator (Dense)
      ▲   │ (LSTM/GRU)        │               │
      │   └───────────────────┘               ▼
      │           ▲          ┌──────────────────────────────────┐
      │           │          │           Addressing Logic       │
      │       Read Vectors   │ ┌─────────┐  ┌───────────┐ ┌─────┤
      │           │          │ │ Content │► │ Graph     │►│ Final Weights │
      │           │          │ └─────────┘  └───────────┘ └─────┤
      │   ┌───────┴───────┐  └──────────────────────────────────┘
      └─ ─┤ Graph Memory  ◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
          │ (Read/Write)  │            Write/Erase Operations
          └───────────────┘
    ```

    **Data Flow per Timestep**:
    1. The `Controller` (LSTM/GRU) receives the current input and the read vectors
       from the previous timestep.
    2. The controller's output is passed to a `Dense` layer to generate a parameter
       vector for all read and write heads.
    3. **Addressing**: For each head, a final attention weighting over memory
       nodes is computed by blending:
        a. **Content-based addressing**: Similarity (cosine) between a generated
           `key` and memory node content.
        b. **Graph-based addressing**: Propagating attention weights across the
           graph's edges, allowing the head to shift focus to related nodes.
    4. **Write Operations**: Write heads update the features of memory nodes using
       the computed weights, an `erase` vector, and an `add` vector.
    5. **Read Operations**: Read heads retrieve information from the updated memory,
       producing `read vectors` for the next timestep.
    6. The final output of the layer at each timestep is a concatenation of the
       controller's output and all read vectors.

    Args:
        num_memory_nodes: Integer, the number of nodes in the memory graph.
            Must be a positive integer.
        memory_dim: Integer, the dimensionality of each memory node's state vector.
            Must be a positive integer.
        controller_units: Integer, the number of units in the recurrent controller.
            Must be a positive integer.
        num_read_heads: Integer, the number of read heads to interact with memory.
        num_write_heads: Integer, the number of write heads to interact with memory.
        controller_type: Literal['lstm', 'gru'], the type of recurrent cell to use
            as the controller. Defaults to 'lstm'.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length,
        controller_units + num_read_heads * memory_dim)`.

    Attributes:
        memory_nodes: The primary state of the memory graph of shape
            (num_memory_nodes, memory_dim).
        adjacency_matrix: The learnable graph structure of shape
            (num_memory_nodes, num_memory_nodes).
        controller: The recurrent controller sub-layer (LSTM or GRU).
        param_generator: The dense layer that generates head parameters.

    Example:
        ```python
        # Configure a GMANN layer for a relational reasoning task
        gmann_layer = GraphMannLayer(
            num_memory_nodes=128,
            memory_dim=40,
            controller_units=100,
            num_read_heads=1,
            num_write_heads=1
        )

        # Build a model
        inputs = keras.Input(shape=(None, 20)) # (batch, seq_len, features)
        gmann_output = gmann_layer(inputs)

        # The output can be processed by subsequent layers
        outputs = layers.Dense(10, activation='softmax')(gmann_output)
        model = keras.Model(inputs, outputs)
        model.summary()
        ```

    References:
        - Graves, et al., 2014. "Neural Turing Machines".
          https://arxiv.org/abs/1410.5401
    """

    def __init__(
        self,
        num_memory_nodes: int,
        memory_dim: int,
        controller_units: int,
        num_read_heads: int,
        num_write_heads: int,
        controller_type: Literal["lstm", "gru"] = "lstm",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_memory_nodes <= 0:
            raise ValueError(
                f"num_memory_nodes must be positive, got {num_memory_nodes}"
            )
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if controller_units <= 0:
            raise ValueError(
                f"controller_units must be positive, got {controller_units}"
            )
        if num_read_heads < 0 or num_write_heads < 0:
            raise ValueError("Number of heads cannot be negative.")
        if controller_type not in ["lstm", "gru"]:
            raise ValueError(
                f"controller_type must be 'lstm' or 'gru', got {controller_type}"
            )

        # Store ALL configuration
        self.num_memory_nodes = num_memory_nodes
        self.memory_dim = memory_dim
        self.controller_units = controller_units
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_type = controller_type

        # Calculate total parameters needed from controller per timestep
        # For each head: key(M), beta(1), gate(1), gamma(1)
        params_per_head = self.memory_dim + 1 + 1 + 1
        # For each write head: erase(M), add(M)
        params_per_write_head = self.memory_dim * 2
        self.total_params_size = (
            (self.num_read_heads + self.num_write_heads) * params_per_head
            + self.num_write_heads * params_per_write_head
        )

        # CREATE sub-layers in __init__ (they are unbuilt), per the guide.
        if self.controller_type == "lstm":
            self.controller = layers.LSTM(
                self.controller_units, return_sequences=True, return_state=True
            )
        else:  # gru
            self.controller = layers.GRU(
                self.controller_units, return_sequences=True, return_state=True
            )

        self.param_generator = layers.Dense(
            self.total_params_size, name="param_generator"
        )

        # Initialize weight attributes - created in build()
        self.memory_nodes = None
        self.adjacency_matrix = None
        self.initial_memory_nodes = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights and build sub-layers."""
        # CREATE layer's own weights using add_weight()
        self.memory_nodes = self.add_weight(
            name="memory_nodes",
            shape=(self.num_memory_nodes, self.memory_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.adjacency_matrix = self.add_weight(
            name="adjacency_matrix",
            shape=(self.num_memory_nodes, self.num_memory_nodes),
            initializer="ones",
            trainable=True,
        )

        self.initial_memory_nodes = self.add_weight(
            name="initial_memory_nodes",
            shape=(self.num_memory_nodes, self.memory_dim),
            initializer=initializers.RandomNormal(mean=0.0, stddev=0.5),
            trainable=True,
        )

        # CRITICAL: Explicitly build each sub-layer for robust serialization.
        # This is the core principle of the "Composite Layer" pattern.
        controller_input_dim = (
            input_shape[-1] + self.num_read_heads * self.memory_dim
        )
        self.controller.build(
            (input_shape[0], input_shape[1], controller_input_dim)
        )

        param_generator_input_shape = (
            input_shape[0],
            input_shape[1],
            self.controller_units,
        )
        self.param_generator.build(param_generator_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _calculate_head_addressing(
        self,
        params: keras.KerasTensor,
        prev_weights: keras.KerasTensor,
        memory: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Computes the final attention weights for a single head.

        This involves blending content-based attention with graph-based traversal.
        """
        # Unpack parameters
        k = params[:, : self.memory_dim]
        beta = ops.softplus(params[:, self.memory_dim])
        g = ops.sigmoid(params[:, self.memory_dim + 1])
        gamma = ops.softplus(params[:, self.memory_dim + 2]) + 1.0

        # 1. Content-based addressing
        norm_k = ops.normalize(k, axis=-1)
        norm_mem = ops.normalize(memory, axis=-1)
        similarity = ops.einsum("bi,ni->bn", norm_k, norm_mem)
        w_c = ops.softmax(beta[:, None] * similarity, axis=-1)

        # 2. Interpolation gate (blend content and previous focus)
        w_g = g[:, None] * w_c + (1 - g[:, None]) * prev_weights

        # 3. Graph-based traversal
        # Propagate weights to neighbors using the learnable adjacency matrix
        normalized_adj = self.adjacency_matrix / (
            ops.sum(self.adjacency_matrix, axis=0, keepdims=True) + 1e-8
        )
        w_shifted = ops.einsum("bn,nm->bm", w_g, normalized_adj)

        # 4. Sharpening
        w_sharp = ops.power(w_shifted, gamma[:, None])
        final_weights = w_sharp / (
            ops.sum(w_sharp, axis=-1, keepdims=True) + 1e-8
        )

        return final_weights

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation over a sequence."""
        batch_size = ops.shape(inputs)[0]
        sequence_length = ops.shape(inputs)[1]

        # Initialize states for the sequence processing loop
        memory = ops.tile(
            ops.expand_dims(self.initial_memory_nodes, 0), [batch_size, 1, 1]
        )

        initial_head_weights = ops.softmax(
            ops.zeros((batch_size, self.num_memory_nodes)), axis=-1
        )
        prev_read_weights = [initial_head_weights] * self.num_read_heads
        prev_write_weights = [initial_head_weights] * self.num_write_heads

        prev_read_vectors = [
            ops.zeros((batch_size, self.memory_dim))
            for _ in range(self.num_read_heads)
        ]

        if self.controller_type == "lstm":
            controller_state = [
                ops.zeros((batch_size, self.controller_units)),
                ops.zeros((batch_size, self.controller_units)),
            ]
        else:  # gru
            controller_state = [ops.zeros((batch_size, self.controller_units))]

        # List to store outputs at each timestep
        outputs = []

        # Main loop over the time sequence
        for t in range(sequence_length):
            x_t = inputs[:, t, :]

            # Concatenate input with previous read vectors
            controller_input = ops.concatenate([x_t] + prev_read_vectors, axis=-1)

            # Run one step of the controller
            controller_output, *controller_state = self.controller(
                ops.expand_dims(controller_input, 1),
                states=controller_state,
                training=training,
            )
            controller_output = ops.squeeze(controller_output, 1)

            # Generate and slice head parameters
            all_params = self.param_generator(
                controller_output, training=training
            )

            param_idx = 0
            params_per_head = self.memory_dim + 1 + 1 + 1

            # Process read heads
            read_weights = self._process_heads(
                all_params,
                param_idx,
                params_per_head,
                self.num_read_heads,
                prev_read_weights,
                memory,
            )
            param_idx += self.num_read_heads * params_per_head

            # Process write heads
            write_weights = self._process_heads(
                all_params,
                param_idx,
                params_per_head,
                self.num_write_heads,
                prev_write_weights,
                memory,
            )
            param_idx += self.num_write_heads * params_per_head

            # Update memory (write operations)
            memory = self._perform_writes(
                memory, all_params, param_idx, write_weights
            )

            # Update read vectors for next timestep
            read_vectors = []
            for i in range(self.num_read_heads):
                r = ops.einsum("bl,bld->bd", read_weights[i], memory)
                read_vectors.append(r)

            # Store final output for this timestep
            timestep_output = ops.concatenate(
                [controller_output] + read_vectors, axis=-1
            )
            outputs.append(timestep_output)

            # Update previous states for next iteration
            prev_read_weights = read_weights
            prev_write_weights = write_weights
            prev_read_vectors = read_vectors

        # Stack outputs into a single tensor
        final_output = ops.stack(outputs, axis=1)
        return final_output

    def _process_heads(
        self,
        all_params: keras.KerasTensor,
        start_idx: int,
        params_per_head: int,
        num_heads: int,
        prev_weights: List[keras.KerasTensor],
        memory: keras.KerasTensor,
    ) -> List[keras.KerasTensor]:
        """Helper to process read or write head addressing."""
        head_weights = []
        for i in range(num_heads):
            param_slice = all_params[
                :, start_idx : start_idx + params_per_head
            ]
            start_idx += params_per_head
            w = self._calculate_head_addressing(
                param_slice, prev_weights[i], memory
            )
            head_weights.append(w)
        return head_weights

    def _perform_writes(
        self,
        memory: keras.KerasTensor,
        all_params: keras.KerasTensor,
        start_idx: int,
        write_weights: List[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Helper to perform all write operations on memory."""
        params_per_write_head = self.memory_dim * 2
        for i in range(self.num_write_heads):
            w = write_weights[i]
            ea_params = all_params[
                :, start_idx : start_idx + params_per_write_head
            ]
            start_idx += params_per_write_head

            erase_vec = ops.sigmoid(ea_params[:, : self.memory_dim])
            add_vec = ea_params[:, self.memory_dim :]

            erase_term = ops.einsum("bl,bd->bld", w, erase_vec)
            add_term = ops.einsum("bl,bd->bld", w, add_vec)

            memory = memory * (1.0 - erase_term) + add_term
        return memory

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (
            input_shape[0],
            input_shape[1],
            self.controller_units + self.num_read_heads * self.memory_dim,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_memory_nodes": self.num_memory_nodes,
                "memory_dim": self.memory_dim,
                "controller_units": self.controller_units,
                "num_read_heads": self.num_read_heads,
                "num_write_heads": self.num_write_heads,
                "controller_type": self.controller_type,
            }
        )
        return config
