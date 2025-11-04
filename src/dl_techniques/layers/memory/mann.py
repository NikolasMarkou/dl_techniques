"""
Memory-Augmented Neural Network (MANN) based on
the Neural Turing Machine (NTM) architecture.

**What is this?**

This layer provides a neural network with a form of external, learnable memory,
analogous to the memory in a traditional computer. It enhances a standard recurrent
"controller" network (like an LSTM or GRU) with a large, addressable memory matrix.
Unlike traditional RNNs that must compress all long-term information into a
fixed-size hidden state, an NTM can explicitly store and retrieve information
over extended periods, overcoming the vanishing gradient problem for long-term
dependencies.

The core innovation lies in its **differentiable read and write operations**. The
controller learns *how* to interact with the memory via gradient descent. This is
achieved through a sophisticated addressing mechanism that combines two strategies:

1.  **Content-Based Addressing**: Allows the controller to focus on memory locations
    based on the similarity of their content to a query (like a "search"). This is
    powerful for associative recall.
2.  **Location-Based Addressing**: Allows the controller to iterate through memory
    locations sequentially or jump to specific offsets (like a "pointer"). This is
    essential for sequential data processing and implementing iterative algorithms.

By blending these two mechanisms, the NTM can learn simple computer programs and
algorithms directly from input-output examples, a capability far beyond that of
standard recurrent networks.

**Key Capabilities:**
- **External Memory**: Offloads the task of storing long-term information from the
  controller's hidden state to a dedicated memory matrix.
- **Algorithmic Learning**: Capable of learning simple algorithms like copying,
  sorting, and associative recall from data.
- **Long-Term Dependencies**: Excels at tasks where information from many timesteps
  in the past is required to make a correct prediction.
- **Differentiable Interface**: The entire memory interaction process is end-to-end
  differentiable, allowing for training with standard backpropagation.

**Primary Use Cases (Based on the NTM Paper):**

The NTM was originally designed and tested on a suite of simple algorithmic tasks
to demonstrate its memory and control capabilities. This layer is well-suited for
replicating these and similar tasks:

- **Copy Task**: The network is shown a sequence of vectors and must reproduce it
  identically. This tests the basic ability to store (write) and recall (read)
  information from memory.
- **Repeat Copy Task**: The network is shown a sequence and must reproduce it a
  variable number of times. This tests the ability to learn a simple loop and
  iteratively access memory.
- **Associative Recall**: The network is shown a sequence of item pairs and must
  later recall a specific item when queried with its pair. This tests content-based
  addressing.
- **Sorting Task**: The network is shown a sequence of numbers and their associated
  priorities and must output the numbers sorted by their priority. This demonstrates
  the ability to learn a more complex, non-trivial algorithm.

**Broader Applications:**

While benchmarked on simple algorithms, the principles of NTMs are applicable to
more complex, real-world problems:

- **Question Answering over Long Documents**: The memory can store key facts from a
  document, which the controller can then query to answer questions.
- **Reinforcement Learning**: An agent's memory can store past experiences (states,
  actions, rewards) to inform long-term planning and strategy.
- **Dialogue Systems**: Maintaining the context of a long conversation by storing
  key entities, facts, and user intentions in memory.
- **Program Synthesis**: Using the memory as a "scratchpad" to store variables and
  intermediate results while generating code.

**How to Use in a Model:**

```python
import keras
from keras import layers

# 1. Define the MANN layer with a specified memory configuration
mann_layer = MannLayer(
    memory_locations=128,  # Number of memory slots
    memory_dim=40,         # Dimension of each memory slot
    controller_units=100,  # Size of the LSTM/GRU controller
    num_read_heads=1,      # Number of read heads
    num_write_heads=1      # Number of write heads
)

# 2. Build a model using the layer
# Input shape: (batch_size, sequence_length, feature_dim)
inputs = keras.Input(shape=(None, 20))
mann_outputs = mann_layer(inputs)

# 3. Add subsequent layers to process the MANN output
# The output includes the controller state and what was read from memory
final_outputs = layers.Dense(10, activation="softmax")(mann_outputs)

model = keras.Model(inputs, final_outputs)
model.summary()

# 4. Compile and train as a standard Keras model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
"""

import keras
from keras import ops, layers, initializers
from typing import Optional, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MannLayer(keras.layers.Layer):
    """
    Memory-Augmented Neural Network (MANN) layer based on Neural Turing Machines.

    This layer implements a Neural Turing Machine (NTM), which enhances a standard
    recurrent controller with an external memory matrix. The controller learns to
    interact with the memory through differentiable read and write operations,
    enabling it to solve tasks requiring long-term memory and algorithmic reasoning.

    **Intent**: Provide a state-of-the-art, configurable MANN component for tasks
    where standard RNNs/LSTMs fail due to limited memory, such as sequence copying,
    sorting, and complex reasoning over long time-dependencies.

    **Architecture**:
    ```
          ┌───────────────────┐
   Input─►│ Controller (RNN)  ├─► Head Parameter Generator (Dense)
      ▲   │ (LSTM/GRU)        │               │
      │   └───────────────────┘               ▼
      │           ▲          ┌────────────────────────────────────────────┐
      │           │          │           Addressing Logic                 │
      │       Read Vectors   │ ┌─────────┐  ┌───────────┐ ┌──────────────┐│
      │           │          │ │ Content │► │ Location  │►│ Final Weights││
      │           │          │ └─────────┘  └───────────┘ └──────────────┘│
      │   ┌───────┴───────┐  └────────────────────────────────────────────┘
      └─ ─┤External Memory│ ◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
          │ (Read/Write)  │            Write/Erase Operations
          └───────────────┘
    ```

    **Data Flow per Timestep**:
    1. The `Controller` (LSTM/GRU) receives the current input and the read vectors
       from the previous timestep.
    2. The controller's output is passed to a `Dense` layer to generate a parameter
       vector for all read and write heads.
    3. **Addressing**: For each head, a final attention weighting over memory
       locations is computed through a combination of:
        a. **Content-based addressing**: Similarity (cosine) between a generated
           `key` and memory content.
        b. **Location-based addressing**: Interpolating with previous weights and
           applying a convolutional shift to iterate through memory.
    4. **Write Operations**: Write heads update the memory using the computed
       weights, an `erase` vector, and an `add` vector.
    5. **Read Operations**: Read heads retrieve information from the updated memory,
       producing `read vectors` for the next timestep.
    6. The final output of the layer at each timestep is a concatenation of the
       controller's output and all read vectors.

    Args:
        memory_locations: Integer, the number of memory slots (rows) in the
            external memory matrix.
        memory_dim: Integer, the dimensionality of each memory slot (columns).
        controller_units: Integer, the number of units in the recurrent controller.
        num_read_heads: Integer, the number of read heads to interact with memory.
        num_write_heads: Integer, the number of write heads to interact with memory.
        controller_type: Literal['lstm', 'gru'], the type of recurrent cell to use
            as the controller. Defaults to 'lstm'.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, controller_units + num_read_heads * memory_dim)`.

    Attributes:
        memory_matrix: The learnable external memory bank of shape
            (memory_locations, memory_dim). Created in `build()`.
        controller: The recurrent controller sub-layer (LSTM or GRU).
        param_generator: The dense layer that generates head parameters.

    Example:
        ```python
        # Configure a MANN layer for a sequence processing task
        mann_layer = MannLayer(
            memory_locations=128,
            memory_dim=40,
            controller_units=100,
            num_read_heads=1,
            num_write_heads=1
        )

        # Build a model
        inputs = keras.Input(shape=(None, 784)) # (batch, seq_len, features)
        mann_output = mann_layer(inputs)
        # The output can be processed by subsequent layers
        outputs = layers.Dense(10, activation='softmax')(mann_output)

        model = keras.Model(inputs, outputs)
        model.summary()
        ```

    References:
        - Graves, et al., 2014. "Neural Turing Machines". https://arxiv.org/abs/1410.5401
    """

    def __init__(
            self,
            memory_locations: int,
            memory_dim: int,
            controller_units: int,
            num_read_heads: int,
            num_write_heads: int,
            controller_type: Literal['lstm', 'gru'] = 'lstm',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if memory_locations <= 0:
            raise ValueError(f"memory_locations must be positive, got {memory_locations}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if controller_units <= 0:
            raise ValueError(f"controller_units must be positive, got {controller_units}")
        if num_read_heads < 0 or num_write_heads < 0:
            raise ValueError("Number of heads cannot be negative.")
        if controller_type not in ['lstm', 'gru']:
            raise ValueError(f"controller_type must be 'lstm' or 'gru', got {controller_type}")

        # Store ALL configuration
        self.memory_locations = memory_locations
        self.memory_dim = memory_dim
        self.controller_units = controller_units
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_type = controller_type

        # Calculate total parameters needed from controller per timestep
        # For each head (read or write): key(M), beta(1), gate(1), shift(3), gamma(1)
        params_per_head = self.memory_dim + 1 + 1 + 3 + 1
        # For each write head: erase(M), add(M)
        params_per_write_head = self.memory_dim * 2
        self.total_params_size = (
                (self.num_read_heads + self.num_write_heads) * params_per_head
                + self.num_write_heads * params_per_write_head
        )

        # CREATE sub-layers in __init__ (they are unbuilt)
        if self.controller_type == 'lstm':
            self.controller = layers.LSTM(
                self.controller_units, return_sequences=True, return_state=True
            )
        else:  # gru
            self.controller = layers.GRU(
                self.controller_units, return_sequences=True, return_state=True
            )

        self.param_generator = layers.Dense(self.total_params_size, name="param_generator")

        # Initialize weight attributes - created in build()
        self.memory_matrix = None
        # This weight is used for initializing the memory matrix at the start of each sequence
        self.initial_memory_vector = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's own weights and build sub-layers.

        This is called automatically when the layer first processes input.
        """
        # Create layer's own weights: the persistent memory matrix.
        # Although it's modified in call(), it's a learnable component.
        self.memory_matrix = self.add_weight(
            name='memory_matrix',
            shape=(self.memory_locations, self.memory_dim),
            initializer='glorot_uniform',
            trainable=True,
        )

        # A learnable initial state for the memory matrix
        self.initial_memory_vector = self.add_weight(
            name="initial_memory_vector",
            shape=(self.memory_locations, self.memory_dim),
            initializer=initializers.RandomNormal(mean=0.0, stddev=0.5),
            trainable=True,
        )

        # Build sub-layers explicitly
        # Controller input is `(batch, seq_len, input_dim + num_read_heads * memory_dim)`
        controller_input_dim = input_shape[-1] + self.num_read_heads * self.memory_dim
        self.controller.build((input_shape[0], input_shape[1], controller_input_dim))

        # Param generator input is controller output
        self.param_generator.build((input_shape[0], input_shape[1], self.controller_units))

        super().build(input_shape)

    def _calculate_head_addressing(
            self, params: keras.KerasTensor, prev_weights: keras.KerasTensor, memory: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Computes the final attention weights for a single head."""
        # Unpack parameters
        k = params[:, :self.memory_dim]
        beta = ops.softplus(params[:, self.memory_dim])
        g = ops.sigmoid(params[:, self.memory_dim + 1])
        s = ops.softmax(params[:, self.memory_dim + 2: self.memory_dim + 5], axis=-1)
        gamma = ops.softplus(params[:, self.memory_dim + 5]) + 1.0

        # 1. Content-based addressing
        norm_k = ops.normalize(k, axis=-1)
        norm_mem = ops.normalize(memory, axis=-1)
        # Cosine similarity -> (batch_size, memory_locations)
        similarity = ops.einsum('bi,ni->bn', norm_k, norm_mem)
        w_c = ops.softmax(beta[:, None] * similarity, axis=-1)

        # 2. Interpolation gate
        w_g = g[:, None] * w_c + (1 - g[:, None]) * prev_weights

        # 3. Convolutional shift
        # We perform a 1D circular convolution
        w_shifted = ops.zeros_like(w_g)
        for i in range(w_g.shape[0]):  # Iterate over batch
            # Convolve using indexing for circular shift
            convolved = ops.convolve(
                ops.expand_dims(ops.pad(w_g[i], [[1, 1]], mode="circular"), axis=[0, 2]),
                ops.expand_dims(ops.flip(s[i]), axis=[0, 2]),
                padding="valid",
            )[0, :, 0]
            w_shifted = ops.scatter_update(w_shifted, [i], convolved)

        # 4. Sharpening
        w_sharp = ops.power(w_shifted, gamma[:, None])
        final_weights = w_sharp / (ops.sum(w_sharp, axis=-1, keepdims=True) + 1e-8)

        return final_weights

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation over a sequence."""
        batch_size = ops.shape(inputs)[0]
        sequence_length = ops.shape(inputs)[1]

        # Initialize states for the sequence processing loop
        memory = ops.tile(ops.expand_dims(self.initial_memory_vector, 0), [batch_size, 1, 1])

        # Initial read/write weights (uniform or random)
        initial_head_weights = ops.softmax(
            ops.zeros((batch_size, self.memory_locations)), axis=-1
        )
        prev_read_weights = [initial_head_weights] * self.num_read_heads
        prev_write_weights = [initial_head_weights] * self.num_write_heads

        # Initial read vectors are zeros
        prev_read_vectors = [
            ops.zeros((batch_size, self.memory_dim)) for _ in range(self.num_read_heads)
        ]

        # Initial controller state
        if self.controller_type == 'lstm':
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
                training=training
            )
            controller_output = ops.squeeze(controller_output, 1)

            # Generate and slice head parameters
            all_params = self.param_generator(controller_output, training=training)

            param_idx = 0
            params_per_head = self.memory_dim + 1 + 1 + 3 + 1

            # Process read heads
            read_weights = []
            for i in range(self.num_read_heads):
                params = all_params[:, param_idx: param_idx + params_per_head]
                param_idx += params_per_head
                w = self._calculate_head_addressing(params, prev_read_weights[i], memory)
                read_weights.append(w)

            # Process write heads
            write_weights = []
            erase_add_params = []
            params_per_write_head = self.memory_dim * 2
            for i in range(self.num_write_heads):
                # Addressing params
                params = all_params[:, param_idx: param_idx + params_per_head]
                param_idx += params_per_head
                w = self._calculate_head_addressing(params, prev_write_weights[i], memory)
                write_weights.append(w)

                # Erase/add params
                ea_params = all_params[:, param_idx: param_idx + params_per_write_head]
                param_idx += params_per_write_head
                erase_add_params.append(ea_params)

            # Update memory (write operations)
            for i in range(self.num_write_heads):
                w = write_weights[i]  # (batch, locations)
                ea = erase_add_params[i]
                erase_vec = ops.sigmoid(ea[:, :self.memory_dim])  # (batch, dim)
                add_vec = ea[:, self.memory_dim:]  # (batch, dim)

                # Outer products for erase and add
                erase_term = ops.einsum('bl,bd->bld', w, erase_vec)
                add_term = ops.einsum('bl,bd->bld', w, add_vec)

                memory = memory * (1. - erase_term) + add_term

            # Update read vectors for next timestep
            read_vectors = []
            for i in range(self.num_read_heads):
                r = ops.einsum('bl,bld->bd', read_weights[i], memory)
                read_vectors.append(r)

            # Store final output for this timestep
            timestep_output = ops.concatenate([controller_output] + read_vectors, axis=-1)
            outputs.append(timestep_output)

            # Update previous states for next iteration
            prev_read_weights = read_weights
            prev_write_weights = write_weights
            prev_read_vectors = read_vectors

        # Stack outputs into a single tensor
        final_output = ops.stack(outputs, axis=1)
        return final_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (
            input_shape[0],
            input_shape[1],
            self.controller_units + self.num_read_heads * self.memory_dim
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'memory_locations': self.memory_locations,
            'memory_dim': self.memory_dim,
            'controller_units': self.controller_units,
            'num_read_heads': self.num_read_heads,
            'num_write_heads': self.num_write_heads,
            'controller_type': self.controller_type,
        })
        return config

# ---------------------------------------------------------------------
