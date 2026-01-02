"""
Baseline Neural Turing Machine Implementation.

This module provides a reference implementation of the NTM described in
Graves et al., 2014, updated for Keras 3 compatibility.
"""

import keras
from keras import ops
import numpy as np
from typing import Optional, Tuple, List, Any, Dict

from .ntm_interface import (
    BaseMemory,
    BaseHead,
    BaseController,
    BaseNTM,
    MemoryState,
    HeadState,
    NTMOutput,
    NTMConfig,
    AddressingMode,
    cosine_similarity,
    circular_convolution,
    sharpen_weights,
)


class NTMMemory(BaseMemory):
    """
    Standard NTM Memory Matrix.
    """

    def initialize_state(self, batch_size: int) -> MemoryState:
        # Initialize memory with small random values or learned initialization
        # Here we use constant initialization for simplicity/stability
        memory = ops.ones((batch_size, self.memory_size, self.memory_dim)) * 1e-6

        # Usage not used in standard NTM, but part of interface
        usage = ops.zeros((batch_size, self.memory_size))

        return MemoryState(memory=memory, usage=usage)

    def read(self, memory_state: MemoryState, read_weights: Any) -> Any:
        # read_weights: (batch, num_slots)
        # memory: (batch, num_slots, memory_dim)
        # output: (batch, memory_dim)

        # Expand weights for broadcasting: (batch, num_slots, 1)
        weights_expanded = ops.expand_dims(read_weights, axis=-1)

        # Weighted sum over memory slots
        read_vector = ops.sum(memory_state.memory * weights_expanded, axis=1)
        return read_vector

    def write(
        self,
        memory_state: MemoryState,
        write_weights: Any,
        erase_vector: Any,
        add_vector: Any,
    ) -> MemoryState:
        # write_weights: (batch, num_slots)
        # erase_vector: (batch, memory_dim)
        # add_vector: (batch, memory_dim)

        prev_memory = memory_state.memory

        # Expand dims for broadcasting
        # weights: (batch, num_slots, 1)
        # vectors: (batch, 1, memory_dim)
        weights_expanded = ops.expand_dims(write_weights, axis=-1)
        erase_expanded = ops.expand_dims(erase_vector, axis=1)
        add_expanded = ops.expand_dims(add_vector, axis=1)

        # Erase: M_t = M_{t-1} * (1 - w_t * e_t)
        erase_matrix = 1.0 - (weights_expanded * erase_expanded)
        erased_memory = prev_memory * erase_matrix

        # Add: M_t = M_{t-1} + w_t * a_t
        add_matrix = weights_expanded * add_expanded
        new_memory = erased_memory + add_matrix

        return MemoryState(
            memory=new_memory,
            usage=memory_state.usage,
            temporal_links=memory_state.temporal_links,
            precedence=memory_state.precedence
        )


class NTMReadHead(BaseHead):
    """
    Standard NTM Read Head.
    """

    def build(self, input_shape):
        # input_shape comes from controller output
        # We need dense layers to project controller output to head parameters

        # Parameters needed:
        # k (key): memory_dim
        # beta (strength): 1
        # g (gate): 1
        # s (shift): shift_range
        # gamma (sharpening): 1

        self.key_dense = keras.layers.Dense(self.memory_dim, name="key")
        self.beta_dense = keras.layers.Dense(1, activation="softplus", name="beta")
        self.gate_dense = keras.layers.Dense(1, activation="sigmoid", name="gate")
        self.shift_dense = keras.layers.Dense(self.shift_range, activation="softmax", name="shift")
        self.gamma_dense = keras.layers.Dense(1, activation="softplus", name="gamma")

        super().build(input_shape)

    def content_addressing(self, key: Any, beta: Any, memory: Any) -> Any:
        return ops.softmax(beta * cosine_similarity(key, memory))

    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> Tuple[Any, HeadState]:

        # 1. Project controller output to head parameters
        key = self.key_dense(controller_output)          # (batch, mem_dim)
        beta = self.beta_dense(controller_output)        # (batch, 1)
        gate = self.gate_dense(controller_output)        # (batch, 1)
        shift = self.shift_dense(controller_output)      # (batch, shift_range)
        gamma = self.gamma_dense(controller_output) + 1.0 # (batch, 1), gamma >= 1

        # 2. Content Addressing
        # w_c = softmax(beta * cosine_sim(k, M))
        # memory_state.memory: (batch, slots, mem_dim)
        # key: (batch, mem_dim) -> expand to (batch, 1, mem_dim)
        key_expanded = ops.expand_dims(key, axis=1)
        content_weights = self.content_addressing(key_expanded, beta, memory_state.memory)

        # 3. Interpolation (Gating)
        # w_g = g * w_c + (1 - g) * w_{t-1}
        gated_weights = gate * content_weights + (1.0 - gate) * prev_weights

        # 4. Convolutional Shift
        # w_tilde = conv(w_g, s)
        shifted_weights = circular_convolution(gated_weights, shift)

        # 5. Sharpening
        # w = w_tilde ^ gamma / sum(...)
        final_weights = sharpen_weights(shifted_weights, gamma)

        new_state = HeadState(
            weights=final_weights,
            key=key,
            beta=beta,
            gate=gate,
            shift=shift,
            gamma=gamma
        )

        return final_weights, new_state


class NTMWriteHead(BaseHead):
    """
    Standard NTM Write Head.
    """

    def build(self, input_shape):
        # Same addressing params as ReadHead, plus erase and add vectors
        self.key_dense = keras.layers.Dense(self.memory_dim, name="key")
        self.beta_dense = keras.layers.Dense(1, activation="softplus", name="beta")
        self.gate_dense = keras.layers.Dense(1, activation="sigmoid", name="gate")
        self.shift_dense = keras.layers.Dense(self.shift_range, activation="softmax", name="shift")
        self.gamma_dense = keras.layers.Dense(1, activation="softplus", name="gamma")

        self.erase_dense = keras.layers.Dense(self.memory_dim, activation="sigmoid", name="erase")
        self.add_dense = keras.layers.Dense(self.memory_dim, activation="tanh", name="add") # or sigmoid/linear

        super().build(input_shape)

    def content_addressing(self, key: Any, beta: Any, memory: Any) -> Any:
        return ops.softmax(beta * cosine_similarity(key, memory))

    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> Tuple[Any, HeadState]:

        # 1. Project parameters
        key = self.key_dense(controller_output)
        beta = self.beta_dense(controller_output)
        gate = self.gate_dense(controller_output)
        shift = self.shift_dense(controller_output)
        gamma = self.gamma_dense(controller_output) + 1.0

        erase = self.erase_dense(controller_output)
        add = self.add_dense(controller_output)

        # 2. Content
        key_expanded = ops.expand_dims(key, axis=1)
        content_weights = self.content_addressing(key_expanded, beta, memory_state.memory)

        # 3. Interpolation
        gated_weights = gate * content_weights + (1.0 - gate) * prev_weights

        # 4. Shift
        shifted_weights = circular_convolution(gated_weights, shift)

        # 5. Sharpen
        final_weights = sharpen_weights(shifted_weights, gamma)

        new_state = HeadState(
            weights=final_weights,
            key=key,
            beta=beta,
            gate=gate,
            shift=shift,
            gamma=gamma,
            erase_vector=erase,
            add_vector=add
        )

        return final_weights, new_state


class NTMController(BaseController):
    """
    Standard NTM Controller (LSTM/FeedForward).
    """

    def __init__(self, controller_dim, controller_type='lstm', **kwargs):
        super().__init__(controller_dim, controller_type, **kwargs)
        self.cell = None

    def build(self, input_shape):
        """
        Explicitly build the internal cell based on input shape.
        input_shape: (batch_size, input_dim + total_read_dim)
        """
        # Ensure input_shape is a tuple of integers/None
        if isinstance(input_shape, (list, tuple)):
             # Handle case where input_shape might be a list of shapes if called incorrectly
             if len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                 input_shape = input_shape[0]

        if self.controller_type == 'lstm':
            self.cell = keras.layers.LSTMCell(self.controller_dim)
        elif self.controller_type == 'gru':
            self.cell = keras.layers.GRUCell(self.controller_dim)
        else:
            self.cell = keras.layers.Dense(self.controller_dim, activation='relu')

        # Manually trigger build on the cell
        if hasattr(self.cell, 'build'):
             self.cell.build(input_shape)

        self.built = True

    def initialize_state(self, batch_size: int) -> Optional[Any]:
        if self.controller_type in ['lstm', 'gru']:
            # Get initial state from the cell
            # Keras cells usually take (batch_size, dtype) for get_initial_state
            # But we might need to construct it manually if using raw cells outside RNN
            return self.cell.get_initial_state(batch_size=batch_size, dtype="float32")
        return None

    def call(
        self,
        inputs: Any,
        state: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Optional[Any]]:

        if self.controller_type in ['lstm', 'gru']:
            output, new_state = self.cell(inputs, state, training=training)
            return output, new_state
        else:
            output = self.cell(inputs, training=training)
            return output, None


class NTMCell(keras.layers.Layer):
    """
    NTM Cell processing a single timestep.
    Combines Controller, Memory, and Heads.
    """

    def __init__(self, config: NTMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.memory = NTMMemory(config.memory_size, config.memory_dim)
        self.controller = NTMController(config.controller_dim, config.controller_type)

        self.read_heads = [
            NTMReadHead(
                config.memory_size,
                config.memory_dim,
                config.addressing_mode,
                config.shift_range
            ) for _ in range(config.num_read_heads)
        ]

        self.write_heads = [
            NTMWriteHead(
                config.memory_size,
                config.memory_dim,
                config.addressing_mode,
                config.shift_range
            ) for _ in range(config.num_write_heads)
        ]

    def build(self, input_shape):
        """
        Build controller and heads with correct shapes.
        input_shape: (batch, feature_dim)
        """
        # Calculate controller input dimension
        # Input to controller = External Input + All Read Vectors
        feature_dim = input_shape[-1]
        total_read_dim = self.config.num_read_heads * self.config.memory_dim
        controller_input_shape = (None, feature_dim + total_read_dim)

        # Build Controller
        self.controller.build(controller_input_shape)

        # Build Heads (input is controller output)
        controller_output_shape = (None, self.config.controller_dim)
        for head in self.read_heads:
            head.build(controller_output_shape)
        for head in self.write_heads:
            head.build(controller_output_shape)

        self.built = True

    @property
    def state_size(self):
        # Return state sizes for RNN wrapper
        # 1. Controller State (list for LSTM)
        # 2. Memory State (tensor)
        # 3. Read Weights (tensor per head)
        # 4. Write Weights (tensor per head)
        # 5. Read Vectors (tensor per head)

        # Note: Keras RNN expects structure matching state tuple
        # We flatten complex states into a list of tensors for compatibility

        sizes = []

        # Controller state
        if self.config.controller_type == 'lstm':
            sizes.extend([self.config.controller_dim, self.config.controller_dim]) # h, c
        elif self.config.controller_type == 'gru':
            sizes.append(self.config.controller_dim)

        # Memory
        sizes.append((self.config.memory_size, self.config.memory_dim))

        # Read Weights
        for _ in range(self.config.num_read_heads):
            sizes.append(self.config.memory_size)

        # Write Weights
        for _ in range(self.config.num_write_heads):
            sizes.append(self.config.memory_size)

        # Read Vectors (needed for next step's input)
        for _ in range(self.config.num_read_heads):
            sizes.append(self.config.memory_dim)

        return sizes

    @property
    def output_size(self):
        return self.config.controller_dim

    def call(self, inputs, states, training=None):
        # Unpack states
        # The order must match state_size property and get_initial_state

        # 1. Controller State
        if self.config.controller_type == 'lstm':
            controller_state = (states[0], states[1])
            idx = 2
        elif self.config.controller_type == 'gru':
            controller_state = states[0]
            idx = 1
        else:
            controller_state = None
            idx = 0

        # 2. Memory State
        memory_val = states[idx]
        idx += 1

        # 3. Read Weights
        prev_read_weights = []
        for _ in range(self.config.num_read_heads):
            prev_read_weights.append(states[idx])
            idx += 1

        # 4. Write Weights
        prev_write_weights = []
        for _ in range(self.config.num_write_heads):
            prev_write_weights.append(states[idx])
            idx += 1

        # 5. Read Vectors
        prev_read_vectors = []
        for _ in range(self.config.num_read_heads):
            prev_read_vectors.append(states[idx])
            idx += 1

        # Reconstruct high-level state objects
        memory_state = MemoryState(memory=memory_val)

        # --- Step Execution ---

        # 1. Prepare Controller Input
        # Concat external input + previous read vectors
        flat_read_vectors = ops.concatenate(prev_read_vectors, axis=-1)
        controller_input = ops.concatenate([inputs, flat_read_vectors], axis=-1)

        # 2. Run Controller
        controller_output, new_controller_state = self.controller(
            controller_input,
            state=controller_state,
            training=training
        )

        # 3. Write Heads
        current_memory_state = memory_state
        new_write_weights_list = []

        for i, head in enumerate(self.write_heads):
            # Compute addressing
            weights, head_state = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_write_weights[i]
            )
            new_write_weights_list.append(weights)

            # Perform Write
            current_memory_state = self.memory.write(
                current_memory_state,
                weights,
                head_state.erase_vector,
                head_state.add_vector
            )

        # 4. Read Heads
        new_read_weights_list = []
        new_read_vectors_list = []

        for i, head in enumerate(self.read_heads):
            # Compute addressing (using UPDATED memory)
            weights, head_state = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_read_weights[i]
            )
            new_read_weights_list.append(weights)

            # Perform Read
            read_vec = self.memory.read(current_memory_state, weights)
            new_read_vectors_list.append(read_vec)

        # 5. Pack Output State
        new_states = []

        # Controller
        if self.config.controller_type == 'lstm':
            new_states.extend([new_controller_state[0], new_controller_state[1]])
        elif self.config.controller_type == 'gru':
            new_states.append(new_controller_state)

        # Memory
        new_states.append(current_memory_state.memory)

        # Weights
        new_states.extend(new_read_weights_list)
        new_states.extend(new_write_weights_list)

        # Read Vectors
        new_states.extend(new_read_vectors_list)

        # Output of the cell is the controller output (will be projected by NTM layer)
        return controller_output, new_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Helper to create initial zero tensors

        states = []

        # Controller
        if self.config.controller_type == 'lstm':
            states.extend([
                ops.zeros((batch_size, self.config.controller_dim), dtype=dtype),
                ops.zeros((batch_size, self.config.controller_dim), dtype=dtype)
            ])
        elif self.config.controller_type == 'gru':
            states.append(ops.zeros((batch_size, self.config.controller_dim), dtype=dtype))

        # Memory
        # Call memory module to get initial state
        mem_state = self.memory.initialize_state(batch_size)
        states.append(mem_state.memory)

        # Read Weights (Init to uniform or one-hot)
        # Using one-hot at index 0 is common, or small random
        # Here: One-hot start
        one_hot_weight = ops.one_hot(
            ops.zeros((batch_size,), dtype="int32"),
            self.config.memory_size
        )
        for _ in range(self.config.num_read_heads):
            states.append(one_hot_weight)

        # Write Weights
        for _ in range(self.config.num_write_heads):
            states.append(one_hot_weight)

        # Read Vectors (Init to zeros)
        for _ in range(self.config.num_read_heads):
            states.append(ops.zeros((batch_size, self.config.memory_dim), dtype=dtype))

        return states


class NeuralTuringMachine(BaseNTM):
    """
    Complete Neural Turing Machine Layer.
    Wraps NTMCell in an RNN layer.
    """

    def __init__(self, config: NTMConfig, output_dim: int, **kwargs):
        super().__init__(config, output_dim=output_dim, **kwargs)
        self.ntm_cell = NTMCell(config)
        self.rnn = keras.layers.RNN(
            self.ntm_cell,
            return_sequences=True,
            return_state=True,
            name="ntm_rnn"
        )
        self.output_projection = keras.layers.Dense(output_dim, name="output_projection")

    def build(self, input_shape):
        # input_shape: (batch, seq_len, input_dim)

        # Build RNN
        # Pass shape (batch, seq_len, input_dim) -> RNN builds cell with (batch, input_dim)
        self.rnn.build(input_shape)

        # Build projection
        # Input to projection is controller output dim
        self.output_projection.build((None, self.config.controller_dim))

        self.built = True

    def initialize_state(self, batch_size: int):
        # This implementation delegates state management to the RNN layer's logic
        # But required by abstract base class.
        # We can reconstruct high-level objects from the cell's initial state
        raw_states = self.ntm_cell.get_initial_state(batch_size=batch_size, dtype="float32")

        # Manual reconstruction logic would go here if needed for step()
        # For standard usage, the RNN layer handles this.
        return None, None, None

    def step(self, inputs, memory_state, head_states, controller_state, training=None):
        # Direct step execution, bypassing RNN layer logic
        # Useful for custom loops, but typically we use call()
        pass

    def call(self, inputs, initial_state=None, training=None, return_sequences=True, return_state=False):
        # inputs: (batch, seq_len, features)

        # 1. Run RNN
        # rnn_output: (batch, seq_len, controller_dim)
        # final_states: list of tensors
        rnn_result = self.rnn(inputs, initial_state=initial_state, training=training)

        if self.rnn.return_state:
            rnn_output = rnn_result[0]
            final_states = rnn_result[1:]
        else:
            rnn_output = rnn_result
            final_states = None

        # 2. Project Output
        output = self.output_projection(rnn_output)

        # Handle return sequences (Dense applies to all steps automatically)
        if not return_sequences:
            output = output[:, -1, :]

        if return_state:
            return output, final_states
        return output

    def get_memory_state(self):
        # Not easily accessible via standard Keras RNN without keeping history
        return None

    def reset_memory(self, batch_size):
        pass

def create_ntm(
    memory_size: int = 128,
    memory_dim: int = 64,
    output_dim: int = 10,
    controller_dim: int = 256,
    controller_type: str = 'lstm',
    num_heads: int = 1
) -> keras.layers.Layer:
    """
    Factory function to create a functional Keras model with NTM.
    """
    config = NTMConfig(
        memory_size=memory_size,
        memory_dim=memory_dim,
        num_read_heads=num_heads,
        num_write_heads=num_heads,
        controller_dim=controller_dim,
        controller_type=controller_type
    )

    layer = NeuralTuringMachine(config, output_dim=output_dim)
    return layer