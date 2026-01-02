"""
Baseline Neural Turing Machine Implementation.

This module provides a reference implementation of the NTM described in
Graves et al., 2014, updated for Keras 3 compatibility with robust serialization
and graph-safe operations.
"""

import keras
from keras import ops
from keras import layers
from typing import Optional, Tuple, Any, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .ntm_interface import (
    BaseMemory,
    BaseHead,
    BaseController,
    BaseNTM,
    MemoryState,
    HeadState,
    NTMConfig,
    cosine_similarity,
    circular_convolution,
    sharpen_weights,
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMMemory(BaseMemory):
    """
    Standard NTM Memory Matrix utility.

    Manages read/write operations on the memory matrix.
    """

    def __init__(self, memory_size: int, memory_dim: int, **kwargs):
        super().__init__(memory_size=memory_size, memory_dim=memory_dim, **kwargs)

    def initialize_state(self, batch_size: int) -> MemoryState:
        # Initialize memory with small random values or learned initialization
        memory = ops.ones((batch_size, self.memory_size, self.memory_dim)) * 1e-6
        usage = ops.zeros((batch_size, self.memory_size))

        return MemoryState(memory=memory, usage=usage)

    def read(self, memory_state: MemoryState, read_weights: Any) -> Any:
        # read_weights: (batch, num_slots)
        # memory: (batch, num_slots, memory_dim)

        # Expand weights for broadcasting: (batch, num_slots, 1)
        weights_expanded = ops.expand_dims(read_weights, axis=-1)

        # Weighted sum over memory slots -> (batch, memory_dim)
        read_vector = ops.sum(memory_state.memory * weights_expanded, axis=1)
        return read_vector

    def write(
        self,
        memory_state: MemoryState,
        write_weights: Any,
        erase_vector: Any,
        add_vector: Any,
    ) -> MemoryState:
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

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # BaseMemory stores memory_size/dim, ensuring they are in config
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMReadHead(BaseHead):
    """
    Standard NTM Read Head.
    Projects controller output to addressing parameters and reads from memory.
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        addressing_mode: str = 'content_and_location',
        shift_range: int = 3,
        **kwargs
    ):
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=addressing_mode,
            shift_range=shift_range,
            **kwargs
        )

        # Create sub-layers in __init__ (Golden Rule)
        self.key_dense = layers.Dense(memory_dim, name="key")
        self.beta_dense = layers.Dense(1, activation="softplus", name="beta")
        self.gate_dense = layers.Dense(1, activation="sigmoid", name="gate")
        self.shift_dense = layers.Dense(shift_range, activation="softmax", name="shift")
        self.gamma_dense = layers.Dense(1, activation="softplus", name="gamma")

    def build(self, input_shape):
        """Build sub-layers explicitly."""
        self.key_dense.build(input_shape)
        self.beta_dense.build(input_shape)
        self.gate_dense.build(input_shape)
        self.shift_dense.build(input_shape)
        self.gamma_dense.build(input_shape)
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
        key_expanded = ops.expand_dims(key, axis=1)
        content_weights = self.content_addressing(key_expanded, beta, memory_state.memory)

        # 3. Interpolation (Gating)
        gated_weights = gate * content_weights + (1.0 - gate) * prev_weights

        # 4. Convolutional Shift
        shifted_weights = circular_convolution(gated_weights, shift)

        # 5. Sharpening
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

    def call(self, inputs, **kwargs):
        # BaseHead doesn't strictly define call since it's used imperatively in Cell,
        # but implementing it enables standard layer checks.
        # This is a placeholder as Heads are typically called via compute_addressing.
        return inputs

    def compute_output_shape(self, input_shape):
        # Input: Controller output (Batch, Ctrl_Dim)
        # Output: Weights (Batch, Memory_Size)
        return (input_shape[0], self.memory_size)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # BaseHead config should handle init params
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMWriteHead(BaseHead):
    """
    Standard NTM Write Head.
    Includes erase and add vectors for memory modification.
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        addressing_mode: str = 'content_and_location',
        shift_range: int = 3,
        **kwargs
    ):
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=addressing_mode,
            shift_range=shift_range,
            **kwargs
        )

        # Create sub-layers in __init__
        self.key_dense = layers.Dense(memory_dim, name="key")
        self.beta_dense = layers.Dense(1, activation="softplus", name="beta")
        self.gate_dense = layers.Dense(1, activation="sigmoid", name="gate")
        self.shift_dense = layers.Dense(shift_range, activation="softmax", name="shift")
        self.gamma_dense = layers.Dense(1, activation="softplus", name="gamma")

        self.erase_dense = layers.Dense(memory_dim, activation="sigmoid", name="erase")
        self.add_dense = layers.Dense(memory_dim, activation="tanh", name="add")

    def build(self, input_shape):
        self.key_dense.build(input_shape)
        self.beta_dense.build(input_shape)
        self.gate_dense.build(input_shape)
        self.shift_dense.build(input_shape)
        self.gamma_dense.build(input_shape)

        self.erase_dense.build(input_shape)
        self.add_dense.build(input_shape)

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

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.memory_size)

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMController(BaseController):
    """
    Standard NTM Controller (LSTM/FeedForward).
    """

    def __init__(self, controller_dim: int, controller_type: str = 'lstm', **kwargs):
        super().__init__(controller_dim=controller_dim, controller_type=controller_type, **kwargs)

        # Create cell in __init__ (Golden Rule)
        if self.controller_type == 'lstm':
            self.cell = layers.LSTMCell(self.controller_dim)
        elif self.controller_type == 'gru':
            self.cell = layers.GRUCell(self.controller_dim)
        else:
            self.cell = layers.Dense(self.controller_dim, activation='relu')

    def build(self, input_shape):
        """
        Explicitly build the internal cell based on input shape.
        """
        if isinstance(input_shape, (list, tuple)):
             if len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                 input_shape = input_shape[0]

        # Manually trigger build on the cell
        if hasattr(self.cell, 'build'):
             self.cell.build(input_shape)

        super().build(input_shape)

    def initialize_state(self, batch_size: int) -> Optional[Any]:
        if self.controller_type in ['lstm', 'gru']:
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

    def compute_output_shape(self, input_shape):
        # Input: (Batch, Input_Dim)
        # Output: (Batch, Controller_Dim)
        return (input_shape[0], self.controller_dim)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # BaseController adds controller_dim, controller_type
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMCell(layers.Layer):
    """
    NTM Cell processing a single timestep.
    Combines Controller, Memory, and Heads.
    """

    def __init__(self, config: Union[NTMConfig, Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)

        # Handle reconstruction from dict (deserialization) or NTMConfig object
        if isinstance(config, dict):
            # Check if this is a serialized NTMConfig
            self.config = NTMConfig(**config)
        else:
            self.config = config

        self.memory = NTMMemory(self.config.memory_size, self.config.memory_dim)
        self.controller = NTMController(self.config.controller_dim, self.config.controller_type)

        self.read_heads = [
            NTMReadHead(
                self.config.memory_size,
                self.config.memory_dim,
                self.config.addressing_mode,
                self.config.shift_range,
                name=f"read_head_{i}"
            ) for i in range(self.config.num_read_heads)
        ]

        self.write_heads = [
            NTMWriteHead(
                self.config.memory_size,
                self.config.memory_dim,
                self.config.addressing_mode,
                self.config.shift_range,
                name=f"write_head_{i}"
            ) for i in range(self.config.num_write_heads)
        ]

    def build(self, input_shape):
        """
        Build controller and heads with correct shapes.
        input_shape: (batch, feature_dim)
        """
        # Calculate controller input dimension
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

        super().build(input_shape)

    @property
    def state_size(self):
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

        # Read Vectors
        for _ in range(self.config.num_read_heads):
            sizes.append(self.config.memory_dim)

        return sizes

    @property
    def output_size(self):
        return self.config.controller_dim + (self.config.num_read_heads * self.config.memory_dim)

    def call(self, inputs, states, training=None):
        # 1. Unpack states
        if self.config.controller_type == 'lstm':
            controller_state = (states[0], states[1])
            idx = 2
        elif self.config.controller_type == 'gru':
            controller_state = states[0]
            idx = 1
        else:
            controller_state = None
            idx = 0

        memory_val = states[idx]
        idx += 1

        prev_read_weights = []
        for _ in range(self.config.num_read_heads):
            prev_read_weights.append(states[idx])
            idx += 1

        prev_write_weights = []
        for _ in range(self.config.num_write_heads):
            prev_write_weights.append(states[idx])
            idx += 1

        prev_read_vectors = []
        for _ in range(self.config.num_read_heads):
            prev_read_vectors.append(states[idx])
            idx += 1

        memory_state = MemoryState(memory=memory_val)

        # 2. Prepare Controller Input
        flat_read_vectors = ops.concatenate(prev_read_vectors, axis=-1)
        controller_input = ops.concatenate([inputs, flat_read_vectors], axis=-1)

        # 3. Run Controller
        controller_output, new_controller_state = self.controller(
            controller_input,
            state=controller_state,
            training=training
        )

        # 4. Write Heads
        current_memory_state = memory_state
        new_write_weights_list = []

        for i, head in enumerate(self.write_heads):
            weights, head_state = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_write_weights[i]
            )
            new_write_weights_list.append(weights)

            current_memory_state = self.memory.write(
                current_memory_state,
                weights,
                head_state.erase_vector,
                head_state.add_vector
            )

        # 5. Read Heads
        new_read_weights_list = []
        new_read_vectors_list = []

        for i, head in enumerate(self.read_heads):
            weights, head_state = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_read_weights[i]
            )
            new_read_weights_list.append(weights)

            read_vec = self.memory.read(current_memory_state, weights)
            new_read_vectors_list.append(read_vec)

        # 6. Pack Output State
        new_states = []

        if self.config.controller_type == 'lstm':
            new_states.extend([new_controller_state[0], new_controller_state[1]])
        elif self.config.controller_type == 'gru':
            new_states.append(new_controller_state)

        new_states.append(current_memory_state.memory)
        new_states.extend(new_read_weights_list)
        new_states.extend(new_write_weights_list)
        new_states.extend(new_read_vectors_list)

        flat_new_read_vectors = ops.concatenate(new_read_vectors_list, axis=-1)
        cell_output = ops.concatenate([controller_output, flat_new_read_vectors], axis=-1)

        return cell_output, new_states

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, input_dim)
        # output: (batch, output_size)
        return (input_shape[0], self.output_size)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
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
        mem_state = self.memory.initialize_state(batch_size)
        states.append(mem_state.memory)

        # Read Weights
        one_hot_weight = ops.one_hot(
            ops.zeros((batch_size,), dtype="int32"),
            self.config.memory_size
        )
        for _ in range(self.config.num_read_heads):
            states.append(one_hot_weight)

        # Write Weights
        for _ in range(self.config.num_write_heads):
            states.append(one_hot_weight)

        # Read Vectors
        for _ in range(self.config.num_read_heads):
            states.append(ops.zeros((batch_size, self.config.memory_dim), dtype=dtype))

        return states

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # Serialize the config dataclass to a dict
        config.update({'config': vars(self.config)})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NTMCell':
        # Reconstructs the NTMConfig object in __init__
        return cls(**config)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NeuralTuringMachine(BaseNTM):
    """
    Complete Neural Turing Machine Layer.
    Wraps NTMCell in an RNN layer.
    """

    def __init__(self, config: Union[NTMConfig, Dict[str, Any]], output_dim: int, **kwargs):
        super().__init__(config=config, output_dim=output_dim, **kwargs)

        # Handle reconstruction
        if isinstance(config, dict):
            self.config = NTMConfig(**config)
        else:
            self.config = config

        self.ntm_cell = NTMCell(self.config)
        self.rnn = layers.RNN(
            self.ntm_cell,
            return_sequences=True,
            return_state=True,
            name="ntm_rnn"
        )
        self.output_projection = layers.Dense(output_dim, name="output_projection")

    def build(self, input_shape):
        # input_shape: (batch, seq_len, input_dim)

        self.rnn.build(input_shape)

        # Input to projection is cell output size (controller + read vectors)
        out_dim = self.ntm_cell.output_size
        self.output_projection.build((None, out_dim))

        super().build(input_shape)

    def initialize_state(self, batch_size: int):
        return None, None, None

    def step(self, inputs, memory_state, head_states, controller_state, training=None):
        pass

    def call(self, inputs, initial_state=None, training=None, return_sequences=True, return_state=False):
        # 1. Run RNN
        rnn_result = self.rnn(inputs, initial_state=initial_state, training=training)

        if self.rnn.return_state:
            rnn_output = rnn_result[0]
            final_states = rnn_result[1:]
        else:
            rnn_output = rnn_result
            final_states = None

        # 2. Project Output
        output = self.output_projection(rnn_output)

        if not return_sequences:
            output = output[:, -1, :]

        if return_state:
            return output, final_states
        return output

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, seq_len, features)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        output_shape = (batch_size, seq_len, self.output_dim)

        # If return_sequences is False (handled via slicing in call,
        # but pure shape inference might need explicit handling if we exposed that arg)
        # Note: call argument return_sequences logic is runtime branching.
        # Standard layer practice assumes configuration state.
        # Since rnn.return_sequences is fixed at init in this wrapper:
        return output_shape

    def get_memory_state(self):
        return None

    def reset_memory(self, batch_size):
        pass

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # Serialize nested config
        config.update({
            'config': vars(self.config),
            'output_dim': self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NeuralTuringMachine':
        return cls(**config)

# ---------------------------------------------------------------------
# utility functions
# ---------------------------------------------------------------------

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