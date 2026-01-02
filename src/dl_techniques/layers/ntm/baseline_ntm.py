"""
Baseline Neural Turing Machine Implementation.

This module provides a complete implementation of the original Neural Turing Machine
as described in Graves et al., 2014 "Neural Turing Machines".

The implementation includes:
    - NTMMemory: External memory matrix with read/write operations.
    - NTMReadHead: Read head with content and location-based addressing.
    - NTMWriteHead: Write head with erase and add operations.
    - NTMController: LSTM or feedforward controller network.
    - NTMCell: Single timestep NTM operation (RNN cell interface).
    - NeuralTuringMachine: Complete NTM layer for sequence processing.

References:
    Graves, A., Wayne, G., & Danihelka, I. (2014).
    Neural Turing Machines. arXiv preprint arXiv:1410.5401.
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Any, Dict, Tuple, List, Literal

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


# ---------------------------------------------------------------------
# Memory Module
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMMemory(BaseMemory):
    """
    External memory module for the Neural Turing Machine.
    
    The memory is a 2D matrix of shape (num_slots, memory_dim) that supports
    differentiable read and write operations through attention mechanisms.
    
    Attributes:
        memory_size: Number of memory slots (N).
        memory_dim: Dimension of each memory slot (M).
        use_memory_init: Whether to learn initial memory state.
        memory_initializer: Initializer for learnable memory.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        use_memory_init: bool = True,
        memory_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the memory module.
        
        Args:
            memory_size: Number of memory slots.
            memory_dim: Dimension of each memory slot.
            use_memory_init: Whether to learn initial memory state.
            memory_initializer: Initializer for learnable memory.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            epsilon=epsilon,
            **kwargs,
        )
        self.use_memory_init = use_memory_init
        self.memory_initializer = initializers.get(memory_initializer)
        
        # Learnable initial memory (optional)
        self._initial_memory: Optional[Any] = None
    
    def build(self, input_shape: Any) -> None:
        """Build the memory module."""
        if self.use_memory_init:
            self._initial_memory = self.add_weight(
                name='initial_memory',
                shape=(1, self.memory_size, self.memory_dim),
                initializer=self.memory_initializer,
                trainable=True,
            )
        self.built = True
    
    def initialize_state(self, batch_size: int) -> MemoryState:
        """
        Initialize memory state for a new sequence.
        
        Args:
            batch_size: Number of sequences in the batch.
            
        Returns:
            MemoryState with initialized memory matrix.
        """
        if self.use_memory_init and self._initial_memory is not None:
            # Tile learnable initial memory across batch
            memory = ops.tile(self._initial_memory, (batch_size, 1, 1))
        else:
            # Initialize with small random values
            memory = ops.ones((batch_size, self.memory_size, self.memory_dim)) * 1e-6
        
        # Initialize usage to uniform
        usage = ops.ones((batch_size, self.memory_size)) / self.memory_size
        
        return MemoryState(
            memory=memory,
            usage=usage,
        )
    
    def read(
        self,
        memory_state: MemoryState,
        read_weights: Any,
    ) -> Any:
        """
        Read from memory using attention weights.
        
        The read operation computes a weighted sum of memory rows:
            r = sum_i(w_i * M[i])
        
        Args:
            memory_state: Current memory state.
            read_weights: Attention weights of shape (batch, num_slots).
            
        Returns:
            Read vector of shape (batch, memory_dim).
        """
        # read_weights: (batch, num_slots) -> (batch, num_slots, 1)
        weights_expanded = ops.expand_dims(read_weights, axis=-1)
        
        # Weighted sum: (batch, num_slots, memory_dim) * (batch, num_slots, 1)
        read_vector = ops.sum(memory_state.memory * weights_expanded, axis=1)
        
        return read_vector
    
    def write(
        self,
        memory_state: MemoryState,
        write_weights: Any,
        erase_vector: Any,
        add_vector: Any,
    ) -> MemoryState:
        """
        Write to memory using erase and add operations.
        
        The write operation consists of two steps:
            1. Erase: M_t[i] = M_{t-1}[i] * (1 - w_i * e)
            2. Add:   M_t[i] = M_t[i] + w_i * a
        
        Args:
            memory_state: Current memory state.
            write_weights: Write weights of shape (batch, num_slots).
            erase_vector: Erase vector of shape (batch, memory_dim).
            add_vector: Add vector of shape (batch, memory_dim).
            
        Returns:
            Updated memory state.
        """
        memory = memory_state.memory
        
        # Expand dimensions for broadcasting
        # write_weights: (batch, num_slots) -> (batch, num_slots, 1)
        weights_expanded = ops.expand_dims(write_weights, axis=-1)
        # erase_vector: (batch, memory_dim) -> (batch, 1, memory_dim)
        erase_expanded = ops.expand_dims(erase_vector, axis=1)
        # add_vector: (batch, memory_dim) -> (batch, 1, memory_dim)
        add_expanded = ops.expand_dims(add_vector, axis=1)
        
        # Erase operation: M = M * (1 - w * e)
        erase_term = weights_expanded * erase_expanded  # (batch, num_slots, memory_dim)
        memory = memory * (1.0 - erase_term)
        
        # Add operation: M = M + w * a
        add_term = weights_expanded * add_expanded  # (batch, num_slots, memory_dim)
        memory = memory + add_term
        
        # Update usage based on write weights
        usage = memory_state.usage
        if usage is not None:
            # Increase usage where we wrote
            usage = usage + write_weights * (1.0 - usage)
        
        return MemoryState(
            memory=memory,
            usage=usage,
            write_weights=write_weights,
            read_weights=memory_state.read_weights,
            temporal_links=memory_state.temporal_links,
            precedence=memory_state.precedence,
            metadata=memory_state.metadata,
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'use_memory_init': self.use_memory_init,
            'memory_initializer': initializers.serialize(self.memory_initializer),
        })
        return config


# ---------------------------------------------------------------------
# Read Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMReadHead(BaseHead):
    """
    Read head for the Neural Turing Machine.
    
    The read head computes attention weights over memory using a combination
    of content-based and location-based addressing:
    
    1. Content addressing: Similarity between key and memory rows.
    2. Interpolation: Blend with previous weights.
    3. Shift: Apply circular convolution for location-based access.
    4. Sharpening: Focus the attention distribution.
    
    Attributes:
        memory_size: Number of memory slots.
        memory_dim: Dimension of each memory slot.
        controller_dim: Dimension of controller output.
        shift_range: Range of allowed shifts.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        controller_dim: int,
        addressing_mode: AddressingMode = AddressingMode.HYBRID,
        shift_range: int = 3,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the read head.
        
        Args:
            memory_size: Number of memory slots.
            memory_dim: Dimension of each memory slot.
            controller_dim: Dimension of controller output.
            addressing_mode: Type of addressing mechanism.
            shift_range: Range of allowed shifts (must be odd).
            kernel_initializer: Initializer for dense layers.
            kernel_regularizer: Regularizer for dense layers.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional arguments.
        """
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=addressing_mode,
            shift_range=shift_range,
            epsilon=epsilon,
            **kwargs,
        )
        self.controller_dim = controller_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        # Ensure shift_range is odd
        if shift_range % 2 == 0:
            self.shift_range = shift_range + 1
    
    def build(self, input_shape: Any) -> None:
        """Build the read head layers."""
        # Key projection: controller_dim -> memory_dim
        self.key_layer = layers.Dense(
            self.memory_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='key_projection',
        )
        
        # Beta (key strength): controller_dim -> 1
        self.beta_layer = layers.Dense(
            1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activation='softplus',  # Ensure positive
            name='beta_projection',
        )
        
        # Gate (interpolation): controller_dim -> 1
        self.gate_layer = layers.Dense(
            1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activation='sigmoid',  # Between 0 and 1
            name='gate_projection',
        )
        
        # Shift distribution: controller_dim -> shift_range
        self.shift_layer = layers.Dense(
            self.shift_range,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activation='softmax',  # Valid distribution
            name='shift_projection',
        )
        
        # Gamma (sharpening): controller_dim -> 1
        self.gamma_layer = layers.Dense(
            1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activation='softplus',  # Ensure positive, add 1 later
            name='gamma_projection',
        )
        
        self.built = True
    
    def content_addressing(
        self,
        key: Any,
        beta: Any,
        memory: Any,
    ) -> Any:
        """
        Compute content-based attention weights.
        
        Content addressing uses cosine similarity between the key and
        each memory row, scaled by beta and normalized via softmax:
            w_c[i] = softmax(beta * cos(key, M[i]))
        
        Args:
            key: Key vector of shape (batch, memory_dim).
            beta: Key strength of shape (batch, 1).
            memory: Memory matrix of shape (batch, num_slots, memory_dim).
            
        Returns:
            Content weights of shape (batch, num_slots).
        """
        # Expand key: (batch, memory_dim) -> (batch, 1, memory_dim)
        key_expanded = ops.expand_dims(key, axis=1)
        
        # Compute cosine similarity with each memory row
        # memory: (batch, num_slots, memory_dim)
        similarity = cosine_similarity(key_expanded, memory, self.epsilon)
        # similarity: (batch, num_slots)
        
        # Scale by beta and apply softmax
        scaled = similarity * beta
        content_weights = ops.softmax(scaled, axis=-1)
        
        return content_weights
    
    def location_addressing(
        self,
        content_weights: Any,
        prev_weights: Any,
        gate: Any,
        shift: Any,
        gamma: Any,
    ) -> Any:
        """
        Compute location-based addressing.
        
        Location addressing applies:
            1. Interpolation: w_g = g * w_c + (1 - g) * w_{t-1}
            2. Shift: w_s = circular_conv(w_g, s)
            3. Sharpening: w = normalize(w_s^gamma)
        
        Args:
            content_weights: Content-based weights of shape (batch, num_slots).
            prev_weights: Previous attention weights of shape (batch, num_slots).
            gate: Interpolation gate of shape (batch, 1).
            shift: Shift distribution of shape (batch, shift_range).
            gamma: Sharpening factor of shape (batch, 1).
            
        Returns:
            Final attention weights of shape (batch, num_slots).
        """
        # Interpolation
        interpolated = gate * content_weights + (1.0 - gate) * prev_weights
        
        # Circular convolution for shift
        shifted = circular_convolution(interpolated, shift)
        
        # Sharpening (gamma >= 1)
        gamma_adjusted = gamma + 1.0  # Ensure >= 1
        final_weights = sharpen_weights(shifted, gamma_adjusted, self.epsilon)
        
        return final_weights
    
    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> Tuple[Any, HeadState]:
        """
        Compute attention weights using the full addressing mechanism.
        
        Args:
            controller_output: Output from controller of shape (batch, controller_dim).
            memory_state: Current memory state.
            prev_weights: Previous attention weights of shape (batch, num_slots).
            
        Returns:
            Tuple of:
                - New attention weights of shape (batch, num_slots).
                - Updated head state.
        """
        # Compute addressing parameters
        key = self.key_layer(controller_output)
        beta = self.beta_layer(controller_output)
        gate = self.gate_layer(controller_output)
        shift = self.shift_layer(controller_output)
        gamma = self.gamma_layer(controller_output)
        
        # Content-based addressing
        content_weights = self.content_addressing(
            key, beta, memory_state.memory
        )
        
        # Location-based addressing (if hybrid mode)
        if self.addressing_mode in [AddressingMode.HYBRID, AddressingMode.LOCATION]:
            weights = self.location_addressing(
                content_weights, prev_weights, gate, shift, gamma
            )
        else:
            weights = content_weights
        
        # Create head state
        head_state = HeadState(
            weights=weights,
            key=key,
            beta=beta,
            gate=gate,
            shift=shift,
            gamma=gamma,
        )
        
        return weights, head_state
    
    def call(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
        training: Optional[bool] = None,
    ) -> Tuple[Any, HeadState]:
        """
        Forward pass of the read head.
        
        Args:
            controller_output: Output from controller.
            memory_state: Current memory state.
            prev_weights: Previous attention weights.
            training: Whether in training mode.
            
        Returns:
            Tuple of attention weights and head state.
        """
        return self.compute_addressing(controller_output, memory_state, prev_weights)
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'controller_dim': self.controller_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Write Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMWriteHead(NTMReadHead):
    """
    Write head for the Neural Turing Machine.
    
    The write head extends the read head with additional projections
    for erase and add vectors.
    
    Write operation:
        1. Erase: M_t[i] = M_{t-1}[i] * (1 - w_i * e)
        2. Add:   M_t[i] = M_t[i] + w_i * a
    """
    
    def build(self, input_shape: Any) -> None:
        """Build the write head layers."""
        # Build parent layers (addressing)
        super().build(input_shape)
        
        # Erase vector: controller_dim -> memory_dim
        self.erase_layer = layers.Dense(
            self.memory_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activation='sigmoid',  # Between 0 and 1
            name='erase_projection',
        )
        
        # Add vector: controller_dim -> memory_dim
        self.add_layer = layers.Dense(
            self.memory_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='add_projection',
        )
    
    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> Tuple[Any, HeadState]:
        """
        Compute attention weights and write vectors.
        
        Args:
            controller_output: Output from controller.
            memory_state: Current memory state.
            prev_weights: Previous attention weights.
            
        Returns:
            Tuple of attention weights and head state with write vectors.
        """
        # Get base addressing
        weights, head_state = super().compute_addressing(
            controller_output, memory_state, prev_weights
        )
        
        # Compute erase and add vectors
        erase_vector = self.erase_layer(controller_output)
        add_vector = self.add_layer(controller_output)
        
        # Update head state with write vectors
        head_state.erase_vector = erase_vector
        head_state.add_vector = add_vector
        
        return weights, head_state


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMController(BaseController):
    """
    Controller network for the Neural Turing Machine.
    
    The controller processes the concatenation of input and read vectors,
    producing an output that parameterizes the memory operations.
    
    Supports LSTM, GRU, or feedforward architectures.
    
    Attributes:
        controller_dim: Dimension of controller hidden state.
        controller_type: Type of controller ('lstm', 'gru', 'feedforward').
        num_layers: Number of controller layers.
        dropout_rate: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        controller_dim: int,
        controller_type: Literal['lstm', 'gru', 'feedforward'] = 'lstm',
        num_layers: int = 1,
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the controller.
        
        Args:
            controller_dim: Dimension of controller hidden state.
            controller_type: Type of controller architecture.
            num_layers: Number of controller layers.
            dropout_rate: Dropout rate for regularization.
            kernel_initializer: Initializer for dense/input kernels.
            recurrent_initializer: Initializer for recurrent kernels.
            kernel_regularizer: Regularizer for kernels.
            **kwargs: Additional arguments.
        """
        super().__init__(
            controller_dim=controller_dim,
            controller_type=controller_type,
            **kwargs,
        )
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        self._layers: List[layers.Layer] = []
    
    def build(self, input_shape: Any) -> None:
        """Build the controller network."""
        if self.controller_type == 'lstm':
            for i in range(self.num_layers):
                self._layers.append(
                    layers.LSTMCell(
                        self.controller_dim,
                        kernel_initializer=self.kernel_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate,
                        name=f'lstm_cell_{i}',
                    )
                )
        elif self.controller_type == 'gru':
            for i in range(self.num_layers):
                self._layers.append(
                    layers.GRUCell(
                        self.controller_dim,
                        kernel_initializer=self.kernel_initializer,
                        recurrent_initializer=self.recurrent_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate,
                        name=f'gru_cell_{i}',
                    )
                )
        else:  # feedforward
            for i in range(self.num_layers):
                self._layers.append(
                    layers.Dense(
                        self.controller_dim,
                        activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name=f'dense_{i}',
                    )
                )
                if self.dropout_rate > 0:
                    self._layers.append(
                        layers.Dropout(self.dropout_rate, name=f'dropout_{i}')
                    )
        
        self.built = True
    
    def initialize_state(self, batch_size: int) -> Optional[List[Tuple[Any, Any]]]:
        """
        Initialize controller state for a new sequence.
        
        Args:
            batch_size: Number of sequences in the batch.
            
        Returns:
            List of initial states for each layer, or None for feedforward.
        """
        if self.controller_type == 'feedforward':
            return None
        
        states = []
        for i in range(self.num_layers):
            if self.controller_type == 'lstm':
                h = ops.zeros((batch_size, self.controller_dim))
                c = ops.zeros((batch_size, self.controller_dim))
                states.append((h, c))
            else:  # GRU
                h = ops.zeros((batch_size, self.controller_dim))
                states.append((h,))
        
        return states
    
    def call(
        self,
        inputs: Any,
        state: Optional[List[Tuple[Any, ...]]] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Optional[List[Tuple[Any, ...]]]]:
        """
        Process inputs through the controller.
        
        Args:
            inputs: Input tensor of shape (batch, input_dim).
            state: Previous controller states.
            training: Whether in training mode.
            
        Returns:
            Tuple of controller output and updated states.
        """
        x = inputs
        
        if self.controller_type == 'feedforward':
            for layer in self._layers:
                if isinstance(layer, layers.Dropout):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            return x, None
        
        new_states = []
        layer_idx = 0
        
        for i, layer in enumerate(self._layers):
            if isinstance(layer, (layers.LSTMCell, layers.GRUCell)):
                x, new_state = layer(x, state[layer_idx], training=training)
                new_states.append(new_state)
                layer_idx += 1
        
        return x, new_states
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# NTM Cell (Single Timestep)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NTMCell(keras.layers.Layer):
    """
    NTM Cell for single timestep operation.
    
    This implements the core NTM computation for one timestep,
    suitable for use with keras.layers.RNN.
    
    Attributes:
        config: NTM configuration.
        output_dim: Dimension of the output.
    """
    
    def __init__(
        self,
        config: NTMConfig,
        output_dim: int,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the NTM cell.
        
        Args:
            config: NTM configuration object.
            output_dim: Dimension of the output.
            kernel_initializer: Initializer for dense layers.
            kernel_regularizer: Regularizer for dense layers.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.config = config
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        # State size for RNN interface
        # Memory: num_slots * memory_dim
        # Read weights per head: num_slots
        # Write weights per head: num_slots
        # Read vectors: num_read_heads * memory_dim
        # Controller state: depends on type
        memory_state_size = config.memory_size * config.memory_dim
        read_weights_size = config.num_read_heads * config.memory_size
        write_weights_size = config.num_write_heads * config.memory_size
        read_vectors_size = config.num_read_heads * config.memory_dim
        
        if config.controller_type == 'lstm':
            controller_state_size = 2 * config.controller_dim  # h and c
        elif config.controller_type == 'gru':
            controller_state_size = config.controller_dim
        else:
            controller_state_size = 0
        
        self._state_size = (
            memory_state_size +
            read_weights_size +
            write_weights_size +
            read_vectors_size +
            controller_state_size
        )
        
        self._output_size = output_dim
    
    @property
    def state_size(self) -> int:
        """Return the state size for RNN compatibility."""
        return self._state_size
    
    @property
    def output_size(self) -> int:
        """Return the output size."""
        return self._output_size
    
    def build(self, input_shape: Any) -> None:
        """Build the NTM cell components."""
        input_dim = input_shape[-1]
        
        # Memory module
        self.memory = NTMMemory(
            memory_size=self.config.memory_size,
            memory_dim=self.config.memory_dim,
            use_memory_init=self.config.use_memory_init,
            epsilon=self.config.epsilon,
            name='memory',
        )
        
        # Controller
        self.controller = NTMController(
            controller_dim=self.config.controller_dim,
            controller_type=self.config.controller_type,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='controller',
        )
        
        # Read heads
        self.read_heads = []
        for i in range(self.config.num_read_heads):
            head = NTMReadHead(
                memory_size=self.config.memory_size,
                memory_dim=self.config.memory_dim,
                controller_dim=self.config.controller_dim,
                addressing_mode=self.config.addressing_mode,
                shift_range=self.config.shift_range,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                epsilon=self.config.epsilon,
                name=f'read_head_{i}',
            )
            self.read_heads.append(head)
        
        # Write heads
        self.write_heads = []
        for i in range(self.config.num_write_heads):
            head = NTMWriteHead(
                memory_size=self.config.memory_size,
                memory_dim=self.config.memory_dim,
                controller_dim=self.config.controller_dim,
                addressing_mode=self.config.addressing_mode,
                shift_range=self.config.shift_range,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                epsilon=self.config.epsilon,
                name=f'write_head_{i}',
            )
            self.write_heads.append(head)
        
        # Output projection
        self.output_layer = layers.Dense(
            self.output_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='output_projection',
        )
        
        # Build sub-layers
        controller_input_dim = input_dim + self.config.num_read_heads * self.config.memory_dim
        self.memory.build((None, self.config.memory_size, self.config.memory_dim))
        self.controller.build((None, controller_input_dim))
        
        for head in self.read_heads:
            head.build((None, self.config.controller_dim))
        for head in self.write_heads:
            head.build((None, self.config.controller_dim))
        
        self.built = True
    
    def get_initial_state(self, batch_size: int) -> Any:
        """
        Get initial state for the NTM cell.
        
        Args:
            batch_size: Batch size.
            
        Returns:
            Flattened initial state tensor.
        """
        # Initialize memory
        memory_state = self.memory.initialize_state(batch_size)
        memory_flat = ops.reshape(
            memory_state.memory,
            (batch_size, -1)
        )
        
        # Initialize read weights (uniform)
        read_weights = ops.ones(
            (batch_size, self.config.num_read_heads, self.config.memory_size)
        ) / self.config.memory_size
        read_weights_flat = ops.reshape(read_weights, (batch_size, -1))
        
        # Initialize write weights (uniform)
        write_weights = ops.ones(
            (batch_size, self.config.num_write_heads, self.config.memory_size)
        ) / self.config.memory_size
        write_weights_flat = ops.reshape(write_weights, (batch_size, -1))
        
        # Initialize read vectors (zeros)
        read_vectors = ops.zeros(
            (batch_size, self.config.num_read_heads, self.config.memory_dim)
        )
        read_vectors_flat = ops.reshape(read_vectors, (batch_size, -1))
        
        # Initialize controller state
        controller_state = self.controller.initialize_state(batch_size)
        if controller_state is not None:
            if self.config.controller_type == 'lstm':
                h, c = controller_state[0]
                controller_flat = ops.concatenate([h, c], axis=-1)
            else:  # GRU
                controller_flat = controller_state[0][0]
        else:
            controller_flat = ops.zeros((batch_size, 0))
        
        # Concatenate all states
        state = ops.concatenate([
            memory_flat,
            read_weights_flat,
            write_weights_flat,
            read_vectors_flat,
            controller_flat,
        ], axis=-1)
        
        return state
    
    def _unpack_state(
        self,
        state: Any,
        batch_size: int,
    ) -> Tuple[MemoryState, List[Any], List[Any], Any, Optional[Any]]:
        """Unpack flattened state into components."""
        idx = 0
        
        # Memory
        memory_size = self.config.memory_size * self.config.memory_dim
        memory_flat = state[:, idx:idx + memory_size]
        memory = ops.reshape(
            memory_flat,
            (batch_size, self.config.memory_size, self.config.memory_dim)
        )
        idx += memory_size
        
        # Read weights
        read_weights_size = self.config.num_read_heads * self.config.memory_size
        read_weights_flat = state[:, idx:idx + read_weights_size]
        read_weights = ops.reshape(
            read_weights_flat,
            (batch_size, self.config.num_read_heads, self.config.memory_size)
        )
        idx += read_weights_size
        
        # Write weights
        write_weights_size = self.config.num_write_heads * self.config.memory_size
        write_weights_flat = state[:, idx:idx + write_weights_size]
        write_weights = ops.reshape(
            write_weights_flat,
            (batch_size, self.config.num_write_heads, self.config.memory_size)
        )
        idx += write_weights_size
        
        # Read vectors
        read_vectors_size = self.config.num_read_heads * self.config.memory_dim
        read_vectors_flat = state[:, idx:idx + read_vectors_size]
        read_vectors = ops.reshape(
            read_vectors_flat,
            (batch_size, self.config.num_read_heads, self.config.memory_dim)
        )
        idx += read_vectors_size
        
        # Controller state
        if self.config.controller_type == 'lstm':
            h = state[:, idx:idx + self.config.controller_dim]
            c = state[:, idx + self.config.controller_dim:idx + 2 * self.config.controller_dim]
            controller_state = [(h, c)]
        elif self.config.controller_type == 'gru':
            h = state[:, idx:idx + self.config.controller_dim]
            controller_state = [(h,)]
        else:
            controller_state = None
        
        memory_state = MemoryState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
        )
        
        # Convert to lists
        prev_read_weights = [read_weights[:, i, :] for i in range(self.config.num_read_heads)]
        prev_write_weights = [write_weights[:, i, :] for i in range(self.config.num_write_heads)]
        
        return memory_state, prev_read_weights, prev_write_weights, read_vectors, controller_state
    
    def _pack_state(
        self,
        memory_state: MemoryState,
        read_weights: List[Any],
        write_weights: List[Any],
        read_vectors: Any,
        controller_state: Optional[Any],
        batch_size: int,
    ) -> Any:
        """Pack state components into flattened tensor."""
        # Memory
        memory_flat = ops.reshape(memory_state.memory, (batch_size, -1))
        
        # Read weights
        read_weights_stacked = ops.stack(read_weights, axis=1)
        read_weights_flat = ops.reshape(read_weights_stacked, (batch_size, -1))
        
        # Write weights
        write_weights_stacked = ops.stack(write_weights, axis=1)
        write_weights_flat = ops.reshape(write_weights_stacked, (batch_size, -1))
        
        # Read vectors
        read_vectors_flat = ops.reshape(read_vectors, (batch_size, -1))
        
        # Controller state
        if controller_state is not None:
            if self.config.controller_type == 'lstm':
                h, c = controller_state[0]
                controller_flat = ops.concatenate([h, c], axis=-1)
            else:  # GRU
                controller_flat = controller_state[0][0]
        else:
            controller_flat = ops.zeros((batch_size, 0))
        
        return ops.concatenate([
            memory_flat,
            read_weights_flat,
            write_weights_flat,
            read_vectors_flat,
            controller_flat,
        ], axis=-1)
    
    def call(
        self,
        inputs: Any,
        states: List[Any],
        training: Optional[bool] = None,
    ) -> Tuple[Any, List[Any]]:
        """
        Process one timestep.
        
        Args:
            inputs: Input tensor of shape (batch, input_dim).
            states: List containing the flattened state tensor.
            training: Whether in training mode.
            
        Returns:
            Tuple of (output, [new_state]).
        """
        state = states[0]
        batch_size = ops.shape(inputs)[0]
        
        # Unpack state
        (
            memory_state,
            prev_read_weights,
            prev_write_weights,
            prev_read_vectors,
            controller_state,
        ) = self._unpack_state(state, batch_size)
        
        # Flatten previous read vectors for controller input
        prev_read_vectors_flat = ops.reshape(prev_read_vectors, (batch_size, -1))
        
        # Concatenate input with read vectors
        controller_input = ops.concatenate([inputs, prev_read_vectors_flat], axis=-1)
        
        # Run controller
        controller_output, new_controller_state = self.controller(
            controller_input, controller_state, training=training
        )
        
        # Write to memory (before reading for this timestep)
        new_write_weights = []
        for i, head in enumerate(self.write_heads):
            weights, head_state = head(
                controller_output,
                memory_state,
                prev_write_weights[i],
                training=training,
            )
            new_write_weights.append(weights)
            
            # Apply write operation
            memory_state = self.memory.write(
                memory_state,
                weights,
                head_state.erase_vector,
                head_state.add_vector,
            )
        
        # Read from memory
        new_read_weights = []
        read_vectors_list = []
        for i, head in enumerate(self.read_heads):
            weights, head_state = head(
                controller_output,
                memory_state,
                prev_read_weights[i],
                training=training,
            )
            new_read_weights.append(weights)
            
            # Read from memory
            read_vector = self.memory.read(memory_state, weights)
            read_vectors_list.append(read_vector)
        
        # Stack read vectors
        new_read_vectors = ops.stack(read_vectors_list, axis=1)
        
        # Compute output
        read_vectors_flat = ops.reshape(new_read_vectors, (batch_size, -1))
        output_input = ops.concatenate([controller_output, read_vectors_flat], axis=-1)
        output = self.output_layer(output_input)
        
        # Pack new state
        new_state = self._pack_state(
            memory_state,
            new_read_weights,
            new_write_weights,
            new_read_vectors,
            new_controller_state,
            batch_size,
        )
        
        return output, [new_state]
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'config': {
                'memory_size': self.config.memory_size,
                'memory_dim': self.config.memory_dim,
                'num_read_heads': self.config.num_read_heads,
                'num_write_heads': self.config.num_write_heads,
                'controller_dim': self.config.controller_dim,
                'controller_type': self.config.controller_type,
                'addressing_mode': self.config.addressing_mode.name,
                'shift_range': self.config.shift_range,
                'use_memory_init': self.config.use_memory_init,
                'clip_value': self.config.clip_value,
                'epsilon': self.config.epsilon,
            },
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NTMCell':
        """Create from configuration."""
        ntm_config_dict = config.pop('config')
        ntm_config_dict['addressing_mode'] = AddressingMode[
            ntm_config_dict['addressing_mode']
        ]
        ntm_config = NTMConfig(**ntm_config_dict)
        return cls(config=ntm_config, **config)


# ---------------------------------------------------------------------
# Complete NTM Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NeuralTuringMachine(BaseNTM):
    """
    Complete Neural Turing Machine layer.
    
    This layer implements the full NTM architecture from Graves et al., 2014,
    suitable for sequence-to-sequence tasks.
    
    Example:
        >>> config = NTMConfig(
        ...     memory_size=128,
        ...     memory_dim=64,
        ...     num_read_heads=1,
        ...     num_write_heads=1,
        ...     controller_dim=256,
        ... )
        >>> ntm = NeuralTuringMachine(config, output_dim=10)
        >>> output = ntm(input_sequence)  # (batch, seq_len, 10)
    
    Attributes:
        config: NTM configuration.
        output_dim: Dimension of the output.
    """
    
    def __init__(
        self,
        config: NTMConfig,
        output_dim: int,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Neural Turing Machine.
        
        Args:
            config: NTM configuration object.
            output_dim: Dimension of the output.
            kernel_initializer: Initializer for dense layers.
            kernel_regularizer: Regularizer for dense layers.
            **kwargs: Additional arguments.
        """
        super().__init__(config=config, **kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        # Will be built in build()
        self._ntm_cell: Optional[NTMCell] = None
        self._rnn: Optional[layers.RNN] = None
        
        # Current state for stateful operation
        self._current_memory_state: Optional[MemoryState] = None
    
    def build(self, input_shape: Any) -> None:
        """Build the NTM layer."""
        # Create NTM cell
        self._ntm_cell = NTMCell(
            config=self.config,
            output_dim=self.output_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='ntm_cell',
        )
        
        # Wrap in RNN layer
        self._rnn = layers.RNN(
            self._ntm_cell,
            return_sequences=True,
            return_state=True,
            name='ntm_rnn',
        )
        
        # Build RNN
        self._rnn.build(input_shape)
        
        # Reference memory for the interface
        self.memory = self._ntm_cell.memory
        self.controller = self._ntm_cell.controller
        self.read_heads = self._ntm_cell.read_heads
        self.write_heads = self._ntm_cell.write_heads
        
        self.built = True
    
    def initialize_state(
        self,
        batch_size: int,
    ) -> Tuple[MemoryState, List[HeadState], Optional[Any]]:
        """
        Initialize all states for a new sequence.
        
        Args:
            batch_size: Number of sequences in the batch.
            
        Returns:
            Tuple of initial states.
        """
        # Get initial state from cell
        initial_state = self._ntm_cell.get_initial_state(batch_size)
        
        # Unpack into components
        (
            memory_state,
            prev_read_weights,
            prev_write_weights,
            read_vectors,
            controller_state,
        ) = self._ntm_cell._unpack_state(initial_state, batch_size)
        
        # Create head states
        head_states = []
        for weights in prev_read_weights:
            head_states.append(HeadState(weights=weights))
        for weights in prev_write_weights:
            head_states.append(HeadState(weights=weights))
        
        return memory_state, head_states, controller_state
    
    def step(
        self,
        inputs: Any,
        memory_state: MemoryState,
        head_states: List[HeadState],
        controller_state: Optional[Any],
        training: Optional[bool] = None,
    ) -> NTMOutput:
        """
        Perform a single timestep of the NTM.
        
        This method is primarily for manual stepping. For normal use,
        call the layer directly which processes the full sequence.
        
        Args:
            inputs: Input at current timestep.
            memory_state: Current memory state.
            head_states: Current head states.
            controller_state: Current controller state.
            training: Whether in training mode.
            
        Returns:
            NTMOutput with all outputs and updated states.
        """
        batch_size = ops.shape(inputs)[0]
        
        # Pack state
        read_weights = [hs.weights for hs in head_states[:self.config.num_read_heads]]
        write_weights = [hs.weights for hs in head_states[self.config.num_read_heads:]]
        
        # Get read vectors from head states or initialize
        read_vectors = ops.zeros(
            (batch_size, self.config.num_read_heads, self.config.memory_dim)
        )
        
        state = self._ntm_cell._pack_state(
            memory_state,
            read_weights,
            write_weights,
            read_vectors,
            [controller_state] if controller_state is not None else None,
            batch_size,
        )
        
        # Run cell
        output, [new_state] = self._ntm_cell(inputs, [state], training=training)
        
        # Unpack new state
        (
            new_memory_state,
            new_read_weights,
            new_write_weights,
            new_read_vectors,
            new_controller_state,
        ) = self._ntm_cell._unpack_state(new_state, batch_size)
        
        # Create head states
        new_head_states = []
        for weights in new_read_weights:
            new_head_states.append(HeadState(weights=weights))
        for weights in new_write_weights:
            new_head_states.append(HeadState(weights=weights))
        
        return NTMOutput(
            output=output,
            memory_state=new_memory_state,
            head_states=new_head_states,
            read_vectors=new_read_vectors,
            controller_state=new_controller_state[0] if new_controller_state else None,
        )
    
    def call(
        self,
        inputs: Any,
        initial_state: Optional[Any] = None,
        training: Optional[bool] = None,
        mask: Optional[Any] = None,
    ) -> Any:
        """
        Process a sequence through the NTM.
        
        Args:
            inputs: Input sequence of shape (batch, seq_len, input_dim).
            initial_state: Optional initial state tensor.
            training: Whether in training mode.
            mask: Optional mask for padded sequences.
            
        Returns:
            Output sequence of shape (batch, seq_len, output_dim).
        """
        batch_size = ops.shape(inputs)[0]
        
        # Get initial state if not provided
        if initial_state is None:
            initial_state = self._ntm_cell.get_initial_state(batch_size)
        
        # Run RNN
        outputs = self._rnn(
            inputs,
            initial_state=[initial_state],
            training=training,
            mask=mask,
        )
        
        # outputs is (output_sequence, final_state)
        output_sequence = outputs[0]
        final_state = outputs[1]
        
        # Store final state for potential stateful operation
        self._current_memory_state = self._ntm_cell._unpack_state(
            final_state, batch_size
        )[0]
        
        return output_sequence
    
    def get_memory_state(self) -> Optional[MemoryState]:
        """Get the current memory state."""
        return self._current_memory_state
    
    def reset_memory(self, batch_size: int) -> None:
        """Reset memory to initial state."""
        self._current_memory_state = self.memory.initialize_state(batch_size)
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NeuralTuringMachine':
        """Create layer from configuration."""
        ntm_config_dict = config.pop('config')
        ntm_config_dict['addressing_mode'] = AddressingMode[
            ntm_config_dict['addressing_mode']
        ]
        ntm_config = NTMConfig(**ntm_config_dict)
        return cls(config=ntm_config, **config)


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------

def create_ntm(
    memory_size: int = 128,
    memory_dim: int = 64,
    output_dim: int = 10,
    num_read_heads: int = 1,
    num_write_heads: int = 1,
    controller_dim: int = 256,
    controller_type: Literal['lstm', 'gru', 'feedforward'] = 'lstm',
    addressing_mode: str = 'hybrid',
    shift_range: int = 3,
    kernel_initializer: str = 'glorot_uniform',
    kernel_regularizer: Optional[str] = None,
    name: str = 'ntm',
    **kwargs: Any,
) -> NeuralTuringMachine:
    """
    Factory function to create a Neural Turing Machine.
    
    Args:
        memory_size: Number of memory slots.
        memory_dim: Dimension of each memory slot.
        output_dim: Dimension of the output.
        num_read_heads: Number of read heads.
        num_write_heads: Number of write heads.
        controller_dim: Dimension of controller hidden state.
        controller_type: Type of controller ('lstm', 'gru', 'feedforward').
        addressing_mode: Addressing mechanism ('content', 'location', 'hybrid').
        shift_range: Range of allowed shifts for location addressing.
        kernel_initializer: Initializer for dense layers.
        kernel_regularizer: Regularizer for dense layers.
        name: Layer name.
        **kwargs: Additional arguments.
        
    Returns:
        NeuralTuringMachine: Configured NTM layer.
    
    Example:
        >>> ntm = create_ntm(
        ...     memory_size=128,
        ...     memory_dim=64,
        ...     output_dim=10,
        ...     controller_type='lstm',
        ... )
        >>> output = ntm(input_sequence)
    """
    # Parse addressing mode
    mode_map = {
        'content': AddressingMode.CONTENT,
        'location': AddressingMode.LOCATION,
        'hybrid': AddressingMode.HYBRID,
        'sparse': AddressingMode.SPARSE,
        'temporal': AddressingMode.TEMPORAL,
        'learned': AddressingMode.LEARNED,
    }
    
    if isinstance(addressing_mode, str):
        addressing_mode = mode_map.get(addressing_mode.lower(), AddressingMode.HYBRID)
    
    # Create configuration
    config = NTMConfig(
        memory_size=memory_size,
        memory_dim=memory_dim,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
        controller_dim=controller_dim,
        controller_type=controller_type,
        addressing_mode=addressing_mode,
        shift_range=shift_range,
    )
    
    # Create and return NTM
    return NeuralTuringMachine(
        config=config,
        output_dim=output_dim,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.get(kernel_regularizer) if kernel_regularizer else None,
        name=name,
        **kwargs,
    )
