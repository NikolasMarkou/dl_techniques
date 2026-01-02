"""
Neural Turing Machine (NTM) Interface Module.

This module defines the abstract base classes for all NTM variants, establishing
a consistent interface for memory-augmented neural network implementations.

The interface is designed based on the original NTM paper (Graves et al., 2014)
and extended to accommodate modern variants like DNC, SANTM, and Titans.

Classes:
    MemoryState: Dataclass representing the state of external memory.
    HeadState: Dataclass representing the state of read/write heads.
    NTMOutput: Dataclass for NTM forward pass outputs.
    AddressingMode: Enum for addressing mechanism types.
    BaseMemory: Abstract base class for memory modules.
    BaseHead: Abstract base class for read/write heads.
    BaseController: Abstract base class for controller networks.
    BaseNTM: Abstract base class for complete NTM architectures.
"""

import keras
from keras import ops
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Optional, 
    Union, 
    Any, 
    Dict, 
    Tuple, 
    List, 
    Literal,
    TypeVar,
    Generic,
)


# ---------------------------------------------------------------------
# Type Variables
# ---------------------------------------------------------------------

T = TypeVar('T')
MemoryStateT = TypeVar('MemoryStateT', bound='MemoryState')
HeadStateT = TypeVar('HeadStateT', bound='HeadState')


# ---------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------

class AddressingMode(Enum):
    """
    Enumeration of addressing mechanism types.
    
    Attributes:
        CONTENT: Content-based addressing using similarity measures.
        LOCATION: Location-based addressing using shifts and interpolation.
        HYBRID: Combined content and location addressing (original NTM).
        SPARSE: Sparse attention-based addressing (SANTM).
        TEMPORAL: Temporal link-based addressing (DNC).
        LEARNED: Fully learned addressing (Titans).
    """
    CONTENT = auto()
    LOCATION = auto()
    HYBRID = auto()
    SPARSE = auto()
    TEMPORAL = auto()
    LEARNED = auto()


class MemoryAccessType(Enum):
    """
    Enumeration of memory access types.
    
    Attributes:
        READ: Read-only memory access.
        WRITE: Write-only memory access.
        READ_WRITE: Combined read and write access.
    """
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()


# ---------------------------------------------------------------------
# State Dataclasses
# ---------------------------------------------------------------------

@dataclass
class MemoryState:
    """
    Represents the state of external memory.
    
    This dataclass encapsulates all information needed to describe
    the current state of the memory matrix and associated metadata.
    
    Attributes:
        memory: The memory matrix of shape (batch, num_slots, memory_dim).
        usage: Memory usage vector of shape (batch, num_slots).
        write_weights: Most recent write weights of shape (batch, num_heads, num_slots).
        read_weights: Most recent read weights of shape (batch, num_heads, num_slots).
        temporal_links: Optional temporal link matrix for DNC-style memory.
        precedence: Optional precedence weights for temporal ordering.
        metadata: Optional dictionary for variant-specific state.
    """
    memory: Any  # (batch, num_slots, memory_dim)
    usage: Optional[Any] = None  # (batch, num_slots)
    write_weights: Optional[Any] = None  # (batch, num_write_heads, num_slots)
    read_weights: Optional[Any] = None  # (batch, num_read_heads, num_slots)
    temporal_links: Optional[Any] = None  # (batch, num_slots, num_slots)
    precedence: Optional[Any] = None  # (batch, num_slots)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def clone(self) -> 'MemoryState':
        """
        Create a shallow copy of the memory state.
        
        Returns:
            MemoryState: A new MemoryState with copied references.
        """
        return MemoryState(
            memory=self.memory,
            usage=self.usage,
            write_weights=self.write_weights,
            read_weights=self.read_weights,
            temporal_links=self.temporal_links,
            precedence=self.precedence,
            metadata=dict(self.metadata),
        )


@dataclass
class HeadState:
    """
    Represents the state of a read or write head.
    
    Attributes:
        weights: Attention weights over memory slots, shape (batch, num_slots).
        read_vector: Last read vector for read heads, shape (batch, memory_dim).
        key: Key vector used for content addressing.
        beta: Key strength (sharpening factor).
        gate: Interpolation gate between content and location addressing.
        shift: Shift distribution for location addressing.
        gamma: Sharpening factor for final weights.
        erase_vector: Erase vector for write heads.
        add_vector: Add vector for write heads.
        metadata: Optional dictionary for variant-specific state.
    """
    weights: Any  # (batch, num_slots)
    read_vector: Optional[Any] = None  # (batch, memory_dim)
    key: Optional[Any] = None  # (batch, memory_dim)
    beta: Optional[Any] = None  # (batch, 1)
    gate: Optional[Any] = None  # (batch, 1)
    shift: Optional[Any] = None  # (batch, shift_range)
    gamma: Optional[Any] = None  # (batch, 1)
    erase_vector: Optional[Any] = None  # (batch, memory_dim)
    add_vector: Optional[Any] = None  # (batch, memory_dim)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NTMOutput:
    """
    Output structure for NTM forward pass.
    
    Attributes:
        output: The network output, shape (batch, output_dim).
        memory_state: Updated memory state after the forward pass.
        head_states: List of updated head states.
        read_vectors: Read vectors from all read heads.
        controller_state: Controller hidden state (for recurrent controllers).
        attention_weights: Attention weights for visualization/analysis.
        auxiliary_losses: Optional auxiliary losses (e.g., regularization).
        metadata: Optional dictionary for additional outputs.
    """
    output: Any  # (batch, output_dim)
    memory_state: MemoryState
    head_states: List[HeadState]
    read_vectors: Any  # (batch, num_read_heads, memory_dim)
    controller_state: Optional[Any] = None
    attention_weights: Optional[Dict[str, Any]] = None
    auxiliary_losses: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NTMConfig:
    """
    Configuration for NTM architectures.
    
    Attributes:
        memory_size: Number of memory slots (N).
        memory_dim: Dimension of each memory slot (M).
        num_read_heads: Number of read heads.
        num_write_heads: Number of write heads.
        controller_dim: Dimension of controller hidden state.
        controller_type: Type of controller ('lstm', 'gru', 'feedforward').
        addressing_mode: Type of addressing mechanism.
        shift_range: Range of allowed shifts for location addressing.
        use_memory_init: Whether to learn initial memory state.
        clip_value: Gradient clipping value for stability.
        epsilon: Small constant for numerical stability.
    """
    memory_size: int = 128
    memory_dim: int = 64
    num_read_heads: int = 1
    num_write_heads: int = 1
    controller_dim: int = 256
    controller_type: Literal['lstm', 'gru', 'feedforward'] = 'lstm'
    addressing_mode: AddressingMode = AddressingMode.HYBRID
    shift_range: int = 3
    use_memory_init: bool = True
    clip_value: float = 10.0
    epsilon: float = 1e-6


# ---------------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------------

class BaseMemory(keras.layers.Layer, ABC):
    """
    Abstract base class for memory modules.
    
    This class defines the interface for memory operations including
    initialization, reading, writing, and state management.
    
    All memory implementations must inherit from this class and implement
    the abstract methods.
    
    Attributes:
        memory_size: Number of memory slots.
        memory_dim: Dimension of each memory slot.
        epsilon: Small constant for numerical stability.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the memory module.
        
        Args:
            memory_size: Number of memory slots.
            memory_dim: Dimension of each memory slot.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.epsilon = epsilon
    
    @abstractmethod
    def initialize_state(self, batch_size: int) -> MemoryState:
        """
        Initialize memory state for a new sequence.
        
        Args:
            batch_size: Number of sequences in the batch.
            
        Returns:
            MemoryState: Initial memory state.
        """
        pass
    
    @abstractmethod
    def read(
        self,
        memory_state: MemoryState,
        read_weights: Any,
    ) -> Any:
        """
        Read from memory using attention weights.
        
        Args:
            memory_state: Current memory state.
            read_weights: Attention weights of shape (batch, num_slots).
            
        Returns:
            Read vector of shape (batch, memory_dim).
        """
        pass
    
    @abstractmethod
    def write(
        self,
        memory_state: MemoryState,
        write_weights: Any,
        erase_vector: Any,
        add_vector: Any,
    ) -> MemoryState:
        """
        Write to memory using erase and add operations.
        
        Args:
            memory_state: Current memory state.
            write_weights: Write attention weights of shape (batch, num_slots).
            erase_vector: Erase vector of shape (batch, memory_dim).
            add_vector: Add vector of shape (batch, memory_dim).
            
        Returns:
            MemoryState: Updated memory state.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'memory_size': self.memory_size,
            'memory_dim': self.memory_dim,
            'epsilon': self.epsilon,
        })
        return config


class BaseHead(keras.layers.Layer, ABC):
    """
    Abstract base class for read and write heads.
    
    Heads are responsible for computing attention weights over memory
    using various addressing mechanisms.
    
    Attributes:
        memory_size: Number of memory slots.
        memory_dim: Dimension of each memory slot.
        addressing_mode: Type of addressing mechanism.
        shift_range: Range of allowed shifts for location addressing.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        addressing_mode: AddressingMode = AddressingMode.HYBRID,
        shift_range: int = 3,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the head.
        
        Args:
            memory_size: Number of memory slots.
            memory_dim: Dimension of each memory slot.
            addressing_mode: Type of addressing mechanism.
            shift_range: Range of allowed shifts for location addressing.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.addressing_mode = addressing_mode
        self.shift_range = shift_range
        self.epsilon = epsilon
    
    @abstractmethod
    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> Tuple[Any, HeadState]:
        """
        Compute attention weights using the addressing mechanism.
        
        Args:
            controller_output: Output from the controller network.
            memory_state: Current memory state.
            prev_weights: Previous attention weights.
            
        Returns:
            Tuple of:
                - New attention weights of shape (batch, num_slots).
                - Updated head state.
        """
        pass
    
    @abstractmethod
    def content_addressing(
        self,
        key: Any,
        beta: Any,
        memory: Any,
    ) -> Any:
        """
        Compute content-based attention weights.
        
        Args:
            key: Key vector of shape (batch, memory_dim).
            beta: Key strength of shape (batch, 1).
            memory: Memory matrix of shape (batch, num_slots, memory_dim).
            
        Returns:
            Content weights of shape (batch, num_slots).
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'memory_size': self.memory_size,
            'memory_dim': self.memory_dim,
            'addressing_mode': self.addressing_mode.name,
            'shift_range': self.shift_range,
            'epsilon': self.epsilon,
        })
        return config


class BaseController(keras.layers.Layer, ABC):
    """
    Abstract base class for controller networks.
    
    The controller processes inputs concatenated with read vectors
    and produces outputs that parameterize the memory operations.
    
    Attributes:
        controller_dim: Dimension of controller hidden state.
        controller_type: Type of controller architecture.
    """
    
    def __init__(
        self,
        controller_dim: int,
        controller_type: Literal['lstm', 'gru', 'feedforward'] = 'lstm',
        **kwargs: Any,
    ) -> None:
        """
        Initialize the controller.
        
        Args:
            controller_dim: Dimension of controller hidden state.
            controller_type: Type of controller architecture.
            **kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.controller_dim = controller_dim
        self.controller_type = controller_type
    
    @abstractmethod
    def initialize_state(self, batch_size: int) -> Optional[Any]:
        """
        Initialize controller state for a new sequence.
        
        Args:
            batch_size: Number of sequences in the batch.
            
        Returns:
            Initial controller state, or None for feedforward controllers.
        """
        pass
    
    @abstractmethod
    def call(
        self,
        inputs: Any,
        state: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Process inputs through the controller.
        
        Args:
            inputs: Input tensor concatenated with read vectors.
            state: Previous controller state.
            training: Whether in training mode.
            
        Returns:
            Tuple of:
                - Controller output.
                - Updated controller state.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'controller_dim': self.controller_dim,
            'controller_type': self.controller_type,
        })
        return config


class BaseNTM(keras.layers.Layer, ABC):
    """
    Abstract base class for Neural Turing Machine architectures.
    
    This class defines the complete interface for NTM variants,
    including initialization, forward pass, and state management.
    
    All NTM implementations must inherit from this class.
    
    Attributes:
        config: NTM configuration object.
        memory: Memory module.
        controller: Controller network.
        read_heads: List of read heads.
        write_heads: List of write heads.
    """
    
    def __init__(
        self,
        config: NTMConfig,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the NTM.
        
        Args:
            config: NTM configuration object.
            **kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.config = config
        
        # Subclasses must initialize these
        self.memory: Optional[BaseMemory] = None
        self.controller: Optional[BaseController] = None
        self.read_heads: List[BaseHead] = []
        self.write_heads: List[BaseHead] = []
    
    @abstractmethod
    def initialize_state(
        self,
        batch_size: int,
    ) -> Tuple[MemoryState, List[HeadState], Optional[Any]]:
        """
        Initialize all states for a new sequence.
        
        Args:
            batch_size: Number of sequences in the batch.
            
        Returns:
            Tuple of:
                - Initial memory state.
                - List of initial head states.
                - Initial controller state.
        """
        pass
    
    @abstractmethod
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
        
        Args:
            inputs: Input at current timestep, shape (batch, input_dim).
            memory_state: Current memory state.
            head_states: Current head states.
            controller_state: Current controller state.
            training: Whether in training mode.
            
        Returns:
            NTMOutput containing all outputs and updated states.
        """
        pass
    
    def call(
        self,
        inputs: Any,
        initial_state: Optional[Tuple[MemoryState, List[HeadState], Any]] = None,
        training: Optional[bool] = None,
        return_sequences: bool = True,
        return_state: bool = False,
    ) -> Union[Any, Tuple[Any, ...]]:
        """
        Process a sequence through the NTM.
        
        Args:
            inputs: Input sequence of shape (batch, seq_len, input_dim).
            initial_state: Optional tuple of initial states.
            training: Whether in training mode.
            return_sequences: Whether to return outputs at all timesteps.
            return_state: Whether to return final states.
            
        Returns:
            Output tensor(s) and optionally final states.
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        
        # Initialize states
        if initial_state is None:
            memory_state, head_states, controller_state = self.initialize_state(
                batch_size
            )
        else:
            memory_state, head_states, controller_state = initial_state
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            step_input = inputs[:, t, :]
            ntm_output = self.step(
                step_input,
                memory_state,
                head_states,
                controller_state,
                training=training,
            )
            
            outputs.append(ntm_output.output)
            memory_state = ntm_output.memory_state
            head_states = ntm_output.head_states
            controller_state = ntm_output.controller_state
        
        # Stack outputs
        if return_sequences:
            output = ops.stack(outputs, axis=1)  # (batch, seq_len, output_dim)
        else:
            output = outputs[-1]  # (batch, output_dim)
        
        if return_state:
            return output, (memory_state, head_states, controller_state)
        return output
    
    @abstractmethod
    def get_memory_state(self) -> Optional[MemoryState]:
        """
        Get the current memory state.
        
        Returns:
            Current memory state or None if not initialized.
        """
        pass
    
    @abstractmethod
    def reset_memory(self, batch_size: int) -> None:
        """
        Reset memory to initial state.
        
        Args:
            batch_size: Number of sequences in the batch.
        """
        pass
    
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
            }
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseNTM':
        """Create layer from configuration."""
        ntm_config_dict = config.pop('config')
        ntm_config_dict['addressing_mode'] = AddressingMode[
            ntm_config_dict['addressing_mode']
        ]
        ntm_config = NTMConfig(**ntm_config_dict)
        return cls(config=ntm_config, **config)


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def cosine_similarity(
    a: Any,
    b: Any,
    epsilon: float = 1e-6,
) -> Any:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        a: First tensor of shape (..., dim).
        b: Second tensor of shape (..., dim).
        epsilon: Small constant for numerical stability.
        
    Returns:
        Cosine similarity of shape (...,).
    """
    a_norm = ops.sqrt(ops.sum(ops.square(a), axis=-1, keepdims=True) + epsilon)
    b_norm = ops.sqrt(ops.sum(ops.square(b), axis=-1, keepdims=True) + epsilon)
    
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    
    return ops.sum(a_normalized * b_normalized, axis=-1)


def circular_convolution(
    weights: Any,
    shift: Any,
) -> Any:
    """
    Perform circular convolution for location-based addressing.
    
    Args:
        weights: Attention weights of shape (batch, num_slots).
        shift: Shift distribution of shape (batch, shift_range).
        
    Returns:
        Shifted weights of shape (batch, num_slots).
    """
    batch_size = ops.shape(weights)[0]
    num_slots = ops.shape(weights)[1]
    shift_range = ops.shape(shift)[1]
    
    # Compute shift center
    shift_center = shift_range // 2
    
    # Pad weights for circular convolution
    padded_weights = ops.concatenate([
        weights[:, -shift_center:],
        weights,
        weights[:, :shift_center],
    ], axis=1)
    
    # Apply convolution
    result = ops.zeros((batch_size, num_slots))
    for i in range(shift_range):
        shifted = padded_weights[:, i:i + num_slots]
        result = result + shift[:, i:i+1] * shifted
    
    return result


def sharpen_weights(
    weights: Any,
    gamma: Any,
    epsilon: float = 1e-6,
) -> Any:
    """
    Sharpen attention weights using gamma parameter.
    
    Args:
        weights: Attention weights of shape (batch, num_slots).
        gamma: Sharpening factor of shape (batch, 1), >= 1.
        epsilon: Small constant for numerical stability.
        
    Returns:
        Sharpened weights of shape (batch, num_slots).
    """
    # Ensure gamma >= 1
    gamma = ops.maximum(gamma, 1.0)
    
    # Raise to power and normalize
    sharpened = ops.power(weights + epsilon, gamma)
    return sharpened / (ops.sum(sharpened, axis=-1, keepdims=True) + epsilon)
