"""
Neural Turing Machine (NTM) Interface Module.

This module defines the abstract base classes and data structures for all NTM
variants, establishing a consistent interface for memory-augmented neural
network implementations.

Based on: "Neural Turing Machines" (Graves et al., 2014)

Classes:
    AddressingMode: Enum for addressing mechanism types.
    MemoryAccessType: Enum for memory access types.
    MemoryState: Dataclass representing memory state.
    HeadState: Dataclass representing head state.
    NTMOutput: Dataclass for NTM forward pass outputs.
    NTMConfig: Configuration dataclass for NTM architectures.
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
from typing import Any, Literal


# ---------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------


class AddressingMode(Enum):
    """
    Enumeration of addressing mechanism types.

    :cvar CONTENT: Content-based addressing using similarity measures.
    :cvar LOCATION: Location-based addressing using shifts and interpolation.
    :cvar HYBRID: Combined content and location addressing (original NTM).
    :cvar SPARSE: Sparse attention-based addressing (SANTM).
    :cvar TEMPORAL: Temporal link-based addressing (DNC).
    :cvar LEARNED: Fully learned addressing (Titans).
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

    :cvar READ: Read-only memory access.
    :cvar WRITE: Write-only memory access.
    :cvar READ_WRITE: Combined read and write access.
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

    :param memory: Memory matrix of shape (batch, num_slots, memory_dim).
    :type memory: Any
    :param usage: Memory usage vector of shape (batch, num_slots).
    :type usage: Any | None
    :param write_weights: Most recent write weights of shape
        (batch, num_heads, num_slots).
    :type write_weights: Any | None
    :param read_weights: Most recent read weights of shape
        (batch, num_heads, num_slots).
    :type read_weights: Any | None
    :param temporal_links: Optional temporal link matrix for DNC-style memory.
    :type temporal_links: Any | None
    :param precedence: Optional precedence weights for temporal ordering.
    :type precedence: Any | None
    :param metadata: Optional dictionary for variant-specific state.
    :type metadata: dict[str, Any]
    """

    memory: Any
    usage: Any | None = None
    write_weights: Any | None = None
    read_weights: Any | None = None
    temporal_links: Any | None = None
    precedence: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "MemoryState":
        """
        Create a shallow copy of the memory state.

        :return: A new MemoryState with copied references.
        :rtype: MemoryState
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

    :param weights: Attention weights over memory slots, shape (batch, num_slots).
    :type weights: Any
    :param read_vector: Last read vector for read heads, shape (batch, memory_dim).
    :type read_vector: Any | None
    :param key: Key vector used for content addressing.
    :type key: Any | None
    :param beta: Key strength (sharpening factor).
    :type beta: Any | None
    :param gate: Interpolation gate between content and location addressing.
    :type gate: Any | None
    :param shift: Shift distribution for location addressing.
    :type shift: Any | None
    :param gamma: Sharpening factor for final weights.
    :type gamma: Any | None
    :param erase_vector: Erase vector for write heads.
    :type erase_vector: Any | None
    :param add_vector: Add vector for write heads.
    :type add_vector: Any | None
    :param metadata: Optional dictionary for variant-specific state.
    :type metadata: dict[str, Any]
    """

    weights: Any
    read_vector: Any | None = None
    key: Any | None = None
    beta: Any | None = None
    gate: Any | None = None
    shift: Any | None = None
    gamma: Any | None = None
    erase_vector: Any | None = None
    add_vector: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NTMOutput:
    """
    Output structure for NTM forward pass.

    :param output: The network output, shape (batch, output_dim).
    :type output: Any
    :param memory_state: Updated memory state after the forward pass.
    :type memory_state: MemoryState
    :param head_states: List of updated head states.
    :type head_states: list[HeadState]
    :param read_vectors: Read vectors from all read heads.
    :type read_vectors: Any
    :param controller_state: Controller hidden state (for recurrent controllers).
    :type controller_state: Any | None
    :param attention_weights: Attention weights for visualization/analysis.
    :type attention_weights: dict[str, Any] | None
    :param auxiliary_losses: Optional auxiliary losses (e.g., regularization).
    :type auxiliary_losses: dict[str, Any] | None
    :param metadata: Optional dictionary for additional outputs.
    :type metadata: dict[str, Any]
    """

    output: Any
    memory_state: MemoryState
    head_states: list[HeadState]
    read_vectors: Any
    controller_state: Any | None = None
    attention_weights: dict[str, Any] | None = None
    auxiliary_losses: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NTMConfig:
    """
    Configuration for NTM architectures.

    :param memory_size: Number of memory slots (N).
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot (M).
    :type memory_dim: int
    :param num_read_heads: Number of read heads.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads.
    :type num_write_heads: int
    :param controller_dim: Dimension of controller hidden state.
    :type controller_dim: int
    :param controller_type: Type of controller ('lstm', 'gru', 'feedforward').
    :type controller_type: str
    :param addressing_mode: Type of addressing mechanism.
    :type addressing_mode: AddressingMode
    :param shift_range: Range of allowed shifts for location addressing.
    :type shift_range: int
    :param use_memory_init: Whether to learn initial memory state.
    :type use_memory_init: bool
    :param clip_value: Gradient clipping value for stability.
    :type clip_value: float
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    """

    memory_size: int = 128
    memory_dim: int = 64
    num_read_heads: int = 1
    num_write_heads: int = 1
    controller_dim: int = 256
    controller_type: Literal["lstm", "gru", "feedforward"] = "lstm"
    addressing_mode: AddressingMode = AddressingMode.HYBRID
    shift_range: int = 3
    use_memory_init: bool = True
    clip_value: float = 10.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {self.memory_size}")
        if self.memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {self.memory_dim}")
        if self.num_read_heads <= 0:
            raise ValueError(
                f"num_read_heads must be positive, got {self.num_read_heads}"
            )
        if self.num_write_heads <= 0:
            raise ValueError(
                f"num_write_heads must be positive, got {self.num_write_heads}"
            )
        if self.controller_dim <= 0:
            raise ValueError(
                f"controller_dim must be positive, got {self.controller_dim}"
            )
        if self.controller_type not in ["lstm", "gru", "feedforward"]:
            raise ValueError(
                f"controller_type must be 'lstm', 'gru', or 'feedforward', "
                f"got {self.controller_type}"
            )
        if self.shift_range <= 0 or self.shift_range % 2 == 0:
            raise ValueError(
                f"shift_range must be a positive odd integer, got {self.shift_range}"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        :return: Configuration as dictionary.
        :rtype: dict[str, Any]
        """
        return {
            "memory_size": self.memory_size,
            "memory_dim": self.memory_dim,
            "num_read_heads": self.num_read_heads,
            "num_write_heads": self.num_write_heads,
            "controller_dim": self.controller_dim,
            "controller_type": self.controller_type,
            "addressing_mode": self.addressing_mode.name,
            "shift_range": self.shift_range,
            "use_memory_init": self.use_memory_init,
            "clip_value": self.clip_value,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "NTMConfig":
        """
        Create configuration from dictionary.

        :param config_dict: Configuration dictionary.
        :type config_dict: dict[str, Any]
        :return: NTMConfig instance.
        :rtype: NTMConfig
        """
        config = config_dict.copy()
        if "addressing_mode" in config and isinstance(config["addressing_mode"], str):
            config["addressing_mode"] = AddressingMode[config["addressing_mode"]]
        return cls(**config)


# ---------------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------------


class BaseMemory(keras.layers.Layer, ABC):
    """
    Abstract base class for memory modules.

    This class defines the interface for memory operations including
    initialization, reading, writing, and state management.

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :param kwargs: Additional arguments for keras.layers.Layer.
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

        :param memory_size: Number of memory slots.
        :type memory_size: int
        :param memory_dim: Dimension of each memory slot.
        :type memory_dim: int
        :param epsilon: Small constant for numerical stability.
        :type epsilon: float
        :param kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.epsilon = epsilon

    @abstractmethod
    def initialize_state(self, batch_size: int) -> MemoryState:
        """
        Initialize memory state for a new sequence.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: Initial memory state.
        :rtype: MemoryState
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

        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param read_weights: Attention weights of shape (batch, num_slots).
        :type read_weights: Any
        :return: Read vector of shape (batch, memory_dim).
        :rtype: Any
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

        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param write_weights: Write attention weights of shape (batch, num_slots).
        :type write_weights: Any
        :param erase_vector: Erase vector of shape (batch, memory_dim).
        :type erase_vector: Any
        :param add_vector: Add vector of shape (batch, memory_dim).
        :type add_vector: Any
        :return: Updated memory state.
        :rtype: MemoryState
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "memory_dim": self.memory_dim,
                "epsilon": self.epsilon,
            }
        )
        return config


class BaseHead(keras.layers.Layer, ABC):
    """
    Abstract base class for read and write heads.

    Heads are responsible for computing attention weights over memory
    using various addressing mechanisms.

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param addressing_mode: Type of addressing mechanism.
    :type addressing_mode: AddressingMode
    :param shift_range: Range of allowed shifts for location addressing.
    :type shift_range: int
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :param kwargs: Additional arguments for keras.layers.Layer.
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

        :param memory_size: Number of memory slots.
        :type memory_size: int
        :param memory_dim: Dimension of each memory slot.
        :type memory_dim: int
        :param addressing_mode: Type of addressing mechanism.
        :type addressing_mode: AddressingMode
        :param shift_range: Range of allowed shifts for location addressing.
        :type shift_range: int
        :param epsilon: Small constant for numerical stability.
        :type epsilon: float
        :param kwargs: Additional arguments for keras.layers.Layer.
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
    ) -> tuple[Any, HeadState]:
        """
        Compute attention weights using the addressing mechanism.

        :param controller_output: Output from the controller network.
        :type controller_output: Any
        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param prev_weights: Previous attention weights.
        :type prev_weights: Any
        :return: Tuple of (new_weights, head_state).
        :rtype: tuple[Any, HeadState]
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

        :param key: Key vector of shape (batch, memory_dim).
        :type key: Any
        :param beta: Key strength of shape (batch, 1).
        :type beta: Any
        :param memory: Memory matrix of shape (batch, num_slots, memory_dim).
        :type memory: Any
        :return: Content weights of shape (batch, num_slots).
        :rtype: Any
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "memory_dim": self.memory_dim,
                "addressing_mode": self.addressing_mode.name,
                "shift_range": self.shift_range,
                "epsilon": self.epsilon,
            }
        )
        return config


class BaseController(keras.layers.Layer, ABC):
    """
    Abstract base class for controller networks.

    The controller processes inputs concatenated with read vectors
    and produces outputs that parameterize the memory operations.

    :param controller_dim: Dimension of controller hidden state.
    :type controller_dim: int
    :param controller_type: Type of controller architecture.
    :type controller_type: str
    :param kwargs: Additional arguments for keras.layers.Layer.
    """

    def __init__(
        self,
        controller_dim: int,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the controller.

        :param controller_dim: Dimension of controller hidden state.
        :type controller_dim: int
        :param controller_type: Type of controller architecture.
        :type controller_type: str
        :param kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.controller_dim = controller_dim
        self.controller_type = controller_type

    @abstractmethod
    def initialize_state(self, batch_size: int) -> Any | None:
        """
        Initialize controller state for a new sequence.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: Initial controller state, or None for feedforward controllers.
        :rtype: Any | None
        """
        pass

    @abstractmethod
    def call(
        self,
        inputs: Any,
        state: Any | None = None,
        training: bool | None = None,
    ) -> tuple[Any, Any | None]:
        """
        Process inputs through the controller.

        :param inputs: Input tensor concatenated with read vectors.
        :type inputs: Any
        :param state: Previous controller state.
        :type state: Any | None
        :param training: Whether in training mode.
        :type training: bool | None
        :return: Tuple of (controller_output, new_state).
        :rtype: tuple[Any, Any | None]
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "controller_dim": self.controller_dim,
                "controller_type": self.controller_type,
            }
        )
        return config


class BaseNTM(keras.layers.Layer, ABC):
    """
    Abstract base class for Neural Turing Machine architectures.

    This class defines the complete interface for NTM variants,
    including initialization, forward pass, and state management.

    :param config: NTM configuration object.
    :type config: NTMConfig
    :param output_dim: Dimension of the output vector.
    :type output_dim: int | None
    :param kwargs: Additional arguments for keras.layers.Layer.
    """

    def __init__(
        self,
        config: NTMConfig,
        output_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the NTM.

        :param config: NTM configuration object.
        :type config: NTMConfig
        :param output_dim: Dimension of the output vector.
        :type output_dim: int | None
        :param kwargs: Additional arguments for keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.config = config
        self.output_dim = output_dim

        # Subclasses must initialize these
        self.memory: BaseMemory | None = None
        self.controller: BaseController | None = None
        self.read_heads: list[BaseHead] = []
        self.write_heads: list[BaseHead] = []

    @abstractmethod
    def initialize_state(
        self,
        batch_size: int,
    ) -> tuple[MemoryState, list[HeadState], Any | None]:
        """
        Initialize all states for a new sequence.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: Tuple of (memory_state, head_states, controller_state).
        :rtype: tuple[MemoryState, list[HeadState], Any | None]
        """
        pass

    @abstractmethod
    def step(
        self,
        inputs: Any,
        memory_state: MemoryState,
        head_states: list[HeadState],
        controller_state: Any | None,
        training: bool | None = None,
    ) -> NTMOutput:
        """
        Perform a single timestep of the NTM.

        :param inputs: Input at current timestep, shape (batch, input_dim).
        :type inputs: Any
        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param head_states: Current head states.
        :type head_states: list[HeadState]
        :param controller_state: Current controller state.
        :type controller_state: Any | None
        :param training: Whether in training mode.
        :type training: bool | None
        :return: NTMOutput containing all outputs and updated states.
        :rtype: NTMOutput
        """
        pass

    def call(
        self,
        inputs: Any,
        initial_state: tuple[MemoryState, list[HeadState], Any] | None = None,
        training: bool | None = None,
        return_sequences: bool = True,
        return_state: bool = False,
    ) -> Any | tuple[Any, ...]:
        """
        Process a sequence through the NTM.

        :param inputs: Input sequence of shape (batch, seq_len, input_dim).
        :type inputs: Any
        :param initial_state: Optional tuple of initial states.
        :type initial_state: tuple[MemoryState, list[HeadState], Any] | None
        :param training: Whether in training mode.
        :type training: bool | None
        :param return_sequences: Whether to return outputs at all timesteps.
        :type return_sequences: bool
        :param return_state: Whether to return final states.
        :type return_state: bool
        :return: Output tensor(s) and optionally final states.
        :rtype: Any | tuple[Any, ...]
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
            output = ops.stack(outputs, axis=1)
        else:
            output = outputs[-1]

        if return_state:
            return output, (memory_state, head_states, controller_state)
        return output

    @abstractmethod
    def get_memory_state(self) -> MemoryState | None:
        """
        Get the current memory state.

        :return: Current memory state or None if not initialized.
        :rtype: MemoryState | None
        """
        pass

    @abstractmethod
    def reset_memory(self, batch_size: int) -> None:
        """
        Reset memory to initial state.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        """
        pass

    def compute_output_shape(self, input_shape: Any) -> Any:
        """
        Compute output shape of the NTM layer.

        :param input_shape: Shape of the input tensor (batch, seq_len, input_dim).
        :type input_shape: Any
        :return: Output shape (batch, seq_len, output_dim).
        :rtype: Any
        """
        if self.output_dim is None:
            raise ValueError(
                "output_dim must be provided in __init__ for compute_output_shape."
            )

        batch_size = input_shape[0]
        seq_len = input_shape[1]
        return (batch_size, seq_len, self.output_dim)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "config": self.config.to_dict(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseNTM":
        """
        Create layer from configuration.

        :param config: Configuration dictionary.
        :type config: dict[str, Any]
        :return: BaseNTM instance.
        :rtype: BaseNTM
        """
        if "config" in config:
            ntm_config_dict = config.pop("config")
            ntm_config = NTMConfig.from_dict(ntm_config_dict)
            return cls(config=ntm_config, **config)
        return cls(**config)


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------


def cosine_similarity(
    query: Any,
    keys: Any,
    epsilon: float = 1e-6,
) -> Any:
    """
    Compute cosine similarity between query and keys.

    :param query: Query tensor of shape (batch, 1, dim) or (batch, dim).
    :type query: Any
    :param keys: Keys tensor of shape (batch, num_slots, dim).
    :type keys: Any
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :return: Cosine similarity of shape (batch, num_slots).
    :rtype: Any
    """
    # Ensure query has 3 dimensions for broadcasting
    if len(ops.shape(query)) == 2:
        query = ops.expand_dims(query, axis=1)

    # Normalize query and keys
    query_norm = ops.sqrt(ops.sum(ops.square(query), axis=-1, keepdims=True) + epsilon)
    keys_norm = ops.sqrt(ops.sum(ops.square(keys), axis=-1, keepdims=True) + epsilon)

    query_normalized = query / query_norm
    keys_normalized = keys / keys_norm

    # Compute similarity
    similarity = ops.sum(query_normalized * keys_normalized, axis=-1)

    # Squeeze if query was 2D
    if len(ops.shape(similarity)) == 2 and ops.shape(similarity)[1] == 1:
        similarity = ops.squeeze(similarity, axis=1)

    return similarity


def circular_convolution(
    weights: Any,
    shift: Any,
) -> Any:
    """
    Perform circular convolution for location-based addressing.

    :param weights: Attention weights of shape (batch, num_slots).
    :type weights: Any
    :param shift: Shift distribution of shape (batch, shift_range).
    :type shift: Any
    :return: Shifted weights of shape (batch, num_slots).
    :rtype: Any
    """
    batch_size = ops.shape(weights)[0]
    num_slots = ops.shape(weights)[1]
    shift_range = ops.shape(shift)[1]

    # Compute shift center
    shift_center = shift_range // 2

    # Pad weights for circular convolution
    padded_weights = ops.concatenate(
        [
            weights[:, -shift_center:],
            weights,
            weights[:, :shift_center],
        ],
        axis=1,
    )

    # Apply convolution
    result = ops.zeros((batch_size, num_slots))
    for i in range(shift_range):
        shifted = padded_weights[:, i : i + num_slots]
        result = result + shift[:, i : i + 1] * shifted

    return result


def sharpen_weights(
    weights: Any,
    gamma: Any,
    epsilon: float = 1e-6,
) -> Any:
    """
    Sharpen attention weights using gamma parameter.

    :param weights: Attention weights of shape (batch, num_slots).
    :type weights: Any
    :param gamma: Sharpening factor of shape (batch, 1), >= 1.
    :type gamma: Any
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :return: Sharpened weights of shape (batch, num_slots).
    :rtype: Any
    """
    # Ensure gamma >= 1
    gamma = ops.maximum(gamma, 1.0)

    # Raise to power and normalize
    sharpened = ops.power(weights + epsilon, gamma)
    return sharpened / (ops.sum(sharpened, axis=-1, keepdims=True) + epsilon)