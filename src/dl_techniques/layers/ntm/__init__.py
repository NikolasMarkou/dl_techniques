"""
Neural Turing Machine (NTM) Package.

This package provides implementations of Neural Turing Machines and related
memory-augmented neural network architectures.

Modules:
    ntm_interface: Abstract base classes and data structures.
    base_layers: Differentiable addressing mechanisms.
    baseline_ntm: Production-ready NTM implementation.

Classes:
    Interface Classes:
        - BaseMemory: Abstract base for memory modules.
        - BaseHead: Abstract base for read/write heads.
        - BaseController: Abstract base for controller networks.
        - BaseNTM: Abstract base for complete NTM architectures.

    State Classes:
        - MemoryState: Dataclass for memory state.
        - HeadState: Dataclass for head state.
        - NTMOutput: Dataclass for forward pass outputs.
        - NTMConfig: Configuration for NTM architectures.

    Enumerations:
        - AddressingMode: Types of addressing mechanisms.
        - MemoryAccessType: Types of memory access.

    Base Layers:
        - DifferentiableAddressingHead: NTM-style addressing head.
        - DifferentiableSelectCopy: Read/write operations with multiple heads.
        - SimpleSelectCopy: Simplified select-copy mechanism.

    Baseline Implementation:
        - NTMMemory: External memory module.
        - NTMReadHead: Read head with content/location addressing.
        - NTMWriteHead: Write head with erase/add operations.
        - NTMController: Configurable LSTM/GRU/feedforward controller.
        - NTMCell: Single timestep NTM cell (RNN compatible).
        - NeuralTuringMachine: Complete NTM layer.

Functions:
    - create_ntm: Factory function to create NTM instances.
    - cosine_similarity: Compute cosine similarity.
    - circular_convolution: Circular convolution for location addressing.
    - sharpen_weights: Sharpen attention weights.

Example:
    >>> from ntm import create_ntm, NTMConfig
    >>>
    >>> # Using factory function
    >>> ntm = create_ntm(
    ...     memory_size=128,
    ...     memory_dim=64,
    ...     output_dim=10,
    ...     controller_type='lstm',
    ... )
    >>> output = ntm(input_sequence)
    >>>
    >>> # Using configuration object
    >>> config = NTMConfig(
    ...     memory_size=128,
    ...     memory_dim=64,
    ...     num_read_heads=2,
    ...     controller_dim=256,
    ... )
    >>> ntm = NeuralTuringMachine(config, output_dim=10)
"""

# Interface classes and utilities
from .ntm_interface import (
    # Enumerations
    AddressingMode,
    MemoryAccessType,
    # State dataclasses
    MemoryState,
    HeadState,
    NTMOutput,
    NTMConfig,
    # Abstract base classes
    BaseMemory,
    BaseHead,
    BaseController,
    BaseNTM,
    # Utility functions
    cosine_similarity,
    circular_convolution,
    sharpen_weights,
)

# Base layers
from .base_layers import (
    DifferentiableAddressingHead,
    DifferentiableSelectCopy,
    SimpleSelectCopy,
)

# Baseline implementation
from .baseline_ntm import (
    # Memory
    NTMMemory,
    # Heads
    NTMReadHead,
    NTMWriteHead,
    # Controller
    NTMController,
    # Complete NTM
    NTMCell,
    NeuralTuringMachine,
    # Factory
    create_ntm,
)

__all__ = [
    # Enumerations
    "AddressingMode",
    "MemoryAccessType",
    # State dataclasses
    "MemoryState",
    "HeadState",
    "NTMOutput",
    "NTMConfig",
    # Abstract base classes
    "BaseMemory",
    "BaseHead",
    "BaseController",
    "BaseNTM",
    # Utility functions
    "cosine_similarity",
    "circular_convolution",
    "sharpen_weights",
    # Base layers
    "DifferentiableAddressingHead",
    "DifferentiableSelectCopy",
    "SimpleSelectCopy",
    # Baseline implementation
    "NTMMemory",
    "NTMReadHead",
    "NTMWriteHead",
    "NTMController",
    "NTMCell",
    "NeuralTuringMachine",
    # Factory
    "create_ntm",
]

__version__ = "1.0.0"