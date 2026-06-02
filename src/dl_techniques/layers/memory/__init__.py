"""
Memory-Augmented Neural Networks Package.

This package consolidates memory-augmented and topographic-memory layer families
that were previously split across `layers/ntm/` and `layers/memory/`. It provides:

* **NTM family** — Neural Turing Machine (Graves et al., 2014): differentiable
  external memory with content + location addressing. Re-exported from
  `ntm_interface` and `baseline_ntm`.
* **MANN** — Memory-Augmented Neural Network (Santoro et al., 2016) style
  standalone layer built on the NTM idea. Defined in `mann.py`.
* **SOM family** — Self-Organizing Maps (Kohonen, 1982): 2D, ND, and soft
  variants. Defined in `som_2d_layer.py`, `som_nd_layer.py`, `som_nd_soft_layer.py`.

Modules:
    ntm_interface: Abstract base classes, dataclasses, enums, and utilities.
    baseline_ntm: Production-ready NTM implementation + factory.
    mann: Standalone Memory-Augmented Neural Network layer.
    som_nd_layer: N-dimensional Self-Organizing Map layer (hard winner).
    som_2d_layer: 2D Self-Organizing Map layer (subclass of SOMLayer).
    som_nd_soft_layer: Soft (differentiable) N-dimensional SOM layer.

NTM Classes:
    Interface:
        - BaseMemory, BaseHead, BaseController, BaseNTM
    State / Config:
        - MemoryState, HeadState, NTMOutput, NTMConfig
    Enumerations:
        - AddressingMode, MemoryAccessType
    Baseline Implementation:
        - NTMMemory, NTMReadHead, NTMWriteHead, NTMController, NTMCell,
          NeuralTuringMachine

NTM Functions:
    - create_ntm, cosine_similarity, circular_convolution, sharpen_weights

MANN Class:
    - MannLayer

SOM Classes:
    - SOMLayer (ND, hard winner), SOM2dLayer (2D specialization), SoftSOMLayer

Example:
    >>> from dl_techniques.layers.memory import (
    ...     create_ntm, NTMConfig, MannLayer, SOMLayer, SOM2dLayer, SoftSOMLayer,
    ... )
    >>> ntm = create_ntm(memory_size=128, memory_dim=64, output_dim=10,
    ...                  controller_type='lstm')
"""

# ---------------------------------------------------------------------------
# NTM family
# ---------------------------------------------------------------------------

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

from .baseline_ntm import (
    NTMMemory,
    NTMReadHead,
    NTMWriteHead,
    NTMController,
    NTMCell,
    NeuralTuringMachine,
    create_ntm,
)

# ---------------------------------------------------------------------------
# MANN
# ---------------------------------------------------------------------------

from .mann import MannLayer

# ---------------------------------------------------------------------------
# SOM family
# ---------------------------------------------------------------------------

from .som_nd_layer import SOMLayer
from .som_2d_layer import SOM2dLayer
from .som_nd_soft_layer import SoftSOMLayer

# ---------------------------------------------------------------------------
# NeuroGrid (topographic memory grid)
# ---------------------------------------------------------------------------

from .neuro_grid import NeuroGrid

# ---------------------------------------------------------------------------
# Factory (recommended construction surface)
# ---------------------------------------------------------------------------

from .factory import create_mann, create_som_2d  # noqa: E402  (after class imports)


__all__ = [
    # NTM — enumerations
    "AddressingMode",
    "MemoryAccessType",
    # NTM — state / config dataclasses
    "MemoryState",
    "HeadState",
    "NTMOutput",
    "NTMConfig",
    # NTM — abstract base classes
    "BaseMemory",
    "BaseHead",
    "BaseController",
    "BaseNTM",
    # NTM — utility functions
    "cosine_similarity",
    "circular_convolution",
    "sharpen_weights",
    # NTM — baseline implementation
    "NTMMemory",
    "NTMReadHead",
    "NTMWriteHead",
    "NTMController",
    "NTMCell",
    "NeuralTuringMachine",
    "create_ntm",
    # MANN
    "MannLayer",
    # SOM family
    "SOMLayer",
    "SOM2dLayer",
    "SoftSOMLayer",
    # NeuroGrid
    "NeuroGrid",
    # Factory functions
    "create_mann",
    "create_som_2d",
]

__version__ = "1.0.0"
