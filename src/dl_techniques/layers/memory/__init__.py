"""
Memory-augmented and topographic-memory layers.

This package is the single home for layer families that were previously split
across `layers/ntm/` and `layers/memory/`. Two families live here, plus a
standalone grid layer:

* **NTM** (Graves et al., 2014): differentiable external memory with content and
  location addressing. Interfaces in `ntm_interface.py`, implementation in
  `baseline_ntm.py`.
* **SOM** (Kohonen, 1982): Self-Organizing Maps, in 2D, N-D and soft variants.
* **NeuroGrid**: a topographic memory grid with differentiable soft assignment.

`__all__` re-exports 25 names. Import from the package, not the modules.

**Package Surface:**

.. code-block:: text

    dl_techniques.layers.memory
      │
      ├─ ntm_interface.py      enums, state dataclasses, ABCs,
      │                        addressing utilities
      ├─ baseline_ntm.py       NTMMemory, NTMReadHead,
      │                        NTMWriteHead, NTMController,
      │                        NTMCell, NeuralTuringMachine,
      │                        create_ntm
      ├─ som_nd_layer.py       SOMLayer (N-D, hard winner)
      ├─ som_2d_layer.py       SOM2dLayer (2D SOMLayer subclass)
      ├─ som_nd_soft_layer.py  SoftSOMLayer (differentiable)
      ├─ neuro_grid.py         NeuroGrid
      └─ factory.py            create_mann, create_som_2d

Exported names, by group:

* Enums: `AddressingMode`.
* State and config: `MemoryState`, `HeadState`, `NTMOutput`, `NTMConfig`.
* Abstract bases: `BaseMemory`, `BaseHead`, `BaseController`, `BaseNTM`.
* Addressing utilities: `cosine_similarity`, `circular_convolution`,
  `sharpen_weights`.
* NTM layers: `NTMMemory`, `NTMReadHead`, `NTMWriteHead`, `NTMController`,
  `NTMCell`, `NeuralTuringMachine`.
* SOM layers: `SOMLayer`, `SOM2dLayer`, `SoftSOMLayer`.
* Grid layer: `NeuroGrid`.
* Builders: `create_ntm`, `create_mann`, `create_som_2d`.

There is no standalone MANN class. `create_mann` returns a configured
`NeuralTuringMachine`; see `factory.py` for why.

Example:
    >>> from dl_techniques.layers.memory import (
    ...     create_ntm, NTMConfig, SOMLayer, SOM2dLayer, SoftSOMLayer,
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

# Imported last so the classes above are already bound.
from .factory import create_mann, create_som_2d


__all__ = [
    # NTM — enumerations
    "AddressingMode",
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
