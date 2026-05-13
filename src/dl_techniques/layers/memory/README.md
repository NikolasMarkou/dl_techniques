# `dl_techniques.layers.memory`

Memory-augmented and topographic-memory layer families. This package consolidates
the formerly-separate `layers/ntm/` and `layers/memory/` packages into a single
domain-aligned home for layers whose distinguishing feature is **explicit,
addressable memory** — either a learned content/location-addressable matrix
(NTM / MANN) or a topographic grid of prototypes (SOM).

## Overview

| Family | Type of memory | Differentiable? | Typical use |
|--------|----------------|-----------------|-------------|
| **NTM** | External memory matrix with read/write heads (content + location addressing) | yes | Algorithmic / sequence tasks needing scratchpad memory |
| **MANN** | Standalone NTM-style layer (single-block memory + controller) | yes | Drop-in memory-augmented sequence layer |
| **SOM** | Topographic grid of prototypes (hard or soft winner-take-all) | hard: no, soft: yes | Unsupervised representation, clustering, quantization |

## Families

### NTM — Neural Turing Machine

References: Graves, Wayne & Danihelka, 2014, *Neural Turing Machines* (arXiv:1410.5401).

Files:

- `ntm_interface.py` — abstract base classes (`BaseMemory`, `BaseHead`,
  `BaseController`, `BaseNTM`), state dataclasses (`MemoryState`, `HeadState`,
  `NTMOutput`, `NTMConfig`), enums (`AddressingMode`, `MemoryAccessType`), and
  addressing utilities (`cosine_similarity`, `circular_convolution`,
  `sharpen_weights`).
- `base_layers.py` — differentiable addressing heads
  (`DifferentiableAddressingHead`, `DifferentiableSelectCopy`,
  `SimpleSelectCopy`).
- `baseline_ntm.py` — production NTM: `NTMMemory`, `NTMReadHead`,
  `NTMWriteHead`, `NTMController`, `NTMCell` (RNN-cell compatible),
  `NeuralTuringMachine`, and the `create_ntm` factory.

### MANN — Memory-Augmented Neural Network

Reference: Santoro et al., 2016, *One-shot Learning with Memory-Augmented
Neural Networks* (arXiv:1605.06065).

File:

- `mann.py` — `MannLayer`. Standalone implementation; does **not** inherit
  from `BaseNTM`. Kept separate by design (different controller/head wiring
  and self-contained configuration).

### SOM — Self-Organizing Map

Reference: Kohonen, 1982, *Self-organized formation of topologically correct
feature maps*, Biological Cybernetics.

Files:

- `som_nd_layer.py` — `SOMLayer`, the N-dimensional hard-winner SOM.
- `som_2d_layer.py` — `SOM2dLayer`, a 2D specialization (subclass of `SOMLayer`).
- `som_nd_soft_layer.py` — `SoftSOMLayer`, fully differentiable soft-winner
  variant suitable for end-to-end gradient training.

## Public Surface

All names below are importable directly from `dl_techniques.layers.memory`.

| Name | Kind | Module | Family |
|------|------|--------|--------|
| `AddressingMode`, `MemoryAccessType` | Enum | `ntm_interface` | NTM |
| `MemoryState`, `HeadState`, `NTMOutput`, `NTMConfig` | dataclass | `ntm_interface` | NTM |
| `BaseMemory`, `BaseHead`, `BaseController`, `BaseNTM` | ABC | `ntm_interface` | NTM |
| `cosine_similarity`, `circular_convolution`, `sharpen_weights` | function | `ntm_interface` | NTM |
| `DifferentiableAddressingHead`, `DifferentiableSelectCopy`, `SimpleSelectCopy` | Layer | `base_layers` | NTM |
| `NTMMemory`, `NTMReadHead`, `NTMWriteHead`, `NTMController`, `NTMCell`, `NeuralTuringMachine` | Layer | `baseline_ntm` | NTM |
| `create_ntm` | factory fn | `baseline_ntm` | NTM |
| `MannLayer` | Layer | `mann` | MANN |
| `SOMLayer`, `SOM2dLayer`, `SoftSOMLayer` | Layer | `som_nd_layer`, `som_2d_layer`, `som_nd_soft_layer` | SOM |

## Usage

### NTM (factory)

```python
from dl_techniques.layers.memory import create_ntm

ntm = create_ntm(
    memory_size=128,
    memory_dim=64,
    output_dim=10,
    controller_type='lstm',
)
y = ntm(x)  # x: (batch, time, features)
```

### NTM (config object + layer)

```python
from dl_techniques.layers.memory import NTMConfig, NeuralTuringMachine

config = NTMConfig(
    memory_size=128, memory_dim=64,
    num_read_heads=2, controller_dim=256,
)
ntm = NeuralTuringMachine(config, output_dim=10)
```

### MANN

```python
from dl_techniques.layers.memory import MannLayer

mann = MannLayer(memory_size=128, memory_dim=40, controller_units=200)
y = mann(x)
```

### SOM (hard winner, N-D grid)

```python
from dl_techniques.layers.memory import SOMLayer

som = SOMLayer(map_size=(10, 10), input_dim=128)
codes = som(x)
```

### SoftSOM (differentiable)

```python
from dl_techniques.layers.memory import SoftSOMLayer

soft_som = SoftSOMLayer(map_size=(10, 10), input_dim=128, temperature=0.5)
codes = soft_som(x)
```

## References

- Graves, A., Wayne, G., Danihelka, I. (2014). *Neural Turing Machines.* arXiv:1410.5401.
- Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., Lillicrap, T. (2016).
  *One-shot Learning with Memory-Augmented Neural Networks.* arXiv:1605.06065.
- Kohonen, T. (1982). *Self-organized formation of topologically correct
  feature maps.* Biological Cybernetics, 43(1), 59-69.
