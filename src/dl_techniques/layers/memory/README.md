# `dl_techniques.layers.memory`

Layers whose distinguishing feature is explicit, addressable memory. Three
families live here: a learned content- and location-addressable memory matrix
(NTM / MANN), a topographic grid of prototypes (SOM), and a differentiable
memory grid with factorized addressing (NeuroGrid). The package merges the
formerly separate `layers/ntm/` and `layers/memory/`.

## Overview

| Family | Type of memory | Differentiable? | Typical use |
|--------|----------------|-----------------|-------------|
| **NTM** | External memory matrix with read/write heads (content + location addressing) | yes | Algorithmic / sequence tasks needing scratchpad memory |
| **MANN** | The NTM memory, configured with MANN knobs via `create_mann(...)` — no separate class | yes | Drop-in memory-augmented sequence layer |
| **SOM** | Topographic grid of prototypes (hard or soft winner-take-all) | hard: no, soft: yes | Unsupervised representation, clustering, quantization |
| **NeuroGrid** | N-D grid of latent vectors, addressed by one softmax per grid axis | yes | Differentiable bottleneck, soft clustering, quality scoring |

## Families

### NTM — Neural Turing Machine

References: Graves, Wayne & Danihelka, 2014, *Neural Turing Machines* (arXiv:1410.5401).

Files:

- `ntm_interface.py` — abstract base classes (`BaseMemory`, `BaseHead`,
  `BaseController`, `BaseNTM`), state dataclasses (`MemoryState`, `HeadState`,
  `NTMOutput`, `NTMConfig`), the `AddressingMode` enum, and
  addressing utilities (`cosine_similarity`, `circular_convolution`,
  `sharpen_weights`).
- `baseline_ntm.py` — production NTM: `NTMMemory`, `NTMReadHead`,
  `NTMWriteHead`, `NTMController`, `NTMCell` (RNN-cell compatible),
  `NeuralTuringMachine`, and the `create_ntm` factory.

`NeuralTuringMachine` wraps `NTMCell` in a `keras.layers.RNN`. It does not run
the step loop `BaseNTM.call` provides, so its `step()` and `initialize_state()`
raise; use `NTMCell.get_initial_state(...)` and `return_state=True` to reach the
state.

### MANN — Memory-Augmented Neural Network

Reference: Santoro et al., 2016, *One-shot Learning with Memory-Augmented
Neural Networks* (arXiv:1605.06065).

There is no standalone MANN class. MANN is available only as a configuration of
the NTM pipeline, through `factory.py`'s `create_mann(...)`, which returns a
`NeuralTuringMachine` whose output width matches the historical MANN contract
(`controller_units + num_read_heads * memory_dim`).

### SOM — Self-Organizing Map

Reference: Kohonen, 1982, *Self-organized formation of topologically correct
feature maps*, Biological Cybernetics.

Files:

- `som_nd_layer.py` — `SOMLayer`, the N-dimensional hard-winner SOM. Its weight
  map is non-trainable and is updated inside `call()` with `assign_add`, not by
  an optimizer.
- `som_2d_layer.py` — `SOM2dLayer`, a 2D specialization (subclass of `SOMLayer`).
- `som_nd_soft_layer.py` — `SoftSOMLayer`, fully differentiable soft-winner
  variant. Its weight map is trainable and moves only through gradients, which
  is the opposite of `SOMLayer`.

### NeuroGrid

Reference: Lample et al., 2019, *Large Memory Layers with Product Keys*
(NeurIPS), for the factorized addressing idea.

Files:

- `neuro_grid.py` — `NeuroGrid`, a grid of latent vectors addressed by one
  `Dense` plus temperature softmax per grid axis. The joint address is the outer
  product of those per-axis distributions, so a grid of `d1 * ... * dn` cells
  costs `sum d_i` logits rather than `prod d_i`. Accepts rank-2 and rank-3
  inputs.

### Factory

- `factory.py` — `create_mann` and `create_som_2d`, plus a re-export of
  `create_ntm` from `baseline_ntm.py`.

## Public Surface

All 26 names below are importable directly from `dl_techniques.layers.memory`,
and are exactly the contents of its `__all__`.

| Name | Kind | Module | Family |
|------|------|--------|--------|
| `AddressingMode` | Enum | `ntm_interface` | NTM |
| `MemoryState`, `HeadState`, `NTMOutput`, `NTMConfig` | dataclass | `ntm_interface` | NTM |
| `BaseMemory`, `BaseHead`, `BaseController`, `BaseNTM` | ABC | `ntm_interface` | NTM |
| `cosine_similarity`, `circular_convolution`, `sharpen_weights` | function | `ntm_interface` | NTM |
| `NTMMemory`, `NTMReadHead`, `NTMWriteHead`, `NTMController`, `NTMCell`, `NeuralTuringMachine` | Layer | `baseline_ntm` | NTM |
| `create_ntm` | factory fn | `baseline_ntm` | NTM |
| `SOMLayer`, `SOM2dLayer`, `SoftSOMLayer` | Layer | `som_nd_layer`, `som_2d_layer`, `som_nd_soft_layer` | SOM |
| `NeuroGrid` | Layer | `neuro_grid` | NeuroGrid |
| `create_mann`, `create_som_2d` | factory fn | `factory` | MANN, SOM |

## Usage

Every example below was run against this package on 2026-08-30. The shapes in
the comments are measured, not assumed.

### NTM (factory)

```python
from dl_techniques.layers.memory import create_ntm

ntm = create_ntm(
    memory_size=128,
    memory_dim=64,
    output_dim=10,
    controller_type='lstm',
)
y = ntm(x)  # x: (batch, time, features)  ->  y: (batch, time, 10)
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

### MANN (factory)

```python
from dl_techniques.layers.memory import create_mann

mann = create_mann(memory_locations=128, memory_dim=40, controller_units=200)
y = mann(x)  # x: (batch, time, features)  ->  y: (batch, time, 240)
```

The output width is `controller_units + num_read_heads * memory_dim`, so
`200 + 1 * 40 = 240`.

### SOM (hard winner, N-D grid)

```python
from dl_techniques.layers.memory import SOMLayer

som = SOMLayer(grid_shape=(10, 10), input_dim=128)
bmu_coords, quantization_errors = som(x, training=True)
# bmu_coords: (batch, 2) int32   quantization_errors: (batch,)
```

The layer returns grid coordinates and a distance, not a code vector. It also
updates its own weight map when `training=True`.

### SOM 2D (factory)

```python
from dl_techniques.layers.memory import create_som_2d

som2d = create_som_2d(map_size=(10, 10), input_dim=128)
bmu_coords, quantization_errors = som2d(x)
grid = som2d.get_weights_as_grid()  # (10, 10, 128)
```

### SoftSOM (differentiable)

```python
from dl_techniques.layers.memory import SoftSOMLayer

soft_som = SoftSOMLayer(grid_shape=(10, 10), input_dim=128, temperature=0.5)
reconstruction = soft_som(x)          # (batch, 128), same shape as x
assignments = soft_som.get_soft_assignments(x)   # (batch, 10, 10)
```

### NeuroGrid

```python
from dl_techniques.layers.memory import NeuroGrid

grid = NeuroGrid(grid_shape=[10, 8], latent_dim=32)
y = grid(x)                                    # (batch, 32)
probs = grid.get_addressing_probabilities(x)['joint']   # (batch, 10, 8)
```

## References

- Graves, A., Wayne, G., Danihelka, I. (2014). *Neural Turing Machines.* arXiv:1410.5401.
- Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., Lillicrap, T. (2016).
  *One-shot Learning with Memory-Augmented Neural Networks.* arXiv:1605.06065.
- Kohonen, T. (1982). *Self-organized formation of topologically correct
  feature maps.* Biological Cybernetics, 43(1), 59-69.
- Lample, G., Sablayrolles, A., Ranzato, M., Denoyer, L., Jegou, H. (2019).
  *Large Memory Layers with Product Keys.* NeurIPS.
