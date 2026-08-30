"""
Factory functions for the `dl_techniques.layers.memory` package.

Build a memory-augmented layer from configuration instead of instantiating the
class yourself. Three builders share one call shape:

* `create_ntm(...)`: Neural Turing Machine. Defined in `baseline_ntm.py` and
  re-exported here.
* `create_mann(...)`: Memory-Augmented Neural Network. Returns a
  `NeuralTuringMachine` configured through MANN-style argument names.
* `create_som_2d(...)`: 2D Self-Organizing Map.

Each one returns a built Keras layer, ready for `keras.Sequential` or the
functional API.

# DECISION plan_2026-05-13_8c1dc6fd/D-002 [STALE]
`create_mann` returns a `NeuralTuringMachine`, not a dedicated MANN class. Do
NOT add one: it duplicates addressing/read/write logic `NTMCell` already has,
and the NTM RNN-cell wrapper subsumes the LSTMCell rewrite proposed as R4.
Output width is unchanged: `controller_units + num_read_heads * memory_dim`.
The plan that owned this decision is gone; this note is the only record left.

# DECISION plan-2026-08-03T161943-02be1d7e/D-004
The legacy `MannLayer` class (`mann.py`) the note above refers to is DELETED.
Do NOT reintroduce a standalone MANN class as a "simpler" or "more faithful"
alternative: it had zero consumers, was unreachable from this module, and kept
its own copy of the location-shift math; the two copies drifted and only one
was fixed. `create_mann(...)` is the only MANN path. See decisions.md D-004.
"""

from __future__ import annotations

from typing import Any, Literal

from .baseline_ntm import NeuralTuringMachine, create_ntm
from .ntm_interface import NTMConfig
from .som_2d_layer import SOM2dLayer


def create_mann(
    memory_locations: int,
    memory_dim: int,
    controller_units: int,
    num_read_heads: int = 1,
    num_write_heads: int = 1,
    controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
    shift_range: int = 3,
    return_sequences: bool = True,
    return_state: bool = False,
    **kwargs: Any,
) -> NeuralTuringMachine:
    """
    Build a Memory-Augmented Neural Network as a configured `NeuralTuringMachine`.

    This function only assembles configuration. It packs the MANN-style keyword
    arguments into an `NTMConfig`, computes
    ``output_dim = controller_units + num_read_heads * memory_dim`` to keep the
    historical MANN output width, and hands both to `NeuralTuringMachine`. The
    architecture you get is the standard NTM: `NTMMemory`, `NTMReadHead`,
    `NTMWriteHead` and `NTMController` inside an `NTMCell`, which
    `NeuralTuringMachine` wraps in a `keras.layers.RNN`.

    Config fields this function does not expose keep their `NTMConfig` defaults,
    including ``addressing_mode=AddressingMode.HYBRID``.

    **Architecture Overview:**

    .. code-block:: text

        memory_locations, memory_dim, controller_units,
        num_read_heads, num_write_heads, controller_type,
        shift_range
                    │
                    ▼
        ┌────────────────────────────────┐
        │ NTMConfig (dataclass)          │  config, no weights
        └────────────────────────────────┘
                    │
                    │  output_dim = controller_units
                    │             + num_read_heads * memory_dim
                    ▼
        ┌────────────────────────────────────────────────┐
        │ NeuralTuringMachine (returned layer)           │
        │                                                │
        │  x (batch, seq_len, input_dim)                 │
        │            │                                   │
        │            ▼                                   │
        │  ┌──────────────────────────────────────────┐  │
        │  │ keras.layers.RNN(NTMCell)                │  │
        │  └──────────────────────────────────────────┘  │
        │            │  last axis = output_dim           │
        │            ▼                                   │
        │  ┌──────────────────────────────────────────┐  │
        │  │ Dense output_projection                  │  │
        │  └──────────────────────────────────────────┘  │
        │            │                                   │
        └────────────┼───────────────────────────────────┘
                     │
        return_sequences ─┬─ True  ► (batch, seq_len, output_dim)
                          └─ False ► (batch, output_dim)

        return_state=True ► (output, final RNN states)

    :param memory_locations: Number of memory slots (N). Becomes
        ``NTMConfig.memory_size``.
    :type memory_locations: int
    :param memory_dim: Width of each memory slot (M).
    :type memory_dim: int
    :param controller_units: Width of the controller hidden state. Becomes
        ``NTMConfig.controller_dim``.
    :type controller_units: int
    :param num_read_heads: Number of read heads. Defaults to 1.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads. Defaults to 1.
    :type num_write_heads: int
    :param controller_type: One of `'lstm'`, `'gru'`, `'feedforward'`.
        Defaults to `'lstm'`.
    :type controller_type: str
    :param shift_range: Range of allowed circular shifts. Must be odd.
        Defaults to 3.
    :type shift_range: int
    :param return_sequences: Return an output at every timestep. Defaults to True.
    :type return_sequences: bool
    :param return_state: Also return the final RNN states. Defaults to False.
    :type return_state: bool
    :param kwargs: Forwarded to `NeuralTuringMachine.__init__`, for example
        `name`, `kernel_initializer` or `kernel_regularizer`.
    :type kwargs: Any
    :return: Configured `NeuralTuringMachine`, called on `(batch, seq_len, input_dim)`.
    :rtype: NeuralTuringMachine
    :raises ValueError: Propagated from `NTMConfig.__post_init__` when a size is
        not positive or `shift_range` is invalid.

    Example:
        >>> mann = create_mann(memory_locations=128, memory_dim=64,
        ...                    controller_units=256, num_read_heads=2)
        >>> mann.output_dim
        384
    """
    output_dim = controller_units + num_read_heads * memory_dim
    config = NTMConfig(
        memory_size=memory_locations,
        memory_dim=memory_dim,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
        controller_dim=controller_units,
        controller_type=controller_type,
        shift_range=shift_range,
    )
    return NeuralTuringMachine(
        config,
        output_dim=output_dim,
        return_sequences=return_sequences,
        return_state=return_state,
        **kwargs,
    )


def create_som_2d(
    map_size: tuple[int, int],
    input_dim: int,
    **kwargs: Any,
) -> SOM2dLayer:
    """
    Build a 2D Self-Organizing Map layer.

    A thin wrapper around `SOM2dLayer`, present so the SOM family is built the
    same way as the NTM family. Every keyword argument is forwarded unchanged.
    `SOM2dLayer` passes `map_size` to `SOMLayer` as `grid_shape`, so the neuron
    weight tensor has shape `(H, W, input_dim)`.

    **Architecture Overview:**

    .. code-block:: text

        map_size = (H, W), input_dim, **kwargs
                    │
                    ▼
        ┌────────────────────────────────────────────┐
        │ SOM2dLayer (returned layer)                │
        │   grid_shape = map_size                    │
        │                                            │
        │  x (batch, input_dim)                      │
        │            │                               │
        │            ▼                               │
        │  squared distance to every neuron          │
        │  weights_map (H, W, input_dim)             │
        │            │  (batch, H*W)                 │
        │            ▼                               │
        │  BMU = argmin over neurons                 │
        │            │                               │
        │            ├──────────────┐                │
        │            ▼              ▼                │
        │  neighborhood h      quantization error    │
        │  (training only)     sqrt(min distance)    │
        │            │              │                │
        │            ▼              │                │
        │  weights_map += alpha*h*(x - w)            │
        │                           │                │
        └───────────────────────────┼────────────────┘
                                    │
        returns ─┬─ bmu_coords (batch, 2), int32
                 └─ quantization_errors (batch,)

    :param map_size: Shape of the 2D grid `(H, W)`. Exactly 2 positive integers.
    :type map_size: tuple[int, int]
    :param input_dim: Width of each input vector.
    :type input_dim: int
    :param kwargs: Forwarded to `SOM2dLayer.__init__`: `initial_learning_rate`,
        `decay_function`, `sigma`, `neighborhood_function`,
        `weights_initializer`, `regularizer`, `name`.
    :type kwargs: Any
    :return: Configured `SOM2dLayer`, called on `(batch, input_dim)`.
    :rtype: SOM2dLayer
    :raises ValueError: From `SOM2dLayer.__init__` when `map_size` is not 2
        positive integers.
    """
    return SOM2dLayer(map_size=map_size, input_dim=input_dim, **kwargs)


__all__ = [
    "create_mann",
    "create_ntm",
    "create_som_2d",
]
