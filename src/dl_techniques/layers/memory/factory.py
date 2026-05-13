"""
Factory functions for the `dl_techniques.layers.memory` package.

This module is the recommended entry point for constructing memory-augmented
layers via configuration rather than direct class instantiation. It exposes a
single, consistent function-call surface across the three families:

* `create_ntm(...)` — Neural Turing Machine (wraps `NTMCell` inside `keras.layers.RNN`).
* `create_mann(...)` — Memory-Augmented Neural Network configured as an NTM
  (uses the NTM RNN-cell pipeline with MANN-equivalent knobs).
* `create_som_2d(...)` — 2D Self-Organizing Map.

All three return fully-built Keras layers ready for use in `keras.Sequential`
or the functional API.

# DECISION plan_2026-05-13_8c1dc6fd/D-002
`create_mann` returns a `NeuralTuringMachine` (not a `MannLayer`). Rationale
(see plan decisions.md D-002): the standalone `MannLayer` class duplicates the
NTM addressing/read/write logic that `NTMCell` already implements, and the
NTM RNN-cell wrapper subsumes the LSTMCell-based rewrite originally proposed
as R4. Output shape is preserved (`controller_units + num_read_heads * memory_dim`)
so callers migrating from `MannLayer(...)` to `create_mann(...)` see no
shape change, only an implementation swap.

Note: `MannLayer` (the legacy class) is intentionally NOT deprecated here —
existing callers (`src/dl_techniques/models/qwen/qwen3_mega.py`) continue to
use it directly. New code should prefer `create_mann(...)`.
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
    Construct a Memory-Augmented Neural Network as a configured `NeuralTuringMachine`.

    The output dimensionality is set to
    ``output_dim = controller_units + num_read_heads * memory_dim`` to preserve
    the historical shape contract of the legacy `MannLayer` class. The internal
    architecture is the standard NTM (NTMMemory + NTMReadHead + NTMWriteHead +
    NTMController wrapped in keras.layers.RNN(NTMCell)).

    :param memory_locations: Number of memory slots (N) in the external memory matrix.
    :type memory_locations: int
    :param memory_dim: Dimension of each memory slot (M).
    :type memory_dim: int
    :param controller_units: Dimension of the controller hidden state.
    :type controller_units: int
    :param num_read_heads: Number of read heads. Defaults to 1.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads. Defaults to 1.
    :type num_write_heads: int
    :param controller_type: One of `'lstm'`, `'gru'`, `'feedforward'`. Defaults to `'lstm'`.
    :type controller_type: str
    :param shift_range: Range of allowed circular shifts (must be odd). Defaults to 3.
    :type shift_range: int
    :param return_sequences: Whether to return outputs at every timestep. Defaults to True.
    :type return_sequences: bool
    :param return_state: Whether to also return final RNN states. Defaults to False.
    :type return_state: bool
    :param kwargs: Forwarded to `NeuralTuringMachine.__init__` (e.g. `name`).
    :type kwargs: Any
    :return: Configured `NeuralTuringMachine` ready to be called on `(batch, seq_len, input_dim)`.
    :rtype: NeuralTuringMachine
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
    Construct a 2D Self-Organizing Map layer.

    Thin factory around `SOM2dLayer` for API uniformity with `create_ntm` /
    `create_mann`. All keyword arguments are forwarded.

    :param map_size: Shape of the 2D grid `(H, W)`. Must be exactly 2 positive integers.
    :type map_size: tuple[int, int]
    :param input_dim: Dimensionality of the input data vectors.
    :type input_dim: int
    :param kwargs: Forwarded to `SOM2dLayer.__init__`
        (`initial_learning_rate`, `decay_function`, `sigma`,
        `neighborhood_function`, `weights_initializer`, `regularizer`, `name`).
    :type kwargs: Any
    :return: Configured `SOM2dLayer` ready to be called on `(batch, input_dim)`.
    :rtype: SOM2dLayer
    """
    return SOM2dLayer(map_size=map_size, input_dim=input_dim, **kwargs)


__all__ = [
    "create_mann",
    "create_ntm",
    "create_som_2d",
]
