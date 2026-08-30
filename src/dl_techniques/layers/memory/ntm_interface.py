"""
Neural Turing Machine (NTM) interface.

This module holds the shared vocabulary of the NTM family: two enums, four
state and configuration dataclasses, four abstract base classes, and three
pure tensor helpers used by the addressing chain.

Nothing here runs an NTM. Every abstract method declared here is implemented
in `baseline_ntm.py`; this module fixes the shapes and the call order that the
implementations must agree on.

Symbols used throughout the docstrings in this file:

    N  = memory_size, the number of memory slots
    M  = memory_dim, the width of one slot
    S  = shift_range, the width of the circular-shift distribution
    H  = the number of heads

A HYBRID head runs the four-stage addressing chain of Graves et al. 2014:

    content      w_c = softmax(beta * cosine_similarity(key, memory))
    interpolate  w_g = g * w_c + (1 - g) * w_prev
    shift        w~  = circular_convolution(w_g, s)
    sharpen      w   = w~^gamma, renormalized

A CONTENT head stops after the first stage and never builds the projections
that produce `g`, `s` or `gamma`.

Contents:
    `AddressingMode` -- enum.
    `MemoryState`, `HeadState`, `NTMOutput`, `NTMConfig` -- dataclasses.
    `BaseMemory`, `BaseHead`, `BaseController`, `BaseNTM` -- abstract bases.
    `cosine_similarity`, `circular_convolution`, `sharpen_weights` -- pure
    tensor helpers, used by the concrete heads in `baseline_ntm.py`.

References:
    [1] Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines.
        arXiv:1410.5401.
"""

import keras
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------


class AddressingMode(Enum):
    """
    Which addressing mechanism a head runs.

    The value picks how many stages of the Graves chain a head executes, and
    therefore how many projections it builds. It is stored on `NTMConfig` and
    passed to `BaseHead`.

    Only the two values actually consumed anywhere are kept. Four speculative
    members (LOCATION-only, SPARSE, TEMPORAL, LEARNED) were removed in
    plan_2026-05-13_8c1dc6fd step 9 (R12) because they had no call site in
    `src/` or `tests/`.

    **Architecture Overview:**

    .. code-block:: text

        member   what selects it          what it changes
        ───────  ───────────────────────  ───────────────────────
        CONTENT  addressing_mode=CONTENT  the content weights ARE
                 on NTMConfig or a head   the final weights; no
                                          gate/shift/gamma exists
        HYBRID   the default on both      content ──► interpolate
                                          ──► circular shift ──►
                                          sharpen

    :cvar CONTENT: Content-only addressing. The content weights are returned
        as the final weights.
    :cvar HYBRID: Content plus location addressing, the original NTM. This is
        the default everywhere in this package.
    """

    CONTENT = auto()
    HYBRID = auto()


# ---------------------------------------------------------------------
# State Dataclasses
# ---------------------------------------------------------------------


@dataclass
class MemoryState:
    """
    The external memory matrix and everything carried alongside it.

    One instance describes memory at one point in a sequence. `memory` is the
    only required field; the rest exist so richer variants (usage tracking)
    can carry their own state through the same object.

    **Architecture Overview:**

    .. code-block:: text

        N = num_slots, M = memory_dim, H = num_heads
        ┌────────────────┬──────────────────────────────┐
        │ memory         │ (batch, N, M)       required │
        │ usage          │ (batch, N)          or None  │
        │ write_weights  │ (batch, H, N)       or None  │
        │ read_weights   │ (batch, H, N)       or None  │
        │ metadata       │ dict, default {}             │
        └────────────────┴──────────────────────────────┘
                 │
                 ▼  clone()
                 same tensor objects, a NEW metadata dict

    :ivar memory: Memory matrix of shape (batch, N, M).
    :vartype memory: Any
    :ivar usage: Per-slot usage vector of shape (batch, N), or None.
    :vartype usage: Any | None
    :ivar write_weights: Most recent write weights of shape (batch, H, N),
        or None.
    :vartype write_weights: Any | None
    :ivar read_weights: Most recent read weights of shape (batch, H, N),
        or None.
    :vartype read_weights: Any | None
    :ivar metadata: Free dictionary for variant-specific state. Defaults to an
        empty dict.
    :vartype metadata: dict[str, Any]
    """

    memory: Any
    usage: Any | None = None
    write_weights: Any | None = None
    read_weights: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "MemoryState":
        """
        Copy the state shallowly.

        Every tensor field is carried over by reference, so the copy shares
        the same tensor objects. Only `metadata` is rebuilt, as a new dict, so
        mutating the copy's metadata does not touch the original's.

        :return: A new MemoryState sharing this one's tensors.
        :rtype: MemoryState
        """
        return MemoryState(
            memory=self.memory,
            usage=self.usage,
            write_weights=self.write_weights,
            read_weights=self.read_weights,
            metadata=dict(self.metadata),
        )


@dataclass
class HeadState:
    """
    One read or write head's state after an addressing step.

    `weights` is always set. Which of the rest are populated depends on the
    head's `AddressingMode` and on whether it reads or writes: a CONTENT head
    fills `key` and `beta` only, a HYBRID head also fills `gate`, `shift` and
    `gamma`, and a write head additionally fills `erase_vector` and
    `add_vector`.

    The read result does not travel here; it travels in
    `NTMOutput.read_vectors`.

    **Architecture Overview:**

    .. code-block:: text

        N = num_slots, M = memory_dim
        ┌──────────────┬───────────────────────────────────┐
        │ weights      │ (batch, N)            always set  │
        │ key          │ (batch, M)            content     │
        │ beta         │ (batch, 1)            content     │
        │ gate         │ (batch, 1)            HYBRID only │
        │ shift        │ (batch, shift_range)  HYBRID only │
        │ gamma        │ (batch, 1)            HYBRID only │
        │ erase_vector │ (batch, M)            write heads │
        │ add_vector   │ (batch, M)            write heads │
        │ metadata     │ dict, default {}                  │
        └──────────────┴───────────────────────────────────┘

    :ivar weights: Attention weights over memory slots, shape (batch, N).
    :vartype weights: Any
    :ivar key: Content-addressing key, shape (batch, M), or None.
    :vartype key: Any | None
    :ivar beta: Key strength, shape (batch, 1), or None.
    :vartype beta: Any | None
    :ivar gate: Interpolation gate between the content weights and the
        previous weights, shape (batch, 1). HYBRID heads only.
    :vartype gate: Any | None
    :ivar shift: Shift distribution, shape (batch, shift_range). HYBRID heads
        only.
    :vartype shift: Any | None
    :ivar gamma: Sharpening exponent, shape (batch, 1). HYBRID heads only.
    :vartype gamma: Any | None
    :ivar erase_vector: Erase vector, shape (batch, M). Write heads only.
    :vartype erase_vector: Any | None
    :ivar add_vector: Add vector, shape (batch, M). Write heads only.
    :vartype add_vector: Any | None
    :ivar metadata: Free dictionary for variant-specific state. Defaults to an
        empty dict.
    :vartype metadata: dict[str, Any]
    """

    weights: Any
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
    Everything one NTM timestep produces.

    `BaseNTM.step()` returns this. The first four fields are required: the
    output tensor plus the three pieces of state the next timestep needs.
    `BaseNTM.call()` keeps `output` and threads `memory_state`, `head_states`
    and `controller_state` into the next `step()`.

    **Architecture Overview:**

    .. code-block:: text

        one timestep, returned by BaseNTM.step()
        ┌───────────────────┬───────────────────────────────┐
        │ output            │ (batch, output_dim)           │
        │ memory_state      │ MemoryState after the step    │
        │ head_states       │ list[HeadState], one per head │
        │ read_vectors      │ the read-head outputs         │
        │ controller_state  │ recurrent state, or None      │
        │ attention_weights │ dict for analysis, or None    │
        │ auxiliary_losses  │ dict, or None                 │
        │ metadata          │ dict, default {}              │
        └───────────────────┴───────────────────────────────┘

    :ivar output: Network output for this timestep, shape (batch, output_dim).
    :vartype output: Any
    :ivar memory_state: Memory state after the step.
    :vartype memory_state: MemoryState
    :ivar head_states: Head states after the step, one per head.
    :vartype head_states: list[HeadState]
    :ivar read_vectors: What the read heads returned this timestep.
    :vartype read_vectors: Any
    :ivar controller_state: Controller hidden state for recurrent
        controllers, or None for feedforward ones.
    :vartype controller_state: Any | None
    :ivar attention_weights: Attention weights kept for analysis or
        visualization, or None.
    :vartype attention_weights: dict[str, Any] | None
    :ivar auxiliary_losses: Extra losses such as regularization terms, or
        None.
    :vartype auxiliary_losses: dict[str, Any] | None
    :ivar metadata: Free dictionary for anything else. Defaults to an empty
        dict.
    :vartype metadata: dict[str, Any]
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
    Every knob an NTM variant is built from.

    This is a plain dataclass, so it validates in `__post_init__` rather than
    in a builder. `to_dict()` and `from_dict()` are the serialization pair;
    they are what `BaseNTM.get_config()` and `from_config()` use.

    **Architecture Overview:**

    .. code-block:: text

        field             default  constraint
        ────────────────  ───────  ────────────────────────────
        memory_size       128      N, slot count; > 0
        memory_dim        64       M, slot width; > 0
        num_read_heads    1        > 0
        num_write_heads   1        > 0
        controller_dim    256      > 0
        controller_type   'lstm'   lstm | gru | feedforward
        addressing_mode   HYBRID   CONTENT or HYBRID
        shift_range       3        positive ODD integer
        use_memory_init   True     learn the initial memory
        memory_init_seed  42       used when the above is False
        epsilon           1e-6     numerical stability

        __post_init__ raises ValueError on a violated constraint.
        to_dict()   ──► dict, addressing_mode stored as its .name
        from_dict() ──► NTMConfig, drops a legacy 'clip_value' key

    Example:
        >>> cfg = NTMConfig(memory_size=128, memory_dim=20, shift_range=3)
        >>> round_tripped = NTMConfig.from_dict(cfg.to_dict())
        >>> round_tripped == cfg
        True

    :ivar memory_size: Number of memory slots, N. Must be positive.
    :vartype memory_size: int
    :ivar memory_dim: Width of one memory slot, M. Must be positive.
    :vartype memory_dim: int
    :ivar num_read_heads: Number of read heads. Must be positive.
    :vartype num_read_heads: int
    :ivar num_write_heads: Number of write heads. Must be positive.
    :vartype num_write_heads: int
    :ivar controller_dim: Width of the controller hidden state. Must be
        positive.
    :vartype controller_dim: int
    :ivar controller_type: One of `'lstm'`, `'gru'`, `'feedforward'`.
    :vartype controller_type: str
    :ivar addressing_mode: `AddressingMode.HYBRID`, the default, runs the full
        chain: content, interpolation, circular shift, sharpening.
        `AddressingMode.CONTENT` returns the content weights directly and
        never creates the gate, shift and gamma projections, so a CONTENT head
        has strictly fewer parameters than a HYBRID one.
    :vartype addressing_mode: AddressingMode
    :ivar shift_range: Width of the circular-shift distribution, S. Must be a
        positive odd integer, so that the shift offsets are symmetric about 0.
    :vartype shift_range: int
    :ivar use_memory_init: Whether the initial memory is a learned variable.
    :vartype use_memory_init: bool
    :ivar memory_init_seed: Seed for the symmetry-breaking random initial
        memory used when `use_memory_init` is False. It is a fixed, stateless
        seed, so repeated `predict` calls on one model return the same values.
        See the `D-058` anchor in `baseline_ntm.py`.
    :vartype memory_init_seed: int
    :ivar epsilon: Small constant added inside square roots and denominators.
    :vartype epsilon: float
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
    memory_init_seed: int = 42
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        """
        Reject any field that violates its constraint.

        Runs automatically after dataclass construction.

        :raises ValueError: If any size is not positive, if `controller_type`
            is not one of the three accepted strings, or if `shift_range` is
            not a positive odd integer.
        """
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
        Flatten the configuration for serialization.

        `addressing_mode` is stored as its member name, a string, so the
        result is JSON-serializable. `from_dict()` converts it back.

        :return: One key per field, all JSON-serializable.
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
            "memory_init_seed": self.memory_init_seed,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "NTMConfig":
        """
        Rebuild a configuration from `to_dict()` output.

        The input is copied, so the caller's dictionary is not mutated. A
        string `addressing_mode` is converted back to the enum member. A
        legacy `clip_value` key is dropped with a warning; any other unknown
        key still raises, which is what the anchor below is about.

        :param config_dict: Output of `to_dict()`, or an older dictionary that
            still carries `clip_value`.
        :type config_dict: dict[str, Any]
        :return: The reconstructed configuration.
        :rtype: NTMConfig
        :raises TypeError: If the dictionary carries any key that is not a
            field of this dataclass, other than `clip_value`.
        :raises ValueError: Propagated from `__post_init__` when a value
            violates its constraint.
        :raises KeyError: If `addressing_mode` is a string that names no
            member of `AddressingMode`.
        """
        config = config_dict.copy()
        # DECISION plan-2026-08-03T130803-4c570ee4/D-003
        # `clip_value` was declared but never read, so it was removed; an old
        # config still carrying it would raise TypeError in `cls(**config)`.
        # Drop ONLY this named key. Do NOT generalize to a blanket unknown-key
        # filter: that swallows typos silently. See decisions.md D-003.
        if "clip_value" in config:
            config.pop("clip_value")
            logger.warning(
                "NTMConfig.from_dict: ignoring removed legacy key 'clip_value' "
                "(it never affected behaviour; use the optimizer's clipnorm instead)."
            )
        if "addressing_mode" in config and isinstance(config["addressing_mode"], str):
            config["addressing_mode"] = AddressingMode[config["addressing_mode"]]
        return cls(**config)


# ---------------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------------


class BaseMemory(keras.layers.Layer, ABC):
    """
    The contract an external memory module must satisfy.

    A subclass owns the memory matrix and the three operations on it: create
    an initial `MemoryState`, read a vector out of it under attention weights,
    and write into it with an erase-then-add update. This base class stores
    the two sizes and serializes them; it creates no weights.

    Three methods are abstract and must be implemented. `__init__` and
    `get_config` are concrete and usually only need `super()` calls.

    **Architecture Overview:**

    .. code-block:: text

        batch_size, read/write weights, erase/add vectors
                 │
                 ▼
        ┌─ abstract: a subclass MUST implement ─────────────────┐
        │ initialize_state(batch_size)                          │
        │     ──► MemoryState, memory (batch, N, M)             │
        │ read(memory_state, read_weights (batch, N))           │
        │     ──► read vector (batch, M)                        │
        │ write(memory_state, write_weights (batch, N),         │
        │       erase_vector (batch, M), add_vector (batch, M)) │
        │     ──► updated MemoryState                           │
        └───────────────────────────────────────────────────────┘

        ┌─ concrete: provided here ─────────────────────────────┐
        │ __init__(memory_size=N, memory_dim=M)                 │
        │ get_config() ──► adds memory_size, memory_dim         │
        │ from_config()  ──► drops a legacy `epsilon` key       │
        └───────────────────────────────────────────────────────┘

    :param memory_size: Number of memory slots, N.
    :type memory_size: int
    :param memory_dim: Width of one memory slot, M.
    :type memory_dim: int
    :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
    :type kwargs: Any

    :ivar memory_size: The N given to `__init__`.
    :vartype memory_size: int
    :ivar memory_dim: The M given to `__init__`.
    :vartype memory_dim: int
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        **kwargs: Any,
    ) -> None:
        """
        Store the sizes. No weights are created here.

        :param memory_size: Number of memory slots, N.
        :type memory_size: int
        :param memory_dim: Width of one memory slot, M.
        :type memory_dim: int
        :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
        :type kwargs: Any
        """
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim

    @abstractmethod
    def initialize_state(self, batch_size: int) -> MemoryState:
        """
        Build the starting memory state for a new sequence. Abstract.

        The returned state's `memory` must have shape (batch_size, N, M).

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: The initial memory state.
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
        Read one vector out of memory under attention weights. Abstract.

        :param memory_state: The current memory state.
        :type memory_state: MemoryState
        :param read_weights: Attention weights of shape (batch, N).
        :type read_weights: Any
        :return: Read vector of shape (batch, M).
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
        Erase then add, under attention weights. Abstract.

        The implementation must not mutate `memory_state`; it returns a new
        one.

        :param memory_state: The current memory state.
        :type memory_state: MemoryState
        :param write_weights: Write attention weights of shape (batch, N).
        :type write_weights: Any
        :param erase_vector: Erase vector of shape (batch, M).
        :type erase_vector: Any
        :param add_vector: Add vector of shape (batch, M).
        :type add_vector: Any
        :return: The updated memory state.
        :rtype: MemoryState
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Serialize the sizes on top of the base layer configuration.

        :return: The base configuration plus `memory_size` and `memory_dim`.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "memory_dim": self.memory_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseMemory":
        """
        Reconstruct from a configuration, tolerating one removed legacy key.

        :param config: A configuration dictionary, possibly one written by a
            version of this class that still emitted `epsilon`.
        :type config: dict[str, Any]
        :return: The reconstructed memory layer.
        :rtype: BaseMemory
        """
        config = dict(config)
        # DECISION plan-2026-08-30T120217-7f6cedd1/D-002
        # `epsilon` was stored and serialized here but read by neither `read`
        # nor `write`, so it was removed; an old config still carrying it would
        # raise in `cls(**config)`. Drop ONLY this named key. Do NOT generalize
        # to a blanket unknown-key filter: that swallows typos and turns a loud
        # constructor error into a wrong-config-that-runs. See decisions.md
        # D-002, and D-003 of plan-2026-08-03T130803-4c570ee4 for the precedent.
        if "epsilon" in config:
            config.pop("epsilon")
            logger.warning(
                "BaseMemory.from_config: ignoring removed legacy key 'epsilon' "
                "(it never affected behaviour on this class; the read/write "
                "heads carry their own, which is live)."
            )
        return cls(**config)


class BaseHead(keras.layers.Layer, ABC):
    """
    The contract a read or write head must satisfy.

    A head turns the controller output into attention weights over the memory
    slots. Two methods are abstract: `compute_addressing`, which runs the
    whole chain and returns the new weights plus a `HeadState`, and
    `content_addressing`, which is the first stage of that chain on its own.
    This base class stores the configuration and serializes it; it creates no
    weights and no projections.

    **Architecture Overview:**

    .. code-block:: text

        controller_output, memory_state, prev_weights
                 │
                 ▼
        ┌─ abstract: a subclass MUST implement ──────────────┐
        │ compute_addressing(controller_output,              │
        │                    memory_state, prev_weights)     │
        │     ──► (weights (batch, N), HeadState)            │
        │ content_addressing(key, beta, memory (batch, N, M))│
        │     ──► content weights (batch, N)                 │
        └────────────────────────────────────────────────────┘

        ┌─ concrete: provided here ──────────────────────────┐
        │ __init__(memory_size, memory_dim, addressing_mode, │
        │          shift_range, epsilon)                     │
        │ get_config() ──► adds all five of the above        │
        └────────────────────────────────────────────────────┘

    :param memory_size: Number of memory slots, N.
    :type memory_size: int
    :param memory_dim: Width of one memory slot, M.
    :type memory_dim: int
    :param addressing_mode: `AddressingMode.HYBRID`, the default, runs the
        full chain: content, interpolation, circular shift, sharpening.
        `AddressingMode.CONTENT` returns the content weights directly and
        never creates the gate, shift and gamma projections, so a CONTENT head
        has strictly fewer parameters than a HYBRID one.
    :type addressing_mode: AddressingMode
    :param shift_range: Width of the circular-shift distribution, S. Only used
        under HYBRID addressing. Defaults to 3.
    :type shift_range: int
    :param epsilon: Small constant for numerical stability. Defaults to 1e-6.
    :type epsilon: float
    :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
    :type kwargs: Any

    :ivar memory_size: The N given to `__init__`.
    :vartype memory_size: int
    :ivar memory_dim: The M given to `__init__`.
    :vartype memory_dim: int
    :ivar addressing_mode: The mode given to `__init__`.
    :vartype addressing_mode: AddressingMode
    :ivar shift_range: The S given to `__init__`.
    :vartype shift_range: int
    :ivar epsilon: The epsilon given to `__init__`.
    :vartype epsilon: float
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
        Store the configuration. No projections are created here.

        :param memory_size: Number of memory slots, N.
        :type memory_size: int
        :param memory_dim: Width of one memory slot, M.
        :type memory_dim: int
        :param addressing_mode: CONTENT or HYBRID. Defaults to HYBRID.
        :type addressing_mode: AddressingMode
        :param shift_range: Width of the circular-shift distribution, S.
            Defaults to 3.
        :type shift_range: int
        :param epsilon: Small constant for numerical stability. Defaults to
            1e-6.
        :type epsilon: float
        :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
        :type kwargs: Any
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
        Run the whole addressing chain for this head. Abstract.

        How many stages run is up to `addressing_mode`. Under CONTENT the
        content weights are the answer and `prev_weights` goes unused.

        :param controller_output: Controller output of shape
            (batch, controller_dim).
        :type controller_output: Any
        :param memory_state: The current memory state.
        :type memory_state: MemoryState
        :param prev_weights: This head's weights from the previous timestep,
            shape (batch, N).
        :type prev_weights: Any
        :return: The new weights of shape (batch, N), and the `HeadState`
            recording the projections that produced them.
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
        Compute the first stage of the chain on its own. Abstract.

        The usual implementation is
        `softmax(beta * cosine_similarity(key, memory))`.

        :param key: Key vector of shape (batch, M) or (batch, 1, M).
        :type key: Any
        :param beta: Key strength of shape (batch, 1). A larger value gives a
            sharper distribution.
        :type beta: Any
        :param memory: Memory matrix of shape (batch, N, M).
        :type memory: Any
        :return: Content weights of shape (batch, N).
        :rtype: Any
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Serialize the head configuration on top of the base layer's.

        `addressing_mode` is stored as its member name, a string.

        :return: The base configuration plus `memory_size`, `memory_dim`,
            `addressing_mode`, `shift_range` and `epsilon`.
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
    The contract a controller network must satisfy.

    The controller sees the timestep input concatenated with the read vectors
    from the previous step, and produces the vector every head projects its
    parameters from. Two methods are abstract: `initialize_state`, which
    returns the starting hidden state (or None for a feedforward controller),
    and `call`, which returns the output together with the next state.

    **Architecture Overview:**

    .. code-block:: text

        inputs (the timestep concatenated with the read vectors)
                 │
                 ▼
        ┌─ abstract: a subclass MUST implement ────────────────┐
        │ initialize_state(batch_size)                         │
        │     ──► initial state, or None for feedforward       │
        │ call(inputs, state=None, training=None)              │
        │     ──► (controller_output, new_state)               │
        └──────────────────────────────────────────────────────┘

        ┌─ concrete: provided here ────────────────────────────┐
        │ __init__(controller_dim, controller_type)            │
        │ get_config() ──► adds controller_dim, controller_type│
        └──────────────────────────────────────────────────────┘

    :param controller_dim: Width of the controller hidden state.
    :type controller_dim: int
    :param controller_type: One of `'lstm'`, `'gru'`, `'feedforward'`.
        Defaults to `'lstm'`. This base class stores the value and does not
        check it; `NTMConfig.__post_init__` is where it is validated.
    :type controller_type: str
    :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
    :type kwargs: Any

    :ivar controller_dim: The width given to `__init__`.
    :vartype controller_dim: int
    :ivar controller_type: The type string given to `__init__`.
    :vartype controller_type: str
    """

    def __init__(
        self,
        controller_dim: int,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration. No sublayers are created here.

        :param controller_dim: Width of the controller hidden state.
        :type controller_dim: int
        :param controller_type: One of `'lstm'`, `'gru'`, `'feedforward'`.
            Defaults to `'lstm'`. Not validated here.
        :type controller_type: str
        :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
        :type kwargs: Any
        """
        super().__init__(**kwargs)
        self.controller_dim = controller_dim
        self.controller_type = controller_type

    @abstractmethod
    def initialize_state(self, batch_size: int) -> Any | None:
        """
        Build the starting hidden state for a new sequence. Abstract.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: The initial state, or None for a feedforward controller.
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
        Run one timestep through the controller. Abstract.

        Unlike a stock Keras layer this returns a pair, not a tensor, so the
        caller can thread the state forward.

        :param inputs: The timestep input concatenated with the previous read
            vectors.
        :type inputs: Any
        :param state: The controller state from the previous timestep, or
            None. Defaults to None.
        :type state: Any | None
        :param training: Keras training flag. Defaults to None.
        :type training: bool | None
        :return: The controller output and the next state.
        :rtype: tuple[Any, Any | None]
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Serialize the controller configuration on top of the base layer's.

        :return: The base configuration plus `controller_dim` and
            `controller_type`.
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
    The contract a complete Neural Turing Machine must satisfy.

    A subclass owns a memory module, a controller and its read and write
    heads, and implements one timestep in `step()`. The `call()` provided here
    loops `step()` over the sequence axis, threads the three pieces of state
    from one timestep into the next, and stacks the per-timestep outputs.

    Four methods are abstract: `initialize_state`, `step`, `get_memory_state`
    and `reset_memory`. The rest -- `call`, `compute_output_shape`,
    `get_config` and `from_config` -- are concrete and work as written for any
    subclass that honours the four.

    The loop in `call()` is a plain Python `for` over `range(seq_len)`, so the
    sequence length must be known statically at trace time.

    That `call()` is a DEFAULT, not a requirement. A subclass may
    override it and never run the loop, in which case `step()` and
    `initialize_state()` exist only to satisfy this class. The shipped
    `NeuralTuringMachine` does exactly that: it wraps its cell in a
    `keras.layers.RNN` and lets the RNN do the stepping.

    **Architecture Overview:**

    .. code-block:: text

        inputs (batch, seq_len, input_dim)  -- INPUT tensor
        initial_state  (optional)
                 │
                 ▼
        ┌──────────────────────────────────────────────────────┐
        │ initialize_state(batch_size)            (abstract)   │
        │ skipped when initial_state was given    (optional)   │
        └──────────────────────────────────────────────────────┘
                 │  memory_state, head_states, controller_state
                 ▼
        ┌──────────────────────────────────────────────────────┐
        │ for t in range(seq_len):                             │
        │   step(inputs[:, t, :], the 3 states)   (abstract)   │
        │   ──► NTMOutput; its 3 states become the current     │
        │   outputs.append(NTMOutput.output)                   │
        └──────────────────────────────────────────────────────┘
                 │  outputs: seq_len tensors of (batch, out)
            ┌────┴──────────────────┐
            ▼                       ▼
            return_sequences        return_sequences
              = True                  = False
            │                       │
            ▼                       ▼
            stack(axis=1)           outputs[-1]
            (batch, seq_len, out)   (batch, out)
            └───────────┬───────────┘
                        ▼
            output; plus (memory_state, head_states,
            controller_state) when return_state=True

    Input shape:
        3D tensor of shape `(batch_size, seq_len, input_dim)`.

    Output shape:
        `(batch_size, seq_len, output_dim)` when `return_sequences` is True,
        otherwise `(batch_size, output_dim)`.

    :param config: The configuration this NTM is built from.
    :type config: NTMConfig
    :param output_dim: Width of the output vector. May be None, but then
        `compute_output_shape` cannot be used. Defaults to None.
    :type output_dim: int | None
    :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
    :type kwargs: Any

    :ivar config: The configuration given to `__init__`.
    :vartype config: NTMConfig
    :ivar output_dim: The output width given to `__init__`.
    :vartype output_dim: int | None
    :ivar memory: Set to None here; a subclass must assign its memory module.
    :vartype memory: BaseMemory | None
    :ivar controller: Set to None here; a subclass must assign its controller.
    :vartype controller: BaseController | None
    :ivar read_heads: Empty here; a subclass must fill it.
    :vartype read_heads: list[BaseHead]
    :ivar write_heads: Empty here; a subclass must fill it.
    :vartype write_heads: list[BaseHead]
    """

    def __init__(
        self,
        config: NTMConfig,
        output_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration and null out the component slots.

        `memory`, `controller`, `read_heads` and `write_heads` are set to
        None or to empty lists. A subclass must fill them.

        :param config: The configuration this NTM is built from.
        :type config: NTMConfig
        :param output_dim: Width of the output vector, or None.
        :type output_dim: int | None
        :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
        :type kwargs: Any
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
        Build the three starting states for a new sequence. Abstract.

        `call()` uses this whenever no `initial_state` is supplied.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: The memory state, one head state per head, and the
            controller state (None for a feedforward controller).
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
        Run one timestep. Abstract, and the only method `call()` needs.

        An implementation runs the controller, has each head address memory,
        reads and writes, and packs everything into an `NTMOutput`.

        :param inputs: Input at this timestep, shape (batch, input_dim).
        :type inputs: Any
        :param memory_state: The memory state entering this timestep.
        :type memory_state: MemoryState
        :param head_states: The head states entering this timestep.
        :type head_states: list[HeadState]
        :param controller_state: The controller state entering this timestep,
            or None.
        :type controller_state: Any | None
        :param training: Keras training flag. Defaults to None.
        :type training: bool | None
        :return: This timestep's output and the three updated states.
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
        Run the whole sequence, one `step()` per timestep.

        States start from `initialize_state(batch_size)` unless
        `initial_state` is given. Each `step()` returns an `NTMOutput`; its
        output is collected and its three states replace the current ones.

        :param inputs: Input sequence of shape (batch, seq_len, input_dim).
        :type inputs: Any
        :param initial_state: Optional tuple of initial states.
        :type initial_state: tuple[MemoryState, list[HeadState], Any] | None
        :param training: Whether in training mode.
        :type training: bool | None
        :param return_sequences: Whether to return outputs at all timesteps.
        :type return_sequences: bool
        :param return_state: Whether to also return the final states.
        :type return_state: bool
        :return: The output tensor: (batch, seq_len, output_dim) when
            `return_sequences` is True, otherwise (batch, output_dim). When
            `return_state` is True, a tuple of that output and the final
            (memory_state, head_states, controller_state).
        :rtype: Any | tuple[Any, ...]
        """
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]

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
            output = keras.ops.stack(outputs, axis=1)
        else:
            output = outputs[-1]

        if return_state:
            return output, (memory_state, head_states, controller_state)
        return output

    @abstractmethod
    def get_memory_state(self) -> MemoryState | None:
        """
        Return the memory state the subclass is holding. Abstract.

        :return: The current memory state, or None if nothing has run yet.
        :rtype: MemoryState | None
        """
        pass

    @abstractmethod
    def reset_memory(self, batch_size: int) -> None:
        """
        Throw the current memory away and start over. Abstract.

        :param batch_size: Number of sequences the fresh memory must cover.
        :type batch_size: int
        """
        pass

    def compute_output_shape(self, input_shape: Any) -> Any:
        """
        Report the output shape for a sequence input.

        This assumes `return_sequences=True`, which is the `call()` default.

        :param input_shape: Input shape (batch, seq_len, input_dim).
        :type input_shape: Any
        :return: (batch, seq_len, output_dim).
        :rtype: Any
        :raises ValueError: If `output_dim` was not given to `__init__`.
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
        Serialize the NTM configuration on top of the base layer's.

        The `NTMConfig` is flattened with `to_dict()`, so the result stays
        JSON-serializable. `from_config()` rebuilds it.

        :return: The base configuration plus `output_dim` and a nested
            `config` dictionary.
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
        Rebuild the layer from `get_config()` output.

        A nested `config` entry is rebuilt into an `NTMConfig` first and
        passed as the `config` keyword. A dictionary without that entry is
        forwarded unchanged.

        :param config: Configuration dictionary.
        :type config: dict[str, Any]
        :return: The reconstructed layer.
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
    Cosine similarity between one query and every key.

    This is the first stage of NTM content addressing: the concrete heads in
    `baseline_ntm.py` feed the result to `softmax(beta * similarity)`.

    `epsilon` is added inside the square root of each L2 norm, so a
    zero-length vector gives `sqrt(epsilon)` instead of dividing by zero.

    **Architecture Overview:**

    .. code-block:: text

        query (batch, dim) or (batch, 1, dim)  -- INPUT
        keys  (batch, N, dim)                  -- INPUT
                 │
                 ▼
        ┌────────────────────────────────────────────────┐
        │ expand_dims(query, axis=1) when query is rank 2│
        │ (optional)                                     │
        └────────────────────────────────────────────────┘
                 │  query (batch, 1, dim)
                 ▼
        ┌────────────────────────────────────────────────┐
        │ L2-normalize BOTH along the last axis:         │
        │ x / sqrt(sum(x^2, -1, keepdims) + epsilon)     │
        └────────────────────────────────────────────────┘
                 │  (batch, 1, dim) and (batch, N, dim)
                 ▼
        ┌────────────────────────────────────────────────┐
        │ sum(q_hat * k_hat, axis=-1)   broadcasts over N│
        └────────────────────────────────────────────────┘
                 │  (batch, N)
                 ▼
        ┌────────────────────────────────────────────────┐
        │ squeeze(axis=1) when N == 1         (optional) │
        └────────────────────────────────────────────────┘
                 │
                 ▼
        similarity (batch, N); (batch,) when N == 1

    Note:
        The trailing squeeze fires when the result has shape (batch, 1), which
        means N == 1. It does NOT undo the rank-2 query expansion. With a
        single memory slot the return is (batch,) whatever rank the query had.

    :param query: Query tensor of shape (batch, dim) or (batch, 1, dim). A
        rank-2 query is expanded to rank 3 before use.
    :type query: Any
    :param keys: Keys tensor of shape (batch, N, dim).
    :type keys: Any
    :param epsilon: Small constant added inside each L2 norm. Defaults to
        1e-6.
    :type epsilon: float
    :return: Similarity of shape (batch, N), or (batch,) when N == 1.
    :rtype: Any
    """
    # Ensure query has 3 dimensions for broadcasting
    if len(keras.ops.shape(query)) == 2:
        query = keras.ops.expand_dims(query, axis=1)

    # Normalize query and keys
    query_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(query), axis=-1, keepdims=True) + epsilon)
    keys_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(keys), axis=-1, keepdims=True) + epsilon)

    query_normalized = query / query_norm
    keys_normalized = keys / keys_norm

    # Compute similarity
    similarity = keras.ops.sum(query_normalized * keys_normalized, axis=-1)

    # Squeeze the slot axis when there is exactly one memory slot.
    if len(keras.ops.shape(similarity)) == 2 and keras.ops.shape(similarity)[1] == 1:
        similarity = keras.ops.squeeze(similarity, axis=1)

    return similarity


def circular_convolution(
    weights: Any,
    shift: Any,
) -> Any:
    """
    Circular convolution of the weights with a shift distribution.

    This is the third stage of HYBRID addressing. It builds one rolled copy of
    `weights` per shift offset, stacks them, and takes the weighted sum under
    `shift`. The offsets run from `-S // 2` to `+S // 2`, so a `shift_range`
    of 3 gives offsets -1, 0, +1.

    The rolls are built with `keras.ops.roll`, `stack` and `sum` rather than a
    gather loop, so the whole operation stays one graph-friendly expression.

    **Architecture Overview:**

    .. code-block:: text

        weights (batch, N)                   shift (batch, S)
        both are INPUT tensors                       │
                 │                                   │ half_shift
                 │                                   │  = S // 2
                 ▼                                   │
        ┌───────────────────────────────────────┐    │
        │ for i in range(S):                    │    │
        │   offset = i - half_shift             │◄───┤
        │   roll(weights, shift=offset, axis=-1)│    │
        └───────────────────────────────────────┘    │
                 │  S rolled tensors of (batch, N)   │
                 ▼                                   ▼
        ┌─────────────────────────────────────────────────┐
        │ stack(axis=1)                      (batch, S, N)│
        │ multiply by expand_dims(shift, -1) (batch, S, 1)│
        │ sum(axis=1)                        (batch, N)   │
        └─────────────────────────────────────────────────┘
                 │
                 ▼
        shifted weights (batch, N)

    Note:
        The Python `for` iterates over `shift.shape[-1]`, the STATIC last
        dimension, so `shift` must have a known width at trace time. The
        `half_shift` offset is derived from `keras.ops.shape(shift)[-1]`.

    :param weights: Attention weights of shape (batch, N).
    :type weights: Any
    :param shift: Shift distribution of shape (batch, S), normally the output
        of a softmax so that it sums to 1 along the last axis.
    :type shift: Any
    :return: Shifted weights of shape (batch, N).
    :rtype: Any
    """
    shift_range = keras.ops.shape(shift)[-1]
    half_shift = shift_range // 2

    # Build all shifted versions and stack them
    shifted_versions = []

    # The loop bound is the STATIC last dimension, so `shift` must have a known
    # width at trace time. Standard NTM usage fixes it at construction.
    for i in range(shift.shape[-1]):
        shift_offset = i - half_shift
        # DECISION plan-2026-08-03T130803-4c570ee4/D-001
        # Graves et al. 2014 eq. 8: w~(i) = sum_j w(j) * s(i - j mod N).
        # keras.ops.roll(a, k)[i] == a[(i - k) mod N], so the tap carrying offset
        # k is roll(w, +k). Do NOT "simplify" this back to shift=-shift_offset:
        # that mirrors the shift (offset +1 lands at slot N-1). See decisions.md D-001.
        rolled = keras.ops.roll(weights, shift=shift_offset, axis=-1)
        shifted_versions.append(rolled)

    # Stack: (batch, num_shifts, memory_size)
    stacked = keras.ops.stack(shifted_versions, axis=1)

    # Weight by shift probabilities: (batch, num_shifts, 1)
    shift_weights = keras.ops.expand_dims(shift, axis=-1)

    # Weighted sum: (batch, memory_size)
    shifted_weights = keras.ops.sum(stacked * shift_weights, axis=1)

    return shifted_weights


def sharpen_weights(
    weights: Any,
    gamma: Any,
    epsilon: float = 1e-6,
) -> Any:
    """
    Raise the weights to a power and renormalize.

    This is the last stage of HYBRID addressing. A `gamma` above 1 pushes mass
    toward the largest weights; `gamma == 1` leaves the distribution alone
    apart from the epsilon terms. `gamma` is clamped up to 1.0 first, so a
    smaller value never flattens the distribution.

    **Architecture Overview:**

    .. code-block:: text

        weights (batch, N)          gamma (batch, 1)
        both are INPUT tensors              │
                 │                          │ maximum(gamma, 1.0)
                 │                          │
                 └────────────┬─────────────┘
                              ▼
        ┌────────────────────────────────────────────────┐
        │ power(weights + epsilon, gamma)                │
        └────────────────────────────────────────────────┘
                 │  (batch, N)
                 ▼
        ┌──────────────────────────────────────────────────┐
        │ divide by (sum(axis=-1, keepdims=True) + epsilon)│
        └──────────────────────────────────────────────────┘
                 │
                 ▼
        sharpened weights (batch, N)

    Note:
        Both epsilon terms are kept, so the result sums to slightly less than
        1 rather than exactly 1.

    :param weights: Attention weights of shape (batch, N).
    :type weights: Any
    :param gamma: Sharpening exponent of shape (batch, 1). Values below 1.0
        are clamped to 1.0.
    :type gamma: Any
    :param epsilon: Small constant added to the base and to the denominator.
        Defaults to 1e-6.
    :type epsilon: float
    :return: Sharpened weights of shape (batch, N).
    :rtype: Any
    """
    # Ensure gamma >= 1
    gamma = keras.ops.maximum(gamma, 1.0)

    # Raise to power and normalize
    sharpened = keras.ops.power(weights + epsilon, gamma)
    return sharpened / (keras.ops.sum(sharpened, axis=-1, keepdims=True) + epsilon)

# ---------------------------------------------------------------------
