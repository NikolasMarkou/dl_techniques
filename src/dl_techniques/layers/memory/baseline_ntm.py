"""
Neural Turing Machine, after Graves et al., 2014.

A Neural Turing Machine is a controller network coupled to an external
memory matrix. Read and write heads address that memory by content and
by location. Every addressing step is differentiable, so the whole
thing trains by ordinary backpropagation.

This module holds the concrete Keras 3 implementation. The contracts it
implements live in `ntm_interface.py`: the abstract base classes, the
`NTMConfig` dataclass, the state records, and the three addressing
helpers (`cosine_similarity`, `circular_convolution`, `sharpen_weights`).

Classes:
    NTMMemory: The memory matrix, with read and erase-then-add write.
    NTMReadHead: Read head, content or content-plus-location addressing.
    NTMWriteHead: Write head, the same addressing plus erase and add.
    NTMController: LSTM, GRU or feedforward controller.
    NTMCell: One timestep, shaped for `keras.layers.RNN`.
    NeuralTuringMachine: The full layer, an RNN wrapped around NTMCell.

Functions:
    create_ntm: Build a NeuralTuringMachine from plain arguments.

Throughout, `N` is the number of memory slots (`memory_size`) and `M` is
the width of one slot (`memory_dim`), matching `ntm_interface.py`.
"""

import keras
from typing import Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer

from .ntm_interface import (
    AddressingMode,
    BaseMemory,
    BaseHead,
    BaseController,
    BaseNTM,
    MemoryState,
    HeadState,
    NTMConfig,
    NTMOutput,
    cosine_similarity,
    circular_convolution,
    sharpen_weights,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# NTMMemory
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.baseline_ntm")
class NTMMemory(BaseMemory):
    """
    The NTM memory matrix, and the read and write operations on it.

    The memory is a STATE tensor of shape `(batch, N, M)`, not a
    weight. This layer creates no weights of its own. It reads that
    tensor and returns a new one.

    A read is a weighted sum over slots. A write is erase-then-add:
    `M' = M * (1 - w e) + w a`, with the write weights `w` broadcast
    over slot width and the erase and add vectors broadcast over
    slots.

    **Architecture Overview:**

    .. code-block:: text

        memory_state.memory, M (batch, N, M) -- a STATE
        tensor. This layer creates NO weights.

        ┌─ read(memory_state, read_weights) ───────────────┐
        │ read_weights w (batch, N)                        │
        │ expand_dims(w, -1)  ──►  (batch, N, 1)           │
        │ sum(M * w, axis=1)                               │
        └───────────────────┬──────────────────────────────┘
                            ▼
        read vector (batch, M)

        ┌─ write(memory_state, w, erase, add) ─────────────┐
        │ w (batch, N, 1);  erase e, add a (batch, 1, M)   │
        │ erase:  M * (1 - w * e)                          │
        │ add:    ... + w * a                              │
        └───────────────────┬──────────────────────────────┘
                            ▼
        MemoryState(memory=M', usage carried through unchanged)

    Note:
        This class takes no `epsilon`. One was declared, stored and
        serialized here but read by neither `read` nor `write`; it was
        removed, and `BaseMemory.from_config` drops the legacy key. The
        heads carry their own `epsilon`, which does reach code.

    :param memory_size: Number of memory slots, N. Must be positive.
    :type memory_size: int
    :param memory_dim: Width of one memory slot, M. Must be positive.
    :type memory_dim: int
    :param memory_init_seed: Seed for the symmetry-breaking draw in
        `initialize_state`. It is fixed, so that draw is stateless and
        repeats exactly. Defaults to 42. Serialized by this class's
        `get_config`, so a `from_config` round-trip preserves it.
        `NTMCell` passes its own `NTMConfig.memory_init_seed` down here,
        so the two seed paths agree.
    :type memory_init_seed: int
    :param kwargs: Forwarded to `BaseMemory.__init__`.
    :type kwargs: Any

    :ivar memory_init_seed: The seed given to `__init__`.
    :vartype memory_init_seed: int
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        memory_init_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Store the two sizes and the init seed.

        `BaseMemory.__init__` stores the sizes; `memory_init_seed` is
        stored here and is the only field this subclass adds.

        :param memory_size: Number of memory slots, N.
        :type memory_size: int
        :param memory_dim: Width of one memory slot, M.
        :type memory_dim: int
        :param memory_init_seed: Seed for the draw in
            `initialize_state`. Defaults to 42.
        :type memory_init_seed: int
        :param kwargs: Forwarded to `BaseMemory.__init__`.
        :type kwargs: Any
        """
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            **kwargs,
        )
        self.memory_init_seed = memory_init_seed

    def initialize_state(self, batch_size: int) -> MemoryState:
        """
        Build a starting memory state for a new sequence.

        Fills memory with a small random draw (stddev 1e-3) so the
        slots are not identical. Content addressing over identical
        slots gives one flat softmax and no useful gradient, so the
        draw is what makes the addressing mechanism trainable at
        step one. Usage is all zeros.

        `NTMCell` does NOT call this. The cell draws its own initial
        memory in `get_initial_state`, using the seed from its
        `NTMConfig`. This method is for a caller driving `BaseMemory`
        directly.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: Memory of shape (batch, N, M) and zero usage of
            shape (batch, N).
        :rtype: MemoryState
        """
        # See the D-058 anchor on `NTMCell.get_initial_state` for why this seed
        # is fixed rather than absent.
        memory = keras.random.normal(
            (batch_size, self.memory_size, self.memory_dim),
            mean=0.0,
            stddev=1e-3,
            seed=self.memory_init_seed,
        )
        usage = keras.ops.zeros((batch_size, self.memory_size))
        return MemoryState(memory=memory, usage=usage)

    def read(
        self,
        memory_state: MemoryState,
        read_weights: Any,
    ) -> Any:
        """
        Read one vector per batch element, by weighted sum over slots.

        The weights are broadcast over slot width and the slot axis
        is summed away.

        :param memory_state: The state holding the memory to read.
        :type memory_state: MemoryState
        :param read_weights: A distribution over slots, shape
            (batch, N). Nothing here requires it to be normalized.
        :type read_weights: Any
        :return: Read vector of shape (batch, M).
        :rtype: Any
        """
        weights_expanded = keras.ops.expand_dims(read_weights, axis=-1)
        read_vector = keras.ops.sum(memory_state.memory * weights_expanded, axis=1)
        return read_vector

    def write(
        self,
        memory_state: MemoryState,
        write_weights: Any,
        erase_vector: Any,
        add_vector: Any,
    ) -> MemoryState:
        """
        Write to memory, erase first and then add.

        `M' = M * (1 - w e) + w a`. The write weights are broadcast
        over slot width and the erase and add vectors over slots, so
        each slot is modified in proportion to its weight. A slot
        with weight 0 is left exactly as it was.

        `usage` is carried through unchanged; this class does not
        maintain it.

        :param memory_state: The state holding the memory to write to.
        :type memory_state: MemoryState
        :param write_weights: A distribution over slots, shape
            (batch, N).
        :type write_weights: Any
        :param erase_vector: Erase vector of shape (batch, M). The
            write head produces it through a sigmoid, so it is in
            [0, 1]; nothing here enforces that.
        :type erase_vector: Any
        :param add_vector: Add vector of shape (batch, M).
        :type add_vector: Any
        :return: A new state carrying the updated memory.
        :rtype: MemoryState
        """
        prev_memory = memory_state.memory

        # Expand dims for broadcasting
        weights_expanded = keras.ops.expand_dims(write_weights, axis=-1)
        erase_expanded = keras.ops.expand_dims(erase_vector, axis=1)
        add_expanded = keras.ops.expand_dims(add_vector, axis=1)

        # Erase: M_t = M_{t-1} * (1 - w_t * e_t)
        erase_matrix = 1.0 - (weights_expanded * erase_expanded)
        erased_memory = prev_memory * erase_matrix

        # Add: M_t = M_{t-1} + w_t * a_t
        add_matrix = weights_expanded * add_expanded
        new_memory = erased_memory + add_matrix

        return MemoryState(
            memory=new_memory,
            usage=memory_state.usage,
        )

    def get_config(self) -> dict[str, Any]:
        """
        Return the constructor arguments, for serialization.

        `BaseMemory.get_config` emits `memory_size` and `memory_dim`;
        this override adds `memory_init_seed`, the one field this
        subclass owns, so a `from_config` round-trip reproduces the seed
        rather than resetting it to the default 42.

        :return: The configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({"memory_init_seed": self.memory_init_seed})
        return config


# ---------------------------------------------------------------------
# NTMReadHead
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.baseline_ntm")
class NTMReadHead(BaseHead):
    """
    NTM read head: turns the controller output into read weights.

    The head projects the controller output to addressing parameters,
    then computes a distribution over the N memory slots. `NTMCell`
    hands those weights to `NTMMemory.read`.

    Which projections exist depends on `addressing_mode`. Under
    `AddressingMode.HYBRID` (the default) the head runs the full NTM
    chain and owns five Dense layers. Under `AddressingMode.CONTENT`
    the content weights ARE the addressing, and the gate, shift and
    gamma projections are never created.

    On the HYBRID path `gamma_dense` is softplus and 1.0 is added to its
    output, so the exponent handed to `sharpen_weights` is always above 1
    and that helper's own `maximum(gamma, 1.0)` clamp never fires here.

    **Architecture Overview:**

    .. code-block:: text

        controller_output (batch, controller_dim) -- INPUT
                          │
                          ▼
        ┌─ Dense projections, always created ────────────┐
        │ key_dense    ──► key  (batch, M)               │
        │ beta_dense   ──► beta (batch, 1), softplus     │
        └─────────────────┬──────────────────────────────┘
                          ▼
        ┌─ content_addressing ───────────────────────────┐
        │ cosine_similarity(key, memory, epsilon)        │
        │ softmax(beta * similarity, axis=-1)            │
        └─────────────────┬──────────────────────────────┘
                          │  content_weights (batch, N)
            ┌─────────────┴──────────────────────────┐
            ▼                                        ▼
            CONTENT                                  HYBRID
            │                                        ▼
            │                           ┌─────────────────────────┐
            │                           │ gate_dense   (optional) │
            │                           │ shift_dense  (optional) │
            │                           │ gamma_dense  (optional) │
            │                           │   gamma = out + 1.0     │
            │                           └────────────┬────────────┘
            │                                        ▼
            │                    ┌──────────────────────────────────────┐
            │                    │ gate * content_weights               │
            │                    │   + (1 - gate) * prev_weights        │
            │                    │ circular_convolution(w, shift)       │
            │                    │ sharpen_weights(w, gamma, epsilon)   │
            │                    └───────────────────┬──────────────────┘
            │                                        │
            └─────────────┬──────────────────────────┘
                          ▼
        weights (batch, N), HeadState

    :param memory_size: Number of memory slots, N. Must be positive.
    :type memory_size: int
    :param memory_dim: Width of one memory slot, M. Must be positive.
    :type memory_dim: int
    :param addressing_mode: Which addressing chain to run.
        `AddressingMode.HYBRID`, the default, runs content, then
        interpolation with `prev_weights`, then circular shift, then
        sharpening. `AddressingMode.CONTENT` returns the content weights
        directly and does not create the gate / shift / gamma projections
        at all, so a CONTENT head has strictly fewer parameters.
    :type addressing_mode: AddressingMode
    :param shift_range: Width of the circular-shift distribution, S. Read
        ONLY on the HYBRID path; a CONTENT head creates no shift
        projection, so the value is silently ignored there. Defaults to 3.
    :type shift_range: int
    :param kernel_initializer: Initializer for the Dense kernels. Each
        projection gets its own clone of it. Defaults to
        `'glorot_uniform'`.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for the Dense biases. Every
        projection this head creates receives it, `key_dense` included,
        and each gets its OWN clone of it -- a single instance handed to
        all of them would draw the same bias for every projection.
        Defaults to `'zeros'`.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer shared by every Dense kernel.
        Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param epsilon: Small constant handed to the two addressing helpers.
        It is added inside each L2 norm in `cosine_similarity`, so a
        zero-length key or an all-zero memory slot gives `sqrt(epsilon)`
        rather than a divide by zero; and on the HYBRID path it is added
        to the base and the denominator in `sharpen_weights`, so an
        exactly-zero weight raised to a large gamma stays differentiable.
        Under CONTENT only the `cosine_similarity` use is reached.
        Defaults to 1e-6.
    :type epsilon: float
    :param kwargs: Forwarded to `BaseHead.__init__`.
    :type kwargs: Any

    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or None.
    :vartype kernel_regularizer: keras.regularizers.Regularizer | None
    :ivar gate_dense: Interpolation gate projection, or None under
        CONTENT.
    :vartype gate_dense: keras.layers.Dense | None
    :ivar shift_dense: Circular-shift projection, or None under CONTENT.
    :vartype shift_dense: keras.layers.Dense | None
    :ivar gamma_dense: Sharpening projection, or None under CONTENT.
    :vartype gamma_dense: keras.layers.Dense | None
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        addressing_mode: AddressingMode = AddressingMode.HYBRID,
        shift_range: int = 3,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Resolve the initializers and create the Dense projections.

        `key_dense` and `beta_dense` are always created. The three
        location-addressing projections are created only under
        `AddressingMode.HYBRID`; otherwise they are set to None.

        :param memory_size: Number of memory slots, N.
        :type memory_size: int
        :param memory_dim: Width of one memory slot, M.
        :type memory_dim: int
        :param addressing_mode: Which addressing chain to run.
            Defaults to `AddressingMode.HYBRID`.
        :type addressing_mode: AddressingMode
        :param shift_range: Width of the circular-shift
            distribution, S. Defaults to 3.
        :type shift_range: int
        :param kernel_initializer: Initializer for the Dense kernels.
            Defaults to `'glorot_uniform'`.
        :type kernel_initializer: str | keras.initializers.Initializer
        :param bias_initializer: Initializer for the Dense biases. Each
            projection gets its own clone. Defaults to `'zeros'`.
        :type bias_initializer: str | keras.initializers.Initializer
        :param kernel_regularizer: Regularizer shared by every Dense
            kernel. Defaults to None.
        :type kernel_regularizer: keras.regularizers.Regularizer | None
        :param epsilon: Small constant handed to `cosine_similarity`
            and, on the HYBRID path, to `sharpen_weights`.
            Defaults to 1e-6.
        :type epsilon: float
        :param kwargs: Forwarded to `BaseHead.__init__`.
        :type kwargs: Any
        """
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=addressing_mode,
            shift_range=shift_range,
            epsilon=epsilon,
            **kwargs,
        )

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create sub-layers in __init__ (Golden Rule)
        self.key_dense = keras.layers.Dense(
            memory_dim,
            # DECISION plan-2026-08-19T163559-499b6f0e/D-068
            # EVERY consumer clones. Do NOT hand the SAME initializer instance to all of a
            # head's projections: that made them bit-identical -- MEASURED before this change,
            # 22 identical pairs of 17 non-constant tensors, and 0 after it. `erase` and `add`
            # are OPPOSITE ops and the two heads address memory independently: see decisions.md D-068.
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="key",
        )
        self.beta_dense = keras.layers.Dense(
            1,
            activation="softplus",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="beta",
        )
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-073
        # Under CONTENT the three location-addressing projections are NOT created at all.
        # Do NOT "simplify" this by always creating them and skipping their use instead:
        # that revives the defect this closes (parameters that exist, train and serialize
        # unread) and makes a CONTENT checkpoint HYBRID-shaped. See decisions.md D-073.
        if self.addressing_mode is AddressingMode.HYBRID:
            self.gate_dense = keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="gate",
            )
            self.shift_dense = keras.layers.Dense(
                shift_range,
                activation="softmax",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="shift",
            )
            self.gamma_dense = keras.layers.Dense(
                1,
                activation="softplus",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="gamma",
            )
        else:
            self.gate_dense = None
            self.shift_dense = None
            self.gamma_dense = None

    def build(self, input_shape: tuple) -> None:
        """
        Build every Dense projection this head created.

        The three location-addressing projections are built only
        under HYBRID, because under CONTENT they do not exist.

        :param input_shape: Shape of the controller output,
            `(batch, controller_dim)`.
        :type input_shape: tuple
        """
        self.key_dense.build(input_shape)
        self.beta_dense.build(input_shape)
        if self.addressing_mode is AddressingMode.HYBRID:
            self.gate_dense.build(input_shape)
            self.shift_dense.build(input_shape)
            self.gamma_dense.build(input_shape)
        super().build(input_shape)

    def content_addressing(
        self,
        key: Any,
        beta: Any,
        memory: Any,
    ) -> Any:
        """
        Score every memory slot against the key.

        Cosine similarity between the key and each slot, scaled by
        the key strength `beta`, then softmax over slots. A large
        `beta` sharpens the distribution toward the closest slot.

        :param key: Key vector of shape (batch, 1, M).
        :type key: Any
        :param beta: Key strength of shape (batch, 1), non-negative
            because the projection is softplus.
        :type beta: Any
        :param memory: Memory matrix of shape (batch, N, M).
        :type memory: Any
        :return: Content weights of shape (batch, N), summing to 1.
        :rtype: Any
        """
        similarity = cosine_similarity(key, memory, epsilon=self.epsilon)
        return keras.ops.softmax(beta * similarity, axis=-1)

    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> tuple[Any, HeadState]:
        """
        Turn the controller output into read weights.

        Projects the key and key strength, scores the memory, and
        then either returns those content weights directly
        (CONTENT) or runs interpolation, circular shift and
        sharpening on top of them (HYBRID).

        `prev_weights` is read only on the HYBRID path.

        :param controller_output: Controller output of shape
            (batch, controller_dim).
        :type controller_output: Any
        :param memory_state: The state holding the memory to score.
        :type memory_state: MemoryState
        :param prev_weights: This head's weights from the previous
            timestep, shape (batch, N). Unused under CONTENT.
        :type prev_weights: Any
        :return: The weights of shape (batch, N), and a `HeadState`
            carrying them plus the projected parameters. Under
            CONTENT the state's `gate`, `shift` and `gamma` stay
            None.
        :rtype: tuple[Any, HeadState]
        """
        # 1. Project controller output to head parameters
        key = self.key_dense(controller_output)
        beta = self.beta_dense(controller_output)

        # 2. Content Addressing
        key_expanded = keras.ops.expand_dims(key, axis=1)
        content_weights = self.content_addressing(
            key_expanded, beta, memory_state.memory
        )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-073
        # Under CONTENT the content weights ARE the addressing: no interpolation with the
        # previous weights, no circular shift, no sharpening. `prev_weights` is therefore
        # not read in this mode, which is exactly what "content-only addressing" means.
        # See decisions.md D-073.
        if self.addressing_mode is not AddressingMode.HYBRID:
            return content_weights, HeadState(
                weights=content_weights,
                key=key,
                beta=beta,
            )

        gate = self.gate_dense(controller_output)
        shift = self.shift_dense(controller_output)
        gamma = self.gamma_dense(controller_output) + 1.0

        # 3. Interpolation (Gating)
        gated_weights = gate * content_weights + (1.0 - gate) * prev_weights

        # 4. Convolutional Shift
        shifted_weights = circular_convolution(gated_weights, shift)

        # 5. Sharpening
        final_weights = sharpen_weights(shifted_weights, gamma, epsilon=self.epsilon)

        new_state = HeadState(
            weights=final_weights,
            key=key,
            beta=beta,
            gate=gate,
            shift=shift,
            gamma=gamma,
        )

        return final_weights, new_state

    def call(
        self,
        inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Return the input unchanged.

        A head does no work through `call`. `NTMCell` drives it
        through `compute_addressing` instead. This method exists so
        the class satisfies the `keras.layers.Layer` interface.

        :param inputs: Any tensor.
        :type inputs: Any
        :param kwargs: Ignored.
        :type kwargs: Any
        :return: `inputs`, unchanged.
        :rtype: Any
        """
        return inputs

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Return the shape of the attention weights.

        :param input_shape: Shape of the controller output.
        :type input_shape: tuple[int | None, ...]
        :return: `(batch, memory_size)`.
        :rtype: tuple[int | None, ...]
        """
        return (input_shape[0], self.memory_size)

    def get_config(self) -> dict[str, Any]:
        """
        Return the constructor arguments, for serialization.

        Extends `BaseHead.get_config` (which emits `memory_size`,
        `memory_dim`, `addressing_mode`, `shift_range` and
        `epsilon`) with the two initializers and the regularizer.

        :return: The configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# NTMWriteHead
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.baseline_ntm")
class NTMWriteHead(BaseHead):
    """
    NTM write head: read weights plus the erase and add vectors.

    The addressing chain is identical to `NTMReadHead`'s. On top of it
    this head projects two more vectors, `erase` and `add`, which
    `NTMCell` hands to `NTMMemory.write` together with the weights.
    Both are always created, in either addressing mode.

    Which ADDRESSING projections exist depends on `addressing_mode`,
    exactly as for `NTMReadHead`: the gate, shift and gamma projections
    are created only under `AddressingMode.HYBRID`.

    On the HYBRID path `gamma_dense` is softplus and 1.0 is added to its
    output, so the exponent handed to `sharpen_weights` is always above 1
    and that helper's own `maximum(gamma, 1.0)` clamp never fires here.

    **Architecture Overview:**

    .. code-block:: text

        controller_output (batch, controller_dim) -- INPUT
                          │
                          ▼
        ┌─ Dense projections, always created ────────────┐
        │ key_dense    ──► key  (batch, M)               │
        │ beta_dense   ──► beta (batch, 1), softplus     │
        │ erase_dense  ──► e    (batch, M), sigmoid      │
        │ add_dense    ──► a    (batch, M), tanh         │
        └─────────────────┬──────────────────────────────┘
                          ▼
        ┌─ content_addressing ───────────────────────────┐
        │ cosine_similarity(key, memory, epsilon)        │
        │ softmax(beta * similarity, axis=-1)            │
        └─────────────────┬──────────────────────────────┘
                          │  content_weights (batch, N)
            ┌─────────────┴──────────────────────────┐
            ▼                                        ▼
            CONTENT                                  HYBRID
            │                                        ▼
            │                           ┌─────────────────────────┐
            │                           │ gate_dense   (optional) │
            │                           │ shift_dense  (optional) │
            │                           │ gamma_dense  (optional) │
            │                           │   gamma = out + 1.0     │
            │                           └────────────┬────────────┘
            │                                        ▼
            │                    ┌──────────────────────────────────────┐
            │                    │ gate * content_weights               │
            │                    │   + (1 - gate) * prev_weights        │
            │                    │ circular_convolution(w, shift)       │
            │                    │ sharpen_weights(w, gamma, epsilon)   │
            │                    └───────────────────┬──────────────────┘
            │                                        │
            └─────────────┬──────────────────────────┘
                          ▼
        weights (batch, N), HeadState

    :param memory_size: Number of memory slots, N. Must be positive.
    :type memory_size: int
    :param memory_dim: Width of one memory slot, M. Must be positive.
    :type memory_dim: int
    :param addressing_mode: Which addressing chain to run.
        `AddressingMode.HYBRID`, the default, runs content, then
        interpolation with `prev_weights`, then circular shift, then
        sharpening. `AddressingMode.CONTENT` returns the content weights
        directly and does not create the gate / shift / gamma projections
        at all, so a CONTENT head has strictly fewer parameters.
    :type addressing_mode: AddressingMode
    :param shift_range: Width of the circular-shift distribution, S. Read
        ONLY on the HYBRID path; a CONTENT head creates no shift
        projection, so the value is silently ignored there. Defaults to 3.
    :type shift_range: int
    :param kernel_initializer: Initializer for the Dense kernels. Each
        projection gets its own clone of it. Defaults to
        `'glorot_uniform'`.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for the Dense biases. Every
        projection this head creates receives it, `key_dense` included,
        and each gets its OWN clone of it -- a single instance handed to
        all of them would draw the same bias for every projection.
        Defaults to `'zeros'`.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer shared by every Dense kernel.
        Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param epsilon: Small constant handed to the two addressing helpers.
        It is added inside each L2 norm in `cosine_similarity`, so a
        zero-length key or an all-zero memory slot gives `sqrt(epsilon)`
        rather than a divide by zero; and on the HYBRID path it is added
        to the base and the denominator in `sharpen_weights`, so an
        exactly-zero weight raised to a large gamma stays differentiable.
        Under CONTENT only the `cosine_similarity` use is reached.
        Defaults to 1e-6.
    :type epsilon: float
    :param kwargs: Forwarded to `BaseHead.__init__`.
    :type kwargs: Any

    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or None.
    :vartype kernel_regularizer: keras.regularizers.Regularizer | None
    :ivar gate_dense: Interpolation gate projection, or None under
        CONTENT.
    :vartype gate_dense: keras.layers.Dense | None
    :ivar shift_dense: Circular-shift projection, or None under CONTENT.
    :vartype shift_dense: keras.layers.Dense | None
    :ivar gamma_dense: Sharpening projection, or None under CONTENT.
    :vartype gamma_dense: keras.layers.Dense | None
    :ivar erase_dense: Erase-vector projection, sigmoid. Always created.
    :vartype erase_dense: keras.layers.Dense
    :ivar add_dense: Add-vector projection, tanh. Always created.
    :vartype add_dense: keras.layers.Dense
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        addressing_mode: AddressingMode = AddressingMode.HYBRID,
        shift_range: int = 3,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Resolve the initializers and create the Dense projections.

        `key_dense`, `beta_dense`, `erase_dense` and `add_dense` are
        always created. The three location-addressing projections
        are created only under `AddressingMode.HYBRID`; otherwise
        they are set to None.

        :param memory_size: Number of memory slots, N.
        :type memory_size: int
        :param memory_dim: Width of one memory slot, M.
        :type memory_dim: int
        :param addressing_mode: Which addressing chain to run.
            Defaults to `AddressingMode.HYBRID`.
        :type addressing_mode: AddressingMode
        :param shift_range: Width of the circular-shift
            distribution, S. Defaults to 3.
        :type shift_range: int
        :param kernel_initializer: Initializer for the Dense kernels.
            Defaults to `'glorot_uniform'`.
        :type kernel_initializer: str | keras.initializers.Initializer
        :param bias_initializer: Initializer for the Dense biases. Each
            projection gets its own clone. Defaults to `'zeros'`.
        :type bias_initializer: str | keras.initializers.Initializer
        :param kernel_regularizer: Regularizer shared by every Dense
            kernel. Defaults to None.
        :type kernel_regularizer: keras.regularizers.Regularizer | None
        :param epsilon: Small constant handed to `cosine_similarity`
            and, on the HYBRID path, to `sharpen_weights`.
            Defaults to 1e-6.
        :type epsilon: float
        :param kwargs: Forwarded to `BaseHead.__init__`.
        :type kwargs: Any
        """
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=addressing_mode,
            shift_range=shift_range,
            epsilon=epsilon,
            **kwargs,
        )

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Addressing parameters
        self.key_dense = keras.layers.Dense(
            memory_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="key",
        )
        self.beta_dense = keras.layers.Dense(
            1,
            activation="softplus",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="beta",
        )
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-073
        # Same branch as NTMReadHead: under CONTENT the location-addressing projections do
        # not exist. See decisions.md D-073.
        if self.addressing_mode is AddressingMode.HYBRID:
            self.gate_dense = keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="gate",
            )
            self.shift_dense = keras.layers.Dense(
                shift_range,
                activation="softmax",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="shift",
            )
            self.gamma_dense = keras.layers.Dense(
                1,
                activation="softplus",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="gamma",
            )
        else:
            self.gate_dense = None
            self.shift_dense = None
            self.gamma_dense = None

        # Write-specific parameters
        self.erase_dense = keras.layers.Dense(
            memory_dim,
            activation="sigmoid",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="erase",
        )
        self.add_dense = keras.layers.Dense(
            memory_dim,
            activation="tanh",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="add",
        )

    def build(self, input_shape: tuple) -> None:
        """
        Build every Dense projection this head created.

        The three location-addressing projections are built only
        under HYBRID, because under CONTENT they do not exist.
        `erase_dense` and `add_dense` are always built.

        :param input_shape: Shape of the controller output,
            `(batch, controller_dim)`.
        :type input_shape: tuple
        """
        self.key_dense.build(input_shape)
        self.beta_dense.build(input_shape)
        if self.addressing_mode is AddressingMode.HYBRID:
            self.gate_dense.build(input_shape)
            self.shift_dense.build(input_shape)
            self.gamma_dense.build(input_shape)
        self.erase_dense.build(input_shape)
        self.add_dense.build(input_shape)
        super().build(input_shape)

    def content_addressing(
        self,
        key: Any,
        beta: Any,
        memory: Any,
    ) -> Any:
        """
        Score every memory slot against the key.

        Cosine similarity between the key and each slot, scaled by
        the key strength `beta`, then softmax over slots. A large
        `beta` sharpens the distribution toward the closest slot.

        :param key: Key vector of shape (batch, 1, M).
        :type key: Any
        :param beta: Key strength of shape (batch, 1), non-negative
            because the projection is softplus.
        :type beta: Any
        :param memory: Memory matrix of shape (batch, N, M).
        :type memory: Any
        :return: Content weights of shape (batch, N), summing to 1.
        :rtype: Any
        """
        similarity = cosine_similarity(key, memory, epsilon=self.epsilon)
        return keras.ops.softmax(beta * similarity, axis=-1)

    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> tuple[Any, HeadState]:
        """
        Turn the controller output into write weights and vectors.

        Projects the key, key strength, erase vector and add vector,
        scores the memory, and then either returns those content
        weights directly (CONTENT) or runs interpolation, circular
        shift and sharpening on top of them (HYBRID).

        The erase and add vectors are projected in both modes and
        always reach the returned `HeadState`.

        :param controller_output: Controller output of shape
            (batch, controller_dim).
        :type controller_output: Any
        :param memory_state: The state holding the memory to score.
        :type memory_state: MemoryState
        :param prev_weights: This head's weights from the previous
            timestep, shape (batch, N). Unused under CONTENT.
        :type prev_weights: Any
        :return: The weights of shape (batch, N), and a `HeadState`
            carrying them, the projected parameters, and the erase
            and add vectors. Under CONTENT the state's `gate`,
            `shift` and `gamma` stay None.
        :rtype: tuple[Any, HeadState]
        """
        # 1. Project parameters
        key = self.key_dense(controller_output)
        beta = self.beta_dense(controller_output)
        erase = self.erase_dense(controller_output)
        add = self.add_dense(controller_output)

        # 2. Content Addressing
        key_expanded = keras.ops.expand_dims(key, axis=1)
        content_weights = self.content_addressing(
            key_expanded, beta, memory_state.memory
        )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-073
        # Content-only addressing: the content weights are the final weights.
        # See decisions.md D-073.
        if self.addressing_mode is not AddressingMode.HYBRID:
            return content_weights, HeadState(
                weights=content_weights,
                key=key,
                beta=beta,
                erase_vector=erase,
                add_vector=add,
            )

        gate = self.gate_dense(controller_output)
        shift = self.shift_dense(controller_output)
        gamma = self.gamma_dense(controller_output) + 1.0

        # 3. Interpolation
        gated_weights = gate * content_weights + (1.0 - gate) * prev_weights

        # 4. Shift
        shifted_weights = circular_convolution(gated_weights, shift)

        # 5. Sharpen
        final_weights = sharpen_weights(shifted_weights, gamma, epsilon=self.epsilon)

        new_state = HeadState(
            weights=final_weights,
            key=key,
            beta=beta,
            gate=gate,
            shift=shift,
            gamma=gamma,
            erase_vector=erase,
            add_vector=add,
        )

        return final_weights, new_state

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Return the shape of the attention weights.

        :param input_shape: Shape of the controller output.
        :type input_shape: tuple[int | None, ...]
        :return: `(batch, memory_size)`.
        :rtype: tuple[int | None, ...]
        """
        return (input_shape[0], self.memory_size)

    def get_config(self) -> dict[str, Any]:
        """
        Return the constructor arguments, for serialization.

        Extends `BaseHead.get_config` (which emits `memory_size`,
        `memory_dim`, `addressing_mode`, `shift_range` and
        `epsilon`) with the two initializers and the regularizer.

        :return: The configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# NTMController
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.baseline_ntm")
class NTMController(BaseController):
    """
    The controller network, the "CPU" of the NTM.

    It reads the timestep input concatenated with the previous read
    vectors, and emits the control signal every head projects from.
    `NTMCell` owns one of these.

    Exactly ONE core is created, in `__init__`, from
    `controller_type`. A feedforward controller is a single
    `Dense(relu)` and owns no recurrent weights at all; its
    `initialize_state` returns None and its `call` returns an empty
    state list.

    **Architecture Overview:**

    .. code-block:: text

        inputs (batch, input_dim + num_read_heads * M)
        state: 2 tensors for lstm, 1 for gru, None for ff
                          │
                          ▼
        ┌─ self.core, ONE branch, chosen in __init__ ────────┐
        │ controller_type 'lstm' ──► keras.layers.LSTMCell   │
        │ controller_type 'gru'  ──► keras.layers.GRUCell    │
        │ anything else           ──► Dense(relu)            │
        └─────────────────┬──────────────────────────────────┘
                          ▼
        controller_output (batch, controller_dim)
        new_states: 2 tensors for lstm, 1 for gru, [] for ff

    :param controller_dim: Width of the controller hidden state.
        Must be positive.
    :type controller_dim: int
    :param controller_type: One of `'lstm'`, `'gru'`,
        `'feedforward'`. Anything other than the first two takes the
        feedforward branch. Defaults to `'lstm'`.
    :type controller_type: str
    :param kernel_initializer: Initializer for the Dense kernels.
        Each consumer gets its own clone of it. Defaults to
        `'glorot_uniform'`.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for the Dense biases.
        Each consumer gets its own clone of it. Defaults to `'zeros'`.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer shared by every Dense
        kernel. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Forwarded to `BaseController.__init__`.
    :type kwargs: Any

    :ivar core: The one cell or layer chosen by `controller_type`.
    :vartype core: keras.layers.Layer
    """

    def __init__(
        self,
        controller_dim: int,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Resolve the initializers and create the one core layer.

        Exactly one of `LSTMCell`, `GRUCell` or `Dense(relu)` is
        created, chosen by `controller_type`. Anything other than
        `'lstm'` or `'gru'` takes the Dense branch.

        :param controller_dim: Width of the controller hidden state.
        :type controller_dim: int
        :param controller_type: One of `'lstm'`, `'gru'`,
            `'feedforward'`. Defaults to `'lstm'`.
        :type controller_type: str
        :param kernel_initializer: Initializer for the Dense kernels.
            Each consumer gets its own clone. Defaults to
            `'glorot_uniform'`.
        :type kernel_initializer: str | keras.initializers.Initializer
        :param bias_initializer: Initializer for the Dense biases.
            Each consumer gets its own clone. Defaults to `'zeros'`.
        :type bias_initializer: str | keras.initializers.Initializer
        :param kernel_regularizer: Regularizer shared by every Dense
            kernel. Defaults to None.
        :type kernel_regularizer: keras.regularizers.Regularizer | None
        :param kwargs: Forwarded to `BaseController.__init__`.
        :type kwargs: Any
        """
        super().__init__(
            controller_dim=controller_dim,
            controller_type=controller_type,
            **kwargs,
        )

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create core cell in __init__ (Golden Rule)
        if self.controller_type == "lstm":
            self.core = keras.layers.LSTMCell(
                self.controller_dim,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="controller_cell",
            )
        elif self.controller_type == "gru":
            self.core = keras.layers.GRUCell(
                self.controller_dim,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="controller_cell",
            )
        else:
            self.core = keras.layers.Dense(
                self.controller_dim,
                activation="relu",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                name="controller_dense",
            )

    def build(self, input_shape: tuple) -> None:
        """
        Build the core layer.

        Unwraps a nested shape first: `keras.layers.RNN` can hand
        down a list of shapes, and the core wants the first one.

        :param input_shape: Shape of the controller input,
            `(batch, input_dim)`, or a list whose first entry is
            that shape.
        :type input_shape: tuple
        """
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                input_shape = input_shape[0]

        if hasattr(self.core, "build"):
            self.core.build(input_shape)

        super().build(input_shape)

    def initialize_state(self, batch_size: int) -> list[keras.KerasTensor] | None:
        """
        Build the starting controller state, all zeros.

        An LSTM core needs two tensors (hidden and cell), a GRU
        core one, and a feedforward core none.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: Two zero tensors of shape
            (batch, controller_dim) for `'lstm'`, one for
            `'gru'`, and None for a feedforward controller.
        :rtype: list[keras.KerasTensor] | None
        """
        if self.controller_type == "lstm":
            return [
                keras.ops.zeros((batch_size, self.controller_dim)),
                keras.ops.zeros((batch_size, self.controller_dim)),
            ]
        elif self.controller_type == "gru":
            return [keras.ops.zeros((batch_size, self.controller_dim))]
        return None

    def call(
        self,
        inputs: Any,
        state: list[keras.KerasTensor] | None = None,
        training: bool | None = None,
    ) -> tuple[Any, list[keras.KerasTensor]]:
        """
        Run the input through the core for one timestep.

        A recurrent core also takes and returns state; a
        feedforward core ignores `state` and returns an empty
        state list. When a recurrent core is called with
        `state=None`, a zero state is built here.

        :param inputs: Input of shape (batch, input_dim).
        :type inputs: Any
        :param state: The state from the previous timestep, or
            None to start from zeros. Defaults to None.
        :type state: list[keras.KerasTensor] | None
        :param training: Keras training flag. Defaults to None.
        :type training: bool | None
        :return: The controller output of shape
            (batch, controller_dim), and the new state list
            (empty for a feedforward controller).
        :rtype: tuple[Any, list[keras.KerasTensor]]
        """
        if self.controller_type in ["lstm", "gru"]:
            if state is None:
                batch_size = keras.ops.shape(inputs)[0]
                state = self.initialize_state(batch_size)

            output, new_states = self.core(inputs, state, training=training)

            if not isinstance(new_states, list):
                new_states = (
                    list(new_states) if hasattr(new_states, "__iter__") else [new_states]
                )
            return output, new_states
        else:
            output = self.core(inputs, training=training)
            return output, []

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[tuple[int | None, ...], list[tuple[int | None, ...]]]:
        """
        Return the output shape and the state shapes.

        :param input_shape: Shape of the controller input.
        :type input_shape: tuple[int | None, ...]
        :return: `(batch, controller_dim)`, and a list of two
            such shapes for `'lstm'`, one for `'gru'`, empty for
            a feedforward controller.
        :rtype: tuple[tuple[int | None, ...], list[tuple[int | None, ...]]]
        """
        batch_size = input_shape[0]
        output_shape = (batch_size, self.controller_dim)

        state_shape = (batch_size, self.controller_dim)
        if self.controller_type == "lstm":
            state_shapes = [state_shape, state_shape]
        elif self.controller_type == "gru":
            state_shapes = [state_shape]
        else:
            state_shapes = []

        return output_shape, state_shapes

    def get_config(self) -> dict[str, Any]:
        """
        Return the constructor arguments, for serialization.

        Extends `BaseController.get_config` (which emits
        `controller_dim` and `controller_type`) with the two
        initializers and the regularizer.

        :return: The configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# NTMCell
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.baseline_ntm")
class NTMCell(keras.layers.Layer):
    """
    One NTM timestep, shaped for `keras.layers.RNN`.

    The cell owns the memory module, the controller and every head,
    and carries the whole NTM state as the flat tensor list
    `keras.layers.RNN` requires. `state_size` and `output_size` are
    computed once in `__init__`.

    Write heads run BEFORE read heads, and each write rebinds the
    memory state, so a read head in the same timestep sees the
    memory the write heads just produced.

    **Architecture Overview:**

    .. code-block:: text

        inputs (batch, input_dim);  states, the flat list below
                          │
                          ▼
        ┌─ unpack states, in this exact order ─────────────────┐
        │ controller  2 x (batch, controller_dim) for lstm,    │
        │               1 x for gru, none for feedforward      │
        │ memory      1 x (batch, N, M)                        │
        │ read vecs   num_read_heads x (batch, M)              │
        │ read wts    num_read_heads x (batch, N)              │
        │ write wts   num_write_heads x (batch, N)             │
        └─────────────────┬────────────────────────────────────┘
                          ▼
        ┌─ controller step ────────────────────────────────────┐
        │ concatenate(inputs, prev read vecs)                  │
        │ self.controller ──► controller_output                │
        └─────────────────┬────────────────────────────────────┘
                          │  (batch, controller_dim)
                          ▼
        ┌─ write heads FIRST ──────────────────────────────────┐
        │ for each write head:                                 │
        │   compute_addressing ──► w, erase, add               │
        │   self.memory.write ──► memory is REBOUND            │
        │ so a read head sees the memory the write             │
        │ heads just produced, not the incoming one            │
        └─────────────────┬────────────────────────────────────┘
                          ▼
        ┌─ read heads SECOND ──────────────────────────────────┐
        │ for each read head:                                  │
        │   compute_addressing ──► w                           │
        │   self.memory.read ──► read vector (batch, M)        │
        └─────────────────┬────────────────────────────────────┘
                          ▼
        cell_output = concatenate(controller_output, read vecs)
          (batch, controller_dim + num_read_heads * M)
        new_states: same order as the unpack above

    Input shape:
        2D tensor of shape `(batch_size, input_dim)`, one timestep.

    Output shape:
        `(batch_size, controller_dim + num_read_heads * memory_dim)`.

    :param config: The NTM configuration, or a dict from
        `NTMConfig.to_dict()`. A dict is rebuilt with
        `NTMConfig.from_dict`.
    :type config: NTMConfig | dict[str, Any]
    :param kernel_initializer: Initializer for the Dense kernels.
        Each consumer gets its own clone of it. Defaults to
        `'glorot_uniform'`.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for the Dense biases.
        Each consumer gets its own clone of it. Defaults to `'zeros'`.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer shared by every Dense
        kernel. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
    :type kwargs: Any

    :ivar config: The resolved `NTMConfig`.
    :vartype config: NTMConfig
    :ivar memory: The memory module.
    :vartype memory: NTMMemory
    :ivar controller: The controller.
    :vartype controller: NTMController
    :ivar read_heads: One `NTMReadHead` per configured read head.
    :vartype read_heads: list[NTMReadHead]
    :ivar write_heads: One `NTMWriteHead` per configured write head.
    :vartype write_heads: list[NTMWriteHead]
    """

    def __init__(
        self,
        config: NTMConfig | dict[str, Any],
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Build the memory, controller and every head.

        A dict `config` is rebuilt into an `NTMConfig` first.
        `state_size` and `output_size` are computed here, once,
        because `keras.layers.RNN` reads them before `build`.

        The learnable initial memory is NOT created here; it is
        a weight and so belongs in `build`.

        :param config: The NTM configuration, or a dict from
            `NTMConfig.to_dict()`.
        :type config: NTMConfig | dict[str, Any]
        :param kernel_initializer: Initializer for the Dense kernels.
            Each consumer gets its own clone. Defaults to
            `'glorot_uniform'`.
        :type kernel_initializer: str | keras.initializers.Initializer
        :param bias_initializer: Initializer for the Dense biases.
            Each consumer gets its own clone. Defaults to `'zeros'`.
        :type bias_initializer: str | keras.initializers.Initializer
        :param kernel_regularizer: Regularizer shared by every Dense
            kernel. Defaults to None.
        :type kernel_regularizer: keras.regularizers.Regularizer | None
        :param kwargs: Forwarded to `keras.layers.Layer.__init__`.
        :type kwargs: Any
        """
        super().__init__(**kwargs)

        # Handle dict config from deserialization
        if isinstance(config, dict):
            self.config = NTMConfig.from_dict(config)
        else:
            self.config = config

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create sub-layers
        self.memory = NTMMemory(
            self.config.memory_size,
            self.config.memory_dim,
            memory_init_seed=self.config.memory_init_seed,
            name="memory",
        )

        self.controller = NTMController(
            self.config.controller_dim,
            self.config.controller_type,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="controller",
        )

        self.read_heads = [
            NTMReadHead(
                self.config.memory_size,
                self.config.memory_dim,
                self.config.addressing_mode,
                self.config.shift_range,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                epsilon=self.config.epsilon,
                name=f"read_head_{i}",
            )
            for i in range(self.config.num_read_heads)
        ]

        self.write_heads = [
            NTMWriteHead(
                self.config.memory_size,
                self.config.memory_dim,
                self.config.addressing_mode,
                self.config.shift_range,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                epsilon=self.config.epsilon,
                name=f"write_head_{i}",
            )
            for i in range(self.config.num_write_heads)
        ]

        # Learnable initial memory (created in build)
        self._initial_memory = None

        # Pre-calculate state sizes
        self._state_size = self._calculate_state_size()
        self._output_size = self.config.controller_dim + (
            self.config.num_read_heads * self.config.memory_dim
        )

    @property
    def state_size(self) -> list[Any]:
        """
        The state sizes, as `keras.layers.RNN` expects them.

        :return: One entry per state tensor, in the order
            `call` unpacks them.
        :rtype: list[Any]
        """
        return self._state_size

    @property
    def output_size(self) -> int:
        """
        The width of one timestep's output.

        :return: `controller_dim + num_read_heads * memory_dim`.
        :rtype: int
        """
        return self._output_size

    def _calculate_state_size(self) -> list[Any]:
        """
        Compute the state sizes, in the order `call` unpacks them.

        The order is: controller state (two entries for
        `'lstm'`, one for `'gru'`, none for feedforward), then
        the memory, then one read vector per read head, then one
        read-weight vector per read head, then one write-weight
        vector per write head.

        :return: One entry per state tensor. The memory entry is
            the tuple `(memory_size, memory_dim)`; the others are
            plain widths.
        :rtype: list[Any]
        """
        sizes = []

        # Controller State
        if self.config.controller_type == "lstm":
            sizes.extend([self.config.controller_dim, self.config.controller_dim])
        elif self.config.controller_type == "gru":
            sizes.append(self.config.controller_dim)

        # Memory Matrix
        sizes.append((self.config.memory_size, self.config.memory_dim))

        # Read Vectors
        for _ in range(self.config.num_read_heads):
            sizes.append(self.config.memory_dim)

        # Read Weights
        for _ in range(self.config.num_read_heads):
            sizes.append(self.config.memory_size)

        # Write Weights
        for _ in range(self.config.num_write_heads):
            sizes.append(self.config.memory_size)

        return sizes

    def build(self, input_shape: tuple) -> None:
        """
        Build the controller and every head, and the initial memory.

        The controller is built on the concatenation of the
        timestep input with every read vector, so its input width
        is `input_dim + num_read_heads * memory_dim`. Every head
        is built on the controller output.

        The learnable initial memory weight is created here, and
        only when `config.use_memory_init` is set.

        :param input_shape: Shape of one timestep's input,
            `(batch, feature_dim)`.
        :type input_shape: tuple
        """
        feature_dim = input_shape[-1]
        total_read_dim = self.config.num_read_heads * self.config.memory_dim
        controller_input_shape = (None, feature_dim + total_read_dim)

        self.controller.build(controller_input_shape)

        controller_output_shape = (None, self.config.controller_dim)

        for head in self.read_heads:
            head.build(controller_output_shape)

        for head in self.write_heads:
            head.build(controller_output_shape)

        # Create learnable initial memory if configured
        if self.config.use_memory_init:
            self._initial_memory = self.add_weight(
                name="initial_memory",
                shape=(1, self.config.memory_size, self.config.memory_dim),
                initializer=keras.initializers.RandomNormal(stddev=1e-3),
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        states: list[keras.KerasTensor],
        training: bool | None = None,
    ) -> tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        """
        Run one NTM timestep.

        Unpacks the flat state list, runs the controller on the
        input concatenated with the previous read vectors, then
        the write heads, then the read heads.

        Write heads run FIRST and each write rebinds the memory
        state, so the read heads see the memory the write heads
        just produced rather than the one that came in.

        :param inputs: This timestep's input, shape
            (batch, input_dim).
        :type inputs: keras.KerasTensor
        :param states: The state tensors, in the order
            `_calculate_state_size` lists them.
        :type states: list[keras.KerasTensor]
        :param training: Keras training flag. Defaults to None.
        :type training: bool | None
        :return: The output of shape
            `(batch, controller_dim + num_read_heads * memory_dim)`,
            and the new state list in the same order as `states`.
        :rtype: tuple[keras.KerasTensor, list[keras.KerasTensor]]
        """
        # Unpack State
        idx = 0

        if self.config.controller_type == "lstm":
            controller_state = [states[idx], states[idx + 1]]
            idx += 2
        elif self.config.controller_type == "gru":
            controller_state = [states[idx]]
            idx += 1
        else:
            controller_state = None

        memory_val = states[idx]
        idx += 1

        prev_read_vectors = []
        for _ in range(self.config.num_read_heads):
            prev_read_vectors.append(states[idx])
            idx += 1

        prev_read_weights = []
        for _ in range(self.config.num_read_heads):
            prev_read_weights.append(states[idx])
            idx += 1

        prev_write_weights = []
        for _ in range(self.config.num_write_heads):
            prev_write_weights.append(states[idx])
            idx += 1

        memory_state = MemoryState(memory=memory_val)

        # Controller Step
        flat_read_vectors = keras.ops.concatenate(prev_read_vectors, axis=-1)
        controller_input = keras.ops.concatenate([inputs, flat_read_vectors], axis=-1)

        controller_output, new_controller_state = self.controller(
            controller_input,
            state=controller_state,
            training=training,
        )

        # Write Heads
        current_memory_state = memory_state
        new_write_weights = []

        for i, head in enumerate(self.write_heads):
            weights, head_state = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_write_weights[i],
            )
            new_write_weights.append(weights)

            current_memory_state = self.memory.write(
                current_memory_state,
                weights,
                head_state.erase_vector,
                head_state.add_vector,
            )

        # Read Heads
        new_read_weights = []
        new_read_vectors = []

        for i, head in enumerate(self.read_heads):
            weights, _ = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_read_weights[i],
            )
            new_read_weights.append(weights)

            read_vec = self.memory.read(current_memory_state, weights)
            new_read_vectors.append(read_vec)

        # Pack Output State
        new_states = []

        if self.config.controller_type == "lstm":
            new_states.extend(new_controller_state)
        elif self.config.controller_type == "gru":
            new_states.extend(new_controller_state)

        new_states.append(current_memory_state.memory)
        new_states.extend(new_read_vectors)
        new_states.extend(new_read_weights)
        new_states.extend(new_write_weights)

        # Output
        flat_new_read_vectors = keras.ops.concatenate(new_read_vectors, axis=-1)
        cell_output = keras.ops.concatenate([controller_output, flat_new_read_vectors], axis=-1)

        return cell_output, new_states

    def get_initial_state(
        self,
        inputs: keras.KerasTensor | None = None,
        batch_size: int | None = None,
        dtype: str | None = None,
    ) -> list[keras.KerasTensor]:
        """
        Build the starting state for a new sequence.

        The controller state, read vectors and both sets of
        weights start at zero, except that the weights start
        uniform (`1 / memory_size` in every slot) rather than at
        zero, so the first timestep reads a plain average rather
        than nothing.

        The memory starts either from the learnable
        `initial_memory` weight, when `config.use_memory_init` is
        set, or from a small seeded random draw.

        :param inputs: A tensor to take the batch size from, used
            only when `batch_size` is None. Defaults to None.
        :type inputs: keras.KerasTensor | None
        :param batch_size: Number of sequences in the batch.
            Defaults to None.
        :type batch_size: int | None
        :param dtype: Accepted for the `keras.layers.RNN`
            interface and not used. Defaults to None.
        :type dtype: str | None
        :return: The state tensors, in the order
            `_calculate_state_size` lists them.
        :rtype: list[keras.KerasTensor]
        """
        if batch_size is None and inputs is not None:
            batch_size = keras.ops.shape(inputs)[0]

        states = []

        # Controller states
        if self.config.controller_type == "lstm":
            states.extend([
                keras.ops.zeros((batch_size, self.config.controller_dim)),
                keras.ops.zeros((batch_size, self.config.controller_dim)),
            ])
        elif self.config.controller_type == "gru":
            states.append(keras.ops.zeros((batch_size, self.config.controller_dim)))

        # Memory — use learnable initial memory if available, else random
        if self._initial_memory is not None:
            memory = keras.ops.broadcast_to(
                self._initial_memory,
                (batch_size, self.config.memory_size, self.config.memory_dim),
            )
        else:
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-058
            # A FIXED seed, so this draw is stateless and repeats exactly. DO NOT drop it: an
            # unseeded `keras.random.normal` draws from the global stateful stream, so with
            # `use_memory_init=False` the memory differed per call and `model.predict(x)` twice
            # returned DIFFERENT values -- at stddev 1e-3, flaky rather than red. See decisions.md D-058.
            memory = keras.random.normal(
                (batch_size, self.config.memory_size, self.config.memory_dim),
                mean=0.0,
                stddev=1e-3,
                seed=self.config.memory_init_seed,
            )
        states.append(memory)

        # Read Vectors
        for _ in range(self.config.num_read_heads):
            states.append(keras.ops.zeros((batch_size, self.config.memory_dim)))

        # Read Weights (uniform)
        uniform_weight = keras.ops.ones((1, self.config.memory_size)) / self.config.memory_size
        for _ in range(self.config.num_read_heads):
            states.append(keras.ops.broadcast_to(uniform_weight, (batch_size, self.config.memory_size)))

        # Write Weights (uniform)
        for _ in range(self.config.num_write_heads):
            states.append(keras.ops.broadcast_to(uniform_weight, (batch_size, self.config.memory_size)))

        return states

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Return the shape of one timestep's output.

        :param input_shape: Shape of one timestep's input.
        :type input_shape: tuple[int | None, ...]
        :return: `(batch, controller_dim + num_read_heads * memory_dim)`.
        :rtype: tuple[int | None, ...]
        """
        return (input_shape[0], self._output_size)

    def get_config(self) -> dict[str, Any]:
        """
        Return the constructor arguments, for serialization.

        The `NTMConfig` is emitted as a plain dict via
        `NTMConfig.to_dict()`; `from_config` rebuilds it.

        :return: The configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "config": self.config.to_dict(),
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NTMCell":
        """
        Rebuild the cell from `get_config`'s output.

        :param config: The output of `get_config`.
        :type config: dict[str, Any]
        :return: A new cell.
        :rtype: NTMCell
        """
        return cls(**config)


# ---------------------------------------------------------------------
# NeuralTuringMachine
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.baseline_ntm")
class NeuralTuringMachine(BaseNTM):
    """
    The complete NTM layer: an RNN wrapped around `NTMCell`.

    `__init__` builds an `NTMCell`, hands it to a
    `keras.layers.RNN`, and adds a `Dense` output projection.
    `call` runs those two in order.

    This class does NOT run the step loop `BaseNTM.call` provides.
    `keras.layers.RNN` drives `NTMCell` instead. Consequently
    `step()` and `initialize_state()` exist only to satisfy the
    abstract base class and are not on the forward path: both raise
    `NotImplementedError`, as does `get_memory_state()`, and
    `reset_memory()` is a no-op. Use `NTMCell.get_initial_state(...)`
    and `return_state=True` to reach the state.

    **Architecture Overview:**

    .. code-block:: text

        inputs (batch, seq_len, input_dim) -- INPUT
        initial_state (optional)
                          │
                          ▼
        ┌─ self.rnn ─────────────────────────────────────────┐
        │ keras.layers.RNN(self.ntm_cell)                    │
        │ drives NTMCell.call once per timestep              │
        └─────────────────┬──────────────────────────────────┘
                          │  (batch, seq_len, cell_out) when
                          │  return_sequences, else (batch, cell_out)
                          ▼
        ┌─ self.output_projection ───────────────────────────┐
        │ Dense(output_dim)                                  │
        └─────────────────┬──────────────────────────────────┘
                          ▼
        output; plus the final RNN states when return_state

    Input shape:
        3D tensor of shape `(batch_size, seq_len, input_dim)`.

    Output shape:
        `(batch_size, seq_len, output_dim)` when `return_sequences`
        is True, otherwise `(batch_size, output_dim)`.

    :param config: The NTM configuration, or a dict from
        `NTMConfig.to_dict()`. A dict is rebuilt with
        `NTMConfig.from_dict`.
    :type config: NTMConfig | dict[str, Any]
    :param output_dim: Width of the output projection. Required.
    :type output_dim: int
    :param return_sequences: Whether to return every timestep rather
        than only the last. Defaults to True.
    :type return_sequences: bool
    :param return_state: Whether to also return the final RNN
        states. Defaults to False.
    :type return_state: bool
    :param kernel_initializer: Initializer for the Dense kernels.
        Each consumer gets its own clone of it. Defaults to
        `'glorot_uniform'`.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for the Dense biases.
        Each consumer gets its own clone of it. Defaults to `'zeros'`.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer shared by every Dense
        kernel. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Forwarded to `BaseNTM.__init__`.
    :type kwargs: Any

    :ivar ntm_cell: The wrapped cell.
    :vartype ntm_cell: NTMCell
    :ivar rnn: The `keras.layers.RNN` driving the cell.
    :vartype rnn: keras.layers.RNN
    :ivar output_projection: The final Dense projection.
    :vartype output_projection: keras.layers.Dense
    """

    def __init__(
        self,
        config: NTMConfig | dict[str, Any],
        output_dim: int,
        return_sequences: bool = True,
        return_state: bool = False,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        # Handle dict config
        """
        Build the cell, the RNN around it, and the projection.

        A dict `config` is rebuilt into an `NTMConfig` first.
        Three sub-layers are created: an `NTMCell`, the
        `keras.layers.RNN` that drives it, and the `Dense`
        output projection.

        :param config: The NTM configuration, or a dict from
            `NTMConfig.to_dict()`.
        :type config: NTMConfig | dict[str, Any]
        :param output_dim: Width of the output projection.
        :type output_dim: int
        :param return_sequences: Whether to return every
            timestep. Defaults to True.
        :type return_sequences: bool
        :param return_state: Whether to also return the final
            RNN states. Defaults to False.
        :type return_state: bool
        :param kernel_initializer: Initializer for the Dense
            kernels. Each consumer gets its own clone.
            Defaults to `'glorot_uniform'`.
        :type kernel_initializer: str | keras.initializers.Initializer
        :param bias_initializer: Initializer for the Dense
            biases. Each consumer gets its own clone.
            Defaults to `'zeros'`.
        :type bias_initializer: str | keras.initializers.Initializer
        :param kernel_regularizer: Regularizer shared by every
            Dense kernel. Defaults to None.
        :type kernel_regularizer: keras.regularizers.Regularizer | None
        :param kwargs: Forwarded to `BaseNTM.__init__`.
        :type kwargs: Any
        """
        if isinstance(config, dict):
            config = NTMConfig.from_dict(config)

        super().__init__(config=config, output_dim=output_dim, **kwargs)

        self.return_sequences = return_sequences
        self.return_state = return_state
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.ntm_cell = NTMCell(
            self.config,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="ntm_cell",
        )

        self.rnn = keras.layers.RNN(
            self.ntm_cell,
            return_sequences=return_sequences,
            return_state=return_state,
            name="ntm_rnn",
        )

        self.output_projection = keras.layers.Dense(
            output_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="output_projection",
        )

    def build(self, input_shape: tuple) -> None:
        """
        Build the RNN and the output projection.

        The projection is built on the RNN's output, which
        keeps the time axis only when `return_sequences` is
        set.

        :param input_shape: Shape of the input sequence,
            `(batch, seq_len, input_dim)`.
        :type input_shape: tuple
        """
        self.rnn.build(input_shape)

        out_dim = self.ntm_cell.output_size
        if self.return_sequences:
            proj_input_shape = (input_shape[0], input_shape[1], out_dim)
        else:
            proj_input_shape = (input_shape[0], out_dim)
        self.output_projection.build(proj_input_shape)

        super().build(input_shape)

    def initialize_state(
        self,
        batch_size: int,
    ) -> tuple[MemoryState, list[HeadState], Any | None]:
        """
        Always raises. Not on this layer's forward path.

        This layer wraps `NTMCell` in a `keras.layers.RNN`,
        and the RNN owns state initialization through
        `NTMCell.get_initial_state(...)`. The `BaseNTM`
        step and state API is only meaningful for a subclass
        that runs the parent's step loop; this one does not.

        :param batch_size: Number of sequences in the batch.
            Not used.
        :type batch_size: int
        :return: Never returns.
        :rtype: tuple[MemoryState, list[HeadState], Any | None]
        :raises NotImplementedError: Always. Use
            `NTMCell.get_initial_state(...)` instead.
        """
        raise NotImplementedError(
            "NeuralTuringMachine.initialize_state is not implemented; this layer "
            "uses keras.layers.RNN(NTMCell) and state is initialized by the wrapped "
            "cell. Use NTMCell.get_initial_state(...) for the BaseNTM step API."
        )

    def step(
        self,
        inputs: Any,
        memory_state: MemoryState,
        head_states: list[HeadState],
        controller_state: Any | None,
        training: bool | None = None,
    ) -> NTMOutput:
        """
        Always raises. Not on this layer's forward path.

        Stepping is done inside `keras.layers.RNN(NTMCell)`,
        which calls `NTMCell.call` once per timestep. Call
        this layer on a whole `(batch, seq_len, input_dim)`
        sequence rather than stepping it yourself.

        :param inputs: This timestep's input. Not used.
        :type inputs: Any
        :param memory_state: The memory state. Not used.
        :type memory_state: MemoryState
        :param head_states: The head states. Not used.
        :type head_states: list[HeadState]
        :param controller_state: The controller state. Not used.
        :type controller_state: Any | None
        :param training: Keras training flag. Not used.
            Defaults to None.
        :type training: bool | None
        :return: Never returns.
        :rtype: NTMOutput
        :raises NotImplementedError: Always. Call the layer
            on a full sequence instead.
        """
        raise NotImplementedError(
            "NeuralTuringMachine.step is not implemented; the wrapped RNN(NTMCell) "
            "performs stepping internally. Call the layer directly on a "
            "(batch, seq_len, input_dim) tensor."
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        initial_state: list[keras.KerasTensor] | None = None,
        training: bool | None = None,
    ) -> keras.KerasTensor | tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        """
        Run a sequence through the RNN and the projection.

        Two steps: the `keras.layers.RNN` drives `NTMCell`
        over the time axis, then the `Dense` projection maps
        the result to `output_dim`.

        This does NOT run the step loop `BaseNTM.call`
        provides. See the class docstring.

        :param inputs: Input sequence of shape
            `(batch, seq_len, input_dim)`.
        :type inputs: keras.KerasTensor
        :param initial_state: State to start the RNN from.
            When None the cell builds its own. Defaults to
            None.
        :type initial_state: list[keras.KerasTensor] | None
        :param training: Keras training flag. Defaults to None.
        :type training: bool | None
        :return: The projected output. When `return_state` is
            set, a tuple of that output and the final RNN
            state list instead.
        :rtype: keras.KerasTensor | tuple[keras.KerasTensor, list[keras.KerasTensor]]
        """
        rnn_result = self.rnn(inputs, initial_state=initial_state, training=training)

        if self.return_state:
            rnn_output = rnn_result[0]
            final_states = list(rnn_result[1:])
        else:
            rnn_output = rnn_result
            final_states = None

        output = self.output_projection(rnn_output, training=training)

        if self.return_state:
            return output, final_states
        return output

    def get_memory_state(self) -> MemoryState | None:
        """
        Always raises. There is no single memory state here.

        The memory tensor lives inside the per-timestep RNN
        state owned by `NTMCell`, so there is nothing on this
        layer to hand back. Construct the layer with
        `return_state=True` and read the final RNN states
        instead.

        :return: Never returns.
        :rtype: MemoryState | None
        :raises NotImplementedError: Always. Use
            `return_state=True`.
        """
        raise NotImplementedError(
            "NeuralTuringMachine.get_memory_state is not implemented; the memory "
            "tensor lives inside NTMCell's per-step RNN state. Construct the layer "
            "with return_state=True and inspect the final RNN states."
        )

    def reset_memory(self, batch_size: int) -> None:
        """
        Does nothing.

        There is no memory held between calls to reset:
        `keras.layers.RNN` builds fresh state at the start of
        every call. The method exists to satisfy `BaseNTM`.

        :param batch_size: Number of sequences in the batch.
            Not used.
        :type batch_size: int
        """
        pass

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Return the shape of the projected output.

        :param input_shape: Shape of the input sequence,
            `(batch, seq_len, input_dim)`.
        :type input_shape: tuple[int | None, ...]
        :return: `(batch, seq_len, output_dim)` when
            `return_sequences` is set, otherwise
            `(batch, output_dim)`.
        :rtype: tuple[int | None, ...]
        """
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        if self.return_sequences:
            return (batch_size, seq_len, self.output_dim)
        return (batch_size, self.output_dim)

    def get_config(self) -> dict[str, Any]:
        """
        Return the constructor arguments, for serialization.

        Extends `BaseNTM.get_config` (which emits the config
        and `output_dim`) with the two flags, the two
        initializers and the regularizer.

        :return: The configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NeuralTuringMachine":
        """
        Rebuild the layer from `get_config`'s output.

        :param config: The output of `get_config`.
        :type config: dict[str, Any]
        :return: A new layer.
        :rtype: NeuralTuringMachine
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------


def create_ntm(
    memory_size: int = 128,
    memory_dim: int = 64,
    output_dim: int = 10,
    controller_dim: int = 256,
    controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
    num_read_heads: int = 1,
    num_write_heads: int = 1,
    shift_range: int = 3,
    return_sequences: bool = True,
    return_state: bool = False,
    # DECISION plan-2026-08-30T063229-ccd6ad17/D-015
    # These four are appended LAST, not grouped with the other config
    # arguments. Do NOT reorder them to mirror NTMConfig's field order: the
    # first 10 parameters are positional-call compatible and moving them
    # silently re-maps every existing positional call. See decisions.md D-015.
    addressing_mode: AddressingMode = AddressingMode.HYBRID,
    use_memory_init: bool = True,
    memory_init_seed: int = 42,
    epsilon: float = 1e-6,
) -> NeuralTuringMachine:
    """
    Build a `NeuralTuringMachine` from plain arguments.

    Packs the memory and controller arguments into an `NTMConfig`,
    then constructs the layer. The returned layer is unbuilt.

    Every `NTMConfig` field is reachable from here; each argument
    that feeds the config defaults to the value `NTMConfig` itself
    declares, so omitting all of them builds the same layer the
    dataclass defaults would.

    The four config arguments come LAST in the signature, after the
    two layer flags, rather than beside the other config arguments:
    they were added to a released signature, and any other position
    would have changed what an existing positional call means.

    **Architecture Overview:**

    .. code-block:: text

        the 14 keyword arguments below
                          │
                          ▼
        ┌─ NTMConfig ────────────────────────────────────────┐
        │ memory_size, memory_dim, num_read_heads,           │
        │ num_write_heads, controller_dim,                   │
        │ controller_type, shift_range,                      │
        │ addressing_mode, use_memory_init,                  │
        │ memory_init_seed, epsilon                          │
        └─────────────────┬──────────────────────────────────┘
                          │  output_dim, return_sequences and
                          │  return_state are layer arguments and
                          │  bypass the config
                          ▼
        ┌─ NeuralTuringMachine ──────────────────────────────┐
        │ config, output_dim, return_sequences, return_state │
        └─────────────────┬──────────────────────────────────┘
                          ▼
        an unbuilt NeuralTuringMachine layer

    Example:
        .. code-block:: python

            import keras
            from dl_techniques.layers.memory.baseline_ntm import create_ntm

            ntm = create_ntm(memory_size=64, memory_dim=32, output_dim=8)
            inputs = keras.Input(shape=(10, 16))
            outputs = ntm(inputs)

    :param memory_size: Number of memory slots, N. Defaults to 128.
    :type memory_size: int
    :param memory_dim: Width of one memory slot, M. Defaults to 64.
    :type memory_dim: int
    :param output_dim: Width of the output projection. Defaults to 10.
    :type output_dim: int
    :param controller_dim: Width of the controller hidden state.
        Defaults to 256.
    :type controller_dim: int
    :param controller_type: One of `'lstm'`, `'gru'`,
        `'feedforward'`. Defaults to `'lstm'`.
    :type controller_type: str
    :param num_read_heads: Number of read heads. Defaults to 1.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads. Defaults to 1.
    :type num_write_heads: int
    :param shift_range: Width of the circular-shift distribution, S.
        Defaults to 3.
    :type shift_range: int
    :param return_sequences: Whether to return every timestep.
        Defaults to True.
    :type return_sequences: bool
    :param return_state: Whether to also return the final RNN
        states. Defaults to False.
    :type return_state: bool
    :param addressing_mode: Which addressing chain the heads run.
        `AddressingMode.HYBRID`, the default, runs content
        addressing then interpolation, shift and sharpening;
        `AddressingMode.CONTENT` stops after content addressing and
        the heads never build the three location projections.
    :type addressing_mode: AddressingMode
    :param use_memory_init: Whether the initial memory is a learned
        variable. Defaults to True.
    :type use_memory_init: bool
    :param memory_init_seed: Seed for the symmetry-breaking initial
        memory draw used when `use_memory_init` is False. Defaults
        to 42.
    :type memory_init_seed: int
    :param epsilon: Small constant for numerical stability, handed
        to the heads' addressing helpers. Defaults to 1e-6.
    :type epsilon: float
    :return: An unbuilt NTM layer.
    :rtype: NeuralTuringMachine
    """
    config = NTMConfig(
        memory_size=memory_size,
        memory_dim=memory_dim,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
        controller_dim=controller_dim,
        controller_type=controller_type,
        shift_range=shift_range,
        addressing_mode=addressing_mode,
        use_memory_init=use_memory_init,
        memory_init_seed=memory_init_seed,
        epsilon=epsilon,
    )

    return NeuralTuringMachine(
        config,
        output_dim=output_dim,
        return_sequences=return_sequences,
        return_state=return_state,
    )