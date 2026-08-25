"""
Neural Turing Machine with a configurable controller, external memory matrix and
optional output projection.

A recurrent network stores everything it knows in a fixed-size hidden vector, so
the capacity to remember and the capacity to compute are the same resource.
Holding a long sequence verbatim therefore costs the network exactly the units it
would otherwise use for processing, and the content it does hold is smeared across
a representation that has to be rewritten in full at every step. The NTM separates
the two: a controller of modest width is coupled to an `N x M` memory matrix it
addresses through a differentiable attention mechanism, so capacity grows by adding
memory slots rather than controller parameters, and a slot written at step 3 is
still bit-for-bit present at step 300 unless a write head explicitly erases it.

What makes the coupling trainable is that addressing is soft. Every head emits a
distribution `w` over the `N` slots and reads or writes the whole memory weighted
by it, so gradients flow to the addressing parameters as well as to the content.
Each head's distribution is produced in four stages. Content addressing scores a
key against every row by cosine similarity and softmaxes it at a learned sharpness
`beta`. An interpolation gate `g` blends that against the head's own weighting from
the previous step, which is what lets a head stay where it was rather than
re-deriving its position from content. A shift distribution `s` is then applied by
circular convolution, `w~(i) = sum_j w(j) * s(i - j mod N)`, giving relative
movement along the memory — the mechanism that turns "the next slot" into a
learnable primitive. Finally a sharpening exponent `gamma >= 1` renormalizes
`w^gamma`, undoing the blur that convolution introduces. The shift's orientation is
the part that is easy to get wrong: `keras.ops.roll(a, k)[i] == a[(i - k) mod N]`,
so the tap carrying offset `k` is `roll(w, +k)`, and negating it mirrors the shift
instead of inverting it. That sign is pinned by a decision anchor in
`layers/memory/ntm_interface.py` because an inverted version of it survived a long
period of green tests.

Writing is erase-then-add, `M_t = M_{t-1} * (1 - w e^T) + w a^T`. Splitting the
update into a multiplicative erase and an additive write means a head can clear a
slot, overwrite it, or accumulate into it, all as smooth functions of head outputs.
Within a timestep the cell runs the controller first, then all write heads in
sequence — each writing into the memory the previous one produced — and only then
the read heads, so reads observe the current step's writes. The controller itself
consumes the *previous* step's read vectors concatenated with the input, which is
the standard one-step delay: a read issued at step `t` can only influence the
controller at `t+1`.

This module is the model-level wrapper. `NTMCell` is a `keras.layers.RNN` cell
whose state tuple carries the controller state, the full memory matrix, and the
read vectors and read/write weightings for every head; wrapping it in `RNN` is what
unrolls it over a sequence. `return_state=False` on that wrapper is deliberate: the
raw memory matrix is a large tensor of internal bookkeeping, and exposing it from
the model's outputs would put it in every `fit()` metric path for the rare caller
that actually wants to inspect it. The cell's own output is the controller output
concatenated with the freshly read vectors, width
`controller_dim + num_read_heads * memory_dim`, which is why the optional dense
projection exists — without it the model's output width is an artifact of the
memory configuration rather than the task.

The three presets scale memory and controller together, since a wide controller
addressing a small memory just relearns to be an LSTM. Shift range stays at 3
(offsets -1, 0, +1) across all of them: relative movement by more than one slot per
step is not what the shift is for, and widening it enlarges the softmax the head
must learn to concentrate.

References:
    - Graves et al., 2014. Neural Turing Machines.
      (https://arxiv.org/abs/1410.5401)
    - Graves et al., 2016. Hybrid computing using a neural network with dynamic
      external memory. Nature 538, 471-476.
    - Weston et al., 2014. Memory Networks. (https://arxiv.org/abs/1410.3916)
"""

import keras
import dataclasses
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.memory import NTMCell, NTMConfig

# ---------------------------------------------------------------------
# NTM Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMModel(keras.Model):
    """
    Neural Turing Machine: an ``NTMCell`` unrolled by ``keras.layers.RNN``.

    Wraps the cell in an ``RNN`` layer to give a sequence-to-sequence or
    sequence-to-vector interface compatible with standard Keras workflows. The
    cell emits the controller output concatenated with the freshly read vectors,
    so its width is ``controller_dim + num_read_heads * memory_dim`` -- an
    artifact of the memory configuration rather than of the task, which is what
    the optional dense projection exists to fix.

    ``return_state=False`` on the ``RNN`` is deliberate and not exposed: the state
    tuple carries the full memory matrix, and surfacing it from the model's
    outputs would drag a large bookkeeping tensor through every ``fit()`` metric
    path for the rare caller who wants to inspect it.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, T, input_dim]             │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ntm_rnn: RNN(NTMCell), unrolled over T                      │
        │                                                              │
        │   per timestep:                                              │
        │     controller( x_t ‖ read_{t-1} )                           │
        │            │                                                 │
        │            ▼   head params: key, β, g, s, γ, erase, add      │
        │     ┌──────────────────────────────────────────────┐         │
        │     │ addressing:  content(key, β) → gate(g)       │         │
        │     │             → shift(s, circular) → sharpen(γ)│         │
        │     └──────────────────────────────────────────────┘         │
        │            │ w over N slots                                  │
        │            ▼                                                 │
        │     write heads:  M ← M·(1 − w eᵀ) + w aᵀ   (erase, then add)│
        │            ▼                                                 │
        │     read heads:   r = wᵀ M      (sees THIS step's writes)    │
        │            ▼                                                 │
        │     output_t = controller_out ‖ r                            │
        │            width = controller_dim + num_read_heads·memory_dim│
        │                                                              │
        │   return_sequences as configured; return_state=False, always │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  output_projection: Dense(output_dim)│
        │   (omitted when use_projection=False)│
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output  return_sequences=True  → [B, T, output_dim]         │
        │          return_sequences=False → [B, output_dim]            │
        │          use_projection=False   → last dim is the cell's     │
        │                                    output_size instead       │
        └──────────────────────────────────────────────────────────────┘

    **Variants:**

    .. code-block:: text

        tiny    memory  32×16   controller  64   1 read / 1 write   shift 3   lstm
        base    memory 128×20   controller 256   1 read / 1 write   shift 3   lstm
        large   memory 256×64   controller 512   2 read / 2 write   shift 3   lstm

        'base' quotes the paper's memory shape (128 × 20, every row of Tables 1
        and 2); its controller width and both other tiers are this repo's own.
        Shift range is 3 (offsets −1, 0, +1) throughout — see the module docstring.

    :param input_shape: Sequence shape ``(seq_len, input_dim)`` excluding the
        batch dimension. ``seq_len`` may be ``None``. Stored for serialization;
        the layers are built from the shape passed to :meth:`build`.
    :type input_shape: Tuple[Optional[int], int]
    :param output_dim: Width of the final output. Used only when
        ``use_projection=True``.
    :type output_dim: int
    :param output_dim: Dimension of the final output.
    :param config: NTM hyperparameters — an :class:`NTMConfig`, or the dict it
        serializes to (the ``from_config`` path). Both spellings are retained on
        the instance so ``get_config`` round-trips without re-deriving either.
    :type config: Union[NTMConfig, Dict[str, Any]]
    :param return_sequences: Whether to return every timestep's output or only
        the last. Defaults to ``True``.
    :type return_sequences: bool
    :param use_projection: Whether to apply the dense projection to the cell's
        output. With it off, the output width is
        ``controller_dim + num_read_heads * memory_dim`` and ``output_dim`` is
        ignored. Defaults to ``True``.
    :type use_projection: bool
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.

    :raises ValueError: From :meth:`from_variant`, if the variant is unknown.

    Input shape:
        3D tensor with shape ``(batch_size, seq_len, input_dim)``.

    Output shape:
        - ``return_sequences=True``: ``(batch_size, seq_len, output_dim)``.
        - ``return_sequences=False``: ``(batch_size, output_dim)``.
        - ``use_projection=False``: the last dimension is the cell's
          ``output_size`` instead of ``output_dim``.

    Example:
        >>> # From a preset variant
        >>> model = NTMModel.from_variant('base', input_shape=(None, 8),
        ...                               output_dim=8)
        >>>
        >>> # Sequence-to-vector, no projection: output width follows the memory
        >>> model = NTMModel.from_variant('tiny', (20, 8), output_dim=8,
        ...                               return_sequences=False,
        ...                               use_projection=False)
        >>>
        >>> # Overriding an NTM hyperparameter through the variant factory
        >>> model = NTMModel.from_variant('base', (None, 8), 8, num_read_heads=2)

    Note:
        Only ``base`` has a published counterpart, and only for its memory shape.
        ``tiny`` and ``large`` are repo-invented tiers; do not read them as
        reproducing anything from the paper.

    Attributes:
        cell: The ``NTMCell``, named ``ntm_cell``.
        rnn: The ``keras.layers.RNN`` wrapper, named ``ntm_rnn``.
        projection: The output ``Dense``, or ``None`` when disabled.
        config_obj: The resolved :class:`NTMConfig`.
        config_dict: Its dict form, as re-emitted by :meth:`get_config`.
    """

    # Only 'base' has a published counterpart. Graves et al. 2014
    # (https://arxiv.org/abs/1410.5401) Table 1 and Table 2 state the memory shape
    # for EVERY experiment in the paper, and it is 128 x 20 in every single row --
    # the paper varies controller width and head count by task, never N or M. So
    # 'base' quotes memory_size=128, memory_dim=20 from those tables, and its
    # controller_dim=256 does NOT come from them (no LSTM-controller row uses 256;
    # Table 2 uses 100, or 2x100 for Priority Sort) -- it is this repo's choice, kept
    # because a 20-wide memory read by a 100-wide LSTM is not a size the rest of the
    # ladder can be built around. 'tiny' and 'large' are repo-invented tiers with no
    # published counterpart at all; do not read them as reproducing anything.
    # Pinned by tests/test_variant_tables_match_upstream_references.py.
    NTM_VARIANTS = {
        'tiny': {
            'memory_size': 32,
            'memory_dim': 16,
            'controller_dim': 64,
            'num_read_heads': 1,
            'num_write_heads': 1,
            'shift_range': 3,
            'controller_type': 'lstm'
        },
        'base': {
            # DECISION plan-2026-08-23T091307-9a110062/D-462
            # 128 x 20 is the paper's memory shape in all 10 experiment rows across
            # Tables 1 and 2. Do NOT "round up" memory_dim to 32 for a tidier ladder:
            # memory_size here is already the paper's 128, so the two numbers are one
            # quoted pair and changing half of it makes the row cite a shape that
            # appears nowhere in the paper.
            'memory_size': 128,
            'memory_dim': 20,
            'controller_dim': 256,
            'num_read_heads': 1,
            'num_write_heads': 1,
            'shift_range': 3,
            'controller_type': 'lstm'
        },
        'large': {
            'memory_size': 256,
            'memory_dim': 64,
            'controller_dim': 512,
            'num_read_heads': 2,
            'num_write_heads': 2,
            'shift_range': 3,
            'controller_type': 'lstm'
        }
    }

    #: Canonical alias of ``NTM_VARIANTS`` (models/CLAUDE.md Axis 2: "where one
    #: of those is the package's only variant table, add MODEL_VARIANTS as a
    #: class-level alias to the same dict"). An ALIAS, never a rename -- the same
    #: object under both names, so ``from_variant``, ``create_ntm_variant`` and
    #: every existing reader stay on one table.
    MODEL_VARIANTS = NTM_VARIANTS

    def __init__(
            self,
            input_shape: Tuple[Optional[int], int],
            output_dim: int,
            config: Union[NTMConfig, Dict[str, Any]],
            return_sequences: bool = True,
            use_projection: bool = True,
            **kwargs: Any
    ) -> None:
        """Resolve the NTM configuration and create the cell, the RNN and the projection.

        The config is accepted either live or as its serialized dict, so
        ``from_config`` and direct construction take the same path. See the class
        docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        self.input_shape_config = input_shape
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.use_projection = use_projection

        # Configuration handling
        if isinstance(config, dict):
            self.config_obj = NTMConfig.from_dict(config)
            self.config_dict = config
        else:
            self.config_obj = config
            self.config_dict = config.to_dict()

        # Create Layers (Golden Rule: Create all layers in __init__)

        # 1. NTM Cell
        self.cell = NTMCell(self.config_obj, name="ntm_cell")

        # 2. RNN Wrapper
        # We wrap the cell in an RNN layer to handle unrolling
        # return_state=False because we usually don't need raw NTM internal states
        # in the high-level model output
        self.rnn = keras.layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=False,
            name="ntm_rnn"
        )

        # 3. Output Projection
        if self.use_projection:
            self.projection = keras.layers.Dense(
                output_dim,
                name="output_projection"
            )
        else:
            self.projection = None

        # Build the model if input shape is provided (optional but good for summary())
        # Note: Keras 3 models often defer build, but we can hint it here.
        # We generally avoid calling self.build() in __init__ to allow flexibility,
        # but we can set the input spec.

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the RNN and, when enabled, the projection.

        The projection is built at the CELL's output width, not the model input's:
        ``NTMCell.output_size`` is ``controller_dim + num_read_heads * memory_dim``.
        Dense builds on the last axis alone, so one shape serves both the
        sequence and the last-step case.

        :param input_shape: Shape of the input tensor,
            ``(batch, seq_len, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # 1. Build RNN
        self.rnn.build(input_shape)

        # 2. Build Projection
        # The cell output size is defined in NTMCell.output_size
        rnn_output_dim = self.cell.output_size

        if self.use_projection:
            # If return_sequences: (batch, seq_len, rnn_out)
            # Else: (batch, rnn_out)
            # Dense builds on the last dimension
            self.projection.build((None, rnn_output_dim))

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Unroll the cell over the sequence, then project.

        :param inputs: Input tensor of shape ``(batch, seq_len, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the call is in training mode; forwarded to the
            RNN, which passes it down to the cell.
        :type training: Optional[bool]
        :return: The unrolled output, projected when ``use_projection`` is set.
            See the class docstring's Output shape for the four cases.
        :rtype: keras.KerasTensor
        """
        x = self.rnn(inputs, training=training)

        if self.use_projection:
            x = self.projection(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, from configuration alone.

        ``return_sequences`` decides the rank and ``use_projection`` decides the
        last dimension, so all four combinations are covered here.

        :param input_shape: Shape of the input tensor,
            ``(batch, seq_len, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, seq_len, last_dim)`` or ``(batch, last_dim)``, where
            ``last_dim`` is ``output_dim`` or the cell's ``output_size``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        last_dim = self.output_dim if self.use_projection else self.cell.output_size

        if self.return_sequences:
            return (batch_size, seq_len, last_dim)
        else:
            return (batch_size, last_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        The NTM hyperparameters are emitted as the stored dict form, so the cell
        is reconstructed from config rather than serialized as a sub-layer.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_config,
            'output_dim': self.output_dim,
            'config': self.config_dict,
            'return_sequences': self.return_sequences,
            'use_projection': self.use_projection,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NTMModel':
        """Create a model from its configuration.

        No ``NTMConfig`` reconstruction is needed here: ``__init__`` accepts the
        dict form directly. The copy keeps the caller's dict unmodified.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: An ``NTMModel`` instance.
        :rtype: NTMModel
        """
        # NTMConfig reconstruction happens in __init__ via dict check
        # We copy to avoid modifying the original config dict
        config = config.copy()
        return cls(**config)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Tuple[Optional[int], int],
            output_dim: int,
            return_sequences: bool = True,
            **kwargs: Any
    ) -> 'NTMModel':
        """Create an NTM model from a predefined variant.

        ``kwargs`` is split against :class:`NTMConfig`'s dataclass fields: keys
        that name an NTM hyperparameter override the preset before the config is
        built, and everything else is forwarded to the constructor. The preset is
        copied first, so the class-level table is never mutated for later callers.

        :param variant: One of ``'tiny'``, ``'base'``, ``'large'``.
        :type variant: str
        :param input_shape: Input sequence shape ``(seq_len, input_dim)``.
        :type input_shape: Tuple[Optional[int], int]
        :param output_dim: Output width.
        :type output_dim: int
        :param return_sequences: Whether to output the full sequence. Defaults to
            ``True``.
        :type return_sequences: bool
        :param kwargs: Overrides for :class:`NTMConfig` fields (e.g.
            ``controller_type``, ``num_read_heads``) and/or constructor arguments
            (e.g. ``use_projection``, ``name``).
        :type kwargs: Any
        :return: The configured ``NTMModel``.
        :rtype: NTMModel
        :raises ValueError: If ``variant`` is not recognized.
        """
        if variant not in cls.NTM_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.NTM_VARIANTS.keys())}")

        variant_config = cls.NTM_VARIANTS[variant].copy()

        # Separate model arguments from NTMConfig arguments
        ntm_field_names = {f.name for f in dataclasses.fields(NTMConfig)}

        config_overrides = {k: v for k, v in kwargs.items() if k in ntm_field_names}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in ntm_field_names}

        # Update variant defaults with overrides
        variant_config.update(config_overrides)

        ntm_config = NTMConfig(**variant_config)

        return cls(
            input_shape=input_shape,
            output_dim=output_dim,
            config=ntm_config,
            return_sequences=return_sequences,
            **model_kwargs
        )


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_ntm_variant(
        variant: str,
        input_shape: Tuple[Optional[int], int],
        output_dim: int,
        return_sequences: bool = True,
        **kwargs: Any
) -> NTMModel:
    """Convenience function to create an NTM model variant.

    :param variant: One of ``'tiny'``, ``'base'``, ``'large'``.
    :type variant: str
    :param input_shape: Sequence shape ``(seq_len, input_dim)``.
    :type input_shape: Tuple[Optional[int], int]
    :param output_dim: Size of the output vector.
    :type output_dim: int
    :param return_sequences: If ``True``, returns ``(batch, seq, out)``; else
        ``(batch, out)``. Defaults to ``True``.
    :type return_sequences: bool
    :param kwargs: Additional overrides, split between :class:`NTMConfig` fields
        and constructor arguments; see :meth:`NTMModel.from_variant`.
    :type kwargs: Any
    :return: An uncompiled ``NTMModel`` instance.
    :rtype: NTMModel

    Example:
        >>> # Copy-task-shaped model
        >>> model = create_ntm_variant('base', input_shape=(None, 8), output_dim=8)
        >>>
        >>> # Sequence-to-vector classifier head
        >>> model = create_ntm_variant('tiny', (20, 8), output_dim=4,
        ...                            return_sequences=False)
    """
    return NTMModel.from_variant(
        variant=variant,
        input_shape=input_shape,
        output_dim=output_dim,
        return_sequences=return_sequences,
        **kwargs
    )

# ---------------------------------------------------------------------