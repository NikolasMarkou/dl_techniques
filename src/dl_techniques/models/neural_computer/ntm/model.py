"""
Neural Turing Machine model wrapper. ``NTMModel`` unrolls an ``NTMCell`` over a
sequence with ``keras.layers.RNN`` and projects its output through an optional
dense layer.

A plain recurrent network stores everything in one fixed-size hidden vector, so
memory and computation share the same resource. The NTM splits them: a
modest-width controller addresses an external ``N x M`` memory matrix through
differentiable read and write heads, so capacity grows by adding memory slots
instead of controller width, and a slot written at step 3 stays present at step
300 unless a write head erases it. Each head builds its weighting over the ``N``
slots in four soft steps: content lookup by cosine similarity, interpolation
with the head's previous position, a circular shift for relative movement, and
a sharpening step that undoes the blur the shift introduces.

``return_state=False`` on the RNN wrapper is fixed: the cell's state carries the
full memory matrix, and this keeps that large tensor out of the model's output
and out of every `fit()` metric path. Only the ``base`` variant's memory shape
(128 x 20) is quoted from the paper; the controller width and the ``tiny`` and
``large`` tiers are this repo's own.

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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# NTM Model
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.ntm.model")
class NTMModel(keras.Model):
    """
    Neural Turing Machine: an ``NTMCell`` unrolled by ``keras.layers.RNN``.

    Wraps the cell in an ``RNN`` layer to give a sequence-to-sequence or
    sequence-to-vector interface. The cell's own output is the controller output
    concatenated with the freshly read vectors, width
    ``controller_dim + num_read_heads * memory_dim``, which depends on the memory
    configuration rather than the task; the optional dense projection maps that
    down to ``output_dim``.

    ``return_state=False`` on the ``RNN`` is not exposed as an option: the state
    tuple carries the full memory matrix, and surfacing it from the model's
    outputs would put a large bookkeeping tensor in every ``fit()`` metric path.

    Architecture:

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

    Variants:

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

    # 'base' quotes memory_size=128, memory_dim=20 from Graves et al. 2014 Tables 1-2.
    # controller_dim=256 and the 'tiny'/'large' tiers have no published counterpart.
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
            # DECISION plan-2026-08-23T091307-9a110062/D-462: memory_size and memory_dim
            # form one quoted pair (128 x 20, Graves et al. Tables 1-2); do not round
            # memory_dim to 32 alone. See decisions.md.
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

    #: Alias of ``NTM_VARIANTS``, the same dict under both names.
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

        if isinstance(config, dict):
            self.config_obj = NTMConfig.from_dict(config)
            self.config_dict = config
        else:
            self.config_obj = config
            self.config_dict = config.to_dict()

        self.cell = NTMCell(self.config_obj, name="ntm_cell")

        self.rnn = keras.layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=False,
            name="ntm_rnn"
        )

        if self.use_projection:
            self.projection = keras.layers.Dense(
                output_dim,
                name="output_projection"
            )
        else:
            self.projection = None

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
        self.rnn.build(input_shape)

        rnn_output_dim = self.cell.output_size

        if self.use_projection:
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