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
from keras import layers
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
    Neural Turing Machine Model.

    A wrapper around the `NTMCell` that creates a fully unrolled Recurrent Neural Network.
    This model provides a sequence-to-sequence or sequence-to-vector interface
    compatible with standard Keras workflows.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, input_dim])
           ↓
    RNN(NTMCell) -> Unrolls over sequence
           ↓
    (Optional) Dense(output_dim)
           ↓
    Output
    ```

    **Presets**:
    - **tiny**: Small memory (32x16), simple controller, good for unit tests.
    - **base**: Standard NTM (128x32), LSTM controller, robust baseline.
    - **large**: Large memory (256x64), deep controller, for complex tasks.

    Args:
        input_shape: Tuple (seq_len, input_dim). seq_len can be None.
        output_dim: Dimension of the final output.
        config: NTMConfig object or dict defining NTM hyperparameters.
        return_sequences: Whether to return the full sequence or just the last output.
        use_projection: Whether to apply a dense projection to the NTM output.
        **kwargs: Additional arguments for Model base class.
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
        self.rnn = layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=False,
            name="ntm_rnn"
        )

        # 3. Output Projection
        if self.use_projection:
            self.projection = layers.Dense(
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
        """
        Build the model layers with explicit shapes.

        Args:
            input_shape: Shape of the input tensor (batch, seq_len, dim).
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
        """
        Forward pass.

        Args:
            inputs: Input tensor (batch, seq_len, input_dim).
            training: Whether to run in training mode (affects Dropout/RNN).

        Returns:
            Output tensor.
        """
        x = self.rnn(inputs, training=training)

        if self.use_projection:
            x = self.projection(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on configuration."""
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        last_dim = self.output_dim if self.use_projection else self.cell.output_size

        if self.return_sequences:
            return (batch_size, seq_len, last_dim)
        else:
            return (batch_size, last_dim)

    def get_config(self) -> Dict[str, Any]:
        """Serialize configuration."""
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
        """Deserialize configuration."""
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
        """
        Create NTM model from a predefined variant.

        Args:
            variant: One of 'tiny', 'base', 'large'.
            input_shape: Input sequence shape (seq_len, dim).
            output_dim: Output dimension.
            return_sequences: Whether to output full sequence.
            **kwargs: Overrides for specific config parameters (e.g., controller_type).

        Returns:
            Configured NTMModel instance.
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
    """
    Factory function to create an NTM model variant.

    Args:
        variant: One of 'tiny', 'base', 'large'.
        input_shape: Tuple (seq_len, input_dim).
        output_dim: Size of output vector.
        return_sequences: If True, returns (batch, seq, out). Else (batch, out).
        **kwargs: Additional overrides for NTM configuration.

    Returns:
        An uncompiled NTMModel instance.
    """
    return NTMModel.from_variant(
        variant=variant,
        input_shape=input_shape,
        output_dim=output_dim,
        return_sequences=return_sequences,
        **kwargs
    )