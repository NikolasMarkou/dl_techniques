"""CliffordUNet causal language model.

Combines AU-Net's hierarchical U-Net architecture (arXiv:2506.14761) with
CliffordNet's geometric algebra blocks (arXiv:2601.06793v2) for causal
language modeling.

The model processes token sequences through a contracting path of
:class:`CausalCliffordNetBlock` stages at increasing channel widths with
causal window pooling, then an expanding path with multi-linear upsampling
and residual skip connections.  Deeper stages operate on shorter, coarser
sequences — amortising compute — while the full-resolution byte/token
level is preserved at the finest stage.

Architecture:

.. code-block:: text

    Input IDs (B, seq_len)
         |
         v
    Token Embedding + Positional Embedding
    -> LayerNorm -> Dropout
         |
         v  Reshape to (B, 1, seq_len, D0)
    +============================+
    | Encoder Stage 0 (D0)       |---> skip0
    +============================+
         | CausalWindowPool (factor k0)
         v
    +============================+
    | Encoder Stage 1 (D1)       |---> skip1
    +============================+
         | CausalWindowPool (factor k1)
         v
    +============================+
    | Encoder Stage 2 / Bottleneck (D2)  |
    +============================+
         | MultiLinearUpsample (factor k1)
         v
    +============================+
    | + skip1                    |
    | Decoder Stage 1 (D1)      |
    +============================+
         | MultiLinearUpsample (factor k0)
         v
    +============================+
    | + skip0                    |
    | Decoder Stage 0 (D0)      |
    +============================+
         |
         v  Squeeze to (B, seq_len, D0)
    LayerNorm -> Dropout -> Dense(vocab_size)
         |
         v
    Logits (B, seq_len, vocab_size)

Causal invariants:

1. CausalCliffordNetBlocks use left-only padded depthwise convolutions
   so position *i* only sees positions ``<= i``.
2. CausalWindowPool selects the **last** element of each window of
   size *k*.  The last position has causal access to all earlier
   positions through prior block processing.
3. MultiLinearUpsample is *strictly causal*: each fine slot is sourced
   from the previous coarse window (slots ``0..k-2``) or from its own
   coarse vector (slot ``k-1``, the anchor).  See the layer docstring
   for the derivation.
4. Skip connections are residual additions (not concatenations).

References:
    Videau, M., et al. (2025). From Bytes to Ideas: Language Modeling
    with Autoregressive U-Nets.  arXiv:2506.14761.

    Brandstetter, J., et al. (2025). CliffordNet: All You Need is
    Geometric Algebra.  arXiv:2601.06793v2.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import keras
from keras import initializers, regularizers

from dl_techniques.utils.logger import logger
from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CausalCliffordNetBlock,
    GatedGeometricResidual,
)

# ---------------------------------------------------------------------------

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Linearly spaced drop-path rates from 0 to ``max_rate``."""
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques")
class CausalWindowPool(keras.layers.Layer):
    """Causal window pooling via last-element selection and projection.

    For each non-overlapping window of ``pool_size`` positions, selects the
    **last** element (which has causal access to all prior positions through
    upstream causal processing) and projects it to the target dimension.

    Input shape:  ``(B, 1, L, D_in)``  where ``L`` is divisible by ``pool_size``.
    Output shape: ``(B, 1, L // pool_size, d_out)``.

    :param d_out: Output channel dimension.
    :param pool_size: Window size for pooling.
    :param kernel_initializer: Initializer for the projection kernel.
    """

    def __init__(
        self,
        d_out: int,
        pool_size: int,
        kernel_initializer: Any = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_out = d_out
        self.pool_size = pool_size
        self._kernel_initializer_cfg = kernel_initializer
        self.proj = keras.layers.Dense(
            d_out,
            kernel_initializer=kernel_initializer,
            name="pool_proj",
        )

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        # x: (B, 1, L, D_in)
        shape = keras.ops.shape(x)
        k = self.pool_size
        # Reshape to (B, 1, L//k, k, D_in), take last element per window
        x = keras.ops.reshape(x, (shape[0], 1, shape[2] // k, k, shape[3]))
        x = x[:, :, :, -1, :]  # (B, 1, L//k, D_in)
        return self.proj(x)     # (B, 1, L//k, d_out)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        B, H, L, _ = input_shape
        new_L = L // self.pool_size if L is not None else None
        return (B, H, new_L, self.d_out)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_out": self.d_out,
            "pool_size": self.pool_size,
            "kernel_initializer": initializers.serialize(
                initializers.get(self._kernel_initializer_cfg),
            ),
        })
        return config


@keras.saving.register_keras_serializable(package="dl_techniques")
class MultiLinearUpsample(keras.layers.Layer):
    """Strictly causal position-specific linear upsampling.

    Expands each coarse vector into ``factor`` fine vectors using
    position-specific linear maps, but with a one-window left shift so that
    no fine output position depends on a coarse vector representing future
    tokens.

    Causal sourcing rule (``j`` = coarse index, ``i`` = fine slot in
    ``[0, factor)``)::

        fine[j*factor + i] = coarse[j-1] @ W_i + b_i,   if i <  factor - 1
        fine[j*factor + i] = coarse[j]   @ W_i + b_i,   if i == factor - 1

    where ``coarse[-1]`` is treated as a zero vector.

    Why this is causal
    ------------------
    Upstream the encoder uses :class:`CausalCliffordNetBlock` (left-padded
    depthwise convolution) and :class:`CausalWindowPool` (last-element
    selection).  Therefore coarse position ``j`` deterministically encodes
    information from fine positions ``0 .. (j+1)*factor - 1``: that is, its
    own anchor is the last fine position of window ``j``.

    For a fine output position ``t = j*factor + i``:

    * If ``i == factor - 1`` (the anchor slot), the source ``coarse[j]``
      represents fine positions ``0..t``.  No future leakage.
    * If ``i <  factor - 1`` (a non-anchor slot), the source ``coarse[j-1]``
      represents fine positions ``0..j*factor - 1`` ``< t``.  No future
      leakage; the slot loses access to its own coarse-window summary, but
      the residual skip connection from the same-resolution encoder stage
      restores it through a strictly-causal path.

    Each fine slot still has its own learnable projection ``(W_i, b_i)``,
    so the per-slot expressivity of the AU-Net design is preserved.  The
    only thing the fix removes is the implicit "I can read my own future"
    that the naive (non-shifted) version provided.

    Note:
        The naive (non-causal) version of this layer leaks up to
        ``factor - 1`` future tokens into every non-anchor position.  When
        composed across a U-Net with multiple pooling stages the leakage
        compounds to ``∏ pool_sizes - 1`` future tokens at the bottom
        decoder stage.  See
        ``analyses/analysis_2026-04-14_92ce16b9/summary.md`` for the
        derivation and the empirical signature (training loss collapses,
        generation degenerates into token soup).

    Input shape:  ``(B, 1, L_coarse, D_in)``
    Output shape: ``(B, 1, L_coarse * factor, d_out)``

    :param d_out: Output channel dimension.
    :param factor: Expansion factor (number of fine positions per coarse).
    :param kernel_initializer: Initializer for the projection kernels.
    """

    def __init__(
        self,
        d_out: int,
        factor: int,
        kernel_initializer: Any = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        self.d_out = d_out
        self.factor = factor
        self._kernel_initializer_cfg = kernel_initializer

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        d_in = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.factor, d_in, self.d_out),
            initializer=self._kernel_initializer_cfg,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.factor, self.d_out),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        # x: (B, 1, L_coarse, D_in)
        k = self.factor

        if k == 1:
            # No upsampling — single causal projection (slot is its own anchor).
            return (
                keras.ops.einsum("bhld,do->bhlo", x, self.kernel[0])
                + self.bias[0]
            )

        # Shifted-coarse construction: pad one zero coarse vector on the
        # left so that non-anchor slots are sourced from the *previous*
        # coarse window. This is what makes the upsample strictly causal.
        coarse_padded = keras.ops.pad(
            x, [[0, 0], [0, 0], [1, 0], [0, 0]],
        )                                              # (B, 1, L+1, D_in)
        coarse_prev = coarse_padded[:, :, :-1, :]      # (B, 1, L, D_in)
        coarse_curr = x                                # (B, 1, L, D_in)

        # Slots 0 .. k-2 read from the previous coarse window.
        kernel_prev = self.kernel[:-1]                 # (k-1, D_in, d_out)
        bias_prev = self.bias[:-1]                     # (k-1, d_out)
        fine_prev = keras.ops.einsum(
            "bhld,fdo->bhlfo", coarse_prev, kernel_prev,
        ) + bias_prev                                  # (B, 1, L, k-1, d_out)

        # Slot k-1 (the anchor slot) reads from its own coarse vector.
        fine_last = keras.ops.einsum(
            "bhld,do->bhlo", coarse_curr, self.kernel[-1],
        ) + self.bias[-1]                              # (B, 1, L, d_out)
        fine_last = keras.ops.expand_dims(fine_last, axis=3)
        # fine_last: (B, 1, L, 1, d_out)

        fine = keras.ops.concatenate([fine_prev, fine_last], axis=3)
        # fine: (B, 1, L, k, d_out)
        shape = keras.ops.shape(fine)
        return keras.ops.reshape(
            fine, (shape[0], shape[1], shape[2] * k, self.d_out),
        )

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        B, H, L, _ = input_shape
        new_L = L * self.factor if L is not None else None
        return (B, H, new_L, self.d_out)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_out": self.d_out,
            "factor": self.factor,
            "kernel_initializer": initializers.serialize(
                initializers.get(self._kernel_initializer_cfg),
            ),
        })
        return config


# ---------------------------------------------------------------------------
# Clifford-native helper layers (the redesign)
# ---------------------------------------------------------------------------
#
# These three layers replace the legacy ``CausalWindowPool`` /
# ``MultiLinearUpsample`` / raw-additive skip merge used by the legacy
# ``CliffordUNetLM``.  They keep the channel dimension constant and operate
# entirely through sparse rolling geometric products, so the full forward
# pass of the redesigned :class:`CliffordUNetLM` lives in **one** Clifford
# algebra from embedding to logits.  The legacy classes above are kept only
# because :mod:`dl_techniques.models.cliffordunet.draft` still imports them.


@keras.saving.register_keras_serializable(package="dl_techniques")
class GeometricStridePool(keras.layers.Layer):
    """Strictly causal Clifford-equivariant downsampling along the sequence axis.

    This is the sequence-axis dual of
    :class:`~dl_techniques.layers.geometric.clifford_block.SparseRollingGeometricProduct`.
    Where the block-level rolling product mixes channels of two same-shape
    streams via a wedge/inner product at offsets ``shifts``, this layer mixes
    the ``k = pool_size`` consecutive fine positions of each window with
    ``k`` learned per-slot multivectors via the same wedge/inner product, sums
    the per-slot contributions, then projects back to ``D``::

        out[j] = proj( sum_{i=0..k-1}  GP_{shifts}( v[j*k + i],  W_i ) )

    where ``GP_{shifts}(a, b)`` denotes the per-shift wedge / inner-product
    components ``[a * roll(b, s) - b * roll(a, s),  silu(a * roll(b, s))]``.

    The output has **the same channel dimensionality ``D``** as the input,
    so the Clifford algebra is preserved across the resampling.  No
    ``Dense`` shortcut, no last-element-and-discard-the-rest pooling — the
    whole window contributes through geometric products.

    Causality
    ---------
    Output coarse position ``j`` is a function of fine positions
    ``j*k .. j*k + k - 1`` only.  The "anchor" fine position of window
    ``j`` is ``j*k + k - 1`` (the latest position summarised), matching the
    convention that the legacy :class:`CausalWindowPool` and the new
    :class:`GeometricStrideUnpool` use.

    :param channels: Feature dimension ``D`` (constant in/out).
    :param pool_size: Window size ``k``. Must divide the input sequence length.
    :param shifts: Channel-shift offsets for the per-slot rolling product.
    :param cli_mode: ``"inner"`` / ``"wedge"`` / ``"full"``.
    :param use_bias: Whether the projection ``Dense`` uses a bias.
    :param kernel_initializer: Initializer for ``W_i`` and the projection.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    """

    def __init__(
        self,
        channels: int,
        pool_size: int,
        shifts: List[int],
        cli_mode: CliMode = "full",
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        if cli_mode not in ("inner", "wedge", "full"):
            raise ValueError(
                f"cli_mode must be 'inner'/'wedge'/'full', got {cli_mode!r}"
            )
        filtered_shifts = [s for s in shifts if 0 < s < channels]
        if not filtered_shifts:
            raise ValueError(
                f"All shifts {shifts} are out of range for channels={channels}"
            )

        self.channels = channels
        self.pool_size = pool_size
        self.shifts = filtered_shifts
        self.cli_mode = cli_mode
        self.use_bias = use_bias
        self._kernel_initializer_cfg = kernel_initializer
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        multiplier = 2 if cli_mode == "full" else 1
        self._proj_input_dim = multiplier * len(self.shifts) * channels

        self.proj = keras.layers.Dense(
            channels,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="pool_proj",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # input_shape: (B, 1, L, D)
        d_in = input_shape[-1]
        if d_in is not None and d_in != self.channels:
            raise ValueError(
                f"GeometricStridePool requires d_in == channels "
                f"({self.channels}), got d_in={d_in}"
            )
        self.W = self.add_weight(
            name="slot_multivectors",
            shape=(self.pool_size, self.channels),
            initializer=self._kernel_initializer_cfg,
        )
        # The Dense projection sees a tensor of shape
        # (B, 1, L_out, proj_input_dim) after the per-slot sum.
        self.proj.build((*input_shape[:-2], None, self._proj_input_dim))
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        k = self.pool_size
        D = self.channels
        s_in = keras.ops.shape(x)
        B, H, L = s_in[0], s_in[1], s_in[2]
        L_out = L // k

        # Window the sequence: (B, 1, L_out, k, D)
        x_win = keras.ops.reshape(x, (B, H, L_out, k, D))

        # Broadcast the slot multivectors.
        W_b = keras.ops.reshape(self.W, (1, 1, 1, k, D))

        components: List[keras.KerasTensor] = []
        for s in self.shifts:
            x_s = keras.ops.roll(x_win, shift=s, axis=-1)   # rolled along channel
            W_s = keras.ops.roll(self.W, shift=s, axis=-1)
            W_s_b = keras.ops.reshape(W_s, (1, 1, 1, k, D))

            if self.cli_mode in ("wedge", "full"):
                wedge = x_win * W_s_b - W_b * x_s
                components.append(wedge)
            if self.cli_mode in ("inner", "full"):
                dot = keras.activations.silu(x_win * W_s_b)
                components.append(dot)

        # Per-slot interaction features: (B, 1, L_out, k, M*D)
        g_per_slot = keras.ops.concatenate(components, axis=-1)

        # Aggregate slots inside each window.
        # The sum is a valid multivector operation (Clifford addition).
        g_summed = keras.ops.sum(g_per_slot, axis=3)        # (B, 1, L_out, M*D)

        return self.proj(g_summed)                           # (B, 1, L_out, D)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        B, H, L, _ = input_shape
        new_L = L // self.pool_size if L is not None else None
        return (B, H, new_L, self.channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "pool_size": self.pool_size,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


@keras.saving.register_keras_serializable(package="dl_techniques")
class GeometricStrideUnpool(keras.layers.Layer):
    """Strictly causal Clifford-equivariant upsampling along the sequence axis.

    Inverse companion of :class:`GeometricStridePool`.  Each coarse vector
    is expanded into ``factor`` fine vectors via per-slot geometric products
    with learned multivectors ``W_0 .. W_{factor-1}``.  To preserve strict
    causality the *non-anchor* slots ``i < factor - 1`` of every coarse
    window are sourced from the **previous** coarse vector — so they cannot
    leak future fine-token information into past positions.  The anchor
    slot ``i == factor - 1`` is sourced from its own coarse vector::

        fine[j*k + i] = proj( GP_{shifts}( coarse[j-1],  W_i ) )   if i <  k-1
        fine[j*k + k-1] = proj( GP_{shifts}( coarse[j],   W_{k-1} ) )

    where ``coarse[-1]`` is treated as a zero vector.  ``GP_{shifts}`` is
    the same wedge/inner sparse rolling product that
    :class:`GeometricStridePool` uses (see that docstring for the formula).

    Constant channel dimensionality
    -------------------------------
    Input ``D`` and output ``D`` are equal — the algebra carries through.
    This is the *Clifford-native* upsample; the legacy
    :class:`MultiLinearUpsample` is a per-slot **linear** map and was kept
    in this file only because :mod:`dl_techniques.models.cliffordunet.draft`
    still imports it.

    Why this is causal — derivation
    -------------------------------
    Upstream, :class:`GeometricStridePool` ensures that coarse position
    ``j`` summarises fine positions ``0 .. (j+1)*k - 1`` (its anchor is
    fine position ``j*k + k - 1``).  For a fine output position
    ``t = j*k + i``:

    * ``i == k - 1`` (anchor): source is ``coarse[j]`` → fine positions
      ``0..t``.  No future leakage.
    * ``i  <  k - 1``        : source is ``coarse[j-1]`` → fine positions
      ``0 .. j*k - 1`` ``< t``.  No future leakage.

    The non-anchor slots lose access to *their own* coarse-window summary,
    but the residual skip from the same-resolution encoder stage (merged
    via :class:`GGRMerge`) restores it through a strictly-causal path.

    Input shape:  ``(B, 1, L_coarse, D)``
    Output shape: ``(B, 1, L_coarse * factor, D)``

    :param channels: Feature dimension ``D`` (constant in/out).
    :param factor: Expansion factor ``k``.
    :param shifts: Channel-shift offsets for the per-slot rolling product.
    :param cli_mode: ``"inner"`` / ``"wedge"`` / ``"full"``.
    :param use_bias: Whether the projection ``Dense`` uses a bias.
    :param kernel_initializer: Initializer for ``W_i`` and the projection.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    """

    def __init__(
        self,
        channels: int,
        factor: int,
        shifts: List[int],
        cli_mode: CliMode = "full",
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        if cli_mode not in ("inner", "wedge", "full"):
            raise ValueError(
                f"cli_mode must be 'inner'/'wedge'/'full', got {cli_mode!r}"
            )
        filtered_shifts = [s for s in shifts if 0 < s < channels]
        if not filtered_shifts:
            raise ValueError(
                f"All shifts {shifts} are out of range for channels={channels}"
            )

        self.channels = channels
        self.factor = factor
        self.shifts = filtered_shifts
        self.cli_mode = cli_mode
        self.use_bias = use_bias
        self._kernel_initializer_cfg = kernel_initializer
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        multiplier = 2 if cli_mode == "full" else 1
        self._proj_input_dim = multiplier * len(self.shifts) * channels

        self.proj = keras.layers.Dense(
            channels,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="unpool_proj",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        d_in = input_shape[-1]
        if d_in is not None and d_in != self.channels:
            raise ValueError(
                f"GeometricStrideUnpool requires d_in == channels "
                f"({self.channels}), got d_in={d_in}"
            )
        self.W = self.add_weight(
            name="slot_multivectors",
            shape=(self.factor, self.channels),
            initializer=self._kernel_initializer_cfg,
        )
        # Dense applied to the last dim of a (B, 1, L_coarse, k, M*D) tensor.
        self.proj.build((*input_shape[:-1], self.factor, self._proj_input_dim))
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        k = self.factor
        D = self.channels
        s_in = keras.ops.shape(x)
        B, H, L = s_in[0], s_in[1], s_in[2]

        if k == 1:
            # No upsampling — single (anchor) slot per coarse position.
            src = keras.ops.expand_dims(x, axis=3)              # (B,1,L,1,D)
        else:
            # Causal-shift trick: slots 0..k-2 read coarse[j-1],
            # slot k-1 reads coarse[j].  coarse[-1] is the implicit zero
            # vector introduced by the left pad.
            coarse_padded = keras.ops.pad(
                x, [[0, 0], [0, 0], [1, 0], [0, 0]],
            )                                                    # (B,1,L+1,D)
            coarse_prev = coarse_padded[:, :, :-1, :]            # (B,1,L,D)
            cp_e = keras.ops.expand_dims(coarse_prev, axis=3)    # (B,1,L,1,D)
            cp_r = keras.ops.repeat(cp_e, k - 1, axis=3)         # (B,1,L,k-1,D)
            cc_e = keras.ops.expand_dims(x, axis=3)              # (B,1,L,1,D)
            src = keras.ops.concatenate([cp_r, cc_e], axis=3)    # (B,1,L,k,D)

        # Same per-slot GP machinery as GeometricStridePool.
        W_b = keras.ops.reshape(self.W, (1, 1, 1, k, D))

        components: List[keras.KerasTensor] = []
        for s in self.shifts:
            src_s = keras.ops.roll(src, shift=s, axis=-1)
            W_s = keras.ops.roll(self.W, shift=s, axis=-1)
            W_s_b = keras.ops.reshape(W_s, (1, 1, 1, k, D))

            if self.cli_mode in ("wedge", "full"):
                components.append(src * W_s_b - W_b * src_s)
            if self.cli_mode in ("inner", "full"):
                components.append(keras.activations.silu(src * W_s_b))

        g_per_slot = keras.ops.concatenate(components, axis=-1)
        # Per-slot projection back to D (Dense acts on the last axis).
        fine_per_slot = self.proj(g_per_slot)                    # (B,1,L,k,D)

        return keras.ops.reshape(fine_per_slot, (B, H, L * k, D))

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        B, H, L, _ = input_shape
        new_L = L * self.factor if L is not None else None
        return (B, H, new_L, self.channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "factor": self.factor,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


@keras.saving.register_keras_serializable(package="dl_techniques")
class GGRMerge(keras.layers.Layer):
    """Skip-junction merge using a Gated Geometric Residual.

    Replaces the raw additive ``decoder + skip`` of the legacy
    :class:`CliffordUNetLM` with a gated geometric residual that lives in
    the same algebra as the surrounding Clifford blocks::

        merged = decoder + GGR( LayerNorm(decoder),  skip )

    :class:`~dl_techniques.layers.geometric.clifford_block.GatedGeometricResidual`
    learns a per-channel sigmoid gate ``alpha = sigmoid(Dense([decoder_norm; skip]))``
    and outputs the LayerScale-attenuated residual term
    ``gamma * (silu(decoder_norm) + alpha * skip)``, which the caller adds
    to the decoder stream.  This is exactly the same machinery the block
    already uses for its own internal residual; reusing it at the skip
    junction means *both* fine and coarse streams are merged inside the
    Clifford block's gating regime instead of by an unweighted sum.

    Input/Output shape: both ``decoder`` and ``skip`` are
    ``(B, 1, L, D)``; output is ``(B, 1, L, D)``.

    :param channels: Feature dimension ``D``.
    :param layer_scale_init: Initial LayerScale ``gamma`` (passed to GGR).
    :param drop_path_rate: Stochastic-depth probability inside GGR.
    :param kernel_initializer: Initializer for the GGR gate kernel.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    """

    def __init__(
        self,
        channels: int,
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.channels = channels
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self._kernel_initializer_cfg = kernel_initializer
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="merge_norm",
        )
        self.ggr = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="merge_ggr",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # input_shape is the decoder shape; skip has the same shape.
        self.norm.build(input_shape)
        self.ggr.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        decoder: keras.KerasTensor,
        skip: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        decoder_norm = self.norm(decoder, training=training)
        residual = self.ggr(decoder_norm, skip, training=training)
        return decoder + residual

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "layer_scale_init": self.layer_scale_init,
            "drop_path_rate": self.drop_path_rate,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------------
# CliffordUNet Language Model
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques")
class CliffordUNetLM(keras.Model):
    """Clifford-native U-Net language model.

    A redesign of the original CliffordUNetLM in which the U-Net adapts to
    the Clifford block, not the other way around.  The architecture is
    deliberately built so that **one Clifford algebra spans the entire
    forward pass** — from the token embedding to the vocabulary logits.

    Five rules the architecture obeys
    ---------------------------------
    1. **Constant channel dimension ``D``** through every stage (encoder,
       bottleneck, decoder).  No per-stage widening — that single design
       choice was what destroyed the algebraic continuity of the legacy
       version, because there is no canonical embedding from a Clifford
       algebra of dim ``D_i`` into one of dim ``D_{i+1}``.
    2. **Pool and upsample are themselves Clifford operations.** The
       contracting path uses :class:`GeometricStridePool`, the expanding
       path uses :class:`GeometricStrideUnpool`.  Both keep ``D`` constant
       and operate via sparse rolling geometric products against learned
       per-slot multivectors — exactly the operator the block uses
       internally on the channel axis, but applied on the sequence axis.
    3. **Skip merge runs through :class:`GGRMerge`**, which wraps the same
       :class:`~dl_techniques.layers.geometric.clifford_block.GatedGeometricResidual`
       the block uses for its own residual.  The two streams are merged
       inside the algebra with a learnable gate, not by raw addition of
       two tensors that happen to share a shape.
    4. **Channel-axis multi-scale (per-stage ``shifts``) and sequence-axis
       multi-scale (``pool_sizes``) are scheduled together.**  Fine stages
       use narrow ``shifts`` because they already have full sequence
       resolution; deeper stages widen ``shifts`` because they operate on
       coarse summaries that must fold in long-range channel structure.
    5. **The U-Net's only justification is sequence receptive field.**  At
       constant ``D`` the param count is bought by ``D`` and depth, so the
       hierarchy must contribute receptive field or it shouldn't exist.
       With two pools by 4 and a kernel-3 causal conv per block the
       bottleneck token comfortably sees the entire 512-token context.

    Architecture::

        Input IDs (B, L)
            │
            ▼
        Token embed + Position embed → LayerNorm → Dropout
            │      [D constant from here through to logits]
            ▼  reshape to (B, 1, L_internal, D)
        ─── Encoder Stage 0 ─────────────────────────────────► skip_0
            │   depth_0 × CausalCliffordNetBlock(D, shifts=S_0)
            ▼
        GeometricStridePool(k_0, shifts=pool_shifts)
            │
        ─── Encoder Stage 1 ─────────────────────────────────► skip_1
            │   depth_1 × CausalCliffordNetBlock(D, shifts=S_1)
            ▼
        GeometricStridePool(k_1, shifts=pool_shifts)
            │
        ─── Bottleneck ──────────────────────────────────────
            │   depth_{n-1} × CausalCliffordNetBlock(D, shifts=S_{n-1})
            ▼
        GeometricStrideUnpool(k_1, shifts=pool_shifts)
            │
        GGRMerge(skip_1, ↑)
            │
        ─── Decoder Stage 1 ─────────────────────────────────
            │   dec_1 × CausalCliffordNetBlock(D, shifts=S_1)
            ▼
        GeometricStrideUnpool(k_0, shifts=pool_shifts)
            │
        GGRMerge(skip_0, ↑)
            │
        ─── Decoder Stage 0 ─────────────────────────────────
            │   dec_0 × CausalCliffordNetBlock(D, shifts=S_0)
            ▼
        squeeze(H) → crop pad → LayerNorm → Dropout → Dense(vocab)
            │
            ▼
        Logits (B, L, vocab_size)

    Causality
    ---------
    Strictly causal end-to-end, by composition.  All three new layers are
    Clifford-equivariant and individually causal:

    * :class:`CausalCliffordNetBlock` uses left-only padded depthwise
      convolutions (block-level causality).
    * :class:`GeometricStridePool` aggregates intra-window fine positions
      ``j*k .. j*k + k - 1`` so coarse position ``j`` only depends on fine
      positions ``≤ (j+1)*k - 1`` (its anchor).
    * :class:`GeometricStrideUnpool` reads ``coarse[j-1]`` for non-anchor
      slots and ``coarse[j]`` only for the anchor slot, so fine output
      ``t = j*k + i`` only depends on fine positions ``≤ t``.
    * :class:`GGRMerge` adds a position-local residual (no sequence-axis
      mixing inside the merge).

    The bit-identical causality test
    (``model([t0..tn, real_future]) == model([t0..tn, pad…])`` at positions
    ``0..n``) passes — see ``analyses/analysis_2026-04-14_92ce16b9/`` for
    the discussion of why this matters.

    :param vocab_size: Vocabulary size (including special tokens).
    :param max_seq_length: Maximum sequence length for positional embeddings.
    :param channels: **Constant** feature dimension ``D`` (single int).
    :param encoder_depths: Blocks per encoder stage. ``len = n_stages``.
    :param decoder_depths: Blocks per decoder stage. ``len = n_stages - 1``.
    :param pool_sizes: Pool factor between adjacent stages. ``len = n_stages - 1``.
    :param stage_shifts: Per-stage channel-shift schedule for the
        :class:`CausalCliffordNetBlock` instances. ``len = n_stages``.
        Decoder stages reuse the encoder list at the same depth.
    :param pool_shifts: Channel-shift schedule used inside every
        :class:`GeometricStridePool` and :class:`GeometricStrideUnpool`
        layer. Defaults to ``[1, 2]``.
    :param cli_mode: ``"inner"`` / ``"wedge"`` / ``"full"``.
    :param ctx_mode: ``"diff"`` / ``"abs"`` (passed through to the block).
    :param use_global_context: Add the block's global context branch.
    :param layer_scale_init: Initial LayerScale gamma value.
    :param stochastic_depth_rate: Maximum DropPath rate (linear schedule).
    :param dropout_rate: Embedding and pre-output dropout rate.
    :param kernel_initializer: Initializer for all dense / projection layers.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.

    Example::

        model = CliffordUNetLM.from_variant("nano", vocab_size=50261)
        input_ids = keras.random.uniform((2, 64), 0, 50261, dtype="int32")
        outputs = model(input_ids)
        print(outputs["logits"].shape)  # (2, 64, 50261)
    """

    LAYERNORM_EPSILON: float = 1e-6

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # ~25M params (most parameters in the embedding tables)
        "nano": dict(
            channels=192,
            encoder_depths=[3, 4, 3],
            decoder_depths=[3, 3],
            pool_sizes=[4, 4],
            stage_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8]],
            pool_shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.05,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        # ~50M params
        "mini": dict(
            channels=256,
            encoder_depths=[4, 6, 4],
            decoder_depths=[4, 4],
            pool_sizes=[4, 4],
            stage_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8]],
            pool_shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        # ~110M params, 4-stage hierarchy with a final pool of 2
        "base": dict(
            channels=384,
            encoder_depths=[4, 6, 8, 4],
            decoder_depths=[6, 4, 4],
            pool_sizes=[4, 4, 2],
            stage_shifts=[
                [1, 2],
                [1, 2, 4],
                [1, 2, 4, 8],
                [1, 2, 4, 8, 16],
            ],
            pool_shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.15,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        channels: int = 192,
        encoder_depths: Optional[List[int]] = None,
        decoder_depths: Optional[List[int]] = None,
        pool_sizes: Optional[List[int]] = None,
        stage_shifts: Optional[List[List[int]]] = None,
        pool_shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(
                f"channels must be a positive int, got {channels!r}"
            )

        encoder_depths = list(encoder_depths or [3, 4, 3])
        decoder_depths = list(decoder_depths or [3, 3])
        pool_sizes = list(pool_sizes or [4, 4])
        if stage_shifts is None:
            stage_shifts = [[1, 2], [1, 2, 4], [1, 2, 4, 8]]
        stage_shifts = [list(s) for s in stage_shifts]
        pool_shifts = list(pool_shifts or [1, 2])

        n_stages = len(encoder_depths)
        if len(decoder_depths) != n_stages - 1:
            raise ValueError(
                f"decoder_depths length ({len(decoder_depths)}) must be "
                f"encoder_depths length - 1 ({n_stages - 1})"
            )
        if len(pool_sizes) != n_stages - 1:
            raise ValueError(
                f"pool_sizes length ({len(pool_sizes)}) must be "
                f"encoder_depths length - 1 ({n_stages - 1})"
            )
        if len(stage_shifts) != n_stages:
            raise ValueError(
                f"stage_shifts length ({len(stage_shifts)}) must match "
                f"encoder_depths length ({n_stages})"
            )

        # Store config for serialization
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.channels = channels
        self.encoder_depths = encoder_depths
        self.decoder_depths = decoder_depths
        self.pool_sizes = pool_sizes
        self.stage_shifts = stage_shifts
        self.pool_shifts = pool_shifts
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.kernel_initializer = (
            initializers.get(kernel_initializer)
            if kernel_initializer else None
        )
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self._n_stages = n_stages

        # Right-pad input sequence length to a multiple of the product of
        # pool sizes so every pool sees an exact integer-length window.
        total_factor = math.prod(pool_sizes) if pool_sizes else 1
        self._total_pool_factor = total_factor
        self._internal_len = (
            ((max_seq_length + total_factor - 1) // total_factor)
            * total_factor
        )

        D = channels

        # --- Embeddings ---
        self.token_embedding = keras.layers.Embedding(
            vocab_size, D, name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            self._internal_len, D, name="position_embedding",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            dropout_rate, name="embed_dropout",
        )

        # --- Stochastic depth schedule across all encoder + decoder blocks ---
        total_blocks = sum(encoder_depths) + sum(decoder_depths)
        drop_rates = _linear_drop_path_rates(total_blocks, stochastic_depth_rate)
        block_idx = 0

        _block_common = dict(
            channels=D,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            layer_scale_init=layer_scale_init,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # --- Encoder blocks (flat list, stage boundaries from encoder_depths) ---
        self._encoder_blocks: List[CausalCliffordNetBlock] = []
        for i in range(n_stages):
            for j in range(encoder_depths[i]):
                blk = CausalCliffordNetBlock(
                    shifts=stage_shifts[i],
                    drop_path_rate=drop_rates[block_idx],
                    name=f"enc_s{i}_b{j}",
                    **_block_common,
                )
                self._encoder_blocks.append(blk)
                block_idx += 1

        # --- Pool layers (between encoder stages) ---
        self._pool_layers: List[GeometricStridePool] = [
            GeometricStridePool(
                channels=D,
                pool_size=pool_sizes[i],
                shifts=pool_shifts,
                cli_mode=cli_mode,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"pool_{i}",
            )
            for i in range(n_stages - 1)
        ]

        # --- Unpool layers (between decoder stages) ---
        self._unpool_layers: List[GeometricStrideUnpool] = [
            GeometricStrideUnpool(
                channels=D,
                factor=pool_sizes[i],
                shifts=pool_shifts,
                cli_mode=cli_mode,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"unpool_{i}",
            )
            for i in range(n_stages - 1)
        ]

        # --- GGR merge layers (at every skip junction) ---
        self._merge_layers: List[GGRMerge] = [
            GGRMerge(
                channels=D,
                layer_scale_init=layer_scale_init,
                drop_path_rate=0.0,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"merge_{i}",
            )
            for i in range(n_stages - 1)
        ]

        # --- Decoder blocks (flat list, stage boundaries from decoder_depths) ---
        self._decoder_blocks: List[CausalCliffordNetBlock] = []
        for i in range(n_stages - 1):
            for j in range(decoder_depths[i]):
                blk = CausalCliffordNetBlock(
                    shifts=stage_shifts[i],
                    drop_path_rate=drop_rates[block_idx],
                    name=f"dec_s{i}_b{j}",
                    **_block_common,
                )
                self._decoder_blocks.append(blk)
                block_idx += 1

        # --- Output head ---
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="head_norm",
        )
        self.head_dropout = (
            keras.layers.Dropout(dropout_rate, name="head_dropout")
            if dropout_rate > 0.0
            else None
        )
        self.output_proj = keras.layers.Dense(
            vocab_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="output_proj",
        )

        logger.info(
            f"Created CliffordUNetLM (Clifford-native, vocab_size={vocab_size}, "
            f"max_seq_length={max_seq_length}, D={D}, stages={n_stages}, "
            f"enc_depths={encoder_depths}, dec_depths={decoder_depths}, "
            f"pool_sizes={pool_sizes}, stage_shifts={stage_shifts}, "
            f"pool_shifts={pool_shifts}, cli_mode={cli_mode}, "
            f"ctx_mode={ctx_mode}, internal_len={self._internal_len})"
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        :param input_ids: Token IDs ``(B, seq_len)``.
        :param training: Whether in training mode.
        :return: Dict with ``"logits"`` key: ``(B, seq_len, vocab_size)``.
        """
        seq_len = keras.ops.shape(input_ids)[1]
        positions = keras.ops.arange(seq_len)

        # --- Embed tokens + positions ---
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_norm(x, training=training)
        x = self.embed_dropout(x, training=training)

        # Reshape to 4D: (B, seq_len, D) -> (B, 1, seq_len, D)
        x = keras.ops.expand_dims(x, axis=1)

        # Right-pad to internal length so every pool sees an integer window.
        # Causal blocks only look left, so right padding cannot influence
        # positions 0..seq_len-1.
        internal_len = self._internal_len
        pad_r = internal_len - seq_len
        x = keras.ops.pad(x, [[0, 0], [0, 0], [0, pad_r], [0, 0]])

        # --- Encoder (contracting path) ---
        skips: List[keras.KerasTensor] = []
        enc_idx = 0
        for stage in range(self._n_stages):
            for _ in range(self.encoder_depths[stage]):
                x = self._encoder_blocks[enc_idx](x, training=training)
                enc_idx += 1
            if stage < self._n_stages - 1:
                # Save the same-resolution encoder output as the skip for
                # this stage, then geometrically pool to the next stage.
                skips.append(x)
                x = self._pool_layers[stage](x)
            # Bottleneck: no skip is saved at the deepest stage.

        # --- Decoder (expanding path) ---
        for stage in reversed(range(self._n_stages - 1)):
            x = self._unpool_layers[stage](x)
            # Clifford-aware skip merge (instead of raw `x = x + skip`).
            x = self._merge_layers[stage](
                x, skips[stage], training=training,
            )
            # Decoder blocks for this stage (read from flat list).
            stage_start = sum(self.decoder_depths[:stage])
            for j in range(self.decoder_depths[stage]):
                x = self._decoder_blocks[stage_start + j](x, training=training)

        # Remove height dim and crop right padding.
        x = keras.ops.squeeze(x, axis=1)   # (B, internal_len, D)
        x = x[:, :seq_len, :]              # (B, seq_len, D)

        # --- Output head ---
        x = self.head_norm(x, training=training)
        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)
        logits = self.output_proj(x)

        return {"logits": logits}

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    @property
    def total_pool_factor(self) -> int:
        """Product of all ``pool_sizes`` — i.e. tokens per coarsest position.

        With the strictly causal :class:`GeometricStrideUnpool` every
        position is safe, but if a caller wants to train only at the
        ``total_pool_factor``-stride coarse anchor positions (the "fix 2"
        loss-mask path) it can use this to build the position mask.
        """
        return self._total_pool_factor

    # ------------------------------------------------------------------
    # Shape / config / serialization
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        return {"logits": (input_shape[0], input_shape[1], self.vocab_size)}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "channels": self.channels,
            "encoder_depths": self.encoder_depths,
            "decoder_depths": self.decoder_depths,
            "pool_sizes": self.pool_sizes,
            "stage_shifts": self.stage_shifts,
            "pool_shifts": self.pool_shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": (
                initializers.serialize(self.kernel_initializer)
                if self.kernel_initializer else None
            ),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordUNetLM":
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        max_seq_length: int = 512,
        **kwargs: Any,
    ) -> "CliffordUNetLM":
        """Create a CliffordUNetLM from a predefined variant.

        :param variant: One of ``"nano"``, ``"mini"``, ``"base"``.
        :param vocab_size: Vocabulary size.
        :param max_seq_length: Maximum sequence length.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordUNetLM` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordUNetLM-{variant.upper()}")
        return cls(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            **defaults,
        )
