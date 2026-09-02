"""The DiTXA transformer block and the three embedders that feed it.

A plain DiT block conditions two sub-ops (self-attention and an MLP) from one
conditioning vector, so its ``adaLN_modulation`` emits ``6 * hidden_size``
numbers. The bidirectional bridge's block conditions **four** things and emits
``12 * hidden_size``: the third triple does not gate a residual at all -- it
FiLM-modulates the conditioning tokens on their way into cross-attention's K/V
projection, from the same ``c`` that drives the query stream's own modulation.

That asymmetry is the whole point of the module. It is also invisible to every
conventional test: a permutation of the twelve chunks, or of the three residual
adds, preserves the parameter count, every tensor shape, ``get_config()`` and a
``.keras`` round trip, and changes only which learned scalar multiplies which
sub-op. It is pinned instead by a per-chunk attribution probe in
``tests/test_models/test_bit_diffusion/test_the_twelve_way_modulation_is_wired.py``.

Ported from ``DiTBlockWithCrossAttention`` (upstream ``dit.py:344-366``).

.. code-block:: text

    c (B, hidden)                                cond_tokens (B, N, hidden)
      |                                                     |
      v                                                     |
    SiLU -> Dense(12*hidden, zero-init kernel AND bias)      |
      |                                                     |
      +-> split 12 -> the chunk order, exactly:              |
      |                                                     |
      |   0 shift_msa  1 scale_msa  2 gate_msa   -- triple 1 |
      |   3 shift_xa   4 scale_xa   5 gate_xa    -- triple 2 |
      |   6 shift_cond 7 scale_cond 8 gate_cond  -- triple 3 |
      |   9 shift_mlp 10 scale_mlp 11 gate_mlp   -- triple 4 |
      |                                                     |
    x (B, N, hidden)                                         |
      |                                                     |
      |--------------------------------+                     |
      v                                |                     |
    norm1 -> modulate(shift_msa, scale_msa)                   |
      v                                |                     |
    attn  (self-attention, fused QKV, non-affine per-head     |
      |    RMSNorm on Q/K, 1/sqrt(head_dim) applied ONCE)     |
      v                                |                     |
    * gate_msa ---------------------> (+)  residual 1         |
                                       |                     |
      +--------------------------------+                     |
      |                                |                     v
      v                                |          norm_cond -> modulate(
    norm_cross -> modulate(shift_xa,   |             shift_cond, scale_cond)
      |            scale_xa)           |                     |
      |            = the QUERY         |            = the K/V stream
      v                                |                     |
    cross_attn(query, kv) <------------|---------------------+
      |                                |
      v                                |     NOTE: gate_cond is emitted and
    * gate_xa ----------------------> (+)    consumed by NO residual add. It is
                                       |     the one dead chunk, reproduced on
      +--------------------------------+     purpose -- see the module docstring.
      |                                |
      v                                |
    norm2 -> modulate(shift_mlp, scale_mlp)
      v                                |
    mlp   (tanh-approximate GELU)      |
      v                                |
    * gate_mlp ---------------------> (+)  residual 3
                                       |
                                       v
                              output (B, N, hidden)

**Reuse.** Nothing here re-implements attention, an MLP or the modulation
broadcast. ``MultiHeadCrossAttention`` covers both attention modes (fused QKV in
self-attention mode, fused KV in cross-attention mode) with the per-head
non-affine RMSNorm the upstream ``qk_norm=True`` / ``elementwise_affine=False``
combination asks for; ``create_ffn_layer('gelu_tanh', ...)`` is exactly
upstream's ``Mlp(act_layer=nn.GELU(approximate="tanh"))``; and the ``modulate``
broadcast contract is imported from ``layers/transformers/sd3_adaln.py`` rather
than re-spelled. The 12-way split itself stays here rather than becoming an
``sd3_adaln.py`` sibling -- see ``decisions.md`` D-004.

**Also here.** :class:`DiTXATimestepEmbedder` and :func:`get_2d_sincos_pos_embed`.
Both are numerically specified, both have a plausible house-layer substitute, and
neither substitute is correct:

* the timestep embedder differs from ``layers/embedding/ScalarSinusoidalEmbedding``
  on three independent numeric axes and one structural one -- see that class's
  docstring, which quotes the measured deltas;
* the positional table is upstream's MAE helper, whose ``np.meshgrid(grid_w,
  grid_h)`` ordering and ``concat([emb_h, emb_w])`` split are invisible to any
  shape assertion, because both halves have the same width.

The third embedder, ``ClassLabelEmbedding``, is genuinely reusable and therefore
does NOT live here: it is a shared layer at
``src/dl_techniques/layers/embedding/class_label_embedding.py`` with a factory key.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import numpy as np

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.multi_head_cross_attention import (
    MultiHeadCrossAttention,
)
from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.transformers.sd3_adaln import modulate
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# The chunk order -- a fact, not a style choice
# ---------------------------------------------------------------------

#: The order the 12 modulation chunks are consumed in, upstream ``dit.py:351-352``.
#: Index ``k`` of ``ops.split(adaLN_modulation(c), 12, axis=-1)`` is
#: ``ADALN_CHUNK_NAMES[k]``. Exported so the guard test can name chunks rather
#: than count slices.
ADALN_CHUNK_NAMES: Tuple[str, ...] = (
    "shift_msa", "scale_msa", "gate_msa",
    "shift_xa", "scale_xa", "gate_xa",
    "shift_cond", "scale_cond", "gate_cond",
    "shift_mlp", "scale_mlp", "gate_mlp",
)

#: Width multiplier of the modulation projection.
NUM_ADALN_CHUNKS: int = len(ADALN_CHUNK_NAMES)


def _unpack_triple_shape(
    input_shape: Any,
) -> Tuple[
    Tuple[Optional[int], ...],
    Tuple[Optional[int], ...],
    Tuple[Optional[int], ...],
]:
    """Split ``[x_shape, c_shape, cond_shape]`` into its three parts.

    :param input_shape: A list or tuple of exactly three shapes: ``x_shape``
        ``(B, N, hidden)``, ``c_shape`` ``(B, hidden)`` and ``cond_shape``
        ``(B, N, hidden)``.
    :type input_shape: Any
    :return: ``(x_shape, c_shape, cond_shape)`` as tuples.
    :rtype: Tuple[Tuple, Tuple, Tuple]
    :raises ValueError: If the input is not a triple of shapes.
    """
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
        raise ValueError(
            "DiTXABlock expects an input_shape triple "
            "[x_shape, c_shape, cond_shape], got "
            f"{input_shape!r}"
        )
    parts = []
    for name, shape in zip(("x_shape", "c_shape", "cond_shape"), input_shape):
        if not isinstance(shape, (list, tuple)):
            raise ValueError(
                "DiTXABlock expects an input_shape triple "
                f"[x_shape, c_shape, cond_shape]; {name} is {shape!r}, which is "
                "not a shape. A bare rank-3 tuple like (B, N, hidden) is the "
                "usual mistake -- the block takes three inputs, not one."
            )
        parts.append(tuple(shape))
    return parts[0], parts[1], parts[2]


@register_dl_technique(package="dl_techniques.models.bit_diffusion.blocks")
class DiTXABlock(keras.layers.Layer):
    """DiT block with cross-attention and a 12-way adaLN-Zero modulation.

    Three residual adds, in the order ``msa -> xa -> mlp``, each gated by its own
    chunk; a fourth chunk triple modulates the conditioning tokens entering
    cross-attention's K/V projection and gates nothing. See the module docstring
    for the diagram and the exact chunk order.

    The conditioning stream is **not** carried forward: ``cond_tokens`` is
    modulated per block and consumed, never updated, so block ``n + 1`` receives
    the same tensor block ``n`` did. Only ``x`` accumulates.

    :param hidden_size: Model width. Must be positive and divisible by
        ``num_heads``.
    :type hidden_size: int
    :param num_heads: Attention head count for both attention sub-layers.
    :type num_heads: int
    :param mlp_ratio: Expansion factor of the MLP's hidden width,
        ``int(hidden_size * mlp_ratio)``. Must be positive.
    :type mlp_ratio: float
    :param norm_epsilon: Epsilon of the four non-affine ``LayerNormalization``
        sub-layers. Explicit on purpose: bare Keras defaults to ``1e-3``, a
        silent 1000x error with no shape symptom.
    :type norm_epsilon: float
    :param qk_norm_epsilon: Epsilon of the per-head Q/K ``RMSNorm``s inside both
        attention sub-layers.
    :type qk_norm_epsilon: float
    :param dropout_rate: Attention-weight dropout, and the MLP's dropout.
    :type dropout_rate: float
    :param use_bias: Whether the attention projections and the MLP carry biases.
        Upstream is ``qkv_bias=True`` throughout.
    :type use_bias: bool
    :param kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :raises ValueError: If ``hidden_size`` is not positive or not divisible by
        ``num_heads``, if ``num_heads`` is not positive, or if ``mlp_ratio`` is
        not positive.

    Input shape:
        A list of three tensors -- ``x`` ``(B, N, hidden_size)``, ``c``
        ``(B, hidden_size)`` and ``cond_tokens`` ``(B, N, hidden_size)``.
        Upstream asserts ``cond_tokens.shape == x.shape``; this is same-length
        cross-attention, not variable-length K/V.

    Output shape:
        ``(B, N, hidden_size)`` -- the query stream only.

    Example:
        >>> import keras
        >>> block = DiTXABlock(hidden_size=64, num_heads=4)
        >>> x = keras.random.normal((2, 16, 64))
        >>> c = keras.random.normal((2, 64))
        >>> cond = keras.random.normal((2, 16, 64))
        >>> block([x, c, cond]).shape
        (2, 16, 64)

    Note:
        At initialisation the modulation ``Dense`` is zero in **both** kernel and
        bias (adaLN-Zero), so all three gates are ``0`` and the block is the
        exact identity on ``x``. That is not a quirk to be optimised away: it is
        what makes a deep stack trainable, and it is also the premise the
        attribution probe rests on.

    Attributes:
        adaln_dense: The zero-init ``Dense(12 * hidden_size)`` producing the chunks.
        attn: Self-attention over ``x`` (fused QKV).
        cross_attn: Cross-attention, ``x`` querying ``cond_tokens`` (fused KV).
        mlp: The tanh-approximate GELU MLP.
        norm1, norm_cross, norm_cond, norm2: Non-affine ``LayerNormalization``s.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_epsilon: float = 1e-6,
        qk_norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.norm_epsilon = float(norm_epsilon)
        self.qk_norm_epsilon = float(qk_norm_epsilon)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)

        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        qk_norm_kwargs = {"use_scale": False, "epsilon": self.qk_norm_epsilon}

        # CREATE sub-layers. Every one gets an explicit name so the weight paths
        # are stable across a `.keras` round trip.
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=self.norm_epsilon, center=False, scale=False, name="norm1"
        )
        self.attn = MultiHeadCrossAttention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            shared_qk_projections=True,
            use_bias=self.use_bias,
            qk_norm_type="rms_norm",
            qk_norm_kwargs=dict(qk_norm_kwargs),
            name="attn",
        )
        self.norm_cross = keras.layers.LayerNormalization(
            epsilon=self.norm_epsilon, center=False, scale=False, name="norm_cross"
        )
        self.norm_cond = keras.layers.LayerNormalization(
            epsilon=self.norm_epsilon, center=False, scale=False, name="norm_cond"
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            shared_qk_projections=False,
            use_bias=self.use_bias,
            qk_norm_type="rms_norm",
            qk_norm_kwargs=dict(qk_norm_kwargs),
            name="cross_attn",
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.norm_epsilon, center=False, scale=False, name="norm2"
        )
        self.mlp = create_ffn_layer(
            "gelu_tanh",
            hidden_dim=self.mlp_hidden_dim,
            output_dim=self.hidden_size,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            name="mlp",
        )
        # Two SEPARATE Zeros() instances. A shared Initializer object draws
        # bit-identically forever; that is invisible for Zeros here, but the XL
        # variant stacks 28 of these blocks and the habit is what matters.
        self.adaln_dense = keras.layers.Dense(
            NUM_ADALN_CHUNKS * self.hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Zeros(),
            name="adaln_modulation",
        )

    def build(self, input_shape: Any) -> None:
        """Build every sub-layer at the shape it will actually see.

        :param input_shape: ``[x_shape, c_shape, cond_shape]``.
        :type input_shape: Any
        :raises ValueError: If ``input_shape`` is not a triple of shapes.
        """
        if self.built:
            return

        x_shape, c_shape, cond_shape = _unpack_triple_shape(input_shape)

        self.norm1.build(x_shape)
        self.attn.build(x_shape)
        self.norm_cross.build(x_shape)
        self.norm_cond.build(cond_shape)
        self.cross_attn.build([x_shape, cond_shape])
        self.norm2.build(x_shape)
        self.mlp.build(x_shape)
        self.adaln_dense.build(c_shape)

        super().build(input_shape)
        logger.debug(
            "Built DiTXABlock '%s': hidden=%d heads=%d mlp_hidden=%d",
            self.name,
            self.hidden_size,
            self.num_heads,
            self.mlp_hidden_dim,
        )

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the three gated residuals in the order ``msa -> xa -> mlp``.

        :param inputs: ``[x, c, cond_tokens]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Keras training flag, threaded to the attention and MLP
            dropouts.
        :type training: Optional[bool]
        :return: The updated query stream, ``(B, N, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        x, c, cond_tokens = inputs[0], inputs[1], inputs[2]

        # DECISION plan-2026-09-02T094601-77d4a04e/D-004
        # This 12-way split stays INSIDE the block. Do NOT hoist it into an
        # `sd3_adaln.py` sibling: the four triples apply to THREE different
        # normed streams (norm1(x), norm_cross(x), norm_cond(cond_tokens)), which
        # is not the one-norm-plus-one-chunk-tuple shape that family has, and a
        # 12-way sibling would have exactly one call site. See decisions.md D-004.
        chunks = keras.ops.split(
            self.adaln_dense(keras.ops.silu(c)), NUM_ADALN_CHUNKS, axis=-1
        )
        (
            shift_msa, scale_msa, gate_msa,
            shift_xa, scale_xa, gate_xa,
            shift_cond, scale_cond, _gate_cond,
            shift_mlp, scale_mlp, gate_mlp,
        ) = chunks
        # `_gate_cond` is chunk 8 and gates NOTHING. Upstream emits it and never
        # consumes it (dit.py:351-366); dropping it here would renumber chunks
        # 9-11 and silently rewire the MLP triple. Reproduced deliberately.

        x = x + keras.ops.expand_dims(gate_msa, axis=1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            training=training,
        )

        x = x + keras.ops.expand_dims(gate_xa, axis=1) * self.cross_attn(
            modulate(self.norm_cross(x), shift_xa, scale_xa),
            kv_input=modulate(self.norm_cond(cond_tokens), shift_cond, scale_cond),
            training=training,
        )

        x = x + keras.ops.expand_dims(gate_mlp, axis=1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp),
            training=training,
        )
        return x

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Return the query stream's shape; the conditioning stream is not carried.

        :param input_shape: ``[x_shape, c_shape, cond_shape]``.
        :type input_shape: Any
        :return: ``(B, N, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not a triple of shapes.
        """
        x_shape, _, _ = _unpack_triple_shape(input_shape)
        return (x_shape[0], x_shape[1], self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "norm_epsilon": self.norm_epsilon,
                "qk_norm_epsilon": self.qk_norm_epsilon,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.models.bit_diffusion.blocks")
class DiTXATimestepEmbedder(keras.layers.Layer):
    """Sinusoidal embedding of a scalar timestep, refined by ``Dense -> SiLU -> Dense``.

    Ported from upstream ``TimestepEmbedder`` (``dit.py:30-68``), which is in
    turn OpenAI's GLIDE ``timestep_embedding``.

    .. code-block:: text

        t  (B,)  -- ALREADY scaled by the caller (DiTXA multiplies by
          |         time_scale = 1000 before calling; this layer does not)
          v
        freqs = exp(-log(max_period) * arange(half) / half)     half = F // 2
          |     a NON-TRAINABLE weight, F = frequency_embedding_size
          v
        args = t[:, None] * freqs[None]                          (B, half)
          v
        concat([cos(args), sin(args)], axis=-1)                  (B, 2*half)
          |   ^^^ COS FIRST. Not a style choice -- see below.
          v
        (odd F only) one trailing zero column                    (B, F)
          v
        Dense(hidden_size) -> SiLU -> Dense(hidden_size)         (B, hidden_size)

    Why this is not :class:`~dl_techniques.layers.embedding.scalar_sinusoidal_embedding.ScalarSinusoidalEmbedding`:
        The house layer looks like a drop-in and is not, on **three independent
        numeric axes** plus a structural one. The numbers below were MEASURED at
        ``dim = 8`` on ``t = [0, 0.25, 0.5, 1]``
        (``plans/.../probes/step6_scalar_sinusoidal_differences.txt``); they are
        not estimates.

        1. **Concat order.** The house layer emits ``concat([sin, cos])``, this
           one emits ``concat([cos, sin])``. At ``t = 0`` the house basis is
           ``[0,0,0,0,1,1,1,1]`` and this one is ``[1,1,1,1,0,0,0,0]``:
           ``max|delta| = 1.0``, the largest a bounded sinusoid can be.
        2. **Frequency-ladder denominator.** The house ladder divides the
           exponent by ``half - 1``, so its last frequency lands on
           ``1e-4`` (``9.9999997e-05`` in float32). This one divides by
           ``half``, so its last frequency is ``exp(-log(1e4)*(half-1)/half)``,
           **10x larger** at ``half = 4`` (``9.9999993e-04``) and never reaching
           ``1 / max_period``. Same ``max_period``, different ladder:
           ``max|delta| = 0.0534341932``.
        3. **Input rescale.** The house layer rescales its input onto
           ``[0, 1e4]`` before the sinusoidal map (``t = 0.25`` becomes
           ``2500.0``, a ``10000x`` larger sinusoidal argument). This layer does
           NO input rescale: the caller pre-scales ``t`` by ``time_scale = 1000``
           and this layer feeds it straight in. ``max|delta| = 1.8825995338``.
        4. **Structural.** The house layer's single ``dim`` sets the basis width
           AND both Dense widths. Here ``frequency_embedding_size`` (256) is
           decoupled from ``hidden_size`` (384/768/896/1024/1152 for the shipped
           DiTXA variants -- never 256), so the MLP is ``256 -> hidden -> hidden``.

        All four are pinned by
        ``tests/test_models/test_bit_diffusion/test_the_embedders.py``, so
        "simplifying" this back to the house layer reddens rather than drifts.

    :param hidden_size: Output width, and the width of both Dense layers.
    :type hidden_size: int
    :param frequency_embedding_size: Width of the sinusoidal basis feeding the
        MLP. Deliberately independent of ``hidden_size``; upstream's default is
        256 for every variant.
    :type frequency_embedding_size: int
    :param max_period: Base of the frequency ladder. Upstream's ``10000``.
    :type max_period: float
    :param kernel_stddev: Standard deviation of the ``RandomNormal`` kernel
        initializer of both Dense layers, upstream's ``nn.init.normal_(std=0.02)``
        (``dit.py:213-214``). A **fresh** initializer instance is constructed per
        Dense: a shared instance draws bit-identically forever.
    :type kernel_stddev: float
    :param kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :raises ValueError: If ``hidden_size`` or ``frequency_embedding_size`` is not
        positive, if ``frequency_embedding_size`` is less than 2 (no frequency
        ladder exists), if ``max_period`` is not greater than 1, or if
        ``kernel_stddev`` is not positive.

    Input shape:
        ``(B,)`` or ``(B, 1)``. A trailing singleton axis is squeezed.

    Output shape:
        ``(B, hidden_size)``.

    Example:
        >>> import keras
        >>> emb = DiTXATimestepEmbedder(hidden_size=64)
        >>> t = keras.ops.convert_to_tensor([0.0, 0.5, 1.0]) * 1000.0
        >>> emb(t).shape
        (3, 64)

    Attributes:
        freqs: Non-trainable ``(half,)`` frequency ladder, materialized in
            :meth:`build` from NumPy and installed through a constant
            initializer. NOT a plain tensor attribute -- that does not round-trip
            through ``.keras`` save/load, which is the bug
            ``ScalarSinusoidalEmbedding``'s own anchor was written to prevent.
        mlp_in, mlp_out: The two Dense layers.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: float = 10000.0,
        kernel_stddev: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if frequency_embedding_size < 2:
            raise ValueError(
                "frequency_embedding_size must be at least 2 (half = "
                f"frequency_embedding_size // 2 must be positive), got "
                f"{frequency_embedding_size}"
            )
        if max_period <= 1.0:
            raise ValueError(
                "max_period must be greater than 1 so that log(max_period) is "
                f"positive and the ladder decreases, got {max_period}"
            )
        if kernel_stddev <= 0.0:
            raise ValueError(f"kernel_stddev must be positive, got {kernel_stddev}")

        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.max_period = float(max_period)
        self.kernel_stddev = float(kernel_stddev)
        self.half = self.frequency_embedding_size // 2

        # A FRESH RandomNormal per Dense. Sharing one instance across layers
        # makes every one of them draw the same numbers forever, and no default
        # exposes it.
        self.mlp_in = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=self.kernel_stddev
            ),
            name="mlp_in",
        )
        self.mlp_out = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=self.kernel_stddev
            ),
            name="mlp_out",
        )

        self.freqs = None

    def build(self, input_shape: Any) -> None:
        """Materialize the frequency ladder and build both Dense sub-layers.

        :param input_shape: ``(B,)`` or ``(B, 1)``.
        :type input_shape: Any
        """
        if self.built:
            return

        # Computed with NumPy, installed as a NON-TRAINABLE WEIGHT through a
        # constant initializer. Do NOT replace this with
        # `self.freqs = keras.ops.exp(...)` in __init__ or build: a plain tensor
        # attribute does not survive a `.keras` round trip, and an `.assign()`
        # inside build() is DISCARDED by StatelessScope, leaving the table all
        # zeros in every real model. Both are recorded repo failures.
        freqs_np = np.exp(
            -math.log(self.max_period)
            * np.arange(self.half, dtype="float32")
            / self.half
        )
        self.freqs = self.add_weight(
            name="freqs",
            shape=(self.half,),
            initializer=keras.initializers.Constant(freqs_np),
            trainable=False,
            dtype="float32",
        )

        batch = tuple(input_shape)[0] if len(tuple(input_shape)) > 0 else None
        self.mlp_in.build((batch, self.frequency_embedding_size))
        self.mlp_out.build((batch, self.hidden_size))

        super().build(input_shape)

    def timestep_embedding(self, t: keras.KerasTensor) -> keras.KerasTensor:
        """Map a scalar timestep onto the sinusoidal basis, cos first.

        Exposed as a method so the guard test can compare the BASIS against the
        house layer's basis without the two MLPs in the way.

        :param t: Timesteps, shape ``(B,)``. Already scaled by the caller.
        :type t: keras.KerasTensor
        :return: ``(B, frequency_embedding_size)``.
        :rtype: keras.KerasTensor
        """
        args = keras.ops.expand_dims(keras.ops.cast(t, "float32"), axis=-1) * self.freqs
        # COS FIRST. Upstream `dit.py:60`. The 1D positional helper in this same
        # module is sin-first; they are independently specified and must not be
        # unified.
        embedding = keras.ops.concatenate(
            [keras.ops.cos(args), keras.ops.sin(args)], axis=-1
        )
        if self.frequency_embedding_size % 2:
            # An odd width leaves the basis one column short; upstream pads one
            # trailing ZERO column, it does not drop a frequency.
            embedding = keras.ops.pad(embedding, [(0, 0), (0, 1)])
        return embedding

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Embed the timestep.

        :param inputs: Timesteps, ``(B,)`` or ``(B, 1)``. **No rescale happens
            here**: the caller is expected to have applied ``time_scale``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to both Dense sub-layers.
        :type training: Optional[bool]
        :return: ``(B, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        t = inputs
        # Static rank read, as `ScalarSinusoidalEmbedding` does: `len(t.shape)`
        # is known at trace time. `len(keras.ops.shape(t))` is not graph-safe.
        if len(t.shape) > 1 and t.shape[-1] == 1:
            t = keras.ops.squeeze(t, axis=-1)
        t_freq = self.timestep_embedding(t)
        h = keras.activations.silu(self.mlp_in(t_freq, training=training))
        return self.mlp_out(h, training=training)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return ``(B, hidden_size)``.

        :param input_shape: ``(B,)`` or ``(B, 1)``.
        :type input_shape: Any
        :return: ``(B, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(input_shape)
        batch = shape[0] if len(shape) > 0 else None
        return (batch, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "frequency_embedding_size": self.frequency_embedding_size,
                "max_period": self.max_period,
                "kernel_stddev": self.kernel_stddev,
            }
        )
        return config


# ---------------------------------------------------------------------
# 2D sin-cos positional table -- pure NumPy, from facebookresearch/mae
# ---------------------------------------------------------------------


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray
) -> np.ndarray:
    """Sinusoidally embed a flat array of positions, **sin first**.

    Ported verbatim from ``dit.py:621-639`` (MAE ``util/pos_embed.py``).

    Two things here contradict :class:`DiTXATimestepEmbedder` and are correct
    anyway, because the two are independently specified upstream:

    * the concat is ``[sin, cos]``, not ``[cos, sin]``;
    * ``omega`` is built in **float64** and the whole computation stays float64,
      whereas the timestep ladder is float32.

    :param embed_dim: Output width per position. Must be even.
    :type embed_dim: int
    :param pos: Positions of any shape; flattened to ``(M,)`` first.
    :type pos: np.ndarray
    :return: ``(M, embed_dim)`` float64 array.
    :rtype: np.ndarray
    :raises ValueError: If ``embed_dim`` is not even or not positive.
    """
    if embed_dim <= 0 or embed_dim % 2 != 0:
        raise ValueError(
            f"embed_dim must be a positive even integer, got {embed_dim}"
        )
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = np.asarray(pos).reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: np.ndarray
) -> np.ndarray:
    """Concatenate two 1D embeddings, one per meshgrid output.

    Ported verbatim from ``dit.py:610-618``.

    .. code-block:: text

        grid (2, 1, G, G)   from np.meshgrid(grid_w, grid_h)  -- "w goes first"
          |
          +-- grid[0]  value at (row, col) == col   (the W / COLUMN position)
          |      -> get_1d(embed_dim // 2)  -> first  half of the output
          |
          +-- grid[1]  value at (row, col) == row   (the H / ROW position)
                 -> get_1d(embed_dim // 2)  -> second half of the output

    Upstream names those halves ``emb_h`` and ``emb_w``, which is **backwards**
    relative to what they encode -- ``emb_h = get_1d(grid[0])`` actually encodes
    the column. That naming inversion is cosmetic and is reproduced here only in
    the sense that the ORDER is reproduced; the names below say what the arrays
    are. What a port must match is the order: **the first ``embed_dim // 2``
    columns encode the column index, the last ``embed_dim // 2`` encode the row
    index.** Swapping them preserves the shape, the dtype, every norm and every
    per-row statistic on a square grid, so only an elementwise comparison sees it.

    :param embed_dim: Total output width. Must be even (each half is again
        halved by the 1D helper, so a multiple of 4 is the practical constraint).
    :type embed_dim: int
    :param grid: ``(2, ...)`` array of positions.
    :type grid: np.ndarray
    :return: ``(H*W, embed_dim)`` float64 array.
    :rtype: np.ndarray
    :raises ValueError: If ``embed_dim`` is not even.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    emb_col = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_row = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    return np.concatenate([emb_col, emb_row], axis=1)  # (H*W, D)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    """Build the fixed 2D sin-cos positional table for a square patch grid.

    Ported verbatim from ``dit.py:592-607``. Pure NumPy: no Keras op runs here,
    so it is safe to call at ``build()`` time and feed to
    ``add_weight(trainable=False, initializer=keras.initializers.Constant(...))``.

    **How to install this, and how not to.** The returned array is a constant,
    but it must still become a non-trainable WEIGHT:

    * NEVER a plain tensor attribute (``self.pos_embed = ops.convert_to_tensor(
      get_2d_sincos_pos_embed(...))``) -- that does not round-trip through
      ``.keras`` save/load. That is the legacy ``TimestepEmbedding`` bug this
      repo already paid for once.
    * NEVER ``add_weight(...)`` followed by ``.assign(...)`` inside ``build()``
      -- ``StatelessScope`` DISCARDS the assign and the table stays all zeros in
      every real model, with no shape symptom.

    The one correct form is a ``Constant`` initializer passed to ``add_weight``.

    :param embed_dim: Width of the embedding per grid position.
    :type embed_dim: int
    :param grid_size: Side length of the square grid; the table has
        ``grid_size ** 2`` rows.
    :type grid_size: int
    :param cls_token: Whether to prepend ``extra_tokens`` zero rows.
    :type cls_token: bool
    :param extra_tokens: Number of zero rows prepended when ``cls_token`` is
        true. Upstream prepends only when BOTH are set; reproduced exactly.
    :type extra_tokens: int
    :return: ``(grid_size**2, embed_dim)``, or
        ``(extra_tokens + grid_size**2, embed_dim)`` with a cls token. float64.
    :rtype: np.ndarray
    :raises ValueError: If ``grid_size`` is not positive or ``embed_dim`` is not
        even.

    Example:
        >>> table = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4)
        >>> table.shape
        (16, 8)
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # "here w goes first" -- upstream's own annotation, and NumPy's default
    # indexing='xy'. So grid[0] holds the COLUMN index and grid[1] the ROW index.
    # Passing (grid_h, grid_w) instead transposes the table with no shape change.
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed
