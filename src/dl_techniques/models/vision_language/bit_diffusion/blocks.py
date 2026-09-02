"""The DiTXA transformer block: 12-way adaLN over self-attention, cross-attention and an MLP.

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
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import keras

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
