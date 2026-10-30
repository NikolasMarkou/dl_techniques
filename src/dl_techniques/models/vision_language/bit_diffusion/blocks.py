"""
DiTXA transformer block: a DiT block with cross-attention and 12-way adaLN-Zero.

:class:`DiTXABlock` updates a token stream ``x`` with self-attention,
cross-attention onto conditioning tokens, and an MLP, each gated from one
conditioning vector ``c``. A plain DiT block emits ``6 * hidden_size``
modulation numbers for its two residuals; this block emits ``12 * hidden_size``,
one triple per residual plus a triple that modulates the conditioning tokens on
their way into cross-attention's K/V. Attention, the MLP and ``modulate`` come
from shared layers (``MultiHeadCrossAttention``,
``create_ffn_layer('gelu_tanh', ...)``, ``layers/transformers/sd3_adaln.py``),
as do the timestep, 2D sin-cos and class-label embedders that feed the block
(``layers/embedding/``).

References:
    - Ported from ``DiTBlockWithCrossAttention``, upstream ``dit.py``.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.multi_head_cross_attention import (
    MultiHeadCrossAttention,
)
from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.transformers.sd3_adaln import modulate
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
#: Order of the 12 modulation chunks, as in upstream ``dit.py``: index ``k`` of
#: ``ops.split(adaLN_modulation(c), 12, axis=-1)`` is ``ADALN_CHUNK_NAMES[k]``.
ADALN_CHUNK_NAMES: Tuple[str, ...] = (
    "shift_msa", "scale_msa", "gate_msa",
    "shift_xa", "scale_xa", "gate_xa",
    "shift_cond", "scale_cond", "gate_cond",
    "shift_mlp", "scale_mlp", "gate_mlp",
)

#: Width multiplier of the modulation projection.
NUM_ADALN_CHUNKS: int = len(ADALN_CHUNK_NAMES)

# ---------------------------------------------------------------------

def _unpack_triple_shape(
    input_shape: Any,
) -> Tuple[
    Tuple[Optional[int], ...],
    Tuple[Optional[int], ...],
    Tuple[Optional[int], ...],
]:
    """Split ``[x_shape, c_shape, cond_shape]`` into its three parts.

    :param input_shape: A list or tuple of exactly three shapes, each itself a
        list or tuple: ``x_shape`` ``(B, N, hidden)``, ``c_shape``
        ``(B, hidden)`` and ``cond_shape`` ``(B, N, hidden)``. Their sizes are
        not checked.
    :type input_shape: Any
    :return: ``(x_shape, c_shape, cond_shape)`` as tuples.
    :rtype: Tuple[Tuple, Tuple, Tuple]
    :raises ValueError: If the input is not a triple, or one entry is not a
        list or tuple.
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
    """Update a token stream with gated self-attention, cross-attention and MLP.

    One ``Dense(12 * hidden_size)`` on ``SiLU(c)`` yields twelve ``(B, hidden)``
    chunks in the order of ``ADALN_CHUNK_NAMES``. Three triples shift, scale and
    gate the three residuals; the fourth shifts and scales the conditioning
    tokens before they become cross-attention's K/V. Its gate, chunk 8, is
    emitted and never used, as upstream. ``cond_tokens`` is consumed, not
    updated, so every block of a stack receives the same tensor; only ``x``
    accumulates. The modulation ``Dense`` starts at zero in kernel and bias
    (adaLN-Zero), so every gate is 0 and the block is the exact identity on
    ``x`` at initialisation, which keeps a deep stack trainable.

    Modulation:

    .. code-block:: text

        c [B, hidden]
                 │
                 ▼
               SiLU
                 │
                 ▼
        ┌──────────────────────────┐
        │ adaln_dense              │  zero-init kernel and bias
        │ Dense(12 * hidden)       │
        └──────────────────────────┘
                 │  [B, 12 * hidden]
                 ▼
        split into 12 chunks of [B, hidden]

         0 shift_msa    1 scale_msa    2 gate_msa     self-attention
         3 shift_xa     4 scale_xa     5 gate_xa      cross-attention query
         6 shift_cond   7 scale_cond   8 gate_cond    cond K/V; gate unused
         9 shift_mlp   10 scale_mlp   11 gate_mlp     MLP

    Residual stream:

    .. code-block:: text

        x [B, N, hidden]
                 │
                 ├────────────────────┐
                 ▼                    │
        norm1, modulate msa           │
                 │                    │
                 ▼                    │
        ┌─────────────────────┐       │
        │ attn  self          │       │
        └─────────────────────┘       │
                 │                    │
                 ▼                    │
             * gate_msa               │
                 │                    │
                 ▼                    │
                 + ◄──────────────────┘
                 │
                 ├────────────────────┐
                 ▼                    │
        norm_cross, modulate xa       │
                 │                    │
                 ▼                    │
        ┌─────────────────────┐       │
        │ cross_attn          │◄──────┼── kv
        └─────────────────────┘       │
                 │                    │
                 ▼                    │
             * gate_xa                │
                 │                    │
                 ▼                    │
                 + ◄──────────────────┘
                 │
                 ├────────────────────┐
                 ▼                    │
        norm2, modulate mlp           │
                 │                    │
                 ▼                    │
        ┌─────────────────────┐       │
        │ mlp  GELU (tanh)    │       │
        └─────────────────────┘       │
                 │                    │
                 ▼                    │
             * gate_mlp               │
                 │                    │
                 ▼                    │
                 + ◄──────────────────┘
                 │
                 ▼
        output [B, N, hidden]

    "modulate msa" is ``modulate(., shift_msa, scale_msa)``, and likewise for
    xa and mlp. ``kv = modulate(norm_cond(cond_tokens), shift_cond,
    scale_cond)``. Both attention sub-layers apply a per-head ``RMSNorm``
    without scale to Q and K; ``attn`` runs in self-attention mode.

    :param hidden_size: Model width. Must be positive and divisible by
        ``num_heads``.
    :type hidden_size: int
    :param num_heads: Attention head count for both attention sub-layers. Must
        be positive.
    :type num_heads: int
    :param mlp_ratio: Expansion factor of the MLP's hidden width,
        ``int(hidden_size * mlp_ratio)``. Must be positive.
    :type mlp_ratio: float
    :param norm_epsilon: Epsilon of the four non-affine ``LayerNormalization``
        sub-layers. The Keras default of ``1e-3`` would differ from this
        ``1e-6`` by 1000x with no shape symptom. Not validated.
    :type norm_epsilon: float
    :param qk_norm_epsilon: Epsilon of the per-head Q/K ``RMSNorm``s inside both
        attention sub-layers. Not validated.
    :type qk_norm_epsilon: float
    :param dropout_rate: Attention-weight dropout, and the MLP's dropout. Not
        validated.
    :type dropout_rate: float
    :param use_bias: Whether the attention projections and the MLP carry biases.
        Upstream is ``qkv_bias=True`` throughout. ``adaln_dense`` always has a
        bias.
    :type use_bias: bool
    :param **kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :ivar adaln_dense: The zero-init ``Dense(12 * hidden_size)`` producing the
        chunks; its layer name, and so its weight path, is ``adaln_modulation``.
    :ivar attn: Self-attention over ``x``.
    :ivar cross_attn: Cross-attention, ``x`` querying the modulated
        ``cond_tokens``.
    :ivar mlp: The tanh-approximate GELU MLP.
    :ivar norm1: Non-affine ``LayerNormalization`` before ``attn``.
    :ivar norm_cross: Non-affine ``LayerNormalization`` on the cross-attention
        query.
    :ivar norm_cond: Non-affine ``LayerNormalization`` on ``cond_tokens``.
    :ivar norm2: Non-affine ``LayerNormalization`` before ``mlp``.

    Input shape:
        A list of three tensors: ``x`` ``(B, N, hidden_size)``, ``c``
        ``(B, hidden_size)`` and ``cond_tokens`` ``(B, N, hidden_size)``.
        Upstream asserts ``cond_tokens.shape == x.shape``; this layer does not
        check it.

    Output shape:
        ``(B, N, hidden_size)``, the updated ``x`` only.

    Example:
        >>> import keras
        >>> block = DiTXABlock(hidden_size=64, num_heads=4)
        >>> x = keras.random.normal((2, 16, 64))
        >>> c = keras.random.normal((2, 64))
        >>> cond = keras.random.normal((2, 16, 64))
        >>> block([x, c, cond]).shape
        (2, 16, 64)
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
        """Validate the sizes and create the sub-layers.

        Arguments are documented on the class.

        :raises ValueError: If ``hidden_size`` or ``num_heads`` is not positive,
            ``hidden_size`` is not divisible by ``num_heads``, or ``mlp_ratio``
            is not positive.
        """
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

        # Every sub-layer gets an explicit name so weight paths stay stable
        # across a .keras round trip.
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
        # Separate Zeros() instances: a shared initializer object draws
        # identically every time, harmless for Zeros but a habit to avoid.
        self.adaln_dense = keras.layers.Dense(
            NUM_ADALN_CHUNKS * self.hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Zeros(),
            name="adaln_modulation",
        )

    def build(self, input_shape: Any) -> None:
        """Build every sub-layer at the shape it will see. Returns at once if built.

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

        :param inputs: ``[x, c, cond_tokens]``, indexed without a length check.
        :type inputs: List[keras.KerasTensor]
        :param training: Keras training flag, threaded to the attention and MLP
            dropouts.
        :type training: Optional[bool]
        :return: The updated query stream, ``(B, N, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        x, c, cond_tokens = inputs[0], inputs[1], inputs[2]

        # DECISION plan-2026-09-02T094601-77d4a04e/D-004: the 12-way split stays here.
        # Its triples feed three normed streams, unlike sd3_adaln.py's one. See decisions.md.
        chunks = keras.ops.split(
            self.adaln_dense(keras.ops.silu(c)), NUM_ADALN_CHUNKS, axis=-1
        )
        (
            shift_msa, scale_msa, gate_msa,
            shift_xa, scale_xa, gate_xa,
            shift_cond, scale_cond, _gate_cond,
            shift_mlp, scale_mlp, gate_mlp,
        ) = chunks
        # _gate_cond (chunk 8) gates nothing, as upstream. Dropping it from the split
        # would renumber chunks 9-11 and silently rewire the MLP triple.

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
        """Return the shape of the updated ``x``; ``cond_tokens`` is not returned.

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
