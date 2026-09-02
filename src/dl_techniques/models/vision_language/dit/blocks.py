"""The plain-DiT transformer block and its final modulated projection.

Two layers, both pure composition of existing ``dl_techniques`` assets:

- :class:`DiTBlock` -- one adaLN-Zero conditioned transformer block. Two gated
  residual branches (self-attention, then MLP), both driven by a single
  ``(B, hidden_size)`` conditioning vector through one
  ``SiLU -> Dense(6 * hidden_size)`` projection whose kernel AND bias are
  zero-initialised.
- :class:`DiTFinalLayer` -- the read-out: a 2-way modulation followed by a
  zero-initialised ``Dense(patch_size**2 * out_channels)``.

Ported from ``reference/models.py:94-135`` (``DiTBlock`` and ``FinalLayer``).
The forward path this module must reproduce, verbatim from
``reference/models.py:111-115``::

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \\
        adaLN_modulation(c).chunk(6, dim=1)
    x = x + gate_msa.unsqueeze(1) * attn(modulate(norm1(x), shift_msa, scale_msa))
    x = x + gate_mlp.unsqueeze(1) * mlp(modulate(norm2(x), shift_mlp, scale_mlp))

**Nothing here is a re-implementation.** The 6-way chunk, its zero-init Dense
and the affine-free ``norm1`` come from
:class:`~dl_techniques.layers.transformers.sd3_adaln.AdaLayerNormZero`; the
2-way chunk from
:class:`~dl_techniques.layers.transformers.sd3_adaln.AdaLayerNormContinuous`;
the broadcast from :func:`~dl_techniques.layers.transformers.sd3_adaln.modulate`;
the attention from stock ``keras.layers.MultiHeadAttention`` (the precedent set
by ``layers/transformers/adaln_zero.py:222-230`` -- the ``'multi_head'`` FFN-style
factory key is a DIFFERENT class with no ``key_dim`` and ``use_bias=False``); and
the MLP from ``create_ffn_layer("gelu_tanh", ...)``, which is the
tanh-approximate GELU of ``reference/models.py:104``, not the exact GELU behind
the ``'mlp'`` key.

``adaln_zero.AdaLNZeroConditionalBlock`` is deliberately NOT used even though it
also emits a 6-way chunk in the same order: it defaults to
``use_causal_mask=True`` and expects ``(B, T, D)`` conditioning, and DiT is
non-causal bidirectional self-attention over image patches conditioned on a
single ``(B, D)`` vector.

**What is invisible without a dedicated guard.** A permutation of the six chunks
preserves the parameter count, every tensor shape, ``get_config()`` and a
``.keras`` round trip, and changes only which learned scalar multiplies which
sub-op -- a reversed permutation is still an exact bijection. It is pinned
instead by a per-chunk attribution probe in
``tests/test_models/test_dit/test_dit_blocks.py``, which drives one chunk at a
time through the modulation ``Dense``'s bias and asks only which sub-op moved.

.. code-block:: text

    c [B, D]                                  x [B, T, D]
      │                                          │
      ▼                                          │
    ┌──────────────────────────────┐             │
    │ AdaLayerNormZero(dim=D)      │◄────────────┤
    │   SiLU → Dense(6*D, zeros)   │             │
    │   split 6, in this order:    │             │
    │     0 shift_msa              │             │
    │     1 scale_msa   ──┐        │             │
    │     2 gate_msa      │        │             │
    │     3 shift_mlp     │        │             │
    │     4 scale_mlp     │        │             │
    │     5 gate_mlp      │        │             │
    │   x_norm = modulate(         │             │
    │     norm1(x), 0, 1)          │             │
    └──────────────────────────────┘             │
      │ x_norm [B, T, D]                         │
      ▼                                          │
    ┌──────────────────────────────┐             │
    │ MultiHeadAttention           │             │
    │   NON-causal, use_bias=True  │             │
    │   key_dim = D // num_heads   │             │
    └──────────────────────────────┘             │
      │                                          │
      ├─ × gate_msa[:, None, :]  (chunk 2)       │
      ▼                                          ▼
      └───────────────────────────────────────► ⊕   residual 1
                                                 │ x [B, T, D]
      ┌──────────────────────────────────────────┤
      ▼                                          │
    ┌──────────────────────────────┐             │
    │ norm2: LayerNormalization    │             │
    │   center=False, scale=False  │             │
    │ modulate(·, shift_mlp,       │             │
    │            scale_mlp)        │  chunks 3, 4│
    └──────────────────────────────┘             │
      │                                          │
      ▼                                          │
    ┌──────────────────────────────┐             │
    │ GELUMLPFFN ('gelu_tanh')     │             │
    │  Dense(mlp_ratio*D)          │             │
    │  → gelu(approximate=True)    │             │
    │  → Dense(D)                  │             │
    └──────────────────────────────┘             │
      │                                          │
      ├─ × gate_mlp[:, None, :]  (chunk 5)       │
      ▼                                          ▼
      └───────────────────────────────────────► ⊕   residual 2
                                                 │
                                                 ▼
                                            out [B, T, D]

At initialisation the modulation ``Dense`` is zero in both kernel and bias, so
both gates are exactly ``0`` and :class:`DiTBlock` is the exact identity on
``x`` while :class:`DiTFinalLayer` emits exactly ``0.0``. That is not a defect
to be optimised away -- it is what makes a 28-block stack trainable, and it is
also the premise the attribution probe rests on. Any test asserting "the block
changes its input at init" is wrong here and will fail correctly.

References:
    - Peebles & Xie, 2022. Scalable Diffusion Models with Transformers.
      (https://arxiv.org/abs/2212.09748)
    - Perez et al., 2018. FiLM: Visual Reasoning with a General Conditioning
      Layer. (https://arxiv.org/abs/1709.07871)
    - Esser et al., 2024. Scaling Rectified Flow Transformers for High-Resolution
      Image Synthesis. (https://arxiv.org/abs/2403.03206)
"""

from typing import Any, Dict, List, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.transformers.sd3_adaln import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    modulate,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# The chunk orders -- facts, not style choices
# ---------------------------------------------------------------------

#: The order the 6 block modulation chunks are produced in, matching
#: ``reference/models.py:112`` exactly. Index ``k`` of
#: ``ops.split(adaLN_modulation(c), 6, axis=-1)`` is ``DIT_ADALN_CHUNK_NAMES[k]``.
#: Owned by :class:`~dl_techniques.layers.transformers.sd3_adaln.AdaLayerNormZero`
#: and re-exported here so the guard test can name chunks rather than count
#: slices.
DIT_ADALN_CHUNK_NAMES: Tuple[str, ...] = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)

#: Width multiplier of the block's modulation projection.
NUM_DIT_ADALN_CHUNKS: int = len(DIT_ADALN_CHUNK_NAMES)

# DECISION plan-2026-09-02T170923-1285ed83/D-011
# `AdaLayerNormContinuous` splits its 2-way projection as `scale, shift`
# (diffusers order, sd3_adaln.py:501), while upstream DiT's `FinalLayer` splits
# as `shift, scale` (reference/models.py:132). Do NOT "fix" this by hand-rolling
# a shift-first copy inside this package: with a zero-init kernel AND bias the
# two orders are the same function class under a permutation of the Dense's
# output units, and that permutation is an exact symmetry of the zero
# initialisation, so training is identical up to relabelling. It matters only
# for loading an upstream checkpoint, which is impossible here
# (`pretrained=True` raises). See decisions.md D-011; pinned by
# `test_dit_blocks.py::TestTheFinalLayerChunkOrderIsScaleFirst`.
DIT_FINAL_CHUNK_NAMES: Tuple[str, ...] = ("scale", "shift")

#: Width multiplier of the final layer's modulation projection.
NUM_DIT_FINAL_CHUNKS: int = len(DIT_FINAL_CHUNK_NAMES)


def _unpack_pair_shape(
    input_shape: Any,
    owner: str,
) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
    """Split a ``[x_shape, c_shape]`` build/compute input into its two parts.

    :param input_shape: A list or tuple of exactly two shapes: ``x_shape``
        ``(B, T, hidden_size)`` and ``c_shape`` ``(B, hidden_size)``.
    :type input_shape: Any
    :param owner: Class name, used in the error message.
    :type owner: str
    :return: ``(x_shape, c_shape)`` as tuples.
    :rtype: Tuple[Tuple, Tuple]
    :raises ValueError: If ``input_shape`` is not a pair of shapes.
    """
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError(
            f"{owner} expects an input_shape pair [x_shape, c_shape], got "
            f"{input_shape!r}"
        )
    parts = []
    for name, shape in zip(("x_shape", "c_shape"), input_shape):
        if not isinstance(shape, (list, tuple)):
            raise ValueError(
                f"{owner} expects an input_shape pair [x_shape, c_shape]; "
                f"{name} is {shape!r}, which is not a shape. A bare rank-3 "
                "tuple like (B, T, hidden_size) is the usual mistake -- the "
                "layer takes two inputs, not one."
            )
        parts.append(tuple(shape))
    return parts[0], parts[1]


# ---------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.dit.blocks")
class DiTBlock(keras.layers.Layer):
    """DiT transformer block with 6-way adaLN-Zero conditioning.

    Two gated residual adds in the order ``msa -> mlp``, each gated by its own
    chunk of a single zero-initialised ``SiLU -> Dense(6 * hidden_size)``
    projection of the conditioning vector. See the module docstring for the
    diagram and the exact chunk order.

    The conditioning vector ``c`` is **not** carried forward: it is consumed
    per block and never updated, so block ``n + 1`` receives the same ``c``
    block ``n`` did. Only ``x`` accumulates.

    :param hidden_size: Model width. Must be positive and divisible by
        ``num_heads``.
    :type hidden_size: int
    :param num_heads: Attention head count. Per-head width is
        ``hidden_size // num_heads``, used for both ``key_dim`` and
        ``value_dim``.
    :type num_heads: int
    :param mlp_ratio: Expansion factor of the MLP's hidden width,
        ``int(hidden_size * mlp_ratio)``. Must be positive.
    :type mlp_ratio: float
    :param norm_epsilon: Epsilon of the affine-free ``LayerNormalization``s.
        Explicit on purpose: bare Keras defaults to ``1e-3``, a silent 1000x
        error with no shape symptom. Upstream is ``eps=1e-6``
        (``reference/models.py:100``).
    :type norm_epsilon: float
    :param dropout_rate: Attention-weight dropout and the MLP's dropout.
        Upstream is ``0`` (``reference/models.py:105``).
    :type dropout_rate: float
    :param use_bias: Whether the attention projections and the MLP carry
        biases. Upstream is ``qkv_bias=True`` (``reference/models.py:101``).
    :type use_bias: bool
    :param kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :raises ValueError: If ``hidden_size`` is not positive or not divisible by
        ``num_heads``, if ``num_heads`` is not positive, if ``mlp_ratio`` is not
        positive, or if ``norm_epsilon`` is not positive.

    Input shape:
        A list of two tensors -- ``x`` ``(B, T, hidden_size)`` and ``c``
        ``(B, hidden_size)``.

    Output shape:
        ``(B, T, hidden_size)``.

    Example:
        >>> import keras
        >>> block = DiTBlock(hidden_size=64, num_heads=4)
        >>> x = keras.random.normal((2, 16, 64))
        >>> c = keras.random.normal((2, 64))
        >>> block([x, c]).shape
        (2, 16, 64)

    Note:
        The attention is NON-causal. ``use_causal_mask`` is never passed to
        ``self.attn``, because DiT attends bidirectionally over image patches;
        a causal mask here would make a later patch invisible to an earlier one
        while every shape assertion still passed.

    Attributes:
        adaln: :class:`~dl_techniques.layers.transformers.sd3_adaln.AdaLayerNormZero`
            -- owns ``norm1``, the zero-init ``Dense(6 * hidden_size)`` and the
            ``shift_msa`` / ``scale_msa`` modulation.
        attn: Stock ``keras.layers.MultiHeadAttention``, non-causal.
        norm2: Affine-free ``LayerNormalization`` before the MLP.
        mlp: The tanh-approximate GELU MLP (``'gelu_tanh'`` factory key).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if norm_epsilon <= 0:
            raise ValueError(
                f"norm_epsilon must be positive, got {norm_epsilon}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}"
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.norm_epsilon = float(norm_epsilon)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)

        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        # CREATE sub-layers unconditionally, each with an explicit name so the
        # weight paths are stable across a `.keras` round trip.
        self.adaln = AdaLayerNormZero(
            dim=self.hidden_size,
            eps=self.norm_epsilon,
            name="adaln",
        )
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            dropout=self.dropout_rate,
            use_bias=self.use_bias,
            name="attn",
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.norm_epsilon,
            center=False,
            scale=False,
            name="norm2",
        )
        self.mlp = create_ffn_layer(
            "gelu_tanh",
            hidden_dim=self.mlp_hidden_dim,
            output_dim=self.hidden_size,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            name="mlp",
        )

    def build(self, input_shape: Any) -> None:
        """Build exactly the sub-layer tree ``call`` runs.

        :param input_shape: ``[x_shape, c_shape]``.
        :type input_shape: Any
        :raises ValueError: If ``input_shape`` is not a pair of shapes.
        """
        if self.built:
            return

        x_shape, c_shape = _unpack_pair_shape(input_shape, "DiTBlock")

        self.adaln.build([x_shape, c_shape])
        self.attn.build(query_shape=x_shape, value_shape=x_shape)
        self.norm2.build(x_shape)
        self.mlp.build(x_shape)

        super().build(input_shape)
        logger.debug(
            "Built DiTBlock '%s': hidden=%d heads=%d mlp_hidden=%d",
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
        """Run the two gated residuals in the order ``msa -> mlp``.

        :param inputs: ``[x, c]`` -- ``x`` is ``(B, T, hidden_size)`` and ``c``
            is ``(B, hidden_size)``.
        :type inputs: List[keras.KerasTensor]
        :param training: Keras training flag, threaded to the attention and MLP
            dropouts.
        :type training: Optional[bool]
        :return: ``(B, T, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        x, c = inputs[0], inputs[1]

        # `adaln` owns norm1 and the shift_msa/scale_msa modulation; it returns
        # the four chunks this block still has to apply itself. Chunk order is
        # DIT_ADALN_CHUNK_NAMES.
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln([x, c])

        # DECISION plan-2026-09-02T170923-1285ed83/D-012
        # `use_causal_mask` is deliberately NOT passed. DiT self-attention is
        # bidirectional over image patches (`reference/models.py:101` -- timm
        # `Attention`, which has no mask at all). Do NOT copy
        # `adaln_zero.AdaLNZeroConditionalBlock`'s `use_causal_mask=True`
        # default across: a causal mask changes no shape, no parameter count and
        # no `get_config()`, it only makes a later patch invisible to an earlier
        # one. See decisions.md D-012; pinned by
        # `test_dit_blocks.py::TestTheAttentionIsNonCausal`.
        x = x + keras.ops.expand_dims(gate_msa, axis=1) * self.attn(
            query=x_norm,
            value=x_norm,
            key=x_norm,
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
        """Return ``(B, T, hidden_size)``, derived from the stored config.

        :param input_shape: ``[x_shape, c_shape]``.
        :type input_shape: Any
        :return: ``(B, T, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not a pair of shapes.
        """
        x_shape, _ = _unpack_pair_shape(input_shape, "DiTBlock")
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
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------
# The read-out
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.dit.blocks")
class DiTFinalLayer(keras.layers.Layer):
    """DiT read-out: 2-way adaLN modulation, then a zero-init patch projection.

    Reproduces ``reference/models.py:118-135``. The modulation ``Dense`` and the
    output ``Dense`` are BOTH zero-initialised in kernel and bias, so the layer
    emits exactly ``0.0`` at initialisation -- which is what makes the whole DiT
    output exactly zero before training.

    .. code-block:: text

        c [B, D]                              x [B, T, D]
          │                                      │
          ▼                                      ▼
        ┌────────────────────────────────────────────────┐
        │ AdaLayerNormContinuous(dim=D)                  │
        │   SiLU → Dense(2*D, zeros) → split 2           │
        │   (chunk 0 = scale, chunk 1 = shift -- see     │
        │    D-011; upstream names them the other way)   │
        │   norm: LayerNormalization(center/scale=False) │
        │   out = norm(x) * (1 ⊕ scale) ⊕ shift          │
        └────────────────────────────────────────────────┘
                                                 │ [B, T, D]
                                                 ▼
                              ┌──────────────────────────────────┐
                              │ Dense(patch_size**2 * out_ch)    │
                              │   kernel = zeros, bias = zeros   │
                              └──────────────────────────────────┘
                                                 │
                                                 ▼
                              out [B, T, patch_size**2 * out_channels]

    :param hidden_size: Model width ``D``. Must be positive.
    :type hidden_size: int
    :param patch_size: Side length of a square patch. Must be positive.
    :type patch_size: int
    :param out_channels: Channels the model predicts per pixel. This is
        ``2 * in_channels`` when ``learn_sigma=True``. Must be positive.
    :type out_channels: int
    :param norm_epsilon: Epsilon of the affine-free ``LayerNormalization``.
        Upstream is ``eps=1e-6`` (``reference/models.py:124``).
    :type norm_epsilon: float
    :param use_bias: Whether the output projection carries a bias. Upstream is
        ``bias=True`` (``reference/models.py:125``); the bias is zero-init
        regardless.
    :type use_bias: bool
    :param kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :raises ValueError: If ``hidden_size``, ``patch_size`` or ``out_channels``
        is not a positive integer, or ``norm_epsilon`` is not positive.

    Input shape:
        A list of two tensors -- ``x`` ``(B, T, hidden_size)`` and ``c``
        ``(B, hidden_size)``.

    Output shape:
        ``(B, T, patch_size ** 2 * out_channels)``.

    Example:
        >>> import keras
        >>> final = DiTFinalLayer(hidden_size=64, patch_size=2, out_channels=8)
        >>> x = keras.random.normal((2, 16, 64))
        >>> c = keras.random.normal((2, 64))
        >>> final([x, c]).shape
        (2, 16, 32)

    Attributes:
        adaln: :class:`~dl_techniques.layers.transformers.sd3_adaln.AdaLayerNormContinuous`
            -- the affine-free norm plus the zero-init ``Dense(2 * hidden_size)``.
        linear: The zero-init ``Dense(patch_size ** 2 * out_channels)``.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        norm_epsilon: float = 1e-6,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(
                f"patch_size must be a positive integer, got {patch_size}"
            )
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(
                f"out_channels must be a positive integer, got {out_channels}"
            )
        if norm_epsilon <= 0:
            raise ValueError(
                f"norm_epsilon must be positive, got {norm_epsilon}"
            )

        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.norm_epsilon = float(norm_epsilon)
        self.use_bias = bool(use_bias)

        self.output_dim = self.patch_size * self.patch_size * self.out_channels

        self.adaln = AdaLayerNormContinuous(
            dim=self.hidden_size,
            eps=self.norm_epsilon,
            name="adaln",
        )
        # Two SEPARATE Zeros() instances. A shared Initializer object draws
        # bit-identically forever; that is invisible for Zeros, but the habit is
        # what matters.
        self.linear = keras.layers.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Zeros(),
            name="linear",
        )

    def build(self, input_shape: Any) -> None:
        """Build exactly the sub-layer tree ``call`` runs.

        :param input_shape: ``[x_shape, c_shape]``.
        :type input_shape: Any
        :raises ValueError: If ``input_shape`` is not a pair of shapes.
        """
        if self.built:
            return

        x_shape, c_shape = _unpack_pair_shape(input_shape, "DiTFinalLayer")

        self.adaln.build([x_shape, c_shape])
        self.linear.build(x_shape)

        super().build(input_shape)
        logger.debug(
            "Built DiTFinalLayer '%s': hidden=%d patch=%d out_channels=%d",
            self.name,
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Modulate, then project to ``patch_size ** 2 * out_channels``.

        :param inputs: ``[x, c]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Keras training flag. Unused -- neither sub-layer has a
            training-dependent branch -- but accepted so callers can thread it.
        :type training: Optional[bool]
        :return: ``(B, T, patch_size ** 2 * out_channels)``.
        :rtype: keras.KerasTensor
        """
        x, c = inputs[0], inputs[1]
        return self.linear(self.adaln([x, c], training=training))

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Return the projected shape, derived from the stored config.

        :param input_shape: ``[x_shape, c_shape]``.
        :type input_shape: Any
        :return: ``(B, T, patch_size ** 2 * out_channels)``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not a pair of shapes.
        """
        x_shape, _ = _unpack_pair_shape(input_shape, "DiTFinalLayer")
        return (x_shape[0], x_shape[1], self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "out_channels": self.out_channels,
                "norm_epsilon": self.norm_epsilon,
                "use_bias": self.use_bias,
            }
        )
        return config
