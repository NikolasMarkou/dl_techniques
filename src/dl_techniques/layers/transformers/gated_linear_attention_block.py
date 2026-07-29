"""
A gated linear-attention block: a recurrent, linear-complexity sequence mixer.

The block keeps one matrix-valued state ``S`` per head and rewrites it once per
timestep with a gated outer product::

    S_t   = alpha_t * S_{t-1} + beta_t * (k_t v_t^(1)T)
    out_t = q_t^T S_t + v_t^(2)

``alpha_t`` (how much of the previous state survives) and ``beta_t`` (how
strongly the current key/value pair is written) are *per-head scalars*, obtained
by applying a ``Dense(num_heads)`` projection and a sigmoid to the raw block
input -- they bypass the normalization, convolution and activation that ``q``,
``k`` and ``v`` go through.

The read-out multiplies ``S_t``, the state *after* the current step's write, so
the equivalent closed form is inclusive in ``j = t``::

    out_t = sum_{j<=t} (prod_{l=j+1..t} alpha_l) * beta_j * (q_t . k_j) * v_j^(1)
            + v_t^(2)

The state transition is a plain per-head scalar rescaling. There is no
error-correction term -- the state is never asked to subtract what it already
predicts for ``k_t``, and the transition is not a projection built from
``k_t``. This module implements exactly the two recurrence lines above and
claims nothing beyond them.

Arithmetic cost is one ``head_dim x head_dim`` outer product plus one
vector-matrix product per timestep per head, i.e. it grows linearly with
sequence length rather than quadratically.

Two implementations compute this same function, and ``gated_linear_scan``
dispatches between them on whether the sequence length is statically known:

* **Chunked** (``_chunked_scan``, used whenever the length is a static
  ``int`` -- the ordinary case). Splits the sequence into ``chunk_size``-wide
  blocks. Only the block-to-block state hand-off stays sequential, so the
  sequential depth drops to ``ceil(seq_len / chunk_size)``; everything within a
  block is one batched matmul. Measured on an RTX 4070, float32,
  ``num_heads=4, head_dim=8, chunk_size=64``: **15.2 ms vs 746 ms at
  ``seq_len=128`` (49x), and 33.3 ms vs 2972 ms at ``seq_len=512`` (89x)**.
* **Sequential** (``_sequential_scan``, used when the length is symbolic).
  One ``ops.while_loop`` iteration per timestep. It is also the reference the
  chunked path is tested against.

The two agree to floating-point reassociation, never bitwise. Measured across
``seq_len in {1, 7, 63, 64, 65, 128, 257} x num_heads in {1, 4} x head_dim in
{8, 32}``: worst-case absolute difference ``4.4e-13`` at float64, and
``1.6`` TF32 ulps of output scale at float32 (``4.7e-06`` relative with TF32
disabled).
"""

import keras
from typing import Any, Callable, Dict, Optional, Tuple, Union
from keras import initializers, layers, ops, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.masking import MaskFactory
from ..ffn.factory import create_ffn_from_config, FFNType, FFN_REGISTRY
from ..norms import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------


def _inclusive_causal_mask(size: int, dtype: str) -> keras.KerasTensor:
    """Return the causal **keep** mask as a float multiplier, diagonal INCLUDED.

    ``mask[i, j] = 1`` iff ``j <= i``, else ``0``. The diagonal must be
    included: the ``j = t`` term of the closed-form sum is the current step's
    own write, and the read-out sees it because it reads the state *after* that
    write. Dropping it would make the read-out exclusive and wrong.

    This is a **polarity + dtype adapter** over the canonical
    :meth:`~dl_techniques.utils.masking.MaskFactory.create_causal_mask`, not a
    reimplementation -- the triangle logic lives there and only there. The
    canonical helper returns a *boolean block* mask (``True`` where a position
    must be suppressed, i.e. ``j > i``); the scan needs the complementary
    *keep* mask as a float it can multiply by. Inverting at the call site is the
    house convention for this family: the caller supplies the keep predicate
    because polarity is per-site.

    Do NOT swap this for ``keras.ops.tril``: ``ops.tril`` raises
    ``TypeError: pred must not be a Python bool`` when traced into a graph on
    this Keras/TF version, which broke every ``Model``-level path (``fit``,
    ``predict``, ``jit_compile``, save/load) while eager tests stayed green.

    :param size: Side length of the square mask.
    :type size: int
    :param dtype: Floating dtype of the returned mask.
    :type dtype: str
    :return: Mask of shape ``(size, size)``.
    :rtype: keras.KerasTensor
    """
    blocked = MaskFactory.create_causal_mask(size, dtype="bool")
    return ops.cast(ops.logical_not(blocked), dtype)


@keras.saving.register_keras_serializable()
class GatedLinearAttentionBlock(keras.layers.Layer):
    """
    Recurrent sequence-mixing block with a gated outer-product state.

    Each head carries a ``(head_dim, head_dim)`` state that is rewritten once per
    timestep as ``S_t = alpha_t * S_{t-1} + beta_t * (k_t v_t^(1)T)`` and read out
    *after* that write as ``out_t = q_t^T S_t + v_t^(2)``. See the module
    docstring for the closed form and for what this recurrence is *not*. Q, K and
    V each pass through a configurable normalization, a causal depthwise
    convolution and an activation before the scan; the block output goes through
    either a built-in gated projection or a factory-built FFN. Because
    ``ops.while_loop`` needs a compile-time iteration bound, a hard
    ``max_seq_len`` is required.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────────────────────────────────────┐
        │          GatedLinearAttentionBlock -- gated outer-product state           │
        │                                                                           │
        │  Input [B, S, dim]                                                        │
        │    │                                                                      │
        │    ├───────────────┬───────────────┬───────────────┐                      │
        │    ▼               ▼               ▼               ▼                      │
        │  q_proj          k_proj          v_proj          alpha_proj / beta_proj   │
        │  [B,S,qk_dim]    [B,S,qk_dim]    [B,S,2*qk_dim]  Dense(num_heads) on the  │
        │    ▼               ▼               ▼             RAW block input, then    │
        │  q_norm          k_norm          v_norm          sigmoid. No norm, no     │
        │  PER-HEAD,       PER-HEAD,       WHOLE-TENSOR,   conv, no activation on   │
        │  over head_dim   over head_dim   over 2*qk_dim   this path.               │
        │    ▼               ▼               ▼               ▼                      │
        │  q_conv          k_conv          v_conv          a_t, b_t  [B,S,H]        │
        │  causal depthwise Conv1D (groups = channels),    one scalar per head      │
        │  then `activation` (default 'silu')                                       │
        │    ▼               ▼               ▼                                      │
        │  [B,S,H,d]       [B,S,H,d]       [B,S,H,2d]                               │
        │    │               │               │  ops.split(2, axis=-1),              │
        │    │               │               │  WITHIN each head                    │
        │    │               │               └───┬───────────────────┐              │
        │    │               │                   ▼                   ▼              │
        │    │               │                v1_t [.,d]          v2_t [.,d]        │
        │    └───────┬───────┘                   │                   │              │
        │            ▼                           ▼                   │              │
        │    gated_linear_scan -- dispatches on shape staticness:      │             │
        │      static seq_len -> _chunked_scan  (chunk_size-wide blocks)│            │
        │      symbolic       -> _sequential_scan (one step per t)     │             │
        │      S_t   = a_t * S_{t-1} + b_t * (k_t ⊗ v1_t)             │             │
        │      out_t = q_t · S_t  +  v2_t  ◄──────────────────────────┘             │
        │      out_t reads S_t, i.e. AFTER this step's write.                       │
        │                            ▼                                              │
        │                  reshape [B, S, qk_dim]                                   │
        │                            ▼                                              │
        │    ffn_type is None (default): p = output_proj(x)                         │
        │                                y = sigmoid(output_gate_linear(p)) * p     │
        │    ffn_type set:               y = output_ffn(x)                          │
        │                            ▼                                              │
        │                     Output [B, S, dim]                                    │
        └───────────────────────────────────────────────────────────────────────────┘

    .. note::
        **The value projection is split in two, and the second half is not an
        identity residual.** ``v_proj`` emits ``v_dim = 2 * num_heads * head_dim``
        channels, twice as many as ``q_proj``/``k_proj``. After the reshape to
        ``(batch, seq, num_heads, 2 * head_dim)``, ``ops.split(v_t, 2, axis=-1)``
        divides each head's channels in half: the first ``head_dim`` of that
        head's channels (``v_t^(1)``) is what the outer-product write puts into
        the state, and the second ``head_dim`` (``v_t^(2)``) is added straight
        onto the read-out. The split is therefore *interleaved per head* in the
        flat ``v_dim`` axis -- head ``h``'s write half is flat channel range
        ``[2*h*head_dim, 2*h*head_dim + head_dim)``, not the leading
        ``num_heads * head_dim`` block (verified by construction and by an
        executed probe).

        The caveat that matters: ``v_t^(2)`` is **not** a plain identity or
        skip connection over the block input. It is a slice of the *same*
        processed tensor as ``v_t^(1)`` -- it has already been through
        ``v_proj``, ``v_norm``, the causal ``v_conv`` and the activation (SiLU by
        default). It carries no un-transformed copy of the input, and it is
        outside the recurrence: it depends only on timestep ``t``, never on the
        state. Read it as "half of V bypasses the state" rather than "the block
        has a residual connection".

    :param dim: Model dimension size. Must be positive.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param max_seq_len: Hard upper bound on the sequence length, used as the
        scan's ``maximum_iterations``. Must be positive. See the overflow note
        below for what is and is not guaranteed when it is exceeded.
    :type max_seq_len: int
    :param head_dim: Dimension per head. If None, defaults to dim // num_heads.
    :type head_dim: Optional[int]
    :param conv_kernel_size: Kernel size for short convolution layers. Defaults
        to 4.
    :type conv_kernel_size: int
    :param dropout_rate: Dropout rate for regularization. Defaults to 0.0.
    :type dropout_rate: float
    :param activation: Activation function after convolutions. Defaults to 'silu'.
    :type activation: Union[str, Callable]
    :param normalization_type: Type of normalization for Q, K, V. Defaults to
        'zero_centered_rms_norm'.

        **Q and K are normalized per head**: the tensor is reshaped to
        ``(batch, seq, num_heads, head_dim)``, normalized over ``head_dim``, and
        reshaped back before the causal convolution. The scale weight is therefore
        ``(head_dim,)`` and is shared across heads -- the standard QK-Norm
        convention. **V is deliberately normalized whole-tensor** over the full
        ``v_dim`` axis; that is an intentional asymmetry, not an oversight.

        This assumes the chosen normalization reduces over the last axis only.
        Measured for the 18 factory types on a rank-4 input: 14 reduce over the
        last axis only (including every RMS/layer/band family member and the
        default ``'zero_centered_rms_norm'``); ``'global_response_norm'`` also
        reduces over the head axis by design; and ``'decoupled_max_logit'``,
        ``'dml_plus_focal'``, ``'dml_plus_center'`` are unusable here at any rank
        (they change or drop the feature axis, which already breaks the
        convolution that follows).
    :type normalization_type: NormalizationType
    :param q_norm_args: Optional arguments for Q normalization layer.
    :type q_norm_args: Optional[Dict[str, Any]]
    :param k_norm_args: Optional arguments for K normalization layer.
    :type k_norm_args: Optional[Dict[str, Any]]
    :param v_norm_args: Optional arguments for V normalization layer.
    :type v_norm_args: Optional[Dict[str, Any]]
    :param ffn_type: Type of FFN for the output stage. If ``None`` (default),
        the block uses its built-in gated projection
        ``y = sigmoid(output_gate_linear(p)) * p`` with ``p = output_proj(x)``;
        otherwise the FFN is built by ``create_ffn_from_config``.
    :type ffn_type: Optional[FFNType]
    :param ffn_args: Optional arguments for the custom FFN layer. Applied last,
        after this layer's own generic defaults, and passed through unfiltered
        -- the factory rejects an unknown key loudly.
    :type ffn_args: Optional[Dict[str, Any]]
    :param intermediate_size: Intermediate size for standard FFNs. Defaults to
        dim * 4 if not provided.
    :type intermediate_size: Optional[int]
    :param chunk_size: Block width for the chunked scan, in timesteps. Defaults
        to 64. A pure performance knob: the result is the same (to floating-point
        reassociation) for every value, which
        ``test_result_is_independent_of_chunk_size`` pins at float64 for
        ``chunk_size in {1, 7, 16, 64, 256}``. Larger blocks mean fewer
        sequential steps but a wider ``(chunk_size, chunk_size)`` intra-block
        matmul, so the cost is quadratic in this value and linear in the step
        count. Only used when the sequence length is statically known; the
        symbolic-length fallback is timestep-sequential and ignores it.
    :type chunk_size: int
    :param use_bias: Whether to use bias in linear layers. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class.

    :raises ValueError: At construction, if any parameter is outside its
        admissible range (including ``intermediate_size <= 0`` when given).
    :raises ValueError: At ``build()`` / ``call()`` time, if the input's
        sequence axis is a *statically known* integer greater than
        ``max_seq_len``.

    .. note::
        **What the overflow guard does and does not cover.** The recurrent scan
        runs under ``ops.while_loop`` with ``maximum_iterations=max_seq_len``,
        so a sequence longer than ``max_seq_len`` cannot be computed: every
        timestep past the cap stays at the zero-initialized buffer value.

        *Guaranteed*: when the sequence length is statically known -- the
        ordinary case of calling the layer on a concrete tensor, or building it
        on a shape with a concrete sequence axis -- an over-long input raises
        ``ValueError`` naming both the offending length and ``max_seq_len``.

        *Not guaranteed*: when the sequence axis is ``None``/symbolic (for
        example ``keras.Input(shape=(None, dim))``), the guard cannot fire --
        ``keras.ops`` offers no portable way to raise from inside a traced
        graph. In that case ``build()`` emits a one-time ``logger.warning`` and
        the silent truncation described above remains possible at runtime. Size
        ``max_seq_len`` for the longest sequence you intend to feed.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: Optional[int] = None,
        conv_kernel_size: int = 4,
        dropout_rate: float = 0.0,
        activation: Union[str, Callable] = "silu",
        normalization_type: NormalizationType = "zero_centered_rms_norm",
        q_norm_args: Optional[Dict[str, Any]] = None,
        k_norm_args: Optional[Dict[str, Any]] = None,
        v_norm_args: Optional[Dict[str, Any]] = None,
        ffn_type: Optional[FFNType] = None,
        ffn_args: Optional[Dict[str, Any]] = None,
        intermediate_size: Optional[int] = None,
        chunk_size: int = 64,
        use_bias: bool = False,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(
            dim,
            num_heads,
            head_dim,
            conv_kernel_size,
            dropout_rate,
            max_seq_len,
            intermediate_size,
            chunk_size,
        )

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.normalization_type = normalization_type
        self.q_norm_args = q_norm_args or (
            {"epsilon": 1e-5, "use_scale": True}
            if normalization_type == "zero_centered_rms_norm"
            else {}
        )
        self.k_norm_args = k_norm_args or (
            {"epsilon": 1e-5, "use_scale": True}
            if normalization_type == "zero_centered_rms_norm"
            else {}
        )
        self.v_norm_args = v_norm_args or (
            {"epsilon": 1e-5, "use_scale": True}
            if normalization_type == "zero_centered_rms_norm"
            else {}
        )
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args or {}
        self.intermediate_size = intermediate_size
        self.chunk_size = chunk_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Compute dimensions
        self.qk_dim = self.num_heads * self.head_dim
        self.v_dim = self.num_heads * self.head_dim * 2

        # Q/K/V projections
        self.q_proj = layers.Dense(self.qk_dim, use_bias=use_bias, name="q_proj")
        self.k_proj = layers.Dense(self.qk_dim, use_bias=use_bias, name="k_proj")
        self.v_proj = layers.Dense(self.v_dim, use_bias=use_bias, name="v_proj")

        # Configurable Normalization layers
        self.q_norm = self._create_normalization_layer("q_norm", self.q_norm_args)
        self.k_norm = self._create_normalization_layer("k_norm", self.k_norm_args)
        self.v_norm = self._create_normalization_layer("v_norm", self.v_norm_args)

        # Short convolution layers (depthwise separable)
        self.q_conv = layers.Conv1D(
            self.qk_dim, conv_kernel_size, padding="causal", groups=self.qk_dim, name="q_conv"
        )
        self.k_conv = layers.Conv1D(
            self.qk_dim, conv_kernel_size, padding="causal", groups=self.qk_dim, name="k_conv"
        )
        self.v_conv = layers.Conv1D(
            self.v_dim, conv_kernel_size, padding="causal", groups=self.v_dim, name="v_conv"
        )

        # Gating parameter projections (alpha and beta)
        self.alpha_proj = layers.Dense(self.num_heads, use_bias=use_bias, name="alpha_proj")
        self.beta_proj = layers.Dense(self.num_heads, use_bias=use_bias, name="beta_proj")

        # Configurable Output FFN
        self.use_default_ffn = self.ffn_type is None
        if self.use_default_ffn:
            self.output_proj = layers.Dense(self.dim, use_bias=use_bias, name="output_proj")
            self.output_gate_linear = layers.Dense(
                self.dim, use_bias=use_bias, name="output_gate_linear"
            )
        else:
            self.output_ffn = self._create_ffn_layer("output_ffn")

        # Configurable activation layer
        self.activation_layer = layers.Activation(self.activation, name="conv_activation")

        # Dropout for regularization
        self.dropout = (
            layers.Dropout(dropout_rate, name="dropout") if dropout_rate > 0.0 else None
        )

        logger.info(
            f"GatedLinearAttentionBlock initialized: dim={dim}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}, "
            f"max_seq_len={self.max_seq_len}, activation='{self.activation}', "
            f"norm='{self.normalization_type}', ffn='{self.ffn_type or 'default_gated'}'"
        )

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int],
        conv_kernel_size: int,
        dropout_rate: float,
        max_seq_len: int,
        intermediate_size: Optional[int] = None,
        chunk_size: int = 64,
    ) -> None:
        """Validate layer initialization parameters.

        :param dim: Model dimension.
        :type dim: int
        :param num_heads: Number of heads.
        :type num_heads: int
        :param head_dim: Per-head dimension.
        :type head_dim: Optional[int]
        :param conv_kernel_size: Convolution kernel size.
        :type conv_kernel_size: int
        :param dropout_rate: Dropout rate.
        :type dropout_rate: float
        :param max_seq_len: Maximum sequence length.
        :type max_seq_len: int
        :param intermediate_size: Intermediate size for the optional FFN stage.
            ``None`` means "derive from ``dim``" and is always valid.
        :type intermediate_size: Optional[int]
        :param chunk_size: Chunk width for the blockwise scan.
        :type chunk_size: int
        :raises ValueError: If any parameter is outside its admissible range.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim is None and dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}) "
                "when head_dim is None"
            )
        if conv_kernel_size <= 0:
            raise ValueError(
                f"conv_kernel_size must be positive, got {conv_kernel_size}"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if intermediate_size is not None and intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive when given, "
                f"got {intermediate_size}"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    def _validate_seq_len(self, seq_len: int) -> None:
        """Reject a statically known sequence length larger than ``max_seq_len``.

        The recurrent scan runs under ``ops.while_loop`` with
        ``maximum_iterations=max_seq_len``, so any timestep past that cap is
        never written and silently reads back as zero. This turns that silent
        corruption into a loud failure.

        :param seq_len: Statically known sequence length.
        :type seq_len: int
        :raises ValueError: If ``seq_len`` exceeds ``self.max_seq_len``.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_seq_len "
                f"({self.max_seq_len}). The recurrent scan is capped at "
                f"max_seq_len steps, so every timestep from index "
                f"{self.max_seq_len} onwards would be silently zero. "
                f"Increase max_seq_len to at least {seq_len}."
            )

    @staticmethod
    def _static_seq_len(inputs: Any) -> Optional[int]:
        """Return the sequence length as a Python ``int``, or ``None``.

        Only a statically known dimension is returned. Symbolic or unknown
        sequence axes yield ``None`` -- a traced tensor's shape entry must never
        reach a Python ``if``, which is exactly why the guard cannot fire under
        a dynamic shape.

        :param inputs: Input tensor (eager, symbolic or Keras) of rank 3.
        :type inputs: Any
        :return: The static sequence length, or ``None`` if it is not static.
        :rtype: Optional[int]
        """
        shape = getattr(inputs, "shape", None)
        if shape is None or len(shape) != 3:
            return None
        try:
            dim = shape[1]
        except (TypeError, IndexError, ValueError):
            return None
        if dim is None:
            return None
        try:
            return int(dim)
        except (TypeError, ValueError):
            return None

    def _create_normalization_layer(
        self, name: str, custom_args: Dict[str, Any]
    ) -> keras.layers.Layer:
        """Create a normalization layer from the factory.

        :param name: Layer name.
        :type name: str
        :param custom_args: Custom arguments for the normalization layer.
        :type custom_args: Dict[str, Any]
        :return: Normalization layer instance.
        :rtype: keras.layers.Layer
        """
        try:
            return create_normalization_layer(
                normalization_type=self.normalization_type, name=name, **custom_args
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create '{self.normalization_type}' norm layer named '{name}'. "
                f"Check parameter compatibility. Custom args: {custom_args}. Error: {e}"
            )

    def _create_ffn_layer(self, name: str) -> keras.layers.Layer:
        """Create an FFN layer from the factory for the output stage.

        :param name: Layer name.
        :type name: str
        :return: FFN layer instance.
        :rtype: keras.layers.Layer
        """
        # Do NOT mutate `self.intermediate_size` here: `get_config()` serializes it, so
        # overwriting a caller's `None` with a computed default made a reloaded layer's
        # config differ from the one that was built. Resolve the effective value locally.
        effective_intermediate = (
            self.dim * 4 if self.intermediate_size is None else self.intermediate_size
        )

        ffn_info = FFN_REGISTRY.get(self.ffn_type)
        if ffn_info is None:
            raise ValueError(
                f"Unknown ffn_type '{self.ffn_type}'. "
                f"Available: {sorted(FFN_REGISTRY)}."
            )
        valid_ffn_params = set(ffn_info["required_params"]) | set(
            ffn_info["optional_params"]
        )

        # `hidden_dim` is now the universal sizing knob across FFN types -- SwiGLU used to
        # be the odd one out (sized only by `ffn_expansion_factor`), so passing it a
        # hidden_dim had the value SILENTLY DROPPED by the factory's kwarg filter and the
        # FFN was built at SwiGLU's default expansion instead. `SwiGLUFFN` now accepts an
        # explicit `hidden_dim` like the rest, so `intermediate_size` is honored uniformly.
        config = {
            "type": self.ffn_type,
            "name": name,
            "output_dim": self.dim,
            "hidden_dim": effective_intermediate,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
        }

        # Drop OUR OWN generic defaults that this ffn_type does not accept. These are this
        # layer's conveniences, not the caller's explicit intent, so filtering them is
        # correct -- unlike `ffn_args`, which the factory now rejects loudly if unknown.
        config = {
            k: v for k, v in config.items()
            if k in valid_ffn_params or k in ("type", "name")
        }
        config.update(self.ffn_args)
        try:
            return create_ffn_from_config(config)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create '{self.ffn_type}' FFN layer. "
                f"Check for parameter incompatibility. Custom args: {self.ffn_args}. Error: {e}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers.

        :param input_shape: Shape tuple (batch_size, sequence_length, dim).
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the rank is not 3, if the feature axis does not
            match ``dim``, or if a *statically known* sequence axis exceeds
            ``max_seq_len``.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")
        batch_size, seq_len, features = input_shape
        if features != self.dim:
            raise ValueError(f"Input feature dim ({features}) must match layer dim ({self.dim})")

        if seq_len is not None:
            self._validate_seq_len(int(seq_len))
        else:
            # DECISION plan-2026-07-29-adbe605f/D-002
            # The overflow guard is STATIC-ONLY, deliberately. Do NOT "fix" this
            # branch by raising here (it would break every legitimate
            # `keras.Input(shape=(None, dim))` model) and do NOT replace it with
            # a graph-time assert: `keras.ops` has no portable way to raise from
            # inside a traced graph, and a backend-specific `tf.debugging.assert`
            # would make this layer non-portable for a guarantee we then could
            # not state uniformly. Under a symbolic sequence axis the
            # `maximum_iterations=max_seq_len` cap still silently truncates, and
            # the docstring says so rather than claiming dynamic coverage.
            logger.warning(
                "GatedLinearAttentionBlock '%s' built with an unknown "
                "(symbolic/dynamic) sequence axis: the max_seq_len=%d overflow "
                "guard CANNOT fire for this layer. If a batch longer than %d is "
                "fed at runtime, the recurrent scan silently returns zeros for "
                "every timestep past index %d.",
                self.name,
                self.max_seq_len,
                self.max_seq_len,
                self.max_seq_len - 1,
            )

        # Set common initializers/regularizers for dense layers
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.alpha_proj, self.beta_proj]:
            layer.kernel_initializer = self.kernel_initializer
            layer.bias_initializer = self.bias_initializer
            layer.kernel_regularizer = self.kernel_regularizer
            layer.bias_regularizer = self.bias_regularizer
        for layer in [self.q_conv, self.k_conv, self.v_conv]:
            layer.kernel_initializer = self.kernel_initializer
            layer.bias_initializer = self.bias_initializer

        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        q_shape = (batch_size, seq_len, self.qk_dim)
        k_shape = (batch_size, seq_len, self.qk_dim)
        v_shape = (batch_size, seq_len, self.v_dim)
        # DECISION plan-2026-07-29-adbe605f/D-003
        # q_norm/k_norm are built on the PER-HEAD shape (batch, seq, heads, head_dim),
        # not on the flat (batch, seq, qk_dim). Their scale weight is therefore
        # (head_dim,), shared across heads -- the standard QK-Norm convention.
        # Do NOT "simplify" this back to `q_shape`: a last-axis RMS statistic taken
        # over the concatenated qk_dim mixes every head into one denominator, which
        # is the defect this step exists to close (measured: perturbing head 0's
        # input moved head 1's output by 1.03 at rank 3, by exactly 0.0 at rank 4).
        # v_norm stays WHOLE-TENSOR on purpose -- see the class docstring.
        qk_head_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        self.q_norm.build(qk_head_shape)
        self.k_norm.build(qk_head_shape)
        self.v_norm.build(v_shape)
        self.q_conv.build(q_shape)
        self.k_conv.build(k_shape)
        self.v_conv.build(v_shape)
        self.alpha_proj.build(input_shape)
        self.beta_proj.build(input_shape)

        self.activation_layer.build(q_shape)

        ffn_input_shape = (batch_size, seq_len, self.qk_dim)
        if self.use_default_ffn:
            self.output_proj.kernel_initializer = self.kernel_initializer
            self.output_proj.bias_initializer = self.bias_initializer
            self.output_proj.build(ffn_input_shape)
            self.output_gate_linear.kernel_initializer = self.kernel_initializer
            self.output_gate_linear.bias_initializer = self.bias_initializer
            self.output_gate_linear.build((batch_size, seq_len, self.dim))
        else:
            self.output_ffn.build(ffn_input_shape)

        super().build(input_shape)

    def gated_linear_scan(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        alpha: keras.KerasTensor,
        beta: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the gated outer-product recurrence, dispatching on shape staticness.

        Two implementations compute the same function:

        * :meth:`_chunked_scan` -- blockwise, ``ceil(seq_len / chunk_size)``
          sequential steps. Chosen whenever the sequence length is a static
          Python ``int``, which is the ordinary eager/compiled case.
        * :meth:`_sequential_scan` -- one step per timestep. Chosen when the
          sequence length is symbolic, because the chunked form needs a concrete
          length to lay out its chunk grid. Also the reference implementation the
          chunked path is tested against.

        The two agree to floating-point reassociation only, never bitwise; see
        the class docstring's note on the equivalence tolerances.

        :param q: Query tensor of shape (batch, seq, heads, head_dim).
        :type q: keras.KerasTensor
        :param k: Key tensor of shape (batch, seq, heads, head_dim).
        :type k: keras.KerasTensor
        :param v: Value tensor of shape (batch, seq, heads, 2*head_dim).
        :type v: keras.KerasTensor
        :param alpha: Persistence gate of shape (batch, seq, heads).
        :type alpha: keras.KerasTensor
        :param beta: Write-strength gate of shape (batch, seq, heads).
        :type beta: keras.KerasTensor
        :param training: Accepted for signature uniformity; the scan has no
            training-dependent behaviour (dropout is applied by ``call()``).
        :type training: Optional[bool]
        :return: Output tensor of shape (batch, seq, heads, head_dim).
        :rtype: keras.KerasTensor
        """
        static_seq_len = None
        shape = getattr(q, "shape", None)
        if shape is not None and len(shape) == 4 and shape[1] is not None:
            try:
                static_seq_len = int(shape[1])
            except (TypeError, ValueError):
                static_seq_len = None

        if static_seq_len is None:
            return self._sequential_scan(q, k, v, alpha, beta)
        return self._chunked_scan(q, k, v, alpha, beta, static_seq_len)

    def _sequential_scan(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        alpha: keras.KerasTensor,
        beta: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Run the recurrence one timestep at a time with ``ops.while_loop``.

        One loop iteration per timestep ``t``::

            S_t   = alpha_t * S_{t-1} + beta_t * (k_t v_t^(1)T)
            out_t = q_t^T S_t + v_t^(2)

        ``S`` starts at zeros with shape ``(batch, heads, head_dim, head_dim)``.
        ``v`` is split in half along its last axis *inside* the loop:
        ``v_t^(1)`` is written into the state, ``v_t^(2)`` is added to the
        read-out. The read-out uses ``S_t``, i.e. the state after this step's
        write, so it is inclusive in ``j = t``.

        The loop carries ``maximum_iterations=self.max_seq_len``; a sequence
        longer than that is rejected up front by ``_validate_seq_len`` whenever
        its length is statically known (see the class docstring for the
        symbolic-shape gap).

        :param q: Query tensor of shape (batch, seq, heads, head_dim).
        :type q: keras.KerasTensor
        :param k: Key tensor of shape (batch, seq, heads, head_dim).
        :type k: keras.KerasTensor
        :param v: Value tensor of shape (batch, seq, heads, 2*head_dim).
        :type v: keras.KerasTensor
        :param alpha: Persistence gate of shape (batch, seq, heads), one scalar
            per head per timestep.
        :type alpha: keras.KerasTensor
        :param beta: Write-strength gate of shape (batch, seq, heads), one
            scalar per head per timestep.
        :type beta: keras.KerasTensor
        :return: Output tensor of shape (batch, seq, heads, head_dim).
        :rtype: keras.KerasTensor
        """
        batch_size, seq_len, _, _ = ops.shape(q)

        i = ops.convert_to_tensor(0, dtype="int32")
        initial_state = ops.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim), dtype=q.dtype
        )
        outputs_transposed = ops.zeros(
            (seq_len, batch_size, self.num_heads, self.head_dim), dtype=q.dtype
        )

        def condition(i, state, outputs):
            return ops.less(i, seq_len)

        def body(i, state, outputs):
            q_t, k_t, v_t = q[:, i], k[:, i], v[:, i]
            alpha_t, beta_t = alpha[:, i], beta[:, i]
            v_t_1, v_t_2 = ops.split(v_t, 2, axis=-1)

            k_exp = ops.expand_dims(k_t, -1)
            v_exp = ops.expand_dims(v_t_1, -2)
            write = ops.matmul(k_exp, v_exp)

            beta_exp = ops.expand_dims(ops.expand_dims(beta_t, -1), -1)
            alpha_exp = ops.expand_dims(ops.expand_dims(alpha_t, -1), -1)
            next_state = alpha_exp * state + beta_exp * write

            q_exp = ops.expand_dims(q_t, -2)
            output_t = ops.squeeze(ops.matmul(q_exp, next_state), axis=-2) + v_t_2

            next_outputs = ops.scatter_update(
                outputs, ops.expand_dims([i], -1), ops.expand_dims(output_t, 0)
            )
            return i + 1, next_state, next_outputs

        _, _, final_outputs = ops.while_loop(
            cond=condition,
            body=body,
            loop_vars=(i, initial_state, outputs_transposed),
            maximum_iterations=self.max_seq_len,
        )
        return ops.transpose(final_outputs, [1, 0, 2, 3])

    def _chunked_scan(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        alpha: keras.KerasTensor,
        beta: keras.KerasTensor,
        seq_len: int,
    ) -> keras.KerasTensor:
        """Compute the same recurrence blockwise, in ``ceil(seq/chunk)`` steps.

        Unrolling the recurrence gives a closed form. With the inclusive
        cumulative log-gate ``d_t = sum_{l<=t} log alpha_l``::

            out_t = sum_{j<=t} exp(d_t - d_j) * beta_j * (q_t . k_j) * v_j^(1)
                    + v_t^(2)

        The ``j = t`` term carries ``exp(0) = 1``, which is what makes the
        read-out inclusive -- the same convention ``_sequential_scan`` gets from
        reading the state *after* the write.

        Splitting the sum at chunk boundaries, with ``D_i`` the cumulative
        log-gate *within* a chunk and ``S_start`` the state entering it::

            intra_i = sum_{i'<=i} exp(D_i - D_i') * beta_i' * (q_i . k_i') * v_i'
            inter_i = exp(D_i) * (q_i . S_start)
            S_end   = exp(D_last) * S_start
                      + sum_i' exp(D_last - D_i') * beta_i' * k_i' v_i'^T

        Only the ``S_start`` chain is sequential, and it advances one chunk at a
        time; both the intra-chunk triangle and the per-chunk state
        contributions are computed for every chunk at once.

        Sequences shorter than a whole number of chunks are padded with
        gate-neutral values (``alpha = 1`` so the cumulative gate stays flat,
        ``beta = 0`` and ``q = k = v = 0`` so nothing is written or read) and the
        output is sliced back.

        :param q: Query tensor of shape (batch, seq, heads, head_dim).
        :type q: keras.KerasTensor
        :param k: Key tensor of shape (batch, seq, heads, head_dim).
        :type k: keras.KerasTensor
        :param v: Value tensor of shape (batch, seq, heads, 2*head_dim).
        :type v: keras.KerasTensor
        :param alpha: Persistence gate of shape (batch, seq, heads).
        :type alpha: keras.KerasTensor
        :param beta: Write-strength gate of shape (batch, seq, heads).
        :type beta: keras.KerasTensor
        :param seq_len: Statically known sequence length.
        :type seq_len: int
        :return: Output tensor of shape (batch, seq, heads, head_dim).
        :rtype: keras.KerasTensor
        """
        compute_dtype = keras.backend.standardize_dtype(q.dtype)
        # DECISION plan-2026-07-29-adbe605f/D-012
        # Every gate factor below is exp() of a DIFFERENCE of cumulative
        # log-gates whose argument is <= 0, so all of them lie in (0, 1].
        # Do NOT "simplify" this into the textbook two-vector form
        #     q~ = q * exp(D),   k~ = k * exp(-D) * beta
        # which is algebraically identical but materializes the unbounded
        # reciprocal exp(-D_j). Measured: that form OVERFLOWS float32 at
        # chunk_size=64 for alpha=0.1, and random-init sigmoid alpha already
        # reaches 0.0111, so it breaks at initialization -- not just in a
        # contrived corner. The pairwise-difference form below never forms a
        # reciprocal; underflow to 0 is the numerically correct answer there
        # (a gate that has decayed by 1e-60 contributes nothing).
        #
        # float32 is a FLOOR for the gate arithmetic, not a target: fp16/bf16
        # are promoted (a float16 log/exp cannot carry a cumulative sum
        # spanning hundreds of nats), but float64 must NOT be demoted.
        gate_dtype = "float64" if compute_dtype == "float64" else "float32"

        batch_size = ops.shape(q)[0]
        chunk = self.chunk_size
        n_chunks = (seq_len + chunk - 1) // chunk
        pad = n_chunks * chunk - seq_len

        if pad > 0:
            q = ops.pad(q, [[0, 0], [0, pad], [0, 0], [0, 0]])
            k = ops.pad(k, [[0, 0], [0, pad], [0, 0], [0, 0]])
            v = ops.pad(v, [[0, 0], [0, pad], [0, 0], [0, 0]])
            beta = ops.pad(beta, [[0, 0], [0, pad], [0, 0]])
            alpha = ops.pad(
                alpha, [[0, 0], [0, pad], [0, 0]], constant_values=1.0
            )

        v_write = v[..., : self.head_dim]
        v_residual = v[..., self.head_dim :]

        def to_chunks(x, last_dim):
            x = ops.reshape(
                x, (batch_size, n_chunks, chunk, self.num_heads, last_dim)
            )
            return ops.transpose(x, [0, 3, 1, 2, 4])

        q_c = to_chunks(q, self.head_dim)
        k_c = to_chunks(k, self.head_dim)
        v_c = to_chunks(v_write, self.head_dim)

        # alpha = sigmoid(.) is strictly positive mathematically but can
        # underflow to exactly 0 in fp16, and log(0) = -inf would poison every
        # difference into NaN. Floor it before the log.
        alpha_g = ops.maximum(ops.cast(alpha, gate_dtype), 1e-30)
        log_alpha = ops.reshape(
            ops.log(alpha_g), (batch_size, n_chunks, chunk, self.num_heads)
        )
        log_alpha = ops.transpose(log_alpha, [0, 3, 1, 2])
        cum = ops.cumsum(log_alpha, axis=-1)
        cum_last = cum[..., -1:]

        beta_g = ops.reshape(
            ops.cast(beta, gate_dtype),
            (batch_size, n_chunks, chunk, self.num_heads),
        )
        beta_g = ops.transpose(beta_g, [0, 3, 1, 2])

        causal = _inclusive_causal_mask(chunk, gate_dtype)
        decay = ops.exp(
            ops.minimum(ops.expand_dims(cum, -1) - ops.expand_dims(cum, -2), 0.0)
        )
        decay = decay * causal

        scores = ops.matmul(q_c, ops.swapaxes(k_c, -1, -2))
        weighted = (
            ops.cast(scores, gate_dtype) * decay * ops.expand_dims(beta_g, -2)
        )
        intra = ops.matmul(ops.cast(weighted, compute_dtype), v_c)

        write_weight = ops.exp(ops.minimum(cum_last - cum, 0.0)) * beta_g
        k_weighted = k_c * ops.expand_dims(
            ops.cast(write_weight, compute_dtype), -1
        )
        chunk_state = ops.matmul(ops.swapaxes(k_weighted, -1, -2), v_c)
        chunk_decay = ops.reshape(
            ops.cast(ops.exp(cum_last), compute_dtype),
            (batch_size, self.num_heads, n_chunks, 1, 1),
        )

        state = ops.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim),
            dtype=compute_dtype,
        )
        entry_states = ops.zeros(
            (n_chunks, batch_size, self.num_heads, self.head_dim, self.head_dim),
            dtype=compute_dtype,
        )
        c = ops.convert_to_tensor(0, dtype="int32")

        def condition(c, state, entries):
            return ops.less(c, n_chunks)

        def body(c, state, entries):
            entries = ops.scatter_update(
                entries, ops.expand_dims([c], -1), ops.expand_dims(state, 0)
            )
            state = chunk_decay[:, :, c] * state + chunk_state[:, :, c]
            return c + 1, state, entries

        _, _, entry_states = ops.while_loop(
            cond=condition,
            body=body,
            loop_vars=(c, state, entry_states),
            maximum_iterations=n_chunks,
        )
        entry_states = ops.transpose(entry_states, [1, 2, 0, 3, 4])

        inter = ops.matmul(q_c, entry_states) * ops.cast(
            ops.expand_dims(ops.exp(cum), -1), compute_dtype
        )

        out = ops.transpose(intra + inter, [0, 2, 3, 1, 4])
        out = ops.reshape(
            out, (batch_size, n_chunks * chunk, self.num_heads, self.head_dim)
        )
        if pad > 0:
            out = out[:, :seq_len]
            v_residual = v_residual[:, :seq_len]
        return out + v_residual

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: project, normalize, convolve, scan, then project out.

        ``alpha``/``beta`` are read off the *raw* ``inputs``, not off the
        normalized/convolved Q/K/V stream. Dropout, when enabled, is applied to
        ``q_heads``/``k_heads``/``v_heads`` immediately before the scan.

        :param inputs: Input tensor of shape (batch, seq_len, dim).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch, seq_len, dim).
        :rtype: keras.KerasTensor
        :raises ValueError: If the input's sequence length is statically known
            and exceeds ``max_seq_len``. A built layer can be re-called at a
            different length than it was built at, so this re-checks rather than
            trusting ``build()``.
        """
        static_seq_len = self._static_seq_len(inputs)
        if static_seq_len is not None:
            self._validate_seq_len(static_seq_len)

        batch_size, seq_len, _ = ops.shape(inputs)

        q = self.q_proj(inputs, training=training)
        k = self.k_proj(inputs, training=training)
        v = self.v_proj(inputs, training=training)

        # DECISION plan-2026-07-29-adbe605f/D-003
        # Per-head Q/K normalization. Only the SCOPE of the statistic changes; the
        # pipeline order `proj -> norm -> conv -> activation` is deliberately kept.
        # Moving the norm after the head reshape in the pipeline would also move it
        # after the causal conv and the SiLU -- a second, unrequested change to what
        # the norm sees. So we reshape to per-head, normalize, and reshape straight
        # back so the conv's input layout is unchanged.
        # v_norm is deliberately NOT per-head (whole-tensor, over v_dim).
        q_heads_pre = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k_heads_pre = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        q_norm = ops.reshape(
            self.q_norm(q_heads_pre, training=training),
            (batch_size, seq_len, self.qk_dim),
        )
        k_norm = ops.reshape(
            self.k_norm(k_heads_pre, training=training),
            (batch_size, seq_len, self.qk_dim),
        )
        v_norm = self.v_norm(v, training=training)

        q_conv = self.activation_layer(self.q_conv(q_norm, training=training))
        k_conv = self.activation_layer(self.k_conv(k_norm, training=training))
        v_conv = self.activation_layer(self.v_conv(v_norm, training=training))

        q_heads = ops.reshape(q_conv, (batch_size, seq_len, self.num_heads, self.head_dim))
        k_heads = ops.reshape(k_conv, (batch_size, seq_len, self.num_heads, self.head_dim))
        v_heads = ops.reshape(v_conv, (batch_size, seq_len, self.num_heads, 2 * self.head_dim))

        alpha = ops.sigmoid(self.alpha_proj(inputs, training=training))
        beta = ops.sigmoid(self.beta_proj(inputs, training=training))

        if training and self.dropout is not None:
            q_heads = self.dropout(q_heads, training=training)
            k_heads = self.dropout(k_heads, training=training)
            v_heads = self.dropout(v_heads, training=training)

        scan_output = self.gated_linear_scan(q_heads, k_heads, v_heads, alpha, beta)
        scan_output = ops.reshape(scan_output, (batch_size, seq_len, self.qk_dim))

        if self.use_default_ffn:
            projected_output = self.output_proj(scan_output, training=training)
            gate = ops.sigmoid(self.output_gate_linear(projected_output, training=training))
            gated_output = gate * projected_output
        else:
            gated_output = self.output_ffn(scan_output, training=training)

        return gated_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape given input shape.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "max_seq_len": self.max_seq_len,
                "head_dim": self.head_dim,
                "conv_kernel_size": self.conv_kernel_size,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "normalization_type": self.normalization_type,
                "q_norm_args": self.q_norm_args,
                "k_norm_args": self.k_norm_args,
                "v_norm_args": self.v_norm_args,
                "ffn_type": self.ffn_type,
                "ffn_args": self.ffn_args,
                "chunk_size": self.chunk_size,
                "intermediate_size": self.intermediate_size,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            }
        )
        return config

# ---------------------------------------------------------------------
