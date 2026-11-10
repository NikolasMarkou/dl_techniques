"""
KimiDeltaAttention, the Kimi Delta Attention (KDA) token mixer from Kimi
Linear: a gated delta-rule linear attention with channel-wise forgetting.

Softmax attention keeps every past key and value in a cache that grows with
the sequence. KDA keeps one fixed-size matrix state per head instead,
``S in R^{d_k x d_v}``, and edits it once per token. Each step first decays
the state, then asks it what value it currently associates with the new key,
and writes back only the error between that prediction and the true value
(the delta rule). The decay is a per-channel diagonal ``Diag(alpha_t)``, not
Gated DeltaNet's single scalar per head, so every key channel of the state
chooses its own forgetting rate.

Per head, at step ``t``::

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

``alpha_t in (0, 1]^{d_k}`` comes from a low-rank projection,
``log alpha_t = -exp(A_log) * softplus(W_up W_down x_t + dt_bias)``.
``beta_t in (0, 1)`` is a per-head sigmoid write strength. ``q`` and ``k``
pass through a causal depthwise short convolution and SiLU, are L2-normalized
per head, and ``q`` is scaled by ``1 / sqrt(d_k)``. The output is per-head
RMS-normalized, gated by ``sigmoid(W_up W_down x)`` and projected back to
``dim``.

The transition ``(I - beta k k^T) Diag(alpha)`` is diagonal-plus-low-rank.
The ``'chunk'`` mode exploits it: inside a chunk of ``C`` tokens the delta
rule's sequential dependency is a unit lower-triangular system, solved once
per chunk (the WY representation), so only ``N / C`` state updates run in
sequence. ``'recurrent'`` runs the per-token recurrence directly and is the
exact reference. Both are causal by construction, cost ``O(N)`` in the
sequence length, and carry an ``O(d_k * d_v)`` state per head regardless of
``N``.

Kimi Linear interleaves KDA with full attention layers (three KDA to one
MLA). That stacking belongs to the transformer block, not to this layer.

References:
    - Kimi Team, 2025. Kimi Linear: An Expressive, Efficient Attention
      Architecture. (https://arxiv.org/abs/2510.26692)
    - Yang et al., 2025. Gated Delta Networks: Improving Mamba2 with Delta
      Rule. (https://arxiv.org/abs/2412.06464)
    - Yang et al., 2024. Parallelizing Linear Transformers with the Delta
      Rule over Sequence Length. (the chunkwise WY form)
      (https://arxiv.org/abs/2406.06484)
    - Schlag et al., 2021. Linear Transformers Are Secretly Fast Weight
      Programmers. (https://arxiv.org/abs/2102.11174)
"""

import math
import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.keras_registration import register_dl_technique

from .common import (
    mask_dtype,
    compute_attention_scale,
    validate_head_divisibility,
)

# ---------------------------------------------------------------------

_MODES: Tuple[str, ...] = ("chunk", "recurrent")

# Default epsilon of the per-head output RMSNorm: fla ``norm_eps``
# (``fla/layers/kda.py``) and HF Kimi-Linear ``rms_norm_eps`` (``config.json``).
_O_NORM_EPSILON: float = 1e-5

# ---------------------------------------------------------------------


def _scan_with_outputs(step, state, xs, y_init):
    """A scan over the leading axis of ``xs`` that XLA can differentiate.

    ``keras.ops.while_loop`` with ``maximum_iterations`` set to the exact trip
    count, not ``keras.ops.scan``: on the Keras 3.8 TensorFlow backend
    ``ops.scan`` builds a ``tf.while_loop`` without ``maximum_iterations``, so
    ``fit`` under ``jit_compile=True`` (the default ``'auto'`` on a GPU) fails
    with "XLA compilation requires a fixed tensor list size", at static and
    dynamic lengths alike. ``layers/transformers/gated_linear_attention_block.py``
    and ``layers/dynamic_chunking/dechunk_layer.py`` use the same form.

    :param step: ``step(state, x) -> (state, y)``.
    :param state: Initial state.
    :param xs: Tuple of tensors scanned along their leading axis.
    :param y_init: Zeros with the shape and dtype of one step's ``y``.
    :return: ``(final_state, ys)`` with ``ys`` stacked along axis 0.
    """
    ops = keras.ops
    n = ops.shape(xs[0])[0]
    ys = ops.repeat(ops.expand_dims(y_init, 0), n, axis=0)

    def body(i, s, out):
        s, y = step(s, tuple(ops.take(t, i, axis=0) for t in xs))
        return i + 1, s, ops.scatter_update(out, ops.reshape(i, (1, 1)), y[None])

    _, state, ys = ops.while_loop(
        lambda i, s, out: i < n, body, (0, state, ys), maximum_iterations=n
    )
    return state, ys

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.attention.kimi_delta_attention")
class KimiDeltaAttention(keras.layers.Layer):
    """
    Kimi Delta Attention: gated delta-rule linear attention with per-channel decay.

    Replaces the ``(N, N)`` score matrix with a per-head recurrent state
    ``S`` of shape ``(d_k, d_v)`` that is decayed channel by channel and
    corrected by the delta rule at every token. No probability normalization
    is involved anywhere, so there is no ``probability_type`` hook; the
    write strength ``beta`` and the decay ``alpha`` play that role.

    Architecture:

    .. code-block:: text

                           x  [B, N, dim]
                                │
           ┌──────────┬─────────┼──────────┬────────────┐
           ▼          ▼         ▼          ▼            ▼
        ┌──────┐  ┌──────┐  ┌──────┐  ┌─────────┐  ┌─────────┐
        │q_proj│  │k_proj│  │v_proj│  │ f_down  │  │ b_proj  │
        └──┬───┘  └──┬───┘  └──┬───┘  │ f_up    │  │ Dense(H)│
           ▼         ▼         ▼      └────┬────┘  └────┬────┘
        causal depthwise conv (conv_size)  │            ▼
        then SiLU, on all three            │      beta = sigmoid
           │         │         │           ▼        [B, N, H]
           ▼         ▼         │   log_alpha = -exp(A_log)
        L2 norm   L2 norm      │     * softplus(f + dt_bias)
        * scale     │          │        [B, N, H, d_k]
           │        │          │           │            │
           └────────┴────┬─────┴───────────┴────────────┘
                         ▼
        ┌──────────────────────────────────────────────┐
        │ delta-rule state update, per head, >= fp32:  │
        │   S = Diag(alpha_t) S                        │
        │   u = beta_t (v_t - S^T k_t)    the delta    │
        │   S = S + k_t u^T               the write    │
        │   o_t = S^T q_t                              │
        │ 'chunk': WY solve per chunk, scan chunks     │
        │ 'recurrent': scan tokens                     │
        └──────────────────────┬───────────────────────┘
                               ▼
                    o  [B, N, H, d_v]
                               ▼
               o_norm (per head, rms_norm)
                               ▼
            * sigmoid(g_up(g_down(x)))   (use_output_gate)
                               ▼
                merge heads [B, N, H*d_v]
                               ▼
                   o_proj -> dropout
                               ▼
                     output [B, N, dim]

    Chunkwise form (``mode='chunk'``), inside one chunk of ``C`` tokens with
    ``g = cumsum(log alpha)`` from the chunk start and ``S0`` the incoming state:

    .. code-block:: text

        A[r, i] = sum_c k[r,c] k[i,c] exp(g[r,c] - g[i,c])     i < r
        P[r, i] = sum_c q[r,c] k[i,c] exp(g[r,c] - g[i,c])     i <= r
        (I + diag(beta) A) [U0 | W] = diag(beta) [V | K*exp(g)]
        U   = U0 - W S0
        O   = (Q * exp(g)) S0 + P U
        S_C = exp(g_C) * S0 + (K * exp(g_C - g))^T U

        Every exponent is <= 0, so no term can overflow however strong
        the decay. The price is a [B, H, N/C, C, C, d_k] intermediate:
        memory grows linearly in chunk_size. Lower chunk_size if it
        does not fit; the result does not depend on it.

    **Chunk-mode memory** is ``O(B * H * (N/C) * C * C * d_k)``, i.e.
    ``O(B * H * N * C * d_k)``, for the pairwise-decay tensors (fla avoids
    them with block-wise Triton kernels). Measured on CPU, forward only,
    float32, ``B=1, H=8, d_k=128``: ``C=64`` costs a peak of +1.4 GB at
    ``N=1024`` (one such tensor is 256 MB) and +2.8 GB at ``N=2048``;
    ``C=16`` at ``N=1024`` costs +0.44 GB. The backward pass keeps more.
    Lower ``chunk_size``, or use ``mode='recurrent'`` (3 to 5 times slower
    per token on CPU), when it does not fit.

    The recurrence runs in ``mask_dtype(compute_dtype)`` (at least float32),
    so the layer is safe under ``mixed_float16``.

    ``call()`` is ``(inputs, attention_mask=None, training=None,
    initial_state=None, return_state=False)``.

    **Carried state.** With ``return_state=True`` the call returns
    ``(output, state)`` where ``state = (S, conv_q, conv_k, conv_v)``:

    - ``S``, shape ``(B, num_heads, head_dim, value_head_dim)``, the
      delta-rule state, in ``mask_dtype(compute_dtype)`` (at least float32);
    - ``conv_q`` and ``conv_k``, shape ``(B, conv_size - 1, num_heads *
      head_dim)``, and ``conv_v``, shape ``(B, conv_size - 1, num_heads *
      value_head_dim)``: the last ``conv_size - 1`` projected inputs BEFORE
      the short convolution, in the compute dtype. This is the conv cache of
      fla ``ShortConvolution`` (``fla/layers/kda.py``) and of HF
      ``modeling_kimi.py``.

    With ``conv_size`` 0 or 1 nothing needs carrying and the three windows
    have length 0; the tuple has four entries for every configuration.
    Passing the tuple back as ``initial_state`` continues the sequence: the
    windows are prepended before the convolution and ``S`` seeds the
    recurrence, so a split call and a token-by-token decode equal one full
    call up to float rounding, in both modes. ``initial_state=None`` means
    all zeros, which is what the causal zero padding of a first call sees.
    Anything other than a 4-tuple (or list) raises ``ValueError``.

    **Masks.** ``attention_mask`` is a rank-2 keep mask ``(B, N)``. A masked
    position neither writes to ``S`` nor decays it (``beta = 0``,
    ``alpha = 1``), is zeroed before the short convolution, is never taken
    into a returned window (the window holds the last ``conv_size - 1``
    KEPT inputs), and has a zero output row. Masked positions therefore act
    exactly as deleted tokens for left padding, right padding, and any
    sequence of calls (a right-padded or an all-padding segment followed by
    a continuation included): the kept outputs, ``S`` and the windows equal
    those of the unpadded sequence. The one exception is an interior hole, a
    masked position between two kept positions of the same row, within a
    call or across calls once a kept input sits in the carried window: the
    short convolution's window still spans the hole's zero, so for
    ``conv_size > 1`` the kept outputs after it differ from deleting the
    token (fla unpads such batches instead). With ``conv_size`` 0 or 1 holes
    are exact too. Masks of rank 3 or more are rejected: the layer is causal
    by construction and has no pairwise score to mask.

    **Not implemented** from fla ``KimiDeltaAttention``: ``allow_neg_eigval``
    (``beta`` in ``(0, 2)``), ``safe_gate`` / ``lower_bound`` (a clamp on the
    decay gate), grouped value attention (``num_v_heads > num_heads``),
    ``cu_seqlens`` variable-length packing, and a bias on the short
    convolution. fla's ``use_short_conv=False`` corresponds to
    ``conv_size=0``, where SiLU is still applied.

    **Weight decay.** ``A_log`` and ``dt_bias`` parameterize the decay; fla
    marks both ``_no_weight_decay``. Exclude them in the optimizer config of
    ``dl_techniques.optimization.optimizer.optimizer_builder``, e.g.
    ``exclude_from_weight_decay=["A_log", "dt_bias", ...]`` (patterns matched
    against variable names with ``re.search``).

    **Initialization.** ``A_log`` and ``dt_bias`` are drawn with the global
    NumPy RNG (reseeded by ``keras.utils.set_random_seed``), as fla does in
    torch. The short-convolution kernels use torch's default ``Conv1d`` init
    that fla keeps, ``U(-1/sqrt(conv_size), 1/sqrt(conv_size))``, not
    ``kernel_initializer``. The output RMSNorm epsilon defaults to ``1e-5``
    (fla ``norm_eps``, HF ``rms_norm_eps``).

    :param dim: Input and output dimension. Must be positive.
    :type dim: int
    :param num_heads: Number of heads. Must be positive. When ``head_dim`` is
        None it must divide ``dim``.
    :type num_heads: int
    :param head_dim: Key (and query) width per head, ``d_k``. Defaults to
        ``dim // num_heads``.
    :type head_dim: Optional[int]
    :param value_head_dim: Value width per head, ``d_v``. Defaults to
        ``head_dim``.
    :type value_head_dim: Optional[int]
    :param conv_size: Kernel size of the causal depthwise short convolution on
        q, k and v. ``0`` disables the convolution (SiLU is still applied).
        Defaults to 4.
    :type conv_size: int
    :param gate_low_rank_dim: Bottleneck width of the low-rank decay and
        output-gate projections. Defaults to ``head_dim``.
    :type gate_low_rank_dim: Optional[int]
    :param use_output_gate: Whether to multiply the normalized output by a
        sigmoid gate computed from the input. Defaults to True.
    :type use_output_gate: bool
    :param output_norm_type: Normalization applied per head to the state
        readout, built through ``create_normalization_layer``. ``None``
        disables it. Defaults to ``'rms_norm'``.
    :type output_norm_type: Optional[str]
    :param output_norm_kwargs: Extra arguments for the output normalization.
        ``epsilon`` defaults to ``1e-5`` unless given here.
    :type output_norm_kwargs: Optional[Dict[str, Any]]
    :param chunk_size: Tokens per chunk in ``'chunk'`` mode. Must be positive.
        Changes memory and speed only, not the result. Defaults to 64.
    :type chunk_size: int
    :param mode: ``'chunk'`` (chunkwise WY form, for training and prefill) or
        ``'recurrent'`` (per-token scan, the exact reference and the natural
        form for decoding). Defaults to ``'chunk'``.
    :type mode: str
    :param a_init_range: ``(low, high)`` range of the per-head decay rate
        ``exp(A_log)`` at initialization, sampled uniformly. Defaults to
        ``(1.0, 16.0)``.
    :type a_init_range: Tuple[float, float]
    :param dt_min: Lower bound of the log-uniform ``softplus(dt_bias)``
        initialization. Defaults to 0.001.
    :type dt_min: float
    :param dt_max: Upper bound of the same initialization. Defaults to 0.1.
    :type dt_max: float
    :param dt_init_floor: Floor applied to the sampled ``dt`` before the
        inverse softplus. Defaults to 1e-4.
    :type dt_init_floor: float
    :param dropout_rate: Dropout on the final output. Must be in ``[0, 1]``.
        Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the q, k, v, beta and output projections carry a
        bias. Defaults to False.
    :type use_bias: bool
    :param eps: Stability constant of the q/k L2 normalization. Defaults to
        1e-6.
    :type eps: float
    :param kernel_initializer: Initializer for the projection kernels (not the
        short convolution, see Initialization above). The first projection
        uses the instance, every other site a clone. Defaults to
        ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for projection biases. Defaults to
        ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for biases.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If any size is not positive, ``conv_size`` is
        negative, ``dim`` is not divisible by ``num_heads`` while ``head_dim``
        is None, ``mode`` is unknown, ``a_init_range`` or ``dt_min``/``dt_max``
        is not an increasing positive pair, ``dropout_rate`` is outside
        ``[0, 1]`` or ``eps`` is not positive.
    :raises ValueError: From ``build()``, if the input is not 3D or its last
        dimension is not ``dim``.
    :raises ValueError: From ``call()``, if ``attention_mask`` is not rank 2,
        or ``initial_state`` is not the 4-tuple described above.

    Example:
        >>> import keras
        >>> from dl_techniques.layers.attention.kimi_delta_attention import (
        ...     KimiDeltaAttention)
        >>> layer = KimiDeltaAttention(dim=256, num_heads=4)
        >>> x = keras.random.normal((2, 100, 256))
        >>> y = layer(x)
        >>> tuple(y.shape)
        (2, 100, 256)
        >>> # Prefill, then continue from the carried state: y2 equals y[:, 64:]
        >>> y1, state = layer(x[:, :64], return_state=True)
        >>> len(state)
        4
        >>> y2 = layer(x[:, 64:], initial_state=state)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            value_head_dim: Optional[int] = None,
            conv_size: int = 4,
            gate_low_rank_dim: Optional[int] = None,
            use_output_gate: bool = True,
            output_norm_type: Optional[str] = "rms_norm",
            output_norm_kwargs: Optional[Dict[str, Any]] = None,
            chunk_size: int = 64,
            mode: str = "chunk",
            a_init_range: Tuple[float, float] = (1.0, 16.0),
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            eps: float = 1e-6,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        ``A_log`` and ``dt_bias`` are this layer's own weights and are created
        in :meth:`build`. See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is None:
            validate_head_divisibility(dim, num_heads)
            head_dim = dim // num_heads
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if value_head_dim is None:
            value_head_dim = head_dim
        if value_head_dim <= 0:
            raise ValueError(f"value_head_dim must be positive, got {value_head_dim}")
        if conv_size < 0:
            raise ValueError(f"conv_size must be non-negative, got {conv_size}")
        if gate_low_rank_dim is None:
            gate_low_rank_dim = head_dim
        if gate_low_rank_dim <= 0:
            raise ValueError(
                f"gate_low_rank_dim must be positive, got {gate_low_rank_dim}"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        a_low, a_high = float(a_init_range[0]), float(a_init_range[1])
        if not 0.0 < a_low <= a_high:
            raise ValueError(
                f"a_init_range must satisfy 0 < low <= high, got {a_init_range}"
            )
        if not 0.0 < dt_min <= dt_max:
            raise ValueError(
                f"dt_min and dt_max must satisfy 0 < dt_min <= dt_max, "
                f"got {dt_min}, {dt_max}"
            )
        if dt_init_floor <= 0.0:
            raise ValueError(f"dt_init_floor must be positive, got {dt_init_floor}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.conv_size = conv_size
        self.gate_low_rank_dim = gate_low_rank_dim
        self.use_output_gate = use_output_gate
        self.output_norm_type = output_norm_type
        self.output_norm_kwargs = output_norm_kwargs
        self.chunk_size = chunk_size
        self.mode = mode
        self.a_init_range = (a_low, a_high)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.dt_init_floor = float(dt_init_floor)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = use_bias
        self.eps = float(eps)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_heads * self.value_head_dim
        # Python float, computed once here and never in call().
        self.scale = compute_attention_scale(self.head_dim)

        def dense(units: int, name: str, use_bias: bool, first: bool = False):
            """One projection with its own initializer clone.

            :param units: Output width.
            :type units: int
            :param name: Sub-layer name; checkpoint paths depend on it.
            :type name: str
            :param use_bias: Whether the projection carries a bias.
            :type use_bias: bool
            :param first: Keep the shared initializer instance instead of a
                clone, so the first site's seeded draw does not move.
            :type first: bool
            :return: An unbuilt Dense layer.
            :rtype: keras.layers.Dense
            """
            return keras.layers.Dense(
                units,
                use_bias=use_bias,
                kernel_initializer=(
                    self.kernel_initializer if first
                    else clone_initializer(self.kernel_initializer)
                ),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=name,
            )

        # Only q_proj keeps the shared instance; every other site clones it,
        # so a seedless initializer does not replay one sample everywhere.
        self.q_proj = dense(self.key_dim, "q_proj", self.use_bias, first=True)
        self.k_proj = dense(self.key_dim, "k_proj", self.use_bias)
        self.v_proj = dense(self.value_dim, "v_proj", self.use_bias)

        # Pre-convolution inputs carried between calls: the short
        # convolution's receptive field minus the current token.
        self.conv_window = max(self.conv_size - 1, 0)

        if self.conv_size > 0:
            def conv(channels: int, name: str) -> keras.layers.Conv1D:
                """Depthwise convolution over the sequence axis, ``'valid'`` padding.

                Causality comes from :meth:`call`, which prepends the carried
                ``conv_size - 1`` inputs (zeros on a first call) instead of
                letting the convolution zero-pad.

                :param channels: Number of channels, one group per channel.
                :type channels: int
                :param name: Sub-layer name.
                :type name: str
                :return: An unbuilt Conv1D layer.
                :rtype: keras.layers.Conv1D
                """
                return keras.layers.Conv1D(
                    filters=channels,
                    kernel_size=self.conv_size,
                    padding="valid",
                    groups=channels,
                    use_bias=False,
                    # torch's default Conv1d init, kept by fla ShortConvolution:
                    # U(+-1/sqrt(fan_in)) with fan_in = conv_size for a
                    # depthwise kernel (std 0.29 at conv_size=4). A fresh
                    # instance per site, so the three kernels draw independently.
                    kernel_initializer=keras.initializers.VarianceScaling(
                        scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
                    ),
                    kernel_regularizer=self.kernel_regularizer,
                    name=name,
                )

            self.q_conv = conv(self.key_dim, "q_conv")
            self.k_conv = conv(self.key_dim, "k_conv")
            self.v_conv = conv(self.value_dim, "v_conv")
        else:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None

        # Low-rank channel-wise decay logits.
        self.f_down = dense(self.gate_low_rank_dim, "f_down", False)
        self.f_up = dense(self.key_dim, "f_up", False)

        # Per-head write strength.
        self.b_proj = dense(self.num_heads, "b_proj", self.use_bias)

        if self.use_output_gate:
            self.g_down = dense(self.gate_low_rank_dim, "g_down", False)
            self.g_up = dense(self.value_dim, "g_up", True)
        else:
            self.g_down = None
            self.g_up = None

        if self.output_norm_type is not None:
            norm_kwargs = dict(self.output_norm_kwargs or {})
            norm_kwargs.setdefault("epsilon", _O_NORM_EPSILON)
            self.o_norm = create_normalization_layer(
                self.output_norm_type,
                name="o_norm",
                **norm_kwargs,
            )
        else:
            self.o_norm = None

        self.o_proj = dense(self.dim, "o_proj", self.use_bias)

        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout_layer = None

        self.A_log = None
        self.dt_bias = None

        logger.debug(
            f"KimiDeltaAttention initialized: dim={dim}, num_heads={num_heads}, "
            f"head_dim={self.head_dim}, value_head_dim={self.value_head_dim}, "
            f"mode={mode}, chunk_size={chunk_size}"
        )

    # -----------------------------------------------------------------
    # initial values of the layer's own weights
    # -----------------------------------------------------------------

    # DECISION plan-2026-11-10T082801-e036869a/D-001: draw A_log/dt_bias with
    # the global NumPy RNG and hand them over as Constant initializers, as
    # layers/ssm/mamba2.py does. Do NOT go back to keras.random: an unseeded
    # draw creates the global SeedGenerator lazily, which fails ('NoneType'
    # object has no attribute 'assign') when the layer is first built inside a
    # traced graph (a cold subclassed model in fit). np.random is reseeded by
    # keras.utils.set_random_seed, so a seeded build stays reproducible.

    def _initial_a_log(self) -> np.ndarray:
        """``log(U(a_low, a_high))``, one decay rate per head.

        :return: float32 array of shape ``(num_heads,)``.
        :rtype: np.ndarray
        """
        low, high = self.a_init_range
        a = np.random.uniform(low, high, size=self.num_heads)
        return np.log(a).astype("float32")

    def _initial_dt_bias(self) -> np.ndarray:
        """Inverse softplus of a log-uniform ``dt`` in ``[dt_min, dt_max]``.

        :return: float32 array of shape ``(num_heads * head_dim,)``.
        :rtype: np.ndarray
        """
        u = np.random.uniform(0.0, 1.0, size=self.key_dim)
        dt = np.exp(
            u * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )
        dt = np.maximum(dt, self.dt_init_floor)
        # softplus^{-1}(dt) = dt + log(1 - exp(-dt))
        return (dt + np.log(-np.expm1(-dt))).astype("float32")

    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create ``A_log`` and ``dt_bias`` and build every sub-layer explicitly.

        :param input_shape: ``(batch, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the input is not 3D or its last axis is not ``dim``.
        """
        if self.built:
            return

        input_shape = tuple(input_shape)
        if len(input_shape) != 3:
            raise ValueError(
                f"KimiDeltaAttention expects a 3D input (batch, seq_len, dim), "
                f"got {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match "
                f"dim ({self.dim})"
            )

        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.num_heads,),
            initializer=keras.initializers.Constant(self._initial_a_log()),
            trainable=True,
            dtype="float32",
        )
        self.dt_bias = self.add_weight(
            name="dt_bias",
            shape=(self.key_dim,),
            initializer=keras.initializers.Constant(self._initial_dt_bias()),
            trainable=True,
            dtype="float32",
        )

        batch, seq_len = input_shape[0], input_shape[1]
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.f_down, self.b_proj):
            layer.build(input_shape)
        if self.q_conv is not None:
            # The convolution sees the carried window in front of the sequence.
            conv_len = None if seq_len is None else seq_len + self.conv_window
            self.q_conv.build((batch, conv_len, self.key_dim))
            self.k_conv.build((batch, conv_len, self.key_dim))
            self.v_conv.build((batch, conv_len, self.value_dim))
        self.f_up.build((batch, seq_len, self.gate_low_rank_dim))
        if self.g_down is not None:
            self.g_down.build(input_shape)
            self.g_up.build((batch, seq_len, self.gate_low_rank_dim))
        if self.o_norm is not None:
            self.o_norm.build((batch, seq_len, self.num_heads, self.value_head_dim))
        self.o_proj.build((batch, seq_len, self.value_dim))
        if self.dropout_layer is not None:
            self.dropout_layer.build((batch, seq_len, self.dim))

        super().build(input_shape)

    # -----------------------------------------------------------------

    def _split_initial_state(
            self, initial_state: Any
    ) -> Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """Unpack and check an incoming ``(S, conv_q, conv_k, conv_v)``.

        :param initial_state: The tuple a previous call returned with
            ``return_state=True``.
        :type initial_state: Any
        :raises ValueError: If it is not a 4-element tuple or list, or a
            window's static length is not ``conv_size - 1``.
        :return: ``(S, (conv_q, conv_k, conv_v))``.
        :rtype: Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]
        """
        if not isinstance(initial_state, (tuple, list)) or len(initial_state) != 4:
            raise ValueError(
                "KimiDeltaAttention: initial_state must be the 4-tuple "
                "(S, conv_q, conv_k, conv_v) returned with return_state=True, "
                f"got {type(initial_state).__name__}"
                + (f" of length {len(initial_state)}"
                   if isinstance(initial_state, (tuple, list)) else "")
            )
        state, *windows = initial_state
        for name, window in zip(("conv_q", "conv_k", "conv_v"), windows):
            width = window.shape[1] if len(window.shape) == 3 else -1
            if width is not None and width != self.conv_window:
                raise ValueError(
                    f"KimiDeltaAttention: initial_state {name} must have shape "
                    f"(batch, {self.conv_window}, channels) for conv_size="
                    f"{self.conv_size}, got {tuple(window.shape)}"
                )
        return state, tuple(windows)

    def _last_kept_inputs(
            self,
            extended: keras.KerasTensor,
            keep_extended: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """The last ``conv_size - 1`` kept rows of ``extended``, in order.

        ``extended`` is the carried window followed by this call's
        pre-convolution inputs; the window rows always count as kept, so there
        are at least ``conv_size - 1`` kept rows. Selecting kept rows only is
        what deleting the masked tokens would leave in the window. Called only
        when ``conv_size - 1 > 0``.

        :param extended: ``(batch, conv_size - 1 + seq_len, channels)``.
        :type extended: keras.KerasTensor
        :param keep_extended: ``(batch, conv_size - 1 + seq_len)`` keep mask, or
            None when every row is kept.
        :type keep_extended: Optional[keras.KerasTensor]
        :return: ``(batch, conv_size - 1, channels)``.
        :rtype: keras.KerasTensor
        """
        ops = keras.ops
        w = self.conv_window
        if keep_extended is None:
            return extended[:, -w:]
        kept = ops.cast(keep_extended > 0, "int32")
        # Kept rows at or after each position; the last kept row has 1.
        rank = ops.sum(kept, axis=1, keepdims=True) - ops.cumsum(kept, axis=1) + kept
        slot_rank = w - ops.arange(w, dtype="int32")                   # w, ..., 1
        pick = ops.logical_and(
            ops.equal(rank[:, None, :], slot_rank[None, :, None]),
            ops.greater(kept[:, None, :], 0),
        )                                                              # (B, w, L)
        # A one-hot contraction, exact: one product per output, the rest zeros.
        return ops.einsum("bjl,blc->bjc", ops.cast(pick, extended.dtype), extended)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            initial_state: Optional[Tuple[keras.KerasTensor, ...]] = None,
            return_state: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]:
        """Run the KDA recurrence over the sequence.

        :param inputs: Tensor of shape ``(batch, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional rank-2 keep mask ``(batch, seq_len)``;
            nonzero means the token is real. Masked tokens are skipped by the
            recurrence, kept out of the carried convolution window and get a
            zero output row (see the class docstring for the exact semantics).
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training flag, forwarded to dropout.
        :type training: Optional[bool]
        :param initial_state: Optional incoming state, the 4-tuple
            ``(S, conv_q, conv_k, conv_v)`` a previous call returned with
            ``return_state=True``. ``None`` is all zeros.
        :type initial_state: Optional[Tuple[keras.KerasTensor, ...]]
        :param return_state: If True, also return the final state tuple.
        :type return_state: bool

        :raises ValueError: If ``attention_mask`` is not rank 2, or
            ``initial_state`` is not a 4-tuple with windows of length
            ``conv_size - 1``.

        :return: Output ``(batch, seq_len, dim)``, or
            ``(output, (S, conv_q, conv_k, conv_v))`` when ``return_state`` is
            True. ``S`` is in at least float32, the windows in the compute dtype.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]
        """
        ops = keras.ops
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        H, dk, dv = self.num_heads, self.head_dim, self.value_head_dim

        state0, windows0 = None, (None, None, None)
        if initial_state is not None:
            state0, windows0 = self._split_initial_state(initial_state)

        keep = None
        x = inputs
        if attention_mask is not None:
            if len(attention_mask.shape) != 2:
                raise ValueError(
                    "KimiDeltaAttention: attention_mask must be a rank-2 keep mask "
                    f"(batch, seq_len), got shape {tuple(attention_mask.shape)}. "
                    "The layer is causal by construction and has no pairwise "
                    "score to mask."
                )
            keep = ops.cast(ops.cast(attention_mask, "float32") > 0.0, inputs.dtype)
            x = inputs * keep[..., None]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if keep is not None:
            # Zero before the convolution so a padded token's bias cannot
            # leak into its kept neighbours.
            keep3 = keep[..., None]
            q, k, v = q * keep3, k * keep3, v * keep3

        # DECISION plan-2026-11-10T082801-e036869a/D-001: the short convolution
        # carries its last conv_size-1 PRE-conv inputs between calls, as fla
        # ShortConvolution / HF Kimi-Linear do. The window (zeros on a first
        # call) is prepended and the convolution runs 'valid', which is the
        # causal convolution of the whole stream. Do NOT carry only S: a split
        # call then differed from a full call by 2.5 to 4.2 at conv_size=4. The
        # outgoing window takes the last KEPT rows, never the last rows, so a
        # right-padded segment carries what deleting the pads would carry.
        windows = []
        if self.q_conv is not None:
            w = self.conv_window
            keep_ext = None
            if keep is not None and return_state and w > 0:
                keep_ext = ops.concatenate(
                    [ops.ones((batch_size, w), dtype=keep.dtype), keep], axis=1
                )
            convolved = []
            for t, conv, window0 in zip(
                    (q, k, v), (self.q_conv, self.k_conv, self.v_conv), windows0
            ):
                if window0 is None:
                    window0 = ops.zeros((batch_size, w, t.shape[-1]), dtype=t.dtype)
                extended = ops.concatenate([ops.cast(window0, t.dtype), t], axis=1)
                convolved.append(conv(extended))
                if return_state:
                    windows.append(
                        self._last_kept_inputs(extended, keep_ext) if w > 0
                        else extended[:, :0]
                    )
            q, k, v = convolved
        elif return_state:
            windows = [ops.zeros((batch_size, 0, t.shape[-1]), dtype=t.dtype)
                       for t in (q, k, v)]
        q, k, v = ops.silu(q), ops.silu(k), ops.silu(v)

        f = self.f_up(self.f_down(x))
        beta = ops.sigmoid(self.b_proj(x))

        # The recurrence runs in >= float32 whatever the policy.
        rd = mask_dtype(getattr(q.dtype, "name", None) or str(q.dtype))
        q = ops.reshape(ops.cast(q, rd), (batch_size, seq_len, H, dk))
        k = ops.reshape(ops.cast(k, rd), (batch_size, seq_len, H, dk))
        v = ops.reshape(ops.cast(v, rd), (batch_size, seq_len, H, dv))
        f = ops.reshape(ops.cast(f, rd), (batch_size, seq_len, H, dk))
        beta = ops.cast(beta, rd)

        dt_bias = ops.reshape(ops.cast(self.dt_bias, rd), (H, dk))
        a_rate = ops.exp(ops.cast(self.A_log, rd))[:, None]
        log_alpha = -a_rate * ops.softplus(f + dt_bias)

        q = q * ops.rsqrt(ops.sum(ops.square(q), axis=-1, keepdims=True) + self.eps)
        q = q * self.scale
        k = k * ops.rsqrt(ops.sum(ops.square(k), axis=-1, keepdims=True) + self.eps)

        if keep is not None:
            keep_rd = ops.cast(keep, rd)
            beta = beta * keep_rd[..., None]
            log_alpha = log_alpha * keep_rd[..., None, None]

        # (B, N, H, .) -> (B, H, N, .)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        log_alpha = ops.transpose(log_alpha, (0, 2, 1, 3))
        beta = ops.transpose(beta, (0, 2, 1))

        if state0 is None:
            state = ops.zeros((batch_size, H, dk, dv), dtype=rd)
        else:
            state = ops.cast(state0, rd)

        if self.mode == "chunk":
            o, state = self._chunk_delta_rule(q, k, v, log_alpha, beta, state)
        else:
            o, state = self._recurrent_delta_rule(q, k, v, log_alpha, beta, state)

        # (B, H, N, dv) -> (B, N, H, dv)
        o = ops.cast(ops.transpose(o, (0, 2, 1, 3)), self.compute_dtype)
        if self.o_norm is not None:
            o = self.o_norm(o, training=training)
        if self.g_up is not None:
            gate = ops.sigmoid(self.g_up(self.g_down(x)))
            o = o * ops.reshape(gate, (batch_size, seq_len, H, dv))
        o = ops.reshape(o, (batch_size, seq_len, self.value_dim))

        output = self.o_proj(o)
        if self.dropout_layer is not None:
            output = self.dropout_layer(output, training=training)
        if keep is not None:
            output = output * ops.cast(keep, output.dtype)[..., None]

        if return_state:
            return output, (state, *windows)
        return output

    # -----------------------------------------------------------------

    @staticmethod
    def _recurrent_delta_rule(
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            log_alpha: keras.KerasTensor,
            beta: keras.KerasTensor,
            state: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Exact per-token recurrence, one scan step per token.

        :param q: ``(B, H, N, d_k)``, normalized and scaled.
        :param k: ``(B, H, N, d_k)``, normalized.
        :param v: ``(B, H, N, d_v)``.
        :param log_alpha: ``(B, H, N, d_k)``, non-positive.
        :param beta: ``(B, H, N)``, in ``[0, 1)``.
        :param state: ``(B, H, d_k, d_v)``.
        :return: ``(o, final_state)``, ``o`` of shape ``(B, H, N, d_v)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        ops = keras.ops
        xs = (
            ops.transpose(q, (2, 0, 1, 3)),
            ops.transpose(k, (2, 0, 1, 3)),
            ops.transpose(v, (2, 0, 1, 3)),
            ops.transpose(log_alpha, (2, 0, 1, 3)),
            ops.transpose(beta, (2, 0, 1)),
        )

        def step(s, xt):
            q_t, k_t, v_t, la_t, b_t = xt
            s = ops.exp(la_t)[..., None] * s
            pred = ops.einsum("bhk,bhkv->bhv", k_t, s)
            u = b_t[..., None] * (v_t - pred)
            s = s + k_t[..., :, None] * u[..., None, :]
            o_t = ops.einsum("bhk,bhkv->bhv", q_t, s)
            return s, o_t

        y_init = ops.zeros_like(v[:, :, 0, :])
        state, o = _scan_with_outputs(step, state, xs, y_init)
        return ops.transpose(o, (1, 2, 0, 3)), state

    def _chunk_delta_rule(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            log_alpha: keras.KerasTensor,
            beta: keras.KerasTensor,
            state: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Chunkwise WY form: one triangular solve per chunk, scan over chunks.

        The sequence is right-padded to a multiple of ``chunk_size`` with
        ``beta = 0`` and ``log_alpha = 0``, so padding leaves the state
        unchanged and its outputs are sliced away.

        :param q: ``(B, H, N, d_k)``, normalized and scaled.
        :param k: ``(B, H, N, d_k)``, normalized.
        :param v: ``(B, H, N, d_v)``.
        :param log_alpha: ``(B, H, N, d_k)``, non-positive.
        :param beta: ``(B, H, N)``, in ``[0, 1)``.
        :param state: ``(B, H, d_k, d_v)``.
        :return: ``(o, final_state)``, ``o`` of shape ``(B, H, N, d_v)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        ops = keras.ops
        C = self.chunk_size
        dk, dv = self.head_dim, self.value_head_dim
        batch_size = ops.shape(q)[0]
        H = self.num_heads
        n = ops.shape(q)[2]
        pad = (C - n % C) % C

        def pad_seq(t):
            widths = [[0, 0], [0, 0], [0, pad]] + [[0, 0]] * (len(t.shape) - 3)
            return ops.pad(t, widths)

        q, k, v, log_alpha, beta = map(pad_seq, (q, k, v, log_alpha, beta))
        m = (n + pad) // C

        def chunks(t, d):
            return ops.reshape(t, (batch_size, H, m, C, d))

        q, k, v, log_alpha = (
            chunks(q, dk), chunks(k, dk), chunks(v, dv), chunks(log_alpha, dk)
        )
        beta = ops.reshape(beta, (batch_size, H, m, C))

        g = ops.cumsum(log_alpha, axis=3)                      # (B,H,m,C,dk)
        g_last = g[:, :, :, -1:, :]                           # (B,H,m,1,dk)

        # Pairwise decay exp(g_r - g_i) for i <= r, zero above the diagonal.
        # The `where` inside the exp keeps the masked (positive) exponents
        # from overflowing and from poisoning the gradient.
        # Triangles from an arange comparison: ops.tril raises under tracing
        # on Keras 3.8 ('pred must not be a Python bool').
        idx = ops.arange(C)
        incl = ops.greater_equal(idx[:, None], idx[None, :])  # i <= r
        incl5 = ops.reshape(incl, (1, 1, 1, C, C, 1))
        diff = g[:, :, :, :, None, :] - g[:, :, :, None, :, :]  # (B,H,m,C,C,dk)
        decay = ops.where(incl5, ops.exp(ops.where(incl5, diff, 0.0)), 0.0)

        strict = ops.cast(ops.greater(idx[:, None], idx[None, :]), q.dtype)  # i < r
        a_kk = ops.einsum("bhmrc,bhmic,bhmric->bhmri", k, k, decay) * strict
        p_qk = ops.einsum("bhmrc,bhmic,bhmric->bhmri", q, k, decay)

        # (I + diag(beta) A) [U0 | W] = diag(beta) [V | K * exp(g)]
        eye = ops.eye(C, dtype=q.dtype)
        lower = eye + beta[..., None] * a_kk
        rhs = beta[..., None] * ops.concatenate([v, k * ops.exp(g)], axis=-1)
        solved = ops.solve_triangular(lower, rhs, lower=True)
        u0 = solved[..., :dv]
        w = solved[..., dv:]

        q_dec = q * ops.exp(g)
        k_dec = k * ops.exp(g_last - g)
        state_decay = ops.exp(g_last[:, :, :, 0, :])            # (B,H,m,dk)

        def to_front(t):
            perm = (2, 0, 1) + tuple(range(3, len(t.shape)))
            return ops.transpose(t, perm)

        xs = tuple(map(to_front, (q_dec, w, u0, p_qk, k_dec, state_decay)))

        def step(s, xc):
            qd, wc, u0c, pc, kd, sd = xc
            u = u0c - ops.einsum("bhck,bhkv->bhcv", wc, s)
            o_c = ops.einsum("bhck,bhkv->bhcv", qd, s) + ops.einsum(
                "bhri,bhiv->bhrv", pc, u
            )
            s = sd[..., :, None] * s + ops.einsum("bhck,bhcv->bhkv", kd, u)
            return s, o_c

        y_init = ops.zeros_like(u0[:, :, 0])
        state, o = _scan_with_outputs(step, state, xs, y_init)   # (m,B,H,C,dv)
        o = ops.transpose(o, (1, 2, 0, 3, 4))
        o = ops.reshape(o, (batch_size, H, m * C, dv))
        return o[:, :, :n, :], state

    # -----------------------------------------------------------------

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape, identical to the input shape.

        :param input_shape: ``(batch, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, seq_len, dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def compute_output_spec(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            initial_state: Optional[Tuple[keras.KerasTensor, ...]] = None,
            return_state: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]:
        """Symbolic output, with the state tuple when ``return_state`` is True.

        Overridden because :meth:`compute_output_shape` cannot see
        ``return_state``: a ``keras.Input`` call must return the same
        structure as an eager call.

        :param inputs: Symbolic ``(batch, seq_len, dim)`` input.
        :type inputs: keras.KerasTensor
        :param attention_mask: Unused here; part of the call signature.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Unused here; part of the call signature.
        :type training: Optional[bool]
        :param initial_state: Unused here; part of the call signature.
        :type initial_state: Optional[Tuple[keras.KerasTensor, ...]]
        :param return_state: Whether the state tuple is returned too.
        :type return_state: bool
        :return: The output spec, or ``(output, (S, conv_q, conv_k, conv_v))``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]]
        """
        output = keras.KerasTensor(
            self.compute_output_shape(inputs.shape), dtype=self.compute_dtype
        )
        if not return_state:
            return output
        batch = inputs.shape[0]
        state = keras.KerasTensor(
            (batch, self.num_heads, self.head_dim, self.value_head_dim),
            dtype=mask_dtype(self.compute_dtype),
        )
        windows = tuple(
            keras.KerasTensor((batch, self.conv_window, channels), dtype=self.compute_dtype)
            for channels in (self.key_dim, self.key_dim, self.value_dim)
        )
        return output, (state, *windows)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "value_head_dim": self.value_head_dim,
            "conv_size": self.conv_size,
            "gate_low_rank_dim": self.gate_low_rank_dim,
            "use_output_gate": self.use_output_gate,
            "output_norm_type": self.output_norm_type,
            "output_norm_kwargs": self.output_norm_kwargs,
            "chunk_size": self.chunk_size,
            "mode": self.mode,
            "a_init_range": tuple(self.a_init_range),
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init_floor": self.dt_init_floor,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "eps": self.eps,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------