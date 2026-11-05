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

# ---------------------------------------------------------------------


def _scan_with_outputs(step, state, xs, y_init):
    """``keras.ops.scan`` that tolerates a per-step output unlike the carry.

    The TensorFlow backend of ``keras.ops.scan`` requires every step's output
    to have the carry's structure and shapes. The output is therefore carried
    alongside the state and emitted as the full carry. The stacked copy of
    the state is discarded; backpropagation through the scan keeps one state
    per step anyway.

    :param step: ``step(state, x) -> (state, y)``.
    :param state: Initial state.
    :param xs: Pytree of tensors scanned along their leading axis.
    :param y_init: Zeros with the shape and dtype of one step's ``y``.
    :return: ``(final_state, ys)`` with ``ys`` stacked along axis 0.
    """

    def wrapped(carry, x):
        s, y = step(carry[0], x)
        return (s, y), (s, y)

    (state, _), (_, ys) = keras.ops.scan(wrapped, (state, y_init), xs)
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

    The recurrence runs in ``mask_dtype(compute_dtype)`` (at least float32),
    so the layer is safe under ``mixed_float16``.

    ``call()`` is ``(inputs, attention_mask=None, training=None,
    initial_state=None, return_state=False)``. ``attention_mask`` is a rank-2
    keep mask ``(B, N)``. A masked position neither writes to the state nor
    decays it (``beta = 0``, ``alpha = 1``) and is zeroed before the short
    convolution, so left and right padding both leave the kept outputs and
    the final state exactly as for the unpadded sequence; its own output row
    is zero. Masks of rank 3 or more are rejected: the layer is causal by
    construction and has no pairwise score to mask.

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
    :param kernel_initializer: Initializer for projection and convolution
        kernels. The first projection uses the instance, every other site a
        clone. Defaults to ``'glorot_uniform'``.
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
    :raises ValueError: From ``call()``, if ``attention_mask`` is not rank 2.

    Example:
        >>> layer = KimiDeltaAttention(dim=256, num_heads=4)
        >>> y = layer(keras.random.normal((2, 100, 256)))          # (2, 100, 256)
        >>>
        >>> # Prefill, then continue from the carried state
        >>> y1, s = layer(x[:, :64], return_state=True)
        >>> y2 = layer(x[:, 64:], initial_state=s)
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

        if self.conv_size > 0:
            def conv(channels: int, name: str) -> keras.layers.Conv1D:
                """Causal depthwise convolution over the sequence axis.

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
                    padding="causal",
                    groups=channels,
                    use_bias=False,
                    kernel_initializer=clone_initializer(self.kernel_initializer),
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
            self.o_norm = create_normalization_layer(
                self.output_norm_type,
                name="o_norm",
                **(self.output_norm_kwargs or {}),
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
    # initializers for the layer's own weights
    # -----------------------------------------------------------------

    def _a_log_initializer(self, shape, dtype=None):
        """``log(U(a_low, a_high))``, one decay rate per head."""
        low, high = self.a_init_range
        a = keras.random.uniform(shape, minval=low, maxval=high, dtype="float32")
        return keras.ops.cast(keras.ops.log(a), dtype or "float32")

    def _dt_bias_initializer(self, shape, dtype=None):
        """Inverse softplus of a log-uniform ``dt`` in ``[dt_min, dt_max]``."""
        u = keras.random.uniform(shape, dtype="float32")
        dt = keras.ops.exp(
            u * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )
        dt = keras.ops.maximum(dt, self.dt_init_floor)
        # softplus^{-1}(dt) = dt + log(1 - exp(-dt))
        inv = dt + keras.ops.log(-keras.ops.expm1(-dt))
        return keras.ops.cast(inv, dtype or "float32")

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
            initializer=self._a_log_initializer,
            trainable=True,
            dtype="float32",
        )
        self.dt_bias = self.add_weight(
            name="dt_bias",
            shape=(self.key_dim,),
            initializer=self._dt_bias_initializer,
            trainable=True,
            dtype="float32",
        )

        batch, seq_len = input_shape[0], input_shape[1]
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.f_down, self.b_proj):
            layer.build(input_shape)
        if self.q_conv is not None:
            self.q_conv.build((batch, seq_len, self.key_dim))
            self.k_conv.build((batch, seq_len, self.key_dim))
            self.v_conv.build((batch, seq_len, self.value_dim))
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

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            initial_state: Optional[keras.KerasTensor] = None,
            return_state: bool = False,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Run the KDA recurrence over the sequence.

        :param inputs: Tensor of shape ``(batch, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional rank-2 keep mask ``(batch, seq_len)``;
            nonzero means the token is real. Masked tokens are skipped by the
            recurrence and get a zero output row.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training flag, forwarded to dropout.
        :type training: Optional[bool]
        :param initial_state: Optional incoming state
            ``(batch, num_heads, head_dim, value_head_dim)``. ``None`` is zeros.
            The short convolution does not carry its own history across calls,
            so continuing from a state is exact only with ``conv_size=0``.
        :type initial_state: Optional[keras.KerasTensor]
        :param return_state: If True, also return the final state.
        :type return_state: bool

        :raises ValueError: If ``attention_mask`` is not rank 2.

        :return: Output ``(batch, seq_len, dim)``, or ``(output, final_state)``
            when ``return_state`` is True. The state is in at least float32.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        ops = keras.ops
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        H, dk, dv = self.num_heads, self.head_dim, self.value_head_dim

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
        if self.q_conv is not None:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)
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

        if initial_state is None:
            state = ops.zeros((batch_size, H, dk, dv), dtype=rd)
        else:
            state = ops.cast(initial_state, rd)

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
            return output, state
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
        incl = ops.cast(ops.tril(ops.ones((C, C))), "bool")
        incl5 = ops.reshape(incl, (1, 1, 1, C, C, 1))
        diff = g[:, :, :, :, None, :] - g[:, :, :, None, :, :]  # (B,H,m,C,C,dk)
        decay = ops.where(incl5, ops.exp(ops.where(incl5, diff, 0.0)), 0.0)

        strict = ops.cast(ops.tril(ops.ones((C, C)), k=-1), q.dtype)
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