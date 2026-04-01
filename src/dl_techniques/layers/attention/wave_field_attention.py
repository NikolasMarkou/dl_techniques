"""
Wave Field Attention - Keras 3 Implementation (V3.6)
=====================================================

Physics-inspired attention mechanism: damped wave field convolution
replaces standard dot-product attention.

Changes from V3.5 Keras port:
    - **Q is now used**: query-dependent gather modulation via
      ``sigmoid(Q / sqrt(d_h))`` element-wise on gathered field output.
      Q acts as a learned per-token feature selector on propagated
      information, distinct from the input-based gate.
    - **Padding mask support**: ``attention_mask`` argument ``(B, N)``
      zeros out deposits from padded tokens and masks final output. Causal masking
      is inherent to the left-aligned kernel (output at field position g
      depends only on positions <= g).
    - **coupling_noise_stddev fixed**: now actually used in
      ``field_coupling`` initialisation (identity + Gaussian noise),
      matching the PyTorch original.

Architecture:
    1. QKV projection
    2. Absolute position mapping (token i -> field position i * stride)
    3. Bilinear scatter: ``deposit = V * ||K||``, masked by padding
    4. FFT damped-wave convolution (per-head causal kernels)
    5. Static multi-field coupling (head mixing)
    6. Bilinear gather from field
    7. Query modulation: ``gathered *= sigmoid(Q / scale)``
    8. Content-dependent gating: ``output *= sigmoid(gate_proj(x))``
    9. Output projection + optional dropout

Causality:
    The left-aligned kernel ``k(t) = exp(-alpha*t) * cos(omega*t + phi)``
    for ``t >= 0`` ensures convolution output at field position g depends
    only on field positions <= g. Since token indices map monotonically
    to field positions (token i -> i * stride), this guarantees causal
    information flow without explicit masking. The coupling matrix mixes
    across heads at the *same* spatial position, preserving causality.

Complexity:
    O(N*D + G*log(G)*H*D_h)
"""

import math
import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union


@keras.saving.register_keras_serializable()
class WaveFieldAttention(keras.layers.Layer):
    """
    Multi-head wave-field attention via FFT-based damped-wave convolution.

    Replaces the standard dot-product attention with a physics-inspired
    mechanism: tokens deposit information onto a 1-D field grid weighted by
    key magnitude, a per-head damped-wave kernel is convolved via FFT, a
    learnable coupling matrix mixes across heads at each field position,
    and each token gathers from the convolved field. A query-dependent
    sigmoid modulation and an input-based content gate further refine the
    output before a final linear projection.

    The damped-wave kernel ``k_h(t) = exp(-alpha*t) * cos(omega*t + phi)``
    for ``t >= 0`` is causal (left-aligned), so the convolution output at
    field position ``g`` depends only on positions ``<= g``, providing
    autoregressive information flow without explicit masking. Complexity is
    ``O(N*D + G*log(G)*H*D_h)`` where ``G`` is the field grid size.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input (B, N, D)                │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  QKV Projection ─► Q, K, V      │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Deposit: V * ||K|| onto field  │
        │  (bilinear scatter)             │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  FFT damped-wave convolution     │
        │  (per-head causal kernels)       │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Multi-field coupling (head mix) │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Bilinear gather from field      │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Query modulation: sigmoid(Q/s)  │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Content gate: sigmoid(gate(x))  │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Output projection + dropout     │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Output (B, N, D)               │
        └──────────────────────────────────┘

    :param dim: Model dimension (must be divisible by ``num_heads``).
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param field_size: 1-D field grid resolution.
    :type field_size: int
    :param max_seq_len: Maximum sequence length (determines stride).
    :type max_seq_len: int
    :param dropout_rate: Dropout rate on output.
    :type dropout_rate: float
    :param use_bias: Whether projections use bias.
    :type use_bias: bool
    :param gate_bias_init: Initial gate bias (positive = starts open).
    :type gate_bias_init: float
    :param coupling_noise_stddev: Std-dev of Gaussian noise added to the
        identity-initialized head coupling matrix.
    :type coupling_noise_stddev: float
    :param kernel_initializer: Initializer for projection kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for projection biases.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for projection kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer for projection biases.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        field_size: int = 512,
        max_seq_len: int = 128,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        gate_bias_init: float = 2.0,
        coupling_noise_stddev: float = 0.01,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if field_size <= 1:
            raise ValueError(f"field_size must be > 1, got {field_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.field_size = field_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.gate_bias_init = gate_bias_init
        self.coupling_noise_stddev = coupling_noise_stddev
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.scale = math.sqrt(self.head_dim)
        self._field_stride = float((field_size - 1) / max(max_seq_len - 1, 1))

        dense_kw: Dict[str, Any] = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        self.qkv_proj = keras.layers.Dense(3 * dim, name="qkv_proj", **dense_kw)
        self.output_proj = keras.layers.Dense(dim, name="output_proj", **dense_kw)

        self.gate_proj = keras.layers.Dense(
            dim,
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(self.gate_bias_init),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="gate_proj",
        )

        self.dropout_layer: Optional[keras.layers.Dropout] = (
            keras.layers.Dropout(self.dropout_rate, name="dropout")
            if self.dropout_rate > 0.0
            else None
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3-D input (batch, seq_len, dim), got {input_shape}")
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(f"Last dim ({input_shape[-1]}) must match dim ({self.dim})")

        H = self.num_heads

        self.qkv_proj.build(input_shape)
        self.gate_proj.build(input_shape)
        proj_shape = (input_shape[0], input_shape[1], self.dim)
        self.output_proj.build(proj_shape)
        if self.dropout_layer is not None:
            self.dropout_layer.build(proj_shape)

        # --- wave kernel parameters (per head) ---
        self.wave_frequency = self.add_weight(
            name="wave_frequency",
            shape=(H,),
            initializer=keras.initializers.Constant(
                [0.3 + (4.0 - 0.3) * i / max(H - 1, 1) for i in range(H)]
            ),
            trainable=True,
        )
        self.wave_damping = self.add_weight(
            name="wave_damping",
            shape=(H,),
            initializer=keras.initializers.Constant(
                [-3.0 + (0.5 - (-3.0)) * i / max(H - 1, 1) for i in range(H)]
            ),
            trainable=True,
        )
        self.wave_phase = self.add_weight(
            name="wave_phase",
            shape=(H,),
            initializer=keras.initializers.Constant(
                [math.pi * i / max(H - 1, 1) for i in range(H)]
            ),
            trainable=True,
        )

        # --- field coupling: identity + noise (matching PyTorch original) ---
        import numpy as np

        coupling_init = (
            np.eye(H, dtype="float32")
            + np.random.randn(H, H).astype("float32") * self.coupling_noise_stddev
        )
        self.field_coupling = self.add_weight(
            name="field_coupling",
            shape=(H, H),
            initializer=keras.initializers.Constant(coupling_init),
            trainable=True,
        )

        super().build(input_shape)

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    def _compute_field_positions(self, seq_len: int) -> keras.KerasTensor:
        """Map token index to field position via ``i * stride``, clamped to ``[0, G-2]``.

        :param seq_len: Current sequence length.
        :type seq_len: int
        :return: Float tensor of field positions with shape ``(seq_len,)``.
        :rtype: keras.KerasTensor
        """
        seq_idx = ops.cast(ops.arange(seq_len), "float32")
        return ops.clip(seq_idx * self._field_stride, 0.0, float(self.field_size - 2))

    def _build_scatter_gather_matrices(
        self, field_pos: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Build bilinear interpolation matrices for scatter ``(G, N)`` and gather ``(N, G)``.

        :param field_pos: Float field positions of shape ``(N,)``.
        :type field_pos: keras.KerasTensor
        :return: Tuple of ``(scatter_mat, gather_mat)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        G = self.field_size

        idx_lo = ops.cast(ops.floor(field_pos), "int32")
        idx_lo = ops.clip(idx_lo, 0, G - 2)
        idx_hi = idx_lo + 1

        frac = ops.clip(field_pos - ops.cast(idx_lo, "float32"), 0.0, 1.0)
        w_lo = 1.0 - frac
        w_hi = frac

        lo_oh = ops.one_hot(idx_lo, G, dtype="float32")
        hi_oh = ops.one_hot(idx_hi, G, dtype="float32")

        scatter_mat = (
            ops.transpose(lo_oh) * ops.expand_dims(w_lo, 0)
            + ops.transpose(hi_oh) * ops.expand_dims(w_hi, 0)
        )  # (G, N)

        gather_mat = ops.transpose(scatter_mat)  # (N, G)
        return scatter_mat, gather_mat

    def _build_wave_kernels_fft(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Build left-aligned causal damped-wave kernels in frequency domain.

        Each head has a kernel
        ``k_h(t) = exp(-softplus(damping_h) * t) * cos(freq_h * t + phase_h)``.
        Kernels are L1-normalised, zero-padded to ``2G``, and rfft'd.

        :return: ``(real, imag)`` each of shape ``(H, G+1)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        G = self.field_size
        t = ops.cast(ops.arange(G), "float32")

        alpha = ops.log(1.0 + ops.exp(self.wave_damping))
        omega = self.wave_frequency
        phi = self.wave_phase

        alpha = ops.expand_dims(alpha, 1)
        omega = ops.expand_dims(omega, 1)
        phi = ops.expand_dims(phi, 1)
        t = ops.expand_dims(t, 0)

        kernels = ops.exp(-alpha * t) * ops.cos(omega * t + phi)

        norm = ops.maximum(ops.sum(ops.abs(kernels), axis=1, keepdims=True), 1e-8)
        kernels = kernels / norm

        kernels_padded = ops.pad(kernels, [[0, 0], [0, G]])
        return keras.ops.rfft(kernels_padded)

    def _wave_convolve(
        self,
        field: keras.KerasTensor,
        kernel_fft: Tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Per-head FFT convolution on the field grid.

        :param field: Field tensor of shape ``(B, H, G, D_h)``.
        :type field: keras.KerasTensor
        :param kernel_fft: Tuple of ``(real, imag)`` kernel spectra.
        :type kernel_fft: Tuple[keras.KerasTensor, keras.KerasTensor]
        :return: Convolved field of shape ``(B, H, G, D_h)``.
        :rtype: keras.KerasTensor
        """
        G = self.field_size

        field_t = ops.transpose(field, (0, 1, 3, 2))  # (B, H, D_h, G)
        field_padded = ops.pad(field_t, [[0, 0], [0, 0], [0, 0], [0, G]])

        field_re, field_im = keras.ops.rfft(field_padded)

        kern_re, kern_im = kernel_fft
        kern_re = ops.reshape(kern_re, (1, self.num_heads, 1, -1))
        kern_im = ops.reshape(kern_im, (1, self.num_heads, 1, -1))

        conv_re = field_re * kern_re - field_im * kern_im
        conv_im = field_re * kern_im + field_im * kern_re

        convolved = keras.ops.irfft((conv_re, conv_im))
        convolved = convolved[..., :G]

        return ops.transpose(convolved, (0, 1, 3, 2))

    def _apply_field_coupling(self, field: keras.KerasTensor) -> keras.KerasTensor:
        """Apply row-softmax coupling across heads at each spatial position.

        :param field: Field tensor of shape ``(B, H, G, D_h)``.
        :type field: keras.KerasTensor
        :return: Coupled field of shape ``(B, H, G, D_h)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(field)[0]
        H = self.num_heads
        G = self.field_size
        D_h = self.head_dim

        coupling = ops.softmax(self.field_coupling, axis=-1)
        field_flat = ops.reshape(field, (batch_size, H, G * D_h))
        coupled = ops.einsum("hk,bkf->bhf", coupling, field_flat)
        return ops.reshape(coupled, (batch_size, H, G, D_h))

    # -----------------------------------------------------------------
    # call
    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass through the wave-field attention mechanism.

        :param inputs: Input tensor of shape ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional float mask ``(B, N)``. ``1.0`` = valid,
            ``0.0`` = padding.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        H = self.num_heads
        D_h = self.head_dim

        # 1. QKV projection -> (B, H, N, D_h) each
        qkv = self.qkv_proj(inputs)
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, H, D_h))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Field positions + scatter/gather matrices
        field_pos = self._compute_field_positions(seq_len)
        scatter_mat, gather_mat = self._build_scatter_gather_matrices(field_pos)

        # 3. Deposit = V * ||K||
        k_mag = ops.sqrt(ops.sum(ops.square(k), axis=-1, keepdims=True) + 1e-8)
        deposit = v * k_mag  # (B, H, N, D_h)

        # Apply padding mask to deposits
        if attention_mask is not None:
            # attention_mask: (B, N) -> (B, 1, N, 1)
            mask_4d = ops.expand_dims(ops.expand_dims(attention_mask, 1), -1)
            deposit = deposit * mask_4d

        # Scatter onto field
        field = ops.einsum("gn,bhnd->bhgd", scatter_mat, deposit)

        # 4. FFT wave convolution
        kernel_fft = self._build_wave_kernels_fft()
        field = self._wave_convolve(field, kernel_fft)

        # 5. Multi-field coupling
        field = self._apply_field_coupling(field)

        # 6. Gather from field
        gathered = ops.einsum("ng,bhgd->bhnd", gather_mat, field)

        # 7. Query-dependent gather modulation (NEW in V3.6)
        #    Q selects which dimensions of propagated info each token reads.
        #    Distinct from the gate (which is input-based, not projection-based).
        q_mod = ops.sigmoid(q / self.scale)  # (B, H, N, D_h)
        gathered = gathered * q_mod

        # 8. Content-dependent gating (input-based)
        gate = ops.sigmoid(self.gate_proj(inputs))  # (B, N, D)
        gate = ops.reshape(gate, (batch_size, seq_len, H, D_h))
        gate = ops.transpose(gate, (0, 2, 1, 3))  # (B, H, N, D_h)
        output = gathered * gate

        # 9. Merge heads + project
        output = ops.transpose(output, (0, 2, 1, 3))  # (B, N, H, D_h)
        output = ops.reshape(output, (batch_size, seq_len, self.dim))
        output = self.output_proj(output)

        # Apply padding mask to output
        if attention_mask is not None:
            output = output * ops.expand_dims(attention_mask, -1)

        # 10. Dropout
        if self.dropout_layer is not None:
            output = self.dropout_layer(output, training=training)

        return output

    # -----------------------------------------------------------------
    # shape inference + serialization
    # -----------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "field_size": self.field_size,
            "max_seq_len": self.max_seq_len,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "gate_bias_init": self.gate_bias_init,
            "coupling_noise_stddev": self.coupling_noise_stddev,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config