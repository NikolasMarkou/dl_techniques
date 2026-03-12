"""
Wave Field Attention - Keras 3 Implementation
==============================================

A physics-inspired attention mechanism that replaces standard dot-product
attention with damped wave field convolution in a continuous spatial medium.

Instead of computing pairwise token interactions via :math:`QK^T`, tokens
deposit information onto a discretized 1-D field at absolute positions,
the field is evolved via per-head damped-wave kernels using FFT convolution,
and the result is gathered back to token positions through bilinear
interpolation.

Architecture:
    1. **Absolute Position Mapping**: Token *i* always maps to field position
       ``i * stride``, regardless of current sequence length. This ensures
       positional consistency during autoregressive generation.

    2. **Bilinear Scatter / Gather**: Values are deposited onto and read from
       the continuous field via bilinear interpolation, implemented as sparse
       matrix--vector products for backend-agnostic differentiability.

    3. **Damped Wave Kernels**: Each head maintains a learnable causal kernel

       .. math::

           k(t) = \\exp(-\\alpha t) \\cos(\\omega t + \\varphi), \\quad t \\geq 0

       parameterised by damping :math:`\\alpha`, frequency :math:`\\omega`,
       and phase :math:`\\varphi`.  Convolution is performed in
       :math:`O(G \\log G)` via zero-padded real FFT.

    4. **Static Multi-Field Coupling**: A learnable :math:`H \\times H` matrix
       (row-softmax normalised) mixes information across heads at the field
       level, enabling cross-head field interactions.

    5. **Content-Dependent Gating**: A sigmoid gate modulates the gathered
       field output, allowing the network to selectively suppress or amplify
       information per position and head.

Complexity:
    :math:`O(N \\cdot D + G \\log G \\cdot H \\cdot D_h)` where *N* is
    sequence length, *G* is field size, *H* is head count, and *D_h* is
    head dimension.

References:
    - Ported from the PyTorch ``WaveFieldAttention`` v3.5 implementation.
    - Damped harmonic oscillator: classical wave mechanics.
    - FFT convolution: Cooley--Tukey (1965).
"""

import math
import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WaveFieldAttention(keras.layers.Layer):
    """
    Multi-head wave-field attention via FFT-based damped-wave convolution.

    Tokens are projected to queries, keys, and values.  Key magnitudes
    modulate value deposits onto a 1-D field at absolute positions.
    Per-head damped-wave kernels are applied via FFT convolution, heads
    are coupled through a learnable mixing matrix, and the result is
    gathered back and gated by a content-dependent sigmoid.

    **Intent**: Provide a drop-in replacement for standard multi-head
    attention that captures long-range interactions through physics-inspired
    wave propagation rather than pairwise dot products.

    **Data Flow**::

        Input [B, N, D]
              |
              v
        QKV Projection --> Q, K, V  [B, H, N, D_h]
              |
              v
        deposit = V * ||K||  (key-magnitude modulated values)
              |
              v
        Bilinear Scatter onto field [B, H, G, D_h]
              |
              v
        FFT Wave Convolution (per-head causal kernels)
              |
              v
        Static Multi-Field Coupling (head mixing)
              |
              v
        Bilinear Gather back to token positions [B, H, N, D_h]
              |
              v
        Content-Dependent Sigmoid Gating
              |
              v
        Output Projection --> [B, N, D]

    Args:
        dim: Integer, model dimension. Must be divisible by ``num_heads``.
        num_heads: Integer, number of attention heads. Each head operates
            its own wave field with independent kernel parameters.
            Defaults to 8.
        field_size: Integer, discretisation resolution of each head's
            1-D field. Larger values allow finer spatial resolution at the
            cost of memory and FFT compute. Defaults to 512.
        max_seq_len: Integer, maximum expected sequence length. Used to
            compute the fixed stride that maps token indices to field
            positions. Defaults to 128.
        dropout_rate: Float in ``[0, 1]``, dropout applied to the gated
            output. Defaults to 0.0.
        use_bias: Boolean, whether projection layers use bias terms.
            Defaults to True.
        gate_bias_init: Float, initial bias for the sigmoid gate. A positive
            value (e.g. 2.0) starts the gate near-open. Defaults to 2.0.
        coupling_noise_stddev: Float, standard deviation of Gaussian noise
            added to the identity initialisation of the coupling matrix.
            Defaults to 0.01.
        kernel_initializer: Initializer for projection kernels.
            Defaults to ``"glorot_uniform"``.
        bias_initializer: Initializer for projection biases.
            Defaults to ``"zeros"``.
        kernel_regularizer: Optional regularizer for projection kernels.
        bias_regularizer: Optional regularizer for projection biases.
        **kwargs: Additional keyword arguments for the Layer base class.

    Call arguments:
        inputs: Input tensor of shape ``(batch, seq_len, dim)``.
        training: Boolean indicating training or inference mode.

    Returns:
        Output tensor of shape ``(batch, seq_len, dim)``.

    Raises:
        ValueError: If ``dim`` is not divisible by ``num_heads`` or
            parameters are out of range.

    Example::

        >>> layer = WaveFieldAttention(dim=256, num_heads=8, field_size=512)
        >>> x = keras.random.normal((2, 64, 256))
        >>> out = layer(x)
        >>> print(out.shape)  # (2, 64, 256)
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
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- validation ---
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if field_size <= 1:
            raise ValueError(f"field_size must be > 1, got {field_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

        # --- store config ---
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

        # --- derived constants ---
        self.scale = math.sqrt(self.head_dim)
        self._field_stride = float(
            (field_size - 1) / max(max_seq_len - 1, 1)
        )

        # --- projection sublayers ---
        dense_kwargs: Dict[str, Any] = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        self.qkv_proj = keras.layers.Dense(
            3 * dim, name="qkv_proj", **dense_kwargs
        )
        self.output_proj = keras.layers.Dense(
            dim, name="output_proj", **dense_kwargs
        )

        # Gate projection: zero-init weights, positive bias to start open
        self.gate_proj = keras.layers.Dense(
            dim,
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(self.gate_bias_init),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="gate_proj",
        )

        # --- dropout ---
        self.dropout_layer: Optional[keras.layers.Dropout] = (
            keras.layers.Dropout(self.dropout_rate, name="dropout")
            if self.dropout_rate > 0.0
            else None
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build projection sublayers and learnable wave / coupling parameters.

        Args:
            input_shape: Shape tuple ``(batch, seq_len, dim)``.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3-D input (batch, seq_len, dim), got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension ({input_shape[-1]}) must match dim ({self.dim})"
            )

        H = self.num_heads

        # --- build projection sublayers ---
        self.qkv_proj.build(input_shape)
        self.gate_proj.build(input_shape)

        proj_input_shape = (input_shape[0], input_shape[1], self.dim)
        self.output_proj.build(proj_input_shape)

        if self.dropout_layer is not None:
            self.dropout_layer.build(proj_input_shape)

        # --- wave kernel parameters (per head) ---
        freq_values = [
            0.3 + (4.0 - 0.3) * i / max(H - 1, 1) for i in range(H)
        ]
        self.wave_frequency = self.add_weight(
            name="wave_frequency",
            shape=(H,),
            initializer=keras.initializers.Constant(freq_values),
            trainable=True,
        )

        damp_values = [
            -3.0 + (0.5 - (-3.0)) * i / max(H - 1, 1) for i in range(H)
        ]
        self.wave_damping = self.add_weight(
            name="wave_damping",
            shape=(H,),
            initializer=keras.initializers.Constant(damp_values),
            trainable=True,
        )

        phase_values = [
            math.pi * i / max(H - 1, 1) for i in range(H)
        ]
        self.wave_phase = self.add_weight(
            name="wave_phase",
            shape=(H,),
            initializer=keras.initializers.Constant(phase_values),
            trainable=True,
        )

        # --- field coupling matrix: identity + small noise ---
        eye_values = [
            [1.0 if i == j else 0.0 for j in range(H)]
            for i in range(H)
        ]
        self.field_coupling = self.add_weight(
            name="field_coupling",
            shape=(H, H),
            initializer=keras.initializers.Constant(eye_values),
            trainable=True,
        )

        super().build(input_shape)

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------

    def _compute_field_positions(
        self, seq_len: int
    ) -> keras.KerasTensor:
        """
        Map token indices to absolute field positions.

        Token *i* is placed at ``i * stride``, clamped to ``[0, G-2]``
        so that bilinear interpolation always has a valid upper neighbour.

        Args:
            seq_len: Current sequence length (dynamic).

        Returns:
            Float tensor of shape ``(seq_len,)`` with field positions.
        """
        seq_idx = ops.cast(ops.arange(seq_len), "float32")
        field_pos = seq_idx * self._field_stride
        return ops.clip(field_pos, 0.0, float(self.field_size - 2))

    def _build_scatter_gather_matrices(
        self, field_pos: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Build bilinear interpolation matrices for scatter and gather.

        For each token position ``p``, the scatter matrix distributes weight
        ``(1 - frac)`` to ``floor(p)`` and ``frac`` to ``ceil(p)`` in the
        field grid.  The gather matrix is its transpose.

        Args:
            field_pos: Float tensor of shape ``(N,)`` with field positions.

        Returns:
            Tuple of:
                - scatter_mat: Float tensor ``(G, N)`` mapping tokens to grid.
                - gather_mat: Float tensor ``(N, G)`` mapping grid to tokens.
        """
        G = self.field_size

        idx_lo = ops.cast(ops.floor(field_pos), "int32")
        idx_lo = ops.clip(idx_lo, 0, G - 2)
        idx_hi = idx_lo + 1

        frac = field_pos - ops.cast(idx_lo, "float32")
        frac = ops.clip(frac, 0.0, 1.0)
        w_lo = 1.0 - frac  # (N,)
        w_hi = frac  # (N,)

        # one-hot: (N, G)
        lo_one_hot = ops.one_hot(idx_lo, G, dtype="float32")
        hi_one_hot = ops.one_hot(idx_hi, G, dtype="float32")

        # weighted one-hot transposed -> (G, N)
        scatter_mat = (
            ops.transpose(lo_one_hot) * ops.expand_dims(w_lo, 0)
            + ops.transpose(hi_one_hot) * ops.expand_dims(w_hi, 0)
        )

        gather_mat = ops.transpose(scatter_mat)  # (N, G)
        return scatter_mat, gather_mat

    def _build_wave_kernels_fft(self) -> keras.KerasTensor:
        """
        Build left-aligned causal damped-wave kernels in the frequency domain.

        Each head's kernel is defined as:

        .. math::

            k_h(t) = \\exp(-\\alpha_h t) \\cos(\\omega_h t + \\varphi_h)

        where :math:`\\alpha_h = \\mathrm{softplus}(\\text{wave\\_damping}_h)`.
        Kernels are L1-normalised, zero-padded to ``2G``, and transformed
        via real FFT for efficient convolution.

        Returns:
            Tuple of two real tensors ``(real, imag)`` each of shape
            ``(H, G+1)`` representing the kernels in the frequency domain.
            This follows the ``keras.ops.rfft`` convention.
        """
        G = self.field_size

        t = ops.cast(ops.arange(G), "float32")  # (G,)

        # softplus for strictly positive damping
        alpha = ops.log(1.0 + ops.exp(self.wave_damping))  # (H,)
        omega = self.wave_frequency  # (H,)
        phi = self.wave_phase  # (H,)

        # expand for broadcasting: (H, 1) vs (1, G)
        alpha = ops.expand_dims(alpha, 1)  # (H, 1)
        omega = ops.expand_dims(omega, 1)  # (H, 1)
        phi = ops.expand_dims(phi, 1)  # (H, 1)
        t = ops.expand_dims(t, 0)  # (1, G)

        # damped cosine kernel: (H, G)
        kernels = ops.exp(-alpha * t) * ops.cos(omega * t + phi)

        # L1 normalisation per head
        kernel_norm = ops.sum(ops.abs(kernels), axis=1, keepdims=True)
        kernel_norm = ops.maximum(kernel_norm, 1e-8)
        kernels = kernels / kernel_norm

        # zero-pad to 2G for linear (non-circular) convolution
        kernels_padded = ops.pad(kernels, [[0, 0], [0, G]])  # (H, 2G)

        # real FFT along last axis -> tuple of (H, G+1) real tensors
        return keras.ops.rfft(kernels_padded)

    def _wave_convolve(
        self,
        field: keras.KerasTensor,
        kernel_fft: Tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> keras.KerasTensor:
        """
        Per-head wave convolution via zero-padded FFT.

        Performs linear (non-circular) convolution of each head's field
        slice with the corresponding damped-wave kernel. All FFTs operate
        on the last tensor axis.

        ``keras.ops.rfft`` returns a tuple ``(real, imag)`` rather than
        a complex tensor, so complex multiplication is performed
        component-wise: ``(a+bi)(c+di) = (ac-bd) + (ad+bc)i``.

        Args:
            field: Real tensor of shape ``(B, H, G, D_h)``.
            kernel_fft: Tuple ``(real, imag)`` each of shape ``(H, G+1)``.

        Returns:
            Convolved field of shape ``(B, H, G, D_h)``.
        """
        G = self.field_size

        # move D_h before G so FFT acts on the spatial axis
        # (B, H, G, D_h) -> (B, H, D_h, G)
        field_t = ops.transpose(field, (0, 1, 3, 2))

        # zero-pad spatial axis to 2G
        field_padded = ops.pad(
            field_t, [[0, 0], [0, 0], [0, 0], [0, G]]
        )  # (B, H, D_h, 2G)

        # forward real FFT -> tuple of (B, H, D_h, G+1) real tensors
        field_re, field_im = keras.ops.rfft(field_padded)

        # unpack kernel FFT: each (H, G+1)
        kern_re, kern_im = kernel_fft

        # broadcast kernel: (H, G+1) -> (1, H, 1, G+1)
        kern_re = ops.reshape(kern_re, (1, self.num_heads, 1, -1))
        kern_im = ops.reshape(kern_im, (1, self.num_heads, 1, -1))

        # complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        conv_re = field_re * kern_re - field_im * kern_im
        conv_im = field_re * kern_im + field_im * kern_re

        # inverse real FFT -> (B, H, D_h, 2G)
        convolved = keras.ops.irfft((conv_re, conv_im))

        # take first G elements (valid linear convolution region)
        convolved = convolved[..., :G]  # (B, H, D_h, G)

        # transpose back: (B, H, G, D_h)
        return ops.transpose(convolved, (0, 1, 3, 2))

    def _apply_field_coupling(
        self, field: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply static multi-field coupling across heads.

        A row-softmax-normalised ``(H, H)`` matrix mixes the flattened
        ``(G * D_h)`` representation of each head.

        Args:
            field: Tensor of shape ``(B, H, G, D_h)``.

        Returns:
            Coupled field of shape ``(B, H, G, D_h)``.
        """
        batch_size = ops.shape(field)[0]
        H = self.num_heads
        G = self.field_size
        D_h = self.head_dim

        coupling = ops.softmax(self.field_coupling, axis=-1)  # (H, H)

        # flatten spatial + feature dims per head
        field_flat = ops.reshape(field, (batch_size, H, G * D_h))  # (B, H, G*D_h)

        # head mixing via einsum: output[b,h,f] = sum_k coupling[h,k] * field_flat[b,k,f]
        coupled = ops.einsum("hk,bkf->bhf", coupling, field_flat)

        return ops.reshape(coupled, (batch_size, H, G, D_h))

    # -----------------------------------------------------------------
    # call
    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass through wave-field attention.

        Shape Legend:
            - B: batch size
            - N: sequence length
            - D: model dimension (``self.dim``)
            - H: number of heads (``self.num_heads``)
            - D_h: head dimension (``self.head_dim``)
            - G: field grid size (``self.field_size``)

        Args:
            inputs: Input tensor of shape ``(B, N, D)``.
            training: Boolean flag for training vs inference.

        Returns:
            Output tensor of shape ``(B, N, D)``.
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        H = self.num_heads
        D_h = self.head_dim

        # --- 1. QKV projection ---
        # (B, N, D) -> (B, N, 3*D)
        qkv = self.qkv_proj(inputs)
        # reshape -> (B, N, 3, H, D_h) -> (3, B, H, N, D_h)
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, H, D_h))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v: (B, H, N, D_h)

        # --- 2. Compute absolute field positions ---
        field_pos = self._compute_field_positions(seq_len)  # (N,)
        scatter_mat, gather_mat = self._build_scatter_gather_matrices(
            field_pos
        )  # (G, N), (N, G)

        # --- 3. Bilinear scatter: deposit onto field ---
        # deposit = V * ||K|| (key magnitude modulates value deposit)
        k_mag = ops.sqrt(
            ops.sum(ops.square(k), axis=-1, keepdims=True) + 1e-8
        )  # (B, H, N, 1)
        deposit = v * k_mag  # (B, H, N, D_h)

        # field = scatter_mat @ deposit along the N axis
        # scatter_mat: (G, N), deposit: (B, H, N, D_h) -> field: (B, H, G, D_h)
        field = ops.einsum("gn,bhnd->bhgd", scatter_mat, deposit)

        # --- 4. FFT wave convolution ---
        kernel_fft = self._build_wave_kernels_fft()
        field = self._wave_convolve(field, kernel_fft)

        # --- 5. Static multi-field coupling ---
        field = self._apply_field_coupling(field)

        # --- 6. Bilinear gather: read from field ---
        # gather_mat: (N, G), field: (B, H, G, D_h) -> gathered: (B, H, N, D_h)
        gathered = ops.einsum("ng,bhgd->bhnd", gather_mat, field)

        # --- 7. Content-dependent gating ---
        gate = ops.sigmoid(self.gate_proj(inputs))  # (B, N, D)
        gate = ops.reshape(gate, (batch_size, seq_len, H, D_h))
        gate = ops.transpose(gate, (0, 2, 1, 3))  # (B, H, N, D_h)

        output = gathered * gate  # (B, H, N, D_h)

        # --- 8. Merge heads and project ---
        output = ops.transpose(output, (0, 2, 1, 3))  # (B, N, H, D_h)
        output = ops.reshape(output, (batch_size, seq_len, self.dim))
        output = self.output_proj(output)  # (B, N, D)

        # --- 9. Optional dropout ---
        if self.dropout_layer is not None:
            output = self.dropout_layer(output, training=training)

        return output

    # -----------------------------------------------------------------
    # shape inference
    # -----------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape -- identical to input shape.

        Args:
            input_shape: Shape tuple ``(batch, seq_len, dim)``.

        Returns:
            Same shape tuple ``(batch, seq_len, dim)``.
        """
        return input_shape

    # -----------------------------------------------------------------
    # serialization
    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return full configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "field_size": self.field_size,
                "max_seq_len": self.max_seq_len,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "gate_bias_init": self.gate_bias_init,
                "coupling_noise_stddev": self.coupling_noise_stddev,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config