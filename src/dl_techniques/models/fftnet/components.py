"""
FFTNet (Spectre)
==============================================

This module implements the FFTNet/Spectre architecture, which provides frequency-domain
token mixing as a scalable alternative to self-attention mechanisms. The implementation
follows modern Keras 3 best practices for custom layer development.

**Architecture Overview**:
The Spectre model performs global token mixing in the frequency domain using:
1. Real FFT (RFFT) to transform sequences to frequency domain
2. Data-dependent frequency gates computed from query pooling
3. Complex-valued modulation with modReLU activation
4. Inverse FFT (IRFFT) to return to time domain

This approach provides O(N log N) complexity compared to O(N²) for standard attention.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops, layers, initializers
from typing import Optional, Union, Tuple, List, Dict, Any, Callable, Literal
import math
import numpy as np

# Conditional TensorFlow import for FFT operations
try:
    import tensorflow as tf

    _HAVE_TF = True
except ImportError:
    _HAVE_TF = False
    print("Warning: TensorFlow not found. FFT operations require TensorFlow backend.")

# Import factories from dl_techniques
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn import create_ffn_layer


# ==============================================================================
# Pooling Layers
# ==============================================================================

@keras.saving.register_keras_serializable()
class MeanPoolingLayer(layers.Layer):
    """
    Simple mean pooling over the sequence dimension.

    Reduces a sequence tensor (B, N, D) to a global descriptor (B, D) by
    averaging across the sequence length N.

    **Intent**: Provide basic global pooling for creating sequence descriptors
    used in gate computation.

    **Architecture**:
    ```
    Input(B, N, D) → Mean(axis=1) → Output(B, D)
    ```

    Args:
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, features)`.

    Output shape:
        2D tensor with shape: `(batch_size, features)`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply mean pooling across sequence dimension.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, features).

        Returns:
            Pooled tensor of shape (batch_size, features).
        """
        return ops.mean(inputs, axis=1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], input_shape[2])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return super().get_config()


@keras.saving.register_keras_serializable()
class AttentionPoolingLayer(layers.Layer):
    """
    Two-layer attention pooling for creating global descriptors with learned weights.

    Uses a small MLP to compute attention scores over sequence positions, then
    pools with a weighted sum. This allows the network to focus on important
    positions when creating global descriptors.

    **Intent**: Provide learned attention-based pooling that can adaptively focus
    on relevant sequence positions for gate computation.

    **Architecture**:
    ```
    Input(B, N, D)
         ↓
    Dense(hidden_dim) → GELU
         ↓
    Dense(1) → Softmax(axis=1)
         ↓
    Weighted Sum → Output(B, D)
    ```

    Args:
        hidden_dim: Hidden dimension for the attention scoring network. Default: 256.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, features)`.

    Output shape:
        2D tensor with shape: `(batch_size, features)`.
    """

    def __init__(self, hidden_dim: int = 256, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

        # Create sub-layers in __init__ (Golden Rule)
        self.dense1 = layers.Dense(hidden_dim, activation='gelu', name="attn_pool_dense1")
        self.dense2 = layers.Dense(1, name="attn_pool_dense2")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers explicitly for robust serialization.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Build sub-layers in computational order
        self.dense1.build(input_shape)
        dense1_output_shape = self.dense1.compute_output_shape(input_shape)
        self.dense2.build(dense1_output_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply attention-based pooling.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, features).

        Returns:
            Attention-pooled tensor of shape (batch_size, features).
        """
        scores = self.dense2(self.dense1(inputs))  # (B, N, 1)
        weights = ops.softmax(scores, axis=1)
        pooled = ops.sum(inputs * weights, axis=1)  # (B, D)
        return pooled

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], input_shape[2])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({'hidden_dim': self.hidden_dim})
        return config


@keras.saving.register_keras_serializable()
class DCTPoolingLayer(layers.Layer):
    """
    DCT-based pooling for gate descriptor creation.

    Applies Discrete Cosine Transform along the sequence dimension and averages
    the first K DCT components to create a frequency-domain descriptor. Falls
    back to mean pooling if TensorFlow is not available.

    **Intent**: Provide frequency-domain pooling that captures global patterns
    in the spectral domain, potentially better for periodic or structured sequences.

    **Architecture**:
    ```
    Input(B, N, D)
         ↓
    Transpose → (B, D, N)
         ↓
    DCT(type=2) → (B, D, N)
         ↓
    Mean(first K components) → Output(B, D)
    ```

    Args:
        dct_components: Number of DCT components to average. Default: 64.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, features)`.

    Output shape:
        2D tensor with shape: `(batch_size, features)`.

    Note:
        Requires TensorFlow backend for DCT operations. Falls back to mean
        pooling if TensorFlow is unavailable.
    """

    def __init__(self, dct_components: int = 64, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dct_components = dct_components

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply DCT-based pooling.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, features).

        Returns:
            DCT-pooled tensor of shape (batch_size, features).
        """
        if _HAVE_TF:
            # Transpose for DCT along the sequence dimension
            x_t = ops.transpose(inputs, (0, 2, 1))  # (B, D, N)
            x_dct = tf.signal.dct(x_t, type=2)
            # Take first K components and average
            x_pool = ops.mean(x_dct[..., :self.dct_components], axis=-1)  # (B, D)
        else:
            # Fallback to mean pooling
            x_pool = ops.mean(inputs, axis=1)
        return x_pool

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], input_shape[2])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({'dct_components': self.dct_components})
        return config


# ==============================================================================
# Complex Number Operation Layers
# ==============================================================================

@keras.saving.register_keras_serializable()
class ComplexModReLULayer(layers.Layer):
    """
    Complex modReLU activation with learnable bias.

    Implements the activation function: z → ReLU(|z| + b) * z / |z|, where b is
    a learned per-feature bias. This provides a smooth, learnable gating mechanism
    for complex-valued tensors in the frequency domain.

    **Intent**: Enable learnable, non-linear activation in the complex domain while
    preserving phase information through multiplicative gating based on magnitude.

    **Mathematical Operation**:
    ```
    |z| = sqrt(Re(z)² + Im(z)²)
    scale = ReLU(|z| + b) / |z|
    output = z * scale
    ```

    Args:
        num_features: Number of features (size of the last dimension).
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Complex tensor of any rank with last dimension equal to `num_features`.

    Output shape:
        Same shape as input, complex-valued.

    Attributes:
        bias: Learnable bias parameter of shape (num_features,).
    """

    def __init__(self, num_features: int, **kwargs: Any) -> None:
        super().__init__(**kwargs, dtype="float32")
        self.num_features = num_features

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the learnable bias parameter.

        Args:
            input_shape: Shape tuple of the input.
        """
        self.bias = self.add_weight(
            name='bias',
            shape=(self.num_features,),
            initializer=initializers.Constant(-0.1),  # Start near-identity
            trainable=True,
            dtype="float32"
        )
        self.eps = ops.convert_to_tensor(1e-4, dtype="float32")
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply complex modReLU activation.

        Args:
            inputs: Complex-valued input tensor.

        Returns:
            Complex-valued output tensor with same shape as input.
        """
        mag = ops.abs(inputs)

        # Stable magnitude to avoid division by zero
        mag_stable = ops.sqrt(ops.square(mag) + ops.square(self.eps))

        # Gating scale
        scale = ops.relu(mag + self.bias) / mag_stable

        # Apply scale to complex input
        scale_complex = ops.cast(scale, inputs.dtype)
        return inputs * scale_complex

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({'num_features': self.num_features})
        return config


@keras.saving.register_keras_serializable()
class ComplexInterpolationLayer(layers.Layer):
    """
    Complex tensor interpolation along the last dimension using bicubic interpolation.

    Splits complex tensor into real and imaginary parts, applies bicubic resizing
    (acting as cubic interpolation for 1D), and recombines. Used to upsample
    frequency-domain anchors to full resolution.

    **Intent**: Provide smooth interpolation of complex-valued frequency gates from
    sparse anchor points to full frequency resolution.

    **Architecture**:
    ```
    Input(B, G, K) [complex]
         ↓
    Split → Real(B, G, K), Imag(B, G, K)
         ↓
    Reshape → (B*G, 1, K, 1)
         ↓
    Bicubic Resize → (B*G, 1, size, 1)
         ↓
    Reshape + Combine → Output(B, G, size) [complex]
    ```

    Args:
        size: Target size for interpolation.
        mode: Interpolation mode. Currently only 'cubic' is supported. Default: 'cubic'.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D complex tensor with shape: `(batch_size, groups, input_size)`.

    Output shape:
        3D complex tensor with shape: `(batch_size, groups, size)`.
    """

    def __init__(self, size: int, mode: str = "cubic", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if mode != "cubic":
            raise ValueError("Only 'cubic' interpolation is currently supported.")
        self.size = size
        self.mode = mode

        # Create resizing layer in __init__
        self.resizing_layer = layers.Resizing(
            height=1,
            width=self.size,
            interpolation="bicubic"
        )

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply complex interpolation.

        Args:
            inputs: Complex input tensor of shape (batch_size, groups, input_size).

        Returns:
            Interpolated complex tensor of shape (batch_size, groups, size).
        """
        input_shape = ops.shape(inputs)
        B, G, K = input_shape[0], input_shape[1], input_shape[2]

        # Split into real and imaginary parts
        x_real = ops.real(inputs)
        x_imag = ops.imag(inputs)

        # Reshape for 4D image resizing: (B*G, 1, K, 1)
        x_real_4d = ops.reshape(x_real, (B * G, 1, K, 1))
        x_imag_4d = ops.reshape(x_imag, (B * G, 1, K, 1))

        # Apply bicubic interpolation
        real_up_4d = self.resizing_layer(x_real_4d)
        imag_up_4d = self.resizing_layer(x_imag_4d)

        # Reshape back and recombine
        real_up = ops.reshape(real_up_4d, (B, G, self.size))
        imag_up = ops.reshape(imag_up_4d, (B, G, self.size))

        return ops.cast(ops.complex(real_up, imag_up), dtype=inputs.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], input_shape[1], self.size)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({'size': self.size, 'mode': self.mode})
        return config


@keras.saving.register_keras_serializable()
class ComplexConv1DLayer(layers.Layer):
    """
    One-dimensional complex convolution with circular padding.

    Performs convolution on complex tensors by splitting into real/imaginary parts
    and applying the complex multiplication formula: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i.
    Uses circular padding to maintain sequence length.

    **Intent**: Apply learnable frequency-domain filtering to complex spectral
    representations with circular boundary conditions suitable for periodic signals.

    **Mathematical Operation**:
    For complex input z and kernel k:
    ```
    z = a + bi, k = c + di
    conv(z, k) = conv(a,c) - conv(b,d) + i[conv(a,d) + conv(b,c)]
    ```

    Args:
        kernel_size: Size of the complex convolution kernel.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D complex tensor with shape: `(batch_size, groups, sequence_length)`.

    Output shape:
        Same shape as input.

    Attributes:
        kernel: Complex-valued learnable kernel of shape (kernel_size,).
    """

    def __init__(self, kernel_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the complex kernel weight.

        Args:
            input_shape: Shape tuple of the input.
        """
        self.kernel = self.add_weight(
            name='complex_kernel',
            shape=(self.kernel_size,),
            initializer=initializers.RandomNormal(
                stddev=1.0 / math.sqrt(self.kernel_size)
            ),
            trainable=True,
            dtype="complex64"
        )
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply complex 1D convolution with circular padding.

        Args:
            inputs: Complex input tensor of shape (batch_size, groups, sequence_length).

        Returns:
            Convolved complex tensor with same shape as input.
        """
        input_shape = ops.shape(inputs)
        B, G, K = input_shape[0], input_shape[1], input_shape[2]

        # Reshape for convolution: (B*G, K, 1)
        x_flat = ops.reshape(inputs, (B * G, K, 1))

        # Circular padding
        x_padded = ops.concatenate(
            [x_flat[:, -self.padding:, :], x_flat, x_flat[:, :self.padding, :]],
            axis=1
        )

        # Split into real/imaginary parts
        x_r, x_i = ops.real(x_padded), ops.imag(x_padded)
        k_r, k_i = ops.real(self.kernel), ops.imag(self.kernel)

        # Reshape kernel for conv1d: (kernel_size, in_channels, out_channels)
        k_r_conv = ops.reshape(k_r, (self.kernel_size, 1, 1))
        k_i_conv = ops.reshape(k_i, (self.kernel_size, 1, 1))

        # Complex multiplication via four convolutions
        conv_ac = ops.conv(x_r, k_r_conv, padding='valid')
        conv_bd = ops.conv(x_i, k_i_conv, padding='valid')
        conv_ad = ops.conv(x_r, k_i_conv, padding='valid')
        conv_bc = ops.conv(x_i, k_r_conv, padding='valid')

        real_part = conv_ac - conv_bd
        imag_part = conv_ad + conv_bc

        # Combine and reshape back
        output_flat = ops.complex(real_part, imag_part)
        return ops.reshape(output_flat, (B, G, K))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


# ==============================================================================
# Main FFTNet (Spectre) Layers
# ==============================================================================

@keras.saving.register_keras_serializable()
class SpectreHead(layers.Layer):
    """
    Frequency-domain token mixer for a single attention head.

    This is the core computational block of the FFTNet/Spectre model, providing
    scalable global token mixing through frequency-domain operations with
    data-dependent gating.

    **Intent**: Provide a scalable O(N log N) alternative to O(N²) self-attention
    by performing global token mixing in the frequency domain with learned,
    data-dependent frequency gates.

    **Architecture**:
    ```
    Input(B, N, D)
         ↓
    Linear Projections: Q, V
         ↓
    V → RFFT → V_freq(B, F_half, D)
         ↓
    Q → Pool → Norm → Gate MLP → Anchors(B, G, B)
         ↓
    Anchors → [Optional Toeplitz] → Interpolate → Gate(B, G, F_half)
         ↓
    Gate → ModReLU → Broadcast
         ↓
    V_freq * Gate → IRFFT → Output(B, N, D)
    ```

    Where:
    - B: Batch size
    - N: Sequence length
    - D: Embedding dimension
    - F_half: FFT size // 2 + 1 (positive frequencies)
    - G: Number of groups (D // G = group size)

    Args:
        embed_dim: Embedding dimension.
        fft_size: FFT size for frequency domain operations.
        num_groups: Number of groups for grouped gating. Must divide embed_dim. Default: 4.
        num_buckets: Number of frequency buckets (anchors). If None, uses sqrt(F_half). Default: None.
        d_gate: Hidden dimension for gate MLP. Default: 256.
        use_toeplitz: Whether to use Toeplitz convolution on anchors. Default: False.
        toeplitz_bw: Bandwidth for Toeplitz convolution. Default: 4.
        dropout_p: Dropout probability. Default: 0.0.
        pooling_type: Type of pooling for query descriptor. Options: 'dct', 'attention', 'mean'. Default: 'dct'.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.

    Output shape:
        Tuple of (output, descriptor):
        - output: 3D tensor with shape `(batch_size, sequence_length, embed_dim)`.
        - descriptor: 2D tensor with shape `(batch_size, embed_dim)`.
    """

    def __init__(
            self,
            embed_dim: int,
            fft_size: int,
            num_groups: int = 4,
            num_buckets: Optional[int] = None,
            d_gate: int = 256,
            use_toeplitz: bool = False,
            toeplitz_bw: int = 4,
            dropout_p: float = 0.0,
            pooling_type: Literal["dct", "attention", "mean"] = "dct",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim % num_groups != 0:
            raise ValueError("embed_dim must be divisible by num_groups")

        # Store configuration
        self.embed_dim = embed_dim
        self.fft_size = fft_size
        self.num_groups = num_groups
        self.d_g = embed_dim // num_groups
        self.F_half = fft_size // 2 + 1
        self.B = num_buckets or max(4, int(math.sqrt(self.F_half)))
        self.d_gate = d_gate
        self.use_toeplitz = use_toeplitz
        self.toeplitz_bw = toeplitz_bw
        self.dropout_p = dropout_p
        self.pooling_type = pooling_type

        # Create sub-layers in __init__ (Golden Rule)
        self.W_q = layers.Dense(embed_dim, use_bias=False, name="W_q")
        self.W_v = layers.Dense(embed_dim, use_bias=False, name="W_v")

        # Gate MLP using existing layer (note: keras.Sequential is fine for simple stacks)
        gate_out_dim = self.B * self.num_groups * 2  # Real and Imaginary parts
        self.gate_mlp = keras.Sequential([
            layers.Dense(d_gate, activation='gelu'),
            layers.Dense(gate_out_dim)
        ], name="gate_mlp")

        # Normalization using factory
        self.q_norm = create_normalization_layer('layer_norm', name="q_norm")

        # Complex operations
        self.modrelu = ComplexModReLULayer(self.F_half * self.num_groups, name="modrelu")
        self.dropout = layers.Dropout(dropout_p)

        # Pooling layer
        if pooling_type == "dct":
            self.pooling = DCTPoolingLayer()
        elif pooling_type == "attention":
            self.pooling = AttentionPoolingLayer()
        else:
            self.pooling = MeanPoolingLayer()

        # Optional Toeplitz convolution
        if self.use_toeplitz:
            self.toeplitz_conv = ComplexConv1DLayer(
                kernel_size=2 * self.toeplitz_bw + 1,
                name="toeplitz_conv"
            )
        else:
            self.toeplitz_conv = None

        self.interp_layer = ComplexInterpolationLayer(size=self.F_half, name="interp")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers explicitly for robust serialization.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Build projections
        self.W_q.build(input_shape)
        self.W_v.build(input_shape)

        # Build pooling
        self.pooling.build(input_shape)

        # Pooling output shape is (B, D)
        pool_output_shape = (input_shape[0], input_shape[-1])
        self.q_norm.build(pool_output_shape)
        self.gate_mlp.build(pool_output_shape)

        # Build complex layers with expected shapes
        gate_anchor_shape = (input_shape[0], self.num_groups, self.B)
        if self.toeplitz_conv:
            self.toeplitz_conv.build(gate_anchor_shape)

        self.interp_layer.build(gate_anchor_shape)
        self.modrelu.build((input_shape[0], self.num_groups * self.F_half))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass through the Spectre head.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embed_dim).
            training: Boolean flag for training mode (affects dropout).

        Returns:
            Tuple of (output, descriptor):
            - output: Transformed tensor of shape (batch_size, sequence_length, embed_dim).
            - descriptor: Global descriptor of shape (batch_size, embed_dim).
        """
        input_shape = ops.shape(inputs)
        B, N = input_shape[0], input_shape[1]

        # 1. Projections
        Q = self.W_q(inputs)  # (B, N, D)
        V = self.W_v(inputs)

        # 2. FFT of V
        V_fft = tf.signal.rfft(V, fft_length=[self.fft_size], axis=1)  # (B, F_half, D)

        # 3. Grouped bucket gate
        q_pool = self.q_norm(self.pooling(Q))  # (B, D)

        # Predict anchors
        gate_params = self.gate_mlp(q_pool)  # (B, G * B * 2)
        gate_rs = ops.reshape(gate_params, (B, self.num_groups, self.B, 2))
        gate_anchor = ops.complex(gate_rs[..., 0], gate_rs[..., 1])  # (B, G, B)

        # Optional Toeplitz convolution on anchors
        if self.toeplitz_conv:
            gate_anchor = gate_anchor + self.toeplitz_conv(gate_anchor)

        # Interpolate anchors to full frequency resolution
        gate_half = self.interp_layer(gate_anchor)  # (B, G, F_half)

        # Apply modReLU
        gate_half_flat = ops.reshape(gate_half, (B, self.num_groups * self.F_half))
        gate_half_activated = self.modrelu(gate_half_flat)
        gate_half = ops.reshape(gate_half_activated, (B, self.num_groups, self.F_half))

        # 4. Broadcast gate and mix
        gate_broadcast = ops.transpose(gate_half, (0, 2, 1))  # (B, F_half, G)
        gate_broadcast = ops.repeat(gate_broadcast, self.d_g, axis=-1)  # (B, F_half, D)

        mixed_half = gate_broadcast * V_fft

        # 5. Inverse FFT
        v_time = tf.signal.irfft(mixed_half, fft_length=[self.fft_size], axis=1)

        # Truncate to original sequence length and apply dropout
        result = self.dropout(v_time[:, :N, :], training=training)

        return result, q_pool

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[
        Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shapes for both outputs."""
        output_shape = input_shape
        descriptor_shape = (input_shape[0], input_shape[2])
        return (output_shape, descriptor_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'fft_size': self.fft_size,
            'num_groups': self.num_groups,
            'num_buckets': self.B,
            'd_gate': self.d_gate,
            'use_toeplitz': self.use_toeplitz,
            'toeplitz_bw': self.toeplitz_bw,
            'dropout_p': self.dropout_p,
            'pooling_type': self.pooling_type,
        })
        return config


@keras.saving.register_keras_serializable()
class SpectreMultiHead(layers.Layer):
    """
    Multi-head Spectre layer combining multiple SpectreHead instances.

    Groups several SpectreHead instances, concatenates their outputs, and applies
    an output projection. This is analogous to multi-head attention but operates
    in the frequency domain.

    **Intent**: Enable the model to attend to different frequency patterns and
    representations simultaneously, similar to multi-head attention's ability to
    focus on different aspects of the input.

    **Architecture**:
    ```
    Input(B, N, D)
         ↓
    Split into H heads of dimension D/H
         ↓
    Process each head independently (SpectreHead)
         ↓
    Concatenate head outputs
         ↓
    Output Projection → Output(B, N, D)
    ```

    Args:
        embed_dim: Total embedding dimension (must be divisible by num_heads).
        num_heads: Number of parallel Spectre heads.
        fft_size: FFT size for frequency domain operations.
        **kwargs: Additional keyword arguments passed to SpectreHead.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            fft_size: int,
            **kwargs: Any
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fft_size = fft_size
        self.head_kwargs = kwargs

        # Create sub-layers in __init__ (Golden Rule)
        self.heads = [
            SpectreHead(
                embed_dim=self.head_dim,
                fft_size=fft_size,
                name=f'spectre_head_{i}',
                **self.head_kwargs
            ) for i in range(num_heads)
        ]
        self.out_proj = layers.Dense(embed_dim, use_bias=False, name="out_proj")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers explicitly for robust serialization.

        Args:
            input_shape: Shape tuple of the input.
        """
        head_input_shape = (*input_shape[:-1], self.head_dim)
        for head in self.heads:
            head.build(head_input_shape)

        self.out_proj.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through multi-head Spectre layer.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embed_dim).
            training: Boolean flag for training mode.

        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        # Split input into heads
        chunks = ops.split(inputs, self.num_heads, axis=-1)

        # Process each head
        head_outputs = [head(chunk, training=training)[0] for head, chunk in zip(self.heads, chunks)]

        # Concatenate and project
        concatenated = ops.concatenate(head_outputs, axis=-1)
        return self.out_proj(concatenated)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'fft_size': self.fft_size,
        })
        config.update(self.head_kwargs)
        return config


@keras.saving.register_keras_serializable()
class SpectreBlock(layers.Layer):
    """
    Complete Transformer-style block using SpectreMultiHead for token mixing.

    This layer is a drop-in replacement for a standard Transformer self-attention
    block, providing the same interface but with O(N log N) frequency-domain mixing
    instead of O(N²) attention.

    **Intent**: Provide a complete Transformer block that can replace standard
    attention blocks in any architecture, offering better scaling for long sequences
    while maintaining similar representational capacity.

    **Architecture**:
    ```
    Input(B, N, D)
         ↓
    LayerNorm1
         ↓
    SpectreMultiHead → Residual(+)
         ↓
    LayerNorm2
         ↓
    FFN → Residual(+)
         ↓
    Output(B, N, D)
    ```

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of parallel Spectre heads.
        fft_size: FFT size for frequency domain operations.
        mlp_ratio: Expansion factor for FFN hidden dimension. Default: 4.
        memory_size: Size of optional spectral memory bank. If 0, no memory. Default: 0.
        ffn_type: Type of FFN to use. Options from FFN factory. Default: 'mlp'.
        normalization_type: Type of normalization. Options from norm factory. Default: 'layer_norm'.
        **kwargs: Additional keyword arguments passed to SpectreMultiHead.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.

    Attributes:
        memory_fft: Optional complex-valued spectral memory bank of shape
            (memory_size, embed_dim). Only created if memory_size > 0.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            fft_size: int,
            mlp_ratio: int = 4,
            memory_size: int = 0,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            **kwargs: Any
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.fft_size = fft_size
        self.mlp_ratio = mlp_ratio
        self.memory_size = memory_size
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.mixer_kwargs = kwargs

        # Create sub-layers in __init__ using factories
        self.ln1 = create_normalization_layer(normalization_type, name="ln1")

        self.mix = SpectreMultiHead(
            embed_dim=embed_dim,
            num_heads=num_heads,
            fft_size=fft_size,
            **self.mixer_kwargs
        )

        self.ln2 = create_normalization_layer(normalization_type, name="ln2")

        # Use FFN factory for feed-forward network
        self.mlp = create_ffn_layer(
            ffn_type,
            hidden_dim=mlp_ratio * embed_dim,
            output_dim=embed_dim,
            name="mlp"
        )

        # Memory will be created in build
        self.memory_fft = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers and create optional spectral memory.

        Args:
            input_shape: Shape tuple of the input.
        """
        self.ln1.build(input_shape)
        self.mix.build(input_shape)
        self.ln2.build(input_shape)
        self.mlp.build(input_shape)

        # Create optional spectral memory bank
        if self.memory_size > 0:
            F_half = self.fft_size // 2 + 1
            mem_freq_bins = min(self.memory_size, F_half) if self.memory_size > 1 else F_half

            self.memory_fft = self.add_weight(
                name="memory_fft",
                shape=(mem_freq_bins, self.embed_dim),
                initializer=initializers.RandomNormal(stddev=1.0 / math.sqrt(self.embed_dim)),
                trainable=False,  # Persistent factual memory bank
                dtype="complex64"
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Spectre block.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embed_dim).
            training: Boolean flag for training mode.

        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        # First residual connection: Spectre mixing
        x = inputs + self.mix(self.ln1(inputs), training=training)

        # Second residual connection: FFN
        x = x + self.mlp(self.ln2(x))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'fft_size': self.fft_size,
            'mlp_ratio': self.mlp_ratio,
            'memory_size': self.memory_size,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
        })
        config.update(self.mixer_kwargs)
        return config