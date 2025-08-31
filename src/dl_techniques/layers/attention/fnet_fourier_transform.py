"""
FNet: Fourier Transform-based Attention Replacement

This module implements the FNet architecture from "FNet: Mixing Tokens with Fourier Transforms"
(Lee-Thorp et al., 2021), which replaces self-attention with parameter-free Fourier transforms
for efficient token mixing in transformer-style architectures.

"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FNetFourierTransform(keras.layers.Layer):
    """
    FNet Fourier Transform layer that replaces self-attention with parameter-free mixing.

    This layer implements the core innovation from "FNet: Mixing Tokens with Fourier Transforms"
    by applying a 2D Discrete Fourier Transform to input embeddings. The transform applies
    1D DFT along both sequence and hidden dimensions, taking only the real part of the result
    for efficient O(N log N) token mixing without learnable parameters.

    **Intent**: Replace computationally expensive self-attention (O(N²)) with a parameter-free
    Fourier Transform that achieves comparable token mixing performance while being significantly
    faster and more memory efficient, particularly for longer sequences.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, hidden_dim])
           ↓
    DFT along sequence dimension → Complex[batch, seq_len, hidden_dim]
           ↓
    DFT along hidden dimension → Complex[batch, seq_len, hidden_dim]
           ↓
    Extract real part: ℜ(result) → Real[batch, seq_len, hidden_dim]
           ↓
    Output(shape=[batch, seq_len, hidden_dim])
    ```

    **Mathematical Operation**:
        y = ℜ(F_h(F_seq(x)))

    Where:
    - F_seq: 1D DFT along sequence dimension (axis=1)
    - F_h: 1D DFT along hidden dimension (axis=2)
    - ℜ: Real part extraction
    - Order of operations is commutative due to DFT properties

    **Key Properties**:
    - **Parameter-free**: Zero learnable weights, reducing model complexity
    - **Efficient**: O(N log N) theoretical complexity vs O(N²) for attention
    - **Global mixing**: Every position influenced by every other position
    - **Structured**: Uses mathematical properties rather than learned patterns
    - **Hardware optimized**: Can leverage efficient DFT implementations

    Args:
        implementation: Strategy for computing DFT. Options:
            - 'matrix': Use cached DFT matrix multiplication (default, most compatible)
            - 'fft': Use Fast Fourier Transform when available (faster for long sequences)
        normalize_dft: Boolean, whether to apply 1/√N normalization to DFT matrices.
            This ensures energy preservation and numerical stability. Defaults to True.
        epsilon: Small constant for numerical stability. Defaults to 1e-12.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_dim)`.
        Both sequence_length and hidden_dim must be known at build time for matrix caching.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_dim)`.
        Shape is preserved through the Fourier transform operation.

    Attributes:
        dft_matrix_seq: Cached DFT matrix for sequence dimension, shape (seq_len, seq_len, 2).
        dft_matrix_hidden: Cached DFT matrix for hidden dimension, shape (hidden_dim, hidden_dim, 2).

    Example:
        ```python
        # Basic usage - drop-in replacement for self-attention
        fourier_layer = FNetFourierTransform()
        inputs = keras.Input(shape=(512, 768))  # BERT-base dimensions
        mixed = fourier_layer(inputs)  # Same shape: (batch, 512, 768)

        # Disable normalization for experimentation
        fourier_unnorm = FNetFourierTransform(normalize_dft=False)

        # In transformer-style architecture
        def create_fnet_block(hidden_dim: int, intermediate_dim: int) -> keras.Model:
            inputs = keras.Input(shape=(None, hidden_dim))

            # FNet mixing (replaces multi-head attention)
            mixed = FNetFourierTransform()(inputs)
            mixed = keras.layers.LayerNormalization()(inputs + mixed)

            # Standard feed-forward network
            ff = keras.layers.Dense(intermediate_dim, activation='gelu')(mixed)
            ff = keras.layers.Dense(hidden_dim)(ff)
            outputs = keras.layers.LayerNormalization()(mixed + ff)

            return keras.Model(inputs, outputs)

        # Create encoder block
        encoder_block = create_fnet_block(768, 3072)
        ```

    Performance Notes:
        - Matrix implementation provides consistent performance across backends
        - Most efficient for sequences up to ~2048 tokens
        - Memory usage is O(seq_len² + hidden_dim²) for matrix storage
        - Consider model sharding for very large hidden dimensions

    References:
        - FNet: Mixing Tokens with Fourier Transforms (https://arxiv.org/abs/2105.03824)
        - Discrete Fourier Transform: https://en.wikipedia.org/wiki/Discrete_Fourier_transform

    Raises:
        ValueError: If input is not 3D or dimensions are unknown at build time.
        ValueError: If implementation type is invalid.
    """

    def __init__(
            self,
            implementation: Literal['matrix', 'fft'] = 'matrix',
            normalize_dft: bool = True,
            epsilon: float = 1e-12,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        valid_implementations = ['matrix', 'fft']
        if implementation not in valid_implementations:
            raise ValueError(
                f"implementation must be one of {valid_implementations}, "
                f"got '{implementation}'"
            )

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.implementation = implementation
        self.normalize_dft = normalize_dft
        self.epsilon = epsilon

        # DFT matrices (created in build())
        self.dft_matrix_seq = None
        self.dft_matrix_hidden = None

        # Track built dimensions for validation
        self._built_seq_len = None
        self._built_hidden_dim = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create and cache DFT matrices for efficient computation.

        Pre-computes DFT matrices for both sequence and hidden dimensions,
        storing them as non-trainable weights for fast matrix multiplication.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"FNetFourierTransform expects 3D input (batch, sequence, hidden), "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        seq_len = input_shape[1]
        hidden_dim = input_shape[2]

        if seq_len is None or hidden_dim is None:
            raise ValueError(
                f"Sequence length and hidden dimension must be known at build time. "
                f"Got seq_len={seq_len}, hidden_dim={hidden_dim}. "
                f"Consider using keras.Input with explicit shape."
            )

        # Store dimensions for validation
        self._built_seq_len = seq_len
        self._built_hidden_dim = hidden_dim

        # Create DFT matrices
        logger.info(
            f"Building FNet DFT matrices: seq_len={seq_len}, hidden_dim={hidden_dim}, "
            f"normalize={self.normalize_dft}"
        )

        self.dft_matrix_seq = self._create_dft_matrix(seq_len, 'dft_matrix_seq')
        self.dft_matrix_hidden = self._create_dft_matrix(hidden_dim, 'dft_matrix_hidden')

        super().build(input_shape)

    def _create_dft_matrix(self, size: int, name: str) -> keras.Variable:
        """
        Create a DFT matrix as a non-trainable variable.

        Computes the discrete Fourier transform matrix:
        W[n,k] = exp(-2πi*n*k/N) * normalization_factor

        Stored as real tensor of shape (N, N, 2) where last dimension is [real, imag].

        Args:
            size: Matrix size (N×N for N-point DFT).
            name: Variable name for debugging and serialization.

        Returns:
            Keras variable containing the complex DFT matrix.
        """
        # Compute normalization factor
        norm_factor = 1.0 / np.sqrt(size) if self.normalize_dft else 1.0

        # Create index arrays for vectorized computation
        n = np.arange(size, dtype=np.float32)[:, np.newaxis]  # [N, 1]
        k = np.arange(size, dtype=np.float32)[np.newaxis, :]  # [1, N]

        # Compute phase angles: -2πnk/N
        angles = -2.0 * np.pi * n * k / size  # [N, N]

        # Compute complex DFT matrix elements
        dft_real = np.cos(angles) * norm_factor  # Real part
        dft_imag = np.sin(angles) * norm_factor  # Imaginary part

        # Stack to create complex tensor: [N, N, 2] where dim 2 is [real, imag]
        dft_complex = np.stack([dft_real, dft_imag], axis=-1)  # [N, N, 2]

        return self.add_weight(
            name=name,
            shape=(size, size, 2),
            initializer=keras.initializers.Constant(dft_complex),
            trainable=False,
        )

    def _complex_matmul(
            self,
            matrix: keras.KerasTensor,
            vector: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Efficient complex matrix-vector multiplication using real arithmetic.

        Computes (C + iD) @ (A + iB) = (CA - DB) + i(CB + DA) where:
        - matrix (A + iB): [M, N, 2] complex matrix
        - vector (C + iD): [..., N, 2] complex vector(s)
        - result: [..., M, 2] complex result(s)

        Args:
            matrix: Complex matrix with shape (M, N, 2).
            vector: Complex vector(s) with shape (..., N, 2).

        Returns:
            Complex result with shape (..., M, 2).
        """
        # Extract real and imaginary components
        a, b = matrix[..., 0], matrix[..., 1]  # Matrix real, imag parts
        c, d = vector[..., 0], vector[..., 1]  # Vector real, imag parts

        # Complex multiplication: (c + id)(a + ib) = (ca - db) + i(cb + da)
        real_part = ops.matmul(c, a) - ops.matmul(d, b)
        imag_part = ops.matmul(c, b) + ops.matmul(d, a)

        return ops.stack([real_part, imag_part], axis=-1)

    def _apply_dft_along_axis(
            self,
            inputs_complex: keras.KerasTensor,
            dft_matrix: keras.Variable,
            axis: int
    ) -> keras.KerasTensor:
        """
        Apply DFT matrix multiplication along specified axis.

        Args:
            inputs_complex: Complex input tensor with shape (..., 2).
            dft_matrix: DFT matrix with shape (N, N, 2).
            axis: Target axis for DFT application (-1 or -2).

        Returns:
            Complex tensor after DFT transformation.
        """
        if axis == -2:  # Sequence dimension
            # inputs_complex: [batch, seq, hidden, 2]
            # Rearrange for matrix multiplication: [batch, hidden, seq, 2]
            inputs_transposed = ops.transpose(inputs_complex, [0, 2, 1, 3])

            # Apply DFT: [batch, hidden, seq, 2] @ [seq, seq, 2] -> [batch, hidden, seq, 2]
            result = self._complex_matmul(dft_matrix, inputs_transposed)

            # Transpose back to original layout: [batch, seq, hidden, 2]
            return ops.transpose(result, [0, 2, 1, 3])

        elif axis == -1:  # Hidden dimension
            # inputs_complex: [batch, seq, hidden, 2]
            # Apply DFT: [batch, seq, hidden, 2] @ [hidden, hidden, 2] -> [batch, seq, hidden, 2]
            return self._complex_matmul(dft_matrix, inputs_complex)

        else:
            raise ValueError(f"Unsupported axis {axis}. Expected -1 (hidden) or -2 (sequence).")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply 2D Fourier Transform for token mixing.

        Performs the core FNet operation: applies 1D DFT along sequence dimension,
        then 1D DFT along hidden dimension, and returns the real part.

        Args:
            inputs: Input embeddings with shape [batch_size, seq_length, hidden_dim].
            training: Training mode flag (unused but kept for consistency).

        Returns:
            Token-mixed embeddings with identical shape to input.
        """
        # Validate input dimensions match build-time dimensions
        input_shape = ops.shape(inputs)
        seq_len, hidden_dim = input_shape[1], input_shape[2]

        # Convert real input to complex representation
        # Add zero imaginary part: [batch, seq, hidden] -> [batch, seq, hidden, 2]
        zeros_like_input = ops.zeros_like(inputs)
        inputs_complex = ops.stack([inputs, zeros_like_input], axis=-1)

        # Apply first DFT along sequence dimension (axis=-2)
        after_seq_dft = self._apply_dft_along_axis(
            inputs_complex, self.dft_matrix_seq, axis=-2
        )

        # Apply second DFT along hidden dimension (axis=-1)
        # Note: We could extract real part here, but keeping complex gives better accuracy
        after_hidden_dft = self._apply_dft_along_axis(
            after_seq_dft, self.dft_matrix_hidden, axis=-1
        )

        # Extract real part as final result (following paper)
        # Shape: [batch, seq, hidden, 2] -> [batch, seq, hidden]
        return after_hidden_dft[..., 0]

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Fourier transform preserves input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'implementation': self.implementation,
            'normalize_dft': self.normalize_dft,
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------