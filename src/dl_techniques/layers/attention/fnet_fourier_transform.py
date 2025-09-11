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
    (If masked) Zero out masked positions
           ↓
    Output(shape=[batch, seq_len, hidden_dim])
    ```

    **Key Properties**:
    - **Masking Support**: Propagates masks and zeros out masked timesteps in the output.
    - **Parameter-free**: Zero learnable weights, reducing model complexity.
    - **Efficient**: O(N log N) theoretical complexity vs O(N²) for attention.

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
        self.supports_masking = True

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
        """Create a DFT matrix as a non-trainable variable."""
        norm_factor = 1.0 / np.sqrt(size) if self.normalize_dft else 1.0
        n = np.arange(size, dtype=np.float32)[:, np.newaxis]
        k = np.arange(size, dtype=np.float32)[np.newaxis, :]
        angles = -2.0 * np.pi * n * k / size
        dft_real = np.cos(angles) * norm_factor
        dft_imag = np.sin(angles) * norm_factor
        dft_complex = np.stack([dft_real, dft_imag], axis=-1)
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
        """Efficient complex matrix-vector multiplication using real arithmetic."""
        a, b = matrix[..., 0], matrix[..., 1]
        c, d = vector[..., 0], vector[..., 1]
        real_part = ops.matmul(c, a) - ops.matmul(d, b)
        imag_part = ops.matmul(c, b) + ops.matmul(d, a)
        return ops.stack([real_part, imag_part], axis=-1)

    def _apply_dft_along_axis(
            self,
            inputs_complex: keras.KerasTensor,
            dft_matrix: keras.Variable,
            axis: int
    ) -> keras.KerasTensor:
        """Apply DFT matrix multiplication along specified axis."""
        if axis == -2:
            inputs_transposed = ops.transpose(inputs_complex, [0, 2, 1, 3])
            result = self._complex_matmul(dft_matrix, inputs_transposed)
            return ops.transpose(result, [0, 2, 1, 3])
        elif axis == -1:
            return self._complex_matmul(dft_matrix, inputs_complex)
        else:
            raise ValueError(f"Unsupported axis {axis}. Expected -1 or -2.")

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply 2D Fourier Transform for token mixing.
        """
        # Convert real input to complex representation
        zeros_like_input = ops.zeros_like(inputs)
        inputs_complex = ops.stack([inputs, zeros_like_input], axis=-1)

        # Apply first DFT along sequence dimension
        after_seq_dft = self._apply_dft_along_axis(
            inputs_complex, self.dft_matrix_seq, axis=-2
        )

        # Apply second DFT along hidden dimension
        after_hidden_dft = self._apply_dft_along_axis(
            after_seq_dft, self.dft_matrix_hidden, axis=-1
        )

        # Extract real part as final result
        output = after_hidden_dft[..., 0]

        # Apply mask to zero out padded tokens after mixing
        if attention_mask is not None:
            # Expand mask from [batch, seq_len] to [batch, seq_len, 1]
            mask_expanded = ops.expand_dims(attention_mask, axis=-1)
            # Ensure mask is same dtype and multiply
            output *= ops.cast(mask_expanded, output.dtype)

        return output

    def compute_mask(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> Optional[keras.KerasTensor]:
        """Propagate the input mask."""
        return mask

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
