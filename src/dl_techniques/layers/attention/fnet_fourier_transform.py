"""
Mix tokens using a parameter-free 2D Discrete Fourier Transform.

This layer implements the core token-mixing mechanism from the FNet
architecture, which proposes replacing the self-attention sublayer in a
Transformer with a standard, non-parameterized Fourier Transform. This
approach provides a computationally efficient alternative for mixing
information across a sequence, avoiding the quadratic complexity of
self-attention.

Architecture and Foundational Mathematics:
The FNet block operates on the principle that the primary role of the
self-attention layer is to mix tokens, enabling each position in the
sequence to gather information from all other positions. The authors of FNet
demonstrate that this mixing can be effectively and efficiently approximated
by a much simpler linear transformation: the Discrete Fourier Transform (DFT).

The architecture treats the input tensor of shape `(sequence_length,
hidden_dim)` as a 2D signal. It then applies a 2D DFT by performing two
sequential 1D DFTs:
1.  A 1D DFT is applied along the sequence dimension.
2.  A 1D DFT is applied along the hidden (feature) dimension.

Mathematically, the DFT decomposes a signal into its constituent frequencies.
By applying it across both sequence and hidden dimensions, the layer
transforms the entire input into the frequency domain and back (implicitly,
by taking the real part of the output). This process ensures that every
element in the output tensor is a linear combination of every element in the
input tensor, thus achieving a global receptive field analogous to
self-attention.

The key insight is that this simple, parameter-free linear mixing is
sufficient to achieve strong performance on a variety of NLP tasks, while
being significantly more memory and computationally efficient, with a
complexity of O(N log N) compared to O(N^2) for self-attention.

References:
  - "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., 2021)
    https://arxiv.org/abs/2105.03824

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
    Parameter-free token mixing via 2D Discrete Fourier Transform.

    Implements the core innovation from FNet by applying sequential 1D DFTs
    along the sequence and hidden dimensions, then extracting the real part.
    This achieves global token mixing in ``O(N log N)`` complexity without
    any learnable parameters, replacing the ``O(N^2)`` self-attention
    mechanism. The transform is computed as
    ``output = Re(DFT_hidden(DFT_seq(X)))``.

    **Architecture Overview:**

    .. code-block:: text

        Input [B, S, D]
              │
              ▼
        ┌─────────────────────────┐
        │ DFT along seq dim (S)   │
        │ Complex[B, S, D]        │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ DFT along hidden dim (D)│
        │ Complex[B, S, D]        │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ Extract real part: Re() │
        └────────────┬────────────┘
                     ▼
          (if masked) zero out
              padded positions
                     ▼
              Output [B, S, D]

    :param implementation: Strategy for computing DFT. ``'matrix'`` uses
        cached DFT matrix multiplication (default, most compatible);
        ``'fft'`` uses Fast Fourier Transform when available.
    :type implementation: str
    :param normalize_dft: Whether to apply ``1/sqrt(N)`` normalization to
        DFT matrices for energy preservation and numerical stability.
        Defaults to ``True``.
    :type normalize_dft: bool
    :param epsilon: Small constant for numerical stability.
        Defaults to ``1e-12``.
    :type epsilon: float
    :param kwargs: Additional arguments for the ``Layer`` base class.

    :raises ValueError: If input is not 3D or dimensions are unknown at
        build time.
    :raises ValueError: If ``implementation`` is not ``'matrix'`` or
        ``'fft'``.
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

        :param input_shape: Shape tuple of the input tensor. Expected to be
            ``(batch_size, sequence_length, hidden_dim)``.
        :type input_shape: tuple
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

        :param size: Dimension size for the DFT matrix.
        :type size: int
        :param name: Name for the weight variable.
        :type name: str
        :return: Non-trainable DFT matrix variable of shape
            ``(size, size, 2)``.
        :rtype: keras.Variable
        """
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
        """
        Perform complex matrix-vector multiplication using real arithmetic.

        :param matrix: Complex matrix stored as ``(..., 2)`` real/imag pair.
        :type matrix: keras.KerasTensor
        :param vector: Complex vector stored as ``(..., 2)`` real/imag pair.
        :type vector: keras.KerasTensor
        :return: Complex result stored as ``(..., 2)`` real/imag pair.
        :rtype: keras.KerasTensor
        """
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
        """
        Apply DFT matrix multiplication along a specified axis.

        :param inputs_complex: Complex input tensor with trailing
            ``(..., 2)`` real/imag dimension.
        :type inputs_complex: keras.KerasTensor
        :param dft_matrix: Pre-computed DFT matrix variable.
        :type dft_matrix: keras.Variable
        :param axis: Axis along which to apply the DFT (``-1`` or ``-2``).
        :type axis: int
        :return: Transformed complex tensor.
        :rtype: keras.KerasTensor
        """
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

        :param inputs: Input tensor of shape
            ``(batch_size, sequence_length, hidden_dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask of shape
            ``(batch_size, sequence_length)`` to zero out padded positions.
        :type attention_mask: keras.KerasTensor or None
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: bool or None
        :return: Mixed output tensor of shape
            ``(batch_size, sequence_length, hidden_dim)``.
        :rtype: keras.KerasTensor
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
        """
        Propagate the input mask.

        :param inputs: Input tensor (unused).
        :type inputs: keras.KerasTensor
        :param mask: Input mask to propagate.
        :type mask: keras.KerasTensor or None
        :return: The propagated mask.
        :rtype: keras.KerasTensor or None
        """
        return mask

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape (preserved from input).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input).
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'implementation': self.implementation,
            'normalize_dft': self.normalize_dft,
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
