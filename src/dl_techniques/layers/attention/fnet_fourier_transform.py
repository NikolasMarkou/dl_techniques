"""FNetFourierTransform, the FNet token-mixing sublayer: a parameter-free 2D
Discrete Fourier Transform in place of self-attention.

A 1D DFT runs along the sequence axis, then another along the hidden axis,
and the real part is kept. Every output element is a linear combination of
every input element, so tokens mix globally through a fixed matrix rather
than a data-dependent score. The layer holds no queries, keys, values, or
score matrix, and none of the attention-package conventions (softmax
temperature, head-divisibility check, additive mask bias) apply to it.

The DFT runs as two dense matrix multiplications, not a true
``O(N log N)`` Fast Fourier Transform; passing ``implementation='fft'``
warns and falls back to the same matrix path rather than raising, since a
DFT weight this way serializes exactly. Masking is multiplicative and
applied after mixing, not before a softmax: padded tokens still
contribute to the real tokens' outputs, and only their own rows are
zeroed.

References:
    - Lee-Thorp et al., 2021. FNet: Mixing Tokens with Fourier Transforms. (https://arxiv.org/abs/2105.03824)
"""

import keras
import numpy as np
from typing import Optional, Tuple, Dict, Any, Literal

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.fnet_fourier_transform")
class FNetFourierTransform(keras.layers.Layer):
    """Mix tokens with a 2D Discrete Fourier Transform.

    Applies a 1D DFT along the sequence axis, then another along the hidden
    axis, then takes the real part: ``output = Re(DFT_hidden(DFT_seq(X)))``.
    Every position ends up mixed with every other one, and the layer holds
    no learnable parameters while doing it: the two DFT matrices are created
    with ``trainable=False`` and stored only so they round-trip through
    ``.keras`` serialization instead of being rebuilt on load.

    This is a token mixer, not a QKV attention layer: no queries, keys, or
    values, no learnable projections, no score matrix. The package's
    attention conventions (the ``1 / sqrt(head_dim)`` temperature, the
    ``dim % num_heads`` precondition, the additive ``-1e9`` mask bias) have
    no counterpart here, which is why the module imports nothing from
    :mod:`~dl_techniques.layers.attention.common`.

    Masking is multiplicative and applied after mixing, since there is no
    softmax to bias here. Padded positions still contribute to the mixed
    values of the real tokens; only the padded tokens' own outputs are
    zeroed.

    The DFT costs ``O(S^2 * D + S * D^2)`` per call for an input
    ``(B, S, D)`` — two complex matmuls, not the ``O(N log N)`` of a true
    FFT. ``numpy`` appears only inside ``_create_dft_matrix``, which runs
    once at ``build()`` time to produce a constant initializer; ``call()``
    uses only ``keras.ops``. The complex DFT is carried as a trailing
    real/imag pair rather than routed through ``keras.ops.fft2``, since
    ``fft2`` returns a tuple of tensors that cannot be stored as one
    serializable weight and recomputes the transform on every call instead
    of reading a cached matrix.

    Architecture:

    .. code-block:: text

                     inputs  [B, S, D]  (real)
                              │
                              ▼
                  stack with zeros on a new
                  trailing axis: [B, S, D, 2]
                  (real, imag) - keras.ops has
                  no backend-agnostic complex
                  dtype
                              │
                              ▼
          ┌───────────────────────────────────────┐
          │ DFT along the sequence axis           │
          │   X' = F_S @ X    (complex matmul)    │
          │   F_S: (S, S, 2)  constant weight,    │
          │        trainable=False                │
          └──────────────────┬────────────────────┘
                             │  [B, S, D, 2]
                             ▼
          ┌───────────────────────────────────────┐
          │ DFT along the hidden axis             │
          │   X'' = X' @ F_D  (complex matmul)    │
          │   F_D: (D, D, 2)  constant weight,    │
          │        trainable=False                │
          └──────────────────┬────────────────────┘
                             │  [B, S, D, 2]
                             ▼
                   take the real part
                   X''[..., 0]  ->  [B, S, D]
                             │
                             ▼
          ┌───────────────────────────────────────┐
          │ attention_mask (optional)             │
          │   multiplicative, applied after mix   │
          │   zeroes only the padded tokens' own  │
          │   rows. Padded tokens still fed every │
          │   real token's mix, upstream.         │
          └──────────────────┬────────────────────┘
                             ▼
                     output  [B, S, D]

        No box above owns a trainable weight. The two DFT matrices are
        the layer's only weights, and both are trainable=False.

    :param implementation: Strategy for computing the DFT. Only ``'matrix'``
        (the default) is implemented: it uses cached DFT-matrix
        multiplication. ``'fft'`` is accepted but not implemented. There
        is no true Fast Fourier Transform path here; passing ``'fft'`` logs a
        one-time warning and falls back to the ``'matrix'`` path with
        identical output. Don't reach for ``'fft'`` expecting a speedup.
        Defaults to ``'matrix'``.
    :type implementation: str
    :param normalize_dft: Whether to apply ``1/sqrt(N)`` normalization to the
        DFT matrices, which makes the transform unitary and keeps activation
        energy stable. Defaults to ``True``.
    :type normalize_dft: bool
    :param epsilon: Nominally a small constant for numerical stability. It is
        validated, stored and serialized but never read on the forward path
        -- no division or logarithm in this layer needs a floor. It is kept
        because it is a ``get_config()`` key and dropping it would break
        existing checkpoints. Defaults to ``1e-12``.
    :type epsilon: float
    :param kwargs: Additional arguments for the ``Layer`` base class.

    :ivar implementation: The configured strategy name.
    :vartype implementation: str
    :ivar normalize_dft: Whether the ``1/sqrt(N)`` factor is applied.
    :vartype normalize_dft: bool
    :ivar dft_matrix_seq: Constant ``(S, S, 2)`` DFT matrix, non-trainable.
    :vartype dft_matrix_seq: keras.Variable or None
    :ivar dft_matrix_hidden: Constant ``(D, D, 2)`` DFT matrix, non-trainable.
    :vartype dft_matrix_hidden: keras.Variable or None

    :raises ValueError: If ``implementation`` is not ``'matrix'`` or
        ``'fft'``, or if ``epsilon`` is not positive.
    :raises ValueError: From ``build()``, if the input is not 3D or if the
        sequence length or hidden dimension is unknown at build time.

    Input shape:
        3D tensor with shape ``(batch_size, sequence_length, hidden_dim)``.
        Both trailing dimensions must be statically known: the DFT matrices
        are sized from them.

    Output shape:
        3D tensor with shape ``(batch_size, sequence_length, hidden_dim)`` -
        unchanged from the input.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.attention import FNetFourierTransform

        x = keras.random.normal((4, 128, 64))
        mixer = FNetFourierTransform()
        y = mixer(x)
        assert len(mixer.trainable_weights) == 0
    """

    def __init__(
            self,
            implementation: Literal['matrix', 'fft'] = 'matrix',
            normalize_dft: bool = True,
            epsilon: float = 1e-12,
            **kwargs: Any
    ) -> None:
        """Validate the configuration; the DFT matrices are made in ``build``.

        :param implementation: ``'matrix'`` or ``'fft'``. ``'fft'`` warns and
            falls back to ``'matrix'``.
        :type implementation: str
        :param normalize_dft: Whether to apply the ``1/sqrt(N)`` factor.
        :type normalize_dft: bool
        :param epsilon: Validated and stored, never read on the forward path.
        :type epsilon: float
        :param kwargs: Additional arguments for the ``Layer`` base class.
        :type kwargs: Any

        :raises ValueError: If ``implementation`` is not one of the two
            accepted strings, or if ``epsilon`` is not positive.
        """
        super().__init__(**kwargs)

        valid_implementations = ['matrix', 'fft']
        if implementation not in valid_implementations:
            raise ValueError(
                f"implementation must be one of {valid_implementations}, "
                f"got '{implementation}'"
            )
        # DECISION plan_2026-06-14_0c5d4a21/D-006: 'fft' is accepted but not an
        # FFT path -- build()/call() always use the matrix DFT. Warn once and fall back rather than raising, since a test constructs 'fft' and expects a finite forward. See decisions.md.
        if implementation == 'fft':
            logger.warning(
                "FNetFourierTransform(implementation='fft') is not implemented; "
                "a true FFT path does not exist. Falling back to the 'matrix' "
                "DFT path (identical output). Use implementation='matrix' to "
                "silence this warning."
            )
        # epsilon is validated, stored and serialized but never read: this
        # layer has no division or log that needs a floor. Kept because it is
        # a get_config() key and dropping it breaks load_model on old checkpoints.
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.implementation = implementation
        self.normalize_dft = normalize_dft
        self.epsilon = epsilon
        self.supports_masking = True

        self.dft_matrix_seq = None
        self.dft_matrix_hidden = None

        self._built_seq_len = None
        self._built_hidden_dim = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create and cache the two constant DFT matrices.

        Both are stored as non-trainable weights, so they serialize with the
        model instead of being rebuilt on load. Their sizes come from the
        input shape, which is why the sequence length and hidden dimension
        must be statically known here.

        :param input_shape: Shape tuple of the input tensor. Expected to be
            ``(batch_size, sequence_length, hidden_dim)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 3, or if the
            sequence length or hidden dimension is ``None``.
        """
        if self.built:
            return

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

        # Kept so a later shape mismatch can be reported against what was built.
        self._built_seq_len = seq_len
        self._built_hidden_dim = hidden_dim

        logger.info(
            f"Building FNet DFT matrices: seq_len={seq_len}, hidden_dim={hidden_dim}, "
            f"normalize={self.normalize_dft}"
        )

        self.dft_matrix_seq = self._create_dft_matrix(seq_len, 'dft_matrix_seq')
        self.dft_matrix_hidden = self._create_dft_matrix(hidden_dim, 'dft_matrix_hidden')

        super().build(input_shape)

    def _create_dft_matrix(self, size: int, name: str) -> keras.Variable:
        """
        Create one DFT matrix as a non-trainable variable.

        The trailing axis of size 2 holds the real and imaginary parts.

        :param size: Dimension size for the DFT matrix.
        :type size: int
        :param name: Name for the weight variable.
        :type name: str
        :return: Non-trainable DFT matrix variable of shape
            ``(size, size, 2)``.
        :rtype: keras.Variable
        """
        # `numpy` is used here and only here. This runs once at build() time to
        # produce a constant initializer, never on the forward path, so the
        # package's "keras.ops only in call()" rule holds. numpy keeps the
        # constant a plain host array that `keras.initializers.Constant` can
        # embed directly.
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
        Multiply two complex tensors using real arithmetic only.

        Each operand carries its real and imaginary parts in a trailing axis of
        size 2, because ``keras.ops`` has no backend-agnostic complex dtype.

        :param matrix: Complex matrix stored as ``(..., 2)`` real/imag pair.
        :type matrix: keras.KerasTensor
        :param vector: Complex vector stored as ``(..., 2)`` real/imag pair.
        :type vector: keras.KerasTensor
        :return: Complex result stored as ``(..., 2)`` real/imag pair.
        :rtype: keras.KerasTensor
        """
        a, b = matrix[..., 0], matrix[..., 1]
        c, d = vector[..., 0], vector[..., 1]
        real_part = keras.ops.matmul(c, a) - keras.ops.matmul(d, b)
        imag_part = keras.ops.matmul(c, b) + keras.ops.matmul(d, a)
        return keras.ops.stack([real_part, imag_part], axis=-1)

    def _apply_dft_along_axis(
            self,
            inputs_complex: keras.KerasTensor,
            dft_matrix: keras.Variable,
            axis: int
    ) -> keras.KerasTensor:
        """
        Multiply by a DFT matrix along one axis.

        ``axis=-2`` transposes the sequence axis into last position, contracts,
        and transposes back. ``axis=-1`` contracts directly.

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
            inputs_transposed = keras.ops.transpose(inputs_complex, [0, 2, 1, 3])
            result = self._complex_matmul(dft_matrix, inputs_transposed)
            return keras.ops.transpose(result, [0, 2, 1, 3])
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
        Mix tokens with the 2D DFT.

        Lift to complex, transform along the sequence axis, transform along the
        hidden axis, take the real part, then apply the mask if one was given.

        :param inputs: Input tensor of shape
            ``(batch_size, sequence_length, hidden_dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask of shape
            ``(batch_size, sequence_length)``. It zeroes the padded positions'
            own outputs, after mixing. It does not stop padded tokens from
            contributing to the real tokens' outputs.
        :type attention_mask: keras.KerasTensor or None
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: bool or None
        :return: Mixed output tensor of shape
            ``(batch_size, sequence_length, hidden_dim)``.
        :rtype: keras.KerasTensor
        """
        # Lift the real input to a complex representation. The trailing axis of
        # size 2 is the (real, imaginary) pair. This layer carries complex
        # numbers as an explicit last dimension rather than a complex dtype,
        # because `keras.ops` has no backend-agnostic complex tensor.
        # Shape: (B, S, H) -> (B, S, H)
        zeros_like_input = keras.ops.zeros_like(inputs)
        # Shape: two (B, S, H) -> (B, S, H, 2)
        inputs_complex = keras.ops.stack([inputs, zeros_like_input], axis=-1)

        # First DFT, along the sequence dimension.
        # Shape: (B, S, H, 2) -> (B, S, H, 2), contracting axis -2 with (S, S)
        after_seq_dft = self._apply_dft_along_axis(
            inputs_complex, self.dft_matrix_seq, axis=-2
        )

        # Second DFT, along the hidden dimension.
        # Shape: (B, S, H, 2) -> (B, S, H, 2), contracting axis -1 with (H, H)
        after_hidden_dft = self._apply_dft_along_axis(
            after_seq_dft, self.dft_matrix_hidden, axis=-1
        )

        # Take the real part.
        # Shape: (B, S, H, 2) -> (B, S, H)
        output = after_hidden_dft[..., 0]

        # Multiplicative and applied after mixing, unlike the additive -1e9
        # pre-softmax bias elsewhere: padded tokens still contribute to every
        # real token's output here, and only their own outputs are zeroed.
        if attention_mask is not None:
            # Shape: (B, S) -> (B, S, 1), which broadcasts over H.
            mask_expanded = keras.ops.expand_dims(attention_mask, axis=-1)
            # Cast so a bool or int mask multiplies cleanly.
            output *= keras.ops.cast(mask_expanded, output.dtype)

        return output

    def compute_mask(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> Optional[keras.KerasTensor]:
        """
        Propagate the input mask unchanged.

        The layer preserves the sequence axis, so the incoming mask is still
        valid for the output.

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
        Compute the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input).
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        ``epsilon`` is included even though nothing reads it: it is part of the
        existing config key set.

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
