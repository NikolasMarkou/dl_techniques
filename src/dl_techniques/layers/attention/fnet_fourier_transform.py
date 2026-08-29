"""
Mix tokens with a parameter-free 2D Discrete Fourier Transform.

This is the token-mixing sublayer from FNet. It replaces self-attention with a
fixed, unlearned linear transform. There is nothing to train here: the layer's
only weights are two constant DFT matrices, and they are created
``trainable=False``.

Architecture:
    Self-attention's main job is to let every position read every other
    position. FNet's claim is that a 2D DFT does enough of that job to be worth
    the saving.

    The input of shape ``(sequence_length, hidden_dim)`` is treated as a 2D
    signal, and two 1D DFTs are applied in turn:

    1.  A 1D DFT along the sequence dimension.
    2.  A 1D DFT along the hidden (feature) dimension.

    **This is NOT a QKV attention layer.** It lives in ``attention/`` because
    it drops into a self-attention sublayer's slot. It has no queries, keys or
    values, no learnable projections and no score matrix. The package's
    attention-specific rules therefore do not apply here - the
    ``1 / sqrt(head_dim)`` softmax temperature, the ``dim % num_heads``
    precondition and the additive ``-1e9`` mask bias have no counterpart. That
    is why nothing is imported from ``common.py``.

    Masking differs in kind for the same reason. There is no softmax to bias,
    so ``attention_mask`` MULTIPLIES padded positions to zero AFTER mixing.
    The consequence: padded tokens still contribute to the mixed values of the
    real tokens. Only their own outputs are zeroed.

Foundational Mathematics:
    The DFT decomposes a signal into frequencies. Applying it across both the
    sequence and hidden axes moves the whole input into the frequency domain,
    and taking the real part brings it back::

        output = Re( F_S @ X @ F_D ),   (F_N)_{nk} = exp(-2*pi*i*n*k/N) / sqrt(N)

    Each DFT matrix is dense, so every output element is a linear combination
    of every input element. That is a global receptive field from a fixed
    mixing matrix rather than a data-dependent one. The ``1/sqrt(N)`` factor
    (``normalize_dft=True``) makes the transform unitary, so it preserves
    signal energy and does not amplify activations as the sequence grows.

    Note on complexity. This layer computes the DFT by multiplying with cached
    DFT matrices, which costs ``O(S^2 * D + S * D^2)`` per mixing step - two
    complex matmuls. That is NOT the ``O(N log N)`` of a true Fast Fourier
    Transform, and the FNet paper's ``O(N log N)`` figure refers to an FFT
    implementation. The matrix path trades the asymptotic for hardware
    simplicity and for DFT weights that serialize exactly. Passing
    ``implementation='fft'`` does not get you an FFT; it warns and falls back.
    The ``__init__`` DECISION anchor says why, and why it must not be turned
    into a ``NotImplementedError``.

References:
  - "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., 2021)
    https://arxiv.org/abs/2105.03824

"""

# ---------------------------------------------------------------------

import keras
import numpy as np
from typing import Optional, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.attention.fnet_fourier_transform")
class FNetFourierTransform(keras.layers.Layer):
    """
    Parameter-free token mixing by a 2D Discrete Fourier Transform.

    Applies a 1D DFT along the sequence axis, then another along the hidden
    axis, then takes the real part: ``output = Re(DFT_hidden(DFT_seq(X)))``.
    Every position ends up mixed with every other one, and the layer holds
    ZERO learnable parameters while doing it.

    **[ZERO TRAINABLE PARAMETERS]** The two DFT matrices are created with
    ``trainable=False``, so ``layer.trainable_weights`` is empty. They are
    stored as weights only so they round-trip through ``.keras``
    serialization instead of being rebuilt on load. Nothing here learns.

    **[NOT A QKV ATTENTION LAYER]** This is a token MIXER. There are no
    queries, keys or values, no learnable projections and no score matrix. The
    attention rules applied to its siblings have no counterpart here: the
    ``1 / sqrt(head_dim)`` temperature, the ``dim % num_heads`` precondition,
    and the additive ``-1e9`` mask bias with its fp16 hazard. They are not
    applicable rather than missing, which is why the module imports nothing
    from :mod:`~dl_techniques.layers.attention.common`.

    **[MASKING SEMANTICS DIFFER]** With no softmax to bias, ``attention_mask``
    is applied MULTIPLICATIVELY and only AFTER mixing. Padded positions still
    contribute to the mixed values of the real tokens; only the padded tokens'
    own outputs are zeroed. Don't assume the usual "masked tokens are
    invisible" guarantee. It does not hold here.

    **[COMPLEXITY]** The DFT is computed by multiplying with cached matrices,
    costing ``O(S^2 * D + S * D^2)`` per call for an input ``(B, S, D)`` - two
    complex matmuls. That is NOT the ``O(N log N)`` of a true FFT. No FFT path
    is implemented; see the ``implementation`` parameter.

    **[FORWARD PATH IS PURE ``keras.ops``]** ``numpy`` appears only inside
    ``_create_dft_matrix``, which runs once at ``build()`` time to produce a
    constant initializer. No ``numpy``, ``tf.`` or backend-specific call exists
    in ``call()``. The complex DFT is hand-rolled over a trailing real/imag
    pair rather than routed through ``keras.ops.fft2`` for two reasons.
    ``fft2`` takes and returns a ``(real, imag)`` TUPLE of tensors, which
    cannot be stored as one serializable weight, and it recomputes the
    transform on every call instead of reading a cached matrix. This is a
    design choice, not an unmigrated ``tf.`` exception.

    **Architecture Overview:**

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
          │   F_S: (S, S, 2)  CONSTANT WEIGHT,    │
          │        trainable=False                │
          └──────────────────┬────────────────────┘
                             │  [B, S, D, 2]
                             ▼
          ┌───────────────────────────────────────┐
          │ DFT along the hidden axis             │
          │   X'' = X' @ F_D  (complex matmul)    │
          │   F_D: (D, D, 2)  CONSTANT WEIGHT,    │
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
          │   MULTIPLICATIVE and POST-mix:        │
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
        multiplication. ``'fft'`` is **accepted but not implemented**. There
        is no true Fast Fourier Transform path here; passing ``'fft'`` logs a
        one-time warning and falls back to the ``'matrix'`` path with
        identical output. Don't reach for ``'fft'`` expecting a speedup.
        Defaults to ``'matrix'``.
    :type implementation: str
    :param normalize_dft: Whether to apply ``1/sqrt(N)`` normalization to the
        DFT matrices, which makes the transform unitary and keeps activation
        energy stable. Defaults to ``True``.
    :type normalize_dft: bool
    :param epsilon: Nominally a small constant for numerical stability. **It is
        validated, stored and serialized but never read on the forward path**
        - no division or logarithm in this layer needs a floor. It is kept
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

        # Validate inputs
        valid_implementations = ['matrix', 'fft']
        if implementation not in valid_implementations:
            raise ValueError(
                f"implementation must be one of {valid_implementations}, "
                f"got '{implementation}'"
            )
        # DECISION plan_2026-06-14_0c5d4a21/D-006
        # The originating plan directory is gone, so this comment is the record.
        # 'fft' is accepted but is NOT an FFT path: build() and call() always
        # use the matrix DFT. Don't raise NotImplementedError here.
        # `test_fnet_fourier_transform.py::test_different_implementations`
        # constructs implementation='fft' and asserts a finite forward, so
        # failing loud turns a green test red. Warn once and fall back instead.
        # Don't implement a real FFT either: the decision was to guard and
        # document this, not to re-implement it.
        if implementation == 'fft':
            logger.warning(
                "FNetFourierTransform(implementation='fft') is not implemented; "
                "a true FFT path does not exist. Falling back to the 'matrix' "
                "DFT path (identical output). Use implementation='matrix' to "
                "silence this warning."
            )
        # KNOWN DEAD PARAMETER, left in place on purpose.
        # `epsilon` is validated, stored and serialized but never read by
        # build(), call() or any helper. This layer has no division or log that
        # needs a floor. Don't delete it: it is a `get_config()` key, so
        # removing it breaks `load_model` on every existing checkpoint.
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
            OWN outputs, AFTER mixing. It does not stop padded tokens from
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

        # Zero the padded tokens' own rows, after mixing.
        #
        # This is MULTIPLICATIVE and post-hoc, unlike the additive `-1e9`
        # pre-softmax bias used elsewhere in this package. There is no softmax
        # here to bias, so nothing from `common` applies. Say the consequence
        # plainly: padded tokens are NOT excluded from the mixing. They
        # contribute to every real token's output, and only their own outputs
        # are zeroed. Don't unify this with the additive mask helper.
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

# ---------------------------------------------------------------------
