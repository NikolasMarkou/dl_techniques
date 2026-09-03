"""Fixed 2D Haar wavelet decomposition filters.

This initializer is deterministic and does not perform random sampling.
Instead, it builds a fixed set of 2x2 convolution kernels that correspond
to the basis functions of the 2D discrete Haar wavelet transform. Its
purpose is to equip a convolutional layer with the ability to perform a
single level of multi-resolution analysis, decomposing an input signal (such
as an image) into distinct frequency sub-bands.

Architecture and Mathematical Foundations:
The Haar wavelet transform is the simplest form of wavelet analysis and is
based on a single prototype wavelet. In two dimensions, the decomposition is
achieved by applying the 1D transform separably along the rows and columns.
The 1D orthonormal Haar pair is ``(a + b) / sqrt(2)`` and ``(a - b) / sqrt(2)``,
so applying it once per axis gives four 2x2 filters whose every tap has
magnitude ``1/2``:

1.  **LL (approximation)**: ``[[0.5, 0.5], [0.5, 0.5]]``
    A 2x2 averager. When applied with a stride of 2, it produces a
    downsampled, lower-resolution version of the input, capturing its
    low-frequency "approximation" coefficients.

2.  **LH**: ``[[0.5, -0.5], [0.5, -0.5]]``
    Averages along the HEIGHT axis and differences along the WIDTH axis. It
    responds to vertical edges.

3.  **HL**: ``[[0.5, 0.5], [-0.5, -0.5]]``
    Differences along the HEIGHT axis and averages along the WIDTH axis. It
    responds to horizontal edges.

4.  **HH**: ``[[0.5, -0.5], [-0.5, 0.5]]``
    Differences along both axes, responding to diagonal detail.

Together these four filters form an ORTHONORMAL basis: the Gram matrix is the
identity, so with ``scale=1.0`` the transform preserves energy exactly
(``sum(c**2) == sum(x**2)`` per 2x2 block), leaves the variance of every
sub-band equal to the input variance, and is inverted by its own transpose.
Passing ``scale != 1.0`` preserves ORTHOGONALITY but not normality: energy is
then multiplied by ``scale**2``.

.. note::
    **Sub-band labels are library-dependent; the axis descriptions above are
    the contract.** The two-letter names follow the separable
    (row filter, column filter) ordering used here, and other wavelet libraries
    attach the words "horizontal" and "vertical" to the opposite band. If you
    are matching coefficients against an external reference, match on which
    axis is differenced, not on the label.

.. note::
    **Keras convolution is cross-correlation**, and no kernel flip is applied
    here. ``LL`` is symmetric and unaffected, but the three detail sub-bands
    carry the opposite SIGN to a true convolution against the same filters.
    This is irrelevant to learning and to energy, and relevant only when
    comparing coefficients with an external wavelet library.

.. note::
    A stride-2 ``'valid'`` decomposition consumes exactly 2x2 blocks, so an odd
    height or width would silently drop the last row/column.
    :func:`create_haar_depthwise_conv2d` rejects that up front.

References:
    - Mallat, S. (1989). *A theory for multiresolution signal
      decomposition: The wavelet representation*. IEEE Transactions on
      Pattern Analysis and Machine Intelligence.
    - Daubechies, I. (1992). *Ten lectures on wavelets*. Society for
      Industrial and Applied Mathematics.

"""

import keras
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

#: Sub-band order of the bank, i.e. the meaning of the last kernel axis.
SUBBAND_NAMES: Tuple[str, str, str, str] = ("LL", "LH", "HL", "HH")

#: The orthonormal 2D Haar basis, one 2x2 filter per entry of SUBBAND_NAMES.
#: Every tap is +/- 0.5, which is what makes the Gram matrix the identity.
HAAR_PATTERNS: np.ndarray = 0.5 * np.array([
    # LL: average both axes (approximation).
    [[1.0, 1.0],
     [1.0, 1.0]],
    # LH: average along height, difference along width (vertical edges).
    [[1.0, -1.0],
     [1.0, -1.0]],
    # HL: difference along height, average along width (horizontal edges).
    [[1.0, 1.0],
     [-1.0, -1.0]],
    # HH: difference along both axes (diagonal detail).
    [[1.0, -1.0],
     [-1.0, 1.0]],
], dtype=np.float64)


def _numpy_dtype(dtype: Any) -> str:
    """Keras dtype spec -> a numpy-acceptable dtype name.

    The Keras-2 ``standardize_dtype`` helper is banned tree-wide (see
    ``tests/test_the_keras2_backend_calls_are_gone.py``, whose predicate is a
    plain substring match, so naming the banned module even in prose trips it);
    this is the sanctioned replacement spelling.
    """
    return getattr(dtype, "name", None) or str(dtype)
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.initializers.haar_wavelet_initializer")
class HaarWaveletInitializer(keras.initializers.Initializer):
    """Haar wavelet initializer for convolutional layers.

    Fills a 2x2 kernel of shape ``(2, 2, in_channels, channel_multiplier)`` with
    the standard 2D Haar decomposition bank:

    - ``LL``: approximation (average both axes)
    - ``LH``: difference along the WIDTH axis (vertical edges)
    - ``HL``: difference along the HEIGHT axis (horizontal edges)
    - ``HH``: difference along both axes (diagonal detail)

    Output slot ``j`` of EVERY input channel receives pattern ``j % 4``, so all
    input channels see the same bank and the sub-band of an output slot is a
    property of ``j`` alone. In a ``DepthwiseConv2D`` the output channels are
    ordered input-channel-major, so output channel ``i * channel_multiplier + j``
    holds sub-band ``j % 4`` of input channel ``i``.

    With ``scale=1.0`` the four filters are ORTHONORMAL (Gram matrix = identity):
    energy is preserved exactly, every sub-band has the same variance as the
    input, and the transform is inverted by its own transpose. ``scale != 1.0``
    keeps them orthogonal but multiplies energy by ``scale ** 2``.

    Args:
        scale: Scaling factor for the wavelet coefficients. Must be positive.
            ``1.0`` (the default) is the orthonormal basis.
        seed: Accepted and IGNORED. This initializer is deterministic and draws
            no random numbers; the argument exists only so that configs saved by
            earlier versions still deserialize. It is not written to
            ``get_config``.

    Raises:
        ValueError: If ``scale`` is not positive.

    Example:
        >>> initializer = HaarWaveletInitializer(scale=1.0)
        >>> # 2x2 kernels, 3 input channels, 4 sub-bands per channel
        >>> weights = initializer((2, 2, 3, 4))
    """

    def __init__(
        self,
        scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Haar wavelet kernel initializer.

        Args:
            scale: Scaling factor for the wavelet coefficients; must be > 0.
            seed: Accepted and ignored (deterministic initializer).

        Raises:
            ValueError: If scale is not positive.
        """
        # NOTE: keras.initializers.Initializer (Keras 3) defines no __init__, so
        # there is nothing to forward to and a **kwargs passthrough could only
        # ever raise TypeError from object.__init__. This signature is closed.
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")

        self.scale = float(scale)

        if seed is not None:
            logger.debug(
                "HaarWaveletInitializer: `seed` is ignored -- the Haar bank is "
                "deterministic and draws no random numbers"
            )

        logger.debug(f"Initialized HaarWaveletInitializer with scale={self.scale}")

    def __call__(
        self,
        shape: Sequence[int],
        dtype: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """Generate the orthonormal Haar wavelet bank.

        Args:
            shape: Required shape ``(2, 2, in_channels, channel_multiplier)``.
            dtype: Data type of the tensor. ``None`` falls back to
                ``keras.config.floatx()``.
            **kwargs: Additional arguments (unused).

        Returns:
            Tensor: The Haar bank, with output slot ``j`` holding sub-band
            ``j % 4`` for every input channel.

        Raises:
            ValueError: If shape is not 4D, the kernel is not 2x2, or either
                channel dimension is < 1.
        """
        if dtype is None:
            dtype = keras.config.floatx()

        if len(shape) != 4:
            raise ValueError(f"Expected 4D shape, got {len(shape)}D")

        kernel_h, kernel_w, in_channels, channel_multiplier = (int(d) for d in shape)

        if kernel_h != 2 or kernel_w != 2:
            raise ValueError(
                f"Haar wavelets require 2x2 kernels, got {kernel_h}x{kernel_w}"
            )

        # A non-positive channel count used to leave the loop body unexecuted and
        # hand back an empty kernel with no error -- the worst failure mode for a
        # layer that is meant to be a fixed transform.
        if in_channels < 1 or channel_multiplier < 1:
            raise ValueError(
                f"in_channels and channel_multiplier must be >= 1, got "
                f"(in_channels={in_channels}, channel_multiplier={channel_multiplier})"
            )

        logger.debug(f"Generating Haar wavelet filters for shape {tuple(shape)}")

        # Scaling preserves orthogonality, but only scale == 1.0 is orthonormal.
        patterns = HAAR_PATTERNS * self.scale

        # Every input channel receives the SAME bank: the sub-band of an output
        # slot is a property of j alone, so downstream code can address it.
        bank = patterns[np.arange(channel_multiplier) % len(patterns)]  # (cm, 2, 2)
        kernel = np.transpose(bank, (1, 2, 0))[:, :, None, :]           # (2, 2, 1, cm)
        kernel = np.repeat(kernel, in_channels, axis=2)

        return keras.ops.convert_to_tensor(
            kernel.astype(_numpy_dtype(dtype)), dtype=dtype
        )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization.

        ``seed`` is deliberately absent: it does not affect the output, so
        persisting it would advertise a dependency that does not exist.

        Returns:
            Dict containing the initializer configuration.
        """
        config = super().get_config()
        config.update({
            'scale': self.scale,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HaarWaveletInitializer':
        """Create initializer from configuration.

        Args:
            config: Configuration dictionary. A ``seed`` key written by an
                earlier version is accepted and ignored.

        Returns:
            HaarWaveletInitializer instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------
# builder utility
# ---------------------------------------------------------------------

def create_haar_depthwise_conv2d(
    input_shape: Tuple[int, int, int],
    channel_multiplier: int = 4,
    scale: float = 1.0,
    use_bias: bool = False,
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    trainable: bool = False,
    name: Optional[str] = None
) -> keras.layers.DepthwiseConv2D:
    """Create a Haar wavelet depthwise convolution layer.

    Implements 2D Haar wavelet decomposition as a depthwise convolution with
    stride=2 for dyadic downsampling. This is the standard approach for
    wavelet decomposition in neural networks.

    Output channels are ordered input-channel-major: channel
    ``i * channel_multiplier + j`` holds sub-band ``SUBBAND_NAMES[j % 4]`` of
    input channel ``i``. With the default ``channel_multiplier=4`` that is
    ``[LL, LH, HL, HH]`` per input channel.

    Args:
        input_shape: Input tensor shape ``(height, width, channels)``. The
            spatial dimensions must be EVEN (or ``None``): a stride-2 ``'valid'``
            convolution consumes whole 2x2 blocks, so an odd size silently drops
            the last row/column and breaks perfect reconstruction.
        channel_multiplier: Output channels per input channel. For a full wavelet
            decomposition this should be 4 (LL, LH, HL, HH). Values above 4 cycle
            the bank and therefore duplicate filters.
        scale: Wavelet coefficient scaling factor. ``1.0`` is orthonormal.
        use_bias: Whether to add bias terms (typically False for wavelets).
        kernel_regularizer: Optional kernel regularization. Measured in Keras
            3.8: a regularizer on a NON-trainable weight contributes nothing to
            ``model.losses`` (0 terms frozen vs 1 term trainable), so combining
            it with ``trainable=False`` is a silent no-op and is warned about.
        trainable: Whether wavelet weights can be trained. Usually False for
            fixed wavelet transforms.
        name: Layer name.

    Returns:
        keras.layers.DepthwiseConv2D: Configured Haar wavelet layer.

    Raises:
        ValueError: If ``input_shape`` is not 3D, either spatial dimension is
            odd, or ``channel_multiplier`` is not positive.

    Example:
        >>> # Create a standard Haar wavelet decomposition layer
        >>> layer = create_haar_depthwise_conv2d(
        ...     input_shape=(256, 256, 3),
        ...     channel_multiplier=4,
        ...     trainable=False
        ... )
        >>> # Input: (batch, 256, 256, 3) -> Output: (batch, 128, 128, 12)
    """
    if len(input_shape) != 3:
        raise ValueError(f"Expected 3D input shape (H,W,C), got {len(input_shape)}D")

    if channel_multiplier <= 0:
        raise ValueError(f"channel_multiplier must be positive, got {channel_multiplier}")

    height, width, _ = input_shape
    odd = [
        f"{axis}={size}"
        for axis, size in (("height", height), ("width", width))
        if size is not None and size % 2 != 0
    ]
    if odd:
        raise ValueError(
            f"a stride-2 'valid' Haar decomposition requires EVEN spatial "
            f"dimensions, got {', '.join(odd)} in input_shape={input_shape}. "
            f"The last row/column would be silently dropped, breaking perfect "
            f"reconstruction -- crop or pad the input to an even size first."
        )

    # Log warning if using non-standard configuration
    if channel_multiplier != 4 and not trainable:
        logger.warning(
            f"Using channel_multiplier={channel_multiplier} with trainable=False. "
            "For standard wavelet decomposition, channel_multiplier should be 4."
        )
    if channel_multiplier > len(SUBBAND_NAMES):
        duplicates = channel_multiplier - len(SUBBAND_NAMES)
        logger.warning(
            f"channel_multiplier={channel_multiplier} cycles the 4-filter Haar "
            f"bank, so {duplicates} of every {channel_multiplier} output slots "
            "per input channel are exact duplicates of an earlier one; with "
            "trainable=False those feature maps stay bit-identical forever."
        )
    if kernel_regularizer is not None and not trainable:
        logger.warning(
            "kernel_regularizer is set on a frozen Haar kernel. Keras does not "
            "collect a regularization loss from a non-trainable weight, so this "
            "is a silent no-op -- set trainable=True or drop the regularizer."
        )

    logger.debug(
        f"Creating Haar wavelet layer: input_shape={input_shape}, "
        f"channel_multiplier={channel_multiplier}, trainable={trainable}"
    )

    return keras.layers.DepthwiseConv2D(
        kernel_size=2,
        strides=2,  # Dyadic downsampling for wavelet decomposition
        padding='valid',
        depth_multiplier=channel_multiplier,
        use_bias=use_bias,
        depthwise_initializer=HaarWaveletInitializer(scale=scale),
        depthwise_regularizer=kernel_regularizer,
        trainable=trainable,
        name=name or 'haar_dwconv'
    )

# ---------------------------------------------------------------------
