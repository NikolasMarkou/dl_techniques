"""Fixed 2D Haar wavelet decomposition filters.

Provides :class:`HaarWaveletInitializer`, which fills a 2x2 convolution kernel
with the four basis filters of the 2D discrete Haar wavelet transform, and
:func:`create_haar_depthwise_conv2d`, which wires that bank into a stride-2
``DepthwiseConv2D``. Together they give a layer one level of multi-resolution
analysis, splitting an image into frequency sub-bands.

The initializer is deterministic and draws no random numbers.

The four filters
----------------
The 1D orthonormal Haar pair is ``(a + b) / sqrt(2)`` and ``(a - b) / sqrt(2)``.
Applying it once per axis gives four 2x2 filters whose every tap has magnitude
``1/2``:

1. **LL (approximation)**: ``[[0.5, 0.5], [0.5, 0.5]]``. A 2x2 averager. At
   stride 2 it produces a downsampled, lower-resolution copy of the input.
2. **LH**: ``[[0.5, -0.5], [0.5, -0.5]]``. Averages along height, differences
   along width. Responds to vertical edges.
3. **HL**: ``[[0.5, 0.5], [-0.5, -0.5]]``. Differences along height, averages
   along width. Responds to horizontal edges.
4. **HH**: ``[[0.5, -0.5], [-0.5, 0.5]]``. Differences along both axes.
   Responds to diagonal detail.

The four form an orthonormal basis: the Gram matrix is the identity. With
``scale=1.0`` the transform preserves energy exactly, ``sum(c**2) ==
sum(x**2)`` per 2x2 block, leaves every sub-band with the input's variance, and
is inverted by its own transpose. ``scale != 1.0`` keeps them orthogonal but
multiplies energy by ``scale**2``.

.. note::
    Sub-band labels are library-dependent; the axis descriptions above are the
    contract. The two-letter names follow the separable (row filter, column
    filter) ordering used here, and other wavelet libraries attach the words
    "horizontal" and "vertical" to the opposite band. When matching
    coefficients against an external reference, match on which axis is
    differenced, not on the label.

.. note::
    Keras convolution is cross-correlation, and no kernel flip is applied here.
    ``LL`` is symmetric and unaffected, but the three detail sub-bands carry the
    opposite sign to a true convolution against the same filters. That matters
    only when comparing coefficients with an external wavelet library, not for
    learning or for energy.

.. note::
    A stride-2 ``'valid'`` decomposition consumes exactly 2x2 blocks, so an odd
    height or width would silently drop the last row or column.
    :func:`create_haar_depthwise_conv2d` rejects that up front.

References:
    - Mallat, S. (1989). *A theory for multiresolution signal decomposition: The
      wavelet representation*. IEEE Transactions on Pattern Analysis and Machine
      Intelligence.
    - Daubechies, I. (1992). *Ten lectures on wavelets*. Society for Industrial
      and Applied Mathematics.
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
    """Convert a Keras dtype spec to a numpy-acceptable dtype name.

    The Keras-2 ``standardize_dtype`` helper is banned tree-wide (see
    ``tests/test_the_keras2_backend_calls_are_gone.py``); this is the sanctioned
    replacement spelling.

    :param dtype: A dtype object or name.
    :return: The dtype name.
    :rtype: str
    """
    return getattr(dtype, "name", None) or str(dtype)
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.initializers.haar_wavelet_initializer")
class HaarWaveletInitializer(keras.initializers.Initializer):
    """Fill a 2x2 kernel with the 2D Haar decomposition bank.

    The kernel shape is ``(2, 2, in_channels, channel_multiplier)``.

    **Architecture overview:**

    .. code-block:: text

        requested shape (2, 2, in_channels, channel_multiplier)
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │ validate: rank 4, kernel 2x2,        │  raises ValueError
        │ both channel dims >= 1               │
        └──────────────────┬───────────────────┘
                           ▼
                  HAAR_PATTERNS * scale
                           │ [4, 2, 2]
                           ▼
        ┌──────────────────────────────────────┐
        │ index by arange(cm) % 4              │
        │ (cycles the bank across output slots)│
        └──────────────────┬───────────────────┘
                           │ [cm, 2, 2]
                           ▼
                   transpose + repeat
                           │
                           ▼
              [2, 2, in_channels, cm]

    **Sub-band per output slot:**

    .. code-block:: text

        slot j   j % 4   filter                    responds to
        ------   -----   -----------------------   ------------------
        0        0       LL  [[ .5, .5],[ .5, .5]]  approximation
        1        1       LH  [[ .5,-.5],[ .5,-.5]]  vertical edges
        2        2       HL  [[ .5, .5],[-.5,-.5]]  horizontal edges
        3        3       HH  [[ .5,-.5],[-.5, .5]]  diagonal detail

    Output slot ``j`` of every input channel receives pattern ``j % 4``, so all
    input channels see the same bank and the sub-band of an output slot is a
    property of ``j`` alone. In a ``DepthwiseConv2D`` the output channels are
    ordered input-channel-major, so output channel
    ``i * channel_multiplier + j`` holds sub-band ``j % 4`` of input channel
    ``i``.

    With ``scale=1.0`` the four filters are orthonormal: energy is preserved
    exactly, every sub-band has the input's variance, and the transform is
    inverted by its own transpose. ``scale != 1.0`` keeps them orthogonal but
    multiplies energy by ``scale ** 2``.

    :param scale: Scaling factor for the wavelet coefficients. Must be positive.
        ``1.0`` is the orthonormal basis.
    :type scale: float
    :param seed: Accepted and ignored. This initializer is deterministic and
        draws no random numbers; the argument exists so that configs saved by
        earlier versions still deserialize. It is not written to
        :meth:`get_config`.
    :type seed: int or None

    :ivar scale: The coerced coefficient scale.
    :vartype scale: float

    :raises ValueError: If ``scale`` is not positive.

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
        """Validate and store the coefficient scale.

        :param scale: Scaling factor for the wavelet coefficients; must be > 0.
        :type scale: float
        :param seed: Accepted and ignored.
        :type seed: int or None
        :raises ValueError: If ``scale`` is not positive.
        """
        # keras.initializers.Initializer (Keras 3) defines no __init__, so there
        # is nothing to forward to and this signature is closed.
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

        :param shape: Required shape ``(2, 2, in_channels, channel_multiplier)``.
        :type shape: sequence of int
        :param dtype: Data type of the result. ``None`` falls back to
            ``keras.config.floatx()``.
        :type dtype: str or None
        :param kwargs: Additional arguments (unused).
        :return: The Haar bank, with output slot ``j`` holding sub-band
            ``j % 4`` for every input channel.
        :rtype: tensor
        :raises ValueError: If the shape is not 4D, the kernel is not 2x2, or
            either channel dimension is below 1.
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

        # A non-positive channel count would otherwise hand back an empty kernel
        # with no error, the worst failure mode for a fixed transform.
        if in_channels < 1 or channel_multiplier < 1:
            raise ValueError(
                f"in_channels and channel_multiplier must be >= 1, got "
                f"(in_channels={in_channels}, channel_multiplier={channel_multiplier})"
            )

        logger.debug(f"Generating Haar wavelet filters for shape {tuple(shape)}")

        # Scaling preserves orthogonality, but only scale == 1.0 is orthonormal.
        patterns = HAAR_PATTERNS * self.scale

        # Every input channel receives the same bank, so the sub-band of an
        # output slot is a property of j alone and downstream code can address it.
        # Shape after this: (cm, 2, 2).
        bank = patterns[np.arange(channel_multiplier) % len(patterns)]
        # Shape after this: (2, 2, 1, cm).
        kernel = np.transpose(bank, (1, 2, 0))[:, :, None, :]
        kernel = np.repeat(kernel, in_channels, axis=2)

        return keras.ops.convert_to_tensor(
            kernel.astype(_numpy_dtype(dtype)), dtype=dtype
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        ``seed`` is absent: it does not affect the output, so persisting it
        would advertise a dependency that does not exist.

        :return: A dict holding ``scale``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'scale': self.scale,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HaarWaveletInitializer':
        """Rebuild an initializer from a config dict.

        :param config: Configuration dictionary. A ``seed`` key written by an
            earlier version is accepted and ignored.
        :type config: dict
        :return: A new initializer.
        :rtype: HaarWaveletInitializer
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
    stride 2 for dyadic downsampling, the standard way to do wavelet
    decomposition in a network.

    **Layer wiring:**

    .. code-block:: text

        input [B, H, W, C]     (H and W must be even)
              │
              ▼
        ┌────────────────────────────────────┐
        │ DepthwiseConv2D                    │
        │   kernel_size = 2                  │
        │   strides = 2, padding = 'valid'   │
        │   depth_multiplier = cm            │
        │   depthwise_initializer =          │
        │       HaarWaveletInitializer(scale)│
        │   trainable = trainable (False by  │
        │       default: a fixed transform)  │
        └────────────────┬───────────────────┘
                         ▼
        output [B, H/2, W/2, C * cm]

        channel ordering, input-channel-major:
          i * cm + j  ->  sub-band SUBBAND_NAMES[j % 4] of input channel i

    With the default ``channel_multiplier=4`` that is ``[LL, LH, HL, HH]`` per
    input channel.

    :param input_shape: Input tensor shape ``(height, width, channels)``. The
        spatial dimensions must be even, or ``None``: a stride-2 ``'valid'``
        convolution consumes whole 2x2 blocks, so an odd size drops the last row
        or column and breaks perfect reconstruction.
    :type input_shape: tuple of int
    :param channel_multiplier: Output channels per input channel. A full wavelet
        decomposition needs 4 (LL, LH, HL, HH). Values above 4 cycle the bank
        and duplicate filters.
    :type channel_multiplier: int
    :param scale: Wavelet coefficient scaling factor. ``1.0`` is orthonormal.
    :type scale: float
    :param use_bias: Whether to add bias terms. Typically ``False`` for wavelets.
    :type use_bias: bool
    :param kernel_regularizer: Optional kernel regularization. Measured in Keras
        3.8: a regularizer on a non-trainable weight contributes nothing to
        ``model.losses`` (0 terms frozen against 1 term trainable), so combining
        it with ``trainable=False`` is a silent no-op and is warned about.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param trainable: Whether the wavelet weights can be trained. Usually
        ``False`` for a fixed transform.
    :type trainable: bool
    :param name: Layer name.
    :type name: str or None
    :return: The configured Haar wavelet layer.
    :rtype: keras.layers.DepthwiseConv2D
    :raises ValueError: If ``input_shape`` is not 3D, either spatial dimension
        is odd, or ``channel_multiplier`` is not positive.

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
        # Stride 2 gives the dyadic downsampling of a wavelet decomposition.
        strides=2,
        padding='valid',
        depth_multiplier=channel_multiplier,
        use_bias=use_bias,
        depthwise_initializer=HaarWaveletInitializer(scale=scale),
        depthwise_regularizer=kernel_regularizer,
        trainable=trainable,
        name=name or 'haar_dwconv'
    )

# ---------------------------------------------------------------------
