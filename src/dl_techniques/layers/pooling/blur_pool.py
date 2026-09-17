"""
BlurPool2D, anti-aliased spatial downsampling with a fixed binomial blur.

A strided convolution or pooling layer subsamples a signal without first
removing content above the new Nyquist frequency, so high-frequency detail
aliases into low frequencies and a one-pixel input shift can change the
output substantially. This layer applies a low-pass filter before
subsampling instead: a 1-D binomial kernel of length ``kernel_size``, outer-
producted with itself into a 2-D kernel that sums to 1, replicated per
channel and applied as a single depthwise convolution with the configured
stride. The kernel is fixed and non-trainable, adds no parameters, and
mixes no channels, so it slots in wherever a strided downsample would sit.
This trades a small amount of genuine high-frequency detail (which the
filter cannot separate from aliased content) for shift-consistency.

``kernel_size=1`` is the degenerate case: the kernel is the scalar ``1.0``
and the layer becomes a pure decimator with no filtering. Use it when the
caller has already low-passed the input, to avoid blurring twice.

References:
    - Zhang, 2019. Making Convolutional Networks Shift-Invariant Again. ICML
      2019. (https://arxiv.org/abs/1904.11486)
    - Azulay and Weiss, 2019. Why do deep convolutional networks generalize so
      poorly to small image transformations? JMLR 20(184).
      (https://arxiv.org/abs/1805.12177)
    - Burt and Adelson, 1983. The Laplacian Pyramid as a Compact Image Code.
      IEEE Transactions on Communications 31(4).
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _binomial_1d(size: int) -> np.ndarray:
    """Return the unnormalised length-``size`` binomial filter.

    ``1 -> [1]``, ``2 -> [1, 1]``, ``3 -> [1, 2, 1]``, ``4 -> [1, 3, 3, 1]``,
    ``5 -> [1, 4, 6, 4, 1]``, and so on.

    :param size: Filter length, a positive int.
    :type size: int
    :return: 1-D binomial coefficients as float32.
    :rtype: np.ndarray
    """
    f = np.array([1.0], dtype=np.float64)
    for _ in range(size - 1):
        f = np.convolve(f, np.array([1.0, 1.0], dtype=np.float64))
    return f.astype(np.float32)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.pooling.blur_pool")
class BlurPool2D(keras.layers.Layer):
    """Anti-aliased depthwise downsampling with a fixed binomial blur.

    The 1-D binomial filter of length ``kernel_size`` is outer-producted with
    itself to form a ``kernel_size x kernel_size`` 2-D kernel that sums to one.
    The kernel is replicated per channel (depthwise) and is fixed,
    non-trainable. Spatial subsampling uses the configured stride.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C]
              │
              ▼
        fixed binomial kernel (kernel_size x kernel_size, non-trainable)
        depthwise conv, stride=strides, padding=padding
              │
              ▼
        Output [B, H', W', C]

    :param strides: Spatial stride. ``2`` is the standard anti-alias-2x downsample.
    :type strides: int
    :param padding: Either ``"same"`` or ``"valid"``.
    :type padding: str
    :param kernel_size: Length of the 1-D binomial filter, a positive int.
        ``4`` is the ``[1, 3, 3, 1] / 8`` filter of Zhang (2019) and the
        default. ``1`` disables filtering and makes the layer a pure
        decimator, for callers that have already low-passed the input.
    :type kernel_size: int
    :param kwargs: Additional keyword arguments for :class:`keras.layers.Layer`.

    :raises ValueError: If ``strides`` or ``kernel_size`` is not a positive
        int, if ``padding`` is neither ``"same"`` nor ``"valid"``, or, at build
        time, if the channel axis is undefined.

    Example:

    .. code-block:: python

        x = keras.layers.Input(shape=(32, 32, 96))
        y = BlurPool2D(strides=2)(x)                  # -> (None, 16, 16, 96)
        z = BlurPool2D(strides=2, kernel_size=1)(x)   # pure decimation
    """

    def __init__(
        self,
        strides: int = 2,
        padding: str = "same",
        kernel_size: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # bool is a subclass of int, so it is excluded first.
        if isinstance(strides, bool) or not isinstance(strides, int) or strides < 1:
            raise ValueError(
                f"strides must be a positive integer, got {strides!r}"
            )
        if (isinstance(kernel_size, bool)
                or not isinstance(kernel_size, int)
                or kernel_size < 1):
            raise ValueError(
                f"kernel_size must be a positive integer, got {kernel_size!r}"
            )
        padding_lc = padding.lower()
        if padding_lc not in {"same", "valid"}:
            raise ValueError(
                f"padding must be 'same' or 'valid', got {padding!r}"
            )

        self.strides = strides
        self.padding = padding_lc
        self.kernel_size = kernel_size

        self.kernel: Optional[keras.Variable] = None

    def build(self, input_shape: Any) -> None:
        if self.built:
            return

        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "BlurPool2D requires a static channel dimension; got None."
            )

        # 1-D binomial -> 2-D outer product, normalised to sum to 1.
        # At kernel_size == 1 this is the scalar 1.0, so the conv is a no-op
        # and only the stride acts.
        f = _binomial_1d(self.kernel_size)
        kernel_2d = np.outer(f, f) / float(f.sum() ** 2)
        # Depthwise kernel shape: (kH, kW, C, 1).
        kernel_dw = np.broadcast_to(
            kernel_2d[:, :, None, None],
            (self.kernel_size, self.kernel_size, channels, 1),
        ).astype(np.float32).copy()

        self.kernel = self.add_weight(
            name="blur_kernel",
            shape=kernel_dw.shape,
            dtype=self.compute_dtype,
            initializer=keras.initializers.Constant(kernel_dw),
            trainable=False,
        )

        logger.debug(
            f"BlurPool2D built: channels={channels}, strides={self.strides}, "
            f"kernel_size={self.kernel_size}"
        )

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return ops.nn.depthwise_conv(
            inputs=inputs,
            kernel=self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format="channels_last",
        )

    def compute_output_shape(self, input_shape: Any) -> Any:
        b, h, w, c = input_shape
        if self.padding == "same":
            new_h = None if h is None else (h + self.strides - 1) // self.strides
            new_w = None if w is None else (w + self.strides - 1) // self.strides
        else:  # valid
            kh = kw = self.kernel_size
            new_h = None if h is None else (h - kh) // self.strides + 1
            new_w = None if w is None else (w - kw) // self.strides + 1
        return (b, new_h, new_w, c)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "strides": self.strides,
                "padding": self.padding,
                "kernel_size": self.kernel_size,
            }
        )
        return config

# ---------------------------------------------------------------------
