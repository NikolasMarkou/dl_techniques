"""
BlurPool2D, anti-aliased spatial downsampling with a fixed binomial blur.

A strided convolution or pooling layer subsamples a signal without first
removing content above the new Nyquist frequency, so high-frequency detail
aliases into low frequencies and a one-pixel input shift can change the
output substantially. This layer applies a low-pass filter before
subsampling instead: the 1-D binomial kernel `[1, 3, 3, 1] / 8`, outer-
producted with itself into a 4x4 kernel that sums to 1, replicated per
channel and applied as a single depthwise convolution with the configured
stride. The kernel is fixed and non-trainable, adds no parameters, and
mixes no channels, so it slots in wherever a strided downsample would sit.
This trades a small amount of genuine high-frequency detail (which the
filter cannot separate from aliased content) for shift-consistency.

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

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.blur_pool")
class BlurPool2D(keras.layers.Layer):
    """Anti-aliased depthwise downsampling with a fixed binomial blur.

    The 1-D binomial filter ``[1, 3, 3, 1] / 8`` is outer-producted with itself
    to form a 4x4 ``[1, 3, 3, 1] x [1, 3, 3, 1] / 64`` 2-D kernel that sums to
    one. The kernel is replicated per channel (depthwise) and is fixed,
    non-trainable. Spatial subsampling uses the configured stride.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C]
              │
              ▼
        fixed binomial kernel (4x4, non-trainable)
        depthwise conv, stride=strides, padding=padding
              │
              ▼
        Output [B, H', W', C]

    :param strides: Spatial stride. ``2`` is the standard anti-alias-2x downsample.
    :type strides: int
    :param padding: Either ``"same"`` or ``"valid"``.
    :type padding: str
    :param kwargs: Additional keyword arguments for :class:`keras.layers.Layer`.

    Example:

    .. code-block:: python

        x = keras.layers.Input(shape=(32, 32, 96))
        y = BlurPool2D(strides=2)(x)  # -> (None, 16, 16, 96)
    """

    def __init__(
        self,
        strides: int = 2,
        padding: str = "same",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(strides, int) or strides < 1:
            raise ValueError(
                f"strides must be a positive integer, got {strides!r}"
            )
        padding_lc = padding.lower()
        if padding_lc not in {"same", "valid"}:
            raise ValueError(
                f"padding must be 'same' or 'valid', got {padding!r}"
            )

        self.strides = strides
        self.padding = padding_lc

        self.kernel: Optional[keras.Variable] = None

    def build(self, input_shape: Any) -> None:
        if self.built:
            return

        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "BlurPool2D requires a static channel dimension; got None."
            )

        # 1-D binomial [1,3,3,1] -> 2-D outer product, normalised to sum to 1.
        f = np.array([1.0, 3.0, 3.0, 1.0], dtype=np.float32)
        kernel_2d = np.outer(f, f) / float(f.sum() ** 2)
        # Depthwise kernel shape: (kH, kW, C, 1).
        kernel_dw = np.broadcast_to(
            kernel_2d[:, :, None, None], (4, 4, channels, 1)
        ).astype(np.float32).copy()

        self.kernel = self.add_weight(
            name="blur_kernel",
            shape=kernel_dw.shape,
            dtype=self.compute_dtype,
            initializer=keras.initializers.Constant(kernel_dw),
            trainable=False,
        )

        logger.debug(
            f"BlurPool2D built: channels={channels}, strides={self.strides}"
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
            kh = kw = 4
            new_h = None if h is None else (h - kh) // self.strides + 1
            new_w = None if w is None else (w - kw) // self.strides + 1
        return (b, new_h, new_w, c)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "strides": self.strides,
                "padding": self.padding,
            }
        )
        return config
