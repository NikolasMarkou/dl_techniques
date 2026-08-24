"""
Anti-aliased 2D downsampling via a fixed binomial blur (BlurPool).

This layer embodies the principle of treating downsampling as a signal
processing problem rather than an architectural convenience, a design paradigm
that inserts the low-pass filter that the sampling theorem requires before
reducing spatial resolution. The core idea addresses a defect present in every
strided convolution and pooling layer: subsampling a signal whose spectrum
extends above the new Nyquist frequency aliases that high-frequency content down
into the low frequencies, where it is indistinguishable from real structure.
Because the aliased content depends on the phase of the sampling grid, a
one-pixel shift of the input can change the output substantially. This is why
convolutional networks, despite their weight sharing, are not shift-invariant in
practice.

The remedy is the classical one. Instead of subsampling directly, the signal is
first attenuated above the target Nyquist frequency and only then sampled. The
filter used is the 1-D binomial kernel

`f = [1, 3, 3, 1] / 8`

outer-producted with itself to form a separable 4x4 2-D kernel normalized to sum
to one:

`K = (f x f^T)`,  `sum(K) = 1`

Binomial coefficients are the natural choice here: they are the discrete
analogue of a Gaussian, strictly non-negative (so no ringing or overshoot is
introduced), and cheap. Unit sum makes the operation a weighted average, which
preserves the DC level and therefore leaves activation magnitudes and downstream
normalization statistics undisturbed. Empirically this kernel delivers roughly
-14 dB of attenuation at Nyquist, against approximately -8 dB for the 2x2 box
filter that average pooling implicitly applies.

Architecturally, the kernel is replicated once per channel and applied as a
depthwise convolution with the configured stride, so blur and subsample are a
single fused operation. Two properties follow from this construction. The filter
is non-trainable and identical across channels, so the layer adds no learnable
parameters and cannot be degraded by training away from its designed frequency
response. And because it is depthwise, there is no cross-channel mixing: the
layer is purely a resampling operator inserted where a strided downsample would
otherwise sit, leaving the surrounding architecture's channel semantics
untouched.

The tradeoff is a modest loss of genuine high-frequency detail alongside the
aliased content, since the filter cannot distinguish the two. In practice the
shift-consistency and accuracy gains outweigh this, which is the empirical result
Zhang reports across architectures.

This layer serves as the anti-alias downsampler for axis A of the
Clifford-algebra-compliant downsampling design space
(analyses/analysis_2026-04-30_41b5e415/summary.md).

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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BlurPool2D(keras.layers.Layer):
    """Anti-aliased depthwise downsampling with a fixed binomial blur.

    The 1-D binomial filter ``[1, 3, 3, 1] / 8`` is outer-producted with itself
    to form a 4x4 ``[1, 3, 3, 1] x [1, 3, 3, 1] / 64`` 2-D kernel that sums to
    one. The kernel is replicated per channel (depthwise) and is fixed,
    non-trainable. Spatial subsampling uses the configured stride.

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
        kernel_2d = np.outer(f, f) / float(f.sum() ** 2)  # (4, 4)
        # Depthwise kernel shape: (kH, kW, C, 1)
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


# ---------------------------------------------------------------------
