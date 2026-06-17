"""
CliffordNet explicit-Laplacian-pyramid image autoencoder.

This module builds a deterministic image autoencoder that fuses an EXPLICIT,
inspectable Laplacian pyramid with CliffordNet geometric-algebra feature
extraction. The defining design choice is that the multi-scale split/merge is a
standalone, serializable helper Layer (``LaplacianPyramidLevel``) operating at
the RAW signal channel count, completely decoupled from any learned Clifford
processing. This makes the high/low frequency decomposition auditable and makes
the reconstruction identity exact to float precision.

Channel-bookkeeping scheme (LOCKED)
-----------------------------------
The low/high decomposition is deliberately *channel-preserving* so the Laplacian
reconstruction identity holds exactly in raw signal space:

- ``low_i  = down(blur(x_i))``  via ``GaussianFilter(strides=(1,1), padding="same")``
  (preserves C) then ``BlurPool2D(strides=2)`` (preserves C). Result has shape
  ``(B, H/2, W/2, C_i)`` -- same channel count as ``x_i``. Both stages are
  anti-aliased.
- ``up`` via ``UpSampling2D(bilinear)`` -- preserves channels, doubles H, W.
- ``high_i = x_i - up(low_i)``: shape ``(B, H, W, C_i)``, same channels as ``x_i``.
  Since ``high_i`` is DEFINED as the residual ``x_i - up(low_i)``, the merge
  ``high_i + up(low_i)`` recovers ``x_i`` exactly (up to float rounding),
  independent of blur quality.

The split is purely signal-level (NO learnable channel change): this is what
makes the reconstruction identity hold. ``PixelUnshuffle2D`` is deliberately NOT
used for the low path -- its 4x channel multiplier breaks the additive identity.
It is reserved as a documented alternative only if a future lossless variant is
wanted; the default path is Gaussian + BlurPool down / bilinear up.

The model class (``CliffordLaplacianUNet``) widens channels via external 1x1
``Conv2D`` before each isotropic ``CliffordNetBlock`` group; the Laplacian SIGNAL
pyramid itself stays at the raw channel count. Clifford blocks process widened
feature copies that contribute learned refinements on the decoder side; they
never sit inside the invertibility path of this helper. (Model class added in a
later step; this module currently defines only the helper layer.)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras

from dl_techniques.layers.gaussian_filter import GaussianFilter
from dl_techniques.layers.blur_pool import BlurPool2D
from dl_techniques.utils.logger import logger


# ===========================================================================
# LaplacianPyramidLevel
# ===========================================================================


@keras.saving.register_keras_serializable()
class LaplacianPyramidLevel(keras.layers.Layer):
    """Explicit, signal-level Laplacian pyramid split/merge for one level.

    split(x) -> (low, high):  low = down(blur(x)) at H/2; high = x - up(low) at H.
    merge(low, high) -> high + up(low) == x  (exact reconstruction identity).
    Pure signal-level (channel-preserving) -- NO learnable channel change, so the
    reconstruction identity holds to float precision.

    :param blur_kernel_size: Height/width of the Gaussian blur kernel.
    :type blur_kernel_size: Tuple[int, int]
    :param blur_sigma: Gaussian sigma; ``-1`` derives it from the kernel size.
    :type blur_sigma: float
    :param blur_trainable: If True the blur kernel is learnable (default False
        keeps the split a fixed, auditable signal operation).
    :type blur_trainable: bool
    :param kwargs: Additional keyword arguments for :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: float = -1,
        blur_trainable: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.blur_trainable = blur_trainable

        # Sublayers created in __init__ (built explicitly in build()).
        self.blur = GaussianFilter(
            kernel_size=blur_kernel_size,
            strides=(1, 1),
            sigma=blur_sigma,
            padding="same",
            trainable=blur_trainable,
        )
        self.down = BlurPool2D(strides=2)
        self.up = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

    def build(self, input_shape) -> None:
        # Build sublayers in computational order; super().build() LAST.
        self.blur.build(input_shape)
        blur_out = self.blur.compute_output_shape(input_shape)
        self.down.build(blur_out)
        low_shape = self.down.compute_output_shape(blur_out)
        self.up.build(low_shape)
        super().build(input_shape)

    def split(self, x):
        """Decompose ``x`` into ``(low, high)`` signal bands.

        :param x: Input tensor ``(B, H, W, C)``.
        :return: ``(low, high)`` where ``low`` is ``(B, H/2, W/2, C)`` and
            ``high`` is ``(B, H, W, C)``.
        """
        low = self.down(self.blur(x))
        high = keras.ops.subtract(x, self.up(low))
        return low, high

    def merge(self, low, high):
        """Reconstruct the level: ``high + up(low)`` (exact inverse of split).

        :param low: Low band ``(B, H/2, W/2, C)``.
        :param high: High band ``(B, H, W, C)``.
        :return: Reconstructed tensor ``(B, H, W, C)``.
        """
        return keras.ops.add(high, self.up(low))

    def call(self, inputs):
        return self.split(inputs)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        low_h = None if h is None else h // 2
        low_w = None if w is None else w // 2
        low_shape = (batch, low_h, low_w, c)
        high_shape = (batch, h, w, c)
        return low_shape, high_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "blur_kernel_size": self.blur_kernel_size,
                "blur_sigma": self.blur_sigma,
                "blur_trainable": self.blur_trainable,
            }
        )
        return config


# ---------------------------------------------------------------------
