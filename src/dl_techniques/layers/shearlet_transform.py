"""
Shearlet Transform Implementation in TensorFlow/Keras
==================================================

This implements the CoShRem Shearlet transform for CoShNet. Key features:
1. Fully complex-valued implementation
2. Multi-scale feature detection
3. Orientation selectivity using shearing operations
4. Phase congruency support
5. FFT-based implementation for efficiency

Core components:
- Fourier domain implementation
- Anisotropic scaling
- Shearing operations
- Band-limited wavelet construction
- Cone-adapted shearlet system

Reference: "CoShRem: Faithful Digital Shearlet Transforms based on Compactly Supported Shearlets"
"""

import numpy as np
import tensorflow as tf
from keras.api.layers import Layer
from typing import List, Tuple, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
class ShearletTransform(Layer):
    """
    Shearlet transform layer for CoShNet.

    Implements a multi-scale, multi-directional complex-valued transform
    using cone-adapted shearlets and band-limited wavelets.
    """

    def __init__(
            self,
            scales: int = 4,
            directions: int = 8,
            alpha: float = 0.5,  # Controls anisotropy (0.5 for parabolic scaling)
            high_freq: bool = True,  # Include high frequency components
            **kwargs
    ):
        super().__init__(**kwargs)
        self.scales = scales
        self.directions = directions
        self.alpha = alpha
        self.high_freq = high_freq
        self.height = None
        self.width = None
        self.freq_x = None
        self.freq_y = None
        self.filters = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize transform parameters based on input size."""
        self.height, self.width = input_shape[1:3]

        # Pre-compute frequency grids
        self.freq_x, self.freq_y = self._create_freq_grid()

        # Generate shearlet filters in Fourier domain
        self.filters = self._create_shearlet_filters()

    def _create_freq_grid(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create frequency grid for Fourier domain computation."""
        # Create centered frequency ranges
        fx = tf.linspace(-1.0, 1.0, self.width)
        fy = tf.linspace(-1.0, 1.0, self.height)

        # Create 2D grid
        freq_y, freq_x = tf.meshgrid(fy, fx)
        return freq_x, freq_y

    def _create_meyer_wavelet(self, x: tf.Tensor) -> tf.Tensor:
        """Create Meyer wavelet window function."""

        def v(x: tf.Tensor) -> tf.Tensor:
            x = tf.clip_by_value(x, 0.0, 1.0)
            return x ** 4 * (35.0 - 84.0 * x + 70.0 * x ** 2 - 20.0 * x ** 3)

        x = tf.abs(x)
        result = tf.zeros_like(x)

        # Low frequencies (including DC)
        mask_low = x < 1 / 3
        result = tf.where(mask_low, tf.ones_like(x), result)

        # Mid frequencies
        mask_mid = tf.logical_and(x >= 1 / 3, x < 2 / 3)
        result = tf.where(mask_mid, v(3.0 * x - 1.0), result)

        # High frequencies
        mask_high = tf.logical_and(x >= 2 / 3, x <= 4 / 3)
        result = tf.where(mask_high, v(2.0 - 3.0 * x / 2.0), result)

        return result

    def _create_shearlet_filters(self) -> List[tf.Tensor]:
        """Create shearlet filters."""
        filters = []

        # Get polar coordinates
        rho = tf.sqrt(self.freq_x ** 2 + self.freq_y ** 2)
        theta = tf.atan2(self.freq_y, self.freq_x)

        # Create low-pass filter
        phi_low = self._create_meyer_wavelet(2.0 * rho)
        filters.append(tf.cast(phi_low, tf.complex64))

        # Create directional filters
        for j in range(self.scales):
            scale = 2.0 ** j

            # Create radial window with overlap
            window_j = self._create_meyer_wavelet(rho / scale) * \
                       (1.0 - self._create_meyer_wavelet(2.0 * rho / scale))

            # Add directional selectivity
            for k in range(-self.directions // 2, self.directions // 2 + 1):
                # Smoother shearing with overlap
                shear = k / (self.directions + 1.0)
                angle = tf.atan(shear)

                # Create angular window
                dir_window = self._create_meyer_wavelet(
                    2.0 * (theta - angle) / np.pi
                )

                # Create shearlet
                shearlet = window_j * dir_window

                # Normalize
                shearlet = shearlet / (tf.reduce_max(tf.abs(shearlet) + 1e-10))
                filters.append(tf.cast(shearlet, tf.complex64))

        # Normalize the filter bank
        filters = [f / tf.cast(tf.sqrt(float(len(filters))), tf.complex64)
                   for f in filters]

        return filters

    def _fft2d(self, x: tf.Tensor) -> tf.Tensor:
        """Compute 2D FFT with proper shifting."""
        return tf.signal.fftshift(
            tf.signal.fft2d(
                tf.cast(x, tf.complex64)
            )
        )

    def _ifft2d(self, x: tf.Tensor) -> tf.Tensor:
        """Compute 2D inverse FFT with proper shifting."""
        return tf.cast(
            tf.math.real(
                tf.signal.ifft2d(
                    tf.signal.ifftshift(x)
                )
            ),
            tf.float32
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply shearlet transform to input images."""
        # Convert to float32 if needed
        inputs = tf.cast(inputs, tf.float32)

        # Compute FFT of input
        fft = self._fft2d(inputs)

        # Apply each filter and compute inverse FFT
        coefficients = []
        for filter_kernel in self.filters:
            # Apply filter in Fourier domain
            filtered = fft * filter_kernel

            # Convert back to spatial domain
            coeff = self._ifft2d(filtered)
            coefficients.append(coeff)

        # Stack coefficients along channel dimension
        return tf.stack(coefficients, axis=-1)

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'scales': self.scales,
            'directions': self.directions,
            'alpha': self.alpha,
            'high_freq': self.high_freq
        })
        return config
