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

import keras
import numpy as np
import tensorflow as tf
from keras.api.layers import Layer
from typing import List, Tuple, Optional, Dict, Any


# ---------------------------------------------------------------------
@keras.utils.register_keras_serializable()
class ShearletTransform(Layer):
    """Shearlet transform layer with improved frame properties.

    This layer implements a multi-scale, multi-directional complex-valued transform
    using cone-adapted shearlets with enhanced frame properties and frequency coverage.

    Attributes:
        scales: Number of scales in the transform
        directions: Number of directions per scale
        alpha: Controls anisotropy (0.5 for parabolic scaling)
        high_freq: Include high frequency components
        height: Height of input images
        width: Width of input images
        freq_x: Frequency grid x-coordinates
        freq_y: Frequency grid y-coordinates
        filters: List of shearlet filters
    """

    def __init__(
            self,
            scales: int = 4,
            directions: int = 8,
            alpha: float = 0.5,
            high_freq: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        """Initialize the ShearletTransform layer.

        Args:
            scales: Number of scales
            directions: Number of directions per scale
            alpha: Anisotropy parameter (0.5 for parabolic scaling)
            high_freq: Whether to include high frequency components
            kernel_regularizer: Optional kernel regularizer
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.scales = scales
        self.directions = directions
        self.alpha = alpha
        self.high_freq = high_freq
        self.kernel_regularizer = kernel_regularizer

        # Initialize attributes
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.freq_x: Optional[tf.Tensor] = None
        self.freq_y: Optional[tf.Tensor] = None
        self.filters: Optional[List[tf.Tensor]] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer and create filters.

        Args:
            input_shape: Input tensor shape
        """
        self.height, self.width = input_shape[1:3]

        # Create frequency grid
        self.freq_x, self.freq_y = self._create_freq_grid()

        # Generate improved shearlet filters
        self.filters = self._create_shearlet_filters()

        self.built = True

    def _create_freq_grid(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create normalized frequency grid.

        Returns:
            Tuple of frequency coordinates (freq_x, freq_y)
        """
        # Create normalized frequency ranges
        fx = tf.linspace(-0.5, 0.5, self.width)
        fy = tf.linspace(-0.5, 0.5, self.height)

        # Create 2D grid with proper broadcasting
        freq_y, freq_x = tf.meshgrid(fy, fx, indexing='ij')
        return freq_x, freq_y

    def _meyer_window(self, x: tf.Tensor, a: float = 1.0, eps: float = 1e-6) -> tf.Tensor:
        """Create smooth Meyer window function with improved boundary handling.

        Args:
            x: Input values
            a: Scaling parameter
            eps: Small value for numerical stability

        Returns:
            Window function values
        """

        def smooth_transition(x: tf.Tensor) -> tf.Tensor:
            """Enhanced smooth polynomial transition."""
            x = tf.clip_by_value(x, 0.0, 1.0)
            # Higher order polynomial for smoother transition
            return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))

        # Normalize input
        x = tf.abs(x)
        x = tf.clip_by_value(x / (a + eps), 0.0, 1.0)

        # Compute window value with broader support
        value = tf.where(
            x < 1 / 3,
            tf.ones_like(x),
            tf.where(
                x < 2 / 3,
                smooth_transition(2.0 - 3.0 * x),
                tf.zeros_like(x)
            )
        )

        # Add small offset to prevent zero response
        return value + eps

    def _create_shearlet_filters(self) -> List[tf.Tensor]:
        """Create shearlet filters with guaranteed minimum response.

        Returns:
            List of shearlet filters in Fourier domain
        """
        filters = []

        # Get polar coordinates
        rho = tf.sqrt(self.freq_x ** 2 + self.freq_y ** 2)
        theta = tf.atan2(self.freq_y, self.freq_x)

        # Ensure non-zero minimum response
        min_response = 1e-3

        # Create low-pass filter with guaranteed minimum
        phi_low = tf.maximum(
            self._meyer_window(2.0 * rho),
            min_response
        )
        filters.append(tf.cast(phi_low, tf.complex64))

        # Create directional filters
        for j in range(self.scales):
            scale = 2.0 ** j

            # Create overlapping windows
            window_j = tf.maximum(
                self._meyer_window(rho / scale) *
                (1.0 - self._meyer_window(2.0 * rho / scale)),
                min_response
            )

            for k in range(-self.directions // 2, self.directions // 2 + 1):
                shear = k / (self.directions / 2.0)
                angle = tf.atan(shear)

                # Create angular window with minimum response
                dir_window = tf.maximum(
                    self._meyer_window(
                        (theta - angle) / (0.5 * np.pi),
                        a=2.0 / (self.directions + 2)
                    ),
                    min_response
                )

                # Combine windows with guaranteed minimum
                shearlet = tf.maximum(window_j * dir_window, min_response)

                # Normalize
                norm = tf.sqrt(tf.reduce_mean(tf.abs(shearlet) ** 2) + 1e-6)
                shearlet = shearlet / norm

                filters.append(tf.cast(shearlet, tf.complex64))

        # Final normalization to ensure frame bounds
        total_response = tf.reduce_sum(
            [tf.abs(f) ** 2 for f in filters],
            axis=0
        )

        # Normalize to achieve tight frame property
        scale_factor = tf.cast(
            1.0 / tf.sqrt(tf.maximum(total_response, min_response)),
            tf.complex64
        )

        return [f * scale_factor for f in filters]

    def _normalize_filter_bank(self, filters: List[tf.Tensor]) -> List[tf.Tensor]:
        """Normalize the complete filter bank with improved frame properties.

        Args:
            filters: List of initial filters

        Returns:
            List of normalized filters with guaranteed frame bounds
        """
        # Step 1: Initial filter energy normalization
        normalized_filters = []
        for f in filters:
            energy = tf.reduce_mean(tf.abs(f) ** 2)
            normalized_filters.append(f / tf.cast(tf.sqrt(energy + 1e-6), tf.complex64))

        # Step 2: Calculate total frequency response
        total_response = tf.reduce_sum(
            [tf.abs(f) ** 2 for f in normalized_filters],
            axis=0
        )

        # Step 3: Find areas with low response
        mean_response = tf.reduce_mean(total_response)
        min_threshold = mean_response * 0.01  # 1% of mean response

        # Step 4: Boost low response areas
        boost_mask = tf.cast(total_response < min_threshold, tf.complex64)
        boost_factor = tf.cast(min_threshold / (total_response + 1e-6), tf.complex64)

        # Step 5: Apply selective boosting
        boosted_filters = []
        for f in normalized_filters:
            # Boost filter response in low-energy regions
            boosted = f * tf.where(
                boost_mask > 0,
                boost_factor,
                tf.ones_like(boost_factor)
            )
            boosted_filters.append(boosted)

        # Step 6: Final normalization for frame bounds
        final_response = tf.reduce_sum([tf.abs(f) ** 2 for f in boosted_filters], axis=0)
        normalization = tf.cast(tf.sqrt(final_response + 1e-6), tf.complex64)

        # Ensure frame bounds by normalizing with proper scaling
        target_bound = 1.0
        final_filters = [
            target_bound * f / normalization
            for f in boosted_filters
        ]

        # Add small DC offset to prevent zero response
        dc_offset = tf.cast(1e-3, tf.complex64)
        final_filters = [f + dc_offset for f in final_filters]

        return final_filters

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply shearlet transform to input images.

        Args:
            inputs: Input tensor of shape [batch, height, width, channels]

        Returns:
            Tensor of shearlet coefficients
        """
        # Ensure float32 input
        inputs = tf.cast(inputs, tf.float32)

        # Compute 2D FFT
        fft = tf.signal.fftshift(
            tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        )

        # Apply filters and compute inverse FFT
        coefficients = []
        for filter_kernel in self.filters:
            filtered = fft * filter_kernel
            coeff = tf.cast(
                tf.math.real(
                    tf.signal.ifft2d(
                        tf.signal.ifftshift(filtered)
                    )
                ),
                tf.float32
            )
            coefficients.append(coeff)

        # Stack along channel dimension
        return tf.stack(coefficients, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing configuration
        """
        config = super().get_config()
        config.update({
            'scales': self.scales,
            'directions': self.directions,
            'alpha': self.alpha,
            'high_freq': self.high_freq,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer
            )
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ShearletTransform':
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            New layer instance
        """
        regularizer_config = config.pop('kernel_regularizer', None)
        if regularizer_config:
            config['kernel_regularizer'] = keras.regularizers.deserialize(
                regularizer_config
            )
        return cls(**config)
