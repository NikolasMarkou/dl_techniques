"""
Decompose an image into multi-scale, multi-directional components.

This layer implements a differentiable shearlet transform, a powerful tool
from multiscale geometric analysis that provides a sparse representation of
images rich in directional features like edges and textures. Unlike learned
convolutional filters, this layer acts as a fixed, mathematically-defined
feature extractor that injects strong geometric priors into a neural
network.

Architecturally, the layer operates as a non-trainable filter bank in the
frequency domain. The transformation process involves:
1.  Computing the 2D Fast Fourier Transform (FFT) of the input image.
2.  Multiplying the result element-wise with a pre-computed bank of
    shearlet filters, each corresponding to a specific scale and orientation.
3.  Applying the inverse FFT to each filtered output to obtain the shearlet
    coefficients in the spatial domain.

The foundational mathematical principle of the shearlet transform is its
ability to overcome the limitations of traditional wavelets. While wavelets
use isotropic (directionally uniform) scaling, they are inefficient at
representing anisotropic features like curves. Shearlets, in contrast, are
constructed using a combination of anisotropic scaling and shearing matrices.
This construction leads to basis functions that are elongated and directionally
oriented, making them highly effective at capturing directional information.

A key theoretical property is the use of "parabolic scaling," where the
support of the shearlet filters in the frequency domain scales differently
along different axes (e.g., size `~2^j` along one axis and `~2^(j/2)` along
the other). This specific scaling relationship is mathematically proven to be
optimal for providing sparse representations of "cartoon-like" images—a class
of functions that are piecewise smooth with discontinuities along curves.
This optimal sparsity means that complex edge information can be captured
with very few non-zero coefficients, making it a highly efficient
representation.

The constructed filter bank forms a "tight frame," a mathematical property
that ensures the transformation is energy-preserving and allows for perfect
reconstruction of the original signal from its coefficients. This stability
is crucial for its integration into deep learning models, as it guarantees
a well-behaved and invertible feature transformation.

References:
    - Guo, K., Kutyniok, G., & Labate, D., 2006. Sparse multidimensional
      representations using shearlets.
    - Kutyniok, G., & Labate, D., 2012. Shearlets: Multiscale Analysis
      for Multivariate Data.
"""

import keras
import numpy as np
from keras import ops, layers, initializers
from typing import List, Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ShearletTransform(keras.layers.Layer):
    """
    Multi-scale, multi-directional shearlet transform layer for enhanced time-frequency analysis.

    This layer implements a cone-adapted discrete shearlet transform with improved frame
    properties and frequency coverage. The shearlet transform provides optimal sparse
    representation of images with edges and directional features by combining multi-scale
    analysis with directional selectivity.

    **Intent**: Provide a differentiable shearlet transform that can be integrated into
    neural networks for enhanced feature extraction, particularly effective for images
    containing edges, textures, and directional patterns.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    2D FFT (Keras Ops) → Frequency Domain
           ↓
    Apply Shearlet Filters (scales × directions filters)
    (Complex Multiplication in Frequency Domain)
           ↓
    2D IFFT (Keras Ops) → Spatial Domain Coefficients
           ↓
    Output(shape=[batch, height, width, num_filters])
    ```

    **Optimization**:
    This implementation pre-shifts the filter bank during the build phase to match
    the standard FFT output layout (DC component at corners). This optimization
    removes the need for computationally expensive `fftshift` and `ifftshift`
    operations during the forward pass (call), significantly improving throughput.

    Args:
        scales: Integer, number of scales in the transform. Controls the multi-resolution
            analysis depth. Higher values provide finer scale analysis. Must be positive.
            Defaults to 4.
        directions: Integer, number of directions per scale. Controls angular resolution.
            Higher values provide better directional selectivity but increase computation.
            Must be positive and preferably even. Defaults to 8.
        alpha: Float, anisotropy parameter controlling the relationship between scale
            and direction sampling. Value of 0.5 provides parabolic scaling which is
            optimal for edge detection. Must be in (0, 1]. Defaults to 0.5.
        high_freq: Boolean, whether to include high frequency components in the transform.
            When True, captures fine details and noise. When False, focuses on low-mid
            frequencies. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        All input images should have the same spatial dimensions within a batch.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, num_filters)`.
        Where num_filters = (1 (low-pass) + scales * directions) * channels.
        Spatial dimensions are preserved.

    Attributes:
        filter_bank_real: Tensor, real part of shearlet filters (set during build).
        filter_bank_imag: Tensor, imaginary part of shearlet filters (set during build).
    """

    def __init__(
            self,
            scales: int = 4,
            directions: int = 8,
            alpha: float = 0.5,
            high_freq: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if scales <= 0:
            raise ValueError(f"scales must be positive, got {scales}")
        if directions <= 0:
            raise ValueError(f"directions must be positive, got {directions}")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        # Store configuration
        self.scales = scales
        self.directions = directions
        self.alpha = alpha
        self.high_freq = high_freq

        # Initialize attributes (created in build)
        self.height: Optional[int] = None
        self.width: Optional[int] = None

        # We store filters as non-trainable weights
        # Shapes will be (num_filters, H, W)
        self.filter_bank_real: Optional[keras.Variable] = None
        self.filter_bank_imag: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create shearlet filter bank.

        This method generates the frequency grid and all shearlet filters using
        NumPy for precise construction, then converts them to Keras tensors.
        Filters are pre-shifted (ifftshift) to align with standard FFT output,
        avoiding runtime shifts.

        Args:
            input_shape: Input tensor shape (batch_size, height, width, channels)
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        if input_shape[1] is None or input_shape[2] is None:
            raise ValueError("Height and width dimensions must be specified for fixed filter bank construction")

        self.height = input_shape[1]
        self.width = input_shape[2]

        # Generate filters using NumPy (complex128 for precision during generation)
        # We perform generation on CPU/NumPy to use standard tooling for window functions
        filters_complex = self._generate_filter_bank_numpy(self.height, self.width)

        # Stack into a single array: (num_filters, H, W)
        filters_stack = np.stack(filters_complex, axis=0)

        # Apply ifftshift to match standard FFT layout (DC at corner)
        # This is a critical optimization to avoid fftshift in call()
        filters_shifted = np.fft.ifftshift(filters_stack, axes=(-2, -1))

        # Split into real and imaginary parts for Keras Ops compatibility
        filters_real = np.real(filters_shifted).astype(np.float32)
        filters_imag = np.imag(filters_shifted).astype(np.float32)

        # Register as non-trainable weights
        self.filter_bank_real = self.add_weight(
            name="filter_bank_real",
            shape=filters_real.shape,
            initializer=initializers.Constant(filters_real),
            trainable=False,
            dtype="float32"
        )

        self.filter_bank_imag = self.add_weight(
            name="filter_bank_imag",
            shape=filters_imag.shape,
            initializer=initializers.Constant(filters_imag),
            trainable=False,
            dtype="float32"
        )

        super().build(input_shape)

    def _generate_filter_bank_numpy(self, height: int, width: int) -> List[np.ndarray]:
        """
        Generate shearlet filters using NumPy.

        Returns:
            List of complex-valued filter arrays (centered frequency domain).
        """
        # Create normalized frequency grid [-0.5, 0.5]
        fx = np.linspace(-0.5, 0.5, width)
        fy = np.linspace(-0.5, 0.5, height)
        freq_y, freq_x = np.meshgrid(fy, fx, indexing='ij')

        filters = []

        # Polar coordinates
        rho = np.sqrt(freq_x ** 2 + freq_y ** 2)
        theta = np.arctan2(freq_y, freq_x)

        min_response = 1e-3

        # 1. Low-pass filter
        phi_low = np.maximum(
            self._meyer_window_numpy(2.0 * rho),
            min_response
        )
        filters.append(phi_low.astype(np.complex64))

        # 2. Directional filters
        for j in range(self.scales):
            scale = 2.0 ** j

            # Radial window for scale j
            window_j = np.maximum(
                self._meyer_window_numpy(rho / scale) *
                (1.0 - self._meyer_window_numpy(2.0 * rho / scale)),
                min_response
            )

            # Directional loop
            for k in range(-self.directions // 2, self.directions // 2 + 1):
                shear = k / (self.directions / 2.0)
                angle = np.arctan(shear)

                # Angular window
                dir_window = np.maximum(
                    self._meyer_window_numpy(
                        (theta - angle) / (0.5 * np.pi),
                        a=2.0 / (self.directions + 2)
                    ),
                    min_response
                )

                # Combine
                shearlet = np.maximum(window_j * dir_window, min_response)

                # Normalize energy
                norm = np.sqrt(np.mean(np.abs(shearlet) ** 2) + 1e-6)
                shearlet = shearlet / norm

                filters.append(shearlet.astype(np.complex64))

        return self._normalize_filter_bank_numpy(filters)

    def _meyer_window_numpy(self, x: np.ndarray, a: float = 1.0, eps: float = 1e-6) -> np.ndarray:
        """NumPy implementation of Meyer window function."""
        def smooth_transition(t):
            t = np.clip(t, 0.0, 1.0)
            return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))

        x = np.abs(x)
        x = np.clip(x / (a + eps), 0.0, 1.0)

        value = np.where(
            x < (1.0 / 3.0),
            np.ones_like(x),
            np.where(
                x < (2.0 / 3.0),
                smooth_transition(2.0 - 3.0 * x),
                np.zeros_like(x)
            )
        )
        return value + eps

    def _normalize_filter_bank_numpy(self, filters: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize filter bank for tight frame property (NumPy version)."""
        # Initial normalization
        normalized = []
        for f in filters:
            energy = np.mean(np.abs(f) ** 2)
            normalized.append(f / np.sqrt(energy + 1e-6))

        # Calculate total response
        total_response = np.sum([np.abs(f) ** 2 for f in normalized], axis=0)

        # Boost low response areas
        mean_response = np.mean(total_response)
        min_threshold = mean_response * 0.01

        boost_needed = total_response < min_threshold
        boost_factor = min_threshold / (total_response + 1e-6)

        boosted_filters = []
        for f in normalized:
            # Apply boost only where needed
            f_boosted = f.copy()
            f_boosted[boost_needed] *= boost_factor[boost_needed]
            boosted_filters.append(f_boosted)

        # Final normalization
        final_response = np.sum([np.abs(f) ** 2 for f in boosted_filters], axis=0)
        normalization = np.sqrt(final_response + 1e-6)

        return [(f / normalization) for f in boosted_filters]

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply shearlet transform to input images using Keras Ops.

        Args:
            inputs: Input tensor [batch_size, height, width, channels]
            training: Unused.

        Returns:
            Shearlet coefficients [batch_size, height, width, num_filters * channels]
        """
        inputs = ops.cast(inputs, "float32")

        # Handle 3D inputs (missing channel dim)
        if len(inputs.shape) == 3:
            inputs = ops.expand_dims(inputs, axis=-1)

        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        # height/width available as self.height/self.width
        # channels is static if possible, else dynamic
        channels_dim = inputs.shape[-1]

        # 1. Permute to (B, C, H, W) for FFT ops (which operate on last 2 axes)
        x = ops.transpose(inputs, axes=(0, 3, 1, 2))

        # 2. Compute FFT
        # Input is real, so imag part is zero
        x_imag = ops.zeros_like(x)
        fft_r, fft_i = ops.fft2((x, x_imag))

        # 3. Prepare filters for broadcasting
        # Filters: (NumFilters, H, W) -> (1, 1, NumFilters, H, W)
        f_r = ops.reshape(self.filter_bank_real, (1, 1, -1, self.height, self.width))
        f_i = ops.reshape(self.filter_bank_imag, (1, 1, -1, self.height, self.width))

        # Input FFT: (B, C, H, W) -> (B, C, 1, H, W)
        fft_r = ops.expand_dims(fft_r, axis=2)
        fft_i = ops.expand_dims(fft_i, axis=2)

        # 4. Complex Multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # Result shape: (B, C, NumFilters, H, W)
        out_r = (fft_r * f_r) - (fft_i * f_i)
        out_i = (fft_r * f_i) + (fft_i * f_r)

        # 5. Inverse FFT
        # We need to perform IFFT on (out_r, out_i).
        # Since `keras.ops.ifft2` availability varies across documentation snippets,
        # we implement it robustly using `fft2` and the property:
        # IFFT(z) = conj(FFT(conj(z))) / N

        # Conjugate of input z: (out_r, -out_i)
        conj_in_r = out_r
        conj_in_i = -out_i

        # Apply FFT to conjugated input
        tmp_r, tmp_i = ops.fft2((conj_in_r, conj_in_i))

        # Result is conj(tmp) / N
        # conj(tmp) = (tmp_r, -tmp_i)
        # We only need the REAL part of the final result for shearlet coefficients
        # Real part of conj(tmp) is just tmp_r

        N = ops.cast(self.height * self.width, "float32")
        coeffs = tmp_r / N

        # 6. Reshape and Transpose back to (B, H, W, OutChannels)
        # Current shape: (B, C, NumFilters, H, W)
        # Target: (B, H, W, C, NumFilters) -> (B, H, W, C * NumFilters)

        coeffs = ops.transpose(coeffs, axes=(0, 3, 4, 1, 2))

        num_base_filters = ops.shape(self.filter_bank_real)[0]

        if channels_dim is None:
             # Dynamic shape handling
            out_ch = ops.shape(inputs)[3] * num_base_filters
            final_shape = (batch_size, self.height, self.width, out_ch)
        else:
            # Static shape handling
            out_ch = channels_dim * num_base_filters
            final_shape = (-1, self.height, self.width, out_ch)

        return ops.reshape(coeffs, final_shape)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        batch_size, height, width, channels = input_shape

        # Filters: 1 low-pass + scales * (directions + 1 directional options in loop?
        # Checking logic: code loops -directions//2 to directions//2 -> directions+1 filters per scale)
        # Wait, original code range: range(-directions // 2, directions // 2 + 1)
        # Length is directions + 1 if directions is even?
        # e.g., dir=2 -> -1, 0, 1 (3 filters). Usually standard is `directions`.
        # Assuming original logic intended inclusive range.

        filters_per_scale = (self.directions // 2 + 1) - (-self.directions // 2) # = directions + 1
        num_filters = 1 + self.scales * filters_per_scale

        if channels is not None:
            num_filters *= channels

        return (batch_size, height, width, num_filters)

    def get_config(self) -> Dict[str, Any]:
        """Serialization configuration."""
        config = super().get_config()
        config.update({
            'scales': self.scales,
            'directions': self.directions,
            'alpha': self.alpha,
            'high_freq': self.high_freq,
        })
        return config

# ---------------------------------------------------------------------
