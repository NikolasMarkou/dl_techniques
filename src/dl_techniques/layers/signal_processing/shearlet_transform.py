"""Fixed multi-scale, multi-directional shearlet transform, built by
:class:`ShearletTransform`.

Wavelets scale isotropically and so represent curved edges inefficiently.
Shearlets replace isotropic scaling with a combination of anisotropic
scaling and shearing, giving elongated, directionally oriented basis
functions that capture edges with far fewer non-zero coefficients. This
layer is a fixed, non-trainable filter bank, not a learned convolution: it
computes the 2D FFT of the input, multiplies element-wise by a
pre-computed bank of shearlet filters (one per scale and orientation),
and inverse-transforms each result back to the spatial domain. The filter
bank forms a tight frame, so the transform is energy-preserving and
invertible.

The filter bank is fixed at build time from the input's height and width,
so this layer requires a statically known spatial shape.

References:
    - Guo, K., Kutyniok, G., Labate, D., 2006. Sparse Multidimensional
      Representations Using Shearlets.
    - Kutyniok, G., Labate, D., 2012. Shearlets: Multiscale Analysis for
      Multivariate Data.
"""

import keras
import numpy as np
from keras import ops, initializers
from typing import List, Tuple, Optional, Dict, Any

from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.signal_processing.shearlet_transform")
class ShearletTransform(keras.layers.Layer):
    """Cone-adapted discrete shearlet transform layer.

    Computes the 2D FFT of the input, multiplies it with a pre-computed
    bank of shearlet filters (each tuned to a specific scale and
    orientation via parabolic scaling and a Meyer window), and applies
    the inverse FFT to obtain spatial-domain coefficients. The filter
    bank is pre-shifted (``ifftshift``) during build to match standard
    FFT layout, avoiding a runtime shift.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────┐
        │  Input [B, H, W, C]                     │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Permute to [B, C, H, W]                │
        │  2D FFT (real → complex)                │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Complex Multiply with Filter Bank      │
        │  [1+S*(D+1) filters, H, W]              │
        │  → [B, C, NumFilters, H, W]             │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Inverse FFT via conj(FFT(conj(·)))/N   │
        │  Take real part → coefficients          │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Reshape to [B, H, W, C * NumFilters]   │
        └─────────────────────────────────────────┘

    :param scales: Number of scales in the transform. Controls multi-resolution
        analysis depth. Must be positive. Defaults to 4.
    :type scales: int
    :param directions: Number of directions per scale. Controls angular resolution.
        Must be positive and preferably even. Defaults to 8.
    :type directions: int
    :param alpha: Anisotropy parameter controlling scale-direction sampling
        relationship. Value of 0.5 provides parabolic scaling optimal for edge
        detection. Must be in (0, 1]. Defaults to 0.5.
    :type alpha: float
    :param high_freq: Whether to include high frequency components. When True,
        captures fine details and noise. Defaults to True.
    :type high_freq: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
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

        if scales <= 0:
            raise ValueError(f"scales must be positive, got {scales}")
        if directions <= 0:
            raise ValueError(f"directions must be positive, got {directions}")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.scales = scales
        self.directions = directions
        self.alpha = alpha
        self.high_freq = high_freq

        self.height: Optional[int] = None
        self.width: Optional[int] = None

        self.filter_bank_real: Optional[keras.Variable] = None
        self.filter_bank_imag: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create shearlet filter bank.

        Generates the frequency grid and all shearlet filters using NumPy for
        precise construction, then converts them to Keras tensors. Filters are
        pre-shifted (ifftshift) to align with standard FFT output, avoiding
        runtime shifts.

        :param input_shape: Input tensor shape (batch_size, height, width, channels).
        :type input_shape: tuple
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        if input_shape[1] is None or input_shape[2] is None:
            raise ValueError("Height and width dimensions must be specified for fixed filter bank construction")

        self.height = input_shape[1]
        self.width = input_shape[2]

        filters_complex = self._generate_filter_bank_numpy(self.height, self.width)

        filters_stack = np.stack(filters_complex, axis=0)

        # ifftshift matches standard FFT layout (DC at corner), avoiding fftshift in call().
        filters_shifted = np.fft.ifftshift(filters_stack, axes=(-2, -1))

        filters_real = np.real(filters_shifted).astype(np.float32)
        filters_imag = np.imag(filters_shifted).astype(np.float32)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-054: keep autocast=False here;
        # without it Keras casts these to float16 under mixed_float16 and fft2 has
        # no float16 kernel. Only the result may return to compute_dtype. See decisions.md.
        self.filter_bank_real = self.add_weight(
            name="filter_bank_real",
            shape=filters_real.shape,
            initializer=initializers.Constant(filters_real),
            trainable=False,
            dtype="float32",
            autocast=False,
        )

        self.filter_bank_imag = self.add_weight(
            name="filter_bank_imag",
            shape=filters_imag.shape,
            initializer=initializers.Constant(filters_imag),
            trainable=False,
            dtype="float32",
            autocast=False,
        )

        super().build(input_shape)

    def _generate_filter_bank_numpy(self, height: int, width: int) -> List[np.ndarray]:
        """
        Generate shearlet filters using NumPy.

        :return: List of complex-valued filter arrays (centered frequency domain).
        :rtype: list[numpy.ndarray]
        """
        fx = np.linspace(-0.5, 0.5, width)
        fy = np.linspace(-0.5, 0.5, height)
        freq_y, freq_x = np.meshgrid(fy, fx, indexing='ij')

        filters = []

        rho = np.sqrt(freq_x ** 2 + freq_y ** 2)
        theta = np.arctan2(freq_y, freq_x)

        min_response = 1e-3

        phi_low = np.maximum(
            self._meyer_window_numpy(2.0 * rho),
            min_response
        )
        filters.append(phi_low.astype(np.complex64))

        for j in range(self.scales):
            scale = 2.0 ** j

            window_j = np.maximum(
                self._meyer_window_numpy(rho / scale) *
                (1.0 - self._meyer_window_numpy(2.0 * rho / scale)),
                min_response
            )

            for k in range(-self.directions // 2, self.directions // 2 + 1):
                shear = k / (self.directions / 2.0)
                angle = np.arctan(shear)

                dir_window = np.maximum(
                    self._meyer_window_numpy(
                        (theta - angle) / (0.5 * np.pi),
                        a=2.0 / (self.directions + 2)
                    ),
                    min_response
                )

                shearlet = np.maximum(window_j * dir_window, min_response)

                norm = np.sqrt(np.mean(np.abs(shearlet) ** 2) + 1e-6)
                shearlet = shearlet / norm

                filters.append(shearlet.astype(np.complex64))

        return self._normalize_filter_bank_numpy(filters)

    def _meyer_window_numpy(
            self,
            x: np.ndarray, a: float = 1.0,
            eps: float = 1e-6
    ) -> np.ndarray:
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

    def _normalize_filter_bank_numpy(
            self,
            filters: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Normalize filter bank for tight frame property (NumPy version)."""
        normalized = []
        for f in filters:
            energy = np.mean(np.abs(f) ** 2)
            normalized.append(f / np.sqrt(energy + 1e-6))

        total_response = np.sum([np.abs(f) ** 2 for f in normalized], axis=0)

        mean_response = np.mean(total_response)
        min_threshold = mean_response * 0.01

        boost_needed = total_response < min_threshold
        boost_factor = min_threshold / (total_response + 1e-6)

        boosted_filters = []
        for f in normalized:
            f_boosted = f.copy()
            f_boosted[boost_needed] *= boost_factor[boost_needed]
            boosted_filters.append(f_boosted)

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

        :param inputs: Input tensor of shape ``[batch_size, height, width, channels]``.
        :type inputs: keras.KerasTensor
        :param training: Unused, kept for interface consistency.
        :type training: bool or None
        :return: Shearlet coefficients of shape ``[batch_size, height, width, num_filters * channels]``.
        :rtype: keras.KerasTensor
        """
        inputs = ops.cast(inputs, "float32")

        if len(inputs.shape) == 3:
            inputs = ops.expand_dims(inputs, axis=-1)

        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        channels_dim = inputs.shape[-1]

        # FFT ops operate on the last 2 axes, so permute channels forward.
        x = ops.transpose(inputs, axes=(0, 3, 1, 2))

        # Input is real, so imag part is zero.
        x_imag = ops.zeros_like(x)
        fft_r, fft_i = ops.fft2((x, x_imag))

        f_r = ops.reshape(self.filter_bank_real, (1, 1, -1, self.height, self.width))
        f_i = ops.reshape(self.filter_bank_imag, (1, 1, -1, self.height, self.width))

        fft_r = ops.expand_dims(fft_r, axis=2)
        fft_i = ops.expand_dims(fft_i, axis=2)

        out_r = (fft_r * f_r) - (fft_i * f_i)
        out_i = (fft_r * f_i) + (fft_i * f_r)

        # ifft2 availability varies; use IFFT(z) = conj(FFT(conj(z))) / N via fft2 instead.

        conj_in_r = out_r
        conj_in_i = -out_i

        tmp_r, tmp_i = ops.fft2((conj_in_r, conj_in_i))


        N = ops.cast(self.height * self.width, "float32")
        coeffs = tmp_r / N


        coeffs = ops.transpose(coeffs, axes=(0, 3, 4, 1, 2))

        num_base_filters = ops.shape(self.filter_bank_real)[0]

        if channels_dim is None:
            out_ch = ops.shape(inputs)[3] * num_base_filters
            final_shape = (batch_size, self.height, self.width, out_ch)
        else:
            out_ch = channels_dim * num_base_filters
            final_shape = (-1, self.height, self.width, out_ch)

        # Spectral island is float32 (see D-054 in build); cast to compute_dtype for the caller.
        return ops.cast(ops.reshape(coeffs, final_shape), self.compute_dtype)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        batch_size, height, width, channels = input_shape

        # 1 low-pass filter, plus directions+1 directional filters per scale:
        # the inclusive range -directions//2..directions//2 yields directions+1 filters.
        filters_per_scale = (self.directions // 2 + 1) - (-self.directions // 2)
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
