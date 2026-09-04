"""
``FFTLayer`` and ``IFFTLayer`` move real-valued spatial feature maps to and
from the 2D frequency domain, used by the Fourier token mixer in PW-FNet.

``FFTLayer`` applies a 2D FFT to a real input ``[B, H, W, C]`` and returns the
real and imaginary spectra concatenated along the channel axis, shape
``[B, H, W, 2*C]``. ``IFFTLayer`` inverts that representation, taking
``[B, H, W, 2*C]`` back to ``[B, H, W, C]`` (the real part of the inverse
transform). Both are stateless and built on ``keras.ops.fft2`` / ``ifft2``,
which operate on the last two axes, so inputs are transposed to
channels-first for the transform and back afterwards.

Both layers run the transform in a float32 island regardless of the compute
dtype policy: neither `fft2` nor `ifft2` has a float16 kernel.
"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.signal_processing.fft_layers")
class FFTLayer(keras.layers.Layer):
    """
    Apply 2D Fast Fourier Transform and output concatenated real/imaginary parts.

    Transforms real-valued spatial domain features into the frequency domain via
    2D FFT, producing a single real-valued tensor where the real and imaginary
    components are concatenated along the channel axis. The output has shape
    [batch, H, W, 2*C] with float32 dtype. This is a core component of the
    Fourier-based token mixer in PW-FNet.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────────┐
        │  Input [batch, H, W, C]               │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  Transpose: [batch, C, H, W]          │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  Create (real, zeros) complex tuple   │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  FFT2D along spatial dims             │
        │  ──▶ (fft_real, fft_imag)             │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  Transpose both: [batch, H, W, C]     │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  Concatenate [real, imag] along ch.   │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  Output [batch, H, W, 2*C] (float32)  │
        └───────────────────────────────────────┘

    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the FFT layer.

        :param kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply 2D FFT to input tensor.

        :param inputs: Input tensor of shape (batch, height, width, channels).
        :type inputs: keras.KerasTensor
        :return: Real tensor of shape (batch, height, width, 2*channels)
            representing concatenated real and imaginary frequency components.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-054: run fft2 in a float32
        # island, not at compute_dtype -- TensorFlow ships no float16 kernel for it. See decisions.md.
        x_permuted = keras.ops.transpose(
            keras.ops.cast(inputs, "float32"), [0, 3, 1, 2]
        )

        real_part = x_permuted
        imag_part = keras.ops.zeros_like(real_part)

        fft_real, fft_imag = keras.ops.fft2((real_part, imag_part))

        fft_real_permuted = keras.ops.transpose(fft_real, [0, 2, 3, 1])
        fft_imag_permuted = keras.ops.transpose(fft_imag, [0, 2, 3, 1])

        return keras.ops.cast(
            keras.ops.concatenate(
                [fft_real_permuted, fft_imag_permuted], axis=-1
            ),
            self.compute_dtype,
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (channels are doubled).

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        batch, height, width, channels = input_shape
        if channels is not None:
            output_channels = channels * 2
        else:
            output_channels = None
        return batch, height, width, output_channels

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.signal_processing.fft_layers")
class IFFTLayer(keras.layers.Layer):
    """
    Apply 2D Inverse FFT to concatenated real/imaginary parts.

    Transforms frequency domain features (represented as a real tensor with
    concatenated real and imaginary parts) back into the spatial domain. It
    performs the inverse operation of ``FFTLayer``, taking input of shape
    [batch, H, W, 2*C] and producing output of shape [batch, H, W, C].

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────┐
        │  Input [batch, H, W, 2*C] (float32)     │
        └──────────────────┬──────────────────────┘
                           ▼
        ┌─────────────────────────────────────────┐
        │  Split into (real, imag) along channels │
        └──────────────────┬──────────────────────┘
                           ▼
        ┌─────────────────────────────────────────┐
        │  Transpose both: [batch, C, H, W]       │
        └──────────────────┬──────────────────────┘
                           ▼
        ┌─────────────────────────────────────────┐
        │  IFFT2D on (real, imag) tuple           │
        │  ──▶ (ifft_real, ifft_imag)             │
        └──────────────────┬──────────────────────┘
                           ▼
        ┌─────────────────────────────────────────┐
        │  Take ifft_real, transpose back         │
        │  [batch, H, W, C]                       │
        └──────────────────┬──────────────────────┘
                           ▼
        ┌─────────────────────────────────────────┐
        │  Output [batch, H, W, C] (float32)      │
        └─────────────────────────────────────────┘

    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the IFFT layer.

        :param kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply 2D IFFT to input tensor and extract real part.

        :param inputs: Real tensor with concatenated real/imag parts, shape
            (batch, height, width, 2*channels).
        :type inputs: keras.KerasTensor
        :return: Real tensor of shape (batch, height, width, channels) in
            spatial domain.
        :rtype: keras.KerasTensor
        """
        # Same float32 island as FFTLayer.call -- ifft2 has no float16 kernel either. See decisions.md D-054.
        real_part, imag_part = keras.ops.split(
            keras.ops.cast(inputs, "float32"), 2, axis=-1
        )

        real_permuted = keras.ops.transpose(real_part, [0, 3, 1, 2])
        imag_permuted = keras.ops.transpose(imag_part, [0, 3, 1, 2])

        ifft_real, _ = keras.ops.ifft2((real_permuted, imag_permuted))

        ifft_permuted = keras.ops.transpose(ifft_real, [0, 2, 3, 1])

        return keras.ops.cast(ifft_permuted, self.compute_dtype)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (channels are halved).

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        batch, height, width, channels = input_shape
        if channels is not None:
            if channels % 2 != 0:
                raise ValueError(
                    f"Input channels for IFFTLayer must be even, got {channels}"
                )
            output_channels = channels // 2
        else:
            output_channels = None
        return batch, height, width, output_channels

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        return config

# ---------------------------------------------------------------------
