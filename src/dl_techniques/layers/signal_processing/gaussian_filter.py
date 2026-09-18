"""
2D Gaussian blur using a depthwise convolution.

Implements a Gaussian filter for smoothing images and reducing high-frequency
noise. The filtering is achieved by convolving the input with a kernel derived
from the 2D Gaussian function G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2) /
(2*sigma^2)), normalized to sum to one. The kernel is applied via depthwise
convolution so each channel is filtered independently with the same kernel,
preventing color bleeding.

``sigma`` is the standard deviation of the Gaussian in PIXELS, per axis. The
kernel is the Gaussian sampled at integer pixel offsets around its center,
truncated to ``kernel_size`` and normalized to sum to one, so a larger ``sigma``
blurs more. Truncation narrows the realized blur once ``sigma`` exceeds about
``(kernel_size - 1) / 6``: a 5-tap kernel with ``sigma=2`` has a measured
standard deviation of 1.29 px, so pick ``kernel_size`` of at least
``6 * sigma + 1`` when the exact scale matters. ``padding="same"`` zero-pads, which darkens the border of a bright
image; ``padding="symmetric"`` mirrors the border instead and returns the same
output size as ``"same"``.

References:
    - Gonzalez, R. C., & Woods, R. E. "Digital Image Processing".
    - Canny, J. "A Computational Approach to Edge Detection".
      https://doi.org/10.1109/TPAMI.1986.4767851
"""

import numbers

import keras
from typing import Tuple, Union, List, Optional, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...utils.logger import logger
from ...utils.tensors import depthwise_gaussian_kernel
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

_WARNED = set()


def _warn_once(key: str, reason: str) -> None:
    """Log that mirrored padding fell back to zero padding, once per reason."""
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(
            f"padding='symmetric' fell back to zero 'same' padding because {reason}; "
            "the border is zero-padded for these calls."
        )


def symmetric_same_pad(
        inputs,
        kernel_size: Sequence[int],
        strides: Sequence[int],
        data_format: str = "channels_last",
):
    """Mirror-pad ``inputs`` so a ``'valid'`` convolution matches a ``'same'`` one.

    The pad widths are the ones ``'same'`` zero padding would use (total
    ``max((ceil(n / s) - 1) * s + k - n, 0)`` per axis, the smaller half first),
    so output size and sampling phase are unchanged and only the border values
    differ. Returns ``None`` (and logs a one-time warning, because the border is
    then zero-padded) when the widths cannot be used: a stride above 1 on an axis
    of unknown length, or a pad wider than its axis, which mirror padding cannot
    fill. The caller then falls back to ``'same'``.

    :param inputs: 4D tensor.
    :param kernel_size: ``(kernel_h, kernel_w)``.
    :param strides: ``(stride_h, stride_w)``.
    :param data_format: ``"channels_last"`` or ``"channels_first"``.
    :return: The padded tensor, or ``None``.
    """
    axes = (1, 2) if data_format == "channels_last" else (2, 3)
    pads = []
    for axis, k, stride in zip(axes, kernel_size, strides):
        size = inputs.shape[axis]
        if size is None:
            if stride != 1:
                _warn_once("dynamic-stride", "an axis has unknown length and a stride above 1")
                return None
            total = k - 1
        else:
            out = -(-size // stride)
            total = max((out - 1) * stride + k - size, 0)
            if total - total // 2 > size:
                _warn_once("pad-wider-than-axis", "the pad is wider than the axis it mirrors")
                return None
        pads.append((total // 2, total - total // 2))

    if data_format == "channels_last":
        pad_width = [[0, 0], list(pads[0]), list(pads[1]), [0, 0]]
    else:
        pad_width = [[0, 0], [0, 0], list(pads[0]), list(pads[1])]
    return keras.ops.pad(inputs, pad_width, mode="symmetric")


@register_dl_technique("dl_techniques.layers.signal_processing.gaussian_filter")
class GaussianFilter(keras.layers.Layer):
    """
    Apply Gaussian blur filter to input images via depthwise convolution.

    Creates a depthwise convolution with Gaussian kernel weights derived from
    G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2)), where sigma
    controls the blur spread. Each channel is processed independently using the
    same normalized kernel to preserve brightness and prevent channel mixing.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input [batch, H, W, C]                  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Depthwise Conv2D with Gaussian kernel   │
        │  (same kernel replicated per channel)    │
        │  kernel_size, strides, padding           │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output [batch, H', W', C]               │
        └──────────────────────────────────────────┘

    :param kernel_size: Height and width of the 2D Gaussian kernel.
    :type kernel_size: Tuple[int, int]
    :param strides: Strides of the convolution along height and width.
    :type strides: Union[Tuple[int, int], List[int]]
    :param sigma: Standard deviation of the Gaussian in pixels, before
        truncation to ``kernel_size`` (see the module note). If a single
        value, same sigma for both dimensions. If a tuple, (sigma_h, sigma_w).
        If -1, None or any non-positive number, a one-pixel standard deviation
        ``(1.0, 1.0)`` is used. Tuple entries must be positive.
    :type sigma: Union[float, Tuple[float, float]]
    :param padding: ``"valid"``, ``"same"`` (zero padding) or ``"symmetric"``
        (mirrored border, output size as for ``"same"``); case-insensitive.
        ``"symmetric"`` needs static spatial dims when a stride exceeds 1, and
        otherwise falls back to zero padding; the image must be at least as
        large as the padding it needs.
    :type padding: str
    :param data_format: Either "channels_last" or "channels_first".
    :type data_format: Optional[str]
    :param trainable: If True allow the weights to change, otherwise static.
    :type trainable: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (5, 5),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            sigma: Union[float, Tuple[float, float]] = -1,
            padding: str = "same",
            data_format: Optional[str] = None,
            trainable: bool = False,
            **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be length 2")
        self.kernel_size = kernel_size

        if not isinstance(strides, (tuple, list)) or len(strides) != 2:
            raise ValueError("strides must be a tuple/list of length 2")
        # ``tuple``: a deserialized list becomes a Keras ``TrackedList``, which
        # ``tuple + list`` arithmetic in later code cannot mix with a tuple.
        self.strides = tuple(strides)

        self.padding = padding.lower()
        if self.padding not in {"valid", "same", "symmetric"}:
            raise ValueError(
                f"padding must be 'valid', 'same' or 'symmetric', got {padding}"
            )

        self.data_format = keras.config.image_data_format() if data_format is None else data_format
        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"data_format must be 'channels_first' or 'channels_last', got {data_format}")

        if isinstance(sigma, bool):
            raise ValueError(f"Invalid sigma value: {sigma}")
        if (sigma is None or
                (isinstance(sigma, numbers.Real) and sigma <= 0)):
            # No sigma given: a one-pixel standard deviation.
            self.sigma = (1.0, 1.0)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            self.sigma = (float(sigma[0]), float(sigma[1]))
            if min(self.sigma) <= 0.0:
                raise ValueError(f"sigma entries must be positive, got {sigma}")
        elif isinstance(sigma, numbers.Real):
            self.sigma = (float(sigma), float(sigma))
        else:
            raise ValueError(f"Invalid sigma value: {sigma}")

        # Will be set in build()
        self.kernel = None

        logger.info(
            f"kernel_size: {self.kernel_size}, "
            f"padding: {self.padding}, "
            f"sigma: {self.sigma}"
        )

    def build(self, input_shape):
        """Build the Gaussian kernel weights based on input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        if self.data_format == "channels_last":
            channels = input_shape[-1]
        else:  # channels_first
            channels = input_shape[1]

        if channels is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                "The 'channels' argument is 'None'."
            )

        kernel_np = depthwise_gaussian_kernel(
            channels=channels,
            kernel_size=self.kernel_size,
            # DECISION plan-2026-09-18T211047-6ac2fa02/D-001: ``sigma`` is a
            # pixel std and is converted here. Do NOT pass it straight through:
            # the util reads its argument as a half-extent in stds, which made a
            # larger sigma blur LESS. The derived default (1.0, 1.0) reproduces
            # the old derived kernel exactly.
            # The util samples ``linspace(-nsig, nsig, k)`` in units of the
            # standard deviation, i.e. nsig is the half-extent in stds.
            nsig=(
                (self.kernel_size[0] - 1) / (2.0 * self.sigma[0]),
                (self.kernel_size[1] - 1) / (2.0 * self.sigma[1]),
            ),
            dtype=self.compute_dtype,
        )

        # Use add_weight to create a properly managed state variable.
        # This ensures the kernel can be used across different tf.function scopes.
        self.kernel = self.add_weight(
            name="gaussian_kernel",
            shape=kernel_np.shape,
            dtype=self.compute_dtype,
            initializer=keras.initializers.Constant(kernel_np),
            trainable=self.trainable,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Apply the Gaussian filter to the input tensor.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (unused).
        :type training: Optional[bool]
        :return: Filtered tensor with the same shape as the input.
        :rtype: keras.KerasTensor
        """
        padding = self.padding
        if padding == "symmetric":
            padded = symmetric_same_pad(
                inputs, self.kernel_size, self.strides, self.data_format
            )
            if padded is None:
                padding = "same"
            else:
                inputs, padding = padded, "valid"

        outputs = keras.ops.nn.depthwise_conv(
            inputs=inputs,
            kernel=self.kernel,
            strides=self.strides,
            padding=padding,
            data_format=self.data_format
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """Compute output spatial dims from strides/padding/kernel_size."""
        if self.data_format == "channels_last":
            batch, h, w, c = input_shape
        else:
            batch, c, h, w = input_shape
        if self.padding in ("same", "symmetric"):
            out_h = (h + self.strides[0] - 1) // self.strides[0] if h is not None else None
            out_w = (w + self.strides[1] - 1) // self.strides[1] if w is not None else None
        else:
            out_h = (h - self.kernel_size[0]) // self.strides[0] + 1 if h is not None else None
            out_w = (w - self.kernel_size[1]) // self.strides[1] + 1 if w is not None else None
        if self.data_format == "channels_last":
            return (batch, out_h, out_w, c)
        return (batch, c, out_h, out_w)

    def get_config(self):
        """Return the configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "sigma": self.sigma,
            "padding": self.padding,
            "data_format": self.data_format
        })
        return config

# ---------------------------------------------------------------------
