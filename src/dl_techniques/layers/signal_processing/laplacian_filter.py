"""Laplacian edge filters and an invertible Laplacian pyramid level.

``LaplacianFilter`` finds edges with a Difference of Gaussians: it blurs the
input and subtracts the blur from the original,

    laplacian = scale_factor * (blurred - input)

which gives a bright spot a negative response. ``AdvancedLaplacianFilter``
reaches the same operator three ways: that Difference of Gaussians, a
convolution with a Laplacian of Gaussian kernel, or a convolution with a small
fixed stencil. ``LaplacianPyramidLevel`` does something different. It splits an
image into a downsampled low band and a same-resolution high band, and adding
the two back reconstructs the input to float precision, because the high band
is defined as the difference rather than estimated.

No kernel here is learned; each one comes from a formula. The blur inside
``LaplacianPyramidLevel`` can be made trainable and is not by default. Both edge
filters hold the blur at stride ``(1, 1)`` so the subtraction lines up, so
``strides`` changes the output shape only for the ``'log'`` and ``'kernel'``
methods.
"""

import keras
import numpy as np
from typing import Tuple, Union, List, Optional, Sequence, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .gaussian_filter import GaussianFilter
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.signal_processing.laplacian_filter")
class LaplacianFilter(keras.layers.Layer):
    """Highlight edges with a Difference of Gaussians.

    The layer blurs the input with a :class:`GaussianFilter` and returns
    ``scale_factor * (blurred - input)``. The blur runs at stride ``(1, 1)``
    whatever ``strides`` says, so the two operands of the subtraction have the
    same shape and the output matches the input. Nothing here is trainable; the
    Gaussian kernel comes from ``sigma`` and ``kernel_size``.

    Architecture:

    .. code-block:: text

                   input [B, H, W, C]
                            │
              ┌─────────────┤
              │             ▼
              │   ┌───────────────────┐
              │   │ GaussianFilter    │
              │   │  strides (1, 1)   │
              │   └─────────┬─────────┘
              │             ▼
              │   ┌───────────────────┐
              └──►│ scale * (blur - x)│
                  └─────────┬─────────┘
                            ▼
                   output [B, H, W, C]

    :param kernel_size: Height and width of the 2D kernel. Must be positive odd
        integers. Defaults to ``(5, 5)``.
    :type kernel_size: Tuple[int, int]
    :param strides: Kept for configuration symmetry and serialization. The blur
        always runs at ``(1, 1)``, so this does not change the output shape.
        Defaults to ``(1, 1)``.
    :type strides: Union[Tuple[int, int], List[int]]
    :param sigma: Standard deviation of the Gaussian. A float applies to both
        axes; a pair is ``(sigma_h, sigma_w)``. ``None`` or a non-positive value
        derives it from ``kernel_size`` as ``(k - 1) / 2`` per axis.
    :type sigma: Optional[Union[float, Tuple[float, float]]]
    :param scale_factor: Multiplier on the Laplacian response. Defaults to
        ``1.0``.
    :type scale_factor: float
    :param kernel_initializer: Stored and serialized, but no weight is created
        from it.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Stored and serialized; no weights to regularize.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        ``(batch_size, height, width, channels)``, unchanged.

    :raises ValueError: If ``kernel_size`` is not length 2, or ``sigma`` is
        neither a number nor a length-2 sequence.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 5),
        strides: Union[Tuple[int, int], List[int]] = (1, 1),
        sigma: Optional[Union[float, Tuple[float, float]]] = 1.0,
        scale_factor: float = 1.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be length 2")

        self.kernel_size = kernel_size
        self.strides = tuple(strides) if isinstance(strides, list) else strides
        self.scale_factor = scale_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        if (sigma is None or
                (isinstance(sigma, (float, int)) and sigma <= 0.0)):
            self.sigma = ((kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            self.sigma = (float(sigma[0]), float(sigma[1]))
        elif isinstance(sigma, (float, int)):
            self.sigma = (float(sigma), float(sigma))
        else:
            raise ValueError(f"Invalid sigma value: {sigma}")

        # Strides fixed to (1, 1) so the blurred output matches the input shape.
        self.gaussian_filter = GaussianFilter(
            kernel_size=self.kernel_size,
            strides=(1, 1),
            sigma=self.sigma,
            name="gaussian_filter"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the Gaussian sub-layer. This layer creates no weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if not self.gaussian_filter.built:
            self.gaussian_filter.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Apply the Laplacian filter to the input tensor.

        :param inputs: Input tensor of shape ``[batch_size, height, width, channels]``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :param kwargs: Additional keyword arguments.
        :return: Tensor with highlighted edges, same shape as input.
        :rtype: keras.KerasTensor
        """
        blurred = self.gaussian_filter(inputs, training=training)

        # The order is blurred minus original, matching the standard Laplacian
        # operator, which gives a bright spot a negative response.
        laplacian = keras.ops.multiply(
            self.scale_factor,
            keras.ops.subtract(blurred, inputs)
        )

        return laplacian

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for the layer.

        :return: Dictionary containing configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "sigma": self.sigma,
            "scale_factor": self.scale_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.signal_processing.laplacian_filter")
class AdvancedLaplacianFilter(keras.layers.Layer):
    """Highlight edges through one of three Laplacian implementations.

    ``'dog'`` blurs and subtracts, as :class:`LaplacianFilter` does. ``'log'``
    convolves with a Laplacian of Gaussian kernel built from ``sigma`` and
    ``kernel_size``. ``'kernel'`` convolves with the 3x3 stencil when
    ``kernel_size`` is ``(3, 3)``, and falls back to the same Laplacian of
    Gaussian kernel for any other size. The convolution is depthwise, so
    channels stay separate, and every kernel is fixed.

    Architecture:

    .. code-block:: text

                           input [B, H, W, C]
                                    │
                ┌───────────────────┬───────────────────┐
                ▼                   ▼                   ▼
              'dog'               'log'             'kernel'
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │ blur, then     │  │ depthwise conv │  │ depthwise conv │
        │  blur - input  │  │  LoG kernel    │  │  3x3 stencil   │
        │  strides (1,1) │  │  uses strides  │  │  uses strides  │
        └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
                └───────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ scale_factor * result         │
                    └───────────────┬───────────────┘
                                    ▼
                          output [B, H', W', C]

    H' and W' equal H and W unless ``'log'`` or ``'kernel'`` runs with a stride
    above 1.

    :param method: ``'dog'``, ``'log'``, or ``'kernel'``. Defaults to ``'dog'``.
    :type method: Literal['dog', 'log', 'kernel']
    :param kernel_size: Height and width of the 2D kernel. Defaults to
        ``(5, 5)``.
    :type kernel_size: Tuple[int, int]
    :param strides: Convolution strides. Read only by ``'log'`` and
        ``'kernel'``; the ``'dog'`` blur is pinned to ``(1, 1)``. Defaults to
        ``(1, 1)``.
    :type strides: Union[Tuple[int, int], List[int]]
    :param sigma: Standard deviation of the Gaussian. A float applies to both
        axes; a pair is ``(sigma_h, sigma_w)``. Defaults to ``1.0``.
    :type sigma: Union[float, Tuple[float, float]]
    :param scale_factor: Multiplier on the Laplacian response. Defaults to
        ``1.0``.
    :type scale_factor: float
    :param kernel_initializer: Stored and serialized, but no weight is created
        from it.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Stored and serialized; no weights to regularize.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``, with the channel
        count known at build time.

    Output shape:
        ``(batch_size, height, width, channels)`` for ``'dog'``, otherwise the
        stride-reduced spatial size with ``'same'`` padding.

    :raises ValueError: If ``method`` is not one of the three names, if
        ``sigma`` is neither a number nor a length-2 sequence, or, at build
        time, if the channel axis is undefined.
    """

    def __init__(
        self,
        method: Literal['dog', 'log', 'kernel'] = 'dog',
        kernel_size: Tuple[int, int] = (5, 5),
        strides: Union[Tuple[int, int], List[int]] = (1, 1),
        sigma: Union[float, Tuple[float, float]] = 1.0,
        scale_factor: float = 1.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if method not in ['dog', 'log', 'kernel']:
            raise ValueError(f"Method '{method}' not supported. Use 'dog', 'log', or 'kernel'.")

        self.method = method
        self.kernel_size = kernel_size
        self.strides = tuple(strides) if isinstance(strides, list) else strides
        self.scale_factor = scale_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        if isinstance(sigma, (int, float)):
            self.sigma = (float(sigma), float(sigma))
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            self.sigma = (float(sigma[0]), float(sigma[1]))
        else:
            raise ValueError(f"Invalid sigma value: {sigma}")

        if self.method == 'dog':
            # Stride (1, 1) keeps the blur the same shape as the input, which
            # the subtraction needs.
            self.gaussian_filter = GaussianFilter(
                kernel_size=self.kernel_size,
                strides=(1, 1),
                sigma=self.sigma,
                name="gaussian_filter"
            )
        else:
            self.gaussian_filter = None

        # Deferred to build(), where the channel count is known.
        self.filter_kernel = None

    def _create_laplacian_kernel(self, channels: int) -> keras.KerasTensor:
        """Create a discrete Laplacian kernel.

        :param channels: Number of input channels.
        :type channels: int
        :return: Laplacian kernel tensor.
        :rtype: keras.KerasTensor
        """
        if self.kernel_size == (3, 3):
            kernel_2d = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=np.float32)
        else:
            # No fixed stencil exists at other sizes, so fall back to LoG.
            kernel_2d = self._create_log_kernel()

        kernel = np.zeros((*self.kernel_size, channels, 1), dtype=np.float32)
        for i in range(channels):
            kernel[:, :, i, 0] = kernel_2d

        return keras.ops.convert_to_tensor(kernel, dtype=self.compute_dtype)

    def _create_log_kernel(self) -> np.ndarray:
        """Create a Laplacian of Gaussian (LoG) kernel.

        :return: LoG kernel as numpy array.
        :rtype: np.ndarray
        """
        sigma_x, sigma_y = self.sigma
        height, width = self.kernel_size

        y, x = np.mgrid[-(height // 2):((height + 1) // 2), -(width // 2):((width + 1) // 2)]

        x_squared_norm = x ** 2 / (2 * sigma_x ** 2)
        y_squared_norm = y ** 2 / (2 * sigma_y ** 2)

        r_squared = x_squared_norm + y_squared_norm

        log_kernel = -1.0 / (np.pi * sigma_x * sigma_y) * (1.0 - r_squared) * np.exp(-r_squared)

        # A Laplacian must leave a constant image at zero, so the kernel is
        # recentred to sum to zero.
        log_kernel = log_kernel - np.mean(log_kernel)

        return log_kernel

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the sub-layer or the fixed kernel for the selected method.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the channel axis of ``input_shape`` is ``None``.
        """
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Last dimension (channels) of input must be defined")

        if self.method == 'dog':
            if not self.gaussian_filter.built:
                self.gaussian_filter.build(input_shape)
        elif self.method == 'log':
            log_kernel = self._create_log_kernel().reshape(*self.kernel_size, 1, 1)
            kernel_tensor = keras.ops.convert_to_tensor(log_kernel, dtype=self.compute_dtype)
            # One copy per channel, since the convolution is depthwise.
            self.filter_kernel = keras.ops.tile(kernel_tensor, [1, 1, channels, 1])
        else:
            self.filter_kernel = self._create_laplacian_kernel(channels)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Apply the Laplacian filter to the input tensor.

        :param inputs: Input tensor of shape ``[batch_size, height, width, channels]``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :param kwargs: Additional keyword arguments.
        :return: Tensor with highlighted edges.
        :rtype: keras.KerasTensor
        """
        if self.method == 'dog':
            blurred = self.gaussian_filter(inputs, training=training)
            result = keras.ops.subtract(blurred, inputs)
            return keras.ops.multiply(self.scale_factor, result)
        else:
            conv_result = keras.ops.depthwise_conv(
                inputs=inputs,
                kernel=self.filter_kernel,
                strides=self.strides,
                padding="same"
            )
            return keras.ops.multiply(self.scale_factor, conv_result)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        if self.method == 'dog':
            return input_shape

        # 'same' padding, so the spatial size is the stride-rounded input size.
        output_h = (input_shape[1] + self.strides[0] - 1) // self.strides[0]
        output_w = (input_shape[2] + self.strides[1] - 1) // self.strides[1]
        return input_shape[0], output_h, output_w, input_shape[3]

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for the layer.

        :return: Dictionary containing configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "method": self.method,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "sigma": self.sigma,
            "scale_factor": self.scale_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.signal_processing.laplacian_filter")
class LaplacianPyramidLevel(keras.layers.Layer):
    """Split one pyramid level into low and high bands, and merge them back.

    The split blurs, downsamples by 2 to get the low band, upsamples that back
    and subtracts it from the input to get the high band. Because the high band
    is the difference rather than an estimate, ``merge(split(x))`` returns ``x``
    to float precision whatever the blur looks like. Both bands keep the
    channel count. Calling the layer returns the split.

    split:

    .. code-block:: text

                      x [B, H, W, C]
                            │
                ┌───────────┤
                │           ▼
                │   ┌───────────────────┐
                │   │ GaussianFilter    │
                │   └─────────┬─────────┘
                │             ├────► low [B, H/2, W/2, C]
                │             ▼
                │   ┌───────────────────┐
                │   │ UpSampling2D(2)   │
                │   └─────────┬─────────┘
                │             ▼
                │   ┌───────────────────┐
                └──►│ x - up(low)       │
                    └─────────┬─────────┘
                              ▼
                     high [B, H, W, C]

    merge:

    .. code-block:: text

        low [B, H/2, W/2, C]              high [B, H, W, C]
                  │                               │
        ┌───────────────────┐                     │
        │ UpSampling2D(2)   │                     │
        └─────────┬─────────┘                     │
                  └───────────┬───────────────────┘
                              ▼
                    ┌───────────────────┐
                    │ add               │
                    └─────────┬─────────┘
                              ▼
                     x_rec [B, H, W, C]

    :param blur_kernel_size: Height and width of the Gaussian blur kernel, as a
        length-2 sequence of positive ints. Defaults to ``(5, 5)``.
    :type blur_kernel_size: Tuple[int, int]
    :param blur_sigma: Gaussian sigma. ``-1`` derives it from the kernel size.
        Defaults to ``-1``.
    :type blur_sigma: float
    :param blur_trainable: Make the blur kernel learnable. Defaults to
        ``False``, which keeps the split a fixed signal operation.
    :type blur_trainable: bool
    :param kwargs: Additional ``Layer`` base-class arguments.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        Tuple of two 4D tensors ``(low, high)``, where ``low`` is
        ``(batch_size, height / 2, width / 2, channels)`` and ``high`` is
        ``(batch_size, height, width, channels)``.

    :raises ValueError: If ``blur_kernel_size`` is not a length-2 sequence of
        positive ints, ``blur_sigma`` is not a number, or ``blur_trainable`` is
        not a bool.
    """

    def __init__(
        self,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: float = -1,
        blur_trainable: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if (not isinstance(blur_kernel_size, Sequence)
                or isinstance(blur_kernel_size, str)
                or len(blur_kernel_size) != 2):
            raise ValueError(
                "blur_kernel_size must be a length-2 sequence (height, width), "
                f"got {blur_kernel_size!r}"
            )
        if not all(isinstance(k, int) and not isinstance(k, bool) and k > 0
                   for k in blur_kernel_size):
            raise ValueError(
                "blur_kernel_size entries must be positive ints, "
                f"got {blur_kernel_size!r}"
            )

        # bool is a subclass of int, so it is excluded first.
        if isinstance(blur_sigma, bool) or not isinstance(blur_sigma, (int, float)):
            raise ValueError(
                f"blur_sigma must be a number, got {blur_sigma!r}"
            )

        if not isinstance(blur_trainable, bool):
            raise ValueError(
                f"blur_trainable must be a bool, got {blur_trainable!r}"
            )

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.blur_trainable = blur_trainable

        # Sublayers created here and built explicitly in build().
        self.blur = GaussianFilter(
            kernel_size=blur_kernel_size,
            strides=(2, 2),
            sigma=blur_sigma,
            padding="same",
            trainable=blur_trainable,
        )
        self.up = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

    def build(self, input_shape) -> None:
        """Build the blur, downsample and upsample sub-layers in order.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        h, w = input_shape[1], input_shape[2]
        for name, dim in (("height", h), ("width", w)):
            if dim is not None and dim % 2 != 0:
                raise ValueError(
                    f"input {name} must be even for a 2x pyramid level, got {dim}"
                )
        self.blur.build(input_shape)
        low_shape = self.blur.compute_output_shape(input_shape)
        self.up.build(low_shape)
        super().build(input_shape)

    def split(self, x):
        """Decompose ``x`` into ``(low, high)`` signal bands.

        :param x: Input tensor ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :return: ``(low, high)`` where ``low`` is ``(B, H/2, W/2, C)`` and
            ``high`` is ``(B, H, W, C)``.
        :rtype: tuple
        """
        low = self.blur(x)
        high = keras.ops.subtract(x, self.up(low))
        return low, high

    def merge(self, low, high):
        """Reconstruct the level as ``high + up(low)``, inverting :meth:`split`.

        :param low: Low band ``(B, H/2, W/2, C)``.
        :type low: keras.KerasTensor
        :param high: High band ``(B, H, W, C)``.
        :type high: keras.KerasTensor
        :return: Reconstructed tensor ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        return keras.ops.add(high, self.up(low))

    def call(self, inputs):
        """Return :meth:`split` of the input.

        :param inputs: Input tensor ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :return: ``(low, high)``.
        :rtype: tuple
        """
        return self.split(inputs)

    def compute_output_shape(self, input_shape):
        """Compute the shapes of both bands.

        :param input_shape: Shape tuple ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(low_shape, high_shape)``.
        :rtype: tuple
        """
        batch, h, w, c = input_shape
        low_h = None if h is None else h // 2
        low_w = None if w is None else w // 2
        low_shape = (batch, low_h, low_w, c)
        high_shape = (batch, h, w, c)
        return low_shape, high_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config dictionary for the layer.

        :return: Dictionary containing configuration parameters.
        :rtype: Dict[str, Any]
        """
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
