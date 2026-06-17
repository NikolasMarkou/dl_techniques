"""Gabor-filter-bank initializer for convolutional layers.

This initializer is deterministic and does not perform random sampling.
Instead, it fills a ``Conv2D`` kernel of shape ``(kh, kw, in_ch, out_ch)`` with
a bank of 2D Gabor filters, implementing the Gabor-filter-bank CNN
initialization of Özbulak & Ekenel. Each output channel ``j`` receives a
distinct Gabor kernel whose five generating parameters are swept across the
paper's Table I intervals; the same 2D Gabor is replicated identically across
all input channels.

Mathematical Foundations:
A 2D Gabor filter is a sinusoidal plane wave modulated by a Gaussian envelope.
For a coordinate grid centered at the origin (``x`` along width, ``y`` along
height), with a rotation ``theta`` applied to the coordinates,

    x_theta =  x * cos(theta) + y * sin(theta)
    y_theta = -x * sin(theta) + y * cos(theta)

    g(x, y) = exp( -(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2) )
              * cos( 2 * pi * x_theta / lambda + psi )

where:

* ``sigma``  controls the width of the Gaussian envelope (a divisor of
  ``2 * sigma**2``, so it must be strictly positive),
* ``theta``  is the orientation of the filter (in DEGREES, converted with
  ``np.deg2rad`` before use),
* ``lambda`` is the wavelength of the sinusoidal carrier,
* ``gamma``  is the spatial aspect ratio of the Gaussian envelope,
* ``psi``    is the phase offset of the sinusoid (in DEGREES).

Gabor filters resemble the receptive fields of simple cells in the mammalian
visual cortex and form effective edge / texture detectors. Initializing the
first convolutional layer of a CNN with such a filter bank (rather than random
weights) gives the network principled, orientation- and frequency-selective
feature extractors from the start; the weights remain trainable and are
refined by gradient descent.

The filter bank is distributed by output channel: with ``n_filters = out_ch``,
each of the five parameters is sampled with ``np.linspace(min, max, n_filters)``
over its Table I interval, and output channel ``j`` uses the ``j``-th sample of
every parameter. All math is performed in numpy ``float32`` and converted to a
Keras tensor at the final step.

References:
    Özbulak, G., & Ekenel, H. K. *Initialization of Convolutional Neural
    Networks by Gabor Filters*.

"""

import keras
import numpy as np
from keras import ops
from typing import Dict, Any, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GaborFiltersInitializer(keras.initializers.Initializer):
    """Gabor-filter-bank initializer for convolutional layers.

    Fills a 4D ``Conv2D`` kernel ``(kh, kw, in_ch, out_ch)`` with a bank of 2D
    Gabor filters (Özbulak & Ekenel). Each output channel ``j`` holds a distinct
    Gabor kernel; the same 2D filter is replicated identically across all input
    channels. With ``n_filters = out_ch``, each of the five Gabor parameters is
    sampled via ``np.linspace(min, max, n_filters)`` across its range, and output
    channel ``j`` uses the ``j``-th sample of every parameter.

    The Gabor kernel for channel ``j`` is computed on a centered coordinate grid
    (``x = arange(kw) - kw // 2``, ``y = arange(kh) - kh // 2``) as::

        x_theta =  x * cos(theta) + y * sin(theta)
        y_theta = -x * sin(theta) + y * cos(theta)
        g = exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
            * cos(2 * pi * x_theta / lambda + psi)

    ``theta`` and ``psi`` are supplied in DEGREES and converted with
    ``np.deg2rad`` before use.

    This initializer is deterministic: it performs no random sampling and takes
    no seed. Two calls with the same shape produce byte-identical tensors. The
    weights are intended to be trainable (refined by gradient descent after the
    Gabor warm start).

    Edge case ``out_ch == 1``: ``np.linspace(min, max, 1)`` returns ``[min]`` for
    every parameter, so the single filter uses each range's minimum endpoint.

    Args:
        sigma_range: ``(min, max)`` interval for the Gaussian envelope width
            ``sigma``. ``min`` must be strictly positive (it divides
            ``2 * sigma**2``). Table I default ``(2.0, 21.0)``.
        theta_range: ``(min, max)`` interval for the filter orientation ``theta``,
            in DEGREES. Table I default ``(0.0, 360.0)``.
        lambda_range: ``(min, max)`` interval for the sinusoid wavelength
            ``lambda``. Table I default ``(8.0, 100.0)``.
        gamma_range: ``(min, max)`` interval for the spatial aspect ratio
            ``gamma``. Table I default ``(0.0, 300.0)``.
        psi_range: ``(min, max)`` interval for the phase offset ``psi``, in
            DEGREES. Table I default ``(0.0, 360.0)``.
        **kwargs: Additional keyword arguments forwarded to the base initializer.

    Raises:
        ValueError: If any range is not exactly two elements, if any
            ``min > max``, or if ``sigma_range[0] <= 0``.

    Example:
        >>> import keras
        >>> from dl_techniques.initializers import GaborFiltersInitializer
        >>> # Use as the first convolutional layer of a CNN (trainable warm start):
        >>> layer = keras.layers.Conv2D(
        ...     filters=96,
        ...     kernel_size=11,
        ...     kernel_initializer=GaborFiltersInitializer(),
        ...     trainable=True,
        ... )
        >>> # Directly produce a kernel bank:
        >>> w = GaborFiltersInitializer()((11, 11, 3, 96))  # (kh, kw, in, out)
    """

    def __init__(
        self,
        sigma_range: Union[Tuple[float, float], Sequence[float]] = (2.0, 21.0),
        theta_range: Union[Tuple[float, float], Sequence[float]] = (0.0, 360.0),
        lambda_range: Union[Tuple[float, float], Sequence[float]] = (8.0, 100.0),
        gamma_range: Union[Tuple[float, float], Sequence[float]] = (0.0, 300.0),
        psi_range: Union[Tuple[float, float], Sequence[float]] = (0.0, 360.0),
        **kwargs: Any,
    ) -> None:
        """Initialize the Gabor-filter-bank initializer.

        Args:
            sigma_range: ``(min, max)`` for the Gaussian envelope width; ``min``
                must be > 0.
            theta_range: ``(min, max)`` for orientation, in DEGREES.
            lambda_range: ``(min, max)`` for the sinusoid wavelength.
            gamma_range: ``(min, max)`` for the spatial aspect ratio.
            psi_range: ``(min, max)`` for the phase offset, in DEGREES.
            **kwargs: Additional keyword arguments forwarded to the base class.

        Raises:
            ValueError: If any range is not 2-element, any ``min > max``, or
                ``sigma_range[0] <= 0``.
        """
        super().__init__(**kwargs)

        # Coerce each range to a tuple (accept list or tuple) and validate.
        # Coercing here (not in get_config) keeps from_config(cls(**config))
        # clean, since .keras serializes tuples as lists.
        ranges = {
            "sigma_range": tuple(sigma_range),
            "theta_range": tuple(theta_range),
            "lambda_range": tuple(lambda_range),
            "gamma_range": tuple(gamma_range),
            "psi_range": tuple(psi_range),
        }

        for name, rng in ranges.items():
            if len(rng) != 2:
                raise ValueError(
                    f"{name} must have exactly 2 elements (min, max), "
                    f"got {len(rng)}: {rng}"
                )
            lo, hi = rng
            if lo > hi:
                raise ValueError(
                    f"{name} must satisfy min <= max, got ({lo}, {hi})"
                )

        if ranges["sigma_range"][0] <= 0:
            raise ValueError(
                f"sigma_range[0] must be > 0 (it divides 2*sigma**2), "
                f"got {ranges['sigma_range'][0]}"
            )

        self.sigma_range = ranges["sigma_range"]
        self.theta_range = ranges["theta_range"]
        self.lambda_range = ranges["lambda_range"]
        self.gamma_range = ranges["gamma_range"]
        self.psi_range = ranges["psi_range"]

        logger.info(
            f"Initialized GaborFiltersInitializer with "
            f"sigma_range={self.sigma_range}, theta_range={self.theta_range}, "
            f"lambda_range={self.lambda_range}, gamma_range={self.gamma_range}, "
            f"psi_range={self.psi_range}"
        )

    def __call__(
        self,
        shape: Sequence[int],
        dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a Gabor filter bank for a 4D Conv2D kernel.

        Args:
            shape: Required 4D shape ``(kh, kw, in_ch, out_ch)``.
            dtype: Data type of the returned tensor. ``None`` falls back to
                ``keras.config.floatx()``.
            **kwargs: Additional arguments (unused).

        Returns:
            Tensor: A ``(kh, kw, in_ch, out_ch)`` tensor holding the Gabor bank.

        Raises:
            ValueError: If ``shape`` is not 4D, or any dimension is < 1.
        """
        if dtype is None:
            dtype = keras.config.floatx()

        if len(shape) != 4:
            raise ValueError(
                f"Expected 4D Conv2D kernel shape (kh, kw, in_ch, out_ch), "
                f"got {len(shape)}D: {tuple(shape)}"
            )

        kh, kw, in_ch, out_ch = shape

        if kh < 1 or kw < 1 or in_ch < 1 or out_ch < 1:
            raise ValueError(
                f"All kernel dimensions must be >= 1, got "
                f"(kh={kh}, kw={kw}, in_ch={in_ch}, out_ch={out_ch})"
            )

        logger.debug(
            f"Generating Gabor filter bank for shape {tuple(shape)} "
            f"({out_ch} filters)"
        )

        # Sweep each parameter across its range, one sample per output channel.
        n = out_ch
        sigmas = np.linspace(self.sigma_range[0], self.sigma_range[1], n)
        thetas_deg = np.linspace(self.theta_range[0], self.theta_range[1], n)
        lambdas = np.linspace(self.lambda_range[0], self.lambda_range[1], n)
        gammas = np.linspace(self.gamma_range[0], self.gamma_range[1], n)
        psis_deg = np.linspace(self.psi_range[0], self.psi_range[1], n)

        # Build the centered coordinate grid once. xx, yy have shape (kh, kw).
        xs = np.arange(kw) - kw // 2
        ys = np.arange(kh) - kh // 2
        xx, yy = np.meshgrid(xs, ys)
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        kernel = np.zeros((kh, kw, in_ch, out_ch), dtype=np.float32)

        for j in range(out_ch):
            theta = np.deg2rad(thetas_deg[j])
            psi = np.deg2rad(psis_deg[j])
            sigma = sigmas[j]
            lambda_ = lambdas[j]
            gamma = gammas[j]

            x_theta = xx * np.cos(theta) + yy * np.sin(theta)
            y_theta = -xx * np.sin(theta) + yy * np.cos(theta)

            gb = np.exp(
                -(x_theta ** 2 + (gamma ** 2) * (y_theta ** 2))
                / (2.0 * sigma ** 2)
            ) * np.cos(2.0 * np.pi * x_theta / lambda_ + psi)
            gb = gb.astype(np.float32)

            # Replicate the same 2D Gabor across all input channels.
            kernel[:, :, :, j] = gb[:, :, None]

        return ops.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization.

        Returns:
            Dict containing the initializer configuration.
        """
        config = super().get_config()
        config.update({
            'sigma_range': self.sigma_range,
            'theta_range': self.theta_range,
            'lambda_range': self.lambda_range,
            'gamma_range': self.gamma_range,
            'psi_range': self.psi_range,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GaborFiltersInitializer':
        """Create initializer from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            GaborFiltersInitializer instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------
# builder utility
# ---------------------------------------------------------------------

def create_gabor_conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]] = 5,
    sigma_range: Union[Tuple[float, float], Sequence[float]] = (2.0, 21.0),
    theta_range: Union[Tuple[float, float], Sequence[float]] = (0.0, 360.0),
    lambda_range: Union[Tuple[float, float], Sequence[float]] = (8.0, 100.0),
    gamma_range: Union[Tuple[float, float], Sequence[float]] = (0.0, 300.0),
    psi_range: Union[Tuple[float, float], Sequence[float]] = (0.0, 360.0),
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = 'same',
    use_bias: bool = False,
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    trainable: bool = True,
    input_shape: Optional[Tuple[int, ...]] = None,
    name: Optional[str] = None,
) -> keras.layers.Conv2D:
    """Create a ``Conv2D`` layer initialized with a Gabor filter bank.

    Builds a standard ``keras.layers.Conv2D`` whose kernel is initialized by a
    ``GaborFiltersInitializer`` (Ozbulak & Ekenel). With ``filters`` output
    channels, the kernel holds ``filters`` distinct 2D Gabor filters swept over
    the paper's Table I parameter intervals (the same 2D filter replicated across
    all input channels). This is the standard recipe for seeding the **first
    convolutional layer** of a CNN with biologically-motivated, orientation- and
    frequency-selective features instead of random weights.

    The defaults follow the paper's first convolutional layer: ``filters=96``
    (per call site), ``kernel_size=5``, ``strides=1``. Unlike
    ``create_haar_depthwise_conv2d`` (which defaults ``trainable=False`` because
    wavelet filters are fixed), this builder defaults ``trainable=True``: the
    Gabor initialization is a strong **starting point** that is then refined by
    gradient descent.

    Note: this returns a *standard* ``Conv2D`` (not a ``DepthwiseConv2D``),
    matching the paper's standard-convolution first layer.

    Args:
        filters: Number of output channels (= number of distinct Gabor filters).
            Must be >= 1.
        kernel_size: Spatial size of the convolution window. Int or
            ``(kh, kw)`` tuple. Defaults to ``5``.
        sigma_range: ``(min, max)`` interval for the Gaussian envelope width
            ``sigma``; ``min`` must be > 0. Table I default ``(2.0, 21.0)``.
        theta_range: ``(min, max)`` interval for orientation, in DEGREES. Table I
            default ``(0.0, 360.0)``.
        lambda_range: ``(min, max)`` interval for the sinusoid wavelength. Table I
            default ``(8.0, 100.0)``.
        gamma_range: ``(min, max)`` interval for the spatial aspect ratio. Table I
            default ``(0.0, 300.0)``.
        psi_range: ``(min, max)`` interval for the phase offset, in DEGREES.
            Table I default ``(0.0, 360.0)``.
        strides: Convolution strides. Int or ``(sh, sw)`` tuple. Defaults to ``1``.
        padding: Padding mode (``'same'`` or ``'valid'``). Defaults to ``'same'``.
        use_bias: Whether to add bias terms. Defaults to ``False``.
        kernel_regularizer: Optional kernel regularization.
        trainable: Whether the Gabor-initialized kernel can be trained. Defaults
            to ``True`` (the Gabor init is a warm start refined by training).
        input_shape: Optional input shape ``(H, W, C)`` for use as a model's first
            layer. Passed through only when not ``None``.
        name: Layer name. Defaults to ``'gabor_conv2d'``.

    Returns:
        keras.layers.Conv2D: Configured Conv2D layer with a Gabor-bank kernel
        initializer.

    Raises:
        ValueError: If ``filters < 1``. (Range validation is delegated to
            ``GaborFiltersInitializer.__init__``.)

    Example:
        >>> from dl_techniques.initializers import create_gabor_conv2d
        >>> # First convolutional layer of a CNN (trainable Gabor warm start):
        >>> layer = create_gabor_conv2d(
        ...     filters=96,
        ...     kernel_size=5,
        ...     trainable=True,
        ...     input_shape=(32, 32, 3),
        ...     name='gabor_first_conv',
        ... )
        >>> # Input: (batch, 32, 32, 3) -> Output: (batch, 32, 32, 96)
    """
    if filters < 1:
        raise ValueError(f"filters must be >= 1, got {filters}")

    logger.info(
        f"Creating Gabor Conv2D layer: filters={filters}, "
        f"kernel_size={kernel_size}, trainable={trainable}"
    )

    layer_kwargs = dict(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=GaborFiltersInitializer(
            sigma_range=sigma_range, theta_range=theta_range,
            lambda_range=lambda_range, gamma_range=gamma_range, psi_range=psi_range,
        ),
        kernel_regularizer=kernel_regularizer,
        trainable=trainable,
        name=name or 'gabor_conv2d',
    )
    if input_shape is not None:
        layer_kwargs['input_shape'] = input_shape

    return keras.layers.Conv2D(**layer_kwargs)

# ---------------------------------------------------------------------
