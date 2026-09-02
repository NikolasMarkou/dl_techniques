"""
Gabor-filter-bank initializer for convolutional layers.

This initializer is deterministic and does not perform random sampling.
Instead, it fills a ``Conv2D`` kernel of shape ``(kh, kw, in_ch, out_ch)`` with
a bank of 2D Gabor filters, following the Gabor-filter-bank CNN initialization
of Özbulak & Ekenel. Each output channel ``j`` receives a Gabor kernel drawn
from a factorized orientation x scale x phase sweep; the same 2D Gabor is
replicated identically across all input channels.

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
* ``lambda`` is the wavelength of the sinusoidal carrier (it divides
  ``2 * pi * x_theta``, so it must be strictly positive),
* ``gamma``  is the spatial aspect ratio of the Gaussian envelope; the envelope
  width along ``y_theta`` is ``sigma / gamma``, so ``gamma`` must be >= 0 and
  conventionally lies in ``[0.2, 3.0]``,
* ``psi``    is the phase offset of the sinusoid (in DEGREES).

Gabor filters resemble the receptive fields of simple cells in the mammalian
visual cortex and form effective edge / texture detectors. Initializing the
first convolutional layer of a CNN with such a filter bank (rather than random
weights) gives the network principled, orientation- and frequency-selective
feature extractors from the start.

Sweep strategy
--------------
``sweep="product"`` (the default) builds a factorized bank: ``n_theta``
orientations x ``n_scale`` scales x ``n_psi`` phases, with ``sigma``, ``lambda``
and ``gamma`` all indexed by the scale axis so that envelope width tracks
carrier wavelength. With the default ``psi_range`` the phase axis
holds ``{0, 180}`` from 4 filters and ``{0, 90, 180, 270}`` from 16, i.e. every
filter has a phase-reversed sibling (which is what makes a rectifying activation
on a frozen signed bank lossless) and, in the larger banks, a quadrature partner
as well.

``sweep="diagonal"`` reproduces the original single-``linspace`` scheme, in
which all five parameters are swept jointly with inclusive endpoints and output
channel ``j`` takes the ``j``-th sample of every parameter. That is a
one-dimensional curve through a five-dimensional parameter space; measured on a
96-filter bank it yields a maximum off-diagonal cosine similarity of 0.9999 and
an effective rank (99% spectral energy) of 52/96. It is retained only for
reproducing the paper's exact construction.

Divergence from the reference implementation
--------------------------------------------
The authors' reference code (``gabor_init.py`` in
``github.com/gokhanozbulak/Gabor-Initialized-CNN``) calls
``cv2.getGaborKernel`` with, for a kernel of size ``k`` and ``n`` filters:

    sigma  in [5, k/2 + 1)              (kernel-size dependent)
    lambda == k                          (constant; ``(k - 2) + 2``)
    theta  in [0, 360) degrees
    gamma  in [1.0, 3.0)                 (a 0..300 slider divided by 100)
    psi    in [-90, 180) degrees         (a 90..360 slider minus 180)

That is the arbiter for the parameter semantics, and it is why ``gamma`` here
defaults to ``(0.5, 1.5)`` and NOT to ``(0.0, 300.0)``: an aspect ratio of 300
collapses the ``y_theta`` envelope to sub-pixel width, turning the filter into a
single line of pixels through the origin. This module deliberately diverges from
the reference in three documented ways: the scale parameters default to
kernel-relative ranges (rather than the reference's ``sigma >= 5``, which is
already flat over an 11x11 window), ``lambda`` is swept rather than held
constant, and the sweep is factorized rather than diagonal. Pass explicit ranges
with ``sweep="diagonal"`` to recover the reference behaviour.

Normalization
-------------
With ``normalize=True`` (the default) each 2D filter has its DC component
removed and is then scaled so that its per-element RMS is ``sqrt(2 / fan_in)``
with ``fan_in = kh * kw * in_ch``. Without it the raw bank is unusable as an
initializer: measured on ``(11, 11, 3, 96)`` the un-normalized per-filter L2
norms spanned 0.12 to 4.60 (a factor of 38) and the per-output-channel gain
``sum |w|`` spanned 0.54 to 100.3, i.e. two orders of magnitude of activation
scale at initialization.

All math is performed in numpy ``float64`` and cast once to the requested dtype
at the final step.

References:
    Özbulak, G., & Ekenel, H. K. *Initialization of Convolutional Neural
    Networks by Gabor Filters*. 26th Signal Processing and Communications
    Applications Conference (SIU), 2018.

"""

import keras
import numpy as np
from typing import Callable, Dict, Any, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

RangeLike = Optional[Union[Tuple[float, float], Sequence[float]]]

#: ``sweep`` values accepted by :class:`GaborFiltersInitializer`.
SWEEP_MODES = ("product", "diagonal")

#: ``sigma_range`` when left as ``None``, as a fraction of ``k = min(kh, kw)``.
SIGMA_FRACTIONS = (0.30, 0.60)

#: ``lambda_range`` when left as ``None``, as a fraction of ``k = min(kh, kw)``.
LAMBDA_FRACTIONS = (0.30, 1.00)

#: Filters below this L2 norm are left untouched by the normalization step.
_NORM_EPS = 1e-12

# ---------------------------------------------------------------------


def _validate_range(name: str, rng: RangeLike) -> Optional[Tuple[float, float]]:
    """Coerce a ``(min, max)`` range to a tuple of floats and validate it.

    Args:
        name: Parameter name, used in error messages.
        rng: ``(min, max)`` pair, or ``None`` (passed through unchanged).

    Returns:
        The coerced ``(min, max)`` tuple, or ``None``.

    Raises:
        ValueError: If the range is not 2-element, holds a non-finite bound, or
            has ``min > max``.
    """
    if rng is None:
        return None

    coerced = tuple(rng)
    if len(coerced) != 2:
        raise ValueError(
            f"{name} must have exactly 2 elements (min, max), "
            f"got {len(coerced)}: {coerced}"
        )

    lo, hi = (float(coerced[0]), float(coerced[1]))

    # A non-finite bound passes every naive comparison (nan > hi is False and
    # nan <= 0 is False) and silently yields an all-NaN kernel, so it is
    # rejected here rather than downstream.
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name} bounds must be finite, got ({lo}, {hi})")
    if lo > hi:
        raise ValueError(f"{name} must satisfy min <= max, got ({lo}, {hi})")

    return (lo, hi)


def _axis(
    lo: float,
    hi: float,
    n: int,
    geometric: bool = False,
) -> np.ndarray:
    """Sample ``n`` values across ``[lo, hi]`` inclusive, midpoint when ``n == 1``.

    A single sample takes the midpoint rather than ``lo``: the endpoints of every
    range are the degenerate extremes, so a one-filter bank must not be pinned to
    one of them.

    Args:
        lo: Lower bound.
        hi: Upper bound.
        n: Number of samples, >= 1.
        geometric: If True, space the samples geometrically (requires ``lo > 0``).

    Returns:
        A ``(n,)`` float64 array.
    """
    if n == 1:
        mid = float(np.sqrt(lo * hi)) if (geometric and lo > 0.0) else 0.5 * (lo + hi)
        return np.asarray([mid], dtype=np.float64)
    if geometric and lo > 0.0:
        return np.geomspace(lo, hi, n, dtype=np.float64)
    return np.linspace(lo, hi, n, dtype=np.float64)


def _factorize_bank(n: int) -> Tuple[int, int, int]:
    """Split ``n`` filters into ``(n_theta, n_scale, n_psi)`` product axes.

    Phase gets four values (a quadrature pair AND its two sign-flipped siblings)
    once the bank is large enough, two below that, one for a tiny bank; the
    remaining budget is split so that orientations outnumber scales, which is the
    shape a Gabor bank wants (a few octaves, many orientations).

    Args:
        n: Number of distinct filters, >= 1.

    Returns:
        ``(n_theta, n_scale, n_psi)`` whose product is >= ``n``.
    """
    n_psi = 4 if n >= 16 else (2 if n >= 4 else 1)
    m = int(np.ceil(n / n_psi))
    n_scale = int(max(1, min(6, round(np.sqrt(m) / 1.5))))
    n_theta = int(np.ceil(m / n_scale))
    return n_theta, n_scale, n_psi


@register_dl_technique("dl_techniques.initializers.gabor_filters_initializer")
class GaborFiltersInitializer(keras.initializers.Initializer):
    """Gabor-filter-bank initializer for convolutional layers.

    Fills a 4D ``Conv2D`` kernel ``(kh, kw, in_ch, out_ch)`` with a bank of 2D
    Gabor filters (Özbulak & Ekenel). Each output channel holds a Gabor kernel;
    the same 2D filter is replicated identically across all input channels, so a
    ``Conv2D`` initialized this way responds to the unweighted SUM of its input
    channels (it is colour-blind at initialization, and its gain scales with
    ``in_ch``). Use :func:`create_gabor_depthwise_conv2d` for a per-channel bank.

    The Gabor kernel is computed on a coordinate grid centered at
    ``((kw - 1) / 2, (kh - 1) / 2)`` as::

        x_theta =  x * cos(theta) + y * sin(theta)
        y_theta = -x * sin(theta) + y * cos(theta)
        g = exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
            * cos(2 * pi * x_theta / lambda + psi)

    ``theta`` and ``psi`` are supplied in DEGREES and converted with
    ``np.deg2rad`` before use.

    This initializer is deterministic: it performs no random sampling and takes
    no seed. Two calls with the same shape produce byte-identical tensors. The
    bank is a warm start, not a fixed transform — with ``normalize=True`` (the
    default) it is scaled to a He-like magnitude precisely so that it can be
    trained on. It is also usable frozen; :func:`create_gabor_depthwise_conv2d`
    defaults to ``trainable=False`` for that use.

    Per-channel (depthwise) use: the same 4D convention serves a
    ``keras.layers.DepthwiseConv2D`` kernel ``(kh, kw, in_ch, depth_multiplier)``
    unchanged. There the last axis is ``depth_multiplier`` (the filter bank) and
    the bank is replicated across ``in_ch`` exactly as for ``Conv2D``; because a
    depthwise convolution does NOT mix channels, the result is each input channel
    convolved independently with the full Gabor bank — ``in_ch * depth_multiplier``
    output channels. See :func:`create_gabor_depthwise_conv2d`.

    Args:
        sigma_range: ``(min, max)`` interval for the Gaussian envelope width
            ``sigma``; ``min`` must be strictly positive. ``None`` (the default)
            resolves at call time to ``(0.30 * k, 0.60 * k)`` with
            ``k = min(kh, kw)``, because a scale that is not relative to the
            kernel is either flat over the whole window or sub-pixel.
        theta_range: ``(min, max)`` interval for the filter orientation ``theta``,
            in DEGREES. Defaults to ``(0.0, 180.0)``; the upper endpoint is
            EXCLUSIVE in ``product`` mode since ``g(theta + 180, psi) ==
            g(theta, -psi)``.
        lambda_range: ``(min, max)`` interval for the sinusoid wavelength
            ``lambda``; ``min`` must be strictly positive (it divides
            ``2 * pi * x_theta``). ``None`` (the default) resolves at call time to
            ``(0.30 * k, 1.00 * k)``.
        gamma_range: ``(min, max)`` interval for the spatial aspect ratio
            ``gamma``; ``min`` must be >= 0. Defaults to ``(0.5, 1.5)``. See the
            module docstring on why this is not ``(0.0, 300.0)``.
        psi_range: ``(min, max)`` interval for the phase offset ``psi``, in
            DEGREES. Defaults to ``(0.0, 360.0)``; the upper endpoint is
            EXCLUSIVE in ``product`` mode, so two phases give the pair
            ``(0, 180)``.
        sweep: ``"product"`` (default) for the factorized orientation x scale x
            phase bank, or ``"diagonal"`` for the original joint-``linspace``
            construction.
        n_filters: Number of DISTINCT filters. ``None`` (the default) means
            ``out_ch``. A smaller value tiles the bank cyclically across the
            output channels.
        normalize: If True (default), remove each filter's DC component and scale
            it to a per-element RMS of ``sqrt(2 / fan_in)``. Set False for the
            raw, un-normalized Gabor responses.

    Raises:
        ValueError: If any range is not exactly two elements, holds a non-finite
            bound, or has ``min > max``; if ``sigma_range[0] <= 0``,
            ``lambda_range[0] <= 0`` or ``gamma_range[0] < 0``; if ``sweep`` is
            not a member of :data:`SWEEP_MODES`; or if ``n_filters < 1``.

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
        sigma_range: RangeLike = None,
        theta_range: RangeLike = (0.0, 180.0),
        lambda_range: RangeLike = None,
        gamma_range: RangeLike = (0.5, 1.5),
        psi_range: RangeLike = (0.0, 360.0),
        sweep: str = "product",
        n_filters: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        """Initialize the Gabor-filter-bank initializer.

        Args:
            sigma_range: ``(min, max)`` for the Gaussian envelope width, or
                ``None`` for a kernel-relative default; ``min`` must be > 0.
            theta_range: ``(min, max)`` for orientation, in DEGREES.
            lambda_range: ``(min, max)`` for the sinusoid wavelength, or ``None``
                for a kernel-relative default; ``min`` must be > 0.
            gamma_range: ``(min, max)`` for the spatial aspect ratio; ``min``
                must be >= 0.
            psi_range: ``(min, max)`` for the phase offset, in DEGREES.
            sweep: ``"product"`` or ``"diagonal"``.
            n_filters: Number of distinct filters, or ``None`` for ``out_ch``.
            normalize: Whether to DC-remove and energy-normalize each filter.

        Raises:
            ValueError: See the class docstring.
        """
        # NOTE: keras.initializers.Initializer (Keras 3) defines no __init__, so
        # there is nothing to forward to and a **kwargs passthrough could only
        # ever raise TypeError from object.__init__. This signature is closed.
        self.sigma_range = _validate_range("sigma_range", sigma_range)
        self.theta_range = _validate_range("theta_range", theta_range)
        self.lambda_range = _validate_range("lambda_range", lambda_range)
        self.gamma_range = _validate_range("gamma_range", gamma_range)
        self.psi_range = _validate_range("psi_range", psi_range)

        if self.sigma_range is not None and self.sigma_range[0] <= 0:
            raise ValueError(
                f"sigma_range[0] must be > 0 (it divides 2*sigma**2), "
                f"got {self.sigma_range[0]}"
            )
        if self.lambda_range is not None and self.lambda_range[0] <= 0:
            raise ValueError(
                f"lambda_range[0] must be > 0 (it divides 2*pi*x_theta), "
                f"got {self.lambda_range[0]}"
            )
        if self.gamma_range is not None and self.gamma_range[0] < 0:
            raise ValueError(
                f"gamma_range[0] must be >= 0 (it is an aspect ratio), "
                f"got {self.gamma_range[0]}"
            )

        if sweep not in SWEEP_MODES:
            raise ValueError(
                f"sweep must be one of {SWEEP_MODES}, got {sweep!r}"
            )
        if n_filters is not None and n_filters < 1:
            raise ValueError(f"n_filters must be >= 1 or None, got {n_filters}")

        self.sweep = sweep
        self.n_filters = None if n_filters is None else int(n_filters)
        self.normalize = bool(normalize)

        logger.debug(
            f"Initialized GaborFiltersInitializer with "
            f"sigma_range={self.sigma_range}, theta_range={self.theta_range}, "
            f"lambda_range={self.lambda_range}, gamma_range={self.gamma_range}, "
            f"psi_range={self.psi_range}, sweep={self.sweep}, "
            f"n_filters={self.n_filters}, normalize={self.normalize}"
        )

    # -----------------------------------------------------------------
    # parameter sweeps
    # -----------------------------------------------------------------

    def _resolved_ranges(self, k: int) -> Dict[str, Tuple[float, float]]:
        """Resolve the ``None`` scale ranges against the kernel size.

        Args:
            k: ``min(kh, kw)``, the reference kernel extent.

        Returns:
            A dict with the five resolved ``(min, max)`` ranges.
        """
        sigma = self.sigma_range
        if sigma is None:
            sigma = (SIGMA_FRACTIONS[0] * k, SIGMA_FRACTIONS[1] * k)
        lambda_ = self.lambda_range
        if lambda_ is None:
            lambda_ = (LAMBDA_FRACTIONS[0] * k, LAMBDA_FRACTIONS[1] * k)

        return {
            "sigma": sigma,
            "theta": self.theta_range,
            "lambda": lambda_,
            "gamma": self.gamma_range,
            "psi": self.psi_range,
        }

    def _sweep_parameters(self, n: int, k: int) -> Tuple[np.ndarray, ...]:
        """Produce ``n`` parameter 5-tuples as five ``(n,)`` float64 arrays.

        Args:
            n: Number of distinct filters.
            k: ``min(kh, kw)``.

        Returns:
            ``(sigmas, thetas_deg, lambdas, gammas, psis_deg)``.
        """
        rng = self._resolved_ranges(k)

        if self.sweep == "diagonal":
            # The original construction: one joint linspace across all five
            # parameters, inclusive endpoints. Kept for reproducibility only.
            return tuple(
                np.linspace(rng[key][0], rng[key][1], n, dtype=np.float64)
                for key in ("sigma", "theta", "lambda", "gamma", "psi")
            )

        n_theta, n_scale, n_psi = _factorize_bank(n)

        # theta and psi are periodic, so their upper endpoint duplicates the
        # lower one: sample them half-open.
        theta_axis = rng["theta"][0] + (
            np.arange(n_theta, dtype=np.float64)
            * (rng["theta"][1] - rng["theta"][0]) / n_theta
        )
        psi_axis = rng["psi"][0] + (
            np.arange(n_psi, dtype=np.float64)
            * (rng["psi"][1] - rng["psi"][0]) / n_psi
        )

        # sigma, lambda and gamma all ride the scale axis, so the envelope width
        # tracks the carrier wavelength instead of drifting independently.
        sigma_axis = _axis(rng["sigma"][0], rng["sigma"][1], n_scale, geometric=True)
        lambda_axis = _axis(rng["lambda"][0], rng["lambda"][1], n_scale, geometric=True)
        gamma_axis = _axis(rng["gamma"][0], rng["gamma"][1], n_scale)

        # Mixed-radix enumeration with psi fastest, so a phase pair stays
        # adjacent and truncating the tail costs whole orientations, not phases.
        idx = np.arange(n_theta * n_scale * n_psi)
        psi_idx = idx % n_psi
        theta_idx = (idx // n_psi) % n_theta
        scale_idx = idx // (n_psi * n_theta)

        take = slice(0, n)
        return (
            sigma_axis[scale_idx][take],
            theta_axis[theta_idx][take],
            lambda_axis[scale_idx][take],
            gamma_axis[scale_idx][take],
            psi_axis[psi_idx][take],
        )

    # -----------------------------------------------------------------
    # call
    # -----------------------------------------------------------------

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

        kh, kw, in_ch, out_ch = (int(d) for d in shape)

        if kh < 1 or kw < 1 or in_ch < 1 or out_ch < 1:
            raise ValueError(
                f"All kernel dimensions must be >= 1, got "
                f"(kh={kh}, kw={kw}, in_ch={in_ch}, out_ch={out_ch})"
            )

        n = out_ch if self.n_filters is None else min(self.n_filters, out_ch)
        k = min(kh, kw)

        logger.debug(
            f"Generating Gabor filter bank for shape {tuple(shape)} "
            f"({n} distinct filters, sweep={self.sweep})"
        )

        sigmas, thetas_deg, lambdas, gammas, psis_deg = self._sweep_parameters(n, k)

        # Broadcast the parameters on a leading filter axis: (n, 1, 1).
        sigma = sigmas[:, None, None]
        theta = np.deg2rad(thetas_deg)[:, None, None]
        lambda_ = lambdas[:, None, None]
        gamma = gammas[:, None, None]
        psi = np.deg2rad(psis_deg)[:, None, None]

        # Centre on (k - 1) / 2 so that an EVEN kernel size is centred on the
        # half-pixel between the two middle taps instead of being off-centre.
        xs = np.arange(kw, dtype=np.float64) - (kw - 1) / 2.0
        ys = np.arange(kh, dtype=np.float64) - (kh - 1) / 2.0
        xx, yy = np.meshgrid(xs, ys)  # both (kh, kw)

        x_theta = xx * np.cos(theta) + yy * np.sin(theta)
        y_theta = -xx * np.sin(theta) + yy * np.cos(theta)

        # The exponent is always <= 0 for sigma > 0, so exp cannot overflow.
        bank = np.exp(
            -(x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (2.0 * sigma ** 2)
        ) * np.cos(2.0 * np.pi * x_theta / lambda_ + psi)  # (n, kh, kw)

        if self.normalize:
            bank = self._normalize_bank(bank, fan_in=kh * kw * in_ch)

        # Tile the distinct filters cyclically across the output channels, then
        # replicate each 2D filter identically across all input channels.
        bank = bank[np.arange(out_ch) % n]                      # (out_ch, kh, kw)
        kernel = np.transpose(bank, (1, 2, 0))[:, :, None, :]   # (kh, kw, 1, out_ch)
        kernel = np.repeat(kernel, in_ch, axis=2)

        return keras.ops.convert_to_tensor(
            kernel.astype(keras.backend.standardize_dtype(dtype)), dtype=dtype
        )

    @staticmethod
    def _normalize_bank(bank: np.ndarray, fan_in: int) -> np.ndarray:
        """DC-remove and energy-normalize every filter of a ``(n, kh, kw)`` bank.

        Args:
            bank: The raw Gabor bank.
            fan_in: ``kh * kw * in_ch`` of the kernel being initialized.

        Returns:
            The normalized bank, each filter with zero mean and per-element RMS
            ``sqrt(2 / fan_in)``. Filters whose energy vanishes after DC removal
            (a filter that is constant over the window) are left at zero, with a
            warning; DC removal is skipped entirely for a 1x1 kernel, where it
            would zero every filter and leave a dead layer.
        """
        if bank.shape[1] * bank.shape[2] > 1:
            bank = bank - bank.mean(axis=(1, 2), keepdims=True)
        norms = np.sqrt((bank ** 2).sum(axis=(1, 2), keepdims=True))
        target = np.sqrt(2.0 / fan_in) * np.sqrt(bank.shape[1] * bank.shape[2])
        dead = int((norms <= _NORM_EPS).sum())
        if dead:
            logger.warning(
                f"GaborFiltersInitializer: {dead} filter(s) have no energy after "
                f"normalization and are initialized to zero"
            )
        scale = np.where(norms > _NORM_EPS, target / np.maximum(norms, _NORM_EPS), 0.0)
        return bank * scale

    # -----------------------------------------------------------------
    # serialization
    # -----------------------------------------------------------------

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
            'sweep': self.sweep,
            'n_filters': self.n_filters,
            'normalize': self.normalize,
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
# builder utilities
# ---------------------------------------------------------------------


def _resolve_filters_per_channel(
    filters_per_channel: Optional[int],
    filters: Optional[int],
) -> int:
    """Resolve the deprecated ``filters`` alias for ``filters_per_channel``.

    Args:
        filters_per_channel: The current parameter.
        filters: The deprecated alias.

    Returns:
        The resolved filter count.

    Raises:
        ValueError: If neither or both are supplied, or the value is < 1.
    """
    if filters is not None:
        if filters_per_channel is not None:
            raise ValueError(
                "pass filters_per_channel, not both it and the deprecated "
                "'filters' alias"
            )
        logger.debug(
            "create_gabor_depthwise_conv2d: 'filters' is a deprecated alias for "
            "'filters_per_channel' (it is a depth_multiplier, not an output "
            "channel count)"
        )
        filters_per_channel = filters

    if filters_per_channel is None:
        raise ValueError("filters_per_channel is required")
    if filters_per_channel < 1:
        raise ValueError(
            f"filters_per_channel must be >= 1, got {filters_per_channel}"
        )
    return int(filters_per_channel)


def create_gabor_depthwise_conv2d(
    filters_per_channel: Optional[int] = None,
    kernel_size: Union[int, Tuple[int, int]] = 11,
    activation: Union[str, Callable, keras.layers.Layer, None] = None,
    sigma_range: RangeLike = None,
    theta_range: RangeLike = (0.0, 180.0),
    lambda_range: RangeLike = None,
    gamma_range: RangeLike = (0.5, 1.5),
    psi_range: RangeLike = (0.0, 360.0),
    sweep: str = "product",
    normalize: bool = True,
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = 'same',
    use_bias: bool = False,
    trainable: bool = False,
    name: Optional[str] = None,
    filters: Optional[int] = None,
) -> keras.layers.DepthwiseConv2D:
    """Create a ``DepthwiseConv2D`` Gabor bank applied PER CHANNEL (no mixing).

    # DECISION plan_2026-06-18_ba4e0079/D-001: per-channel (depthwise) Gabor.
    Builds a ``keras.layers.DepthwiseConv2D`` with
    ``depth_multiplier=filters_per_channel`` whose depthwise kernel is initialized
    by a :class:`GaborFiltersInitializer` (Özbulak & Ekenel). Each of the
    ``filters_per_channel`` Gabor filters is applied **independently to each input
    channel** — a depthwise convolution does NOT sum across input channels, so
    there is **no cross-channel mixing**. For an input with ``C`` channels the
    output therefore has ``C * filters_per_channel`` channels (e.g. 3 channels x
    96 filters -> 288 output channels). Each input channel sees the same bank.

    This is the correct primitive for a fixed orientation/frequency-selective
    front-end. It deliberately does NOT take a target output-channel count: if a
    specific output width is needed, **follow this layer with a ``1x1`` Conv2D
    projection** (e.g. the cliffordnet autoencoder's ``enc_proj[0]``). For a
    cross-channel ``Conv2D`` warm start see :func:`create_gabor_conv2d`.

    Defaults to ``trainable=False`` (a frozen, deterministic Gabor front-end),
    matching :func:`create_haar_depthwise_conv2d`. The depthwise kernel has shape
    ``(kh, kw, in_channels, filters_per_channel)`` and is exposed as
    ``layer.kernel`` in Keras 3.8 (not ``layer.depthwise_kernel``).

    Args:
        filters_per_channel: Number of Gabor filters applied per channel
            (``depth_multiplier``). Output channels =
            ``in_channels * filters_per_channel``. Must be >= 1.
        kernel_size: Spatial size of the convolution window. Int or ``(kh, kw)``
            tuple. Defaults to ``11`` (the Özbulak & Ekenel first-layer size).
        activation: Optional activation applied to the depthwise Gabor responses.
            Passed straight to the ``DepthwiseConv2D`` constructor; accepts a
            Keras activation name, a callable, or a layer. Defaults to ``None``
            (linear passthrough — the raw signed Gabor responses).

            # DECISION plan_2026-07-13_f44e2cb0/D-001: not validated on purpose.
            Only positively homogeneous activations (``relu``, ``leaky_relu``,
            ``linear``) preserve the degree-1 homogeneity ``D(a*x) = a*D(x)`` that
            the bias-free denoisers in ``models/vision/bias_free_denoisers/`` rely on;
            ``gelu``/``elu``/``tanh``/``sigmoid``/``mish`` break it. This builder is
            generic (``models/vision/cliffordnet/`` uses it too and is not bias-free), so
            it does NOT reject them — the caller owns that contract. Note that in
            the default ``sweep="product"`` mode with >= 4 filters the bank holds
            phase-reversed ``(psi, psi + 180)`` pairs, so a rectifying activation
            keeps every filter's negative lobe on its sibling channel; under
            ``sweep="diagonal"`` it does not, and that lobe is discarded.
        sigma_range: ``(min, max)`` interval for the Gaussian envelope width, or
            ``None`` (default) for the kernel-relative default.
        theta_range: ``(min, max)`` interval for orientation, in DEGREES.
        lambda_range: ``(min, max)`` interval for the sinusoid wavelength, or
            ``None`` (default) for the kernel-relative default.
        gamma_range: ``(min, max)`` interval for the spatial aspect ratio.
        psi_range: ``(min, max)`` interval for the phase offset, in DEGREES.
        sweep: ``"product"`` (default) or ``"diagonal"``.
        normalize: Whether to DC-remove and energy-normalize each filter.
            Defaults to True.
        strides: Convolution strides. Int or ``(sh, sw)`` tuple. Defaults to ``1``.
        padding: Padding mode (``'same'`` or ``'valid'``). Defaults to ``'same'``.
        use_bias: Whether to add bias terms. Defaults to ``False``.
        trainable: Whether the Gabor kernel can be trained. Defaults to ``False``
            (frozen per-channel front-end).
        name: Layer name. Defaults to ``'gabor_depthwise_conv2d'``.
        filters: Deprecated alias for ``filters_per_channel``. It reads as a
            Keras output-channel count, which this is not.

    Returns:
        keras.layers.DepthwiseConv2D: Configured depthwise layer whose
        ``depthwise_initializer`` is a Gabor filter bank. Output channels =
        ``in_channels * filters_per_channel``.

    Raises:
        ValueError: If ``filters_per_channel < 1``, or if both it and the
            deprecated ``filters`` alias are given. (Range validation is delegated
            to ``GaborFiltersInitializer.__init__``.)

    Example:
        >>> from dl_techniques.initializers import create_gabor_depthwise_conv2d
        >>> # Frozen per-channel Gabor front-end:
        >>> layer = create_gabor_depthwise_conv2d(filters_per_channel=96, kernel_size=11)
        >>> # Input: (batch, 32, 32, 3) -> Output: (batch, 32, 32, 3 * 96 = 288)
        >>> # For a specific output width, follow with a 1x1 Conv2D, e.g.:
        >>> # proj = keras.layers.Conv2D(64, 1)  # 288 -> 64
    """
    filters_per_channel = _resolve_filters_per_channel(filters_per_channel, filters)

    logger.debug(
        f"Creating depthwise Gabor layer: filters/channel={filters_per_channel}, "
        f"kernel_size={kernel_size}, activation={activation}, "
        f"trainable={trainable} (output channels = in_channels * "
        f"{filters_per_channel})"
    )

    return keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        depth_multiplier=filters_per_channel,
        strides=strides,
        padding=padding,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=GaborFiltersInitializer(
            sigma_range=sigma_range, theta_range=theta_range,
            lambda_range=lambda_range, gamma_range=gamma_range,
            psi_range=psi_range, sweep=sweep, normalize=normalize,
        ),
        trainable=trainable,
        name=name or 'gabor_depthwise_conv2d',
    )


def create_gabor_conv2d(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]] = 11,
    activation: Union[str, Callable, keras.layers.Layer, None] = None,
    sigma_range: RangeLike = None,
    theta_range: RangeLike = (0.0, 180.0),
    lambda_range: RangeLike = None,
    gamma_range: RangeLike = (0.5, 1.5),
    psi_range: RangeLike = (0.0, 360.0),
    sweep: str = "product",
    normalize: bool = True,
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = 'same',
    use_bias: bool = False,
    trainable: bool = True,
    name: Optional[str] = None,
) -> keras.layers.Conv2D:
    """Create a ``Conv2D`` whose kernel is a Gabor bank (a trainable warm start).

    This is the paper's own use case: the first convolutional layer of a CNN, warm
    started with Gabor filters and then refined by gradient descent — hence
    ``trainable=True`` by default, the opposite of
    :func:`create_gabor_depthwise_conv2d`.

    Because the same 2D filter is replicated across all input channels, output
    channel ``j`` sees the Gabor response of the unweighted SUM of the input
    channels: the layer is colour-blind at initialization and training has to
    break that symmetry. ``normalize=True`` (the default) is what keeps the layer
    trainable at all — it gives every output channel the same He-like gain.

    Args:
        filters: Number of output channels (the Keras meaning), which is also the
            number of Gabor filters. Must be >= 1.
        kernel_size: Spatial size of the convolution window. Defaults to ``11``.
        activation: Optional activation, passed to ``Conv2D``.
        sigma_range: ``(min, max)`` envelope width, or ``None`` for the
            kernel-relative default.
        theta_range: ``(min, max)`` orientation, in DEGREES.
        lambda_range: ``(min, max)`` wavelength, or ``None`` for the
            kernel-relative default.
        gamma_range: ``(min, max)`` spatial aspect ratio.
        psi_range: ``(min, max)`` phase offset, in DEGREES.
        sweep: ``"product"`` (default) or ``"diagonal"``.
        normalize: Whether to DC-remove and energy-normalize each filter.
        strides: Convolution strides. Defaults to ``1``.
        padding: Padding mode. Defaults to ``'same'``.
        use_bias: Whether to add bias terms. Defaults to ``False``.
        trainable: Whether the kernel is trainable. Defaults to ``True``.
        name: Layer name. Defaults to ``'gabor_conv2d'``.

    Returns:
        keras.layers.Conv2D: A ``Conv2D`` warm started with the Gabor bank.

    Raises:
        ValueError: If ``filters < 1``. (Range validation is delegated to
            ``GaborFiltersInitializer.__init__``.)

    Example:
        >>> from dl_techniques.initializers import create_gabor_conv2d
        >>> layer = create_gabor_conv2d(filters=96, kernel_size=11)
        >>> # Input: (batch, 32, 32, 3) -> Output: (batch, 32, 32, 96)
    """
    if filters < 1:
        raise ValueError(f"filters must be >= 1, got {filters}")

    logger.debug(
        f"Creating Gabor Conv2D: filters={filters}, kernel_size={kernel_size}, "
        f"activation={activation}, trainable={trainable}"
    )

    return keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=GaborFiltersInitializer(
            sigma_range=sigma_range, theta_range=theta_range,
            lambda_range=lambda_range, gamma_range=gamma_range,
            psi_range=psi_range, sweep=sweep, normalize=normalize,
        ),
        trainable=trainable,
        name=name or 'gabor_conv2d',
    )

# ---------------------------------------------------------------------
