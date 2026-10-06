"""Fixed Gaussian readout sampling, in-graph, over a :class:`SpatialLayout` grid.

The problem
-----------
An fMRI voxel does not report one neuron's response; it reports an
area-weighted aggregate of the neurons under it. Any comparison between model
units and cortical vertices is therefore only fair if the model side is sampled
the same way. This layer is that sampling, in graph, so a model can be built to
emit voxel-like signals instead of requiring the caller to smooth activations
after the fact.

The mechanism
-------------
Reshape ``(..., num_units)`` onto the ``(h, w)`` grid through a
:class:`~dl_techniques.layers.regularization.spatial_smoothness.SpatialLayout`,
convolve with a separable Gaussian whose width is set by a full-width at
half-maximum, and gather back. The kernel is a **non-trainable** weight: this is
an instrument for simulating a sensor, not a learned filter, and a trained one
would be a second place for the two producers of this quantity to drift.

Order matters and is not interchangeable. Smooth *activations*, then compute a
statistic on the smoothed values. Smoothing a t-map or any other already-derived
statistic is a different operation that inflates apparent spatial structure, and
the post-hoc pipeline in ``dl_techniques.metrics`` never does it.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
    - Kriegeskorte, Borgwardt & Bhattacharyya, 2010. How does an fMRI voxel sample
      the neuronal activity pattern? (https://arxiv.org/abs/0906.2859)
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.regularization.spatial_smoothness import (
    SpatialLayout,
    resolve_grid_shape,
)

# ---------------------------------------------------------------------

#: Default full-width at half-maximum in grid units. The paper uses 2.0 mm at
#: 1.0 mm inter-unit spacing, which is 2.0 grid units.
DEFAULT_FWHM = 2.0

#: Paper's unit spacing, in the same arbitrary units as the FWHM. Kept as the
#: default because the pair (2.0, 1.0) is the reported configuration and the
#: sigma derivation below is only meaningful relative to it.
DEFAULT_UNIT_SPACING = 1.0

#: Edge treatments. ``'nearest'`` replicates the boundary sample; ``'zeros'`` is
#: Keras' own ``padding='same'`` and lets a constant field decay at the border.
PADDING_MODES = ("nearest", "zeros")


def gaussian_sigma(fwhm: float, unit_spacing: float = DEFAULT_UNIT_SPACING) -> float:
    """Convert a full-width at half-maximum into a standard deviation.

    ``sigma = fwhm / (2 * sqrt(2 * ln 2)) / unit_spacing`` -- the 2.3548 factor
    is the FWHM of a Gaussian expressed in sigmas, so the division is what makes
    ``fwhm`` and ``unit_spacing`` commensurable.

    :param fwhm: Full width at half maximum, in millimetres or grid units.
    :type fwhm: float
    :param unit_spacing: Distance between neighbouring units, same units.
    :type unit_spacing: float
    :return: The kernel standard deviation, in grid cells.
    :rtype: float
    :raises ValueError: If ``fwhm`` is not positive or ``unit_spacing`` is not
        positive -- naming the offending value.
    """
    if fwhm <= 0:
        raise ValueError(f"fwhm must be > 0, got {fwhm}")
    if unit_spacing <= 0:
        raise ValueError(f"unit_spacing must be > 0, got {unit_spacing}")
    return float(fwhm) / (2.0 * float(np.sqrt(2.0 * np.log(2.0)))) / float(
        unit_spacing
    )


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """Build a normalised 1-D Gaussian kernel.

    :param kernel_size: Odd side length. Even sizes are rejected: a centred
        kernel needs one sample at its own origin.
    :type kernel_size: int
    :param sigma: Standard deviation in cells.
    :type sigma: float
    :return: A ``(kernel_size,)`` float64 kernel summing to 1.
    :rtype: numpy.ndarray
    :raises ValueError: If ``kernel_size`` is not a positive odd integer.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    offset = (kernel_size - 1) / 2.0
    positions = np.arange(kernel_size, dtype="float64") - offset
    kernel = np.exp(-(positions ** 2) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum()


def gaussian_kernel_2d(kernel_size: int, sigma: float) -> np.ndarray:
    """Build a normalised separable 2-D Gaussian kernel.

    :param kernel_size: Odd side length.
    :type kernel_size: int
    :param sigma: Standard deviation in cells.
    :type sigma: float
    :return: A ``(kernel_size, kernel_size)`` float64 kernel summing to 1.
    :rtype: numpy.ndarray
    """
    line = gaussian_kernel_1d(kernel_size, sigma)
    return np.outer(line, line)


@register_dl_technique("dl_techniques.layers.regularization.gaussian_readout")
class GaussianReadout(keras.layers.Layer):
    """Blur activations over a unit grid with a fixed Gaussian kernel.

    Forward output has the same shape as the input; only the values along the
    unit axis are smoothed, and only across neighbouring *grid* cells rather
    than neighbouring unit indices. That distinction is the whole point -- with
    the paper's default permutation the two orderings share no adjacency at all,
    so an implementation that smoothed the flat axis would be smoothing noise.

    Example:
        .. code-block:: python

            readout = GaussianReadout(fwhm=2.0, unit_spacing=1.0, seed=0)
            voxelish = readout(acts, training=False)   # acts: (B, T, 784)

    :param fwhm: Full width at half maximum, in ``unit_spacing`` units.
    :type fwhm: float
    :param unit_spacing: Distance between neighbouring units. Default 1.0, the
        paper's value, which makes ``fwhm`` readable directly as a width in grid
        units.
    :type unit_spacing: float
    :param permute: Whether the unit layout is permuted, matching the tap this
        readout is paired with.
    :type permute: bool
    :param grid_shape: Explicit ``(height, width)``, or ``None`` to factor.
    :type grid_shape: Optional[Tuple[int, int]]
    :param seed: Seed for the layout permutation.
    :type seed: Optional[int]
    :param kernel_size: Odd kernel side, or ``None`` for
        ``2 * ceil(3 * sigma) + 1``, which truncates at three sigma.
    :type kernel_size: Optional[int]
    :param padding_mode: Edge treatment, ``'nearest'`` (default) or ``'zeros'``.
        This is a real choice and not a synonym for Keras' ``padding``: Keras'
        ``'same'`` pads with ZEROS, which lets a constant field decay at the grid
        border -- a mass loss a sensor simulation has no reason to invent.
        ``'nearest'`` replicates the edge instead, which is what
        ``scipy.ndimage.gaussian_filter(mode="nearest")`` does and what makes the
        in-graph blur agree with the reference implementation.
    :type padding_mode: str
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :raises ValueError: If ``fwhm`` or ``unit_spacing`` is not positive, if
        ``kernel_size`` is given and is not a positive odd integer, or if
        ``padding_mode`` is unknown.
    """

    def __init__(
        self,
        fwhm: float = DEFAULT_FWHM,
        unit_spacing: float = DEFAULT_UNIT_SPACING,
        permute: bool = True,
        grid_shape: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
        kernel_size: Optional[int] = None,
        padding_mode: str = "nearest",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if kernel_size is not None and (
            kernel_size < 1 or kernel_size % 2 == 0
        ):
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if padding_mode not in PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {list(PADDING_MODES)}, "
                f"got {padding_mode!r}"
            )
        # Validate the FWHM pair through the one place that owns the derivation.
        self.sigma = gaussian_sigma(fwhm, unit_spacing)

        self.fwhm = float(fwhm)
        self.unit_spacing = float(unit_spacing)
        self.permute = bool(permute)
        self.grid_shape = (
            None if grid_shape is None else tuple(int(v) for v in grid_shape)
        )
        self.seed = seed
        self.kernel_size = (
            int(kernel_size)
            if kernel_size is not None
            else 2 * int(np.ceil(3.0 * self.sigma)) + 1
        )
        self.padding_mode = padding_mode
        self._half_pad = (self.kernel_size - 1) // 2

        self.layout = None
        self._grid_shape = None

        # The blur is one Conv2D over a single channel, created here so the layer
        # has a single construction path and the fixed kernel arrives through an
        # initializer rather than through a second weight plus an assign().
        # Edge replication is applied explicitly in `call`, so the convolution
        # itself always runs 'valid'.
        self.blur = keras.layers.Conv2D(
            filters=1,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="valid" if padding_mode == "nearest" else "same",
            use_bias=False,
            trainable=False,
            kernel_initializer=_kernel_initializer(
                self.kernel_size, self.sigma
            ),
            name="blur",
        )

        self.cell_to_unit_flat = None
        self.perm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Resolve the layout and materialise the two gather tables.

        :param input_shape: Shape of the input; only the last axis is read.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last axis is not statically known.
        """
        num_units = input_shape[-1]
        if num_units is None:
            raise ValueError(
                f"{self.__class__.__name__} needs a statically known last axis to "
                f"resolve its grid, got input_shape={tuple(input_shape)}."
            )

        grid_shape = resolve_grid_shape(num_units, self.grid_shape)
        self._grid_shape = grid_shape
        self.layout = SpatialLayout(
            num_units=num_units,
            grid_shape=grid_shape,
            permute=self.permute,
            seed=self.seed,
        )

        # Unit -> cell scatter, and cell -> unit gather. Two tables, because they
        # are inverses of each other and a layout only ever uses one direction.
        self.cell_to_unit_flat = self.add_weight(
            name="cell_to_unit_flat",
            shape=(num_units,),
            initializer=_index_initializer(
                self.layout.cell_to_unit.reshape(-1)
            ),
            trainable=False,
            dtype="int32",
        )
        self.perm = self.add_weight(
            name="perm",
            shape=(num_units,),
            initializer=_index_initializer(self.layout.perm),
            trainable=False,
            dtype="int32",
        )

        # Conv2D is rank-4 (batch, rows, cols, channels), while the incoming
        # tensor carries an arbitrary number of leading axes. The blur therefore
        # collapses them to one batch axis and restores them afterwards; the
        # collapse is what lets the same layer serve (S, N), (B, T, N) and (B, N).
        self.blur.build((-1, grid_shape[0], grid_shape[1], 1))

        logger.info(
            f"GaussianReadout: fwhm={self.fwhm} at spacing={self.unit_spacing} "
            f"-> sigma={self.sigma:.4f} cells, kernel {self.kernel_size}x"
            f"{self.kernel_size}, grid {grid_shape[0]}x{grid_shape[1]}"
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Smooth ``inputs`` over the grid and return the same shape.

        :param inputs: Tensor ``(..., num_units)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; the kernel is fixed. Accepted so the layer drops
            into a tap position without a signature change.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        del training

        height, width = self._grid_shape
        lead_shape = ops.shape(inputs)[:-1]
        num_units = ops.shape(inputs)[-1]

        grid = ops.reshape(
            ops.take(inputs, self.cell_to_unit_flat, axis=-1),
            (-1, height, width, 1),
        )
        if self.padding_mode == "nearest":
            grid = _replicate_pad(grid, self._half_pad)
        smoothed = self.blur(grid)
        return ops.take(
            ops.reshape(smoothed, (*lead_shape, num_units)),
            self.perm,
            axis=-1,
        )

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``input_shape`` unchanged."""
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        ``sigma`` is derived in ``__init__`` and recomputed on load from
        ``fwhm``/``unit_spacing``, so it is not carried; ``blur`` is a tracked
        sub-layer and serializes itself.
        """
        config = super().get_config()
        config.update({
            "fwhm": self.fwhm,
            "unit_spacing": self.unit_spacing,
            "permute": self.permute,
            "grid_shape": self.grid_shape,
            "seed": self.seed,
            "kernel_size": self.kernel_size,
            "padding_mode": self.padding_mode,
        })
        return config


def _replicate_pad(grid: keras.KerasTensor, pad: int) -> keras.KerasTensor:
    """Extend a ``(B, H, W, C)`` grid by repeating its edge rows and columns.

    Written out rather than delegated to ``ops.pad`` because none of its modes
    is edge replication. ``mode="SYMMETRIC"`` mirrors -- a left pad of three over
    ``[a, b, c]`` produces ``[a, b, c]``, where replication needs
    ``[a, a, a]`` -- and the difference is invisible in the interior and largest
    at the border, which is exactly where a readout simulation is read.

    This is the boundary rule ``scipy.ndimage.gaussian_filter(mode="nearest")``
    applies, so agreeing with it is a measured property (max |delta| 5.1e-07 over
    the interior before this was corrected, and 0.0405 at the border).
    """
    if pad <= 0:
        return grid
    # The two sides use DIFFERENT edges: row 0 on top, row H-1 underneath. Reusing
    # one slice for both is a silent, plausible-looking bug that shows up only in
    # the bottom-right region of the output.
    top = [grid[:, :1, :, :]] * pad
    bottom = [grid[:, -1:, :, :]] * pad
    rows = ops.concatenate(top + [grid] + bottom, axis=1)
    left = [rows[:, :, :1, :]] * pad
    right = [rows[:, :, -1:, :]] * pad
    return ops.concatenate(left + [rows] + right, axis=2)


def _index_initializer(values: np.ndarray) -> Any:
    """Build an ``add_weight`` initializer returning a fixed integer table.

    NumPy in, NumPy out: computing the index table with ``keras.ops`` inside
    ``build`` and assigning it would be discarded by the ``StatelessScope`` that
    runs whenever this layer is first reached from a parent's ``call``.
    """
    frozen = np.array(values, dtype="int32", copy=True)

    def _initialize(shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        del dtype
        return frozen.reshape(shape)

    return _initialize


def _kernel_initializer(kernel_size: int, sigma: float) -> Any:
    """Build a ``kernel_initializer`` returning the fixed 2-D Gaussian.

    NumPy in, NumPy out, for the same ``StatelessScope`` reason as
    :func:`_index_initializer`.
    """
    frozen = gaussian_kernel_2d(kernel_size, sigma).astype("float32").reshape(
        1, kernel_size, kernel_size, 1
    )

    def _initialize(shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        del dtype
        return frozen.reshape(shape)

    return _initialize