"""
Normalized spatial coordinate grids, and sampling a feature map at continuous
coordinates.

This module owns one concept -- *where a spatial location sits, expressed as a
number a network can read* -- and the two operations built on it: constructing a
coordinate grid, and reading a feature grid at arbitrary continuous coordinates.

Coordinate injection breaks translation equivariance on demand, a design
paradigm that supplies a network with information its architecture is
structurally incapable of representing. A convolution applies the same kernel at
every position, so its response depends on what a feature looks like but not on
where it sits in the grid. That invariance is a valuable inductive bias for
recognition, and a hard failure mode for any task whose answer is a position:
coordinate regression, rendering from a latent, spatially conditioned
generation. Supplying the coordinates as ordinary input channels lets the
network learn to use absolute position where it helps and ignore it where it
does not, without altering the convolution operator itself. This is the
mechanism introduced as CoordConv. The same grids, read at *query* coordinates
that need not align with any pixel, are what an arbitrary-scale implicit decoder
consumes.

The grid convention is three independent choices, and this module makes each one
an explicit knob rather than a hard-coded assumption
-------------------------------------------------------------------------------

**Alignment.** ``'endpoints'`` samples ``linspace(-0.5, 0.5, n)``: the first and
last samples land exactly on the interval's edges. ``'centers'`` samples
``linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n)``: the centers of ``n`` equal cells
tiling ``[-0.5, 0.5]``. The distinction is not cosmetic. Pixel *centers* are the
convention under which "the coordinate of pixel ``i``" is stable as ``n``
changes, which is what an arbitrary-scale sampler requires -- an endpoint grid
re-scales every interior coordinate when the resolution changes, so a query
computed at one scale does not name the same point at another.

**Channel order.** ``'ij'`` stacks ``[h_coord, w_coord]``, matching
``np.meshgrid(..., indexing='ij')`` and array indexing order. ``'xy'`` stacks
``[w_coord, h_coord]``, matching the cartesian reading of a picture. Both are in
use in the wild and neither is inferable from a shape, so a wrong choice here is
a silent transpose: every assertion on rank and extent still passes.

**Normalization.** Concatenating raw coordinates alongside learned activations
mixes two distributions with unrelated scales, and the coordinate channels,
being large and perfectly structured, can dominate the gradient signal early in
training before the network has learned to weight them appropriately.
``'zscore'`` maps each channel to ``z = (x - mu) / (sigma + eps)``, placing the
coordinate features on the same statistical footing as typical activations. A
useful consequence is that the initial interval becomes irrelevant: any affine
reparameterization of the ``linspace`` span collapses to the same standardized
values, so ``[-0.5, 0.5]`` is a convention rather than a tuned constant. The
epsilon guards the degenerate case of a single-element axis, where the
coordinate has zero variance. ``'none'`` leaves the raw span, which is what a
sampler needs: the coordinate must remain interpretable as a position, and
z-scoring destroys that.

History: these three knobs were previously two modules. A
``layers/grid_sample.py`` implemented the ``('centers', 'ij', 'none')`` corner
for the THERA neural heat field and carried a docstring explaining that
``SpatialLayer`` -- hard-coded to ``('endpoints', 'xy', 'zscore')`` -- was "NOT
equivalent" and so could not be reused. That was true of the code and false of
the concept. The two are the same generator under different conventions, and
they are one module now.

Sampling
--------
:func:`interpolate_grid` reads a feature grid of shape ``(B, H', W', C)`` at
query coordinates of shape ``(B, Hq, Wq, 2)`` given in ``[-0.5, 0.5]`` with
channel order ``[h, w]``. The coordinate -> continuous-pixel-index map per axis
is::

    pix = coord * size + (size - 1) / 2

where ``size`` is ``H'`` for axis 0 and ``W'`` for axis 1. Border handling is
edge replication (``mode='nearest'``: CLAMP to ``[0, size-1]``). ``order=0``
rounds to the nearest integer index; ``order=1`` performs a 4-corner bilinear
lerp. Output shape is ``(B, Hq, Wq, C)``.

The sampler is a bare function, not a Layer, on purpose: it is stateless, and
its one differentiating consumer (the THERA aliasing-TV Jacobian) must call it
inside a nested gradient tape, where a Layer's call machinery is an obstacle
rather than a service.

Differentiability
-----------------
THERA's forward pass uses ``order=0`` (nearest) to sample the phi-params and the
source coordinates. The aliasing TV Jacobian is the Jacobian of the heat FIELD
w.r.t. its local spatial input ``rel_coords``, where
``rel_coords = coords - interpolate_grid(coords, source_coords)``. The nearest
sampling term has zero gradient almost everywhere, so the coordinate gradient
flows through the DIRECT ``coords`` term, not through the sampler. Therefore
``interpolate_grid`` need not be differentiable for THERA's exact-Jacobian
deliverable. The ``order=1`` path nonetheless is: its lerp weights are computed
from the UNCLAMPED fractional part, so they stay a smooth function of ``coords``
and gradients propagate to the query coordinates.

References:
    - Liu et al., 2018. An Intriguing Failing of Convolutional Neural Networks
      and the CoordConv Solution. (https://arxiv.org/abs/1807.03247)
    - Becker et al., 2025. Thera: Aliasing-Free Arbitrary-Scale Super-Resolution
      with Neural Heat Fields. (https://arxiv.org/abs/2311.17643)

"""

import keras
import numpy as np
from typing import Tuple, Optional, Any, Literal, Union

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

AlignmentType = Literal["centers", "endpoints"]
ChannelOrderType = Literal["ij", "xy"]
NormalizationType = Literal["none", "zscore"]

_VALID_ALIGNMENTS = ("centers", "endpoints")
_VALID_CHANNEL_ORDERS = ("ij", "xy")
_VALID_NORMALIZATIONS = ("none", "zscore")
_VALID_RESIZE_METHODS = ("nearest", "bilinear")

#: Guards the zero-variance axis (``n == 1``) in the ``'zscore'`` path.
ZSCORE_EPSILON = 1e-7

# ---------------------------------------------------------------------


def _axis_positions(n: int, alignment: str) -> np.ndarray:
    """Sample positions along one axis of length ``n``, spanning ``[-0.5, 0.5]``.

    :param n: Axis length. Must be positive.
    :type n: int
    :param alignment: ``'centers'`` or ``'endpoints'``.
    :type alignment: str
    :return: A ``(n,)`` float64 array of positions.
    :rtype: numpy.ndarray
    """
    if alignment == "centers":
        offset = 1.0 / (2.0 * n)
        return np.linspace(-0.5 + offset, 0.5 - offset, n)
    return np.linspace(-0.5, 0.5, n)


def coordinate_grid(
    size: Union[int, Tuple[int, int]],
    alignment: AlignmentType = "centers",
    channel_order: ChannelOrderType = "ij",
    normalization: NormalizationType = "none",
    dtype: str = "float32",
) -> np.ndarray:
    """Build a normalized 2-D coordinate grid.

    Both axes span ``[-0.5, 0.5]`` and are sampled independently using their own
    length, so a rectangular grid is not a stretched square one.

    :param size: Grid size. An ``int`` ``n`` yields an ``(n, n)`` grid; an
        ``(h, w)`` pair yields an ``(h, w)`` grid. Both sides must be positive
        Python ints -- this is a static, eager constructor.
    :type size: int or tuple[int, int]
    :param alignment: ``'centers'`` samples the centers of ``n`` equal cells
        tiling ``[-0.5, 0.5]`` (``linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n)``);
        ``'endpoints'`` samples ``linspace(-0.5, 0.5, n)``. Defaults to
        ``'centers'``.
    :type alignment: str
    :param channel_order: ``'ij'`` stacks ``[h_coord, w_coord]``; ``'xy'`` stacks
        ``[w_coord, h_coord]``. Defaults to ``'ij'``.
    :type channel_order: str
    :param normalization: ``'none'`` keeps the raw span; ``'zscore'`` maps each
        channel to zero mean and unit standard deviation with an epsilon of
        :data:`ZSCORE_EPSILON`. Defaults to ``'none'``.
    :type normalization: str
    :param dtype: Output dtype. Defaults to ``'float32'``.
    :type dtype: str
    :return: An array of shape ``(h, w, 2)``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``size`` is not an int or a length-2 pair of positive
        ints, or if any of the three convention arguments is unrecognized.
    """
    if alignment not in _VALID_ALIGNMENTS:
        raise ValueError(
            f"alignment must be one of {list(_VALID_ALIGNMENTS)}, got {alignment!r}"
        )
    if channel_order not in _VALID_CHANNEL_ORDERS:
        raise ValueError(
            f"channel_order must be one of {list(_VALID_CHANNEL_ORDERS)}, "
            f"got {channel_order!r}"
        )
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {list(_VALID_NORMALIZATIONS)}, "
            f"got {normalization!r}"
        )

    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2:
        raise ValueError(f"size must be an int or a pair (h, w), got {size!r}")
    h, w = int(size[0]), int(size[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"size values must be positive, got {(h, w)!r}")

    grid_h, grid_w = np.meshgrid(
        _axis_positions(h, alignment), _axis_positions(w, alignment), indexing="ij"
    )

    if normalization == "zscore":
        grid_h = (grid_h - grid_h.mean()) / (grid_h.std() + ZSCORE_EPSILON)
        grid_w = (grid_w - grid_w.mean()) / (grid_w.std() + ZSCORE_EPSILON)

    channels = [grid_h, grid_w] if channel_order == "ij" else [grid_w, grid_h]
    return np.stack(channels, axis=-1).astype(dtype)


# ---------------------------------------------------------------------


def _gather_hw(
    grid: "keras.KerasTensor", idx_h: "keras.KerasTensor", idx_w: "keras.KerasTensor"
) -> "keras.KerasTensor":
    """Gather ``grid[b, idx_h, idx_w, :]`` for batched integer index maps.

    ``keras.ops`` has no ``gather_nd`` and ``keras.ops.take`` has no
    ``batch_dims``, so this uses the repo's linearized-index idiom (see
    ``losses/yolo12_multitask_loss.py``): flatten the leading three axes, form
    ``(b * H' + idx_h) * W' + idx_w``, take along axis 0, reshape back.

    :param grid: ``(B, H', W', C)`` feature grid.
    :type grid: keras.KerasTensor
    :param idx_h: ``(B, Hq, Wq)`` int32 clamped row indices.
    :type idx_h: keras.KerasTensor
    :param idx_w: ``(B, Hq, Wq)`` int32 clamped column indices.
    :type idx_w: keras.KerasTensor
    :return: ``(B, Hq, Wq, C)`` gathered values.
    :rtype: keras.KerasTensor
    """
    grid_shape = keras.ops.shape(grid)
    size_h, size_w, channels = grid_shape[1], grid_shape[2], grid_shape[3]

    lead = keras.ops.shape(idx_h)  # (B, Hq, Wq)
    batch_idx = keras.ops.reshape(
        keras.ops.arange(0, lead[0], dtype="int32"), (-1, 1, 1)
    )
    batch_idx = keras.ops.broadcast_to(batch_idx, lead)

    flat_idx = (batch_idx * size_h + idx_h) * size_w + idx_w
    flat_grid = keras.ops.reshape(grid, (-1, channels))
    gathered = keras.ops.take(flat_grid, keras.ops.reshape(flat_idx, (-1,)), axis=0)
    return keras.ops.reshape(gathered, (lead[0], lead[1], lead[2], channels))


def interpolate_grid(
    coords: Union["keras.KerasTensor", np.ndarray],
    grid: Union["keras.KerasTensor", np.ndarray],
    order: int = 0,
) -> "keras.KerasTensor":
    """Sample a feature grid at continuous query coordinates (edge-clamped).

    :param coords: ``(B, Hq, Wq, 2)`` query coordinates in ``[-0.5, 0.5]``,
        channel order ``[h, w]`` -- i.e. ``channel_order='ij'``.
    :type coords: keras.KerasTensor or numpy.ndarray
    :param grid: ``(B, H', W', C)`` feature grid to sample.
    :type grid: keras.KerasTensor or numpy.ndarray
    :param order: Interpolation order. ``0`` = nearest-neighbour (default);
        ``1`` = bilinear (4-corner lerp, differentiable w.r.t. ``coords``).
    :type order: int
    :return: ``(B, Hq, Wq, C)`` sampled values, in ``grid``'s own dtype.
    :rtype: keras.KerasTensor
    :raises ValueError: If ``order`` is not ``0`` or ``1``.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}")

    # DECISION plan-2026-08-19T163559-499b6f0e/D-046
    # The coordinate arithmetic is INDEX math and always runs in float32; the
    # sampled VALUES come back in `grid`'s own dtype. This used to be
    # `tf.convert_to_tensor(coords, dtype=tf.float32)`, which does not convert —
    # it RAISES `ValueError` on a float16 tensor — so every caller under
    # `mixed_float16` died here. Do NOT restore the float32-only contract: a
    # float32 return would then meet a float16 activation in the caller, which is
    # the same defect one frame later. Do NOT instead run the index math in
    # float16 either: `pix = coord * size + (size - 1) / 2` is a pixel index and
    # float16 cannot resolve adjacent integers past 2048.
    # See decisions.md D-046.
    #
    # Ported from raw `tf` to `keras.ops` on 2026-08-31 together with the merge
    # of `layers/grid_sample.py` into this module. The port was measured
    # bit-identical to the `tf.gather_nd` original: max abs diff 0.0 over three
    # shape sets x order 0/1, and 0.0 with a float16 grid at both orders, dtype
    # preserved. The dtype contract above is unchanged and is what makes that
    # true.
    coords = keras.ops.cast(keras.ops.convert_to_tensor(coords), "float32")
    grid = keras.ops.convert_to_tensor(grid)
    if not str(keras.backend.standardize_dtype(grid.dtype)).startswith("float"):
        grid = keras.ops.cast(grid, "float32")
    value_dtype = grid.dtype

    grid_shape = keras.ops.shape(grid)
    size_h = grid_shape[-3]
    size_w = grid_shape[-2]
    size_h_f = keras.ops.cast(size_h, "float32")
    size_w_f = keras.ops.cast(size_w, "float32")

    # coord -> continuous pixel index per axis: pix = coord * size + (size - 1) / 2
    coord_h = coords[..., 0]  # (B, Hq, Wq)
    coord_w = coords[..., 1]
    pix_h = coord_h * size_h_f + (size_h_f - 1.0) / 2.0
    pix_w = coord_w * size_w_f + (size_w_f - 1.0) / 2.0

    max_h = size_h - 1
    max_w = size_w - 1

    if order == 0:
        # Nearest: round, then clamp to [0, size-1].
        idx_h = keras.ops.cast(keras.ops.round(pix_h), "int32")
        idx_w = keras.ops.cast(keras.ops.round(pix_w), "int32")
        idx_h = keras.ops.clip(idx_h, 0, max_h)
        idx_w = keras.ops.clip(idx_w, 0, max_w)
        return _gather_hw(grid, idx_h, idx_w)

    # order == 1: bilinear. Floor/ceil corners, clamp each, lerp by frac part.
    # Floor in float, derive fractional weights, THEN clamp integer corners. The
    # weights are computed from the UNCLAMPED frac so they stay a smooth function
    # of coords (gradient to coords); clamping only affects which pixel is read,
    # reproducing mode='nearest' edge replication.
    h0_f = keras.ops.floor(pix_h)
    w0_f = keras.ops.floor(pix_w)
    frac_h = pix_h - h0_f  # (B, Hq, Wq), differentiable in coords
    frac_w = pix_w - w0_f

    h0 = keras.ops.cast(h0_f, "int32")
    w0 = keras.ops.cast(w0_f, "int32")
    h1 = h0 + 1
    w1 = w0 + 1

    h0c = keras.ops.clip(h0, 0, max_h)
    h1c = keras.ops.clip(h1, 0, max_h)
    w0c = keras.ops.clip(w0, 0, max_w)
    w1c = keras.ops.clip(w1, 0, max_w)

    v00 = _gather_hw(grid, h0c, w0c)  # (B, Hq, Wq, C)
    v01 = _gather_hw(grid, h0c, w1c)
    v10 = _gather_hw(grid, h1c, w0c)
    v11 = _gather_hw(grid, h1c, w1c)

    # Expand weights to (B, Hq, Wq, 1) to broadcast over channels, and bring
    # them down to the value dtype so the lerp does not mix precisions.
    fh = keras.ops.cast(keras.ops.expand_dims(frac_h, axis=-1), value_dtype)
    fw = keras.ops.cast(keras.ops.expand_dims(frac_w, axis=-1), value_dtype)

    top = v00 * (1.0 - fw) + v01 * fw
    bot = v10 * (1.0 - fw) + v11 * fw
    return top * (1.0 - fh) + bot * fh


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.spatial_layer")
class SpatialLayer(keras.layers.Layer):
    """
    Spatial coordinate grid generator for injecting positional information into models.

    This non-trainable layer emits a coordinate grid shaped like its input's
    spatial extent. It reads only the input's shape, never its values, and the
    grid it returns is meant to be concatenated onto the feature map by the
    caller -- widening the following convolution's input by two channels.

    The grid convention is set by three knobs (``alignment``, ``channel_order``,
    ``normalization``) whose meaning is documented at module level and whose
    construction is delegated to :func:`coordinate_grid`, the single
    implementation shared with every other consumer in the library.

    **Two build modes:**

    .. code-block:: text

        resolution = (rh, rw)              resolution = None
        ─────────────────────              ─────────────────
        prototype [1, rh, rw, 2]           grid [1, H, W, 2]
                 │                                  │
        resize -> [1, H, W, 2]             (no resize -- exact)
                 │                                  │
        tile  -> [B, H, W, 2]              tile -> [B, H, W, 2]

    A fixed prototype plus resizing keeps grid construction outside the forward
    path and tolerates spatial shapes not known until call time; the tradeoff is
    that interpolation only approximately preserves standardization when the
    target resolution differs substantially from the prototype, and that
    ``'nearest'`` resampling produces piecewise-constant coordinate steps rather
    than a smooth ramp. ``resolution=None`` removes both effects by building the
    grid at the input's own size, at the cost of requiring static spatial dims.

    :param resolution: ``(height, width)`` of the prototype grid, both positive;
        or ``None`` to build the grid directly at the input's static spatial
        size and skip resizing entirely. Defaults to ``(4, 4)``.
    :type resolution: tuple[int, int] or None
    :param resize_method: Interpolation method used when ``resolution`` is not
        ``None``. One of ``'nearest'`` or ``'bilinear'``. Defaults to
        ``'nearest'``.
    :type resize_method: str
    :param alignment: ``'centers'`` or ``'endpoints'``. Defaults to
        ``'endpoints'``.
    :type alignment: str
    :param channel_order: ``'ij'`` (``[h, w]``) or ``'xy'`` (``[w, h]``).
        Defaults to ``'xy'``.
    :type channel_order: str
    :param normalization: ``'none'`` or ``'zscore'``. Defaults to ``'zscore'``.
    :type normalization: str
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    :raises ValueError: If ``resolution`` is not ``None`` and not a pair of
        positive ints, or if any convention argument is unrecognized.
    """

    def __init__(
        self,
        resolution: Optional[Tuple[int, int]] = (4, 4),
        resize_method: Literal['nearest', 'bilinear'] = 'nearest',
        alignment: AlignmentType = 'endpoints',
        channel_order: ChannelOrderType = 'xy',
        normalization: NormalizationType = 'zscore',
        **kwargs: Any
    ) -> None:
        # Force non-trainable since this is a deterministic coordinate generator
        kwargs['trainable'] = False
        super().__init__(**kwargs)

        # Validate inputs
        if resolution is not None:
            if len(resolution) != 2:
                raise ValueError(f"resolution must be a tuple of 2 integers, got {resolution}")
            if resolution[0] <= 0 or resolution[1] <= 0:
                raise ValueError(f"resolution values must be positive, got {resolution}")
            resolution = (int(resolution[0]), int(resolution[1]))

        if resize_method not in _VALID_RESIZE_METHODS:
            raise ValueError(
                f"resize_method must be one of {list(_VALID_RESIZE_METHODS)}, "
                f"got '{resize_method}'"
            )
        if alignment not in _VALID_ALIGNMENTS:
            raise ValueError(
                f"alignment must be one of {list(_VALID_ALIGNMENTS)}, got '{alignment}'"
            )
        if channel_order not in _VALID_CHANNEL_ORDERS:
            raise ValueError(
                f"channel_order must be one of {list(_VALID_CHANNEL_ORDERS)}, "
                f"got '{channel_order}'"
            )
        if normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {list(_VALID_NORMALIZATIONS)}, "
                f"got '{normalization}'"
            )

        # Store configuration
        self.resolution = resolution
        self.resize_method = resize_method
        self.alignment = alignment
        self.channel_order = channel_order
        self.normalization = normalization

        # Coordinate grid attribute - created in build()
        self.xy_grid = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the coordinate grid during layer building.

        :param input_shape: Shape tuple of the input tensor. Must be 4D.
        :type input_shape: tuple
        :raises ValueError: If the input is not 4D, or if ``resolution is None``
            and either spatial dimension is not statically known.
        """
        if len(input_shape) != 4:
            raise ValueError(f"SpatialLayer expects 4D input, got shape {input_shape}")

        if self.resolution is None:
            if input_shape[1] is None or input_shape[2] is None:
                raise ValueError(
                    "resolution=None builds the grid at the input's own spatial "
                    "size, which must therefore be statically known; got "
                    f"input_shape={input_shape}. Pass an explicit resolution to "
                    "use the prototype-and-resize path instead."
                )
            grid_size = (int(input_shape[1]), int(input_shape[2]))
        else:
            grid_size = self.resolution

        # ONE implementation of the coordinate convention, shared with every
        # other consumer in the library. Built in numpy at graph-construction
        # time: it is a constant, not a weight.
        grid = coordinate_grid(
            grid_size,
            alignment=self.alignment,
            channel_order=self.channel_order,
            normalization=self.normalization,
            dtype="float32",
        )

        # Add batch dimension for later broadcasting -> (1, gh, gw, 2)
        self.xy_grid = keras.ops.convert_to_tensor(grid[None, ...])

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """
        Emit the coordinate grid at the input's spatial dimensions.

        :param inputs: Input tensor with shape ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Unused, kept for interface consistency.
        :type training: bool or None
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        :return: Coordinate grid tensor with shape ``(batch_size, height, width, 2)``.
        :rtype: keras.KerasTensor
        """
        # Get input spatial dimensions
        input_shape = keras.ops.shape(inputs)
        batch_size = input_shape[0]

        if self.resolution is None:
            # Built at exactly this spatial size -- no resampling at all.
            xy_grid_resized = self.xy_grid
        else:
            # Resize the coordinate grid to match input spatial dimensions.
            # Use keras.ops.image.resize for backend-agnostic resizing.
            xy_grid_resized = keras.ops.image.resize(
                images=self.xy_grid,
                size=(input_shape[1], input_shape[2]),
                interpolation=self.resize_method,
                data_format='channels_last'
            )

        # Tile the grid to match the batch size
        # ops.repeat repeats along specified axis
        xy_grid_batched = keras.ops.repeat(
            xy_grid_resized,
            repeats=batch_size,
            axis=0
        )

        # The grid is built in float32 for coordinate precision; hand it back in
        # the layer's own compute dtype so it can be concatenated onto a feature
        # map under a mixed-precision policy without a dtype mismatch.
        return keras.ops.cast(xy_grid_batched, self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: tuple
        :return: Output shape tuple ``(batch_size, height, width, 2)``.
        :rtype: tuple
        :raises ValueError: If the input shape is not 4D.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")

        # Output preserves batch and spatial dimensions, but has 2 channels for (x, y)
        return (input_shape[0], input_shape[1], input_shape[2], 2)

    def get_config(self) -> dict:
        """
        Return the configuration dictionary for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'resolution': self.resolution,
            'resize_method': self.resize_method,
            'alignment': self.alignment,
            'channel_order': self.channel_order,
            'normalization': self.normalization,
        })
        return config

# ---------------------------------------------------------------------
