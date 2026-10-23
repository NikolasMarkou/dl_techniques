"""Spatial smoothness regularization over a fixed 2D or 3D layout of a layer's units.

The problem
-----------
A transformer's units are indexed by nothing but their position on the last axis
of a weight matrix. Two units that end up encoding the same thing have no reason
to sit near each other, so a model's representation has no spatial structure to
read off. Topographic deep artificial neural networks fix this by giving the
units of a layer positions on a sheet of tissue and asking neighbouring units to
respond alike, which makes functional clustering emerge as a consequence of the
objective rather than as a post-hoc clustering of whatever the model happened to
learn. This module is that constraint, in the form the loss takes: a sampled
pairwise-correlation penalty.

The mechanism
-------------
Every unit of the tapped tensor is assigned a distinct cell of a ``h x w`` grid,
with ``h * w == num_units``, by a permutation that is drawn once per layer and
then frozen. With ``grid_dim=3`` (a repo-authored extension; the paper's layout
is 2D) the grid is an ``h x w x d`` box with ``h * w * d == num_units`` and a
neighbourhood is a cube. For one square (3D: cubic) neighbourhood of radius
``rho`` centred on the grid, covering ``P = (2*rho + 1)**grid_dim`` units, the
loss compares two things that should agree:

* ``r``, the Pearson correlation of every unit pair's activations across the
  ``M`` samples in the batch;
* ``d``, the inverse distance ``1 / (dist + 1)`` of the same pairs, mean-centred
  and unit-normalised.

.. code-block:: text

    SL = 0.5 * (1 - corr(r, d))          in [0, 1], 0 == smooth

The ``0.5`` is what puts the range at ``[0, 1]``: ``corr`` is in ``[-1, 1]``.
Minimising ``SL`` makes near units co-activate and pushes distant units apart in
correlation, which is an efficient proxy for short wiring length.

Two decisions that are not the obvious ones
-------------------------------------------
**The permutation is not cosmetic.** Without it the residual stream hands every
layer the same spatial pattern, the loss is satisfied by *copying one map forward*
rather than by each layer learning its own, and the resulting model has one map
rather than ``depth`` of them. It is on by default.

**The whole grid is never scored at once.** ``r`` for all ``N`` units is
``N*(N-1)/2`` numbers -- 306 936 of them at the paper's ``N = 784`` (a 2D
grid), computed from a ``K x M x P`` gather every step. The loss instead
samples ``K`` neighbourhoods per step, which is what makes it affordable, and is
why neighbourhood centres are restricted to those whose patch fits entirely
inside the grid: a constant ``P`` means ``d`` is identical for every
neighbourhood and can be precomputed once, at build time, and shared.

**Rank, axis order and cost in 3D.** ``grid_dim`` is a stated argument (2 or 3,
default 2) and is never inferred from ``len(grid_shape)``;
:func:`build_patch_tables` is the one place that reads the rank from its array
(``cell_to_unit.ndim``), so the two cannot disagree. ``grid_shape ==
cell_to_unit.shape``, row-major, and nothing reorders the axes. An implicit 3D
grid is the most cubic ``(h, w, d)`` with ``h <= w <= d`` (784 -> (7, 8, 14),
343 -> (7, 7, 7)), and a neighbourhood needs every axis at least ``2*rho + 1``
across. The pair count ``P*(P-1)/2`` and the correlation matmul
(``K * M * P**2``) grow with ``P**2``: a radius-5 cube (``P = 1331``, 885 115
pairs) is a factor of 121 over the paper's 2D neighbourhood (``P = 121``, 7260
pairs). Derived, not measured; measured numbers are in
``models/language/topolm/README.md``. ``GaussianReadout``
(``gaussian_readout.py``) shares this module's resolver and stays 2D only.

Numerics
--------
The correlation block runs at ``promote_types(input_dtype, float32)``. A hard
cast to ``float32`` is the wrong instruction under a ``float64`` policy, where it
*narrows* the reduction and costs about seven digits; under ``mixed_float16`` the
promotion is what keeps ``sum(x*x)`` over ``M`` samples from overflowing.
Dead or constant units are handled by an ``eps`` floor under the standard
deviation, which sends their correlations to ``0`` and therefore the loss to
``0.5`` -- finite, and finite in its gradient too.

Reuse sweep (why this is a new layer)
--------------------------------------
Checked before authoring, per the layer-reuse policy:

* :class:`~dl_techniques.layers.statistics.residual_acf.ResidualACFLayer` has
  exactly the right *shape* -- a pass-through tap that ``add_loss``\\ es on
  ``training is True`` -- but the wrong *term*. It penalises an autocorrelation of
  residuals against a target lag structure; this penalises pairwise correlation
  between units against an inverse-distance prior.
* ``layers/memory/SoftSOMLayer._topological_loss`` is the nearest existing
  ``1/(dist)``-weighted cross-batch correlation in the tree, and it is cited as a
  sibling in that layer. It is a soft-assignment *quantiser*: its correlation is
  between assignment profiles, and adopting it here would make the model a
  vector-quantised autoencoder rather than a regularised transformer.
* ``layers/generative/topographic_product.py`` carries reusable grid/window
  machinery, but in one dimension over capsules, not a square or cubic
  neighbourhood over a transformer's residual branches.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM:
      Brain-like spatio-functional organization in a topographic language model.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
    - Lee, Margalit, Jozwik, Cohen, Kanwisher & DiCarlo, 2020. Topographic deep
      artificial neural networks reproduce the hallmarks of the primate inferior
      temporal cortex face processing network. (https://arxiv.org/abs/2007.09019)
    - Margalit, Lee, Finzi, DiCarlo, Grill-Spector & Yamins, 2024. A unifying
      framework for functional organization in early and higher ventral visual
      cortex. Neuron 112(14).
    - Kriegeskorte, Borgwardt & Bhattacharyya, 2010. How does an fMRI voxel
      sample the neuronal activity pattern? (https://arxiv.org/abs/0906.2859)
"""

import math
from typing import Any, Dict, NamedTuple, Optional, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.dtype_policy import INDEX_WEIGHT_DTYPE

# ---------------------------------------------------------------------

#: Distance metrics a neighbourhood's inverse-distance vector can be built from.
#: ``'linf'`` (Chebyshev, max of the row/column offsets) is the metric TopoLM
#: trains with, and it is the same contiguity the post-hoc analysis uses to grow
#: clusters, so the objective and the measurement agree.
DISTANCE_METRICS = ("linf", "l1", "l2")

#: Float32 unit roundoff, used to report a measured ``d_unit`` normalisation.
_F32_EPS = float(np.finfo(np.float32).eps)


def _largest_divisor_at_most(num_units: int, bound: float) -> int:
    """Largest divisor of ``num_units`` that does not exceed ``bound``.

    Integer comparison throughout: ``isqrt(num_units)`` rather than a float
    ``sqrt``, so the choice of grid height cannot depend on how the platform
    rounds a square root.
    """
    limit = math.isqrt(num_units)
    for candidate in range(limit, 0, -1):
        if num_units % candidate == 0:
            return candidate
    return 1


def _factor_3d_cubic(num_units: int) -> Tuple[int, int, int]:
    """Factor ``num_units`` into the most cubic ``(h, w, d)`` triple.

    Returns the triple with ``h <= w <= d`` and ``h * w * d == num_units`` that
    minimises ``d - h``; among equal objectives the one with the larger ``h``
    wins. Integer arithmetic only: ``h`` ascends while ``h ** 3 <= num_units``
    (that loop bound IS the integer cube root, so a float ``**`` never decides a
    grid and no extra helper is needed), and ``w`` runs from ``h`` to
    ``isqrt(num_units // h)``. The `<=` below is what sends ties to the larger
    ``h``, because ``h`` ascends. :func:`_largest_divisor_at_most` is
    deliberately not reused: it ignores its ``bound`` argument.

    :param num_units: A positive integer.
    :type num_units: int
    :return: ``(h, w, d)``; ``(1, 1, num_units)`` when ``num_units`` is prime.
    :rtype: Tuple[int, int, int]
    """
    best_gap = None
    best = (1, 1, num_units)
    h = 1
    while h ** 3 <= num_units:
        if num_units % h == 0:
            rest = num_units // h
            for w in range(h, math.isqrt(rest) + 1):
                if rest % w == 0:
                    d = rest // w
                    if best_gap is None or d - h <= best_gap:
                        best_gap, best = d - h, (h, w, d)
        h += 1
    return best


def validate_grid_dim(grid_dim: Any) -> int:
    """Return ``grid_dim`` as an ``int`` if it is 2 or 3, else raise.

    Shared by :func:`resolve_grid_shape`, :class:`SpatialSmoothness` and the
    TopoLM block and model (``models/language/topolm``), so the accepted values
    and the message cannot drift apart. ``bool`` and non-integers
    (``"3"``, ``3.0``) are refused: a stringly or float rank would otherwise
    compare unequal to 3 further down and fall through as 2D.

    :param grid_dim: Candidate number of grid axes.
    :type grid_dim: Any
    :return: ``int(grid_dim)``, 2 or 3.
    :rtype: int
    :raises ValueError: ``grid_dim must be 2 or 3, got <repr>``.
    """
    if (
        isinstance(grid_dim, bool)
        or not isinstance(grid_dim, (int, np.integer))
        or grid_dim not in (2, 3)
    ):
        raise ValueError(f"grid_dim must be 2 or 3, got {grid_dim!r}")
    return int(grid_dim)


def resolve_grid_shape(
    num_units: int,
    grid_shape: Optional[Tuple[int, ...]] = None,
    grid_dim: int = 2,
) -> Tuple[int, ...]:
    """Resolve the grid a layout of ``num_units`` units sits on.

    This is the ONE place grid arithmetic lives. ``build``, ``call`` and
    ``compute_output_shape`` all route through it, so a change to the factoring
    rule cannot reach one of them and miss the others.

    :param num_units: Number of units on the last axis of the tapped tensor.
    :type num_units: int
    :param grid_shape: Explicit ``(height, width)`` for ``grid_dim=2`` or an
        explicit three-axis shape for ``grid_dim=3``, or ``None`` to factor
        ``num_units`` into the most square rectangle (2D) or the most cubic box
        (3D, :func:`_factor_3d_cubic`) available.
    :type grid_shape: Optional[Tuple[int, ...]]
    :param grid_dim: Number of grid axes, ``2`` or ``3``. The rank is always this
        stated value: it is never inferred from ``len(grid_shape)``, so a caller
        that asks for the default 2D rank and passes a triple is refused.
    :type grid_dim: int
    :return: The resolved shape, ``grid_dim`` entries whose product is
        ``num_units``. For ``grid_dim=3`` the axes are in the order given (or
        ``h <= w <= d`` when factored) and nothing downstream reorders them.
    :rtype: Tuple[int, ...]
    :raises ValueError: If ``num_units`` is not positive, if ``grid_dim`` is not
        2 or 3, if ``grid_shape`` does not have ``grid_dim`` positive integer
        entries, or if its product is not ``num_units`` -- naming the offending
        value in each case.
    """
    if num_units <= 0:
        raise ValueError(
            f"num_units must be positive, got {num_units}"
        )
    grid_dim = validate_grid_dim(grid_dim)

    if grid_shape is None:
        if grid_dim == 3:
            return _factor_3d_cubic(num_units)
        height = _largest_divisor_at_most(num_units, math.isqrt(num_units))
        return height, num_units // height

    # DECISION plan-2026-10-23T141429-390f3815/D-004
    # The rank is the stated ``grid_dim`` and is checked against the length of
    # ``grid_shape``; it is NEVER inferred from that length. Do NOT "simplify"
    # this to ``grid_dim = len(grid_shape)``: ``GaussianReadout`` calls this with
    # the default rank and relies on a 3-tuple raising here, because it would
    # otherwise accept a 3D layout and fail inside its ``Conv2D``. Guarded by
    # test_gaussian_readout.py::TestGridRankRefusal and
    # test_spatial_smoothness_3d.py::TestResolveGridShape3D::
    # test_a_pair_is_refused_when_the_rank_is_three.
    if len(tuple(grid_shape)) != grid_dim:
        if grid_dim == 2:
            raise ValueError(
                f"grid_shape must be a (height, width) pair, got {tuple(grid_shape)}"
            )
        raise ValueError(
            f"grid_shape must have grid_dim={grid_dim} entries, got "
            f"{tuple(grid_shape)}"
        )
    shape = tuple(int(v) for v in grid_shape)
    if min(shape) <= 0:
        raise ValueError(
            f"grid_shape entries must be positive, got "
            f"({', '.join(str(v) for v in shape)})"
        )
    cells = math.prod(shape)
    if cells != num_units:
        raise ValueError(
            f"grid_shape {'x'.join(str(v) for v in shape)} has {cells} cells "
            f"but the tapped tensor has {num_units} units; the layout must be "
            f"a bijection. Pass grid_shape=None to factor {num_units}."
        )
    return shape


def permutation_seed(base_seed: Optional[int], tap_index: int) -> int:
    """Derive the layout seed of one tap from the model's base seed.

    Distinct taps must get distinct permutations, and the derivation has to be
    reproducible from configuration alone -- a reloaded model cannot consult a
    weight to work out which layout it is supposed to have, and a caller who
    never writes the per-tap seeds down must still get the same model twice.

    :param base_seed: The model's base seed, or ``None`` to derive from
        ``tap_index`` alone.
    :type base_seed: Optional[int]
    :param tap_index: Index of this tap within the model's tap sequence.
    :type tap_index: int
    :return: A non-negative integer seed.
    :rtype: int
    """
    if tap_index < 0:
        raise ValueError(f"tap_index must be non-negative, got {tap_index}")
    if base_seed is None:
        return tap_index
    return int(base_seed) * 1000 + int(tap_index)


class SpatialLayout:
    """A bijection between a tensor's last axis and the cells of a 2D or 3D grid.

    Unit ``i`` sits at grid cell ``perm[i]``, row-major (the last grid axis
    varies fastest). ``cell_to_unit`` is the inverse, reshaped to the grid, and is
    what post-hoc analysis indexes with when it grows clusters or computes a
    spatial statistic.

    One axis order everywhere: ``grid_shape == cell_to_unit.shape``, and
    ``positions[:, k]`` is the coordinate along axis ``k`` of that same shape.
    The axes carry no names (no height/width/depth) and nothing reorders them.

    The layout is pure NumPy and holds no tensors: it is a description, and the
    layer that owns it decides whether to keep it as weights.

    Example:
        .. code-block:: python

            layout = SpatialLayout(784, seed=0)          # 28x28
            grid = layout.to_grid(acts)                  # (..., 28, 28)
            back = layout.from_grid(grid)                # (..., 784)

            volume = SpatialLayout(784, seed=0, grid_dim=3)   # 7x8x14
            vol = volume.to_grid(acts)                   # (..., 7, 8, 14)

    :param num_units: Number of units, i.e. the size of the last axis.
    :type num_units: int
    :param grid_shape: Explicit shape with ``grid_dim`` entries, or ``None`` to
        factor (most square for 2D, most cubic for 3D).
    :type grid_shape: Optional[Tuple[int, ...]]
    :param permute: Whether to draw a random permutation of the unit order. Set
        ``False`` to place unit ``i`` at cell ``i``; that is the paper's Fig. 12
        ablation, kept because without it the residual stream satisfies the loss
        by propagating one map through every layer.
    :type permute: bool
    :param seed: Seed for the permutation. The permutation is drawn before and
        independently of the grid rank, so a seed gives the same ``perm`` for
        ``grid_dim`` 2 and 3.
    :type seed: Optional[int]
    :param grid_dim: Number of grid axes, ``2`` (default) or ``3``.
    :type grid_dim: int
    :raises ValueError: If ``num_units`` is not positive, ``grid_dim`` is not 2
        or 3, or ``grid_shape`` does not describe ``num_units`` cells on
        ``grid_dim`` axes. See :func:`resolve_grid_shape`.
    """

    def __init__(
        self,
        num_units: int,
        grid_shape: Optional[Tuple[int, ...]] = None,
        permute: bool = True,
        seed: Optional[int] = None,
        grid_dim: int = 2,
    ) -> None:
        self.grid_shape = resolve_grid_shape(num_units, grid_shape, grid_dim)
        self.grid_dim = len(self.grid_shape)
        self.num_units = int(num_units)
        self.permute = bool(permute)
        self.seed = seed

        if permute:
            self.perm = np.random.default_rng(seed).permutation(
                self.num_units
            ).astype("int32")
        else:
            self.perm = np.arange(self.num_units, dtype="int32")

        # DECISION plan-2026-10-23T141429-390f3815/D-003
        # ONE axis order: ``positions`` columns, ``grid_shape`` and
        # ``cell_to_unit.shape`` all follow the same row-major axes, last axis
        # fastest. Do NOT reorder, transpose or rename axes (no "depth, height,
        # width" translation) here or in any consumer: a centre count or a shape
        # assert is symmetric under an axis permutation, so a mirrored layout
        # would pass every structural test and silently score the wrong
        # neighbourhoods. Guarded by test_spatial_smoothness_3d.py::
        # TestSpatialLayout3D::test_positions_are_row_major_with_axis_2_fastest
        # and ::test_positions_index_cell_to_unit_back_to_the_unit (both go RED
        # when the positions axes are reversed), and by TestPatchTables3D::
        # test_a_non_cubic_grid_pins_the_axis_order_by_value for the tables.
        self.positions = np.stack(
            np.unravel_index(self.perm, self.grid_shape), axis=-1
        ).astype("int32")
        self.cell_to_unit = np.argsort(self.perm).reshape(
            self.grid_shape
        ).astype("int32")

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        """Scatter a ``(..., num_units)`` array onto the ``(..., *grid_shape)`` grid.

        :param values: Array whose last axis indexes units.
        :type values: numpy.ndarray
        :return: The same values with the last axis replaced by the
            ``grid_dim`` grid axes.
        :rtype: numpy.ndarray
        :raises ValueError: If the last axis is not ``num_units``.
        """
        values = np.asarray(values)
        if values.shape[-1] != self.num_units:
            raise ValueError(
                f"to_grid expects a last axis of {self.num_units} units, got "
                f"{values.shape[-1]} for an array of shape {values.shape}"
            )
        return values[..., self.cell_to_unit.reshape(-1)].reshape(
            *values.shape[:-1], *self.grid_shape
        )

    def from_grid(self, grid: np.ndarray) -> np.ndarray:
        """Gather a ``(..., *grid_shape)`` grid back onto the ``(..., num_units)`` axis.

        Exact inverse of :meth:`to_grid`.

        :param grid: Array whose last ``grid_dim`` axes are the grid.
        :type grid: numpy.ndarray
        :return: The same values with the grid axes replaced by the unit axis.
        :rtype: numpy.ndarray
        :raises ValueError: If the last ``grid_dim`` axes are not the grid shape.
        """
        grid = np.asarray(grid)
        rank = len(self.grid_shape)
        if tuple(grid.shape[-rank:]) != tuple(self.grid_shape):
            raise ValueError(
                f"from_grid expects trailing axes {self.grid_shape}, got "
                f"{tuple(grid.shape[-rank:])} for an array of shape {grid.shape}"
            )
        return grid.reshape(*grid.shape[:-rank], self.num_units)[..., self.perm]

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this layout."""
        return {
            "num_units": self.num_units,
            "grid_shape": tuple(int(v) for v in self.grid_shape),
            "permute": self.permute,
            "seed": self.seed,
            "grid_dim": self.grid_dim,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpatialLayout":
        """Rebuild a layout from :meth:`get_config` output.

        A config written before ``grid_dim`` existed has no such key and is a 2D
        layout, so the key defaults to ``2``.
        """
        return cls(
            num_units=config["num_units"],
            grid_shape=tuple(config["grid_shape"]),
            permute=config.get("permute", True),
            seed=config.get("seed"),
            grid_dim=config.get("grid_dim", 2),
        )


class NeighborhoodTables(NamedTuple):
    """Precomputed, shareable tables for one neighbourhood geometry.

    One row of ``patch_table`` is the unit ids of one square (2D) or cubic (3D)
    neighbourhood, in row-major patch order. Because every patch has the same shape and the same
    geometry, ``d_unit`` does not depend on which patch was drawn -- it is
    computed once and shared by every row, which is the entire reason the centre
    restriction in :func:`build_patch_tables` exists.

    ``side`` and ``patch_units`` are different numbers and conflating them is the
    easy mistake: a radius-5 patch has side ``11`` and contains ``121`` units,
    and a row of ``patch_table`` is 121 wide while ``triu_flat`` indexes a 121x121
    matrix. At the paper's settings those are 324x11, 324x121 and 14641. In 3D
    the same radius has side ``11`` and ``1331`` units.

    :param patch_table: ``(num_centers, patch_units)`` int32 unit ids.
    :param triu_flat: ``(num_pairs,)`` int32 flat indices of the strict upper
        triangle of a ``(patch_units, patch_units)`` matrix.
    :param d_unit: ``(num_pairs,)`` float32, mean-centred and unit-normalised
        inverse distances.
    :param grid_shape: The full ``grid_dim``-entry grid the patches were cut from
        (``(height, width)`` in 2D).
    :param radius: Neighbourhood radius.
    :param side: Patch side length ``2 * radius + 1``.
    :param patch_units: ``side ** grid_dim``, the width of one ``patch_table``
        row.
    :param num_pairs: ``patch_units * (patch_units - 1) / 2``.
    :param num_centers: Number of admissible patch centres.
    :param grid_dim: Number of grid axes, ``2`` or ``3``. Trailing and defaulted
        so that positional construction of a 2D record is unchanged.
    """

    patch_table: np.ndarray
    triu_flat: np.ndarray
    d_unit: np.ndarray
    grid_shape: Tuple[int, ...]
    radius: int
    side: int
    patch_units: int
    num_pairs: int
    num_centers: int
    grid_dim: int = 2


def build_patch_tables(
    cell_to_unit: np.ndarray,
    radius: int,
    distance: str = "linf",
) -> NeighborhoodTables:
    """Enumerate every full square or cubic neighbourhood of a grid and its prior.

    The rank is the array's own: ``grid_dim = cell_to_unit.ndim``, so the tables
    cannot disagree with the grid they were cut from. Centres run over
    ``radius .. n - 1 - radius`` on every axis, so a patch never hangs off the
    edge. That keeps ``patch_units`` constant, which keeps ``d_unit``
    well-defined as a single shared vector rather than a per-centre table.
    Centres are enumerated lexicographically (first axis slowest), which in 2D is
    the row-major order of the original nested loop.

    :param cell_to_unit: ``(height, width)`` or three-axis int array of unit ids;
        its axes are the grid axes in the order of :class:`SpatialLayout`.
    :type cell_to_unit: numpy.ndarray
    :param radius: Neighbourhood radius; the patch side is ``2 * radius + 1``.
    :type radius: int
    :param distance: One of :data:`DISTANCE_METRICS`. ``'l2'`` is the Euclidean
        offset in either rank (3D adds the third squared offset).
    :type distance: str
    :return: The tables, with ``d_unit`` already centred and normalised in
        float64 before the float32 cast.
    :rtype: NeighborhoodTables
    :raises ValueError: If ``cell_to_unit`` does not have 2 or 3 axes, if
        ``radius`` is not positive, if ``distance`` is unknown, or if ANY axis is
        shorter than the patch side (the message names every short axis; a
        total-size check would accept ``4 x 9 x 9``, which holds no 5x5x5 cube).
    """
    cell_to_unit = np.asarray(cell_to_unit)
    if cell_to_unit.ndim not in (2, 3):
        raise ValueError(
            f"cell_to_unit must have 2 or 3 axes ((height, width) or a "
            f"3-axis grid), got shape {cell_to_unit.shape}"
        )
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    if distance not in DISTANCE_METRICS:
        raise ValueError(
            f"distance must be one of {list(DISTANCE_METRICS)}, got {distance!r}"
        )

    grid_dim = cell_to_unit.ndim
    shape = tuple(int(n) for n in cell_to_unit.shape)
    side = 2 * radius + 1
    if min(shape) < side:
        head = (
            f"a radius-{radius} neighbourhood needs a "
            f"{'x'.join([str(side)] * grid_dim)} patch, but the grid is "
            f"{'x'.join(str(n) for n in shape)}"
        )
        if grid_dim == 2:
            raise ValueError(
                f"{head}. Lower `radius`, or use a wider `grid_shape` (the "
                f"layout needs at least {side ** 2} units)."
            )
        short = "; ".join(
            f"axis {k} has {n}" for k, n in enumerate(shape) if n < side
        )
        raise ValueError(
            f"{head}: {short}, and every axis needs at least {side}. Lower "
            f"`radius`, or use a wider `grid_shape` (the layout needs at least "
            f"{side ** grid_dim} units, with every axis >= {side})."
        )

    rows = [
        cell_to_unit[tuple(slice(o, o + side) for o in offset)].reshape(-1)
        for offset in np.ndindex(*(n - 2 * radius for n in shape))
    ]
    patch_table = np.stack(rows).astype("int32")
    patch_units = side ** grid_dim

    offsets = np.unravel_index(np.arange(patch_units), (side,) * grid_dim)
    deltas = [np.abs(o[:, None] - o[None, :]) for o in offsets]
    dy, dx = deltas[0], deltas[1]
    if distance == "linf":
        separation = np.maximum(dy, dx)
        for extra in deltas[2:]:
            separation = np.maximum(separation, extra)
    elif distance == "l1":
        separation = dy + dx
        for extra in deltas[2:]:
            separation = separation + extra
    else:
        squared = dy ** 2 + dx ** 2
        for extra in deltas[2:]:
            squared = squared + extra ** 2
        separation = np.sqrt(squared)
    inverse_distance = 1.0 / (separation.astype("float64") + 1.0)

    upper = np.triu_indices(patch_units, k=1)
    triu_flat = (upper[0] * patch_units + upper[1]).astype("int32")
    d_pairs = inverse_distance[upper]

    # Centring and normalising once, here, is what keeps the traced step down to a
    # single dot product. Computing the norm per step would give the same number
    # from a second producer of the same quantity, and the two would drift.
    d_pairs = d_pairs - d_pairs.mean()
    d_norm = float(np.sqrt(np.sum(d_pairs ** 2)))
    if d_norm == 0.0:
        raise ValueError(
            f"the inverse-distance vector for radius={radius} under "
            f"distance={distance!r} is constant, so it carries no spatial "
            f"information; the neighbourhood is degenerate."
        )
    d_unit = (d_pairs / d_norm).astype("float32")

    return NeighborhoodTables(
        patch_table=patch_table,
        triu_flat=triu_flat,
        d_unit=d_unit,
        grid_shape=shape,
        radius=int(radius),
        side=int(side),
        patch_units=int(patch_units),
        num_pairs=int(triu_flat.shape[0]),
        num_centers=int(patch_table.shape[0]),
        grid_dim=int(grid_dim),
    )


def spatial_smoothness_loss(
    activations: keras.KerasTensor,
    patch_indices: keras.KerasTensor,
    triu_flat: keras.KerasTensor,
    d_unit: keras.KerasTensor,
    eps: float = 1e-6,
) -> keras.KerasTensor:
    """Penalise correlation that does not track inverse distance.

    Pure function of its arguments so the transcribed reference in
    ``tests/test_layers/test_regularization/spatial_smoothness_oracle.py`` can be
    compared against it term for term.

    The gather lands as ``(M, K, P)`` and is transposed to ``(K, P, M)`` -- units
    before samples -- because ``matmul`` reduces over its last two axes. Left as
    ``(K, M, P)`` it contracts the samples away and returns ``(K, M, M)``, whose
    flat layout is then read at ``P x P`` offsets and silently yields
    uncorrelated garbage: every map scores ``0.5``, which is exactly the value a
    correct implementation gives for noise, so nothing looks broken.

    :param activations: ``(num_samples, num_units)`` activations, at least
        ``float32`` wide.
    :type activations: keras.KerasTensor
    :param patch_indices: ``(num_neighborhoods, patch_units)`` unit ids.
    :type patch_indices: keras.KerasTensor
    :param triu_flat: ``(num_pairs,)`` flat upper-triangle indices.
    :type triu_flat: keras.KerasTensor
    :param d_unit: ``(num_pairs,)`` centred, unit-normalised inverse distances.
    :type d_unit: keras.KerasTensor
    :param eps: Floor under the per-unit standard deviation, which is what keeps
        a dead or constant unit finite.
    :type eps: float
    :return: Scalar in ``[0, 1]``; ``0`` means the correlation vector tracks the
        distance prior exactly.
    :rtype: keras.KerasTensor
    """
    gathered = keras.ops.take(activations, patch_indices, axis=1)
    gathered = keras.ops.transpose(gathered, (1, 2, 0))

    gathered = gathered - keras.ops.mean(gathered, axis=2, keepdims=True)
    gathered = gathered / (keras.ops.std(gathered, axis=2, keepdims=True) + eps)

    num_samples = keras.ops.cast(keras.ops.shape(gathered)[2], gathered.dtype)
    correlations = keras.ops.matmul(gathered, keras.ops.transpose(gathered, (0, 2, 1)))
    correlations = correlations / num_samples

    num_patches = keras.ops.shape(correlations)[0]
    flat = keras.ops.reshape(correlations, (num_patches, -1))
    pairs = keras.ops.take(flat, triu_flat, axis=1)
    pairs = pairs - keras.ops.mean(pairs, axis=1, keepdims=True)

    # Read the prior through the SAME working dtype as the correlations. Under
    # `mixed_float16`, reading the float32 `d_unit` variable inside an autocast
    # scope yields a float16 tensor, and multiplying a float32 norm by it raises a
    # dtype error -- the variable's dtype and the value's dtype are different
    # things, and only the cast reconciles them.
    prior = keras.ops.cast(d_unit, pairs.dtype)
    numerator = keras.ops.sum(pairs * prior, axis=1)
    denominator = keras.ops.norm(pairs, axis=1) * keras.ops.norm(prior) + eps
    return keras.ops.mean(0.5 * (1.0 - numerator / denominator))


@register_dl_technique(
    "dl_techniques.layers.regularization.spatial_smoothness"
)
class SpatialSmoothness(keras.layers.Layer):
    """Identity tap that adds a spatial smoothness penalty on a layer's units.

    The forward pass returns its input unchanged; all of the work happens in the
    training-only loss. Place it on a tensor whose last axis is the model's unit
    dimension and where spatial neighbours *should* respond alike -- on the
    residual-path output of an attention or feed-forward branch, before
    normalization and before the residual add.

    .. code-block:: text

        x ──► norm ──► attention ──► attn_out
                                       │
                                  ┌────┴────┐
                                  │   TAP   │  identity forward,
                                  └────┬────┘  alpha * SL when training
                                       ▼
        x ──────────────────────► x + attn_out

    :param alpha: Weight on the added loss. ``0`` is a real control, not a no-op:
        the tap still builds its tables, so an ``alpha=0`` model has an identical
        weight set and derives its layouts identically to an ``alpha>0`` one.
    :type alpha: float
    :param radius: Neighbourhood radius; the patch side is ``2 * radius + 1``.
    :type radius: int
    :param num_neighborhoods: Neighbourhoods sampled per step, averaged.
    :type num_neighborhoods: int
    :param distance: Distance metric behind the prior. One of
        :data:`DISTANCE_METRICS`.
    :type distance: str
    :param permute: Whether to draw a random unit permutation. ``False`` is the
        paper's Fig. 12 ablation; see :class:`SpatialLayout`.
    :type permute: bool
    :param grid_shape: Explicit shape with ``grid_dim`` entries (``(height,
        width)`` for ``grid_dim=2``), or ``None`` to factor the unit count into
        the most square rectangle (``grid_dim=2``) or the most cubic box
        (``grid_dim=3``) available.
    :type grid_shape: Optional[Tuple[int, ...]]
    :param seed: Seed for the layout permutation and the neighbourhood sampling.
    :type seed: Optional[int]
    :param max_samples: Cap on the ``M`` sample rows used for the correlation, or
        ``None`` to use every ``batch * sequence`` position. A cap keeps the cost
        ``K * M * P**2`` bounded when sequences are long.
    :type max_samples: Optional[int]
    :param eps: Floor under the per-unit standard deviation. Must exceed ``0``;
        it is the only thing standing between a constant unit and a NaN.
    :type eps: float
    :param grid_dim: Number of grid axes, ``2`` (default, the paper's layout) or
        ``3``. With ``3`` the units are laid out as a volume and a neighbourhood
        is a cube of ``(2 * radius + 1) ** 3`` units. Stated, never inferred from
        ``grid_shape``; a config written before this keyword existed has no such
        key and builds as 2D.
    :type grid_dim: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :raises ValueError: If ``alpha`` is negative, ``radius`` or
        ``num_neighborhoods`` is below ``1``, ``distance`` is unknown, ``eps`` is
        not positive, ``max_samples`` is below ``2``, ``grid_dim`` is not 2 or 3,
        or ``grid_shape`` is malformed -- naming the offending value in each
        case.
    """

    def __init__(
        self,
        alpha: float = 2.5,
        radius: int = 5,
        num_neighborhoods: int = 5,
        distance: str = "linf",
        permute: bool = True,
        grid_shape: Optional[Tuple[int, ...]] = None,
        seed: Optional[int] = None,
        max_samples: Optional[int] = None,
        eps: float = 1e-6,
        grid_dim: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        if num_neighborhoods < 1:
            raise ValueError(
                f"num_neighborhoods must be >= 1, got {num_neighborhoods}"
            )
        if distance not in DISTANCE_METRICS:
            raise ValueError(
                f"distance must be one of {list(DISTANCE_METRICS)}, "
                f"got {distance!r}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        grid_dim = validate_grid_dim(grid_dim)
        if max_samples is not None and max_samples < 2:
            # Fewer than two samples leaves the per-unit standard deviation at
            # zero, so every correlation is 0/0 and the layer reports 0.5 while
            # measuring nothing.
            raise ValueError(
                f"max_samples must be >= 2 (a correlation needs at least two "
                f"samples), got {max_samples}"
            )

        self.alpha = float(alpha)
        self.radius = int(radius)
        self.num_neighborhoods = int(num_neighborhoods)
        self.distance = distance
        self.permute = bool(permute)
        self.grid_shape = (
            None if grid_shape is None else tuple(int(v) for v in grid_shape)
        )
        self.seed = seed
        self.max_samples = max_samples
        self.eps = float(eps)
        self.grid_dim = grid_dim

        self.seed_generator = keras.random.SeedGenerator(seed)

        # Layout and tables. They are resolved in build(), where the unit count is
        # known, and kept as weights rather than as Python state: a checkpoint
        # that does not carry its own layout silently disagrees with the one it
        # was trained under. Declared here so the attributes exist (and read as
        # None) before build.
        self.layout = None
        self.perm = None
        self.patch_table = None
        self.triu_flat = None
        self.d_unit = None
        self.last_spatial_loss = None
        self._num_units = None
        self._grid_shape = None
        self._num_centers = None
        self._patch_units = None
        self._num_pairs = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Resolve the layout and materialise the neighbourhood tables as weights.

        Every table is produced inside an ``add_weight`` initializer closure.
        Computing them here and calling ``.assign()`` would look equivalent and
        is not: Keras runs this pass inside a ``StatelessScope`` whenever the
        layer is first reached from a parent's ``call()``, the scope records the
        assignment and discards it, and the table stays at its initializer value
        in every real model.

        ``perm``, ``patch_table`` and ``triu_flat`` are int64 weights
        (``utils.dtype_policy.INDEX_WEIGHT_DTYPE``), not int32 ones: TensorFlow
        keeps an int32 variable in host memory, where an XLA-compiled step on a
        GPU cannot read it. The NumPy tables they are built from stay int32.

        :param input_shape: Shape of the tapped tensor; only the last axis is read.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last axis is not statically known, or if the
            resolved grid cannot hold a ``radius`` neighbourhood.
        """
        num_units = input_shape[-1]
        if num_units is None:
            raise ValueError(
                f"{self.__class__.__name__} needs a statically known last axis to "
                f"resolve its unit layout, got input_shape={tuple(input_shape)}. "
                f"Build the tapped tensor with a fixed width, or pass an explicit "
                f"unit count."
            )

        grid_shape = resolve_grid_shape(num_units, self.grid_shape, self.grid_dim)
        layout = SpatialLayout(
            num_units=num_units,
            grid_shape=grid_shape,
            permute=self.permute,
            seed=self.seed,
            grid_dim=self.grid_dim,
        )
        # ``build_patch_tables`` reads the rank from ``layout.cell_to_unit.ndim``,
        # so the tables cannot disagree with the layout they were cut from.
        tables = build_patch_tables(
            layout.cell_to_unit, self.radius, self.distance
        )

        self._num_units = int(num_units)
        self._grid_shape = grid_shape
        self._num_centers = tables.num_centers
        self._patch_units = tables.patch_units
        self._num_pairs = tables.num_pairs
        self.layout = layout

        self.perm = self.add_weight(
            name="perm",
            shape=(num_units,),
            initializer=_table_initializer(layout.perm),
            trainable=False,
            dtype=INDEX_WEIGHT_DTYPE,
        )
        self.patch_table = self.add_weight(
            name="patch_table",
            shape=(tables.num_centers, tables.patch_units),
            initializer=_table_initializer(tables.patch_table),
            trainable=False,
            dtype=INDEX_WEIGHT_DTYPE,
        )
        self.triu_flat = self.add_weight(
            name="triu_flat",
            shape=(tables.num_pairs,),
            initializer=_table_initializer(tables.triu_flat),
            trainable=False,
            dtype=INDEX_WEIGHT_DTYPE,
        )
        self.d_unit = self.add_weight(
            name="d_unit",
            shape=(tables.num_pairs,),
            initializer=_table_initializer(tables.d_unit),
            trainable=False,
            dtype="float32",
        )
        self.last_spatial_loss = self.add_weight(
            name="last_spatial_loss",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )

        logger.info(
            f"SpatialSmoothness: {num_units} units on a "
            f"{'x'.join(str(axis) for axis in grid_shape)} grid, "
            f"{tables.num_centers} admissible radius-"
            f"{self.radius} neighbourhoods of "
            f"{'x'.join([str(tables.side)] * len(grid_shape))} = "
            f"{tables.patch_units} units ({tables.num_pairs} pairs each), "
            f"alpha={self.alpha}, distance={self.distance!r}, "
            f"permute={self.permute}"
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Return ``inputs`` unchanged, adding the penalty when training.

        :param inputs: The tapped tensor, ``(..., num_units)``.
        :type inputs: keras.KerasTensor
        :param training: Training flag. The loss is added only when this is
            exactly ``True``, so validation loss stays the pure task loss.
        :type training: Optional[bool]
        :return: ``inputs``, unchanged.
        :rtype: keras.KerasTensor
        """
        if training is True and self.alpha > 0.0 and not isinstance(
            inputs, keras.KerasTensor
        ):
            # Skips SYMBOLIC inputs. What that buys, measured on Keras 3.8 / TF
            # backend rather than assumed: nothing observable. With the guard
            # REMOVED, the loss is unchanged on every path tried -- three that
            # hand the tap a `KerasTensor` (a bare `keras.Input`, a subclassed
            # model called with one, and a functional model wrapping the tap) all
            # end with an empty `losses` either way, because Keras discards the
            # losses collected during a symbolic-input call. So the guard is
            # belt-and-braces, not load-bearing.
            #
            # It is kept for two reasons. The first is that the skip is cheap and
            # the alternative -- a sampled reduction (7260 terms per
            # neighbourhood at the paper's 2D settings) against an
            # unrealisable shape -- is work whose result is thrown away. The
            # second is that the ORIGINAL justification, recorded in an earlier
            # draft of this comment, was that `add_loss` accepts a `KerasTensor`,
            # stores it, and the stored tensor never becomes a value, so a
            # training loss would silently lack the spatial term. That could NOT
            # be reproduced and was withdrawn. Two parts of it were wrong in
            # opposite directions: nothing is left in `losses`, and inside a
            # `tf.function` -- the path a training step actually takes -- the tap
            # receives a CONCRETE traced tensor, the guard does not apply, and the
            # term is added and stored exactly as it should be.
            #
            # A third thing this comment used to assert, that the guard is
            # reachable, is not asserted anywhere in the test suite because it
            # proved PATH-DEPENDENT: identical code saw a `KerasTensor` in one
            # context and an eager tensor in another.
            #
            # What the guard does NOT do is protect the model below: its `build()`
            # materialises the tree without a forward pass, so no tap is reached
            # with a `KerasTensor` in that model's lifecycle. MEASURED: 0 tap
            # calls after `TopoLM.build(...)`, 4 eager calls and 4 losses after the
            # first `training=True` step.
            flat = keras.ops.reshape(
                keras.ops.cast(inputs, _promoted_dtype(inputs.dtype)),
                (-1, self._num_units),
            )

            num_samples = keras.ops.shape(flat)[0]
            if self.max_samples is not None:
                rows = keras.random.randint(
                    (self.max_samples,),
                    0,
                    num_samples,
                    seed=self.seed_generator,
                )
                flat = keras.ops.take(flat, rows, axis=0)

            centers = keras.random.randint(
                (self.num_neighborhoods,),
                0,
                self._num_centers,
                seed=self.seed_generator,
            )
            patches = keras.ops.take(self.patch_table, centers, axis=0)

            penalty = spatial_smoothness_loss(
                flat, patches, self.triu_flat, self.d_unit, self.eps
            )
            self.add_loss(self.alpha * penalty)
            self.last_spatial_loss.assign(penalty)

        return inputs

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``input_shape`` unchanged; the tap is the identity."""
        return tuple(input_shape)

    def reset_statistics(self) -> None:
        """Zero the recorded loss, so a logger can measure a fresh window."""
        if self.last_spatial_loss is not None:
            self.last_spatial_loss.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        The layout is deliberately absent: it travels as weights, so a reloaded
        layer restores the permutation it was trained with rather than the one
        this seed would draw today.
        """
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "radius": self.radius,
            "num_neighborhoods": self.num_neighborhoods,
            "distance": self.distance,
            "permute": self.permute,
            "grid_shape": self.grid_shape,
            "seed": self.seed,
            "max_samples": self.max_samples,
            "eps": self.eps,
            "grid_dim": self.grid_dim,
        })
        return config


def _promoted_dtype(dtype: Any) -> str:
    """``float32`` at minimum, wider if the input already is.

    A bare cast to ``float32`` is the wrong instruction under a ``float64``
    policy, where it narrows the reduction instead of widening it -- measured at
    about seven digits of the result. This is pure dtype arithmetic, done in
    NumPy so it works for every backend; ``bfloat16`` is handled by hand because
    NumPy does not know it.
    """
    name = str(dtype)
    if name == "bfloat16":
        return "float32"
    try:
        return str(np.promote_types(name, "float32"))
    except TypeError:
        return "float32"


def _table_initializer(values: np.ndarray) -> Any:
    """Build an ``add_weight`` initializer that returns a NumPy constant.

    The table is materialised in NumPy and handed to Keras, which converts it.
    Wrapping it in ``keras.ops.convert_to_tensor`` and closing over the result would
    bind the tensor to whichever graph happened to be tracing. The value is cast
    to the weight's dtype here, so an int32 NumPy table lands in an int64
    weight without a conversion inside Keras.
    """
    frozen = np.array(values, copy=True)

    def _initialize(shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        table = frozen.reshape(shape)
        return table if dtype is None else table.astype(dtype)

    return _initialize