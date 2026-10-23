"""Independent transcriptions of the spatial smoothness objective, for testing.

An oracle written by the same hand as the implementation is a second copy of it:
it agrees when both are wrong and disagrees when only one is, and nothing about
reading it tells you which. These are therefore transcribed from the *paper's*
Eq. 1 and its neighbourhood definition, in the plainest NumPy that can express
them, with the vectorised implementation nowhere in sight.

The two entry points differ deliberately:

* :func:`reference_smoothness_loss` follows the paper exactly -- explicit nested
  loops over every neighbourhood and every pair, and a Pearson correlation
  recomputed from first principles. Slow, and therefore not the thing under test.
* :func:`smooth_field_activations` and :func:`graded_field_activations` build
  the *stimuli* whose spatial scale the loss is supposed to be sensitive to.
  They matter as much as the loss: the first version of these fixtures made a
  constant-per-unit pattern plus independent noise, whose unit-to-unit variation
  is dominated by the independent part, so a perfectly smooth field scored
  exactly what noise scores. A field has to vary ACROSS SAMPLES with a spatially
  smooth structure for the loss to have anything to detect.

References:
    - Rathi et al., 2025. TopoLM, Eq. 1 and Section 3. (https://arxiv.org/abs/2410.11516)
"""

from typing import Tuple

import numpy as np


def reference_smoothness_loss(
    activations: np.ndarray,
    patch_units: np.ndarray,
    d_unit: np.ndarray,
) -> float:
    """Transcription of Eq. 1, with explicit loops and no shared helper.

    ``SL = 0.5 * (1 - corr(r, d))`` where ``r`` is the vector of pairwise Pearson
    correlations of unit activations across the batch and ``d`` the vector of
    pairwise inverse distances.

    :param activations: ``(num_samples, num_units)`` activations.
    :type activations: numpy.ndarray
    :param patch_units: ``(patch_units_per_neighborhood,)`` unit ids of ONE
        neighbourhood.
    :type patch_units: numpy.ndarray
    :param d_unit: ``(num_pairs,)`` the matching inverse distances.
    :type d_unit: numpy.ndarray
    :return: The loss for that one neighbourhood.
    :rtype: float
    """
    values = np.asarray(activations, dtype=np.float64)[:, patch_units]
    num_units = values.shape[1]

    correlations = []
    for i in range(num_units):
        for j in range(i + 1, num_units):
            column_i = values[:, i]
            column_j = values[:, j]
            centred_i = column_i - column_i.mean()
            centred_j = column_j - column_j.mean()
            denominator = np.sqrt((centred_i ** 2).sum()) * np.sqrt(
                (centred_j ** 2).sum()
            )
            correlations.append(
                float((centred_i * centred_j).sum() / denominator)
                if denominator > 0.0
                else 0.0
            )

    r = np.array(correlations, dtype=np.float64)
    d = np.asarray(d_unit, dtype=np.float64)
    r_centred = r - r.mean()
    d_centred = d - d.mean()
    denominator = np.sqrt((r_centred ** 2).sum()) * np.sqrt((d_centred ** 2).sum())
    pearson = (
        float((r_centred * d_centred).sum() / denominator)
        if denominator > 0.0
        else 0.0
    )
    return float(0.5 * (1.0 - pearson))


def reference_patch_units(
    cell_to_unit: np.ndarray, center: Tuple[int, int], radius: int
) -> np.ndarray:
    """Unit ids of the square patch around one grid centre, row-major.

    :param cell_to_unit: ``(height, width)`` unit ids.
    :type cell_to_unit: numpy.ndarray
    :param center: ``(row, col)`` of the patch centre.
    :type center: Tuple[int, int]
    :param radius: Patch radius.
    :type radius: int
    :return: The patch's unit ids.
    :rtype: numpy.ndarray
    """
    row, col = center
    return cell_to_unit[
        row - radius:row + radius + 1, col - radius:col + radius + 1
    ].reshape(-1)


def reference_d_unit(patch_size: int, distance: str = "linf") -> np.ndarray:
    """Centred, unit-normalised inverse distances over one patch.

    Transcribed from ``d_ij = 1 / (dist(pos_i, pos_j) + 1)`` rather than from the
    implementation, so that a change to the implementation's normalisation shows
    up as a disagreement instead of being mirrored.

    :param patch_size: Units per neighbourhood, ``side ** 2``.
    :type patch_size: int
    :param distance: One of ``linf``, ``l1``, ``l2``.
    :type distance: str
    :return: ``(num_pairs,)`` float32.
    :rtype: numpy.ndarray
    """
    side = int(round(np.sqrt(patch_size)))
    if side * side != patch_size:
        raise ValueError(f"patch_size {patch_size} is not a perfect square")

    rows, cols = np.divmod(np.arange(patch_size), side)
    delta_row = np.abs(rows[:, None] - rows[None, :])
    delta_col = np.abs(cols[:, None] - cols[None, :])
    if distance == "linf":
        separation = np.maximum(delta_row, delta_col)
    elif distance == "l1":
        separation = delta_row + delta_col
    else:
        separation = np.sqrt(delta_row ** 2 + delta_col ** 2)

    upper = np.triu_indices(patch_size, k=1)
    d_pairs = 1.0 / (separation[upper].astype(np.float64) + 1.0)
    d_pairs = d_pairs - d_pairs.mean()
    return (d_pairs / np.sqrt((d_pairs ** 2).sum())).astype(np.float32)


def smooth_field_activations(
    layout,
    num_samples: int,
    seed: int = 0,
    frequencies=((0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 1)),
) -> np.ndarray:
    """Activations whose unit-to-unit correlation decays smoothly with distance.

    Each sample draws coefficients on low-frequency Fourier modes of the grid; a
    unit's response is its own cell's projection onto that sample's pattern. Two
    units therefore respond to the same latent factors in proportion to how close
    their cells are, which is exactly the structure the loss rewards.

    :param layout: The :class:`SpatialLayout` the activations are indexed by.
    :param num_samples: Number of samples ``M`` to synthesise.
    :type num_samples: int
    :param seed: Seed for the per-sample coefficients.
    :type seed: int
    :param frequencies: ``(row, col)`` mode indices. Lower indices mean longer
        wavelengths, i.e. a smoother field.
    :return: ``(num_samples, num_units)`` activations in UNIT order.
    :rtype: numpy.ndarray
    """
    height, width = layout.grid_shape
    row_index, col_index = np.mgrid[0:height, 0:width]
    basis = []
    for row_freq, col_freq in frequencies:
        mode = np.cos(
            2.0 * np.pi
            * (row_freq * row_index / height + col_freq * col_index / width)
        ).reshape(-1)
        basis.append((mode - mode.mean()) / (mode.std() + 1e-8))
    basis_matrix = np.stack(basis)

    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(num_samples, basis_matrix.shape[0]))
    per_cell = coefficients @ basis_matrix
    return per_cell[:, layout.perm]


def graded_field_activations(
    layout, num_samples: int, frequency: int, seed: int = 0
) -> np.ndarray:
    """Activations from a single spatial frequency, for a scale sweep.

    Sweeping ``frequency`` from 1 upward coarsens the field and drives the loss
    up through 0.5 and past it: a field whose wavelength is much finer than the
    neighbourhood is anti-smooth at that neighbourhood's scale. That crossing is
    the loss's whole dynamic range in one measurement.

    :param layout: The layout the activations are indexed by.
    :param num_samples: Number of samples ``M``.
    :type num_samples: int
    :param frequency: Mode index; ``0`` is flat and ``1`` is the smoothest
        non-constant mode on the grid.
    :type frequency: int
    :param seed: Seed for the per-sample coefficients.
    :type seed: int
    :return: ``(num_samples, num_units)`` activations in UNIT order.
    :rtype: numpy.ndarray
    """
    frequencies = [(0, 0), (frequency, 0), (0, frequency), (frequency, frequency)]
    return smooth_field_activations(
        layout, num_samples, seed=seed, frequencies=frequencies
    )


# ---------------------------------------------------------------------
# 3D transcriptions
#
# Written with the linear index spelled out and no ``np.unravel_index``: the
# implementation turns a flat index into grid coordinates with that function, so
# an oracle that used it would share the one place a 3D axis-order slip can live.
# One convention, everywhere: the three axes are (axis 0, axis 1, axis 2) of
# ``cell_to_unit``, row-major, axis 2 fastest. No depth/height/width naming.
# ---------------------------------------------------------------------


def reference_patch_units_3d(
    cell_to_unit: np.ndarray, center: Tuple[int, int, int], radius: int
) -> np.ndarray:
    """Unit ids of the cubic patch around one grid centre, axis 2 fastest.

    Three explicit loops, outermost over axis 0, so the returned order is the
    row-major order of the block ``cell_to_unit[c0 - r:c0 + r + 1, ...]``.

    :param cell_to_unit: ``(axis0, axis1, axis2)`` unit ids.
    :type cell_to_unit: numpy.ndarray
    :param center: ``(i0, i1, i2)`` grid index of the patch centre.
    :type center: Tuple[int, int, int]
    :param radius: Patch radius.
    :type radius: int
    :return: ``((2 * radius + 1) ** 3,)`` unit ids.
    :rtype: numpy.ndarray
    """
    c0, c1, c2 = center
    units = []
    for i0 in range(c0 - radius, c0 + radius + 1):
        for i1 in range(c1 - radius, c1 + radius + 1):
            for i2 in range(c2 - radius, c2 + radius + 1):
                units.append(cell_to_unit[i0, i1, i2])
    return np.array(units)


def reference_d_unit_3d(patch_size: int, distance: str = "linf") -> np.ndarray:
    """Centred, unit-normalised inverse distances over one cubic patch.

    Transcribed from ``d_ij = 1 / (dist(pos_i, pos_j) + 1)`` with every
    coordinate written out: the patch cell with linear index
    ``(a * side + b) * side + c`` sits at offsets ``(a, b, c)``, and pairs run
    ``i < j`` with ``i`` outermost, the order of ``np.triu_indices``.

    :param patch_size: Units per neighbourhood, ``side ** 3``.
    :type patch_size: int
    :param distance: One of ``linf``, ``l1``, ``l2``.
    :type distance: str
    :return: ``(num_pairs,)`` float32.
    :rtype: numpy.ndarray
    """
    side = int(round(patch_size ** (1.0 / 3.0)))
    if side ** 3 != patch_size:
        raise ValueError(f"patch_size {patch_size} is not a perfect cube")

    coordinates = []
    for a in range(side):
        for b in range(side):
            for c in range(side):
                coordinates.append((a, b, c))

    d_pairs = []
    for i in range(patch_size):
        for j in range(i + 1, patch_size):
            delta = [abs(coordinates[i][k] - coordinates[j][k]) for k in range(3)]
            if distance == "linf":
                separation = float(max(delta))
            elif distance == "l1":
                separation = float(sum(delta))
            else:
                separation = float(np.sqrt(sum(d * d for d in delta)))
            d_pairs.append(1.0 / (separation + 1.0))

    d_pairs = np.array(d_pairs, dtype=np.float64)
    d_pairs = d_pairs - d_pairs.mean()
    return (d_pairs / np.sqrt((d_pairs ** 2).sum())).astype(np.float32)


def smooth_field_activations_3d(
    layout,
    num_samples: int,
    seed: int = 0,
    frequencies=(
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 2), (1, 1, 0),
        (0, 1, 1), (1, 0, 1),
    ),
) -> np.ndarray:
    """3D twin of :func:`smooth_field_activations`.

    A unit's response is its own cell's projection onto that sample's pattern,
    where the modes are cosines over the three grid axes. ``frequencies`` are
    ``(f0, f1, f2)`` indices on ``(axis 0, axis 1, axis 2)`` of
    ``layout.grid_shape`` -- the same order ``cell_to_unit`` has.

    :param layout: Anything with ``grid_shape`` (3 entries) and ``perm``; a real
        3D :class:`SpatialLayout` or a stand-in built from the same two lines.
    :param num_samples: Number of samples ``M``.
    :type num_samples: int
    :param seed: Seed for the per-sample coefficients.
    :type seed: int
    :param frequencies: ``(f0, f1, f2)`` mode indices.
    :return: ``(num_samples, num_units)`` activations in UNIT order.
    :rtype: numpy.ndarray
    """
    n0, n1, n2 = layout.grid_shape
    index0, index1, index2 = np.mgrid[0:n0, 0:n1, 0:n2]
    basis = []
    for f0, f1, f2 in frequencies:
        mode = np.cos(
            2.0 * np.pi * (f0 * index0 / n0 + f1 * index1 / n1 + f2 * index2 / n2)
        ).reshape(-1)
        basis.append((mode - mode.mean()) / (mode.std() + 1e-8))
    basis_matrix = np.stack(basis)

    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(num_samples, basis_matrix.shape[0]))
    per_cell = coefficients @ basis_matrix
    return per_cell[:, layout.perm]
