"""Delta-impulse orientation probes on a NON-SQUARE grid (audit rule R-140).

A square-only spatial test cannot see a transposed stride: if a downsampling or
upsampling path swaps its row and column strides, every intermediate shape stays
correct, every parameter count stays correct and every ``assert_allclose``
against a self-generated reference still passes. This repo's own history records
three such defects that survived large green suites -- a sign error in
``ops.roll`` (249 tests), a transposed relative-position bias (219 tests) and a
shifted CLS slice (91/91).

This module is a shared instrument, not a test module: it carries no ``test_``
prefix and is imported by ``test_the_delta_impulse_orientation_probes.py``.

The instrument
--------------

Feed a single delta impulse into an otherwise all-zero input and subtract the
all-zero forward:

.. code-block:: text

    r(row, col) = | f(delta_at(row, col)) - f(0) |   summed over channels

Subtracting ``f(0)`` is what makes this exact rather than approximate. Outside
the impulse's receptive field the two inputs are *identical*, so the two outputs
are identical bit-for-bit and ``r`` is exactly ``0.0`` there -- biases, running
statistics and any input-independent term cancel. Inside the receptive field
``r`` is the impulse response.

Two assertions are built on ``r``, and which one applies depends on whether the
path is purely local:

``assert_impulse_support_box``
    For a path whose response has bounded support (a convolutional stem with no
    global mixing), the nonzero support is a small box and its location is an
    EXACT statement: a stride-4 non-overlapping stem maps an impulse at
    ``(row, col)`` to exactly ``(row // 4, col // 4)``.

``assert_orientation_is_diagonal``
    For a path that mixes globally (attention in the trunk or the VAE mid-block)
    the support is the whole map, so location is not usable. What survives is the
    *centroid response matrix* ``M``: shift the impulse one stride step in rows
    only, then in columns only, and record how the response centroid moves.

    .. code-block:: text

        M[0, 0] = d(centroid_row) / d(row shift)      M[0, 1] = d(centroid_row) / d(col shift)
        M[1, 0] = d(centroid_col) / d(row shift)      M[1, 1] = d(centroid_col) / d(col shift)

    A correctly oriented path gives a DIAGONAL ``M`` with both diagonal entries
    positive. A transposed stride gives an ANTI-diagonal one. The measured
    off-diagonal terms across the nine stride paths this repo probes are between
    exactly ``0.0`` and ``2.4e-3`` against diagonal terms of ``0.0125`` to
    ``8.0``; the smallest measured diagonal-to-off-diagonal ratio is ``95``
    (``bfunet``), which is where :data:`DEFAULT_MIN_RATIO` comes from.

Both helpers REFUSE a square grid. That refusal is the whole point of R-140:
on a square grid a transposed stride produces the correct output shape and the
instrument is blind to it, so a square-grid "orientation probe" is a test that
cannot fail for the reason it exists.
"""

from typing import Callable, Optional, Tuple

import keras
import numpy as np

__all__ = [
    "DEFAULT_MIN_RATIO",
    "assert_impulse_support_box",
    "assert_orientation_is_diagonal",
    "centroid_response_matrix",
    "impulse_energy_map",
    "impulse_support_box",
    "transposed_stride_injection",
]

#: Smallest diagonal-to-off-diagonal ratio :func:`assert_orientation_is_diagonal`
#: accepts. The worst measured subject in this repo scores ``95``, so ``20``
#: leaves a ~4.8x margin while still failing an axis swap by many orders of
#: magnitude (a swap makes the diagonal terms the small ones).
DEFAULT_MIN_RATIO: float = 20.0

Forward = Callable[[np.ndarray], object]


def _require_non_square(height: int, width: int, label: str) -> None:
    """Refuse a square probe grid -- R-140's entire discriminating condition."""
    assert height != width, (
        f"{label}: the probe grid is {height}x{width}, i.e. SQUARE. A square "
        "grid cannot see a transposed stride (rule R-140): the output shape is "
        "correct either way, so the probe cannot fail for the reason it exists. "
        "Choose height != width."
    )


def impulse_energy_map(
    forward: Forward,
    shape: Tuple[int, int, int],
    row: int,
    col: int,
    *,
    label: str = "impulse probe",
) -> np.ndarray:
    """Return the per-position impulse-response energy ``|f(delta) - f(0)|``.

    :param forward: Callable taking a ``(1, H, W, C)`` array and returning a
        rank-4 channels-last tensor.
    :param shape: ``(H, W, C)`` of the probe input. ``H`` must differ from ``W``.
    :param row: Impulse row in the input grid.
    :param col: Impulse column in the input grid.
    :param label: Name used in assertion messages.
    :return: A ``(H_out, W_out)`` float64 array, channel-summed absolute delta.
    """
    height, width, channels = shape
    _require_non_square(height, width, label)
    zeros = np.zeros((1, height, width, channels), dtype="float32")
    impulse = zeros.copy()
    impulse[0, row, col, :] = 1.0
    out_zero = np.asarray(keras.ops.convert_to_numpy(forward(zeros)), dtype="float64")
    out_imp = np.asarray(keras.ops.convert_to_numpy(forward(impulse)), dtype="float64")
    assert out_zero.ndim == 4, (
        f"{label}: forward returned rank {out_zero.ndim}, expected a rank-4 "
        "channels-last map. Wrap the callable to select one."
    )
    return np.abs(out_imp - out_zero).sum(axis=-1)[0]


def impulse_support_box(energy: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(row_min, row_max, col_min, col_max)`` of the exactly-nonzero support.

    :param energy: A map from :func:`impulse_energy_map`.
    :return: The bounding box, or ``None`` when the response is identically zero
        (which means the impulse never reached the output -- a dead path).
    """
    nonzero = np.argwhere(energy > 0.0)
    if nonzero.size == 0:
        return None
    return (
        int(nonzero[:, 0].min()),
        int(nonzero[:, 0].max()),
        int(nonzero[:, 1].min()),
        int(nonzero[:, 1].max()),
    )


def assert_impulse_support_box(
    forward: Forward,
    shape: Tuple[int, int, int],
    row: int,
    col: int,
    expected: Tuple[int, int, int, int],
    *,
    label: str,
) -> Tuple[int, int, int, int]:
    """Assert a local path maps the impulse to an EXACT output support box.

    Use this only for paths with no global mixing; a single attention block
    anywhere makes the support the whole map and this assertion meaningless.
    Use :func:`assert_orientation_is_diagonal` there instead.

    :param forward: See :func:`impulse_energy_map`.
    :param shape: ``(H, W, C)`` probe input shape, non-square.
    :param row: Impulse row.
    :param col: Impulse column.
    :param expected: The exact ``(row_min, row_max, col_min, col_max)`` box.
    :param label: Name used in assertion messages.
    :return: The measured box.
    """
    energy = impulse_energy_map(forward, shape, row, col, label=label)
    box = impulse_support_box(energy)
    assert box is not None, (
        f"{label}: the impulse response is identically zero everywhere. The "
        "impulse never reached the output -- this is a dead path, not an "
        "orientation result."
    )
    assert box == expected, (
        f"{label}: impulse at (row={row}, col={col}) on a "
        f"{shape[0]}x{shape[1]} grid produced support box {box}, expected "
        f"{expected}. A row/column swap in this path's stride is exactly what "
        "this reports."
    )
    return box


def _centroid(energy: np.ndarray, label: str) -> Tuple[float, float]:
    total = float(energy.sum())
    assert total > 0.0, (
        f"{label}: the impulse response has zero total energy, so its centroid "
        "is undefined. The impulse never reached the output."
    )
    rows = np.arange(energy.shape[0], dtype="float64")
    cols = np.arange(energy.shape[1], dtype="float64")
    return (
        float((energy.sum(axis=1) * rows).sum() / total),
        float((energy.sum(axis=0) * cols).sum() / total),
    )


def centroid_response_matrix(
    forward: Forward,
    shape: Tuple[int, int, int],
    base: Tuple[int, int],
    step: int,
    *,
    label: str,
) -> np.ndarray:
    """Return the 2x2 centroid response matrix ``M`` described in the module docstring.

    :param forward: See :func:`impulse_energy_map`.
    :param shape: ``(H, W, C)`` probe input shape, non-square.
    :param base: ``(row, col)`` of the reference impulse. Both ``row + step`` and
        ``col + step`` must stay inside the grid.
    :param step: Input-pixel shift applied to one axis at a time. Use at least
        one full stride of the path so the shift is visible downstream.
    :param label: Name used in assertion messages.
    :return: ``M`` as a ``(2, 2)`` float64 array.
    """
    row, col = base
    ref = _centroid(impulse_energy_map(forward, shape, row, col, label=label), label)
    shifted_row = _centroid(
        impulse_energy_map(forward, shape, row + step, col, label=label), label
    )
    shifted_col = _centroid(
        impulse_energy_map(forward, shape, row, col + step, label=label), label
    )
    return np.array(
        [
            [shifted_row[0] - ref[0], shifted_col[0] - ref[0]],
            [shifted_row[1] - ref[1], shifted_col[1] - ref[1]],
        ],
        dtype="float64",
    )


def assert_orientation_is_diagonal(
    forward: Forward,
    shape: Tuple[int, int, int],
    base: Tuple[int, int],
    step: int,
    *,
    label: str,
    min_ratio: float = DEFAULT_MIN_RATIO,
) -> np.ndarray:
    """Assert the centroid response matrix is diagonal with positive diagonal.

    :param forward: See :func:`impulse_energy_map`.
    :param shape: ``(H, W, C)`` probe input shape, non-square.
    :param base: Reference impulse ``(row, col)``.
    :param step: Per-axis input shift.
    :param label: Name used in assertion messages.
    :param min_ratio: Required ratio of the smaller diagonal magnitude to the
        larger off-diagonal magnitude.
    :return: The measured ``M``.
    """
    matrix = centroid_response_matrix(forward, shape, base, step, label=label)
    printed = np.array2string(matrix, precision=6)
    assert matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0, (
        f"{label}: the centroid response matrix has a non-positive diagonal "
        f"entry, so a positive shift along an axis does NOT move the response "
        f"the same way along that axis.\nM =\n{printed}"
    )
    off = max(abs(float(matrix[0, 1])), abs(float(matrix[1, 0])))
    diag = min(abs(float(matrix[0, 0])), abs(float(matrix[1, 1])))
    if off == 0.0:
        return matrix
    ratio = diag / off
    assert ratio >= min_ratio, (
        f"{label}: the centroid response matrix is not diagonal enough "
        f"(diag/off = {ratio:.4g}, required >= {min_ratio}). A row shift is "
        "leaking into the column response or vice versa, which is what a "
        f"transposed stride looks like.\nM =\n{printed}"
    )
    return matrix


def transposed_stride_injection(forward: Forward) -> Forward:
    """Wrap ``forward`` so its OUTPUT spatial axes are swapped -- a one-sided swap.

    This is the DEAD-COMPONENT injection every probe in
    ``test_the_delta_impulse_orientation_probes`` is proven RED against. It
    reproduces the exact defect R-140 exists to catch: a path that lands the
    response at ``(col // s, row // s)`` instead of ``(row // s, col // s)``.

    **Do NOT "fix" this to transpose the input as well.** That was the first
    version written here and all nine RED proofs FAILED TO RAISE against it,
    because conjugating an isotropic convolutional stack by a transpose --
    ``transpose . f . transpose`` -- is a SYMMETRY of that stack, not a
    corruption of it. Every stride path probed in this repo uses the same kernel
    and stride on both axes, so the two-sided wrapper reproduces the original
    response matrix exactly and the probes stay green while measuring nothing.
    An injection that moves BOTH sides of the comparison proves nothing. See
    ``decisions.md`` D-077.

    :param forward: The callable to corrupt.
    :return: A callable whose response is transposed relative to ``forward``.
    """

    # DECISION plan-2026-08-19T163559-499b6f0e/D-077: the swap is ONE-SIDED.
    # Do NOT also transpose the input. `transpose . f . transpose` is a
    # CONJUGATION, and every stride path in this repo uses the same kernel and
    # stride on both axes, so it is a SYMMETRY of the path, not a corruption:
    # measured, all NINE RED proofs in
    # test_the_delta_impulse_orientation_probes.py FAILED TO RAISE against the
    # two-sided form while the 14 real probes stayed green.
    def _wrapped(x: np.ndarray) -> object:
        out = keras.ops.convert_to_numpy(forward(np.asarray(x)))
        return np.transpose(np.asarray(out), (0, 2, 1, 3))

    return _wrapped
