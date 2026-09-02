"""The 2-D sin-cos table is built ``w``-first, and the halves say which is which.

``get_2d_sincos_pos_embed`` calls ``np.meshgrid(grid_w, grid_h)`` -- "here w
goes first" -- so ``grid[0]`` holds the COLUMN index and ``grid[1]`` the ROW
index, and ``get_2d_sincos_pos_embed_from_grid`` puts the column half FIRST.

**Why this file exists.** Swapping the two meshgrid arguments, or swapping the
two output halves, is a pure permutation of a square table. Shape, dtype, row
sums, column sums, every norm, the min, the max, the histogram and a ``.keras``
round trip are all IDENTICAL. A model trained on the transposed table trains
perfectly well and is incompatible with every published checkpoint. Only an
elementwise comparison against an **independently computed destination index**
can see it.

**How the independence is obtained.** Every expected value below comes from
``_sincos_1d``, a closed form transcribed by hand from
``reference/models.py:301-320``, and every destination index comes from
``_flat_index``, which states the row-major flattening as arithmetic on
``(row, col)``. Neither calls the module under test. Re-invoking the module's
own meshgrid, or deriving the expected table from ``get_2d_sincos_pos_embed``
itself, would make the guard pass under the transposition it exists to reject
-- the reversed permutation is still an exact bijection.

Each positive arm has a **negative sibling** that asserts the transposed
reading FAILS at an asymmetric grid position, so the arm is known to
discriminate rather than merely to pass.
"""

import numpy as np
import pytest

from dl_techniques.layers.embedding.sincos_pos_embed_2d import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
)
from dl_techniques.models.vision_language.bit_diffusion.blocks import (
    get_2d_sincos_pos_embed as sibling_get_2d_sincos_pos_embed,
)


def _sincos_1d(embed_dim: int, pos: float) -> np.ndarray:
    """The 1-D MAE embedding of ONE scalar position, written out longhand.

    Transcribed from ``reference/models.py:301-320``. Deliberately does not
    call :func:`get_1d_sincos_pos_embed_from_grid`.
    """
    half = embed_dim // 2
    omega = np.array(
        [1.0 / 10000 ** (j / (embed_dim / 2.0)) for j in range(half)],
        dtype=np.float64,
    )
    args = float(pos) * omega
    return np.concatenate([np.sin(args), np.cos(args)])


def _flat_index(row: int, col: int, grid_size: int) -> int:
    """Row-major destination index of grid cell ``(row, col)``.

    ``get_2d_sincos_pos_embed`` reshapes the stacked grid to
    ``(2, 1, G, G)`` and the 1-D helper flattens with ``reshape(-1)``, so the
    last axis (the column) varies fastest.
    """
    return row * grid_size + col


class TestTheGridIsWFirst:
    """``grid[0]`` is the column. Asserted per cell, not per statistic."""

    @pytest.mark.parametrize("grid_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("embed_dim", [8, 16])
    def test_every_cell_is_column_then_row(self, grid_size, embed_dim):
        table = get_2d_sincos_pos_embed(embed_dim, grid_size)
        assert table.shape == (grid_size * grid_size, embed_dim)
        half = embed_dim // 2

        for row in range(grid_size):
            for col in range(grid_size):
                t = _flat_index(row, col, grid_size)
                np.testing.assert_allclose(
                    table[t, :half],
                    _sincos_1d(half, col),
                    rtol=0.0,
                    atol=1e-12,
                    err_msg=(
                        f"row {t} (grid row {row}, col {col}): the FIRST half "
                        f"must encode the COLUMN index {col}"
                    ),
                )
                np.testing.assert_allclose(
                    table[t, half:],
                    _sincos_1d(half, row),
                    rtol=0.0,
                    atol=1e-12,
                    err_msg=(
                        f"row {t} (grid row {row}, col {col}): the SECOND half "
                        f"must encode the ROW index {row}"
                    ),
                )

    def test_the_transposed_reading_fails_at_an_asymmetric_cell(self):
        # Anti-vacuity for the arm above. At (row=0, col=2) on a 4x4 grid the
        # two halves encode DIFFERENT positions, so the swapped reading must be
        # wrong. Without this sibling the arm above would be equally satisfied
        # by a table whose halves are swapped on the diagonal-only cells it
        # happened to visit.
        grid_size, embed_dim, half = 4, 8, 4
        table = get_2d_sincos_pos_embed(embed_dim, grid_size)
        t = _flat_index(0, 2, grid_size)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                table[t, :half], _sincos_1d(half, 0), rtol=0.0, atol=1e-12
            )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                table[t, half:], _sincos_1d(half, 2), rtol=0.0, atol=1e-12
            )

    def test_the_table_is_not_symmetric_under_a_grid_transpose(self):
        # A single scalar statement of the same fact: if the meshgrid arguments
        # were swapped, the table would equal its own (row, col)-transposed
        # permutation. It does not.
        grid_size, embed_dim = 4, 8
        table = get_2d_sincos_pos_embed(embed_dim, grid_size)
        perm = [
            _flat_index(c, r, grid_size)
            for r in range(grid_size)
            for c in range(grid_size)
        ]
        assert not np.allclose(table, table[perm])

    @pytest.mark.parametrize("grid_size", [1, 2, 5, 16])
    def test_swapping_the_meshgrid_arguments_is_an_exact_no_op(self, grid_size):
        # The OBVIOUS injection for this guard -- `np.meshgrid(grid_h, grid_w)`
        # -- proves nothing, because `grid_h` and `grid_w` are the same
        # `np.arange(grid_size)` for a square grid and `np.meshgrid` is
        # argument-order invariant on identical arguments. Anyone reaching for
        # it to demonstrate the guard works gets a false GREEN. This arm records
        # the inertness so the next reader picks a mutation that bites: either
        # `indexing="ij"`, or swapping `grid[0]`/`grid[1]` in
        # `get_2d_sincos_pos_embed_from_grid`. Both were proven RED.
        h = np.arange(grid_size, dtype=np.float32)
        w = np.arange(grid_size, dtype=np.float32)
        np.testing.assert_array_equal(
            np.stack(np.meshgrid(w, h), axis=0),
            np.stack(np.meshgrid(h, w), axis=0),
        )
        # ...and the mutation that IS a transposition really is one.
        if grid_size > 1:
            assert not np.array_equal(
                np.stack(np.meshgrid(w, h), axis=0),
                np.stack(np.meshgrid(w, h, indexing="ij"), axis=0),
            )

    def test_a_one_cell_grid_is_the_degenerate_control(self):
        # G == 1 is the one grid where the transposition IS the identity, so
        # this arm cannot discriminate and is recorded as a shape/constant
        # check only: position 0 embeds to [sin(0)... , cos(0)...] == [0, 1].
        table = get_2d_sincos_pos_embed(8, 1)
        assert table.shape == (1, 8)
        np.testing.assert_array_equal(table[0, :2], np.zeros(2))
        np.testing.assert_array_equal(table[0, 6:], np.ones(2))


class TestTheHalvesOfTheFromGridHelper:
    """``grid[0]`` -> first half, ``grid[1]`` -> second half, unconditionally."""

    def test_the_two_grid_planes_land_in_the_declared_halves(self):
        embed_dim = 8
        # Two planes chosen so no value appears in both -- a shared value would
        # make a swap invisible.
        plane_a = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        plane_b = np.array([[[10.0, 20.0], [30.0, 40.0]]])
        grid = np.concatenate([plane_a, plane_b], axis=0)

        out = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        assert out.shape == (4, embed_dim)
        half = embed_dim // 2
        for k, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            np.testing.assert_allclose(
                out[k, :half], _sincos_1d(half, value), rtol=0.0, atol=1e-12
            )
        for k, value in enumerate([10.0, 20.0, 30.0, 40.0]):
            np.testing.assert_allclose(
                out[k, half:], _sincos_1d(half, value), rtol=0.0, atol=1e-12
            )

    def test_swapping_the_planes_changes_the_table(self):
        embed_dim = 8
        plane_a = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        plane_b = np.array([[[10.0, 20.0], [30.0, 40.0]]])
        straight = get_2d_sincos_pos_embed_from_grid(
            embed_dim, np.concatenate([plane_a, plane_b], axis=0)
        )
        swapped = get_2d_sincos_pos_embed_from_grid(
            embed_dim, np.concatenate([plane_b, plane_a], axis=0)
        )
        assert not np.allclose(straight, swapped)


class TestTheOneDimensionalHelperIsSinFirst:
    """MAE is sin-first; the timestep ladder is cos-first. Both, deliberately."""

    def test_position_zero_is_zeros_then_ones(self):
        out = get_1d_sincos_pos_embed_from_grid(8, np.array([0.0]))
        np.testing.assert_array_equal(out[0, :4], np.zeros(4))
        np.testing.assert_array_equal(out[0, 4:], np.ones(4))

    @pytest.mark.parametrize("embed_dim", [2, 4, 16, 64])
    def test_it_matches_the_longhand_closed_form(self, embed_dim):
        positions = np.array([0.0, 1.0, 7.0, 63.0])
        out = get_1d_sincos_pos_embed_from_grid(embed_dim, positions)
        expected = np.stack([_sincos_1d(embed_dim, p) for p in positions])
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)

    def test_it_stays_float64(self):
        # The table is a constant computed once; narrowing it to float32 would
        # quantize the high-frequency columns for no benefit.
        assert (
            get_1d_sincos_pos_embed_from_grid(8, np.arange(4)).dtype
            == np.float64
        )


class TestItAgreesWithTheBitDiffusionSibling:
    """The recorded duplication (D-001) is pinned elementwise, not by prose."""

    @pytest.mark.parametrize(
        "embed_dim,grid_size", [(8, 4), (64, 3), (16, 1), (32, 7)]
    )
    def test_the_tables_are_identical_at_atol_zero(self, embed_dim, grid_size):
        np.testing.assert_array_equal(
            get_2d_sincos_pos_embed(embed_dim, grid_size),
            sibling_get_2d_sincos_pos_embed(embed_dim, grid_size),
        )


class TestTheGuardsOnTheArguments:

    @pytest.mark.parametrize("embed_dim", [0, -2, 3, 7])
    def test_a_bad_one_dimensional_width_raises(self, embed_dim):
        with pytest.raises(ValueError):
            get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(3))

    def test_an_odd_two_dimensional_width_raises(self):
        with pytest.raises(ValueError):
            get_2d_sincos_pos_embed_from_grid(7, np.zeros((2, 1, 2, 2)))

    @pytest.mark.parametrize("grid_size", [0, -1])
    def test_a_non_positive_grid_size_raises(self, grid_size):
        with pytest.raises(ValueError):
            get_2d_sincos_pos_embed(8, grid_size)

    def test_the_cls_token_prepends_zero_rows_only_when_both_are_set(self):
        assert get_2d_sincos_pos_embed(8, 2, cls_token=True, extra_tokens=1).shape == (5, 8)
        # BOTH must be set -- upstream's exact condition.
        assert get_2d_sincos_pos_embed(8, 2, cls_token=True, extra_tokens=0).shape == (4, 8)
        assert get_2d_sincos_pos_embed(8, 2, cls_token=False, extra_tokens=1).shape == (4, 8)
        prepended = get_2d_sincos_pos_embed(8, 2, cls_token=True, extra_tokens=1)
        np.testing.assert_array_equal(prepended[0], np.zeros(8))
