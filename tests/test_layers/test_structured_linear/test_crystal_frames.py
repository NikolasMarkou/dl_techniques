"""Test suite for the shared crystal frame constructors.

These are the geometric oracle the crystal layers are tested against, so the
properties pinned here are the ones the layers rely on: unit-norm columns, the
``-1/N`` equiangular inner product for a single simplex, orthogonality of the
Haar rotation, and the claim that packing yields more distinct directions than
tiling.
"""

import numpy as np
import pytest

from dl_techniques.layers.structured_linear.crystal_frames import (
    crystal_frame,
    haar_orthogonal,
    hyperspherical_simplex,
    tiled_simplex_frame
)


def _max_off_diagonal(gram: np.ndarray) -> float:
    """Largest off-diagonal magnitude in a Gram matrix."""
    return float(np.abs(gram[~np.eye(gram.shape[0], dtype=bool)]).max())


class TestHypersphericalSimplex:
    """The single-simplex construction: unit vertices, equiangular at -1/N."""

    @pytest.mark.parametrize('dimensions', [1, 2, 3, 4, 8, 16, 64])
    def test_vertices_are_unit_norm(self, dimensions):
        vertices = hyperspherical_simplex(dimensions)

        assert vertices.shape == (dimensions + 1, dimensions)
        np.testing.assert_allclose(
            np.linalg.norm(vertices, axis=1), 1.0, atol=1e-12, rtol=0
        )

    @pytest.mark.parametrize('dimensions', [1, 2, 3, 4, 8, 16, 64])
    def test_every_pair_sits_at_minus_one_over_n(self, dimensions):
        """The defining property of the ETF the layers build on."""
        vertices = hyperspherical_simplex(dimensions)
        gram = vertices @ vertices.T

        off_diagonal = gram[~np.eye(dimensions + 1, dtype=bool)]
        np.testing.assert_allclose(
            off_diagonal,
            -1.0 / dimensions,
            atol=1e-12,
            rtol=0,
            err_msg=(
                f"a {dimensions}-simplex must have every pairwise inner product "
                f"equal to -1/{dimensions}"
            )
        )

    @pytest.mark.parametrize('dimensions', [2, 4, 8])
    def test_vertices_sum_to_zero(self, dimensions):
        """Centroid at the origin is what makes the set tight."""
        np.testing.assert_allclose(
            hyperspherical_simplex(dimensions).sum(axis=0),
            0.0,
            atol=1e-12,
            rtol=0
        )

    @pytest.mark.parametrize('dimensions', [0, -1])
    def test_rejects_a_non_positive_dimension(self, dimensions):
        with pytest.raises(ValueError, match='dimensions must be >= 1'):
            hyperspherical_simplex(dimensions)


class TestHaarOrthogonal:
    """The rotation used to place several simplexes without coincidence."""

    @pytest.mark.parametrize('dimensions', [1, 2, 8, 16])
    def test_the_result_is_orthogonal(self, dimensions):
        rotation = haar_orthogonal(dimensions, np.random.default_rng(0))

        np.testing.assert_allclose(
            rotation.T @ rotation,
            np.eye(dimensions),
            atol=1e-10,
            rtol=0
        )

    def test_the_same_seed_gives_the_same_rotation(self):
        first = haar_orthogonal(8, np.random.default_rng(5))
        second = haar_orthogonal(8, np.random.default_rng(5))

        np.testing.assert_array_equal(first, second)

    def test_different_draws_give_different_rotations(self):
        """Anti-vacuity: a function returning a constant would pass the above."""
        first = haar_orthogonal(8, np.random.default_rng(0))
        second = haar_orthogonal(8, np.random.default_rng(1))

        assert not np.allclose(first, second)

    def test_rejects_a_non_positive_dimension(self):
        with pytest.raises(ValueError, match='dimensions must be >= 1'):
            haar_orthogonal(0, np.random.default_rng(0))


class TestCrystalFrame:
    """Packing: distinct directions for widths past one simplex's capacity."""

    @pytest.mark.parametrize(
        'dimensions,width',
        [(4, 4), (4, 5), (8, 9), (16, 100), (8, 30), (4, 37)]
    )
    def test_shape_dtype_and_unit_columns(self, dimensions, width):
        frame = crystal_frame(dimensions, width, seed=0)

        assert frame.shape == (dimensions, width)
        assert frame.dtype == np.float32
        np.testing.assert_allclose(
            np.linalg.norm(frame, axis=0), 1.0, atol=1e-6, rtol=0
        )

    @pytest.mark.parametrize('dimensions,width', [(8, 30), (16, 100), (4, 37)])
    def test_every_column_is_a_distinct_direction(self, dimensions, width):
        """The property tiling cannot provide, and the reason this exists."""
        frame = crystal_frame(dimensions, width, seed=1)
        distinct = {tuple(np.round(column, 6)) for column in frame.T}

        assert len(distinct) == width, (
            f"expected {width} distinct columns, got {len(distinct)}"
        )

    def test_packing_separates_more_than_tiling_does(self):
        """Direct comparison against the degenerate construction."""
        dimensions, width = 8, 30

        packed = crystal_frame(dimensions, width, seed=2).astype(np.float64)
        tiled = tiled_simplex_frame(dimensions, width).astype(np.float64)

        assert _max_off_diagonal(packed.T @ packed) < _max_off_diagonal(tiled.T @ tiled)

    def test_the_seed_makes_the_frame_reproducible(self):
        np.testing.assert_array_equal(
            crystal_frame(8, 30, seed=7),
            crystal_frame(8, 30, seed=7)
        )

    def test_seed_none_still_produces_a_valid_frame(self):
        """``None`` means an independent draw, not a broken one.

        Pinned on validity rather than on difference: two ``None`` draws being
        equal is possible and would not be a defect.
        """
        frame = crystal_frame(8, 30)

        assert frame.shape == (8, 30)
        np.testing.assert_allclose(
            np.linalg.norm(frame, axis=0), 1.0, atol=1e-6, rtol=0
        )

    def test_candidate_trials_of_one_skips_the_search(self):
        """A single trial is a legal request, not an error."""
        assert crystal_frame(8, 30, candidate_trials=1, seed=0).shape == (8, 30)

    def test_a_broader_search_improves_separation_on_average(self):
        """More trials should not make the packing worse.

        Measured over 12 seeds at ``dimensions=16, width=60``: mean maximum
        pairwise inner product falls from 0.7465 at ``candidate_trials=1`` to
        0.7271 at ``candidate_trials=8``. The per-seed effect is noisy and can
        move either way, so this asserts on the mean over seeds -- a per-seed
        comparison is not a claim the search makes.
        """
        def _mean_coherence(trials: int) -> float:
            return float(np.mean([
                _max_off_diagonal(
                    crystal_frame(16, 60, candidate_trials=trials, seed=seed)
                    .astype(np.float64).T
                    @ crystal_frame(16, 60, candidate_trials=trials, seed=seed)
                    .astype(np.float64)
                )
                for seed in range(12)
            ]))

        assert _mean_coherence(8) < _mean_coherence(1)

    def test_candidate_trials_changes_the_frame(self):
        """Knob sensitivity: the search width must actually reach the result."""
        one_trial = crystal_frame(16, 60, candidate_trials=1, seed=4)
        many_trials = crystal_frame(16, 60, candidate_trials=8, seed=4)

        assert not np.allclose(one_trial, many_trials), (
            "candidate_trials did not change the packed frame"
        )

    @pytest.mark.parametrize(
        'dimensions,width,trials,match',
        [
            (0, 10, 5, 'dimensions must be >= 1'),
            (8, 0, 5, 'width must be >= 1'),
            (8, 10, 0, 'candidate_trials must be >= 1'),
        ]
    )
    def test_validation(self, dimensions, width, trials, match):
        with pytest.raises(ValueError, match=match):
            crystal_frame(dimensions, width, candidate_trials=trials)


class TestTiledSimplexFrame:
    """The retained legacy construction, pinned so it stays what it is."""

    def test_at_or_below_the_simplex_capacity_it_is_exact(self):
        frame = tiled_simplex_frame(16, 17)

        assert frame.shape == (16, 17)
        np.testing.assert_allclose(
            np.linalg.norm(frame, axis=0), 1.0, atol=1e-6, rtol=0
        )
        assert _max_off_diagonal(frame.astype(np.float64).T @ frame.astype(np.float64)) < 0.1

    def test_beyond_the_capacity_it_repeats_vertices(self):
        """Documented degeneracy, asserted so a future 'fix' is a deliberate act."""
        frame = tiled_simplex_frame(16, 100).astype(np.float64)
        distinct = {tuple(np.round(column, 6)) for column in frame.T}

        assert len(distinct) == 17, "tiled_simplex_frame no longer tiles"
        assert _max_off_diagonal(frame.T @ frame) > 0.999

    @pytest.mark.parametrize('dimensions,width', [(0, 4), (4, 0)])
    def test_validation(self, dimensions, width):
        with pytest.raises(ValueError, match='must be >= 1'):
            tiled_simplex_frame(dimensions, width)


class TestTheGuardHasSubjects:
    """Anti-vacuity for the module: an import failure would silence everything."""

    def test_the_geometry_module_exposes_all_four_builders(self):
        import dl_techniques.layers.structured_linear.crystal_frames as module

        for name in ('hyperspherical_simplex', 'haar_orthogonal',
                     'crystal_frame', 'tiled_simplex_frame'):
            assert callable(getattr(module, name, None)), (
                f"{name} is missing from crystal_frames"
            )
