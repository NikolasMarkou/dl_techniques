"""Three-dimensional unit grids for TopoLM: config-level validation.

This file grows across the 3D plan. At this step it covers ``config.py`` only:
``grid_shape_of`` with a ``grid_dim``, the rank-aware floor, the per-axis check
that the total unit count cannot replace, and ``validate_all_variants`` judging a
row by its own ``radius`` and ``grid_dim``.

The numbers pinned here are derived, not measured: 784 factors to (7, 8, 14) in
3D, so a radius-4 patch (side 9) clears the floor of 9**3 = 729 yet does not fit
along the first axis; 1000 is a cube (10, 10, 10), 343 is (7, 7, 7), 216 is
(6, 6, 6).
"""

import pytest

from dl_techniques.models.language.topolm import (
    MODEL_VARIANTS,
    grid_shape_of,
    validate_variant,
)
from dl_techniques.models.language.topolm.config import (
    DEFAULT_RADIUS,
    validate_all_variants,
)


def _row(embed_dim: int, num_heads: int = 1) -> dict:
    return {"embed_dim": embed_dim, "num_heads": num_heads}


class TestGridShapeOf:
    def test_the_two_dimensional_call_is_unchanged(self):
        assert grid_shape_of(784) == (28, 28)
        assert grid_shape_of(512) == (16, 32)
        assert grid_shape_of(784, 2) == (28, 28)

    def test_three_dimensional_shapes(self):
        assert grid_shape_of(784, 3) == (7, 8, 14)
        assert grid_shape_of(1000, 3) == (10, 10, 10)
        assert grid_shape_of(343, 3) == (7, 7, 7)
        assert grid_shape_of(216, 3) == (6, 6, 6)

    @pytest.mark.parametrize("bad", [1, 4, "3"])
    def test_an_unsupported_rank_is_refused(self, bad):
        with pytest.raises(ValueError):
            grid_shape_of(512, bad)


class TestThreeDimensionalValidation:
    def test_the_floor_passes_but_an_axis_is_too_short(self):
        """The point of the per-axis check. 784 >= 9**3 = 729, so a floor-only
        validator accepts this and the tap build fails far from the cause."""
        assert 784 >= (2 * 4 + 1) ** 3
        with pytest.raises(ValueError, match=r"axis 0 has 7") as info:
            validate_variant("wide", _row(784, 16), 4, grid_dim=3)
        message = str(info.value)
        assert "'wide'" in message
        assert "7x8x14" in message
        assert "every axis >= 9" in message
        assert "largest radius this grid can use is 3" in message

    def test_the_floor_itself_names_the_cube(self):
        with pytest.raises(ValueError, match=r"below the 9x9x9 = 729 units"):
            validate_variant("narrow", _row(728, 8), 4, grid_dim=3)

    @pytest.mark.parametrize(
        "embed_dim,num_heads,radius",
        [(1000, 8, 4), (343, 7, 2), (216, 8, 2), (729, 9, 4)],
    )
    def test_widths_whose_every_axis_fits_pass(self, embed_dim, num_heads, radius):
        validate_variant("ok", _row(embed_dim, num_heads), radius, grid_dim=3)

    def test_the_default_radius_names_the_maximum_usable_radius(self):
        """10x10x10 hosts radius 4 (side 9) but not the default radius 5 (side
        11). A cube below the side is always below the floor too, so the floor
        message is the one that fires and it carries the advice."""
        with pytest.raises(ValueError, match=r"below the 11x11x11 = 1331") as info:
            validate_variant("cube", _row(1000, 8), DEFAULT_RADIUS, grid_dim=3)
        assert "largest radius this grid can use is 4" in str(info.value)

    def test_the_floor_advice_is_three_dimensional_only(self):
        with pytest.raises(ValueError) as info:
            validate_variant("cube", _row(64, 4), DEFAULT_RADIUS)
        assert "largest radius" not in str(info.value)

    def test_a_grid_no_radius_can_use_says_so(self):
        # 1009 is prime: in 3D it resolves to (1, 1, 1009) and the floor at
        # radius 1 (27) is cleared, but no radius works on a unit-thick grid.
        assert grid_shape_of(1009, 3) == (1, 1, 1009)
        with pytest.raises(ValueError, match=r"No radius can work") as info:
            validate_variant("prime", _row(1009), 1, grid_dim=3)
        # Every short axis is listed, not just the first.
        assert "axis 0 has 1; axis 1 has 1" in str(info.value)
        assert "largest radius" not in str(info.value)


class TestTwoDimensionalHole:
    """A 2D configuration that clears the floor and still cannot build.

    This only moves WHERE an unbuildable configuration fails: 997 is prime, so
    its 2D grid is (1, 997); the floor at radius 1 is 9 and 997 clears it, and
    before this check the error fired at the tap's build. Nothing that built
    before is rejected, because ``build_patch_tables`` demands exactly
    ``min(grid) >= 2 * radius + 1``.
    """

    def test_997_clears_the_floor_and_is_refused_by_axis(self):
        assert grid_shape_of(997) == (1, 997)
        assert 997 >= (2 * 1 + 1) ** 2
        with pytest.raises(ValueError, match=r"axis 0 has 1") as info:
            validate_variant("prime", _row(997), 1)
        message = str(info.value)
        assert "'prime'" in message
        assert "1x997" in message
        assert "No radius can work" in message

    def test_the_two_dimensional_floor_message_is_unchanged(self):
        with pytest.raises(ValueError) as info:
            validate_variant("broken", _row(64, 4), DEFAULT_RADIUS)
        assert str(info.value) == (
            "variant 'broken': embed_dim (64) is below the 11x11 = 121 units a "
            "radius-5 neighbourhood needs. Lower the radius, or pick a wider "
            "variant."
        )

    def test_every_shipped_row_still_builds_its_own_patch(self):
        for name, config in MODEL_VARIANTS.items():
            grid = grid_shape_of(config["embed_dim"])
            assert min(grid) >= 2 * DEFAULT_RADIUS + 1, name


class TestValidateAllVariants:
    def test_the_shipped_table_still_passes(self):
        validate_all_variants(DEFAULT_RADIUS)

    def test_a_three_dimensional_row_is_judged_by_its_own_values(self, monkeypatch):
        """343 units are the 7x7x7 cube: legal at the row's own radius 2 and
        grid_dim 3, illegal at the sweep's radius 5 and rank 2 (the 2D grid is
        7 x 49). A sweep that ignored the row's keys would raise here."""
        snapshot = {n: dict(c) for n, c in MODEL_VARIANTS.items()}
        monkeypatch.setitem(
            MODEL_VARIANTS,
            "probe3d",
            {**_row(343, 7), "grid_dim": 3, "radius": 2},
        )
        validate_all_variants(DEFAULT_RADIUS)
        monkeypatch.undo()
        assert {n: dict(c) for n, c in MODEL_VARIANTS.items()} == snapshot

    def test_a_three_dimensional_row_that_misfits_its_own_radius_is_named(
        self, monkeypatch
    ):
        monkeypatch.setitem(
            MODEL_VARIANTS,
            "probe3d_bad",
            {**_row(784, 16), "grid_dim": 3, "radius": 4},
        )
        with pytest.raises(ValueError, match=r"'probe3d_bad'.*axis 0 has 7"):
            validate_all_variants(DEFAULT_RADIUS)

    def test_a_row_without_the_keys_is_judged_at_the_sweep_values(self, monkeypatch):
        monkeypatch.setitem(MODEL_VARIANTS, "probe2d", _row(64, 4))
        with pytest.raises(ValueError, match=r"'probe2d'.*radius-5"):
            validate_all_variants(DEFAULT_RADIUS)
        # A lower sweep radius makes the same row legal: 64 = 8x8 hosts side 7.
        monkeypatch.setitem(MODEL_VARIANTS, "probe2d", _row(64, 4))
        validate_all_variants(3)


class TestValidationAgreesWithTheTapBuild:
    """The validator must reject exactly what ``build_patch_tables`` rejects.

    Looser means an unbuildable model constructs; tighter means a model that
    built before (in 2D) is now refused. Swept over every width below 300, radii
    1 to 3 and both ranks (1794 cases).
    """

    def test_verdicts_agree_for_every_width_radius_and_rank(self):
        import numpy as np

        from dl_techniques.layers.regularization.spatial_smoothness import (
            build_patch_tables,
        )

        disagreements = []
        for grid_dim in (2, 3):
            for width in range(1, 300):
                grid = grid_shape_of(width, grid_dim)
                for radius in (1, 2, 3):
                    try:
                        validate_variant(
                            "x", _row(width), radius, grid_dim=grid_dim
                        )
                        validated = True
                    except ValueError:
                        validated = False
                    try:
                        build_patch_tables(np.arange(width).reshape(grid), radius)
                        built = True
                    except ValueError:
                        built = False
                    if validated != built:
                        disagreements.append((grid_dim, width, radius))
        assert disagreements == []
