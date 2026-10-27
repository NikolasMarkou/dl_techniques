"""Three-dimensional unit grids for TopoLM: config validation, block, model.

This file grows across the 3D plan (plan-2026-10-23T141429-390f3815). Step 4
covers ``config.py``: ``grid_shape_of`` with a ``grid_dim``, the rank-aware floor,
the per-axis check that the total unit count cannot replace, and
``validate_all_variants`` judging a row by its own ``radius`` and ``grid_dim``.
Step 5 covers ``components.py`` and ``model.py``: ``grid_dim`` on
:class:`TopoLMBlock` and :class:`TopoLM`, forwarded to every tap, written by
``get_config``, absent from an old config (which is then 2D), validated, and
refused by name when a checkpoint of the other rank is loaded into the model.

The numbers pinned here are derived, not measured: 784 factors to (7, 8, 14) in
3D, so a radius-4 patch (side 9) clears the floor of 9**3 = 729 yet does not fit
along the first axis; 1000 is a cube (10, 10, 10), 343 is (7, 7, 7), 216 is
(6, 6, 6), 210 is (5, 6, 7), 1344 is (8, 12, 14).

Fast model shapes: embed 216 on a (6, 6, 6) grid at radius 2 (8 centres, 125
units per patch, 7750 pairs), embed 64 on (4, 4, 4) at radius 1 (8 centres), and
embed 210 on the non-cubic (5, 6, 7) at radius 2 (6 centres).
"""

import json
import zipfile
from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

import keras

from dl_techniques.models.language.topolm import (
    MODEL_VARIANTS,
    TAP_SITES,
    TopoLM,
    TopoLMBlock,
    create_topolm,
    grid_shape_of,
    validate_variant,
)
from dl_techniques.models.common.lm_pretraining import CausalLanguageModel
from dl_techniques.layers.heads.nlp import TASK_HEAD_NAME
from dl_techniques.models.language.topolm.config import (
    DEFAULT_RADIUS,
    validate_all_variants,
)
from dl_techniques.utils.weight_transfer import load_backbone_from_checkpoint
from tests.index_weights import int32_variables, requires_gpu
from tests.test_layers.test_regularization.spatial_smoothness_oracle import (
    reference_patch_units_3d,
)

from .test_topolm import (
    VOCAB,
    _build_head,
    _dataset,
    _relative_paths,
    _tokens,
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

    def test_every_shipped_two_dimensional_row_still_builds_its_own_patch(self):
        """Rows with no ``grid_dim`` are the 2D ones and keep the default radius
        (the 3D rows carry their own and are judged in TestThreeDimensionalVariantRows).
        Anti-vacuity: the filter must leave the three rows that predate them."""
        rows = {n: c for n, c in MODEL_VARIANTS.items() if "grid_dim" not in c}
        assert sorted(rows) == ["paper", "small", "tiny"]
        for name, config in rows.items():
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


# ---------------------------------------------------------------------
# step 5: TopoLMBlock and TopoLM carry grid_dim
# ---------------------------------------------------------------------

#: 216 = 6 x 6 x 6; a radius-2 patch is 5 per side, 8 centres, 125 units.
_THREE: Dict[str, Any] = {
    "embed_dim": 216, "depth": 2, "num_heads": 4, "max_seq_len": 64,
    "radius": 2, "grid_dim": 3,
}


def _build3d(**kwargs) -> TopoLM:
    """A small 3D model, named like the 2D suite's so weight paths compare."""
    keras.utils.set_random_seed(0)
    config = dict(vocab_size=VOCAB, name="topolm", **_THREE)
    config.update(kwargs)
    return TopoLM(**config)


def _np(value):
    return keras.ops.convert_to_numpy(value)


def _block(**kwargs) -> TopoLMBlock:
    config = dict(
        hidden_size=216, num_heads=4, intermediate_size=216, radius=2,
        name="block",
    )
    config.update(kwargs)
    return TopoLMBlock(**config)


class TestBlockGridDim:
    def test_the_default_is_two_and_the_config_carries_it(self):
        block = _block()
        assert block.grid_dim == 2
        assert block.get_config()["grid_dim"] == 2
        assert _block(grid_dim=3).get_config()["grid_dim"] == 3

    def test_the_config_round_trips(self):
        block = _block(grid_dim=3, seed=4)
        rebuilt = TopoLMBlock.from_config(block.get_config())
        assert rebuilt.get_config() == block.get_config()
        assert rebuilt.grid_dim == 3

    def test_a_config_without_the_key_is_a_two_dimensional_block(self):
        config = _block(grid_dim=3).get_config()
        config.pop("grid_dim")
        block = TopoLMBlock.from_config(config)
        assert block.grid_dim == 2
        assert all(tap.grid_dim == 2 for tap in block.taps)

    @pytest.mark.parametrize("bad", [1, 4, "3", 3.0, True, None])
    def test_an_unsupported_rank_is_refused_naming_the_value(self, bad):
        with pytest.raises(ValueError, match=r"grid_dim must be 2 or 3, got"):
            _block(grid_dim=bad)

    def test_a_numpy_integer_rank_is_stored_as_a_plain_int(self):
        """The validator returns ``int(grid_dim)``; the block stores THAT, so a
        config built from a numpy rank still serializes."""
        block = _block(grid_dim=np.int64(3))
        assert type(block.grid_dim) is int
        json.dumps(block.get_config()["grid_dim"])

    def test_a_bad_rank_is_refused_before_any_sublayer_is_created(self, monkeypatch):
        """Each tap validates its own ``grid_dim`` too, so the block's check is
        only distinguishable by WHEN it fires: before the block builds the
        normalization, attention and feed-forward layers, not at the taps."""
        from dl_techniques.models.language.topolm import components

        def explode(*args, **kwargs):
            raise AssertionError("a sublayer was created before grid_dim was checked")

        monkeypatch.setattr(components, "create_normalization_layer", explode)
        with pytest.raises(ValueError, match=r"grid_dim must be 2 or 3, got 5"):
            _block(grid_dim=5)

    def test_the_taps_are_built_in_the_blocks_rank(self):
        """The forwarding site. A dropped ``grid_dim=`` in ``_make_tap`` leaves a
        block that reports 3 and taps that are silently 2D: they resolve their
        grid lazily, at their own build, so only the built tables show it."""
        block = _block(grid_dim=3)
        block.build((2, 6, 216))
        for tap in block.taps:
            assert tap.grid_dim == 3
            assert tap.layout.grid_shape == (6, 6, 6)
            assert tuple(tap.patch_table.shape) == (8, 125)
        flat = _block(grid_dim=2)
        flat.build((2, 6, 216))
        # 216 = 12 x 18 in 2D: (12-4)(18-4) = 112 centres of 25 units.
        assert tuple(flat.taps[0].patch_table.shape) == (112, 25)


class TestModelGridDim:
    def test_the_rank_is_stored_and_the_grid_is_reported_in_it(self):
        model = _build3d()
        assert model.grid_dim == 3
        assert model.grid_shape == (6, 6, 6)
        assert model.grid_shape is model.grid_shape  # cached after first read
        assert _build3d(embed_dim=210, num_heads=6, grid_dim=3).grid_shape == (5, 6, 7)

    def test_the_default_is_the_papers_two_dimensional_layout(self):
        model = TopoLM(vocab_size=VOCAB, name="topolm", embed_dim=256, depth=1,
                       num_heads=4, max_seq_len=64)
        assert model.grid_dim == 2
        assert model.grid_shape == (16, 16)
        assert model.get_config()["grid_dim"] == 2

    def test_the_config_carries_the_rank_and_round_trips(self):
        model = _build3d(seed=21, alpha=1.5)
        config = model.get_config()
        assert config["grid_dim"] == 3
        rebuilt = TopoLM.from_config(config)
        assert rebuilt.get_config() == config
        assert rebuilt.grid_dim == 3 and rebuilt.grid_shape == (6, 6, 6)

    def test_the_config_has_every_constructor_argument_and_grid_dim_exactly_once(self):
        config = _build3d().get_config()
        assert list(config).count("grid_dim") == 1
        for key in ("grid_shape", "grid_dim", "radius", "embed_dim"):
            assert key in config

    def test_a_config_without_the_key_loads_as_two_dimensional(self):
        """An archive written before ``grid_dim`` existed has no such key. The
        same widths and radius are legal in both ranks (216 is 12 x 18 in 2D),
        so a default that were 3 would build a different model and raise nothing."""
        config = _build3d().get_config()
        config.pop("grid_dim")
        model = TopoLM.from_config(config)
        assert model.grid_dim == 2
        assert model.grid_shape == (12, 18)
        model(_tokens(), training=True)
        assert tuple(model.tap_layers[0].patch_table.shape) == (112, 25)

    def test_every_tap_is_built_in_the_models_rank(self):
        """The model forwarding site (``_build_architecture``): all 2 * depth taps."""
        model = _build3d(depth=3)
        assert all(block.grid_dim == 3 for block in model.blocks)
        model(_tokens())
        assert len(model.tap_layers) == 3 * len(TAP_SITES)
        for tap in model.tap_layers:
            assert tap.grid_dim == 3
            assert tap.layout.grid_shape == (6, 6, 6)
            assert tap.layout.cell_to_unit.shape == (6, 6, 6)
            assert tuple(tap.patch_table.shape) == (8, 125)

    def test_the_forward_is_finite_and_adds_one_loss_per_tap(self):
        model = _build3d()
        outputs = model(_tokens(), training=True)
        assert outputs["logits"].shape == (2, 12, VOCAB)
        assert outputs["last_hidden_state"].shape == (2, 12, 216)
        assert np.all(np.isfinite(_np(outputs["logits"])))
        assert len(model.losses) == 2 * len(TAP_SITES)
        assert all(np.isfinite(float(loss)) for loss in model.losses)

    def test_a_small_cube_with_eight_centres_trains(self):
        """64 = 4 x 4 x 4 at radius 1: a 3 x 3 x 3 patch, (4-2)**3 = 8 centres."""
        model = _build3d(embed_dim=64, num_heads=4, depth=1, radius=1)
        model(_tokens(), training=True)
        assert tuple(model.tap_layers[0].patch_table.shape) == (8, 27)
        assert len(model.losses) == 2

    def test_one_training_step_moves_every_trainable_variable(self):
        keras.utils.set_random_seed(0)
        model = _build3d()
        head = _build_head(model, aggregate=True)
        head.compile(optimizer=keras.optimizers.SGD(1e-2))
        head(np.zeros((1, 12), dtype="int32"), training=False)
        before = {
            variable.path: _np(variable).copy()
            for variable in model.trainable_variables
        }
        head.fit(_dataset(), epochs=1, verbose=0)
        moved = {
            variable.path for variable in model.trainable_variables
            if not np.array_equal(before[variable.path], _np(variable))
        }
        assert moved == set(before), sorted(set(before) - moved)


class TestTheAlphaControlInThreeDimensions:
    def test_alpha_zero_keeps_every_weight(self):
        live = _build3d(alpha=2.5)
        live(_tokens())
        control = _build3d(alpha=0.0)
        control(_tokens())
        assert _relative_paths(live) == _relative_paths(control)

    def test_alpha_zero_keeps_every_permutation_and_table_bit_identical(self):
        live = _build3d(alpha=2.5, seed=5)
        live(_tokens())
        control = _build3d(alpha=0.0, seed=5)
        control(_tokens())
        for a, b in zip(live.tap_layers, control.tap_layers):
            for name in ("perm", "patch_table", "triu_flat", "d_unit"):
                np.testing.assert_array_equal(
                    _np(getattr(a, name)), _np(getattr(b, name)), err_msg=name
                )

    def test_alpha_zero_adds_no_loss(self):
        live = _build3d(alpha=2.5)
        live(_tokens(), training=True)
        control = _build3d(alpha=0.0)
        control(_tokens(), training=True)
        assert len(live.losses) == 2 * len(TAP_SITES)
        assert control.losses == []

    def test_the_reported_training_loss_includes_the_spatial_term(self):
        """The 3D analogue of the 2D aggregate-loss test, re-measured, not copied.

        216 units on (6, 6, 6), radius 2, depth 2, ``alpha = 2.5``, 4 taps,
        ``SGD(1e-3)``, one epoch over 2 batches, CPU, float32. Measured:

          * the reported training loss is 10.377 with ``aggregate_backbone_losses``
            and 5.373 without, a gap of 5.004;
          * the 4 taps' recorded penalties at initialisation are 1.229 to 1.266
            each (raw ``0.5 * (1 - corr)`` of about 0.5, times alpha), so the
            expected gap is 4 * 2.5 * 0.5 = 5.0;
          * the evaluation loss is 5.323 either way (the taps add nothing when
            ``training`` is not True).

        The 2D test's ``+ 5.0`` is NOT reused: measured on its own config the 2D
        gap is 5.008, so that threshold clears by 0.008 and would clear a 3D gap
        of 5.004 by 0.004. The bounds below are 80 % of the derived gap, and an
        upper bound of 120 % so a double-counted term also fails.
        """
        totals = {}
        for aggregate in (True, False):
            keras.utils.set_random_seed(0)
            backbone = _build3d(alpha=2.5)
            head = _build_head(backbone, aggregate)
            head.compile(optimizer=keras.optimizers.SGD(1e-3))
            head(np.zeros((1, 12), dtype="int32"), training=False)
            history = head.fit(_dataset(), epochs=1, verbose=0)
            evaluation = head.evaluate(_dataset(), verbose=0, return_dict=True)
            totals[aggregate] = (history.history["loss"][0], evaluation["loss"])

        gap = totals[True][0] - totals[False][0]
        expected = 4 * 2.5 * 0.5
        assert 0.8 * expected < gap < 1.2 * expected, totals
        assert totals[True][1] == pytest.approx(totals[False][1], rel=0.05)


class TestThreeDimensionalValidationThroughTheModel:
    def test_784_units_at_radius_4_name_the_short_axis(self):
        """The floor (729) is cleared and axis 0 of (7, 8, 14) is not."""
        with pytest.raises(ValueError, match=r"axis 0 has 7") as info:
            TopoLM(vocab_size=VOCAB, embed_dim=784, depth=1, num_heads=16,
                   max_seq_len=32, radius=4, grid_dim=3)
        message = str(info.value)
        assert "7x8x14" in message
        assert "every axis >= 9" in message
        assert "largest radius this grid can use is 3" in message

    def test_1000_units_at_radius_4_construct(self):
        """The both-ways twin of the 784 case."""
        model = TopoLM(vocab_size=VOCAB, embed_dim=1000, depth=1, num_heads=8,
                       max_seq_len=32, radius=4, grid_dim=3)
        assert model.grid_shape == (10, 10, 10)

    def test_the_default_radius_on_a_narrow_cube_names_the_largest_usable_radius(self):
        """1000 = 10 x 10 x 10 is below the radius-5 floor 11**3 = 1331."""
        with pytest.raises(ValueError, match=r"below the 1331 units") as info:
            TopoLM(vocab_size=VOCAB, embed_dim=1000, depth=1, num_heads=8,
                   max_seq_len=32, grid_dim=3)
        message = str(info.value)
        assert "(11 x 11 x 11)" in message
        assert "largest radius this grid can use is 4" in message

    def test_the_default_radius_on_a_wide_but_thin_box_names_the_axis(self):
        """1344 clears the 1331 floor and factors to (8, 12, 14): axis 0 is short."""
        assert 1344 >= 1331
        with pytest.raises(ValueError, match=r"axis 0 has 8") as info:
            TopoLM(vocab_size=VOCAB, embed_dim=1344, depth=1, num_heads=8,
                   max_seq_len=32, grid_dim=3)
        assert "largest radius this grid can use is 3" in str(info.value)

    def test_the_two_dimensional_floor_message_is_unchanged(self):
        with pytest.raises(ValueError) as info:
            TopoLM(vocab_size=VOCAB, embed_dim=64, depth=1, num_heads=4,
                   max_seq_len=32, radius=5)
        assert str(info.value) == (
            "embed_dim (64) is below the 121 units a radius-5 neighbourhood "
            "needs (11 x 11). Lower `radius`, or choose a wider variant."
        )

    def test_the_floor_is_the_cube_not_the_square(self):
        """100 units clear the 2D floor 5**2 and miss the 3D floor 5**3 = 125."""
        TopoLM(vocab_size=VOCAB, embed_dim=100, depth=1, num_heads=4,
               max_seq_len=32, radius=2, grid_dim=2)
        with pytest.raises(ValueError, match=r"below the 125 units"):
            TopoLM(vocab_size=VOCAB, embed_dim=100, depth=1, num_heads=4,
                   max_seq_len=32, radius=2, grid_dim=3)

    def test_a_two_dimensional_model_still_constructs_with_a_short_axis(self):
        """The per-axis check is 3D-only in the constructor, on purpose.

        An explicit (8, 32) grid at the default radius 5 has always constructed
        and reported its grid; only the tap's build refuses it, and the 2D suite
        pins that. The variant table is stricter in both ranks. Tightening this
        is a 2D behaviour change and belongs to a decision, not to this test.
        """
        model = TopoLM(vocab_size=VOCAB, embed_dim=256, depth=1, num_heads=4,
                       max_seq_len=32, grid_shape=(8, 32))
        assert model.grid_shape == (8, 32)
        # The same refusal in 3D, from an explicit grid: 756 = 7 x 9 x 12 clears
        # the 729 floor and its first axis cannot hold a 9-wide patch.
        with pytest.raises(ValueError, match=r"axis 0 has 7"):
            TopoLM(vocab_size=VOCAB, embed_dim=756, depth=1, num_heads=4,
                   max_seq_len=32, radius=4, grid_dim=3, grid_shape=(7, 9, 12))

    @pytest.mark.parametrize("bad", [1, 4, "3", 3.0, True])
    def test_an_unsupported_rank_is_refused_naming_the_value(self, bad):
        with pytest.raises(ValueError, match=r"grid_dim must be 2 or 3, got"):
            _build3d(grid_dim=bad)

    def test_a_pair_cannot_describe_a_three_dimensional_grid(self):
        with pytest.raises(ValueError, match=r"grid_shape"):
            _build3d(grid_shape=(12, 18))

    def test_a_triple_cannot_describe_a_two_dimensional_grid(self):
        """The rank is the stated ``grid_dim``, never ``len(grid_shape)``."""
        with pytest.raises(ValueError, match=r"grid_shape"):
            _build3d(grid_dim=2, grid_shape=(6, 6, 6))

    def test_a_triple_whose_product_is_not_the_width_is_refused(self):
        with pytest.raises(ValueError, match=r"has 180 cells but the tapped tensor"):
            _build3d(grid_shape=(6, 6, 5))

    def test_an_explicit_triple_is_reported_in_the_order_given(self):
        model = _build3d(embed_dim=210, num_heads=6, grid_shape=(7, 6, 5))
        assert model.grid_shape == (7, 6, 5)
        assert model.get_config()["grid_shape"] == (7, 6, 5)

    def test_only_the_two_documented_rows_are_three_dimensional(self):
        """``paper``, ``small`` and ``tiny`` are 2D; ``paper3d`` and ``tiny3d`` are
        the only rows that ask for another rank."""
        assert {
            name for name, config in MODEL_VARIANTS.items()
            if config.get("grid_dim", 2) != 2
        } == {"paper3d", "tiny3d"}
        assert all(
            "grid_dim" not in MODEL_VARIANTS[n] for n in ("paper", "small", "tiny")
        )


# ---------------------------------------------------------------------
# axis order reaches the taps
# ---------------------------------------------------------------------


def _table(model: TopoLM, index: int = 0) -> np.ndarray:
    return _np(model.tap_layers[index].patch_table)


def _identity_model(grid_shape):
    """210 units on a non-cubic grid with NO permutation: unit i sits at cell i."""
    model = _build3d(
        embed_dim=210, num_heads=6, depth=1, ffn_intermediate_size=210,
        permute=False, grid_shape=grid_shape,
    )
    model.build((None, 12))
    return model


class TestTheAxisOrderReachesTheTaps:
    """A centre count is symmetric under a permutation of the axes.

    ``(5, 6, 7)`` and ``(7, 6, 5)`` at radius 2 both give 6 centres of 125 units,
    so a table SHAPE cannot tell a reordered grid from the right one. The pins
    below are VALUES, derived by hand from the row-major cell numbering
    ``cell (i0, i1, i2) = i0 * s0 + i1 * s1 + i2`` of an identity layout.
    """

    def test_a_5_6_7_grid_pins_the_table_values(self):
        """Strides 42, 7, 1. Centres, lexicographic: (2,2,2) (2,2,3) (2,2,4)
        (2,3,2) (2,3,3) (2,3,4); a block spans centre -2 .. centre +2, so its
        first cell is (0,0,0) (0,0,1) (0,0,2) (0,1,0) (0,1,1) (0,1,2) = 0 1 2 7 8
        9 and its last is (4,4,4) (4,4,5) (4,4,6) (4,5,4) (4,5,5) (4,5,6)."""
        model = _identity_model((5, 6, 7))
        table = _table(model)
        assert table.shape == (6, 125)
        assert table[:, 0].tolist() == [0, 1, 2, 7, 8, 9]
        assert table[:, -1].tolist() == [200, 201, 202, 207, 208, 209]
        assert table[0, 1] == 1      # axis 2 varies fastest
        assert table[0, 5] == 7      # one step on axis 1
        assert table[0, 25] == 42    # one step on axis 0

    def test_the_reversed_grid_gives_other_values_of_the_same_shape(self):
        """Strides 30, 5, 1 on (7, 6, 5); centres (2..4) x (2,3) x (2,)."""
        model = _identity_model((7, 6, 5))
        table = _table(model)
        assert table.shape == (6, 125)
        assert table[:, 0].tolist() == [0, 5, 30, 35, 60, 65]
        assert table[0, 25] == 30

    def test_the_implicit_factoring_is_the_ascending_order(self):
        implicit = _identity_model(None)
        assert implicit.grid_shape == (5, 6, 7)
        np.testing.assert_array_equal(_table(implicit), _table(_identity_model((5, 6, 7))))

    def test_the_reported_grid_is_the_taps_cell_array_shape(self):
        model = _identity_model((5, 6, 7))
        for tap in model.tap_layers:
            assert tap.layout.cell_to_unit.shape == model.grid_shape == (5, 6, 7)

    def test_a_permuted_layout_matches_the_independent_oracle_in_the_models_order(self):
        """With a real permutation the pin is the oracle on the tap's own
        ``perm`` and the model's reported grid, and the oracle on the REVERSED
        grid does not agree (anti-vacuity: the pin discriminates)."""
        keras.utils.set_random_seed(0)
        model = _build3d(
            embed_dim=210, num_heads=6, depth=1, ffn_intermediate_size=210,
            seed=3,
        )
        model.build((None, 12))
        tap = model.tap_layers[0]
        perm = _np(tap.perm)
        assert not np.array_equal(perm, np.arange(210))

        def oracle(grid):
            cell_to_unit = np.argsort(perm).reshape(grid)
            rows = [
                reference_patch_units_3d(cell_to_unit, (c0, c1, c2), 2)
                for c0 in range(2, grid[0] - 2)
                for c1 in range(2, grid[1] - 2)
                for c2 in range(2, grid[2] - 2)
            ]
            return np.stack(rows)

        np.testing.assert_array_equal(_np(tap.patch_table), oracle(model.grid_shape))
        assert not np.array_equal(_np(tap.patch_table), oracle((7, 6, 5)))


# ---------------------------------------------------------------------
# serialization, precision, device
# ---------------------------------------------------------------------


class TestThreeDimensionalSerialization:
    def test_a_keras_round_trip_preserves_values_and_every_layout(self, tmp_path):
        model = _build3d(depth=2, seed=4)
        tokens = _tokens(seed=5)
        original = _np(model(tokens, training=False)["logits"])
        saved = {w.path: _np(w).copy() for w in model.weights}

        path = str(tmp_path / "topolm3d.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert loaded.grid_dim == 3 and loaded.grid_shape == (6, 6, 6)
        restored = {w.path: _np(w) for w in loaded.weights}
        assert sorted(saved) == sorted(restored)
        for name, values in saved.items():
            np.testing.assert_array_equal(values, restored[name], err_msg=name)
        for before, after in zip(model.tap_layers, loaded.tap_layers):
            assert after.grid_dim == 3
            np.testing.assert_array_equal(_np(before.perm), _np(after.perm))
            assert tuple(after.patch_table.shape) == (8, 125)
        np.testing.assert_allclose(
            _np(loaded(tokens, training=False)["logits"]), original,
            atol=1e-6, rtol=0,
        )

    def test_a_legacy_archive_without_the_key_loads_as_two_dimensional(self, tmp_path):
        """Strip ``grid_dim`` from the archive's recorded config, as an archive
        written before the key existed would be, and load it."""
        model = TopoLM(vocab_size=VOCAB, embed_dim=256, depth=1, num_heads=4,
                       max_seq_len=64, name="topolm", seed=2)
        tokens = _tokens()
        original = _np(model(tokens, training=False)["logits"])
        path = str(tmp_path / "legacy.keras")
        model.save(path)
        stripped = str(tmp_path / "legacy_stripped.keras")
        removed = _strip_key_from_archive(path, stripped, "grid_dim")
        # Only the model's own config records it: blocks and taps are rebuilt
        # from that config on load, so there is exactly one occurrence.
        assert removed == 1

        loaded = keras.models.load_model(stripped)
        assert loaded.grid_dim == 2 and loaded.grid_shape == (16, 16)
        np.testing.assert_allclose(
            _np(loaded(tokens, training=False)["logits"]), original,
            atol=1e-6, rtol=0,
        )


def _strip_key_from_archive(source: str, target: str, key: str) -> int:
    """Copy a ``.keras`` archive with ``key`` removed from every config dict in
    ``config.json``. Returns how many occurrences were removed."""
    removed = 0

    def strip(node):
        nonlocal removed
        if isinstance(node, dict):
            if key in node:
                del node[key]
                removed += 1
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(data)
                strip(config)
                data = json.dumps(config).encode()
            dst.writestr(item, data)
    return removed


class TestThreeDimensionalPrecisionAndDevice:
    def test_the_forward_is_finite_under_mixed_float16(self):
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            model = _build3d()
            outputs = model(_tokens(), training=True)
            assert np.all(np.isfinite(_np(outputs["logits"])))
            assert all(np.isfinite(float(value)) for value in model.losses)
            assert len(model.losses) == 2 * len(TAP_SITES)
            assert {
                tap.patch_table.dtype for tap in model.tap_layers
            } == {"int64"}
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_no_variable_of_the_model_is_int32(self):
        """Device-independent arm of the XLA-on-GPU placement rule."""
        model = _build3d()
        model(_tokens(), training=False)
        assert int32_variables(model) == []
        assert {
            table.dtype
            for tap in model.tap_layers
            for table in (tap.perm, tap.patch_table, tap.triu_flat)
        } == {"int64"}

    def test_a_tf_function_trace_matches_eager(self):
        model = _build3d()
        tokens = _tokens()

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
        def traced(ids):
            return model(ids, training=False)["logits"]

        np.testing.assert_allclose(
            _np(traced(tokens)),
            _np(model(tokens, training=False)["logits"]),
            atol=1e-5, rtol=0,
        )

    @requires_gpu
    def test_a_jit_compiled_fit_runs_on_a_gpu(self):
        """Written in step 5, run on GPU 0 in step 7. The 3D tables are int64
        like the 2D ones; the arm exists because the placement rule is a device
        property that a CPU suite cannot see."""
        keras.utils.set_random_seed(0)
        model = _build3d(depth=1)
        head = _build_head(model, aggregate=True)
        head.compile(optimizer=keras.optimizers.SGD(1e-2), jit_compile=True)

        history = head.fit(_dataset(), epochs=1, verbose=0)

        assert np.isfinite(history.history["loss"][0])
        devices = {tap.patch_table.value.device for tap in model.tap_layers}
        assert all("GPU" in device for device in devices), devices

    def test_a_jit_compiled_step_actually_optimises(self):
        """The same arm on whatever device is visible (XLA on CPU compiles too)."""
        keras.utils.set_random_seed(0)
        model = _build3d(depth=1)
        head = _build_head(model, aggregate=True)
        head.compile(optimizer=keras.optimizers.SGD(1e-2), jit_compile=True)
        head(np.zeros((1, 12), dtype="int32"), training=False)
        before = {v.path: _np(v).copy() for v in model.trainable_variables}
        history = head.fit(_dataset(), epochs=2, verbose=0)
        assert np.isfinite(history.history["loss"][-1])
        assert history.history["loss"][-1] < history.history["loss"][0]
        assert any(
            not np.array_equal(before[v.path], _np(v))
            for v in model.trainable_variables
        )


# ---------------------------------------------------------------------
# a checkpoint of the other rank
# ---------------------------------------------------------------------

#: Small enough to save and load in a test. The feed-forward width is explicit
#: because ``create_topolm("tiny", ...)`` would otherwise take the row's 1024 and
#: the two sides would differ in more than the rank.
_LOAD = {
    "vocab_size": 64, "depth": 1, "max_seq_len": 16,
    "ffn_intermediate_size": 128,
}


def _save(model: TopoLM, tmp_path, name: str) -> str:
    model(_tokens(vocab=64, length=8), training=False)
    path = str(tmp_path / name)
    model.save(path)
    return path


class TestACheckpointOfTheOtherRankIsRefused:
    """Measured before the guard existed (CPU, float32, 2026-10-23):

      * 216 units, radius 2 in both ranks, 2D -> 3D and 3D -> 2D: raised
        ``Shape mismatch on layer 'block_0'`` (the tables are (112, 25) in 2D and
        (8, 125) in 3D). It raised, but never said ``grid_dim``.
      * 729 units, 2D 27 x 27 at radius 13 and 3D 9 x 9 x 9 at radius 4, both
        directions: LOADED, silently. Each tap holds one 729-unit patch in either
        rank, so ``perm`` (729,), ``patch_table`` (1, 729) and ``triu_flat``
        (265356,) have identical shapes, the by-name shape check passes, and a 3D
        model ends up with a 2D layout (``perm`` and ``patch_table`` copied).

    ``TopoLM.from_variant(pretrained=path)`` now reads the archive's recorded
    ``grid_dim`` and refuses a mismatch by name before any copy.
    """

    @pytest.mark.parametrize(
        "source_dim,target_dim", [(2, 3), (3, 2)], ids=["2D-into-3D", "3D-into-2D"]
    )
    def test_equal_width_and_radius_raise_naming_grid_dim(
        self, tmp_path, source_dim, target_dim
    ):
        base = {**_LOAD, "embed_dim": 216, "num_heads": 4, "radius": 2}
        path = _save(
            TopoLM(**base, grid_dim=source_dim, name="topolm"), tmp_path, "src.keras"
        )
        with pytest.raises(ValueError, match=r"grid_dim") as info:
            create_topolm("tiny", pretrained=path, **base, grid_dim=target_dim)
        message = str(info.value)
        assert f"saved with grid_dim={source_dim}" in message
        assert f"this model has grid_dim={target_dim}" in message

    @pytest.mark.parametrize(
        "source,target",
        [
            ({"radius": 13, "grid_dim": 2}, {"radius": 4, "grid_dim": 3}),
            ({"radius": 4, "grid_dim": 3}, {"radius": 13, "grid_dim": 2}),
        ],
        ids=["2D-r13-into-3D-r4", "3D-r4-into-2D-r13"],
    )
    def test_tables_of_identical_shape_are_still_refused(
        self, tmp_path, source, target
    ):
        """The case a shape check cannot see. Anti-vacuity first: the two models'
        tables really do have the same shapes, so only the rank differs."""
        base = {**_LOAD, "embed_dim": 729, "num_heads": 9}
        left = TopoLM(**base, **source, name="topolm")
        right = TopoLM(**base, **target, name="topolm")
        for model in (left, right):
            model(_tokens(vocab=64, length=8), training=False)
        assert [tuple(w.shape) for w in left.tap_layers[0].weights] == [
            tuple(w.shape) for w in right.tap_layers[0].weights
        ], "fixture invalid: the tables must have identical shapes in both ranks"

        path = _save(left, tmp_path, "same_shapes.keras")
        with pytest.raises(ValueError, match=r"saved with grid_dim="):
            create_topolm("tiny", pretrained=path, **base, **target)

    def test_the_same_rank_still_loads_and_restores_every_layout(self, tmp_path):
        """The guard must not over-refuse: 3D into 3D copies perm and tables."""
        base = {**_LOAD, "embed_dim": 216, "num_heads": 4, "radius": 2,
                "grid_dim": 3}
        source = TopoLM(**base, name="topolm", seed=9)
        path = _save(source, tmp_path, "three.keras")
        restored = create_topolm("tiny", pretrained=path, **base, seed=1)
        restored(_tokens(vocab=64, length=8), training=False)
        for a, b in zip(source.tap_layers, restored.tap_layers):
            np.testing.assert_array_equal(_np(a.perm), _np(b.perm))
            np.testing.assert_array_equal(_np(a.patch_table), _np(b.patch_table))

    def test_a_legacy_checkpoint_without_the_key_loads_into_a_two_dimensional_model(
        self, tmp_path
    ):
        base = {**_LOAD, "embed_dim": 256, "num_heads": 4, "radius": 2}
        path = _save(TopoLM(**base, name="topolm", seed=9), tmp_path, "two.keras")
        legacy = str(tmp_path / "two_legacy.keras")
        assert _strip_key_from_archive(path, legacy, "grid_dim") == 1
        restored = create_topolm("tiny", pretrained=legacy, **base)
        assert restored.grid_dim == 2

    def test_a_legacy_checkpoint_is_refused_by_a_three_dimensional_model(
        self, tmp_path
    ):
        """No key means 2D, so a 3D model must not take it."""
        base = {**_LOAD, "embed_dim": 216, "num_heads": 4, "radius": 2}
        path = _save(TopoLM(**base, name="topolm"), tmp_path, "two.keras")
        legacy = str(tmp_path / "two_legacy.keras")
        _strip_key_from_archive(path, legacy, "grid_dim")
        with pytest.raises(ValueError, match=r"saved with grid_dim=2"):
            create_topolm("tiny", pretrained=legacy, **base, grid_dim=3)



def _wrapped_archive(tmp_path, grid_dim: int, name: str) -> str:
    """A ``CausalLanguageModel`` around a TopoLM, saved whole -- the archive the
    README's 3D recipe ("build it with the model API and a CausalLanguageModel
    head") produces. Its top-level config is the wrapper's, with no ``grid_dim``
    key; the TopoLM config is nested under ``backbone``."""
    backbone = TopoLM(
        **_LOAD, embed_dim=216, num_heads=4, radius=2, grid_dim=grid_dim,
        name="topolm", seed=3,
    )
    wrapper = CausalLanguageModel(
        backbone=backbone, vocab_size=64, skip_head=True, output_key="logits",
        pre_shifted=True, aggregate_backbone_losses=True, verify_causality=False,
    )
    wrapper(_tokens(vocab=64, length=8), training=False)
    path = str(tmp_path / name)
    wrapper.save(path)
    return path


class TestTheGuardReadsOnlyATopoLMConfig:
    """D-017: the rank (and the grid) is read from a recorded TopoLM config and
    from nothing else.

    Measured before the fix (CPU, 2026-10-23): ``recorded.get("grid_dim", 2)``
    treated the wrapper's config, which has no such key, as 2D, so a 3D TopoLM
    wrapped in ``CausalLanguageModel`` loaded into a 3D model raised ``was saved
    with grid_dim=2`` -- false, and it hid the loader's real error. Measured
    after: all four wrapper/target combinations raise the loader's own
    ``ValueError`` ``No overlapping layers between target and source checkpoint``
    (the wrapper's layers are not a bare TopoLM's, so ``from_variant`` has never
    been able to warm-start from one, in either rank).
    """

    @pytest.mark.parametrize(
        "source_dim,target_dim",
        [(3, 3), (3, 2), (2, 2), (2, 3)],
        ids=["3D-into-3D", "3D-into-2D", "2D-into-2D", "2D-into-3D"],
    )
    def test_a_wrapped_archive_gets_the_loaders_error_not_a_guessed_rank(
        self, tmp_path, source_dim, target_dim
    ):
        path = _wrapped_archive(tmp_path, source_dim, "wrapped.keras")
        base = {**_LOAD, "embed_dim": 216, "num_heads": 4, "radius": 2}
        with pytest.raises(ValueError, match=r"No overlapping layers") as info:
            create_topolm("tiny", pretrained=path, **base, grid_dim=target_dim)
        message = str(info.value)
        assert "saved with grid_dim" not in message
        assert "Build the model with grid_dim" not in message

    def test_the_wrapper_config_really_has_no_rank_key(self, tmp_path):
        """Anti-vacuity: the fixture is the case the old default got wrong."""
        path = _wrapped_archive(tmp_path, 3, "wrapped.keras")
        with zipfile.ZipFile(path) as archive:
            top = json.loads(archive.read("config.json"))
        assert top["class_name"] == "CausalLanguageModel"
        assert "grid_dim" not in top["config"]
        assert top["config"]["backbone"]["config"]["grid_dim"] == 3


class TestACheckpointOnAnotherGridIsRefused:
    """D-017: the resolved grid is compared too, not the rank alone.

    Measured before the fix: a 3D checkpoint on ``(10, 5, 20)`` loaded into a
    model on ``(20, 5, 10)`` returned normally (every table has the same shape,
    because the centre count ``(a-2r)(b-2r)(c-2r)`` is symmetric in the axes),
    reporting ``(20, 5, 10)`` while holding the checkpoint's tables. After: the
    load raises ``ValueError`` naming both grids. The 2D analogue, ``(8, 32)``
    against ``(32, 8)``, behaved the same and is refused the same way.
    """

    @staticmethod
    def _base(rank: int) -> dict:
        if rank == 3:
            return {**_LOAD, "embed_dim": 1000, "num_heads": 4, "radius": 2,
                    "grid_dim": 3}
        return {**_LOAD, "embed_dim": 256, "num_heads": 4, "radius": 2}

    @pytest.mark.parametrize(
        "rank,saved,wanted",
        [
            (3, (10, 5, 20), (20, 5, 10)),
            (3, (10, 5, 20), (5, 10, 20)),
            (2, (8, 32), (32, 8)),
        ],
        ids=["3D-permuted", "3D-another-permutation", "2D-transposed"],
    )
    def test_a_permuted_grid_is_refused_naming_both(
        self, tmp_path, rank, saved, wanted
    ):
        base = self._base(rank)
        source = TopoLM(**base, grid_shape=saved, name="topolm", seed=3)
        path = _save(source, tmp_path, "src.keras")
        with pytest.raises(ValueError, match=r"unit grid") as info:
            create_topolm("tiny", pretrained=path, **base, grid_shape=wanted)
        message = str(info.value)
        assert f"unit grid {saved}" in message
        assert f"this model's grid is {wanted}" in message
        assert f"grid_shape={saved}" in message
        assert "saved with grid_dim" not in message

    @pytest.mark.parametrize(
        "rank,saved,default",
        [(3, (10, 5, 20), (10, 10, 10)), (2, (8, 32), (16, 16))],
        ids=["3D", "2D"],
    )
    def test_an_explicit_grid_is_refused_by_a_model_left_on_the_default(
        self, tmp_path, rank, saved, default
    ):
        """Both sides are RESOLVED, so a ``grid_shape=None`` model is compared by
        the grid it factors to."""
        base = self._base(rank)
        path = _save(
            TopoLM(**base, grid_shape=saved, name="topolm"), tmp_path, "src.keras"
        )
        with pytest.raises(ValueError, match=r"unit grid") as info:
            create_topolm("tiny", pretrained=path, **base)
        assert f"this model's grid is {default}" in str(info.value)

    @pytest.mark.parametrize(
        "rank,saved", [(3, (10, 5, 20)), (2, (8, 32))], ids=["3D", "2D"]
    )
    def test_the_same_grid_still_loads_and_restores_every_layout(
        self, tmp_path, rank, saved
    ):
        """Positive control, and a different model seed on the same grid."""
        base = self._base(rank)
        source = TopoLM(**base, grid_shape=saved, name="topolm", seed=9)
        path = _save(source, tmp_path, "src.keras")
        for seed in (9, 1):
            restored = create_topolm(
                "tiny", pretrained=path, **{**base, "seed": seed}, grid_shape=saved
            )
            restored(_tokens(vocab=64, length=8), training=False)
            assert restored.grid_shape == saved
            for a, b in zip(source.tap_layers, restored.tap_layers):
                np.testing.assert_array_equal(_np(a.perm), _np(b.perm))
                np.testing.assert_array_equal(
                    _np(a.patch_table), _np(b.patch_table)
                )

    def test_a_default_grid_archive_loads_into_the_same_grid_given_explicitly(
        self, tmp_path
    ):
        """``grid_shape=None`` and the tuple it factors to are the same grid
        (2D byte-identical behaviour: this load succeeded before the guard)."""
        base = self._base(2)
        path = _save(TopoLM(**base, name="topolm"), tmp_path, "src.keras")
        restored = create_topolm("tiny", pretrained=path, **base, grid_shape=(16, 16))
        assert restored.grid_shape == (16, 16)

    def test_a_different_width_keeps_the_loaders_shape_error(self, tmp_path):
        """The grid is compared only at an equal width; a width mismatch is the
        loader's ``Shape mismatch`` (measured: ``word_embeddings``), which names
        the real cause."""
        base = self._base(2)
        path = _save(TopoLM(**base, name="topolm"), tmp_path, "src.keras")
        with pytest.raises(ValueError, match=r"Shape mismatch on layer") as info:
            create_topolm("tiny", pretrained=path, **{**base, "embed_dim": 144})
        assert "unit grid" not in str(info.value)


class TestADocumentedLimitationOfTheGuard:
    """The guard lives in ``TopoLM.from_variant(pretrained=)``. A DIRECT call to
    ``load_backbone_from_checkpoint`` is not covered, and the topolm README says so
    ("a direct call to `load_backbone_from_checkpoint` is not covered by this
    check, only `from_variant(pretrained=)` is").

    This test pins that truth. Measured (CPU, 2026-10-23): a 3D 9 x 9 x 9 radius-4
    archive loaded into a 2D 27 x 27 radius-13 model of equal table shapes returns
    a ``TransferReport`` normally and the model keeps reporting ``(27, 27)``. If a
    later change guards the loader itself, this assertion has to be updated on
    purpose, together with that README sentence.
    """

    def test_a_direct_loader_call_does_not_see_the_rank(self, tmp_path):
        base = {**_LOAD, "embed_dim": 729, "num_heads": 9}
        path = _save(
            TopoLM(**base, radius=4, grid_dim=3, name="topolm", seed=3),
            tmp_path, "three.keras",
        )
        target = TopoLM(**base, radius=13, name="topolm")
        target.build((1, 8))
        assert (target.grid_dim, target.grid_shape) == (2, (27, 27))

        report = load_backbone_from_checkpoint(
            target,
            path,
            head_prefixes=(TASK_HEAD_NAME,),
            vocab_layers=("word_embeddings",),
            tables=("word_embeddings/embeddings", "lm_head/kernel"),
            tops=("lm_head",),
        )

        assert report is not None
        assert (target.grid_dim, target.grid_shape) == (2, (27, 27))
        with pytest.raises(ValueError, match=r"saved with grid_dim=3"):
            create_topolm("tiny", pretrained=path, **base, radius=13)

_ROWS_3D = {
    # name: (embed_dim, depth, heads, ffn, max_seq_len, radius, grid, centres, P)
    "paper3d": (1000, 12, 8, 4000, 1024, 4, (10, 10, 10), 8, 729),
    "tiny3d": (343, 4, 7, 1372, 256, 2, (7, 7, 7), 27, 125),
}


class TestThreeDimensionalVariantRows:
    """``paper3d`` and ``tiny3d``: repo-authored rows carrying ``grid_dim`` and
    ``radius`` in ``MODEL_VARIANTS`` (step 6).

    The expected numbers are written out, not read back from the table, so a row
    edited to another width fails here rather than agreeing with itself. Centres
    are ``(n - 2r) ** 3``: (10 - 8) ** 3 = 8 and (7 - 4) ** 3 = 27; a patch holds
    ``(2r + 1) ** 3``: 729 and 125.
    """

    @pytest.mark.parametrize("name", sorted(_ROWS_3D))
    def test_the_row_carries_exactly_the_documented_values(self, name):
        embed, depth, heads, ffn, seq, radius, _, _, _ = _ROWS_3D[name]
        row = MODEL_VARIANTS[name]
        assert {k: v for k, v in row.items() if k != "description"} == {
            "embed_dim": embed, "depth": depth, "num_heads": heads,
            "ffn_intermediate_size": ffn, "max_seq_len": seq,
            "radius": radius, "grid_dim": 3,
        }

    @pytest.mark.parametrize("name", sorted(_ROWS_3D))
    def test_the_description_says_it_is_ours_and_not_the_papers(self, name):
        description = MODEL_VARIANTS[name]["description"]
        assert "repo-authored" in description.lower()
        assert "not in the paper" in description
        assert "Rathi" not in description

    def test_the_rows_that_predate_them_are_two_dimensional_and_untouched(self):
        for name in ("paper", "small", "tiny"):
            assert "grid_dim" not in MODEL_VARIANTS[name], name
            assert "radius" not in MODEL_VARIANTS[name], name
        paper = MODEL_VARIANTS["paper"]
        assert (paper["embed_dim"], paper["depth"], paper["num_heads"]) == (784, 12, 16)

    @pytest.mark.parametrize("name", sorted(_ROWS_3D))
    def test_each_row_validates_at_its_own_radius_and_would_not_at_the_default(
        self, name
    ):
        config = MODEL_VARIANTS[name]
        validate_variant(name, config, config["radius"], grid_dim=3)
        validate_all_variants(DEFAULT_RADIUS)
        # Anti-vacuity: the default radius 5 needs 11 cells on every axis.
        with pytest.raises(ValueError, match=rf"'{name}'.*radius-5"):
            validate_variant(name, config, DEFAULT_RADIUS, grid_dim=3)

    @pytest.mark.parametrize("name", sorted(_ROWS_3D))
    def test_from_variant_builds_with_the_rows_rank_radius_and_grid(self, name):
        _, _, _, _, _, radius, grid, _, _ = _ROWS_3D[name]
        model = TopoLM.from_variant(name, vocab_size=64, depth=1, max_seq_len=16)
        assert model.grid_dim == 3
        assert model.radius == radius
        assert model.grid_shape == grid
        assert "description" not in model.get_config()
        assert model.get_config()["grid_dim"] == 3

    @pytest.mark.parametrize("name", sorted(_ROWS_3D))
    def test_every_tap_is_cut_on_the_rows_cube(self, name):
        _, _, _, _, _, _, grid, centres, patch_units = _ROWS_3D[name]
        model = TopoLM.from_variant(name, vocab_size=64, depth=1, max_seq_len=16)
        model(_tokens(vocab=64, length=8), training=False)
        assert len(model.tap_layers) == 2
        for tap in model.tap_layers:
            assert tuple(tap.layout.grid_shape) == grid
            assert tuple(tap.patch_table.shape) == (centres, patch_units)

    def test_the_rows_radius_can_be_overridden(self):
        model = TopoLM.from_variant(
            "tiny3d", vocab_size=64, depth=1, max_seq_len=16, radius=1
        )
        assert model.radius == 1
        assert model.grid_dim == 3
        model(_tokens(vocab=64, length=8), training=False)
        # (7 - 2) ** 3 centres of 3 ** 3 units.
        assert tuple(model.tap_layers[0].patch_table.shape) == (125, 27)

    def test_an_override_too_large_for_the_cube_names_the_largest_usable_radius(self):
        """343 = 7 x 7 x 7 holds one radius-3 patch (side 7) and no radius-4 one."""
        with pytest.raises(ValueError, match=r"below the 729 units.*radius this grid can use is 3"):
            TopoLM.from_variant("tiny3d", vocab_size=64, depth=1, radius=4)
        model = TopoLM.from_variant(
            "tiny3d", vocab_size=64, depth=1, max_seq_len=16, radius=3
        )
        model(_tokens(vocab=64, length=8), training=False)
        assert tuple(model.tap_layers[0].patch_table.shape) == (1, 343)

    def test_the_factory_delegates_the_row_unchanged(self):
        model = create_topolm("tiny3d", vocab_size=64, depth=1, max_seq_len=16)
        assert (model.grid_dim, model.radius) == (3, 2)
