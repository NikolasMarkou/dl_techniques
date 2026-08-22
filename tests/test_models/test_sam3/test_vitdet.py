"""Tests for SAM 3's ViTDet trunk (`models/SAM/SAM3/vitdet.py`).

Every value oracle here is an INDEPENDENT float64 NumPy computation derived from
the reference's published arithmetic, never from the implementation under test.
Where a wrong candidate exists (interpolated position embedding, rounded MLP
width, unit rotary position scale, multi-map trunk output, post-stack `ln_pre`)
the probe set is required to SEPARATE it, and the separation itself is asserted
so the oracle cannot silently become vacuous.

The tiny variant used throughout has a 8x8 token grid with window size 4, i.e.
the grid is exactly 2x the window in BOTH axes. That is the minimum at which a
windowed block and a global block are distinguishable at all: below it every
block sees the whole grid and the receptive-field guards are vacuous by
construction.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.SAM.SAM3.vitdet import (
    Sam3ViTDetBackbone,
    Sam3ViTDetBlock,
    _window_partition,
    _window_unpartition,
)

# ---------------------------------------------------------------------
# tiny variant -- grid 8x8, window 4 (grid == 2 x window in both axes)
# ---------------------------------------------------------------------

TINY = dict(
    img_size=16,
    patch_size=2,
    in_channels=3,
    embed_dim=8,
    depth=4,
    num_heads=2,
    mlp_ratio=4.0,
    window_size=4,
    global_att_blocks=(1, 3),
    pretrain_img_size=8,
    drop_path_rate=0.0,
    dropout_rate=0.0,
)

TINY_GRID = TINY["img_size"] // TINY["patch_size"]          # 8
TINY_PRETRAIN_GRID = TINY["pretrain_img_size"] // TINY["patch_size"]  # 4

# The shipped SAM 3 trunk, re-read from the pinned upstream clone's
# `_create_vit_backbone()`.
SHIPPED = dict(
    img_size=1008,
    patch_size=14,
    embed_dim=1024,
    depth=32,
    num_heads=16,
    mlp_ratio=4.625,
    window_size=24,
    global_att_blocks=(7, 15, 23, 31),
    pretrain_img_size=336,
    pretrain_use_cls_token=True,
    bias_patch_embed=False,
    ln_pre=True,
    ln_post=False,
    use_interp_rope=True,
    init_values=None,
)


@pytest.fixture()
def tiny_trunk():
    trunk = Sam3ViTDetBackbone(**TINY)
    trunk.build((None, TINY["img_size"], TINY["img_size"], 3))
    return trunk


@pytest.fixture()
def tiny_image():
    return np.random.RandomState(7).randn(
        2, TINY["img_size"], TINY["img_size"], 3
    ).astype("float32")


# ---------------------------------------------------------------------
# independent float64 oracles
# ---------------------------------------------------------------------


def _oracle_tile_crop(pretrain_grid: np.ndarray, grid: int) -> np.ndarray:
    """Tile-then-crop oracle: `tile(g // side + 1) [:g, :g]`, float64."""
    side = pretrain_grid.shape[0]
    tiles = grid // side + 1
    tiled = np.tile(pretrain_grid.astype(np.float64), (tiles, tiles, 1))
    return tiled[:grid, :grid, :]


def _candidate_bilinear(pretrain_grid: np.ndarray, grid: int) -> np.ndarray:
    """A WRONG candidate: align_corners=False bilinear resize, float64."""
    side = pretrain_grid.shape[0]
    src = (np.arange(grid, dtype=np.float64) + 0.5) * side / grid - 0.5
    src = np.clip(src, 0.0, side - 1.0)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, side - 1)
    frac = (src - lo)[:, None]
    rows = (
        pretrain_grid.astype(np.float64)[lo] * (1.0 - frac[..., None])
        + pretrain_grid.astype(np.float64)[hi] * frac[..., None]
    )
    frac_w = frac[None, :, :]
    return rows[:, lo] * (1.0 - frac_w) + rows[:, hi] * frac_w


def _oracle_axial_rope(
    q: np.ndarray, head_dim: int, grid: int, theta: float, scale_pos: float
) -> np.ndarray:
    """Independent float64 axial-RoPE oracle, transcribed from the published form.

    `freqs = 1 / theta ** (arange(0, D, 4)[:D//4] / D)`, coordinates
    `t_x = t % grid`, `t_y = t // grid` scaled by `scale_pos`, angles
    `concat([outer(t_x, f), outer(t_y, f)])`, applied as a complex multiply over
    ADJACENT channel pairs.
    """
    bands = np.arange(0, head_dim, 4, dtype=np.float64)[: head_dim // 4]
    freqs = 1.0 / (theta ** (bands / head_dim))
    flat = np.arange(grid * grid, dtype=np.float64)
    t_x = (flat % grid) * scale_pos
    t_y = (flat // grid) * scale_pos
    angles = np.concatenate([np.outer(t_x, freqs), np.outer(t_y, freqs)], axis=-1)
    cos_e = np.repeat(np.cos(angles), 2, axis=-1)
    sin_e = np.repeat(np.sin(angles), 2, axis=-1)
    x = q.astype(np.float64)
    pairs = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    rotated = np.stack([-pairs[..., 1], pairs[..., 0]], axis=-1)
    rotated = rotated.reshape(x.shape)
    return x * cos_e + rotated * sin_e


# ---------------------------------------------------------------------


class TestConstruction:
    def test_tiny_forward_shape(self, tiny_trunk, tiny_image):
        out = tiny_trunk(tiny_image, training=False)
        assert tuple(out.shape) == (2, TINY_GRID, TINY_GRID, TINY["embed_dim"])

    def test_compute_output_shape_matches_forward(self, tiny_trunk, tiny_image):
        declared = tiny_trunk.compute_output_shape(tiny_image.shape)
        actual = tuple(tiny_trunk(tiny_image, training=False).shape)
        assert declared == actual

    def test_test_variant_grid_is_at_least_twice_the_window(self):
        """Pre-registered STOP-IF: below 2x the guards below are vacuous."""
        assert TINY_GRID >= 2 * TINY["window_size"]
        assert TINY_GRID % TINY["window_size"] == 0

    def test_non_divisible_window_raises_rather_than_padding(self):
        with pytest.raises(ValueError, match="must divide both axes"):
            Sam3ViTDetBlock(
                dim=8, num_heads=2, input_size=(7, 7), window_size=4
            )

    def test_empty_global_blocks_raises(self):
        with pytest.raises(ValueError, match="at least one block"):
            Sam3ViTDetBackbone(**{**TINY, "global_att_blocks": ()})

    def test_out_of_range_global_block_raises(self):
        with pytest.raises(ValueError, match=r"\[0, depth"):
            Sam3ViTDetBackbone(**{**TINY, "global_att_blocks": (0, 99)})

    def test_head_width_not_multiple_of_four_raises(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            Sam3ViTDetBlock(dim=6, num_heads=3, input_size=(4, 4))

    def test_window_partition_roundtrip_is_exact(self):
        x = np.random.RandomState(1).randn(2, 8, 8, 5).astype("float32")
        back = _window_unpartition(_window_partition(x, 4), 4, 8, 8)
        assert np.max(np.abs(np.asarray(back) - x)) == 0.0


class TestTileAbsPos:
    """Oracle (a): position embedding is tile-then-crop, NOT interpolation."""

    @staticmethod
    def _pretrain_grid(trunk):
        table = keras.ops.convert_to_numpy(trunk.pos_embed).astype(np.float64)
        if trunk.pretrain_use_cls_token:
            table = table[:, 1:]
        side = trunk.pretrain_grid_size
        return table.reshape(side, side, trunk.embed_dim)

    def test_matches_float64_tile_crop_oracle(self, tiny_trunk):
        pretrain = self._pretrain_grid(tiny_trunk)
        oracle = _oracle_tile_crop(pretrain, TINY_GRID)
        got = keras.ops.convert_to_numpy(tiny_trunk._abs_pos())[0]
        assert np.max(np.abs(got.astype(np.float64) - oracle)) < 1e-6

    def test_periodicity_that_no_interpolation_can_produce(self, tiny_trunk):
        pretrain = self._pretrain_grid(tiny_trunk)
        got = keras.ops.convert_to_numpy(tiny_trunk._abs_pos())[0].astype(np.float64)
        side = TINY_PRETRAIN_GRID
        for i in range(TINY_GRID):
            for j in range(TINY_GRID):
                assert np.max(np.abs(got[i, j] - pretrain[i % side, j % side])) < 1e-6

    def test_bilinear_candidate_is_separated_by_the_probe_set(self, tiny_trunk):
        """The wrong candidate must FAIL the oracle -- proven, not assumed."""
        pretrain = self._pretrain_grid(tiny_trunk)
        tile = _oracle_tile_crop(pretrain, TINY_GRID)
        bilinear = _candidate_bilinear(pretrain, TINY_GRID)
        separation = np.max(np.abs(tile - bilinear))
        assert separation > 1e-3, (
            "the bilinear candidate is indistinguishable from tiling on this "
            f"fixture (max diff {separation}); the oracle is vacuous"
        )
        # And it is separated at a specific interior row, not only in aggregate.
        assert np.max(np.abs(tile[5] - bilinear[5])) > 1e-3

    def test_redundant_last_tile_is_discarded_in_its_entirety(self, tiny_trunk):
        """The `+ 1` in the tile count makes the last tile fully redundant.

        This is the probe INSIDE the discarded tile's would-be span: the
        pre-crop tensor is built explicitly, its final `side` rows are shown to
        be a complete copy of the pre-training grid, and the shipped output is
        shown to contain none of them.
        """
        side = TINY_PRETRAIN_GRID
        tiles = TINY_GRID // side + 1
        assert tiles == 3
        assert tiles * side - TINY_GRID == side  # exactly one whole tile dropped

        pretrain = self._pretrain_grid(tiny_trunk)
        pre_crop = np.tile(pretrain, (tiles, tiles, 1))
        assert np.max(np.abs(pre_crop[2 * side:3 * side, :side] - pretrain)) == 0.0

        got = keras.ops.convert_to_numpy(tiny_trunk._abs_pos())[0]
        assert got.shape[0] == TINY_GRID < tiles * side

    def test_each_pretrain_row_appears_exactly_twice_not_three_times(
        self, tiny_trunk
    ):
        pretrain = self._pretrain_grid(tiny_trunk)
        got = keras.ops.convert_to_numpy(tiny_trunk._abs_pos())[0].astype(np.float64)
        counts = []
        for r in range(TINY_PRETRAIN_GRID):
            counts.append(
                sum(
                    1 for i in range(TINY_GRID)
                    if np.max(np.abs(got[i, 0] - pretrain[r, 0])) < 1e-6
                )
            )
        assert counts == [2, 2, 2, 2]

    def test_equal_grid_skips_tiling_entirely(self):
        trunk = Sam3ViTDetBackbone(**{**TINY, "pretrain_img_size": TINY["img_size"]})
        trunk.build((None, TINY["img_size"], TINY["img_size"], 3))
        assert trunk.pretrain_grid_size == TINY_GRID
        table = keras.ops.convert_to_numpy(trunk.pos_embed)[0, 1:]
        got = keras.ops.convert_to_numpy(trunk._abs_pos())[0]
        assert np.max(
            np.abs(got.reshape(-1, trunk.embed_dim) - table)
        ) == 0.0


class TestMlpWidth:
    """Oracle (b): `int()` TRUNCATION, never `round()`."""

    @pytest.mark.parametrize(
        "dim,ratio,expected_int,expected_round",
        [
            (8, 4.6, 36, 37),      # 36.8  -- separates int from round
            (16, 4.6, 73, 74),     # 73.6  -- separates int from round
            (8, 3.3, 26, 26),      # 26.4  -- floor and round coincide (recorded)
            (1024, 4.625, 4736, 4736),  # the SHIPPED coincidence point
        ],
    )
    def test_hidden_width_is_truncated(self, dim, ratio, expected_int, expected_round):
        block = Sam3ViTDetBlock(
            dim=dim, num_heads=2, input_size=(4, 4), mlp_ratio=ratio
        )
        assert block.mlp_hidden_dim == expected_int
        assert int(dim * ratio) == expected_int
        assert int(round(dim * ratio)) == expected_round

    def test_probe_set_contains_a_point_that_separates_round(self):
        """Guard against the whole parametrization degenerating to coincidences."""
        separating = [
            (dim, ratio) for dim, ratio in [(8, 4.6), (16, 4.6), (8, 3.3),
                                            (1024, 4.625)]
            if int(dim * ratio) != int(round(dim * ratio))
        ]
        assert separating, "every probe is a coincidence point; oracle is vacuous"

    def test_shipped_ratio_is_a_coincidence_point(self):
        """Recorded explicitly so nobody treats the shipped config as a guard."""
        assert int(1024 * 4.625) == int(round(1024 * 4.625)) == 4736

    def test_shipped_hidden_width(self):
        block = Sam3ViTDetBlock(
            dim=1024, num_heads=16, input_size=(24, 24), mlp_ratio=4.625
        )
        assert block.mlp_hidden_dim == 4736


class TestWindowedVersusGlobal:
    """Oracle (c): assert the RECEPTIVE FIELD, never the config alone."""

    @staticmethod
    def _block(window_size):
        block = Sam3ViTDetBlock(
            dim=8,
            num_heads=2,
            input_size=(TINY_GRID, TINY_GRID),
            window_size=window_size,
            rope_pt_size=TINY["window_size"],
            use_interp_rope=True,
            mlp_ratio=4.0,
        )
        block.build((None, TINY_GRID, TINY_GRID, 8))
        return block

    @staticmethod
    def _far_token_response(block):
        x = np.random.RandomState(3).randn(1, TINY_GRID, TINY_GRID, 8).astype("float32")
        base = keras.ops.convert_to_numpy(block(x, training=False))
        pert = x.copy()
        # A SINGLE CHANNEL, not the whole token: `norm1` is a per-token
        # LayerNorm, so a uniform across-channel bump is mean-centred away and
        # the probe measures exactly 0.0 for BOTH block kinds. That was measured
        # here, not theorized -- the first draft of this probe was vacuous.
        pert[0, TINY_GRID - 1, TINY_GRID - 1, 0] += 25.0
        moved = keras.ops.convert_to_numpy(block(pert, training=False))
        return float(np.max(np.abs(moved[0, 0, 0] - base[0, 0, 0])))

    def test_windowed_block_does_not_move_a_token_in_another_window(self):
        assert self._far_token_response(self._block(TINY["window_size"])) == 0.0

    def test_global_block_does_move_that_distant_token(self):
        assert self._far_token_response(self._block(0)) > 1e-4

    @staticmethod
    def _near_token_response(block):
        """Positive arm: a token INSIDE the same window must move."""
        x = np.random.RandomState(3).randn(
            1, TINY_GRID, TINY_GRID, 8
        ).astype("float32")
        base = keras.ops.convert_to_numpy(block(x, training=False))
        pert = x.copy()
        pert[0, 1, 1, 0] += 25.0
        moved = keras.ops.convert_to_numpy(block(pert, training=False))
        return float(np.max(np.abs(moved[0, 0, 0] - base[0, 0, 0])))

    def test_windowed_block_does_move_a_token_inside_its_own_window(self):
        """Liveness arm for the exact-zero guard below.

        Without this, "a windowed block does not move a distant token" is an
        ABSENCE assertion that a completely dead attention satisfies -- the
        dead-component probe measured precisely that.
        """
        assert self._near_token_response(self._block(TINY["window_size"])) > 1e-4

    # DECISION plan-2026-08-22T035419-a11304c8/D-036
    # The response is bounded in ULP of the output, NOT pinned at `== 0.0`.
    # The exact-zero form was RED at baseline
    # (`assert 1.7881393432617188e-07 == 0.0`) and is unsatisfiable: the
    # mean-centring is exact in real arithmetic and only sub-ulp-exact in fp32.
    # Measured over 20 freshly initialized blocks: the residual is
    # 5.960464e-08 .. 2.980232e-07, which is **0.25 .. 2.50 ulp** of the output
    # amplitude at that seed -- it never reached 0.0 in any of the 20. Over the
    # same 20 blocks the SINGLE-CHANNEL bump this trap is contrasted with moves
    # the distant token by 5.393e-03 .. 3.937e-02, four to five orders of
    # magnitude more, so an 8-ulp budget (3.2x over the worst residual observed)
    # cannot swallow the real signal and the vacuity claim survives intact.
    def test_a_uniform_across_channel_bump_is_mean_centred_away(self):
        """MEASURED: the naive form of the probe above is VACUOUS.

        A perturbation applied to every channel of one token is removed by
        `norm1`'s per-token mean subtraction, so a global block's distant-token
        response collapses to fp32 rounding -- indistinguishable from a windowed
        block's exact zero. This test pins that trap permanently so the probe
        above cannot silently regress to the vacuous form.
        """
        block = self._block(0)
        x = np.random.RandomState(3).randn(
            1, TINY_GRID, TINY_GRID, 8
        ).astype("float32")
        base = keras.ops.convert_to_numpy(block(x, training=False))
        uniform = x.copy()
        uniform[0, TINY_GRID - 1, TINY_GRID - 1] += 25.0
        moved = keras.ops.convert_to_numpy(block(uniform, training=False))

        response = float(np.max(np.abs(moved[0, 0, 0] - base[0, 0, 0])))
        amplitude = float(np.max(np.abs(base[0, 0, 0])))
        ulp = float(np.spacing(np.float32(amplitude)))
        assert response <= 8 * ulp, (
            f"a uniform across-channel bump moved the distant token by "
            f"{response:.6e} = {response / ulp:.2f} ulp of the output amplitude "
            f"{amplitude:.4f}; `norm1` is no longer mean-centring it away, so "
            "the single-channel form of this probe is no longer necessary"
        )

    def test_backbone_assigns_windows_and_globals_as_configured(self, tiny_trunk):
        for i, block in enumerate(tiny_trunk.blocks):
            if i in TINY["global_att_blocks"]:
                assert block.window_size == 0
                assert block.attn.input_size == (TINY_GRID, TINY_GRID)
            else:
                assert block.window_size == TINY["window_size"]
                assert block.attn.input_size == (
                    TINY["window_size"], TINY["window_size"]
                )


class TestRopeScalePos:
    """M2.3's oracle: the global blocks' rotary position scale is NOT 1.0."""

    def test_windowed_scale_is_one_global_scale_is_the_ratio(self, tiny_trunk):
        pt = TINY["window_size"]
        for i, block in enumerate(tiny_trunk.blocks):
            if i in TINY["global_att_blocks"]:
                assert block.rope_scale_pos == pytest.approx(pt / TINY_GRID)
            else:
                assert block.rope_scale_pos == 1.0

    def test_shipped_global_scale_is_one_third(self):
        trunk = Sam3ViTDetBackbone(**SHIPPED)
        assert trunk.blocks[31].rope_scale_pos == pytest.approx(24.0 / 72.0)
        assert trunk.blocks[0].rope_scale_pos == 1.0

    def test_global_rope_matches_independent_float64_oracle(self, tiny_trunk):
        block = tiny_trunk.blocks[TINY["global_att_blocks"][0]]
        head_dim = block.attn.head_dim
        q = np.random.RandomState(11).randn(
            1, block.attn.num_heads, TINY_GRID * TINY_GRID, head_dim
        ).astype("float32")
        got = keras.ops.convert_to_numpy(block.attn.rope(ops.convert_to_tensor(q)))
        oracle = _oracle_axial_rope(
            q, head_dim, TINY_GRID, block.rope_theta,
            TINY["window_size"] / TINY_GRID,
        )
        assert np.max(np.abs(got.astype(np.float64) - oracle)) < 1e-5

    def test_unit_scale_candidate_is_separated(self, tiny_trunk):
        """A `scale_pos = 1.0` global block must FAIL the oracle above."""
        block = tiny_trunk.blocks[TINY["global_att_blocks"][0]]
        head_dim = block.attn.head_dim
        q = np.random.RandomState(11).randn(
            1, block.attn.num_heads, TINY_GRID * TINY_GRID, head_dim
        ).astype("float32")
        correct = _oracle_axial_rope(
            q, head_dim, TINY_GRID, block.rope_theta,
            TINY["window_size"] / TINY_GRID,
        )
        wrong = _oracle_axial_rope(q, head_dim, TINY_GRID, block.rope_theta, 1.0)
        assert np.max(np.abs(correct - wrong)) > 1e-2


class TestTrunkOutputArity:
    """M2.4: the trunk emits ONE map, and it is the LAST global block's."""

    def test_output_is_a_single_tensor_not_a_sequence(self, tiny_trunk, tiny_image):
        out = tiny_trunk(tiny_image, training=False)
        assert not isinstance(out, (list, tuple))
        assert len(out.shape) == 4

    def test_last_global_block_index_is_the_maximum(self, tiny_trunk):
        assert tiny_trunk.last_global_block == max(TINY["global_att_blocks"])

    def test_output_equals_the_last_global_block_not_an_earlier_one(
        self, tiny_trunk, tiny_image
    ):
        x = tiny_trunk.patch_embed(tiny_image)
        x = x + ops.cast(tiny_trunk._abs_pos(), x.dtype)
        x = tiny_trunk.norm_pre(x)
        maps = []
        for block in tiny_trunk.blocks:
            x = block(x, training=False)
            maps.append(keras.ops.convert_to_numpy(x))
        got = keras.ops.convert_to_numpy(tiny_trunk(tiny_image, training=False))
        last = TINY["global_att_blocks"][-1]
        assert np.max(np.abs(got - maps[last])) < 1e-5
        earlier = TINY["global_att_blocks"][0]
        assert np.max(np.abs(got - maps[earlier])) > 1e-4

    def test_a_trunk_with_blocks_past_the_last_global_one_cannot_be_BUILT(self):
        """F-17: this construction is now REFUSED, not merely wasteful.

        This test previously asserted the opposite -- it BUILT a
        ``depth=6, global_att_blocks=(1, 3)`` trunk and measured that zeroing
        block 5 moved the output by exactly 0.0, i.e. it PINNED the dead-block
        behaviour as if it were a feature. It was documenting the early return,
        but the configuration it used to do so is one no caller should be able
        to create: blocks 4 and 5 (1,744 parameters, measured) were built,
        parameter-counted and handed to the optimizer while contributing
        nothing.

        The early-return behaviour it meant to test is still covered, by
        ``test_output_equals_the_last_global_block_not_an_earlier_one`` above,
        which reads the LEGAL ``TINY`` trunk. The refusal itself, and the proof
        that every shipped variant survives it, live in
        ``test_vitdet_last_block_is_global.py``.
        """
        with pytest.raises(ValueError, match="must name the LAST block"):
            Sam3ViTDetBackbone(**{**TINY, "depth": 6,
                                  "global_att_blocks": (1, 3)})


class TestLnPrePlacement:
    """M2.5: `ln_pre` runs BEFORE the block stack, not after it."""

    def test_correct_order_reproduces_the_trunk(self, tiny_trunk, tiny_image):
        x = tiny_trunk.patch_embed(tiny_image)
        x = x + ops.cast(tiny_trunk._abs_pos(), x.dtype)
        x = tiny_trunk.norm_pre(x)
        for i, block in enumerate(tiny_trunk.blocks):
            x = block(x, training=False)
            if i == tiny_trunk.last_global_block:
                break
        got = keras.ops.convert_to_numpy(tiny_trunk(tiny_image, training=False))
        assert np.max(np.abs(got - keras.ops.convert_to_numpy(x))) < 1e-5

    def test_post_stack_order_is_measurably_different(self, tiny_trunk, tiny_image):
        x = tiny_trunk.patch_embed(tiny_image)
        x = x + ops.cast(tiny_trunk._abs_pos(), x.dtype)
        for i, block in enumerate(tiny_trunk.blocks):
            x = block(x, training=False)
            if i == tiny_trunk.last_global_block:
                break
        wrong = keras.ops.convert_to_numpy(tiny_trunk.norm_pre(x))
        got = keras.ops.convert_to_numpy(tiny_trunk(tiny_image, training=False))
        delta = float(np.max(np.abs(got - wrong)))
        assert delta > 1e-3, (
            "moving ln_pre after the stack is indistinguishable here "
            f"(max diff {delta}); the placement guard is vacuous"
        )

    def test_ln_pre_is_a_real_norm_and_ln_post_is_identity_at_the_settled_config(
        self,
    ):
        trunk = Sam3ViTDetBackbone(**SHIPPED)
        assert isinstance(trunk.norm_pre, keras.layers.LayerNormalization)
        assert isinstance(trunk.norm_post, keras.layers.Identity)


class TestLayerScale:
    def test_init_values_none_gives_identity_and_no_extra_parameters(self):
        block = Sam3ViTDetBlock(dim=8, num_heads=2, input_size=(4, 4),
                                init_values=None)
        assert isinstance(block.ls1, keras.layers.Identity)
        assert isinstance(block.ls2, keras.layers.Identity)
        block.build((None, 4, 4, 8))
        without = block.count_params()
        scaled = Sam3ViTDetBlock(dim=8, num_heads=2, input_size=(4, 4),
                                init_values=1e-5)
        scaled.build((None, 4, 4, 8))
        assert scaled.count_params() == without + 2 * 8

    def test_layer_scale_gain_is_unconstrained(self):
        block = Sam3ViTDetBlock(dim=8, num_heads=2, input_size=(4, 4),
                                init_values=1e-5)
        assert block.ls1.constraint is None


class TestParameterAudit:
    """Closed-form per-component counts, asserted EXACTLY."""

    @staticmethod
    def _closed_form(cfg):
        dim = cfg["embed_dim"]
        hidden = int(dim * cfg["mlp_ratio"])
        patch = (
            cfg["patch_size"] ** 2 * cfg.get("in_channels", 3) * dim
            + (dim if cfg.get("bias_patch_embed", False) else 0)
        )
        pos_tokens = (cfg["pretrain_img_size"] // cfg["patch_size"]) ** 2 + int(
            cfg.get("pretrain_use_cls_token", True)
        )
        pos = pos_tokens * dim
        pre = 2 * dim if cfg.get("ln_pre", True) else 0
        post = 2 * dim if cfg.get("ln_post", False) else 0
        per_block = (
            2 * dim + 2 * dim                     # norm1 + norm2
            + dim * 3 * dim + 3 * dim             # fused qkv (+ bias)
            + dim * dim + dim                     # attn out proj
            + dim * hidden + hidden               # mlp fc1
            + hidden * dim + dim                  # mlp fc2
        )
        return {
            "patch_embed": patch,
            "pos_embed": pos,
            "ln_pre": pre,
            "ln_post": post,
            "per_block": per_block,
            "total": patch + pos + pre + post + per_block * cfg["depth"],
        }

    def test_tiny_counts_match_exactly(self, tiny_trunk):
        expected = self._closed_form(TINY)
        assert tiny_trunk.count_params() == expected["total"]
        assert tiny_trunk.blocks[0].count_params() == expected["per_block"]
        assert tiny_trunk.patch_embed.count_params() == expected["patch_embed"]
        assert int(np.prod(tiny_trunk.pos_embed.shape)) == expected["pos_embed"]

    def test_shipped_closed_form_is_the_documented_number(self):
        expected = self._closed_form(SHIPPED)
        assert expected["per_block"] == 13_907_584
        assert expected["patch_embed"] == 602_112
        assert expected["pos_embed"] == 577 * 1024
        assert expected["ln_post"] == 0
        assert expected["total"] == 446_237_696

    def test_shipped_variant_instantiates_and_matches_the_closed_form(self):
        """A-6 MEASURED, not assumed: the shipped trunk DOES fit on GPU1.

        Recorded peak of a standalone batch-1 1008x1008 forward pass on the
        12 GB card: 8,534.3 MiB. Only the parameter count is asserted here --
        the forward pass is deliberately not repeated inside the suite.
        """
        trunk = Sam3ViTDetBackbone(**SHIPPED)
        trunk.build((None, 1008, 1008, 3))
        assert trunk.count_params() == self._closed_form(SHIPPED)["total"]
        assert trunk.compute_output_shape((None, 1008, 1008, 3)) == (
            None, 72, 72, 1024
        )

    def test_audit_is_not_vacuous_deleting_a_component_changes_the_total(self):
        """Executes the vacuity claim instead of asserting it."""
        no_pos = self._closed_form({**TINY, "pretrain_img_size": TINY["patch_size"]})
        full = self._closed_form(TINY)
        assert no_pos["total"] != full["total"]
        shallow = self._closed_form({**TINY, "depth": TINY["depth"] - 1})
        assert full["total"] - shallow["total"] == full["per_block"]


class TestSerialization:
    def test_block_config_roundtrip(self):
        block = Sam3ViTDetBlock(
            dim=8, num_heads=2, input_size=(8, 8), window_size=4,
            mlp_ratio=4.6, rope_pt_size=4, use_interp_rope=True,
            drop_path_rate=0.05, init_values=1e-5,
        )
        restored = Sam3ViTDetBlock.from_config(block.get_config())
        for key, value in block.get_config().items():
            assert restored.get_config()[key] == value

    def test_block_config_covers_every_init_parameter(self):
        import inspect

        params = set(inspect.signature(Sam3ViTDetBlock.__init__).parameters)
        params -= {"self", "kwargs"}
        config = Sam3ViTDetBlock(
            dim=8, num_heads=2, input_size=(4, 4)
        ).get_config()
        assert params <= set(config)

    def test_backbone_config_covers_every_init_parameter(self):
        import inspect

        params = set(inspect.signature(Sam3ViTDetBackbone.__init__).parameters)
        params -= {"self", "kwargs"}
        assert params <= set(Sam3ViTDetBackbone(**TINY).get_config())

    def test_full_keras_roundtrip_preserves_outputs(self, tiny_image, tmp_path):
        inputs = keras.Input(shape=tiny_image.shape[1:])
        trunk = Sam3ViTDetBackbone(**TINY)
        model = keras.Model(inputs, trunk(inputs))
        before = keras.ops.convert_to_numpy(model(tiny_image, training=False))
        path = tmp_path / "sam3_vitdet.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(tiny_image, training=False))
        assert np.max(np.abs(after - before)) == 0.0


class TestTrainingBehaviour:
    def test_gradients_reach_every_trainable_weight(self, tiny_trunk, tiny_image):
        with tf.GradientTape() as tape:
            out = tiny_trunk(tiny_image, training=True)
            loss = ops.mean(ops.square(out))
        grads = tape.gradient(loss, tiny_trunk.trainable_variables)
        missing = [
            v.path for v, g in zip(tiny_trunk.trainable_variables, grads)
            if g is None
        ]
        assert missing == []

    def test_inference_is_deterministic(self, tiny_trunk, tiny_image):
        a = keras.ops.convert_to_numpy(tiny_trunk(tiny_image, training=False))
        b = keras.ops.convert_to_numpy(tiny_trunk(tiny_image, training=False))
        assert np.max(np.abs(a - b)) == 0.0

    def test_drop_path_is_active_in_training_only(self):
        trunk = Sam3ViTDetBackbone(**{**TINY, "drop_path_rate": 0.9})
        trunk.build((None, TINY["img_size"], TINY["img_size"], 3))
        assert trunk.blocks[0].drop_path.drop_path_rate == 0.0
        assert trunk.blocks[-1].drop_path.drop_path_rate == pytest.approx(0.9)


class TestPurity:
    def test_library_file_uses_no_tensorflow(self):
        import re

        from dl_techniques.models.SAM.SAM3 import vitdet

        source = open(vitdet.__file__, encoding="utf-8").read()
        assert "import tensorflow" not in source
        assert re.search(r"\btf\.[A-Za-z_]", source) is None
