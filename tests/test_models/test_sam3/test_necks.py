"""Tests for SAM 3's dual SimpleFPN neck (`models/sam3/necks.py`).

The positional-encoding oracle here is an INDEPENDENT float64 NumPy computation
of the sine formula, written channels-LAST. That orientation is deliberate: the
layer the neck reuses returns channels-FIRST, so an oracle written channels-last
is the only kind that can see a forgotten transpose.

The tiny variant is chosen so that ONE of its four scales is square AND has a
side equal to ``d_model`` (grid 8, ``d_model`` 8). At that scale a forgotten
transpose is shape-COMPATIBLE and therefore silent, which is exactly the
condition under which the axis-order defect must still be caught by value. Every
other scale catches it by the call-site axis-order check instead.

Iteration 1 measured a conv index-mirroring defect INERT because Keras infers a
conv's input width at build time (D-015): any CONSISTENT conv-to-level
assignment is numerically identical. The guards below therefore pin the channel
WIDTHS (through the closed-form parameter count) and the weight BINDING (through
an independence probe), never "did the output move".
"""

import math
import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.sam3.necks import (
    SUPPORTED_SCALES,
    Sam3DualViTDetNeck,
    _build_scale_stack,
    _encode_position,
)

# ---------------------------------------------------------------------
# tiny variant -- trunk grid 8x8, dim 16, d_model 8
# ---------------------------------------------------------------------

TINY_DIM = 16
TINY_D_MODEL = 8
TINY_GRID = 8

# The resolution ladder is a pure function of the trunk grid and the scales.
TINY_LADDER = [32, 16, 8, 4]

# The shipped SAM 3 neck, re-read from the pinned upstream clone's
# `_create_vit_neck()` / `_create_position_encoding()`:
# d_model=256, scale_factors=[4.0, 2.0, 1.0, 0.5], add_sam2_neck=True (the
# builder passes `enable_inst_interactivity`, which defaults to True), and a
# sine encoding whose upstream `num_pos_feats=256` is a TOTAL width.
SHIPPED_DIM = 1024
SHIPPED_D_MODEL = 256
SHIPPED_TRUNK_GRID = 72
SHIPPED_LADDER = [288, 144, 72, 36]


@pytest.fixture()
def tiny_neck():
    neck = Sam3DualViTDetNeck(dim=TINY_DIM, d_model=TINY_D_MODEL)
    neck.build((None, TINY_GRID, TINY_GRID, TINY_DIM))
    return neck


@pytest.fixture()
def tiny_trunk_map():
    return np.random.RandomState(11).randn(
        2, TINY_GRID, TINY_GRID, TINY_DIM
    ).astype("float32")


# ---------------------------------------------------------------------
# independent float64 oracles
# ---------------------------------------------------------------------


def _oracle_sine_pe(
        height: int,
        width: int,
        num_pos_feats: int,
        temperature: float = 10000.0,
) -> np.ndarray:
    """Channels-LAST float64 sine positional encoding, `(h, w, 2*num_pos_feats)`."""
    scale = 2.0 * math.pi
    eps = 1e-6
    rows = np.arange(1, height + 1, dtype=np.float64)[:, None] * np.ones(
        (1, width), dtype=np.float64
    )
    cols = np.ones((height, 1), dtype=np.float64) * np.arange(
        1, width + 1, dtype=np.float64
    )[None, :]
    rows = (rows - 0.5) / (height + eps) * scale
    cols = (cols - 0.5) / (width + eps) * scale

    dim_t = temperature ** (
        2.0 * (np.arange(num_pos_feats, dtype=np.float64) // 2) / num_pos_feats
    )

    def _interleave(angles: np.ndarray) -> np.ndarray:
        sin_part = np.sin(angles[..., 0::2])
        cos_part = np.cos(angles[..., 1::2])
        stacked = np.stack([sin_part, cos_part], axis=-1)
        return stacked.reshape(*angles.shape[:-1], -1)

    pos_y = _interleave(rows[..., None] / dim_t)
    pos_x = _interleave(cols[..., None] / dim_t)
    return np.concatenate([pos_y, pos_x], axis=-1)


def _oracle_branch_params(dim: int, d_model: int, scale: float) -> int:
    """Closed-form trainable-parameter count of one scale branch."""
    if scale == 4.0:
        total = 4 * dim * (dim // 2) + (dim // 2)
        total += 4 * (dim // 2) * (dim // 4) + (dim // 4)
        out_dim = dim // 4
    elif scale == 2.0:
        total = 4 * dim * (dim // 2) + (dim // 2)
        out_dim = dim // 2
    elif scale == 1.0:
        total = 0
        out_dim = dim
    elif scale == 0.5:
        total = 0
        out_dim = dim
    else:
        raise AssertionError(f"unsupported probe scale {scale}")
    total += out_dim * d_model + d_model           # conv 1x1
    total += 9 * d_model * d_model + d_model       # conv 3x3
    return total


def _oracle_neck_params(
        dim: int, d_model: int, scales=SUPPORTED_SCALES, dual: bool = True
) -> int:
    single = sum(_oracle_branch_params(dim, d_model, s) for s in scales)
    return single * (2 if dual else 1)


# ---------------------------------------------------------------------


class TestConstruction:
    def test_forward_returns_the_four_declared_keys(
            self, tiny_neck, tiny_trunk_map
    ):
        out = tiny_neck(tiny_trunk_map)
        assert set(out) == set(Sam3DualViTDetNeck.FEATURE_KEYS)

    def test_every_key_holds_one_entry_per_scale(self, tiny_neck, tiny_trunk_map):
        out = tiny_neck(tiny_trunk_map)
        for key in Sam3DualViTDetNeck.FEATURE_KEYS:
            assert len(out[key]) == len(SUPPORTED_SCALES)

    def test_single_neck_leaves_the_sam2_lists_empty_without_changing_the_keys(
            self,
    ):
        neck = Sam3DualViTDetNeck(
            dim=TINY_DIM, d_model=TINY_D_MODEL, add_sam2_neck=False
        )
        out = neck(np.zeros((1, TINY_GRID, TINY_GRID, TINY_DIM), "float32"))
        assert set(out) == set(Sam3DualViTDetNeck.FEATURE_KEYS)
        assert out["sam2_features"] == [] and out["sam2_pos"] == []
        assert len(out["sam3_features"]) == len(SUPPORTED_SCALES)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="multiple of 4"):
            Sam3DualViTDetNeck(dim=18, d_model=8)

    def test_odd_d_model_raises(self):
        with pytest.raises(ValueError, match="positive and even"):
            Sam3DualViTDetNeck(dim=16, d_model=7)

    def test_unsupported_scale_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            Sam3DualViTDetNeck(dim=16, d_model=8, scale_factors=(3.0,))

    def test_empty_scale_factors_raises(self):
        with pytest.raises(ValueError, match="at least one scale"):
            Sam3DualViTDetNeck(dim=16, d_model=8, scale_factors=())

    def test_wrong_trunk_width_raises(self):
        neck = Sam3DualViTDetNeck(dim=16, d_model=8)
        with pytest.raises(ValueError, match="must equal the configured dim"):
            neck.build((None, 8, 8, 32))

    def test_rank_three_input_raises(self):
        neck = Sam3DualViTDetNeck(dim=16, d_model=8)
        with pytest.raises(ValueError, match="rank-4"):
            neck.build((None, 8, 16))

    def test_odd_trunk_grid_raises_because_the_half_scale_cannot_halve_it(self):
        neck = Sam3DualViTDetNeck(dim=16, d_model=8)
        with pytest.raises(ValueError, match="odd extent"):
            neck.build((None, 7, 7, 16))


class TestResolutionLadder:
    """The ladder is asserted BY VALUE, at the tiny variant and at 72x72."""

    def test_tiny_ladder_is_32_16_8_4(self, tiny_neck, tiny_trunk_map):
        out = tiny_neck(tiny_trunk_map)
        assert [int(f.shape[1]) for f in out["sam3_features"]] == TINY_LADDER
        assert [int(f.shape[2]) for f in out["sam3_features"]] == TINY_LADDER

    def test_tiny_ladder_holds_for_the_second_neck_too(
            self, tiny_neck, tiny_trunk_map
    ):
        out = tiny_neck(tiny_trunk_map)
        assert [int(f.shape[1]) for f in out["sam2_features"]] == TINY_LADDER

    def test_positional_encodings_follow_the_same_ladder(
            self, tiny_neck, tiny_trunk_map
    ):
        out = tiny_neck(tiny_trunk_map)
        assert [int(p.shape[1]) for p in out["sam3_pos"]] == TINY_LADDER
        assert [int(p.shape[1]) for p in out["sam2_pos"]] == TINY_LADDER

    def test_every_scale_is_projected_to_d_model(self, tiny_neck, tiny_trunk_map):
        out = tiny_neck(tiny_trunk_map)
        for key in Sam3DualViTDetNeck.FEATURE_KEYS:
            assert all(int(t.shape[-1]) == TINY_D_MODEL for t in out[key])

    def test_shipped_ladder_at_a_72x72_trunk_is_288_144_72_36(self):
        neck = Sam3DualViTDetNeck(dim=SHIPPED_DIM, d_model=SHIPPED_D_MODEL)
        shapes = neck.compute_output_shape(
            (None, SHIPPED_TRUNK_GRID, SHIPPED_TRUNK_GRID, SHIPPED_DIM)
        )
        assert [s[1] for s in shapes["sam3_features"]] == SHIPPED_LADDER
        assert [s[2] for s in shapes["sam3_features"]] == SHIPPED_LADDER
        assert all(s[-1] == SHIPPED_D_MODEL for s in shapes["sam3_features"])

    def test_compute_output_shape_agrees_with_the_forward_pass(
            self, tiny_neck, tiny_trunk_map
    ):
        declared = tiny_neck.compute_output_shape(
            (2, TINY_GRID, TINY_GRID, TINY_DIM)
        )
        out = tiny_neck(tiny_trunk_map)
        for key in Sam3DualViTDetNeck.FEATURE_KEYS:
            assert [tuple(t.shape) for t in out[key]] == [
                tuple(s) for s in declared[key]
            ]

    def test_a_single_deconv_on_the_four_scale_would_break_the_ladder(self):
        """Separation guard: the ladder's 4x entry is not reachable with one deconv."""
        one_deconv = Sam3DualViTDetNeck(
            dim=TINY_DIM, d_model=TINY_D_MODEL, scale_factors=(2.0,)
        )
        shapes = one_deconv.compute_output_shape(
            (None, TINY_GRID, TINY_GRID, TINY_DIM)
        )
        assert shapes["sam3_features"][0][1] == 16 != TINY_LADDER[0]


class TestBranchComposition:
    """The per-scale resamplers, by layer type and by declared width."""

    @pytest.mark.parametrize("scale,expected_types", [
        (4.0, ["Conv2DTranspose", "Activation", "Conv2DTranspose",
               "Conv2D", "Conv2D"]),
        (2.0, ["Conv2DTranspose", "Conv2D", "Conv2D"]),
        (1.0, ["Conv2D", "Conv2D"]),
        (0.5, ["MaxPooling2D", "Conv2D", "Conv2D"]),
    ])
    def test_branch_layer_sequence(self, scale, expected_types):
        branch = _build_scale_stack(TINY_DIM, TINY_D_MODEL, scale, "p")
        assert [type(l).__name__ for l in branch] == expected_types

    def test_four_scale_narrows_dim_to_half_then_quarter(self):
        branch = _build_scale_stack(TINY_DIM, TINY_D_MODEL, 4.0, "p")
        assert branch[0].filters == TINY_DIM // 2
        assert branch[2].filters == TINY_DIM // 4

    def test_the_gelu_sits_between_the_two_deconvs(self):
        branch = _build_scale_stack(TINY_DIM, TINY_D_MODEL, 4.0, "p")
        assert type(branch[1]).__name__ == "Activation"

    def test_gelu_is_exact_not_the_tanh_approximation(self):
        """`nn.GELU()` upstream is the erf form; the tanh form is a different fn."""
        branch = _build_scale_stack(TINY_DIM, TINY_D_MODEL, 4.0, "p")
        probe = np.linspace(-5.0, 5.0, 201, dtype="float64")[None, :]
        exact = np.asarray(
            keras.activations.gelu(ops.convert_to_tensor(probe), approximate=False)
        )
        approx = np.asarray(
            keras.activations.gelu(ops.convert_to_tensor(probe), approximate=True)
        )
        # The probe SEPARATES the two candidates before it is used as an oracle.
        # The erf/tanh gap peaks near |x| ~ 2, so a probe confined to small |x|
        # or to a coarse grid measures a difference indistinguishable from
        # float32 noise -- that is a coincidence point, not a pass.
        assert np.abs(exact - approx).max() > 2e-4
        np.testing.assert_allclose(
            np.asarray(branch[1](ops.convert_to_tensor(probe))), exact, atol=2e-6
        )
        # The tolerance must stay far below the separation, or the oracle is
        # blind to the tanh candidate: 2e-6 vs a 2e-4 floor is a 100x margin.
        assert np.abs(exact - approx).max() > 100 * 2e-6

    def test_both_projection_convs_carry_a_bias(self):
        for scale in SUPPORTED_SCALES:
            branch = _build_scale_stack(TINY_DIM, TINY_D_MODEL, scale, "p")
            assert branch[-2].use_bias is True
            assert branch[-1].use_bias is True

    def test_the_three_by_three_conv_preserves_resolution(self):
        branch = _build_scale_stack(TINY_DIM, TINY_D_MODEL, 1.0, "p")
        assert branch[-1].kernel_size == (3, 3)
        assert branch[-1].padding == "same"

    def test_there_is_no_normalization_anywhere_in_the_neck(self, tiny_neck):
        """`Sam3DualViTDetNeck` has NO `neck_norm`; the SAM 3.1 tri-neck does."""
        names = [type(l).__name__ for l in tiny_neck._flatten_layers()]
        assert not any(
            "Norm" in n or n.endswith("Normalization") for n in names
        ), names


class TestDualIndependence:
    """M3.1's guards: a weight COUNT and a weight-BINDING probe.

    A shared-stack port has identical output shapes AND identical forward values
    on a fresh model, so an output comparison is blind to it by construction.
    """

    def test_tiny_parameter_count_matches_the_closed_form(self, tiny_neck):
        assert tiny_neck.count_params() == _oracle_neck_params(
            TINY_DIM, TINY_D_MODEL
        )

    def test_the_dual_neck_has_exactly_twice_a_single_neck_s_parameters(self):
        dual = Sam3DualViTDetNeck(dim=TINY_DIM, d_model=TINY_D_MODEL)
        single = Sam3DualViTDetNeck(
            dim=TINY_DIM, d_model=TINY_D_MODEL, add_sam2_neck=False
        )
        dual.build((None, TINY_GRID, TINY_GRID, TINY_DIM))
        single.build((None, TINY_GRID, TINY_GRID, TINY_DIM))
        assert dual.count_params() == 2 * single.count_params()

    def test_shipped_parameter_count_matches_the_closed_form(self):
        neck = Sam3DualViTDetNeck(dim=SHIPPED_DIM, d_model=SHIPPED_D_MODEL)
        neck.build((None, SHIPPED_TRUNK_GRID, SHIPPED_TRUNK_GRID, SHIPPED_DIM))
        expected = _oracle_neck_params(SHIPPED_DIM, SHIPPED_D_MODEL)
        assert expected == 15_604_224
        assert neck.count_params() == expected

    def test_the_two_stacks_share_no_layer_object(self, tiny_neck):
        sam3_ids = {id(l) for l in tiny_neck.sam3_convs}
        sam2_ids = {id(l) for l in tiny_neck.sam2_convs}
        assert sam3_ids and sam2_ids
        assert sam3_ids.isdisjoint(sam2_ids)

    def test_the_two_stacks_share_no_weight_variable(self, tiny_neck):
        sam3_ids = {id(w) for l in tiny_neck.sam3_convs for w in l.weights}
        sam2_ids = {id(w) for l in tiny_neck.sam2_convs for w in l.weights}
        assert len(sam3_ids) == len(sam2_ids) > 0
        assert sam3_ids.isdisjoint(sam2_ids)

    def test_perturbing_the_sam3_stack_does_not_move_the_sam2_output(
            self, tiny_neck, tiny_trunk_map
    ):
        before = tiny_neck(tiny_trunk_map)
        sam2_before = [np.asarray(t) for t in before["sam2_features"]]
        sam3_before = [np.asarray(t) for t in before["sam3_features"]]

        target = tiny_neck.branches(tiny_neck.sam3_convs)[0][-1]
        kernel = np.asarray(target.kernel)
        target.kernel.assign(kernel + 3.0)

        after = tiny_neck(tiny_trunk_map)
        # The OTHER neck must not move -- exactly zero, not "within tolerance".
        for old, new in zip(sam2_before, after["sam2_features"]):
            assert np.abs(old - np.asarray(new)).max() == 0.0
        # And the perturbation must be live, or the probe above is vacuous.
        assert np.abs(
            sam3_before[0] - np.asarray(after["sam3_features"][0])
        ).max() > 1e-3

    def test_perturbing_the_sam2_stack_does_not_move_the_sam3_output(
            self, tiny_neck, tiny_trunk_map
    ):
        before = tiny_neck(tiny_trunk_map)
        sam3_before = [np.asarray(t) for t in before["sam3_features"]]
        sam2_before = np.asarray(before["sam2_features"][1])

        target = tiny_neck.branches(tiny_neck.sam2_convs)[1][-1]
        target.kernel.assign(np.asarray(target.kernel) - 2.5)

        after = tiny_neck(tiny_trunk_map)
        for old, new in zip(sam3_before, after["sam3_features"]):
            assert np.abs(old - np.asarray(new)).max() == 0.0
        assert np.abs(
            sam2_before - np.asarray(after["sam2_features"][1])
        ).max() > 1e-3

    def test_every_scale_of_both_necks_emits_a_non_constant_map(
            self, tiny_neck, tiny_trunk_map
    ):
        """Positive LIVENESS arm, added because the dead-component probe needed it.

        Zeroing one neck's 3x3 projection makes that whole neck emit exactly
        zero, and it left 57 of 57 guards GREEN: the independence probes are
        ABSENCE assertions a dead branch satisfies by construction, the parameter
        counts are unchanged, the ladder is unchanged, and the sine encoding is a
        function of the GRID only. The carried lesson is that a constant output
        has a fingerprint -- count unique values, never read a magnitude.
        """
        out = tiny_neck(tiny_trunk_map)
        for key in ("sam3_features", "sam2_features"):
            for scale_index, tensor in enumerate(out[key]):
                values = np.asarray(tensor)
                assert np.unique(values).size > 1, (key, scale_index)
                assert float(np.std(values)) > 1e-6, (key, scale_index)

    def test_at_initialization_the_two_necks_already_differ(
            self, tiny_neck, tiny_trunk_map
    ):
        """Independent init, so a shared-stack port is ALSO caught here...

        ...but only because Keras seeds each conv separately. This assertion is
        recorded as WEAKER than the two above: it would hold trivially for any
        two differently-seeded stacks and would FAIL on a deliberately
        weight-tied-at-init port that still shared no variables.
        """
        out = tiny_neck(tiny_trunk_map)
        deltas = [
            np.abs(np.asarray(a) - np.asarray(b)).max()
            for a, b in zip(out["sam3_features"], out["sam2_features"])
        ]
        assert min(deltas) > 0.0


class TestPositionalEncoding:
    """M3.2 / the transpose mutation: per-scale values, channels-LAST."""

    def test_matches_the_float64_channels_last_oracle_at_every_scale(
            self, tiny_neck, tiny_trunk_map
    ):
        out = tiny_neck(tiny_trunk_map)
        for pos, side in zip(out["sam3_pos"], TINY_LADDER):
            oracle = _oracle_sine_pe(side, side, TINY_D_MODEL // 2)
            got = np.asarray(pos)[0]
            np.testing.assert_allclose(got, oracle, atol=1e-5)

    def test_the_square_equal_width_scale_exists_so_a_lost_transpose_is_silent(
            self,
    ):
        """The 1.0 scale is 8x8x8: transposing it is SHAPE-compatible."""
        assert TINY_GRID == TINY_D_MODEL
        assert 1.0 in SUPPORTED_SCALES

    def test_value_oracle_at_the_square_equal_width_scale_alone(self):
        """The probe that catches a lost transpose BY VALUE, not by shape."""
        neck = Sam3DualViTDetNeck(
            dim=TINY_DIM, d_model=TINY_D_MODEL, scale_factors=(1.0,)
        )
        feat = np.random.RandomState(3).randn(
            1, TINY_GRID, TINY_GRID, TINY_DIM
        ).astype("float32")
        pos = np.asarray(neck(feat)["sam3_pos"][0])[0]
        oracle = _oracle_sine_pe(TINY_GRID, TINY_GRID, TINY_D_MODEL // 2)
        # Separation: the transposed candidate is shape-legal here and WRONG.
        transposed = np.transpose(oracle, (2, 0, 1))
        assert transposed.shape == oracle.shape
        assert np.abs(transposed - oracle).max() > 0.1
        np.testing.assert_allclose(pos, oracle, atol=1e-5)

    def test_the_encoding_width_is_d_model_not_twice_d_model(
            self, tiny_neck, tiny_trunk_map
    ):
        """G-1e: the reused layer emits `2 * num_pos_feats`."""
        out = tiny_neck(tiny_trunk_map)
        assert tiny_neck.position_encoding.num_pos_feats == TINY_D_MODEL // 2
        for pos in out["sam3_pos"]:
            assert int(pos.shape[-1]) == TINY_D_MODEL

    def test_each_scale_gets_its_own_encoding_not_a_resampled_one(
            self, tiny_neck, tiny_trunk_map
    ):
        """M3.2's guard: a single encoding reused across scales is WRONG.

        The normalized coordinate pitch depends on the grid, so the coarse
        encoding nearest-resampled to a finer grid differs measurably from the
        finer grid's own encoding.
        """
        out = tiny_neck(tiny_trunk_map)
        fine = np.asarray(out["sam3_pos"][0])[0]          # 32x32
        coarse = np.asarray(out["sam3_pos"][2])[0]        # 8x8
        upsampled = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)
        assert upsampled.shape == fine.shape
        assert np.abs(upsampled - fine).max() > 1e-2

    def test_the_encoding_is_not_constant_along_either_axis(
            self, tiny_neck, tiny_trunk_map
    ):
        out = tiny_neck(tiny_trunk_map)
        pos = np.asarray(out["sam3_pos"][1])[0]
        assert np.abs(pos[0, :, :] - pos[-1, :, :]).max() > 1e-3   # varies in h
        assert np.abs(pos[:, 0, :] - pos[:, -1, :]).max() > 1e-3   # varies in w

    def test_the_encoding_is_independent_of_the_feature_values(
            self, tiny_neck
    ):
        """A sine PE is a function of the GRID only -- a learned one is not."""
        a = np.zeros((1, TINY_GRID, TINY_GRID, TINY_DIM), "float32")
        b = np.random.RandomState(5).randn(
            1, TINY_GRID, TINY_GRID, TINY_DIM
        ).astype("float32") * 10.0
        pos_a = np.asarray(tiny_neck(a)["sam3_pos"][0])
        pos_b = np.asarray(tiny_neck(b)["sam3_pos"][0])
        assert np.abs(pos_a - pos_b).max() == 0.0

    def test_the_two_necks_get_identical_encodings_at_the_same_scale(
            self, tiny_neck, tiny_trunk_map
    ):
        out = tiny_neck(tiny_trunk_map)
        for p3, p2 in zip(out["sam3_pos"], out["sam2_pos"]):
            assert np.abs(np.asarray(p3) - np.asarray(p2)).max() == 0.0

    def test_encode_position_rejects_a_mis_oriented_encoding(self):
        """The call-site axis-order guard, exercised directly.

        The helper unconditionally transposes channels-first to channels-last,
        so an encoder that already returns channels-LAST comes out mis-oriented
        and must be REFUSED rather than added to the feature. This is the same
        check that catches the reverse defect (a lost transpose) at every scale
        whose channel count differs from its spatial extent.
        """
        class _AlreadyChannelsLast(keras.layers.Layer):
            def call(self, inputs):
                return ops.zeros_like(inputs)

        feat = ops.zeros((1, 4, 6, TINY_D_MODEL))
        with pytest.raises(ValueError, match="not transposed"):
            _encode_position(_AlreadyChannelsLast(), feat, TINY_D_MODEL)

    def test_encode_position_rejects_a_double_width_encoding(self, tiny_neck):
        """The call-site WIDTH guard: `2 * num_pos_feats`, not `num_pos_feats`."""
        from dl_techniques.layers.embedding.positional_embedding_sine_2d import (
            PositionEmbeddingSine2D,
        )
        wide = PositionEmbeddingSine2D(num_pos_feats=TINY_D_MODEL)
        feat = ops.zeros((1, 4, 6, TINY_D_MODEL))
        with pytest.raises(ValueError, match="must match the feature"):
            _encode_position(wide, feat, TINY_D_MODEL)


class TestTrunkCoupling:
    """The neck consumes the trunk's ONE channels-last map (D-092)."""

    def test_accepts_the_real_trunk_output_shape_and_dtype(self):
        from dl_techniques.models.sam3.vitdet import Sam3ViTDetBackbone
        trunk = Sam3ViTDetBackbone(
            img_size=16, patch_size=2, in_channels=3, embed_dim=TINY_DIM,
            depth=2, num_heads=2, window_size=4, global_att_blocks=(1,),
            pretrain_img_size=8, drop_path_rate=0.0,
        )
        image = np.random.RandomState(2).randn(1, 16, 16, 3).astype("float32")
        feature = trunk(image)
        assert len(feature.shape) == 4 and int(feature.shape[-1]) == TINY_DIM
        neck = Sam3DualViTDetNeck(dim=TINY_DIM, d_model=TINY_D_MODEL)
        out = neck(feature)
        assert [int(f.shape[1]) for f in out["sam3_features"]] == TINY_LADDER

    def test_a_channels_first_trunk_map_is_rejected(self):
        neck = Sam3DualViTDetNeck(dim=TINY_DIM, d_model=TINY_D_MODEL)
        with pytest.raises(ValueError, match="must equal the configured dim"):
            neck.build((None, TINY_DIM, TINY_GRID, TINY_GRID))


class TestSerialization:
    def test_config_covers_every_init_parameter(self):
        import inspect
        params = set(inspect.signature(
            Sam3DualViTDetNeck.__init__
        ).parameters) - {"self", "kwargs"}
        config = Sam3DualViTDetNeck(dim=16, d_model=8).get_config()
        assert params.issubset(config)

    def test_config_roundtrip_preserves_every_value(self):
        neck = Sam3DualViTDetNeck(
            dim=32, d_model=10, scale_factors=(2.0, 1.0),
            add_sam2_neck=False, pe_temperature=5000.0,
        )
        clone = Sam3DualViTDetNeck.from_config(neck.get_config())
        for key, value in neck.get_config().items():
            assert clone.get_config()[key] == value

    def test_full_keras_roundtrip_preserves_outputs(self, tmp_path):
        neck = Sam3DualViTDetNeck(dim=TINY_DIM, d_model=TINY_D_MODEL)
        inputs = keras.Input(shape=(TINY_GRID, TINY_GRID, TINY_DIM))
        out = neck(inputs)
        flat = out["sam3_features"] + out["sam3_pos"] + out["sam2_features"]
        model = keras.Model(inputs, flat)
        probe = np.random.RandomState(13).randn(
            2, TINY_GRID, TINY_GRID, TINY_DIM
        ).astype("float32")
        before = [np.asarray(t) for t in model.predict(probe, verbose=0)]

        path = tmp_path / "neck.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = [np.asarray(t) for t in restored.predict(probe, verbose=0)]
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_the_sub_layer_lists_are_flat_not_nested(self, tiny_neck):
        """D-098: nesting them silently loses weights on `.keras` round trip."""
        for stack in (tiny_neck.sam3_convs, tiny_neck.sam2_convs):
            assert stack
            assert all(isinstance(l, keras.layers.Layer) for l in stack)
            assert not any(isinstance(l, (list, tuple)) for l in stack)

    def test_branches_reslices_the_flat_stack_back_into_scales(self, tiny_neck):
        branches = tiny_neck.branches(tiny_neck.sam3_convs)
        assert len(branches) == len(SUPPORTED_SCALES)
        assert [len(b) for b in branches] == [5, 3, 2, 3]
        assert sum(len(b) for b in branches) == len(tiny_neck.sam3_convs)
        assert tiny_neck.branches([]) == []

    def test_the_framework_defect_D_098_guards_against_is_real(self, tmp_path):
        """A NESTED sub-layer list loses its weights; a FLAT one does not.

        This is a framework-behaviour regression test, not a test of this
        module: it fails loudly if a future Keras release fixes the nesting
        case, at which point D-098's constraint can be revisited. Both arms run,
        so neither direction is asserted on faith.
        """
        @keras.saving.register_keras_serializable(package="sam3_d098_probe")
        class _Probe(keras.layers.Layer):
            def __init__(self, nested=True, **kwargs):
                super().__init__(**kwargs)
                self.nested = nested
                self.groups = (
                    [[keras.layers.Dense(4, name=f"d{i}{j}") for j in range(2)]
                     for i in range(2)]
                    if nested else
                    [keras.layers.Dense(4, name=f"f{i}") for i in range(4)]
                )

            def build(self, input_shape):
                groups = self.groups if self.nested else [self.groups]
                for group in groups:
                    shape = input_shape
                    for layer in group:
                        layer.build(shape)
                        shape = layer.compute_output_shape(shape)
                super().build(input_shape)

            def call(self, inputs):
                if not self.nested:
                    return sum(l(inputs) for l in self.groups)
                total = None
                for group in self.groups:
                    x = inputs
                    for layer in group:
                        x = layer(x)
                    total = x if total is None else total + x
                return total

            def get_config(self):
                config = super().get_config()
                config["nested"] = self.nested
                return config

        deltas = {}
        for nested in (True, False):
            layer = _Probe(nested=nested)
            inputs = keras.Input(shape=(4,))
            model = keras.Model(inputs, layer(inputs))
            probe = np.random.RandomState(0).randn(3, 4).astype("float32")
            before = np.asarray(model.predict(probe, verbose=0))
            path = tmp_path / f"probe_{nested}.keras"
            model.save(path)
            after = np.asarray(
                keras.models.load_model(path).predict(probe, verbose=0)
            )
            deltas[nested] = float(np.abs(before - after).max())

        assert deltas[False] == 0.0, deltas
        assert deltas[True] > 1e-3, (
            "nested sub-layer lists now round-trip correctly; D-098's flat "
            f"storage constraint can be revisited. deltas={deltas}"
        )


class TestTrainingBehaviour:
    def test_gradients_reach_every_trainable_weight_of_both_necks(
            self, tiny_neck, tiny_trunk_map
    ):
        import tensorflow as tf
        with tf.GradientTape() as tape:
            out = tiny_neck(tf.convert_to_tensor(tiny_trunk_map))
            loss = sum(
                ops.sum(ops.square(t))
                for t in out["sam3_features"] + out["sam2_features"]
            )
        grads = tape.gradient(loss, tiny_neck.trainable_weights)
        assert len(grads) == len(tiny_neck.trainable_weights) > 0
        assert all(g is not None for g in grads)

    def test_the_positional_encoding_contributes_no_trainable_weights(
            self, tiny_neck
    ):
        assert tiny_neck.position_encoding.trainable_weights == []

    def test_inference_is_deterministic(self, tiny_neck, tiny_trunk_map):
        first = tiny_neck(tiny_trunk_map)
        second = tiny_neck(tiny_trunk_map)
        for key in Sam3DualViTDetNeck.FEATURE_KEYS:
            for a, b in zip(first[key], second[key]):
                assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0


class TestPurity:
    def test_library_file_uses_no_tensorflow(self):
        import dl_techniques.models.sam3.necks as module
        source = open(module.__file__, "r", encoding="utf-8").read()
        assert "import tensorflow" not in source
        assert "tf." not in source
