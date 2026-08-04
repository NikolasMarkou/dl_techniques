"""
Guards for SAM 2's FPN neck and image encoder (plan step 4, guards G4.1-G4.5).

Both mechanisms guarded here are SILENT when ported wrong: the model builds,
forward-passes, trains and serializes either way.

    * ``scalp`` drops the COARSEST level, and it does so BEFORE
      ``vision_features`` is read. Dropping the finest end instead keeps the
      returned level COUNT at three, so a count assertion cannot see it.
    * The top-down fusion is GATED to the two coarsest levels. Fusing every
      level changes only values, never shapes.

Guard map:

    G4.1  ``TestScalpCount``            -- exactly 3 levels and 3 encodings
    G4.2  ``TestScalpIdentity``         -- WHICH 3, by shape triple
    G4.3  ``TestTopDownGating``         -- zero the coarsest trunk level
    G4.4  ``TestPositionEncodingWidth`` -- 256 channels HERE (64 in step 5)
    G4.5  ``TestDeadComponentPartition``-- the MEASURED red/green partition

G4.1 and G4.2 deliberately guard the same defect twice, from opposite sides:
G4.1 sees a missing drop, G4.2 sees a drop from the wrong end. Neither alone is
sufficient.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.sam2.hiera import Hiera
from dl_techniques.models.sam2.neck import SAM2FpnNeck, SAM2ImageEncoder

from ..test_sam.dead_component_oracle import zeroed_variables

# ---------------------------------------------------------------------
# Test geometry.
#
# Everything is read from the two variant tables. Nothing below re-states a
# variant number; the constants are the DERIVED consequences a reader can check
# by hand.
# ---------------------------------------------------------------------

TINY_TRUNK: Dict[str, Any] = Hiera.MODEL_VARIANTS["tiny"]
TINY_ENCODER: Dict[str, Any] = SAM2ImageEncoder.MODEL_VARIANTS["tiny"]
IMAGE_SIZE: int = TINY_TRUNK["image_size"]
BATCH = 2
SEED = 8642

#: The four trunk levels at `tiny`, ascending stage order: the stem divides by
#: 4 and each of the three query-pooling transitions halves the grid and
#: doubles the width.
TRUNK_LEVELS: List[Tuple[int, int, int]] = [
    (16, 16, 16),
    (8, 8, 32),
    (4, 4, 64),
    (2, 2, 128),
]

#: What the neck produces: the same four grids, all widened to `d_model`.
NECK_LEVELS: List[Tuple[int, int, int]] = [
    (h, w, TINY_ENCODER["d_model"]) for h, w, _ in TRUNK_LEVELS
]

#: What the encoder returns after `scalp=1`: the FINEST three of the above.
RETAINED_LEVELS: List[Tuple[int, int, int]] = NECK_LEVELS[:-1]

#: The wrong end. Same length, different levels -- this is why G4.1 is not
#: enough on its own.
WRONG_END_LEVELS: List[Tuple[int, int, int]] = NECK_LEVELS[1:]

# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _encoder(**overrides: Any) -> SAM2ImageEncoder:
    """Build the `tiny` image encoder through the real variant path."""
    return SAM2ImageEncoder.from_variant("tiny", **overrides)


def _neck(**overrides: Any) -> SAM2FpnNeck:
    """Build a neck matching the `tiny` trunk's channel list."""
    config = dict(TINY_ENCODER)
    config.pop("scalp")
    config.update(overrides)
    trunk_channels = tuple(
        channels for _, _, channels in reversed(TRUNK_LEVELS))
    return SAM2FpnNeck(backbone_channel_list=trunk_channels, **config)


def _image(batch: int = BATCH) -> np.ndarray:
    """A seeded image batch at the `tiny` resolution."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(
        (batch, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")


def _trunk_levels(batch: int = BATCH) -> List[np.ndarray]:
    """A seeded stand-in for the trunk's four ascending-stage outputs."""
    rng = np.random.default_rng(SEED + 1)
    return [
        rng.standard_normal((batch, h, w, c)).astype("float32")
        for h, w, c in TRUNK_LEVELS
    ]


def _built_neck(batch: int = BATCH) -> Tuple[SAM2FpnNeck, List[np.ndarray]]:
    """A neck built against the `tiny` trunk shapes, plus matching inputs."""
    neck = _neck()
    levels = _trunk_levels(batch)
    neck.build([level.shape for level in levels])
    return neck, levels


def _triples(tensors: Any) -> List[Tuple[int, int, int]]:
    """The ``(H, W, C)`` triple of each tensor, batch axis dropped."""
    return [tuple(int(d) for d in tensor.shape[1:]) for tensor in tensors]


def _max_abs_diff(a: Any, b: Any) -> float:
    """Max absolute elementwise difference, as a Python float."""
    return float(np.max(np.abs(
        ops.convert_to_numpy(a) - ops.convert_to_numpy(b))))


# ---------------------------------------------------------------------
# Index orientation.
# ---------------------------------------------------------------------


class TestLateralIndexOrientation:
    """`convs[n - i]` is applied to `xs[i]` -- the two lists run opposite ways."""

    def test_backbone_channel_list_is_descending_and_matches_the_trunk(
            self) -> None:
        """The neck's channel list is the trunk's own `channel_list`."""
        encoder = _encoder()
        assert tuple(encoder.trunk.channel_list) == \
            encoder.neck.backbone_channel_list
        assert list(encoder.neck.backbone_channel_list) == sorted(
            encoder.neck.backbone_channel_list, reverse=True), (
            "backbone_channel_list must be DESCENDING (widest/coarsest first)"
        )

    def test_each_lateral_conv_is_built_for_its_mirrored_level(self) -> None:
        """`convs[j]`'s input width is `backbone_channel_list[j]`."""
        neck, _ = _built_neck()
        for index, conv in enumerate(neck.convs):
            assert int(conv.kernel.shape[2]) == \
                neck.backbone_channel_list[index], (
                f"lateral_conv_{index} is built for "
                f"{int(conv.kernel.shape[2])} channels but "
                f"backbone_channel_list[{index}] is "
                f"{neck.backbone_channel_list[index]}"
            )

    def test_reversed_input_order_raises_rather_than_training_silently(
            self) -> None:
        """A reversed level list is a shape error, not a silent transposition."""
        neck = _neck()
        levels = _trunk_levels()
        with pytest.raises(ValueError, match="ASCENDING stage order"):
            neck.build([level.shape for level in reversed(levels)])

    def test_wrong_level_count_raises(self) -> None:
        """Three levels into a four-level neck is a construction error."""
        neck = _neck()
        levels = _trunk_levels()
        with pytest.raises(ValueError, match="4 trunk levels"):
            neck.build([level.shape for level in levels[:3]])


# ---------------------------------------------------------------------
# G4.1 -- count.
# ---------------------------------------------------------------------


class TestScalpCount:
    """G4.1: the neck builds 4 levels, the encoder returns 3.

    Mutation this is proven against: `scalp=0`.
    """

    def test_neck_returns_four_levels(self) -> None:
        """The neck itself drops nothing."""
        neck, levels = _built_neck()
        out, pos = neck(levels)
        assert len(out) == 4
        assert len(pos) == 4

    def test_encoder_returns_exactly_three_levels_and_three_encodings(
            self) -> None:
        """The encoder emits one fewer level than the neck built."""
        encoder = _encoder()
        out = encoder(_image())
        assert len(out["backbone_fpn"]) == 3, (
            f"scalp=1 must leave 3 of the neck's 4 levels; got "
            f"{len(out['backbone_fpn'])}"
        )
        assert len(out["vision_pos_enc"]) == 3, (
            f"one positional encoding per RETAINED level; got "
            f"{len(out['vision_pos_enc'])}"
        )

    def test_compute_output_shape_agrees_on_the_count(self) -> None:
        """The static shape contract drops the same level the forward pass does."""
        encoder = _encoder()
        shapes = encoder.compute_output_shape((None, IMAGE_SIZE, IMAGE_SIZE, 3))
        assert len(shapes["backbone_fpn"]) == 3
        assert len(shapes["vision_pos_enc"]) == 3

    def test_scalp_zero_returns_all_four(self) -> None:
        """The control: with the drop disabled the count is 4, not 3.

        This is the plan's own G4.1 mutation, executed as a positive test so
        the count assertion above is demonstrably not vacuous.
        """
        encoder = _encoder(scalp=0)
        out = encoder(_image())
        assert len(out["backbone_fpn"]) == 4
        assert _triples(out["backbone_fpn"]) == NECK_LEVELS


# ---------------------------------------------------------------------
# G4.2 -- identity. The guard that matters.
# ---------------------------------------------------------------------


class TestScalpIdentity:
    """G4.2: WHICH three levels are retained, not how many.

    Mutation this is proven against: `levels = levels[scalp:]` -- dropping the
    FINEST level instead of the coarsest. The count stays 3, so every assertion
    in `TestScalpCount` stays green; only the shape triples below move.
    """

    def test_retained_levels_are_the_finest_three(self) -> None:
        """The returned `(H, W, C)` triples equal the finest three."""
        encoder = _encoder()
        out = encoder(_image())
        assert _triples(out["backbone_fpn"]) == RETAINED_LEVELS, (
            f"the retained levels must be the FINEST three "
            f"{RETAINED_LEVELS}; got {_triples(out['backbone_fpn'])}"
        )

    def test_retained_levels_are_not_the_coarsest_three(self) -> None:
        """The explicit negative: dropping the other end is a different set.

        Stated separately because the two lists have the SAME length and the
        same channel width -- resolution is the only discriminator.
        """
        assert RETAINED_LEVELS != WRONG_END_LEVELS
        encoder = _encoder()
        out = encoder(_image())
        assert _triples(out["backbone_fpn"]) != WRONG_END_LEVELS

    def test_vision_features_is_the_third_finest_level(self) -> None:
        """`vision_features` is read AFTER the drop, so it is the third finest."""
        encoder = _encoder()
        out = encoder(_image())
        assert tuple(int(d) for d in out["vision_features"].shape[1:]) == \
            NECK_LEVELS[2], (
            f"vision_features must be the coarsest RETAINED level "
            f"{NECK_LEVELS[2]}, not the coarsest level the neck built "
            f"{NECK_LEVELS[3]}"
        )

    def test_vision_features_is_bit_identical_to_the_last_retained_level(
            self) -> None:
        """It is the same tensor, not a recomputation."""
        encoder = _encoder()
        out = encoder(_image())
        assert _max_abs_diff(
            out["vision_features"], out["backbone_fpn"][-1]) == 0.0

    def test_taking_vision_features_before_the_drop_is_a_different_tensor(
            self) -> None:
        """The order of the two operations is observable.

        Runs the neck directly and takes both candidates. Their shapes differ
        by a factor of 2 in each spatial axis, so a port that reads
        `vision_features` before scalping is measurably wrong -- but only if
        something looks, which nothing downstream of this boundary does.
        """
        neck, levels = _built_neck()
        out, _ = neck(levels)
        after_drop = out[: -TINY_ENCODER["scalp"]][-1]
        before_drop = out[-1]
        assert tuple(after_drop.shape[1:3]) != tuple(before_drop.shape[1:3])
        assert tuple(int(d) for d in after_drop.shape[1:]) == NECK_LEVELS[2]
        assert tuple(int(d) for d in before_drop.shape[1:]) == NECK_LEVELS[3]

    def test_high_resolution_skips_are_the_two_finest(self) -> None:
        """`backbone_fpn[0]` / `[1]` are what the mask decoder consumes."""
        encoder = _encoder()
        out = encoder(_image())
        assert _triples(out["backbone_fpn"][:2]) == NECK_LEVELS[:2]

    def test_positional_encodings_track_their_levels(self) -> None:
        """Each retained encoding has its level's grid."""
        encoder = _encoder()
        out = encoder(_image())
        for level, position in zip(out["backbone_fpn"], out["vision_pos_enc"]):
            assert tuple(level.shape[1:3]) == tuple(position.shape[1:3])


class TestStrideReconciliation:
    """A-1 as a consistency check: stride 16 at `image_size=1024`.

    The plan settled the scalp ordering by READING it. This asserts the
    downstream consequence independently: the memory attention's rotary tables
    are built for a 64x64 grid, and only the read ordering produces one.
    """

    def test_hiera_l_vision_features_is_64x64_stride_16(self) -> None:
        """Shapes only -- `hiera_l` is never forward-passed in tests."""
        encoder = SAM2ImageEncoder.from_variant("hiera_l")
        shapes = encoder.compute_output_shape((None, 1024, 1024, 3))
        assert shapes["vision_features"][1:3] == (64, 64), (
            f"vision_features must be a 64x64 grid to match the memory "
            f"attention's feat_sizes; got {shapes['vision_features'][1:3]}"
        )
        assert 1024 // shapes["vision_features"][1] == 16

    def test_hiera_l_retained_strides_are_4_8_16(self) -> None:
        """The three retained levels, by stride."""
        encoder = SAM2ImageEncoder.from_variant("hiera_l")
        shapes = encoder.compute_output_shape((None, 1024, 1024, 3))
        strides = [1024 // shape[1] for shape in shapes["backbone_fpn"]]
        assert strides == [4, 8, 16]

    def test_hiera_l_without_scalp_would_be_stride_32(self) -> None:
        """The refuted alternative, made explicit."""
        encoder = SAM2ImageEncoder.from_variant("hiera_l", scalp=0)
        shapes = encoder.compute_output_shape((None, 1024, 1024, 3))
        assert shapes["vision_features"][1:3] == (32, 32)

    def test_tiny_reproduces_the_same_stride_ladder(self) -> None:
        """The small geometry is structurally faithful on this axis too."""
        encoder = _encoder()
        out = encoder(_image())
        strides = [
            IMAGE_SIZE // int(level.shape[1]) for level in out["backbone_fpn"]
        ]
        assert strides == [4, 8, 16]


# ---------------------------------------------------------------------
# G4.3 -- top-down gating.
# ---------------------------------------------------------------------


class TestTopDownGating:
    """G4.3: only levels 2 and 3 see the coarsest trunk level.

    Mutation this is proven against: gate on ALL levels
    (`fpn_top_down_levels=None`), which is upstream's own signature default and
    therefore the most likely wrong port.
    """

    @staticmethod
    def _levels_with_and_without_the_coarsest(
            neck: SAM2FpnNeck) -> Tuple[List[Any], List[Any]]:
        """Run the neck twice, the second time with `xs[-1]` zeroed."""
        levels = _trunk_levels()
        neck.build([level.shape for level in levels])
        reference, _ = neck(levels)
        perturbed_inputs = list(levels)
        perturbed_inputs[-1] = np.zeros_like(perturbed_inputs[-1])
        perturbed, _ = neck(perturbed_inputs)
        return reference, perturbed

    def test_shipped_gating_is_the_two_coarsest_levels(self) -> None:
        """Config, as the precondition of the behavioural test below."""
        neck = _neck()
        assert neck.fpn_top_down_levels == (2, 3)

    def test_fine_levels_are_bit_identical_when_the_coarsest_changes(
            self) -> None:
        """Levels 0 and 1 are lateral-only: max-abs-diff EXACTLY 0.0."""
        neck = _neck()
        reference, perturbed = self._levels_with_and_without_the_coarsest(neck)
        for index in (0, 1):
            diff = _max_abs_diff(reference[index], perturbed[index])
            assert diff == 0.0, (
                f"level {index} moved by {diff} when the COARSEST trunk level "
                f"was zeroed -- it must be lateral-only, with no top-down "
                f"fusion at all"
            )

    def test_coarse_levels_move_when_the_coarsest_changes(self) -> None:
        """Levels 2 and 3 must move -- the other half of the guard.

        A one-sided assertion is vacuous: a neck that ignored its input
        entirely would pass the bit-identity half.
        """
        neck = _neck()
        reference, perturbed = self._levels_with_and_without_the_coarsest(neck)
        for index in (2, 3):
            diff = _max_abs_diff(reference[index], perturbed[index])
            assert diff > 1e-5, (
                f"level {index} did not move ({diff}) when the coarsest trunk "
                f"level was zeroed"
            )

    def test_ungated_neck_propagates_all_the_way_to_the_finest_level(
            self) -> None:
        """The mutation, executed as a positive control.

        With gating disabled the zeroed coarsest level reaches levels 0 and 1,
        so the bit-identity assertion above would fail. This proves that
        assertion discriminates rather than merely passing.
        """
        neck = _neck(fpn_top_down_levels=None)
        reference, perturbed = self._levels_with_and_without_the_coarsest(neck)
        for index in (0, 1):
            diff = _max_abs_diff(reference[index], perturbed[index])
            assert diff > 1e-5, (
                f"with gating disabled level {index} should have moved; the "
                f"top-down chain may not be connected at all"
            )

    def test_top_down_addition_is_additive_not_a_replacement(self) -> None:
        """Level 2 depends on its OWN trunk level as well as on level 3."""
        neck = _neck()
        levels = _trunk_levels()
        neck.build([level.shape for level in levels])
        reference, _ = neck(levels)
        perturbed_inputs = list(levels)
        perturbed_inputs[2] = np.zeros_like(perturbed_inputs[2])
        perturbed, _ = neck(perturbed_inputs)
        assert _max_abs_diff(reference[2], perturbed[2]) > 1e-5

    def test_nearest_upsample_replicates_rather_than_interpolates(self) -> None:
        """The 2x step is nearest: each coarse cell becomes a constant 2x2 tile.

        Isolated by zeroing every lateral contribution except the coarsest
        level's, so the only signal in level 2 is the upsampled level 3.
        """
        neck = _neck()
        levels = _trunk_levels(batch=1)
        neck.build([level.shape for level in levels])
        quiet = [np.zeros_like(level) for level in levels]
        quiet[-1] = levels[-1]
        with zeroed_variables(
                [conv.bias for conv in neck.convs]):
            out, _ = neck(quiet)
        level2 = ops.convert_to_numpy(out[2])
        # Every 2x2 tile came from one cell of level 3.
        for row in range(0, level2.shape[1], 2):
            for col in range(0, level2.shape[2], 2):
                tile = level2[0, row:row + 2, col:col + 2, :]
                assert float(np.max(np.abs(tile - tile[0, 0]))) == 0.0, (
                    f"tile at ({row}, {col}) is not constant -- the top-down "
                    f"step interpolated instead of replicating"
                )


# ---------------------------------------------------------------------
# G4.4 -- positional-encoding width.
# ---------------------------------------------------------------------


class TestPositionEncodingWidth:
    """G4.4: the neck's sine encoding is 256 channels wide.

    The literal 256 is written HERE and only here. The memory encoder's
    encoding is 64 channels wide and its test writes that literal in ITS own
    file. These two widths are deliberately NOT unified behind a shared
    constant: a single constant would make one of the two sites unable to fail.
    """

    def test_shipped_neck_encoding_is_256_channels(self) -> None:
        """At `d_model=256` -- the shipped configuration."""
        neck = SAM2FpnNeck()
        assert neck.pos_enc_channels == 256, (
            f"the FPN neck's positional encoding is 256 channels wide; got "
            f"{neck.pos_enc_channels}"
        )

    def test_shipped_encoding_width_is_measured_not_declared(self) -> None:
        """Forward-pass the real layer and read the channel axis."""
        neck = SAM2FpnNeck(
            d_model=256, backbone_channel_list=(16, 8),
            fpn_top_down_levels=(1,))
        inputs = [
            np.zeros((1, 4, 4, 8), dtype="float32"),
            np.zeros((1, 2, 2, 16), dtype="float32"),
        ]
        neck.build([x.shape for x in inputs])
        _, positions = neck(inputs)
        assert int(positions[0].shape[-1]) == 256

    def test_encoding_matches_the_feature_width_it_is_added_to(self) -> None:
        """The invariant behind the number: PE width == d_model.

        The memory attention adds the encoding to its `d_model`-wide input, so
        any other width is an addition error waiting to happen.
        """
        for d_model in (32, 64, 256):
            neck = SAM2FpnNeck(
                d_model=d_model, backbone_channel_list=(8, 4),
                fpn_top_down_levels=(1,))
            assert neck.pos_enc_channels == d_model

    def test_encoding_is_cast_to_the_feature_dtype(self) -> None:
        """`PositionEmbeddingSine2D` returns float32 at every policy.

        Measured for this repo's layer, so the neck casts explicitly. Without
        the cast a `mixed_float16` consumer either upcasts silently or raises
        on the addition.
        """
        previous = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            neck, levels = _built_neck(batch=1)
            out, positions = neck(levels)
            assert positions[0].dtype == out[0].dtype, (
                f"positional encoding dtype {positions[0].dtype} does not "
                f"match the feature dtype {out[0].dtype}"
            )
            assert keras.backend.standardize_dtype(
                positions[0].dtype) == "float16"
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_encoding_is_channels_last(self) -> None:
        """The reused layer emits channels-FIRST; the neck transposes back.

        Without the transpose the encoding would be
        `(batch, channels, H, W)` and would still broadcast against a square
        feature map, which is the silent case this asserts against.
        """
        neck = _neck()
        levels = _trunk_levels(batch=1)
        neck.build([level.shape for level in levels])
        _, positions = neck(levels)
        # Level 0 is 16x16 and d_model is 32: a channels-first encoding would
        # be (1, 32, 16, 16), which is a DIFFERENT shape only because the two
        # sizes differ. Chosen deliberately.
        assert tuple(positions[0].shape) == (
            1, TRUNK_LEVELS[0][0], TRUNK_LEVELS[0][1],
            TINY_ENCODER["d_model"])

    def test_encoding_varies_across_positions(self) -> None:
        """A constant encoding would satisfy every width assertion above."""
        neck, levels = _built_neck(batch=1)
        _, positions = neck(levels)
        position = ops.convert_to_numpy(positions[0])
        assert float(np.std(position, axis=(1, 2)).max()) > 1e-4

    def test_encoding_does_not_depend_on_the_feature_values(self) -> None:
        """It is a fixed function of the grid, not a learned projection."""
        neck, levels = _built_neck(batch=1)
        _, reference = neck(levels)
        _, perturbed = neck([level * 3.0 + 1.0 for level in levels])
        assert _max_abs_diff(reference[0], perturbed[0]) == 0.0

    def test_encoding_is_weightless(self) -> None:
        """It contributes no parameters to the `hiera_l` audit in step 8."""
        neck, _ = _built_neck()
        assert neck.position_encoding.weights == []


# ---------------------------------------------------------------------
# Forward pass, shapes, errors, serialization, trace, gradients.
# ---------------------------------------------------------------------


class TestForwardAndShapes:
    """The shape contract, and that the static one matches the dynamic one."""

    def test_neck_widens_every_level_to_d_model(self) -> None:
        neck, levels = _built_neck()
        out, _ = neck(levels)
        assert _triples(out) == NECK_LEVELS

    def test_neck_preserves_every_grid(self) -> None:
        neck, levels = _built_neck()
        out, _ = neck(levels)
        for source, fused in zip(levels, out):
            assert tuple(source.shape[1:3]) == tuple(fused.shape[1:3])

    def test_compute_output_shape_agrees_with_the_forward_pass(self) -> None:
        encoder = _encoder()
        images = _image()
        out = encoder(images)
        shapes = encoder.compute_output_shape((BATCH,) + images.shape[1:])
        assert [tuple(level.shape) for level in out["backbone_fpn"]] == \
            [tuple(shape) for shape in shapes["backbone_fpn"]]
        assert [tuple(p.shape) for p in out["vision_pos_enc"]] == \
            [tuple(shape) for shape in shapes["vision_pos_enc"]]
        assert tuple(out["vision_features"].shape) == \
            tuple(shapes["vision_features"])

    def test_batch_size_does_not_change_per_sample_output(self) -> None:
        """No sample sees another sample's data.

        Measured on the NECK alone, deliberately. The same probe run through
        the whole encoder drifts by ~3e-3 at `tiny`, which is the trunk's
        cuDNN batch-dependent kernel selection, not a batch-axis leak -- an
        encoder-level tolerance loose enough to absorb that would no longer
        detect the leak this test exists for.
        """
        neck = _neck()
        levels = _trunk_levels(batch=3)
        neck.build([level.shape for level in levels])
        batched, _ = neck(levels)
        single, _ = neck([level[:1] for level in levels])
        for index in range(len(batched)):
            assert _max_abs_diff(batched[index][:1], single[index]) < 1e-6

    def test_neck_output_is_a_pair_of_lists(self) -> None:
        neck, levels = _built_neck()
        out = neck(levels)
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], list) and isinstance(out[1], list)

    def test_encoder_output_keys(self) -> None:
        encoder = _encoder()
        out = encoder(_image())
        assert set(out) == {
            "vision_features", "vision_pos_enc", "backbone_fpn"}


class TestConstructionErrors:
    """Every invalid configuration raises at construction, not at call."""

    def test_non_positive_d_model(self) -> None:
        with pytest.raises(ValueError, match="d_model must be positive"):
            SAM2FpnNeck(d_model=0)

    def test_odd_d_model(self) -> None:
        with pytest.raises(ValueError, match="d_model must be even"):
            SAM2FpnNeck(d_model=255)

    def test_empty_backbone_channel_list(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SAM2FpnNeck(backbone_channel_list=())

    def test_non_positive_channel_width(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SAM2FpnNeck(backbone_channel_list=(16, 0))

    def test_top_down_level_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            SAM2FpnNeck(
                backbone_channel_list=(16, 8), fpn_top_down_levels=(2,))

    def test_negative_scalp(self) -> None:
        with pytest.raises(ValueError, match="scalp must not be negative"):
            _encoder(scalp=-1)

    def test_scalp_discards_every_level(self) -> None:
        with pytest.raises(ValueError, match="would discard every"):
            _encoder(scalp=4)

    def test_trunk_and_neck_channel_mismatch(self) -> None:
        trunk = Hiera.from_variant("tiny")
        neck = SAM2FpnNeck(
            d_model=32, backbone_channel_list=(128, 64, 32, 8))
        with pytest.raises(ValueError, match="does not match the neck"):
            SAM2ImageEncoder(trunk=trunk, neck=neck, scalp=1)

    def test_trunk_and_neck_level_count_mismatch(self) -> None:
        trunk = Hiera.from_variant("tiny")
        neck = SAM2FpnNeck(
            d_model=32, backbone_channel_list=(128, 64, 32),
            fpn_top_down_levels=(2,))
        with pytest.raises(ValueError, match="produces 4 levels"):
            SAM2ImageEncoder(trunk=trunk, neck=neck, scalp=1)

    def test_unknown_variant(self) -> None:
        with pytest.raises(ValueError, match="Unknown SAM2ImageEncoder"):
            SAM2ImageEncoder.from_variant("huge")

    def test_rank_3_level_shape(self) -> None:
        neck = _neck()
        with pytest.raises(ValueError, match="rank-4"):
            neck.build([(None, 16, 16), (None, 8, 8, 32),
                        (None, 4, 4, 64), (None, 2, 2, 128)])


class TestSerialization:
    """`get_config` round-trips, including the nested trunk and neck."""

    def test_neck_config_round_trip(self) -> None:
        neck = _neck()
        restored = SAM2FpnNeck.from_config(neck.get_config())
        assert restored.get_config() == neck.get_config()

    def test_encoder_config_round_trip_by_value(self) -> None:
        encoder = _encoder()
        images = _image(batch=1)
        reference = encoder(images)["vision_features"]

        restored = SAM2ImageEncoder.from_config(encoder.get_config())
        restored.build((None,) + images.shape[1:])
        for source, target in zip(encoder.weights, restored.weights):
            target.assign(source)
        assert _max_abs_diff(reference, restored(images)["vision_features"]) \
            == 0.0

    def test_encoder_config_preserves_scalp(self) -> None:
        config = _encoder(scalp=0).get_config()
        assert config["scalp"] == 0
        assert SAM2ImageEncoder.from_config(config).scalp == 0

    def test_registered_keys_are_present_exactly_once(self) -> None:
        registry = keras.saving.get_custom_objects()
        for name in ("SAM2FpnNeck", "SAM2ImageEncoder"):
            matches = [key for key in registry if key.endswith(f">{name}")]
            assert len(matches) == 1, (
                f"'{name}' is registered {len(matches)} times: {matches}"
            )


class TestGraphTrace:
    """The encoder traces under `tf.function` with a static input signature."""

    def test_call_traces_with_static_input_signature(self) -> None:
        encoder = _encoder()
        encoder.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

        @tf.function(input_signature=[
            tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)])
        def traced(images: Any) -> Any:
            return encoder(images)["vision_features"]

        concrete = traced.get_concrete_function()
        assert concrete.output_shapes[1:] == tuple(NECK_LEVELS[2])

    def test_trace_guard_is_not_vacuous(self) -> None:
        """The measured liveness proof from step 3: `ops.convert_to_numpy`.

        Step 3 measured that `float(<traced tensor>)` is INERT inside a
        `tf.function` body (AutoGraph rewrites it into a cast). This uses the
        substitute that actually fires, and records the exception TYPE that was
        measured rather than one that was predicted.
        """
        encoder = _encoder()
        encoder.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

        @tf.function(input_signature=[
            tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)])
        def traced(images: Any) -> Any:
            out = encoder(images)["vision_features"]
            return out + float(np.sum(ops.convert_to_numpy(out)))

        with pytest.raises(NotImplementedError):
            traced.get_concrete_function()


class TestGradientFlow:
    """Every lateral convolution carries gradient."""

    def test_all_neck_variables_receive_a_gradient(self) -> None:
        neck, levels = _built_neck(batch=1)
        tensors = [tf.convert_to_tensor(level) for level in levels]
        with tf.GradientTape() as tape:
            out, _ = neck(tensors)
            loss = sum(ops.sum(ops.square(level)) for level in out)
        grads = tape.gradient(loss, neck.trainable_variables)
        dead = [
            variable.name
            for variable, grad in zip(neck.trainable_variables, grads)
            if grad is None or float(np.max(np.abs(
                ops.convert_to_numpy(grad)))) == 0.0
        ]
        assert dead == [], f"variables with no gradient: {dead}"

    def test_the_scalped_level_still_carries_gradient_in_the_encoder(
            self) -> None:
        """The dropped level is not dead code: level 2 depends on it.

        This is why `scalp` is a return-value filter and not a reason to skip
        computing the coarsest level.
        """
        encoder = _encoder()
        images = tf.convert_to_tensor(_image(batch=1))
        coarsest_conv = encoder.neck.convs[0]
        with tf.GradientTape() as tape:
            out = encoder(images)
            loss = ops.sum(ops.square(out["vision_features"]))
        grads = tape.gradient(loss, coarsest_conv.trainable_variables)
        assert all(grad is not None for grad in grads)
        assert float(np.max(np.abs(ops.convert_to_numpy(grads[0])))) > 0.0


# ---------------------------------------------------------------------
# G4.5 -- the measured dead-component partition.
# ---------------------------------------------------------------------


class TestDeadComponentPartition:
    """G4.5: which guards actually go RED under a dead component.

    "All guards go red" is a hypothesis, and steps 1, 2 and 3 all measured it
    FALSE. What follows is the partition MEASURED here, including the guards
    that stay GREEN -- a guard that cannot go red is the thing worth knowing
    about. Measured with the neck's lateral convolution KERNELS zeroed, which
    is the strongest dead component this layer has: every level collapses to
    its bias.
    """

    @staticmethod
    def _dead_kernels(neck: SAM2FpnNeck) -> List[Any]:
        return [conv.kernel for conv in neck.convs]

    def test_count_guard_stays_green(self) -> None:
        """G4.1 is structural: no weight value can change a list length."""
        encoder = _encoder()
        encoder.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))
        with zeroed_variables(self._dead_kernels(encoder.neck)):
            out = encoder(_image())
        assert len(out["backbone_fpn"]) == 3

    def test_identity_guard_stays_green(self) -> None:
        """G4.2 is also structural -- it reads shapes, not values.

        This is the measured limit of G4.2: it discriminates WHICH end is
        dropped and is blind to a dead pyramid. That is acceptable only
        because G4.3 and the gradient tests cover liveness.
        """
        encoder = _encoder()
        encoder.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))
        with zeroed_variables(self._dead_kernels(encoder.neck)):
            out = encoder(_image())
        assert _triples(out["backbone_fpn"]) == RETAINED_LEVELS

    def test_gating_guard_goes_half_red(self) -> None:
        """G4.3 splits: the movement arm goes RED, the identity arm stays GREEN.

        With the kernels dead every level is its bias, so zeroing the coarsest
        trunk level moves nothing. The bit-identity half therefore passes
        VACUOUSLY -- which is exactly why the movement half exists alongside
        it, and why a one-sided gating assertion would be worthless.
        """
        neck = _neck()
        levels = _trunk_levels()
        neck.build([level.shape for level in levels])
        perturbed_inputs = list(levels)
        perturbed_inputs[-1] = np.zeros_like(perturbed_inputs[-1])

        with zeroed_variables(self._dead_kernels(neck)):
            reference, _ = neck(levels)
            perturbed, _ = neck(perturbed_inputs)

        for index in (0, 1):
            assert _max_abs_diff(reference[index], perturbed[index]) == 0.0, (
                "the bit-identity arm went red under a dead component, so it "
                "is not the vacuous half"
            )
        for index in (2, 3):
            assert _max_abs_diff(reference[index], perturbed[index]) == 0.0, (
                f"level {index} moved with every lateral kernel dead -- the "
                f"movement arm was expected to go RED here"
            )

    def test_position_encoding_guard_stays_green(self) -> None:
        """G4.4 is weight-independent by construction: the PE has no weights."""
        neck = _neck()
        levels = _trunk_levels(batch=1)
        neck.build([level.shape for level in levels])
        with zeroed_variables(self._dead_kernels(neck)):
            _, positions = neck(levels)
        assert int(positions[0].shape[-1]) == TINY_ENCODER["d_model"]
        assert float(np.std(
            ops.convert_to_numpy(positions[0]), axis=(1, 2)).max()) > 1e-4

    def test_a_dead_trunk_stem_flattens_every_level(self) -> None:
        """The floor case: with the stem dead no information enters the neck."""
        encoder = _encoder()
        encoder.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))
        with zeroed_variables(encoder.trunk.patch_embed.proj.weights):
            out = encoder(_image(batch=1))
        finest = ops.convert_to_numpy(out["backbone_fpn"][0])
        assert float(np.std(finest, axis=(1, 2)).max()) < 1e-5

# ---------------------------------------------------------------------
