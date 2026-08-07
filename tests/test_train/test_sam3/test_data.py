"""Tests for ``train/sam3/data.py`` -- the synthetic text-prompted source.

**A bug in this file's subject is indistinguishable from a model bug**, which is
why this suite runs BEFORE any learnability claim. The guards are shaped so that
none of them can pass by construction:

- **The box oracle is derived from the RASTERIZED MASK, never from the drawn
  parameters.** The pipeline computes each box ANALYTICALLY from the sampled
  extent; the oracle here recomputes it from the emitted 64x64 binary mask's own
  extents. The two are genuinely different computations, so a rasterization or
  placement bug is visible instead of cancelling out. The oracle's own RED arm
  (a deliberately displaced box) proves it discriminates.
- **Prompt-vs-target correspondence is checked against the RENDERED IMAGE.**
  Connected components of the composited image are segmented and classified by
  shape statistics, with no reference to the record's target arrays. A pipeline
  that ignored the prompt and returned every instance would satisfy every
  box/mask assertion and fails this one on the COUNT.
- **The zero-instance rate is measured, not assumed**, with both degenerate
  rates (0.0 and 1.0) as liveness arms.
- **The packed width is compared against the model's own
  ``packed_target_spec()``** at both ``include_masks`` values.
"""

import hashlib
import inspect
import os
import tempfile

import cv2
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.sam3_detection_loss import (
    Sam3DetectionLoss,
    packed_channel_count,
    unpack_targets,
)
from dl_techniques.models.sam3 import Sam3Image
from dl_techniques.models.sam3.training_model import (
    Sam3TrainingModel,
    compile_sam3_trainer,
)
from train.sam3.data import (
    CATEGORIES,
    CATEGORY_ID_PAD,
    CATEGORY_PHRASES,
    PAD_ID,
    WORD_TO_ID,
    _downsample,
    _rasterize,
    _sample_extent,
    all_instance_capacity,
    build_sam3_dataset,
    encode_phrase,
    synthetic_prompt_samples,
)

#: The shipped geometry the learnability run will use (`small`): 224 px images,
#: a 64x64 mask grid, `context_length=32`.
IMAGE_SIZE = 224
MASK_GRID = (64, 64)
CONTEXT = 32
N_MAX = 8

#: Box-vs-mask tolerance, in MASK CELLS, converted to the normalized frame the
#: boxes live in. Rasterizing at 224 px and area-downsampling to 64 moves each
#: edge by up to about one cell, so a width or height can move by two.
#: MEASURED over 570 instances: worst component deviation 1.40 cells. Pinned at
#: 2.5 cells, which still leaves a placement bug (tens of cells) and a
#: width/height swap (>= 9 cells for a `bar`) far outside.
BOX_TOLERANCE_CELLS = 2.5
BOX_TOLERANCE = BOX_TOLERANCE_CELLS / MASK_GRID[0]

#: Shape-statistic decision boundaries used ONLY by the image-side oracle below.
#: MEASURED per-instance ranges at 224 px are disjoint with wide gaps:
#: triangle fill [.504, .524], circle [.753, .797], square/bar 1.000; aspect
#: 1.0 for the first three and [2.29, 2.53] for `bar`. The boundaries sit in the
#: middle of those gaps.
FILL_TRIANGLE_MAX = 0.62
FILL_CIRCLE_MAX = 0.90
ASPECT_BAR_MIN = 1.60

#: The sha256 of the first batch at ``seed=99`` on the ``tiny`` geometry,
#: computed in a SEPARATE process (twice, under two different
#: ``PYTHONHASHSEED`` values, which agreed). See
#: ``test_the_first_batch_is_bit_reproducible_across_processes``.
PINNED_FIRST_BATCH_SHA256 = (
    "36c573f038bbf43d48be67869d1178bca8c1c3c766ba10d055467f1ba87e0f76")

#: The SEVEN keys of ``build_sam3_dataset``'s default ``signature``. The
#: eval-only all-instance keys are deliberately absent.
SIGNATURE_KEYS = (
    "image", "token_ids", "token_padding_mask",
    "target_boxes", "target_valid", "target_masks", "is_exhaustive")

#: **The before/after evidence for the all-instance export (plan SC-A / I-1).**
#: Per-key sha256 over 64 records at the shipped geometry, at two seeds,
#: computed by running the generator AS IT STOOD AT COMMIT ``fde3d6395`` --
#: BEFORE the all-instance arrays existed. They are therefore a genuine
#: pre-change reference and not a transcription of the current implementation's
#: own output. Any single differing byte in any of the seven keys moves a
#: digest. The property pinned is RNG NEUTRALITY: the export loop consumes no
#: ``rng`` draw, so adding the all-instance arrays cannot perturb the stream by
#: one draw and every published ``box_iou`` stays comparable.
PINNED_PRE_CHANGE_KEY_SHA256 = {
    7: {
        "image":
            "fe593609203c1b879f2f0b24f8d382f7186fe3c5fdf4eee45737aa1d0f378d6c",
        "token_ids":
            "01d8dd0e6c04706d594b2c97508cdfdd3998e434939ea27a1321f8ecbf58b2c6",
        "token_padding_mask":
            "562e7ee236adc0ac5e8656c529cc9a4986bdfca23544d897bed1f9678b654207",
        "target_boxes":
            "30dd70c5ca49b28a2cb13e9c4c55b525c8fbec6e6333cdba3bb9f840e84f9422",
        "target_valid":
            "64621e020f130a913940e9870252185268b0db67de268a8ccbf31079960c17a7",
        "target_masks":
            "64e3a7a3f3debf72771d25de46f1d8204885e932e47e4be7ecf1fd4459eb5f13",
        "is_exhaustive":
            "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0",
    },
    21: {
        "image":
            "25b600a4e4f756b873b7d5527b5e6a7cf24eaee4dd8fcc81eb38480277b9299b",
        "token_ids":
            "1a3fa32698aecaa69fb4f3168583319c801ff320da003b13ed0ef7d92c3e3d66",
        "token_padding_mask":
            "736cf8f98f03a46495551258387e79423cc4dc33122550b63505df9f68a37a14",
        "target_boxes":
            "9d70fb3b6d7ce8502ceb24c6a009d5bc316302308e5241d1b1aa2baac84dc70e",
        "target_valid":
            "d23d9d2d965db1842a79632f57696351b6f46b63790151eb260266d7a9964035",
        "target_masks":
            "33413a89c2587cc6fa20b8badd7d43a29ecf024cb9257ae8319242ca643365af",
        "is_exhaustive":
            "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0",
    },
}
#: Records per seed behind the digests above. Asserted, so a comparison that
#: silently covered zero records cannot pass.
PINNED_RECORD_COUNT = 64

#: Box tolerance for the all-instance oracle, at the IMAGE resolution (224 px),
#: in PIXELS, converted to the normalized frame. The oracle reads each 224 px
#: connected component's own extents; the pipeline computes the box
#: analytically from the float extent it sampled, and `cv2` rounds to integer
#: pixels, so each edge can move by about one pixel and a width by two.
#: MEASURED over 814 instances at four seeds: worst component deviation 1.97
#: px. Pinned at 3.0 px (1.53x margin), which still leaves a placement bug
#: (tens of px) and a width/height swap on a `bar` (>= 20 px) far outside; the
#: liveness arm below displaces by 5 px and must go RED.
ALL_BOX_TOLERANCE_PX = 3.0
ALL_BOX_TOLERANCE = ALL_BOX_TOLERANCE_PX / IMAGE_SIZE


# ---------------------------------------------------------------------
# Oracles. Neither calls anything in the module under test.
# ---------------------------------------------------------------------
def box_from_mask(mask: np.ndarray) -> np.ndarray:
    """Normalized ``cxcywh`` from a binary mask's OWN extents.

    This is the M1 oracle: it reads the rasterized geometry, not the parameters
    the pipeline drew from and not the pipeline's own box expression.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size, "empty mask has no box"
    height, width = mask.shape
    x_1, x_2 = float(cols[0]), float(cols[-1] + 1)
    y_1, y_2 = float(rows[0]), float(rows[-1] + 1)
    return np.asarray([
        (x_1 + x_2) * 0.5 / width,
        (y_1 + y_2) * 0.5 / height,
        (x_2 - x_1) / width,
        (y_2 - y_1) / height,
    ], dtype="float64")


def mask_statistics(component: np.ndarray) -> np.ndarray:
    """``(fill ratio, aspect ratio)`` of a binary component."""
    rows = np.flatnonzero(component.any(axis=1))
    cols = np.flatnonzero(component.any(axis=0))
    height = float(rows[-1] - rows[0] + 1)
    width = float(cols[-1] - cols[0] + 1)
    return np.asarray(
        [float(component.sum()) / (width * height), width / height])


def shape_statistics(category: str, downsample: bool,
                     count: int = 120, seed: int = 3) -> np.ndarray:
    """``(count, 2)`` statistics from DIRECT rasterization of one category.

    The reference distribution, produced without the sample generator, so a
    comparison against it is not a comparison of the pipeline with itself.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        mask = _rasterize(
            category, _sample_extent(category, IMAGE_SIZE, rng), IMAGE_SIZE)
        if downsample:
            mask = _downsample(mask, *MASK_GRID)
        rows.append(mask_statistics(mask))
    return np.asarray(rows)


def classify_component(component: np.ndarray) -> str:
    """Name a shape from its binary component, by fill ratio and aspect."""
    fill, aspect = mask_statistics(component)
    if fill < FILL_TRIANGLE_MAX:
        return "triangle"
    if fill < FILL_CIRCLE_MAX:
        return "circle"
    return "bar" if aspect > ASPECT_BAR_MIN else "square"


def components_of(image: np.ndarray) -> list:
    """Segment the composited image into per-instance binary components.

    Instances are drawn NON-OVERLAPPING and on a dark background (0.05-0.25 of
    full scale) in bright colours (0.55-1.0), so a mid-grey threshold plus
    connected components recovers them exactly -- with no reference to the
    record's target arrays.
    """
    grey = image.max(axis=-1)
    binary = (grey > 128.0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary)
    return [(labels == index) for index in range(1, count)]


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def records():
    """200 records at the shipped geometry, one fixed seed."""
    return list(synthetic_prompt_samples(
        num_samples=200, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
        context_length=CONTEXT, max_instances=N_MAX,
        zero_instance_rate=0.25, seed=7))


@pytest.fixture(scope="module")
def tiny_model():
    return Sam3TrainingModel(Sam3Image.from_variant("tiny"))


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------
class TestVocabulary:
    def test_the_map_is_injective_over_the_closed_set(self):
        encodings = {name: tuple(encode_phrase(name, CONTEXT)[0].tolist())
                     for name in CATEGORIES}
        assert len(set(encodings.values())) == len(CATEGORIES)

    def test_the_same_phrase_always_maps_to_the_same_ids(self):
        for name in CATEGORIES:
            first = encode_phrase(name, CONTEXT)[0]
            second = encode_phrase(name, CONTEXT)[0]
            np.testing.assert_array_equal(first, second)

    def test_padding_id_is_zero_and_the_mask_marks_exactly_it(self):
        assert PAD_ID == 0
        for name in CATEGORIES:
            ids, mask = encode_phrase(name, CONTEXT)
            assert mask.dtype == np.bool_
            np.testing.assert_array_equal(mask, ids == PAD_ID)
            # True AT PADDING -- the key-padding polarity, not a keep mask.
            assert not bool(mask[0]) and bool(mask[-1])

    def test_the_padding_mask_is_not_a_constant_across_the_set(self):
        # A constant mask would make the padding channel untestable: the
        # phrases deliberately differ in length (2 words vs 3).
        widths = {int((~encode_phrase(name, CONTEXT)[1]).sum())
                  for name in CATEGORIES}
        assert len(widths) > 1

    def test_every_id_is_inside_the_small_variants_vocabulary(self):
        assert max([2] + list(WORD_TO_ID.values())) < 512

    def test_a_phrase_that_does_not_fit_raises_rather_than_truncating(self):
        longest = max(len(p.split()) for p in CATEGORY_PHRASES.values())
        with pytest.raises(ValueError, match="Truncation is refused"):
            encode_phrase("bar", longest + 1)


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
class TestEmittedGeometryAgreesWithTheDrawnShapes:
    def test_every_emitted_box_matches_the_box_of_its_own_mask(self, records):
        deviations = []
        for record in records:
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                oracle = box_from_mask(record["target_masks"][slot])
                emitted = record["target_boxes"][slot].astype("float64")
                deviations.append(np.abs(oracle - emitted))
        assert len(deviations) > 100, "not enough instances to be a gate"
        worst = np.max(np.stack(deviations))
        assert worst < BOX_TOLERANCE, (
            f"worst box deviation {worst:.5f} exceeds {BOX_TOLERANCE:.5f} "
            f"({worst * MASK_GRID[0]:.2f} mask cells)")

    def test_that_oracle_goes_red_on_a_displaced_box(self, records):
        # M3 liveness arm. Without this, a tolerance loose enough to absorb the
        # rasterization jitter could also absorb a real placement bug.
        displaced = []
        for record in records:
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                oracle = box_from_mask(record["target_masks"][slot])
                wrong = record["target_boxes"][slot].astype("float64").copy()
                wrong[0] += 4.0 / MASK_GRID[0]   # four cells, not two
                displaced.append(np.max(np.abs(oracle - wrong)))
        assert min(displaced) > BOX_TOLERANCE

    def test_the_mask_area_matches_each_shapes_ANALYTIC_area(self):
        # A pure-geometry oracle: pi/4, 1, 1/2, 1 are the fill ratios of a
        # disc, a square, an isoceles triangle and a rectangle. Nothing about
        # the rasterizer or the downsampler enters it, which is why this is the
        # arm that sees a downsample-threshold mutation -- the box oracle above
        # does NOT (measured: mutation M-9 was INERT against it).
        analytic = {"circle": np.pi / 4.0, "square": 1.0,
                    "triangle": 0.5, "bar": 1.0}
        errors = {name: [] for name in CATEGORIES}
        for record in synthetic_prompt_samples(
                num_samples=150, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.0, seed=31):
            name = record["prompt_category"]
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                _, _, width, height = record["target_boxes"][slot]
                expected = (analytic[name] * width * MASK_GRID[1]
                            * height * MASK_GRID[0])
                observed = float(record["target_masks"][slot].sum())
                errors[name].append(abs(observed - expected) / expected)
        # MEASURED: per-category mean relative error 0.032-0.079 on the correct
        # code and 0.209-0.411 with the downsample threshold mutated to 0.0.
        # The 0.15 boundary sits between the two with ~2x margin either side.
        for name in CATEGORIES:
            assert len(errors[name]) >= 20, name
            assert float(np.mean(errors[name])) < 0.15, (
                f"{name}: mean relative area error "
                f"{float(np.mean(errors[name])):.3f}")

    def test_every_emitted_mask_is_binary_and_non_empty_where_valid(
            self, records):
        for record in records:
            for slot in range(N_MAX):
                mask = record["target_masks"][slot]
                assert set(np.unique(mask)).issubset({0.0, 1.0})
                if record["target_valid"][slot] > 0.0:
                    assert mask.sum() > 0.0
                else:
                    assert mask.sum() == 0.0

    def test_padded_rows_carry_exactly_zero_boxes(self, records):
        for record in records:
            padded = record["target_valid"] == 0.0
            assert np.all(record["target_boxes"][padded] == 0.0)

    def test_boxes_are_normalized_cxcywh_inside_the_unit_square(self, records):
        for record in records:
            valid = record["target_valid"] > 0.0
            boxes = record["target_boxes"][valid]
            if boxes.size == 0:
                continue
            assert np.all(boxes[:, 2:] > 0.0)
            assert np.all(boxes[:, 0] - boxes[:, 2] * 0.5 >= 0.0)
            assert np.all(boxes[:, 1] - boxes[:, 3] * 0.5 >= 0.0)
            assert np.all(boxes[:, 0] + boxes[:, 2] * 0.5 <= 1.0)
            assert np.all(boxes[:, 1] + boxes[:, 3] * 0.5 <= 1.0)

    @pytest.mark.parametrize("downsample", [False, True])
    def test_the_categories_are_separable_at_both_resolutions(self, downsample):
        ranges = {name: shape_statistics(name, downsample=downsample)
                  for name in CATEGORIES}
        # At the IMAGE resolution -- what the model classifies from -- every
        # pair must be separated on at least one axis with NO range overlap.
        # At the mask grid the same is asserted on the MEANS only, because
        # circle/triangle fill ranges do overlap on a 6-cell instance (a
        # measured property of the supervision target, recorded in data.py).
        for first in CATEGORIES:
            for second in CATEGORIES:
                if first >= second:
                    continue
                gaps = []
                for axis in (0, 1):
                    a, b = ranges[first][:, axis], ranges[second][:, axis]
                    if downsample:
                        gaps.append(abs(a.mean() - b.mean())
                                    > 3.0 * max(a.std(), b.std()))
                    else:
                        gaps.append(a.min() > b.max() or b.min() > a.max())
                assert any(gaps), f"{first} vs {second} not separable"


# ---------------------------------------------------------------------
# Prompt-vs-target correspondence
# ---------------------------------------------------------------------
class TestThePromptSelectsTheTargets:
    @pytest.mark.parametrize("seed", [0, 1, 2, 11])
    def test_targets_are_the_prompted_category_and_none_of_the_distractors(
            self, seed):
        distractor_images = 0
        checked = 0
        for record in synthetic_prompt_samples(
                num_samples=40, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25, seed=seed):
            prompt = record["prompt_category"]
            drawn = [classify_component(c)
                     for c in components_of(record["image"])]
            expected = sum(1 for name in drawn if name == prompt)
            emitted = int(record["target_valid"].sum())
            # THE discriminating assertion: a pipeline that ignored the prompt
            # and returned every instance would give `emitted == len(drawn)`.
            assert emitted == expected, (
                f"seed {seed}: prompt {prompt!r}, image holds {drawn}, "
                f"pipeline emitted {emitted} target(s)")
            if any(name != prompt for name in drawn):
                distractor_images += 1
            checked += 1
        assert checked == 40
        # Without distractors the text channel would not be load-bearing.
        assert distractor_images >= 30

    def test_the_emitted_masks_carry_the_prompted_categorys_statistics(self):
        # An AGGREGATE oracle, deliberately, because at the 64x64 mask grid the
        # circle and triangle per-instance fill ranges do overlap at their
        # extremes (measured; recorded in data.py). The reference means below
        # come from DIRECT rasterization, never from the pipeline.
        reference = {name: shape_statistics(name, downsample=True).mean(axis=0)
                     for name in CATEGORIES}
        emitted = {name: [] for name in CATEGORIES}
        for record in synthetic_prompt_samples(
                num_samples=200, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.0, seed=5):
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                emitted[record["prompt_category"]].append(
                    mask_statistics(record["target_masks"][slot]))
        for name in CATEGORIES:
            assert len(emitted[name]) >= 20, name
            centre = np.mean(np.asarray(emitted[name]), axis=0)
            distances = {other: float(np.linalg.norm(centre - reference[other]))
                         for other in CATEGORIES}
            nearest = min(distances, key=distances.get)
            assert nearest == name, (
                f"masks emitted for prompt {name!r} sit closest to "
                f"{nearest!r}: {distances}")

    def test_the_token_ids_name_the_prompted_category(self, records):
        for record in records:
            expected, mask = encode_phrase(
                record["prompt_category"], CONTEXT)
            np.testing.assert_array_equal(record["token_ids"], expected)
            np.testing.assert_array_equal(record["token_padding_mask"], mask)


# ---------------------------------------------------------------------
# The zero-instance case
# ---------------------------------------------------------------------
class TestZeroInstanceIsGenuinelySampled:
    def test_the_observed_rate_tracks_the_configured_rate(self):
        total = 600
        rate = 0.25
        observed = sum(
            1 for record in synthetic_prompt_samples(
                num_samples=total, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=rate, seed=21)
            if record["target_valid"].sum() == 0.0) / total
        # 3 sigma of a Binomial(600, 0.25) proportion is 0.053; the tolerance
        # is stated rather than tuned to the number that came out.
        assert abs(observed - rate) < 0.06, f"observed {observed:.4f}"

    @pytest.mark.parametrize("rate,expect_zero", [(0.0, False), (1.0, True)])
    def test_the_degenerate_rates_are_liveness_arms(self, rate, expect_zero):
        counts = [record["target_valid"].sum() == 0.0
                  for record in synthetic_prompt_samples(
                      num_samples=60, image_size=IMAGE_SIZE,
                      mask_grid=MASK_GRID, context_length=CONTEXT,
                      max_instances=N_MAX, zero_instance_rate=rate, seed=4)]
        assert all(counts) if expect_zero else not any(counts)

    def test_keep_loss_is_zero_on_exactly_the_zero_instance_samples(
            self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=64, batch_size=8, max_instances=N_MAX,
            zero_instance_rate=0.5, seed=13)
        zeros = 0
        for _, packed in dataset:
            fields = unpack_targets(packed, include_masks=False)
            keep = np.asarray(fields["keep_loss"]).reshape(-1)
            valid = np.asarray(fields["target_valid"]).sum(axis=-1)
            np.testing.assert_array_equal(keep, (valid > 0.0).astype("float32"))
            np.testing.assert_array_equal(
                np.asarray(fields["num_boxes"]), valid)
            zeros += int((keep == 0.0).sum())
        # Both arms present: neither all-zero nor all-one would be a gate.
        assert 0 < zeros < 64

    def test_is_exhaustive_is_one_on_every_row(self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=16, batch_size=8, max_instances=N_MAX,
            seed=2)
        for _, packed in dataset:
            fields = unpack_targets(packed, include_masks=False)
            np.testing.assert_array_equal(
                np.asarray(fields["is_exhaustive"]), np.ones(8, "float32"))


# ---------------------------------------------------------------------
# The three-way width contract, from the pipeline's side
# ---------------------------------------------------------------------
class TestThePackedWidthIsDerived:
    @pytest.mark.parametrize("include_masks", [False, True])
    def test_the_emitted_width_equals_the_models_own_spec(self, include_masks):
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=include_masks)
        dataset = build_sam3_dataset(
            model, num_samples=8, batch_size=4, max_instances=N_MAX, seed=1)
        spec = model.packed_target_spec(N_MAX)
        for _, packed in dataset:
            assert tuple(packed.shape[1:]) == tuple(spec)
            # And the width is the LAYOUT's, not merely the model's.
            assert packed.shape[-1] == packed_channel_count(model.mask_size)

    def test_the_input_dict_is_exactly_the_models_contract(self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=4, batch_size=4, max_instances=N_MAX,
            seed=1)
        inputs, _ = next(iter(dataset))
        assert set(inputs) == {"image", "token_ids", "token_padding_mask"}
        assert inputs["token_padding_mask"].dtype == tf.bool

    def test_the_batch_axis_is_static(self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=10, batch_size=4, max_instances=N_MAX,
            seed=1)
        assert dataset.element_spec[1].shape[0] == 4
        # `drop_remainder=True`: 10 samples at batch 4 gives 2 batches, not 3.
        assert sum(1 for _ in dataset) == 2


# ---------------------------------------------------------------------
# Hygiene and reproducibility
# ---------------------------------------------------------------------
class TestPipelineHygiene:
    def test_nothing_is_written_to_disk(self, tiny_model):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            os.chdir(workdir)
            try:
                dataset = build_sam3_dataset(
                    tiny_model, num_samples=8, batch_size=4,
                    max_instances=N_MAX, seed=1)
                for _ in dataset:
                    pass
                leftovers = sorted(os.listdir(workdir))
            finally:
                os.chdir(original)
        assert leftovers == [], f"the source wrote {leftovers}"

    def test_the_first_batch_is_bit_reproducible_across_processes(self):
        # The expected digest was computed in a DIFFERENT process. An
        # in-process "two datasets agree" check cannot see a per-process hash
        # salt or a parallel stateful RNG; a pinned constant can.
        # `include_masks=True` deliberately: at the default the packed target
        # carries no mask block and the digest would be blind to the masks.
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=True)
        dataset = build_sam3_dataset(
            model, num_samples=8, batch_size=4, max_instances=N_MAX,
            seed=99)
        inputs, packed = next(iter(dataset))
        digest = hashlib.sha256()
        for tensor in (inputs["image"], inputs["token_ids"],
                       inputs["token_padding_mask"], packed):
            digest.update(np.asarray(tensor).tobytes())
        assert digest.hexdigest() == PINNED_FIRST_BATCH_SHA256

    def test_two_seeds_do_not_produce_the_same_data(self, tiny_model):
        # Liveness arm for the digest above: a source that ignored its seed
        # would satisfy the pin trivially.
        first = next(iter(build_sam3_dataset(
            tiny_model, num_samples=4, batch_size=4, seed=99)))[0]["image"]
        second = next(iter(build_sam3_dataset(
            tiny_model, num_samples=4, batch_size=4, seed=100)))[0]["image"]
        assert not np.array_equal(np.asarray(first), np.asarray(second))


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------
class TestEndToEnd:
    def test_one_batch_through_the_model_and_loss_gives_a_finite_loss(
            self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=8, batch_size=4, max_instances=N_MAX,
            zero_instance_rate=0.5, seed=17)
        loss_fn = Sam3DetectionLoss(include_masks=False)
        inputs, packed = next(iter(dataset))
        predictions = tiny_model(inputs, training=False)
        value = float(loss_fn(packed, predictions))
        assert np.isfinite(value) and value > 0.0

    def test_a_stock_fit_step_completes_on_this_dataset(self):
        model = Sam3TrainingModel(Sam3Image.from_variant("tiny"))
        compile_sam3_trainer(model, optimizer="sgd")
        dataset = build_sam3_dataset(
            model, num_samples=8, batch_size=4, max_instances=N_MAX,
            zero_instance_rate=0.5, seed=23)
        history = model.fit(dataset, epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][0])
        assert model.jit_compile is False


# ---------------------------------------------------------------------
# The eval-only all-instance export (plan step 2: SC-A / SC-B, I-1..I-3)
# ---------------------------------------------------------------------
def key_digests(seed: int, count: int) -> dict:
    """Per-key sha256 over ``count`` records of the SEVEN signature keys."""
    digests = {name: hashlib.sha256() for name in SIGNATURE_KEYS}
    seen = 0
    for record in synthetic_prompt_samples(
            num_samples=count, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
            context_length=CONTEXT, max_instances=N_MAX,
            zero_instance_rate=0.25, seed=seed):
        for name in SIGNATURE_KEYS:
            digests[name].update(np.asarray(record[name]).tobytes())
        seen += 1
    return {"count": seen,
            **{name: digest.hexdigest() for name, digest in digests.items()}}


def all_instances_of(record: dict) -> list:
    """``[(category name, box)]`` read off the eval-only arrays."""
    keep = np.flatnonzero(record["all_valid"] > 0.0)
    return [(CATEGORIES[int(record["all_category_ids"][index])],
             record["all_boxes"][index].astype("float64"))
            for index in keep]


def image_side_instances(image: np.ndarray) -> list:
    """``[(category name, box)]`` derived FROM THE IMAGE, not from the record.

    The structural oracle for the all-instance export: connected components of
    the composited 224 px image, each classified by its shape statistics and
    boxed from its own extents. It never reads any target array, so it cannot
    agree with the pipeline by construction -- a pipeline that filled
    ``all_boxes`` from ``target_boxes`` disagrees on the COUNT.
    """
    return [(classify_component(component), box_from_mask(component))
            for component in components_of(image)]


def pair_by_centre(emitted: list, oracle: list) -> list:
    """Pair two instance lists by nearest centre; assert it is a bijection."""
    assert len(emitted) == len(oracle)
    taken = set()
    pairs = []
    for name, box in oracle:
        order = sorted(
            range(len(emitted)),
            key=lambda index: float(np.hypot(*(emitted[index][1][:2] - box[:2])))
        )
        choice = next(index for index in order if index not in taken)
        taken.add(choice)
        pairs.append((emitted[choice], (name, box)))
    assert len(taken) == len(oracle), "centre pairing was not a bijection"
    return pairs


class TestTheAllInstanceExportIsRngNeutral:
    """SC-A. What these pin: adding the all-instance arrays changed NOTHING
    about the seven keys the training path consumes, byte for byte."""

    @pytest.mark.parametrize("seed", sorted(PINNED_PRE_CHANGE_KEY_SHA256))
    def test_the_seven_keys_match_their_PRE_CHANGE_digests(self, seed):
        observed = key_digests(seed, PINNED_RECORD_COUNT)
        # A comparison over zero records must be impossible.
        assert observed["count"] == PINNED_RECORD_COUNT
        expected = PINNED_PRE_CHANGE_KEY_SHA256[seed]
        differing = [name for name in SIGNATURE_KEYS
                     if observed[name] != expected[name]]
        assert not differing, (
            f"seed {seed}: {differing} differ from the pre-change generator "
            f"over {observed['count']} records -- RNG neutrality is BROKEN")

    def test_the_digests_discriminate(self):
        # Liveness for the pin above: a different seed must not match, or the
        # digests would be satisfiable by any stream at all.
        observed = key_digests(3, PINNED_RECORD_COUNT)
        assert observed["target_boxes"] != (
            PINNED_PRE_CHANGE_KEY_SHA256[7]["target_boxes"])

    def test_the_export_loop_contains_no_rng_reference(self):
        # The STRUCTURAL half of the RNG-neutrality argument (finding F-10):
        # every draw for a record happens above the export loop, so the loop
        # cannot perturb the stream however much it computes. This goes RED the
        # moment anyone puts a draw in it.
        source = inspect.getsource(synthetic_prompt_samples)
        start = source.index("boxes = np.zeros((max_instances, 4)")
        end = source.index("token_ids, padding_mask = tokens[prompt]")
        assert start < end, "the export-loop markers moved"
        # Comments are stripped: the claim is about EXECUTED code, and the
        # block's own explanatory comment names `rng` on purpose.
        code = "\n".join(line for line in source[start:end].splitlines()
                         if not line.strip().startswith("#"))
        assert "rng" not in code, (
            "the export loop now references `rng` -- every published box_iou "
            "loses comparability")

    @pytest.mark.parametrize("include_all_instances", [False, True])
    def test_the_pinned_first_batch_digest_holds_at_both_arms(
            self, include_all_instances):
        # The cross-process pin, re-run through the opt-in. The digest covers
        # image + tokens + mask + the packed boxes/valid/masks/exhaustive.
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=True)
        dataset = build_sam3_dataset(
            model, num_samples=8, batch_size=4, max_instances=N_MAX,
            seed=99, include_all_instances=include_all_instances)
        element = next(iter(dataset))
        inputs, packed = element[0], element[1]
        digest = hashlib.sha256()
        for tensor in (inputs["image"], inputs["token_ids"],
                       inputs["token_padding_mask"], packed):
            digest.update(np.asarray(tensor).tobytes())
        assert digest.hexdigest() == PINNED_FIRST_BATCH_SHA256


class TestTheAllInstanceOptInIsAdditive:
    """SC-A / I-3. What these pin: the opt-in is OFF by default and, when on,
    nothing new reaches the model's inputs or the packed target."""

    def test_the_default_element_spec_is_unchanged(self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=8, batch_size=4, max_instances=N_MAX,
            seed=1)
        assert len(dataset.element_spec) == 2
        assert set(dataset.element_spec[0]) == {
            "image", "token_ids", "token_padding_mask"}

    def test_the_opt_in_adds_a_THIRD_element_and_touches_nothing_else(
            self, tiny_model):
        plain = build_sam3_dataset(
            tiny_model, num_samples=8, batch_size=4, max_instances=N_MAX,
            seed=1)
        extended = build_sam3_dataset(
            tiny_model, num_samples=8, batch_size=4, max_instances=N_MAX,
            seed=1, include_all_instances=True)
        assert len(extended.element_spec) == 3
        # The model input contract and the packed target width are IDENTICAL.
        assert extended.element_spec[0] == plain.element_spec[0]
        assert extended.element_spec[1] == plain.element_spec[1]
        assert extended.element_spec[1].shape[1:] == tuple(
            tiny_model.packed_target_spec(N_MAX))
        capacity = all_instance_capacity(3)
        extras = extended.element_spec[2]
        assert set(extras) == {"all_boxes", "all_valid", "all_category_ids",
                               "prompt_category_id"}
        assert tuple(extras["all_boxes"].shape) == (4, capacity, 4)
        assert tuple(extras["all_valid"].shape) == (4, capacity)
        assert tuple(extras["all_category_ids"].shape) == (4, capacity)
        assert extras["all_category_ids"].dtype == tf.int32

    def test_the_batches_are_byte_identical_across_the_two_arms(
            self, tiny_model):
        plain = build_sam3_dataset(
            tiny_model, num_samples=64, batch_size=8, max_instances=N_MAX,
            seed=5)
        extended = build_sam3_dataset(
            tiny_model, num_samples=64, batch_size=8, max_instances=N_MAX,
            seed=5, include_all_instances=True)
        compared = 0
        for left, right in zip(plain, extended):
            for name in ("image", "token_ids", "token_padding_mask"):
                assert (np.asarray(left[0][name]).tobytes()
                        == np.asarray(right[0][name]).tobytes()), name
            assert (np.asarray(left[1]).tobytes()
                    == np.asarray(right[1]).tobytes()), "packed target"
            compared += int(np.asarray(left[1]).shape[0])
        assert compared == 64, f"compared {compared} records, expected 64"

    def test_the_capacity_is_derived_from_the_generators_own_limits(self):
        # Written FROM THE STRUCTURE: at most `len(CATEGORIES) - 1` categories
        # are drawn (the count draw is upper-exclusive), each contributing at
        # most `max_per_category`.
        for max_per_category in (1, 2, 3, 5):
            assert all_instance_capacity(max_per_category) == (
                (len(CATEGORIES) - 1) * max_per_category)
        with pytest.raises(ValueError, match="must be >= 1"):
            all_instance_capacity(0)

    @pytest.mark.parametrize("max_per_category", [1, 2, 3])
    def test_the_derived_capacity_is_never_exceeded_in_practice(
            self, max_per_category):
        capacity = all_instance_capacity(max_per_category)
        worst = 0
        for record in synthetic_prompt_samples(
                num_samples=120, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25,
                max_per_category=max_per_category, seed=41):
            assert record["all_valid"].shape == (capacity,)
            worst = max(worst, int(record["all_valid"].sum()))
        assert worst <= capacity
        # And the capacity is not absurdly slack -- otherwise "never exceeded"
        # would be uninformative.
        assert worst >= capacity - max_per_category


class TestTheAllInstanceGeometryMatchesTheIMAGE:
    """SC-B. The oracle is the rendered image, never the record's own arrays."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 11])
    def test_every_all_instance_box_and_category_matches_the_image(self, seed):
        checked = 0
        deviations = []
        for record in synthetic_prompt_samples(
                num_samples=40, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25, seed=seed):
            emitted = all_instances_of(record)
            oracle = image_side_instances(record["image"])
            # THE discriminating assertion on the count: a pipeline that filled
            # `all_boxes` from `target_boxes` emits only the prompted subset.
            assert len(emitted) == len(oracle), (
                f"seed {seed}: image holds {[n for n, _ in oracle]}, the "
                f"all-instance arrays hold {[n for n, _ in emitted]}")
            for (got_name, got_box), (want_name, want_box) in pair_by_centre(
                    emitted, oracle):
                assert got_name == want_name, (
                    f"seed {seed}: box at {got_box[:2]} is {got_name!r} in the "
                    f"record and {want_name!r} in the image")
                deviations.append(np.abs(got_box - want_box))
                checked += 1
        assert checked > 100, "not enough instances to be a gate"
        worst = float(np.max(np.stack(deviations)))
        assert worst < ALL_BOX_TOLERANCE, (
            f"worst all-instance box deviation {worst:.5f} exceeds "
            f"{ALL_BOX_TOLERANCE:.5f} ({worst * IMAGE_SIZE:.2f} px)")

    def test_that_oracle_goes_red_on_a_displaced_box(self):
        # Liveness arm for the tolerance: four pixels must not fit inside it.
        displaced = []
        for record in synthetic_prompt_samples(
                num_samples=40, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25, seed=0):
            oracle = image_side_instances(record["image"])
            emitted = all_instances_of(record)
            for (_, got_box), (_, want_box) in pair_by_centre(emitted, oracle):
                wrong = got_box.copy()
                wrong[0] += 5.0 / IMAGE_SIZE
                displaced.append(float(np.max(np.abs(wrong - want_box))))
        assert displaced and min(displaced) > ALL_BOX_TOLERANCE

    def test_the_all_instance_set_is_a_STRICT_superset_of_the_targets(self):
        # The single assertion that would fire if `all_boxes` were a copy of
        # `target_boxes`, or empty, or all-zeros.
        strictly_larger = 0
        total = 0
        for record in synthetic_prompt_samples(
                num_samples=120, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25, seed=13):
            prompted = int(record["target_valid"].sum())
            everything = int(record["all_valid"].sum())
            assert everything >= prompted, (
                f"{everything} all-instance rows < {prompted} target rows")
            strictly_larger += int(everything > prompted)
            total += 1
        assert total == 120
        # At least ONE category is always absent and 2-3 are drawn, so a
        # distractor exists on nearly every image.
        assert strictly_larger >= 100, (
            f"only {strictly_larger}/120 images carry a distractor -- the "
            f"all-instance arrays are not carrying non-prompted geometry")

    def test_the_distractor_subset_is_non_empty_and_excludes_the_prompt(self):
        distractors = 0
        for record in synthetic_prompt_samples(
                num_samples=120, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25, seed=13):
            prompt_id = int(record["prompt_category_id"])
            assert CATEGORIES[prompt_id] == record["prompt_category"]
            keep = record["all_valid"] > 0.0
            ids = record["all_category_ids"]
            mask = keep & (ids != prompt_id)
            distractors += int(mask.sum())
            boxes = record["all_boxes"][mask]
            if boxes.size:
                # Real geometry, not zeros.
                assert np.all(boxes[:, 2:] > 0.0)
            # Padding carries the sentinel id and a zero box.
            assert np.all(ids[~keep] == CATEGORY_ID_PAD)
            assert np.all(record["all_boxes"][~keep] == 0.0)
        assert distractors > 200, (
            f"only {distractors} distractor instance(s) over 120 images")

    def test_the_prompted_subset_agrees_with_target_boxes(self):
        # I-2 from the other side: the `:371` filter is SUPPLEMENTED, so the
        # prompted rows of the all-instance arrays reproduce `target_boxes`.
        compared = 0
        for record in synthetic_prompt_samples(
                num_samples=80, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25, seed=29):
            prompt_id = int(record["prompt_category_id"])
            keep = (record["all_valid"] > 0.0) & (
                record["all_category_ids"] == prompt_id)
            mine = record["all_boxes"][keep]
            theirs = record["target_boxes"][record["target_valid"] > 0.0]
            assert len(mine) == len(theirs)
            if len(mine):
                np.testing.assert_array_equal(
                    mine[np.lexsort(mine.T)], theirs[np.lexsort(theirs.T)])
                compared += len(mine)
        assert compared > 100
